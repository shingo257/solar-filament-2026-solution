"""排他化（disjoint化）の画素割当方式を Panoptic Quality で比較する。

Kaggleは同一画像内のmask重複を拒否するため、重なった候補から画素を必ず削る。
V30bはconfidence降順の先着で削った。しかしPQのTP条件はIoU > 0.5の硬い崖なので、
削られて縮んだmaskが崖を割ると、得点ゼロどころかFP+FNの二重減点になる。

ここで比較するのは「GTを見ずに決められる」割当規則だけ。加えて2つの参照値を出す。

  raw     : 重複を許したまま採点。提出不可だが「排他化が何を失っているか」の基準
  oracle  : GTを見て、各候補を最も良く当てるGTへ画素を割り当てた場合の上限

maskごとに一定の優先度を与える貪欲法は、画素ごとのargmaxと厳密に一致するため、
(n, H, W) のスコア配列を作らずに済む。

使い方:
    python scratch/run_ownership_ablation.py [--conf 0.26] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics_official import PanopticQuality  # noqa: E402

DRIVE = Path("G:/マイドライブ")
V30 = DRIVE / "solar_filament_v30"
V34 = DRIVE / "solar_filament_v34"
TRAIN_JSON = V30 / "competition_data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FOLDS_CSV = V30 / "artifacts/grouped_folds.csv"
OOF_CACHE = V34 / "prediction_cache/oof"

EVAL_SIZE = 2048
MIN_AREA = 20
CONTAINMENT = 0.90


def to_rle(annotation: dict, height: int, width: int) -> dict:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_util.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def masks_to_rles(masks: list[np.ndarray]) -> list[dict]:
    if not masks:
        return []
    return mask_util.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))


def iou_matrix(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    return np.asarray(mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64).T


def load_yolo_cache(path: Path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    masks = [np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed]
    return masks, data["confidences"].astype(np.float32)


def greedy_disjoint(masks, order, min_area=MIN_AREA):
    """優先度順の貪欲な画素占有。maskごとに一定の優先度なら画素ごとargmaxと一致する。"""
    occupied = np.zeros((EVAL_SIZE, EVAL_SIZE), dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept


def drop_contained(masks, confidences, threshold=CONTAINMENT):
    """他のmaskへ threshold 以上包含されている小さいmaskを捨てる（縮めない）。"""
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = masks_to_rles(masks)
    areas = np.asarray(mask_util.area(rles), dtype=np.float64)
    self_iou = iou_matrix(rles, rles)
    dropped = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            value = self_iou[i, j]
            if value <= 0 or i in dropped or j in dropped:
                continue
            intersection = value * (areas[i] + areas[j]) / (1.0 + value)
            smaller, larger = (i, j) if areas[i] <= areas[j] else (j, i)
            if intersection / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]


def drop_any_overlap(masks, confidences):
    """重なりがあれば低confidence側を丸ごと捨てる。一切縮めない極端案。"""
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = masks_to_rles(masks)
    self_iou = iou_matrix(rles, rles)
    kept: list[int] = []
    for index in np.argsort(-confidences):
        if all(self_iou[index, other] <= 0 for other in kept):
            kept.append(int(index))
    return kept


def oracle_disjoint(masks, gt_rles, min_area=MIN_AREA):
    """上限参照: GTを見て、各候補を最も良く当てるGTとの IoU が最大になる順で占有する。

    提出には使えない。「割当規則をどれだけ改善しても届く先はここまで」を示すためだけ。
    """
    if not masks:
        return []
    if not gt_rles:
        return greedy_disjoint(masks, list(range(len(masks))))
    matrix = iou_matrix(gt_rles, masks_to_rles(masks))     # (n_gt, n_pred)
    best = matrix.max(axis=0) if matrix.size else np.zeros(len(masks))
    return greedy_disjoint(masks, list(np.argsort(-best)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.26)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("output/v35"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {i["id"]: i for i in coco["images"]}
    annotations: dict = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    file_to_ids: dict = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)
    gt_rles = {
        image_id: [to_rle(a, int(images[image_id]["height"]), int(images[image_id]["width"]))
                   for a in anns]
        for image_id, anns in annotations.items()
    }

    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    observations = sorted(file_to_ids)
    available = [n for n in observations
                 if (OOF_CACHE / f"yolo_fold_{fold_of_file[n]}" / f"{Path(n).stem}.npz").exists()]
    if args.limit:
        available = available[:args.limit]
    print({"観測": len(available), "conf": args.conf})

    strategies = ["raw", "conf_first", "area_asc", "area_desc",
                  "drop_contained_then_conf", "drop_contained_then_area_asc",
                  "drop_any_overlap", "oracle"]
    accumulators = {name: PanopticQuality() for name in strategies}
    near_miss = {name: 0 for name in strategies}
    instance_counts = {name: [] for name in strategies}
    shrink_rows = []

    started = time.perf_counter()
    for position, name in enumerate(available, 1):
        fold = fold_of_file[name]
        masks, confidences = load_yolo_cache(
            OOF_CACHE / f"yolo_fold_{fold}" / f"{Path(name).stem}.npz")
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= args.conf and m.sum() >= MIN_AREA]
        masks = [masks[i] for i in keep]
        confidences = confidences[keep] if len(keep) else np.zeros(0, np.float32)
        areas = np.array([m.sum() for m in masks], dtype=np.float64)
        variants = [gt_rles[i] for i in file_to_ids[name]]
        reference_gt = variants[0]

        candidates: dict = {}
        candidates["raw"] = list(masks)
        candidates["conf_first"] = greedy_disjoint(masks, list(np.argsort(-confidences)))
        candidates["area_asc"] = greedy_disjoint(masks, list(np.argsort(areas)))
        candidates["area_desc"] = greedy_disjoint(masks, list(np.argsort(-areas)))
        subset = drop_contained(masks, confidences)
        candidates["drop_contained_then_conf"] = greedy_disjoint(
            masks, sorted(subset, key=lambda i: -confidences[i]))
        candidates["drop_contained_then_area_asc"] = greedy_disjoint(
            masks, sorted(subset, key=lambda i: areas[i]))
        candidates["drop_any_overlap"] = [masks[i] for i in drop_any_overlap(masks, confidences)]
        candidates["oracle"] = oracle_disjoint(masks, reference_gt)

        for strategy, predictions in candidates.items():
            rles = masks_to_rles(predictions)
            instance_counts[strategy].append(len(predictions))
            for truth in variants:
                matrix = iou_matrix(truth, rles)
                accumulators[strategy].add_matrix(matrix, len(truth), len(rles))
                if matrix.size:
                    near_miss[strategy] += int(
                        ((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

        if masks:
            kept_area = sum(int(m.sum()) for m in candidates["conf_first"])
            shrink_rows.append({
                "file_name": name, "fold": fold,
                "proposals": len(masks),
                "raw_area": float(areas.sum()),
                "conf_first_area": float(kept_area),
                "lost_ratio": float(1.0 - kept_area / max(areas.sum(), 1.0)),
            })

        if position % 50 == 0 or position == len(available):
            print(f"  {position}/{len(available)}  {time.perf_counter() - started:.0f}s  "
                  f"conf_first={accumulators['conf_first'].score:.4f} "
                  f"oracle={accumulators['oracle'].score:.4f}", flush=True)

    table = pd.DataFrame({
        name: {
            "PQ": accumulators[name].summary()["pq"],
            "SQ": accumulators[name].summary()["sq"],
            "RQ": accumulators[name].summary()["rq"],
            "TP": accumulators[name].summary()["tp"],
            "FP": accumulators[name].summary()["fp"],
            "FN": accumulators[name].summary()["fn"],
            "near_miss": near_miss[name],
            "inst/img": float(np.mean(instance_counts[name])),
        } for name in strategies
    }).T
    table.index.name = "strategy"
    baseline = table.loc["conf_first", "PQ"]
    table["ΔPQ_vs_conf_first"] = table["PQ"] - baseline
    print("\n=== 排他化の割当方式比較 (conf=%.2f, 全fold OOF) ===" % args.conf)
    print(table.round(4).to_string())

    shrink = pd.DataFrame(shrink_rows)
    shrink.to_csv(args.out / "ownership_shrink.csv", index=False)
    print("\n=== conf先着で失われた画素の割合 ===")
    print(shrink["lost_ratio"].describe().round(4).to_string())

    table.to_csv(args.out / "ownership_ablation.csv")
    report = {
        "conf": args.conf,
        "observations": len(available),
        "baseline_strategy": "conf_first",
        "baseline_pq": float(baseline),
        "best_submittable": str(
            table.drop(index=["raw", "oracle"])["PQ"].idxmax()),
        "best_submittable_pq": float(table.drop(index=["raw", "oracle"])["PQ"].max()),
        "raw_pq_not_submittable": float(table.loc["raw", "PQ"]),
        "oracle_pq_upper_bound": float(table.loc["oracle", "PQ"]),
        "headroom_from_ownership": float(table.loc["oracle", "PQ"] - baseline),
        "mean_pixels_lost_ratio": float(shrink["lost_ratio"].mean()),
        "table": json.loads(table.round(6).to_json(orient="index")),
    }
    (args.out / "ownership_ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "table"},
                            ensure_ascii=False, indent=2))
    print(f"\n所要 {time.perf_counter() - started:.0f}s / 保存先 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
