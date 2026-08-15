"""MediSeg refiner の後処理方式・閾値を Panoptic Quality で選び直す (CPU、GPU不要)。

V31 は同じ比較を Hungarian Dice で行い「guided7 と YOLO native はほぼ同値
(0.552553 vs 0.552290)」と結論した。しかし PQ の TP 条件は IoU > 0.5 の硬い崖なので、
Dice では見えなかった差が出る可能性がある。near-miss (IoU 0.3〜0.5) を崖の向こうへ
押し出せるかが論点であり、それは PQ の分母を変えずに分子だけを増やす。

リーク管理 (§18.2):
  refiner の checkpoint は fold 0 を学習から除いた `mediseg_refiner_fold0_resume.pt`
  のみを使い、評価対象も fold 0 の held-out 画像に限る。`all_data` refiner は
  OOF 画像を学習に含むため、ここでは使わない。
  したがって本スクリプトの結論は **方式の順位** についてのものであり、
  絶対値をそのまま全fold ゲートにしてはいけない。

使い方:
    python scratch/run_refiner_pq_ablation.py [--conf 0.26] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402

DRIVE = Path("G:/マイドライブ")
V30 = DRIVE / "solar_filament_v30"
V34 = DRIVE / "solar_filament_v34"
TRAIN_JSON = V30 / "competition_data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FOLDS_CSV = V30 / "artifacts/grouped_folds.csv"
OOF_CACHE = V34 / "prediction_cache/oof"
PREPROC = V30 / "preprocessed_v30_q96"
REFINER_WEIGHT = V30 / "artifacts/mediseg_refiner_fold0_resume.pt"

EVAL_SIZE = 2048
MIN_AREA = 20
CROP_SIZE = 320          # V31 CFG と同一
CROP_CONTEXT = 1.65
EVAL_FOLD = 0            # fold0 refiner なので fold0 の held-out のみ
METHODS = ("shrink", "guided3", "guided7")
THRESHOLDS = (0.45, 0.55, 0.65, 0.75)


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


def masks_to_rles(masks):
    if not masks:
        return []
    return mask_util.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))


def iou_matrix(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    return np.asarray(mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64).T


def load_yolo_cache(path: Path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))


def imread_unicode(path: Path):
    """cv2.imread は Windows で非ASCIIパスを開けない (Driveが `G:\\マイドライブ` にある)。

    バイト列を自前で読んで imdecode する。Colab 側は ASCII パスなので影響しないが、
    ローカル実行では必須。
    """
    if not path.exists():
        return None
    buffer = np.frombuffer(path.read_bytes(), np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def square_box(mask, context=CROP_CONTEXT, minimum=96):
    """V31 の square_box と同一。"""
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, width, height
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def build_variant(proposal, box, probability, method, threshold):
    """V31 の build_variant と同一。"""
    if method == "yolo":
        return proposal.copy()
    x0, y0, x1, y1 = box
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    if method == "shrink":
        return refined & proposal
    kernel = 3 if method == "guided3" else 7
    support = cv2.dilate(proposal.astype(np.uint8), np.ones((kernel, kernel), np.uint8)).astype(bool)
    return refined & support


def greedy_disjoint(masks, order, min_area=MIN_AREA):
    occupied = np.zeros((EVAL_SIZE, EVAL_SIZE), dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.26)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("output/v35"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

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
    targets = [n for n in sorted(file_to_ids)
               if fold_of_file[n] == EVAL_FOLD
               and (OOF_CACHE / f"yolo_fold_{EVAL_FOLD}" / f"{Path(n).stem}.npz").exists()]
    if args.limit:
        targets = targets[:args.limit]

    refiner = MediSegUNet().eval()
    checkpoint = torch.load(REFINER_WEIGHT, map_location="cpu")
    state = checkpoint.get("best_model") or checkpoint["model"]
    refiner.load_state_dict(state, strict=True)
    print({"評価対象": len(targets), "fold": EVAL_FOLD, "conf": args.conf,
           "refiner": REFINER_WEIGHT.name, "threads": args.threads,
           "リーク": "なし (fold0を学習から除いたcheckpointをfold0 held-outへ適用)"})

    skipped: list[str] = []
    variants = ["yolo"] + [f"{m}@{t:.2f}" for m in METHODS for t in THRESHOLDS]
    accumulators = {v: PanopticQuality() for v in variants}
    near_miss = {v: 0 for v in variants}
    counts = {v: [] for v in variants}

    started = time.perf_counter()
    for position, name in enumerate(targets, 1):
        masks, confidences = load_yolo_cache(
            OOF_CACHE / f"yolo_fold_{EVAL_FOLD}" / f"{Path(name).stem}.npz")
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= args.conf and m.sum() >= MIN_AREA]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
        order = list(np.argsort(-scores))
        gt_variants = [gt_rles[i] for i in file_to_ids[name]]

        image = imread_unicode(PREPROC / name)
        if image is None:
            skipped.append(name)
            # 静かに飛ばすと全ゼロの表が「結果」に見えてしまう。多発したら止める。
            if len(skipped) > max(3, len(targets) // 20):
                raise RuntimeError(
                    f"前処理画像を読めない observation が {len(skipped)} 件に達した。"
                    f"例: {skipped[:3]} / 探索先: {PREPROC}")
            continue

        # refiner確率は候補ごとに1回だけ計算し、方式×閾値で使い回す
        boxes, probabilities = [], []
        with torch.no_grad():
            for proposal in proposals:
                x0, y0, x1, y1 = square_box(proposal)
                crop = cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
                tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float() / 255.0
                probability = refiner(tensor).sigmoid()[0, 0].numpy()
                probabilities.append(cv2.resize(probability, (x1 - x0, y1 - y0),
                                                interpolation=cv2.INTER_LINEAR))
                boxes.append((x0, y0, x1, y1))

        for variant in variants:
            if variant == "yolo":
                built = [p.copy() for p in proposals]
            else:
                method, threshold = variant.split("@")
                built = [build_variant(proposals[i], boxes[i], probabilities[i],
                                       method, float(threshold))
                         for i in range(len(proposals))]
            predictions = greedy_disjoint(built, order)
            rles = masks_to_rles(predictions)
            counts[variant].append(len(predictions))
            for truth in gt_variants:
                matrix = iou_matrix(truth, rles)
                accumulators[variant].add_matrix(matrix, len(truth), len(rles))
                if matrix.size:
                    near_miss[variant] += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

        if position % 25 == 0 or position == len(targets):
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                  f"yolo={accumulators['yolo'].score:.4f} "
                  f"guided7@0.55={accumulators['guided7@0.55'].score:.4f}", flush=True)

    table = pd.DataFrame({
        v: {**{k: accumulators[v].summary()[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")},
            "near_miss": near_miss[v], "inst/img": float(np.mean(counts[v]))}
        for v in variants
    }).T
    table.index.name = "variant"
    table["ΔPQ_vs_yolo"] = table["pq"] - table.loc["yolo", "pq"]
    table = table.sort_values("pq", ascending=False)
    print(f"\n=== refiner方式×閾値のPQ比較 (fold{EVAL_FOLD} held-out, conf={args.conf}) ===")
    print(table.round(4).to_string())

    table.to_csv(args.out / "refiner_pq_ablation.csv")
    best = table.index[0]
    report = {
        "scope": f"fold{EVAL_FOLD} held-out のみ ({len(targets) - len(skipped)}観測)。リークなし。",
        "skipped_missing_image": len(skipped),
        "caveat": "単一foldなので絶対値を全foldゲートに使わない。方式の順位のみ読む。",
        "conf": args.conf,
        "yolo_native_pq": float(table.loc["yolo", "pq"]),
        "best_variant": str(best),
        "best_pq": float(table.loc[best, "pq"]),
        "gain_over_yolo": float(table.loc[best, "ΔPQ_vs_yolo"]),
        "v31_dice_conclusion": "guided7 0.552553 vs yolo native 0.552290 (差ほぼなし)",
        "table": json.loads(table.round(6).to_json(orient="index")),
    }
    (args.out / "refiner_pq_ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "table"},
                            ensure_ascii=False, indent=2))
    print(f"\n所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
