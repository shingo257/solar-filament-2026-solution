"""refiner の後処理方式・閾値を **全fold OOF** の Panoptic Quality で選ぶ (CPU、GPU不要)。

`run_refiner_pq_ablation.py` は fold0 の checkpoint しか無かったため fold0 限定だった。
fold1 / fold2 の refiner を学習したので、ようやく V31 以来の「fold0だけで後処理を
選んでいる」選択バイアスを外して確定できる。

リーク管理 (§18.2):
  各観測には、その観測の fold を **学習から除いた** checkpoint だけを適用する。
    fold0 の観測 → mediseg_refiner_fold0_resume.pt (fold1+2 で学習)
    fold1 の観測 → mediseg_refiner_fold1_resume.pt (fold0+2 で学習)
    fold2 の観測 → mediseg_refiner_fold2_resume.pt (fold0+1 で学習)
  all-data refiner はここでは絶対に使わない。

高速化:
  1画像分の crop をまとめて1回の forward に通す (fold0版は1 crop ずつだった)。
  variant は fold0 の結果で shrink 優位・閾値ほぼ平坦と分かったので、
  shrink を厚く、guided は比較対照として最小限だけ残す。

使い方:
    python scratch/run_refiner_pq_allfold.py [--conf 0.26] [--limit N] [--threads 14]
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
ARTIFACTS = V30 / "artifacts"

EVAL_SIZE = 2048
MIN_AREA = 20
CROP_SIZE = 320
CROP_CONTEXT = 1.65
FOLDS = (0, 1, 2)
REFINER_BATCH = 8

# fold0 の結果: shrink が全閾値で guided を上回り、閾値は shrink 内でほぼ平坦。
# shrink を厚く取り、guided は比較対照 (V31が選んだ guided7@0.45 を含む) に絞る。
VARIANTS = (
    ("shrink", 0.45), ("shrink", 0.55), ("shrink", 0.65), ("shrink", 0.75), ("shrink", 0.85),
    ("guided3", 0.65), ("guided7", 0.45), ("guided7", 0.65),
)


def to_rle(annotation, height, width):
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
    """cv2.imread は Windows で非ASCIIパス (`G:\\マイドライブ`) を開けない。"""
    if not path.exists():
        return None
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def square_box(mask, context=CROP_CONTEXT, minimum=96):
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


def refine_batch(model, image, proposals):
    """1画像分の crop をまとめて forward する。戻り値は (boxes, probability maps)。"""
    boxes, crops = [], []
    for proposal in proposals:
        x0, y0, x1, y1 = square_box(proposal)
        crop = cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE),
                          interpolation=cv2.INTER_LINEAR)
        boxes.append((x0, y0, x1, y1))
        crops.append(crop.transpose(2, 0, 1))
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(crops), REFINER_BATCH):
            chunk = np.stack(crops[start:start + REFINER_BATCH])
            tensor = torch.from_numpy(chunk.copy()).float() / 255.0
            output = model(tensor).sigmoid()[:, 0].numpy()
            for offset in range(output.shape[0]):
                x0, y0, x1, y1 = boxes[start + offset]
                probabilities.append(cv2.resize(output[offset], (x1 - x0, y1 - y0),
                                                interpolation=cv2.INTER_LINEAR))
    return boxes, probabilities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.26)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("output/v35"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {i["id"]: i for i in coco["images"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    file_to_ids = defaultdict(list)
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
               if (OOF_CACHE / f"yolo_fold_{fold_of_file[n]}" / f"{Path(n).stem}.npz").exists()]
    if args.limit:
        targets = targets[:args.limit]

    # foldごとの refiner。対象foldを学習から除いた checkpoint だけを使う。
    refiners = {}
    for fold in FOLDS:
        path = ARTIFACTS / f"mediseg_refiner_fold{fold}_resume.pt"
        if not path.exists():
            raise FileNotFoundError(f"fold{fold} の refiner がない: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        model = MediSegUNet(3, 1, False).eval()
        model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)
        refiners[fold] = model
        print(f"fold{fold} refiner: best crop Dice {checkpoint.get('best_dice'):.6f}")
    print({"観測": len(targets), "conf": args.conf, "threads": args.threads,
           "リーク": "なし (各観測へ、そのfoldを学習から除いたcheckpointのみ適用)"})

    names = ["yolo"] + [f"{m}@{t:.2f}" for m, t in VARIANTS]
    overall = {n: PanopticQuality() for n in names}
    by_fold = {n: {f: PanopticQuality() for f in FOLDS} for n in names}
    near_miss = {n: 0 for n in names}
    counts = {n: [] for n in names}
    skipped = []

    started = time.perf_counter()
    for position, name in enumerate(targets, 1):
        fold = fold_of_file[name]
        masks, confidences = load_yolo_cache(
            OOF_CACHE / f"yolo_fold_{fold}" / f"{Path(name).stem}.npz")
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= args.conf and m.sum() >= MIN_AREA]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
        order = list(np.argsort(-scores))
        variants_gt = [gt_rles[i] for i in file_to_ids[name]]

        image = imread_unicode(PREPROC / name)
        if image is None:
            skipped.append(name)
            if len(skipped) > max(3, len(targets) // 20):
                raise RuntimeError(f"前処理画像を読めない観測が{len(skipped)}件: {skipped[:3]}")
            continue

        boxes, probabilities = refine_batch(refiners[fold], image, proposals) if proposals else ([], [])

        for variant in names:
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
            for truth in variants_gt:
                matrix = iou_matrix(truth, rles)
                overall[variant].add_matrix(matrix, len(truth), len(rles))
                by_fold[variant][fold].add_matrix(matrix, len(truth), len(rles))
                if matrix.size:
                    near_miss[variant] += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

        if position % 25 == 0 or position == len(targets):
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                  f"yolo={overall['yolo'].score:.4f} "
                  f"shrink@0.65={overall['shrink@0.65'].score:.4f}", flush=True)

    table = pd.DataFrame({
        n: {**{k: overall[n].summary()[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")},
            **{f"pq_fold{f}": by_fold[n][f].score for f in FOLDS},
            "near_miss": near_miss[n], "inst/img": float(np.mean(counts[n]))}
        for n in names
    }).T
    table.index.name = "variant"
    table["ΔPQ_vs_yolo"] = table["pq"] - table.loc["yolo", "pq"]
    table["worst_fold"] = table[[f"pq_fold{f}" for f in FOLDS]].min(axis=1)
    table = table.sort_values("pq", ascending=False)
    print(f"\n=== 全fold OOF: refiner方式×閾値のPQ (conf={args.conf}, {len(targets) - len(skipped)}観測) ===")
    print(table.round(4).to_string())

    best = table.index[0]
    table.to_csv(args.out / "refiner_pq_allfold.csv")
    report = {
        "scope": f"全fold OOF ({len(targets) - len(skipped)}観測)。リークなし。",
        "conf": args.conf,
        "skipped_missing_image": len(skipped),
        "yolo_native_pq": float(table.loc["yolo", "pq"]),
        "best_variant": str(best),
        "best_pq": float(table.loc[best, "pq"]),
        "best_worst_fold_pq": float(table.loc[best, "worst_fold"]),
        "gain_over_yolo": float(table.loc[best, "ΔPQ_vs_yolo"]),
        "gain_over_v31_choice": float(table.loc[best, "pq"] - table.loc["guided7@0.45", "pq"]),
        "new_gate_pq": round(float(table.loc[best, "pq"]) + 0.010, 6),
        "fold0_only_prediction_was": "shrink@0.65 (fold0 236観測での1位)",
        "table": json.loads(table.round(6).to_json(orient="index")),
    }
    (args.out / "refiner_pq_allfold.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "table"},
                            ensure_ascii=False, indent=2))
    print(f"\n所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
