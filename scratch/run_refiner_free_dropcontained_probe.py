"""free@0.86を固定し、drop_contained閾値(現行0.90はshrink向けにチューニング
された値のまま)を再調整してfold0で検証する。

⚠️ **2026-08-29監査での訂正: 前提のfree@0.86自体がfold0選択バイアスと
判明(fold1 Δ-0.0157、fold2 Δ-0.0106)。この結果も不採用。README.md §2.0.1参照。**

## 背景

free variant(候補マスクとの交差なし、確率閾値のみ)の閾値スイープで、
閾値0.86がPQ 0.4361(基準0.4393比Δ-0.0032)まで肉薄した。shrinkとの
ハイブリッドは単独のfree@0.86を超えられなかった。free variantは候補外へ
広がるため、shrink由来のマスクより重複(隣接個体との被り)が発生しやすい
可能性があり、drop_contained(重複除去)の最適値がshrink時代の0.90から
ずれている可能性がある。

## このスクリプトでやること

free@0.86固定で、drop_contained閾値のみをスイープする。

使い方:
    python scratch/run_refiner_free_dropcontained_probe.py [--limit N]
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS, FOLDS_CSV, MIN_AREA, OOF_CACHE, PREPROC, TRAIN_JSON,
    imread_unicode, iou_matrix, load_yolo_cache, masks_to_rles,
    refine_batch, to_rle,
)
from pycocotools import mask as mask_util  # noqa: E402

CONF = 0.26
FOLD = 0
BASELINE_FOLD0_PQ = 0.4393
FREE_THRESHOLD = 0.86


def filter_proposals(masks, confidences):
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CONF and m.sum() >= MIN_AREA]
    return [masks[i] for i in keep]


def free_variant(proposal, box, probability, threshold):
    """候補マスクとの交差を取らず、確率マップをそのまま閾値化する。"""
    x0, y0, x1, y1 = box
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    return refined


def greedy_disjoint_local(masks, order, min_area=MIN_AREA, canvas=2048):
    occupied = np.zeros((canvas, canvas), dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept


def drop_contained_indices_local(masks, threshold):
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = mask_util.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))
    areas = np.asarray(mask_util.area(rles), dtype=np.float64)
    self_iou = np.asarray(mask_util.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
    dropped = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped:
                continue
            value = self_iou[i, j]
            if value <= 0:
                continue
            intersection = value * (areas[i] + areas[j]) / (1.0 + value)
            smaller = i if areas[i] <= areas[j] else j
            if intersection / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]


def apply_ownership_local(masks, threshold):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices_local(masks, threshold)
    return greedy_disjoint_local(masks, sorted(subset, key=lambda i: areas[i]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("output/refiner_free"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

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
    targets = [
        n for n in sorted(file_to_ids)
        if fold_of_file[n] == FOLD
        and (OOF_CACHE / f"yolo_fold_{FOLD}" / f"{Path(n).stem}.npz").exists()
    ]
    if args.limit:
        targets = targets[:args.limit]
    print(f"観測数: {len(targets)}", flush=True)

    checkpoint = torch.load(
        ARTIFACTS / "mediseg_refiner_fold0_resume.pt", map_location="cpu", weights_only=False)
    refiner = MediSegUNet(3, 1, False).eval()
    refiner.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    # free@0.86固定で確率マップは画像ごとに一度だけ推論し、複数のdrop_contained
    # 閾値で使い回す(全観測分をメモリに保持しない設計を維持)
    drop_thresholds = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
    meters = {t: PanopticQuality() for t in drop_thresholds}
    started = time.perf_counter()

    for position, name in enumerate(targets, 1):
        variants = [gt_rles[i] for i in file_to_ids[name]]
        image = imread_unicode(PREPROC / name)
        if image is None:
            continue

        masks, confidences = load_yolo_cache(OOF_CACHE / f"yolo_fold_{FOLD}" / f"{Path(name).stem}.npz")
        proposals = filter_proposals(masks, confidences)
        if not proposals:
            for threshold in drop_thresholds:
                for truth in variants:
                    meters[threshold].add_matrix(np.zeros((len(truth), 0)), len(truth), 0)
            continue

        boxes, probabilities = refine_batch(refiner, image, proposals)
        refined = [free_variant(proposals[i], boxes[i], probabilities[i], FREE_THRESHOLD)
                   for i in range(len(proposals))]

        for drop_threshold in drop_thresholds:
            final = apply_ownership_local(refined, drop_threshold)
            rles = masks_to_rles(final)
            for truth in variants:
                matrix = iou_matrix(truth, rles)
                meters[drop_threshold].add_matrix(matrix, len(truth), len(rles))

        if position % 25 == 0 or position == len(targets):
            scores = {t: round(m.score, 4) for t, m in meters.items()}
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  {scores}", flush=True)

    report = {
        "scope": f"{len(targets)}観測、fold0のみ。free@{FREE_THRESHOLD}固定でdrop_contained閾値をスイープ",
        "baseline_shrink045_fold0_pq": BASELINE_FOLD0_PQ,
        "results": {
            str(t): {
                **m.summary(), "delta_vs_baseline": round(m.summary()["pq"] - BASELINE_FOLD0_PQ, 6),
                "beats_baseline": bool(m.summary()["pq"] > BASELINE_FOLD0_PQ)}
            for t, m in meters.items()
        },
    }
    (args.out / "refiner_free086_dropcontained_sweep.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
