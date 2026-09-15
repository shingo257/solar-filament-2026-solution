"""WBF統合後マスクに対するdrop_contained重複除去閾値(既定0.90)を再スイープする。

drop_contained_then_area_ascのIoU≥0.90という重複除去閾値は、単一seed時代の
確定値のまま使ってきた。WBFのクラスタ化(iou≥0.5)で既にある程度重複を
吸収しているため、後段のdrop_containedの最適値が変わっている可能性がある。

固定: WBF iou_threshold=0.5, vote_threshold=0.6、refiner shrink閾値0.45
(いずれも改訂11までの確定値)。可変: drop_contained閾値のみ。

使い方:
    python scratch/run_wbf_dropcontained_threshold_probe.py --drop-threshold 0.80 [--limit N]
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
from src.mask_wbf import weighted_mask_fusion  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS, FOLDS_CSV, MIN_AREA, OOF_CACHE, PREPROC, TRAIN_JSON,
    build_variant, imread_unicode, iou_matrix, load_yolo_cache,
    masks_to_rles, refine_batch, to_rle,
)
from pycocotools import mask as mask_util  # noqa: E402

CONF = 0.26
REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45
FOLD = 0
BASELINE_FOLD0_PQ = 0.4393
WBF_IOU = 0.5
WBF_VOTE = 0.6


def filter_proposals(masks, confidences):
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CONF and m.sum() >= MIN_AREA]
    return [masks[i] for i in keep], (confidences[keep] if keep else np.zeros(0, np.float32))


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


def apply_ownership_with_threshold(masks, threshold):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices_local(masks, threshold)
    return greedy_disjoint_local(masks, sorted(subset, key=lambda i: areas[i]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--drop-threshold", type=float, default=0.90)
    parser.add_argument(
        "--seed2-cache", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "kaggle_downloads" / "v42" / "oof_cache" / "yolo_seed8407_fold_0",
    )
    parser.add_argument("--out", type=Path, default=Path("output/v42"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.seed2_cache.exists():
        print(f"seed2キャッシュが見つからない: {args.seed2_cache}")
        return 1

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
        and (args.seed2_cache / f"{Path(n).stem}.npz").exists()
    ]
    if args.limit:
        targets = targets[:args.limit]
    print(f"観測数: {len(targets)}  drop_threshold={args.drop_threshold}", flush=True)

    checkpoint = torch.load(
        ARTIFACTS / "mediseg_refiner_fold0_resume.pt", map_location="cpu", weights_only=False)
    refiner = MediSegUNet(3, 1, False).eval()
    refiner.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    meter = PanopticQuality()
    near_miss = 0
    started = time.perf_counter()

    for position, name in enumerate(targets, 1):
        variants = [gt_rles[i] for i in file_to_ids[name]]
        image = imread_unicode(PREPROC / name)
        if image is None:
            continue

        masks1, conf1 = load_yolo_cache(OOF_CACHE / f"yolo_fold_{FOLD}" / f"{Path(name).stem}.npz")
        masks2, conf2 = load_yolo_cache(args.seed2_cache / f"{Path(name).stem}.npz")
        proposals1, scores1 = filter_proposals(masks1, conf1)
        proposals2, scores2 = filter_proposals(masks2, conf2)

        combined_masks = proposals1 + proposals2
        combined_scores = (np.concatenate([scores1, scores2])
                            if (proposals1 or proposals2) else np.zeros(0, np.float32))
        source_ids = [0] * len(proposals1) + [1] * len(proposals2)
        fused_masks, fused_scores, _ = weighted_mask_fusion(
            combined_masks, combined_scores, source_ids,
            iou_threshold=WBF_IOU, vote_threshold=WBF_VOTE)

        if fused_masks:
            boxes, probabilities = refine_batch(refiner, image, fused_masks)
            refined = [build_variant(fused_masks[i], boxes[i], probabilities[i],
                                     REFINER_METHOD, REFINER_THRESHOLD)
                       for i in range(len(fused_masks))]
            final = apply_ownership_with_threshold(refined, args.drop_threshold)
        else:
            final = []
        rles = masks_to_rles(final)

        for truth in variants:
            matrix = iou_matrix(truth, rles)
            meter.add_matrix(matrix, len(truth), len(rles))
            if matrix.size:
                near_miss += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

        if position % 25 == 0 or position == len(targets):
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                  f"pq={meter.score:.4f}", flush=True)

    summary = meter.summary()
    report = {
        "scope": f"{len(targets)}観測、fold0のみ。WBF(iou={WBF_IOU},vote={WBF_VOTE})→drop_contained閾値スイープ",
        "drop_threshold": args.drop_threshold,
        **summary,
        "near_miss": near_miss,
        "baseline_fold0_pq": BASELINE_FOLD0_PQ,
        "delta_vs_baseline": round(summary["pq"] - BASELINE_FOLD0_PQ, 6),
        "continue_fold12": bool(summary["pq"] > BASELINE_FOLD0_PQ),
    }
    suffix = f"_limit{args.limit}" if args.limit else ""
    (args.out / f"wbf_dropcontained_threshold{args.drop_threshold:.2f}{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
