"""癒着分離(src/blob_split.py)をV35確定パイプラインに挿入した場合のPQを測る。

near-miss被覆モード診断(改訂7)でundercoverage/overcoverageがほぼ拮抗していた。
近接する2個体をYOLOが1つの塊として検出すると、一方はovercoverage、
もう一方はundercoverage(検出漏れ)になる、という「癒着」仮説を検証する。

挿入位置: refinerでshrinkした後、drop_contained_then_area_asc(排他化)の前。
GTは使わない(mask形状だけで分割可否を決める)ので、選択バイアスの心配はない。

使い方:
    python scratch/run_blob_split_probe.py [--limit N] [--conf 0.26] [--threads 14]
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
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402
from src.blob_split import split_proposals  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS, FOLDS, FOLDS_CSV, MIN_AREA, OOF_CACHE, PREPROC, TRAIN_JSON,
    build_variant, greedy_disjoint, imread_unicode, iou_matrix, load_yolo_cache,
    masks_to_rles, refine_batch, to_rle,
)
from scratch.run_pipeline_stack import apply_ownership  # noqa: E402

CONF = 0.26
REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("output/v35"))
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
    targets = [n for n in sorted(file_to_ids)
               if (OOF_CACHE / f"yolo_fold_{fold_of_file[n]}" / f"{Path(n).stem}.npz").exists()]
    if args.limit:
        targets = targets[:args.limit]

    refiners = {}
    for fold in FOLDS:
        path = ARTIFACTS / f"mediseg_refiner_fold{fold}_resume.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = MediSegUNet(3, 1, False).eval()
        model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)
        refiners[fold] = model
    print(f"観測数: {len(targets)}", flush=True)

    meter_baseline = PanopticQuality()
    meter_split = PanopticQuality()
    near_miss_baseline = 0
    near_miss_split = 0
    split_events = 0
    started = time.perf_counter()

    for position, name in enumerate(targets, 1):
        fold = fold_of_file[name]
        masks, confidences = load_yolo_cache(
            OOF_CACHE / f"yolo_fold_{fold}" / f"{Path(name).stem}.npz")
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= CONF and m.sum() >= MIN_AREA]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
        variants = [gt_rles[i] for i in file_to_ids[name]]

        image = imread_unicode(PREPROC / name)
        if image is None:
            continue

        if proposals:
            boxes, probabilities = refine_batch(refiners[fold], image, proposals)
            refined = [build_variant(proposals[i], boxes[i], probabilities[i],
                                     REFINER_METHOD, REFINER_THRESHOLD)
                       for i in range(len(proposals))]
        else:
            refined = []

        # ベースライン(V35確定構成)
        baseline_final = apply_ownership(refined, scores, "drop_contained_then_area_asc") if refined else []
        baseline_rles = masks_to_rles(baseline_final)

        # 癒着分離あり
        if refined:
            split_masks, split_scores = split_proposals(refined, scores)
            split_events += len(split_masks) - len(refined)
        else:
            split_masks, split_scores = [], np.zeros(0, np.float32)
        split_final = apply_ownership(split_masks, split_scores, "drop_contained_then_area_asc") if split_masks else []
        split_rles = masks_to_rles(split_final)

        for truth in variants:
            matrix_b = iou_matrix(truth, baseline_rles)
            meter_baseline.add_matrix(matrix_b, len(truth), len(baseline_rles))
            if matrix_b.size:
                near_miss_baseline += int(((matrix_b > 0.3) & (matrix_b <= 0.5)).any(axis=0).sum())

            matrix_s = iou_matrix(truth, split_rles)
            meter_split.add_matrix(matrix_s, len(truth), len(split_rles))
            if matrix_s.size:
                near_miss_split += int(((matrix_s > 0.3) & (matrix_s <= 0.5)).any(axis=0).sum())

        if position % 25 == 0 or position == len(targets):
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                  f"baseline={meter_baseline.score:.4f}  split={meter_split.score:.4f}  "
                  f"split_events={split_events}", flush=True)

    summary_baseline = meter_baseline.summary()
    summary_split = meter_split.summary()
    report = {
        "scope": f"{len(targets)}観測(リークなしrefiner)。V35確定構成 vs 癒着分離挿入版",
        "baseline": {**summary_baseline, "near_miss": near_miss_baseline},
        "split": {**summary_split, "near_miss": near_miss_split},
        "delta_pq": round(summary_split["pq"] - summary_baseline["pq"], 6),
        "split_events": split_events,
    }
    (args.out / "blob_split_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
