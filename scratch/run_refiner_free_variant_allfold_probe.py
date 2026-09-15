"""free variant(refinerの確率マップをそのまま閾値化、候補との交差制約なし)を
fold1・fold2でも独立に検証する — 監査で発覚したギャップの解消。

## 監査で発覚した問題(2026-08-29)

`run_refiner_free_variant_probe.py`・`run_refiner_free_dropcontained_probe.py`は
いずれも`FOLD = 0`にハードコードされており、free@0.86(fold0 OOF PQ 0.4361、
基準比Δ-0.0032)という「現時点の最良非ベースライン構成」は**一度もfold1・fold2で
検証されていなかった**。このプロジェクトでは過去に同種の問題(WBFの
drop_contained@0.80がfold0で基準突破(Δ+0.0007)したが、fold1(Δ-0.0027)・
fold2(Δ-0.0062)で悪化し不採用になった事例、2026-08-22)が実際に起きている。
free@0.86も同じ「fold0選択バイアス」の可能性を検証する必要がある。

## このスクリプトでやること

`--fold`引数でfold0/1/2を切り替え、各foldの専用refinerチェックポイント
(`mediseg_refiner_fold{N}_resume.pt`)・専用OOFキャッシュ(`yolo_fold_{N}`)を
使って、free variantの閾値スイープ(fold0の元スイープと同じ粗密構成)を
独立に行う。fold0の0.86という値に引きずられないよう、広い範囲を再スイープする。

使い方:
    python scratch/run_refiner_free_variant_allfold_probe.py --fold 1
    python scratch/run_refiner_free_variant_allfold_probe.py --fold 2
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
DROP_THRESHOLD = 0.90

# pipeline_stack.json (2026-08-16確定、全fold OOF) のfold別baseline PQ (shrink@0.45確定構成)
BASELINE_PQ_BY_FOLD = {0: 0.439288, 1: 0.441499, 2: 0.447052}


def filter_proposals(masks, confidences):
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CONF and m.sum() >= MIN_AREA]
    return [masks[i] for i in keep]


def free_variant(proposal, box, probability, threshold):
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
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("output/refiner_free"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    fold = args.fold
    baseline_pq = BASELINE_PQ_BY_FOLD[fold]

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
        if fold_of_file[n] == fold
        and (OOF_CACHE / f"yolo_fold_{fold}" / f"{Path(n).stem}.npz").exists()
    ]
    if args.limit:
        targets = targets[:args.limit]
    print(f"fold{fold} 観測数: {len(targets)}  baseline_pq: {baseline_pq}", flush=True)

    checkpoint = torch.load(
        ARTIFACTS / f"mediseg_refiner_fold{fold}_resume.pt", map_location="cpu", weights_only=False)
    refiner = MediSegUNet(3, 1, False).eval()
    refiner.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    # fold0の0.86に引きずられないよう、広い範囲を独立にスイープする
    thresholds = [0.30, 0.50, 0.65, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.95]
    meters = {t: PanopticQuality() for t in thresholds}
    started = time.perf_counter()

    for position, name in enumerate(targets, 1):
        variants = [gt_rles[i] for i in file_to_ids[name]]
        image = imread_unicode(PREPROC / name)
        if image is None:
            continue

        masks, confidences = load_yolo_cache(OOF_CACHE / f"yolo_fold_{fold}" / f"{Path(name).stem}.npz")
        proposals = filter_proposals(masks, confidences)
        if not proposals:
            for threshold in thresholds:
                for truth in variants:
                    meters[threshold].add_matrix(np.zeros((len(truth), 0)), len(truth), 0)
            continue

        boxes, probabilities = refine_batch(refiner, image, proposals)

        for threshold in thresholds:
            refined = [free_variant(proposals[i], boxes[i], probabilities[i], threshold)
                       for i in range(len(proposals))]
            final = apply_ownership_local(refined, DROP_THRESHOLD)
            rles = masks_to_rles(final)
            for truth in variants:
                matrix = iou_matrix(truth, rles)
                meters[threshold].add_matrix(matrix, len(truth), len(rles))

        if position % 25 == 0 or position == len(targets):
            scores = {t: round(m.score, 4) for t, m in meters.items()}
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  {scores}", flush=True)

    best_threshold = max(thresholds, key=lambda t: meters[t].score)
    report = {
        "scope": f"{len(targets)}観測、fold{fold}のみ。free variant閾値スイープ(fold0選択バイアス検証)",
        "fold": fold,
        "baseline_shrink045_pq": baseline_pq,
        "best_threshold": best_threshold,
        "best_pq": meters[best_threshold].score,
        "delta_vs_baseline_at_best": round(meters[best_threshold].score - baseline_pq, 6),
        "pq_at_fold0_selected_086": meters[0.86].score,
        "delta_vs_baseline_at_086": round(meters[0.86].score - baseline_pq, 6),
        "results": {
            str(t): {**m.summary(), "delta_vs_baseline": round(m.summary()["pq"] - baseline_pq, 6),
                     "beats_baseline": bool(m.summary()["pq"] > baseline_pq)}
            for t, m in meters.items()
        },
    }
    out_path = args.out / f"refiner_free_variant_fold{fold}_allfold_check.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
    print(f"所要 {time.perf_counter() - started:.0f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
