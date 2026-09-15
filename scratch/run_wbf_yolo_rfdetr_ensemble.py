"""YOLO(V35確定)とRF-DETR-Seg(V45、768px・epoch40・conf0.40)をWBFで統合し、
V35確定後処理(refiner shrink@0.45 + drop_contained_then_area_asc@0.90)込みで
fold0 OOF PQをV35基準(0.4393)と比較する。

## 狙い

RF-DETR単体はraw PQ 0.3410で生YOLO基準0.3977に届かず不採用済み(V44/V45)。
ただしNMSベースのYOLOとHungarian matchingベースのRF-DETRは誤り方の相関が
低い可能性がある(multi-seed YOLOアンサンブルが失敗したのは同一アーキテクチャ
由来の誤りが強く相関していたためという仮説)。RF-DETRが弱くても、YOLOが
見落とす個体を拾えるなら、V35確定パイプラインへ足すことでPQが伸びる可能性が
まだ残っている。

## 設計判断: confidenceのスケールを揃えない(重み1.0の単純平均)

YOLOのconfidence分布とRF-DETRのconfidence分布は最適閾値からして違う
(YOLO conf 0.26、RF-DETR conf 0.40、RF-DETRの中央値はわずか0.0995)。
スケールを揃えずにWBFの加重平均へ渡すと、値の大きいYOLO側にマスクが
引っ張られてRF-DETRの寄与が実質ゼロになる。今回はまず単純化のため、
各ソースを自身の最適閾値でフィルタした後は重み1.0固定(クラスタ内単純平均)
として統合する。有望なら次のステップでスケール正規化を検討する。

使い方:
    python scratch/run_wbf_yolo_rfdetr_ensemble.py [--limit N]
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

YOLO_CONF = 0.26
RFDETR_CONF = 0.40
REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45
FOLD = 0
BASELINE_FOLD0_PQ = 0.4393  # V35確定、YOLO単独
RAW_YOLO_FOLD0_PQ = 0.3977
RAW_RFDETR_FOLD0_PQ = 0.3410
WBF_IOU = 0.5
WBF_VOTE = 0.6
DROP_THRESHOLD = 0.90  # V35確定値(0.80はfold0選択バイアスと判明済み、V42参照)


def filter_proposals(masks, confidences, conf_threshold):
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= conf_threshold and m.sum() >= MIN_AREA]
    return [masks[i] for i in keep]


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
    parser.add_argument(
        "--rfdetr-cache", type=Path,
        default=Path("G:/マイドライブ/solar_filament_v35/v45_rfdetr_oof_cache_fold0"),
    )
    parser.add_argument("--out", type=Path, default=Path("output/ensemble"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.rfdetr_cache.exists():
        print(f"RF-DETRキャッシュが見つからない: {args.rfdetr_cache}")
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
        and (args.rfdetr_cache / f"{Path(n).stem}.npz").exists()
    ]
    if args.limit:
        targets = targets[:args.limit]
    print(f"観測数: {len(targets)}", flush=True)

    checkpoint = torch.load(
        ARTIFACTS / "mediseg_refiner_fold0_resume.pt", map_location="cpu", weights_only=False)
    refiner = MediSegUNet(3, 1, False).eval()
    refiner.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    meter = PanopticQuality()
    both_source_clusters = 0
    total_clusters = 0
    started = time.perf_counter()

    for position, name in enumerate(targets, 1):
        variants = [gt_rles[i] for i in file_to_ids[name]]
        image = imread_unicode(PREPROC / name)
        if image is None:
            continue

        yolo_masks, yolo_conf = load_yolo_cache(OOF_CACHE / f"yolo_fold_{FOLD}" / f"{Path(name).stem}.npz")
        rfdetr_masks, rfdetr_conf = load_yolo_cache(args.rfdetr_cache / f"{Path(name).stem}.npz")
        yolo_proposals = filter_proposals(yolo_masks, yolo_conf, YOLO_CONF)
        rfdetr_proposals = filter_proposals(rfdetr_masks, rfdetr_conf, RFDETR_CONF)

        combined_masks = yolo_proposals + rfdetr_proposals
        combined_scores = np.ones(len(combined_masks), dtype=np.float32)  # 重み1.0固定(スケール不一致を避ける)
        source_ids = [0] * len(yolo_proposals) + [1] * len(rfdetr_proposals)
        fused_masks, fused_scores, fused_source_counts = weighted_mask_fusion(
            combined_masks, combined_scores, source_ids,
            iou_threshold=WBF_IOU, vote_threshold=WBF_VOTE)

        if fused_masks:
            boxes, probabilities = refine_batch(refiner, image, fused_masks)
            refined = [build_variant(fused_masks[i], boxes[i], probabilities[i],
                                     REFINER_METHOD, REFINER_THRESHOLD)
                       for i in range(len(fused_masks))]
            final = apply_ownership_local(refined, DROP_THRESHOLD)
        else:
            final = []
        rles = masks_to_rles(final)

        for truth in variants:
            matrix = iou_matrix(truth, rles)
            meter.add_matrix(matrix, len(truth), len(rles))

        if position % 25 == 0 or position == len(targets):
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                  f"pq={meter.score:.4f}", flush=True)

    summary = meter.summary()
    report = {
        "scope": f"{len(targets)}観測、fold0のみ。YOLO(conf{YOLO_CONF})+RFDETR(conf{RFDETR_CONF})を"
                 f"WBF(iou={WBF_IOU},vote={WBF_VOTE},重み1.0固定)で統合→refiner shrink@0.45→"
                 f"drop_contained_then_area_asc@{DROP_THRESHOLD}",
        **summary,
        "baseline_yolo_only_fold0_pq": BASELINE_FOLD0_PQ,
        "delta_vs_yolo_only": round(summary["pq"] - BASELINE_FOLD0_PQ, 6),
        "beats_yolo_only": bool(summary["pq"] > BASELINE_FOLD0_PQ),
        "raw_yolo_fold0_pq": RAW_YOLO_FOLD0_PQ,
        "raw_rfdetr_fold0_pq": RAW_RFDETR_FOLD0_PQ,
    }
    (args.out / "wbf_yolo_rfdetr_ensemble_fold0.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
