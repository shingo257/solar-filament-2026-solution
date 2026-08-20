"""V35確定fold0(seed3407)とV42(seed8407)をWeighted Mask Fusion(WBF)で統合する。

単純連結アンサンブル(V42、改訂9 §27.5)はFP+190がTP+53を上回り不合格だった。
それぞれのseed固有の誤検出が drop_contained(IoU≥0.90の重複除去のみ)では
吸収しきれなかったため。WBF(`src/mask_wbf.py`)はIoU≥0.5程度で緩くクラスタ化し、
confidence加重平均でmaskを統合する。2seedが同じ個体を検出した場合は
なめらかに1個へ統合され、片方だけの誤検出は単独クラスタのまま残るので、
単純連結よりFP抑制効果があるという仮説を検証する。

WBFは refiner の**前**(生YOLO proposal同士)で行う。統合後のmaskをrefinerに
通し、最後にdrop_contained_then_area_ascを掛ける。

使い方:
    python scratch/run_multiseed_wbf_probe.py [--limit N] [--iou-threshold 0.5] [--threads 14]
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
from scratch.run_pipeline_stack import apply_ownership  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

CONF = 0.26
REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45
FOLD = 0
BASELINE_FOLD0_PQ = 0.4393


def filter_proposals(masks, confidences):
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CONF and m.sum() >= MIN_AREA]
    return [masks[i] for i in keep], (confidences[keep] if keep else np.zeros(0, np.float32))


def refine_and_finalize(refiner, image, proposals, scores):
    if not proposals:
        return []
    boxes, probabilities = refine_batch(refiner, image, proposals)
    refined = [build_variant(proposals[i], boxes[i], probabilities[i],
                             REFINER_METHOD, REFINER_THRESHOLD)
               for i in range(len(proposals))]
    return apply_ownership(refined, scores, "drop_contained_then_area_asc")


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


def drop_contained_indices_local(masks, threshold=0.90):
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


def refine_and_finalize_consensus_first(refiner, image, proposals, scores, consensus_counts):
    """コンセンサス数(降順)→面積(昇順)の順で排他化する。

    通常のdrop_contained_then_area_ascは面積の小さい順に領土を確保させるが、
    WBFで得られる「何seedが一致したか」という情報を使っていない。
    2seed一致のクラスタを単独seedのクラスタより先に確保させることで、
    単独seed(誤検出の疑いが強い)がconsensusクラスタと重なる場合に
    自然に排除されるようにする。
    """
    if not proposals:
        return []
    boxes, probabilities = refine_batch(refiner, image, proposals)
    refined = [build_variant(proposals[i], boxes[i], probabilities[i],
                             REFINER_METHOD, REFINER_THRESHOLD)
               for i in range(len(proposals))]
    areas = np.array([m.sum() for m in refined], dtype=np.float64)
    subset = drop_contained_indices_local(refined)
    order = sorted(subset, key=lambda i: (-consensus_counts[i], areas[i]))
    return greedy_disjoint_local(refined, order)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--require-consensus", action="store_true",
                        help="2seedとも検出した(コンセンサス)クラスタだけを残す")
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
    print(f"観測数: {len(targets)}  iou_threshold={args.iou_threshold}", flush=True)

    checkpoint = torch.load(
        ARTIFACTS / "mediseg_refiner_fold0_resume.pt", map_location="cpu", weights_only=False)
    refiner = MediSegUNet(3, 1, False).eval()
    refiner.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    meter_single = PanopticQuality()
    meter_wbf = PanopticQuality()
    meter_consensus_order = PanopticQuality()
    near_miss_single = 0
    near_miss_wbf = 0
    near_miss_consensus_order = 0
    consensus_counter = {1: 0, 2: 0}
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

        single_final = refine_and_finalize(refiner, image, proposals1, scores1)
        single_rles = masks_to_rles(single_final)

        combined_masks = proposals1 + proposals2
        combined_scores = np.concatenate([scores1, scores2]) if (proposals1 or proposals2) else np.zeros(0, np.float32)
        source_ids = [0] * len(proposals1) + [1] * len(proposals2)
        fused_masks, fused_scores, fused_counts = weighted_mask_fusion(
            combined_masks, combined_scores, source_ids, iou_threshold=args.iou_threshold)
        for count in fused_counts:
            consensus_counter[count] = consensus_counter.get(count, 0) + 1

        # コンセンサス優先順位付け(全クラスタ保持、排他化の順序だけをconsensus_counts降順にする)
        consensus_order_final = refine_and_finalize_consensus_first(
            refiner, image, fused_masks, fused_scores, fused_counts)
        consensus_order_rles = masks_to_rles(consensus_order_final)

        if args.require_consensus:
            keep_idx = [i for i, c in enumerate(fused_counts) if c >= 2]
            fused_masks = [fused_masks[i] for i in keep_idx]
            fused_scores = fused_scores[keep_idx] if len(keep_idx) else np.zeros(0, np.float32)

        wbf_final = refine_and_finalize(refiner, image, fused_masks, fused_scores)
        wbf_rles = masks_to_rles(wbf_final)

        for truth in variants:
            matrix_s = iou_matrix(truth, single_rles)
            meter_single.add_matrix(matrix_s, len(truth), len(single_rles))
            if matrix_s.size:
                near_miss_single += int(((matrix_s > 0.3) & (matrix_s <= 0.5)).any(axis=0).sum())

            matrix_w = iou_matrix(truth, wbf_rles)
            meter_wbf.add_matrix(matrix_w, len(truth), len(wbf_rles))
            if matrix_w.size:
                near_miss_wbf += int(((matrix_w > 0.3) & (matrix_w <= 0.5)).any(axis=0).sum())

            matrix_c = iou_matrix(truth, consensus_order_rles)
            meter_consensus_order.add_matrix(matrix_c, len(truth), len(consensus_order_rles))
            if matrix_c.size:
                near_miss_consensus_order += int(((matrix_c > 0.3) & (matrix_c <= 0.5)).any(axis=0).sum())

        if position % 25 == 0 or position == len(targets):
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                  f"single={meter_single.score:.4f}  wbf={meter_wbf.score:.4f}  "
                  f"consensus_order={meter_consensus_order.score:.4f}  "
                  f"consensus={consensus_counter}", flush=True)

    summary_single = meter_single.summary()
    summary_wbf = meter_wbf.summary()
    summary_consensus_order = meter_consensus_order.summary()
    report = {
        "scope": f"{len(targets)}観測、fold0のみ。V35確定(seed3407) vs WBF(seed3407+8407)",
        "iou_threshold": args.iou_threshold,
        "single_seed3407": {**summary_single, "near_miss": near_miss_single},
        "wbf_seed3407_8407": {**summary_wbf, "near_miss": near_miss_wbf},
        "wbf_consensus_first_order": {**summary_consensus_order, "near_miss": near_miss_consensus_order},
        "consensus_cluster_counts": consensus_counter,
        "baseline_fold0_pq": BASELINE_FOLD0_PQ,
        "delta_wbf_vs_baseline": round(summary_wbf["pq"] - BASELINE_FOLD0_PQ, 6),
        "delta_wbf_vs_single_local": round(summary_wbf["pq"] - summary_single["pq"], 6),
        "delta_consensus_order_vs_baseline": round(summary_consensus_order["pq"] - BASELINE_FOLD0_PQ, 6),
        "delta_consensus_order_vs_wbf": round(summary_consensus_order["pq"] - summary_wbf["pq"], 6),
        "continue_fold12": bool(summary_wbf["pq"] > BASELINE_FOLD0_PQ
                                 or summary_consensus_order["pq"] > BASELINE_FOLD0_PQ),
    }
    suffix = "_consensus2" if args.require_consensus else ""
    (args.out / f"multiseed_wbf_probe_iou{args.iou_threshold:.2f}{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
