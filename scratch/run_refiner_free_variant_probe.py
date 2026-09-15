"""refinerの確率マップを候補マスクとの交差(shrink)に制約せず、そのまま閾値化する
「無制約(free)」variantをfold0で検証する。

⚠️ **2026-08-29監査での訂正: このスクリプトの結果(閾値0.86が最良)はfold0
選択バイアスであり、fold1・fold2では基準を大きく下回ることが判明した
(fold1 Δ-0.0157、fold2 Δ-0.0106、`scratch/run_refiner_free_variant_allfold_probe.py`
で検証)。near-miss境界refiner路線全体が不採用。詳細はREADME.md §2.0.1。**

## 背景

refinerの学習コード(build_v30_prize_colab.py InstanceCropDataset)は、
GTアノテーションのbboxを中心にcontext 1.65倍でクロップし、そのGTマスク自体を
予測するよう学習している。一方、推論時は(GTではなく)YOLO候補のbboxを中心に
同じcontextでクロップする。候補が過小検出(undercoverage)の場合でも、この
1.65倍のcontextにGTの欠けている部分が入っていれば、refinerの確率マップ自体は
正しく広い範囲を予測できている可能性がある。しかし`shrink`後処理は
`refined & proposal`で出力を候補マスクの内側に強制するため、その正しい予測を
必ず捨ててしまう。近接シーンのnear-miss診断で過小検出(45%)と過大検出(42%)が
拮抗していたことから、過小検出側を一切救済できない`shrink`の構造的限界が
near-miss全体の半分を頭打ちにしている可能性が高い。

## このスクリプトでやること

追加学習なしで、確率マップをそのまま閾値化する(候補マスクとの交差なし)
`free`variantを閾値スイープし、`shrink`(既定0.45)と比較する。

使い方:
    python scratch/run_refiner_free_variant_probe.py [--limit N]
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
DROP_THRESHOLD = 0.90


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

    # 画像ごとに一度だけ推論し、複数閾値で使い回す(全観測分をメモリに保持しない設計を維持)
    thresholds = [0.81, 0.82, 0.83, 0.84, 0.86, 0.87, 0.88]
    meters = {t: PanopticQuality() for t in thresholds}
    started = time.perf_counter()

    for position, name in enumerate(targets, 1):
        variants = [gt_rles[i] for i in file_to_ids[name]]
        image = imread_unicode(PREPROC / name)
        if image is None:
            continue

        masks, confidences = load_yolo_cache(OOF_CACHE / f"yolo_fold_{FOLD}" / f"{Path(name).stem}.npz")
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

    report = {
        "scope": f"{len(targets)}観測、fold0のみ。free variant(候補との交差なし)閾値スイープ",
        "baseline_shrink045_fold0_pq": BASELINE_FOLD0_PQ,
        "results": {
            str(t): {**m.summary(), "delta_vs_baseline": round(m.summary()["pq"] - BASELINE_FOLD0_PQ, 6),
                     "beats_baseline": bool(m.summary()["pq"] > BASELINE_FOLD0_PQ)}
            for t, m in meters.items()
        },
    }
    (args.out / "refiner_free_variant_sweep_fine085.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
