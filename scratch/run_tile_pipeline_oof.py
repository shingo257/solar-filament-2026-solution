"""タイル推論込みの prod パイプライン全経路を fold2 OOF で採点する。

`run_tile_infer_probe.py` で detector-blind FN の救出が確認できた (fold2:
blind 143 -> 60, うち IoU>=0.5 が 60 件) ため、本評価で end-to-end の
ΔPQ を測る。両 arm は refiner・ownership・採点まで同一で、検出方式と
conf keep 閾値だけが違う。

プローブが示した通り救出提案の大半は conf<0.26 に分布するため、評価は
keep 閾値グリッドで行う:

- base@0.26 / base@0.195 : prod 参照と conf フロア統制
- tile@0.26              : prod 同一 operating point への drop-in
- tile@0.10 / tile@0.05  : 低 conf 提案を refiner 閾値 0.45 に委ねる arm
- tile@0.10|nms0.3       : タイル跨ぎ断片を強め NMS で抑える変種
- union@0.26 / union@0.10: base ∪ tile の合併候補を cross-source NMS で
                           間引いた本命 arm (tile 救出を base 検出に加える形)

refined mask は提案ごとに決定的なので、全 arm の union (tile nms0.5/0.3
生存者 ∪ union-NMS の tile 由来生存者) に一度だけ refiner を掛け、
各 arm は subset 選択 + ownership + PQ のみで使い回す (検出コスト1回で
グリッド全点を評価)。

リーク管理 (run_refiner_pq_allfold と同じ):
  fold2 の held-out 観測 → yolo_fold_2 + mediseg_edgeattnet_refiner_fold2_resume.pt
  (どちらも fold2 を学習から除外)。all-data refiner は使わない。
採点は annotator x image 基準の公式 PanopticQuality (run_pq_oof_rescore と同じ)。

注意: keep 閾値グリッドは fold2 上の探索であり、どれかが有望でも採用前に
fold0/1 で事前宣言した閾値の独立検証が必要 (単一 fold 選択は overfit)。

使い方:
    python scratch/run_tile_pipeline_oof.py [--fold 2] [--limit N] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_refiner_pq_allfold import refine_crops, square_box  # noqa: E402
from run_tile_infer_probe import (  # noqa: E402
    predict_masks, tiled_predict, to_rle)
from solar_filament.pipeline.ownership import apply_ownership  # noqa: E402
from src.mediseg_models import MediSegEdgeAttUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402

DRIVE = Path("G:/マイドライブ")
V30 = DRIVE / "solar_filament_v30"
TRAIN_JSON = V30 / "competition_data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FOLDS_CSV = V30 / "artifacts/grouped_folds.csv"
PREPROC = V30 / "preprocessed_v30_q96"
RUNS = V30 / "runs"
ARTIFACTS = V30 / "artifacts"

YOLO_IMGSZ = 1280
PREDICT_IOU = 0.55
BASE_DETECT_CONF = 0.195     # prod cell12 の predict_conf (base は prod 同一)
TILE_DETECT_CONF = 0.05      # tile は低フロアで検出し keep 閾値は arm 側で変える
MIN_AREA = 20
REFINER_THRESHOLD = 0.45
DROP_THRESHOLD = 0.90

# (arm 名, 提案ソース, keep conf, NMS IoU or None)
ARMS = [
    ("base@0.26", "base", 0.26, None),
    ("base@0.195", "base", 0.195, None),
    ("tile@0.26", "tile", 0.26, 0.5),
    ("tile@0.10", "tile", 0.10, 0.5),
    ("tile@0.05", "tile", 0.05, 0.5),
    ("tile@0.10|nms0.3", "tile", 0.10, 0.3),
    ("union@0.26", "union", 0.26, 0.5),
    ("union@0.10", "union", 0.10, 0.5),
]


def cross_tile_nms_indices(masks, confidences, iou_thr):
    """conf 降順に見て、既採用提案と mask-IoU>=iou_thr なら間引く。
    戻り値は採用提案の index (元の並び順)。IoU は kept との遅延比較。"""
    order = np.argsort(-confidences)
    kept = []
    for index in order:
        if any(((masks[index] & masks[j]).sum() /
                max((masks[index] | masks[j]).sum(), 1)) >= iou_thr
               for j in kept):
            continue
        kept.append(int(index))
    return sorted(kept)


def pack_masks(masks):
    if not masks:
        return np.zeros((0, 0), np.uint8), 0
    flat = np.stack([m.astype(np.uint8).reshape(-1) for m in masks])
    return np.packbits(flat, axis=1), len(masks)


def unpack_masks(packed, count):
    if count == 0:
        return []
    flat = np.unpackbits(packed, axis=1)[:, :2048 * 2048]
    return [row.reshape(2048, 2048).astype(bool) for row in flat]


def refine_proposals(refiner, image, proposals):
    """提案1つごとに refined mask を返す (prod_pipeline と同一式)。"""
    boxes = [square_box(p) for p in proposals]
    probabilities = refine_crops(refiner, image, boxes) if boxes else []
    refined = []
    for prop, (x0, y0, x1, y1), prob in zip(proposals, boxes, probabilities):
        mask = np.zeros_like(prop, dtype=bool)
        mask[y0:y1, x0:x1] = prob >= REFINER_THRESHOLD
        refined.append(mask & prop)
    return refined


def masks_to_rles(masks):
    if not masks:
        return []
    return mask_util.encode(
        np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))


def iou_matrix(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    return np.asarray(
        mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64).T


def detections_fresh(saved, key, want_conf):
    """キャッシュの検出 conf が要求値と一致するか (prod 等価には厳密一致が要る)。"""
    if key not in saved.files:
        return False
    return abs(float(saved[key]) - want_conf) < 1e-9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cache", type=Path,
                        default=Path("output/tile_pipeline_oof/cache"))
    parser.add_argument("--out", type=Path, default=Path("output/tile_pipeline_oof"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
    annotations: dict = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    file_to_ids: dict = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)

    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    observations = sorted(n for n in file_to_ids if fold_of_file.get(n) == args.fold)
    if args.limit:
        observations = observations[:args.limit]

    weight = RUNS / f"yolo_fold_{args.fold}" / "weights" / "best.pt"
    refiner_path = ARTIFACTS / f"mediseg_edgeattnet_refiner_fold{args.fold}_resume.pt"
    print({"fold": args.fold, "観測": len(observations),
           "yolo": str(weight), "refiner": str(refiner_path),
           "arms": [a[0] for a in ARMS]}, flush=True)

    model = YOLO(str(weight))
    refiner = MediSegEdgeAttUNet(3, 1, False).eval().to(args.device)
    checkpoint = torch.load(refiner_path, map_location="cpu")
    refiner.load_state_dict(
        checkpoint.get("best_model") or checkpoint["model"], strict=True)

    pq = {name: PanopticQuality() for name, *_ in ARMS}
    counts = {name: {"final": 0} for name, *_ in ARMS}
    rows = []
    started = time.perf_counter()
    for index, name in enumerate(observations, 1):
        stem = Path(name).stem
        cache_file = args.cache / f"{stem}_fold{args.fold}.npz"
        base_masks = tile_masks = None
        if cache_file.exists():
            saved = np.load(cache_file)
            if detections_fresh(saved, "base_detect_conf", BASE_DETECT_CONF):
                base_masks = unpack_masks(saved["base_packed"], int(saved["base_n"]))
                base_confs = saved["base_confs"].astype(np.float32)
            if detections_fresh(saved, "tile_detect_conf", TILE_DETECT_CONF):
                tile_masks = unpack_masks(saved["tile_packed"], int(saved["tile_n"]))
                tile_confs = saved["tile_confs"].astype(np.float32)
        image = cv2.imread(str(PREPROC / name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(PREPROC / name)
        if base_masks is None:
            base_masks, base_confs = predict_masks(
                model, image, YOLO_IMGSZ, BASE_DETECT_CONF, args.device)
        if tile_masks is None:
            tile_masks, tile_confs = tiled_predict(
                model, image, TILE_DETECT_CONF, args.device)
        base_packed, base_n = pack_masks(base_masks)
        tile_packed, tile_n = pack_masks(tile_masks)
        np.savez_compressed(
            cache_file, base_packed=base_packed, base_n=base_n,
            tile_packed=tile_packed, tile_n=tile_n,
            base_confs=base_confs, tile_confs=tile_confs,
            base_detect_conf=BASE_DETECT_CONF, tile_detect_conf=TILE_DETECT_CONF)

        # refiner 対象 = 全 arm の union:
        #   tile nms 生存者 ∪ union-NMS (base+tile 合併) の tile 由来生存者
        # (nms0.3 生存者は nms0.5 生存者の部分集合とは限らない: suppressor が
        #  低い閾値で落ちる経路があるため)。
        n_base = len(base_masks)
        combined_masks = base_masks + tile_masks
        combined_confs = np.concatenate([base_confs, tile_confs])
        nms = {thr: cross_tile_nms_indices(tile_masks, tile_confs, thr)
               for thr in {a[3] for a in ARMS if a[1] == "tile"}}
        union_nms = {thr: cross_tile_nms_indices(
            combined_masks, combined_confs, thr)
            for thr in {a[3] for a in ARMS if a[1] == "union"}}
        needed = set()
        for ids in nms.values():
            needed |= set(ids)
        for ids in union_nms.values():
            needed |= {i - n_base for i in ids if i >= n_base}
        needed = sorted(needed)
        base_refined = refine_proposals(refiner, image, base_masks)
        tile_refined = dict(zip(needed, refine_proposals(
            refiner, image, [tile_masks[i] for i in needed])))

        arm_final = {}
        base_refined_of = dict(enumerate(base_refined))
        for arm_name, source, keep_conf, nms_thr in ARMS:
            if source == "base":
                proposal_ids = list(range(n_base))
                props, pconfs = base_masks, base_confs
                refined_of = base_refined_of
            elif source == "tile":
                proposal_ids = nms[nms_thr]
                props = [tile_masks[i] for i in proposal_ids]
                pconfs = tile_confs[proposal_ids]
                refined_of = tile_refined
            else:
                proposal_ids = union_nms[nms_thr]
                props = [combined_masks[i] for i in proposal_ids]
                pconfs = combined_confs[proposal_ids]
                refined_of = {i: (base_refined_of[i] if i < n_base
                                  else tile_refined[i - n_base])
                              for i in proposal_ids}
            keep = [k for k, (p, s) in enumerate(zip(props, pconfs))
                    if s >= keep_conf and p.sum() >= MIN_AREA]
            if keep:
                final = apply_ownership(
                    [refined_of[proposal_ids[k]] for k in keep],
                    pconfs[keep],
                    rule="drop_contained_then_area_asc",
                    containment_threshold=DROP_THRESHOLD, min_area=MIN_AREA)
            else:
                final = []
            arm_final[arm_name] = masks_to_rles(final)
            counts[arm_name]["final"] += len(final)

        for image_id in file_to_ids[name]:
            info = images[image_id]
            gt = [to_rle(a, int(info["height"]), int(info["width"]))
                  for a in annotations[image_id]]
            for arm_name in arm_final:
                rles = arm_final[arm_name]
                pq[arm_name].add_matrix(
                    iou_matrix(gt, rles), len(gt), len(rles))
        rows.append({"file_name": name,
                     **{f"final_{a}": len(arm_final[a]) for a, *_ in ARMS}})
        if index % 10 == 0 or index == len(observations):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(observations)}  {elapsed:.0f}s  "
                  f"残見積 {elapsed / index * (len(observations) - index):.0f}s",
                  flush=True)

    pd.DataFrame(rows).to_csv(args.out / f"pipeline_fold{args.fold}.csv", index=False)
    base_pq = float(pq["base@0.26"].score)
    summary = {"fold": args.fold, "観測": len(observations),
               "base_pq": base_pq, "arms": {},
               "wall_s": round(time.perf_counter() - started, 1)}
    for arm_name, *_ in ARMS:
        arm_pq = float(pq[arm_name].score)
        summary["arms"][arm_name] = {
            "pq": arm_pq, "delta_vs_base": arm_pq - base_pq,
            **pq[arm_name].summary(),
            "final_mean": round(counts[arm_name]["final"] / len(observations), 2)}
    (args.out / f"pipeline_fold{args.fold}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
