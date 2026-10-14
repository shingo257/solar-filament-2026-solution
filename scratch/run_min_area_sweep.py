"""prod の min_area (ownership 内の排他面積フロア) を fold2 OOF で一括 sweep する。

§6.27 の dariushafshar カーネルが「人間レベルでは <300px 除去は wash」と
示した一方、モデル精度では別途 OOF 評価が必要と留保した点を潰す。
prod は `apply_ownership(refined, scores)` に min_area=20 の1箇所のみで
適用される (raw 提案に面積 pre-filter は無い) ので、検出+refiner は
prod 同一 operating point (YOLO@0.26, fold2 EdgeAtt, thr 0.45) で1回だけ
走り、min_area だけ変えて ownership+PQ を再評価する。

leak 管理は run_tile_pipeline_oof と同じ: fold2 held-out 観測に
yolo_fold_2 + mediseg_edgeattnet_refiner_fold2_resume.pt。
fold2 上の探索なので採用前に fold0/1 の独立検証が必要 (同じ注意書き)。

使い方:
    python scratch/run_min_area_sweep.py [--fold 2] [--limit N] [--device cpu]
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
from run_tile_infer_probe import predict_masks, to_rle  # noqa: E402
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
DETECT_CONF = 0.26          # prod 同一: YOLO predict conf
REFINER_THRESHOLD = 0.45
DROP_THRESHOLD = 0.90
MIN_AREAS = [0, 5, 10, 20, 50, 100, 300]
PROD_MIN_AREA = 20          # sanity gate: この arm は 0.4442 を再現するはず


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", type=Path,
                        default=Path("output/min_area_sweep/cache"))
    parser.add_argument("--out", type=Path, default=Path("output/min_area_sweep"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    weight = RUNS / f"yolo_fold_{args.fold}" / "weights" / "best.pt"
    assert weight.exists(), f"YOLO 重みなし: {weight}"
    model = YOLO(str(weight))
    refiner = MediSegEdgeAttUNet(3, 1, False).to(args.device).eval()
    ckpt = ARTIFACTS / f"mediseg_edgeattnet_refiner_fold{args.fold}_resume.pt"
    assert ckpt.exists(), f"refiner 重みなし: {ckpt}"
    checkpoint = torch.load(ckpt, map_location="cpu")
    refiner.load_state_dict(
        checkpoint.get("best_model") or checkpoint["model"], strict=True)

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
    observations = sorted(n for n, f in fold_of_file.items()
                          if f == args.fold and n in file_to_ids)
    if args.limit:
        observations = observations[:args.limit]
    print({"fold": args.fold, "観測": len(observations)})

    pq = {a: PanopticQuality() for a in MIN_AREAS}
    counts = {a: 0 for a in MIN_AREAS}
    rows = []
    started = time.perf_counter()
    for index, name in enumerate(observations, 1):
        stem = Path(name).stem
        cache_file = args.cache / f"{stem}_fold{args.fold}.npz"
        if cache_file.exists():
            saved = np.load(cache_file)
            proposals = unpack_masks(saved["prop_packed"], int(saved["prop_n"]))
            confs = saved["confs"].astype(np.float32)
            refined = unpack_masks(saved["ref_packed"], int(saved["ref_n"]))
        else:
            image = cv2.imdecode(
                np.fromfile(str(PREPROC / name), dtype=np.uint8),
                cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(PREPROC / name)
            proposals, confs = predict_masks(
                model, image, YOLO_IMGSZ, DETECT_CONF, args.device)
            refined = refine_proposals(refiner, image, proposals)
            prop_packed, prop_n = pack_masks(proposals)
            ref_packed, ref_n = pack_masks(refined)
            np.savez_compressed(
                cache_file, prop_packed=prop_packed, prop_n=prop_n,
                ref_packed=ref_packed, ref_n=ref_n, confs=confs,
                detect_conf=DETECT_CONF)

        arm_final = {}
        for area in MIN_AREAS:
            final = apply_ownership(
                refined, confs,
                rule="drop_contained_then_area_asc",
                containment_threshold=DROP_THRESHOLD, min_area=area)
            arm_final[area] = masks_to_rles(final)
            counts[area] += len(final)

        for image_id in file_to_ids[name]:
            info = images[image_id]
            gt = [to_rle(a, int(info["height"]), int(info["width"]))
                  for a in annotations[image_id]]
            for area, rles in arm_final.items():
                pq[area].add_matrix(iou_matrix(gt, rles), len(gt), len(rles))
        rows.append({"file_name": name, "proposals": len(proposals),
                     **{f"final_ma{a}": len(arm_final[a]) for a in MIN_AREAS}})
        if index % 10 == 0 or index == len(observations):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(observations)}  {elapsed:.0f}s  "
                  f"残見積 {elapsed / index * (len(observations) - index):.0f}s",
                  flush=True)

    pd.DataFrame(rows).to_csv(args.out / f"per_obs_fold{args.fold}.csv",
                              index=False)
    prod_pq = float(pq[PROD_MIN_AREA].score)
    summary = {"fold": args.fold, "観測": len(observations),
               "prod_min_area": PROD_MIN_AREA, "prod_pq": prod_pq,
               "min_areas": {},
               "wall_s": round(time.perf_counter() - started, 1)}
    for area in MIN_AREAS:
        arm_pq = float(pq[area].score)
        summary["min_areas"][str(area)] = {
            "pq": arm_pq, "delta_vs_prod": arm_pq - prod_pq,
            **pq[area].summary()}
    (args.out / f"summary_fold{args.fold}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
