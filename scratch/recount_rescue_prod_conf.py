"""P2指摘の修正版 prod-conf 救出数を再計算する (probe の検証用 recount)。

`run_tile_infer_probe.py` 初回実行では、base-blind GT の prod 救出判定が
「best-IoU タイル提案自身が conf>=0.26」のみを見ており、別提案が
IoU>=0.5 かつ conf>=0.26 を満たすケースを数え落とす (Codex P2)。

本スクリプトは probe と同一条件 (base@0.05 全画面・tile@0.05 2x2
native) で GT 単位の再計算を行う。tile 提案は pipeline cache
(`output/tile_pipeline_oof/cache`, tile_detect_conf=0.05 で同一検出)
から読み、base のみ再検出して計算時間を半減させる。

使い方:
    python scratch/recount_rescue_prod_conf.py [--fold 2] [--device cpu]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tile_infer_probe import (  # noqa: E402
    FOLDS_CSV, PREPROC, PROD_CONF, PROBE_CONF, RUNS, TRAIN_JSON,
    YOLO_IMGSZ, iou_matrix_per_gt, predict_masks, to_rle)
from run_tile_pipeline_oof import unpack_masks  # noqa: E402

CACHE = Path("output/tile_pipeline_oof/cache")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

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
    print({"fold": args.fold, "観測": len(observations), "weight": str(weight)},
          flush=True)
    model = YOLO(str(weight))

    totals = dict(blind=0, rescued05=0, prod_old=0, prod_new=0)
    started = time.perf_counter()
    for index, name in enumerate(observations, 1):
        cache_file = CACHE / f"{Path(name).stem}_fold{args.fold}.npz"
        saved = np.load(cache_file)
        assert float(saved["tile_detect_conf"]) == PROBE_CONF, \
            f"{name}: cache tile_detect_conf != {PROBE_CONF}"
        tile_masks = unpack_masks(saved["tile_packed"], int(saved["tile_n"]))
        tile_confs = saved["tile_confs"].astype(np.float32)
        image = cv2.imread(str(PREPROC / name), cv2.IMREAD_COLOR)
        base_masks, _ = predict_masks(
            model, image, YOLO_IMGSZ, PROBE_CONF, args.device)

        for image_id in file_to_ids[name]:
            info = images[image_id]
            gt_rles = [to_rle(a, int(info["height"]), int(info["width"]))
                       for a in annotations[image_id]]
            base_matrix = iou_matrix_per_gt(gt_rles, base_masks) \
                if gt_rles and base_masks else np.zeros((len(gt_rles), 0))
            tile_matrix = iou_matrix_per_gt(gt_rles, tile_masks) \
                if gt_rles and tile_masks else np.zeros((len(gt_rles), 0))
            base_best = (base_matrix.max(axis=1) if base_matrix.size
                         else np.zeros(len(gt_rles)))
            tile_best = (tile_matrix.max(axis=1) if tile_matrix.size
                         else np.zeros(len(gt_rles)))
            base_blind = base_best < 0.1
            if tile_matrix.size:
                argmax = tile_matrix.argmax(axis=1)
                best_conf = tile_confs[argmax]
                prod_any = ((tile_matrix >= 0.5)
                            & (tile_confs >= PROD_CONF)[None, :]).any(axis=1)
            else:
                best_conf = np.zeros(len(gt_rles), dtype=np.float32)
                prod_any = np.zeros(len(gt_rles), dtype=bool)
            totals["blind"] += int(base_blind.sum())
            totals["rescued05"] += int((base_blind & (tile_best >= 0.5)).sum())
            totals["prod_old"] += int(
                (base_blind & (tile_best >= 0.5) & (best_conf >= PROD_CONF)).sum())
            totals["prod_new"] += int((base_blind & prod_any).sum())
        if index % 25 == 0 or index == len(observations):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(observations)}  {elapsed:.0f}s", flush=True)

    out = {"fold": args.fold, "観測": len(observations),
           "base_blind": totals["blind"],
           "rescued_IoU0.5": totals["rescued05"],
           "rescued_05_prod_old": totals["prod_old"],
           "rescued_05_prod_corrected": totals["prod_new"],
           "wall_s": round(time.perf_counter() - started, 1)}
    print(json.dumps(out, indent=2))
    Path("output/tile_infer_probe").mkdir(parents=True, exist_ok=True)
    Path(f"output/tile_infer_probe/prod_rescue_recount_fold{args.fold}.json"
         ).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
