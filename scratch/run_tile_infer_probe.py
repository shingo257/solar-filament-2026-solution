"""タイル推論(SAHI相当)で detector-blind FN が救えるかを検出レベルで測る。

背景: FN1071 の68%は提案不在(blind+weak、v62解剖)。全画面 2048→1280 への
0.625 ダウンサンプルで細いフィラメントが消えている可能性が残る唯一の
検出系未検証レバーとして、全画面 @1280 vs 2x2 タイル(1152px, overlap256,
native 解像度)の提案を blind-FN 救出率で比較する。

リーク管理: fold2 の held-out 観測には yolo_fold_2 (fold0+1 で学習) を適用。
refiner/ownership/PQ は回さず、提案の存在性(IoU)のみを見る軽量プローブ。
救出が実質ゼロならパイプライン評価には進まない(早期棄却)。

使い方:
    python scratch/run_tile_infer_probe.py [--fold 2] [--limit N] [--device cpu]
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
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DRIVE = Path("G:/マイドライブ")
V30 = DRIVE / "solar_filament_v30"
TRAIN_JSON = V30 / "competition_data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FOLDS_CSV = V30 / "artifacts/grouped_folds.csv"
PREPROC = V30 / "preprocessed_v30_q96"
RUNS = V30 / "runs"

IMG = 2048
YOLO_IMGSZ = 1280            # prod 標準
PREDICT_IOU = 0.55
PROBE_CONF = 0.05            # 提案「存在」判定フロア (v64 と同じ)
PROD_CONF = 0.26             # prod 提出で実際に残るフロア
TILE = 1152
STRIDE = 896                 # 2x2, overlap 256


def to_rle(annotation: dict, height: int, width: int) -> dict:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_util.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def predict_masks(model, image, imgsz, conf, device):
    result = model.predict(
        image, imgsz=imgsz, conf=conf, iou=PREDICT_IOU, agnostic_nms=True,
        retina_masks=True, device=device, verbose=False)[0]
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = (
        result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        if result.boxes is not None else np.zeros(0, np.float32))
    return output, confidences


def tiled_predict(model, image, conf, device):
    """TILE x TILE を STRIDE 刻みで切り出し、各タイルを native 解像度で推論して
    全画面キャンバスへ貼り戻す。NMS は掛けず全提案を保持する。"""
    height, width = image.shape[:2]
    masks, confidences = [], []
    for y0 in range(0, height - TILE + 1, STRIDE):
        for x0 in range(0, width - TILE + 1, STRIDE):
            tile = image[y0:y0 + TILE, x0:x0 + TILE]
            tile_masks, tile_confs = predict_masks(
                model, tile, TILE, conf, device)
            for mask, score in zip(tile_masks, tile_confs):
                canvas = np.zeros((height, width), dtype=bool)
                canvas[y0:y0 + TILE, x0:x0 + TILE] = mask
                masks.append(canvas)
                confidences.append(score)
    return masks, np.asarray(confidences, dtype=np.float32)


def best_iou_per_gt(gt_rles, pred_masks):
    """GT ごとの best proposal IoU。"""
    if not gt_rles or not pred_masks:
        return np.zeros(len(gt_rles), dtype=np.float64)
    matrix = iou_matrix_per_gt(gt_rles, pred_masks)
    if not matrix.size:
        return np.zeros(len(gt_rles), dtype=np.float64)
    return matrix.max(axis=1)


def iou_matrix_per_gt(gt_rles, pred_masks):
    """(n_gt, n_pred) の IoU 行列。"""
    pred_rles = mask_util.encode(
        np.asfortranarray(np.stack(pred_masks, axis=-1).astype(np.uint8)))
    matrix = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    return np.asarray(matrix, dtype=np.float64).T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=Path("output/tile_infer_probe"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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
    print({"fold": args.fold, "観測": len(observations), "weight": str(weight)})

    model = YOLO(str(weight))
    stats = []
    started = time.perf_counter()
    for index, name in enumerate(observations, 1):
        image = cv2.imread(str(PREPROC / name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(PREPROC / name)
        base_masks, base_confs = predict_masks(
            model, image, YOLO_IMGSZ, PROBE_CONF, args.device)
        tile_masks, tile_confs = tiled_predict(
            model, image, PROBE_CONF, args.device)

        for image_id in file_to_ids[name]:
            info = images[image_id]
            gt_rles = [to_rle(a, int(info["height"]), int(info["width"]))
                       for a in annotations[image_id]]
            base_best = best_iou_per_gt(gt_rles, base_masks)
            tile_matrix = iou_matrix_per_gt(gt_rles, tile_masks)
            tile_best = (tile_matrix.max(axis=1) if tile_matrix.size
                         else np.zeros(len(gt_rles)))
            base_blind = base_best < 0.1
            rescued05 = base_blind & (tile_best >= 0.5)
            # prod 救出 = IoU>=0.5 かつ conf>=0.26 を「同時に」満たす提案の存在
            # (best-IoU 提案の conf だけを見ると別提案の高conf適合を数え落とす)
            if tile_matrix.size:
                prod_hit = ((tile_matrix >= 0.5)
                            & (tile_confs >= PROD_CONF)[None, :]).any(axis=1)
            else:
                prod_hit = np.zeros(len(gt_rles), dtype=bool)
            stats.append({
                "file_name": name, "image_id": image_id, "n_gt": len(gt_rles),
                "base_blind": int(base_blind.sum()),
                "tile_blind": int((tile_best < 0.1).sum()),
                "rescued_01": int((base_blind & (tile_best >= 0.1)).sum()),
                "rescued_03": int((base_blind & (tile_best >= 0.3)).sum()),
                "rescued_05": int(rescued05.sum()),
                # prod conf (0.26) に届く救出だけが提出スコアに乗る
                "rescued_05_prod": int((base_blind & prod_hit).sum()),
                "base_props": len(base_masks),
                "tile_props": len(tile_masks),
                "base_props_prod": int((base_confs >= PROD_CONF).sum()),
                "tile_props_prod": int((tile_confs >= PROD_CONF).sum()),
            })
        if index % 10 == 0 or index == len(observations):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(observations)}  {elapsed:.0f}s  "
                  f"残見積 {elapsed / index * (len(observations) - index):.0f}s",
                  flush=True)

    frame = pd.DataFrame(stats)
    frame.to_csv(args.out / f"tile_probe_fold{args.fold}.csv", index=False)
    summary = {
        "fold": args.fold, "records": int(len(frame)),
        "gt_total": int(frame["n_gt"].sum()),
        "blind_base": int(frame["base_blind"].sum()),
        "blind_tile": int(frame["tile_blind"].sum()),
        "rescued_IoU0.1": int(frame["rescued_01"].sum()),
        "rescued_IoU0.3": int(frame["rescued_03"].sum()),
        "rescued_IoU0.5": int(frame["rescued_05"].sum()),
        "rescued_IoU0.5_prod_conf": int(frame["rescued_05_prod"].sum()),
        "props_base_mean": round(float(frame["base_props"].mean()), 2),
        "props_tile_mean": round(float(frame["tile_props"].mean()), 2),
        "props_prod_base_mean": round(float(frame["base_props_prod"].mean()), 2),
        "props_prod_tile_mean": round(float(frame["tile_props_prod"].mean()), 2),
        "wall_s": round(time.perf_counter() - started, 1),
    }
    (args.out / f"tile_probe_fold{args.fold}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
