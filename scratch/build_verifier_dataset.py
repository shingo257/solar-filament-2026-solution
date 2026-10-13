"""深層 verifier 学習用のクロップデータセットを OOF 提案キャッシュから構築する。

背景 (RESEARCH_LOG §2.21 / v64): perfect verifier oracle は +0.046 の最大天井
だが、v64 の静的特徴 selector は「最良点 = selector不使用」に沈んだ。
手作り特徴ではなく画像そのものを見る CNN verifier は未検証。

本スクリプトは提案ごとの学習サンプルを作る:
  - 提案源: prediction_cache/oof/yolo_fold_{k} (conf>=~0.08 の OOF 提案)
  - 入力: square_box(proposal) のクロップ RGB + 提案マスクを4ch目に (128x128)
  - ラベル: その観測の全 GT (annotator x image の全 record) との
            best IoU > 0.5 か = 公式PQのTP条件 (境界値は陽性にしない)。
            GT は annotator 間で union — 公式PQは annotator×image の
            group 毎に採点し、どれか1つの group で TP になる提案は
            その group の TP。verifier の予測対象「提案がTPになり
            得るか」には union を採用

fold 分割 (v64 と同型):
  train = fold0 / 操作点選択 = fold1 / 凍結評価 = fold2
(各 fold の提案は「その fold を学習から除外した検出器」由来なので、
 fold0 で学習した verifier は fold0 検出器のバイアスを見ない)

使い方:
    python scratch/build_verifier_dataset.py --folds 0 1 2 --out output/verifier_v1
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_refiner_pq_allfold import square_box  # noqa: E402
from run_tile_infer_probe import TRAIN_JSON, to_rle  # noqa: E402

DRIVE = Path("G:/マイドライブ")
V30 = DRIVE / "solar_filament_v30"
FOLDS_CSV = V30 / "artifacts/grouped_folds.csv"
PREPROC = V30 / "preprocessed_v30_q96"
OOF = DRIVE / "solar_filament_v34/prediction_cache/oof"

CROP = 128                    # verifier 入力解像度 (320 の refiner より軽量)
POS_IOU = 0.5                 # 公式PQのTP条件
MIN_AREA = 20


def imread_unicode(path):
    """cv2.imread は非ASCIIパス(マイドライブ)を読めないことがある (v68教訓)。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_proposals(npz_path):
    saved = np.load(npz_path)
    packed, (h, w), confs = saved["packed"], saved["shape"], saved["confidences"]
    n = packed.shape[0]
    if n == 0:
        return [], confs.astype(np.float32)
    flat = np.unpackbits(packed.reshape(n, -1), axis=1)[:, : h * w]
    return [row.reshape(h, w).astype(bool) for row in flat], confs.astype(np.float32)


def make_crop(image_bgr, prop):
    """BGR 画像 + 提案マスク → (CROP,CROP,4) の RGB+mask uint8。"""
    x0, y0, x1, y1 = square_box(prop)
    crop_img = cv2.resize(image_bgr[y0:y1, x0:x1], (CROP, CROP),
                          interpolation=cv2.INTER_LINEAR)
    crop_img = crop_img[:, :, ::-1]                      # OpenCV BGR → RGB
    crop_mask = cv2.resize(prop[y0:y1, x0:x1].astype(np.uint8), (CROP, CROP),
                           interpolation=cv2.INTER_NEAREST)
    return np.concatenate([crop_img, crop_mask[..., None]], axis=-1)  # (C,C,4)


def positive_label(best_iou):
    """公式PQのTP条件 (IoU > 0.5、境界値は陽性にしない)。"""
    return int(best_iou > POS_IOU)


def best_ious(prop, gt_rles):
    if not gt_rles:
        return 0.0
    rle = mask_util.encode(np.asfortranarray(prop.astype(np.uint8)[:, :, None]))[0]
    return float(mask_util.iou([rle], gt_rles, [0] * len(gt_rles))[0].max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("output/verifier_v1"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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

    for fold in args.folds:
        observations = sorted(
            n for n in file_to_ids if fold_of_file.get(n) == fold)
        if args.limit:
            observations = observations[:args.limit]
        cache_dir = OOF / f"yolo_fold_{fold}"
        crops, rows = [], []
        started = time.perf_counter()
        for index, name in enumerate(observations, 1):
            npz_path = cache_dir / f"{Path(name).stem}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(npz_path)
            proposals, confs = load_proposals(npz_path)
            image = imread_unicode(PREPROC / name)
            if image is None:
                raise FileNotFoundError(PREPROC / name)
            gt_rles = []
            for image_id in file_to_ids[name]:
                info = images[image_id]
                gt_rles += [to_rle(a, int(info["height"]), int(info["width"]))
                            for a in annotations[image_id]]
            for pi, (prop, conf) in enumerate(zip(proposals, confs)):
                if prop.sum() < MIN_AREA:
                    continue
                crops.append(make_crop(image, prop))
                rows.append({"fold": fold, "file_name": name, "prop_idx": pi,
                             "conf": float(conf),
                             "best_iou": best_ious(prop, gt_rles)})
            if index % 25 == 0 or index == len(observations):
                elapsed = time.perf_counter() - started
                print(f"fold{fold} {index}/{len(observations)}  {elapsed:.0f}s  "
                      f"crops={len(crops)}", flush=True)
        crop_arr = np.stack(crops).astype(np.uint8)          # (N,128,128,4)
        np.save(args.out / f"crops_fold{fold}.npy", crop_arr)
        frame = pd.DataFrame(rows)
        frame["label"] = frame["best_iou"].map(positive_label)
        frame.to_csv(args.out / f"labels_fold{fold}.csv", index=False)
        summary = {"fold": fold, "observations": len(observations),
                   "crops": int(len(frame)),
                   "pos": int(frame["label"].sum()),
                   "pos_rate": round(float(frame["label"].mean()), 4),
                   "conf_auc": None}
        # conf 単独の AUC を基準として記録
        from sklearn.metrics import roc_auc_score
        summary["conf_auc"] = round(float(
            roc_auc_score(frame["label"], frame["conf"])), 4)
        (args.out / f"dataset_fold{fold}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
