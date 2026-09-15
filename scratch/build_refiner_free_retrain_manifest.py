"""refinerをfree-variant向けに再学習するための学習データmanifestを作る(fold0用)。

## 背景

現行refinerはGTのbboxを中心にクロップしGTマスクを予測するよう学習されて
いるが、推論時はYOLO候補のbboxを中心にクロップする。この訓練/推論の不一致が
`shrink`(候補との交差)を使わない`free`variantの性能を頭打ちにしている
可能性が高い(free@0.86でfold0 OOF PQ 0.4361、基準0.4393にΔ-0.0032)。

## このスクリプトでやること

fold0の学習セット(fold1+fold2画像)について、GTのbboxではなく**YOLOの
OOF候補のbbox**を中心にクロップし、その候補に最もよく対応するGTマスクを
ターゲットとするmanifestを作る。リーク防止のため、fold1画像には
`yolo_fold_1`キャッシュ(fold0+fold2で学習したモデル、fold1未見)、
fold2画像には`yolo_fold_2`キャッシュを使う(逆に混ぜない)。

対応するGTが無い候補(IoU<0.10、純粋な誤検出)は全ゼロターゲットとして
一定割合だけ含める(モデルに「ここには何も無い」も学習させる)。

使い方:
    python scratch/build_refiner_free_retrain_manifest.py --target-fold 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    FOLDS_CSV, MIN_AREA, OOF_CACHE, PREPROC, TRAIN_JSON,
    iou_matrix, load_yolo_cache, to_rle,
)
from pycocotools import mask as mask_util  # noqa: E402

CONF = 0.26
CROP_CONTEXT = 1.65
CROP_MIN = 96
SEED = 3407


def square_box_from_mask(mask: np.ndarray, context: float = CROP_CONTEXT, minimum: int = CROP_MIN):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, width, height
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-fold", type=int, default=0,
                         help="このfoldの検証用refinerを学習する(学習データは他のfoldから取る)")
    parser.add_argument("--positive-min-iou", type=float, default=0.10,
                         help="これ以上のIoU一致を正例(GT対応あり)として採用する閾値。"
                              "V46は0.10(緩い)。教師信号の質を絞るなら0.30などへ引き上げる")
    parser.add_argument("--negative-sample-rate", type=float, default=0.15,
                         help="誤検出候補(positive-min-iou未満)を全ゼロターゲットとして"
                              "学習に混ぜる割合。V46は0.15")
    parser.add_argument("--out", type=Path, default=Path("output/refiner_retrain"))
    parser.add_argument("--out-suffix", type=str, default="",
                         help="出力ファイル名に付与するsuffix(例: '_v2')。既存manifestを上書きしない")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {i["id"]: i for i in coco["images"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    file_to_ids = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)

    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))

    train_folds = sorted({0, 1, 2} - {args.target_fold})
    print(f"学習に使うfold: {train_folds}(検証対象fold{args.target_fold}は除外)")

    import os
    temp_root = Path(os.environ.get("TEMP", "")) / "solar_oof_cache"

    samples = []
    negative_pool = []
    for fold in train_folds:
        local_cache_dir = temp_root / f"yolo_fold_{fold}"
        cache_dir = local_cache_dir if local_cache_dir.exists() else OOF_CACHE / f"yolo_fold_{fold}"
        print(f"  fold{fold} cache: {cache_dir}")
        file_names = [
            n for n in file_to_ids
            if fold_of_file.get(n) == fold and (PREPROC / n).exists()
            and (cache_dir / f"{Path(n).stem}.npz").exists()
        ]
        print(f"  fold{fold}: {len(file_names)}観測")

        for file_name in file_names:
            masks, confidences = load_yolo_cache(cache_dir / f"{Path(file_name).stem}.npz")
            keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                    if s >= CONF and m.sum() >= MIN_AREA]
            if not keep:
                continue
            proposals = [masks[i] for i in keep]

            # 全アノテータ分のGTを候補にする(V35のall-labeler規則を踏襲)
            gt_entries = []  # (image_id, ann, rle)
            for image_id in file_to_ids[file_name]:
                info = images[image_id]
                for ann in annotations[image_id]:
                    if ann.get("iscrowd", 0) or ann.get("area", 0) <= 0:
                        continue
                    rle = to_rle(ann, int(info["height"]), int(info["width"]))
                    gt_entries.append((image_id, ann, rle))
            if not gt_entries:
                continue
            gt_rles = [g[2] for g in gt_entries]

            proposal_rles = mask_util.encode(
                np.asfortranarray(np.stack(proposals, axis=-1).astype(np.uint8)))
            matrix = iou_matrix(gt_rles, proposal_rles)  # (n_gt, n_pred)

            for proposal_index, proposal_mask in enumerate(proposals):
                if matrix.shape[0] == 0:
                    best_iou, best_gt_index = 0.0, -1
                else:
                    best_gt_index = int(np.argmax(matrix[:, proposal_index]))
                    best_iou = float(matrix[best_gt_index, proposal_index])

                box = square_box_from_mask(proposal_mask)
                if best_iou >= args.positive_min_iou:
                    image_id, matched_ann, _ = gt_entries[best_gt_index]
                    samples.append({
                        "file_name": file_name,
                        "target_image_id": image_id,
                        "target_ann_id": matched_ann.get("id"),
                        "box": box,
                        "is_negative": False,
                        "matched_iou": round(best_iou, 4),
                    })
                else:
                    negative_pool.append({
                        "file_name": file_name,
                        "box": box,
                        "is_negative": True,
                        "matched_iou": round(best_iou, 4),
                    })

    # 誤検出(全ゼロターゲット)を一定割合だけサンプリングして混ぜる
    n_positive = len(samples)
    n_negative = int(n_positive * args.negative_sample_rate)
    random.shuffle(negative_pool)
    samples.extend(negative_pool[:n_negative])
    random.shuffle(samples)

    print(f"正例(GT対応あり、IoU>={args.positive_min_iou}): {n_positive}件")
    print(f"負例(誤検出、全ゼロターゲット): {min(n_negative, len(negative_pool))}件"
          f" (プール{len(negative_pool)}件から{args.negative_sample_rate:.0%}サンプリング)")
    print(f"合計: {len(samples)}件")

    out_path = args.out / f"refiner_retrain_manifest_fold{args.target_fold}{args.out_suffix}.json"
    out_path.write_text(json.dumps({
        "target_fold": args.target_fold,
        "crop_context": CROP_CONTEXT,
        "crop_min": CROP_MIN,
        "conf": CONF,
        "positive_min_iou": args.positive_min_iou,
        "negative_sample_rate": args.negative_sample_rate,
        "samples": samples,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"manifest -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
