"""near-miss症例(GT vs shrink vs free refiner)を人間が見比べられるビューア用データを作る。

fold0のnear-miss(IoU 0.3〜0.5)ペアからサンプルを取り、各症例について
- 画像クロップ(JPEG、data URI)
- GTマスクの輪郭(SVGポリゴン点列)
- shrink@0.45(現行baseline)の輪郭
- free@threshold(新variant、候補マスクとの交差なし)の輪郭
- IoU値・過小/過大検出の分類

をJSONへ書き出す。HTMLビューア(build_near_miss_viewer.py)がこれを読み込んで
data URIとして埋め込む。

使い方:
    python scratch/build_near_miss_viewer_data.py --free-threshold 0.7 --limit 24
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS, FOLDS_CSV, MIN_AREA, OOF_CACHE, PREPROC, TRAIN_JSON,
    imread_unicode, iou_matrix, load_yolo_cache, refine_batch, to_rle,
)
from pycocotools import mask as mask_util  # noqa: E402
import pandas as pd  # noqa: E402

CONF = 0.26
FOLD = 0
CROP_PAD = 60
CROP_SIZE_OUT = 420


def free_variant(box, probability, threshold, shape):
    x0, y0, x1, y1 = box
    refined = np.zeros(shape, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    return refined


def shrink_variant(proposal, box, probability, threshold):
    x0, y0, x1, y1 = box
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    return refined & proposal


def mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    """boolマスク -> 輪郭点列(複数リング対応)。ビューア側でSVG polygonとして描く。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        approx = cv2.approxPolyDP(contour, 1.5, True)
        polygons.append([[float(p[0][0]), float(p[0][1])] for p in approx])
    return polygons


def classify_coverage(gt_mask: np.ndarray, pred_mask: np.ndarray) -> str:
    gt_only = (gt_mask & ~pred_mask).sum()
    pred_only = (pred_mask & ~gt_mask).sum()
    if gt_only > pred_only * 1.3:
        return "undercoverage"
    if pred_only > gt_only * 1.3:
        return "overcoverage"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free-threshold", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("output/near_miss_viewer"))
    args = parser.parse_args()
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
    print(f"観測数: {len(targets)}", flush=True)

    checkpoint = torch.load(
        ARTIFACTS / "mediseg_refiner_fold0_resume.pt", map_location="cpu", weights_only=False)
    refiner = MediSegUNet(3, 1, False).eval()
    refiner.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    cases = []
    for name in targets:
        if len(cases) >= args.limit * 3:  # 候補は多めに集めてから多様性を見て間引く
            break
        image = imread_unicode(PREPROC / name)
        if image is None:
            continue
        masks, confidences = load_yolo_cache(OOF_CACHE / f"yolo_fold_{FOLD}" / f"{Path(name).stem}.npz")
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= CONF and m.sum() >= MIN_AREA]
        proposals = [masks[i] for i in keep]
        if not proposals:
            continue
        boxes, probabilities = refine_batch(refiner, image, proposals)

        shrink_masks = [shrink_variant(proposals[i], boxes[i], probabilities[i], 0.45)
                         for i in range(len(proposals))]
        free_masks = [free_variant(boxes[i], probabilities[i], args.free_threshold, proposals[i].shape)
                      for i in range(len(proposals))]

        variants = file_to_ids[name]
        for image_id in variants:
            gt_list = gt_rles[image_id]
            if not gt_list:
                continue
            gt_masks = [mask_util.decode(g).astype(bool) for g in gt_list]
            # shrinkマスクとのIoU行列でnear-missを探す
            shrink_rles = mask_util.encode(np.asfortranarray(np.stack(shrink_masks, axis=-1).astype(np.uint8))) if shrink_masks else []
            matrix = iou_matrix(gt_list, shrink_rles) if shrink_rles else np.zeros((len(gt_list), 0))
            for gt_index, gt_mask in enumerate(gt_masks):
                if matrix.shape[1] == 0:
                    continue
                best_pred_index = int(np.argmax(matrix[gt_index]))
                best_iou = float(matrix[gt_index, best_pred_index])
                if not (0.3 < best_iou <= 0.5):
                    continue
                # クロップ範囲: GT+shrink+freeのbboxを包含
                free_mask = free_masks[best_pred_index]
                shrink_mask_arr = shrink_masks[best_pred_index]
                combined = gt_mask | shrink_mask_arr | free_mask
                ys, xs = np.where(combined)
                if len(xs) == 0:
                    continue
                x0, y0 = max(0, xs.min() - CROP_PAD), max(0, ys.min() - CROP_PAD)
                x1, y1 = min(image.shape[1], xs.max() + CROP_PAD), min(image.shape[0], ys.max() + CROP_PAD)
                scale = CROP_SIZE_OUT / max(x1 - x0, y1 - y0)

                def to_crop_coords(polygons):
                    return [[[(px - x0) * scale, (py - y0) * scale] for px, py in poly] for poly in polygons]

                crop = image[y0:y1, x0:x1]
                crop_resized = cv2.resize(crop, (int((x1 - x0) * scale), int((y1 - y0) * scale)))
                ok, buf = cv2.imencode(".jpg", crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if not ok:
                    continue
                data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

                free_iou = float(iou_matrix([gt_list[gt_index]],
                                             mask_util.encode(np.asfortranarray(free_mask[:, :, None].astype(np.uint8))))[0, 0])

                cases.append({
                    "file_name": name,
                    "coverage_mode": classify_coverage(gt_mask, shrink_mask_arr),
                    "shrink_iou": round(best_iou, 4),
                    "free_iou": round(free_iou, 4),
                    "improved": bool(free_iou > best_iou),
                    "crossed_threshold": bool(free_iou > 0.5 and best_iou <= 0.5),
                    "image": data_uri,
                    "width": crop_resized.shape[1],
                    "height": crop_resized.shape[0],
                    "gt_polygons": to_crop_coords(mask_to_polygons(gt_mask[y0:y1, x0:x1])),
                    "shrink_polygons": to_crop_coords(mask_to_polygons(shrink_mask_arr[y0:y1, x0:x1])),
                    "free_polygons": to_crop_coords(mask_to_polygons(free_mask[y0:y1, x0:x1])),
                })
        if len(cases) % 10 == 0:
            print(f"  症例収集中... {len(cases)}件", flush=True)

    # 多様性を確保して間引く: undercoverage/overcoverage/mixedからバランスよく、改善例と非改善例も混ぜる
    by_mode = defaultdict(list)
    for case in cases:
        by_mode[case["coverage_mode"]].append(case)
    selected = []
    modes = list(by_mode.keys())
    index = 0
    while len(selected) < args.limit and any(by_mode.values()):
        mode = modes[index % len(modes)]
        if by_mode[mode]:
            selected.append(by_mode[mode].pop(0))
        index += 1
        if index > args.limit * 10:
            break

    out_path = args.out / "near_miss_cases.json"
    out_path.write_text(json.dumps({
        "free_threshold": args.free_threshold,
        "cases": selected,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"選定症例数: {len(selected)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
