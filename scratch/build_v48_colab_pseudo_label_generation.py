"""V48 Colab版 — pseudo-label生成(3-fold YOLOアンサンブル+refiner+確定後処理)。

## 背景

707観測は2011-2017年が87%、2018-2022年がわずか13%と年代分布が偏っている
(README.md §2.1)。この偏りを是正するため、GONGアーカイブから追加取得した
150枚の2018-2022年画像(`scratch/fetch_gong_pseudo_label_pilot.py`で取得、
`scratch/convert_gong_fits_to_jpeg.py`で階調変換、
`scratch/preprocess_pseudo_label_images.py`で前処理済み)にpseudo-labelを
付与する。

**重要: 後処理は`shrink@0.45`(V35確定構成)を使う。** free variantは
2026-08-29の監査でfold0選択バイアスと判明し不採用のため、ここでは使わない。

## このノートでやること

1. 3-fold YOLOモデル(`yolo_fold_{0,1,2}/weights/best.pt`)でそれぞれ独立に推論
2. `weighted_mask_fusion`でIoUクラスタ化、**2/3以上のfoldが合意した候補のみ**採用
   (単一foldだけの誤検出を除外する狙い)
3. confidence閾値を本番の0.26より厳しい0.5でフィルタ
4. all_data refiner(全707観測で学習、pseudo-label画像に対して最も汎化する
   はずのrefiner)を適用、`shrink@0.45`(確定構成)で後処理
5. `drop_contained_then_area_asc`(閾値0.90)で排他化
6. 結果をnpzキャッシュ(既存のOOFキャッシュと同形式)+manifestとして保存

**採用判定はしない。** 次段階(人間QAビューア→リークなし拡張学習)で判断する。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v48_colab_pseudo_label_generation.ipynb"
MEDISEG = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")
MASK_WBF = (ROOT / "solar_filament" / "pipeline" / "mask_wbf.py").read_text(encoding="utf-8")
MASK_WBF = MASK_WBF.replace("from __future__ import annotations\n", "", 1)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V48 Colab版 — pseudo-label生成

3-fold YOLOアンサンブル(2/3合意フィルタ)+ all_data refiner + **確定後処理
shrink@0.45**(free variantは監査でfold0選択バイアスと判明、使わない)で
GONG追加画像150枚にpseudo-labelを付与する。

ランタイム: **GPU**(L4/A100等)。
"""
    ),
    code(
        """# [1/5] 環境確認・依存関係・Drive mount
import json, math, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from google.colab import drive
from pycocotools import mask as mask_utils

drive.mount("/content/drive")

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("gpu:", gpu_name)
if not torch.cuda.is_available():
    raise RuntimeError("GPUランタイムを選択してください")

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
from ultralytics import YOLO

class CFG:
    V30_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    V35_ROOT = Path("/content/drive/MyDrive/solar_filament_v35")
    ARTIFACTS_ROOT = V30_ROOT / "artifacts"
    YOLO_WEIGHTS = {f: V30_ROOT / "runs" / f"yolo_fold_{f}" / "weights" / "best.pt" for f in (0, 1, 2)}
    REFINER_CKPT = ARTIFACTS_ROOT / "mediseg_refiner_all_data_resume.pt"
    PSEUDO_DIR = V35_ROOT / "pseudo_label" / "preprocessed"
    OUT_DIR = V35_ROOT / "pseudo_label" / "cache"
    CROP_SIZE = 320
    CONF_YOLO = 0.10          # WBF融合前は緩めに集める(融合後にconf_finalで絞る)
    MIN_AREA = 20
    CONF_FINAL = 0.50         # 本番0.26より厳しいpseudo-label採用閾値
    MIN_SOURCE_AGREEMENT = 2  # 3-foldのうち2つ以上が同一個体を検出した場合のみ採用
    WBF_IOU_THRESHOLD = 0.5
    WBF_VOTE_THRESHOLD = 0.5
    SHRINK_THRESHOLD = 0.45   # 確定構成(free variantではない)
    DROP_THRESHOLD = 0.90

for required in list(CFG.YOLO_WEIGHTS.values()) + [CFG.REFINER_CKPT, CFG.PSEUDO_DIR]:
    if not required.exists():
        raise FileNotFoundError(f"{required} が見つからない")

CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
targets = sorted(p.name for p in CFG.PSEUDO_DIR.glob("*.jpeg"))
print({"pseudo_images": len(targets), "gpu": gpu_name})"""
    ),
    code(
        "# [2/5] MediSeg UNetアーキテクチャ・WBF(solar_filament/と同一)\n"
        + MEDISEG
        + "\n\n"
        + MASK_WBF
        + """
def soft_dice_loss(logits, targets):
    probs = logits.sigmoid()
    intersection = (probs * targets).sum((1, 2, 3))
    denominator = probs.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()

def imread_unicode(path):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
"""
    ),
    code(
        """# [3/5] YOLO 3-fold・refinerのロード
yolo_models = {f: YOLO(str(path)) for f, path in CFG.YOLO_WEIGHTS.items()}

refiner_ckpt = torch.load(CFG.REFINER_CKPT, map_location="cpu", weights_only=False)
refiner = MediSegUNet(3, 1, False).cuda().eval()
refiner.load_state_dict(refiner_ckpt.get("best_model") or refiner_ckpt["model"], strict=True)
print("loaded: 3x YOLO fold models + all_data refiner")

def square_box(mask, context=1.65, minimum=96):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, width, height
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > width: x0 -= x1 - width; x1 = width
    if y1 > height: y0 -= y1 - height; y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

def refine_batch(refiner_model, image, proposals, batch=32):
    boxes, crops = [], []
    for proposal in proposals:
        x0, y0, x1, y1 = square_box(proposal)
        crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        boxes.append((x0, y0, x1, y1))
        crops.append(crop.transpose(2, 0, 1))
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            chunk = np.stack(crops[start:start + batch])
            tensor = torch.from_numpy(chunk.copy()).float().cuda() / 255.0
            output = refiner_model(tensor).sigmoid()[:, 0].cpu().numpy()
            for offset in range(output.shape[0]):
                x0, y0, x1, y1 = boxes[start + offset]
                probabilities.append(cv2.resize(output[offset], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR))
    return boxes, probabilities

def shrink_variant(proposal, box, probability, threshold):
    x0, y0, x1, y1 = box
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    return refined & proposal  # shrink = refined AND proposal(確定構成)

def greedy_disjoint_local(masks, order, min_area=CFG.MIN_AREA, canvas=2048):
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
    rles = mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))
    areas = np.asarray(mask_utils.area(rles), dtype=np.float64)
    self_iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
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
"""
    ),
    code(
        """# [4/5] 画像ごとに: 3-fold YOLO推論 -> WBF融合(2/3合意フィルタ) -> refiner(shrink@0.45) -> 排他化
started = time.perf_counter()
manifest = []
for position, name in enumerate(targets, 1):
    out_path = CFG.OUT_DIR / f"{Path(name).stem}.npz"
    image_path = CFG.PSEUDO_DIR / name
    image = imread_unicode(image_path)

    all_masks, all_scores, all_sources = [], [], []
    for fold, model in yolo_models.items():
        result = model.predict(str(image_path), conf=CFG.CONF_YOLO, verbose=False)[0]
        if result.masks is None:
            continue
        for mask_xy, box_conf in zip(result.masks.data.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            mask_full = cv2.resize(mask_xy, (2048, 2048), interpolation=cv2.INTER_LINEAR) >= 0.5
            if mask_full.sum() < CFG.MIN_AREA:
                continue
            all_masks.append(mask_full)
            all_scores.append(float(box_conf))
            all_sources.append(fold)

    if not all_masks:
        np.savez_compressed(out_path, packed=np.empty((0, 0, 0), np.uint8),
                            shape=np.asarray([2048, 2048], np.int32), confidences=np.zeros(0, np.float32),
                            agreement=np.zeros(0, np.int32))
        manifest.append({"file_name": name, "n_proposals": 0, "n_kept": 0})
        continue

    fused_masks, fused_scores, fused_agreement = weighted_mask_fusion(
        all_masks, np.asarray(all_scores, dtype=np.float32), all_sources,
        iou_threshold=CFG.WBF_IOU_THRESHOLD, vote_threshold=CFG.WBF_VOTE_THRESHOLD)

    # 2/3以上のfoldが合意 かつ confidence >= 0.5 のみ採用
    keep_indices = [i for i in range(len(fused_masks))
                    if fused_agreement[i] >= CFG.MIN_SOURCE_AGREEMENT and fused_scores[i] >= CFG.CONF_FINAL]
    proposals = [fused_masks[i] for i in keep_indices]
    kept_agreement = [fused_agreement[i] for i in keep_indices]
    kept_scores = [float(fused_scores[i]) for i in keep_indices]

    if not proposals:
        np.savez_compressed(out_path, packed=np.empty((0, 0, 0), np.uint8),
                            shape=np.asarray([2048, 2048], np.int32), confidences=np.zeros(0, np.float32),
                            agreement=np.zeros(0, np.int32))
        manifest.append({"file_name": name, "n_proposals": len(fused_masks), "n_kept": 0})
        continue

    boxes, probabilities = refine_batch(refiner, image, proposals)
    refined = [shrink_variant(proposals[i], boxes[i], probabilities[i], CFG.SHRINK_THRESHOLD)
               for i in range(len(proposals))]
    final_order = sorted(range(len(refined)), key=lambda i: -kept_scores[i])
    final_masks_ordered = [refined[i] for i in final_order]
    final = apply_ownership_local(final_masks_ordered, CFG.DROP_THRESHOLD)
    # apply_ownership_localはarea_asc昇順に並べ替えて処理するため、final要素順はsize順。
    # confidence/agreementは元のmaskとサイズで対応付け直すのではなく、
    # final各要素についてどのproposalに由来するか面積一致で逆引きする簡易対応。
    final_scores, final_agreement = [], []
    for f_mask in final:
        best_i, best_overlap = 0, -1.0
        for i, r_mask in enumerate(final_masks_ordered):
            overlap = np.logical_and(f_mask, r_mask).sum()
            if overlap > best_overlap:
                best_overlap, best_i = overlap, i
        final_scores.append(kept_scores[final_order[best_i]])
        final_agreement.append(kept_agreement[final_order[best_i]])

    if final:
        packed = np.packbits(np.stack(final).astype(np.uint8), axis=-1)
        shape = np.asarray(final[0].shape, np.int32)
    else:
        packed = np.empty((0, 0, 0), np.uint8)
        shape = np.asarray([2048, 2048], np.int32)
    np.savez_compressed(out_path, packed=packed, shape=shape,
                        confidences=np.asarray(final_scores, np.float32),
                        agreement=np.asarray(final_agreement, np.int32))
    manifest.append({"file_name": name, "n_proposals": len(fused_masks), "n_kept": len(final)})

    if position % 25 == 0 or position == len(targets):
        total_kept = sum(m["n_kept"] for m in manifest)
        print(f"  {position}/{len(targets)}  {time.perf_counter()-started:.0f}s  累計採用{total_kept}件", flush=True)

manifest_path = CFG.OUT_DIR.parent / "pseudo_label_manifest.json"
manifest_path.write_text(json.dumps({
    "scope": f"{len(targets)}枚のGONG追加画像。3-fold YOLO WBF(2/3合意)+refiner(shrink@0.45確定構成)",
    "conf_final": CFG.CONF_FINAL, "min_source_agreement": CFG.MIN_SOURCE_AGREEMENT,
    "total_kept": sum(m["n_kept"] for m in manifest),
    "images": manifest,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved manifest:", manifest_path)
print("total_kept:", sum(m["n_kept"] for m in manifest), "/ ", len(targets), "images")"""
    ),
    code(
        """# [5/5] QA用サマリ: 検出数分布・信頼度分布を確認(人間QAビューアへ渡す前の一次チェック)
kept_counts = [m["n_kept"] for m in manifest]
print({
    "images_with_zero_pseudo_labels": sum(1 for c in kept_counts if c == 0),
    "mean_kept_per_image": float(np.mean(kept_counts)),
    "max_kept_per_image": max(kept_counts),
    "total_pseudo_labels": sum(kept_counts),
})"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for cell in cells:
    if cell["cell_type"] == "code":
        ast.parse(re.sub(r"^[%!].*$", "", cell["source"], flags=re.M))
print("syntax OK")

text = json.dumps(notebook, ensure_ascii=False, indent=1)
assert "shrink_variant" in text
assert "SHRINK_THRESHOLD = 0.45" in text
assert "free" not in text.lower() or "free_variant" not in text  # free variantを使っていないことの確認
OUTPUT.write_text(text, encoding="utf-8")
print(OUTPUT)
