"""P2-3 探針ノート: V30 の fold0 重みを学習せず、推論 imgsz だけ 1536 にする。

P2-2 は「1536 で学習し直す」が fold0 PQ 0.4295 < 0.4393 で不合格だった。
そこでは学習解像度・batch 2・Ultralytics 8.4 が交絡している。

この探針は V30 の `yolo_fold_0/weights/best.pt`（imgsz 1280 で学習済み）を
そのまま使い、predict の imgsz だけ 1536 にする。学習しない。
V30 の runs は上書きしない。

fold0 OOF PQ > 0.4393 のときだけ fold1/2 の推論に進む。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v36_infer_imgsz1536_fold0_colab.ipynb"
MEDISEG = (ROOT / "src" / "mediseg_models.py").read_text(encoding="utf-8")
METRICS = (ROOT / "src" / "metrics_official.py").read_text(encoding="utf-8")
METRICS = METRICS.replace("from __future__ import annotations\n", "", 1)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V36 P2-3 探針 — V30 重みのまま推論 imgsz 1536（fold0 のみ）

P2-2（1536 で再学習）は fold0 PQ **0.4295 < 0.4393** で不合格。
交絡を外すため、**学習はせず** V30 fold0 重みで predict の解像度だけ変える。

| | V35 提出 | P2-2（外れ） | この探針 |
|---|---|---|---|
| 重み | V30 `yolo_fold_*`（1280 学習） | 1536 で 70 epoch 再学習 | **V30 fold0 のまま** |
| 推論 imgsz | 1280 | 1536 | **1536** |
| 後処理 | shrink@0.45 + drop_contained | 同じ | 同じ |
| 比較 | fold0 PQ 0.4393 | 0.4295 | 0.4393 を上回れば fold1/2 推論へ |

**学習しない。V30 の `runs/yolo_fold_*` は触らない。test 推論もしない。**
"""
    ),
    code(
        """# [1/6] 依存関係・Drive・上書き禁止の確認
%pip install -q "ultralytics>=8.3,<9" pycocotools

import json, math, os, random, shutil, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO

drive.mount("/content/drive")

class CFG:
    SEED = 3407
    V30_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    V35_ROOT = Path("/content/drive/MyDrive/solar_filament_v35")
    DATA_CACHE = V30_ROOT / "competition_data"
    ARTIFACT_ROOT = V30_ROOT / "artifacts"
    PREPROC_CACHE = V30_ROOT / "preprocessed_v30_q96"
    WEIGHTS = V30_ROOT / "runs" / "yolo_fold_0" / "weights" / "best.pt"
    LOCAL_ROOT = Path("/content/solar_v36_infer")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    CACHE_ROOT = V35_ROOT / "prediction_cache" / "oof"
    CACHE_NAME = "yolo_v30w_imgsz1536_fold_0"
    YOLO_IMGSZ = 1536
    FOLD = 0
    CONF = 0.26
    MIN_AREA = 20
    REFINER_THRESHOLD = 0.45
    CROP_SIZE = 320
    BASELINE_FOLD0_PQ = 0.4393

assert CFG.WEIGHTS.exists(), f"V30 fold0 重みがない: {CFG.WEIGHTS}"
assert "solar_filament_v30" not in str(CFG.CACHE_ROOT)
CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
CFG.CACHE_ROOT.mkdir(parents=True, exist_ok=True)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理": CFG.PREPROC_CACHE,
    "fold0 refiner": CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt",
    "V30 fold0 weights": CFG.WEIGHTS,
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0), "imgsz": CFG.YOLO_IMGSZ,
       "weights": str(CFG.WEIGHTS), "cache": str(CFG.CACHE_ROOT / CFG.CACHE_NAME)})
"""
    ),
    code(
        """# [2/6] COCO・fold・前処理画像
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"annotation JSONが見つからない: {base}")
    return matches[0].parents[2]

TRAIN_JSON = next(find_dataset_root(CFG.DATA_CACHE).rglob(
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

images = {item["id"]: item for item in coco["images"]}
coco_anns = defaultdict(list)
for ann in coco["annotations"]:
    coco_anns[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

train_files = sorted(file_to_ids)
fold_frame = pd.read_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
assert all(name in fold_of_file for name in train_files)

for name in tqdm(train_files, desc="前処理画像をSSDへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)

print({"観測": len(train_files),
       "fold0": sum(v == 0 for v in fold_of_file.values()),
       "annotation": len(coco["annotations"])})
"""
    ),
    code(
        """# [3/6] V30 fold0 重みで imgsz 1536 の OOF 推論（学習しない）
def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    return output, confidences

def save_yolo_cache(path, masks, confidences):
    path.parent.mkdir(parents=True, exist_ok=True)
    if masks:
        packed = np.packbits(np.stack(masks).astype(np.uint8), axis=-1)
        shape = np.asarray(masks[0].shape, np.int32)
    else:
        packed = np.empty((0, 0, 0), np.uint8)
        shape = np.asarray([2048, 2048], np.int32)
    np.savez_compressed(path, packed=packed, shape=shape, confidences=confidences)

model = YOLO(str(CFG.WEIGHTS))
oof_files = [name for name in train_files if fold_of_file[name] == CFG.FOLD]
cache_dir = CFG.CACHE_ROOT / CFG.CACHE_NAME
for name in tqdm(oof_files, desc="OOF推論 imgsz1536"):
    path = cache_dir / f"{Path(name).stem}.npz"
    if path.exists():
        continue
    masks, confidences = result_masks(model.predict(
        str(CFG.PREPROC_DIR / name), imgsz=CFG.YOLO_IMGSZ, conf=0.08,
        iou=0.55, agnostic_nms=True, retina_masks=True, verbose=False,
    )[0])
    save_yolo_cache(path, masks, confidences)
print({"oof": len(oof_files), "cache": str(cache_dir),
       "npz": len(list(cache_dir.glob('*.npz'))), "weights": str(CFG.WEIGHTS)})
"""
    ),
    code("# [4/6] MediSegUNet（src/mediseg_models.py と同一）\n" + MEDISEG),
    code("# [5/6] 公式 PQ（src/metrics_official.py と同一）\n" + METRICS),
    code(
        """# [6/6] fold0 OOF を V35 確定後処理で採点し、0.4393 と比較
def load_yolo_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))

def to_rle(annotation, height, width):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

def mask_square_box(mask, context=1.65, minimum=96):
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

@torch.no_grad()
def refine_proposal(image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal

def greedy_disjoint(masks, order, min_area=CFG.MIN_AREA):
    occupied = np.zeros((2048, 2048), dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept

def drop_contained_indices(masks, threshold=0.90):
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

def apply_ownership(masks, scores):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices(masks)
    return greedy_disjoint(masks, sorted(subset, key=lambda i: areas[i]))

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

refiner = MediSegUNet(3, 1, False).cuda().eval()
try:
    ckpt = torch.load(CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt",
                      map_location="cuda", weights_only=False)
except TypeError:
    ckpt = torch.load(CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt",
                      map_location="cuda")
refiner.load_state_dict(ckpt.get("best_model") or ckpt["model"], strict=True)

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in coco_anns[image_id]
                         if not ann.get("iscrowd", 0)]

meter = PanopticQuality()
near_miss = 0
counts = []
cache_dir = CFG.CACHE_ROOT / CFG.CACHE_NAME
oof_files = [name for name in train_files if fold_of_file[name] == CFG.FOLD]
for name in tqdm(oof_files, desc="PQ採点"):
    masks, confidences = load_yolo_cache(cache_dir / f"{Path(name).stem}.npz")
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
    proposals = [masks[i] for i in keep]
    scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
    image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
    refined = [refine_proposal(image, m) for m in proposals]
    predictions = apply_ownership(refined, scores)
    counts.append(len(predictions))
    pred_rles = []
    if predictions:
        encoded = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
        pred_rles = encoded if isinstance(encoded, list) else [encoded]
    for image_id in file_to_ids[name]:
        truth = gt_rles[image_id]
        matrix = iou_matrix(truth, pred_rles)
        meter.add_matrix(matrix, len(truth), len(pred_rles))
        if matrix.size:
            near_miss += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

summary = meter.summary()
delta = summary["pq"] - CFG.BASELINE_FOLD0_PQ
continue_folds = summary["pq"] > CFG.BASELINE_FOLD0_PQ
report = {
    "probe": "P2-3 infer-only imgsz1536 on V30 fold0 weights",
    "train_imgsz": 1280,
    "infer_imgsz": CFG.YOLO_IMGSZ,
    "fold": CFG.FOLD,
    "observations": len(oof_files),
    "pq": round(summary["pq"], 6),
    "sq": round(summary["sq"], 6),
    "rq": round(summary["rq"], 6),
    "near_miss": int(near_miss),
    "inst/img": float(np.mean(counts)) if counts else 0.0,
    "baseline_fold0_pq": CFG.BASELINE_FOLD0_PQ,
    "delta": round(delta, 6),
    "continue_fold12": bool(continue_folds),
    "decision": ("fold1/2 を同じ推論 imgsz で回してよい"
                 if continue_folds else
                 "推論 1536 もベースラインを超えていない。解像度レバーはここで止める"),
}
(CFG.V35_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
(CFG.V35_ROOT / "artifacts" / "p2_3_infer_imgsz1536_fold0.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"v36i-{index:02d}"
    if cell["cell_type"] != "code":
        continue
    stripped = re.sub(r"^[%!].*$", "", cell["source"], flags=re.M)
    ast.parse(stripped)

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
generated = json.loads(OUTPUT.read_text(encoding="utf-8"))
text = "\n".join(
    "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    for cell in generated["cells"]
)
assert "class MediSegUNet" in text
assert "class PanopticQuality" in text
assert "YOLO_IMGSZ = 1536" in text
assert "yolo_fold_0" in text
assert "model.train" not in text
assert "from __future__ import annotations" not in text
assert "coco_anns" in text
print(OUTPUT)
