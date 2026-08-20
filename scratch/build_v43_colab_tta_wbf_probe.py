"""Colab版ビルダー: TTA(test-time augmentation)+WBFプローブ(fold0のみ)。

## 背景

imgsz・augmentation強化・モデルサイズ拡大という検出器強化系6本と、
multi-seed単純連結アンサンブル(V42)が全て不合格だった(改訂9)。
WBF(`src/mask_wbf.py`)による統合は単純連結より改善した
(fold0ローカルΔ: 単純連結-0.0078 → WBF-0.0027)が、まだ基準を下回る。

この探針は**追加学習を一切せず**、既存のV30 fold0確定モデル(best.pt)に対して
複数の向き(元画像・左右反転・上下反転・180度回転)で推論し、その4視点分の
検出結果をWBFで統合する。学習コストがゼロ(推論のみ)なので、GPU予算が
ほぼ枯渇した状況でも試せる。

仮説: 同一モデルでも視点を変えると偽陽性の出方が変わるため、複数視点の
合意で偽陽性を抑えつつ、真陽性は複数視点で繰り返し検出されて生き残る
(multi-seedと同じ理屈を「モデルの違い」ではなく「視点の違い」で作る)。

## なぜfold0だけか

**採用判定はしない。** fold0 OOF PQ が V35 fold0 実測 **0.4393** を上回ったら
全fold・testで使う価値があると判断する。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v43_colab_tta_wbf_probe.ipynb"
MEDISEG = (ROOT / "src" / "mediseg_models.py").read_text(encoding="utf-8")
METRICS = (ROOT / "src" / "metrics_official.py").read_text(encoding="utf-8")
METRICS = METRICS.replace("from __future__ import annotations\n", "", 1)
MASK_WBF = (ROOT / "src" / "mask_wbf.py").read_text(encoding="utf-8")
MASK_WBF = MASK_WBF.replace("from __future__ import annotations\n", "", 1)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V43 Colab探針 — TTA(4視点)+WBF（fold0のみ、追加学習なし）

検出器強化系6本・multi-seed単純連結が全て不合格だった(改訂9)。WBFでの統合は
単純連結より改善したが(ローカルΔ: -0.0078→-0.0027)まだ基準未満。

この探針は**追加学習ゼロ**。既存のV30 fold0確定モデル(`best.pt`)に対し、
元画像・左右反転・上下反転・180度回転の4視点で推論し、WBFで統合する。
GPU予算がほぼ枯渇していても試せる(推論のみ、数分〜十数分程度)。

**採用判定はしない。** fold0 OOF PQ が V35 fold0 実測 **0.4393** を上回ったら
全fold・testで使う価値があると判断する。

GPU必須（推論のみなのでT4でも十分速い）。
"""
    ),
    code(
        """# [1/6] 依存関係・Drive・前提確認
%pip install -q "ultralytics>=8.3,<9" pycocotools

import json, math
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
    V30_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    V35_ROOT = Path("/content/drive/MyDrive/solar_filament_v35")
    DATA_CACHE = V30_ROOT / "competition_data"
    ARTIFACT_ROOT = V30_ROOT / "artifacts"
    PREPROC_CACHE = V30_ROOT / "preprocessed_v30_q96"
    FOLD0_WEIGHTS = V30_ROOT / "runs" / "yolo_fold_0" / "weights" / "best.pt"
    FOLD = 0
    CONF = 0.26
    RAW_CONF = 0.08           # OOFキャッシュ生成時と同じ緩い閾値でproposalを集める
    MIN_AREA = 20
    REFINER_THRESHOLD = 0.45
    CROP_SIZE = 320
    EVAL_SIZE = 2048
    BASELINE_FOLD0_PQ = 0.4393
    WBF_IOU = 0.5

required = {
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理": CFG.PREPROC_CACHE,
    "fold0確定重み": CFG.FOLD0_WEIGHTS,
    "fold0 refiner": CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt",
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0)})
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

fold_frame = pd.read_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
fold0_files = sorted(
    name for name in file_to_ids
    if fold_of_file.get(name) == CFG.FOLD and (CFG.PREPROC_CACHE / name).exists()
)
print({"fold0観測": len(fold0_files)})
"""
    ),
    code(
        """# [3/6] 4視点(元画像/左右反転/上下反転/180度回転)でYOLO推論
def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = (result.boxes.conf.detach().cpu().numpy().astype(np.float32)
                   if result.boxes is not None else np.zeros(0, np.float32))
    return output, confidences

# (順変換, 逆変換)。画像はimshow用ではなくnp配列そのものに掛ける。
VIEWS = {
    "identity": (lambda im: im, lambda m: m),
    "hflip": (lambda im: cv2.flip(im, 1), lambda m: np.flip(m, axis=1)),
    "vflip": (lambda im: cv2.flip(im, 0), lambda m: np.flip(m, axis=0)),
    "rot180": (lambda im: cv2.rotate(im, cv2.ROTATE_180), lambda m: np.rot90(m, 2)),
}

model = YOLO(str(CFG.FOLD0_WEIGHTS))

def imread_unicode(path):
    if not path.exists():
        return None
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)

tta_cache = {}   # file_name -> {view_name: (masks, scores)}
for name in tqdm(fold0_files, desc="TTA推論(4視点)"):
    image = imread_unicode(CFG.PREPROC_CACHE / name)
    if image is None:
        continue
    per_view = {}
    for view_name, (forward, backward) in VIEWS.items():
        transformed = forward(image)
        result = model.predict(transformed, imgsz=1280, conf=CFG.RAW_CONF,
                               iou=0.55, agnostic_nms=True, retina_masks=True,
                               verbose=False)[0]
        masks, scores = result_masks(result)
        masks = [np.ascontiguousarray(backward(m)) for m in masks]
        per_view[view_name] = (masks, scores)
    tta_cache[name] = per_view

print({"推論済み観測": len(tta_cache)})
"""
    ),
    code("# [4/6] WBF（src/mask_wbf.py と同一）\n" + MASK_WBF),
    code("# [4.5/6] MediSegUNet・公式PQ（src/mediseg_models.py, src/metrics_official.py と同一）\n"
         + MEDISEG + "\n\n" + METRICS),
    code(
        """# [5/6] refiner読み込み・共通後処理関数
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
def refine_proposal(refiner, image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal

def greedy_disjoint(masks, order, min_area=CFG.MIN_AREA):
    occupied = np.zeros((CFG.EVAL_SIZE, CFG.EVAL_SIZE), dtype=bool)
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

def masks_to_rles(masks):
    if not masks:
        return []
    return mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))

def filter_proposals(masks, confidences):
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
    return [masks[i] for i in keep], (confidences[keep] if len(keep) else np.zeros(0, np.float32))

refiner = MediSegUNet(3, 1, False).cuda().eval()
try:
    ckpt = torch.load(CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt",
                      map_location="cuda", weights_only=False)
except TypeError:
    ckpt = torch.load(CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt", map_location="cuda")
refiner.load_state_dict(ckpt.get("best_model") or ckpt["model"], strict=True)
print("refiner loaded")
"""
    ),
    code(
        """# [6/6] 単一視点(identity) vs TTA+WBF をV35確定後処理で採点
gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in coco_anns[image_id]
                         if not ann.get("iscrowd", 0)]

meter_single = PanopticQuality()
meter_tta = PanopticQuality()
near_miss_single = 0
near_miss_tta = 0
consensus_counter = {}

for name in tqdm(fold0_files, desc="採点"):
    if name not in tta_cache:
        continue
    image = imread_unicode(CFG.PREPROC_CACHE / name)
    per_view = tta_cache[name]

    # 単一視点(identity)のみ = V35確定と同一構成の再現(サニティチェック)
    masks_id, conf_id = per_view["identity"]
    proposals_id, scores_id = filter_proposals(masks_id, conf_id)
    if proposals_id:
        refined_id = [refine_proposal(refiner, image, m) for m in proposals_id]
        final_single = apply_ownership(refined_id, scores_id)
    else:
        final_single = []
    rles_single = masks_to_rles(final_single)

    # TTA+WBF: 4視点の全proposalを結合してWBF
    combined_masks, combined_scores, source_ids = [], [], []
    for view_index, view_name in enumerate(VIEWS):
        masks_v, conf_v = per_view[view_name]
        proposals_v, scores_v = filter_proposals(masks_v, conf_v)
        combined_masks += proposals_v
        combined_scores += list(scores_v)
        source_ids += [view_index] * len(proposals_v)
    combined_scores = np.asarray(combined_scores, dtype=np.float32)
    fused_masks, fused_scores, fused_counts = weighted_mask_fusion(
        combined_masks, combined_scores, source_ids, iou_threshold=CFG.WBF_IOU)
    for count in fused_counts:
        consensus_counter[count] = consensus_counter.get(count, 0) + 1
    if fused_masks:
        refined_tta = [refine_proposal(refiner, image, m) for m in fused_masks]
        final_tta = apply_ownership(refined_tta, fused_scores)
    else:
        final_tta = []
    rles_tta = masks_to_rles(final_tta)

    for image_id in file_to_ids[name]:
        truth = gt_rles[image_id]
        matrix_s = iou_matrix(truth, rles_single)
        meter_single.add_matrix(matrix_s, len(truth), len(rles_single))
        if matrix_s.size:
            near_miss_single += int(((matrix_s > 0.3) & (matrix_s <= 0.5)).any(axis=0).sum())
        matrix_t = iou_matrix(truth, rles_tta)
        meter_tta.add_matrix(matrix_t, len(truth), len(rles_tta))
        if matrix_t.size:
            near_miss_tta += int(((matrix_t > 0.3) & (matrix_t <= 0.5)).any(axis=0).sum())

summary_single = meter_single.summary()
summary_tta = meter_tta.summary()
report = {
    "experiment": "v43_tta_wbf_probe",
    "views": list(VIEWS),
    "wbf_iou_threshold": CFG.WBF_IOU,
    "fold": CFG.FOLD,
    "observations": len(fold0_files),
    "single_identity": {**summary_single, "near_miss": near_miss_single},
    "tta_wbf": {**summary_tta, "near_miss": near_miss_tta},
    "consensus_cluster_counts": consensus_counter,
    "baseline_fold0_pq": CFG.BASELINE_FOLD0_PQ,
    "delta_tta_vs_baseline": round(summary_tta["pq"] - CFG.BASELINE_FOLD0_PQ, 6),
    "delta_tta_vs_single_local": round(summary_tta["pq"] - summary_single["pq"], 6),
    "continue_fold12": bool(summary_tta["pq"] > CFG.BASELINE_FOLD0_PQ),
}
(CFG.V35_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
(CFG.V35_ROOT / "artifacts" / "v43_tta_wbf_fold0.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"v43c-{index:02d}"
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
assert "def weighted_mask_fusion" in text
assert "VIEWS = {" in text
assert "yolo_fold_0" in text
print(OUTPUT)
