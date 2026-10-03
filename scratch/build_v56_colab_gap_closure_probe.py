"""スケルトンベースのギャップ閉合 OOF検証 notebook builder — WINNING_STRATEGY_2026-08-29.md §6.12優先度3。

## 目的

2026-09-15の競合公開コード調査で見つかった改善レバーの優先度3:
`solar_filament.pipeline.gap_close.skeleton_gap_close`(ローカルで7件のテスト
全パス済み、`tests/test_gap_close.py`)を、refiner適用後・ownership適用前に
挿入するとOOF PQが改善するかを検証する。

## GPU学習は不要

このレバーは古典的画像処理の後処理のみで、**refinerの再学習は不要。**
既存のfold別標準refinerチェックポイント(`mediseg_refiner_fold{N}_resume.pt`、
V51/matched-parameter ablationで使用済み)とYOLO検出器をそのまま使い、
推論結果に対する後処理だけを変える。そのためV54/V55のような1〜1.5時間の
学習は発生せず、推論のみで10〜20分程度の見込み。

`max_gap`(橋渡しを許す最大距離)を{0(=ギャップ閉合なし、基準), 4, 8, 16}で
スイープし、1回の推論結果を使い回して4通りの後処理を比較する(推論を4回
繰り返さない)。

## 使い方

    python scratch/build_v56_colab_gap_closure_probe.py --fold 0
    python scratch/build_v56_colab_gap_closure_probe.py --fold 1
    python scratch/build_v56_colab_gap_closure_probe.py --fold 2

出力: solar_filament_v56_gap_closure_probe_fold{N}_colab.ipynb
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GAP_CLOSE_SOURCE = (ROOT / "solar_filament" / "pipeline" / "gap_close.py").read_text(encoding="utf-8")

# 標準refinerの全fold OOF基準(README「現在地」、build_v51_edgeattnet_refiner.pyと同一)
BASELINE_PQ = {0: 0.4393, 1: 0.4415, 2: 0.4471}
MAX_GAP_GRID = (0.0, 4.0, 8.0, 16.0)  # 0.0 = ギャップ閉合なし(現行のまま)
BRIDGE_WIDTH = 3


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(fold):
    baseline_pq = BASELINE_PQ[fold]

    cells = [
        markdown(
            """# スケルトンベースのギャップ閉合 OOF検証(fold{fold})

WINNING_STRATEGY_2026-08-29.md §6.12優先度3。`skeleton_gap_close`
(`solar_filament/pipeline/gap_close.py`、ローカルで7件のテスト全パス済み)を
refiner適用後・ownership適用前に挿入し、OOF PQへの影響を`max_gap`のスイープ
{grid}で検証する(`max_gap=0.0`はギャップ閉合なし=現行のまま、基準点)。

**GPU学習は不要。** 既存のfold{fold}標準refinerチェックポイントと
YOLO検出器をそのまま使い、推論結果への後処理だけを変える。推論は1回だけ
行い、4通りの`max_gap`で後処理を分岐させて比較する。

fold{fold}基準(標準refiner、matched-parameter ablation実測): {baseline_pq}。
GPU必須(推論のみ、10〜20分程度の見込み)。
""".format(fold=fold, grid=MAX_GAP_GRID, baseline_pq=baseline_pq)
        ),
        code(
            """# [1/6] Drive接続、設定、前提の確認
import json, math, shutil, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from tqdm.auto import tqdm

drive.mount("/content/drive")

class CFG:
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_gap_closure_probe")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    FOLD = %d
    CONF = 0.26
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ = %r
    MAX_GAP_GRID = %r
    BRIDGE_WIDTH = %d

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像": CFG.PREPROC_CACHE,
    f"fold{CFG.FOLD} YOLO(OOF推論用)": CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{CFG.FOLD}" / "weights" / "best.pt",
    f"fold{CFG.FOLD} 標準refiner": CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0)})
"""
            % (fold, baseline_pq, MAX_GAP_GRID, BRIDGE_WIDTH)
        ),
        code(
            """# [2/6] COCOとfoldを読み、前処理画像をローカルSSDへ展開
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"学習annotation JSONが見つかりません: {base}")
    return matches[0].parents[2]

TRAIN_JSON = next(find_dataset_root(CFG.DATA_CACHE).rglob(
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

images = {item["id"]: item for item in coco["images"]}
annotations_by_image = defaultdict(list)
for ann in coco["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)

fold_frame = pd.read_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]

for name in tqdm(valid_files_fold, desc="検証画像をSSDへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)

try:
    from pycocotools import mask as coco_mask
except ImportError:
    !pip -q install pycocotools
    from pycocotools import mask as coco_mask

print({"検証(OOF)画像": len(valid_files_fold)})
"""
        ),
        code(
            "# [3/6] MediSegUNet (solar_filament/models/mediseg_models.py と同一。標準refinerのロードに使用)\n"
            + (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")
        ),
        code(
            "# [4/6] skeleton_gap_close (solar_filament/pipeline/gap_close.py と同一、7件のテスト全パス済み)\n"
            + GAP_CLOSE_SOURCE
        ),
        code(
            """# [5/6] 推論・後処理・採点関数の定義
import shutil
!pip -q install ultralytics
from ultralytics import YOLO
from pycocotools import mask as mask_utils

yolo_model = YOLO(str(CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{CFG.FOLD}" / "weights" / "best.pt"))
refiner = MediSegUNet(3, 1, False).cuda().eval()
_ckpt = torch.load(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
                    map_location="cuda", weights_only=False)
refiner.load_state_dict(_ckpt.get("best_model") or _ckpt["model"], strict=True)
del _ckpt

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
    return refined & proposal  # shrink(確定構成)

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

def drop_contained_indices(masks, threshold=CFG.DROP_THRESHOLD):
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

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

class PanopticQuality:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.tp_ious = []
        self.fp_count = 0
        self.fn_count = 0
    def add_matrix(self, matrix, n_gt, n_pred):
        if n_gt == 0:
            self.fp_count += n_pred; return
        if n_pred == 0:
            self.fn_count += n_gt; return
        hits = matrix > self.iou_threshold
        scores = matrix[hits].tolist()
        self.tp_ious.extend(scores)
        self.fp_count += int((hits.sum(axis=0) == 0).sum())
        self.fn_count += int((hits.sum(axis=1) == 0).sum())
    @property
    def denominator(self):
        return len(self.tp_ious) + 0.5 * self.fp_count + 0.5 * self.fn_count
    @property
    def score(self):
        d = self.denominator
        return float(sum(self.tp_ious) / d) if d > 0 else 0.0
    def sq_rq(self):
        tp = len(self.tp_ious)
        sq = float(sum(self.tp_ious) / tp) if tp > 0 else 0.0
        rq = float(tp / (tp + 0.5 * self.fp_count + 0.5 * self.fn_count)) if (tp + self.fp_count + self.fn_count) > 0 else 0.0
        return sq, rq
    def summary(self):
        sq, rq = self.sq_rq()
        return {"pq": self.score, "sq": sq, "rq": rq, "tp": len(self.tp_ious),
                "fp": self.fp_count, "fn": self.fn_count, "denominator": self.denominator}

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in annotations_by_image[image_id] if not ann.get("iscrowd", 0)]

print("推論・後処理・採点関数の定義OK")
"""
        ),
        code(
            """# [6/6] 1回の推論を使い回し、max_gapグリッドごとに後処理を分岐して採点
from scipy import ndimage as ndi_label

meters = {max_gap: PanopticQuality() for max_gap in CFG.MAX_GAP_GRID}
fragmentation_counts = []  # 診断用: refine直後の連結成分数(gap_close適用前)の分布

for name in tqdm(valid_files_fold, desc=f"fold{CFG.FOLD} OOF採点(max_gapグリッド)"):
    image_path = CFG.PREPROC_DIR / name
    result = yolo_model.predict(str(image_path), imgsz=1280, conf=0.08, iou=0.55,
                                 agnostic_nms=True, retina_masks=True, verbose=False)[0]
    masks, confidences = result_masks(result)
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences)) if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
    proposals = [masks[i] for i in keep]
    scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    refined = [refine_proposal(image, m) for m in proposals]  # 1回だけ計算(高コスト部分)

    for m in refined:
        _, n_components = ndi_label.label(m, structure=np.ones((3, 3)))
        fragmentation_counts.append(n_components)

    for max_gap in CFG.MAX_GAP_GRID:
        if max_gap > 0:
            closed = [skeleton_gap_close(m, max_gap=max_gap, bridge_width=CFG.BRIDGE_WIDTH) for m in refined]
        else:
            closed = refined  # max_gap=0.0はギャップ閉合なし(現行のまま、基準)
        predictions = apply_ownership(closed, scores)
        pred_rles = []
        if predictions:
            encoded = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
            pred_rles = encoded if isinstance(encoded, list) else [encoded]
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, pred_rles)
            meters[max_gap].add_matrix(matrix, len(truth), len(pred_rles))

fragmentation_counts = np.array(fragmentation_counts)
print(json.dumps({
    "fragmentation_diagnostic": {
        "n_proposals": int(len(fragmentation_counts)),
        "multi_component_fraction": float((fragmentation_counts > 1).mean()) if len(fragmentation_counts) else 0.0,
        "max_components_seen": int(fragmentation_counts.max()) if len(fragmentation_counts) else 0,
    }
}, indent=2))

summary_by_max_gap = {}
for max_gap in CFG.MAX_GAP_GRID:
    summary = meters[max_gap].summary()
    delta = summary["pq"] - CFG.BASELINE_PQ
    summary_by_max_gap[str(max_gap)] = {**summary, "delta_vs_baseline": delta}
    print(json.dumps({"max_gap": max_gap, **summary, "delta_vs_baseline": round(delta, 6)}, indent=2))

manifest = {
    "experiment_id": f"v56_gap_closure_probe_fold{CFG.FOLD}",
    "purpose": "WINNING_STRATEGY_2026-08-29.md §6.12優先度3。skeleton_gap_closeが"
               "OOF PQを改善するか、標準refiner・fold{CFG.FOLD}でmax_gapスイープにより検証",
    "fold": CFG.FOLD,
    "bridge_width": CFG.BRIDGE_WIDTH,
    "max_gap_grid": list(CFG.MAX_GAP_GRID),
    "reference_baseline_pq": CFG.BASELINE_PQ,
    "fragmentation_diagnostic": {
        "n_proposals": int(len(fragmentation_counts)),
        "multi_component_fraction": float((fragmentation_counts > 1).mean()) if len(fragmentation_counts) else 0.0,
    },
    "summary_by_max_gap": summary_by_max_gap,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}
report_path = CFG.ARTIFACT_ROOT / f"v56_gap_closure_probe_fold{CFG.FOLD}_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\\n保存: {report_path}")
print("\\n次: fold0で有望なmax_gapが見つかれば、fold1・fold2でも同じ検証を行い、"
      "「単一foldのbestを一般化しない」という規律に従って独立2fold以上での符号一致を確認する。")
"""
        ),
    ]
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["0", "1", "2"], help="評価対象fold")
    args = parser.parse_args()
    fold = int(args.fold)

    cells = build_cells(fold)
    for index, cell in enumerate(cells):
        cell["id"] = f"v56-gapclose-f{fold}-{index:02d}"

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

    output = ROOT / f"solar_filament_v56_gap_closure_probe_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
