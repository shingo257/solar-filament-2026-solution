"""定性比較図 + fold0詳細監査(TP/FP/FN/SQ/RQ/IoU分布/Dice分布/one-to-many/many-to-one)
のノートブックビルダー。

## 背景

技術報告書(`report/main.tex`)の2回目査読(2026-08-30)で、以下2点が指摘された:

  1. 公式ルーブリック(定量70%)が求めるDice/IoU分布・one-to-many/many-to-one
     分布の報告が不足している
  2. edge-attention refinerの効果を実証する定性的な比較図(入力/GT/raw/
     plain-refiner/edge-refiner)がFigure 2の1例だけでは足りない、
     成功例・失敗例(split/merge/細いfilament/端部欠落)を追加すべき

このスクリプトが生成するnotebookは、この日のColabセッションで対話的に
実行し実際に動作確認済みのコードを、単一の再実行可能な形にまとめたもの
(Colab接続がセッション途中で切断されたため、`colab-mcp`経由の対話実行から
「1回で最後まで回せるnotebook」方式に切り替えた)。

## 生成される成果物

  - `mediseg_refiner_fold0_resume.pt`と`mediseg_edgeattnet_refiner_fold0_resume.pt`
    を使い、fold0 OOF(236画像)全体でTP/FP/FN/SQ/RQ・IoU分布(中央値/四分位)・
    Dice分布(中央値/四分位、IoUから2*IoU/(1+IoU)で導出)・one-to-many/
    many-to-one件数を両refinerで測定 → `v51_fold0_detailed_audit_manifest.json`
    (実際に2026-08-30に一度実行し値を確認済み: 標準refiner PQ=0.439288
    ≈既知値と完全一致、EdgeAttNet refiner PQ=0.445492≈既知値と完全一致)
  - 改善例(1画像に3個体、全て大幅改善)+失敗例(GT3個体に対し生き残った
    予測1個、マージ/部分検出の疑い)の4パネル(GT/raw/plain/edge)比較図を
    `qualitative_comparison.png`として保存

## 使い方

    python scratch/build_qualitative_analysis.py

生成物: solar_filament_qualitative_analysis_colab.ipynb
Colabで上から順に実行するだけで、上記2つの成果物が
`solar_filament_v30/artifacts/`と`/content/`に保存される。
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_qualitative_analysis_colab.ipynb"
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")

assert "class MediSegEdgeAttUNet" in MODELS, "MediSegEdgeAttUNet が埋め込みソースに見つからない"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# 定性比較図 + fold0詳細監査(査読対応、2026-08-30)

技術報告書の2回目査読で指摘された(1)Dice/IoU分布・one-to-many/many-to-one
分布の欠如、(2)定性的な比較図の不足、に対応するnotebook。

(A) セットアップ → (B) fold0 OOF全体でのTP/FP/FN/SQ/RQ・分布の詳細監査
(標準refiner・EdgeAttNet refiner両方) → (C) 改善例+失敗例の4パネル比較図。

GPU必須。A100/T4どちらでも動く(A100なら(B)は数分)。
"""
    ),
    markdown("## (A) セットアップ"),
    code(
        """# Drive接続・設定・COCO/fold読み込み
import json, math, os, random, shutil, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm

drive.mount("/content/drive")

class CFG:
    SEED = 3407
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    V34_CACHE = Path("/content/drive/MyDrive/solar_filament_v34/prediction_cache/oof")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_qual")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    CROP_SIZE = 320
    CONF = 0.26
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    FOLD = 0

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)

def find_dataset_root(base):
    return list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))[0].parents[2]

TRAIN_JSON = next(find_dataset_root(CFG.DATA_CACHE).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)
images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)
fold_frame = pd.read_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))

required = {
    "fold0 標準refiner": CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt",
    "fold0 EdgeAttNet refiner": CFG.ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold0_resume.pt",
    "fold0 YOLO候補キャッシュ": CFG.V34_CACHE / "yolo_fold_0",
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")

for name in tqdm(train_files, desc="前処理画像をローカルへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)

assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({"GPU": torch.cuda.get_device_name(0), "観測": len(train_files)})
"""
    ),
    code("# MediSegUNet / EdgeGuidedMHSA / MediSegEdgeAttUNet 定義\n" + MODELS),
    code(
        """# fold0 refiner(標準・EdgeAttNet)をロード
standard_refiner = MediSegUNet(3, 1, False).cuda().eval()
std_ckpt = torch.load(CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt", map_location="cuda", weights_only=False)
standard_refiner.load_state_dict(std_ckpt.get("best_model") or std_ckpt["model"], strict=True)

edge_refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
edge_ckpt = torch.load(CFG.ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold0_resume.pt", map_location="cuda", weights_only=False)
edge_refiner.load_state_dict(edge_ckpt.get("best_model") or edge_ckpt["model"], strict=True)
print("refiners loaded: standard + edge-attention (fold0)")
"""
    ),
    code(
        """# 排他化・採点インフラ
def load_yolo_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))

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
def refine_with(refiner, image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= 0.45
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

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in annotations[image_id] if not ann.get("iscrowd", 0)]

FOLD_CACHE = CFG.V34_CACHE / f"yolo_fold_{CFG.FOLD}"
oof_files = [name for name in train_files if fold_of_file[name] == CFG.FOLD]
print("排他化・採点インフラ定義完了。fold0 OOF画像数:", len(oof_files))
"""
    ),
    markdown(
        """## (B) fold0 OOF詳細監査

TP/FP/FN/SQ/RQ、IoU分布(中央値・四分位)、Dice分布(IoUから導出)、
one-to-many/many-to-one件数を、標準refiner・EdgeAttNet refinerそれぞれで
fold0 OOF全236画像に対して測定する。"""
    ),
    code(
        """# fold0 OOF全体で両refinerの詳細統計を測定
def full_stats(refiner_model, desc):
    tp_ious, fp_count, fn_count = [], 0, 0
    one_to_many, many_to_one = 0, 0
    for name in tqdm(oof_files, desc=desc):
        masks, confidences = load_yolo_cache(FOLD_CACHE / f"{Path(name).stem}.npz")
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences)) if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        refined = [refine_with(refiner_model, image, m) for m in proposals]
        predictions = apply_ownership(refined, scores)
        pred_rles = []
        if predictions:
            encoded = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
            pred_rles = encoded if isinstance(encoded, list) else [encoded]
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, pred_rles)
            n_gt, n_pred = len(truth), len(pred_rles)
            if n_gt == 0:
                fp_count += n_pred
                continue
            if n_pred == 0:
                fn_count += n_gt
                continue
            hits = matrix > 0.5
            scores_tp = matrix[hits].tolist()
            tp_ious.extend(scores_tp)
            fp_count += int((hits.sum(axis=0) == 0).sum())
            fn_count += int((hits.sum(axis=1) == 0).sum())
            one_to_many += int((hits.sum(axis=1) > 1).sum())
            many_to_one += int((hits.sum(axis=0) > 1).sum())
    denom = len(tp_ious) + 0.5 * fp_count + 0.5 * fn_count
    pq = float(sum(tp_ious) / denom) if denom > 0 else 0.0
    sq = float(np.mean(tp_ious)) if tp_ious else 0.0
    rq = float(len(tp_ious) / denom) if denom > 0 else 0.0
    dice_vals = [2 * v / (1 + v) for v in tp_ious]
    return {
        "pq": pq, "sq": sq, "rq": rq,
        "tp": len(tp_ious), "fp": fp_count, "fn": fn_count, "denominator": denom,
        "iou_median": float(np.median(tp_ious)) if tp_ious else None,
        "iou_q1": float(np.percentile(tp_ious, 25)) if tp_ious else None,
        "iou_q3": float(np.percentile(tp_ious, 75)) if tp_ious else None,
        "dice_median": float(np.median(dice_vals)) if dice_vals else None,
        "dice_q1": float(np.percentile(dice_vals, 25)) if dice_vals else None,
        "dice_q3": float(np.percentile(dice_vals, 75)) if dice_vals else None,
        "one_to_many_gt": one_to_many, "many_to_one_pred": many_to_one,
    }

standard_stats = full_stats(standard_refiner, "標準refiner採点(fold0 OOF)")
edge_stats = full_stats(edge_refiner, "EdgeAttNet refiner採点(fold0 OOF)")

print("=== 標準refiner ===")
print(json.dumps(standard_stats, ensure_ascii=False, indent=2))
print("\\n=== EdgeAttNet refiner ===")
print(json.dumps(edge_stats, ensure_ascii=False, indent=2))

detail_manifest = {
    "experiment": "v51_fold0_detailed_audit",
    "purpose": "査読(2026-08-30)対応: edge-attention runsのper-image内訳・Dice/IoU分布・"
               "one-to-many/many-to-one分布を実測(fold0 OOF、236画像、既存V34キャッシュ再利用)",
    "fold": 0, "n_images": len(oof_files),
    "standard_refiner": standard_stats,
    "edge_attention_refiner": edge_stats,
    "note": "one_to_many_gt/many_to_one_predが両方0なのは、排他化ステップ(drop_contained+"
            "greedy_disjoint)が予測集合を強制的に非重複にしているため。",
}
out_path = CFG.ARTIFACT_ROOT / "v51_fold0_detailed_audit_manifest.json"
out_path.write_text(json.dumps(detail_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("\\nsaved:", out_path)
"""
    ),
    markdown(
        """## (C) 定性比較図: 改善例 + 失敗例

改善例: 1画像に3個体、EdgeAttNet refinerがいずれも標準refinerより大きく
GTに近づく(事前スキャンで確認済み: gain +0.22/+0.18/+0.17)。

失敗例: GTは3個体あるが生き残った予測は1個のみ(マージまたは部分検出の
疑い、IoU=0.25と近接ミス寄りの値)。"""
    ),
    code(
        """# 改善例・失敗例の4パネル(GT/raw/plain-refiner/edge-refiner)比較図
import matplotlib.pyplot as plt

def decode_gt_union(image_id):
    truth = gt_rles[image_id]
    if not truth:
        return None
    decoded = [mask_utils.decode(rle).astype(bool) for rle in truth]
    union = np.zeros_like(decoded[0])
    for d in decoded:
        union |= d
    return union, decoded

def render_case(ax_row, file_name, image_id, pad=40, title_prefix=""):
    image = cv2.imread(str(CFG.PREPROC_DIR / file_name), cv2.IMREAD_COLOR)
    masks, confidences = load_yolo_cache(FOLD_CACHE / f"{Path(file_name).stem}.npz")
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences)) if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
    proposals = [masks[i] for i in keep]
    scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
    raw_pred = apply_ownership(proposals, scores) if proposals else []
    std_pred = apply_ownership([refine_with(standard_refiner, image, m) for m in proposals], scores) if proposals else []
    edge_pred = apply_ownership([refine_with(edge_refiner, image, m) for m in proposals], scores) if proposals else []
    print(f"  {file_name} / {image_id}: raw={len(raw_pred)} std={len(std_pred)} edge={len(edge_pred)} preds in full image")

    gt_union, gt_list = decode_gt_union(image_id)
    ys, xs = np.where(gt_union)
    y0, y1 = max(0, ys.min() - pad), min(image.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad)
    crop_rgb = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

    panels = [("Ground truth", gt_list), ("Raw detector", raw_pred), ("Plain refiner", std_pred), ("Edge-attn refiner", edge_pred)]
    for ax, (label, mask_list) in zip(ax_row, panels):
        ax.imshow(crop_rgb)
        drawn = 0
        for m in mask_list:
            crop_mask = m[y0:y1, x0:x1].astype(np.uint8)
            if crop_mask.sum() == 0:
                continue
            contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                pts = contour.squeeze(1)
                if len(pts) > 2:
                    drawn += 1
                    ax.plot(pts[:, 0], pts[:, 1],
                            color="#39ff6a" if "Edge" in label else ("#ffcf33" if label == "Ground truth" else "#38bdf8"),
                            linewidth=1.8)
        if drawn == 0 and label != "Ground truth":
            ax.text(0.5, 0.5, "no prediction\\nin this crop", color="white", fontsize=9, ha="center", va="center",
                     transform=ax.transAxes, bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"))
        ax.set_title(f"{title_prefix}{label}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(2, 4, figsize=(13, 7), dpi=150)
render_case(axes[0], "20120424072414Th.jpeg", "010401-20120424072414Th", pad=60, title_prefix="(a) ")
render_case(axes[1], "20190602025430Lh.jpeg", "010102-20190602025430Lh", pad=80, title_prefix="(b) ")
fig.text(0.02, 0.72, "Improvement\\ncase", fontsize=10, fontweight="bold", rotation=90, va="center")
fig.text(0.02, 0.28, "Failure\\ncase", fontsize=10, fontweight="bold", rotation=90, va="center")
plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig("/content/qualitative_comparison.png", dpi=150, bbox_inches="tight")
DRIVE_FIG = CFG.ARTIFACT_ROOT / "qualitative_comparison.png"
plt.savefig(DRIVE_FIG, dpi=150, bbox_inches="tight")
plt.show()
print("saved: /content/qualitative_comparison.png and", DRIVE_FIG)
"""
    ),
    markdown(
        """## 次のステップ

このnotebookが生成した2つの成果物をダウンロードし、リポジトリへ配置する:

  - `v51_fold0_detailed_audit_manifest.json` → `output/v51/`
  - `qualitative_comparison.png` → `report/figures/`

その後、`report/main.tex`のTable 1にedge-attention行のTP/FP/FN/SQ/RQを
追加し、Figure 2の下(または新しいFigure)に定性比較図を挿入する。"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"qual-{index:02d}"

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
print(f"wrote {OUTPUT} ({len(cells)} cells)")
