"""V35 最終提出ノートブックのビルダー（骨格。重畳検証の結果確定後にパラメータを埋める）。

## なぜこれが必要か

V30本体のtest推論 (`build_v30_prize_colab.py` セル[11/12]) を精査した結果、
今回のOOF実験で確定した知見が**反映されていない**ことが分かった。

  - refiner適用: V30本体は `probability >= threshold` に加えて
    「31px膨張した支持領域に限定」「YOLO候補と0.8:0.2でブレンド」という
    独自の第4の方式を使っている。これは V31 や今回の OOF 実験
    (`run_refiner_pq_allfold.py`) で比較した shrink / guided3 / guided7 の
    どれとも異なり、比較検証を一度も受けていない。
  - 排他化: V30本体のセルには排他化ロジックが存在しない。重複除去は
    別スクリプト (`build_v30_disjoint_masks.py`) で後から行われていた。

一方、**3-fold の統合（fold間IoUクラスタリング＋多数決投票、`ensemble_proposals`）は
そのまま踏襲する。** これは OOF では検証できない部分だと HANDOVER §2.1.7 に
明記した通りで、変更する根拠がない。

## このノートブックの設計

1. fold間統合 (`ensemble_proposals`): V30本体からそのまま移植。変更しない。
2. refiner適用: `run_refiner_pq_allfold.py` の `build_variant` (shrink方式) を使う。
   OOF実験で全fold・リークなしで確定した方式に統一する。
3. 排他化: `run_pipeline_stack.py` の `apply_ownership` を使う。規則は
   重畳検証 (`pipeline_stack.json`) の結果で確定する。
4. refinerの重みは all-data 版 (`mediseg_refiner_all_data_resume.pt`) を使う。
   これは test 推論専用で、OOF評価には使えない (§18.2)。

## 実行前に必ずすること

`CFG` 内の `REFINER_THRESHOLD` / `CONF` / `OWNERSHIP_RULE` は
`output/v35/pipeline_stack.json` の `best_config` で上書きすること。
現在入っている値は重畳検証の暫定途中経過であり、確定値ではない。
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v35_final_submission_colab.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# 太陽フィラメント V35 — 最終提出（PQ整合版）

**実行前に `[2/9]` の `SUBMISSION_CONF` を `output/v35/pipeline_stack.json` の
`best_config` で更新すること。** 現在の値は重畳検証の暫定途中経過。

このノートブックは V30 の 3-fold 統合ロジックをそのまま使い、refiner適用と
排他化だけを、今回の全fold OOF実験（リークなし）で確定した実装に差し替える。

- fold間統合: V30本体と同一（IoUクラスタリング＋多数決投票）。OOFでは検証不能な部分
- refiner: `shrink` 方式に統一（V30本体が使っていた未検証の独自ブレンド方式から変更）
- 排他化: `output/v35/pipeline_stack.json` で確定した規則

GPU必須（YOLO推論・refiner推論）。T4で可。
"""
    ),
    code(
        """# [1/9] Drive接続と前提artifactの確認
import json, math, shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm

drive.mount("/content/drive")

class CFG:
    V30_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = V30_ROOT / "competition_data"
    ARTIFACT_ROOT = V30_ROOT / "artifacts"
    RUNS_ROOT = V30_ROOT / "runs"
    PREPROC_CACHE = V30_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v35_final")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    CROP_SIZE = 320
    YOLO_IMGSZ = 1280
    PREDICT_IOU = 0.55

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)

required = {
    "YOLO fold0": CFG.RUNS_ROOT / "yolo_fold_0/weights/best.pt",
    "YOLO fold1": CFG.RUNS_ROOT / "yolo_fold_1/weights/best.pt",
    "YOLO fold2": CFG.RUNS_ROOT / "yolo_fold_2/weights/best.pt",
    "all-data refiner": CFG.ARTIFACT_ROOT / "mediseg_refiner_all_data_resume.pt",
    "competition data": CFG.DATA_CACHE,
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({k: "OK" for k in required}, {"GPU": torch.cuda.get_device_name(0)})
"""
    ),
    code(
        """# [2/9] ★ 提出パラメータ (重畳検証の結果でここを更新してから実行する) ★
#
# 出典: output/v35/pipeline_stack.json の "best_config"
# 現在の値は 2026-08-16 時点、707観測中350観測までの暫定途中経過。
# 確定後、この形式の文字列 "confX.XX|rule_name" をパースして下2つに反映すること。

SUBMISSION_CONF = 0.26                              # TODO: 確定値で上書き
SUBMISSION_OWNERSHIP_RULE = "drop_contained_then_area_asc"  # TODO: 確定値で上書き
REFINER_METHOD = "shrink"                           # 確定済み (HANDOVER §2.1.8)
REFINER_THRESHOLD = 0.45                            # 確定済み。0.45〜0.65はほぼフラット
MIN_AREA = 20                                       # OOF実験と統一

print({
    "conf": SUBMISSION_CONF, "ownership": SUBMISSION_OWNERSHIP_RULE,
    "refiner": f"{REFINER_METHOD}@{REFINER_THRESHOLD}", "min_area": MIN_AREA,
})
print("★ 上記が output/v35/pipeline_stack.json の best_config と一致するか必ず確認 ★")
"""
    ),
    code(
        """# [3/9] MediSegUNet定義とYOLO読み込み
!pip -q install ultralytics pycocotools

from ultralytics import YOLO
import sys
sys.path.insert(0, str(CFG.V30_ROOT))  # ローカルsrc配置に合わせて調整すること

exec(open("/content/mediseg_models.py").read()) if Path("/content/mediseg_models.py").exists() else None
# 上記が失敗する場合は src/mediseg_models.py の内容をこのセルに直接貼り付けること。
# (ローカルからColabへ同期する手段はプロジェクトの運用に合わせて選ぶ)

yolo_models = {
    fold: YOLO(str(CFG.RUNS_ROOT / f"yolo_fold_{fold}/weights/best.pt"))
    for fold in (0, 1, 2)
}
refiner = MediSegUNet(3, 1, False).cuda().eval()
refiner_checkpoint = torch.load(CFG.ARTIFACT_ROOT / "mediseg_refiner_all_data_resume.pt",
                                map_location="cuda")
refiner.load_state_dict(refiner_checkpoint.get("best_model") or refiner_checkpoint["model"],
                        strict=True)
print("refiner best crop Dice (学習時参考値):", refiner_checkpoint.get("best_dice"))
"""
    ),
    code(
        """# [4/9] test画像一覧とfold間統合 (V30本体から変更なし)
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"annotation JSONが見つからない: {base}")
    return matches[0].parents[2]

DATA_ROOT = find_dataset_root(CFG.DATA_CACHE)
TEST_DIR = next(DATA_ROOT.rglob("test_images"))
test_files = sorted(p.name for p in TEST_DIR.glob("*.jpeg"))
print({"test画像": len(test_files)})

for file_name in tqdm(test_files, desc="前処理画像をローカルへ"):
    cached = CFG.PREPROC_CACHE / file_name
    local = CFG.PREPROC_DIR / file_name
    if not local.exists():
        if cached.exists():
            shutil.copy2(cached, local)
        else:
            raise FileNotFoundError(f"前処理キャッシュがない: {cached}")

def result_masks(result):
    if result.masks is None:
        return []
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    return output

def mask_iou(first, second):
    intersection = np.logical_and(first, second).sum()
    return intersection / max(np.logical_or(first, second).sum(), 1)

def ensemble_proposals(image_path):
    \"\"\"V30本体と同一。fold間の統合はOOFで検証できないため変更しない (HANDOVER §2.1.7)。\"\"\"
    raw = []
    for fold, model in yolo_models.items():
        result = model.predict(
            str(image_path), imgsz=CFG.YOLO_IMGSZ,
            conf=max(0.08, SUBMISSION_CONF * 0.75),
            iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False,
        )[0]
        masks = result_masks(result)
        confidences = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes is not None else []
        raw.extend((float(score), fold, mask) for score, mask in zip(confidences, masks))
    clusters = []
    for score, fold, mask in sorted(raw, reverse=True, key=lambda item: item[0]):
        best_index, best_overlap = None, 0.0
        for index, cluster in enumerate(clusters):
            if fold in cluster["folds"]:
                continue
            overlap = mask_iou(mask, cluster["representative"])
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is not None and best_overlap >= 0.30:
            cluster = clusters[best_index]
            cluster["masks"].append(mask)
            cluster["scores"].append(score)
            cluster["folds"].add(fold)
            votes = np.stack(cluster["masks"]).sum(0)
            cluster["representative"] = votes >= math.ceil(len(cluster["masks"]) / 2)
        else:
            clusters.append({"masks": [mask], "scores": [score], "folds": {fold}, "representative": mask})
    proposals = []
    for cluster in clusters:
        votes = len(cluster["folds"])
        mean_conf = float(np.mean(cluster["scores"]))
        if votes >= 2 or mean_conf >= max(0.42, SUBMISSION_CONF + 0.12):
            vote_sum = np.stack(cluster["masks"]).sum(0)
            proposals.append((mean_conf, vote_sum >= math.ceil(len(cluster["masks"]) / 2)))
    return proposals
"""
    ),
    code(
        """# [5/9] refiner適用 (OOF実験と同一の shrink 方式に統一)
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
    \"\"\"OOF実験 (run_refiner_pq_allfold.build_variant) と同一定義。shrink固定。\"\"\"
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= REFINER_THRESHOLD
    if REFINER_METHOD == "shrink":
        return refined & proposal
    kernel = 3 if REFINER_METHOD == "guided3" else 7
    support = cv2.dilate(proposal.astype(np.uint8), np.ones((kernel, kernel), np.uint8)).astype(bool)
    return refined & support
"""
    ),
    code(
        """# [6/9] 排他化 (output/v35/run_pipeline_stack.py の apply_ownership と同一)
def greedy_disjoint(masks, order, min_area=MIN_AREA):
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

def apply_ownership(masks, scores, rule=SUBMISSION_OWNERSHIP_RULE):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    if rule == "conf_first":
        return greedy_disjoint(masks, list(np.argsort(-scores)))
    if rule == "area_asc":
        return greedy_disjoint(masks, list(np.argsort(areas)))
    if rule == "drop_contained_then_area_asc":
        subset = drop_contained_indices(masks)
        return greedy_disjoint(masks, sorted(subset, key=lambda i: areas[i]))
    raise ValueError(rule)
"""
    ),
    code(
        """# [7/9] test 180枚に対して推論し、submission CSVを生成
rows, audit = [], []
for file_name in tqdm(test_files, desc="test推論"):
    image_path = CFG.PREPROC_DIR / file_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    proposals = ensemble_proposals(image_path)          # (confidence, mask) のリスト。fold統合済み
    scores = np.array([c for c, _ in proposals], dtype=np.float32)
    refined = [refine_proposal(image, m) for _, m in proposals]
    final_masks = apply_ownership(refined, scores)

    image_id = Path(file_name).stem
    for index, mask in enumerate(final_masks, start=1):
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rows.append({"filament_id": f"{image_id}_{index}",
                     "segmentation_rle": encoded["counts"].decode("utf-8")})
    audit.append({"file_name": file_name, "proposals": len(proposals), "submitted": len(final_masks)})

submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
EXPERIMENT_ID = "v35_final_pq_aligned"  # 実行のたびに更新する (HANDOVER §33)
OUTPUT_PATH = Path(f"/content/submission_{EXPERIMENT_ID}.csv")
submission.to_csv(OUTPUT_PATH, index=False, lineterminator="\\n")
DRIVE_OUT = CFG.ARTIFACT_ROOT / OUTPUT_PATH.name
shutil.copy2(OUTPUT_PATH, DRIVE_OUT)
pd.DataFrame(audit).to_csv(CFG.ARTIFACT_ROOT / f"{EXPERIMENT_ID}_audit.csv", index=False)
print({"rows": len(submission), "images_with_predictions": len(set(audit_row['file_name'] for audit_row in audit if audit_row['submitted'] > 0)),
       "drive_copy": str(DRIVE_OUT)})
"""
    ),
    code(
        """# [8/9] 公式RLE厳密検証 (提出前チェックリスト §4.4 に対応)
checked = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
assert checked.columns.tolist() == ["filament_id", "segmentation_rle"]
assert len(checked) > 0
assert not checked.isna().any().any()
assert not checked["filament_id"].duplicated().any()
assert not checked["segmentation_rle"].str.strip().eq("").any()

test_ids = {Path(name).stem for name in test_files}
submitted_ids = {value.rsplit("_", 1)[0] for value in checked["filament_id"]}
assert submitted_ids <= test_ids

areas = []
occupancy = {}
for row in tqdm(checked.itertuples(index=False), total=len(checked), desc="decode + overlap検証"):
    decoded = mask_utils.decode({"size": [2048, 2048], "counts": row.segmentation_rle.encode("utf-8")})
    assert decoded.shape == (2048, 2048)
    area = int(decoded.sum())
    assert area > 0
    areas.append(area)
    image_id = row.filament_id.rsplit("_", 1)[0]
    occ = occupancy.setdefault(image_id, np.zeros((2048, 2048), dtype=bool))
    assert not np.logical_and(occ, decoded.astype(bool)).any(), f"重複: {image_id}"
    occ |= decoded.astype(bool)

print({"rows": len(checked), "images": len(submitted_ids), "area_min": min(areas),
       "area_median": int(np.median(areas)), "area_max": max(areas), "overlap_pixels": 0})
"""
    ),
    code(
        """# [9/9] manifest保存 (HANDOVER §34)
import hashlib

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "experiment_id": EXPERIMENT_ID,
    "metric_definition_version": "official-v1-2026-08-15",
    "submission_conf": SUBMISSION_CONF,
    "ownership_rule": SUBMISSION_OWNERSHIP_RULE,
    "refiner_method": REFINER_METHOD,
    "refiner_threshold": REFINER_THRESHOLD,
    "min_area": MIN_AREA,
    "source_pipeline_stack_json": "output/v35/pipeline_stack.json",
    "fold_ensemble": "V30本体のensemble_proposalsを変更せず流用 (OOF検証対象外)",
    "rows": len(checked),
    "submission_sha256": sha256_of(OUTPUT_PATH),
    "kaggle_public_score": "unknown",
    "decision": "pending",
}
(CFG.ARTIFACT_ROOT / f"{EXPERIMENT_ID}_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("\\n次: ローカルの tests/test_submission.py 相当の検証を通し、日次提出枠を確認してから提出。")
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"v35f-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
