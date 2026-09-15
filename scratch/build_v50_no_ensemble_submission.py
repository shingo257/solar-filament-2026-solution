"""V50 最終提出ノートブックのビルダー — アンサンブル無効化版（診断実験）。

## 背景・目的

2026-08-29、ユーザーから「ホームページに載せた全fold OOF PQ 0.4426は、
実際のPublic LB 0.37とどの程度比較可能なのか」と問われ、`build_v35_final_submission.py`
を精査した結果、**両者は同じ指標(公式PQ)だが構造的に異なる条件で測っている**ことが
判明した:

  - 0.4426 (全fold OOF): fold0/1/2それぞれの単一モデルを、そのモデル自身の
    学習から除外された観測だけで評価。3モデルのアンサンブルは一切関与しない。
  - 0.37 (Public LB): 実際の提出では、3つのfoldモデル全部を動かし、
    IoUクラスタリング＋多数決投票 (`ensemble_proposals`、V30本体からそのまま
    移植) で統合してから提出している。このアンサンブル統合ロジックは
    **fold間の統合はOOFでは検証できない**ため(fold1/fold2モデルはfold0の画像を
    学習時に見ているため、fold0データでの3モデル統合テストはリークになる)、
    V30から一度も妥当性検証されないまま使われ続けている。

この約0.07のギャップの一因が、この未検証のアンサンブル統合ロジックそのもの
である可能性を検証するため、**アンサンブル統合を無効化し、OOF実験で確定した
単一モデル+shrink+drop_contained_then_area_askをそのままtest推論に適用した
バリアントを一度提出し、Public LBの差分を直接確認する。**

## 設計

`build_v35_final_submission.py` との差分は [3/9]〜[4/9] のみ:

  - YOLOモデルは3-fold全部ではなく、**OOF PQが最良だったfold2単体のみ**
    (fold2 OOF PQ 0.4471、fold0 0.4393・fold1 0.4415より高い。アンサンブルに
    とって最も厳しい比較対象になるよう、単純に一番良い個体を選ぶ)。
  - `ensemble_proposals`(IoUクラスタリング＋多数決投票)は使わず、fold2モデルの
    生出力をそのままrefiner・排他化へ渡す。
  - refiner・排他化は V35 と完全に同一 (all-data refiner, shrink@0.45,
    drop_contained_then_area_asc) — 差分をアンサンブル統合の有無だけに絞る。

## 判定

  - Public LBが0.37を上回る → アンサンブル統合が実際には害になっていた
    可能性。統合ロジックの見直しが最優先課題になる。
  - Public LBが0.37を大きく下回る → アンサンブル統合は実際に有効に機能して
    おり、0.4426との残差は主に学習/テスト間の分布差(年代・サイト)に起因する
    可能性が高まる。pseudo-labeling路線の正当性がむしろ補強される。
  - Public LBが0.37に近い → どちらの要因も支配的ではなく、ギャップの原因は
    それ以外(分布差と統合ロジックの両方が小さく効いている等)にある可能性。

いずれの結果も情報量があり、「デメリットのない」診断実験である
(ユーザー承認: 2026-08-29)。
"""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v50_no_ensemble_submission_colab.ipynb"
MEDISEG = (ROOT / "src" / "mediseg_models.py").read_text(encoding="utf-8")


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# 太陽フィラメント V50 — アンサンブル無効化版（診断実験）

V35(`submission_v35_final_pq_aligned.csv`, Public 0.37)との唯一の違いは、
3-foldのIoUクラスタリング＋多数決投票アンサンブル(`ensemble_proposals`、
V30本体から一度も検証されずに継承)を無効化し、**OOF PQが最良だった
fold2単体のモデルのみ**を使う点。refiner(shrink@0.45)・排他化
(drop_contained_then_area_asc)はV35と完全に同一。

目的: 全fold OOF PQ 0.4426とPublic LB 0.37の間にある約0.07のギャップの
うち、どれだけがこの未検証のアンサンブル統合ロジックに起因するかを、
実際の提出スコアの差分で直接確認する診断実験(2026-08-29)。

GPU必須（YOLO推論・refiner推論）。T4で可。
`MediSegUNet` は `src/mediseg_models.py` をノートへ埋め込んであるので、
Colab へ別ファイルをアップロードする必要はない。上から順に実行する。
"""
    ),
    code(
        """# [1/9] 依存関係・Drive接続と前提artifactの確認
%pip install -q "ultralytics>=8.3,<9" pycocotools

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
from ultralytics import YOLO

drive.mount("/content/drive")

class CFG:
    V30_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = V30_ROOT / "competition_data"
    ARTIFACT_ROOT = V30_ROOT / "artifacts"
    RUNS_ROOT = V30_ROOT / "runs"
    PREPROC_CACHE = V30_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v50_no_ensemble")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    CROP_SIZE = 320
    YOLO_IMGSZ = 1280
    PREDICT_IOU = 0.55
    SINGLE_FOLD = 2  # OOF PQ最良(0.4471)。アンサンブルにとって最も厳しい比較対象

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)

required = {
    f"YOLO fold{CFG.SINGLE_FOLD}": CFG.RUNS_ROOT / f"yolo_fold_{CFG.SINGLE_FOLD}/weights/best.pt",
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
        """# [2/9] 提出パラメータ (V35と同一。差分はアンサンブル統合の有無のみ)
#
# 出典: output/v35/pipeline_stack.json の "best_config"

SUBMISSION_CONF = 0.26
SUBMISSION_OWNERSHIP_RULE = "drop_contained_then_area_asc"
REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45
MIN_AREA = 20

print({
    "conf": SUBMISSION_CONF, "ownership": SUBMISSION_OWNERSHIP_RULE,
    "refiner": f"{REFINER_METHOD}@{REFINER_THRESHOLD}", "min_area": MIN_AREA,
    "single_fold": CFG.SINGLE_FOLD,
})
print("★ V35との唯一の差分はアンサンブル無効化(fold2単体)であることを確認 ★")
"""
    ),
    code(
        "# [3/9] MediSegUNet定義とYOLO読み込み(fold2単体のみ、アンサンブルなし)\n"
        "# src/mediseg_models.py を V30 / fold1-2 学習ノートと同じく埋め込む。\n"
        "# Colab へ別ファイルをアップロードする必要はない。\n\n"
        + MEDISEG
        + "\n\n"
        + """assert "MediSegUNet" in globals(), "MediSegUNet が定義されていません"

yolo_model = YOLO(str(CFG.RUNS_ROOT / f"yolo_fold_{CFG.SINGLE_FOLD}/weights/best.pt"))
refiner = MediSegUNet(3, 1, False).cuda().eval()
refiner_path = CFG.ARTIFACT_ROOT / "mediseg_refiner_all_data_resume.pt"
try:
    refiner_checkpoint = torch.load(refiner_path, map_location="cuda", weights_only=False)
except TypeError:
    refiner_checkpoint = torch.load(refiner_path, map_location="cuda")
refiner.load_state_dict(refiner_checkpoint.get("best_model") or refiner_checkpoint["model"],
                        strict=True)
print("refiner best crop Dice (学習時参考値):", refiner_checkpoint.get("best_dice"))
print({"single_fold_model": CFG.SINGLE_FOLD, "refiner_params": sum(p.numel() for p in refiner.parameters())})
"""
    ),
    code(
        """# [4/9] test画像一覧と単一モデル推論(アンサンブルなし。V30本体のensemble_proposalsは使わない)
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"annotation JSONが見つからない: {base}")
    return matches[0].parents[2]

DATA_ROOT = find_dataset_root(CFG.DATA_CACHE)
TEST_DIR = next(DATA_ROOT.rglob("test_images"))
test_files = sorted(p.name for p in TEST_DIR.glob("*.jpeg"))
if len(test_files) != 180:
    raise RuntimeError(f"test画像が180枚ではありません: {len(test_files)}")
print({"test画像": len(test_files), "test_dir": str(TEST_DIR)})

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

def single_fold_proposals(image_path):
    \"\"\"アンサンブルなし。fold2単体の生出力をそのまま返す(V35のensemble_proposalsの代わり)。\"\"\"
    result = yolo_model.predict(
        str(image_path), imgsz=CFG.YOLO_IMGSZ,
        conf=max(0.08, SUBMISSION_CONF * 0.75),
        iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False,
    )[0]
    masks = result_masks(result)
    confidences = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes is not None else []
    keep = [(score, mask) for score, mask in zip(confidences, masks) if score >= SUBMISSION_CONF]
    return keep
"""
    ),
    code(
        """# [5/9] refiner適用 (V35・OOF実験と同一定義。shrink固定。変更なし)
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
    return refined & proposal  # shrink固定(REFINER_METHOD="shrink")
"""
    ),
    code(
        """# [6/9] 排他化 (output/v35/run_pipeline_stack.py の apply_ownership と同一。変更なし)
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
        """# [7/9] test 180枚に対して推論し、submission CSVを生成(単一モデル、アンサンブルなし)
rows, audit = [], []
for file_name in tqdm(test_files, desc="test推論(fold2単体)"):
    image_path = CFG.PREPROC_DIR / file_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"前処理画像を読めない: {image_path}")
    proposals = single_fold_proposals(image_path)        # (confidence, mask) のリスト。アンサンブルなし
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
EXPERIMENT_ID = "v50_no_ensemble_fold2_only"
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
        """# [8/9] 公式RLE厳密検証 (提出前チェックリスト §4.4 に対応。V35と同一)
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
        """# [9/9] manifest保存
import hashlib

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "experiment_id": EXPERIMENT_ID,
    "purpose": "全fold OOF PQ(0.4426)とPublic LB(0.37, V35)の差分のうち、"
               "未検証のfold間アンサンブル統合(ensemble_proposals)がどれだけ寄与するかを診断",
    "diff_vs_v35": "アンサンブル統合(IoUクラスタリング+多数決投票)を無効化し、"
                   "OOF PQ最良のfold2単体モデルのみを使用。refiner・排他化はV35と完全同一",
    "single_fold": CFG.SINGLE_FOLD,
    "single_fold_oof_pq": 0.4471,
    "metric_definition_version": "official-v1-2026-08-15",
    "submission_conf": SUBMISSION_CONF,
    "ownership_rule": SUBMISSION_OWNERSHIP_RULE,
    "refiner_method": REFINER_METHOD,
    "refiner_threshold": REFINER_THRESHOLD,
    "min_area": MIN_AREA,
    "compare_against": {"experiment_id": "v35_final_pq_aligned", "kaggle_public_score": 0.37},
    "rows": len(checked),
    "submission_sha256": sha256_of(OUTPUT_PATH),
    "kaggle_public_score": "unknown",
    "decision": "pending",
}
(CFG.ARTIFACT_ROOT / f"{EXPERIMENT_ID}_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("\\n次: ローカルへ submission_v50_no_ensemble_fold2_only.csv をダウンロードし、"
      "kaggle competitions submit で提出。")
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"v50-{index:02d}"
    if cell["cell_type"] != "code":
        continue
    stripped = re.sub(r"^[%!].*$", "", cell["source"], flags=re.M)
    ast.parse(stripped)

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

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUTPUT} ({len(cells)} cells)")
