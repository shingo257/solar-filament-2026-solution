import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v31_heldout_postprocess_ablation.ipynb"
MEDISEG = (ROOT / "src" / "mediseg_models.py").read_text(encoding="utf-8")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    md(
        """# 太陽フィラメント V31 — リークなしheld-out後処理アブレーション（再学習なし）

V30bは`0.35`まで改善しましたが、MediSeg精密化後に66万画素の個体間重複が発生し、
排他化で97 maskを除外しました。またV30の閾値調整は排他化前・単一foldであり、
最終提出の3-fold処理と一致していませんでした。

V31では保存済み重みだけを使い、YOLO fold 0と、fold 0を学習に使っていないMediSeg refinerで次を比較します。

- YOLO native mask
- MediSegでYOLO maskの内側だけを削る`shrink`
- YOLO境界から3pxまたは7pxだけ拡張を許す`guided`
- confidence、refiner threshold、最小面積

すべて**個体maskを排他化した後**にfold 0 held-outのHungarian個体Diceを計算します。再学習はありません。
最後に、held-out最良方式の3-fold ensemble、同条件のfold 0単体、YOLO native ensembleの3 CSVを生成し、
同一画像内の重複が0ピクセルであることを再復号検証します。

**Colab設定:** A100推奨（T4でも実行可）、Internet ON、標準RAMで十分です。High-RAMは不要です。Driveの`solar_filament_v30`成果物を再利用します。
"""
    ),
    code(
        """# [1/9] 必要パッケージをインストール
%pip install -q "ultralytics>=8.3,<9" pycocotools
"""
    ),
    code(
        """# [2/9] ライブラリ、Drive、設定
import gc
import json
import math
import os
import random
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")
drive.mount("/content/drive")

class CFG:
    SEED = 3407
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs"
    LOCAL_ROOT = Path("/content/solar_v31")
    LOCAL_DATA = LOCAL_ROOT / "competition_data"
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    IMGSZ = 1280
    CROP_SIZE = 320
    OUTPUT_SIZE = 2048
    CROP_CONTEXT = 1.65
    COMPACT_SIZE = 256
    PREDICT_CONF_FLOOR = 0.10
    PREDICT_IOU = 0.55
    CONF_GRID = (0.18, 0.22, 0.26, 0.30, 0.34)
    MIN_AREA_GRID = (20, 28, 60)
    REFINER_THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65)

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
CFG.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
assert torch.cuda.is_available(), "GPUランタイムを選択してください。"
print({"GPU": torch.cuda.get_device_name(0), "成果物": str(CFG.ARTIFACT_ROOT)})
"""
    ),
    code(
        """# [3/9] Driveの競技データ、注釈、fold定義を読み込む
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"注釈JSONが見つかりません: {base}")
    return matches[0].parents[2]

try:
    DATA_ROOT = find_dataset_root(CFG.LOCAL_DATA)
except FileNotFoundError:
    if CFG.LOCAL_DATA.exists():
        shutil.rmtree(CFG.LOCAL_DATA)
    print("DriveのキャッシュをColab SSDへコピーします...")
    shutil.copytree(CFG.DATA_CACHE, CFG.LOCAL_DATA)
    DATA_ROOT = find_dataset_root(CFG.LOCAL_DATA)

TRAIN_JSON = next(DATA_ROOT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
TRAIN_DIR = TRAIN_JSON.parent / "train_images"
TEST_DIR = next(DATA_ROOT.rglob("test_images"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

fold_path = CFG.ARTIFACT_ROOT / "grouped_folds.csv"
if not fold_path.exists():
    raise FileNotFoundError("V30のgrouped_folds.csvがありません。V30成果物を確認してください。")
fold_frame = pd.read_csv(fold_path)
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
train_files = sorted(file_to_ids)
test_files = sorted(path.name for path in TEST_DIR.glob("*.jpeg"))
assert set(train_files) == set(fold_of_file)

weight_paths = {fold: CFG.RUNS_ROOT / f"yolo_fold_{fold}" / "weights" / "best.pt" for fold in range(3)}
for path in weight_paths.values():
    if not path.exists():
        raise FileNotFoundError(path)
REFINER_PATH = CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt"
if not REFINER_PATH.exists():
    raise FileNotFoundError(REFINER_PATH)

print({
    "train画像": len(train_files), "test画像": len(test_files),
    "YOLO重み": {k: str(v) for k, v in weight_paths.items()},
    "refiner": str(REFINER_PATH),
})
"""
    ),
    code(
        """# [4/9] V30と同じ太陽画像前処理（既存ファイルは再利用）
def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)

def solar_three_channel(gray):
    gray = gray.astype(np.float32)
    disk = gray > max(4.0, np.percentile(gray, 8))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=80, sigmaY=80)
    flattened = gray / np.maximum(background, 1.0)
    flat_u8 = robust_uint8(flattened, disk, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    raw_u8 = robust_uint8(gray, disk, 0.5, 99.5)
    detail = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    detail = robust_uint8(detail, disk, 0.0, 99.5)
    output = np.dstack([raw_u8, clahe, detail])
    output[~disk] = 0
    return output

sources = [TRAIN_DIR / name for name in train_files] + [TEST_DIR / name for name in test_files]
expected_names = {source.name for source in sources}
CFG.PREPROC_CACHE.mkdir(parents=True, exist_ok=True)

# ランタイム再接続時はDriveキャッシュをSSDへ戻し、20分程度の前処理を繰り返さない。
cached_names = {path.name for path in CFG.PREPROC_CACHE.glob("*.jpeg")}
if cached_names == expected_names:
    for source in tqdm(CFG.PREPROC_CACHE.glob("*.jpeg"), total=len(cached_names), desc="前処理キャッシュをSSDへコピー"):
        destination = CFG.PREPROC_DIR / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
else:
    for source in tqdm(sources, desc="太陽画像を前処理"):
        destination = CFG.PREPROC_DIR / source.name
        if destination.exists():
            continue
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(source)
        cv2.imwrite(str(destination), solar_three_channel(gray), [cv2.IMWRITE_JPEG_QUALITY, 96])
    for source in tqdm(CFG.PREPROC_DIR.glob("*.jpeg"), total=len(expected_names), desc="前処理をDriveへ保存"):
        destination = CFG.PREPROC_CACHE / source.name
        if not destination.exists():
            shutil.copy2(source, destination)

actual_names = {path.name for path in CFG.PREPROC_DIR.glob("*.jpeg")}
assert actual_names == expected_names
print("前処理済み:", len(actual_names), "Driveキャッシュ:", str(CFG.PREPROC_CACHE))
"""
    ),
    code(
        """# [5/9] MediSeg実装と保存済みモデルを読み込む
""" + MEDISEG + """

yolo_models = {fold: YOLO(str(path)) for fold, path in weight_paths.items()}
refiner_state = torch.load(REFINER_PATH, map_location="cpu")
refiner = MediSegUNet(in_channels=3, out_channels=1, bilinear=False)
refiner.load_state_dict(refiner_state.get("best_model", refiner_state.get("model", refiner_state)))
refiner.cuda().eval()
print({"YOLOモデル数": len(yolo_models), "refiner_best_crop_dice": refiner_state.get("best_dice")})
"""
    ),
    code(
        """# [6/9] 個体mask、refiner、排他化、Hungarian Diceの共通関数
def annotation_mask(image_id, ann):
    info = images[image_id]
    mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
    for polygon in ann.get("segmentation", []):
        if len(polygon) >= 6:
            points = np.rint(np.asarray(polygon).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)

def gt_variants(file_name, compact=True):
    output = []
    for image_id in file_to_ids[file_name]:
        masks = [annotation_mask(image_id, ann) for ann in annotations[image_id]]
        if compact:
            masks = [cv2.resize(m.astype(np.float32), (CFG.COMPACT_SIZE, CFG.COMPACT_SIZE), interpolation=cv2.INTER_AREA) > 0 for m in masks]
        output.append(masks)
    return output

def prediction_masks(result):
    if result.masks is None:
        return []
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    return output

def square_box(mask, context=CFG.CROP_CONTEXT, minimum=96):
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
def refiner_probability(image, proposal):
    x0, y0, x1, y1 = square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return (x0, y0, x1, y1), probability

def build_variant(proposal, box, probability, method, threshold):
    if method == "yolo":
        return proposal.copy()
    x0, y0, x1, y1 = box
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    if method == "shrink":
        return refined & proposal
    kernel = 3 if method == "guided3" else 7
    support = cv2.dilate(proposal.astype(np.uint8), np.ones((kernel, kernel), np.uint8)).astype(bool)
    return refined & support

def compact_mask(mask):
    return cv2.resize(mask.astype(np.float32), (CFG.COMPACT_SIZE, CFG.COMPACT_SIZE), interpolation=cv2.INTER_AREA) > 0

def make_disjoint(candidates, min_area, compact=False):
    if not candidates:
        return []
    shape = candidates[0][1].shape
    occupied = np.zeros(shape, dtype=bool)
    output = []
    for score, mask, full_area in sorted(candidates, key=lambda item: item[0], reverse=True):
        if full_area < min_area:
            continue
        exclusive = mask & ~occupied
        if compact:
            scale = (CFG.OUTPUT_SIZE / CFG.COMPACT_SIZE) ** 2
            if exclusive.sum() * scale < min_area:
                continue
        elif exclusive.sum() < min_area:
            continue
        output.append(exclusive)
        occupied |= exclusive
    return output

def instance_dice(gt_masks, pred_masks):
    if not gt_masks and not pred_masks:
        return 1.0
    if not gt_masks or not pred_masks:
        return 0.0
    matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            intersection = np.logical_and(gt, pred).sum()
            matrix[i, j] = 2 * intersection / max(gt.sum() + pred.sum(), 1)
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].sum() / max(len(gt_masks), len(pred_masks)))

def mask_dice(first, second):
    intersection = np.logical_and(first, second).sum()
    return 2 * intersection / max(first.sum() + second.sum(), 1)
"""
    ),
    code(
        """# [7/9] リークのないfold 0 held-outで、排他化後の後処理を比較する
variant_keys = [("yolo", None)]
for method in ("shrink", "guided3", "guided7"):
    variant_keys.extend((method, threshold) for threshold in CFG.REFINER_THRESHOLDS)

configs = []
for method, threshold in variant_keys:
    for conf in CFG.CONF_GRID:
        for min_area in CFG.MIN_AREA_GRID:
            configs.append((method, threshold, conf, min_area))

score_sums = defaultdict(float)
score_counts = defaultdict(int)
count_errors = defaultdict(float)

fold = 0
model = yolo_models[fold]
valid_files = [name for name in train_files if fold_of_file[name] == fold]
for file_name in tqdm(valid_files, desc="fold 0 held-out後処理比較"):
        image_path = CFG.PREPROC_DIR / file_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        result = model.predict(
            str(image_path), imgsz=CFG.IMGSZ, conf=CFG.PREDICT_CONF_FLOOR,
            iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False,
        )[0]
        proposals = prediction_masks(result)
        confidences = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes is not None else []
        prepared = []
        for confidence, proposal in zip(confidences, proposals):
            box, probability = refiner_probability(image, proposal)
            variants = {}
            for method, threshold in variant_keys:
                mask = build_variant(proposal, box, probability, method, threshold)
                variants[(method, threshold)] = (compact_mask(mask), int(mask.sum()))
            prepared.append((float(confidence), variants))

        gt_sets = gt_variants(file_name, compact=True)
        mean_gt_count = float(np.mean([len(items) for items in gt_sets]))
        for config in configs:
            method, threshold, conf, min_area = config
            candidates = [
                (score, variants[(method, threshold)][0], variants[(method, threshold)][1])
                for score, variants in prepared if score >= conf
            ]
            predictions = make_disjoint(candidates, min_area, compact=True)
            score = float(np.mean([instance_dice(gt, predictions) for gt in gt_sets]))
            score_sums[config] += score
            score_counts[config] += 1
            count_errors[config] += abs(len(predictions) - mean_gt_count)
        del prepared, result

rows = []
for config in configs:
    method, threshold, conf, min_area = config
    rows.append({
        "method": method,
        "threshold": threshold,
        "conf": conf,
        "min_area": min_area,
        "heldout_instance_dice": score_sums[config] / score_counts[config],
        "mean_abs_count_error": count_errors[config] / score_counts[config],
    })

heldout_results = pd.DataFrame(rows).sort_values(
    ["heldout_instance_dice", "mean_abs_count_error"], ascending=[False, True]
).reset_index(drop=True)
heldout_results.to_csv(CFG.ARTIFACT_ROOT / "v31_fold0_heldout_postprocess_ablation.csv", index=False)
display(heldout_results.head(20))

def clean_params(row):
    threshold = row["threshold"]
    return {
        "method": str(row["method"]),
        "threshold": None if pd.isna(threshold) else float(threshold),
        "conf": float(row["conf"]),
        "min_area": int(row["min_area"]),
        "heldout_instance_dice": float(row["heldout_instance_dice"]),
        "mean_abs_count_error": float(row["mean_abs_count_error"]),
    }

best = clean_params(heldout_results.iloc[0])
best_yolo = clean_params(heldout_results[heldout_results["method"] == "yolo"].iloc[0])
(CFG.ARTIFACT_ROOT / "v31_heldout_selected_params.json").write_text(
    json.dumps({"best": best, "best_yolo": best_yolo}, indent=2), encoding="utf-8"
)
print({"held-out最良": best, "YOLO単体最良": best_yolo})
"""
    ),
    code(
        """# [8/9] held-out最良方式、fold 0単体、YOLO native ensembleを生成
def cluster_proposals(raw_predictions, minimum_dice=0.25):
    clusters = []
    for score, fold, mask in sorted(raw_predictions, key=lambda item: item[0], reverse=True):
        selected, selected_dice = None, 0.0
        for index, cluster in enumerate(clusters):
            if fold in cluster["folds"]:
                continue
            overlap = mask_dice(mask, cluster["representative"])
            if overlap > selected_dice:
                selected, selected_dice = index, overlap
        if selected is not None and selected_dice >= minimum_dice:
            cluster = clusters[selected]
            cluster["folds"].add(fold)
            cluster["scores"].append(score)
            if score > cluster["best_score"]:
                cluster["best_score"] = score
                cluster["representative"] = mask
        else:
            clusters.append({
                "folds": {fold}, "scores": [score], "best_score": score,
                "representative": mask,
            })
    output = []
    for cluster in clusters:
        votes = len(cluster["folds"])
        mean_score = float(np.mean(cluster["scores"]))
        adjusted_score = mean_score + 0.04 * (votes - 1)
        output.append((adjusted_score, votes, cluster["representative"]))
    return output

def params_to_key(params):
    threshold = None if params["method"] == "yolo" else float(params["threshold"])
    return params["method"], threshold, float(params["conf"]), int(params["min_area"])

BEST_KEY = params_to_key(best)
YOLO_KEY = params_to_key(best_yolo)

def apply_method(image, proposal, method, threshold):
    if method == "yolo":
        return proposal.copy()
    box, probability = refiner_probability(image, proposal)
    return build_variant(proposal, box, probability, method, threshold)

def predict_raw(file_name):
    image_path = CFG.PREPROC_DIR / file_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    raw = []
    for fold in range(3):
        result = yolo_models[fold].predict(
            str(image_path), imgsz=CFG.IMGSZ, conf=CFG.PREDICT_CONF_FLOOR,
            iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False,
        )[0]
        masks = prediction_masks(result)
        scores = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes is not None else []
        raw.extend((float(score), fold, mask) for score, mask in zip(scores, masks))
    return image, raw

def select_proposals(raw, ensemble, conf):
    if ensemble:
        clustered = cluster_proposals(raw)
        return [
            (score, proposal) for score, votes, proposal in clustered
            if score >= conf and (votes >= 2 or score >= conf + 0.12)
        ]
    return [(score, proposal) for score, fold, proposal in raw if fold == 0 and score >= conf]

def infer_from_raw(image, raw, ensemble, params, refinement_cache):
    method, threshold, conf, min_area = params
    selected = select_proposals(raw, ensemble, conf)
    candidates = []
    for score, proposal in selected:
        cache_key = (id(proposal), method, threshold)
        if cache_key not in refinement_cache:
            refinement_cache[cache_key] = apply_method(image, proposal, method, threshold)
        mask = refinement_cache[cache_key]
        candidates.append((score, mask, int(mask.sum())))
    masks = make_disjoint(candidates, min_area, compact=False)
    raw_count = len(raw) if ensemble else sum(fold == 0 for _, fold, _ in raw)
    return masks, raw_count, len(selected)

submission_rows = {"best_ensemble": [], "best_fold0": [], "yolo_ensemble": []}
audit_rows = []
for file_name in tqdm(test_files, desc="V31 test推論"):
    image, raw = predict_raw(file_name)
    refinement_cache = {}
    variants = {
        "best_ensemble": infer_from_raw(image, raw, True, BEST_KEY, refinement_cache),
        "best_fold0": infer_from_raw(image, raw, False, BEST_KEY, refinement_cache),
        "yolo_ensemble": infer_from_raw(image, raw, True, YOLO_KEY, refinement_cache),
    }
    image_id = Path(file_name).stem
    for variant_name, (masks, raw_count, selected_count) in variants.items():
        for index, mask in enumerate(masks, start=1):
            encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            submission_rows[variant_name].append({
                "filament_id": f"{image_id}_{index}",
                "segmentation_rle": encoded["counts"].decode("utf-8"),
            })
        audit_rows.append({
            "file_name": file_name, "variant": variant_name,
            "raw": raw_count, "selected": selected_count, "submitted": len(masks),
        })

output_paths = {
    "best_ensemble": Path("/content/submission_v31a_heldout_best_ensemble.csv"),
    "best_fold0": Path("/content/submission_v31b_heldout_best_fold0.csv"),
    "yolo_ensemble": Path("/content/submission_v31c_yolo_native_ensemble.csv"),
}
for name, rows in submission_rows.items():
    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    frame.to_csv(output_paths[name], index=False, lineterminator="\\n")
    shutil.copy2(output_paths[name], CFG.ARTIFACT_ROOT / output_paths[name].name)

audit = pd.DataFrame(audit_rows)
audit.to_csv(CFG.ARTIFACT_ROOT / "v31_test_variant_audit.csv", index=False)
display(audit.groupby("variant")["submitted"].describe())
print({name: str(path) for name, path in output_paths.items()})
"""
    ),
    code(
        """# [9/9] 3つのCSVを再復号し、null・ID・RLE・個体間重複を厳密検証
def validate_submission(path):
    frame = pd.read_csv(path, keep_default_na=False)
    assert frame.columns.tolist() == ["filament_id", "segmentation_rle"]
    assert len(frame) > 0
    assert not frame.isna().any().any()
    assert not frame["filament_id"].duplicated().any()
    assert not frame["segmentation_rle"].str.strip().eq("").any()
    occupied_by_image = {}
    areas = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=path.name):
        mask = mask_utils.decode({
            "size": [2048, 2048], "counts": row.segmentation_rle.encode("utf-8")
        }).astype(bool)
        assert mask.shape == (2048, 2048) and mask.any()
        image_id = row.filament_id.rsplit("_", 1)[0]
        occupied = occupied_by_image.setdefault(image_id, np.zeros((2048, 2048), dtype=bool))
        assert not np.logical_and(occupied, mask).any(), image_id
        occupied |= mask
        areas.append(int(mask.sum()))
    return {
        "file": str(path), "rows": len(frame), "images": len(occupied_by_image),
        "overlap_pixels": 0, "area_min": min(areas),
        "area_median": int(np.median(areas)), "area_max": max(areas),
    }

reports = [validate_submission(path) for path in output_paths.values()]
manifest = {
    "v30b_public_score": 0.35,
    "heldout_best": best,
    "heldout_best_yolo": best_yolo,
    "submissions": reports,
    "recommended_first": str(output_paths["best_ensemble"]),
}
(CFG.ARTIFACT_ROOT / "v31_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
"""
    ),
]


for index, cell in enumerate(cells):
    cell["id"] = f"v31-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
