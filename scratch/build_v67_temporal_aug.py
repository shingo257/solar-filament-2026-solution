"""v67 temporal-neighbor augmentation builder — Phase B 第三弾。

## 目的

定量レーン枯渇宣言(13機構全棄却)の後、残る唯一の合法レバーは
**この枠組みに無い情報源の持ち込み**。主催者裁定済み(2026-10)で、
学習時のみ近傍・共時刻の無ラベルGONGフレームを使用可能(推論は単一画像必須)。

filament は数十分スケールでほぼ不変 → 各train観測の ±N分 フレームに
**同じannotationを流用する天然の増強**。これは augmentation の多様化ではなく
「同じfilamentの別seeing/別ノイズ実現」で、blind-FN の68%を占める
「弱い・かすれたfilamentの検出」に効く可能性がある。

## 設計

- fold2 pilot。train側(fold0+1)の各観測について GONG haf(FITS)から
  ±NEIGH_MINUTES分内の同サイトフレームを最大PER_OBS件取得
- FITS → match_histograms(既存train画像を参照)→ 8bit gray →
  solar_three_channel → jpeg96 で本物と同じ前処理画像を生成
- YOLO dataset: train = 原本(全labeler重複)+ 近傍(同labelerのラベル流用)
  val = fold2 原本のみ(近傍はvalに混入しない)
- YOLO学習はprodと同一ハイパラ(seed=3407+fold)
- 評価: 既存fold2 refiner + ownership + PQ(conf0.26)。
  **同一セル内でprod検出器も採点するペア比較**(run間の環境差を排除)

## 採否(事前宣言)

- ΔPQ(v67−prod、fold2 val) > +0.005 → fold0で独立確認
- |Δ| < 0.002 → ノイズ棄却。負 → 時間近傍増強も棄却、定量レーン完全終了

    python scratch/build_v67_temporal_aug.py --fold 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")

FROZEN_FOLDS = 'def load_frozen_folds(path, train_files, n_folds, expected_sha256):\n    import csv\n    import hashlib\n    import io\n    import re\n\n    if type(n_folds) is not int or n_folds < 2:\n        raise ValueError("n_folds must be an integer >= 2")\n    if (not isinstance(train_files, list) or not train_files\n            or any(not isinstance(name, str) or not name or name in (".", "..")\n                   or name.strip() != name or any(c in name for c in "/\\\\:")\n                   or any(ord(c) < 32 or ord(c) == 127 for c in name) for name in train_files)\n            or len(set(train_files)) != len(train_files)):\n        raise ValueError("Expected observations must be unique safe basenames")\n    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):\n        raise ValueError("Provide the externally frozen grouped_folds.csv SHA256")\n    payload = Path(path).read_bytes()\n    actual_sha256 = hashlib.sha256(payload).hexdigest()\n    if actual_sha256 != expected_sha256:\n        raise ValueError("Frozen fold CSV SHA256 differs")\n    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)\n    fields = reader.fieldnames\n    if (not fields or len(fields) != len(set(fields))\n            or not {"file_name", "fold"}.issubset(fields)):\n        raise ValueError("Fold CSV needs unique headers including file_name and fold")\n    expected, fold_of_file = set(train_files), {}\n    try:\n        for row in reader:\n            if None in row or any(value is None for value in row.values()):\n                raise ValueError("Malformed fold CSV row")\n            name, value = row["file_name"], row["fold"]\n            if name not in expected or name in fold_of_file:\n                raise ValueError("Duplicate or unexpected fold observation")\n            if not re.fullmatch(r"0|[1-9][0-9]*", value):\n                raise ValueError("Fold must be an exact nonnegative integer")\n            fold = int(value)\n            if fold >= n_folds:\n                raise ValueError("Fold is outside configured range")\n            fold_of_file[name] = fold\n    except csv.Error as error:\n        raise ValueError("Malformed fold CSV") from error\n    if set(fold_of_file) != expected or set(fold_of_file.values()) != set(range(n_folds)):\n        raise ValueError("Fold CSV must cover every observation and every configured fold")\n    return fold_of_file, actual_sha256\n\n'

NEIGH_MINUTES = 20
PER_OBS = 2
FOLD_BASELINE = {0: 0.4393, 1: 0.4415, 2: 0.4432}  # fold2は v64/65 の val baseline 実測値


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def build(fold: int) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
        },
        "cells": [
            markdown(
                f"""# v67 temporal-neighbor augmentation — fold{fold} パイロット

**目的**: 枯渇宣言後の唯一残る合法定量レバー — 時間近傍GONGフレームを
学習時のみ増強(主催者裁定済み合法・推論は単一画像必須)。

各train観測の ±{NEIGH_MINUTES}分 以内の同サイトFITSフレームを最大{PER_OBS}件取得し、
原本と同一annotationを流用してYOLO学習データを増量。
val には近傍を混入しない(純粋なfold{fold} valでprodとペア比較)。

**中断時は必ず「ランタイム→すべて実行」**。セル単独実行は未定義名で失敗する。
実行前に学習モデルと同じ分割の外部凍結SHA256を `CFG.GROUPED_FOLDS_SHA256` に設定する。
未設定・不一致・不正な分割では前処理前に停止する。ハッシュ一致だけで独立評価は保証しない。
全段階はDrive ckpt/キャッシュで再開可能。status: `v67_status_fold{fold}.json`

**採否(事前宣言)**: fold{fold} val ΔPQ(v67−prod同条件) > +0.005 → 別fold確認。
|Δ|<0.002 ノイズ棄却、負 → このレバーも棄却。
"""
            ),
            code(
                """# [1/7] 依存・設定・Drive・status
%%pip install -q "ultralytics>=8.3,<9" pycocotools astropy scikit-image

import json, math, random, re, shutil, time, urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from tqdm.auto import tqdm

drive.mount("/content/drive")

class CFG:
    SEED = 3407
    N_FOLDS = 3
    GROUPED_FOLDS_SHA256 = ""  # Independently frozen training split SHA256 required
    FOLD = %d
    NEIGH_MINUTES = %d
    PER_OBS = %d
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v67")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"          # val原本のみ(推論用)
    TN_FITS = DRIVE_ROOT / "temporal_neighbors" / "fits"
    TN_PREPROC = DRIVE_ROOT / "temporal_neighbors" / "preprocessed"
    TN_MANIFEST = DRIVE_ROOT / "temporal_neighbors" / "manifest.jsonl"
    YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v67_fold%d"
    # --- YOLO検出器(prodと同一) ---
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    # --- 後処理(prodと同一) ---
    SUBMISSION_CONF = 0.26
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    PREDICT_IOU = 0.55
    CROP_SIZE = 320
    # --- v67管理 ---
    BASELINE_PQ = %f
    MANIFEST = DRIVE_ROOT / "artifacts" / "v67_temporal_aug_fold%d_manifest.json"
    STATUS_FILE = DRIVE_ROOT / "artifacts" / "v67_status_fold%d.json"

_T0 = time.time()
def write_status(stage, done=-1, total=-1, detail=""):
    payload = {"stage": stage, "done": done, "total": total, "detail": detail,
               "elapsed_min": round((time.time() - _T0) / 60, 1),
               "updated": time.strftime("%%Y-%%m-%%d %%H:%%M:%%S")}
    try:
        CFG.STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass

for path in [CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.TN_FITS, CFG.TN_PREPROC,
             CFG.YOLO_ROOT]:
    path.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像キャッシュ": CFG.PREPROC_CACHE,
    f"fold{CFG.FOLD} prod検出器": CFG.RUNS_ROOT / f"yolo_fold_{CFG.FOLD}" / "weights" / "best.pt",
    f"fold{CFG.FOLD} 標準refiner": CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0)})
write_status("setup-ok")
""" % (fold, NEIGH_MINUTES, PER_OBS, fold, FOLD_BASELINE[fold], fold, fold)
            ),
            code(
                FROZEN_FOLDS + """# [2/7] COCO + fold分割。val原本をSSDへ、近傍取得対象=train側のみ
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"annotation JSONが見つかりません: {base}")
    return matches[0].parents[2]

DATA_ROOT = find_dataset_root(CFG.DATA_CACHE)
TRAIN_JSON = next(DATA_ROOT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
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

fold_of_file, frozen_fold_sha256 = load_frozen_folds(
    CFG.ARTIFACT_ROOT / "grouped_folds.csv", train_files, CFG.N_FOLDS, CFG.GROUPED_FOLDS_SHA256)
train_files_fold = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]

# val原本をSSDへ(推論用)。train原本はYOLO dataset構築時にcacheから直接参照
for _ci, name in enumerate(tqdm(valid_files_fold, desc="val前処理画像をSSDへ")):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)
    if _ci % 50 == 0:
        write_status("copy-val", _ci, len(valid_files_fold), name)
write_status("copy-done")

print({"除外train": len(train_files_fold), "検証(OOF)": len(valid_files_fold)})
"""
            ),
            code(
                """# [3/7] 近傍フレーム選定 + FITS ダウンロード(Driveキャッシュで再開可)
GONG_BASE = "https://gong2.nso.edu/ftp/HA/haf"
FILE_RE = re.compile(r'href="(\\d{14}([A-Za-z])h\\.fits\\.fz)"')
NAME_RE = re.compile(r"(\\d{14})([A-Za-z])h", re.IGNORECASE)

def fetch_url(url, binary=False):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        return data if binary else data.decode("utf-8", errors="replace")
    except Exception as exc:
        return None

def parse_obs(file_name):
    m = NAME_RE.search(file_name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), m.group(2).upper()

_day_cache = {}
def list_day(day):
    key = f"{day:%Y%m%d}"
    if key not in _day_cache:
        html = fetch_url(f"{GONG_BASE}/{day:%Y%m}/{key}/")
        _day_cache[key] = [(m.group(1), m.group(2).upper())
                          for m in FILE_RE.finditer(html)] if html else []
    return _day_cache[key]

def neighbors_for(obs_time, site):
    cands = []
    for dd in (-1, 0, 1):
        day = obs_time + timedelta(days=dd)
        for fname, fsite in list_day(day):
            if fsite != site:
                continue
            fm = NAME_RE.search(fname)
            if not fm:
                continue
            ftime = datetime.strptime(fm.group(1), "%Y%m%d%H%M%S")
            diff = (ftime - obs_time).total_seconds() / 60.0
            if diff == 0 or abs(diff) > CFG.NEIGH_MINUTES:
                continue
            cands.append({"file": fname, "day": f"{day:%Y%m%d}",
                          "dt_min": round(diff, 2)})
    cands.sort(key=lambda c: abs(c["dt_min"]))
    return cands[:CFG.PER_OBS]

# manifest.jsonl: 1行1観測 {"src": name, "neighbors": [...]}
done_srcs = set()
if CFG.TN_MANIFEST.exists():
    for line in CFG.TN_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done_srcs.add(json.loads(line)["src"])

mf = open(CFG.TN_MANIFEST, "a", encoding="utf-8")
n_cand = 0
for i, name in enumerate(tqdm(train_files_fold, desc="近傍選定")):
    if name in done_srcs:
        continue
    parsed = parse_obs(name)
    if not parsed:
        continue
    obs_time, site = parsed
    cands = neighbors_for(obs_time, site)
    mf.write(json.dumps({"src": name, "obs_utc": obs_time.isoformat(),
                          "site": site, "neighbors": cands}) + "\\n")
    mf.flush()
    n_cand += len(cands)
    if i % 25 == 0:
        write_status("select-neighbors", i, len(train_files_fold), name)
mf.close()

# ダウンロード(manifest全件、ファイル単位スキップ再開)
entries = [json.loads(l) for l in CFG.TN_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
todo = [n for e in entries for n in e["neighbors"]]
n_dl = 0
for i, n in enumerate(tqdm(todo, desc="FITS DL")):
    dest = CFG.TN_FITS / n["file"]
    if dest.exists() and dest.stat().st_size > 100_000:
        continue
    url = f"{GONG_BASE}/{n['day'][:6]}/{n['day']}/{n['file']}"
    data = fetch_url(url, binary=True)
    if data and len(data) > 100_000:
        dest.write_bytes(data)
        n_dl += 1
    else:
        print(f"  取得失敗: {url}")
    if i % 20 == 0:
        write_status("download", i, len(todo), n["file"])
write_status("download-done", len(todo), len(todo))
print({"観測": len(entries), "近傍候補": len(todo), "新規DL": n_dl})
"""
            ),
            code(
                """# [4/7] FITS → 階調マッチ → 3ch前処理JPEG(原本と同じパイプライン)
from astropy.io import fits
from skimage.exposure import match_histograms

def find_train_dir():
    return next(DATA_ROOT.rglob("train_images"))

# 参照ヒストグラム: train原本5枚(ラベル非依存の画素分布のみ利用)
RAW_TRAIN_DIR = find_train_dir()
ref_names = train_files_fold[:5]
reference = np.vstack([
    cv2.imread(str(RAW_TRAIN_DIR / n), cv2.IMREAD_GRAYSCALE) for n in ref_names
])

def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)

def solar_three_channel(gray):
    gray = gray.astype(np.float32)
    disk = gray > max(4.0, np.percentile(gray, 8))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=80, sigmaY=80)
    flat_u8 = robust_uint8(gray / np.maximum(background, 1.0), disk, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    raw_u8 = robust_uint8(gray, disk, 0.5, 99.5)
    dark = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    dark = robust_uint8(dark, disk, 0.0, 99.5)
    out = np.dstack([raw_u8, clahe, dark])
    out[~disk] = 0
    return out

def fits_to_preprocessed(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[1].data.astype(np.float32) if len(hdul) > 1 else hdul[0].data.astype(np.float32)
    gray = np.clip(match_histograms(data, reference), 0, 255).astype(np.uint8)
    return solar_three_channel(gray)

fits_files = sorted(CFG.TN_FITS.glob("*.fits.fz"))
n_conv, n_fail = 0, 0
for i, fp in enumerate(tqdm(fits_files, desc="FITS→前処理JPEG")):
    out = CFG.TN_PREPROC / (fp.name.replace(".fits.fz", "") + "_tn.jpeg")
    if out.exists():
        continue
    try:
        img = fits_to_preprocessed(fp)
        if img.shape != (2048, 2048, 3):
            img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 96])
        n_conv += 1
    except Exception as exc:
        n_fail += 1
        print(f"  変換失敗: {fp.name} ({exc})")
    if i % 20 == 0:
        write_status("convert", i, len(fits_files), fp.name)
write_status("convert-done", len(fits_files), len(fits_files))
print({"変換": n_conv, "失敗": n_fail, "総近傍画像": len(list(CFG.TN_PREPROC.glob('*.jpeg')))})
"""
            ),
            code(
                """# [5/7] YOLO dataset構築: train=原本(全labeler)+近傍(同labeler流用)、val=原本のみ
def polygon_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    polygon = max(polygons, key=len)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{v:.6f}" for v in points.reshape(-1))

def label_lines(image_id, info):
    return [polygon_line(ann, info["width"], info["height"])
            for ann in annotations_by_image[image_id] if not ann.get("iscrowd", 0)]

def write_variant(split, image_path, stem, lines):
    image_out = CFG.YOLO_ROOT / "images" / split / f"{stem}.jpeg"
    label_out = CFG.YOLO_ROOT / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    if not image_out.exists():
        shutil.copy2(image_path, image_out)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(l for l in lines if l), encoding="utf-8")

# src file_name -> 近傍前処理画像リスト の対応表
src_to_neighbors = defaultdict(list)
for e in entries:
    for n in e["neighbors"]:
        pp = CFG.TN_PREPROC / (n["file"].replace(".fits.fz", "") + "_tn.jpeg")
        if pp.exists():
            src_to_neighbors[e["src"]].append((n["file"], pp))

n_orig = n_neigh = 0
for name in tqdm(train_files_fold, desc="train側構築"):
    for image_id in file_to_ids[name]:
        info = images[image_id]
        lines = label_lines(image_id, info)
        stem = Path(name).stem + f"__labeler_{image_id}"
        write_variant("train", CFG.PREPROC_CACHE / name, stem, lines)
        n_orig += 1
        # 近傍: 同じlabelerのラベルを流用(filamentは±数分でほぼ不変)
        for fits_name, pp in src_to_neighbors.get(name, []):
            nstem = Path(pp).stem + f"__labeler_{image_id}"
            write_variant("train", pp, nstem, lines)
            n_neigh += 1

for name in tqdm(valid_files_fold, desc="val側構築"):
    image_id = file_to_ids[name][0]  # reference labeler のみ(prodと同じ)
    info = images[image_id]
    write_variant("val", CFG.PREPROC_DIR / name, Path(name).stem,
                   label_lines(image_id, info))

yaml_path = CFG.YOLO_ROOT / "data.yaml"
yaml_path.write_text(
    f"path: {CFG.YOLO_ROOT}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
    encoding="utf-8")
write_status("dataset-done")
print({"train原本variant": n_orig, "近傍variant": n_neigh,
       "増強倍率": round((n_orig + n_neigh) / max(n_orig, 1), 2), "yaml": str(yaml_path)})
"""
            ),
            code(
                """# [6/7] YOLO fold%d を近傍増強データで学習(prodと同一ハイパラ)
from ultralytics import YOLO

run_dir = CFG.RUNS_ROOT / f"v67_yolo_fold_{CFG.FOLD}"
best_path = run_dir / "weights" / "best.pt"
done_path = run_dir / "TRAINING_COMPLETE"
if done_path.exists() and best_path.exists():
    print(f"学習済みcheckpointを再利用 {best_path}")
else:
    model = YOLO(CFG.YOLO_MODEL)
    model.train(
        data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
        batch=CFG.YOLO_BATCH, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True,
        cache="disk",
        optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(CFG.RUNS_ROOT), name=f"v67_yolo_fold_{CFG.FOLD}", exist_ok=True,
    )
    done_path.write_text("ok", encoding="utf-8")
write_status("train-done")
print({"v67_weight": str(best_path)})
""" % fold
            ),
            code(
                """# [7/7] ペア評価: prod検出器 vs v67検出器 → refiner → ownership → PQ(fold%d val)
from pycocotools import mask as mask_utils
from ultralytics import YOLO

""" % fold
                + MODELS
                + """
refiner = MediSegUNet(3, 1, False).cuda().eval()
_ck = torch.load(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
                  map_location="cuda", weights_only=False)
refiner.load_state_dict(_ck.get("best_model") or _ck["model"], strict=True)
del _ck

def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    h, w = result.orig_shape
    out = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(mask >= 0.5)
    confs = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    return out, confs

def mask_square_box(mask, context=1.65, minimum=96):
    h, w = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, w, h
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(h, w))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= x1 - w; x1 = w
    if y1 > h: y0 -= y1 - h; y1 = h
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

@torch.no_grad()
def refine_one(image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    prob = refiner(t).sigmoid()[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = prob >= CFG.REFINER_THRESHOLD
    return refined & proposal

def greedy_disjoint(masks, order, min_area=CFG.MIN_AREA):
    occupied = np.zeros((2048, 2048), dtype=bool)
    kept = []
    for idx in order:
        ex = masks[idx] & ~occupied
        if ex.sum() < min_area:
            continue
        kept.append(ex); occupied |= ex
    return kept

def drop_contained(masks, thr=CFG.DROP_THRESHOLD):
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = mask_utils.encode(np.asfortranarray(np.stack(masks, -1).astype(np.uint8)))
    areas = np.asarray(mask_utils.area(rles), np.float64)
    si = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), np.float64)
    dropped = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped:
                continue
            v = si[i, j]
            if v <= 0:
                continue
            inter = v * (areas[i] + areas[j]) / (1.0 + v)
            s = i if areas[i] <= areas[j] else j
            if inter / max(areas[s], 1.0) >= thr:
                dropped.add(s)
    return [i for i in range(len(masks)) if i not in dropped]

def apply_ownership(masks, scores):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], np.float64)
    return greedy_disjoint(masks, sorted(drop_contained(masks), key=lambda i: areas[i]))

def to_rle(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    if isinstance(seg.get("counts"), list):
        return mask_utils.frPyObjects(seg, h, w)
    r = dict(seg)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode()
    return r

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)))
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth))).T

class PanopticQuality:
    def __init__(self, thr=0.5):
        self.thr, self.tp_ious, self.fp, self.fn = thr, [], 0, 0
    def add(self, m, n_gt, n_pred):
        if n_gt == 0:
            self.fp += n_pred; return
        if n_pred == 0:
            self.fn += n_gt; return
        hits = m > self.thr
        self.tp_ious.extend(m[hits].tolist())
        self.fp += int((hits.sum(0) == 0).sum())
        self.fn += int((hits.sum(1) == 0).sum())
    def summary(self):
        tp = len(self.tp_ious)
        den = tp + 0.5 * self.fp + 0.5 * self.fn
        return {"pq": float(sum(self.tp_ious) / den) if den else 0.0,
                "sq": float(sum(self.tp_ious) / tp) if tp else 0.0,
                "rq": float(tp / den) if den else 0.0,
                "tp": tp, "fp": self.fp, "fn": self.fn}

gt_rles = {iid: [to_rle(a, int(info["height"]), int(info["width"]))
                  for a in annotations_by_image[iid] if not a.get("iscrowd", 0)]
           for iid, info in images.items()}

def evaluate(yolo_weight, tag):
    model = YOLO(str(yolo_weight))
    meter = PanopticQuality()
    for i, name in enumerate(tqdm(valid_files_fold, desc=f"{tag} 評価")):
        path = CFG.PREPROC_DIR / name
        res = model.predict(str(path), imgsz=CFG.YOLO_IMGSZ,
                             conf=max(0.08, CFG.SUBMISSION_CONF * 0.75),
                             iou=CFG.PREDICT_IOU, agnostic_nms=True,
                             retina_masks=True, verbose=False)[0]
        masks, confs = result_masks(res)
        props = [(c, m) for c, m in zip(confs, masks) if c >= CFG.SUBMISSION_CONF]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        refined = [refine_one(image, m) for _, m in props]
        preds = apply_ownership(refined, np.ones(len(refined), np.float32))
        prles = []
        if preds:
            enc = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8)))
            prles = enc if isinstance(enc, list) else [enc]
        for iid in file_to_ids[name]:
            truth = gt_rles[iid]
            meter.add(iou_matrix(truth, prles), len(truth), len(prles))
        if i % 20 == 0:
            write_status(f"eval:{tag}", i, len(valid_files_fold), name)
    del model
    torch.cuda.empty_cache()
    return meter.summary()

res_prod = evaluate(CFG.RUNS_ROOT / f"yolo_fold_{CFG.FOLD}" / "weights" / "best.pt", "prod")
res_v67 = evaluate(CFG.RUNS_ROOT / f"v67_yolo_fold_{CFG.FOLD}" / "weights" / "best.pt", "v67")

delta = res_v67["pq"] - res_prod["pq"]
manifest = {"fold": CFG.FOLD, "folds_sha256": frozen_fold_sha256, "baseline_recorded": CFG.BASELINE_PQ,
             "prod_rerun": res_prod, "v67": res_v67, "delta_pq": delta,
             "neigh_minutes": CFG.NEIGH_MINUTES, "per_obs": CFG.PER_OBS,
             "verdict": ("adopt-candidate" if delta > 0.005
                           else "noise-reject" if abs(delta) < 0.002 else "reject")}
CFG.MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
write_status("done")
print(f"prod(再採点): {res_prod}")
print(f"v67(近傍増強): {res_v67}")
print(f"ΔPQ: {delta:+.4f}  -> {manifest['verdict']}")
"""
            ),
        ],
    }

    out = ROOT / f"solar_filament_v67_temporal_aug_fold{fold}_colab.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")

    import ast
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("%"))
            try:
                ast.parse(src)
            except SyntaxError as e:
                raise SystemExit(f"cell {i} syntax error: {e}\n{src[:400]}")
    print("AST validation passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2)
    args = ap.parse_args()
    build(args.fold)
