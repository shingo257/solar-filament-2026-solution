# [1/8] 依存関係・Drive・前提artifactの確認
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
    PSEUDO_IMAGE_DIR = V35_ROOT / "pseudo_label" / "preprocessed"
    PSEUDO_CACHE_DIR = V35_ROOT / "pseudo_label" / "cache"
    PSEUDO_MANIFEST = V35_ROOT / "pseudo_label" / "pseudo_label_manifest.json"
    LOCAL_ROOT = Path("/content/solar_v49")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    YOLO_ROOT = LOCAL_ROOT / "yolo_folds"
    RUNS_ROOT = V35_ROOT / "runs"
    CACHE_ROOT = V35_ROOT / "prediction_cache" / "oof"
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    FOLD = 0
    CONF = 0.26
    MIN_AREA = 20
    REFINER_THRESHOLD = 0.45   # 確定構成(V35と同一、free variantではない)
    DROP_THRESHOLD = 0.90
    CROP_SIZE = 320
    BASELINE_FOLD0_PQ = 0.4393

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
CFG.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
CFG.CACHE_ROOT.mkdir(parents=True, exist_ok=True)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理(実データ)": CFG.PREPROC_CACHE,
    "fold0 refiner": CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt",
    "pseudo-label画像": CFG.PSEUDO_IMAGE_DIR,
    "pseudo-labelキャッシュ": CFG.PSEUDO_CACHE_DIR,
    "pseudo-label manifest": CFG.PSEUDO_MANIFEST,
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0), "runs": str(CFG.RUNS_ROOT)})