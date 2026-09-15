# [1/5] 環境確認・依存関係・Drive mount
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
print({"pseudo_images": len(targets), "gpu": gpu_name})