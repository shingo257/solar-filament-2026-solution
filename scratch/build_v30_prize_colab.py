import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v30_prize_colab_instance_ensemble.ipynb"
MEDISEG = (ROOT / "src" / "mediseg_models.py").read_text(encoding="utf-8")


def markdown(source):
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
    markdown(
        """# Solar Filament Segmentation V30 — prize-oriented instance ensemble (Colab A100)

V27 reached only **0.19** because it learned a semantic union mask and split it
with connected components, while the competition evaluates **individual
filaments** and penalizes fragmentation/merging. This notebook changes the
training target and validation loop, rather than only changing a threshold.

Pipeline:

1. group split by physical observation (no duplicate-labeler leakage),
2. train three high-resolution YOLO11m-seg instance models,
3. retain every independent MAGFiLO annotation as a training label variant,
4. refine each detected instance with the repository's 31M-parameter
   `MediSegUNet`, using edge-aware loss and solar-specific three-channel input,
5. tune confidence, mask threshold, and minimum area against a Hungarian-matched
   per-instance Dice proxy averaged across labelers,
6. ensemble folds and export official compressed COCO RLE.

The public YOLO+U-Net notebook (reported public score 0.69) is the floor for the
detector/refiner pattern. V30 adds grouped CV, all-labeler training, stronger
resolution/model, competition-aligned calibration, fold consensus, and robust
Drive checkpoints.

**Colab settings:** Runtime → Change runtime type → **A100 GPU** and High-RAM.
Internet must be enabled. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab
Secrets, and accept the competition rules. Run all cells in order. Disconnects
are safe: completed folds and refiner epochs are restored from Drive.

Research basis: [competition task](https://www.kaggle.com/competitions/filament-segmentation-2026/overview),
[MAGFiLO](https://www.mlecofi.net/magfilo), and
[EdgeAttNet](https://arxiv.org/abs/2509.02964).
"""
    ),
    code(
        """# [1/12] Install runtime dependencies
%pip install -q "ultralytics>=8.3,<9" pycocotools kaggle
"""
    ),
    code(
        """# [2/12] Imports, Drive, and configuration
import gc
import json
import math
import os
import random
import shutil
import subprocess
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive, userdata
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")
drive.mount("/content/drive")

class CFG:
    SEED = 3407
    DEBUG = False
    COMPETITION = "filament-segmentation-2026"
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    LOCAL_ROOT = Path("/content/solar_v30")
    LOCAL_DATA = LOCAL_ROOT / "competition_data"
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    YOLO_ROOT = LOCAL_ROOT / "yolo_folds"
    RUNS_ROOT = DRIVE_ROOT / "runs"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"

    N_FOLDS = 3
    FOLDS_TO_TRAIN = (0, 1, 2)
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 2 if DEBUG else 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2

    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    REFINER_EPOCHS = 2 if DEBUG else 14
    REFINER_FINAL_EPOCHS = 1 if DEBUG else 5
    REFINER_LR = 2e-4

    PREDICT_CONF_FLOOR = 0.08
    PREDICT_IOU = 0.55
    DEFAULT_CONF = 0.22
    DEFAULT_REFINER_THRESHOLD = 0.50
    DEFAULT_MIN_AREA = 35

for path in [CFG.DRIVE_ROOT, CFG.RUNS_ROOT, CFG.ARTIFACT_ROOT, CFG.LOCAL_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()
assert torch.cuda.is_available(), "Select a GPU runtime before continuing."
print({
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    "drive_root": str(CFG.DRIVE_ROOT),
})
"""
    ),
    code(
        """# [3/12] Download once to Drive, then stage data on the local SSD
def find_dataset_root(base):
    base = Path(base)
    matches = list(base.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"Training annotation JSON not found below {base}")
    return matches[0].parents[2]

try:
    cached_root = find_dataset_root(CFG.DATA_CACHE)
    print("Using cached competition data:", cached_root)
except FileNotFoundError:
    CFG.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    username = userdata.get("KAGGLE_USERNAME")
    key = userdata.get("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError("Add KAGGLE_USERNAME and KAGGLE_KEY to Colab Secrets.")
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    zip_path = CFG.DRIVE_ROOT / f"{CFG.COMPETITION}.zip"
    subprocess.run([
        "kaggle", "competitions", "download", "-c", CFG.COMPETITION,
        "-p", str(CFG.DRIVE_ROOT), "--force"
    ], check=True)
    shutil.unpack_archive(str(zip_path), str(CFG.DATA_CACHE))
    cached_root = find_dataset_root(CFG.DATA_CACHE)

if not CFG.LOCAL_DATA.exists():
    print("Copying cached data from Drive to the Colab SSD...")
    shutil.copytree(CFG.DATA_CACHE, CFG.LOCAL_DATA)

DATA_ROOT = find_dataset_root(CFG.LOCAL_DATA)
TRAIN_JSON = next(DATA_ROOT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
TRAIN_DIR = TRAIN_JSON.parent / "train_images"
TEST_DIRS = list(DATA_ROOT.rglob("test_images"))
if not TEST_DIRS:
    raise FileNotFoundError(f"test_images not found below {DATA_ROOT}")
TEST_DIR = TEST_DIRS[0]

with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

def annotation_rank(image_id):
    anns = annotations[image_id]
    return len(anns), sum(float(a.get("area", 0)) for a in anns)

file_to_reference_id = {
    name: max(ids, key=annotation_rank) for name, ids in file_to_ids.items()
}
train_files = sorted(file_to_ids)
test_files = sorted(p.name for p in TEST_DIR.glob("*.jpeg"))
print({
    "data_root": str(DATA_ROOT),
    "unique_train_observations": len(train_files),
    "annotation_records": len(images),
    "individual_filaments": len(coco["annotations"]),
    "multi_labeler_observations": sum(len(v) > 1 for v in file_to_ids.values()),
    "test_observations": len(test_files),
})
"""
    ),
    code(
        """# [4/12] Solar preprocessing and leakage-safe folds
def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)

def solar_three_channel(gray):
    # Inspired by EdgeAttNet: suppress the radial/background gradient and expose
    # faint dark filaments without throwing away the original observation.
    gray = gray.astype(np.float32)
    disk = gray > max(4.0, np.percentile(gray, 8))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=80, sigmaY=80)
    flattened = gray / np.maximum(background, 1.0)
    flat_u8 = robust_uint8(flattened, disk, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    raw_u8 = robust_uint8(gray, disk, 0.5, 99.5)
    dark_detail = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    dark_detail = robust_uint8(dark_detail, disk, 0.0, 99.5)
    out = np.dstack([raw_u8, clahe, dark_detail])
    out[~disk] = 0
    return out

CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
all_source_paths = [TRAIN_DIR / name for name in train_files] + [TEST_DIR / name for name in test_files]
for path in tqdm(all_source_paths, desc="solar preprocessing"):
    destination = CFG.PREPROC_DIR / path.name
    if destination.exists():
        continue
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    cv2.imwrite(str(destination), solar_three_channel(gray), [cv2.IMWRITE_JPEG_QUALITY, 96])

reference_counts = np.array([len(annotations[file_to_reference_id[name]]) for name in train_files])
try:
    strata = pd.qcut(reference_counts, q=min(5, len(np.unique(reference_counts))), labels=False, duplicates="drop")
except ValueError:
    strata = np.zeros(len(train_files), dtype=int)

fold_of_file = {}
splitter = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
for fold, (_, valid_idx) in enumerate(splitter.split(train_files, strata)):
    for idx in valid_idx:
        fold_of_file[train_files[idx]] = fold

fold_frame = pd.DataFrame({
    "file_name": train_files,
    "fold": [fold_of_file[name] for name in train_files],
    "reference_instances": reference_counts,
    "labeler_count": [len(file_to_ids[name]) for name in train_files],
})
fold_frame.to_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv", index=False)
display(fold_frame.groupby("fold").agg(
    observations=("file_name", "size"),
    instances=("reference_instances", "sum"),
    label_variants=("labeler_count", "sum"),
))
"""
    ),
    code(
        """# [5/12] Build all-labeler YOLO instance datasets
def polygon_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    # MAGFiLO specifies one polygon per filament. Largest is a safe fallback for
    # malformed/multipart records without turning one filament into many labels.
    polygon = max(polygons, key=len)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.reshape(-1))

def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)

def write_variant(fold_root, split, file_name, image_id, suffix):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{stem}.jpeg"
    label_out = fold_root / "labels" / split / f"{stem}.txt"
    link_or_copy((CFG.PREPROC_DIR / file_name).resolve(), image_out)
    lines = []
    for ann in annotations[image_id]:
        if ann.get("iscrowd", 0):
            continue
        line = polygon_line(ann, info["width"], info["height"])
        if line:
            lines.append(line)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(lines), encoding="utf-8")

def build_fold_dataset(fold):
    root = CFG.YOLO_ROOT / f"fold_{fold}"
    if root.exists():
        shutil.rmtree(root)
    for file_name in tqdm(train_files, desc=f"build fold {fold}"):
        if fold_of_file[file_name] == fold:
            image_id = file_to_reference_id[file_name]
            write_variant(root, "val", file_name, image_id, "")
        else:
            # Every independent annotation is useful training supervision.
            for image_id in file_to_ids[file_name]:
                write_variant(root, "train", file_name, image_id, f"__labeler_{image_id}")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
        encoding="utf-8",
    )
    return yaml_path

fold_yamls = {fold: build_fold_dataset(fold) for fold in CFG.FOLDS_TO_TRAIN}
print({fold: str(path) for fold, path in fold_yamls.items()})
"""
    ),
    code(
        """# [6/12] Train/resume three high-resolution instance models
def train_yolo_fold(fold, yaml_path):
    run_dir = CFG.RUNS_ROOT / f"yolo_fold_{fold}"
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    done_path = run_dir / "TRAINING_COMPLETE"
    if done_path.exists() and best_path.exists():
        print(f"fold {fold}: using completed checkpoint {best_path}")
        return best_path

    if last_path.exists():
        print(f"fold {fold}: resuming {last_path}")
        model = YOLO(str(last_path))
        model.train(resume=True)
    else:
        model = YOLO(CFG.YOLO_MODEL)
        model.train(
            data=str(yaml_path),
            epochs=CFG.YOLO_EPOCHS,
            imgsz=CFG.YOLO_IMGSZ,
            batch=CFG.YOLO_BATCH,
            device=0,
            workers=4,
            patience=CFG.YOLO_PATIENCE,
            seed=CFG.SEED + fold,
            deterministic=True,
            single_cls=True,
            mask_ratio=CFG.YOLO_MASK_RATIO,
            overlap_mask=False,
            amp=True,
            cache="disk",
            optimizer="AdamW",
            lr0=7e-4,
            lrf=0.05,
            weight_decay=5e-4,
            warmup_epochs=3,
            degrees=180,
            translate=0.04,
            scale=0.25,
            fliplr=0.5,
            flipud=0.5,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.18,
            mosaic=0.20,
            close_mosaic=10,
            project=str(CFG.RUNS_ROOT),
            name=f"yolo_fold_{fold}",
            exist_ok=True,
            plots=True,
            save=True,
            save_period=5,
            verbose=True,
        )
    if not best_path.exists():
        raise FileNotFoundError(f"Training ended without {best_path}")
    done_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return best_path

yolo_weight_paths = {
    fold: train_yolo_fold(fold, fold_yamls[fold]) for fold in CFG.FOLDS_TO_TRAIN
}
print({fold: str(path) for fold, path in yolo_weight_paths.items()})
"""
    ),
    code(
        """# [7/12] MediSeg-AI architecture (repository implementation)
""" + MEDISEG
    ),
    code(
        """# [8/12] Individual-instance crop dataset and edge-aware objective
def ann_mask(image_id, ann):
    info = images[image_id]
    mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
    for polygon in ann.get("segmentation", []):
        if len(polygon) < 6:
            continue
        points = np.rint(np.asarray(polygon).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

def square_box_from_bbox(bbox, height, width, context=CFG.CROP_CONTEXT, minimum=96):
    x, y, w, h = map(float, bbox)
    cx, cy = x + w / 2, y + h / 2
    side = int(math.ceil(max(w, h) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > width: x0 -= x1 - width; x1 = width
    if y1 > height: y0 -= y1 - height; y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

class InstanceCropDataset(Dataset):
    def __init__(self, file_names, all_labelers, augment=False):
        self.samples = []
        self.augment = augment
        for file_name in file_names:
            ids = file_to_ids[file_name] if all_labelers else [file_to_reference_id[file_name]]
            for image_id in ids:
                for ann in annotations[image_id]:
                    if not ann.get("iscrowd", 0) and ann.get("area", 0) > 0:
                        self.samples.append((file_name, image_id, ann))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, image_id, ann = self.samples[index]
        info = images[image_id]
        image = cv2.imread(str(CFG.PREPROC_DIR / file_name), cv2.IMREAD_COLOR)
        target = ann_mask(image_id, ann)
        x0, y0, x1, y1 = square_box_from_bbox(ann["bbox"], info["height"], info["width"])
        image = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        target = torch.from_numpy(target[None].copy()).float()
        if self.augment:
            k = random.randrange(4)
            image, target = torch.rot90(image, k, (1, 2)), torch.rot90(target, k, (1, 2))
            if random.random() < 0.5:
                image, target = torch.flip(image, (2,)), torch.flip(target, (2,))
            if random.random() < 0.5:
                image, target = torch.flip(image, (1,)), torch.flip(target, (1,))
            if random.random() < 0.5:
                image[:2] = torch.clamp(image[:2] ** random.uniform(0.85, 1.20), 0, 1)
        return image, target

def soft_dice_loss(logits, targets):
    probs = logits.sigmoid()
    intersection = (probs * targets).sum((1, 2, 3))
    denominator = probs.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()

def boundary_map(tensor):
    dilation = F.max_pool2d(tensor, 5, stride=1, padding=2)
    erosion = -F.max_pool2d(-tensor, 5, stride=1, padding=2)
    return torch.clamp(dilation - erosion, 0, 1)

def refiner_loss(logits, targets):
    region = 0.35 * F.binary_cross_entropy_with_logits(logits, targets) + 0.65 * soft_dice_loss(logits, targets)
    edge = F.l1_loss(boundary_map(logits.sigmoid()), boundary_map(targets))
    return region + 0.15 * edge

fold0_train_files = [name for name in train_files if fold_of_file[name] != 0]
fold0_valid_files = [name for name in train_files if fold_of_file[name] == 0]
train_crop_ds = InstanceCropDataset(fold0_train_files, all_labelers=True, augment=True)
valid_crop_ds = InstanceCropDataset(fold0_valid_files, all_labelers=True, augment=False)
print({"train_instance_crops": len(train_crop_ds), "valid_instance_crops": len(valid_crop_ds)})

def choose_refiner_batch_size(dataset):
    for candidate in CFG.REFINER_BATCH_CANDIDATES:
        try:
            loader = DataLoader(dataset, batch_size=candidate, shuffle=True, num_workers=2, pin_memory=True)
            image, target = next(iter(loader))
            probe = MediSegUNet(in_channels=3, out_channels=1, bilinear=False).cuda()
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(probe(image.cuda()), target.cuda())
            loss.backward()
            del probe, image, target, loss, loader
            torch.cuda.empty_cache()
            return candidate
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("No refiner batch size fits GPU memory.")

REFINER_BATCH_SIZE = choose_refiner_batch_size(train_crop_ds)
print("REFINER_BATCH_SIZE =", REFINER_BATCH_SIZE)
"""
    ),
    code(
        """# [9/12] Train/resume the held-out MediSeg refiner
def crop_dice(model, loader, threshold=0.5):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for image, target in loader:
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True).bool()
            pred = model(image).sigmoid() >= threshold
            intersection = (pred & target).sum((1, 2, 3)).float()
            denominator = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
            total += ((2 * intersection + 1) / (denominator + 1)).sum().item()
            count += len(image)
    return total / max(count, 1)

def train_refiner(train_dataset, valid_dataset, checkpoint_path, epochs, warm_start=None):
    train_loader = DataLoader(train_dataset, batch_size=REFINER_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=REFINER_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model = MediSegUNet(in_channels=3, out_channels=1, bilinear=False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_dice = 0, -1.0
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_dice = state["epoch"] + 1, state["best_dice"]
        print(f"Resuming refiner at epoch {start_epoch}")
    elif warm_start and Path(warm_start).exists():
        state = torch.load(warm_start, map_location="cpu")
        model.load_state_dict(state.get("best_model", state["model"]))

    best_state = None
    if checkpoint_path.exists():
        best_state = torch.load(checkpoint_path, map_location="cpu").get("best_model")
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        for image, target in tqdm(train_loader, desc=f"refiner {epoch + 1}/{epochs}", leave=False):
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(model(image), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(image)
        val_dice = crop_dice(model, valid_loader)
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "best_model": best_state,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
        }, checkpoint_path)
        print({"epoch": epoch + 1, "train_loss": running / len(train_dataset), "crop_dice": val_dice, "best": best_dice})
    model.load_state_dict(best_state)
    return model, best_dice

VALID_REFINER_CKPT = CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt"
valid_refiner, best_crop_dice = train_refiner(
    train_crop_ds, valid_crop_ds, VALID_REFINER_CKPT, CFG.REFINER_EPOCHS
)
print({"best_crop_dice": best_crop_dice})
"""
    ),
    code(
        """# [10/12] Competition-aligned instance metric and calibration
def result_masks(result):
    if result.masks is None:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    height, width = result.orig_shape
    output = []
    for mask in masks:
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    return output

def instance_dice_score(gt_masks, pred_masks):
    if not gt_masks and not pred_masks:
        return 1.0
    if not gt_masks or not pred_masks:
        return 0.0
    dice = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            intersection = np.logical_and(gt, pred).sum()
            dice[i, j] = (2 * intersection) / max(gt.sum() + pred.sum(), 1)
    rows, cols = linear_sum_assignment(-dice)
    return float(dice[rows, cols].sum() / max(len(gt_masks), len(pred_masks)))

def gt_variants(file_name):
    variants = []
    for image_id in file_to_ids[file_name]:
        variants.append([ann_mask(image_id, ann).astype(bool) for ann in annotations[image_id]])
    return variants

def mask_square_box(mask, context=CFG.CROP_CONTEXT, minimum=96):
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, mask.shape[1], mask.shape[0]
    bbox = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
    return square_box_from_bbox(bbox, mask.shape[0], mask.shape[1], context, minimum)

@torch.no_grad()
def refine_candidate(image, proposal, model):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = model(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    support = cv2.dilate(proposal.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    proposal_crop = proposal[y0:y1, x0:x1].astype(np.float32)
    probability = 0.80 * probability + 0.20 * proposal_crop
    probability[~support[y0:y1, x0:x1]] = 0
    return (x0, y0, x1, y1), probability

fold0_model = YOLO(str(yolo_weight_paths[0]))
calibration_cache = {}
valid_refiner.eval()
for file_name in tqdm(fold0_valid_files, desc="cache held-out predictions"):
    image_path = CFG.PREPROC_DIR / file_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    result = fold0_model.predict(
        str(image_path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.PREDICT_CONF_FLOOR,
        iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False,
    )[0]
    masks = result_masks(result)
    confidences = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes is not None else []
    candidates = []
    for confidence, proposal in zip(confidences, masks):
        box, probability = refine_candidate(image, proposal, valid_refiner)
        candidates.append((float(confidence), proposal, box, probability))
    calibration_cache[file_name] = candidates

def materialize_candidates(candidates, conf, threshold, min_area):
    predictions = []
    for confidence, proposal, (x0, y0, x1, y1), probability in candidates:
        if confidence < conf:
            continue
        mask = np.zeros_like(proposal, dtype=bool)
        mask[y0:y1, x0:x1] = probability >= threshold
        if mask.sum() >= min_area:
            predictions.append(mask)
    return predictions

grid = []
for conf in [0.12, 0.16, 0.20, 0.24, 0.28]:
    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for min_area in [20, 35, 60, 100]:
            file_scores = []
            for file_name, candidates in calibration_cache.items():
                predictions = materialize_candidates(candidates, conf, threshold, min_area)
                labeler_scores = [instance_dice_score(gt, predictions) for gt in gt_variants(file_name)]
                file_scores.append(float(np.mean(labeler_scores)))
            grid.append({"conf": conf, "threshold": threshold, "min_area": min_area, "instance_dice": float(np.mean(file_scores))})

calibration = pd.DataFrame(grid).sort_values("instance_dice", ascending=False).reset_index(drop=True)
display(calibration.head(12))
best_params = calibration.iloc[0].to_dict()
(CFG.ARTIFACT_ROOT / "calibrated_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
calibration.to_csv(CFG.ARTIFACT_ROOT / "calibration_grid.csv", index=False)
print("selected:", best_params)
"""
    ),
    code(
        """# [11/12] Final all-data refiner and three-fold ensemble inference
all_crop_ds = InstanceCropDataset(train_files, all_labelers=True, augment=True)
# The full-data stage starts from the leakage-safe best model and uses a short,
# lower-LR fit. A held-out loader is retained only for progress diagnostics.
original_lr = CFG.REFINER_LR
CFG.REFINER_LR = original_lr * 0.35
FINAL_REFINER_CKPT = CFG.ARTIFACT_ROOT / "mediseg_refiner_all_data_resume.pt"
final_refiner, _ = train_refiner(
    all_crop_ds, valid_crop_ds, FINAL_REFINER_CKPT,
    CFG.REFINER_FINAL_EPOCHS, warm_start=VALID_REFINER_CKPT,
)
CFG.REFINER_LR = original_lr
final_refiner.eval()
yolo_models = {fold: YOLO(str(path)) for fold, path in yolo_weight_paths.items()}

def mask_iou(first, second):
    intersection = np.logical_and(first, second).sum()
    return intersection / max(np.logical_or(first, second).sum(), 1)

def ensemble_proposals(image_path):
    raw = []
    for fold, model in yolo_models.items():
        result = model.predict(
            str(image_path), imgsz=CFG.YOLO_IMGSZ,
            conf=max(CFG.PREDICT_CONF_FLOOR, float(best_params["conf"]) * 0.75),
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
        if votes >= 2 or mean_conf >= max(0.42, float(best_params["conf"]) + 0.12):
            vote_sum = np.stack(cluster["masks"]).sum(0)
            proposals.append((mean_conf, vote_sum >= math.ceil(len(cluster["masks"]) / 2)))
    return proposals

rows = []
audit = []
for file_name in tqdm(test_files, desc="three-fold ensemble + MediSeg"):
    image_path = CFG.PREPROC_DIR / file_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    proposals = ensemble_proposals(image_path)
    final_masks = []
    for confidence, proposal in proposals:
        box, probability = refine_candidate(image, proposal, final_refiner)
        x0, y0, x1, y1 = box
        mask = np.zeros_like(proposal, dtype=bool)
        mask[y0:y1, x0:x1] = probability >= float(best_params["threshold"])
        if mask.sum() >= int(best_params["min_area"]):
            final_masks.append((confidence, mask))
    final_masks.sort(key=lambda item: item[0], reverse=True)
    image_id = Path(file_name).stem
    for index, (_, mask) in enumerate(final_masks, start=1):
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rows.append({
            "filament_id": f"{image_id}_{index}",
            "segmentation_rle": encoded["counts"].decode("utf-8"),
        })
    audit.append({"file_name": file_name, "proposals": len(proposals), "submitted": len(final_masks)})

submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
OUTPUT_PATH = Path("/content/submission_v30_prize_instance_ensemble.csv")
submission.to_csv(OUTPUT_PATH, index=False, lineterminator="\\n")
shutil.copy2(OUTPUT_PATH, CFG.ARTIFACT_ROOT / OUTPUT_PATH.name)
pd.DataFrame(audit).to_csv(CFG.ARTIFACT_ROOT / "test_instance_audit.csv", index=False)
print({"submission": str(OUTPUT_PATH), "rows": len(submission), "drive_copy": str(CFG.ARTIFACT_ROOT / OUTPUT_PATH.name)})
display(pd.DataFrame(audit).describe())
"""
    ),
    code(
        """# [12/12] Strict official-RLE validation and artifact manifest
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
for counts in tqdm(checked["segmentation_rle"], desc="decode official RLE"):
    decoded = mask_utils.decode({"size": [2048, 2048], "counts": counts.encode("utf-8")})
    assert decoded.shape == (2048, 2048)
    area = int(decoded.sum())
    assert area > 0
    areas.append(area)

manifest = {
    "submission": str(OUTPUT_PATH),
    "rows": len(checked),
    "images_with_predictions": len(submitted_ids),
    "test_images": len(test_ids),
    "area_min": min(areas),
    "area_median": int(np.median(areas)),
    "area_max": max(areas),
    "calibration": best_params,
    "fold_weights": {str(k): str(v) for k, v in yolo_weight_paths.items()},
    "refiner_checkpoint": str(FINAL_REFINER_CKPT),
}
(CFG.ARTIFACT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(manifest)

# Quick sanity plot: submitted instance count distribution.
audit_df = pd.read_csv(CFG.ARTIFACT_ROOT / "test_instance_audit.csv")
audit_df["submitted"].hist(bins=range(int(audit_df["submitted"].max()) + 2))
plt.title("V30 submitted instances per test image")
plt.xlabel("instances")
plt.ylabel("images")
plt.tight_layout()
plt.savefig(CFG.ARTIFACT_ROOT / "instance_count_histogram.png", dpi=140)
plt.show()
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
