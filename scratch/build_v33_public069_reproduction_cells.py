"""Embedded cell payloads for the build_v33_public069_reproduction notebook builder.

Verbatim extraction of the cell-source literals from
build_v33_public069_reproduction.py so the builder stays readable.
No logic; loaded by the builder via sibling import.
"""


def markdown_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    markdown_cell(
        """# 太陽フィラメント V33 — 公開スコア0.69の忠実再現（Colab）

V30/V32系は公開スコアが0.35で頭打ちになりました。本ノートブックは、Apache-2.0で公開され、
Kaggle公開スコア **0.69** を記録した
[YOLO + U-Net | Solar Filament Segmentation](https://www.kaggle.com/code/ektarr/yolo-unet-solar-filament-segmentation)
の学習条件を基準に戻します。

中核は **raw grayscale + YOLO11s-seg（1024px・30 epoch）+ ImageNet事前学習ResNet18 U-Net refiner** です。
公開実装の閾値やaugmentationは変えず、以下の運用上の修正だけを加えています。

- Google Driveへepochごとのcheckpointを保存し、Colab切断後に「すべてのセルを実行」で再開
- COCOの文字列IDをそのまま扱い、`int(...)`変換エラーを防止
- 提出前にnull、RLE、空mask、個体間重複を全件検証
- 重複が本当に発生した画像だけ、YOLO confidence順に排他化

**推奨ランタイム:** A100 GPU、通常RAM。High-RAMは不要です。  
**Internet:** 初回はON（Ultralytics、YOLO11s、ImageNet ResNet18の取得に必要）。  
既存の`MyDrive/solar_filament_v30/competition_data`を自動再利用します。
""",
        "v33-title",
    ),
    code_cell(
        """# [1/11] 必要パッケージをインストール
%pip install -q "ultralytics>=8.3,<9" pycocotools kaggle
""",
        "v33-install",
    ),
    code_cell(
        """# [2/11] ライブラリ、Google Drive、共通設定
import gc
import json
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
from google.colab import drive, userdata
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")
drive.mount("/content/drive")

import ultralytics.utils.tqdm as ultra_tqdm
ultra_tqdm.is_noninteractive_console = lambda: True

DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v33_public069")
RUNS_ROOT = DRIVE_ROOT / "runs"
ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
TORCH_CACHE = DRIVE_ROOT / "torch_cache"
LOCAL_ROOT = Path("/content/solar_v33_public069")
LOCAL_DATA = LOCAL_ROOT / "competition_data"

for path in (DRIVE_ROOT, RUNS_ROOT, ARTIFACT_ROOT, TORCH_CACHE, LOCAL_ROOT):
    path.mkdir(parents=True, exist_ok=True)

# ImageNet事前学習重みをDriveへキャッシュし、再接続時の再ダウンロードを避ける。
os.environ["TORCH_HOME"] = str(TORCH_CACHE)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

assert torch.cuda.is_available(), "ランタイムのタイプをGPUに変更してください。"
system_ram_gb = int(Path("/proc/meminfo").read_text().splitlines()[0].split()[1]) / 1024**2
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print({
    "GPU": torch.cuda.get_device_name(0),
    "GPUメモリ_GB": round(gpu_gb, 1),
    "システムRAM_GB": round(system_ram_gb, 1),
    "High-RAM必須": False,
    "Drive保存先": str(DRIVE_ROOT),
})
""",
        "v33-common",
    ),
    code_cell(
        """# [3/11] Driveの既存競技データをColab SSDへ配置
def find_dataset_root(base):
    base = Path(base)
    matches = list(base.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(base)
    # JSONは MAGFiLO_1.0_Kaggle_2026/train/ 配下にある。
    return matches[0].parents[1]

cache_candidates = [
    Path("/content/drive/MyDrive/solar_filament_v30/competition_data"),
    DRIVE_ROOT / "competition_data",
]
cached_container = None
for candidate in cache_candidates:
    try:
        find_dataset_root(candidate)
        cached_container = candidate
        break
    except FileNotFoundError:
        pass

if cached_container is None:
    cached_container = DRIVE_ROOT / "competition_data"
    cached_container.mkdir(parents=True, exist_ok=True)
    username = userdata.get("KAGGLE_USERNAME")
    key = userdata.get("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError("Colab SecretsにKAGGLE_USERNAMEとKAGGLE_KEYを登録してください。")
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    zip_path = DRIVE_ROOT / "filament-segmentation-2026.zip"
    subprocess.run([
        "kaggle", "competitions", "download", "-c", "filament-segmentation-2026",
        "-p", str(DRIVE_ROOT), "--force",
    ], check=True)
    shutil.unpack_archive(str(zip_path), str(cached_container))

try:
    DATA_ROOT = find_dataset_root(LOCAL_DATA)
    print("Colab SSD上の既存データを再利用します。")
except FileNotFoundError:
    if LOCAL_DATA.exists():
        shutil.rmtree(LOCAL_DATA)
    print("DriveのキャッシュをColab SSDへコピーしています...")
    shutil.copytree(cached_container, LOCAL_DATA)
    DATA_ROOT = find_dataset_root(LOCAL_DATA)

print({"競技データ": str(DATA_ROOT)})
""",
        "v33-data-stage",
    ),
    code_cell(
        """# [4/11] 公開0.69実装と同じ設定でCOCOをYOLO-segへ変換
class CFG:
    ROOT = str(DATA_ROOT)
    TRAIN_DIR = str(DATA_ROOT / "train" / "train_images")
    TEST_DIR = str(DATA_ROOT / "test" / "test_images")
    TRAIN_JSON = str(DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    YOLO_DIR = str(LOCAL_ROOT / "yolo_dataset")

    DEVICE = 0
    DEBUG = False
    IMGSZ = 1024
    BATCH_SIZE = 4
    EPOCHS = 1 if DEBUG else 30
    PATIENCE = 5
    MODEL = "yolo11s-seg.pt"
    USE_CATEGORIES = True
    MASK_RATIO = 2
    CONF_THRESHOLD = 0.25
    MIN_COMPONENT_SIZE = 20
    SEED = SEED

with open(CFG.TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

# MAGFiLOのimage idは文字列を含むため、数値変換しない。
images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)

file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

def observation_completeness(image_id):
    anns = annotations.get(image_id, [])
    return len(anns), sum(float(ann.get("area", 0)) for ann in anns)

# 公開実装どおり、複数ラベラーのうち注釈が最も完全な1件を教師に使う。
file_to_best_id = {
    file_name: max(ids, key=observation_completeness)
    for file_name, ids in file_to_ids.items()
}
# 公開実装と同じCOCO JSON内の挿入順を維持する。
unique_files = list(file_to_best_id.keys())
train_files, valid_files = train_test_split(
    unique_files, test_size=0.2, random_state=CFG.SEED
)
train_ids = [file_to_best_id[name] for name in train_files]
valid_ids = [file_to_best_id[name] for name in valid_files]

if CFG.USE_CATEGORIES and coco.get("categories"):
    categories = sorted(coco["categories"], key=lambda item: item["id"])
    cat_id_to_yolo = {item["id"]: index for index, item in enumerate(categories)}
    class_names = [item["name"] for item in categories]
else:
    cat_id_to_yolo = None
    class_names = ["filament"]

def coco_polygon_to_yolo_line(polygon, width, height, class_id):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    coordinates = " ".join(f"{value:.6f}" for value in points.reshape(-1))
    return f"{class_id} {coordinates}"

def build_yolo_split(split_name, image_ids):
    image_dir = Path(CFG.YOLO_DIR) / "images" / split_name
    label_dir = Path(CFG.YOLO_DIR) / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for image_id in tqdm(image_ids, desc=f"YOLO {split_name}作成"):
        info = images[image_id]
        source = Path(CFG.TRAIN_DIR) / info["file_name"]
        destination = image_dir / info["file_name"]
        if not destination.exists():
            destination.symlink_to(source)
        lines = []
        for ann in annotations.get(image_id, []):
            if ann.get("iscrowd", 0):
                skipped += 1
                continue
            class_id = cat_id_to_yolo[ann["category_id"]] if cat_id_to_yolo else 0
            for polygon in ann.get("segmentation", []):
                if len(polygon) < 6:
                    skipped += 1
                    continue
                lines.append(coco_polygon_to_yolo_line(
                    polygon, info["width"], info["height"], class_id
                ))
        (label_dir / f"{Path(info['file_name']).stem}.txt").write_text(
            "\\n".join(lines), encoding="utf-8"
        )
    return skipped

if Path(CFG.YOLO_DIR).exists():
    shutil.rmtree(CFG.YOLO_DIR)
skipped_train = build_yolo_split("train", train_ids)
skipped_valid = build_yolo_split("val", valid_ids)

names_yaml = "\\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
data_yaml = f"path: {CFG.YOLO_DIR}\\ntrain: images/train\\nval: images/val\\nnames:\\n{names_yaml}\\n"
YOLO_YAML = Path(CFG.YOLO_DIR) / "data.yaml"
YOLO_YAML.write_text(data_yaml, encoding="utf-8")

print({
    "学習画像": len(train_ids), "検証画像": len(valid_ids),
    "複数ラベラー画像": sum(len(ids) > 1 for ids in file_to_ids.values()),
    "クラス": class_names,
    "除外polygon_train": skipped_train, "除外polygon_valid": skipped_valid,
})
""",
        "v33-yolo-data",
    ),
    code_cell(
        """# [5/11] YOLO11s-segを学習、またはDriveのlast.ptから再開
YOLO_RUN_DIR = RUNS_ROOT / "yolo_public069"
YOLO_BEST = YOLO_RUN_DIR / "weights" / "best.pt"
YOLO_LAST = YOLO_RUN_DIR / "weights" / "last.pt"
YOLO_DONE = YOLO_RUN_DIR / "TRAINING_COMPLETE"

if YOLO_DONE.exists() and YOLO_BEST.exists():
    print("完了済みYOLO重みを再利用します:", YOLO_BEST)
elif YOLO_LAST.exists():
    print("中断したYOLO学習を再開します:", YOLO_LAST)
    resume_model = YOLO(str(YOLO_LAST))
    resume_model.train(resume=True)
    if not YOLO_BEST.exists():
        raise FileNotFoundError(YOLO_BEST)
    YOLO_DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
else:
    print("公開0.69条件でYOLO学習を開始します。")
    train_model = YOLO(CFG.MODEL)
    train_model.train(
        data=str(YOLO_YAML),
        epochs=CFG.EPOCHS,
        imgsz=CFG.IMGSZ,
        batch=CFG.BATCH_SIZE,
        device=CFG.DEVICE,
        patience=CFG.PATIENCE,
        seed=CFG.SEED,
        single_cls=(not CFG.USE_CATEGORIES),
        mask_ratio=CFG.MASK_RATIO,
        overlap_mask=False,
        hsv_h=0.0,
        hsv_s=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.3,
        degrees=10,
        scale=0.3,
        hsv_v=0.3,
        close_mosaic=5,
        project=str(RUNS_ROOT),
        name="yolo_public069",
        exist_ok=True,
        save=True,
        save_period=1,
        plots=True,
    )
    if not YOLO_BEST.exists():
        raise FileNotFoundError(YOLO_BEST)
    YOLO_DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")

best_model = YOLO(str(YOLO_BEST))
print({"YOLO_best": str(YOLO_BEST)})
""",
        "v33-yolo-train",
    ),
    code_cell(
        """# [6/11] 公開実装と同じImageNet事前学習ResNet18 U-Net refiner
CROP_SIZE = 256
CONTEXT = 1.8
MIN_CROP = 96

def square_bounds(mask, context=CONTEXT, minimum=MIN_CROP):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, width, height
    center_x = (xs.min() + xs.max() + 1) / 2
    center_y = (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > width: x0 -= x1 - width; x1 = width
    if y1 > height: y0 -= y1 - height; y1 = height
    return int(x0), int(y0), int(x1), int(y1)

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

class CropUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], 1))
        x = self.dec3(torch.cat([self.up3(x), x2], 1))
        x = self.dec2(torch.cat([self.up2(x), x1], 1))
        x = self.dec1(torch.cat([self.up1(x), x0], 1))
        return self.head(self.up0(x))

def build_crop_arrays(image_ids):
    crop_images, crop_masks = [], []
    for image_id in tqdm(image_ids, desc="GT個体crop作成"):
        info = images[image_id]
        image = cv2.imread(str(Path(CFG.TRAIN_DIR) / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        for ann in annotations.get(image_id, []):
            mask = np.zeros(image.shape, dtype=np.uint8)
            for polygon in ann.get("segmentation", []):
                points = np.rint(np.asarray(polygon, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
            if not mask.any():
                continue
            x0, y0, x1, y1 = square_bounds(mask)
            crop_images.append(cv2.resize(
                image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR
            ))
            crop_masks.append(cv2.resize(
                mask[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST
            ))
    return np.stack(crop_images), np.stack(crop_masks)

train_crop_images, train_crop_masks = build_crop_arrays(train_ids)
valid_crop_images, valid_crop_masks = build_crop_arrays(valid_ids)

class CropDataset(Dataset):
    def __init__(self, crop_images, crop_masks, augment=False):
        self.crop_images = crop_images
        self.crop_masks = crop_masks
        self.augment = augment

    def __len__(self):
        return len(self.crop_images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.crop_images[index].copy()).float() / 255
        mask = torch.from_numpy(self.crop_masks[index].copy()).float().unsqueeze(0)
        if self.augment:
            k = random.randrange(4)
            image = torch.rot90(image, k, (0, 1))
            mask = torch.rot90(mask, k, (1, 2))
            if random.random() < 0.5:
                image, mask = torch.flip(image, (1,)), torch.flip(mask, (2,))
            if random.random() < 0.5:
                image = image ** random.uniform(0.85, 1.20)
            if random.random() < 0.5:
                image = torch.clamp(image * random.uniform(0.90, 1.10), 0, 1)
        return image.unsqueeze(0).repeat(3, 1, 1), mask

# crop配列は約1GBであり、通常RAMで十分。GPUメモリに応じてbatchだけ調整する。
REFINER_BATCH_SIZE = 16 if gpu_gb >= 24 else 8 if gpu_gb >= 14 else 4
train_crop_loader = DataLoader(
    CropDataset(train_crop_images, train_crop_masks, augment=True),
    batch_size=REFINER_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
)
valid_crop_loader = DataLoader(
    CropDataset(valid_crop_images, valid_crop_masks, augment=False),
    batch_size=max(REFINER_BATCH_SIZE, 16), shuffle=False, num_workers=2, pin_memory=True,
)
print({
    "train_crops": len(train_crop_images), "valid_crops": len(valid_crop_images),
    "REFINER_BATCH_SIZE": REFINER_BATCH_SIZE,
})
""",
        "v33-refiner-data",
    ),
    code_cell(
        """# [7/11] refinerを8 epoch学習、またはDrive checkpointから再開
def soft_dice_loss(logits, targets):
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()

REFINER_RESUME = ARTIFACT_ROOT / "refiner_public069_resume.pt"
REFINER_BEST = ARTIFACT_ROOT / "best_refiner_public069.pt"
REFINER_DONE = ARTIFACT_ROOT / "REFINER_TRAINING_COMPLETE"
REFINER_EPOCHS = 8
device = torch.device("cuda:0")

if REFINER_DONE.exists() and REFINER_BEST.exists():
    print("完了済みrefiner重みを再利用します:", REFINER_BEST)
    refiner = CropUNet(pretrained=False).to(device)
    refiner.load_state_dict(torch.load(REFINER_BEST, map_location=device))
else:
    has_resume = REFINER_RESUME.exists()
    refiner = CropUNet(pretrained=not has_resume).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=REFINER_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_dice, best_state = 0, -1.0, None
    if has_resume:
        state = torch.load(REFINER_RESUME, map_location="cpu")
        refiner.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_dice = float(state["best_dice"])
        best_state = state["best_model"]
        print(f"refinerをepoch {start_epoch + 1}から再開します。")

    for epoch in range(start_epoch, REFINER_EPOCHS):
        refiner.train()
        running_loss = 0.0
        for batch_images, batch_masks in tqdm(
            train_crop_loader, desc=f"refiner {epoch + 1}/{REFINER_EPOCHS}", leave=False
        ):
            batch_images = batch_images.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = refiner(batch_images)
                loss = 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, batch_masks)
                loss += 0.5 * soft_dice_loss(logits, batch_masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(batch_images)

        refiner.eval()
        dice_sum = count = 0
        with torch.no_grad():
            for batch_images, batch_masks in valid_crop_loader:
                batch_images = batch_images.to(device, non_blocking=True)
                batch_masks = batch_masks.to(device, non_blocking=True).bool()
                prediction = refiner(batch_images).sigmoid() >= 0.65
                intersection = (prediction & batch_masks).sum((1, 2, 3)).float()
                denominator = prediction.sum((1, 2, 3)) + batch_masks.sum((1, 2, 3))
                dice_sum += ((2 * intersection + 1e-7) / (denominator + 1e-7)).sum().item()
                count += len(batch_images)
        validation_dice = dice_sum / max(count, 1)
        if validation_dice > best_dice:
            best_dice = validation_dice
            best_state = {key: value.detach().cpu().clone() for key, value in refiner.state_dict().items()}
            torch.save(best_state, REFINER_BEST)
        scheduler.step()
        torch.save({
            "epoch": epoch, "model": refiner.state_dict(), "best_model": best_state,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
        }, REFINER_RESUME)
        print({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_crop_images),
            "crop_dice_0.65": validation_dice,
            "best": best_dice,
        })

    refiner.load_state_dict(torch.load(REFINER_BEST, map_location=device))
    REFINER_DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")

refiner.eval()
REFINER_THRESHOLD = 0.65
""",
        "v33-refiner-train",
    ),
    code_cell(
        """# [8/11] YOLO個体maskを公開実装と同じrefinerで精密化
@torch.no_grad()
def refine_masks(image_gray, yolo_masks, threshold=REFINER_THRESHOLD):
    refined = []
    for mask in yolo_masks:
        mask = mask.astype(bool)
        if not mask.any():
            refined.append(mask)
            continue
        x0, y0, x1, y1 = square_bounds(mask)
        crop = cv2.resize(
            image_gray[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR
        )
        tensor = torch.from_numpy(crop).float()[None, None].repeat(1, 3, 1, 1).to(device) / 255
        probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
        crop_mask = cv2.resize(
            probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR
        ) >= threshold
        if crop_mask.any():
            full = np.zeros_like(mask, dtype=bool)
            full[y0:y1, x0:x1] = crop_mask
            refined.append(full)
        else:
            refined.append(mask)
    return refined

def yolo_masks_from_result(result):
    if result.masks is None:
        return [], []
    height, width = result.orig_shape
    masks = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(mask.astype(bool))
    scores = result.boxes.conf.detach().cpu().numpy().tolist()
    return masks, [float(score) for score in scores]

def make_disjoint_only_if_needed(masks, scores, min_area):
    if not masks:
        return [], 0
    occupied = np.zeros_like(masks[0], dtype=bool)
    overlap_pixels = 0
    for mask in masks:
        overlap_pixels += int(np.logical_and(occupied, mask).sum())
        occupied |= mask
    if overlap_pixels == 0:
        return [mask for mask in masks if mask.sum() >= min_area], 0

    # Kaggleは同一画像の個体mask重複を拒否するため、重複時だけconfidence順に排他化する。
    output = []
    occupied[:] = False
    for score, mask in sorted(zip(scores, masks), key=lambda item: item[0], reverse=True):
        exclusive = mask & ~occupied
        if exclusive.sum() >= min_area:
            output.append(exclusive)
            occupied |= exclusive
    return output, overlap_pixels
""",
        "v33-refine",
    ),
    code_cell(
        """# [9/11] held-outで個体Diceと重複量を診断（閾値の再調整はしない）
def annotation_masks(image_id, compact_size=512):
    info = images[image_id]
    output = []
    for ann in annotations.get(image_id, []):
        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for polygon in ann.get("segmentation", []):
            points = np.rint(np.asarray(polygon).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
        if mask.any():
            output.append(cv2.resize(
                mask, (compact_size, compact_size), interpolation=cv2.INTER_AREA
            ) > 0)
    return output

def compact_masks(masks, compact_size=512):
    return [cv2.resize(
        mask.astype(np.uint8), (compact_size, compact_size), interpolation=cv2.INTER_AREA
    ) > 0 for mask in masks]

def instance_dice(gt_masks, prediction_masks):
    if not gt_masks and not prediction_masks:
        return 1.0
    if not gt_masks or not prediction_masks:
        return 0.0
    matrix = np.zeros((len(gt_masks), len(prediction_masks)), dtype=np.float32)
    for gt_index, gt_mask in enumerate(gt_masks):
        for pred_index, pred_mask in enumerate(prediction_masks):
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            matrix[gt_index, pred_index] = 2 * intersection / max(gt_mask.sum() + pred_mask.sum(), 1)
    rows, columns = linear_sum_assignment(-matrix)
    return float(matrix[rows, columns].sum() / max(len(gt_masks), len(prediction_masks)))

diagnostics = []
for image_id in tqdm(valid_ids, desc="held-out診断"):
    info = images[image_id]
    image_path = str(Path(CFG.TRAIN_DIR) / info["file_name"])
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    result = best_model.predict(
        image_path, imgsz=CFG.IMGSZ, conf=CFG.CONF_THRESHOLD,
        agnostic_nms=True, retina_masks=True, verbose=False,
    )[0]
    yolo_masks, scores = yolo_masks_from_result(result)
    refined_masks = refine_masks(image_gray, yolo_masks) if yolo_masks else []
    disjoint_masks, overlap_pixels = make_disjoint_only_if_needed(
        refined_masks, scores, CFG.MIN_COMPONENT_SIZE
    )
    diagnostics.append({
        "file_name": info["file_name"],
        "gt_count": len(annotations.get(image_id, [])),
        "prediction_count": len(disjoint_masks),
        "overlap_pixels_before_repair": overlap_pixels,
        "instance_dice_512": instance_dice(
            annotation_masks(image_id), compact_masks(disjoint_masks)
        ),
    })

diagnostic_frame = pd.DataFrame(diagnostics)
diagnostic_frame.to_csv(ARTIFACT_ROOT / "v33_heldout_diagnostics.csv", index=False)
display(diagnostic_frame.describe())
print({
    "heldout_instance_dice_512": float(diagnostic_frame["instance_dice_512"].mean()),
    "重複発生画像": int((diagnostic_frame["overlap_pixels_before_repair"] > 0).sum()),
})
""",
        "v33-heldout",
    ),
    code_cell(
        """# [10/11] test推論と提出CSV作成
test_files = sorted(Path(CFG.TEST_DIR).glob("*.jpeg"))
rows = []
audit_rows = []

for image_path in tqdm(test_files, desc="V33 test推論"):
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    result = best_model.predict(
        str(image_path), imgsz=CFG.IMGSZ, conf=CFG.CONF_THRESHOLD,
        agnostic_nms=True, retina_masks=True, verbose=False,
    )[0]
    yolo_masks, scores = yolo_masks_from_result(result)
    refined_masks = refine_masks(image_gray, yolo_masks) if yolo_masks else []
    final_masks, overlap_pixels = make_disjoint_only_if_needed(
        refined_masks, scores, CFG.MIN_COMPONENT_SIZE
    )

    for index, mask in enumerate(final_masks, start=1):
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rows.append({
            "filament_id": f"{image_path.stem}_{index}",
            "segmentation_rle": encoded["counts"].decode("utf-8"),
        })
    audit_rows.append({
        "file_name": image_path.name,
        "yolo_count": len(yolo_masks),
        "submitted_count": len(final_masks),
        "overlap_pixels_before_repair": overlap_pixels,
    })

OUTPUT_PATH = Path("/content/submission_v33_public069_reproduction_disjoint.csv")
submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
submission.to_csv(OUTPUT_PATH, index=False, lineterminator="\\n")
audit = pd.DataFrame(audit_rows)
audit.to_csv(ARTIFACT_ROOT / "v33_test_audit.csv", index=False)
shutil.copy2(OUTPUT_PATH, ARTIFACT_ROOT / OUTPUT_PATH.name)

display(audit[["yolo_count", "submitted_count", "overlap_pixels_before_repair"]].describe())
print({
    "提出CSV": str(OUTPUT_PATH),
    "Driveコピー": str(ARTIFACT_ROOT / OUTPUT_PATH.name),
    "行数": len(submission),
})
""",
        "v33-submit",
    ),
    code_cell(
        """# [11/11] 公式RLE、null、ID、個体間重複を全件検証
checked = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
assert checked.columns.tolist() == ["filament_id", "segmentation_rle"]
assert len(checked) > 0
assert not checked.isna().any().any()
assert not checked["filament_id"].duplicated().any()
assert not checked["segmentation_rle"].str.strip().eq("").any()

test_ids = {path.stem for path in test_files}
submitted_ids = {value.rsplit("_", 1)[0] for value in checked["filament_id"]}
assert submitted_ids <= test_ids

occupied_by_image = {}
areas = []
for row in tqdm(checked.itertuples(index=False), total=len(checked), desc="提出CSV全件検証"):
    decoded = mask_utils.decode({
        "size": [2048, 2048], "counts": row.segmentation_rle.encode("utf-8")
    }).astype(bool)
    assert decoded.shape == (2048, 2048) and decoded.any()
    image_id = row.filament_id.rsplit("_", 1)[0]
    occupied = occupied_by_image.setdefault(image_id, np.zeros((2048, 2048), dtype=bool))
    assert not np.logical_and(occupied, decoded).any(), f"個体間重複: {image_id}"
    occupied |= decoded
    areas.append(int(decoded.sum()))

manifest = {
    "source_public_score": 0.69,
    "source_notebook": "https://www.kaggle.com/code/ektarr/yolo-unet-solar-filament-segmentation",
    "submission": str(OUTPUT_PATH),
    "drive_copy": str(ARTIFACT_ROOT / OUTPUT_PATH.name),
    "rows": len(checked),
    "test_images": len(test_ids),
    "images_with_predictions": len(submitted_ids),
    "overlap_pixels": 0,
    "area_min": min(areas),
    "area_median": int(np.median(areas)),
    "area_max": max(areas),
    "heldout_instance_dice_512": float(diagnostic_frame["instance_dice_512"].mean()),
}
(ARTIFACT_ROOT / "v33_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("検証完了。このCSVだけをKaggleへ提出してください:", OUTPUT_PATH)
""",
        "v33-validate",
    ),
]
