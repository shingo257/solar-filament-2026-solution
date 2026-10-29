"""Embedded cell payloads for the V34 semantic-instance hybrid notebook builder.

Verbatim extraction of the md()/code() cell-source literals from
build_v34_semantic_instance_hybrid.py so the builder stays readable.
No logic; loaded by the builder via sibling import.
"""

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
        """# 太陽フィラメント V34 — 高解像度semantic-first個体分割（Colab A100）

V30～V33で判明した主なボトルネックは、crop refinerの精度ではなく、YOLOが見逃した個体と
縮小maskで失われた細い枝を後段で復元できないことです。V34は処理順を逆転します。

1. 複数アノテータをsoft consensusとして使い、2048画像から高解像度patchでbinary semantic modelを3-fold学習
2. 太陽円盤の背景勾配補正、CLAHE、dark-detailを入力チャンネルとして利用
3. BCE + Dice + clDice + boundary lossで細いspine/barbを保持
4. V30の単一クラスYOLO 3-foldは最終maskではなく個体分離用のseedとしてだけ再利用
5. semantic foregroundをwatershedで分割し、個体maskを最初から排他的に生成
6. fold 0で後処理を調整し、全3-fold OOFで再確認してからtest推論を許可

**提出ゲート:** 全fold OOFの厳格instance DiceがV31の0.552553を明確に超え、個体数誤差も改善した場合だけ
test CSVを生成します。既定では未達時の強制提出を禁止します。

**必要環境:** Colab Pro+ A100を推奨。通常RAMで動作し、High-RAMは不要です。Google Driveの
`solar_filament_v30`にV30のcompetition data、fold定義、3個のYOLO weightが必要です。
外部MAGFiLOデータはルールとtest重複を確認できていないため、この版では使用しません。
"""
    ),
    code(
        """# [1/13] 必要パッケージ
%pip install -q "segmentation-models-pytorch>=0.4,<0.6" "timm>=1.0,<2" "ultralytics>=8.3,<9" pycocotools scikit-image kaggle
"""
    ),
    code(
        """# [2/13] ライブラリ、Drive、設定
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
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive, userdata
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold, StratifiedKFold
from skimage.segmentation import watershed
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")
drive.mount("/content/drive")

class CFG:
    SEED = 3407
    DEBUG = False
    COMPETITION = "filament-segmentation-2026"
    SOURCE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v34")
    DATA_CACHE = SOURCE_ROOT / "competition_data"
    SOURCE_ARTIFACTS = SOURCE_ROOT / "artifacts"
    SOURCE_RUNS = SOURCE_ROOT / "runs"
    LOCAL_ROOT = Path("/content/solar_v34")
    LOCAL_DATA = LOCAL_ROOT / "competition_data"
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    CONSENSUS_DIR = LOCAL_ROOT / "consensus"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_q97"
    CONSENSUS_CACHE = DRIVE_ROOT / "consensus_soft"
    RUNS_ROOT = DRIVE_ROOT / "runs"
    CACHE_ROOT = DRIVE_ROOT / "prediction_cache"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"

    N_FOLDS = 3
    FOLDS_TO_TRAIN = (0, 1, 2)
    TRAIN_PATCH = 768
    INFER_TILE = 1024
    INFER_OVERLAP = 256
    TRAIN_SAMPLES_PER_IMAGE = 1 if DEBUG else 5
    EPOCHS = 2 if DEBUG else 32
    PATIENCE = 7
    LR = 2e-4
    WEIGHT_DECAY = 2e-4
    BATCH_SIZE = 2
    ACCUM_STEPS = 2
    WORKERS = 2
    ENCODER = "timm-efficientnet-b4"
    FALLBACK_ENCODER = "resnet34"
    TTA_FLIPS = ("none", "h")

    YOLO_IMGSZ = 1280
    YOLO_PREDICT_CONF = 0.08
    YOLO_IOU = 0.55
    V31_STRICT_DICE = 0.5525530052850139
    V31_COUNT_ERROR = 1.8531073446327686
    REQUIRED_DICE_GAIN = 0.007
    ALLOW_TEST_WITHOUT_GATE = False

for path in [CFG.DRIVE_ROOT, CFG.RUNS_ROOT, CFG.CACHE_ROOT, CFG.ARTIFACT_ROOT,
             CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.CONSENSUS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()
assert torch.cuda.is_available(), "GPUランタイムを選択してください。"
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
if GPU_GB < 30:
    CFG.BATCH_SIZE = 1
    CFG.ACCUM_STEPS = 4
print({
    "torch": torch.__version__,
    "GPU": torch.cuda.get_device_name(0),
    "GPUメモリGB": round(GPU_GB, 1),
    "batch": CFG.BATCH_SIZE,
    "gradient_accumulation": CFG.ACCUM_STEPS,
    "High-RAM": "不要（画像は必要時にSSDから読み込み）",
})
"""
    ),
    code(
        """# [3/13] V30のcompetition dataをSSDへコピーし、COCOを安全に解析
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"学習annotation JSONが見つかりません: {base}")
    return matches[0].parents[2]

try:
    cached_root = find_dataset_root(CFG.DATA_CACHE)
except FileNotFoundError:
    CFG.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    username = userdata.get("KAGGLE_USERNAME")
    key = userdata.get("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError("V30 cacheがない場合はColab SecretsへKAGGLE_USERNAME/KAGGLE_KEYを登録してください。")
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    subprocess.run([
        "kaggle", "competitions", "download", "-c", CFG.COMPETITION,
        "-p", str(CFG.SOURCE_ROOT), "--force"
    ], check=True)
    archive = CFG.SOURCE_ROOT / f"{CFG.COMPETITION}.zip"
    shutil.unpack_archive(str(archive), str(CFG.DATA_CACHE))
    cached_root = find_dataset_root(CFG.DATA_CACHE)

if not CFG.LOCAL_DATA.exists():
    print("Driveのcompetition dataをColab SSDへコピーします...")
    shutil.copytree(CFG.DATA_CACHE, CFG.LOCAL_DATA)

DATA_ROOT = find_dataset_root(CFG.LOCAL_DATA)
TRAIN_JSON = next(DATA_ROOT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
TRAIN_DIR = TRAIN_JSON.parent / "train_images"
TEST_DIR = next(DATA_ROOT.rglob("test_images"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

# MAGFiLOのimage idは文字列の場合があるためintへ変換しない。
images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

train_files = sorted(file_to_ids)
test_files = sorted(path.name for path in TEST_DIR.glob("*.jpeg"))

def annotation_rank(image_id):
    anns = annotations[image_id]
    return len(anns), sum(float(ann.get("area", 0)) for ann in anns)

file_to_reference_id = {
    name: max(ids, key=annotation_rank) for name, ids in file_to_ids.items()
}
print({
    "train観測": len(train_files),
    "test観測": len(test_files),
    "複数アノテータ観測": sum(len(ids) > 1 for ids in file_to_ids.values()),
    "annotation数": len(coco["annotations"]),
    "data_root": str(DATA_ROOT),
})
assert len(test_files) == 180, "想定外のtest画像数です。データ構成を確認してください。"
"""
    ),
    code(
        """# [4/13] 太陽画像前処理、soft consensus教師、リークしないfold
def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)

def solar_three_channel(gray):
    # raw、半径方向の背景補正+CLAHE、dark-detailの3チャンネル。
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

def annotation_mask(image_id, ann):
    info = images[image_id]
    mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
    for polygon in ann.get("segmentation", []):
        if len(polygon) >= 6:
            points = np.rint(np.asarray(polygon, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
    return mask

def union_mask(image_id):
    info = images[image_id]
    output = np.zeros((info["height"], info["width"]), dtype=np.uint8)
    for ann in annotations[image_id]:
        output |= annotation_mask(image_id, ann)
    return output

def build_soft_consensus(file_name):
    variants = [union_mask(image_id).astype(np.float32) for image_id in file_to_ids[file_name]]
    return np.rint(np.mean(variants, axis=0) * 255).astype(np.uint8)

sources = [TRAIN_DIR / name for name in train_files] + [TEST_DIR / name for name in test_files]
expected_preproc = {path.name for path in sources}
CFG.PREPROC_CACHE.mkdir(parents=True, exist_ok=True)
CFG.CONSENSUS_CACHE.mkdir(parents=True, exist_ok=True)

for source in tqdm(sources, desc="太陽画像前処理"):
    cached = CFG.PREPROC_CACHE / source.name
    if not cached.exists():
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(source)
        cv2.imwrite(str(cached), solar_three_channel(gray), [cv2.IMWRITE_JPEG_QUALITY, 97])
    local = CFG.PREPROC_DIR / source.name
    if not local.exists():
        shutil.copy2(cached, local)

for file_name in tqdm(train_files, desc="soft consensus教師"):
    png_name = Path(file_name).with_suffix(".png").name
    cached = CFG.CONSENSUS_CACHE / png_name
    if not cached.exists():
        cv2.imwrite(str(cached), build_soft_consensus(file_name))
    local = CFG.CONSENSUS_DIR / png_name
    if not local.exists():
        shutil.copy2(cached, local)

fold_path = CFG.SOURCE_ARTIFACTS / "grouped_folds.csv"
if fold_path.exists():
    fold_frame = pd.read_csv(fold_path)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
else:
    counts = np.array([len(annotations[file_to_reference_id[name]]) for name in train_files])
    try:
        strata = pd.qcut(counts, q=min(5, len(np.unique(counts))), labels=False, duplicates="drop")
        splitter = StratifiedKFold(CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
        splits = splitter.split(train_files, np.asarray(strata))
    except ValueError:
        splitter = KFold(CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
        splits = splitter.split(train_files)
    fold_of_file = {}
    for fold, (_, valid_index) in enumerate(splits):
        for index in valid_index:
            fold_of_file[train_files[index]] = fold
    fold_frame = pd.DataFrame({"file_name": train_files, "fold": [fold_of_file[x] for x in train_files]})
    fold_frame.to_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv", index=False)

assert set(train_files) == set(fold_of_file)
display(pd.Series(fold_of_file).value_counts().sort_index().rename("観測数").to_frame())
"""
    ),
    code(
        """# [5/13] 高解像度patch datasetと細線・境界を守る損失
def random_patch(image, target, size, positive_probability=0.75):
    height, width = target.shape
    if random.random() < positive_probability and np.any(target >= 96):
        ys, xs = np.where(target >= 96)
        index = random.randrange(len(xs))
        cx, cy = int(xs[index]), int(ys[index])
        cx += random.randint(-size // 4, size // 4)
        cy += random.randint(-size // 4, size // 4)
        x0 = np.clip(cx - size // 2, 0, max(width - size, 0))
        y0 = np.clip(cy - size // 2, 0, max(height - size, 0))
    else:
        x0 = random.randint(0, max(width - size, 0))
        y0 = random.randint(0, max(height - size, 0))
    x0, y0 = int(x0), int(y0)
    return image[y0:y0 + size, x0:x0 + size], target[y0:y0 + size, x0:x0 + size]

class SemanticPatchDataset(Dataset):
    def __init__(self, file_names, training):
        self.file_names = list(file_names)
        self.training = training
        self.repeats = CFG.TRAIN_SAMPLES_PER_IMAGE if training else 1

    def __len__(self):
        return len(self.file_names) * self.repeats

    def __getitem__(self, index):
        name = self.file_names[index % len(self.file_names)]
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        target = cv2.imread(
            str(CFG.CONSENSUS_DIR / Path(name).with_suffix(".png").name),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None or target is None:
            raise FileNotFoundError(name)
        if self.training:
            image, target = random_patch(image, target, CFG.TRAIN_PATCH)
            k = random.randrange(4)
            image, target = np.rot90(image, k).copy(), np.rot90(target, k).copy()
            if random.random() < 0.5:
                image, target = image[:, ::-1].copy(), target[:, ::-1].copy()
            if random.random() < 0.5:
                image, target = image[::-1].copy(), target[::-1].copy()
            gain = random.uniform(0.88, 1.12)
            gamma = random.uniform(0.85, 1.18)
            image = np.clip((image.astype(np.float32) / 255.0) ** gamma * gain, 0, 1)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0, 0.012, image.shape), 0, 1)
        else:
            # epoch中のcheckpoint選択用。最終評価は2048 tile OOFで別に行う。
            image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            target = cv2.resize(target, (768, 768), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        target = torch.from_numpy(target.copy()).float().unsqueeze(0) / 255.0
        return image, target

def soft_dice_loss(logits, target):
    prob = logits.sigmoid()
    intersection = (prob * target).sum((1, 2, 3))
    denominator = prob.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()

def soft_erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.minimum(p1, p2)

def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)

def soft_open(x):
    return soft_dilate(soft_erode(x))

def soft_skeleton(x, iterations=8):
    opened = soft_open(x)
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton

def cldice_loss(logits, target):
    # 計算量を抑えるため1/2解像度でtopologyを評価する。
    prob = F.avg_pool2d(logits.sigmoid(), 2)
    truth = F.avg_pool2d(target, 2)
    pred_skeleton = soft_skeleton(prob)
    true_skeleton = soft_skeleton(truth)
    precision = (pred_skeleton * truth).sum((1, 2, 3)) / (pred_skeleton.sum((1, 2, 3)) + 1e-6)
    sensitivity = (true_skeleton * prob).sum((1, 2, 3)) / (true_skeleton.sum((1, 2, 3)) + 1e-6)
    return (1 - (2 * precision * sensitivity + 1e-6) / (precision + sensitivity + 1e-6)).mean()

def boundary_loss(logits, target):
    prob = logits.sigmoid()
    dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=prob.device, dtype=prob.dtype).view(1, 1, 3, 3)
    dy = dx.transpose(2, 3)
    pred_edge = torch.sqrt(F.conv2d(prob, dx, padding=1).square() + F.conv2d(prob, dy, padding=1).square() + 1e-6)
    true_edge = torch.sqrt(F.conv2d(target, dx, padding=1).square() + F.conv2d(target, dy, padding=1).square() + 1e-6)
    return F.l1_loss(pred_edge, true_edge)

def combined_loss(logits, target):
    positive = target.sum().clamp_min(1)
    negative = target.numel() - positive
    pos_weight = (negative / positive).clamp(3, 25).detach()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return 0.30 * bce + 0.45 * soft_dice_loss(logits, target) + 0.20 * cldice_loss(logits, target) + 0.05 * boundary_loss(logits, target)
"""
    ),
    code(
        """# [6/13] 3-fold semantic modelを学習（epochごとにDriveへ保存・再開可能）
def make_model(encoder=CFG.ENCODER):
    try:
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",
        )
        return model, encoder
    except Exception as error:
        print(f"{encoder}を作成できないため{CFG.FALLBACK_ENCODER}へ切替: {error}")
        model = smp.Unet(
            encoder_name=CFG.FALLBACK_ENCODER,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",
        )
        return model, CFG.FALLBACK_ENCODER

@torch.no_grad()
def validation_union_dice(model, loader):
    model.eval()
    values = []
    for images_batch, targets_batch in loader:
        images_batch = images_batch.cuda(non_blocking=True)
        targets_batch = targets_batch.cuda(non_blocking=True) >= 0.5
        with torch.autocast("cuda", dtype=torch.float16):
            predictions = model(images_batch).sigmoid() >= 0.45
        intersection = (predictions & targets_batch).sum((1, 2, 3)).float()
        denominator = predictions.sum((1, 2, 3)) + targets_batch.sum((1, 2, 3))
        values.extend(((2 * intersection + 1) / (denominator + 1)).cpu().tolist())
    return float(np.mean(values))

def train_fold(fold):
    run_dir = CFG.RUNS_ROOT / f"semantic_fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    done_path = run_dir / "TRAINING_COMPLETE"
    if done_path.exists() and best_path.exists():
        print(f"fold {fold}: 完了済みweightを使用 {best_path}")
        return best_path

    train_names = [name for name in train_files if fold_of_file[name] != fold]
    valid_names = [name for name in train_files if fold_of_file[name] == fold]
    train_loader = DataLoader(
        SemanticPatchDataset(train_names, True), batch_size=CFG.BATCH_SIZE,
        shuffle=True, num_workers=CFG.WORKERS, pin_memory=True,
        persistent_workers=CFG.WORKERS > 0, drop_last=True,
    )
    valid_loader = DataLoader(
        SemanticPatchDataset(valid_names, False), batch_size=max(1, CFG.BATCH_SIZE * 2),
        shuffle=False, num_workers=CFG.WORKERS, pin_memory=True,
        persistent_workers=CFG.WORKERS > 0,
    )
    model, encoder_used = make_model()
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=CFG.LR * 0.05)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_score, stale = 1, -1.0, 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_score = state["best_score"]
        stale = state.get("stale", 0)
        print(f"fold {fold}: epoch {start_epoch}から再開")

    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = []
        progress = tqdm(train_loader, desc=f"fold {fold} epoch {epoch}/{CFG.EPOCHS}")
        for step, (images_batch, targets_batch) in enumerate(progress, 1):
            images_batch = images_batch.cuda(non_blocking=True)
            targets_batch = targets_batch.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = combined_loss(model(images_batch), targets_batch) / CFG.ACCUM_STEPS
            scaler.scale(loss).backward()
            if step % CFG.ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running.append(loss.item() * CFG.ACCUM_STEPS)
            progress.set_postfix(loss=f"{np.mean(running[-30:]):.4f}")
        score = validation_union_dice(model, valid_loader)
        scheduler.step()
        improved = score > best_score
        if improved:
            best_score, stale = score, 0
            torch.save({"model": model.state_dict(), "encoder": encoder_used, "score": score}, best_path)
        else:
            stale += 1
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_score": best_score, "stale": stale, "encoder": encoder_used,
        }
        torch.save(state, last_path)
        print({"fold": fold, "epoch": epoch, "loss": np.mean(running), "val_union_dice_768": score, "best": best_score})
        if stale >= CFG.PATIENCE:
            print(f"fold {fold}: early stopping")
            break

    done_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    del model, optimizer, train_loader, valid_loader
    gc.collect(); torch.cuda.empty_cache()
    return best_path

semantic_weight_paths = {fold: train_fold(fold) for fold in CFG.FOLDS_TO_TRAIN}
print({fold: str(path) for fold, path in semantic_weight_paths.items()})
"""
    ),
    code(
        """# [7/13] 2048画像をoverlap tile + flip TTAでsemantic推論
def load_semantic_model(weight_path):
    state = torch.load(weight_path, map_location="cpu")
    model, _ = make_model(state.get("encoder", CFG.ENCODER))
    model.load_state_dict(state["model"])
    return model.cuda().eval()

def tile_positions(length, tile, overlap):
    if length <= tile:
        return [0]
    stride = tile - overlap
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions

def flip_array(array, mode):
    if mode == "h": return array[:, ::-1].copy()
    if mode == "v": return array[::-1].copy()
    if mode == "hv": return array[::-1, ::-1].copy()
    return array

@torch.no_grad()
def predict_semantic_full(model, image, tta_flips=CFG.TTA_FLIPS):
    height, width = image.shape[:2]
    tile = CFG.INFER_TILE
    ys = tile_positions(height, tile, CFG.INFER_OVERLAP)
    xs = tile_positions(width, tile, CFG.INFER_OVERLAP)
    axis = np.linspace(-1, 1, tile, dtype=np.float32)
    window_1d = np.clip(1 - np.abs(axis), 0.15, 1.0)
    window = np.outer(window_1d, window_1d)
    total = np.zeros((height, width), np.float32)
    weight = np.zeros((height, width), np.float32)
    for mode in tta_flips:
        transformed = flip_array(image, mode)
        mode_total = np.zeros((height, width), np.float32)
        mode_weight = np.zeros((height, width), np.float32)
        for y0 in ys:
            for x0 in xs:
                patch = transformed[y0:y0 + tile, x0:x0 + tile]
                tensor = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float().unsqueeze(0).cuda() / 255.0
                with torch.autocast("cuda", dtype=torch.float16):
                    probability = model(tensor).sigmoid()[0, 0].float().cpu().numpy()
                mode_total[y0:y0 + tile, x0:x0 + tile] += probability * window
                mode_weight[y0:y0 + tile, x0:x0 + tile] += window
        mode_probability = mode_total / np.maximum(mode_weight, 1e-6)
        mode_probability = flip_array(mode_probability, mode)
        total += mode_probability
        weight += 1
    return total / np.maximum(weight, 1)

def semantic_cache_path(split, fold, file_name):
    directory = CFG.CACHE_ROOT / split / f"semantic_fold_{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / Path(file_name).with_suffix(".png").name

def save_probability(path, probability):
    cv2.imwrite(str(path), np.rint(np.clip(probability, 0, 1) * 65535).astype(np.uint16))

def load_probability(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
"""
    ),
    code(
        """# [8/13] YOLO seed、semantic foregroundの排他的watershed分割、評価指標
yolo_weight_paths = {
    fold: CFG.SOURCE_RUNS / f"yolo_fold_{fold}" / "weights" / "best.pt"
    for fold in range(CFG.N_FOLDS)
}
for path in yolo_weight_paths.values():
    if not path.exists():
        raise FileNotFoundError(f"V30 YOLO weightがありません: {path}")
yolo_models = {fold: YOLO(str(path)) for fold, path in yolo_weight_paths.items()}

def mask_iou(left, right):
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0

def nms_masks(masks, confidences, iou_threshold=0.45):
    if not masks:
        return [], np.empty(0, np.float32)
    order = np.argsort(confidences)[::-1]
    kept_masks, kept_confidences = [], []
    for index in order:
        candidate = masks[index].astype(bool)
        if any(mask_iou(candidate, kept) >= iou_threshold for kept in kept_masks):
            continue
        kept_masks.append(candidate)
        kept_confidences.append(float(confidences[index]))
    return kept_masks, np.asarray(kept_confidences, np.float32)

def predict_yolo_seeds(model, image_path):
    result = model.predict(
        str(image_path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.YOLO_PREDICT_CONF,
        iou=CFG.YOLO_IOU, retina_masks=True, agnostic_nms=True, verbose=False,
    )[0]
    if result.masks is None:
        return [], np.empty(0, np.float32)
    height, width = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).shape
    masks = []
    for tensor in result.masks.data:
        mask = tensor.cpu().numpy()
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        masks.append(mask >= 0.5)
    confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
    return nms_masks(masks, confidences)

def yolo_cache_path(split, fold, file_name):
    directory = CFG.CACHE_ROOT / split / f"yolo_fold_{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{Path(file_name).stem}.npz"

def save_yolo_cache(path, masks, confidences):
    if masks:
        packed = np.packbits(np.stack(masks).astype(np.uint8), axis=-1)
        shape = np.asarray(masks[0].shape, np.int32)
    else:
        packed = np.empty((0, 0, 0), np.uint8)
        shape = np.asarray([2048, 2048], np.int32)
    np.savez_compressed(path, packed=packed, shape=shape, confidences=confidences)

def load_yolo_cache(path):
    data = np.load(path)
    height, width = map(int, data["shape"])
    packed = data["packed"]
    masks = [np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed]
    return masks, data["confidences"].astype(np.float32)

def resize_masks(masks, size):
    return [cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool) for mask in masks]

def semantic_to_instances(probability, seed_masks, confidences, threshold, seed_conf, min_area, recover=True):
    foreground = probability >= threshold
    if not foreground.any():
        return []
    selected = [(mask, conf) for mask, conf in zip(seed_masks, confidences) if conf >= seed_conf]
    selected_masks, selected_confidences = nms_masks(
        [item[0] for item in selected], [item[1] for item in selected], iou_threshold=0.48
    ) if selected else ([], np.empty(0, np.float32))
    markers = np.zeros(foreground.shape, np.int32)
    next_label = 1
    kernel = np.ones((3, 3), np.uint8)
    for mask, _ in zip(selected_masks, selected_confidences):
        core = mask & (probability >= max(0.20, threshold * 0.65))
        core = cv2.erode(core.astype(np.uint8), kernel, iterations=1).astype(bool)
        core &= markers == 0
        if core.any():
            markers[core] = next_label
            next_label += 1

    if recover:
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
        for component in range(1, component_count):
            region = components == component
            if stats[component, cv2.CC_STAT_AREA] < max(min_area * 2, 40):
                continue
            if np.any(markers[region] > 0):
                continue
            values = np.where(region, probability, -1)
            y, x = np.unravel_index(np.argmax(values), values.shape)
            if probability[y, x] >= min(0.98, threshold + 0.10):
                markers[y, x] = next_label
                next_label += 1

    if next_label == 1:
        return []
    labels = watershed(-probability, markers=markers, mask=foreground, watershed_line=True)
    instances = []
    for label in range(1, next_label):
        mask = labels == label
        if mask.sum() >= min_area:
            instances.append(mask)
    return instances

def gt_variants(file_name, size=None):
    variants = []
    for image_id in file_to_ids[file_name]:
        masks = [annotation_mask(image_id, ann).astype(bool) for ann in annotations[image_id]]
        if size is not None:
            masks = resize_masks(masks, size)
        variants.append(masks)
    return variants

def strict_instance_dice(ground_truth, predictions):
    if not ground_truth and not predictions:
        return 1.0
    if not ground_truth or not predictions:
        return 0.0
    matrix = np.zeros((len(ground_truth), len(predictions)), np.float32)
    for i, truth in enumerate(ground_truth):
        truth_area = truth.sum()
        for j, prediction in enumerate(predictions):
            denominator = truth_area + prediction.sum()
            matrix[i, j] = 2 * np.logical_and(truth, prediction).sum() / max(denominator, 1)
    rows, columns = linear_sum_assignment(-matrix)
    return float(matrix[rows, columns].sum() / max(len(ground_truth), len(predictions)))

def fragmentation_counts(ground_truth, predictions, overlap_fraction=0.10):
    splits = 0
    for truth in ground_truth:
        matches = sum(np.logical_and(truth, pred).sum() / max(truth.sum(), 1) >= overlap_fraction for pred in predictions)
        splits += max(matches - 1, 0)
    merges = 0
    for pred in predictions:
        matches = sum(np.logical_and(truth, pred).sum() / max(pred.sum(), 1) >= overlap_fraction for truth in ground_truth)
        merges += max(matches - 1, 0)
    return splits, merges
"""
    ),
    code(
        """# [9/13] 各foldのheld-out semantic確率とYOLO seedをDriveへキャッシュ
for fold in CFG.FOLDS_TO_TRAIN:
    valid_names = [name for name in train_files if fold_of_file[name] == fold]
    missing_semantic = [name for name in valid_names if not semantic_cache_path("oof", fold, name).exists()]
    if missing_semantic:
        semantic_model = load_semantic_model(semantic_weight_paths[fold])
        for name in tqdm(missing_semantic, desc=f"fold {fold} semantic OOF"):
            image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
            probability = predict_semantic_full(semantic_model, image)
            save_probability(semantic_cache_path("oof", fold, name), probability)
        del semantic_model
        gc.collect(); torch.cuda.empty_cache()
    for name in tqdm(valid_names, desc=f"fold {fold} YOLO OOF"):
        path = yolo_cache_path("oof", fold, name)
        if path.exists():
            continue
        masks, confidences = predict_yolo_seeds(yolo_models[fold], CFG.PREPROC_DIR / name)
        save_yolo_cache(path, masks, confidences)
print("OOF cache完了")
"""
    ),
    code(
        """# [10/13] fold 0で後処理を探索し、上位候補を全3-fold OOFで確認
CALIB_SIZE = 512

def evaluate_parameters(file_names, threshold, seed_conf, min_area, recover=True, size=CALIB_SIZE):
    rows = []
    for name in tqdm(file_names, leave=False):
        fold = fold_of_file[name]
        probability = load_probability(semantic_cache_path("oof", fold, name))
        probability = cv2.resize(probability, (size, size), interpolation=cv2.INTER_AREA)
        masks, confidences = load_yolo_cache(yolo_cache_path("oof", fold, name))
        masks = resize_masks(masks, size)
        scaled_min_area = max(2, int(round(min_area * (size / 2048) ** 2)))
        predictions = semantic_to_instances(
            probability, masks, confidences, threshold, seed_conf, scaled_min_area, recover
        )
        variant_scores, split_values, merge_values, counts = [], [], [], []
        for truth in gt_variants(name, size):
            variant_scores.append(strict_instance_dice(truth, predictions))
            split_count, merge_count = fragmentation_counts(truth, predictions)
            split_values.append(split_count)
            merge_values.append(merge_count)
            counts.append(abs(len(truth) - len(predictions)))
        rows.append({
            "dice": np.mean(variant_scores), "count_error": np.mean(counts),
            "splits": np.mean(split_values), "merges": np.mean(merge_values),
        })
    frame = pd.DataFrame(rows)
    return {
        "threshold": threshold, "seed_conf": seed_conf, "min_area": min_area, "recover": recover,
        "strict_instance_dice": frame["dice"].mean(),
        "mean_abs_count_error": frame["count_error"].mean(),
        "mean_splits": frame["splits"].mean(), "mean_merges": frame["merges"].mean(),
    }

fold0_names = [name for name in train_files if fold_of_file[name] == 0]
grid = []
for threshold in (0.34, 0.40, 0.46, 0.52):
    for seed_conf in (0.10, 0.16, 0.22):
        for min_area in (20, 60):
            grid.append(evaluate_parameters(fold0_names, threshold, seed_conf, min_area, True))
grid_frame = pd.DataFrame(grid).sort_values(
    ["strict_instance_dice", "mean_abs_count_error"], ascending=[False, True]
).reset_index(drop=True)
display(grid_frame.head(15))

all_fold_candidates = []
for candidate in grid_frame.head(3).to_dict("records"):
    result = evaluate_parameters(
        train_files, candidate["threshold"], candidate["seed_conf"],
        int(candidate["min_area"]), bool(candidate["recover"]),
    )
    all_fold_candidates.append(result)
all_fold_frame = pd.DataFrame(all_fold_candidates).sort_values(
    ["strict_instance_dice", "mean_abs_count_error"], ascending=[False, True]
).reset_index(drop=True)
display(all_fold_frame)
best_parameters = all_fold_frame.iloc[0].to_dict()
all_fold_frame.to_csv(CFG.ARTIFACT_ROOT / "v34_oof_calibration.csv", index=False)
print("V34 OOF最良:", best_parameters)
"""
    ),
    code(
        """# [11/13] 提出ゲートと代表画像の可視化
PASS_DICE = best_parameters["strict_instance_dice"] >= CFG.V31_STRICT_DICE + CFG.REQUIRED_DICE_GAIN
PASS_COUNT = best_parameters["mean_abs_count_error"] < CFG.V31_COUNT_ERROR
PASS_GATE = bool(PASS_DICE and PASS_COUNT)
gate_report = {
    "V31_strict_dice": CFG.V31_STRICT_DICE,
    "V34_strict_dice_512": best_parameters["strict_instance_dice"],
    "必要改善幅": CFG.REQUIRED_DICE_GAIN,
    "V31_count_error": CFG.V31_COUNT_ERROR,
    "V34_count_error": best_parameters["mean_abs_count_error"],
    "Dice合格": bool(PASS_DICE), "個体数誤差合格": bool(PASS_COUNT),
    "test推論許可": PASS_GATE or CFG.ALLOW_TEST_WITHOUT_GATE,
}
print(json.dumps(gate_report, ensure_ascii=False, indent=2))
with open(CFG.ARTIFACT_ROOT / "v34_gate_report.json", "w", encoding="utf-8") as handle:
    json.dump({"gate": gate_report, "parameters": best_parameters}, handle, ensure_ascii=False, indent=2)

sample_names = fold0_names[:4]
figure, axes = plt.subplots(len(sample_names), 3, figsize=(15, 4 * len(sample_names)))
for row, name in enumerate(sample_names):
    probability = load_probability(semantic_cache_path("oof", 0, name))
    masks, confidences = load_yolo_cache(yolo_cache_path("oof", 0, name))
    predictions = semantic_to_instances(
        probability, masks, confidences, best_parameters["threshold"],
        best_parameters["seed_conf"], int(best_parameters["min_area"]), True,
    )
    image = cv2.cvtColor(cv2.imread(str(CFG.PREPROC_DIR / name)), cv2.COLOR_BGR2RGB)
    truth_union = np.logical_or.reduce(gt_variants(name)[0])
    prediction_union = np.logical_or.reduce(predictions) if predictions else np.zeros_like(truth_union)
    overlay = image.copy()
    overlay[truth_union & prediction_union] = [0, 255, 0]
    overlay[truth_union & ~prediction_union] = [255, 255, 0]
    overlay[~truth_union & prediction_union] = [255, 0, 0]
    axes[row, 0].imshow(image); axes[row, 0].set_title(name); axes[row, 0].axis("off")
    axes[row, 1].imshow(probability, cmap="magma", vmin=0, vmax=1); axes[row, 1].set_title("semantic probability"); axes[row, 1].axis("off")
    axes[row, 2].imshow(cv2.addWeighted(image, 0.45, overlay, 0.55, 0)); axes[row, 2].set_title(f"GT/予測: {len(gt_variants(name)[0])}/{len(predictions)}"); axes[row, 2].axis("off")
plt.tight_layout()
plt.savefig(CFG.ARTIFACT_ROOT / "v34_oof_visual_audit.png", dpi=140)
plt.show()
"""
    ),
    code(
        """# [12/13] ゲート合格時だけ3-fold semantic + YOLO seedでtest推論
SUBMISSION_PATH = Path("/content/submission_v34_semantic_first_instance_hybrid.csv")
DRIVE_SUBMISSION = CFG.ARTIFACT_ROOT / SUBMISSION_PATH.name

if not (PASS_GATE or CFG.ALLOW_TEST_WITHOUT_GATE):
    print("OOFゲート未達のためtest推論と提出CSV生成を停止しました。calibration表と可視化を確認してください。")
else:
    semantic_models = {fold: load_semantic_model(path) for fold, path in semantic_weight_paths.items()}
    rows, count_rows = [], []
    for name in tqdm(test_files, desc="V34 test推論"):
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        fold_probabilities = []
        all_masks, all_confidences = [], []
        for fold in CFG.FOLDS_TO_TRAIN:
            probability_path = semantic_cache_path("test", fold, name)
            if probability_path.exists():
                probability = load_probability(probability_path)
            else:
                probability = predict_semantic_full(semantic_models[fold], image)
                save_probability(probability_path, probability)
            fold_probabilities.append(probability)

            seed_path = yolo_cache_path("test", fold, name)
            if seed_path.exists():
                masks, confidences = load_yolo_cache(seed_path)
            else:
                masks, confidences = predict_yolo_seeds(yolo_models[fold], CFG.PREPROC_DIR / name)
                save_yolo_cache(seed_path, masks, confidences)
            all_masks.extend(masks)
            all_confidences.extend(confidences.tolist())

        probability = np.mean(fold_probabilities, axis=0)
        seed_masks, seed_confidences = nms_masks(all_masks, all_confidences, iou_threshold=0.42)
        instances = semantic_to_instances(
            probability, seed_masks, seed_confidences,
            float(best_parameters["threshold"]), float(best_parameters["seed_conf"]),
            int(best_parameters["min_area"]), bool(best_parameters["recover"]),
        )
        image_key = Path(name).stem
        for index, mask in enumerate(instances, 1):
            encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rows.append({
                "filament_id": f"{image_key}_{index}",
                "segmentation_rle": encoded["counts"].decode("utf-8"),
            })
        count_rows.append({"file_name": name, "instances": len(instances), "seeds": len(seed_masks)})

    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    submission.to_csv(SUBMISSION_PATH, index=False)
    shutil.copy2(SUBMISSION_PATH, DRIVE_SUBMISSION)
    count_frame = pd.DataFrame(count_rows)
    count_frame.to_csv(CFG.ARTIFACT_ROOT / "v34_test_counts.csv", index=False)
    display(count_frame[["instances", "seeds"]].describe())
    print({"提出CSV": str(SUBMISSION_PATH), "Driveコピー": str(DRIVE_SUBMISSION), "行数": len(submission)})
"""
    ),
    code(
        """# [13/13] 提出CSVの全件検証（null、重複ID、RLE、個体間重複）
if not SUBMISSION_PATH.exists():
    print("CSVは生成されていません。OOFゲート結果を確認してください。")
else:
    submission = pd.read_csv(SUBMISSION_PATH, keep_default_na=False)
    assert list(submission.columns) == ["filament_id", "segmentation_rle"]
    assert len(submission) > 0
    assert not submission.isna().any().any()
    assert (submission["segmentation_rle"].str.len() > 0).all()
    assert submission["filament_id"].is_unique
    grouped = defaultdict(list)
    area_values = []
    for row in tqdm(submission.itertuples(index=False), total=len(submission), desc="提出CSV全件検証"):
        image_key = row.filament_id.rsplit("_", 1)[0]
        decoded = mask_utils.decode({
            "size": [2048, 2048], "counts": row.segmentation_rle.encode("utf-8")
        }).astype(bool)
        assert decoded.shape == (2048, 2048)
        assert decoded.any()
        grouped[image_key].append(decoded)
        area_values.append(int(decoded.sum()))
    overlap_pixels = 0
    for masks in grouped.values():
        occupancy = np.zeros((2048, 2048), np.uint16)
        for mask in masks:
            occupancy += mask
        overlap_pixels += int((occupancy > 1).sum())
    assert overlap_pixels == 0, f"個体間重複があります: {overlap_pixels} pixels"
    report = {
        "submission": str(SUBMISSION_PATH), "drive_copy": str(DRIVE_SUBMISSION),
        "rows": len(submission), "test_images": len(test_files),
        "images_with_predictions": len(grouped), "overlap_pixels": overlap_pixels,
        "area_min": int(np.min(area_values)), "area_median": int(np.median(area_values)),
        "area_max": int(np.max(area_values)), "oof": best_parameters, "gate": gate_report,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    with open(CFG.ARTIFACT_ROOT / "v34_submission_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    from google.colab import files
    files.download(str(SUBMISSION_PATH))
"""
    ),
]
