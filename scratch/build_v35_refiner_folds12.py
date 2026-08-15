"""fold1 / fold2 の MediSeg refiner を学習する Colab ノートブックのビルダー。

なぜ必要か:
  refiner の checkpoint は現在 fold0 用 (`mediseg_refiner_fold0_resume.pt`) と
  test 用 all-data 版しか存在しない。all-data 版は全 train を学習に含むので OOF
  評価に使えず、結果として **後処理の選択が fold0 だけで行われてきた**。
  V31 → V34 を通じて繰り返し問題になった選択バイアスの根本原因がこれである。
  fold1 / fold2 の refiner を用意すれば、全 fold OOF でリークなしに後処理を選べる。

重要な設計方針:
  **V30 の学習条件を1つも変えない。** crop size, context, loss, epoch数, LR,
  scheduler, augmentation, さらに「crop Dice で best を選ぶ」点まで同一にする。
  ここで条件を改善してしまうと fold0 の既存 checkpoint と比較できなくなり、
  fold 間の差が「モデル差」なのか「学習条件差」なのか判別できなくなる。
  refiner 自体の改良は、全 fold の土台が揃ってから別の実験として行う。

コスト目安: 1 epoch 約1分 (§28) x 14 epoch x 2 fold + データ準備。T4 で 1 CU 未満。
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v35_refiner_folds12_colab.ipynb"
MODELS = (ROOT / "src" / "mediseg_models.py").read_text(encoding="utf-8")


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V35 — fold1 / fold2 の MediSeg refiner を学習する

## 目的

後処理の選択を **全 fold OOF** で、リークなしに行えるようにする。

現在 refiner の checkpoint は 2 つしかない。

| checkpoint | 学習データ | OOF評価に使えるか |
|---|---|---|
| `mediseg_refiner_fold0_resume.pt` | fold1 + fold2 | fold0 の held-out にのみ使える |
| `mediseg_refiner_all_data_resume.pt` | 全 train | **使えない**（全画像を学習済み） |

このため V31 以降の後処理選択はすべて fold0 限定で行われてきた。
これが V31 → V34 で繰り返し問題になった選択バイアスの根本原因である。

## なぜ今これをやるか

ローカルPCでの実測で、**refiner の方式と閾値の選び方が Panoptic Quality を大きく動かす**
ことが分かった。V31 は Hungarian Dice を根拠に `guided7@0.45` を選んだが、
PQ ではそれが精製版の中で最下位だった。V31a の Public が 0.35 → 0.34 と下がった
のはこれで説明がつく。

正しい方式を **全 fold** で選び直すために、fold1 / fold2 の refiner が要る。

## 設計方針: V30 の学習条件を1つも変えない

crop size 320、context 1.65、loss = 0.35·BCE + 0.65·softDice + 0.15·edge、
14 epoch、LR 2e-4、CosineAnnealing、AMP、そして **best を crop Dice で選ぶ**点まで
そのまま踏襲する。

ここで条件を「改善」すると fold0 の既存 checkpoint と比較不能になり、
fold 間の差がモデル差なのか学習条件差なのか判別できなくなる。
refiner 自体の改良は、全 fold の土台が揃ってから別実験として行う。

GPU が要る。T4 で十分（1 epoch 約1分、2 fold で 30 分程度、1 CU 未満）。
"""
    ),
    code(
        """# [1/6] Drive接続、設定、前提の確認
import json, math, os, random, shutil, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from google.colab import drive
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

drive.mount("/content/drive")

class CFG:
    SEED = 3407
    DEBUG = False
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v35")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    # --- 以下は V30 と完全に同一。触らないこと ---
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_EPOCHS = 2 if DEBUG else 14
    REFINER_LR = 2e-4
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    FOLDS_TO_TRAIN = (1, 2)

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像": CFG.PREPROC_CACHE,
    "fold0 refiner (比較用)": CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt",
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0)})
"""
    ),
    code(
        """# [2/6] COCOとfoldを読み、前処理画像をローカルSSDへ展開
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"学習annotation JSONが見つかりません: {base}")
    return matches[0].parents[2]

TRAIN_JSON = next(find_dataset_root(CFG.DATA_CACHE).rglob(
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

# image id は文字列のことがある。int へ変換しない。
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
assert all(name in fold_of_file for name in train_files), "fold未割当の観測がある"

# Driveから直接読むと1画像ごとのI/Oが遅い。SSDへコピーする。
for name in tqdm(train_files, desc="前処理画像をSSDへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)

try:
    from pycocotools import mask as coco_mask
except ImportError:
    !pip -q install pycocotools
    from pycocotools import mask as coco_mask

def ann_mask(image_id, ann):
    info = images[image_id]
    height, width = int(info["height"]), int(info["width"])
    segmentation = ann["segmentation"]
    if isinstance(segmentation, list):
        rle = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get("counts"), list):
        rle = coco_mask.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return coco_mask.decode(rle).astype(np.uint8)

print({"観測": len(train_files),
       "fold別": {f: sum(v == f for v in fold_of_file.values()) for f in (0, 1, 2)},
       "annotation": len(coco["annotations"])})
"""
    ),
    code("# [3/6] MediSegUNet (src/mediseg_models.py と同一)\n" + MODELS),
    code(
        """# [4/6] Dataset・loss・学習ループ (すべてV30と同一)
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
    def __init__(self, file_names, all_labelers=True, augment=False):
        self.samples, self.augment = [], augment
        for file_name in file_names:
            for image_id in (file_to_ids[file_name] if all_labelers else []):
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

def choose_batch_size(dataset):
    for candidate in CFG.REFINER_BATCH_CANDIDATES:
        try:
            loader = DataLoader(dataset, batch_size=candidate, shuffle=True, num_workers=2, pin_memory=True)
            image, target = next(iter(loader))
            probe = MediSegUNet(3, 1, False).cuda()
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(probe(image.cuda()), target.cuda())
            loss.backward()
            del probe, image, target, loss, loader
            torch.cuda.empty_cache()
            return candidate
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("どのbatch sizeもGPUメモリに収まらない")

def train_refiner(fold, epochs=CFG.REFINER_EPOCHS):
    \"\"\"対象foldをvalidにし、それ以外で学習する。V30のtrain_refinerと同一。\"\"\"
    train_files_fold = [n for n in train_files if fold_of_file[n] != fold]
    valid_files_fold = [n for n in train_files if fold_of_file[n] == fold]
    train_ds = InstanceCropDataset(train_files_fold, augment=True)
    valid_ds = InstanceCropDataset(valid_files_fold, augment=False)
    # リーク検査: validのfileがtrainへ混入していないこと (§36 即時停止条件)
    assert not (set(train_files_fold) & set(valid_files_fold)), "fold分割が壊れている"
    print({"fold": fold, "train_crops": len(train_ds), "valid_crops": len(valid_ds)})

    batch_size = choose_batch_size(train_ds)
    print("batch_size =", batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    checkpoint_path = CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{fold}_resume.pt"
    model = MediSegUNet(3, 1, False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_dice, best_state = 0, -1.0, None
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_dice = state["epoch"] + 1, state["best_dice"]
        best_state = state.get("best_model")
        print(f"epoch {start_epoch} から再開")

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"fold{fold} refiner {epoch + 1}/{epochs}")
        for image, target in progress:
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(model(image), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{running / (progress.n + 1):.4f}")
        scheduler.step()
        dice = crop_dice(model, valid_loader)
        if dice > best_dice:
            best_dice = dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"epoch": epoch, "model": model.state_dict(), "best_model": best_state,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice}, checkpoint_path)
        print(f"fold{fold} epoch {epoch + 1}: crop Dice {dice:.6f} (best {best_dice:.6f})")
    return checkpoint_path, best_dice
"""
    ),
    code(
        """# [5/6] fold1 と fold2 を学習
results = {}
for fold in CFG.FOLDS_TO_TRAIN:
    started = time.time()
    path, best = train_refiner(fold)
    results[fold] = {"checkpoint": str(path), "best_crop_dice": best,
                     "minutes": round((time.time() - started) / 60, 1)}
    print(results[fold])
print(json.dumps(results, ensure_ascii=False, indent=2))
"""
    ),
    code(
        """# [6/6] 検証とmanifest
import hashlib

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "experiment_id": "v35_refiner_folds12",
    "purpose": "全fold OOFでリークなしに後処理を選べるようにする",
    "recipe": "V30と完全に同一 (crop320/context1.65/14epoch/LR2e-4/crop Diceでbest選択)",
    "folds_trained": list(CFG.FOLDS_TO_TRAIN),
    "results": results,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "checkpoints": {},
}
for fold in (0, 1, 2):
    path = CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{fold}_resume.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu")
        # best_model が None のまま保存されていないことを確認する
        assert state.get("best_model") is not None, f"fold{fold}: best_model が空"
        manifest["checkpoints"][f"fold{fold}"] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            "best_dice": state.get("best_dice"),
            "epoch": state.get("epoch"),
        }
    else:
        manifest["checkpoints"][f"fold{fold}"] = "missing"

report_path = CFG.ARTIFACT_ROOT / "v35_refiner_folds12_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"\\n保存: {report_path}")
print("\\n次: ローカルPCで scratch/run_refiner_pq_ablation.py を全fold対応にして実行し、"
      "PQ基準で後処理方式を全fold OOFから選ぶ。")
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"v35r-{index:02d}"

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
