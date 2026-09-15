"""V51 — EdgeAttNet風refiner(EG-MHSA)の学習ノートブックビルダー(全fold対応)。

## 目的

WINNING_STRATEGY_2026-08-29.md §1.4 / Phase 2。現行refiner(`MediSegUNet`、
プレーンU-Net、31M)を、bottleneckにEdge-Guided Multi-Head Self-Attention
(EG-MHSA)を挿入した`MediSegEdgeAttUNet`に置き換える効果を検証する。

参考文献: "EdgeAttNet: Towards Barb-Aware Filament Segmentation"
(arXiv:2509.02964)。MAGFiLO上でmIoU_pairwise 0.5724(U-Net)→0.6451
(EdgeAttNet)を報告。ただし著者の公開リポジトリには実際のアーキテクチャ
コードが存在しないことを確認済み(`solar_filament/models/mediseg_models.py`
のクラス定義コメント参照)。本ノートブックは論文の記述のみを根拠にした
独自の再実装を使う。

## 使い方

    python scratch/build_v51_edgeattnet_refiner.py --fold 0   # fold0を学習(他2foldはvalid)
    python scratch/build_v51_edgeattnet_refiner.py --fold 1
    python scratch/build_v51_edgeattnet_refiner.py --fold 2
    python scratch/build_v51_edgeattnet_refiner.py --fold all # 提出用: 全707画像でfine-tune

`--fold 0/1/2` はそれぞれのfoldを検証専用に除外して学習し(リークなし
OOF評価が可能)、`--fold all` はfold0のcheckpointをウォームスタートして
全707画像へ低LR・短epochで再学習する(V30本体のall-data refinerと
同じレシピ、提出モデル用)。

fold0/1/2の3本すべてが2026-08-29〜30の検証で「3-fold全結合(n=707)で
Δ+0.0028、95%CI[0.000915,0.004756]、ゼロを完全に除外」という結果を
再現しており(README §2.1.5)、このスクリプトはその結果を再現するための
正本である。

## 設計方針: 学習条件は`build_v35_refiner_folds12.py`と同一

crop size 320、context 1.65、loss = 0.35·BCE + 0.65·softDice + 0.15·edge、
14 epoch(all-dataは5 epoch・低LR)、LR 2e-4、CosineAnnealing、AMP、
crop Diceでbest選択——**モデルアーキテクチャ以外は一切変えない**。
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")
# 注意: src/mediseg_models.py はモノレポリファクタ後方互換用のシム
# (`from solar_filament.models.mediseg_models import *`)であり、Colabへ
# 埋め込んでも実体を持たない。実体は solar_filament/models/ 側にある。

BASELINE_PQ = {0: 0.4393, 1: 0.4415, 2: 0.4471}  # 標準refinerの全fold OOF基準(README §2「現在地」)


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def build_cells(fold_arg):
    is_all = fold_arg == "all"
    fold_num = None if is_all else int(fold_arg)
    title_suffix = "all-data fine-tune(提出用)" if is_all else f"fold{fold_num}"

    cells = [
        markdown(
            f"""# V51 — EdgeAttNet風refiner(EG-MHSA)の学習: {title_suffix}

現行refiner(`MediSegUNet`)のbottleneckにEdge-Guided Multi-Head
Self-Attentionを挿入した`MediSegEdgeAttUNet`を学習する。
{'fold0のcheckpointをウォームスタートし、全707画像(train+valid区別なし)へ低LR(0.35倍)・5epochで再学習する(提出モデル用、V30本体のall-data refinerと同じレシピ)。' if is_all else f'fold{fold_num}を検証専用に除外し(リークなしOOF評価)、確定後処理(shrink@0.45 + drop_contained_then_area_asc)込みのOOF PQを基準{BASELINE_PQ[fold_num]}と比較する。'}

GPU必須。A100推奨(Attentionの分だけメモリ消費が増えるため)。
"""
        ),
        code(
            f"""# [1/{'6' if is_all else '7'}] Drive接続、設定、前提の確認
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
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v51")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    # --- 以下はbuild_v35_refiner_folds12.pyと完全に同一。触らないこと ---
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_EPOCHS = {5 if is_all else 14}
    REFINER_LR = 2e-4{' * 0.35  # all-data fine-tuneは低LR' if is_all else ''}
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    {'FOLD = None  # all-dataモードでは未使用' if is_all else f'FOLD = {fold_num}'}
    # 確定後処理(README.md「現在地」節)
    CONF = 0.26
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    {'' if is_all else f'BASELINE_PQ = {BASELINE_PQ[fold_num]}'}

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)

required = {{
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像": CFG.PREPROC_CACHE,
{'    "fold0 EdgeAttNet refiner(ウォームスタート元)": CFG.ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold0_resume.pt",' if is_all else f'    "fold{fold_num} 標準refiner(比較基準、任意)": CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{fold_num}_resume.pt",'}
{'' if is_all else f'    "fold{fold_num} YOLO(OOF推論用)": CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{fold_num}" / "weights" / "best.pt",'}
}}
missing = {{k: str(v) for k, v in required.items() if not v.exists()}}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {{missing}}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({{k: "OK" for k in required}})
print({{"GPU": torch.cuda.get_device_name(0)}})
"""
        ),
        code(
            """# [2/{n}] COCOとfoldを読み、前処理画像をローカルSSDへ展開
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"学習annotation JSONが見つかりません: {base}")
    return matches[0].parents[2]

TRAIN_JSON = next(find_dataset_root(CFG.DATA_CACHE).rglob(
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

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
""".replace("{n}", "6" if is_all else "7")
        ),
        code("# [3/{n}] MediSegUNet / EdgeGuidedMHSA / MediSegEdgeAttUNet (solar_filament/models/mediseg_models.py と同一)\n".replace("{n}", "6" if is_all else "7") + MODELS),
        code(
            """# [3.5/{n}] 形状サニティチェック(学習開始前に必ず確認する)
_probe = MediSegEdgeAttUNet(3, 1, False).cuda()
_dummy = torch.randn(2, 3, CFG.CROP_SIZE, CFG.CROP_SIZE).cuda()
with torch.no_grad():
    _out = _probe(_dummy)
print("output shape:", _out.shape, "(期待値: [2, 1, 320, 320])")
assert _out.shape == (2, 1, CFG.CROP_SIZE, CFG.CROP_SIZE), "出力形状が想定と異なる"

n_params_edgeatt = sum(p.numel() for p in _probe.parameters())
n_params_base = sum(p.numel() for p in MediSegUNet(3, 1, False).parameters())
print({"MediSegEdgeAttUNet_params": n_params_edgeatt,
       "MediSegUNet_params(参考)": n_params_base,
       "差分": n_params_edgeatt - n_params_base})
del _probe, _dummy, _out
torch.cuda.empty_cache()
print("形状チェックOK")
""".replace("{n}", "6" if is_all else "7")
        ),
        code(
            """# [4/{n}] Dataset・loss・学習ループ (build_v35_refiner_folds12.pyと同一。モデルクラスのみ差し替え)
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
            probe = MediSegEdgeAttUNet(3, 1, False).cuda()
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(probe(image.cuda()), target.cuda())
            loss.backward()
            del probe, image, target, loss, loader
            torch.cuda.empty_cache()
            return candidate
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("どのbatch sizeもGPUメモリに収まらない")
""".replace("{n}", "6" if is_all else "7")
        ),
    ]

    if is_all:
        cells.append(code(
            """# [5/6] 全707画像でfine-tune(fold0からウォームスタート、低LR・短epoch)
all_ds = InstanceCropDataset(train_files, augment=True)
print({"all_data_crops": len(all_ds)})

batch_size = choose_batch_size(all_ds)
print("batch_size =", batch_size)
loader = DataLoader(all_ds, batch_size=batch_size, shuffle=True,
                     num_workers=4, pin_memory=True, persistent_workers=True)

WARM_START_CKPT = CFG.ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold0_resume.pt"
ALLDATA_CKPT = CFG.ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_all_data_resume.pt"

model = MediSegEdgeAttUNet(3, 1, False).cuda()
warm_state = torch.load(WARM_START_CKPT, map_location="cpu")
model.load_state_dict(warm_state.get("best_model") or warm_state["model"], strict=True)
print("ウォームスタート元:", WARM_START_CKPT)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.REFINER_EPOCHS)
scaler = torch.amp.GradScaler("cuda")

started = time.time()
for epoch in range(CFG.REFINER_EPOCHS):
    model.train()
    running = 0.0
    progress = tqdm(loader, desc=f"all-data EdgeAttNet refiner {epoch + 1}/{CFG.REFINER_EPOCHS}")
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
    print(f"epoch {epoch + 1}: mean loss {running / len(loader):.4f}")

torch.save({"model": model.state_dict(), "best_model": model.state_dict(),
            "warm_start_from": str(WARM_START_CKPT), "epochs": CFG.REFINER_EPOCHS,
            "lr": CFG.REFINER_LR}, ALLDATA_CKPT)
print({"checkpoint": str(ALLDATA_CKPT), "minutes": round((time.time() - started) / 60, 1)})
"""
        ))
        cells.append(code(
            """# [6/6] manifest保存
import hashlib

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "experiment_id": "v51_edgeattnet_refiner_all_data",
    "purpose": "V52提出用: fold0 checkpointをウォームスタートし全707画像へ再学習",
    "reference": "arXiv:2509.02964 EdgeAttNet (著者コードは未検証、論文記述からの独自再実装)",
    "warm_start_from": str(WARM_START_CKPT),
    "epochs": CFG.REFINER_EPOCHS,
    "lr": CFG.REFINER_LR,
    "n_params": n_params_edgeatt,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "checkpoint_sha256": sha256_of(ALLDATA_CKPT),
}
report_path = CFG.ARTIFACT_ROOT / "v51_edgeattnet_refiner_all_data_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"\\n保存: {report_path}")
print("次: build_v52_edgeattnet_submission.py でこのcheckpointを使い提出物を生成する。")
"""
        ))
    else:
        cells.append(code(
            f"""# [5/7] fold{fold_num}を学習(該当foldを検証専用に除外)
started = time.time()
checkpoint_path = CFG.ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{fold_num}_resume.pt"
train_files_fold = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]
train_ds = InstanceCropDataset(train_files_fold, augment=True)
valid_ds = InstanceCropDataset(valid_files_fold, augment=False)
assert not (set(train_files_fold) & set(valid_files_fold)), "fold分割が壊れている"
print({{"fold": CFG.FOLD, "train_crops": len(train_ds), "valid_crops": len(valid_ds)}})

batch_size = choose_batch_size(train_ds)
print("batch_size =", batch_size)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

model = MediSegEdgeAttUNet(3, 1, False).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.REFINER_EPOCHS)
scaler = torch.amp.GradScaler("cuda")
best_dice, best_state = -1.0, None

for epoch in range(CFG.REFINER_EPOCHS):
    model.train()
    running = 0.0
    progress = tqdm(train_loader, desc=f"fold{{CFG.FOLD}} EdgeAttNet refiner {{epoch + 1}}/{{CFG.REFINER_EPOCHS}}")
    for image, target in progress:
        image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = refiner_loss(model(image), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
        progress.set_postfix(loss=f"{{running / (progress.n + 1):.4f}}")
    scheduler.step()
    dice = crop_dice(model, valid_loader)
    if dice > best_dice:
        best_dice = dice
        best_state = {{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    torch.save({{"epoch": epoch, "model": model.state_dict(), "best_model": best_state,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_dice": best_dice}}, checkpoint_path)
    print(f"fold{{CFG.FOLD}} epoch {{epoch + 1}}: crop Dice {{dice:.6f}} (best {{best_dice:.6f}})")

best_crop_dice = best_dice
print({{"checkpoint": str(checkpoint_path), "best_crop_dice": best_crop_dice,
       "minutes": round((time.time() - started) / 60, 1)}})
"""
        ))
        cells.append(code(
            f"""# [6/7] fold{fold_num} OOF(リークなし)を確定後処理で採点し、基準{BASELINE_PQ[fold_num]}と比較
!pip -q install ultralytics
from ultralytics import YOLO

yolo_model = YOLO(str(CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{fold_num}" / "weights" / "best.pt"))

edge_refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
edge_refiner.load_state_dict(ckpt.get("best_model") or ckpt["model"], strict=True)

def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    return output, confidences

oof_files = [name for name in train_files if fold_of_file[name] == CFG.FOLD]

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
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = edge_refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal  # shrink(確定構成)

def greedy_disjoint(masks, order, min_area=CFG.MIN_AREA):
    occupied = np.zeros((2048, 2048), dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept

from pycocotools import mask as mask_utils

def drop_contained_indices(masks, threshold=CFG.DROP_THRESHOLD):
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

def apply_ownership(masks, scores):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices(masks)
    return greedy_disjoint(masks, sorted(subset, key=lambda i: areas[i]))

def to_rle(annotation, height, width):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

class PanopticQuality:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.tp_ious = []
        self.fp_count = 0
        self.fn_count = 0
    def add_matrix(self, matrix, n_gt, n_pred):
        if n_gt == 0:
            self.fp_count += n_pred; return
        if n_pred == 0:
            self.fn_count += n_gt; return
        hits = matrix > self.iou_threshold
        scores = matrix[hits].tolist()
        self.tp_ious.extend(scores)
        self.fp_count += int((hits.sum(axis=0) == 0).sum())
        self.fn_count += int((hits.sum(axis=1) == 0).sum())
    @property
    def denominator(self):
        return len(self.tp_ious) + 0.5 * self.fp_count + 0.5 * self.fn_count
    @property
    def score(self):
        d = self.denominator
        return float(sum(self.tp_ious) / d) if d > 0 else 0.0
    def summary(self):
        d = self.denominator
        return {{"pq": self.score, "tp": len(self.tp_ious), "fp": self.fp_count, "fn": self.fn_count, "denominator": d}}

gt_rles = {{}}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in annotations[image_id] if not ann.get("iscrowd", 0)]

meter = PanopticQuality()
for name in tqdm(oof_files, desc="OOF推論+採点(EdgeAttNet refiner)"):
    image_path = CFG.PREPROC_DIR / name
    result = yolo_model.predict(str(image_path), imgsz=1280, conf=0.08, iou=0.55,
                                 agnostic_nms=True, retina_masks=True, verbose=False)[0]
    masks, confidences = result_masks(result)
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences)) if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
    proposals = [masks[i] for i in keep]
    scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    refined = [refine_proposal(image, m) for m in proposals]
    predictions = apply_ownership(refined, scores)
    pred_rles = []
    if predictions:
        encoded = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
        pred_rles = encoded if isinstance(encoded, list) else [encoded]
    for image_id in file_to_ids[name]:
        truth = gt_rles[image_id]
        matrix = iou_matrix(truth, pred_rles)
        meter.add_matrix(matrix, len(truth), len(pred_rles))

summary = meter.summary()
delta = summary["pq"] - CFG.BASELINE_PQ
print(json.dumps({{
    "fold": CFG.FOLD, "pq": round(summary["pq"], 6), "baseline_pq": CFG.BASELINE_PQ,
    "delta": round(delta, 6), "beats_baseline": bool(delta > 0),
    "best_crop_dice": best_crop_dice,
}}, ensure_ascii=False, indent=2))
"""
        ))
        cells.append(code(
            f"""# [7/7] manifest保存
import hashlib

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {{
    "experiment_id": "v51_edgeattnet_refiner_fold{fold_num}",
    "purpose": "WINNING_STRATEGY_2026-08-29.md Phase 2. refinerをMediSegUNet"
               "からMediSegEdgeAttUNet(EG-MHSA)へ置き換える効果をfold{fold_num}で検証",
    "reference": "arXiv:2509.02964 EdgeAttNet (著者コードは未検証、論文記述からの独自再実装)",
    "recipe": "build_v35_refiner_folds12.pyと完全同一(crop320/context1.65/"
              "14epoch/LR2e-4/crop Diceでbest選択)。モデルアーキテクチャのみ変更",
    "fold": CFG.FOLD,
    "n_params": n_params_edgeatt,
    "n_params_baseline": n_params_base,
    "best_crop_dice": best_crop_dice,
    "oof_pq": round(summary["pq"], 6),
    "baseline_pq": CFG.BASELINE_PQ,
    "delta": round(delta, 6),
    "beats_baseline": bool(delta > 0),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "checkpoint_sha256": sha256_of(checkpoint_path),
}}
report_path = CFG.ARTIFACT_ROOT / "v51_edgeattnet_refiner_fold{fold_num}_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"\\n保存: {{report_path}}")
print("\\n次: 他のfoldも学習後、ペア化観測クラスタ・ブートストラップで3-fold全結合の有意性を確認する"
      "(README §2.1.5 / v51_edgeattnet_refiner_3fold_manifest.json 参照)。")
"""
        ))

    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["0", "1", "2", "all"],
                         help="学習対象。0/1/2はその他foldで学習しそのfoldをOOF検証、"
                              "allはfold0からウォームスタートして全707画像で提出用モデルを再学習")
    args = parser.parse_args()

    cells = build_cells(args.fold)
    for index, cell in enumerate(cells):
        cell["id"] = f"v51-{args.fold}-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output = ROOT / f"solar_filament_v51_edgeattnet_refiner_{args.fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
