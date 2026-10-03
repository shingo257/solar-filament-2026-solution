"""EMA(重み平均)パイロット notebook builder — WINNING_STRATEGY_2026-08-29.md §6.12優先度1。

## 目的

2026-09-15の競合公開コード調査(`gopi470/VisionX`)で見つかった、自チームが
一度も試していない改善レバーのうち最優先(低コスト・確認済みで未試行)の
EMA(Exponential Moving Average、重み平均)を検証する。

**実装前の確認事項(重要):** boundary loss(境界重視損失)は既にV30から
`refiner_loss`に組み込まれており未試行ではなかった(README.md §2.1.12参照)。
このnotebookは`refiner_loss`を一切変更せず、EMAの有無だけを変数にする。

## 設計

アーキテクチャはmatched-parameter ablationで最も長い実績を持つ`base`
(MediSegUNet、標準refiner)に固定し、fold0で2構成 × seed(3407, 8407) =
4本を学習・OOF評価する:

  1. `baseline` — 現行のV51/matched-parameter ablationと完全同一(EMAなし)
  2. `ema`      — 上記に加え、optimizer.step()ごとにEMA shadow重みを更新
                  (decay=0.999)。エポックごとの検証はEMA shadow重みで行い、
                  最良エポックのEMA shadow重みをcheckpointとして保存する

損失関数・学習率スケジュール・augmentation・batch size選択ロジックは
`build_matched_param_ablation.py`と完全同一(変数はEMAの有無のみ)。

fold別基準(標準refiner、matched-parameter ablation実測、2seed平均):
fold0 0.4414 / fold1 0.4405 / fold2 0.4434(README.md §2.1.10)。

## 使い方

    python scratch/build_v54_colab_ema_pilot.py --fold 0
    python scratch/build_v54_colab_ema_pilot.py --fold 1
    python scratch/build_v54_colab_ema_pilot.py --fold 2

出力: solar_filament_v54_ema_pilot_fold{N}_colab.ipynb (Colab A100/L4推奨)
4本の学習(各14epoch)を直列に回すため、1〜1.5時間程度の見込み。

複数foldを1セッションで直列に回す統合版(fold0の結果を受けたfold1/2の確認用):

    python scratch/build_v54_colab_ema_pilot.py --folds 1 2

出力: solar_filament_v54_ema_pilot_fold1_2_colab.ipynb。単独fold版と同じセルを
foldごとに並べ、(1) Driveのpartial JSONから完了済みrunを飛ばす途中再開、
(2) ピークVRAM・所要時間のmanifest記録、(3) Drive上の全foldのmanifestを
読む最終判定セル、を追加する。学習・評価のレシピは単独fold版と完全同一。
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")

# matched-parameter ablation実測、base/2seed平均(README.md §2.1.10)
BASELINE_PQ = {0: 0.4414, 1: 0.4405, 2: 0.4434}
EMA_DECAY = 0.999
SEEDS = (3407, 8407)
CONFIGS = ("baseline", "ema")


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(fold):
    baseline_pq = BASELINE_PQ[fold]

    cells = [
        markdown(
            """# EMA(重み平均)パイロット — fold{fold}、baseアーキテクチャ固定

WINNING_STRATEGY_2026-08-29.md §6.12優先度1。`refiner_loss`(BCE+Dice+
boundary、V30から不変)はそのまま、**optimizer.step()ごとのEMA shadow重み
追跡の有無だけ**を変数にする。アーキテクチャは`MediSegUNet`(標準refiner)に
固定してfold{fold}で2構成 × seed 2本 = 4本を学習・OOF評価する。

| 構成 | 内容 |
|---|---|
| `baseline` | 現行のmatched-parameter ablationと完全同一(EMAなし) |
| `ema` | 上記に加えEMA shadow重み追跡(decay=0.999)、検証・checkpointもEMA重みで行う |

fold{fold}基準(標準refiner、matched-parameter ablation実測、2seed平均): {baseline_pq}。
GPU必須。A100/L4推奨(4本直列で合計1〜1.5時間見込み)。
""".format(fold=fold, baseline_pq=baseline_pq)
        ),
        code(
            """# [1/7] Drive接続、設定、前提の確認
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
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_ema_pilot")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    # --- build_v51_edgeattnet_refiner.py / build_matched_param_ablation.py と同一。触らないこと ---
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_EPOCHS = 14
    REFINER_LR = 2e-4
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    FOLD = %d
    CONF = 0.26
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ = %r
    EMA_DECAY = %r
    SEEDS = %r
    CONFIGS = %r  # ("baseline", "ema")

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像": CFG.PREPROC_CACHE,
    f"fold{CFG.FOLD} YOLO(OOF推論用)": CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{CFG.FOLD}" / "weights" / "best.pt",
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0)})
"""
            % (fold, baseline_pq, EMA_DECAY, SEEDS, CONFIGS)
        ),
        code(
            """# [2/7] COCOとfoldを読み、前処理画像をローカルSSDへ展開
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

train_files_fold = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]
assert not (set(train_files_fold) & set(valid_files_fold)), "fold分割が壊れている"
print({"観測": len(train_files), "除外train": len(train_files_fold),
       "検証(OOF)": len(valid_files_fold), "annotation": len(coco["annotations"])})
"""
        ),
        code(
            "# [3/7] MediSegUNet (solar_filament/models/mediseg_models.py と同一。base構成のみ使用)\n"
            + MODELS
        ),
        code(
            """# [3.5/7] 形状サニティチェック
_dummy = torch.randn(2, 3, CFG.CROP_SIZE, CFG.CROP_SIZE).cuda()
_probe = MediSegUNet(3, 1, False).cuda()
with torch.no_grad():
    _out = _probe(_dummy)
assert _out.shape == (2, 1, CFG.CROP_SIZE, CFG.CROP_SIZE), "出力形状が想定と異なる"
print({"n_params": sum(p.numel() for p in _probe.parameters())})
del _probe, _dummy, _out
torch.cuda.empty_cache()
print("形状チェックOK")
"""
        ),
        code(
            """# [4/7] Dataset・loss・EMA・学習/評価ループ
# refiner_loss は build_v51_edgeattnet_refiner.py / build_matched_param_ablation.py と完全同一
# (V30から不変。boundary loss は既にここに含まれているため変更しない — README.md §2.1.12)
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

def crop_dice_from_state(state_dict, loader, threshold=0.5):
    model = MediSegUNet(3, 1, False).cuda()
    model.load_state_dict(state_dict, strict=True)
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
    del model
    torch.cuda.empty_cache()
    return total / max(count, 1)

class EMA:
    \"\"\"標準的な指数移動平均。BatchNormの非浮動小数点バッファ(num_batches_tracked)は
    平均化せず最新値をそのまま追跡する(浮動小数点でないものをEMA更新すると壊れるため)。\"\"\"
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            shadow_v = self.shadow[k]
            if shadow_v.dtype.is_floating_point:
                shadow_v.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                shadow_v.copy_(v)

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

def train_one_run(config, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    use_ema = config == "ema"
    train_ds = InstanceCropDataset(train_files_fold, augment=True)
    valid_ds = InstanceCropDataset(valid_files_fold, augment=False)
    batch_size = choose_batch_size(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model = MediSegUNet(3, 1, False).cuda()
    ema = EMA(model, CFG.EMA_DECAY) if use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.REFINER_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    best_dice, best_state = -1.0, None
    started = time.time()

    for epoch in range(CFG.REFINER_EPOCHS):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"{config}/seed{seed} epoch {epoch + 1}/{CFG.REFINER_EPOCHS}")
        for image, target in progress:
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(model(image), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if use_ema:
                ema.update(model)
            running += loss.item()
            progress.set_postfix(loss=f"{running / (progress.n + 1):.4f}")
        scheduler.step()

        eval_state = ema.shadow if use_ema else model.state_dict()
        dice = crop_dice_from_state(eval_state, valid_loader)
        if dice > best_dice:
            best_dice = dice
            best_state = {k: v.detach().cpu().clone() for k, v in eval_state.items()}
        print(f"{config}/seed{seed} epoch {epoch + 1}: crop Dice {dice:.6f} (best {best_dice:.6f})")

    checkpoint_path = CFG.ARTIFACT_ROOT / f"v54_ema_pilot_{config}_seed{seed}_fold{CFG.FOLD}.pt"
    torch.save({"model": model.state_dict(), "best_model": best_state, "best_dice": best_dice,
                "config": config, "ema_decay": CFG.EMA_DECAY if use_ema else None},
               checkpoint_path)
    minutes = round((time.time() - started) / 60, 1)
    print(f"{config}/seed{seed}: checkpoint={checkpoint_path} minutes={minutes}")
    del model, optimizer, scheduler, train_loader, valid_loader, train_ds, valid_ds, ema
    torch.cuda.empty_cache()
    return {"checkpoint": str(checkpoint_path), "best_crop_dice": best_dice, "minutes": minutes,
            "batch_size": batch_size}
"""
        ),
        code(
            """# [5/7] OOF採点用の後処理(V51/matched-parameter ablationと同一: shrink@0.45 + drop_contained@0.90)
!pip -q install ultralytics
from ultralytics import YOLO
from pycocotools import mask as mask_utils

yolo_model = YOLO(str(CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{CFG.FOLD}" / "weights" / "best.pt"))

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
def refine_proposal(refiner, image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
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
    def sq_rq(self):
        tp = len(self.tp_ious)
        sq = float(sum(self.tp_ious) / tp) if tp > 0 else 0.0
        rq = float(tp / (tp + 0.5 * self.fp_count + 0.5 * self.fn_count)) if (tp + self.fp_count + self.fn_count) > 0 else 0.0
        return sq, rq
    def summary(self):
        sq, rq = self.sq_rq()
        return {"pq": self.score, "sq": sq, "rq": rq, "tp": len(self.tp_ious),
                "fp": self.fp_count, "fn": self.fn_count, "denominator": self.denominator}

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in annotations[image_id] if not ann.get("iscrowd", 0)]

@torch.no_grad()
def evaluate_oof(refiner):
    refiner.eval()
    meter = PanopticQuality()
    for name in tqdm(valid_files_fold, desc=f"fold{CFG.FOLD} OOF採点"):
        image_path = CFG.PREPROC_DIR / name
        result = yolo_model.predict(str(image_path), imgsz=1280, conf=0.08, iou=0.55,
                                     agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, confidences = result_masks(result)
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences)) if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        refined = [refine_proposal(refiner, image, m) for m in proposals]
        predictions = apply_ownership(refined, scores)
        pred_rles = []
        if predictions:
            encoded = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
            pred_rles = encoded if isinstance(encoded, list) else [encoded]
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, pred_rles)
            meter.add_matrix(matrix, len(truth), len(pred_rles))
    return meter.summary()

print("後処理・採点関数の定義OK")
"""
        ),
        code(
            """# [6/7] 2構成 x seed 2本 = 4本を直列に学習・OOF評価(合計で1〜1.5時間かかる見込み)
results = []
for config in CFG.CONFIGS:
    for seed in CFG.SEEDS:
        print(f"\\n{'=' * 60}\\n{config} / seed {seed}\\n{'=' * 60}")
        train_summary = train_one_run(config, seed)

        refiner = MediSegUNet(3, 1, False).cuda().eval()
        ckpt = torch.load(train_summary["checkpoint"], map_location="cuda", weights_only=False)
        refiner.load_state_dict(ckpt["best_model"] or ckpt["model"], strict=True)
        oof = evaluate_oof(refiner)
        delta_vs_base = oof["pq"] - CFG.BASELINE_PQ
        print(json.dumps({"config": config, "seed": seed, **oof,
                          "delta_vs_baseline": round(delta_vs_base, 6)}, indent=2))

        results.append({
            "config": config, "seed": seed, "ema_decay": CFG.EMA_DECAY if config == "ema" else None,
            **train_summary, **oof, "delta_vs_baseline": delta_vs_base,
        })
        del refiner, ckpt
        torch.cuda.empty_cache()

        # 途中経過を都度Driveへ保存(セッション切断への備え)
        (CFG.ARTIFACT_ROOT / f"v54_ema_pilot_fold{CFG.FOLD}_partial.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print("\\n全4本完了")
"""
        ),
        code(
            """# [7/7] 集計・manifest保存
import statistics

by_config = {}
for row in results:
    by_config.setdefault(row["config"], []).append(row["pq"])

summary_by_config = {
    config: {
        "seeds": CFG.SEEDS,
        "pq_per_seed": pqs,
        "pq_mean": statistics.mean(pqs),
        "pq_std": statistics.pstdev(pqs) if len(pqs) > 1 else 0.0,
    }
    for config, pqs in by_config.items()
}

manifest = {
    "experiment_id": f"v54_ema_pilot_fold{CFG.FOLD}",
    "purpose": "WINNING_STRATEGY_2026-08-29.md §6.12優先度1。EMA(重み平均)が"
               "refiner学習を改善するか、baseアーキテクチャ・fold%dで検証",
    "fold": CFG.FOLD,
    "ema_decay": CFG.EMA_DECAY,
    "recipe": "build_matched_param_ablation.pyのbase構成と完全同一"
              "(crop320/context1.65/14epoch/LR2e-4/refiner_loss不変)。"
              "EMAの有無のみ変更",
    "reference_baseline_pq": CFG.BASELINE_PQ,
    "runs": results,
    "summary_by_config": summary_by_config,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}
report_path = CFG.ARTIFACT_ROOT / f"v54_ema_pilot_fold{CFG.FOLD}_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary_by_config, ensure_ascii=False, indent=2))
print(f"\\n保存: {report_path}")
print("\\n次: fold%dでemaがbaselineを上回れば、他foldでも同じ検証を行い、"
      "「単一foldのbestを一般化しない」という規律に従って独立2fold以上での符号一致を確認する。")
""" % (fold, fold)
        ),
    ]
    return cells


RESUME_OLD = """results = []
for config in CFG.CONFIGS:
    for seed in CFG.SEEDS:
        print("""

RESUME_NEW = """partial_path = CFG.ARTIFACT_ROOT / f"v54_ema_pilot_fold{CFG.FOLD}_partial.json"
results = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else []
done = {(row["config"], row["seed"]) for row in results}
if done:
    print(f"途中再開: 完了済み {sorted(done)} を飛ばす")
torch.cuda.reset_peak_memory_stats()
fold_started = time.time()
for config in CFG.CONFIGS:
    for seed in CFG.SEEDS:
        if (config, seed) in done:
            continue
        print("""

MANIFEST_OLD = """    "gpu": torch.cuda.get_device_name(0),
}"""

MANIFEST_NEW = """    "gpu": torch.cuda.get_device_name(0),
    # 統合版のみ: このセッションで実行した分のピークVRAMと所要時間(途中再開時は再開後の分のみ)
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "session_minutes": round((time.time() - fold_started) / 60, 1),
}"""


def build_combined_cells(folds):
    """単独fold版のセルをfoldごとに並べた統合版。レシピは単独fold版と完全同一。"""
    fold_list = ", ".join(f"fold{f}" for f in folds)
    cells = [
        markdown(
            f"""# EMA(重み平均)パイロット — {fold_list} 統合版、baseアーキテクチャ固定

WINNING_STRATEGY_2026-08-29.md §6.15 の続き。fold0パイロットで ema が Δ+0.0005
(2/2seedで正)だったため、**独立foldでの符号一致**を確認する。各foldの中身は
単独fold版(`solar_filament_v54_ema_pilot_fold{{N}}_colab.ipynb`)と完全同一で、
それを{len(folds)}fold分、直列に並べただけ。

- 所要時間の見込み: 1foldあたり約1.5〜1.7時間(fold0実測: 1runあたり約25分×4本+OOF採点)、
  合計 **{len(folds) * 1.5:.0f}〜{len(folds) * 1.7:.1f}時間**。Colab Pro+のバックグラウンド実行推奨。
- **途中で切断されたら**: そのまま「すべてのセルを実行」し直せばよい。完了済みのrun
  (Driveの `v54_ema_pilot_fold{{N}}_partial.json` に記録済み)は自動で飛ばす。
  完了済みfoldもmanifestが残るが、学習セルは全runを飛ばしてOOF再採点もしない。
- **実行後に手で記録すること**: 消費コンピューティングユニット(Colabのリソース画面)。
  GPU名・ピークVRAM・所要時間は各foldのmanifestに自動で残る。
- 判定: 最後のセルが、Drive上にある全foldのmanifest(fold0含む)から採否の目安を出す。
"""
        )
    ]
    for fold in folds:
        fold_cells = build_cells(fold)[1:]
        for cell in fold_cells:
            if cell["cell_type"] != "code":
                continue
            if RESUME_OLD in cell["source"]:
                cell["source"] = cell["source"].replace(RESUME_OLD, RESUME_NEW)
            if MANIFEST_OLD in cell["source"]:
                cell["source"] = cell["source"].replace(MANIFEST_OLD, MANIFEST_NEW)
        assert any("partial_path" in c["source"] for c in fold_cells), "途中再開の差し込みに失敗"
        assert any("peak_vram_gb" in c["source"] for c in fold_cells), "VRAM記録の差し込みに失敗"
        cells.append(markdown(f"---\n\n## fold{fold}\n\n以下は単独fold版 fold{fold} と同一のセル。"))
        cells.extend(fold_cells)

    cells.append(
        code(
            """# [最終] Drive上の全foldのmanifestから採否の目安を出す
import json, statistics
from pathlib import Path

ARTIFACT_ROOT = Path("/content/drive/MyDrive/solar_filament_v30/artifacts")
MEANINGFUL = 0.005  # プロジェクト一貫の「意味ある改善」の目安
rows = []
for fold in (0, 1, 2):
    path = ARTIFACT_ROOT / f"v54_ema_pilot_fold{fold}_manifest.json"
    if not path.exists():
        print(f"fold{fold}: manifestなし(未実行)")
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))["summary_by_config"]
    base, ema = summary["baseline"], summary["ema"]
    paired = [e - b for e, b in zip(ema["pq_per_seed"], base["pq_per_seed"])]
    rows.append({"fold": fold, "baseline": round(base["pq_mean"], 6), "ema": round(ema["pq_mean"], 6),
                 "delta": round(ema["pq_mean"] - base["pq_mean"], 6),
                 "paired_delta_per_seed": [round(d, 6) for d in paired],
                 "seeds_positive": f"{sum(d > 0 for d in paired)}/{len(paired)}"})
for row in rows:
    print(json.dumps(row, ensure_ascii=False))

if rows:
    deltas = [row["delta"] for row in rows]
    positive_folds = sum(d > 0 for d in deltas)
    print({"folds": len(rows), "positive_folds": positive_folds,
           "mean_delta": round(statistics.mean(deltas), 6)})
    if positive_folds == len(rows) and statistics.mean(deltas) >= MEANINGFUL:
        print("判定の目安: 全foldで正、かつ平均Δが+0.005以上 → 採用を検討")
    elif positive_folds == len(rows):
        print("判定の目安: 全foldで符号は一致するが、平均Δが+0.005未満 → 効果は実務閾値未満(採用は費用対効果で判断)")
    else:
        print("判定の目安: foldによって符号が割れる → 不採用")
"""
        )
    )
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", choices=["0", "1", "2"], help="評価対象fold")
    group.add_argument("--folds", nargs="+", choices=["0", "1", "2"],
                       help="1セッションで直列に回すfold(統合版notebookを生成)")
    args = parser.parse_args()

    if args.folds:
        folds = sorted({int(f) for f in args.folds})
        cells = build_combined_cells(folds)
        tag = "_".join(str(f) for f in folds)
        cell_prefix = f"v54-ema-f{tag}"
    else:
        fold = int(args.fold)
        cells = build_cells(fold)
        tag = str(fold)
        cell_prefix = f"v54-ema-f{fold}"
    for index, cell in enumerate(cells):
        cell["id"] = f"{cell_prefix}-{index:02d}"

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

    output = ROOT / f"solar_filament_v54_ema_pilot_fold{tag}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
