"""Matched-parameter ablation notebook builder — report/main.tex Future Work item 1.

## 目的

report/main.tex(3-5回の外部査読で繰り返し指摘された未解決の交絡)を検証する:
「refiner(MediSegEdgeAttUNet)の改善は edge prior そのものの効果か、それとも
単にbottleneckのattention容量が増えただけの効果か」を切り分ける。

3構成を指定foldで比較する(いずれもそのfoldを検証専用に除外、他は全て同一レシピ):

  1. MediSegUNet          — プレーンU-Net baseline (31.04M params)
  2. MediSegPlainMHSAUNet — bottleneckに標準MHSA追加、edge priorなし (35.24M params)
  3. MediSegEdgeAttUNet   — bottleneckにEG-MHSA追加、edge prior あり (35.24M params)

2と3はパラメータ数がほぼ完全に一致する(差は edge_conv+edge_proj の約2,076個、
全体の0.006%)。2の存在により「1→2の差」(attention容量を足しただけの効果)と
「2→3の差」(edge priorを足した追加効果)を分離できる。これがEdgeAttNet論文
自身のablation("U-Net + MHSA, no positional encoding")と同じ切り分け。

各構成をseed 3407・8407の2本ずつ、計6本学習する(V42のmulti-seed実験と同じ
seed規約)。既存のfold別YOLO検出器・確定後処理(shrink@0.45 + drop_contained@0.90)
はそのまま流用し、refinerアーキテクチャとseedだけを変える。

## 背景(2026-08-31、fold0実測)

fold0(n=236)でこのablationを実行した結果、edge-attention refinerは
baselineを上回らず、3構成中最低だった(mean PQ: base 0.441412 >
plain_mhsa 0.441259 > edge_att 0.440474)。report/main.texはこの反転を
反映して全面改訂済み。旧報告の3-fold delta(fold0 +0.0062、fold1 +0.0009、
fold2 +0.0012)のうちfold1/fold2は今回実測したseed間ノイズ(std 6-8e-4)
より小さく、fold0限定では断定できない。fold1・fold2でも同じablationを
回し、この反転がfold0固有の現象か、プロジェクト全体に共通する現象かを
確認する必要がある(report Future Work item 1)。

## 使い方

    python scratch/build_matched_param_ablation.py --fold 0
    python scratch/build_matched_param_ablation.py --fold 1
    python scratch/build_matched_param_ablation.py --fold 2

出力: solar_filament_matched_param_ablation_fold{N}_colab.ipynb (Colab A100/L4推奨)
6本の学習(各14epoch)を直列に回すため、fold毎に2〜3時間かかる見込み。
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")

# 標準refiner(MediSegUNet)の全fold OOF基準(README「現在地」、build_v51_edgeattnet_refiner.pyと同一)
BASELINE_PQ = {0: 0.4393, 1: 0.4415, 2: 0.4471}
# 旧V51報告のedge-attention refiner実測値(fold0は直接実測、fold1/2はbaseline+報告済みdeltaから逆算)
EDGEATT_PQ_ORIGINAL = {0: 0.4454, 1: round(0.4415 + 0.0009, 4), 2: round(0.4471 + 0.0012, 4)}
SEEDS = (3407, 8407)
VARIANTS = ("base", "plain_mhsa", "edge_att")


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(fold):
    baseline_pq = BASELINE_PQ[fold]
    edgeatt_pq = EDGEATT_PQ_ORIGINAL[fold]

    cells = [
        markdown(
            """# Matched-parameter ablation — edge prior vs. attention capacity (fold{fold})

`report/main.tex` の Future Work item 1。**fold0で実行した同じablationは
edge-attentionがbaselineを上回らないという反転結果になった**(report参照)。
このnotebookはfold{fold}で同じ検証を行い、その反転がfold0固有の現象か、
プロジェクト全体に共通する現象かを確認する。3構成 × seed 2本 = 6本を
fold{fold}で学習・OOF評価する:

| 構成 | 内容 | パラメータ数 |
|---|---|---|
| `base` | MediSegUNet(プレーンU-Net baseline) | 31.04M |
| `plain_mhsa` | MediSegPlainMHSAUNet(標準MHSA、edge priorなし) | 35.24M |
| `edge_att` | MediSegEdgeAttUNet(EG-MHSA、edge priorあり、V51と同一) | 35.24M |

`plain_mhsa`と`edge_att`はパラメータ数がほぼ完全一致(差は約2,076個
=全体の0.006%)。`base→plain_mhsa`の差はattention容量を足しただけの効果、
`plain_mhsa→edge_att`の差がedge prior固有の効果、という分離ができる。

学習レシピ(crop320・context1.65・14epoch・LR2e-4・crop Diceでbest選択)は
V51(`build_v51_edgeattnet_refiner.py`)と完全同一。**モデルクラスとseedだけ
が変数。** GPU必須。A100/L4推奨(6本直列で合計2〜3時間かかる見込み)。

fold{fold}基準(旧V51報告、単一seed): 標準refiner PQ={base_pq}、
edge-attention refiner PQ={edge_pq}(delta {delta:+.4f})。
""".format(fold=fold, base_pq=baseline_pq, edge_pq=edgeatt_pq, delta=edgeatt_pq - baseline_pq)
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
    LOCAL_ROOT = Path("/content/solar_ablation_fold{fold}")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    # --- 以下はbuild_v51_edgeattnet_refiner.pyと完全に同一。触らないこと ---
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_EPOCHS = 14
    REFINER_LR = 2e-4
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    FOLD = {fold}
    CONF = 0.26
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ = %r
    EDGEATT_PQ = %r
    SEEDS = %r
    VARIANTS = %r  # ("base", "plain_mhsa", "edge_att")

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)

required = {{
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像": CFG.PREPROC_CACHE,
    "fold{fold} YOLO(OOF推論用)": CFG.DRIVE_ROOT / "runs" / "yolo_fold_{fold}" / "weights" / "best.pt",
}}
missing = {{k: str(v) for k, v in required.items() if not v.exists()}}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {{missing}}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({{k: "OK" for k in required}})
print({{"GPU": torch.cuda.get_device_name(0)}})
""".format(fold=fold)
            % (baseline_pq, edgeatt_pq, SEEDS, VARIANTS)
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
            "# [3/7] MediSegUNet / EdgeGuidedMHSA / MediSegEdgeAttUNet / PlainMHSA / MediSegPlainMHSAUNet"
            " (solar_filament/models/mediseg_models.py と同一)\n" + MODELS
        ),
        code(
            """# [3.5/7] 形状サニティチェック + パラメータ数の確認(学習開始前に必ず確認する)
MODEL_CLASSES = {"base": MediSegUNet, "plain_mhsa": MediSegPlainMHSAUNet, "edge_att": MediSegEdgeAttUNet}

_dummy = torch.randn(2, 3, CFG.CROP_SIZE, CFG.CROP_SIZE).cuda()
param_counts = {}
for name, cls in MODEL_CLASSES.items():
    _probe = cls(3, 1, False).cuda()
    with torch.no_grad():
        _out = _probe(_dummy)
    assert _out.shape == (2, 1, CFG.CROP_SIZE, CFG.CROP_SIZE), f"{name}: 出力形状が想定と異なる"
    param_counts[name] = sum(p.numel() for p in _probe.parameters())
    del _probe
    torch.cuda.empty_cache()

print(json.dumps(param_counts, indent=2))
diff = param_counts["edge_att"] - param_counts["plain_mhsa"]
print(f"edge_att - plain_mhsa = {diff} params ({diff / param_counts['edge_att'] * 100:.4f}% of total)"
      " — ほぼ完全なmatched-parameter対照になっていることを確認")
del _dummy, _out
torch.cuda.empty_cache()
print("形状チェックOK")
"""
        ),
        code(
            """# [4/7] Dataset・loss・学習/評価ループ (build_v51_edgeattnet_refiner.pyと同一)
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

def choose_batch_size(dataset, model_cls):
    for candidate in CFG.REFINER_BATCH_CANDIDATES:
        try:
            loader = DataLoader(dataset, batch_size=candidate, shuffle=True, num_workers=2, pin_memory=True)
            image, target = next(iter(loader))
            probe = model_cls(3, 1, False).cuda()
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(probe(image.cuda()), target.cuda())
            loss.backward()
            del probe, image, target, loss, loader
            torch.cuda.empty_cache()
            return candidate
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("どのbatch sizeもGPUメモリに収まらない")

def train_one_run(variant, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model_cls = MODEL_CLASSES[variant]
    train_ds = InstanceCropDataset(train_files_fold, augment=True)
    valid_ds = InstanceCropDataset(valid_files_fold, augment=False)
    batch_size = choose_batch_size(train_ds, model_cls)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model = model_cls(3, 1, False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.REFINER_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    best_dice, best_state = -1.0, None
    started = time.time()

    for epoch in range(CFG.REFINER_EPOCHS):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"{variant}/seed{seed} epoch {epoch + 1}/{CFG.REFINER_EPOCHS}")
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
        print(f"{variant}/seed{seed} epoch {epoch + 1}: crop Dice {dice:.6f} (best {best_dice:.6f})")

    checkpoint_path = CFG.ARTIFACT_ROOT / f"ablation_{variant}_seed{seed}_fold{CFG.FOLD}.pt"
    torch.save({"model": model.state_dict(), "best_model": best_state, "best_dice": best_dice},
               checkpoint_path)
    minutes = round((time.time() - started) / 60, 1)
    print(f"{variant}/seed{seed}: checkpoint={checkpoint_path} minutes={minutes}")
    del model, optimizer, scheduler, train_loader, valid_loader, train_ds, valid_ds
    torch.cuda.empty_cache()
    return {"checkpoint": str(checkpoint_path), "best_crop_dice": best_dice, "minutes": minutes,
            "batch_size": batch_size}
"""
        ),
        code(
            """# [5/7] OOF採点用の後処理(V51と同一: shrink@0.45 + drop_contained@0.90)
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
            """# [6/7] 3構成 x seed 2本 = 6本を直列に学習・OOF評価(合計で2〜3時間かかる見込み)
results = []
for variant in CFG.VARIANTS:
    for seed in CFG.SEEDS:
        print(f"\\n{'=' * 60}\\n{variant} / seed {seed}\\n{'=' * 60}")
        train_summary = train_one_run(variant, seed)

        refiner = MODEL_CLASSES[variant](3, 1, False).cuda().eval()
        ckpt = torch.load(train_summary["checkpoint"], map_location="cuda", weights_only=False)
        refiner.load_state_dict(ckpt["best_model"] or ckpt["model"], strict=True)
        oof = evaluate_oof(refiner)
        delta_vs_base = oof["pq"] - CFG.BASELINE_PQ
        print(json.dumps({"variant": variant, "seed": seed, **oof,
                          "delta_vs_base_refiner": round(delta_vs_base, 6)}, indent=2))

        results.append({
            "variant": variant, "seed": seed, "n_params": param_counts[variant],
            **train_summary, **oof, "delta_vs_base_refiner": delta_vs_base,
        })
        del refiner, ckpt
        torch.cuda.empty_cache()

        # 途中経過を都度Driveへ保存(セッション切断への備え)
        (CFG.ARTIFACT_ROOT / f"matched_param_ablation_fold{CFG.FOLD}_partial.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print("\\n全6本完了")
"""
        ),
        code(
            """# [7/7] 集計・manifest保存
import statistics

by_variant = {}
for row in results:
    by_variant.setdefault(row["variant"], []).append(row["pq"])

summary_by_variant = {
    variant: {
        "n_params": param_counts[variant],
        "seeds": CFG.SEEDS,
        "pq_per_seed": pqs,
        "pq_mean": statistics.mean(pqs),
        "pq_std": statistics.pstdev(pqs) if len(pqs) > 1 else 0.0,
    }
    for variant, pqs in by_variant.items()
}

manifest = {
    "experiment_id": f"matched_param_ablation_fold{CFG.FOLD}",
    "purpose": "report/main.tex Future Work item 1. fold0で見つかったedge-attention"
               "反転結果が、他foldでも再現するかを確認",
    "fold": CFG.FOLD,
    "recipe": "build_v51_edgeattnet_refiner.pyと完全同一(crop320/context1.65/"
              "14epoch/LR2e-4/crop Diceでbest選択)。モデルクラスとseedのみ変更",
    "reference_baseline_pq_original_single_seed": CFG.BASELINE_PQ,
    "reference_edgeatt_pq_original_single_seed_v51": CFG.EDGEATT_PQ,
    "runs": results,
    "summary_by_variant": summary_by_variant,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}
report_path = CFG.ARTIFACT_ROOT / f"matched_param_ablation_fold{CFG.FOLD}_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary_by_variant, ensure_ascii=False, indent=2))
print(f"\\n保存: {report_path}")
print("\\n次: fold0の結果と合わせて、report/main.texのFuture workを更新する。")
"""
        ),
    ]
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["0", "1", "2"], help="学習対象fold")
    args = parser.parse_args()
    fold = int(args.fold)

    cells = build_cells(fold)
    for index, cell in enumerate(cells):
        cell["id"] = f"ablation-f{fold}-{index:02d}"

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

    output = ROOT / f"solar_filament_matched_param_ablation_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
