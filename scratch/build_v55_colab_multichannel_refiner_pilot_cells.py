"""Embedded cell payloads and builder helpers for the build_v55_colab_multichannel_refiner_pilot notebook builder.

Verbatim extraction of the cell-generating functions and the constants they close over
from build_v55_colab_multichannel_refiner_pilot.py so the builder stays readable.
Loaded by the builder via sibling import.
"""


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")


FOLD = 0


BASELINE_PQ = 0.4414  # matched-parameter ablation実測、base/2seed平均(README.md §2.1.10)


SEEDS = (3407, 8407)


CONFIGS = ("baseline", "multichannel")


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells():
    cells = [
        markdown(
            """# マルチチャンネルrefiner入力パイロット — fold0、baseアーキテクチャ固定

WINNING_STRATEGY_2026-08-29.md §6.12優先度2。現行の3ch(色画像のみ)入力に、
**seedマスク**(学習時: GTのランダム侵食/膨張による疑似proposal、推論時: 実際の
YOLO候補マスク)と**正規化距離変換**(seedのdistance transform、クロップごとに
0-1へ動的正規化)の2chを追加した5ch入力を比較する。

| 構成 | 入力チャンネル |
|---|---|
| `baseline` | 3ch(色画像のみ、現行と同一) |
| `multichannel` | 5ch(色画像3ch + seedマスク1ch + 正規化距離変換1ch) |

**train/inference不一致に関する設計判断(V46の教訓を踏まえる):** 学習時の
seedはGTをそのまま使わずランダムに侵食/膨張させた疑似proposalを使う
(GT=targetそのままだと恒等写像を学習するだけになるため)。この摂動は実際の
YOLO誤差の完全な代替にはならない既知の簡略化だが、恒等写像の罠だけは確実に
回避する。

fold0基準(標準refiner、matched-parameter ablation実測、2seed平均): {baseline_pq}。
GPU必須。A100/L4推奨(4本直列で合計1〜1.5時間見込み)。
""".format(baseline_pq=BASELINE_PQ)
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
    LOCAL_ROOT = Path("/content/solar_multichannel_pilot")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    # --- build_v51_edgeattnet_refiner.py / build_v54_colab_ema_pilot.py と同一。触らないこと ---
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
    SEEDS = %r
    CONFIGS = %r  # ("baseline", "multichannel")
    CHANNELS_BY_CONFIG = {"baseline": 3, "multichannel": 5}

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
            % (FOLD, BASELINE_PQ, SEEDS, CONFIGS)
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
            """# [3.5/7] 形状サニティチェック(3ch/5chの両方)
for n_channels in (3, 5):
    _dummy = torch.randn(2, n_channels, CFG.CROP_SIZE, CFG.CROP_SIZE).cuda()
    _probe = MediSegUNet(n_channels, 1, False).cuda()
    with torch.no_grad():
        _out = _probe(_dummy)
    assert _out.shape == (2, 1, CFG.CROP_SIZE, CFG.CROP_SIZE), f"{n_channels}ch: 出力形状が想定と異なる"
    print({"in_channels": n_channels, "n_params": sum(p.numel() for p in _probe.parameters())})
    del _probe, _dummy, _out
    torch.cuda.empty_cache()
print("形状チェックOK")
"""
        ),
        code(
            """# [4/7] Dataset(5ch常時生成)・seed摂動・距離変換・loss・学習/評価ループ
# refiner_loss は build_v51_edgeattnet_refiner.py と完全同一(V30から不変、変更しない)
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

def perturb_seed(mask_u8, rng):
    \"\"\"学習時のseedチャンネル生成。GTをそのまま使うと恒等写像を学習するだけに
    なる(V46が踏んだ訓練/推論不一致の罠と同種)ため、ランダム侵食/膨張で
    疑似proposalを作る。細い構造が侵食で消える場合は軽い膨張へフォールバックする
    (ローカルで2000試行・出現率0%を確認済み)。\"\"\"
    op = rng.choice(["erode", "dilate", "none"])
    if op == "none" or mask_u8.sum() == 0:
        return mask_u8
    kernel = np.ones((3, 3), np.uint8)
    iterations = int(rng.integers(1, 7))
    if op == "erode":
        perturbed = cv2.erode(mask_u8, kernel, iterations=iterations)
        if perturbed.sum() == 0:
            perturbed = cv2.dilate(mask_u8, kernel, iterations=1)
        return perturbed
    return cv2.dilate(mask_u8, kernel, iterations=iterations)

def normalized_distance_transform(mask_u8):
    \"\"\"クロップごとの最大値で0-1へ動的正規化する(固定定数で正規化すると細い
    構造で常に極小値になり情報量が失われることをローカルで確認済み)。\"\"\"
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    peak = dist.max()
    if peak > 0:
        dist = dist / peak
    return dist.astype(np.float32)

_seed_rng = np.random.default_rng(20260915)

class InstanceCropDataset(Dataset):
    \"\"\"常に5ch([色3ch, seed1ch, 正規化距離変換1ch])を返す。baseline構成は
    先頭3chだけを使う(train_one_run側でスライス、augmentationのRNG消費を
    両構成で完全に揃えるため)。\"\"\"
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
        target_full = ann_mask(image_id, ann)
        x0, y0, x1, y1 = square_box_from_bbox(ann["bbox"], info["height"], info["width"])
        image = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target_full[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_NEAREST)

        seed = perturb_seed(target.copy(), _seed_rng)
        dist = normalized_distance_transform(seed)

        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        seed_t = torch.from_numpy(seed[None].copy()).float()
        dist_t = torch.from_numpy(dist[None].copy()).float()
        stacked = torch.cat([image_t, seed_t, dist_t], dim=0)  # (5, H, W)
        target_t = torch.from_numpy(target[None].copy()).float()

        if self.augment:
            k = random.randrange(4)
            stacked, target_t = torch.rot90(stacked, k, (1, 2)), torch.rot90(target_t, k, (1, 2))
            if random.random() < 0.5:
                stacked, target_t = torch.flip(stacked, (2,)), torch.flip(target_t, (2,))
            if random.random() < 0.5:
                stacked, target_t = torch.flip(stacked, (1,)), torch.flip(target_t, (1,))
            if random.random() < 0.5:
                stacked[:2] = torch.clamp(stacked[:2] ** random.uniform(0.85, 1.20), 0, 1)
        return stacked, target_t

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

def crop_dice(model, loader, n_channels, threshold=0.5):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for stacked, target in loader:
            stacked, target = stacked[:, :n_channels].cuda(non_blocking=True), target.cuda(non_blocking=True).bool()
            pred = model(stacked).sigmoid() >= threshold
            intersection = (pred & target).sum((1, 2, 3)).float()
            denominator = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
            total += ((2 * intersection + 1) / (denominator + 1)).sum().item()
            count += len(stacked)
    return total / max(count, 1)

def choose_batch_size(dataset, n_channels):
    for candidate in CFG.REFINER_BATCH_CANDIDATES:
        try:
            loader = DataLoader(dataset, batch_size=candidate, shuffle=True, num_workers=2, pin_memory=True)
            stacked, target = next(iter(loader))
            probe = MediSegUNet(n_channels, 1, False).cuda()
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(probe(stacked[:, :n_channels].cuda()), target.cuda())
            loss.backward()
            del probe, stacked, target, loss, loader
            torch.cuda.empty_cache()
            return candidate
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("どのbatch sizeもGPUメモリに収まらない")

def train_one_run(config, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    n_channels = CFG.CHANNELS_BY_CONFIG[config]
    train_ds = InstanceCropDataset(train_files_fold, augment=True)
    valid_ds = InstanceCropDataset(valid_files_fold, augment=False)
    batch_size = choose_batch_size(train_ds, n_channels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model = MediSegUNet(n_channels, 1, False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.REFINER_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    best_dice, best_state = -1.0, None
    started = time.time()

    for epoch in range(CFG.REFINER_EPOCHS):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"{config}/seed{seed} epoch {epoch + 1}/{CFG.REFINER_EPOCHS}")
        for stacked, target in progress:
            stacked = stacked[:, :n_channels].cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(model(stacked), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{running / (progress.n + 1):.4f}")
        scheduler.step()
        dice = crop_dice(model, valid_loader, n_channels)
        if dice > best_dice:
            best_dice = dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{config}/seed{seed} epoch {epoch + 1}: crop Dice {dice:.6f} (best {best_dice:.6f})")

    checkpoint_path = CFG.ARTIFACT_ROOT / f"v55_multichannel_pilot_{config}_seed{seed}_fold{CFG.FOLD}.pt"
    torch.save({"model": model.state_dict(), "best_model": best_state, "best_dice": best_dice,
                "config": config, "n_channels": n_channels}, checkpoint_path)
    minutes = round((time.time() - started) / 60, 1)
    print(f"{config}/seed{seed}: checkpoint={checkpoint_path} minutes={minutes}")
    del model, optimizer, scheduler, train_loader, valid_loader, train_ds, valid_ds
    torch.cuda.empty_cache()
    return {"checkpoint": str(checkpoint_path), "best_crop_dice": best_dice, "minutes": minutes,
            "batch_size": batch_size, "n_channels": n_channels}
"""
        ),
        code(
            """# [5/7] OOF採点用の後処理(V51と同一の枠組み。refine_proposalのみconfig対応に拡張)
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
def refine_proposal(refiner, image, proposal, n_channels):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float() / 255.0
    if n_channels == 5:
        # 推論時のseedは実際のYOLO候補マスクそのもの(摂動なし、本物の不完全な入力)
        proposal_u8 = proposal[y0:y1, x0:x1].astype(np.uint8)
        seed_crop = cv2.resize(proposal_u8, (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_NEAREST)
        dist_crop = normalized_distance_transform(seed_crop)
        seed_t = torch.from_numpy(seed_crop[None, None].copy()).float()
        dist_t = torch.from_numpy(dist_crop[None, None].copy()).float()
        tensor = torch.cat([tensor, seed_t, dist_t], dim=1)
    tensor = tensor.cuda()
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
def evaluate_oof(refiner, n_channels):
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
        refined = [refine_proposal(refiner, image, m, n_channels) for m in proposals]
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
    n_channels = CFG.CHANNELS_BY_CONFIG[config]
    for seed in CFG.SEEDS:
        print(f"\\n{'=' * 60}\\n{config} ({n_channels}ch) / seed {seed}\\n{'=' * 60}")
        train_summary = train_one_run(config, seed)

        refiner = MediSegUNet(n_channels, 1, False).cuda().eval()
        ckpt = torch.load(train_summary["checkpoint"], map_location="cuda", weights_only=False)
        refiner.load_state_dict(ckpt["best_model"] or ckpt["model"], strict=True)
        oof = evaluate_oof(refiner, n_channels)
        delta_vs_base = oof["pq"] - CFG.BASELINE_PQ
        print(json.dumps({"config": config, "seed": seed, **oof,
                          "delta_vs_baseline": round(delta_vs_base, 6)}, indent=2))

        results.append({
            "config": config, "seed": seed, "n_channels": n_channels,
            **train_summary, **oof, "delta_vs_baseline": delta_vs_base,
        })
        del refiner, ckpt
        torch.cuda.empty_cache()

        # 途中経過を都度Driveへ保存(セッション切断への備え)
        (CFG.ARTIFACT_ROOT / f"v55_multichannel_pilot_fold{CFG.FOLD}_partial.json").write_text(
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
        "n_channels": CFG.CHANNELS_BY_CONFIG[config],
        "seeds": CFG.SEEDS,
        "pq_per_seed": pqs,
        "pq_mean": statistics.mean(pqs),
        "pq_std": statistics.pstdev(pqs) if len(pqs) > 1 else 0.0,
    }
    for config, pqs in by_config.items()
}

manifest = {
    "experiment_id": f"v55_multichannel_refiner_pilot_fold{CFG.FOLD}",
    "purpose": "WINNING_STRATEGY_2026-08-29.md §6.12優先度2。refiner入力への"
               "seedマスク+正規化距離変換の追加が改善するか、baseアーキテクチャ・"
               "fold0で検証",
    "fold": CFG.FOLD,
    "seed_perturbation": "学習時はGTをランダム侵食(1-6px)/膨張(1-6px)/無変換で"
                         "摂動した疑似proposal。推論時は実際のYOLO候補マスクを"
                         "そのまま使用(摂動なし)。GT=seedの恒等写像の罠(V46と"
                         "同種)を回避するための設計",
    "recipe": "build_v54_colab_ema_pilot.pyのbaseline構成と完全同一"
              "(crop320/context1.65/14epoch/LR2e-4/refiner_loss不変)。"
              "入力チャンネル数のみ変更(EMAは併用しない)",
    "reference_baseline_pq": CFG.BASELINE_PQ,
    "runs": results,
    "summary_by_config": summary_by_config,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}
report_path = CFG.ARTIFACT_ROOT / f"v55_multichannel_pilot_fold{CFG.FOLD}_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary_by_config, ensure_ascii=False, indent=2))
print(f"\\n保存: {report_path}")
print("\\n次: fold0でmultichannelがbaselineを上回れば、fold1・fold2でも同じ検証を行い、"
      "「単一foldのbestを一般化しない」という規律に従って独立2fold以上での符号一致を確認する。")
"""
        ),
    ]
    return cells
