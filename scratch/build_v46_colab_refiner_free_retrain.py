"""V46 Colab版 — refinerをfree-variant向けに再学習し、fold0で検証する。

## 背景

free variant(候補マスクとの交差制約を撤廃、確率閾値のみ)は追加学習なしで
fold0 OOF PQ 0.4361まで到達し、基準(shrink@0.45、0.4393)にΔ-0.0032まで
肉薄した。ただしdrop_contained閾値の再調整では追加改善がほぼ得られず、
閾値だけのチューニングは頭打ちに見える。

現行refinerは`build_v30_prize_colab.py`のInstanceCropDatasetで、**GTの
bboxを中心に**context1.65倍でクロップしGTマスク自体を予測するよう学習
されている。一方、推論時は(GTではなく)**YOLO候補のbboxを中心に**同じ
contextでクロップする。この訓練/推論の不一致が、free variantでの性能上限を
規定している可能性が高い。

## このノートでやること

`scratch/build_refiner_free_retrain_manifest.py`で作った、YOLO OOF候補の
bboxを中心にクロップしGTマスクをターゲットとする学習データ(リークなし、
fold0検証用にfold1/fold2のOOFキャッシュを使用)でrefinerを再学習する。
学習後、fold0のYOLO OOF候補に新refinerを適用し、free@0.86 +
drop_contained@0.90でOOF PQを測り、基準0.4393および現行free@0.86の
0.4361と比較する。

**採用判定はしない。** fold0で基準を上回った場合のみfold1/2への展開を検討する。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v46_colab_refiner_free_retrain.ipynb"
MEDISEG = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")
METRICS = (ROOT / "solar_filament" / "metrics" / "metrics_official.py").read_text(encoding="utf-8")
METRICS = METRICS.replace("from __future__ import annotations\n", "", 1)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V46 Colab版 — refinerをfree-variant向けに再学習(fold0検証)

free variant(交差制約なし)は追加学習なしでfold0 OOF PQ **0.4361**まで到達
(基準0.4393にΔ-0.0032)。閾値調整はほぼ頭打ちなので、refiner自体を
**YOLO候補のbboxを中心にクロップしGTを予測する**よう再学習する
(現行はGTのbboxを中心にクロップしており、訓練/推論の不一致がある)。

**採用判定はしない。** fold0 OOF PQが基準0.4393を上回ったら fold1/2 展開を検討する。

ランタイム: **T4/L4 GPU** を選択すること。
"""
    ),
    code(
        """# [1/6] 環境確認・依存関係・Drive mount
import json, math, shutil, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils

drive.mount("/content/drive")

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("gpu:", gpu_name)
if not torch.cuda.is_available():
    raise RuntimeError("GPUランタイムを選択してください")

class CFG:
    V30_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    V34_ROOT = Path("/content/drive/MyDrive/solar_filament_v34")
    V35_ROOT = Path("/content/drive/MyDrive/solar_filament_v35")
    ARTIFACTS_ROOT = V30_ROOT / "artifacts"
    PREPROC_DIR = V30_ROOT / "preprocessed_v30_q96"
    FOLDS_CSV = ARTIFACTS_ROOT / "grouped_folds.csv"
    TRAIN_JSON = (V30_ROOT / "competition_data" / "MAGFiLO_1.0_Kaggle_2026"
                  / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    MANIFEST = V35_ROOT / "refiner_retrain" / "refiner_retrain_manifest_fold0.json"
    OOF_CACHE = V34_ROOT / "prediction_cache" / "oof"
    OLD_REFINER_CKPT = ARTIFACTS_ROOT / "mediseg_refiner_fold0_resume.pt"
    WORK = Path("/content/v46_refiner_retrain")
    OUT_CKPT = V35_ROOT / "refiner_retrain" / "mediseg_refiner_free_fold0.pt"
    FOLD = 0
    CONF = 0.26
    MIN_AREA = 20
    CROP_SIZE = 320
    REFINER_LR = 2e-4
    REFINER_EPOCHS = 14
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    FREE_THRESHOLD = 0.86
    DROP_THRESHOLD = 0.90
    BASELINE_FOLD0_PQ = 0.4393
    FREE_BASELINE_FOLD0_PQ = 0.4361  # 現行refinerでのfree@0.86(基準)

for required in (CFG.ARTIFACTS_ROOT, CFG.PREPROC_DIR, CFG.FOLDS_CSV, CFG.TRAIN_JSON, CFG.MANIFEST):
    if not required.exists():
        raise FileNotFoundError(f"{required} が見つからない。Driveのマウント・パスを確認")

CFG.WORK.mkdir(parents=True, exist_ok=True)
CFG.OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
print({"images": len(list(CFG.PREPROC_DIR.glob("*.jpeg"))), "gpu": gpu_name})
"""
    ),
    code(
        """# [2/6] MediSeg UNetアーキテクチャ(solar_filament/models/mediseg_models.py と同一)
"""
        + MEDISEG
        + """
def soft_dice_loss(logits, targets):
    probs = logits.sigmoid()
    intersection = (probs * targets).sum((1, 2, 3))
    denominator = probs.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()

def boundary_map(tensor):
    dilation = torch.nn.functional.max_pool2d(tensor, 5, stride=1, padding=2)
    erosion = -torch.nn.functional.max_pool2d(-tensor, 5, stride=1, padding=2)
    return torch.clamp(dilation - erosion, 0, 1)

def refiner_loss(logits, targets):
    region = 0.35 * torch.nn.functional.binary_cross_entropy_with_logits(logits, targets) \\
             + 0.65 * soft_dice_loss(logits, targets)
    edge = torch.nn.functional.l1_loss(boundary_map(logits.sigmoid()), boundary_map(targets))
    return region + 0.15 * edge
"""
    ),
    code(
        """# [3/6] manifestからProposalCropDataset(YOLO候補のbboxを中心にGTを予測)を作る
def imread_unicode(path):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)

with open(CFG.TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)
images_by_id = {i["id"]: i for i in coco["images"]}
ann_by_id = {a["id"]: a for a in coco["annotations"]}

def ann_mask(ann, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in ann.get("segmentation", []):
        if len(polygon) < 6:
            continue
        points = np.rint(np.asarray(polygon).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

with open(CFG.MANIFEST, encoding="utf-8") as handle:
    manifest = json.load(handle)
samples_raw = manifest["samples"]
print({"total_samples": len(samples_raw),
       "negative": sum(1 for s in samples_raw if s["is_negative"])})

# 90/10でtrain/validに分割(manifest全体をシャッフル済みなので先頭からで良い)
split_index = int(len(samples_raw) * 0.9)
train_samples = samples_raw[:split_index]
valid_samples = samples_raw[split_index:]
print({"train": len(train_samples), "valid": len(valid_samples)})

class ProposalCropDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = imread_unicode(CFG.PREPROC_DIR / sample["file_name"])
        x0, y0, x1, y1 = sample["box"]
        if sample["is_negative"]:
            target = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        else:
            ann = ann_by_id[sample["target_ann_id"]]
            info = images_by_id[sample["target_image_id"]]
            full_target = ann_mask(ann, int(info["height"]), int(info["width"]))
            target = full_target[y0:y1, x0:x1]

        crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE),
                          interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_NEAREST)
        crop_t = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float() / 255.0
        target_t = torch.from_numpy(target[None].copy()).float()
        if self.augment:
            k = np.random.randint(4)
            crop_t, target_t = torch.rot90(crop_t, k, (1, 2)), torch.rot90(target_t, k, (1, 2))
            if np.random.random() < 0.5:
                crop_t, target_t = torch.flip(crop_t, (2,)), torch.flip(target_t, (2,))
            if np.random.random() < 0.5:
                crop_t, target_t = torch.flip(crop_t, (1,)), torch.flip(target_t, (1,))
        return crop_t, target_t

train_ds = ProposalCropDataset(train_samples, augment=True)
valid_ds = ProposalCropDataset(valid_samples, augment=False)
print({"train_ds": len(train_ds), "valid_ds": len(valid_ds)})
"""
    ),
    code(
        """# [4/6] refinerを学習(既存のtrain_refinerロジックと同一)
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

REFINER_BATCH_SIZE = choose_refiner_batch_size(train_ds)
print("REFINER_BATCH_SIZE =", REFINER_BATCH_SIZE)

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

train_started = time.perf_counter()
train_loader = DataLoader(train_ds, batch_size=REFINER_BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
valid_loader = DataLoader(valid_ds, batch_size=REFINER_BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
model = MediSegUNet(in_channels=3, out_channels=1, bilinear=False).cuda()
# 既存(shrink向け)refinerの重みから warm start する
old_state = torch.load(CFG.OLD_REFINER_CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(old_state.get("best_model") or old_state["model"], strict=True)
print("warm start: 既存refinerの重みから開始")

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.REFINER_EPOCHS)
scaler = torch.amp.GradScaler("cuda")
best_dice, best_state = -1.0, None

for epoch in range(CFG.REFINER_EPOCHS):
    model.train()
    running = 0.0
    for image, target in tqdm(train_loader, desc=f"epoch {epoch + 1}/{CFG.REFINER_EPOCHS}", leave=False):
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
    print({"epoch": epoch + 1, "train_loss": running / len(train_ds), "crop_dice": val_dice, "best": best_dice})

model.load_state_dict(best_state)
torch.save({"model": model.state_dict(), "best_model": best_state, "best_dice": best_dice},
           CFG.OUT_CKPT)
train_seconds = time.perf_counter() - train_started
print(f"学習時間={train_seconds/60:.1f}分  best_crop_dice={best_dice:.4f}  保存先={CFG.OUT_CKPT}")
"""
    ),
    code(
        """# [5/6] fold0 YOLO OOFへ新refinerを適用し、free@0.86 + drop_contained@0.90でnpzキャッシュ相当を評価
def load_yolo_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))

def filter_proposals(masks, confidences):
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
    return [masks[i] for i in keep]

def square_box(mask, context=1.65, minimum=96):
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

def refine_batch(refiner_model, image, proposals, batch=16):
    boxes, crops = [], []
    for proposal in proposals:
        x0, y0, x1, y1 = square_box(proposal)
        crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        boxes.append((x0, y0, x1, y1))
        crops.append(crop.transpose(2, 0, 1))
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            chunk = np.stack(crops[start:start + batch])
            tensor = torch.from_numpy(chunk.copy()).float().cuda() / 255.0
            output = refiner_model(tensor).sigmoid()[:, 0].cpu().numpy()
            for offset in range(output.shape[0]):
                x0, y0, x1, y1 = boxes[start + offset]
                probabilities.append(cv2.resize(output[offset], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR))
    return boxes, probabilities

def free_variant(proposal, box, probability, threshold):
    x0, y0, x1, y1 = box
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    return refined

def greedy_disjoint_local(masks, order, min_area=CFG.MIN_AREA, canvas=2048):
    occupied = np.zeros((canvas, canvas), dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept

def drop_contained_indices_local(masks, threshold):
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

def apply_ownership_local(masks, threshold):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices_local(masks, threshold)
    return greedy_disjoint_local(masks, sorted(subset, key=lambda i: areas[i]))

fold_frame = pd.read_csv(CFG.FOLDS_CSV)
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
file_to_ids = defaultdict(list)
for image_id, info in images_by_id.items():
    file_to_ids[info["file_name"]].append(image_id)
anns_by_image = defaultdict(list)
for ann in coco["annotations"]:
    anns_by_image[ann["image_id"]].append(ann)

fold0_files = sorted(
    n for n in file_to_ids
    if fold_of_file.get(n) == CFG.FOLD and (CFG.PREPROC_DIR / n).exists()
    and (CFG.OOF_CACHE / f"yolo_fold_{CFG.FOLD}" / f"{Path(n).stem}.npz").exists()
)
print(f"fold0観測数: {len(fold0_files)}")

model.eval()
cache_dir = CFG.WORK / "oof_cache_newrefiner"
started = time.perf_counter()
for position, name in enumerate(fold0_files, 1):
    out_path = cache_dir / f"{Path(name).stem}.npz"
    if out_path.exists():
        continue
    image = imread_unicode(CFG.PREPROC_DIR / name)
    masks, confidences = load_yolo_cache(CFG.OOF_CACHE / f"yolo_fold_{CFG.FOLD}" / f"{Path(name).stem}.npz")
    proposals = filter_proposals(masks, confidences)
    if not proposals:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, packed=np.empty((0, 0, 0), np.uint8),
                            shape=np.asarray([2048, 2048], np.int32), confidences=np.zeros(0, np.float32))
        continue
    boxes, probabilities = refine_batch(model, image, proposals)
    refined = [free_variant(proposals[i], boxes[i], probabilities[i], CFG.FREE_THRESHOLD)
               for i in range(len(proposals))]
    final = apply_ownership_local(refined, CFG.DROP_THRESHOLD)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if final:
        packed = np.packbits(np.stack(final).astype(np.uint8), axis=-1)
        shape = np.asarray(final[0].shape, np.int32)
    else:
        packed = np.empty((0, 0, 0), np.uint8)
        shape = np.asarray([2048, 2048], np.int32)
    np.savez_compressed(out_path, packed=packed, shape=shape,
                        confidences=np.ones(len(final), np.float32))
    if position % 25 == 0 or position == len(fold0_files):
        print(f"  {position}/{len(fold0_files)}  {time.perf_counter() - started:.0f}s", flush=True)
print(cache_dir)
"""
    ),
    code(
        (
            "# [6/6] 公式PQで新refiner+free@0.86の結果を採点し、基準0.4393・現行free@0.86の0.4361と比較\n"
            + METRICS
            + """

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

def masks_to_rles(masks):
    if not masks:
        return []
    return mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))

def iou_matrix(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    return np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64).T

def load_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))

def polygon_ok(ann):
    segs = [p for p in ann.get("segmentation", []) if isinstance(p, list) and len(p) >= 6]
    return bool(segs) and not ann.get("iscrowd", 0)

meter = PanopticQuality()
for name in fold0_files:
    gt_by_variant = []
    for image_id in file_to_ids[name]:
        info = images_by_id[image_id]
        gt_by_variant.append([to_rle(a, info["height"], info["width"])
                               for a in anns_by_image[image_id] if polygon_ok(a)])
    cache_path = cache_dir / f"{Path(name).stem}.npz"
    if not cache_path.exists():
        continue
    masks, confidences = load_cache(cache_path)
    # このキャッシュは既にfree+drop_contained適用済みなのでそのまま使う
    pred_rles = masks_to_rles(masks)
    for gt_rles in gt_by_variant:
        matrix = iou_matrix(gt_rles, pred_rles)
        meter.add_matrix(matrix, len(gt_rles), len(pred_rles))

summary = meter.summary()
report = {
    "scope": f"{len(fold0_files)}観測、fold0のみ。free-variant向け再学習refiner + free@{CFG.FREE_THRESHOLD} + drop_contained@{CFG.DROP_THRESHOLD}",
    **summary,
    "baseline_shrink045_fold0_pq": CFG.BASELINE_FOLD0_PQ,
    "delta_vs_baseline": round(summary["pq"] - CFG.BASELINE_FOLD0_PQ, 6),
    "beats_baseline": bool(summary["pq"] > CFG.BASELINE_FOLD0_PQ),
    "free_baseline_fold0_pq": CFG.FREE_BASELINE_FOLD0_PQ,
    "delta_vs_free_baseline": round(summary["pq"] - CFG.FREE_BASELINE_FOLD0_PQ, 6),
    "gpu_name": gpu_name,
}
local_report = CFG.WORK / "v46_refiner_retrain_fold0_report.json"
local_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
drive_report = CFG.V35_ROOT / "refiner_retrain" / "v46_refiner_retrain_fold0_report.json"
drive_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("saved:", local_report, drive_report)
"""
        )
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for cell in cells:
    if cell["cell_type"] == "code":
        ast.parse(re.sub(r"^[%!].*$", "", cell["source"], flags=re.M))
print("syntax OK")

text = json.dumps(notebook, ensure_ascii=False, indent=1)
assert "MediSegUNet" in text
assert "FREE_THRESHOLD = 0.86" in text
assert "REFINER_EPOCHS = 14" in text
OUTPUT.write_text(text, encoding="utf-8")
print(OUTPUT)
