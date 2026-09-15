"""V47 Colab版 — refinerをゼロから(warm startなし)再学習し、fold0で検証する。

## 背景

V46(`build_v46_colab_refiner_free_retrain.py`)は既存refinerの重みから
warm startして再学習したが、fold0 OOF PQは基準0.4393はもちろん、既存
refinerの閾値チューニングのみ(free@0.86=0.4361)にも届かなかった(最良でも
0.4353)。warm startは「GTの位置を渡された前提」の局所解に最適化済みで、
14epochでは新しい入力分布(YOLO候補中心クロップ)向けの解へ十分動けなかった
可能性が高いと判断した(train_loss 0.249→0.234と下降が緩やか)。

## このノートでV46から変更する点(優先度上位から複数を同時に適用)

1. **warm startを廃止し、ランダム初期化から学習する。**(最有力候補)
2. **正例マッチングのIoU閾値を0.10→0.30へ引き上げる。**
   `build_refiner_free_retrain_manifest.py --positive-min-iou 0.30` で
   再構築したmanifest(`refiner_retrain_manifest_fold0_v2.json`)を使う。
   ノイズの多い教師信号(悪いproposal-GTペア)を減らす狙い。
3. **負例(誤検出、全ゼロターゲット)混入率を0.15→0.05へ下げる。**
   同じmanifest_v2に反映済み(既存の強みを薄めた疑いへの対処)。
4. **epoch数を14→30へ増やす。** ランダム初期化なのでwarm startより
   多くのepochが要ると想定。学習率はCosineAnnealingLRのままT_maxを合わせる。
5. **最終評価は free閾値 x drop_contained閾値の同時グリッドサーチにする。**
   V46はdrop_contained0.90固定のままfree閾値だけ再スイープしており、
   新モデルでの相互作用点の変化を見落としている可能性があったため。

candidate #6(GT中心クロップの一部混合)は今回は見送り(効果検証を単純化するため)。

**採用判定はしない。** fold0 OOF PQが基準0.4393を上回ったら fold1/2 展開を検討する。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v47_colab_refiner_scratch_retrain.ipynb"
MEDISEG = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V47 Colab版 — refinerをゼロから再学習(warm startなし、fold0検証)

V46(warm start、既存重みから開始)は基準0.4393はおろか、既存refinerの
閾値チューニングのみ(free@0.86=**0.4361**)にも届かなかった(最良0.4353)。
今回はwarm startを廃止しゼロから再学習、加えて正例マッチングIoU閾値を
0.10→0.30に厳格化、負例混入率を0.15→0.05に低減、epoch数を14→30に増やし、
最終評価はfree閾値×drop_contained閾値の同時グリッドサーチにする。

**採用判定はしない。** fold0 OOF PQが基準0.4393を上回ったら fold1/2 展開を検討する。

ランタイム: **L4 High-RAM GPU**。
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
    MANIFEST = V35_ROOT / "refiner_retrain" / "refiner_retrain_manifest_fold0_v2.json"
    OOF_CACHE = V34_ROOT / "prediction_cache" / "oof"
    WORK = Path("/content/v47_refiner_scratch_retrain")
    OUT_CKPT = V35_ROOT / "refiner_retrain" / "mediseg_refiner_scratch_fold0.pt"
    FOLD = 0
    CONF = 0.26
    MIN_AREA = 20
    CROP_SIZE = 320
    REFINER_LR = 2e-4
    REFINER_EPOCHS = 30
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    DROP_THRESHOLD_DEFAULT = 0.90
    BASELINE_FOLD0_PQ = 0.4393
    FREE_BASELINE_FOLD0_PQ = 0.4361  # 既存refinerでのfree@0.86(V46までの最良)

for required in (CFG.ARTIFACTS_ROOT, CFG.PREPROC_DIR, CFG.FOLDS_CSV, CFG.TRAIN_JSON, CFG.MANIFEST):
    if not required.exists():
        raise FileNotFoundError(f"{required} が見つからない。manifest_v2をDriveへコピー済みか確認")

CFG.WORK.mkdir(parents=True, exist_ok=True)
CFG.OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
print({"images": len(list(CFG.PREPROC_DIR.glob("*.jpeg"))), "gpu": gpu_name})"""
    ),
    code(
        "# [2/6] MediSeg UNetアーキテクチャ(solar_filament/models/mediseg_models.py と同一)\n"
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
        """# [3/6] manifest_v2からProposalCropDataset(YOLO候補のbboxを中心にGTを予測)を作る
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
       "negative": sum(1 for s in samples_raw if s["is_negative"]),
       "positive_min_iou": manifest.get("positive_min_iou"),
       "negative_sample_rate": manifest.get("negative_sample_rate")})

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
print({"train_ds": len(train_ds), "valid_ds": len(valid_ds)})"""
    ),
    code(
        """# [4/6] refinerをゼロから学習(warm startなし)。epochごとにDriveへ保存(クラッシュ耐性)
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
print("ゼロから初期化(warm startなし)")

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
    torch.save({"epoch": epoch + 1, "model": model.state_dict(), "best_model": best_state,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_dice": best_dice}, CFG.OUT_CKPT)

model.load_state_dict(best_state)
torch.save({"model": model.state_dict(), "best_model": best_state, "best_dice": best_dice},
           CFG.OUT_CKPT)
train_seconds = time.perf_counter() - train_started
print(f"学習時間={train_seconds/60:.1f}分  best_crop_dice={best_dice:.4f}  保存先={CFG.OUT_CKPT}")"""
    ),
    code(
        """# [5/6] 推論ヘルパー(V46と同一)
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

def polygon_ok(ann):
    segs = [p for p in ann.get("segmentation", []) if isinstance(p, list) and len(p) >= 6]
    return bool(segs) and not ann.get("iscrowd", 0)

PQ_IOU_THRESHOLD = 0.5
METRIC_VERSION = "official-v1-2026-08-15"

class PanopticQuality:
    def __init__(self, iou_threshold=PQ_IOU_THRESHOLD):
        self.iou_threshold = iou_threshold
        self.tp_ious = []
        self.fp_count = 0
        self.fn_count = 0

    def add_matrix(self, matrix, n_gt, n_pred):
        if n_gt == 0:
            self.fp_count += n_pred
            return
        if n_pred == 0:
            self.fn_count += n_gt
            return
        hits = matrix > self.iou_threshold
        scores = matrix[hits].tolist()
        false_positives = int((hits.sum(axis=0) == 0).sum())
        false_negatives = int((hits.sum(axis=1) == 0).sum())
        self.tp_ious.extend(scores)
        self.fp_count += false_positives
        self.fn_count += false_negatives

    @property
    def denominator(self):
        return len(self.tp_ious) + 0.5 * self.fp_count + 0.5 * self.fn_count

    @property
    def score(self):
        denominator = self.denominator
        return float(sum(self.tp_ious) / denominator) if denominator > 0 else 0.0

    def summary(self):
        tp = len(self.tp_ious)
        segmentation_quality = float(np.mean(self.tp_ious)) if tp else 0.0
        denominator = self.denominator
        recognition_quality = float(tp / denominator) if denominator > 0 else 0.0
        return {
            "pq": self.score, "sq": segmentation_quality, "rq": recognition_quality,
            "tp": tp, "fp": self.fp_count, "fn": self.fn_count,
            "denominator": denominator, "iou_threshold": self.iou_threshold,
            "metric_version": METRIC_VERSION,
        }

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
print(f"fold0観測数: {len(fold0_files)}")"""
    ),
    code(
        """# [6/6] free閾値 x drop_contained閾値の同時グリッドサーチ(候補#4)
# 確率マップは画像ごとに1回だけ推論し、全閾値の組み合わせで使い回す
free_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.86, 0.90]
drop_thresholds = [0.80, 0.85, 0.90, 0.95]
meters = {(f, d): PanopticQuality() for f in free_thresholds for d in drop_thresholds}

model.eval()
started = time.perf_counter()
for position, name in enumerate(fold0_files, 1):
    gt_by_variant = []
    for image_id in file_to_ids[name]:
        info = images_by_id[image_id]
        gt_by_variant.append([to_rle(a, info["height"], info["width"])
                               for a in anns_by_image[image_id] if polygon_ok(a)])

    image = imread_unicode(CFG.PREPROC_DIR / name)
    masks, confidences = load_yolo_cache(CFG.OOF_CACHE / f"yolo_fold_{CFG.FOLD}" / f"{Path(name).stem}.npz")
    proposals = filter_proposals(masks, confidences)
    if not proposals:
        for key in meters:
            for gt_rles in gt_by_variant:
                meters[key].add_matrix(np.zeros((len(gt_rles), 0)), len(gt_rles), 0)
        continue

    boxes, probabilities = refine_batch(model, image, proposals)
    for free_threshold in free_thresholds:
        refined = [free_variant(proposals[i], boxes[i], probabilities[i], free_threshold)
                   for i in range(len(proposals))]
        for drop_threshold in drop_thresholds:
            final = apply_ownership_local(refined, drop_threshold)
            pred_rles = masks_to_rles(final)
            for gt_rles in gt_by_variant:
                matrix = iou_matrix(gt_rles, pred_rles)
                meters[(free_threshold, drop_threshold)].add_matrix(matrix, len(gt_rles), len(pred_rles))

    if position % 25 == 0 or position == len(fold0_files):
        best_so_far = max(meters.items(), key=lambda kv: kv[1].score)
        print(f"  {position}/{len(fold0_files)}  {time.perf_counter() - started:.0f}s"
              f"  best-so-far {best_so_far[0]} pq={best_so_far[1].score:.4f}", flush=True)

grid_report = {
    "scope": f"{len(fold0_files)}観測、fold0のみ。ゼロから再学習refiner、free x drop_contained同時グリッド",
    "baseline_shrink045_fold0_pq": CFG.BASELINE_FOLD0_PQ,
    "free_old_model_086_fold0_pq": CFG.FREE_BASELINE_FOLD0_PQ,
    "manifest": str(CFG.MANIFEST),
    "positive_min_iou": manifest.get("positive_min_iou"),
    "negative_sample_rate": manifest.get("negative_sample_rate"),
    "epochs": CFG.REFINER_EPOCHS,
    "warm_start": False,
    "best_crop_dice": best_dice,
    "gpu_name": gpu_name,
    "results": {
        f"free{f}_drop{d}": {
            **m.summary(),
            "delta_vs_baseline": round(m.summary()["pq"] - CFG.BASELINE_FOLD0_PQ, 6),
            "delta_vs_free_baseline": round(m.summary()["pq"] - CFG.FREE_BASELINE_FOLD0_PQ, 6),
            "beats_baseline": bool(m.summary()["pq"] > CFG.BASELINE_FOLD0_PQ),
        }
        for (f, d), m in meters.items()
    },
}
best_key = max(grid_report["results"], key=lambda k: grid_report["results"][k]["pq"])
grid_report["best_combo"] = best_key
grid_report["best_pq"] = grid_report["results"][best_key]["pq"]

local_report = CFG.WORK / "v47_refiner_scratch_fold0_grid_report.json"
local_report.write_text(json.dumps(grid_report, ensure_ascii=False, indent=2), encoding="utf-8")
drive_report = CFG.V35_ROOT / "refiner_retrain" / "v47_refiner_scratch_fold0_grid_report.json"
drive_report.write_text(json.dumps(grid_report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"best: {best_key}  pq={grid_report['best_pq']:.4f}"
      f"  delta_vs_baseline={grid_report['results'][best_key]['delta_vs_baseline']:+.4f}"
      f"  delta_vs_free_baseline={grid_report['results'][best_key]['delta_vs_free_baseline']:+.4f}")
print("saved:", local_report, drive_report)"""
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
assert "warm start" not in text or "warm startなし" in text
assert "REFINER_EPOCHS = 30" in text
OUTPUT.write_text(text, encoding="utf-8")
print(OUTPUT)
