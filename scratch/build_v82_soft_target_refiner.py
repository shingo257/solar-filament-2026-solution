"""V82 — soft annotator-mean 目標の refiner 学習ノートブックビルダー (C2, PLAN §7.2)。

## 目的

現行 refiner の学習単位は (file, image_id, ann) = **record 単位の硬いマスク**。
同一フィラメントを複数アノテータが描いていれば、中心・輪郭の違う 3 サンプル
として独立に学習される。V82 は GT インスタンスを IoU>0.3 でクラスタ化し、
`S_k = (1/R) Σ_r M_{k,r}` (R = 画像のアノテータ数)を soft 目標として与える
効果を検証する (PLAN_2026-10-09_FINAL_SPRINT.md §7.2)。モデルは現行 refiner
と同じ `MediSegUNet` のまま、目標の定義だけを変える。

arms:
  soft       : S_k ∈ [0,1] の soft 目標 (INTER_LINEAR リサイズ)
  hard_union : クラスタの union を 1/0 目標にする対照 (クラスタ化のみの効果を分離)

hard_record (現行) は既存 `mediseg_refiner_fold{f}_resume.pt` がそのまま対照。

## 使い方

    python scratch/build_v82_soft_target_refiner.py --fold 0 --arm soft --seed 3407
    python scratch/build_v82_soft_target_refiner.py --fold 0 --arm hard_union --seed 3407

GPU必須 (Colab L4 推奨、各run ≈1.5h)。fold 専用 OOF 評価は確定後処理
(shrink + drop_contained_then_area_asc)込みで、標準 refiner の基準 PQ と比較。
採用ゲート: soft − hard_record が 3fold・両seedで正、かつ pooled CI がゼロを除外。
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")
SOFT_TARGET = (ROOT / "solar_filament" / "pipeline" / "soft_target.py").read_text(encoding="utf-8")


def _embed_source(src: str) -> str:
    """ノートブックセルに埋め込むソースから `from __future__` 行を除去する。

    `from __future__ import annotations` はコンパイラフラグの設定に加えて
    名前 `annotations` を `_Feature` オブジェクトとして実行名前空間に
    束縛する。全セルが同一グローバルで動くため、setup セルの
    `annotations = defaultdict(list)`(coco アノテーション)を上書きし、
    `annotations[i]` が `'_Feature' object is not subscriptable` で落ちる。
    Colab は Python>=3.10 で注釈が標準評価できるため除去して安全。
    """
    return "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("from __future__")
    ) + "\n"

BASELINE_PQ = {0: 0.4393, 1: 0.4415, 2: 0.4471}  # 標準refinerの全fold OOF基準(README §2「現在地」)
THRESHOLD_SWEEP = (0.35, 0.40, 0.45, 0.50)  # §7.2.4 事前宣言の補助掃引(主結果は 0.45)


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def build_cells(fold_num, arm, seed):
    assert arm in ("soft", "hard_union")
    soft = arm == "soft"

    cells = [
        markdown(
            f"""# V82 — soft annotator-mean 目標 refiner: fold{fold_num} / arm={arm} / seed={seed}

GT インスタンスを IoU>0.3 でクラスタ化し、{'`S_k = (1/R) Σ_r M_{k,r}` の soft 目標で学習' if soft else 'クラスタ union の 1/0 目標で学習(クラスタ化のみの対照)'}。
モデル・損失・後処理は現行 refiner と同一。目的は「soft/クラスタ化された目標が
near-miss(IoU 0.3〜0.5)を TP に押し上げるか」の検証。採用ゲートは
fold{fold_num} OOF PQ vs 基準 {BASELINE_PQ[fold_num]}、さらに 3fold・両seed符号一致。

GPU必須。
"""
        ),
        code(
            f"""# [1/7] Drive接続、設定、前提の確認
import json, math, random, shutil, time
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
    SEED = {seed}
    FOLD = {fold_num}
    ARM = "{arm}"
    CLUSTER_IOU = 0.30
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v82")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    # --- 以下は build_v35_refiner_folds12.py と同一。触らないこと ---
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_EPOCHS = 14
    REFINER_LR = 2e-4
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    REPEAT_FACTOR = 1.5   # record 単位(≈8.2k)と反復数を揃える
    # 確定後処理(README.md「現在地」節)
    CONF = 0.26
    REFINER_THRESHOLD = 0.45
    THRESHOLD_SWEEP = {THRESHOLD_SWEEP}
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ = {BASELINE_PQ[fold_num]}

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)

required = {{
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像": CFG.PREPROC_CACHE,
    "fold{fold_num} 標準refiner(比較基準、任意)": CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{fold_num}_resume.pt",
    "fold{fold_num} YOLO(OOF推論用)": CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{fold_num}" / "weights" / "best.pt",
}}
missing = {{k: str(v) for k, v in required.items() if not v.exists()}}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {{missing}}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({{k: "OK" for k in required}})
print({{"GPU": torch.cuda.get_device_name(0), "arm": CFG.ARM, "seed": CFG.SEED}})
"""
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

def ann_rle(image_id, ann):
    info = images[image_id]
    height, width = int(info["height"]), int(info["width"])
    segmentation = ann["segmentation"]
    if isinstance(segmentation, list):
        return coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return coco_mask.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

print({"観測": len(train_files),
       "fold別": {f: sum(v == f for v in fold_of_file.values()) for f in (0, 1, 2)},
       "annotation": len(coco["annotations"])})
"""
        ),
        code(
            "# [3/7] MediSegUNet (solar_filament/models/mediseg_models.py と同一)\n" + _embed_source(MODELS)
        ),
        code(
            "# [3.5/7] soft_target ユーティリティ (solar_filament/pipeline/soft_target.py と同一)\n"
            + _embed_source(SOFT_TARGET)
        ),
        code(
            """# [3.7/7] 形状サニティチェック(学習開始前に必ず確認する)
_probe = MediSegUNet(3, 1, False).cuda()
_dummy = torch.randn(2, 3, CFG.CROP_SIZE, CFG.CROP_SIZE).cuda()
with torch.no_grad():
    _out = _probe(_dummy)
print("output shape:", _out.shape, "(期待値: [2, 1, 320, 320])")
assert _out.shape == (2, 1, CFG.CROP_SIZE, CFG.CROP_SIZE), "出力形状が想定と異なる"
del _probe, _dummy, _out
torch.cuda.empty_cache()
print("形状チェックOK")
"""
        ),
        code(
            """# [4/7] ClusterCropDataset・loss・学習ループ
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

class ClusterCropDataset(Dataset):
    \"\"\"(file, cluster) 単位の soft/union 目標。record 単位の InstanceCropDataset
    と同じ画像側処理で、目標だけが変わる。検証セットも同じコードを通る
    (augment=False で完全に決定的)\"\"\"

    def __init__(self, file_names, augment=False, repeat_factor=CFG.REPEAT_FACTOR):
        self.augment = augment
        self.files = {}
        self.samples = []
        for file_name in tqdm(file_names, desc="GTクラスタ化"):
            ids = file_to_ids[file_name]
            rles = [[ann_rle(i, a) for a in annotations[i] if not a.get("iscrowd", 0)]
                    for i in ids]
            # クラスタ化だけのために decode し、RLE のみ保持する (2048² bool を
            # 全件保持すると ~35GB になるため)
            masks = [[coco_mask.decode(r).astype(bool) for r in rr] for rr in rles]
            clusters = cluster_gt_instances(masks, CFG.CLUSTER_IOU)
            self.files[file_name] = (rles, clusters, len(ids))
            for cluster_index in range(len(clusters)):
                self.samples.append((file_name, cluster_index))
        target_len = int(len(self.samples) * repeat_factor)
        base = list(self.samples)
        while len(self.samples) < target_len:
            need = target_len - len(self.samples)
            self.samples += base[:need]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, cluster_index = self.samples[index]
        rles, clusters, n_records = self.files[file_name]
        cluster = clusters[cluster_index]
        info_id = file_to_ids[file_name][0]
        height, width = int(images[info_id]["height"]), int(images[info_id]["width"])

        # クラスタの union bbox → square box。decode はクラスタメンバだけに
        # 絞る (モジュール関数は cluster の (r,i) しか参照しないため None でよい)。
        member_idx = defaultdict(set)
        for r, i in cluster:
            member_idx[r].add(i)
        # member_idx[r] (subscript): defaultdict が空 set を返す。.get(r) だと
        # 未登録 record で None になり `i in None` で TypeError するので注意。
        masks_per_record = [
            [coco_mask.decode(rr[i]).astype(bool) if i in member_idx[r] else None
             for i in range(len(rr))]
            for r, rr in enumerate(rles)
        ]
        bx0, by0, bx1, by1 = cluster_bbox(masks_per_record, cluster)
        bbox = [float(bx0), float(by0), float(bx1 - bx0), float(by1 - by0)]
        x0, y0, x1, y1 = square_box_from_bbox(bbox, height, width)

        image = cv2.imread(str(CFG.PREPROC_DIR / file_name), cv2.IMREAD_COLOR)
        image = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0

        soft_np = soft_target_crop(masks_per_record, cluster, (x0, y0, x1, y1))
        if CFG.ARM == "soft":
            target_np = cv2.resize(soft_np, (CFG.CROP_SIZE, CFG.CROP_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
        else:  # hard_union: 1recordでも塗れば1 (S>0 と同値)
            target_np = cv2.resize((soft_np > 0).astype(np.uint8),
                                   (CFG.CROP_SIZE, CFG.CROP_SIZE),
                                   interpolation=cv2.INTER_NEAREST).astype(np.float32)
        target = torch.from_numpy(target_np[None].copy()).float()

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
"""
        ),
        code(
            f"""# [5/7] fold{fold_num}を学習(該当foldを検証専用に除外)
started = time.time()
ckpt_name = f"mediseg_{arm}_refiner_fold{fold_num}_s{seed}_resume.pt"
checkpoint_path = CFG.ARTIFACT_ROOT / ckpt_name
train_files_fold = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]
train_ds = ClusterCropDataset(train_files_fold, augment=True)
valid_ds = ClusterCropDataset(valid_files_fold, augment=False, repeat_factor=1.0)
assert not (set(train_files_fold) & set(valid_files_fold)), "fold分割が壊れている"
print({{"fold": CFG.FOLD, "arm": CFG.ARM, "seed": CFG.SEED,
       "train_crops": len(train_ds), "valid_crops": len(valid_ds)}})

batch_size = choose_batch_size(train_ds)
print("batch_size =", batch_size)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

model = MediSegUNet(3, 1, False).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.REFINER_EPOCHS)
scaler = torch.amp.GradScaler("cuda")
best_dice, best_state = -1.0, None

for epoch in range(CFG.REFINER_EPOCHS):
    model.train()
    running = 0.0
    progress = tqdm(train_loader, desc=f"fold{{CFG.FOLD}} {{CFG.ARM}} refiner {{epoch + 1}}/{{CFG.REFINER_EPOCHS}}")
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
                "best_dice": best_dice, "arm": CFG.ARM, "seed": CFG.SEED}}, checkpoint_path)
    print(f"fold{{CFG.FOLD}} {{CFG.ARM}} s{{CFG.SEED}} epoch {{epoch + 1}}: crop Dice {{dice:.6f}} (best {{best_dice:.6f}})")

best_crop_dice = best_dice
print({{"checkpoint": str(checkpoint_path), "best_crop_dice": best_crop_dice,
       "minutes": round((time.time() - started) / 60, 1)}})
"""
        ),
        code(
            f"""# [6/7] fold{fold_num} OOF(リークなし)を確定後処理で採点し、基準{BASELINE_PQ[fold_num]}と比較
# refiner 閾値は 0.45 が主結果、{{0.35,0.40,0.45,0.50}} は事前宣言の補助掃引 (§7.2.4)
!pip -q install ultralytics
from ultralytics import YOLO

yolo_model = YOLO(str(CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{fold_num}" / "weights" / "best.pt"))

refiner = MediSegUNet(3, 1, False).cuda().eval()
ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
refiner.load_state_dict(ckpt.get("best_model") or ckpt["model"], strict=True)

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
def refine_probability(image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    return cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), (x0, y0, x1, y1)

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

# refiner の確率マップは閾値ごとに再利用するため先に全件計算して保持する
cache = {{}}
for name in tqdm(oof_files, desc="OOF推論(確率マップを保持)"):
    image_path = CFG.PREPROC_DIR / name
    result = yolo_model.predict(str(image_path), imgsz=1280, conf=0.08, iou=0.55,
                                 agnostic_nms=True, retina_masks=True, verbose=False)[0]
    masks, confidences = result_masks(result)
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences)) if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
    proposals = [masks[i] for i in keep]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    cache[name] = [(refine_probability(image, m), m) for m in proposals]

meters = {{t: PanopticQuality() for t in CFG.THRESHOLD_SWEEP}}
for name in oof_files:
    for threshold in CFG.THRESHOLD_SWEEP:
        refined = []
        for (probability, box), proposal in cache[name]:
            x0, y0, x1, y1 = box
            mask = np.zeros_like(proposal, dtype=bool)
            mask[y0:y1, x0:x1] = probability >= threshold
            refined.append(mask & proposal)  # shrink
        predictions = apply_ownership(refined, np.ones(len(refined)))
        pred_rles = []
        if predictions:
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, pred_rles)
            meters[threshold].add_matrix(matrix, len(truth), len(pred_rles))

report = {{}}
for threshold in CFG.THRESHOLD_SWEEP:
    summary = meters[threshold].summary()
    report[str(threshold)] = {{"pq": round(summary["pq"], 6),
                              "delta_vs_baseline": round(summary["pq"] - CFG.BASELINE_PQ, 6)}}
print(json.dumps({{"fold": CFG.FOLD, "arm": CFG.ARM, "seed": CFG.SEED,
                  "baseline_pq": CFG.BASELINE_PQ, "by_threshold": report,
                  "best_crop_dice": best_crop_dice}}, ensure_ascii=False, indent=2))
"""
        ),
        code(
            """# [7/7] manifest保存
import hashlib

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "experiment_id": "v82_soft_target_refiner",
    "arm": CFG.ARM, "fold": CFG.FOLD, "seed": CFG.SEED,
    "cluster_iou": CFG.CLUSTER_IOU, "repeat_factor": CFG.REPEAT_FACTOR,
    "model": "MediSegUNet(3,1) — 現行 refiner と同一",
    "epochs": CFG.REFINER_EPOCHS, "lr": CFG.REFINER_LR,
    "best_crop_dice": best_crop_dice,
    "baseline_pq": CFG.BASELINE_PQ,
    "oof_pq_by_threshold": report,
    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_of(checkpoint_path),
    "gate": "soft - hard_record が 3fold・両seedで正、pooled CI がゼロを除外",
}
report_path = CFG.ARTIFACT_ROOT / f"v82_{CFG.ARM}_fold{CFG.FOLD}_s{CFG.SEED}_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"\\n保存: {report_path}")
"""
        ),
    ]
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--arm", type=str, required=True, choices=("soft", "hard_union"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cells = build_cells(args.fold, args.arm, args.seed)
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = args.out or ROOT / f"solar_filament_v82_{args.arm}_fold{args.fold}_s{args.seed}_colab.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
