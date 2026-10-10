"""公開コードリポジトリ用の統合パイプラインnotebookビルダー。

## 目的

Kaggle公式ルール「Final Submission」節の要件:
「a jupyter notebook should illustrate the entire pipeline. The committee
should not require any additional file to reproduce the submitted
predictions.」

これまでの実験は`build_v30_prize_colab.py`(検出器+旧refiner+アンサンブル)、
`build_v51_edgeattnet_refiner.py`(新refiner、fold別に分割生成)、
`build_v52_edgeattnet_submission.py`(提出物生成)と目的別に分かれていたが、
「単一notebookでパイプライン全体を示す」という要件に合わせ、この
スクリプトが全ステージ(前処理→YOLO検出器3-fold学習→EdgeAttNet refiner
3-fold+all-data学習→V52推論・提出物生成→RLE厳密検証)を1本のnotebookへ
統合する。

## 設計方針

各ステージは既存の検証済みスクリプト(`build_v30_prize_colab.py`の
データセット構築・YOLO学習ロジック、`build_v51_edgeattnet_refiner.py`の
refiner学習ロジック、`build_v52_edgeattnet_submission.py`の推論ロジック)
と同一のハイパーパラメータ・処理順序を保つ。数値を変えると再現性が
崩れるため、変更する場合は元スクリプトと本スクリプトの両方を更新すること。

チェックポイントが既にDrive上に存在する場合は学習をスキップし
(`TRAINING_COMPLETE`マーカー方式、YOLO)、または既存checkpointから
再開する(refiner)ため、GPU予算をゼロから使い切らなくても実行できる。

## 使い方

    python scratch/build_master_pipeline_notebook.py

生成物: solar_filament_master_pipeline_colab.ipynb
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_master_pipeline_colab.ipynb"
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")

assert "class MediSegEdgeAttUNet" in MODELS, "MediSegEdgeAttUNet が埋め込みソースに見つからない"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# Solar Filament Segmentation Challenge 2026 — 統合パイプライン

このnotebookは提出物(V52: fold2単体YOLO + EdgeAttNet refiner、Public LB
0.39)を再現するための**パイプライン全体**を示す: (A) 前処理・fold分割、
(B) YOLO11-seg検出器の3-fold学習、(C) EdgeAttNet風refinerの3-fold学習
+ 全データfine-tune、(D) test推論・提出物生成、(E) 公式RLE厳密検証。

チェックポイントが既にDrive上にあれば各学習ステージは自動的にスキップ
される(ゼロから全部再学習する必要はない)。GPU必須。A100を推奨するが
T4でも動く(学習時間が伸びるのみ)。詳細な設計根拠・実験の経緯は
リポジトリの`README.md`・`report/main.tex`を参照。

fold分割は既存の `artifacts/grouped_folds.csv` と、外部で凍結した同ファイルの
SHA256を `CFG.GROUPED_FOLDS_SHA256` に設定して読み込む。未設定・不整合なら
前処理前に停止し、新しい分割は生成しない。ハッシュの一致は公式正本や
時間・観測所を跨ぐ独立性の証明ではない。
"""
    ),
    markdown("## (A) セットアップ・データ読み込み・fold分割"),
    code(
        """# 依存関係・Drive接続
%pip install -q "ultralytics>=8.3,<9" pycocotools

import json, math, os, random, shutil, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from google.colab import drive
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from ultralytics import YOLO

drive.mount("/content/drive")

class CFG:
    SEED = 3407
    N_FOLDS = 3
    GROUPED_FOLDS_SHA256 = ""  # 外部で凍結した grouped_folds.csv の SHA256 が必須
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs"
    YOLO_ROOT = DRIVE_ROOT / "yolo_datasets"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_master")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    # --- YOLO検出器(build_v30_prize_colab.pyと同一) ---
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    # --- refiner(build_v35_refiner_folds12.py / build_v51_edgeattnet_refiner.pyと同一) ---
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_EPOCHS = 14
    REFINER_ALLDATA_EPOCHS = 5
    REFINER_LR = 2e-4
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    # --- 確定後処理(README.md「現在地」節、V52提出構成) ---
    SUBMISSION_CONF = 0.26
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    PREDICT_IOU = 0.55
    SINGLE_FOLD = 2  # V50/V52で確認済み: OOF PQ最良、アンサンブルより計算コスト1/3で同等以上

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.ARTIFACT_ROOT, CFG.YOLO_ROOT):
    path.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)

assert CFG.DATA_CACHE.exists(), f"競技データが見つからない: {CFG.DATA_CACHE}"
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({"GPU": torch.cuda.get_device_name(0)})
"""
    ),
    code(
        """# COCO注釈読み込み、凍結済みfold検証、前処理画像展開

def load_frozen_folds(path, train_files, n_folds, expected_sha256):
    import csv
    import hashlib
    import io
    import re

    if type(n_folds) is not int or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    if (not isinstance(train_files, list) or not train_files
            or any(not isinstance(name, str) or not name or name in (".", "..")
                   or name.strip() != name or any(c in name for c in "/\\\\:")
                   or any(ord(c) < 32 or ord(c) == 127 for c in name) for name in train_files)
            or len(set(train_files)) != len(train_files)):
        raise ValueError("Expected observations must be unique safe basenames")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Provide the externally frozen grouped_folds.csv SHA256")
    payload = Path(path).read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("Frozen fold CSV SHA256 differs")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    fields = reader.fieldnames
    if (not fields or len(fields) != len(set(fields))
            or not {"file_name", "fold"}.issubset(fields)):
        raise ValueError("Fold CSV needs unique headers including file_name and fold")
    expected, fold_of_file = set(train_files), {}
    try:
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed fold CSV row")
            name, value = row["file_name"], row["fold"]
            if name not in expected or name in fold_of_file:
                raise ValueError("Duplicate or unexpected fold observation")
            if not re.fullmatch(r"0|[1-9][0-9]*", value):
                raise ValueError("Fold must be an exact nonnegative integer")
            fold = int(value)
            if fold >= n_folds:
                raise ValueError("Fold is outside configured range")
            fold_of_file[name] = fold
    except csv.Error as error:
        raise ValueError("Malformed fold CSV") from error
    if set(fold_of_file) != expected or set(fold_of_file.values()) != set(range(n_folds)):
        raise ValueError("Fold CSV must cover every observation and every configured fold")
    return fold_of_file, actual_sha256

def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"annotation JSONが見つからない: {base}")
    return matches[0].parents[2]

DATA_ROOT = find_dataset_root(CFG.DATA_CACHE)
TRAIN_JSON = next(DATA_ROOT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

image_ids = [item["id"] for item in coco["images"]]
if (any(not (type(image_id) is int or
             (isinstance(image_id, str) and image_id.strip())) for image_id in image_ids)
        or len(set(image_ids)) != len(image_ids)):
    raise ValueError("COCO image IDs must be unique integers or nonempty strings; labeler filenames may repeat")
images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)

grouped_folds_path = CFG.ARTIFACT_ROOT / "grouped_folds.csv"
fold_of_file, frozen_fold_sha256 = load_frozen_folds(
    grouped_folds_path, train_files, CFG.N_FOLDS, CFG.GROUPED_FOLDS_SHA256)
print({"frozen_fold_csv": str(grouped_folds_path), "sha256": frozen_fold_sha256})

for name in tqdm(train_files, desc="前処理画像をローカルへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)

print({"観測": len(train_files),
       "fold別": {f: sum(v == f for v in fold_of_file.values()) for f in (0, 1, 2)},
       "annotation": len(coco["annotations"])})
"""
    ),
    markdown(
        """## (B) YOLO11-seg検出器の3-fold学習

各foldにつき、そのfoldを検証専用に除外し、残り2foldの**全アノテータ
分の独立ラベル**(同一画像でもアノテータごとに別サンプルとして学習)で
学習する。これによりリークなしOOF評価が可能になる。学習済みfoldは
`TRAINING_COMPLETE`マーカーで検出しスキップする。"""
    ),
    code(
        """# YOLOデータセット構築(all-labeler方式)とtrain関数
def polygon_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    polygon = max(polygons, key=len)  # MAGFiLOは1個体1polygon。複数返る不正データへの安全策
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.reshape(-1))

def write_variant(fold_root, split, file_name, image_id, suffix):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{stem}.jpeg"
    label_out = fold_root / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    if not image_out.exists():
        shutil.copy2((CFG.PREPROC_DIR / file_name).resolve(), image_out)
    lines = [polygon_line(ann, info["width"], info["height"])
             for ann in annotations[image_id] if not ann.get("iscrowd", 0)]
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(line for line in lines if line), encoding="utf-8")

def build_fold_dataset(fold):
    root = CFG.YOLO_ROOT / f"fold_{fold}"
    reference_id = {name: file_to_ids[name][0] for name in train_files}
    for file_name in tqdm(train_files, desc=f"fold {fold} データセット構築"):
        if fold_of_file[file_name] == fold:
            write_variant(root, "val", file_name, reference_id[file_name], "")
        else:
            for image_id in file_to_ids[file_name]:
                write_variant(root, "train", file_name, image_id, f"__labeler_{image_id}")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
                          encoding="utf-8")
    return yaml_path

def train_yolo_fold(fold):
    run_dir = CFG.RUNS_ROOT / f"yolo_fold_{fold}"
    best_path = run_dir / "weights" / "best.pt"
    done_path = run_dir / "TRAINING_COMPLETE"
    if done_path.exists() and best_path.exists():
        print(f"fold {fold}: 学習済みcheckpointを再利用 {best_path}")
        return best_path
    yaml_path = build_fold_dataset(fold)
    model = YOLO(CFG.YOLO_MODEL)
    model.train(
        data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
        batch=CFG.YOLO_BATCH, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED + fold, deterministic=True, single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True,
        cache="disk",
        # 査読(2026-08-31)で指摘: これらは "Ultralytics defaults" ではなく
        # build_v30_prize_colab.pyで明示的に設定した値。省略すると別ハイパラで
        # 再学習することになり再現性が壊れるため、必ず全部渡すこと。
        optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(CFG.RUNS_ROOT), name=f"yolo_fold_{fold}", exist_ok=True,
    )
    done_path.write_text("ok", encoding="utf-8")
    return best_path

yolo_weight_paths = {fold: train_yolo_fold(fold) for fold in (0, 1, 2)}
print({fold: str(path) for fold, path in yolo_weight_paths.items()})
"""
    ),
    markdown(
        """## (C) EdgeAttNet風refiner(EG-MHSA)の学習

境界refinerのボトルネックへEdge-Guided Multi-Head Self-Attentionを
追加した`MediSegEdgeAttUNet`(arXiv:2509.02964の独自再実装、著者コードは
未検証)を、各foldで(検証専用に除外して)学習し、続けてfold0の
checkpointをウォームスタートして全707画像へ低LR・短epochで
fine-tuneする(提出モデル用)。"""
    ),
    code("# MediSegUNet / EdgeGuidedMHSA / MediSegEdgeAttUNet 定義\n" + MODELS),
    code(
        """# 形状サニティチェック
_probe = MediSegEdgeAttUNet(3, 1, False).cuda()
_dummy = torch.randn(2, 3, CFG.CROP_SIZE, CFG.CROP_SIZE).cuda()
with torch.no_grad():
    _out = _probe(_dummy)
assert _out.shape == (2, 1, CFG.CROP_SIZE, CFG.CROP_SIZE), "出力形状が想定と異なる"
print({"output_shape": tuple(_out.shape),
       "params": sum(p.numel() for p in _probe.parameters())})
del _probe, _dummy, _out
torch.cuda.empty_cache()
"""
    ),
    code(
        """# Dataset・loss・学習ループ定義(fold引数で汎用化)
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

def ann_mask(image_id, ann):
    info = images[image_id]
    height, width = int(info["height"]), int(info["width"])
    segmentation = ann["segmentation"]
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(np.uint8)

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

def train_refiner_fold(fold, epochs=CFG.REFINER_EPOCHS):
    checkpoint_path = CFG.ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{fold}_resume.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if state.get("epoch", -1) + 1 >= epochs:
            print(f"fold {fold}: 学習済みcheckpointを再利用 {checkpoint_path}")
            return checkpoint_path

    train_files_fold = [n for n in train_files if fold_of_file[n] != fold]
    valid_files_fold = [n for n in train_files if fold_of_file[n] == fold]
    train_ds = InstanceCropDataset(train_files_fold, augment=True)
    valid_ds = InstanceCropDataset(valid_files_fold, augment=False)
    batch_size = choose_batch_size(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MediSegEdgeAttUNet(3, 1, False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    best_dice, best_state = -1.0, None
    for epoch in range(epochs):
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
            best_dice, best_state = dice, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"epoch": epoch, "model": model.state_dict(), "best_model": best_state,
                    "best_dice": best_dice}, checkpoint_path)
        print(f"fold{fold} epoch {epoch + 1}: crop Dice {dice:.6f} (best {best_dice:.6f})")
    return checkpoint_path

for fold in (0, 1, 2):
    train_refiner_fold(fold)
"""
    ),
    code(
        """# all-data fine-tune(fold0からウォームスタート、低LR・短epoch、提出モデル用)
ALLDATA_CKPT = CFG.ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_all_data_resume.pt"
if ALLDATA_CKPT.exists():
    print(f"学習済みcheckpointを再利用 {ALLDATA_CKPT}")
else:
    all_ds = InstanceCropDataset(train_files, augment=True)
    batch_size = choose_batch_size(all_ds)
    loader = DataLoader(all_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    warm_start_path = CFG.ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold0_resume.pt"
    model = MediSegEdgeAttUNet(3, 1, False).cuda()
    warm_state = torch.load(warm_start_path, map_location="cpu")
    model.load_state_dict(warm_state.get("best_model") or warm_state["model"], strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR * 0.35, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.REFINER_ALLDATA_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(CFG.REFINER_ALLDATA_EPOCHS):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"all-data refiner {epoch + 1}/{CFG.REFINER_ALLDATA_EPOCHS}")
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
                "warm_start_from": str(warm_start_path)}, ALLDATA_CKPT)
    print(f"saved: {ALLDATA_CKPT}")
"""
    ),
    markdown(
        """## (D) test推論・提出物生成

提出構成(V52): fold2単体YOLO(アンサンブルなし、V50/V52で3-foldアンサンブルより
同等以上と確認済み) + all-data EdgeAttNet refiner + shrink@0.45 +
drop_contained_then_area_asc。"""
    ),
    code(
        """# test画像一覧、単一モデル推論、refiner適用、排他化、submission CSV生成
def find_test_dir():
    return next(DATA_ROOT.rglob("test_images"))

TEST_DIR = find_test_dir()
test_files = sorted(p.name for p in TEST_DIR.glob("*.jpeg"))
assert len(test_files) == 180, f"test画像が180枚ではありません: {len(test_files)}"
for file_name in tqdm(test_files, desc="test画像をローカルへ"):
    local = CFG.PREPROC_DIR / file_name
    if not local.exists():
        cached = CFG.PREPROC_CACHE / file_name
        shutil.copy2(cached, local) if cached.exists() else shutil.copy2(TEST_DIR / file_name, local)

yolo_model = YOLO(str(yolo_weight_paths[CFG.SINGLE_FOLD]))
refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
refiner_ckpt = torch.load(ALLDATA_CKPT, map_location="cuda", weights_only=False)
refiner.load_state_dict(refiner_ckpt.get("best_model") or refiner_ckpt["model"], strict=True)

def result_masks(result):
    if result.masks is None:
        return []
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    return output

def single_fold_proposals(image_path):
    result = yolo_model.predict(str(image_path), imgsz=CFG.YOLO_IMGSZ,
                                 conf=max(0.08, CFG.SUBMISSION_CONF * 0.75), iou=CFG.PREDICT_IOU,
                                 agnostic_nms=True, retina_masks=True, verbose=False)[0]
    masks = result_masks(result)
    confidences = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes is not None else []
    return [(s, m) for s, m in zip(confidences, masks) if s >= CFG.SUBMISSION_CONF]

@torch.no_grad()
def refine_proposal(image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal

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

rows, audit = [], []
for file_name in tqdm(test_files, desc="test推論"):
    image_path = CFG.PREPROC_DIR / file_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    proposals = single_fold_proposals(image_path)
    scores = np.array([c for c, _ in proposals], dtype=np.float32)
    refined = [refine_proposal(image, m) for _, m in proposals]
    final_masks = apply_ownership(refined, scores)
    image_id = Path(file_name).stem
    for index, mask in enumerate(final_masks, start=1):
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rows.append({"filament_id": f"{image_id}_{index}",
                     "segmentation_rle": encoded["counts"].decode("utf-8")})
    audit.append({"file_name": file_name, "proposals": len(proposals), "submitted": len(final_masks)})

submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
OUTPUT_PATH = Path("/content/submission.csv")
submission.to_csv(OUTPUT_PATH, index=False, lineterminator="\\n")
shutil.copy2(OUTPUT_PATH, CFG.ARTIFACT_ROOT / "submission_master_pipeline.csv")
print({"rows": len(submission), "images_with_predictions":
       len(set(a["file_name"] for a in audit if a["submitted"] > 0))})
"""
    ),
    markdown("## (E) 公式RLE厳密検証"),
    code(
        """# 提出前チェックリスト: 列・重複・空値・重畳ピクセルゼロを確認
checked = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
assert checked.columns.tolist() == ["filament_id", "segmentation_rle"]
assert not checked.isna().any().any()
assert not checked["filament_id"].duplicated().any()
assert not checked["segmentation_rle"].str.strip().eq("").any()

test_ids = {Path(name).stem for name in test_files}
submitted_ids = {value.rsplit("_", 1)[0] for value in checked["filament_id"]}
assert submitted_ids <= test_ids

occupancy = {}
for row in tqdm(checked.itertuples(index=False), total=len(checked), desc="decode + overlap検証"):
    decoded = mask_utils.decode({"size": [2048, 2048], "counts": row.segmentation_rle.encode("utf-8")})
    assert decoded.shape == (2048, 2048) and decoded.sum() > 0
    image_id = row.filament_id.rsplit("_", 1)[0]
    occ = occupancy.setdefault(image_id, np.zeros((2048, 2048), dtype=bool))
    assert not np.logical_and(occ, decoded.astype(bool)).any(), f"重複: {image_id}"
    occ |= decoded.astype(bool)

print({"rows": len(checked), "images": len(submitted_ids), "overlap_pixels": 0,
       "status": "全検証パス。submission.csv は提出可能な状態"})
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"master-{index:02d}"

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

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {OUTPUT} ({len(cells)} cells)")
