"""距離変換 boundary loss パイロット notebook builder — WINNING_STRATEGY_2026-08-29.md §6.12優先度4。

## 目的

現行 `refiner_loss` の境界項は morphological-gradient(dilate−erode 5px)の
L1 で、V30 から不変の粗い定式化(README.md §2.1.12)。これを Kervadec らの
距離変換ベース境界損失(mean(sigmoid·φ_G)、φ_G は GT の正規化済み符号付き
距離写像)に置き換え、IoU>0.5 崖の近傍で TP 化できる境界精度が上がるかを
検証する。

v57(§6.16)で「検出器リコールを上げてもFPが食う」と確定したため、評価側
(提案品質・境界精度)のレバーに進む本パイロット。refiner側の変更なので
検出器seed差ノイズ(σ≫0.001)を持ち込まず、refiner学習のseed差(σ≈0.001)
だけで判定できる。

## 設計

`build_v54_colab_ema_pilot.py` と同一構造。アーキテクチャは `MediSegUNet`
(標準refiner)に固定、fold{0,1,2}で2構成 × seed(3407, 8407) = 4本:

  1. `baseline` — 現行 `refiner_loss` 完全同一(0.35·BCE + 0.65·Dice + 0.15·L1(morph-grad))
  2. `dtb`      — region項は同一、境界項だけを `0.15 · mean(sigmoid·φ_G)` に置換
                  (φ_G は augmented target から batch 内で計算。重みは既存
                  edge項の0.15をそのまま引き継ぐ単変数置換)

旧 v54 baseline は既定では再利用しない。明示 opt-in と完全な入力・checkpoint 契約一致が必要。
由来を確認できなければ baseline も新しい paired pilot として学習する。
評価経路もv54と同一(既存 `yolo_fold_{N}` ckpt の固定proposals → refiner →
shrink@0.45 → ownership → OOF PQ)。

## 使い方

    python scratch/build_v59_boundary_dt_pilot.py --fold 0

出力: solar_filament_v59_boundary_dt_fold{N}_colab.ipynb (Colab A100/L4推奨)
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")
BOUNDARY_DT = (ROOT / "solar_filament" / "pipeline" / "boundary_dt.py").read_text(encoding="utf-8")

# matched-parameter ablation実測、base/2seed平均(README.md §2.1.10)
BASELINE_PQ = {0: 0.4414, 1: 0.4405, 2: 0.4434}
BDT_WEIGHT = 0.15  # 既存edge項の重みをそのまま引き継ぐ
SEEDS = (3407, 8407)
CONFIGS = ("baseline", "dtb")
PROTOCOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n") + MODELS.replace("\r\n", "\n").encode() + BOUNDARY_DT.replace("\r\n", "\n").encode()).hexdigest()
GUARDS = '\ndef atomic_save(path, value, is_checkpoint=False):\n    import os\n    import tempfile\n    path = Path(path)\n    payload = None if is_checkpoint else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")\n    temporary = None\n    try:\n        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as handle:\n            temporary = Path(handle.name)\n            if is_checkpoint:\n                torch.save(value, handle)\n            else:\n                handle.write(payload)\n            handle.flush()\n            os.fsync(handle.fileno())\n        os.replace(temporary, path)\n    finally:\n        if temporary is not None and temporary.exists():\n            temporary.unlink()\n\ndef validate_paired_rows(rows):\n    pairs = {}\n    for row in rows:\n        pairs.setdefault(row["seed"], []).append(row)\n    for arms in pairs.values():\n        if len({r["batch_size"] for r in arms}) != 1 or len({r["initial_state_sha256"] for r in arms}) != 1:\n            raise ValueError("Paired arms must share batch size and actual initial state")\n\n\ndef file_sha256(path):\n    import hashlib\n    digest = hashlib.sha256()\n    with Path(path).open("rb") as handle:\n        for chunk in iter(lambda: handle.read(1024 * 1024), b""):\n            digest.update(chunk)\n    return digest.hexdigest()\n\ndef validate_checkpoint(checkpoint, config, seed):\n    if (not isinstance(checkpoint, dict) or checkpoint.get("training_status") != "complete"\n            or checkpoint.get("input_contract") != PILOT_CONTRACT\n            or checkpoint.get("config") != config or checkpoint.get("seed") != seed\n            or checkpoint.get("fold") != CFG.FOLD or not checkpoint.get("best_model")\n            or not checkpoint.get("training_summary")):\n        raise ValueError("Checkpoint is incomplete or differs from this pilot; retain it and use a fresh path")\n    initial = checkpoint.get("initial_state_sha256")\n    if not isinstance(initial, str) or len(initial) != 64 or any(c not in "0123456789abcdef" for c in initial):\n        raise ValueError("Checkpoint has no actual initial-state fingerprint")\n    return checkpoint\n\ndef validate_run_row(row):\n    if (not isinstance(row, dict) or row.get("config") not in CFG.CONFIGS\n            or row.get("seed") not in CFG.SEEDS or row.get("fold") != CFG.FOLD\n            or row.get("input_contract") != PILOT_CONTRACT\n            or row.get("checkpoint_selection") != "best_model"):\n        raise ValueError("Run fold/arm/input contract does not match this pilot")\n    for metric in ("pq", "sq", "rq"):\n        value = row.get(metric)\n        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:\n            raise ValueError("Run has missing or invalid quality metrics")\n    for metric in ("tp", "fp", "fn"):\n        if type(row.get(metric)) is not int or row[metric] < 0:\n            raise ValueError("Run has missing or invalid match counts")\n    for field in ("checkpoint_sha256", "initial_state_sha256"):\n        value = row.get(field)\n        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):\n            raise ValueError("Run is missing checkpoint/initial-state fingerprints")\n    checkpoint = Path(row.get("checkpoint", ""))\n    if not checkpoint.is_file() or file_sha256(checkpoint) != row["checkpoint_sha256"]:\n        raise ValueError("Recorded checkpoint is absent or its bytes changed")\n    if type(row.get("batch_size")) is not int or row["batch_size"] <= 0:\n        raise ValueError("Run has no valid training batch size")\n    return row\n\ndef read_pilot_manifest():\n    if not CFG.MANIFEST.exists():\n        return {"experiment_id": f"v59_boundary_dt_fold{CFG.FOLD}", "input_contract": PILOT_CONTRACT, "runs": []}\n    manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8"))\n    if (not isinstance(manifest, dict) or manifest.get("input_contract") != PILOT_CONTRACT\n            or manifest.get("experiment_id") != f"v59_boundary_dt_fold{CFG.FOLD}"\n            or not isinstance(manifest.get("runs"), list)):\n        raise ValueError("Manifest contract differs or is absent; retain it and use a fresh path")\n    seen = set()\n    for row in manifest["runs"]:\n        validate_run_row(row)\n        key = (row["config"], row["seed"])\n        if key in seen:\n            raise ValueError("Duplicate pilot run")\n        seen.add(key)\n    validate_paired_rows(manifest["runs"])\n    return manifest\n\ndef reuse_baselines(source):\n    if not CFG.REUSE_V54_BASELINE:\n        return []\n    if not isinstance(source, dict) or source.get("input_contract") != PILOT_CONTRACT:\n        raise ValueError("Baseline equivalence is unverified; rerun the paired baseline")\n    result, seen = [], set()\n    for row in source.get("runs", []):\n        if row.get("config") != "baseline":\n            continue\n        validate_run_row(row)\n        key = (row["config"], row["seed"])\n        if key in seen:\n            raise ValueError("Duplicate reused baseline")\n        seen.add(key)\n        result.append({**row, "reused_from": "explicit_contract_checked_baseline"})\n    validate_paired_rows(result)\n    return result\n\nPILOT_CONTRACT = {\n    "schema_version": 1, "fold": CFG.FOLD, "protocol_sha256": CFG.PROTOCOL_SHA256,\n    "annotations_sha256": file_sha256(TRAIN_JSON),\n    "folds_sha256": file_sha256(CFG.ARTIFACT_ROOT / "grouped_folds.csv"),\n    "detector_sha256": file_sha256(CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{CFG.FOLD}" / "weights/best.pt"),\n    "preprocessed_sha256": hashlib.sha256(json.dumps(preprocessed_hashes, sort_keys=True).encode()).hexdigest(),\n    "recipe": {name: getattr(CFG, name) for name in ("CROP_SIZE", "CROP_CONTEXT", "REFINER_EPOCHS", "REFINER_LR",\n        "REFINER_BATCH_CANDIDATES", "CONF", "REFINER_THRESHOLD", "DROP_THRESHOLD", "MIN_AREA", "BDT_WEIGHT")},\n    "seeds": list(CFG.SEEDS), "configs": list(CFG.CONFIGS),\n    "runtime_versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__, "ultralytics": version("ultralytics"), "pycocotools": version("pycocotools")},\n}\nPILOT_CONTRACT = json.loads(json.dumps(PILOT_CONTRACT, sort_keys=True, allow_nan=False))\nread_pilot_manifest()  # Reject old/mixed results before model loading or historical writes.\n'


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(fold):
    baseline_pq = BASELINE_PQ[fold]

    cells = [
        markdown(
            f"""# 距離変換 boundary loss パイロット — fold{fold}、baseアーキテクチャ固定

WINNING_STRATEGY_2026-08-29.md §6.12優先度4。region損失(0.35·BCE + 0.65·Dice)
は不変で、**境界項だけ**を morphological-gradient L1 から距離変換ベース
(`mean(sigmoid·φ_G)`、Kervadec系)に置き換える。φ_G は GT の符号付き距離写像
(内部負・外部正の離散距離近似、サンプルごとに±1正規化)。

| 構成 | 内容 |
|---|---|
| `baseline` | 現行 `refiner_loss` 完全同一(V30から不変) |
| `dtb` | 境界項だけ `0.15·mean(sigmoid·φ_G)` に置換(重み0.15はedge項から引継ぎ) |

fold{fold}基準(標準refiner、matched-parameter ablation実測、2seed平均): {baseline_pq}。
旧 baseline 再利用は既定 OFF。入力契約・fold・実 checkpoint の一致が確認できなければ
baseline も未実行として残す。これは CPU で構造検証した pilot であり、新しい GPU 受入結果ではない。
"""
        ),
        code(
            """# [1/8] Drive接続、設定、前提の確認
!pip -q install ultralytics pycocotools
import hashlib, json, math, os, random, shutil, sys, time
from datetime import datetime, timezone
from importlib.metadata import version
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
    LOCAL_ROOT = Path("/content/solar_v59_bdt")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    # --- build_v51_edgeattnet_refiner.py / build_matched_param_ablation.py / v54 と同一。触らないこと ---
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_EPOCHS = 14
    REFINER_LR = 2e-4
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    FOLD = %d
    PROTOCOL_SHA256 = %r
    REUSE_V54_BASELINE = False  # Legacy v54 has no equivalence proof; baseline remains pending.
    CONF = 0.26
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ = %r
    BDT_WEIGHT = %r
    SEEDS = %r
    CONFIGS = %r  # ("baseline", "dtb")
    V54_MANIFEST = DRIVE_ROOT / "artifacts" / "v54_ema_pilot_fold%d_manifest.json"
    MANIFEST = DRIVE_ROOT / "artifacts" / "v59_boundary_dt_fold%d_manifest.json"

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
"""  # noqa: UP031 - preserve braces in embedded Python source
            % (fold, PROTOCOL_SHA256, baseline_pq, BDT_WEIGHT, SEEDS, CONFIGS, fold, fold)
        ),
        code(
            "# [2/8] boundary_dt (solar_filament/pipeline/boundary_dt.py と同一、テスト済み)\n"
            + BOUNDARY_DT
        ),
        code(
            """# [3/8] COCOとfoldを読み、前処理画像をローカルSSDへ展開
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
if fold_frame["file_name"].duplicated().any() or not fold_frame["fold"].isin([0, 1, 2]).all():
    raise ValueError("Grouped folds must have unique files and integer folds 0/1/2")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
assert all(name in fold_of_file for name in train_files), "fold未割当の観測がある"

preprocessed_hashes = {}
for name in tqdm(train_files, desc="前処理画像をSSDへ・SHA256"):
    local = CFG.PREPROC_DIR / name
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG.PREPROC_CACHE / name, local)
    preprocessed_hashes[name] = hashlib.sha256(local.read_bytes()).hexdigest()

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
if not train_files_fold or not valid_files_fold:
    raise ValueError("Both training and validation observations are required")
""" + GUARDS
        ),
        code(
            "# [4/8] MediSegUNet (solar_filament/models/mediseg_models.py と同一。base構成のみ使用)\n"
            + MODELS
        ),
        code(
            """# [4.5/8] 形状サニティチェック
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
            """# [5/8] Dataset・loss・学習/評価ループ
# baselineのrefiner_lossはV30から不変。dtb構成は境界項だけを距離変換版に置換する
# (region項・重み0.15・学習・評価ループは同一 = 単変数置換)
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

def refiner_loss_dtb(logits, targets):
    \"\"\"region項はrefiner_lossと同一。境界項を距離変換版 mean(sigmoid·φ_G) に置換。
    φ_G はaugment適用後のtargetから計算する(augmentと同期するためloss内で計算)。\"\"\"
    region = 0.35 * F.binary_cross_entropy_with_logits(logits, targets) + 0.65 * soft_dice_loss(logits, targets)
    phi_np = batch_signed_distance_maps(targets.detach().cpu().numpy())
    phi = torch.from_numpy(phi_np).to(logits.device)
    boundary = (logits.sigmoid()[:, 0] * phi).mean()
    return region + CFG.BDT_WEIGHT * boundary

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

def probe_batch_size(dataset, candidate, loss_fn):
    loader = DataLoader(dataset, batch_size=candidate, shuffle=True, num_workers=2, pin_memory=True)
    image = target = probe = loss = None
    try:
        image, target = next(iter(loader))
        probe = MediSegUNet(3, 1, False).cuda()
        with torch.amp.autocast("cuda"):
            loss = loss_fn(probe(image.cuda()), target.cuda())
        loss.backward()
    finally:
        del probe, image, target, loss, loader
        torch.cuda.empty_cache()

def choose_batch_size(dataset):
    for candidate in CFG.REFINER_BATCH_CANDIDATES:
        try:
            for loss_fn in (refiner_loss, refiner_loss_dtb):
                probe_batch_size(dataset, candidate, loss_fn)
            return candidate
        except torch.cuda.OutOfMemoryError:
            pass
    raise RuntimeError("No common batch size fits both paired losses")

def train_one_run(config, seed):
    checkpoint_path = CFG.ARTIFACT_ROOT / f"v59_bdt_pilot_{config}_seed{seed}_fold{CFG.FOLD}.pt"
    if checkpoint_path.exists():
        checkpoint = validate_checkpoint(torch.load(checkpoint_path, map_location="cpu", weights_only=False), config, seed)
        return {"checkpoint": str(checkpoint_path), **checkpoint["training_summary"]}
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    loss_fn = refiner_loss_dtb if config == "dtb" else refiner_loss
    train_ds = InstanceCropDataset(train_files_fold, augment=True)
    valid_ds = InstanceCropDataset(valid_files_fold, augment=False)
    batch_size = choose_batch_size(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = MediSegUNet(3, 1, False).cuda()
    initial = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        initial.update(name.encode()); initial.update(str(value.dtype).encode()); initial.update(str(tuple(value.shape)).encode())
        initial.update(value.detach().cpu().contiguous().numpy().tobytes())
    initial_state_sha256 = initial.hexdigest()
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
                loss = loss_fn(model(image), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{running / (progress.n + 1):.4f}")
        scheduler.step()

        dice = crop_dice_from_state(model.state_dict(), valid_loader)
        if dice > best_dice:
            best_dice = dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{config}/seed{seed} epoch {epoch + 1}: crop Dice {dice:.6f} (best {best_dice:.6f})")

    minutes = round((time.time() - started) / 60, 1)
    summary = {"best_crop_dice": best_dice, "minutes": minutes, "batch_size": batch_size,
               "training_minutes_scope": "original_complete_training_invocation"}
    atomic_save(checkpoint_path, {"model": model.state_dict(), "best_model": best_state, "best_dice": best_dice,
                "config": config, "seed": seed, "fold": CFG.FOLD, "input_contract": PILOT_CONTRACT,
                "training_status": "complete", "initial_state_sha256": initial_state_sha256,
                "training_summary": summary}, is_checkpoint=True)
    del model, optimizer, scheduler, train_loader, valid_loader, train_ds, valid_ds
    torch.cuda.empty_cache()
    return {"checkpoint": str(checkpoint_path), **summary}
"""
        ),
        code(
            """# [6/8] OOF採点用の後処理(V54/V51/matched-parameter ablationと同一: shrink@0.45 + drop_contained@0.90)
from ultralytics import YOLO
from pycocotools import mask as mask_utils

detector_path = CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{CFG.FOLD}" / "weights" / "best.pt"
if file_sha256(detector_path) != PILOT_CONTRACT["detector_sha256"]:
    raise ValueError("Detector bytes changed before loading")
yolo_model = YOLO(str(detector_path))

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
            """# [7/8] Serial paired runs; legacy baselines are not implicitly relabelled.
manifest = read_pilot_manifest()
rows = manifest["runs"]
done_keys = {(r["config"], r["seed"]) for r in rows}
if CFG.REUSE_V54_BASELINE and CFG.V54_MANIFEST.exists():
    for row in reuse_baselines(json.loads(CFG.V54_MANIFEST.read_text(encoding="utf-8"))):
        if (row["config"], row["seed"]) not in done_keys:
            rows.append(row)
            done_keys.add((row["config"], row["seed"]))
manifest.pop("failure", None)
manifest["status"] = "in_progress"
try:
    for config in CFG.CONFIGS:
        for seed in CFG.SEEDS:
            if (config, seed) in done_keys:
                continue
            checkpoint_path = CFG.ARTIFACT_ROOT / f"v59_bdt_pilot_{config}_seed{seed}_fold{CFG.FOLD}.pt"
            scope = "evaluation_only" if checkpoint_path.exists() else "training_and_evaluation"
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            train_summary = train_one_run(config, seed)
            ckpt = validate_checkpoint(torch.load(train_summary["checkpoint"], map_location="cpu", weights_only=False), config, seed)
            refiner = MediSegUNet(3, 1, False).cuda().eval()
            refiner.load_state_dict(ckpt["best_model"], strict=True)
            if file_sha256(detector_path) != PILOT_CONTRACT["detector_sha256"]:
                raise ValueError("Detector bytes changed after input-contract construction")
            oof = evaluate_oof(refiner)
            row = {"config": config, "seed": seed, "fold": CFG.FOLD,
                   "bdt_weight": CFG.BDT_WEIGHT if config == "dtb" else None,
                   **train_summary, **oof, "input_contract": PILOT_CONTRACT,
                   "checkpoint_sha256": file_sha256(train_summary["checkpoint"]),
                   "initial_state_sha256": ckpt["initial_state_sha256"], "checkpoint_selection": "best_model",
                   "execution": {"service": "Google Colab", "project": str(CFG.DRIVE_ROOT),
                       "notebook": f"solar_filament_v59_boundary_dt_fold{CFG.FOLD}_colab.ipynb",
                       "gpu_name": torch.cuda.get_device_name(0),
                       "measurement_scope": scope, "duration_seconds": round(time.time() - started, 3),
                       "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                       "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "compute_units_used": None, "compute_units_note": "Unavailable from runtime API; record from Colab UI."}}
            validate_run_row(row)
            validate_paired_rows(rows + [row])
            rows.append(row)
            done_keys.add((config, seed))
            manifest["runs"] = rows
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_save(CFG.MANIFEST, manifest)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False))
            del refiner, ckpt
            torch.cuda.empty_cache()
except (Exception, KeyboardInterrupt) as error:
    manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    try:
        failure_peak = int(torch.cuda.max_memory_allocated())
        failure_note = None
    except Exception as stats_error:
        failure_peak = None
        failure_note = f"CUDA memory statistics unavailable: {type(stats_error).__name__}"
    manifest["failure"] = {"type": type(error).__name__, "at_utc": datetime.now(timezone.utc).isoformat(),
        "service": "Google Colab", "project": str(CFG.DRIVE_ROOT),
        "notebook": f"solar_filament_v59_boundary_dt_fold{CFG.FOLD}_colab.ipynb",
        "gpu_name": torch.cuda.get_device_name(0), "measurement_scope": scope,
        "duration_seconds": round(time.time() - started, 3), "peak_vram_bytes": failure_peak,
        "peak_vram_unavailable_reason": failure_note, "compute_units_used": None,
        "compute_units_note": "Unavailable from runtime API; record from Colab UI."}
    atomic_save(CFG.MANIFEST, manifest)
    raise
manifest["status"] = "complete"
atomic_save(CFG.MANIFEST, manifest)

# --- 集計と採否の目安 ---
import statistics
pair = {}
for row in rows:
    pair.setdefault(row["seed"], {})[row["config"]] = row
print("\\n=== seed別 paired Δ(dtb - baseline) ===")
deltas = []
for seed, arms in sorted(pair.items()):
    if "baseline" in arms and "dtb" in arms:
        delta = arms["dtb"]["pq"] - arms["baseline"]["pq"]
        deltas.append(delta)
        print(f"seed {seed}: ΔPQ {delta:+.6f} (dtb {arms['dtb']['pq']:.6f} vs baseline {arms['baseline']['pq']:.6f})")
if deltas:
    print(f"\\nmean Δ {statistics.mean(deltas):+.6f} / 正のseed {sum(d > 0 for d in deltas)}/{len(deltas)}")
print("\\n採否の目安: 2seedで符号一致して初めてfold1・2へ独立確認。境界項の効果は"
      "SQ(分割質)に現れるはずなので、PQが動かなくてもSQの系統的改善があれば次の手を検討。")
"""
        ),
    ]
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["0", "1", "2"], help="評価対象fold")
    args = parser.parse_args()
    fold = int(args.fold)

    cells = build_cells(fold)
    for index, cell in enumerate(cells):
        cell["id"] = f"v59-bdt-f{fold}-{index:02d}"

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

    output = ROOT / f"solar_filament_v59_boundary_dt_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
