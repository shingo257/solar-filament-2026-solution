"""Build v129 Colab notebook: 320x320 Spine-Guided EdgeAttUNet Refiner A/B/C Experiment.

Implements Idea 5 (v129):
- Tests whether supervising human ``spine`` centerlines at full ``320x320`` crop resolution
  inside ``MediSegEdgeAttUNet`` (via a train-time-only ``1x1`` ``spine_head`` on decoder
  ``up4`` features and/or mass-preserving foreground spine BCE weights) improves native
  2048 PQ, SQ, and near-miss conversion while keeping test-time inference 100% identical
  to ``prod`` (vanilla ``MediSegEdgeAttUNet`` + strict ``shrink_mask`` inside ``proposal``).
- Strictly isolated paths (``/content/solar_v129``, ``runs/v129_spine_refiner``) so it can
  run concurrently alongside ``v127`` and ``v130`` without any file collisions.
- Total wall-clock on Colab L4: ~15-16 minutes (~4 min x 3 paired 14-epoch refiners + ~3 min OOF eval).

Usage:
    python scratch/build_v129_spine_refiner_colab.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _strip_module_for_inline(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^\s*from __future__ import annotations\s*\n", "", text, flags=re.MULTILINE)
    for mod in (
        r"solar_filament\.models\.mediseg_models",
        r"solar_filament\.pipeline\.spine_aux",
        r"solar_filament\.pipeline\.spine_yolo",
    ):
        text = re.sub(
            rf"^\s*from {mod} import (?:\([^)]*\)|[^\n]+)\n",
            "",
            text,
            flags=re.MULTILINE,
        )
    return text.rstrip() + "\n"


def _cell(cell_type: str, source: str) -> dict:
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def build_notebook() -> dict:
    md_intro = """# v129 — `320×320` Spine-Guided EdgeAttUNet Refiner 厳密対照実験 (Idea 5)

**Colab L4 約 15〜16 分で完結**（`v127` / `v130` と完全分離された保存先で同時並列実行可能）。

### なぜ `v129` が有望か（`v128` 失敗からの構造的フィードバック）
- `v128` により、推論時に `proposal` の外側へ踏み出す後処理（Hysteresis）や PCA 直線分割は隣接フィラメント混入・湾曲単一チャネル誤分割を招き、`shrink_mask` (`(prob >= 0.45) & proposal`) が推論契約として最適であることが確定した。
- 一方、`v127` は YOLO の粗い `160×160` P3 グリッド（`12.8 px`/cell）にのみ `spine` 補助教師を入れており、最終マスク境界を決定する **`320×320` 高解像度クロップ Refiner（`1〜2 px`/cell）には一度も `spine` 教師が使われていない**。
- 本実験では **推論アーキテクチャと `shrink_mask` (`thr=0.45`) を 100% 無変更に保ったまま**、同一初期重み・同一クロップ順序で以下 3 系統の `MediSegEdgeAttUNet` (各 14 epoch, 約 4 分) を厳密対照学習する:
  1. `ref_ctrl`: `aux_weight=0.0, spine_gamma=0.0`（現行 `prod` と完全同一損失の対照群）
  2. `ref_spine_aux`: `aux_weight=0.25, spine_gamma=0.0`（`up4` `320×320` 特徴への `1×1 spine_head` マルチタスク表現学習のみ）
  3. `ref_spine_bal`: `aux_weight=0.25, spine_gamma=1.5`（`spine_head` + 前景質量保存型の中心線強調 BCE `sum_{x in M} w(x) == |M|`）
"""

    cell1_cfg = """# [1/6] 依存パッケージ・Drive マウント・実験設定 (v127 / v130 と完全分離)
%pip install -q "ultralytics==8.4.120" pycocotools scipy
import hashlib, io, json, math, os, pickle, random, shutil, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO

drive.mount("/content/drive")

class CFG:
    SPLIT_MODE = "pilot"  # "pilot" (365 dev fit / 180 dev∩fold2 eval, locked_eval 非接触) or "temporal_confirm" (545 fit / 160 locked eval)
    SEED = 3407

    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    TIME_BLOCK_SHA256 = "9a9b17252133c9500d44bf3b6ec56879a47e4b32d45cb14be080ba29148c732d"
    EXCLUSIONS_SHA256 = "a56372ea6493a6d3571910e60f291606b864a4f84a61baf1df475a6627270b0a"

    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    EXP_DIR = DRIVE_ROOT / "runs" / "v129_spine_refiner"
    LOCAL_ROOT = Path("/content/solar_v129")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PROPOSAL_CACHE_DIR = LOCAL_ROOT / "oof_proposals"

    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    SPINE_RADIUS_PX = 2
    REFINER_EPOCHS = 14
    REFINER_BATCH = 8
    REFINER_LR = 2e-4
    REFINER_WD = 1e-4

    YOLO_IMGSZ = 1280
    PRIMARY_CONF = 0.26
    PREDICT_IOU = 0.55
    PRIMARY_THR = 0.45
    DIAG_THRS = (0.42, 0.45, 0.48)
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    SMALL_GT_MAX_AREA = 1024.0
    SPARSE_R = 0.77
    BOOTSTRAP_ITERS = 2000

    ARMS = {
        "ref_ctrl":      {"aux_weight": 0.00, "spine_gamma": 0.0},
        "ref_spine_aux": {"aux_weight": 0.25, "spine_gamma": 0.0},
        "ref_spine_bal": {"aux_weight": 0.25, "spine_gamma": 1.5},
    }

for p in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.PROPOSAL_CACHE_DIR, CFG.EXP_DIR):
    p.mkdir(parents=True, exist_ok=True)

random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
RUNTIME_INFO = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "gpu": torch.cuda.get_device_name(0),
    "split_mode": CFG.SPLIT_MODE,
    "python": sys.version,
    "torch": torch.__version__,
    "ultralytics": version("ultralytics"),
}
print(json.dumps(RUNTIME_INFO, indent=2, ensure_ascii=False))
"""

    mediseg_src = _strip_module_for_inline(ROOT / "solar_filament" / "models" / "mediseg_models.py")
    spine_aux_src = _strip_module_for_inline(ROOT / "solar_filament" / "pipeline" / "spine_aux.py")
    spine_yolo_src = _strip_module_for_inline(ROOT / "solar_filament" / "pipeline" / "spine_yolo.py")
    spine_refiner_src = _strip_module_for_inline(
        ROOT / "solar_filament" / "pipeline" / "spine_refiner.py"
    )
    cell2_modules = (
        "# [2/6] MediSegEdgeAttUNet + SpineGuidedEdgeAttUNet / 質量保存型 Spine Refiner 損失\n"
        + mediseg_src
        + "\n"
        + spine_aux_src
        + "\n"
        + spine_yolo_src
        + "\n"
        + spine_refiner_src
    )

    exclusions_text = (
        ROOT / "output" / "s0" / "spine_aux_prereg_exclusions_2026-11-13.csv"
    ).read_text(encoding="utf-8")

    cell3_data = f'''# [3/6] COCO 読込・台帳 SHA256 検証・分割固定・ローカル SSD ステージング
PINNED_EXCLUSIONS_CSV = """{exclusions_text}"""

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

TRAIN_JSON = next(Path(CFG.DATA_CACHE).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
coco = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
images = {{item["id"]: item for item in coco["images"]}}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)

folds_path = CFG.ARTIFACT_ROOT / "grouped_folds.csv"
folds_sha = file_sha256(folds_path)
if folds_sha != CFG.GROUPED_FOLDS_SHA256:
    raise ValueError(f"grouped_folds.csv SHA256 mismatch: {{folds_sha}}")
fold_df = pd.read_csv(folds_path)

tb_path = CFG.ARTIFACT_ROOT / "time_block_assignment.csv"
if tb_path.exists():
    tb_sha = file_sha256(tb_path)
    if tb_sha != CFG.TIME_BLOCK_SHA256:
        raise ValueError(f"time_block_assignment.csv SHA256 mismatch: {{tb_sha}}")
    tb_df = pd.read_csv(tb_path)
else:
    tb_df = fold_df.copy()
    tb_df["year"] = tb_df["file_name"].str.slice(0, 4).astype(int)
    tb_df["time_block_role"] = np.where(tb_df["year"] <= 2016, "development", "locked_evaluation")
    tb_sha = "derived_year_le_2016"

excl_path = CFG.ARTIFACT_ROOT / "spine_aux_prereg_exclusions_2026-11-13.csv"
if excl_path.exists():
    excl_sha = file_sha256(excl_path)
    if excl_sha != CFG.EXCLUSIONS_SHA256:
        raise ValueError(f"exclusions CSV SHA256 mismatch: {{excl_sha}}")
    excl_df = pd.read_csv(excl_path)
else:
    raw_bytes = PINNED_EXCLUSIONS_CSV.encode("utf-8")
    excl_df = pd.read_csv(io.StringIO(PINNED_EXCLUSIONS_CSV))
    excl_sha = hashlib.sha256(raw_bytes).hexdigest()

excluded_set = set(excl_df["file_name"].tolist())
role_of_file = dict(zip(tb_df["file_name"], tb_df["time_block_role"]))
fold_of_file = dict(zip(fold_df["file_name"], fold_df["fold"].astype(int)))

if CFG.SPLIT_MODE == "pilot":
    fit_files = [f for f in train_files if role_of_file[f] == "development" and fold_of_file[f] in (0, 1) and f not in excluded_set]
    valid_files = [f for f in train_files if role_of_file[f] == "development" and fold_of_file[f] == 2 and f not in excluded_set]
    assert len(fit_files) == 365 and len(valid_files) == 180, (len(fit_files), len(valid_files))
    assert not any(role_of_file[f] == "locked_evaluation" for f in fit_files + valid_files)
elif CFG.SPLIT_MODE == "temporal_confirm":
    fit_files = [f for f in train_files if role_of_file[f] == "development" and f not in excluded_set]
    valid_files = [f for f in train_files if role_of_file[f] == "locked_evaluation"]
    assert len(fit_files) == 545 and len(valid_files) == 160, (len(fit_files), len(valid_files))
else:
    raise ValueError(f"Unknown SPLIT_MODE: {{CFG.SPLIT_MODE}}")
assert set(fit_files).isdisjoint(valid_files), "fit/eval overlap detected!"

for name in tqdm(sorted(set(fit_files) | set(valid_files)), desc="前処理画像をローカル SSD へコピー"):
    dst = CFG.PREPROC_DIR / name
    if not dst.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, dst)

def to_rle(annotation, height, width):
    seg = annotation["segmentation"]
    if isinstance(seg, list):
        return mask_utils.merge(mask_utils.frPyObjects(seg, height, width))
    if isinstance(seg.get("counts"), list):
        return mask_utils.frPyObjects(seg, height, width)
    rle = dict(seg)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

gt_rles, gt_areas = {{}}, {{}}
for image_id, info in images.items():
    h, w = int(info["height"]), int(info["width"])
    r_list, a_list = [], []
    for a in annotations[image_id]:
        if a.get("iscrowd", 0):
            continue
        r = to_rle(a, h, w)
        r_list.append(r)
        a_list.append(float(a.get("area", mask_utils.area(r))))
    gt_rles[image_id] = r_list
    gt_areas[image_id] = np.asarray(a_list, dtype=np.float64)

print({{"split_mode": CFG.SPLIT_MODE, "fit_obs": len(fit_files), "eval_obs": len(valid_files),
       "folds_sha": folds_sha[:12], "tb_sha": tb_sha[:12], "excl_sha": excl_sha[:12]}})
'''

    cell4_train = """# [4/6] SpineCropDataset 幾何同期検査 + 同一初期重み・同一乱数による 3 Arm 厳密対照学習
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

class SpineCropDataset(torch.utils.data.Dataset):
    def __init__(self, obs_files, augment=True):
        self.augment = augment
        self.records = []
        for fname in obs_files:
            for iid in file_to_ids[fname]:
                info = images[iid]
                h, w = int(info["height"]), int(info["width"])
                valid_anns = [a for a in annotations[iid] if not a.get("iscrowd", 0)]
                for ann, rle in zip(valid_anns, gt_rles[iid]):
                    box = square_box_from_bbox(ann["bbox"], h, w)
                    sp = ann.get("spine", [])
                    sp_xy = np.asarray(sp, dtype=np.float64).reshape(-1, 2) if len(sp) >= 4 else np.zeros((0, 2), dtype=np.float64)
                    self.records.append((fname, rle, box, sp_xy))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fname, rle, box, sp_xy = self.records[idx]
        x0, y0, x1, y1 = box
        img = cv2.imread(str(CFG.PREPROC_DIR / fname), cv2.IMREAD_COLOR)
        mask = mask_utils.decode(rle).astype(np.uint8)
        crop = cv2.resize(img[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        mcrop = (cv2.resize(mask[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        scrop = crop_spine_target(
            sp_xy, box, crop_size=CFG.CROP_SIZE, radius_px=CFG.SPINE_RADIUS_PX, gt_crop_mask=mcrop
        ).astype(np.float32)

        if self.augment:
            k = random.randint(0, 3)
            if k:
                crop = np.rot90(crop, k).copy()
                mcrop = np.rot90(mcrop, k).copy()
                scrop = np.rot90(scrop, k).copy()
            if random.random() < 0.5:
                crop = np.fliplr(crop).copy()
                mcrop = np.fliplr(mcrop).copy()
                scrop = np.fliplr(scrop).copy()
            if random.random() < 0.5:
                crop = np.flipud(crop).copy()
                mcrop = np.flipud(mcrop).copy()
                scrop = np.flipud(scrop).copy()

        x_t = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float() / 255.0
        if self.augment:
            for ch in (0, 1):
                if random.random() < 0.5:
                    g = random.uniform(0.85, 1.15)
                    x_t[ch] = torch.clamp(x_t[ch], 1e-6, 1.0).pow(g)
        return x_t, torch.from_numpy(mcrop[None].copy()).float(), torch.from_numpy(scrop[None].copy()).float()

rng_split = np.random.default_rng(CFG.SEED)
perm_fit = list(fit_files)
rng_split.shuffle(perm_fit)
n_hold = max(20, int(0.10 * len(perm_fit)))
ref_val_files, ref_tr_files = perm_fit[:n_hold], perm_fit[n_hold:]

tr_ds = SpineCropDataset(ref_tr_files, augment=True)
va_ds = SpineCropDataset(ref_val_files, augment=False)

# 事前幾何検査: 拡張後サンプル 48 件で spine ターゲットが 100% GT マスク内部にあることを確認
_nonempty_sp = 0
for _i in range(min(48, len(tr_ds))):
    _, _m, _s = tr_ds[_i]
    assert bool(((_s > 0.5) <= (_m > 0.5)).all()), f"Spine spilled outside GT crop at idx={_i}"
    _nonempty_sp += int((_s > 0.5).any())
print(f"[Crop Spine Sanity] tr_crops={len(tr_ds)}, va_crops={len(va_ds)}, checked=48, nonempty_spines={_nonempty_sp}/48 (100% inside GT)")

# 全 Arm 共通の初期重みを生成し、初期値分散を完全に排除
torch.manual_seed(CFG.SEED)
_init_net = SpineGuidedEdgeAttUNet(3, 1, False)
SHARED_INIT_SD = {k: v.detach().cpu().clone() for k, v in _init_net.state_dict().items()}
del _init_net

def _seed_worker(worker_id):
    s = CFG.SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def train_refiner_arm(arm_name: str, aux_weight: float, spine_gamma: float):
    ckpt_path = CFG.EXP_DIR / f"{arm_name}_{CFG.SPLIT_MODE}_fit{len(fit_files)}.pt"
    done_path = CFG.EXP_DIR / f"{arm_name}_{CFG.SPLIT_MODE}_fit{len(fit_files)}.DONE"
    if done_path.exists() and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"再利用: {arm_name} ({ckpt_path.name}, best_dice={ck.get('best_dice', -1):.4f})")
        return ckpt_path, {"reused": True, "best_dice": float(ck.get("best_dice", -1.0)), "last_dice": float(ck.get("last_dice", -1.0))}

    random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED); torch.cuda.manual_seed_all(CFG.SEED)
    gen = torch.Generator()
    gen.manual_seed(CFG.SEED)
    tr_loader = torch.utils.data.DataLoader(
        tr_ds, batch_size=CFG.REFINER_BATCH, shuffle=True, num_workers=2,
        pin_memory=True, worker_init_fn=_seed_worker, generator=gen,
    )
    va_loader = torch.utils.data.DataLoader(
        va_ds, batch_size=CFG.REFINER_BATCH, shuffle=False, num_workers=2, pin_memory=True,
    )

    model = SpineGuidedEdgeAttUNet(3, 1, False).cuda()
    model.load_state_dict(SHARED_INIT_SD, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=CFG.REFINER_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.REFINER_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    best_dice, best_sd = -1.0, None
    history = []
    t0 = time.time()
    for ep in range(1, CFG.REFINER_EPOCHS + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for xb, yb, sb in tqdm(tr_loader, desc=f"{arm_name} ep {ep}/{CFG.REFINER_EPOCHS}", leave=False):
            xb = xb.cuda(non_blocking=True)
            yb = yb.cuda(non_blocking=True)
            sb = sb.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                mask_logits, spine_logits = model.forward_with_aux(xb)
                loss = spine_refiner_loss(
                    mask_logits, yb,
                    spine_logits=spine_logits,
                    spine_targets=sb,
                    aux_weight=aux_weight,
                    spine_gamma=spine_gamma,
                )
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.detach().item()); n_batches += 1
        sched.step()

        model.eval()
        d_sum, d_cnt = 0.0, 0
        with torch.no_grad():
            for xb, yb, _ in va_loader:
                xb = xb.cuda(non_blocking=True)
                yb = yb.cuda(non_blocking=True).bool()
                pr = model(xb).sigmoid() >= 0.5
                inter = (pr & yb).sum((1, 2, 3)).float()
                den = pr.sum((1, 2, 3)) + yb.sum((1, 2, 3))
                d_sum += ((2.0 * inter + 1.0) / (den + 1.0)).sum().item()
                d_cnt += len(xb)
        dice = d_sum / max(d_cnt, 1)
        history.append({"ep": ep, "train_loss": round(loss_sum / max(n_batches, 1), 5), "fit_holdout_dice": round(dice, 5)})
        print(f"  [{arm_name}] ep={ep:02d} loss={loss_sum / max(n_batches, 1):.4f} fit-holdout dice={dice:.4f}")
        if dice > best_dice:
            best_dice = dice
            best_sd = model.export_base_state_dict()

    last_sd = model.export_base_state_dict()
    last_dice = history[-1]["fit_holdout_dice"]
    # Exported state_dict が純正 MediSegEdgeAttUNet に strict=True で読めることを即座に保証
    _vanilla = MediSegEdgeAttUNet(3, 1, False)
    _vanilla.load_state_dict(best_sd, strict=True)
    _vanilla.load_state_dict(last_sd, strict=True)
    del _vanilla

    mins = round((time.time() - t0) / 60.0, 2)
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save({
        "best_model": best_sd,
        "last_model": last_sd,
        "best_dice": best_dice,
        "last_dice": last_dice,
        "aux_weight": aux_weight,
        "spine_gamma": spine_gamma,
        "history": history,
        "train_minutes": mins,
    }, tmp)
    os.replace(tmp, ckpt_path)
    done_path.write_text("ok", encoding="utf-8")
    del model, opt, scaler; torch.cuda.empty_cache()
    return ckpt_path, {"reused": False, "best_dice": best_dice, "last_dice": last_dice, "train_minutes": mins, "history": history}

ARM_CKPTS = {}
TRAIN_SUMMARY = {}
for _name, _cfg in CFG.ARMS.items():
    ARM_CKPTS[_name], TRAIN_SUMMARY[_name] = train_refiner_arm(_name, _cfg["aux_weight"], _cfg["spine_gamma"])
print("Refiner training summary:", {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in TRAIN_SUMMARY.items()})
"""

    cell5_proposals = """# [5/6] 厳密 OOF YOLO 提案の単一パス抽出・320x320 クロップキャッシュ構築
def mask_square_box(mask, context=CFG.CROP_CONTEXT, minimum=96):
    h, w = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, w, h
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(h, w))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= x1 - w; x1 = w
    if y1 > h: y0 -= y1 - h; y1 = h
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

cache_file = CFG.PROPOSAL_CACHE_DIR / f"oof_proposals_{CFG.SPLIT_MODE}_{len(valid_files)}.pkl"
YOLO_FOLD_SHAS = {}
eval_folds = sorted({fold_of_file[n] for n in valid_files})
for f in eval_folds:
    ypt = CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{f}" / "weights" / "best.pt"
    assert ypt.exists(), f"Missing OOF YOLO checkpoint: {ypt}"
    YOLO_FOLD_SHAS[f"fold_{f}"] = file_sha256(ypt)

if cache_file.exists():
    EVAL_PROPOSALS = pickle.loads(cache_file.read_bytes())
    print(f"再利用: OOF 提案キャッシュ ({len(EVAL_PROPOSALS)} obs)")
else:
    by_fold = defaultdict(list)
    for n in valid_files:
        by_fold[fold_of_file[n]].append(n)
    EVAL_PROPOSALS = {}
    for f in eval_folds:
        ypt = CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{f}" / "weights" / "best.pt"
        yolo_model = YOLO(str(ypt))
        for name in tqdm(by_fold[f], desc=f"OOF YOLO fold {f} 提案抽出"):
            img_path = CFG.PREPROC_DIR / name
            res = yolo_model.predict(
                str(img_path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.PRIMARY_CONF,
                iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False,
            )[0]
            h, w = res.orig_shape
            masks = []
            if res.masks is not None:
                for m in res.masks.data.detach().cpu().numpy():
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                    masks.append(m >= 0.5)
            conf = res.boxes.conf.detach().cpu().numpy().astype(np.float32) if res.boxes is not None else np.zeros(0, np.float32)
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            items = []
            for i, pm in enumerate(masks):
                if pm.sum() < CFG.MIN_AREA:
                    continue
                x0, y0, x1, y1 = mask_square_box(pm)
                crop_u8 = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                items.append({
                    "conf": float(conf[i]),
                    "raw_rle": mask_utils.encode(np.asfortranarray(pm.astype(np.uint8))),
                    "box": (x0, y0, x1, y1),
                    "crop_u8": crop_u8,
                })
            EVAL_PROPOSALS[name] = {"name": name, "fold": f, "shape": (h, w), "items": items}
        del yolo_model; torch.cuda.empty_cache()
    cache_file.write_bytes(pickle.dumps(EVAL_PROPOSALS, protocol=pickle.HIGHEST_PROTOCOL))
print("Total cached eval proposals:", sum(len(EVAL_PROPOSALS[n]["items"]) for n in valid_files))
"""

    cell6_eval = """# [6/6] バッチ GPU Refiner 推論・厳密対照 PQ / Near-Miss / 対応あり Bootstrap 検証 & 保存
def drop_contained_indices(masks, threshold=CFG.DROP_THRESHOLD):
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))
    areas = np.asarray(mask_utils.area(rles), dtype=np.float64)
    self_iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
    dropped = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped or self_iou[i, j] <= 0:
                continue
            inter = self_iou[i, j] * (areas[i] + areas[j]) / (1.0 + self_iou[i, j])
            smaller = i if areas[i] <= areas[j] else j
            if inter / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]

def apply_ownership_prod(masks):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    occupied = np.zeros(masks[0].shape, dtype=bool)
    kept = []
    for idx in sorted(drop_contained_indices(masks), key=lambda i: areas[i]):
        excl = masks[idx] & ~occupied
        if excl.sum() < CFG.MIN_AREA:
            continue
        kept.append(excl); occupied |= excl
    return kept

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

def sparse_label_pq(sum_iou, tp, fp, fn, r=0.77):
    den = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return float(r * sum_iou / den) if den > 0 else 0.0

def build_five_day_clusters(obs_names):
    parsed = sorted((datetime.strptime(n[:14], "%Y%m%d%H%M%S"), n) for n in obs_names)
    clusters, cur, prev_dt = [], [parsed[0][1]], parsed[0][0]
    for dt, name in parsed[1:]:
        if (dt - prev_dt).total_seconds() > 5 * 86400:
            clusters.append(cur); cur = [name]
        else:
            cur.append(name)
        prev_dt = dt
    clusters.append(cur)
    return clusters

def paired_bootstrap_pq(per_obs_a, per_obs_b, obs_names, block_mode="obs", n_boot=2000, seed=3407):
    rng = np.random.default_rng(seed)
    units = [[n] for n in obs_names] if block_mode == "obs" else build_five_day_clusters(obs_names)
    n_u = len(units)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_u, size=n_u)
        sa, sb = [0.0, 0, 0, 0], [0.0, 0, 0, 0]
        for ui in pick:
            for name in units[ui]:
                for k in range(4):
                    sa[k] += per_obs_a[name][k]
                    sb[k] += per_obs_b[name][k]
        den_a = sa[1] + 0.5 * sa[2] + 0.5 * sa[3]
        den_b = sb[1] + 0.5 * sb[2] + 0.5 * sb[3]
        deltas[b] = (sb[0] / den_b if den_b > 0 else 0.0) - (sa[0] / den_a if den_a > 0 else 0.0)
    return {
        "n_units": n_u,
        "block_mode": block_mode,
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "prob_positive": float((deltas > 0).mean()),
    }

def predict_probs_for_refiner(ckpt_path: Path, state_key: str = "best_model"):
    net = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    net.load_state_dict(ck[state_key], strict=True)
    del ck
    probs_by_obs = {}
    with torch.no_grad():
        for name in valid_files:
            items = EVAL_PROPOSALS[name]["items"]
            if not items:
                probs_by_obs[name] = []
                continue
            arr = np.stack([it["crop_u8"].transpose(2, 0, 1) for it in items], axis=0)
            out_list = []
            for s in range(0, len(arr), 32):
                ten = torch.from_numpy(arr[s:s + 32]).float().cuda() / 255.0
                pr = net(ten).sigmoid()[:, 0].cpu().numpy()
                out_list.append(pr)
            probs_by_obs[name] = list(np.concatenate(out_list, axis=0))
    del net; torch.cuda.empty_cache()
    return probs_by_obs

def score_probs(probs_by_obs, thr: float):
    sum_iou, tp, fp, fn = 0.0, 0, 0, 0
    near_miss_fn = 0
    small_gt_total, small_hit = 0, 0
    per_obs = {}
    for name in valid_files:
        entry = EVAL_PROPOSALS[name]
        items = entry["items"]
        probs = probs_by_obs[name]
        refined_masks = []
        for it, prob_320 in zip(items, probs):
            pm = mask_utils.decode(it["raw_rle"]) != 0
            x0, y0, x1, y1 = it["box"]
            p_box = cv2.resize(prob_320, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            rm = np.zeros_like(pm, dtype=bool)
            rm[y0:y1, x0:x1] = p_box >= thr
            rm &= pm  # strict prod shrink_mask contract
            if rm.sum() >= CFG.MIN_AREA:
                refined_masks.append(rm)
        final_masks = apply_ownership_prod(refined_masks)
        pred_rles = mask_utils.encode(np.asfortranarray(np.stack(final_masks, -1).astype(np.uint8))) if final_masks else []
        obs_stat = [0.0, 0, 0, 0]
        for iid in file_to_ids[name]:
            truth = gt_rles[iid]
            areas = gt_areas[iid]
            mat = iou_matrix(truth, pred_rles)
            if len(truth) == 0:
                fp += len(pred_rles); obs_stat[2] += len(pred_rles)
            elif len(pred_rles) == 0:
                fn += len(truth); obs_stat[3] += len(truth)
            else:
                hits = mat > 0.5
                s_i = float(mat[hits].sum())
                t_i = int(hits.sum())
                fp_i = int((hits.sum(axis=0) == 0).sum())
                fn_i = int((hits.sum(axis=1) == 0).sum())
                sum_iou += s_i; tp += t_i; fp += fp_i; fn += fn_i
                obs_stat[0] += s_i; obs_stat[1] += t_i; obs_stat[2] += fp_i; obs_stat[3] += fn_i
                max_gt_iou = mat.max(axis=1)
                near_miss_fn += int(((max_gt_iou >= 0.30) & (max_gt_iou <= 0.50)).sum())
            sm_mask = areas <= CFG.SMALL_GT_MAX_AREA
            if sm_mask.any():
                sm_truth = [truth[k] for k in np.flatnonzero(sm_mask)]
                small_gt_total += len(sm_truth)
                if pred_rles:
                    pmat = iou_matrix(sm_truth, pred_rles)
                    small_hit += int((pmat.max(axis=1) > 0.5).sum())
        per_obs[name] = tuple(obs_stat)
    den = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / den if den > 0 else 0.0
    sq = sum_iou / tp if tp else 0.0
    rq = tp / den if den > 0 else 0.0
    return {
        "pq": float(pq),
        "sq": float(sq),
        "rq": float(rq),
        "pq_sparse_r077": sparse_label_pq(sum_iou, tp, fp, fn, CFG.SPARSE_R),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "near_miss_fn_030_050": near_miss_fn,
        "small_gt_recall": small_hit / max(small_gt_total, 1),
    }, per_obs

bmode = "obs" if CFG.SPLIT_MODE == "pilot" else "5d_cluster"
PROBS_BEST = {arm: predict_probs_for_refiner(ARM_CKPTS[arm], "best_model") for arm in CFG.ARMS}
PROBS_LAST = {arm: predict_probs_for_refiner(ARM_CKPTS[arm], "last_model") for arm in CFG.ARMS}

PRIMARY_RESULTS = {}
PRIMARY_OBS = {}
THR_SWEEP = {}
LAST_MODEL_RESULTS = {}

for arm in CFG.ARMS:
    m_prim, obs_prim = score_probs(PROBS_BEST[arm], CFG.PRIMARY_THR)
    PRIMARY_RESULTS[arm] = m_prim
    PRIMARY_OBS[arm] = obs_prim
    m_last, _ = score_probs(PROBS_LAST[arm], CFG.PRIMARY_THR)
    LAST_MODEL_RESULTS[arm] = m_last
    THR_SWEEP[arm] = {f"thr_{t:.2f}": score_probs(PROBS_BEST[arm], t)[0] for t in CFG.DIAG_THRS}

COMPARISONS = {}
ctrl_pq = PRIMARY_RESULTS["ref_ctrl"]["pq"]
for arm in ("ref_spine_aux", "ref_spine_bal"):
    boot = paired_bootstrap_pq(PRIMARY_OBS["ref_ctrl"], PRIMARY_OBS[arm], valid_files, block_mode=bmode, n_boot=CFG.BOOTSTRAP_ITERS, seed=CFG.SEED)
    delta_pq = PRIMARY_RESULTS[arm]["pq"] - ctrl_pq
    delta_last = LAST_MODEL_RESULTS[arm]["pq"] - LAST_MODEL_RESULTS["ref_ctrl"]["pq"]
    COMPARISONS[f"{arm}_vs_ctrl"] = {
        "delta_pq_best_ckpt": delta_pq,
        "delta_pq_last_ckpt": delta_last,
        "delta_sq": PRIMARY_RESULTS[arm]["sq"] - PRIMARY_RESULTS["ref_ctrl"]["sq"],
        "delta_rq": PRIMARY_RESULTS[arm]["rq"] - PRIMARY_RESULTS["ref_ctrl"]["rq"],
        "delta_tp": PRIMARY_RESULTS[arm]["tp"] - PRIMARY_RESULTS["ref_ctrl"]["tp"],
        "delta_fp": PRIMARY_RESULTS[arm]["fp"] - PRIMARY_RESULTS["ref_ctrl"]["fp"],
        "delta_near_miss_fn": PRIMARY_RESULTS[arm]["near_miss_fn_030_050"] - PRIMARY_RESULTS["ref_ctrl"]["near_miss_fn_030_050"],
        "bootstrap": boot,
        "gate_pass": bool((delta_pq >= 0.005) and (boot["ci95_low"] > 0.0)),
    }

manifest_out = {
    "experiment_id": f"v129_spine_refiner_{CFG.SPLIT_MODE}",
    "runtime": RUNTIME_INFO,
    "hashes": {
        "annotations_sha256": file_sha256(TRAIN_JSON),
        "grouped_folds_sha256": folds_sha,
        "time_block_sha256": tb_sha,
        "exclusions_sha256": excl_sha,
        "yolo_oof_shas": YOLO_FOLD_SHAS,
        "refiner_arm_shas": {k: file_sha256(v) for k, v in ARM_CKPTS.items()},
    },
    "train_summary": TRAIN_SUMMARY,
    "primary_thr045_best_ckpt": PRIMARY_RESULTS,
    "last_epoch14_thr045": LAST_MODEL_RESULTS,
    "paired_comparisons_vs_ctrl": COMPARISONS,
    "threshold_sensitivity": THR_SWEEP,
}
manifest_path = CFG.ARTIFACT_ROOT / f"v129_spine_refiner_{CFG.SPLIT_MODE}_manifest.json"
manifest_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== v129 SUMMARY TABLE (thr=0.45, strict shrink_mask inside OOF YOLO proposals) ===")
for arm in CFG.ARMS:
    r = PRIMARY_RESULTS[arm]
    rl = LAST_MODEL_RESULTS[arm]
    d = r["pq"] - ctrl_pq
    dl = rl["pq"] - LAST_MODEL_RESULTS["ref_ctrl"]["pq"]
    print(
        f"{arm:14s} | best_PQ={r['pq']:.5f} ({d:+.5f}) | last_PQ={rl['pq']:.5f} ({dl:+.5f}) | "
        f"SQ={r['sq']:.5f} RQ={r['rq']:.5f} | TP={r['tp']} FP={r['fp']} FN={r['fn']} near_miss={r['near_miss_fn_030_050']}"
    )
print("=== PAIRED BOOTSTRAP VS ref_ctrl ===")
print(json.dumps(COMPARISONS, indent=2, ensure_ascii=False))
print("Saved manifest:", manifest_path)
"""

    cells = [
        _cell("markdown", md_intro),
        _cell("code", cell1_cfg),
        _cell("code", cell2_modules),
        _cell("code", cell3_data),
        _cell("code", cell4_train),
        _cell("code", cell5_proposals),
        _cell("code", cell6_eval),
    ]
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            code_txt = "".join(ln for ln in c["source"] if not ln.lstrip().startswith("%"))
            ast.parse(code_txt, filename=f"v129_cell_{idx}.py")

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def main() -> None:
    out_path = ROOT / "solar_filament_v129_spine_refiner_colab.ipynb"
    nb = build_notebook()
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {out_path.name} ({len(nb['cells'])} cells, AST verified)")


if __name__ == "__main__":
    main()
