"""Build v133 Colab notebooks for Small-Instance TAL Equalization (max_boost=4.0):

1. ``solar_filament_v133p_pilot_best_check_colab.ipynb`` (~4 min, zero training):
   Evaluates already-completed ``v127 Arm A`` vs ``v130 Small-TAL`` on BOTH ``best.pt``
   and ``last.pt`` plus 1-to-1 2-detector fusion (``Arm A + Small-TAL``) on the 180 pilot
   val observations.
2. ``solar_filament_v133a_small_tal_f0_colab.ipynb`` (Honest Fold-0):
   Trains ``Small-TAL (max_boost=4.0)`` on ``folds (1, 2)`` with exact ``prod`` recipe
   (``single_cls=True, seed=3407, patience=14, epochs=70``), evaluates ``best.pt`` &
   ``last.pt`` + 1-to-1 ``prod + small_tal`` fusion vs frozen ``prod`` Fold-0 (``0.44414``)
   with 2,000-iter paired bootstrap CI.
3. ``solar_filament_v133b_small_tal_f2_colab.ipynb`` (Honest Fold-2):
   Trains ``Small-TAL (max_boost=4.0)`` on ``folds (0, 1)`` with exact ``prod`` recipe
   (``single_cls=True, seed=3409, patience=14, epochs=70``), evaluates ``best.pt`` &
   ``last.pt`` + 1-to-1 ``prod + small_tal`` fusion vs frozen ``prod`` Fold-2 (``0.44424``)
   with 2,000-iter paired bootstrap CI.
4. ``solar_filament_v133c_small_tal_all_colab.ipynb`` (Speculative All-Data 70ep):
   Trains ``Small-TAL (max_boost=4.0)`` on all 707 observations ``folds (0, 1, 2)``
   (``val2_train012``, exact twin of ``v96-full`` ``yolo_full_fold2_best.pt``) so the
   submission-ready all-data checkpoint is immediately available if ``v133a + v133b`` pass
   the honest 471-obs 2-fold gate.

Usage:
    python scratch/build_v133_small_tal_folds_colab.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_V96 = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
SMALL_TAL_PATH = ROOT / "solar_filament" / "pipeline" / "small_tal_yolo.py"
SPINE_YOLO_PATH = ROOT / "solar_filament" / "pipeline" / "spine_yolo.py"
CONSENSUS_PATH = ROOT / "solar_filament" / "pipeline" / "consensus_fusion.py"
MEDISEG_PATH = ROOT / "solar_filament" / "models" / "mediseg_models.py"


def _cell(cell_type: str, source: str) -> dict:
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _strip_package_imports(code_txt: str) -> str:
    kept = []
    for ln in code_txt.splitlines():
        if ln.startswith(("from solar_filament.", "import solar_filament.")):
            continue
        if ln.strip() == "from __future__ import annotations":
            continue
        kept.append(ln)
    return "\n".join(kept)


def _verify_cells(cells: list[dict], tag: str) -> None:
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            txt = "".join(ln for ln in c["source"] if not ln.lstrip().startswith("%"))
            ast.parse(txt, filename=f"{tag}_cell_{idx}.py")


def _nb_dict(cells: list[dict]) -> dict:
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


def build_fold_or_all_notebook(
    *,
    variant_id: str,
    fold: int,
    train_folds: tuple[int, ...],
    arm_name: str,
    baseline_pq: float,
    is_all_data: bool,
) -> dict:
    base_nb = json.loads(SOURCE_V96.read_text(encoding="utf-8"))
    cell2_data = _source(base_nb["cells"][2])
    cell4_mediseg = _source(base_nb["cells"][4])

    spine_src = _strip_package_imports(SPINE_YOLO_PATH.read_text(encoding="utf-8"))
    tal_src = _strip_package_imports(SMALL_TAL_PATH.read_text(encoding="utf-8"))
    cons_src = _strip_package_imports(CONSENSUS_PATH.read_text(encoding="utf-8"))

    mode_label = (
        f"all-data 投機学習 (train={train_folds}, val=fold-{fold} in-sample)"
        if is_all_data
        else f"honest Fold-{fold} OOF 学習・検証 (train={train_folds}, val=fold-{fold} 非接触)"
    )

    md_intro = f"""# {variant_id} — Small-TAL (`max_boost=4.0`) {mode_label}

- `v130` pilot 実験で **`+0.01414` PQ（`0.43245 -> 0.44660`, 95% CI `[+0.00509, +0.02319]`, `P(>0)=99.75%`, 小個体 recall `+6.08 pt`）** を記録した `Small-Instance TAL Equalization` を、`prod` (`yolo_fold_{fold}`) / `v96-full` と **100% 同一の `single_cls=True` レシピ**（変更は `PureSmallTALSegmentationLoss(max_boost=4.0)` の 1 点のみ）で検証・学習する。
- データセットは Colab ローカル NVMe (`/content/solar_{variant_id}/yolo_datasets`) に構築して高速 I/O で学習し、重みと manifest は Google Drive (`runs/v133_small_tal/`, `artifacts/`) に永続保存する。
- `best.pt`・`last.pt` の単独 PQ に加え、既にメモリ上にある `prod` キャッシュとの **1-to-1 2 検出器連続融合 (`prod + small_tal`)** および 2,000 回 paired bootstrap 95% CI を同時算出する。
"""

    cell1_cfg = f"""# [1/6] 依存・Drive・設定
%pip install -q "ultralytics==8.4.120" pycocotools
import hashlib, json, math, os, pickle, random, shutil, sys, tempfile, time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO

drive.mount("/content/drive")

class CFG:
    SEED = 3407
    FOLD = {fold}
    ARMS = ({arm_name!r},)
    TRAIN_FOLDS = {train_folds!r}
    IS_ALL_DATA = {is_all_data!r}
    MAX_BOOST = 4.0
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v133_small_tal"
    CACHE_OUT_DIR = ARTIFACT_ROOT / "v133_oof_caches"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_{variant_id}")
    YOLO_ROOT = LOCAL_ROOT / "yolo_datasets"
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PROD_YOLO = DRIVE_ROOT / "runs" / f"yolo_fold_{{FOLD}}" / "weights" / "best.pt"
    REFINER_CKPT = ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{{FOLD}}_resume.pt"
    MANIFEST = ARTIFACT_ROOT / "{variant_id}_colab_manifest.json"
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    YOLO_CLS = 0.5
    PREDICT_CONF_FLOOR = 0.08
    DIAG_CONF_FLOOR = 0.02
    PREDICT_IOU = 0.55
    CONF_SWEEP = (0.20, 0.24, 0.26, 0.28, 0.32)
    PRIMARY_CONF = 0.26
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ = {baseline_pq!r}
    BOOTSTRAP_ITERS = 2000
    SPARSE_R = 0.77

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.YOLO_ROOT, CFG.RUNS_ROOT, CFG.CACHE_OUT_DIR):
    path.mkdir(parents=True, exist_ok=True)

if not CFG.REFINER_CKPT.exists():
    _cands = sorted(CFG.ARTIFACT_ROOT.glob(f"*refiner*fold{{CFG.FOLD}}*.pt"))
    if _cands:
        CFG.REFINER_CKPT = _cands[0]
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
required = {{"competition data": CFG.DATA_CACHE, "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
            "前処理画像": CFG.PREPROC_CACHE, "prod YOLO": CFG.PROD_YOLO, "EdgeAtt refiner": CFG.REFINER_CKPT}}
missing = {{k: str(v) for k, v in required.items() if not v.exists()}}
if missing:
    raise FileNotFoundError(f"前提 artifact が見つからない: {{missing}}")
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
RUNTIME_INFO = {{"started_utc": datetime.now(timezone.utc).isoformat(), "GPU": torch.cuda.get_device_name(0),
                "python": sys.version, "torch": torch.__version__, "ultralytics": version("ultralytics"),
                "pycocotools": version("pycocotools"), "opencv": cv2.__version__, "numpy": np.__version__}}
print(RUNTIME_INFO)
"""

    cell3_train = f"""# [3/6] PureSmallTALSegmentationTrainer 組込み + prod 完全準拠データセット構築・学習
# --- BEGIN spine_yolo.py ---
{spine_src}
# --- END spine_yolo.py ---

# --- BEGIN small_tal_yolo.py ---
{tal_src}
# --- END small_tal_yolo.py ---

# --- BEGIN consensus_fusion.py ---
{cons_src}
# --- END consensus_fusion.py ---

# max_boost=1.0 と標準 v8SegmentationLoss の数値一致検査 (FP32)
from ultralytics.cfg import get_cfg as _get_cfg
from ultralytics.nn.tasks import SegmentationModel as _SegModel
from ultralytics.utils.loss import v8SegmentationLoss as _v8SegLoss

torch.manual_seed(42)
_m = _SegModel("yolo11n-seg.yaml", ch=3, nc=1, verbose=False)
_m.args = _get_cfg(overrides={{"task": "segment", "overlap_mask": False}})
_lv8 = _v8SegLoss(_m)
_lp1_cls, _ = build_pure_small_tal_trainer_classes(max_boost=1.0)
_lp1 = _lp1_cls(_m)
_x = torch.randn(2, 3, 128, 128)
_pr = _m(_x)
_mk = torch.zeros((2, 32, 32), dtype=torch.float32); _mk[0, 4:18, 4:20] = 1.0; _mk[1, 22:25, 22:25] = 1.0
_bt = {{"img": _x, "batch_idx": torch.tensor([0.0, 1.0]), "cls": torch.tensor([[0.0], [0.0]]),
       "bboxes": torch.tensor([[0.375, 0.34375, 0.5, 0.4375], [0.734375, 0.734375, 0.09375, 0.09375]]), "masks": _mk}}
_tb, _ = _lv8(_pr, _bt); _t1, _ = _lp1(_pr, _bt)
PARITY_MAX_DIFF = float((_tb - _t1).abs().max().item())
assert PARITY_MAX_DIFF < 1e-5, f"Parity check failed: {{PARITY_MAX_DIFF}}"
print(f"[Sanity] PureSmallTAL(max_boost=1.0) vs v8SegmentationLoss max_diff={{PARITY_MAX_DIFF:.2e}} OK")
del _m, _lv8, _lp1, _x, _pr, _mk, _bt, _tb, _t1

def polygon_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    polygon = max(polygons, key=len)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{{value:.6f}}" for value in points.reshape(-1))

def write_variant(fold_root, split, file_name, image_id, suffix):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{{stem}}.jpeg"
    label_out = fold_root / "labels" / split / f"{{stem}}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    if not image_out.exists():
        shutil.copy2((CFG.PREPROC_DIR / file_name).resolve(), image_out)
    lines = [polygon_line(ann, info["width"], info["height"])
             for ann in annotations[image_id] if not ann.get("iscrowd", 0)]
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(line for line in lines if line), encoding="utf-8")

def build_fold_dataset(fold):
    root = CFG.YOLO_ROOT / f"val{{fold}}_train{{''.join(map(str, CFG.TRAIN_FOLDS))}}"
    reference_id = {{name: file_to_ids[name][0] for name in train_files}}
    stats = {{"train_samples": 0, "train_instances": 0, "val_samples": 0, "val_instances": 0}}
    for file_name in tqdm(train_files, desc=f"fold {{fold}} データセット構築 (local SSD)"):
        if fold_of_file[file_name] == fold:
            write_variant(root, "val", file_name, reference_id[file_name], "")
            stats["val_samples"] += 1
            stats["val_instances"] += len(annotations[reference_id[file_name]])
        if fold_of_file[file_name] in CFG.TRAIN_FOLDS:
            for image_id in file_to_ids[file_name]:
                write_variant(root, "train", file_name, image_id, f"__labeler_{{image_id}}")
                stats["train_samples"] += 1
                stats["train_instances"] += len(annotations[image_id])
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {{root}}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
                         encoding="utf-8")
    return yaml_path, stats

_, PureSmallTALTrainer = build_pure_small_tal_trainer_classes(max_boost=CFG.MAX_BOOST)

def train_arm(arm):
    run_name = f"yolo_{{arm}}_fold{{CFG.FOLD}}"
    run_dir = CFG.RUNS_ROOT / run_name
    best_path, last_path, done_path = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt", run_dir / "TRAINING_COMPLETE"
    yaml_path, stats = build_fold_dataset(CFG.FOLD)
    if done_path.exists() and best_path.exists():
        print(f"{{arm}}: 学習済み checkpoint を再利用 {{best_path}}")
        return best_path, stats
    started = time.time()
    if last_path.exists() and not done_path.exists():
        print(f"{{arm}}: 中断 last.pt から PureSmallTALTrainer で再開 {{last_path}}")
        YOLO(str(last_path)).train(resume=True, trainer=PureSmallTALTrainer)
        stats["resumed_from_last_pt"] = True
    else:
        YOLO(CFG.YOLO_MODEL).train(
            trainer=PureSmallTALTrainer,
            data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
            batch=CFG.YOLO_BATCH, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
            seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
            mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
            cls=CFG.YOLO_CLS,
            optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
            degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
            project=str(CFG.RUNS_ROOT), name=run_name, exist_ok=True,
        )
    stats["train_minutes"] = round((time.time() - started) / 60, 1)
    done_path.write_text("ok", encoding="utf-8")
    return best_path, stats
"""

    cell5_eval = """# [5/6] 評価関数 (prod 完全準拠 + バッチ化高速 Refiner + paired bootstrap + 1-to-1 融合)
def sparse_label_pq(sum_iou, tp, fp, fn, r=0.77):
    denominator = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return float(r * sum_iou / denominator) if denominator > 0 else 0.0

refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
_ckpt = torch.load(CFG.REFINER_CKPT, map_location="cuda", weights_only=False)
refiner.load_state_dict(_ckpt.get("best_model") or _ckpt["model"], strict=True)
del _ckpt
REFINER_SHA256 = file_sha256(CFG.REFINER_CKPT)

def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    out = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        out.append(mask >= 0.5)
    conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    return out, conf

def mask_square_box(mask, context=CFG.CROP_CONTEXT, minimum=96):
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
def refine_proposals_batched(image, proposals, batch_size=32):
    if not proposals:
        return []
    boxes = [mask_square_box(p) for p in proposals]
    crops = [cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
             for (x0, y0, x1, y1) in boxes]
    probs_all = []
    for s in range(0, len(crops), batch_size):
        arr = np.stack([c.transpose(2, 0, 1) for c in crops[s:s + batch_size]], axis=0)
        ten = torch.from_numpy(arr).float().cuda() / 255.0
        probs_all.append(refiner(ten).sigmoid()[:, 0].cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)
    out = []
    for pmask, (x0, y0, x1, y1), pr in zip(proposals, boxes, probs):
        pr_up = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        ref = np.zeros_like(pmask, dtype=bool)
        ref[y0:y1, x0:x1] = pr_up >= CFG.REFINER_THRESHOLD
        out.append(ref & pmask)
    return out

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

def apply_ownership(masks):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    occupied = np.zeros(masks[0].shape, dtype=bool)
    kept = []
    for index in sorted(drop_contained_indices(masks), key=lambda i: areas[i]):
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < CFG.MIN_AREA:
            continue
        kept.append(exclusive); occupied |= exclusive
    return kept

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

def build_detector_cache(weights_path, label):
    model = YOLO(str(weights_path))
    cache = []
    for name in tqdm(valid_files, desc=f"{label} 推論+refine"):
        path = CFG.PREPROC_DIR / name
        result = model.predict(str(path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.PREDICT_CONF_FLOOR,
                               iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, conf = result_masks(result)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        valid_idx = [i for i, m in enumerate(masks) if float(m.sum()) >= CFG.MIN_AREA]
        vmasks = [masks[i] for i in valid_idx]
        vconf = conf[valid_idx] if len(valid_idx) else np.zeros(0, np.float32)
        ref_masks = refine_proposals_batched(image, vmasks)
        raw_rles = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in vmasks]
        ref_rles = [mask_utils.encode(np.asfortranarray(rm.astype(np.uint8))) for rm in ref_masks]
        cache.append((name, vconf, [float(m.sum()) for m in vmasks], raw_rles, ref_rles))
    del model; torch.cuda.empty_cache()
    return cache

def score_cache(cache):
    out, per_obs_026 = {}, {}
    for c in CFG.CONF_SWEEP:
        sum_iou = 0.0; tp = 0; fp = 0; fn = 0; n_pred = 0
        obs_dict = {}
        for name, conf, areas, raw_rles, refined in cache:
            keep = [i for i in range(len(conf)) if conf[i] >= c and areas[i] >= CFG.MIN_AREA]
            preds = apply_ownership([mask_utils.decode(refined[i]) != 0 for i in keep])
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
            n_pred += len(pred_rles)
            os_s = [0.0, 0, 0, 0]
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                mat = iou_matrix(truth, pred_rles)
                if len(truth) == 0:
                    fp += len(pred_rles); os_s[2] += len(pred_rles)
                elif len(pred_rles) == 0:
                    fn += len(truth); os_s[3] += len(truth)
                else:
                    hits = mat > 0.5
                    s_i = float(mat[hits].sum()); t_i = int(hits.sum())
                    fp_i = int((hits.sum(axis=0) == 0).sum()); fn_i = int((hits.sum(axis=1) == 0).sum())
                    sum_iou += s_i; tp += t_i; fp += fp_i; fn += fn_i
                    os_s[0] += s_i; os_s[1] += t_i; os_s[2] += fp_i; os_s[3] += fn_i
            obs_dict[name] = tuple(os_s)
        den = tp + 0.5 * fp + 0.5 * fn
        pq = sum_iou / den if den > 0 else 0.0
        out[f"{c:.2f}"] = {
            "pq": float(pq), "sq": float(sum_iou / tp) if tp else 0.0, "rq": float(tp / den) if den > 0 else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "pq_sparse_r077": sparse_label_pq(sum_iou, tp, fp, fn, CFG.SPARSE_R),
            "preds_per_image": n_pred / max(len(valid_files), 1),
        }
        if abs(c - CFG.PRIMARY_CONF) < 1e-6:
            per_obs_026 = obs_dict
    return out, per_obs_026

def score_2det_fusion(cache_prod, cache_tal, thr_list=(0.18, 0.20, 0.22, 0.24, 0.26)):
    out = {}
    obs_best = {}
    best_pq = -1.0; best_key = None
    for thr in thr_list:
        sum_iou = 0.0; tp = 0; fp = 0; fn = 0; obs_d = {}
        for (name, c_p, a_p, raw_p, ref_p), (_, c_t, a_t, raw_t, ref_t) in zip(cache_prod, cache_tal):
            props = []
            for i in range(len(c_p)):
                if a_p[i] >= CFG.MIN_AREA:
                    props.append(DetectorProposal("prod", float(c_p[i]), raw_p[i], i))
            for j in range(len(c_t)):
                if a_t[j] >= CFG.MIN_AREA:
                    props.append(DetectorProposal("tal", float(c_t[j]), raw_t[j], j))
            clusters = match_detector_proposals_1to1(props, iou_threshold=0.45)
            chosen = []
            for cl in clusters:
                sc = 0.5 * cl.score_of("prod") + 0.5 * cl.score_of("tal")
                if sc >= thr:
                    bm = cl.best_member()
                    rr = ref_p[bm.prop_idx] if bm.det_name == "prod" else ref_t[bm.prop_idx]
                    chosen.append(mask_utils.decode(rr) != 0)
            preds = apply_ownership(chosen)
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
            os_s = [0.0, 0, 0, 0]
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                mat = iou_matrix(truth, pred_rles)
                if len(truth) == 0:
                    fp += len(pred_rles); os_s[2] += len(pred_rles)
                elif len(pred_rles) == 0:
                    fn += len(truth); os_s[3] += len(truth)
                else:
                    hits = mat > 0.5
                    s_i = float(mat[hits].sum()); t_i = int(hits.sum())
                    fp_i = int((hits.sum(axis=0) == 0).sum()); fn_i = int((hits.sum(axis=1) == 0).sum())
                    sum_iou += s_i; tp += t_i; fp += fp_i; fn += fn_i
                    os_s[0] += s_i; os_s[1] += t_i; os_s[2] += fp_i; os_s[3] += fn_i
            obs_d[name] = tuple(os_s)
        den = tp + 0.5 * fp + 0.5 * fn
        pq = sum_iou / den if den > 0 else 0.0
        k = f"mean2_{thr:.2f}"
        out[k] = {"pq": float(pq), "sq": float(sum_iou / tp) if tp else 0.0, "rq": float(tp / den) if den > 0 else 0.0,
                  "tp": tp, "fp": fp, "fn": fn, "pq_sparse_r077": sparse_label_pq(sum_iou, tp, fp, fn, CFG.SPARSE_R)}
        if pq > best_pq:
            best_pq = pq; best_key = k; obs_best = obs_d
    return out, best_key, obs_best

def paired_bootstrap_obs(obs_base, obs_cand, files, n_boot=2000, seed=3407):
    rng = np.random.default_rng(seed)
    ba = np.array([obs_base[f] for f in files], dtype=np.float64)
    ca = np.array([obs_cand[f] for f in files], dtype=np.float64)
    n = len(files)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = ba[idx].sum(axis=1); cs = ca[idx].sum(axis=1)
    bpq = np.where((bs[:, 1] + 0.5 * bs[:, 2] + 0.5 * bs[:, 3]) > 0, bs[:, 0] / (bs[:, 1] + 0.5 * bs[:, 2] + 0.5 * bs[:, 3]), 0.0)
    cpq = np.where((cs[:, 1] + 0.5 * cs[:, 2] + 0.5 * cs[:, 3]) > 0, cs[:, 0] / (cs[:, 1] + 0.5 * cs[:, 2] + 0.5 * cs[:, 3]), 0.0)
    d = cpq - bpq
    return {"n_obs": n, "mean_delta": float(d.mean()), "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)), "prob_positive": float((d > 0).mean())}
"""

    cell6_run = f"""# [6/6] 実行: prod baseline 検証 → Small-TAL 学習 → best/last + 2検出器融合評価 → manifest 保存
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

cache_prod = build_detector_cache(CFG.PROD_YOLO, "prod")
prod_by_conf, obs_prod_026 = score_cache(cache_prod)
prod_pq = prod_by_conf["0.26"]["pq"]
print({{"prod@0.26": round(prod_pq, 6), "既知値": CFG.BASELINE_PQ, "差": round(prod_pq - CFG.BASELINE_PQ, 6)}})

arm = CFG.ARMS[0]
best_w, ds_stats = train_arm(arm)
last_w = Path(best_w).parent / "last.pt"

cache_best = build_detector_cache(best_w, arm + "_best")
best_by_conf, obs_best_026 = score_cache(cache_best)
boot_best = paired_bootstrap_obs(obs_prod_026, obs_best_026, valid_files, n_boot=CFG.BOOTSTRAP_ITERS, seed=CFG.SEED)
fus_best, fus_best_key, obs_fus_best = score_2det_fusion(cache_prod, cache_best)
boot_fus_best = paired_bootstrap_obs(obs_prod_026, obs_fus_best, valid_files, n_boot=CFG.BOOTSTRAP_ITERS, seed=CFG.SEED)

last_res = {{}}
if last_w.exists():
    cache_last = build_detector_cache(last_w, arm + "_last")
    last_by_conf, obs_last_026 = score_cache(cache_last)
    boot_last = paired_bootstrap_obs(obs_prod_026, obs_last_026, valid_files, n_boot=CFG.BOOTSTRAP_ITERS, seed=CFG.SEED)
    fus_last, fus_last_key, obs_fus_last = score_2det_fusion(cache_prod, cache_last)
    boot_fus_last = paired_bootstrap_obs(obs_prod_026, obs_fus_last, valid_files, n_boot=CFG.BOOTSTRAP_ITERS, seed=CFG.SEED)
    last_res = {{
        "weights": str(last_w),
        "weights_sha256": file_sha256(last_w),
        "by_conf": last_by_conf,
        "delta_026_vs_prod": last_by_conf["0.26"]["pq"] - prod_pq,
        "bootstrap_vs_prod_026": boot_last,
        "fusion_2det_with_prod": fus_last,
        "fusion_best_key": fus_last_key,
        "bootstrap_fusion_vs_prod": boot_fus_last,
    }}

# Honest OOF 時は v132 4検出器融合用にキャッシュも保存
if not CFG.IS_ALL_DATA:
    (CFG.CACHE_OUT_DIR / f"small_tal_f{{CFG.FOLD}}_best_cache.pkl").write_bytes(pickle.dumps(cache_best))
    if last_w.exists():
        (CFG.CACHE_OUT_DIR / f"small_tal_f{{CFG.FOLD}}_last_cache.pkl").write_bytes(pickle.dumps(cache_last))

manifest = {{
    "experiment_id": {variant_id!r},
    "is_all_data_in_sample": CFG.IS_ALL_DATA,
    "fold": CFG.FOLD,
    "train_folds": list(CFG.TRAIN_FOLDS),
    "max_boost": CFG.MAX_BOOST,
    "parity_max_diff": PARITY_MAX_DIFF,
    "runtime": RUNTIME_INFO,
    "hashes": {{
        "annotations_sha256": file_sha256(TRAIN_JSON),
        "folds_sha256": folds_sha256,
        "refiner_sha256": REFINER_SHA256,
        "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO),
    }},
    "dataset_stats": ds_stats,
    "prod": {{"by_conf": prod_by_conf}},
    "small_tal_best": {{
        "weights": str(best_w),
        "weights_sha256": file_sha256(best_w),
        "by_conf": best_by_conf,
        "delta_026_vs_prod": best_by_conf["0.26"]["pq"] - prod_pq,
        "bootstrap_vs_prod_026": boot_best,
        "fusion_2det_with_prod": fus_best,
        "fusion_best_key": fus_best_key,
        "bootstrap_fusion_vs_prod": boot_fus_best,
    }},
    "small_tal_last": last_res,
}}
save_manifest(manifest)

print("=== {variant_id} SUMMARY (in_sample=" + str(CFG.IS_ALL_DATA) + ") ===")
print(f"prod (best.pt @0.26)      | PQ={{prod_pq:.5f}} | sparse={{prod_by_conf['0.26']['pq_sparse_r077']:.5f}} | TP={{prod_by_conf['0.26']['tp']}} FP={{prod_by_conf['0.26']['fp']}} FN={{prod_by_conf['0.26']['fn']}}")
print(f"small_tal best.pt (@0.26) | PQ={{best_by_conf['0.26']['pq']:.5f}} ({{best_by_conf['0.26']['pq']-prod_pq:+.5f}}) | CI95=[{{boot_best['ci95_low']:+.5f}}, {{boot_best['ci95_high']:+.5f}}] P(>0)={{boot_best['prob_positive']:.4f}}")
if last_res:
    lpq = last_res["by_conf"]["0.26"]["pq"]
    bl = last_res["bootstrap_vs_prod_026"]
    print(f"small_tal last.pt (@0.26) | PQ={{lpq:.5f}} ({{lpq-prod_pq:+.5f}}) | CI95=[{{bl['ci95_low']:+.5f}}, {{bl['ci95_high']:+.5f}}] P(>0)={{bl['prob_positive']:.4f}}")
print(f"2-det fusion (prod+best)  | key={{fus_best_key}} PQ={{fus_best[fus_best_key]['pq']:.5f}} ({{fus_best[fus_best_key]['pq']-prod_pq:+.5f}}) | CI95=[{{boot_fus_best['ci95_low']:+.5f}}, {{boot_fus_best['ci95_high']:+.5f}}]")
if last_res:
    fk = last_res["fusion_best_key"]
    fpq = last_res["fusion_2det_with_prod"][fk]["pq"]
    bfl = last_res["bootstrap_fusion_vs_prod"]
    print(f"2-det fusion (prod+last)  | key={{fk}} PQ={{fpq:.5f}} ({{fpq-prod_pq:+.5f}}) | CI95=[{{bfl['ci95_low']:+.5f}}, {{bfl['ci95_high']:+.5f}}]")
print("Saved manifest:", CFG.MANIFEST)
"""

    cells = [
        _cell("markdown", md_intro),
        _cell("code", cell1_cfg),
        _cell("code", cell2_data),
        _cell("code", cell3_train),
        _cell("code", cell4_mediseg),
        _cell("code", cell5_eval),
        _cell("code", cell6_run),
    ]
    _verify_cells(cells, variant_id)
    return _nb_dict(cells)


def build_pilot_best_check_notebook() -> dict:
    exclusions_text = (
        ROOT / "output" / "s0" / "spine_aux_prereg_exclusions_2026-11-13.csv"
    ).read_text(encoding="utf-8")
    mediseg_src = _strip_package_imports(MEDISEG_PATH.read_text(encoding="utf-8"))
    cons_src = _strip_package_imports(CONSENSUS_PATH.read_text(encoding="utf-8"))

    md = """# v133p — Pilot `best.pt` 対照検証 & `Arm A + Small-TAL` 1-to-1 融合 (学習ゼロ・約 4 分)

- `v130`（`last.pt` で `+0.01414` PQ）で既に Drive に保存されている `Arm A` (`v127`) と `Small-TAL` (`v130`) の **`best.pt` 同士の厳密比較**、および **2 検出器 1-to-1 連続融合 (`Arm A + Small-TAL`)** を 180 枚の pilot val で即座に採点する。
"""

    c1 = """# [1/3] Drive マウント・設定
%pip install -q "ultralytics==8.4.120" pycocotools
import hashlib, io, json, math, os, shutil, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
DATA_CACHE = DRIVE_ROOT / "competition_data"
ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
LOCAL_PREPROC = Path("/content/solar_v133p/preprocessed")
LOCAL_PREPROC.mkdir(parents=True, exist_ok=True)

ARM_A_BEST = DRIVE_ROOT / "runs" / "v127_spine_ab" / "yolo_pilot_arm_a" / "weights" / "best.pt"
ARM_A_LAST = DRIVE_ROOT / "runs" / "v127_spine_ab" / "yolo_pilot_arm_a" / "weights" / "last.pt"
TAL_BEST = DRIVE_ROOT / "runs" / "v130_small_tal" / "yolo_pilot_arm_small_tal" / "weights" / "best.pt"
TAL_LAST = DRIVE_ROOT / "runs" / "v130_small_tal" / "yolo_pilot_arm_small_tal" / "weights" / "last.pt"
REFINER_CKPT = DRIVE_ROOT / "runs" / "v127_spine_ab" / "refiner_edgeatt_pilot_fit365.pt"

for p in (ARM_A_BEST, ARM_A_LAST, TAL_BEST, TAL_LAST, REFINER_CKPT):
    assert p.exists(), f"Missing: {p}"
print("All 5 pilot checkpoints verified on Drive.")
"""

    c2 = f'''# [2/3] 分割契約・モデル・1-to-1 融合モジュール
PINNED_EXCLUSIONS_CSV = """{exclusions_text}"""

# --- BEGIN mediseg_models.py ---
{mediseg_src}
# --- END mediseg_models.py ---

# --- BEGIN consensus_fusion.py ---
{cons_src}
# --- END consensus_fusion.py ---

TRAIN_JSON = next(DATA_CACHE.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
coco = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
images = {{im["id"]: im for im in coco["images"]}}
annotations = defaultdict(list)
for a in coco["annotations"]:
    annotations[a["image_id"]].append(a)
file_to_ids = defaultdict(list)
for iid, im in images.items():
    file_to_ids[im["file_name"]].append(iid)
train_files = sorted(file_to_ids)

fold_df = pd.read_csv(ARTIFACT_ROOT / "grouped_folds.csv")
fold_of_file = dict(zip(fold_df["file_name"], fold_df["fold"].astype(int)))
tb_path = ARTIFACT_ROOT / "time_block_assignment.csv"
if tb_path.exists():
    tb_df = pd.read_csv(tb_path)
else:
    tb_df = fold_df.copy()
    tb_df["year"] = tb_df["file_name"].str.slice(0, 4).astype(int)
    tb_df["time_block_role"] = np.where(tb_df["year"] <= 2016, "development", "locked_evaluation")
role_of_file = dict(zip(tb_df["file_name"], tb_df["time_block_role"]))
excl_path = ARTIFACT_ROOT / "spine_aux_prereg_exclusions_2026-11-13.csv"
excl_df = pd.read_csv(excl_path) if excl_path.exists() else pd.read_csv(io.StringIO(PINNED_EXCLUSIONS_CSV))
excl = set(excl_df["file_name"].tolist())
valid_files = [n for n in train_files if role_of_file[n] == "development" and fold_of_file[n] == 2 and n not in excl]
assert len(valid_files) == 180, len(valid_files)

for n in valid_files:
    lp = LOCAL_PREPROC / n
    if not lp.exists():
        shutil.copy2(PREPROC_CACHE / n, lp)

def to_rle(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    if isinstance(seg.get("counts"), list):
        return mask_utils.frPyObjects(seg, h, w)
    rle = dict(seg)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

gt_rles = {{iid: [to_rle(a, int(im["height"]), int(im["width"])) for a in annotations[iid] if not a.get("iscrowd", 0)]
           for iid, im in images.items()}}
print("Pilot val_files:", len(valid_files))
'''

    c3 = """# [3/3] 4 チェックポイント一括キャッシュ + best.pt 比較 + 1-to-1 連続融合採点
refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
refiner.load_state_dict(torch.load(REFINER_CKPT, map_location="cuda", weights_only=False)["best_model"], strict=True)

def mask_square_box(mask, context=1.65, minimum=96):
    h, w = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, w, h
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = min(max(int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context)), minimum), min(h, w))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= x1 - w; x1 = w
    if y1 > h: y0 -= y1 - h; y1 = h
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

@torch.no_grad()
def refine_batch(image, proposals, batch_size=32):
    if not proposals:
        return []
    boxes = [mask_square_box(p) for p in proposals]
    crops = [cv2.resize(image[y0:y1, x0:x1], (320, 320), interpolation=cv2.INTER_LINEAR) for (x0, y0, x1, y1) in boxes]
    pa = []
    for s in range(0, len(crops), batch_size):
        ten = torch.from_numpy(np.stack([c.transpose(2, 0, 1) for c in crops[s:s + batch_size]], 0)).float().cuda() / 255.0
        pa.append(refiner(ten).sigmoid()[:, 0].cpu().numpy())
    probs = np.concatenate(pa, 0)
    out = []
    for pm, (x0, y0, x1, y1), pr in zip(proposals, boxes, probs):
        ref = np.zeros_like(pm, dtype=bool)
        ref[y0:y1, x0:x1] = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) >= 0.45
        out.append(ref & pm)
    return out

def drop_contained(masks, thr=0.90):
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = mask_utils.encode(np.asfortranarray(np.stack(masks, -1).astype(np.uint8)))
    areas = np.asarray(mask_utils.area(rles), np.float64)
    siou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), np.float64)
    drop = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in drop or j in drop or siou[i, j] <= 0:
                continue
            inter = siou[i, j] * (areas[i] + areas[j]) / (1.0 + siou[i, j])
            sm = i if areas[i] <= areas[j] else j
            if inter / max(areas[sm], 1.0) >= thr:
                drop.add(sm)
    return [i for i in range(len(masks)) if i not in drop]

def apply_own(masks):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], np.float64)
    occ = np.zeros(masks[0].shape, bool); kept = []
    for idx in sorted(drop_contained(masks), key=lambda i: areas[i]):
        ex = masks[idx] & ~occ
        if ex.sum() >= 20:
            kept.append(ex); occ |= ex
    return kept

def cache_det(pt_path, label):
    m = YOLO(str(pt_path))
    cache = []
    for name in tqdm(valid_files, desc=label):
        ip = LOCAL_PREPROC / name
        r = m.predict(str(ip), imgsz=1280, conf=0.08, iou=0.55, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        h, w = r.orig_shape
        ms = []
        if r.masks is not None:
            for d in r.masks.data.detach().cpu().numpy():
                ms.append((cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) if d.shape != (h, w) else d) >= 0.5)
        cf = r.boxes.conf.detach().cpu().numpy().astype(np.float32) if r.boxes is not None else np.zeros(0, np.float32)
        vi = [i for i, x in enumerate(ms) if x.sum() >= 20]
        vms = [ms[i] for i in vi]; vcf = cf[vi] if vi else np.zeros(0, np.float32)
        im = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        rms = refine_batch(im, vms)
        cache.append((name, vcf, [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in vms],
                      [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in rms]))
    del m; torch.cuda.empty_cache()
    return cache

def eval_single(cache, conf_thr=0.26):
    s_iou = 0.0; tp = 0; fp = 0; fn = 0; obs = {}
    for name, cf, raw_r, ref_r in cache:
        preds = apply_own([mask_utils.decode(ref_r[i]) != 0 for i in range(len(cf)) if cf[i] >= conf_thr])
        pr = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
        os_s = [0.0, 0, 0, 0]
        for iid in file_to_ids[name]:
            tr = gt_rles[iid]
            if not tr:
                fp += len(pr); os_s[2] += len(pr)
            elif not pr:
                fn += len(tr); os_s[3] += len(tr)
            else:
                mat = np.asarray(mask_utils.iou(pr, tr, [0] * len(tr)), np.float64).T
                hits = mat > 0.5
                si = float(mat[hits].sum()); ti = int(hits.sum()); fpi = int((hits.sum(0) == 0).sum()); fni = int((hits.sum(1) == 0).sum())
                s_iou += si; tp += ti; fp += fpi; fn += fni
                os_s[0] += si; os_s[1] += ti; os_s[2] += fpi; os_s[3] += fni
        obs[name] = tuple(os_s)
    den = tp + 0.5 * fp + 0.5 * fn
    return {"pq": s_iou / den, "sq": s_iou / tp, "rq": tp / den, "tp": tp, "fp": fp, "fn": fn}, obs

def eval_fusion(ca, ct, thr=0.22):
    s_iou = 0.0; tp = 0; fp = 0; fn = 0; obs = {}
    for (name, c1, r1, rf1), (_, c2, r2, rf2) in zip(ca, ct):
        ps = [DetectorProposal("a", float(c1[i]), r1[i], i) for i in range(len(c1))] + [DetectorProposal("t", float(c2[j]), r2[j], j) for j in range(len(c2))]
        cls = match_detector_proposals_1to1(ps, iou_threshold=0.45)
        ch = []
        for cl in cls:
            if 0.5 * cl.score_of("a") + 0.5 * cl.score_of("t") >= thr:
                bm = cl.best_member()
                ch.append(mask_utils.decode(rf1[bm.prop_idx] if bm.det_name == "a" else rf2[bm.prop_idx]) != 0)
        preds = apply_own(ch)
        pr = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
        os_s = [0.0, 0, 0, 0]
        for iid in file_to_ids[name]:
            tr = gt_rles[iid]
            if not tr:
                fp += len(pr); os_s[2] += len(pr)
            elif not pr:
                fn += len(tr); os_s[3] += len(tr)
            else:
                mat = np.asarray(mask_utils.iou(pr, tr, [0] * len(tr)), np.float64).T
                hits = mat > 0.5
                si = float(mat[hits].sum()); ti = int(hits.sum()); fpi = int((hits.sum(0) == 0).sum()); fni = int((hits.sum(1) == 0).sum())
                s_iou += si; tp += ti; fp += fpi; fn += fni
                os_s[0] += si; os_s[1] += ti; os_s[2] += fpi; os_s[3] += fni
        obs[name] = tuple(os_s)
    den = tp + 0.5 * fp + 0.5 * fn
    return {"pq": s_iou / den, "sq": s_iou / tp, "rq": tp / den, "tp": tp, "fp": fp, "fn": fn}, obs

def boot(ob, oc):
    rng = np.random.default_rng(3407)
    ba = np.array([ob[f] for f in valid_files], np.float64); ca = np.array([oc[f] for f in valid_files], np.float64)
    idx = rng.integers(0, len(valid_files), size=(2000, len(valid_files)))
    bs = ba[idx].sum(1); cs = ca[idx].sum(1)
    d = cs[:, 0] / (cs[:, 1] + 0.5 * cs[:, 2] + 0.5 * cs[:, 3]) - bs[:, 0] / (bs[:, 1] + 0.5 * bs[:, 2] + 0.5 * bs[:, 3])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_pos": float((d > 0).mean())}

ca_best = cache_det(ARM_A_BEST, "arm_a_best")
ct_best = cache_det(TAL_BEST, "small_tal_best")
ca_last = cache_det(ARM_A_LAST, "arm_a_last")
ct_last = cache_det(TAL_LAST, "small_tal_last")

ra_b, oa_b = eval_single(ca_best, 0.26)
rt_b, ot_b = eval_single(ct_best, 0.26)
ra_l, oa_l = eval_single(ca_last, 0.26)
rt_l, ot_l = eval_single(ct_last, 0.26)

fus_b = {f"{t:.2f}": eval_fusion(ca_best, ct_best, t) for t in (0.18, 0.20, 0.22, 0.24, 0.26)}
fus_l = {f"{t:.2f}": eval_fusion(ca_last, ct_last, t) for t in (0.18, 0.20, 0.22, 0.24, 0.26)}
bk_b = max(fus_b, key=lambda k: fus_b[k][0]["pq"])
bk_l = max(fus_l, key=lambda k: fus_l[k][0]["pq"])

out = {
    "arm_a_best_026": ra_b,
    "small_tal_best_026": rt_b,
    "boot_tal_best_vs_arm_a_best": boot(oa_b, ot_b),
    "arm_a_last_026": ra_l,
    "small_tal_last_026": rt_l,
    "boot_tal_last_vs_arm_a_last": boot(oa_l, ot_l),
    "fusion_best_pt": {k: v[0] for k, v in fus_b.items()},
    "boot_fusion_best_vs_arm_a_best": boot(oa_b, fus_b[bk_b][1]),
    "fusion_last_pt": {k: v[0] for k, v in fus_l.items()},
    "boot_fusion_last_vs_arm_a_best": boot(oa_b, fus_l[bk_l][1]),
}
mp = ARTIFACT_ROOT / "v133p_pilot_best_and_fusion_manifest.json"
mp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== v133p SUMMARY (180 pilot val obs) ===")
print(f"Arm A best.pt     @0.26 | PQ={ra_b['pq']:.5f} | TP={ra_b['tp']} FP={ra_b['fp']} FN={ra_b['fn']}")
print(f"Small-TAL best.pt @0.26 | PQ={rt_b['pq']:.5f} ({rt_b['pq']-ra_b['pq']:+.5f}) | CI95={out['boot_tal_best_vs_arm_a_best']['ci95']}")
print(f"Arm A last.pt     @0.26 | PQ={ra_l['pq']:.5f} | TP={ra_l['tp']} FP={ra_l['fp']} FN={ra_l['fn']}")
print(f"Small-TAL last.pt @0.26 | PQ={rt_l['pq']:.5f} ({rt_l['pq']-ra_l['pq']:+.5f}) | vs ArmA_best={rt_l['pq']-ra_b['pq']:+.5f}")
print(f"Fusion (best+best @{bk_b}) | PQ={fus_b[bk_b][0]['pq']:.5f} (vs ArmA_best {fus_b[bk_b][0]['pq']-ra_b['pq']:+.5f}) | CI95={out['boot_fusion_best_vs_arm_a_best']['ci95']}")
print(f"Fusion (last+last @{bk_l}) | PQ={fus_l[bk_l][0]['pq']:.5f} (vs ArmA_best {fus_l[bk_l][0]['pq']-ra_b['pq']:+.5f}) | CI95={out['boot_fusion_last_vs_arm_a_best']['ci95']}")
print("Saved:", mp)
"""

    cells = [_cell("markdown", md), _cell("code", c1), _cell("code", c2), _cell("code", c3)]
    _verify_cells(cells, "v133p")
    return _nb_dict(cells)


def main() -> None:
    specs = [
        (
            "solar_filament_v133p_pilot_best_check_colab.ipynb",
            build_pilot_best_check_notebook(),
        ),
        (
            "solar_filament_v133a_small_tal_f0_colab.ipynb",
            build_fold_or_all_notebook(
                variant_id="v133a_small_tal_f0",
                fold=0,
                train_folds=(1, 2),
                arm_name="small_tal",
                baseline_pq=0.44413694566994816,
                is_all_data=False,
            ),
        ),
        (
            "solar_filament_v133b_small_tal_f2_colab.ipynb",
            build_fold_or_all_notebook(
                variant_id="v133b_small_tal_f2",
                fold=2,
                train_folds=(0, 1),
                arm_name="small_tal",
                baseline_pq=0.444238,
                is_all_data=False,
            ),
        ),
        (
            "solar_filament_v133c_small_tal_all_colab.ipynb",
            build_fold_or_all_notebook(
                variant_id="v133c_small_tal_all",
                fold=2,
                train_folds=(0, 1, 2),
                arm_name="small_tal_all",
                baseline_pq=0.444238,
                is_all_data=True,
            ),
        ),
    ]
    for fname, nb in specs:
        out = ROOT / fname
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"built {fname} ({len(nb['cells'])} cells, AST verified)")


if __name__ == "__main__":
    main()
