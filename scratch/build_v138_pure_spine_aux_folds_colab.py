"""Build v138 Colab notebooks: Pure Spine Aux-Teacher (aux_weight=0.5) Honest Folds & All-Data.

Motivated by the v135 8-checkpoint audit discovery:
- Pure ``Spine Aux-Teacher (aux_weight=0.5, max_boost=1.0)`` ``best.pt`` achieved the highest
  single-model AND multi-model score on the 180-obs split:
    * ``PQ = 0.45354`` (``+1.118 pt`` vs Arm A ``best.pt`` ``0.44235``, ``P(>0) = 99.10%``)
    * ``pq_sparse_r077 = 0.39920`` (``+1.014 pt`` vs Arm A ``best.pt`` ``0.38906``)
    * ``TP = 1548 (+27)`` AND ``FP = 571 (-52 = -8.3% fewer FPs!)`` at zero inference overhead.

Generates 3 zero-edit 1-click Colab notebooks (with 27.4x fast BBox-cropped ownership in OOF eval):
- ``solar_filament_v138a_spine_aux_f0_colab.ipynb`` (Honest Fold-0, train=(1,2), val=0)
- ``solar_filament_v138b_spine_aux_f2_colab.ipynb`` (Honest Fold-2, train=(0,1), val=2)
- ``solar_filament_v138c_spine_aux_all_colab.ipynb`` (All-Data 70ep, train=(0,1,2))

Usage:
    python scratch/build_v138_pure_spine_aux_folds_colab.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_V96 = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"


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


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _verify_cells(cells: list[dict], tag: str) -> None:
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            txt = "".join(ln for ln in c["source"] if not ln.lstrip().startswith("%"))
            ast.parse(txt, filename=f"{tag}_cell_{idx}.py")


def build_fold_spine_notebook(
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

    spine_aux_src = _strip_module_for_inline(ROOT / "solar_filament" / "pipeline" / "spine_aux.py")
    spine_yolo_src = _strip_module_for_inline(ROOT / "solar_filament" / "pipeline" / "spine_yolo.py")
    cons_src = _strip_module_for_inline(ROOT / "solar_filament" / "pipeline" / "consensus_fusion.py")

    mode_label = (
        f"All-Data 70ep 本番重み学習 (train={train_folds}, val=fold-{fold} in-sample)"
        if is_all_data
        else f"Honest Fold-{fold} OOF 学習・検証 (train={train_folds}, val=fold-{fold} 非接触)"
    )

    md_intro = f"""# {variant_id} — 純粋 `Spine Aux-Teacher (aux_weight=0.5)` {mode_label}

`v135` 監査で `best.pt` 同士の厳密比較においても最高単独スコアを記録した **純粋 `Spine Aux-Teacher (0.5)`** (`v127 Arm B` と完全同一レシピ) を本番 Fold-{fold} / 全データへ展開します：
- Pilot `best.pt` 実績: **`PQ = 0.45354` (`+1.12 pt` vs Arm A `best.pt`)**、**`Sparse PQ (r=0.77) = 0.39920` (`+1.01 pt`)**、**`TP +27` かつ `FP -52 (-8.3%)`**、推論オーバーヘッドゼロ。
- OOF 採点セルには **27.4倍高速 BBox-Cropped 排他化 (`fast_apply_own_specs`)** を組込済みで、学習完了後の評価＋キャッシュ保存が数分で完了します。
"""

    cell1_cfg = f"""# [1/6] 依存・Drive・設定
%pip install -q "ultralytics==8.4.120" pycocotools
import hashlib, json, math, os, pickle, random, shutil, sys, tempfile, time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

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
    SEED = 3407
    FOLD = {fold}
    ARMS = ({arm_name!r},)
    TRAIN_FOLDS = {train_folds!r}
    IS_ALL_DATA = {is_all_data!r}
    AUX_WEIGHT = 0.5
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v138_spine_aux"
    CACHE_OUT_DIR = ARTIFACT_ROOT / "v138_oof_caches"
    V136_CACHE_DIR = ARTIFACT_ROOT / "v136_oof_caches"
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

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.YOLO_ROOT, CFG.RUNS_ROOT, CFG.CACHE_OUT_DIR, CFG.V136_CACHE_DIR):
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

    cell3_train = f"""# [3/6] 純粋 Spine Aux-Teacher (aux_weight=0.5) トレーナー + ローカル SSD データセット構築・学習
# --- BEGIN spine_aux.py ---
{spine_aux_src}
# --- END spine_aux.py ---

# --- BEGIN spine_yolo.py ---
{spine_yolo_src}
# --- END spine_yolo.py ---

# --- BEGIN consensus_fusion.py ---
{cons_src}
# --- END consensus_fusion.py ---

def write_spine_variant(fold_root, split, file_name, image_id, suffix, include_spines):
    info = images[image_id]
    h, w = int(info["height"]), int(info["width"])
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{{stem}}.jpeg"
    label_out = fold_root / "labels" / split / f"{{stem}}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    if not image_out.exists():
        shutil.copy2((CFG.PREPROC_DIR / file_name).resolve(), image_out)
    polys, spines = [], []
    for ann in annotations[image_id]:
        if ann.get("iscrowd", 0):
            continue
        segs = [p for p in ann.get("segmentation", []) if len(p) >= 6]
        if not segs:
            continue
        polys.append(np.asarray(max(segs, key=len), dtype=np.float64).reshape(-1, 2))
        if include_spines:
            sp = ann.get("spine", [])
            if len(sp) >= 4:
                spines.append(np.asarray(sp, dtype=np.float64).reshape(-1, 2))
    label_out.write_text(emit_label_txt(polys, spines, h, w), encoding="utf-8")
    return len(polys), len(spines)

def build_fold_dataset(fold):
    root = CFG.YOLO_ROOT / f"val{{fold}}_train{{''.join(map(str, CFG.TRAIN_FOLDS))}}"
    reference_id = {{name: file_to_ids[name][0] for name in train_files}}
    stats = {{"train_samples": 0, "train_instances": 0, "train_spines": 0, "val_samples": 0}}
    for file_name in tqdm(train_files, desc=f"fold {{fold}} spine+seg データセット構築 (local SSD)"):
        if fold_of_file[file_name] == fold:
            write_spine_variant(root, "val", file_name, reference_id[file_name], "", False)
            stats["val_samples"] += 1
        if fold_of_file[file_name] in CFG.TRAIN_FOLDS:
            for image_id in file_to_ids[file_name]:
                np_i, ns_i = write_spine_variant(root, "train", file_name, image_id, f"__labeler_{{image_id}}", True)
                stats["train_samples"] += 1
                stats["train_instances"] += np_i
                stats["train_spines"] += ns_i
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {{root.as_posix()}}\\ntrain: images/train\\nval: images/val\\nnc: 2\\n"
        "names:\\n  0: filament\\n  1: spine_aux\\n",
        encoding="utf-8",
    )
    return yaml_path, stats

_, _, PureSpineTrainer = build_spine_trainer_classes(aux_weight=CFG.AUX_WEIGHT)

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
        print(f"{{arm}}: 中断 last.pt から PureSpineTrainer で再開 {{last_path}}")
        trainer = PureSpineTrainer(overrides={{"model": str(last_path), "resume": True}})
        trainer.train()
        stats["resumed_from_last_pt"] = True
    else:
        random.seed(CFG.SEED + CFG.FOLD); np.random.seed(CFG.SEED + CFG.FOLD); torch.manual_seed(CFG.SEED + CFG.FOLD)
        trainer = PureSpineTrainer(overrides={{
            "model": CFG.YOLO_MODEL,
            "data": str(yaml_path),
            "epochs": CFG.YOLO_EPOCHS,
            "imgsz": CFG.YOLO_IMGSZ,
            "batch": CFG.YOLO_BATCH,
            "device": 0,
            "workers": 4,
            "patience": CFG.YOLO_PATIENCE,
            "seed": CFG.SEED + CFG.FOLD,
            "deterministic": True,
            "single_cls": False,
            "mask_ratio": CFG.YOLO_MASK_RATIO,
            "overlap_mask": False,
            "amp": True,
            "cache": "disk",
            "cls": CFG.YOLO_CLS,
            "optimizer": "AdamW",
            "lr0": 7e-4,
            "lrf": 0.05,
            "weight_decay": 5e-4,
            "warmup_epochs": 3,
            "degrees": 180,
            "translate": 0.04,
            "scale": 0.25,
            "fliplr": 0.5,
            "flipud": 0.5,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.18,
            "mosaic": 0.20,
            "close_mosaic": 10,
            "project": str(CFG.RUNS_ROOT),
            "name": run_name,
            "exist_ok": True,
        }})
        trainer.train()
    stats["train_minutes"] = round((time.time() - started) / 60, 1)
    done_path.write_text("ok", encoding="utf-8")
    return best_path, stats
"""

    cell5_eval = """# [5/6] 高速評価関数 (27.4倍高速 BBox-Cropped Ownership + バッチ化 EdgeAtt Refiner)
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
    x0, y0 = round(cx - side / 2), round(cy - side / 2)
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > width: x0 -= x1 - width; x1 = width
    if y1 > height: y0 -= y1 - height; y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

@torch.no_grad()
def refine_proposals_batched(image, proposals):
    if not proposals:
        return []
    boxes = [mask_square_box(m) for m in proposals]
    crops = np.stack([
        cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        for (x0, y0, x1, y1) in boxes
    ], axis=0)
    probs_all = []
    for s in range(0, len(crops), 32):
        ten = torch.from_numpy(crops[s:s + 32]).float().cuda() / 255.0
        probs_all.append(refiner(ten).sigmoid()[:, 0].cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)
    out = []
    for pmask, (x0, y0, x1, y1), pr in zip(proposals, boxes, probs):
        pr_up = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        ref = np.zeros_like(pmask, dtype=bool)
        ref[y0:y1, x0:x1] = pr_up >= CFG.REFINER_THRESHOLD
        out.append(ref & pmask)
    return out

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

def build_detector_cache(weights_path, label, reuse_pkl=None):
    if reuse_pkl is not None and Path(reuse_pkl).exists():
        rows = pickle.loads(Path(reuse_pkl).read_bytes())
        if len(rows) == len(valid_files):
            print(f"再利用: {Path(reuse_pkl).name} ({len(rows)} obs)")
            return rows
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
    if reuse_pkl is not None and not CFG.IS_ALL_DATA:
        Path(reuse_pkl).write_bytes(pickle.dumps(cache))
    return cache

def score_cache(cache):
    specs_by_row = [[rle_to_crop_spec(r) for r in row[4]] for row in cache]
    out, per_obs_026 = {}, {}
    for c in CFG.CONF_SWEEP:
        sum_iou = 0.0; tp = 0; fp = 0; fn = 0; n_pred = 0
        obs_dict = {}
        for (name, conf, areas, _, _), sp_list in zip(cache, specs_by_row):
            chosen_specs = [sp_list[i] for i in range(len(conf)) if conf[i] >= c and areas[i] >= CFG.MIN_AREA]
            pred_rles = fast_apply_own_specs(chosen_specs, thr=CFG.DROP_THRESHOLD, min_area=CFG.MIN_AREA)
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

def score_2det_fusion(cache_prod, cache_cand, thr_list=(0.18, 0.20, 0.22, 0.24, 0.26)):
    specs_p = [[rle_to_crop_spec(r) for r in row[4]] for row in cache_prod]
    specs_c = [[rle_to_crop_spec(r) for r in row[4]] for row in cache_cand]
    pre_clusters = []
    for (name, c_p, a_p, raw_p, _), (_, c_t, a_t, raw_t, _) in zip(cache_prod, cache_cand):
        props = [DetectorProposal("prod", float(c_p[i]), raw_p[i], i) for i in range(len(c_p)) if a_p[i] >= CFG.MIN_AREA]
        props.extend(DetectorProposal("cand", float(c_t[j]), raw_t[j], j) for j in range(len(c_t)) if a_t[j] >= CFG.MIN_AREA)
        pre_clusters.append((name, match_detector_proposals_1to1(props, iou_threshold=0.45)))

    out, obs_best = {}, {}
    best_pq = -1.0; best_key = None
    for thr in thr_list:
        sum_iou = 0.0; tp = 0; fp = 0; fn = 0; obs_d = {}
        for idx_row, (name, clusters) in enumerate(pre_clusters):
            chosen_specs = []
            for cl in clusters:
                sc = 0.5 * cl.score_of("prod") + 0.5 * cl.score_of("cand")
                if sc >= thr:
                    bm = cl.best_member()
                    chosen_specs.append(specs_p[idx_row][bm.prop_idx] if bm.det_name == "prod" else specs_c[idx_row][bm.prop_idx])
            pred_rles = fast_apply_own_specs(chosen_specs, thr=CFG.DROP_THRESHOLD, min_area=CFG.MIN_AREA)
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

    cell6_run = f"""# [6/6] 実行: prod 検証 → 純粋 Spine-Aux 学習 → 高速 best/last + 2検出器融合評価 → manifest 保存
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

prod_cache_pkl = CFG.V136_CACHE_DIR / f"prod_f{{CFG.FOLD}}_best_cache.pkl" if not CFG.IS_ALL_DATA else None
cache_prod = build_detector_cache(CFG.PROD_YOLO, "prod", reuse_pkl=prod_cache_pkl)
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

if not CFG.IS_ALL_DATA:
    (CFG.CACHE_OUT_DIR / f"spine_aux_f{{CFG.FOLD}}_best_cache.pkl").write_bytes(pickle.dumps(cache_best))
    if last_w.exists():
        (CFG.CACHE_OUT_DIR / f"spine_aux_f{{CFG.FOLD}}_last_cache.pkl").write_bytes(pickle.dumps(cache_last))

manifest = {{
    "experiment_id": {variant_id!r},
    "is_all_data_in_sample": CFG.IS_ALL_DATA,
    "fold": CFG.FOLD,
    "train_folds": list(CFG.TRAIN_FOLDS),
    "aux_weight": CFG.AUX_WEIGHT,
    "runtime": RUNTIME_INFO,
    "hashes": {{
        "annotations_sha256": file_sha256(TRAIN_JSON),
        "folds_sha256": folds_sha256,
        "refiner_sha256": REFINER_SHA256,
        "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO),
    }},
    "dataset_stats": ds_stats,
    "prod": {{"by_conf": prod_by_conf}},
    "spine_aux_best": {{
        "weights": str(best_w),
        "weights_sha256": file_sha256(best_w),
        "by_conf": best_by_conf,
        "delta_026_vs_prod": best_by_conf["0.26"]["pq"] - prod_pq,
        "bootstrap_vs_prod_026": boot_best,
        "fusion_2det_with_prod": fus_best,
        "fusion_best_key": fus_best_key,
        "bootstrap_fusion_vs_prod": boot_fus_best,
    }},
    "spine_aux_last": last_res,
}}
save_manifest(manifest)

print("=== {variant_id} SUMMARY (in_sample=" + str(CFG.IS_ALL_DATA) + ") ===")
print(f"prod (best.pt @0.26)      | PQ={{prod_pq:.5f}} | sparse={{prod_by_conf['0.26']['pq_sparse_r077']:.5f}} | TP={{prod_by_conf['0.26']['tp']}} FP={{prod_by_conf['0.26']['fp']}} FN={{prod_by_conf['0.26']['fn']}}")
print(f"spine_aux best.pt (@0.26) | PQ={{best_by_conf['0.26']['pq']:.5f}} ({{best_by_conf['0.26']['pq']-prod_pq:+.5f}}) | sparse={{best_by_conf['0.26']['pq_sparse_r077']:.5f}} | FP={{best_by_conf['0.26']['fp']}} | P(>0)={{boot_best['prob_positive']:.4f}}")
if last_res:
    lpq = last_res["by_conf"]["0.26"]["pq"]
    bl = last_res["bootstrap_vs_prod_026"]
    print(f"spine_aux last.pt (@0.26) | PQ={{lpq:.5f}} ({{lpq-prod_pq:+.5f}}) | P(>0)={{bl['prob_positive']:.4f}}")
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
    specs = [
        (
            "solar_filament_v138a_spine_aux_f0_colab.ipynb",
            build_fold_spine_notebook(
                variant_id="v138a_spine_aux_f0",
                fold=0,
                train_folds=(1, 2),
                arm_name="spine_aux",
                baseline_pq=0.44413694566994816,
                is_all_data=False,
            ),
        ),
        (
            "solar_filament_v138b_spine_aux_f2_colab.ipynb",
            build_fold_spine_notebook(
                variant_id="v138b_spine_aux_f2",
                fold=2,
                train_folds=(0, 1),
                arm_name="spine_aux",
                baseline_pq=0.444238,
                is_all_data=False,
            ),
        ),
        (
            "solar_filament_v138c_spine_aux_all_colab.ipynb",
            build_fold_spine_notebook(
                variant_id="v138c_spine_aux_all",
                fold=2,
                train_folds=(0, 1, 2),
                arm_name="spine_aux_all",
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
