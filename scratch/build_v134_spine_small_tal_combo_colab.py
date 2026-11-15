"""Build v134 Colab notebooks: Orthogonal Combined Spine Aux-Teacher (0.5) + Small-TAL (4.0).

Combines the two mechanisms that independently achieved P(>0) >= 99.75% on the 180-obs pilot:
1. ``Spine Aux-Teacher`` (``aux_weight=0.5``, v127 Arm B: ``+0.01288`` PQ, ``FP -1.15%``,
   ``P(>0) = 99.85%``) — regularizes the shared FPN P3 neck feature map on centerline geometry.
2. ``Small-TAL Anchor Equalization`` (``max_boost=4.0``, v130: ``+0.01414`` PQ, small-instance
   recall ``+6.08 pt``, ``P(>0) = 99.75%``) — equalizes detection-head positive anchor weights.

Generates 4 zero-edit 1-click Colab notebooks:
- ``solar_filament_v134p_spine_small_tal_pilot_colab.ipynb`` (Pilot 365 fit / 180 val, ~1h 50m)
- ``solar_filament_v134a_spine_small_tal_f0_colab.ipynb`` (Honest Fold-0, train=(1,2), ~2h 15m)
- ``solar_filament_v134b_spine_small_tal_f2_colab.ipynb`` (Honest Fold-2, train=(0,1), ~2h 15m)
- ``solar_filament_v134c_spine_small_tal_all_colab.ipynb`` (All-Data 70ep, train=(0,1,2), ~2h 40m)

Usage:
    python scratch/build_v134_spine_small_tal_combo_colab.py
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
        r"solar_filament\.pipeline\.small_tal_yolo",
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


def build_pilot_combo_notebook() -> dict:
    """Clone v130 notebook and switch AUX_WEIGHT=0.5 + isolated v134 paths + 4-way comparison."""
    nb = json.loads((ROOT / "solar_filament_v130_small_tal_colab.ipynb").read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = _cell(
        "markdown",
        """# v134p — 直交統合 `Spine Aux-Teacher (0.5) + Small-TAL (4.0)` Pilot 実験

独立に **`P(>0) >= 99.75%`** を達成した 2 大機構を同時有効化します（推論時オーバーヘッドゼロ）：
- **`Spine Aux-Teacher (aux_weight=0.5)`** (`v127 Arm B`): `+1.29 pt` PQ・**FP `-1.15%`**（FPN P3 ネック特徴の幾何正則化）
- **`Small-TAL (max_boost=4.0)`** (`v130`): `+1.41 pt` PQ・**小個体 recall `+6.08 pt`**（検出ヘッド正例アンカー重みの面積格差是正）

既存の `v127` データセット・共通 Refiner・`Arm A`・`Arm B (Spine-Aux)`・`v130 Small-TAL` をすべて読み取り専用で再利用し、**単一 70ep 学習のみ（L4 約 1 時間 50 分）** で `best.pt` / `last.pt` 両面の 4 群比較を行います。
""",
    )

    c1 = _source(cells[1])
    assert "AUX_WEIGHT = 0.0" in c1
    c1 = c1.replace(
        "    AUX_WEIGHT = 0.0      # Idea 1 (v127) と直交分離するため P3 spine 補助損失は 0.0 に固定",
        "    AUX_WEIGHT = 0.5      # v134: Spine Aux-Teacher (0.5) + Small-TAL (4.0) 直交統合",
        1,
    )
    c1 = c1.replace('LOCAL_ROOT = Path("/content/solar_v130")', 'LOCAL_ROOT = Path("/content/solar_v134p")', 1)
    c1 = c1.replace('EXP_DIR = DRIVE_ROOT / "runs" / "v130_small_tal"', 'EXP_DIR = DRIVE_ROOT / "runs" / "v134_spine_small_tal"', 1)
    cells[1] = _cell("code", c1)

    c5 = _source(cells[5])
    c5 = c5.replace('run_name = f"yolo_{CFG.SPLIT_MODE}_arm_small_tal"', 'run_name = f"yolo_{CFG.SPLIT_MODE}_arm_spine_small_tal"', 1)
    cells[5] = _cell("code", c5)

    c7 = _source(cells[7])
    c7 = c7.replace("v130_small_tal_", "v134_spine_small_tal_")
    c7 = c7.replace("v130 PRIMARY COMPARISON", "v134p PRIMARY COMPARISON (Spine-Aux 0.5 + Small-TAL 4.0)")
    cells[7] = _cell("code", c7)

    _verify_cells(cells, "v134p")
    return _nb_dict(cells)


def build_fold_combo_notebook(
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
    tal_src = _strip_module_for_inline(ROOT / "solar_filament" / "pipeline" / "small_tal_yolo.py")
    cons_src = _strip_module_for_inline(ROOT / "solar_filament" / "pipeline" / "consensus_fusion.py")

    mode_label = (
        f"all-data 投機学習 (train={train_folds}, val=fold-{fold} in-sample)"
        if is_all_data
        else f"honest Fold-{fold} OOF 学習・検証 (train={train_folds}, val=fold-{fold} 非接触)"
    )

    md_intro = f"""# {variant_id} — 直交統合 `Spine Aux-Teacher (0.5) + Small-TAL (4.0)` {mode_label}

- `v127 Arm B` (`Spine Aux-Teacher 0.5`, `+1.29 pt` PQ, FP `-1.15%`, `P(>0)=99.85%`) と `v130` (`Small-TAL 4.0`, `+1.41 pt` PQ, 小個体 recall `+6.08 pt`, `P(>0)=99.75%`) を同時有効化し、Fold-{fold} / 全データ本番レシピで学習・評価します。
- データセットは Colab ローカル NVMe (`/content/solar_{variant_id}/yolo_datasets`) に高速構築し、重み・OOF キャッシュ・manifest は Google Drive (`runs/v134_spine_small_tal/`, `artifacts/`) に永続保存します。
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
    MAX_BOOST = 4.0
    AUX_WEIGHT = 0.5
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v134_spine_small_tal"
    CACHE_OUT_DIR = ARTIFACT_ROOT / "v134_oof_caches"
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

    cell3_train = f"""# [3/6] Spine-Aux + Small-TAL 統合トレーナー + Spine 付きローカル SSD データセット構築・学習
# --- BEGIN spine_aux.py ---
{spine_aux_src}
# --- END spine_aux.py ---

# --- BEGIN spine_yolo.py ---
{spine_yolo_src}
# --- END spine_yolo.py ---

# --- BEGIN small_tal_yolo.py ---
{tal_src}
# --- END small_tal_yolo.py ---

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

_, _, ComboTrainer = build_small_tal_trainer_classes(max_boost=CFG.MAX_BOOST, aux_weight=CFG.AUX_WEIGHT)

def train_arm(arm):
    run_name = f"yolo_{{arm}}_fold{{CFG.FOLD}}"
    run_dir = CFG.RUNS_ROOT / run_name
    best_path, last_path, done_path = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt", run_dir / "TRAINING_COMPLETE"
    yaml_path, stats = build_fold_dataset(CFG.FOLD)
    if done_path.exists() and best_path.exists():
        print(f"{{arm}}: 学習済み checkpoint を再利用 {{best_path}}")
        return best_path, stats
    started = time.time()
    try:
        ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
    except Exception:
        ram_gb = 16.0
    fast_cache = "ram" if ram_gb >= 36.0 else "disk"
    fast_workers = 8 if ram_gb >= 36.0 else 4
    print(f"[HW] RAM={{ram_gb:.1f}} GB -> cache={{fast_cache!r}}, workers={{fast_workers}}")
    if last_path.exists() and not done_path.exists():
        print(f"{{arm}}: 中断 last.pt から ComboTrainer で高速再開 {{last_path}}")
        trainer = ComboTrainer(overrides={{
            "model": str(last_path),
            "data": str(yaml_path),
            "resume": True,
            "device": 0,
            "cache": fast_cache,
            "workers": fast_workers,
        }})
        trainer.train()
        stats["resumed_from_last_pt"] = True
    else:
        random.seed(CFG.SEED + CFG.FOLD); np.random.seed(CFG.SEED + CFG.FOLD); torch.manual_seed(CFG.SEED + CFG.FOLD)
        trainer = ComboTrainer(overrides={{
            "model": CFG.YOLO_MODEL,
            "data": str(yaml_path),
            "epochs": CFG.YOLO_EPOCHS,
            "imgsz": CFG.YOLO_IMGSZ,
            "batch": CFG.YOLO_BATCH,
            "device": 0,
            "workers": fast_workers,
            "patience": CFG.YOLO_PATIENCE,
            "seed": CFG.SEED + CFG.FOLD,
            "deterministic": True,
            "single_cls": False,
            "mask_ratio": CFG.YOLO_MASK_RATIO,
            "overlap_mask": False,
            "amp": True,
            "cache": fast_cache,
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

    v138_nb = json.loads((ROOT / "solar_filament_v138a_spine_aux_f0_colab.ipynb").read_text(encoding="utf-8"))
    v133_nb = json.loads((ROOT / "solar_filament_v133a_small_tal_f0_colab.ipynb").read_text(encoding="utf-8"))
    cell5_eval = _source(v138_nb["cells"][5])
    cell6_run = _source(v133_nb["cells"][6])
    cell6_run = cell6_run.replace('"v133a_small_tal_f0"', f"{variant_id!r}")
    cell6_run = cell6_run.replace("v133a_small_tal_f0 SUMMARY", f"{variant_id} SUMMARY")
    cell6_run = cell6_run.replace("small_tal_f{CFG.FOLD}", "spine_small_tal_f{CFG.FOLD}")
    cell6_run = cell6_run.replace('"parity_max_diff": PARITY_MAX_DIFF,\n    ', "")
    cell6_run = cell6_run.replace(
        '"max_boost": CFG.MAX_BOOST,',
        '"max_boost": CFG.MAX_BOOST,\n    "aux_weight": CFG.AUX_WEIGHT,',
    )

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


def main() -> None:
    specs = [
        (
            "solar_filament_v134p_spine_small_tal_pilot_colab.ipynb",
            build_pilot_combo_notebook(),
        ),
        (
            "solar_filament_v134a_spine_small_tal_f0_colab.ipynb",
            build_fold_combo_notebook(
                variant_id="v134a_spine_small_tal_f0",
                fold=0,
                train_folds=(1, 2),
                arm_name="spine_small_tal",
                baseline_pq=0.44413694566994816,
                is_all_data=False,
            ),
        ),
        (
            "solar_filament_v134b_spine_small_tal_f2_colab.ipynb",
            build_fold_combo_notebook(
                variant_id="v134b_spine_small_tal_f2",
                fold=2,
                train_folds=(0, 1),
                arm_name="spine_small_tal",
                baseline_pq=0.444238,
                is_all_data=False,
            ),
        ),
        (
            "solar_filament_v134c_spine_small_tal_all_colab.ipynb",
            build_fold_combo_notebook(
                variant_id="v134c_spine_small_tal_all",
                fold=2,
                train_folds=(0, 1, 2),
                arm_name="spine_small_tal_all",
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
