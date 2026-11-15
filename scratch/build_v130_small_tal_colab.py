"""Build v130 Colab notebook: Small-Instance Task-Aligned Assigner Anchor Equalization (YOLO11m-seg).

Implements Idea 6 (v130):
- Fixes Ultralytics v8SegmentationLoss's 5x-10x positive-anchor starvation on small filaments
  (``area <= 1024 px^2`` at 2048 = 42.4% of MAGFiLO GT, which physically cover only 1-3 stride-8
  anchors vs 10 anchors for large filaments) via mass-conserving ``compute_anchor_equalization_weights``
  (``max_boost=4.0``).
- Cuts Colab GPU wall-clock time in HALF (~1h 50m on L4) by reusing (strictly read-only)
  ``v127``'s ``pilot_fit365_val180`` dataset, fit-only common refiner, and ``yolo_pilot_arm_a``
  70-epoch control checkpoint while writing all ``v130`` outputs exclusively to isolated
  ``/content/solar_v130`` and ``runs/v130_small_tal``.

Usage:
    python scratch/build_v130_small_tal_colab.py
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
    md_intro = """# v130 — Small-Instance TAL アンカー均等化 YOLO11m-seg 実験 (Idea 6)

**単一 YOLO Arm 学習のみ（Colab L4 約 1 時間 50 分）** — `v127` の対照群 `yolo_pilot_arm_a` と共通 in-split Refiner を読み取り専用で再利用し、保存先は `runs/v130_small_tal` に完全分離（`v127` / `v129` と別タブで同時実行可能）。

### 狙う数理ボトルネック
- MAGFiLO 全 GT の **42.4%** を占める小フィラメント（`area ≤ 1024 px²` @ `2048` = `≤ 400 px²` @ `1280`）は、bbox 内部に入る stride-8 P3 アンカー中心が物理的に **1〜3 個** しかない一方、大型フィラメントは常に `TaskAlignedAssigner(topk=10)` の上限 **10 個** を獲得する。
- Ultralytics 純正の `v8SegmentationLoss` は正例アンカー和をとるため、小フィラメント 1 個体あたりの勾配寄与が `cls_loss`・`box_loss`・`dfl_loss`・`seg_loss` の全項で **1/5〜1/10 に希釈** されている（現行 `prod` の未検出 FN 3,132 件の最大要因）。
- `SmallTALSegmentationLoss` (`max_boost=4.0`) は、割当アンカー数 $c$ 個の GT に属する正例アンカーへ $\\text{clamp}(10 / c, 1, 4)$ を掛けた上で、バッチ全体の正例重み総和 $\\sum w_{\\text{cls}} \\cdot t = \\sum t$ および $\\sum_{\\text{fg}} w_{\\text{seg}} = |\\text{fg}|$ を厳密保存する（`conf = 0.26` の閾値較正と実効マスク学習率を不変に維持）。
"""

    cell1_cfg = """# [1/7] 依存パッケージ・Drive マウント・実験設定 (v127 読み取り専用再利用 + 完全分離出力)
%pip install -q "ultralytics==8.4.120" pycocotools scipy
import hashlib, io, json, math, os, random, shutil, sys, time
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
    SPLIT_MODE = "pilot"  # "pilot" (365 dev fit / 180 dev∩fold2 eval) or "temporal_confirm" (545 fit / 160 locked eval)
    SEED = 3407
    MAX_BOOST = 4.0
    AUX_WEIGHT = 0.0      # Idea 1 (v127) と直交分離するため P3 spine 補助損失は 0.0 に固定
    PEER_WAIT_TIMEOUT_MIN = 240

    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    TIME_BLOCK_SHA256 = "9a9b17252133c9500d44bf3b6ec56879a47e4b32d45cb14be080ba29148c732d"
    EXCLUSIONS_SHA256 = "a56372ea6493a6d3571910e60f291606b864a4f84a61baf1df475a6627270b0a"

    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"

    # v127 参照用パス (読み取り専用・一切上書きしない)
    V127_EXP_DIR = DRIVE_ROOT / "runs" / "v127_spine_ab"
    V127_DS_ROOT = DRIVE_ROOT / "yolo_datasets" / "v127_spine_ab"

    # v130 専用出力パス
    EXP_DIR = DRIVE_ROOT / "runs" / "v130_small_tal"
    YOLO_DS_FALLBACK_ROOT = DRIVE_ROOT / "yolo_datasets" / "v130_small_tal"
    LOCAL_ROOT = Path("/content/solar_v130")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"

    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 70
    YOLO_MASK_RATIO = 2
    YOLO_CLS = 0.5

    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_EPOCHS = 14
    REFINER_LR = 2e-4
    REFINER_WD = 1e-4

    PREDICT_CONF_FLOOR = 0.08
    PRIMARY_CONF = 0.26
    CONF_SWEEP = (0.20, 0.24, 0.26, 0.28, 0.32)
    PREDICT_IOU = 0.55
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    SMALL_GT_MAX_AREA = 1024.0
    SPARSE_R = 0.77
    BOOTSTRAP_ITERS = 2000

for p in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.EXP_DIR, CFG.YOLO_DS_FALLBACK_ROOT):
    p.mkdir(parents=True, exist_ok=True)

random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
RUNTIME_INFO = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "gpu": torch.cuda.get_device_name(0),
    "split_mode": CFG.SPLIT_MODE,
    "max_boost": CFG.MAX_BOOST,
    "python": sys.version,
    "torch": torch.__version__,
    "ultralytics": version("ultralytics"),
}
print(json.dumps(RUNTIME_INFO, indent=2, ensure_ascii=False))
"""

    mediseg_src = _strip_module_for_inline(ROOT / "solar_filament" / "models" / "mediseg_models.py")
    spine_aux_src = _strip_module_for_inline(ROOT / "solar_filament" / "pipeline" / "spine_aux.py")
    spine_yolo_src = _strip_module_for_inline(ROOT / "solar_filament" / "pipeline" / "spine_yolo.py")
    small_tal_src = _strip_module_for_inline(
        ROOT / "solar_filament" / "pipeline" / "small_tal_yolo.py"
    )
    cell2_modules = (
        "# [2/7] MediSegEdgeAttUNet + Small-Instance TAL アンカー均等化トレーナー層\n"
        + mediseg_src
        + "\n"
        + spine_aux_src
        + "\n"
        + spine_yolo_src
        + "\n"
        + small_tal_src
    )

    exclusions_text = (
        ROOT / "output" / "s0" / "spine_aux_prereg_exclusions_2026-11-13.csv"
    ).read_text(encoding="utf-8")

    cell3_data = f'''# [3/7] COCO 読込・台帳 SHA256 検証・分割固定・ローカル SSD ステージング
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

    cell4_ds_and_sanity = """# [4/7] ローカル SSD データセット準備 (v127 読み取り専用コピー) & Small-TAL 損失恒等性検査
def prepare_local_yolo_dataset():
    ds_sub = f"{CFG.SPLIT_MODE}_fit{len(fit_files)}_val{len(valid_files)}"
    v127_src = CFG.V127_DS_ROOT / ds_sub
    v130_src = CFG.YOLO_DS_FALLBACK_ROOT / ds_sub
    local_ds = CFG.LOCAL_ROOT / f"yolo_ds_{CFG.SPLIT_MODE}"

    if (v127_src / "DATASET_READY_V127").exists():
        src_root = v127_src
        marker_name = "DATASET_READY_V127"
        print(f"読み取り専用再利用: v127 データセット {src_root}")
    else:
        src_root = v130_src
        marker_name = "DATASET_READY_V130"
        if not (src_root / marker_name).exists():
            stats = {"train_samples": 0, "train_instances": 0, "train_spines": 0, "val_samples": 0}
            for fname in tqdm(fit_files, desc="train 画像+ラベル生成"):
                for iid in file_to_ids[fname]:
                    info = images[iid]
                    h, w = int(info["height"]), int(info["width"])
                    stem = f"{Path(fname).stem}__labeler_{iid}"
                    img_out = src_root / "images" / "train" / f"{stem}.jpeg"
                    lbl_out = src_root / "labels" / "train" / f"{stem}.txt"
                    img_out.parent.mkdir(parents=True, exist_ok=True)
                    lbl_out.parent.mkdir(parents=True, exist_ok=True)
                    if not img_out.exists():
                        shutil.copy2(CFG.PREPROC_DIR / fname, img_out)
                    polys, spines = [], []
                    for ann in annotations[iid]:
                        if ann.get("iscrowd", 0):
                            continue
                        segs = [p for p in ann.get("segmentation", []) if len(p) >= 6]
                        if not segs:
                            continue
                        polys.append(np.asarray(max(segs, key=len), dtype=np.float64).reshape(-1, 2))
                        sp = ann.get("spine", [])
                        if len(sp) >= 4:
                            spines.append(np.asarray(sp, dtype=np.float64).reshape(-1, 2))
                    lbl_out.write_text(emit_label_txt(polys, spines, h, w), encoding="utf-8")
                    stats["train_samples"] += 1
                    stats["train_instances"] += len(polys)
                    stats["train_spines"] += len(spines)
            for fname in tqdm(valid_files, desc="val 画像+ラベル生成"):
                iid = file_to_ids[fname][0]
                info = images[iid]
                h, w = int(info["height"]), int(info["width"])
                stem = Path(fname).stem
                img_out = src_root / "images" / "val" / f"{stem}.jpeg"
                lbl_out = src_root / "labels" / "val" / f"{stem}.txt"
                img_out.parent.mkdir(parents=True, exist_ok=True)
                lbl_out.parent.mkdir(parents=True, exist_ok=True)
                if not img_out.exists():
                    shutil.copy2(CFG.PREPROC_DIR / fname, img_out)
                polys = []
                for ann in annotations[iid]:
                    if ann.get("iscrowd", 0):
                        continue
                    segs = [p for p in ann.get("segmentation", []) if len(p) >= 6]
                    if segs:
                        polys.append(np.asarray(max(segs, key=len), dtype=np.float64).reshape(-1, 2))
                lbl_out.write_text(emit_label_txt(polys, [], h, w), encoding="utf-8")
                stats["val_samples"] += 1
            (src_root / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
            (src_root / marker_name).write_text("ok", encoding="utf-8")

    # ローカル SSD に独立コピー (cache='disk' の .npy/.cache もローカルに閉じ込め Drive 競合ゼロ)
    if not (local_ds / marker_name).exists():
        if local_ds.exists():
            shutil.rmtree(local_ds)
        shutil.copytree(src_root, local_ds)
        # 古い cache ファイルがあれば再生成させてパス整合を保証
        for cfile in (local_ds / "labels").rglob("*.cache"):
            cfile.unlink()
    yaml_local = local_ds / "data.yaml"
    yaml_local.write_text(
        f"path: {local_ds.as_posix()}\\ntrain: images/train\\nval: images/val\\nnc: 2\\n"
        "names:\\n  0: filament\\n  1: spine_aux\\n",
        encoding="utf-8",
    )
    stats = json.loads((local_ds / "dataset_stats.json").read_text(encoding="utf-8"))
    return yaml_local, stats

YAML_PATH, DS_STATS = prepare_local_yolo_dataset()
print("Dataset stats:", DS_STATS)

def verify_small_tal_parity():
    \"\"\"学習前検証: max_boost=1.0 が Arm A 基底損失と bit 単位一致し、max_boost=4.0 が有限勾配を生むことを確認。\"\"\"
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel
    torch.manual_seed(CFG.SEED)
    m = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=1, verbose=False).cuda()
    m.args = get_cfg(overrides={"task": "segment", "overlap_mask": False})
    _, base_loss_cls, _ = build_spine_trainer_classes(aux_weight=0.0)
    _, tal1_cls, _ = build_small_tal_trainer_classes(max_boost=1.0, aux_weight=0.0)
    _, tal4_cls, _ = build_small_tal_trainer_classes(max_boost=CFG.MAX_BOOST, aux_weight=0.0)
    lb, l1, l4 = base_loss_cls(m), tal1_cls(m), tal4_cls(m)
    x = torch.randn(2, 3, 256, 256, device="cuda")
    masks = torch.zeros((2, 64, 64), dtype=torch.float32, device="cuda")
    masks[0, 8:44, 8:50] = 1.0
    masks[1, 48:51, 48:51] = 1.0
    batch = {
        "img": x,
        "batch_idx": torch.tensor([0.0, 1.0], device="cuda"),
        "cls": torch.tensor([[0.0], [0.0]], device="cuda"),
        "bboxes": torch.tensor([[0.45, 0.40, 0.65, 0.56], [0.77, 0.77, 0.05, 0.05]], device="cuda"),
        "masks": masks,
        "spine_segments_px": ([], []),
    }
    with torch.amp.autocast("cuda"):
        preds = m(x)
        tb, _ = lb(preds, batch)
        t1, _ = l1(preds, batch)
        t4, _ = l4(preds, batch)
    max_diff = float((tb - t1).abs().max().item())
    assert max_diff < 1e-5, f"max_boost=1.0 parity violation: {max_diff}"
    assert bool(torch.isfinite(t4).all()), "Non-finite loss under AMP!"
    del m, lb, l1, l4, x, masks, preds; torch.cuda.empty_cache()
    print(f"[Pre-train Small-TAL Parity OK] AMP max_abs_diff(max_boost=1 vs Arm A) = {max_diff:.2e}")
    return max_diff

PARITY_MAX_DIFF = verify_small_tal_parity()
"""

    cell5_train = """# [5/7] Small-TAL YOLO11m-seg 70ep 学習 (単一 Arm のみ実行 -> 壁時計時間を半減)
def train_small_tal_arm():
    run_name = f"yolo_{CFG.SPLIT_MODE}_arm_small_tal"
    run_dir = CFG.EXP_DIR / run_name
    last_pt = run_dir / "weights" / "last.pt"
    done_flag = run_dir / "TRAINING_COMPLETE"
    if done_flag.exists() and last_pt.exists():
        print(f"再利用: Small-TAL 完了済み重み {last_pt}")
        return last_pt, {"reused": True}

    _, _, trainer_cls = build_small_tal_trainer_classes(max_boost=CFG.MAX_BOOST, aux_weight=CFG.AUX_WEIGHT)
    t0 = time.time()
    if last_pt.exists() and not done_flag.exists():
        print(f"中断再開: {last_pt} から resume=True")
        trainer = trainer_cls(overrides={"model": str(last_pt), "resume": True})
    else:
        random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
        trainer = trainer_cls(overrides={
            "model": CFG.YOLO_MODEL,
            "data": str(YAML_PATH),
            "epochs": CFG.YOLO_EPOCHS,
            "imgsz": CFG.YOLO_IMGSZ,
            "batch": CFG.YOLO_BATCH,
            "device": 0,
            "workers": 4,
            "patience": CFG.YOLO_PATIENCE,
            "seed": CFG.SEED,
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
            "project": str(CFG.EXP_DIR),
            "name": run_name,
            "exist_ok": True,
        })
    trainer.train()
    assert last_pt.exists(), f"last.pt missing after training: {last_pt}"
    mins = round((time.time() - t0) / 60.0, 2)
    done_flag.write_text(
        json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(), "minutes": mins, "max_boost": CFG.MAX_BOOST}),
        encoding="utf-8",
    )
    torch.cuda.empty_cache()
    return last_pt, {"reused": False, "train_minutes": mins, "max_boost": CFG.MAX_BOOST}

SMALL_TAL_PT, SMALL_TAL_LOG = train_small_tal_arm()
print("Small-TAL training log:", SMALL_TAL_LOG)
"""

    cell6_resolve_peers = """# [6/7] 共通 in-split Refiner & 対照 Arm A (v127) の解決・フォールバック
def resolve_common_refiner():
    ref_name = f"refiner_edgeatt_{CFG.SPLIT_MODE}_fit{len(fit_files)}"
    v127_pt = CFG.V127_EXP_DIR / f"{ref_name}.pt"
    v127_done = CFG.V127_EXP_DIR / f"{ref_name}.DONE"
    if v127_done.exists() and v127_pt.exists():
        print(f"読み取り専用再利用: v127 共通 Refiner {v127_pt}")
        return v127_pt

    own_pt = CFG.EXP_DIR / f"{ref_name}.pt"
    own_done = CFG.EXP_DIR / f"{ref_name}.DONE"
    if own_done.exists() and own_pt.exists():
        print(f"再利用: v130 共通 Refiner {own_pt}")
        return own_pt

    # 万一 v127 側に無ければ v130 専用フォルダ内で fit-only 学習
    def _sq_box(bbox, height, width, context=CFG.CROP_CONTEXT, minimum=96):
        x, y, w, h = map(float, bbox)
        cx, cy = x + w / 2, y + h / 2
        side = min(max(int(math.ceil(max(w, h) * context)), minimum), min(height, width))
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x1, y1 = x0 + side, y0 + side
        if x0 < 0: x1 -= x0; x0 = 0
        if y0 < 0: y1 -= y0; y0 = 0
        if x1 > width: x0 -= x1 - width; x1 = width
        if y1 > height: y0 -= y1 - height; y1 = height
        return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

    class _FitCropDS(torch.utils.data.Dataset):
        def __init__(self, obs_files, augment=True):
            self.augment = augment
            self.records = []
            for fn in obs_files:
                for iid in file_to_ids[fn]:
                    info = images[iid]
                    h, w = int(info["height"]), int(info["width"])
                    for ann, rle in zip([a for a in annotations[iid] if not a.get("iscrowd", 0)], gt_rles[iid]):
                        self.records.append((fn, rle, _sq_box(ann["bbox"], h, w)))
        def __len__(self): return len(self.records)
        def __getitem__(self, idx):
            fn, rle, (x0, y0, x1, y1) = self.records[idx]
            img = cv2.imread(str(CFG.PREPROC_DIR / fn), cv2.IMREAD_COLOR)
            mask = mask_utils.decode(rle).astype(np.uint8)
            crop = cv2.resize(img[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            mcrop = (cv2.resize(mask[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
            if self.augment:
                k = random.randint(0, 3)
                if k: crop = np.rot90(crop, k).copy(); mcrop = np.rot90(mcrop, k).copy()
                if random.random() < 0.5: crop = np.fliplr(crop).copy(); mcrop = np.fliplr(mcrop).copy()
                if random.random() < 0.5: crop = np.flipud(crop).copy(); mcrop = np.flipud(mcrop).copy()
            xt = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float() / 255.0
            if self.augment:
                for ch in (0, 1):
                    if random.random() < 0.5:
                        xt[ch] = torch.clamp(xt[ch], 1e-6, 1.0).pow(random.uniform(0.85, 1.15))
            return xt, torch.from_numpy(mcrop[None].copy()).float()

    rng = np.random.default_rng(CFG.SEED)
    perm = list(fit_files); rng.shuffle(perm)
    n_hold = max(20, int(0.10 * len(perm)))
    tr_loader = torch.utils.data.DataLoader(_FitCropDS(perm[n_hold:], True), batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    va_loader = torch.utils.data.DataLoader(_FitCropDS(perm[:n_hold], False), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    model = MediSegEdgeAttUNet(3, 1, False).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.REFINER_LR, weight_decay=CFG.REFINER_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.REFINER_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    best_dice, best_sd = -1.0, None
    for ep in range(1, CFG.REFINER_EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                lg = model(xb)
                pr = torch.sigmoid(lg.float())
                inter = (pr * yb).sum((1, 2, 3))
                sdice = (1.0 - (2.0 * inter + 1.0) / (pr.sum((1, 2, 3)) + yb.sum((1, 2, 3)) + 1.0)).mean()
                bm_p = torch.clamp(F.max_pool2d(pr, 5, 1, 2) + F.max_pool2d(-pr, 5, 1, 2), 0.0, 1.0)
                bm_t = torch.clamp(F.max_pool2d(yb, 5, 1, 2) + F.max_pool2d(-yb, 5, 1, 2), 0.0, 1.0)
                loss = 0.35 * F.binary_cross_entropy_with_logits(lg.float(), yb) + 0.65 * sdice + 0.15 * F.l1_loss(bm_p, bm_t)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); ds_sum, dc = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.cuda(), yb.cuda().bool()
                p = model(xb).sigmoid() >= 0.5
                ds_sum += ((2.0 * (p & yb).sum((1, 2, 3)).float() + 1.0) / (p.sum((1, 2, 3)) + yb.sum((1, 2, 3)) + 1.0)).sum().item()
                dc += len(xb)
        dice = ds_sum / max(dc, 1)
        if dice > best_dice:
            best_dice = dice
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tmp = own_pt.with_suffix(".tmp")
    torch.save({"best_model": best_sd, "best_dice": best_dice}, tmp)
    os.replace(tmp, own_pt)
    own_done.write_text("ok", encoding="utf-8")
    del model, opt, scaler; torch.cuda.empty_cache()
    return own_pt

def resolve_control_arm_a():
    run_name = f"yolo_{CFG.SPLIT_MODE}_arm_a"
    v127_dir = CFG.V127_EXP_DIR / run_name
    v127_pt = v127_dir / "weights" / "last.pt"
    v127_done = v127_dir / "TRAINING_COMPLETE"
    if v127_done.exists() and v127_pt.exists():
        print(f"読み取り専用再利用: v127 Arm A 完了済み重み {v127_pt}")
        return v127_pt

    own_dir = CFG.EXP_DIR / run_name
    own_pt = own_dir / "weights" / "last.pt"
    own_done = own_dir / "TRAINING_COMPLETE"
    if own_done.exists() and own_pt.exists():
        print(f"再利用: v130 内 Arm A 完了済み重み {own_pt}")
        return own_pt

    if v127_dir.exists():
        deadline = time.time() + CFG.PEER_WAIT_TIMEOUT_MIN * 60
        while time.time() < deadline:
            if v127_done.exists() and v127_pt.exists():
                print(f"待機完了: v127 Arm A {v127_pt}")
                return v127_pt
            print("v127 タブの Arm A (yolo_pilot_arm_a) 完了を待機中...")
            time.sleep(30)
        raise TimeoutError(f"Timed out waiting for {v127_done}")

    # v127 側で該当 SPLIT_MODE の Arm A が未起動の場合のみ v130 内で対照学習
    print(f"{v127_dir} が存在しないため、v130 内で対照 Arm A を学習します...")
    _, _, base_tr_cls = build_spine_trainer_classes(aux_weight=0.0)
    tr = base_tr_cls(overrides={
        "model": CFG.YOLO_MODEL, "data": str(YAML_PATH), "epochs": CFG.YOLO_EPOCHS,
        "imgsz": CFG.YOLO_IMGSZ, "batch": CFG.YOLO_BATCH, "device": 0, "workers": 4,
        "patience": CFG.YOLO_PATIENCE, "seed": CFG.SEED, "deterministic": True,
        "single_cls": False, "mask_ratio": CFG.YOLO_MASK_RATIO, "overlap_mask": False,
        "amp": True, "cache": "disk", "cls": CFG.YOLO_CLS, "optimizer": "AdamW",
        "lr0": 7e-4, "lrf": 0.05, "weight_decay": 5e-4, "warmup_epochs": 3,
        "degrees": 180, "translate": 0.04, "scale": 0.25, "fliplr": 0.5, "flipud": 0.5,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.18, "mosaic": 0.20, "close_mosaic": 10,
        "project": str(CFG.EXP_DIR), "name": run_name, "exist_ok": True,
    })
    tr.train()
    own_done.write_text("ok", encoding="utf-8")
    torch.cuda.empty_cache()
    return own_pt

REFINER_CKPT_PATH = resolve_common_refiner()
CONTROL_ARM_A_PT = resolve_control_arm_a()
"""

    cell7_eval = """# [7/7] Small-TAL vs Arm A 厳密対応評価・サイズ別 Recall・閾値較正確認・Bootstrap 保存
def sparse_label_pq(sum_iou, tp, fp, fn, r=0.77):
    den = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return float(r * sum_iou / den) if den > 0 else 0.0

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

def infer_and_cache_detector(weights_path: Path, refiner_model, label: str):
    yolo_net = YOLO(str(weights_path))
    cache = []
    t0 = time.time()
    for name in tqdm(valid_files, desc=f"{label} 推論+refine"):
        img_path = CFG.PREPROC_DIR / name
        res = yolo_net.predict(
            str(img_path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.PREDICT_CONF_FLOOR,
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
        raw_rles, ref_rles, ref_areas = [], [], []
        with torch.no_grad():
            for pm in masks:
                raw_rles.append(mask_utils.encode(np.asfortranarray(pm.astype(np.uint8))))
                x0, y0, x1, y1 = mask_square_box(pm)
                crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                ten = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
                prob = refiner_model(ten).sigmoid()[0, 0].cpu().numpy()
                prob = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                rm = np.zeros_like(pm, dtype=bool)
                rm[y0:y1, x0:x1] = prob >= CFG.REFINER_THRESHOLD
                rm &= pm
                ref_rles.append(mask_utils.encode(np.asfortranarray(rm.astype(np.uint8))))
                ref_areas.append(float(rm.sum()))
        cache.append({
            "name": name,
            "conf": conf,
            "raw_rles": raw_rles,
            "ref_rles": ref_rles,
            "ref_areas": np.asarray(ref_areas, dtype=np.float64),
        })
    sec_per_img = (time.time() - t0) / max(len(valid_files), 1)
    del yolo_net; torch.cuda.empty_cache()
    return cache, sec_per_img

def score_cache_at_conf(cache, conf_thr: float, sec_per_img: float = 0.0):
    sum_iou, tp, fp, fn = 0.0, 0, 0, 0
    small_gt_total, small_hit_floor, small_hit_final = 0, 0, 0
    large_gt_total, large_hit_final = 0, 0
    total_preds = 0
    per_obs = {}
    for row in cache:
        name = row["name"]
        keep_idx = [i for i in range(len(row["conf"])) if row["conf"][i] >= conf_thr and row["ref_areas"][i] >= CFG.MIN_AREA]
        kept_masks = [mask_utils.decode(row["ref_rles"][i]) != 0 for i in keep_idx]
        final_masks = apply_ownership_prod(kept_masks)
        pred_rles = mask_utils.encode(np.asfortranarray(np.stack(final_masks, -1).astype(np.uint8))) if final_masks else []
        total_preds += len(pred_rles)
        floor_rles = row["raw_rles"]
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
            sm_mask = areas <= CFG.SMALL_GT_MAX_AREA
            lg_mask = ~sm_mask
            if sm_mask.any():
                sm_truth = [truth[k] for k in np.flatnonzero(sm_mask)]
                small_gt_total += len(sm_truth)
                if floor_rles:
                    small_hit_floor += int((iou_matrix(sm_truth, floor_rles).max(axis=1) > 0.5).sum())
                if pred_rles:
                    small_hit_final += int((iou_matrix(sm_truth, pred_rles).max(axis=1) > 0.5).sum())
            if lg_mask.any():
                lg_truth = [truth[k] for k in np.flatnonzero(lg_mask)]
                large_gt_total += len(lg_truth)
                if pred_rles:
                    large_hit_final += int((iou_matrix(lg_truth, pred_rles).max(axis=1) > 0.5).sum())
        per_obs[name] = tuple(obs_stat)
    den = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / den if den > 0 else 0.0
    sq = sum_iou / tp if tp else 0.0
    rq = tp / den if den > 0 else 0.0
    n_records = sum(len(file_to_ids[n]) for n in valid_files)
    return {
        "pq": float(pq),
        "sq": float(sq),
        "rq": float(rq),
        "pq_sparse_r077": sparse_label_pq(sum_iou, tp, fp, fn, CFG.SPARSE_R),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_per_record": fp / max(n_records, 1),
        "preds_per_image": total_preds / max(len(valid_files), 1),
        "small_gt_total": small_gt_total,
        "small_recall_floor008": small_hit_floor / max(small_gt_total, 1),
        "small_recall_final": small_hit_final / max(small_gt_total, 1),
        "large_gt_total": large_gt_total,
        "large_recall_final": large_hit_final / max(large_gt_total, 1),
        "sec_per_image": round(sec_per_img, 4),
    }, per_obs

refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
_rck = torch.load(REFINER_CKPT_PATH, map_location="cuda", weights_only=False)
refiner.load_state_dict(_rck["best_model"], strict=True)
del _rck

cache_a, sec_a = infer_and_cache_detector(CONTROL_ARM_A_PT, refiner, "arm_a_ctrl_last70")
cache_tal, sec_tal = infer_and_cache_detector(SMALL_TAL_PT, refiner, "arm_small_tal_last70")

res_a, obs_a = score_cache_at_conf(cache_a, CFG.PRIMARY_CONF, sec_a)
res_tal, obs_tal = score_cache_at_conf(cache_tal, CFG.PRIMARY_CONF, sec_tal)

bmode = "obs" if CFG.SPLIT_MODE == "pilot" else "5d_cluster"
boot = paired_bootstrap_pq(obs_a, obs_tal, valid_files, block_mode=bmode, n_boot=CFG.BOOTSTRAP_ITERS, seed=CFG.SEED)

delta_pq = res_tal["pq"] - res_a["pq"]
delta_small_floor_pt = 100.0 * (res_tal["small_recall_floor008"] - res_a["small_recall_floor008"])
delta_small_final_pt = 100.0 * (res_tal["small_recall_final"] - res_a["small_recall_final"])
delta_large_final_pt = 100.0 * (res_tal["large_recall_final"] - res_a["large_recall_final"])
fp_rel_change = (res_tal["fp_per_record"] - res_a["fp_per_record"]) / max(res_a["fp_per_record"], 1e-9)

if CFG.SPLIT_MODE == "pilot":
    gate_pass = bool((delta_pq >= 0.008) and (delta_small_floor_pt >= 2.0) and (fp_rel_change <= 0.06))
else:
    gate_pass = bool((delta_pq >= 0.005) and (boot["ci95_low"] > 0.0) and (delta_small_floor_pt > 0.0) and (fp_rel_change <= 0.06))

sweep_a = {f"conf_{c:.2f}": score_cache_at_conf(cache_a, c, sec_a)[0] for c in CFG.CONF_SWEEP}
sweep_tal = {f"conf_{c:.2f}": score_cache_at_conf(cache_tal, c, sec_tal)[0] for c in CFG.CONF_SWEEP}

manifest_out = {
    "experiment_id": f"v130_small_tal_{CFG.SPLIT_MODE}",
    "runtime": RUNTIME_INFO,
    "hashes": {
        "annotations_sha256": file_sha256(TRAIN_JSON),
        "grouped_folds_sha256": folds_sha,
        "time_block_sha256": tb_sha,
        "exclusions_sha256": excl_sha,
        "refiner_sha256": file_sha256(REFINER_CKPT_PATH),
        "arm_a_last_sha256": file_sha256(CONTROL_ARM_A_PT),
        "arm_small_tal_last_sha256": file_sha256(SMALL_TAL_PT),
    },
    "dataset_stats": DS_STATS,
    "parity_max_diff": PARITY_MAX_DIFF,
    "arm_a_control": res_a,
    "arm_small_tal": res_tal,
    "comparison": {
        "delta_pq_at_026": delta_pq,
        "delta_small_recall_floor008_pt": delta_small_floor_pt,
        "delta_small_recall_final_pt": delta_small_final_pt,
        "delta_large_recall_final_pt": delta_large_final_pt,
        "fp_rel_change": fp_rel_change,
        "bootstrap": boot,
        "gate_pass": gate_pass,
    },
    "conf_sweep_arm_a": sweep_a,
    "conf_sweep_small_tal": sweep_tal,
}
manifest_path = CFG.ARTIFACT_ROOT / f"v130_small_tal_{CFG.SPLIT_MODE}_manifest.json"
manifest_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== v130 PRIMARY COMPARISON (Small-TAL max_boost=4.0 vs Arm A @ conf=0.26, last.pt) ===")
print(f"Arm A (ctrl)   | PQ={res_a['pq']:.5f} SQ={res_a['sq']:.5f} RQ={res_a['rq']:.5f} | TP={res_a['tp']} FP={res_a['fp']} FN={res_a['fn']} | small_R@0.08={res_a['small_recall_floor008']:.4f} small_R={res_a['small_recall_final']:.4f} large_R={res_a['large_recall_final']:.4f}")
print(f"Arm Small-TAL  | PQ={res_tal['pq']:.5f} SQ={res_tal['sq']:.5f} RQ={res_tal['rq']:.5f} | TP={res_tal['tp']} FP={res_tal['fp']} FN={res_tal['fn']} | small_R@0.08={res_tal['small_recall_floor008']:.4f} small_R={res_tal['small_recall_final']:.4f} large_R={res_tal['large_recall_final']:.4f}")
print(json.dumps(manifest_out["comparison"], indent=2, ensure_ascii=False))
print("=== CONFIDENCE SWEEP (PQ) ===")
for c in CFG.CONF_SWEEP:
    k = f"conf_{c:.2f}"
    print(f"  conf={c:.2f} | Arm A={sweep_a[k]['pq']:.5f} | Small-TAL={sweep_tal[k]['pq']:.5f} ({sweep_tal[k]['pq'] - sweep_a[k]['pq']:+.5f})")
print("Saved manifest:", manifest_path)
"""

    cells = [
        _cell("markdown", md_intro),
        _cell("code", cell1_cfg),
        _cell("code", cell2_modules),
        _cell("code", cell3_data),
        _cell("code", cell4_ds_and_sanity),
        _cell("code", cell5_train),
        _cell("code", cell6_resolve_peers),
        _cell("code", cell7_eval),
    ]
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            code_txt = "".join(ln for ln in c["source"] if not ln.lstrip().startswith("%"))
            ast.parse(code_txt, filename=f"v130_cell_{idx}.py")

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
    out_path = ROOT / "solar_filament_v130_small_tal_colab.ipynb"
    nb = build_notebook()
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {out_path.name} ({len(nb['cells'])} cells, AST verified)")


if __name__ == "__main__":
    main()
