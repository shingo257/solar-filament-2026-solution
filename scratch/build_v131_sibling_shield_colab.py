"""Build v131 Colab notebook: Cross-Annotator Unlabeled-Sibling Negative Shielding (YOLO11m-seg).

Implements Idea 7 (v131):
- Fixes multi-annotator negative-label poisoning in per-record YOLO replication (prod):
  under TAL soft positive targets (t_+ ~ 0.50), a 1-of-2 filament (R=2, k=1, marginal PQ
  gain +0.65 >> 0.445 per RESEARCH_LOG §2.66) converges to p* = t_+ / 2 ~ 0.25 < 0.26
  because the sibling annotator's copy trains every anchor on that filament as a hard
  background negative (t = 0, w = 1.0).
- Passes non-overlapping sibling annotators' polygons (IoU <= 0.30) through Ultralytics's
  exact Mosaic + RandomPerspective + Flip pipeline as cls=1 (``sibling_ignore``), strips
  them before Format so the model stays strictly nc=1, and sets negative BCE weight inside
  sibling masks to ``sib_neg_weight = 0.20`` (preserving crisp single-annotator positive
  masks, true solar-disk background weight 1.0, and consensus rank order 3/3 > 2/3 > 1/2 > 1/3).
- Reuses ``v127``'s completed ``yolo_pilot_arm_a`` and ``refiner_edgeatt_pilot_fit365.pt``
  read-only so only ONE 70-epoch YOLO arm is trained (~1h 50m on L4).

Usage:
    python scratch/build_v131_sibling_shield_colab.py
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
    md_intro = """# v131 — Cross-Annotator 未注釈 Sibling 負例遮蔽 YOLO11m-seg 実験 (Idea 7)

**単一 YOLO Arm 学習のみ（Colab L4 約 1 時間 50 分）** — `v127` の完了済み対照群 `yolo_pilot_arm_a` と共通 in-split Refiner を読み取り専用で再利用し、保存先は `runs/v131_sibling_shield` に完全分離（他タブと同時実行可能）。

### 狙う数理ボトルネック（`v87` で実装断念されていた未踏機構）
- `prod` のレコード複製学習では、$R$ 人が注釈した同一画像上で $k$ 人が描いたフィラメントに対し、TAL 正例ソフトターゲット $t_+ \\approx 0.50$ が $k$ 回与えられる一方、描かなかった残り $(R-k)$ 枚のコピーでは全アンカーに **背景ハード負例 ($t = 0, w = 1.0$)** が課されるため、停留確信度は
  $$p^* = \\frac{k \\cdot t_+}{k + (R - k) w_{\\text{sib}}}$$
  へ収束する。
- $w_{\\text{sib}} = 1.0$ (`prod`) では、**採点上の限界価値が明確に正（marginal ratio `0.65 >> 0.445`, `RESEARCH_LOG` §2.66）である $R=2, k=1$ 個体が $p^* \\approx 0.25 < 0.26$ となって閾値直下で脱落**し、これが「primary FN の 57〜58% が `conf ∈ [0.02, 0.26)` に眠る」主因となっていた。
- `v87` はポリゴンを union 結合したため境界膨張で失敗したが、本実験では **正例ポリゴンを `prod` と 100% 同一の単一アノテータマスクに維持**したまま、同一画像の他アノテータが描いた非重複ポリゴン（`IoU ≤ 0.30`）を擬似クラス `cls=1` として Mosaic/Affine 変換に通し、未割当負例アンカーの BCE 重みのみ $w_{\\text{sib}} = 0.20$ に緩和する（真の太陽面背景 $k=0$ は全コピーで $w=1.0$ を維持し、合意順序 $3/3 > 2/3 > 1/2 > 1/3 \\gg 0/R$ も厳密保持）。
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
    SPLIT_MODE = "pilot"          # "pilot" (365 dev fit / 180 dev∩fold2 eval) or "temporal_confirm" (545 fit / 160 locked eval)
    SEED = 3407
    SIB_NEG_WEIGHT = 0.20         # 他アノテータのみが描いた領域の負例 BCE 重み (1.0 = prod 完全一致)
    SIB_OVERLAP_MAX_IOU = 0.30    # 自身のアノテーションと IoU <= 0.30 の他者個体のみ sibling_ignore 化
    SIB_DILATE_CELLS = 1
    PEER_WAIT_TIMEOUT_MIN = 240

    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    TIME_BLOCK_SHA256 = "9a9b17252133c9500d44bf3b6ec56879a47e4b32d45cb14be080ba29148c732d"
    EXCLUSIONS_SHA256 = "a56372ea6493a6d3571910e60f291606b864a4f84a61baf1df475a6627270b0a"

    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    V127_EXP_DIR = DRIVE_ROOT / "runs" / "v127_spine_ab"

    EXP_DIR = DRIVE_ROOT / "runs" / "v131_sibling_shield"
    YOLO_DS_ROOT = DRIVE_ROOT / "yolo_datasets" / "v131_sibling_shield"
    LOCAL_ROOT = Path("/content/solar_v131")
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

for p in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.EXP_DIR, CFG.YOLO_DS_ROOT):
    p.mkdir(parents=True, exist_ok=True)

random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
RUNTIME_INFO = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "gpu": torch.cuda.get_device_name(0),
    "split_mode": CFG.SPLIT_MODE,
    "sib_neg_weight": CFG.SIB_NEG_WEIGHT,
    "python": sys.version,
    "torch": torch.__version__,
    "ultralytics": version("ultralytics"),
}
print(json.dumps(RUNTIME_INFO, indent=2, ensure_ascii=False))
"""

    mediseg_src = _strip_module_for_inline(ROOT / "solar_filament" / "models" / "mediseg_models.py")
    sib_src = _strip_module_for_inline(
        ROOT / "solar_filament" / "pipeline" / "sibling_shield_yolo.py"
    )
    cell2_modules = (
        "# [2/7] MediSegEdgeAttUNet + Sibling Negative Shielding YOLO トレーナー層\n"
        + mediseg_src
        + "\n"
        + sib_src
    )

    exclusions_text = (
        ROOT / "output" / "s0" / "spine_aux_prereg_exclusions_2026-11-13.csv"
    ).read_text(encoding="utf-8")

    cell3_data = f'''# [3/7] COCO 読込・台帳 SHA256 検証・分割固定・アノテータ間合意クラスタ計算
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

gt_rles, gt_areas, gt_polys = {{}}, {{}}, {{}}
for image_id, info in images.items():
    h, w = int(info["height"]), int(info["width"])
    r_list, a_list, p_list = [], [], []
    for a in annotations[image_id]:
        if a.get("iscrowd", 0):
            continue
        segs = [p for p in a.get("segmentation", []) if len(p) >= 6]
        if not segs:
            continue
        r = to_rle(a, h, w)
        r_list.append(r)
        a_list.append(float(a.get("area", mask_utils.area(r))))
        p_list.append(np.asarray(max(segs, key=len), dtype=np.float64).reshape(-1, 2))
    gt_rles[image_id] = r_list
    gt_areas[image_id] = np.asarray(a_list, dtype=np.float64)
    gt_polys[image_id] = p_list

# 各 GT が multi-annotator 画像上で consensus (他 record と IoU>0.5 一致あり) か singleton かを事前ラベル付け (診断用)
gt_is_consensus = {{}}
for fname in valid_files:
    iids = file_to_ids[fname]
    if len(iids) == 1:
        gt_is_consensus[iids[0]] = np.zeros(len(gt_rles[iids[0]]), dtype=bool)
        continue
    for iid in iids:
        my_rles = gt_rles[iid]
        other_rles = [r for oid in iids if oid != iid for r in gt_rles[oid]]
        if not my_rles or not other_rles:
            gt_is_consensus[iid] = np.zeros(len(my_rles), dtype=bool)
        else:
            mat = np.asarray(mask_utils.iou(other_rles, my_rles, [0] * len(my_rles)), dtype=np.float64).T
            gt_is_consensus[iid] = mat.max(axis=1) > 0.5

print({{"split_mode": CFG.SPLIT_MODE, "fit_obs": len(fit_files), "eval_obs": len(valid_files),
       "folds_sha": folds_sha[:12], "tb_sha": tb_sha[:12], "excl_sha": excl_sha[:12]}})
'''

    cell4_ds = """# [4/7] Sibling-Shield データセット構築 & 学習前パリティ検査
def build_sibling_dataset():
    ds_root = CFG.YOLO_DS_ROOT / f"{CFG.SPLIT_MODE}_fit{len(fit_files)}_val{len(valid_files)}"
    done_marker = ds_root / "DATASET_READY_V131"
    local_ds = CFG.LOCAL_ROOT / f"yolo_ds_{CFG.SPLIT_MODE}"

    if not done_marker.exists():
        stats = {"train_samples": 0, "pos_instances": 0, "sibling_ignore_instances": 0, "val_samples": 0}
        for fname in tqdm(fit_files, desc="train 画像+sibling_ignoreラベル生成"):
            iids = file_to_ids[fname]
            for iid in iids:
                info = images[iid]
                h, w = int(info["height"]), int(info["width"])
                stem = f"{Path(fname).stem}__labeler_{iid}"
                img_out = ds_root / "images" / "train" / f"{stem}.jpeg"
                lbl_out = ds_root / "labels" / "train" / f"{stem}.txt"
                img_out.parent.mkdir(parents=True, exist_ok=True)
                lbl_out.parent.mkdir(parents=True, exist_ok=True)
                if not img_out.exists():
                    shutil.copy2(CFG.PREPROC_DIR / fname, img_out)

                pos_polys = gt_polys[iid]
                pos_rles = gt_rles[iid]
                sib_polys = []
                # 同一物理画像の他アノテータが描いた個体のうち、自身の GT と IoU <= 0.30 のものを抽出
                for oid in iids:
                    if oid == iid:
                        continue
                    for o_rle, o_poly in zip(gt_rles[oid], gt_polys[oid]):
                        if not pos_rles:
                            sib_polys.append(o_poly)
                        else:
                            ious = np.asarray(mask_utils.iou(pos_rles, [o_rle], [0]), dtype=np.float64).reshape(-1)
                            if float(ious.max()) <= CFG.SIB_OVERLAP_MAX_IOU:
                                sib_polys.append(o_poly)
                lbl_out.write_text(emit_sibling_shield_label_txt(pos_polys, sib_polys, h, w), encoding="utf-8")
                stats["train_samples"] += 1
                stats["pos_instances"] += len(pos_polys)
                stats["sibling_ignore_instances"] += len(sib_polys)

        for fname in tqdm(valid_files, desc="val 画像+ラベル生成"):
            iid = file_to_ids[fname][0]
            info = images[iid]
            h, w = int(info["height"]), int(info["width"])
            stem = Path(fname).stem
            img_out = ds_root / "images" / "val" / f"{stem}.jpeg"
            lbl_out = ds_root / "labels" / "val" / f"{stem}.txt"
            img_out.parent.mkdir(parents=True, exist_ok=True)
            lbl_out.parent.mkdir(parents=True, exist_ok=True)
            if not img_out.exists():
                shutil.copy2(CFG.PREPROC_DIR / fname, img_out)
            lbl_out.write_text(emit_sibling_shield_label_txt(gt_polys[iid], [], h, w), encoding="utf-8")
            stats["val_samples"] += 1
        (ds_root / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        done_marker.write_text("ok", encoding="utf-8")

    if not (local_ds / "DATASET_READY_V131").exists():
        if local_ds.exists():
            shutil.rmtree(local_ds)
        shutil.copytree(ds_root, local_ds)
    yaml_local = local_ds / "data.yaml"
    yaml_local.write_text(
        f"path: {local_ds.as_posix()}\\ntrain: images/train\\nval: images/val\\nnc: 2\\n"
        "names:\\n  0: filament\\n  1: sibling_ignore\\n",
        encoding="utf-8",
    )
    stats = json.loads((local_ds / "dataset_stats.json").read_text(encoding="utf-8"))
    return yaml_local, stats

YAML_PATH, DS_STATS = build_sibling_dataset()
print("Dataset stats:", DS_STATS)

def verify_sibling_shield_sanity():
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.loss import v8SegmentationLoss
    torch.manual_seed(CFG.SEED)
    m = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=1, verbose=False).cuda()
    m.args = get_cfg(overrides={"task": "segment", "overlap_mask": False})
    _, sib1_cls, _ = build_sibling_shield_trainer_classes(sib_neg_weight=1.0, dilate_cells=CFG.SIB_DILATE_CELLS)
    _, sibw_cls, _ = build_sibling_shield_trainer_classes(sib_neg_weight=CFG.SIB_NEG_WEIGHT, dilate_cells=CFG.SIB_DILATE_CELLS)
    l_orig = v8SegmentationLoss(m)
    l_sib1 = sib1_cls(m)
    l_sibw = sibw_cls(m)
    x = torch.randn(2, 3, 256, 256, device="cuda")
    masks = torch.zeros((2, 64, 64), dtype=torch.float32, device="cuda")
    masks[0, 8:28, 8:28] = 1.0
    masks[1, 8:28, 8:28] = 1.0
    sib_poly = np.array([[120, 120], [220, 120], [220, 220], [120, 220]], dtype=np.float32)
    batch = {
        "img": x,
        "batch_idx": torch.tensor([0.0, 1.0], device="cuda"),
        "cls": torch.tensor([[0.0], [0.0]], device="cuda"),
        "bboxes": torch.tensor([[0.28, 0.28, 0.30, 0.30], [0.28, 0.28, 0.30, 0.30]], device="cuda"),
        "masks": masks,
        "sibling_polys_px": [[sib_poly], []],
    }
    with torch.amp.autocast("cuda"):
        preds = m(x)
        t0, _ = l_orig(preds, batch)
        t1, _ = l_sib1(preds, batch)
        tw, _ = l_sibw(preds, batch)
    diff = float((t0 - t1).abs().max().item())
    assert diff < 1e-5, f"Parity mismatch at sib_neg_weight=1.0: {diff}"
    assert float(tw[2]) < float(t1[2]), "Sibling shielding did not reduce negative cls loss!"
    del m, l_orig, l_sib1, l_sibw, x, masks, preds; torch.cuda.empty_cache()
    print(f"[Pre-train Sibling-Shield Sanity OK] parity_diff={diff:.2e}")
    return diff

PARITY_DIFF = verify_sibling_shield_sanity()
"""

    cell5_train = """# [5/7] Sibling-Shield YOLO11m-seg 70ep 学習 (単一 Arm のみ実行)
def train_sibling_shield_arm():
    run_name = f"yolo_{CFG.SPLIT_MODE}_arm_sibling_shield"
    run_dir = CFG.EXP_DIR / run_name
    last_pt = run_dir / "weights" / "last.pt"
    done_flag = run_dir / "TRAINING_COMPLETE"
    if done_flag.exists() and last_pt.exists():
        print(f"再利用: Sibling-Shield 完了済み重み {last_pt}")
        return last_pt, {"reused": True}

    _, _, trainer_cls = build_sibling_shield_trainer_classes(
        sib_neg_weight=CFG.SIB_NEG_WEIGHT, dilate_cells=CFG.SIB_DILATE_CELLS
    )
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
        json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(), "minutes": mins, "sib_neg_weight": CFG.SIB_NEG_WEIGHT}),
        encoding="utf-8",
    )
    torch.cuda.empty_cache()
    return last_pt, {"reused": False, "train_minutes": mins, "sib_neg_weight": CFG.SIB_NEG_WEIGHT}

SIB_SHIELD_PT, SIB_TRAIN_LOG = train_sibling_shield_arm()
print("Sibling-Shield training log:", SIB_TRAIN_LOG)
"""

    cell6_peers = """# [6/7] 共通 in-split Refiner & 対照 Arm A (v127) 解決
ref_name = f"refiner_edgeatt_{CFG.SPLIT_MODE}_fit{len(fit_files)}"
REFINER_CKPT_PATH = CFG.V127_EXP_DIR / f"{ref_name}.pt"
assert REFINER_CKPT_PATH.exists(), f"Missing fit-only refiner: {REFINER_CKPT_PATH}"

CONTROL_ARM_A_PT = CFG.V127_EXP_DIR / f"yolo_{CFG.SPLIT_MODE}_arm_a" / "weights" / "last.pt"
_arm_a_done = CFG.V127_EXP_DIR / f"yolo_{CFG.SPLIT_MODE}_arm_a" / "TRAINING_COMPLETE"
if not _arm_a_done.exists():
    deadline = time.time() + CFG.PEER_WAIT_TIMEOUT_MIN * 60
    while time.time() < deadline and not _arm_a_done.exists():
        print("Waiting for v127 Arm A TRAINING_COMPLETE...")
        time.sleep(30)
assert CONTROL_ARM_A_PT.exists(), f"Missing Arm A control weights: {CONTROL_ARM_A_PT}"
print("Resolved Refiner & Arm A control:", REFINER_CKPT_PATH.name, CONTROL_ARM_A_PT)
"""

    cell7_eval = """# [7/7] Sibling-Shield vs Arm A 対応評価・Singleton/Consensus 別 Recall 分解・Bootstrap 保存
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

def infer_and_cache_batched(weights_path: Path, refiner_model, label: str):
    yolo_net = YOLO(str(weights_path))
    cache = []
    t0 = time.time()
    for name in tqdm(valid_files, desc=f"{label} 推論+バッチrefine"):
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
        if masks:
            boxes = [mask_square_box(pm) for pm in masks]
            crops = np.stack([
                cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
                for (x0, y0, x1, y1) in boxes
            ], axis=0)
            probs = []
            with torch.no_grad():
                for s in range(0, len(crops), 32):
                    ten = torch.from_numpy(crops[s:s + 32]).float().cuda() / 255.0
                    probs.append(refiner_model(ten).sigmoid()[:, 0].cpu().numpy())
            probs_320 = np.concatenate(probs, axis=0)
            for pm, (x0, y0, x1, y1), p320 in zip(masks, boxes, probs_320):
                raw_rles.append(mask_utils.encode(np.asfortranarray(pm.astype(np.uint8))))
                p_box = cv2.resize(p320, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                rm = np.zeros_like(pm, dtype=bool)
                rm[y0:y1, x0:x1] = p_box >= CFG.REFINER_THRESHOLD
                rm &= pm
                ref_rles.append(mask_utils.encode(np.asfortranarray(rm.astype(np.uint8))))
                ref_areas.append(float(rm.sum()))
        cache.append({"name": name, "conf": conf, "raw_rles": raw_rles, "ref_rles": ref_rles, "ref_areas": np.asarray(ref_areas, dtype=np.float64)})
    sec_per_img = (time.time() - t0) / max(len(valid_files), 1)
    del yolo_net; torch.cuda.empty_cache()
    return cache, sec_per_img

def score_cache_at_conf(cache, conf_thr: float, sec_per_img: float = 0.0):
    sum_iou, tp, fp, fn = 0.0, 0, 0, 0
    multi_singleton_tot, multi_singleton_hit = 0, 0
    multi_consensus_tot, multi_consensus_hit = 0, 0
    small_tot, small_hit = 0, 0
    per_obs = {}
    for row in cache:
        name = row["name"]
        keep_idx = [i for i in range(len(row["conf"])) if row["conf"][i] >= conf_thr and row["ref_areas"][i] >= CFG.MIN_AREA]
        kept_masks = [mask_utils.decode(row["ref_rles"][i]) != 0 for i in keep_idx]
        final_masks = apply_ownership_prod(kept_masks)
        pred_rles = mask_utils.encode(np.asfortranarray(np.stack(final_masks, -1).astype(np.uint8))) if final_masks else []
        obs_stat = [0.0, 0, 0, 0]
        iids = file_to_ids[name]
        for iid in iids:
            truth = gt_rles[iid]
            areas = gt_areas[iid]
            is_cons = gt_is_consensus[iid]
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
            gt_covered = (mat.max(axis=1) > 0.5) if (len(truth) > 0 and len(pred_rles) > 0) else np.zeros(len(truth), dtype=bool)
            if len(iids) >= 2 and len(truth) > 0:
                multi_singleton_tot += int((~is_cons).sum())
                multi_singleton_hit += int((gt_covered & ~is_cons).sum())
                multi_consensus_tot += int(is_cons.sum())
                multi_consensus_hit += int((gt_covered & is_cons).sum())
            if len(truth) > 0:
                sm = areas <= CFG.SMALL_GT_MAX_AREA
                small_tot += int(sm.sum())
                small_hit += int((gt_covered & sm).sum())
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
        "multi_singleton_recall": multi_singleton_hit / max(multi_singleton_tot, 1),
        "multi_consensus_recall": multi_consensus_hit / max(multi_consensus_tot, 1),
        "small_recall_final": small_hit / max(small_tot, 1),
        "sec_per_image": round(sec_per_img, 4),
    }, per_obs

refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
_rck = torch.load(REFINER_CKPT_PATH, map_location="cuda", weights_only=False)
refiner.load_state_dict(_rck["best_model"], strict=True)
del _rck

cache_a, sec_a = infer_and_cache_batched(CONTROL_ARM_A_PT, refiner, "arm_a_ctrl_last70")
cache_sib, sec_sib = infer_and_cache_batched(SIB_SHIELD_PT, refiner, "arm_sibling_shield_last70")

res_a, obs_a = score_cache_at_conf(cache_a, CFG.PRIMARY_CONF, sec_a)
res_sib, obs_sib = score_cache_at_conf(cache_sib, CFG.PRIMARY_CONF, sec_sib)

bmode = "obs" if CFG.SPLIT_MODE == "pilot" else "5d_cluster"
boot = paired_bootstrap_pq(obs_a, obs_sib, valid_files, block_mode=bmode, n_boot=CFG.BOOTSTRAP_ITERS, seed=CFG.SEED)
delta_pq = res_sib["pq"] - res_a["pq"]
delta_sparse_pq = res_sib["pq_sparse_r077"] - res_a["pq_sparse_r077"]
fp_rel_change = (res_sib["fp_per_record"] - res_a["fp_per_record"]) / max(res_a["fp_per_record"], 1e-9)

gate_pass = bool((delta_pq >= 0.006) and (boot["ci95_low"] > 0.0)) if CFG.SPLIT_MODE != "pilot" else bool(delta_pq >= 0.008)

sweep_a = {f"conf_{c:.2f}": score_cache_at_conf(cache_a, c, sec_a)[0] for c in CFG.CONF_SWEEP}
sweep_sib = {f"conf_{c:.2f}": score_cache_at_conf(cache_sib, c, sec_sib)[0] for c in CFG.CONF_SWEEP}

manifest_out = {
    "experiment_id": f"v131_sibling_shield_{CFG.SPLIT_MODE}",
    "runtime": RUNTIME_INFO,
    "hashes": {
        "annotations_sha256": file_sha256(TRAIN_JSON),
        "grouped_folds_sha256": folds_sha,
        "time_block_sha256": tb_sha,
        "exclusions_sha256": excl_sha,
        "refiner_sha256": file_sha256(REFINER_CKPT_PATH),
        "arm_a_last_sha256": file_sha256(CONTROL_ARM_A_PT),
        "arm_sibling_shield_last_sha256": file_sha256(SIB_SHIELD_PT),
    },
    "dataset_stats": DS_STATS,
    "arm_a_control": res_a,
    "arm_sibling_shield": res_sib,
    "comparison": {
        "delta_pq_at_026": delta_pq,
        "delta_sparse_pq_r077": delta_sparse_pq,
        "delta_singleton_recall_pt": 100.0 * (res_sib["multi_singleton_recall"] - res_a["multi_singleton_recall"]),
        "delta_consensus_recall_pt": 100.0 * (res_sib["multi_consensus_recall"] - res_a["multi_consensus_recall"]),
        "fp_rel_change": fp_rel_change,
        "bootstrap": boot,
        "gate_pass": gate_pass,
    },
    "conf_sweep_arm_a": sweep_a,
    "conf_sweep_sibling_shield": sweep_sib,
}
manifest_path = CFG.ARTIFACT_ROOT / f"v131_sibling_shield_{CFG.SPLIT_MODE}_manifest.json"
manifest_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== v131 PRIMARY COMPARISON (Sibling-Shield w_sib=0.20 vs Arm A @ conf=0.26, last.pt) ===")
print(f"Arm A (ctrl)     | PQ={res_a['pq']:.5f} sparsePQ={res_a['pq_sparse_r077']:.5f} | TP={res_a['tp']} FP={res_a['fp']} FN={res_a['fn']} | sing_R={res_a['multi_singleton_recall']:.4f} cons_R={res_a['multi_consensus_recall']:.4f}")
print(f"Sibling-Shield   | PQ={res_sib['pq']:.5f} sparsePQ={res_sib['pq_sparse_r077']:.5f} | TP={res_sib['tp']} FP={res_sib['fp']} FN={res_sib['fn']} | sing_R={res_sib['multi_singleton_recall']:.4f} cons_R={res_sib['multi_consensus_recall']:.4f}")
print(json.dumps(manifest_out["comparison"], indent=2, ensure_ascii=False))
print("=== CONFIDENCE SWEEP (PQ / sparsePQ) ===")
for c in CFG.CONF_SWEEP:
    k = f"conf_{c:.2f}"
    print(f"  conf={c:.2f} | Arm A={sweep_a[k]['pq']:.5f} | SibShield={sweep_sib[k]['pq']:.5f} ({sweep_sib[k]['pq'] - sweep_a[k]['pq']:+.5f}) | ΔsparsePQ={sweep_sib[k]['pq_sparse_r077'] - sweep_a[k]['pq_sparse_r077']:+.5f}")
print("Saved manifest:", manifest_path)
"""

    cells = [
        _cell("markdown", md_intro),
        _cell("code", cell1_cfg),
        _cell("code", cell2_modules),
        _cell("code", cell3_data),
        _cell("code", cell4_ds),
        _cell("code", cell5_train),
        _cell("code", cell6_peers),
        _cell("code", cell7_eval),
    ]
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            code_txt = "".join(ln for ln in c["source"] if not ln.lstrip().startswith("%"))
            ast.parse(code_txt, filename=f"v131_cell_{idx}.py")

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
    out_path = ROOT / "solar_filament_v131_sibling_shield_colab.ipynb"
    nb = build_notebook()
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {out_path.name} ({len(nb['cells'])} cells, AST verified)")


if __name__ == "__main__":
    main()
