"""Build v128 Colab notebook: 3-Fold OOF Geodesic Hysteresis (Barbs/Tips) & Merge-Split Eval.

Zero retraining (~8 min total on Colab GPU). Evaluates Idea 3 & Idea 4 across all 3 OOF folds
using each fold's strictly out-of-fold ``runs/yolo_fold_{f}/weights/best.pt`` and strictly
out-of-fold ``mediseg_edgeattnet_refiner_fold{f}_resume.pt``:

- Reports metrics cleanly split into:
  1. **Development 3-fold OOF** (547 obs, 2011–2016) — used for arm selection without touching locked holdout
  2. **Locked Temporal Holdout** (160 obs, 2017–2022, 58 independent 5-day clusters) — one-shot temporal confirmation
  3. **Full 707-obs 3-fold OOF** — direct comparison against historical RESEARCH_LOG 3-fold baselines
- Caches per-fold compact ``(box, raw_rle, prob_uint8)`` on local SSD so sweeping all 7 arms
  plus observation/cluster paired bootstrap takes seconds after a single GPU pass.

Usage:
    python scratch/build_v128_hysteresis_split_colab.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _strip_module_for_inline(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "from __future__ import annotations":
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


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
    md_intro = """# v128 — 条件付き Geodesic Hysteresis 境界拡張 (Barb/先端復元) & 併合提案 Split 3-Fold OOF 検証

**学習ゼロ・Colab GPU 約 8〜10 分で全 3-fold (707 観測 / 1,154 注釈レコード) 完結**。

### 狙う誤差モード
1. **Idea 3 — Geodesic Hysteresis 境界拡張 (`hysteresis_expand_mask`)**:
   - 主催者回答 (Topic 753665) で「視認できる barb（棘・毛状突起）も正解マスクに含める」と明言された一方、現行 `prod` は `refined & proposal` で YOLO 1/4 解像度 proto マスクの外側を無条件に切り落としている（`near_miss` `0.3 ≤ IoU ≤ 0.5` が FN 817 件 + FP 588 件）。
   - `proposal` 内のコア領域 (`prob ≥ 0.45`) と**8近傍連結している成分に限り**、`proposal` 外縁 `max_expand_px` まで `prob ≥ high_thr` の画素を連結拡張する（隣接する別フィラメントの混入を幾何的に遮断）。
2. **Idea 4 — 並走フィラメント過併合の分離 (`split_disconnected_merges`)**:
   - YOLO が近接 2 個体を 1 つの bbox に包み、Refiner が隙間を分離できても単一 RLE として出力される過併合（約 250 件）に対し、主成分の主軸から横方向にオフセットした高確信の第 2 連結成分のみを別個体へ分離する（同一チャネル上の途切れは維持）。

### リーク防止の評価設計
- 各 fold `f ∈ {0, 1, 2}` について、厳密 OOF の `runs/yolo_fold_{f}/weights/best.pt` と `mediseg_edgeattnet_refiner_fold{f}_resume.pt` のみを使用。
- `development` (2011–2016, **547 obs**) で最良 arm を選び、独立な `locked_evaluation` (2017–2022, **160 obs**, 58 時間クラスタ) で一発確認する 2 段階判定を同時出力。
"""

    cell1_cfg = """# [1/5] 依存・Drive マウント・設定
%pip install -q "ultralytics==8.4.118" pycocotools
import hashlib, json, math, os, pickle, shutil, sys, time
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
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v128")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PROB_CACHE_DIR = LOCAL_ROOT / "oof_prob_cache"
    MANIFEST = ARTIFACT_ROOT / "v128_hysteresis_split_3fold_manifest.json"

    FOLDS = (0, 1, 2)
    YOLO_IMGSZ = 1280
    PRIMARY_CONF = 0.26
    PREDICT_IOU = 0.55
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    LOW_THR = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    SPARSE_R = 0.77
    BOOTSTRAP_ITERS = 2000

    # 事前固定 7 arms (過剰探索を排したコンパクト設計)
    ARMS = {
        "prod":                 {"high_thr": 1.0,  "max_expand_px": 0,  "split": False},
        "hyst_h72_r8":          {"high_thr": 0.72, "max_expand_px": 8,  "split": False},
        "hyst_h65_r10":         {"high_thr": 0.65, "max_expand_px": 10, "split": False},
        "hyst_h58_r14":         {"high_thr": 0.58, "max_expand_px": 14, "split": False},
        "split_a160_p14":       {"high_thr": 1.0,  "max_expand_px": 0,  "split": True, "min_sec_area": 160, "min_sec_prob": 0.62, "min_perp": 14.0},
        "split_a260_p18":       {"high_thr": 1.0,  "max_expand_px": 0,  "split": True, "min_sec_area": 260, "min_sec_prob": 0.68, "min_perp": 18.0},
        "combo_hyst72_split160":{"high_thr": 0.72, "max_expand_px": 8,  "split": True, "min_sec_area": 160, "min_sec_prob": 0.62, "min_perp": 14.0},
    }

for p in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.PROB_CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
print({"gpu": torch.cuda.get_device_name(0), "ultralytics": version("ultralytics")})
"""

    mediseg_src = _strip_module_for_inline(
        ROOT / "solar_filament" / "models" / "mediseg_models.py"
    )
    hyst_src = _strip_module_for_inline(
        ROOT / "solar_filament" / "pipeline" / "hysteresis_split.py"
    )
    cell2_models = (
        "# [2/5] MediSegEdgeAttUNet + Hysteresis / Merge-Split モジュール\n"
        + mediseg_src
        + "\n"
        + hyst_src
    )

    cell3_data = """# [3/5] COCO・3-fold 契約検証・前処理画像ステージング
def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

TRAIN_JSON = next(Path(CFG.DATA_CACHE).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
coco = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for iid, info in images.items():
    file_to_ids[info["file_name"]].append(iid)
train_files = sorted(file_to_ids)

folds_path = CFG.ARTIFACT_ROOT / "grouped_folds.csv"
folds_sha = file_sha256(folds_path)
assert folds_sha == CFG.GROUPED_FOLDS_SHA256, f"folds SHA mismatch: {folds_sha}"
fold_df = pd.read_csv(folds_path)
fold_of_file = dict(zip(fold_df["file_name"], fold_df["fold"].astype(int)))
role_of_file = {n: ("development" if int(n[:4]) <= 2016 else "locked_evaluation") for n in train_files}

for name in tqdm(train_files, desc="前処理画像をローカル SSD へコピー"):
    dst = CFG.PREPROC_DIR / name
    if not dst.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, dst)

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

gt_rles = {iid: [to_rle(a, int(info["height"]), int(info["width"]))
                 for a in annotations[iid] if not a.get("iscrowd", 0)]
           for iid, info in images.items()}
print({"total_obs": len(train_files),
       "dev_obs": sum(v == "development" for v in role_of_file.values()),
       "locked_obs": sum(v == "locked_evaluation" for v in role_of_file.values())})
"""

    cell4_cache = """# [4/5] 3-Fold OOF YOLO + EdgeAtt Refiner 推論キャッシュ (uint8 確率マップ保持)
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

FOLD_WEIGHT_SHAS = {}
OOF_PROPOSALS = {}

for f in CFG.FOLDS:
    yolo_pt = CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{f}" / "weights" / "best.pt"
    ref_pt = CFG.ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{f}_resume.pt"
    assert yolo_pt.exists() and ref_pt.exists(), (yolo_pt, ref_pt)
    FOLD_WEIGHT_SHAS[f"fold_{f}"] = {"yolo": file_sha256(yolo_pt), "refiner": file_sha256(ref_pt)}

    cache_pkl = CFG.PROB_CACHE_DIR / f"fold{f}_oof_probs.pkl"
    if cache_pkl.exists():
        fold_rows = pickle.loads(cache_pkl.read_bytes())
        for r in fold_rows:
            OOF_PROPOSALS[r["name"]] = r
        print(f"再利用: fold {f} キャッシュ ({len(fold_rows)} obs)")
        continue

    yolo_model = YOLO(str(yolo_pt))
    refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
    ck = torch.load(ref_pt, map_location="cuda", weights_only=False)
    refiner.load_state_dict(ck.get("best_model") or ck["model"], strict=True)
    del ck

    f_files = [n for n in train_files if fold_of_file[n] == f]
    fold_rows = []
    for name in tqdm(f_files, desc=f"fold {f} OOF YOLO+Refiner"):
        img_path = CFG.PREPROC_DIR / name
        res = yolo_model.predict(str(img_path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.PRIMARY_CONF,
                                 iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False)[0]
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
        with torch.no_grad():
            for i, pm in enumerate(masks):
                if pm.sum() < CFG.MIN_AREA:
                    continue
                x0, y0, x1, y1 = mask_square_box(pm)
                crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                ten = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
                prob_320 = refiner(ten).sigmoid()[0, 0].cpu().numpy()
                items.append({
                    "conf": float(conf[i]),
                    "raw_rle": mask_utils.encode(np.asfortranarray(pm.astype(np.uint8))),
                    "box": (x0, y0, x1, y1),
                    "prob_320_u8": np.clip(np.rint(prob_320 * 255.0), 0, 255).astype(np.uint8),
                })
        row = {"name": name, "fold": f, "items": items}
        fold_rows.append(row)
        OOF_PROPOSALS[name] = row
    cache_pkl.write_bytes(pickle.dumps(fold_rows, protocol=pickle.HIGHEST_PROTOCOL))
    del yolo_model, refiner; torch.cuda.empty_cache()
print("全 3-fold OOF 推論完了:", len(OOF_PROPOSALS))
"""

    cell5_eval = """# [5/5] 全 7 Arm 評価 (dev 547 / locked 160 / full 707) + 対応あり Bootstrap 検証
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

def paired_bootstrap(per_obs_ref, per_obs_arm, obs_names, block_mode="obs", n_boot=2000, seed=3407):
    rng = np.random.default_rng(seed)
    units = [[n] for n in obs_names] if block_mode == "obs" else build_five_day_clusters(obs_names)
    n_u = len(units)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_u, size=n_u)
        sr, sa = [0.0, 0, 0, 0], [0.0, 0, 0, 0]
        for ui in pick:
            for nm in units[ui]:
                for k in range(4):
                    sr[k] += per_obs_ref[nm][k]
                    sa[k] += per_obs_arm[nm][k]
        dr = sr[1] + 0.5 * sr[2] + 0.5 * sr[3]
        da = sa[1] + 0.5 * sa[2] + 0.5 * sa[3]
        deltas[b] = (sa[0] / da if da > 0 else 0.0) - (sr[0] / dr if dr > 0 else 0.0)
    return {
        "n_units": n_u,
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }

def aggregate_subset(per_obs, obs_subset):
    s = [0.0, 0, 0, 0]
    for n in obs_subset:
        for k in range(4):
            s[k] += per_obs[n][k]
    den = s[1] + 0.5 * s[2] + 0.5 * s[3]
    pq = s[0] / den if den > 0 else 0.0
    sq = s[0] / s[1] if s[1] else 0.0
    rq = s[1] / den if den > 0 else 0.0
    return {"pq": float(pq), "sq": float(sq), "rq": float(rq), "tp": int(s[1]), "fp": int(s[2]), "fn": int(s[3])}

dev_files = [n for n in train_files if role_of_file[n] == "development"]
locked_files = [n for n in train_files if role_of_file[n] == "locked_evaluation"]

PER_OBS_BY_ARM = {}
SUMMARY_BY_ARM = {}

for arm_name, params in CFG.ARMS.items():
    per_obs = {}
    n_split_events = 0
    for name in tqdm(train_files, desc=f"採点 {arm_name}"):
        row = OOF_PROPOSALS[name]
        cand_masks = []
        for it in row["items"]:
            pm = mask_utils.decode(it["raw_rle"]) != 0
            x0, y0, x1, y1 = it["box"]
            prob = cv2.resize(it["prob_320_u8"].astype(np.float32) / 255.0,
                              (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            m = hysteresis_expand_mask(
                pm, prob, (x0, y0, x1, y1),
                low_thr=CFG.LOW_THR,
                high_thr=params["high_thr"],
                max_expand_px=params["max_expand_px"],
            )
            if m.sum() < CFG.MIN_AREA:
                continue
            if params.get("split", False):
                parts = split_disconnected_merges(
                    m, prob, (x0, y0, x1, y1),
                    min_secondary_area=params["min_sec_area"],
                    min_secondary_prob=params["min_sec_prob"],
                    min_perp_offset_px=params["min_perp"],
                )
                if len(parts) > 1:
                    n_split_events += len(parts) - 1
                cand_masks.extend(parts)
            else:
                cand_masks.append(m)
        final_masks = apply_ownership_prod(cand_masks)
        pred_rles = mask_utils.encode(np.asfortranarray(np.stack(final_masks, -1).astype(np.uint8))) if final_masks else []
        obs_stat = [0.0, 0, 0, 0]
        for iid in file_to_ids[name]:
            truth = gt_rles[iid]
            mat = iou_matrix(truth, pred_rles)
            if len(truth) == 0:
                obs_stat[2] += len(pred_rles)
            elif len(pred_rles) == 0:
                obs_stat[3] += len(truth)
            else:
                hits = mat > 0.5
                obs_stat[0] += float(mat[hits].sum())
                obs_stat[1] += int(hits.sum())
                obs_stat[2] += int((hits.sum(axis=0) == 0).sum())
                obs_stat[3] += int((hits.sum(axis=1) == 0).sum())
        per_obs[name] = tuple(obs_stat)

    PER_OBS_BY_ARM[arm_name] = per_obs
    SUMMARY_BY_ARM[arm_name] = {
        "params": params,
        "split_extra_instances": n_split_events,
        "dev_547": aggregate_subset(per_obs, dev_files),
        "locked_160": aggregate_subset(per_obs, locked_files),
        "full_707": aggregate_subset(per_obs, train_files),
        "by_fold": {
            f"fold_{f}": aggregate_subset(per_obs, [n for n in train_files if fold_of_file[n] == f])
            for f in CFG.FOLDS
        },
    }

prod_obs = PER_OBS_BY_ARM["prod"]
for arm_name, entry in SUMMARY_BY_ARM.items():
    if arm_name == "prod":
        continue
    obs_arm = PER_OBS_BY_ARM[arm_name]
    entry["delta_dev_vs_prod"] = entry["dev_547"]["pq"] - SUMMARY_BY_ARM["prod"]["dev_547"]["pq"]
    entry["delta_locked_vs_prod"] = entry["locked_160"]["pq"] - SUMMARY_BY_ARM["prod"]["locked_160"]["pq"]
    entry["delta_full_vs_prod"] = entry["full_707"]["pq"] - SUMMARY_BY_ARM["prod"]["full_707"]["pq"]
    entry["boot_dev_obs"] = paired_bootstrap(prod_obs, obs_arm, dev_files, "obs", CFG.BOOTSTRAP_ITERS)
    entry["boot_locked_5d"] = paired_bootstrap(prod_obs, obs_arm, locked_files, "5d_cluster", CFG.BOOTSTRAP_ITERS)

manifest = {
    "experiment_id": "v128_hysteresis_and_merge_split_3fold",
    "weights_sha256": FOLD_WEIGHT_SHAS,
    "folds_sha256": folds_sha,
    "results": SUMMARY_BY_ARM,
}
CFG.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== SUMMARY TABLE (dev 547 -> locked 160 -> full 707) ===")
for k, v in SUMMARY_BY_ARM.items():
    d_dev = v.get("delta_dev_vs_prod", 0.0)
    d_lck = v.get("delta_locked_vs_prod", 0.0)
    d_all = v.get("delta_full_vs_prod", 0.0)
    print(f"{k:24s} | dev={v['dev_547']['pq']:.5f} ({d_dev:+.5f}) | "
          f"locked={v['locked_160']['pq']:.5f} ({d_lck:+.5f}) | "
          f"full3f={v['full_707']['pq']:.5f} ({d_all:+.5f}) | splits={v['split_extra_instances']}")
print("Saved manifest:", CFG.MANIFEST)
"""

    cells = [
        _cell("markdown", md_intro),
        _cell("code", cell1_cfg),
        _cell("code", cell2_models),
        _cell("code", cell3_data),
        _cell("code", cell4_cache),
        _cell("code", cell5_eval),
    ]
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            code_txt = "".join(
                ln for ln in c["source"] if not ln.lstrip().startswith("%")
            )
            ast.parse(code_txt, filename=f"v128_cell_{idx}.py")

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
    out_path = ROOT / "solar_filament_v128_hysteresis_split_colab.ipynb"
    nb = build_notebook()
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {out_path.name} ({len(nb['cells'])} cells, AST verified)")


if __name__ == "__main__":
    main()
