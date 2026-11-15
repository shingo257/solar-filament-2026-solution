"""Build v135 Colab notebook: 4-Way Pilot Checkpoint Audit (best.pt vs last.pt) & Orthogonal Fusion.

Evaluates all 8 already-completed pilot checkpoints on Google Drive (~6 min on Colab GPU, ZERO training):
- Arm A (Control, v127): best.pt & last.pt
- Arm B (Spine Aux-Teacher 0.5, v127): best.pt & last.pt
- Small-TAL (max_boost=4.0, v130): best.pt & last.pt
- Sibling-Shield (sib_neg_weight=0.20, v131): best.pt & last.pt

And tests 1-to-1 continuous multi-detector fusion across our strong orthogonal detectors
(replacing weak croptrain from v132):
1. ``spine_plus_tal`` (0.50 * Spine-Aux + 0.50 * Small-TAL)
2. ``trio_base_spine_tal`` ((Arm A + Spine-Aux + Small-TAL) / 3)
3. ``cons2of3_strong`` (discrete >=2 of 3 @ 0.26 across Arm A, Spine-Aux, Small-TAL)
4. ``quad_wmean`` (0.35 * Small-TAL + 0.35 * Spine-Aux + 0.15 * Arm A + 0.15 * Sibling-Shield)

Usage:
    python scratch/build_v135_pilot_4way_audit_colab.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDISEG_PATH = ROOT / "solar_filament" / "models" / "mediseg_models.py"
CONSENSUS_PATH = ROOT / "solar_filament" / "pipeline" / "consensus_fusion.py"
EXCLUSIONS_PATH = ROOT / "output" / "s0" / "spine_aux_prereg_exclusions_2026-11-13.csv"


def _cell(cell_type: str, source: str) -> dict:
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def _strip_package_imports(code_txt: str) -> str:
    kept = []
    for ln in code_txt.splitlines():
        if ln.startswith(("from solar_filament.", "import solar_filament.")):
            continue
        if ln.strip() == "from __future__ import annotations":
            continue
        kept.append(ln)
    return "\n".join(kept)


def build_notebook() -> dict:
    mediseg_src = _strip_package_imports(MEDISEG_PATH.read_text(encoding="utf-8"))
    cons_src = _strip_package_imports(CONSENSUS_PATH.read_text(encoding="utf-8"))
    exclusions_text = EXCLUSIONS_PATH.read_text(encoding="utf-8")

    md = """# v135 — 全 4 Pilot モデル (`best.pt` / `last.pt`) 一括監査 & 直交アンサンブル検証 (学習ゼロ・約 6 分)

既に Drive 上で 70 epoch 完走済みの 4 手法 × 2 重み（計 8 チェックポイント）を同一 180 枚の pilot val で一括採点します：
1. **Arm A (`v127` Control)**
2. **Arm B (`v127` Spine Aux-Teacher `0.5`)** — `last.pt` で `+1.29 pt` PQ・FP `-1.15%`
3. **Small-TAL (`v130` `max_boost=4.0`)** — `last.pt` で `+1.41 pt` PQ・小個体 recall `+6.08 pt`
4. **Sibling-Shield (`v131` `sib_neg_weight=0.20`)** — `last.pt` で `+0.85 pt` PQ・単独個体 recall `+6.41 pt`

**検証する 2 つの核心命題**:
- `best.pt` 同士の比較でも `Spine-Aux`・`Small-TAL`・`Sibling-Shield` が `Arm A best.pt` を上回るか？
- `v132` で足を引っ張っていた弱い `croptrain` の代わりに、**強い直交検出器同士（`Spine-Aux + Small-TAL` 2 検出器／`+ Arm A` 3 検出器／`+ Sibling-Shield` 4 検出器）を 1-to-1 連続融合**したとき PQ・Sparse PQ (`r=0.77`) がどこまで伸びるか？
"""

    c1 = """# [1/3] Drive マウント・8 チェックポイント存在確認
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
LOCAL_PREPROC = Path("/content/solar_v135/preprocessed")
LOCAL_PREPROC.mkdir(parents=True, exist_ok=True)

REFINER_CKPT = DRIVE_ROOT / "runs" / "v127_spine_ab" / "refiner_edgeatt_pilot_fit365.pt"
CKPTS = {
    "arm_a": {
        "best": DRIVE_ROOT / "runs" / "v127_spine_ab" / "yolo_pilot_arm_a" / "weights" / "best.pt",
        "last": DRIVE_ROOT / "runs" / "v127_spine_ab" / "yolo_pilot_arm_a" / "weights" / "last.pt",
    },
    "spine_aux": {
        "best": DRIVE_ROOT / "runs" / "v127_spine_ab" / "yolo_pilot_arm_b" / "weights" / "best.pt",
        "last": DRIVE_ROOT / "runs" / "v127_spine_ab" / "yolo_pilot_arm_b" / "weights" / "last.pt",
    },
    "small_tal": {
        "best": DRIVE_ROOT / "runs" / "v130_small_tal" / "yolo_pilot_arm_small_tal" / "weights" / "best.pt",
        "last": DRIVE_ROOT / "runs" / "v130_small_tal" / "yolo_pilot_arm_small_tal" / "weights" / "last.pt",
    },
    "sib_shield": {
        "best": DRIVE_ROOT / "runs" / "v131_sibling_shield" / "yolo_pilot_arm_sibling_shield" / "weights" / "best.pt",
        "last": DRIVE_ROOT / "runs" / "v131_sibling_shield" / "yolo_pilot_arm_sibling_shield" / "weights" / "last.pt",
    },
}

assert REFINER_CKPT.exists(), f"Missing refiner: {REFINER_CKPT}"
for mname, d in CKPTS.items():
    for tag, p in d.items():
        assert p.exists(), f"Missing {mname}/{tag}: {p}"
print("All 8 pilot YOLO checkpoints + common Refiner verified on Drive.")
"""

    c2 = f'''# [2/3] 180 枚 Pilot 分割固定・モデル・1-to-1 連続融合モジュール
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

for n in tqdm(valid_files, desc="前処理画像をローカル SSD へ"):
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
print("Verified exact 180 pilot val observations.")
'''

    c3 = """# [3/3] 8 チェックポイント高速バッチ推論 + 単独比較 + 直交アンサンブル評価
def sparse_label_pq(sum_iou, tp, fp, fn, r=0.77):
    den = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return float(r * sum_iou / den) if den > 0 else 0.0

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
    cache = {}
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
        cache[name] = (vcf, [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in vms],
                       [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in rms])
    del m; torch.cuda.empty_cache()
    return cache

def score_pred_map(pred_masks_by_file):
    s_iou = 0.0; tp = 0; fp = 0; fn = 0; obs = {}
    for name in valid_files:
        preds = apply_own(pred_masks_by_file[name])
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
    return {
        "pq": float(s_iou / den),
        "pq_sparse_r077": sparse_label_pq(s_iou, tp, fp, fn, 0.77),
        "sq": float(s_iou / tp) if tp else 0.0,
        "rq": float(tp / den),
        "tp": tp, "fp": fp, "fn": fn,
    }, obs

def eval_single(cache, thr=0.26):
    pm = {n: [mask_utils.decode(cache[n][2][i]) != 0 for i in range(len(cache[n][0])) if cache[n][0][i] >= thr]
          for n in valid_files}
    return score_pred_map(pm)

def eval_weighted_fusion(caches_by_det, weights_dict, thr, discrete_vote=None):
    pm = {}
    for n in valid_files:
        props = []
        for dname, cdict in caches_by_det.items():
            cf, raw_r, _ = cdict[n]
            for i in range(len(cf)):
                props.append(DetectorProposal(dname, float(cf[i]), raw_r[i], i))
        clusters = match_detector_proposals_1to1(props, iou_threshold=0.45)
        chosen = []
        for cl in clusters:
            if discrete_vote is not None:
                ok = cl.count_above(thr) >= discrete_vote
            else:
                ok = cl.weighted_score(weights_dict) >= thr
            if ok:
                bm = cl.best_member()
                chosen.append(mask_utils.decode(caches_by_det[bm.det_name][n][2][bm.prop_idx]) != 0)
        pm[n] = chosen
    return score_pred_map(pm)

def boot(ob, oc):
    rng = np.random.default_rng(3407)
    ba = np.array([ob[f] for f in valid_files], np.float64); ca = np.array([oc[f] for f in valid_files], np.float64)
    idx = rng.integers(0, len(valid_files), size=(2000, len(valid_files)))
    bs = ba[idx].sum(1); cs = ca[idx].sum(1)
    d = cs[:, 0] / (cs[:, 1] + 0.5 * cs[:, 2] + 0.5 * cs[:, 3]) - bs[:, 0] / (bs[:, 1] + 0.5 * bs[:, 2] + 0.5 * bs[:, 3])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_pos": float((d > 0).mean())}

CACHES = {"best": {}, "last": {}}
for tag in ("best", "last"):
    for mname in ("arm_a", "spine_aux", "small_tal", "sib_shield"):
        CACHES[tag][mname] = cache_det(CKPTS[mname][tag], f"{mname}_{tag}")

report = {"single": {"best": {}, "last": {}}, "fusion": {"best": {}, "last": {}}}

for tag in ("best", "last"):
    print(f"\\n================ SINGLE MODELS ({tag}.pt) ================")
    base_res, base_obs = eval_single(CACHES[tag]["arm_a"], 0.26)
    for mname in ("arm_a", "spine_aux", "small_tal", "sib_shield"):
        sweep = {}
        for t in (0.22, 0.24, 0.26, 0.28, 0.30):
            r, o = eval_single(CACHES[tag][mname], t)
            sweep[f"{t:.2f}"] = r
            if abs(t - 0.26) < 1e-6:
                r026, o026 = r, o
        bt = boot(base_obs, o026)
        best_t = max(sweep, key=lambda k: sweep[k]["pq"])
        report["single"][tag][mname] = {"at_026": r026, "sweep": sweep, "best_thr": best_t, "boot_vs_arm_a_026": bt}
        print(f"{mname:12s} @0.26 | PQ={r026['pq']:.5f} ({r026['pq']-base_res['pq']:+.5f}) | sparse={r026['pq_sparse_r077']:.5f} | TP={r026['tp']} FP={r026['fp']} FN={r026['fn']} | P(>0)={bt['p_pos']:.4f} | peak@{best_t}={sweep[best_t]['pq']:.5f}")

    print(f"\\n================ ORTHOGONAL FUSIONS ({tag}.pt vs arm_a_{tag} @0.26) ================")
    fusions = [
        ("spine_tal_t20", {"spine_aux": 0.5, "small_tal": 0.5}, 0.20, None),
        ("spine_tal_t22", {"spine_aux": 0.5, "small_tal": 0.5}, 0.22, None),
        ("spine_tal_t24", {"spine_aux": 0.5, "small_tal": 0.5}, 0.24, None),
        ("trio_a_sp_tal_t20", {"arm_a": 1/3, "spine_aux": 1/3, "small_tal": 1/3}, 0.20, None),
        ("trio_a_sp_tal_t22", {"arm_a": 1/3, "spine_aux": 1/3, "small_tal": 1/3}, 0.22, None),
        ("cons2of3_strong_c26", {"arm_a": 1.0, "spine_aux": 1.0, "small_tal": 1.0}, 0.26, 2),
        ("quad_wmean_t21", {"small_tal": 0.35, "spine_aux": 0.35, "arm_a": 0.15, "sib_shield": 0.15}, 0.21, None),
        ("quad_wmean_t23", {"small_tal": 0.35, "spine_aux": 0.35, "arm_a": 0.15, "sib_shield": 0.15}, 0.23, None),
    ]
    for fname, wdict, thr, dvote in fusions:
        sub_c = {k: CACHES[tag][k] for k in wdict}
        fr, fo = eval_weighted_fusion(sub_c, wdict, thr, discrete_vote=dvote)
        bt = boot(base_obs, fo)
        report["fusion"][tag][fname] = {"metrics": fr, "boot_vs_arm_a_026": bt}
        print(f"{fname:21s} | PQ={fr['pq']:.5f} ({fr['pq']-base_res['pq']:+.5f}) | sparse={fr['pq_sparse_r077']:.5f} ({fr['pq_sparse_r077']-base_res['pq_sparse_r077']:+.5f}) | TP={fr['tp']} FP={fr['fp']} FN={fr['fn']} | CI95=[{bt['ci95'][0]:+.5f}, {bt['ci95'][1]:+.5f}] P(>0)={bt['p_pos']:.4f}")

mp = ARTIFACT_ROOT / "v135_pilot_4way_audit_manifest.json"
mp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("\\nSaved manifest:", mp)
"""

    cells = [_cell("markdown", md), _cell("code", c1), _cell("code", c2), _cell("code", c3)]
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            txt = "".join(ln for ln in c["source"] if not ln.lstrip().startswith("%"))
            ast.parse(txt, filename=f"v135_cell_{idx}.py")
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
    out = ROOT / "solar_filament_v135_pilot_4way_audit_colab.ipynb"
    nb = build_notebook()
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {out.name} ({len(nb['cells'])} cells, AST verified)")


if __name__ == "__main__":
    main()
