"""Build v136 Colab notebook: Honest 2-Fold (471-Obs) Strong Orthogonal Multi-Detector Fusion.

Upgrades v132 by replacing weak ``croptrain`` (Fold-0 mAP50=0.598) with our newly trained
honest OOF detectors:
- ``prod``: ``yolo_fold_0`` / ``yolo_fold_2`` (YOLO11m baseline)
- ``yolo26``: ``v121_yolo26_f0`` / ``v117_yolo26`` (YOLO26m architecture diversity)
- ``small_tal``: ``v133a`` / ``v133b`` (Small-Instance TAL Anchor Equalization, max_boost=4.0)
- ``spine_small_tal``: ``v134a`` / ``v134b`` (Spine Aux-Teacher 0.5 + Small-TAL 4.0, optional if still training)

Speed & Persistence Design:
- Caches ``prod`` and ``yolo26`` refined proposals into ``artifacts/v136_oof_caches/`` on Drive
  (same tuple schema as ``v133_oof_caches`` and ``v134_oof_caches``).
- Pre-computes ``match_detector_proposals_1to1`` ONCE per active detector subset per image,
  making 30+ continuous fusion arm evaluations across 471 observations run in ~1-2 minutes.
- Runs strict out-of-fold cross-selection (select on Fold 2 -> blind Fold 0, and vice versa)
  for BOTH standard ``pq`` and sparse-label ``pq_sparse_r077`` (Public LB proxy).

Usage:
    python scratch/build_v136_strong_fusion_2fold_colab.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDISEG_PATH = ROOT / "solar_filament" / "models" / "mediseg_models.py"
CONSENSUS_PATH = ROOT / "solar_filament" / "pipeline" / "consensus_fusion.py"


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

    md = """# v136 — 2-Fold Honest (471 観測) 強検出器 1対1 連続融合 & 交差盲検セレクタ

`v132` で Fold-0 側の足を引っ張っていた `croptrain` (`mAP50=0.598`) を外し、漏洩なしの強検出器群を統合評価します：
- **`prod`**: `yolo_fold_0` / `yolo_fold_2`
- **`yolo26`**: `v121_yolo26_f0` / `v117_yolo26`
- **`small_tal`**: `v133a` / `v133b` (`artifacts/v133_oof_caches` 自動再利用)
- **`spine_small_tal`**: `v134a` / `v134b` (`artifacts/v134_oof_caches` が揃っていれば自動で 4 検出器モードに拡張、学習中なら 3 検出器モードで即実行可能)

`prod` と `yolo26` の推論結果も `artifacts/v136_oof_caches/` に永続化するため、2 回目以降（`v134` 完走後の再実行など）は **GPU 推論ゼロ・約 2 分** で完了します。
"""

    c1 = """# [1/4] 環境・Drive マウント・キャッシュ/重み確認
%pip install -q "ultralytics==8.4.120" pycocotools
import hashlib, json, math, os, pickle, shutil, tempfile, time
from collections import defaultdict
from dataclasses import dataclass
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

DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
DATA_CACHE = DRIVE_ROOT / "competition_data"
ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
LOCAL_PREPROC = Path("/content/solar_v136/preprocessed")
LOCAL_PREPROC.mkdir(parents=True, exist_ok=True)

V136_CACHE_DIR = ARTIFACT_ROOT / "v136_oof_caches"
V136_CACHE_DIR.mkdir(parents=True, exist_ok=True)
V133_CACHE_DIR = ARTIFACT_ROOT / "v133_oof_caches"
V134_CACHE_DIR = ARTIFACT_ROOT / "v134_oof_caches"
V138_CACHE_DIR = ARTIFACT_ROOT / "v138_oof_caches"

GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
FOLDS = (0, 2)

BASE_CKPTS = {
    0: {
        "prod": DRIVE_ROOT / "runs" / "yolo_fold_0" / "weights" / "best.pt",
        "yolo26": DRIVE_ROOT / "runs" / "v121_yolo26_f0" / "yolo_yolo26m_fold0" / "weights" / "best.pt",
        "refiner": ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold0_resume.pt",
    },
    2: {
        "prod": DRIVE_ROOT / "runs" / "yolo_fold_2" / "weights" / "best.pt",
        "yolo26": DRIVE_ROOT / "runs" / "v117_yolo26" / "yolo_yolo26m_fold2" / "weights" / "best.pt",
        "refiner": ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold2_resume.pt",
    },
}

for f, mp in BASE_CKPTS.items():
    for k, pt in mp.items():
        assert pt.exists(), f"Missing fold {f} {k}: {pt}"

has_v133_best = all((V133_CACHE_DIR / f"small_tal_f{f}_best_cache.pkl").exists() for f in FOLDS)
has_v133_last = all((V133_CACHE_DIR / f"small_tal_f{f}_last_cache.pkl").exists() for f in FOLDS)
has_v134_best = all((V134_CACHE_DIR / f"spine_small_tal_f{f}_best_cache.pkl").exists() for f in FOLDS)
has_v134_last = all((V134_CACHE_DIR / f"spine_small_tal_f{f}_last_cache.pkl").exists() for f in FOLDS)
has_v138_best = all((V138_CACHE_DIR / f"spine_aux_f{f}_best_cache.pkl").exists() for f in FOLDS)
has_v138_last = all((V138_CACHE_DIR / f"spine_aux_f{f}_last_cache.pkl").exists() for f in FOLDS)

assert has_v133_best, "v133a/v133b の OOF キャッシュがまだ揃っていません (v133_oof_caches)"
print({
    "has_v133_best": has_v133_best,
    "has_v133_last": has_v133_last,
    "has_v134_best": has_v134_best,
    "has_v134_last": has_v134_last,
    "has_v138_best": has_v138_best,
    "has_v138_last": has_v138_last,
})
"""

    c2 = (
        "# [2/4] MediSegEdgeAttUNet & 1-to-1 Consensus Fusion モジュール定義\n"
        + mediseg_src
        + "\n\n"
        + cons_src
    )

    c3 = """# [3/4] GT 読込 & prod / yolo26 キャッシュ構築 (既存なら Drive から即時ロード)
def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sparse_label_pq(sum_iou, tp, fp, fn, r=0.77):
    den = r * tp + 0.5 * fp + 0.5 * (1.0 - r) * tp + 0.5 * r * fn
    return float(r * sum_iou / den) if den > 0 else 0.0

TRAIN_JSON = next(DATA_CACHE.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
coco = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
images = {im["id"]: im for im in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for iid, im in images.items():
    file_to_ids[im["file_name"]].append(iid)
train_files = sorted(file_to_ids)

folds_path = ARTIFACT_ROOT / "grouped_folds.csv"
assert file_sha256(folds_path) == GROUPED_FOLDS_SHA256
fold_df = pd.read_csv(folds_path)
fold_of_file = dict(zip(fold_df["file_name"], fold_df["fold"].astype(int)))
eval_files = {f: [n for n in train_files if fold_of_file[n] == f] for f in FOLDS}
all_files = eval_files[0] + eval_files[2]
assert len(eval_files[0]) == 236 and len(eval_files[2]) == 235

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

gt_rles = {
    iid: [to_rle(a, int(im["height"]), int(im["width"])) for a in annotations[iid] if not a.get("iscrowd", 0)]
    for iid, im in images.items()
}

def sq_box(mask, context=1.65, minimum=96):
    h, w = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, w, h
    cx, cy = (xs.min() + xs.max() + 1) / 2.0, (ys.min() + ys.max() + 1) / 2.0
    side = min(max(int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context)), minimum), min(h, w))
    x0, y0 = round(cx - side / 2.0), round(cy - side / 2.0)
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= x1 - w; x1 = w
    if y1 > h: y0 -= y1 - h; y1 = h
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

def refine_batch(refiner, img_bgr, masks):
    if not masks:
        return []
    boxes = [sq_box(m) for m in masks]
    crops = np.stack([
        cv2.resize(img_bgr[y0:y1, x0:x1], (320, 320), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        for x0, y0, x1, y1 in boxes
    ], 0)
    t = torch.from_numpy(crops).float().cuda() / 255.0
    outs = []
    with torch.no_grad():
        for i in range(0, len(t), 32):
            outs.append(torch.sigmoid(refiner(t[i:i + 32])).squeeze(1).cpu().numpy())
    probs = np.concatenate(outs, 0)
    res = []
    for m, (x0, y0, x1, y1), pr in zip(masks, boxes, probs):
        pb = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) >= 0.45
        fm = np.zeros_like(m, bool)
        fm[y0:y1, x0:x1] = pb
        res.append(fm & m)
    return res

def ensure_base_cache(fold, det_key):
    out_pkl = V136_CACHE_DIR / f"{det_key}_f{fold}_best_cache.pkl"
    if out_pkl.exists():
        data = pickle.loads(out_pkl.read_bytes())
        print(f"再利用: {out_pkl.name} ({len(data)} obs)")
        return {row[0]: row for row in data}

    assert torch.cuda.is_available(), "初回キャッシュ作成には GPU が必要です"
    for name in eval_files[fold]:
        dst = LOCAL_PREPROC / name
        if not dst.exists():
            shutil.copy2(PREPROC_CACHE / name, dst)

    refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
    ck = torch.load(BASE_CKPTS[fold]["refiner"], map_location="cuda", weights_only=False)
    refiner.load_state_dict(ck.get("best_model") or ck["model"], strict=True)
    del ck

    model = YOLO(str(BASE_CKPTS[fold][det_key]))
    cache_rows = []
    for name in tqdm(eval_files[fold], desc=f"f{fold}_{det_key} 推論+refine"):
        ip = LOCAL_PREPROC / name
        r = model.predict(str(ip), imgsz=1280, conf=0.08, iou=0.55, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        h, w = r.orig_shape
        ms = []
        if r.masks is not None:
            for d in r.masks.data.detach().cpu().numpy():
                ms.append((cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) if d.shape != (h, w) else d) >= 0.5)
        cf = r.boxes.conf.detach().cpu().numpy().astype(np.float32) if r.boxes is not None else np.zeros(0, np.float32)
        vi = [i for i, x in enumerate(ms) if float(x.sum()) >= 20.0]
        vms = [ms[i] for i in vi]
        vcf = cf[vi] if vi else np.zeros(0, np.float32)
        areas = [float(x.sum()) for x in vms]
        im = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        rms = refine_batch(refiner, im, vms)
        raw_rles = [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in vms]
        ref_rles = [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in rms]
        cache_rows.append((name, vcf, areas, raw_rles, ref_rles))
    del model, refiner
    torch.cuda.empty_cache()
    out_pkl.write_bytes(pickle.dumps(cache_rows))
    print(f"保存完了: {out_pkl.name}")
    return {row[0]: row for row in cache_rows}

def load_oof_pkl(pkl_path):
    rows = pickle.loads(pkl_path.read_bytes())
    return {row[0]: row for row in rows}

DET_CACHES = {0: {}, 2: {}}
for f in FOLDS:
    DET_CACHES[f]["prod"] = ensure_base_cache(f, "prod")
    DET_CACHES[f]["yolo26"] = ensure_base_cache(f, "yolo26")
    DET_CACHES[f]["tal_best"] = load_oof_pkl(V133_CACHE_DIR / f"small_tal_f{f}_best_cache.pkl")
    if has_v133_last:
        DET_CACHES[f]["tal_last"] = load_oof_pkl(V133_CACHE_DIR / f"small_tal_f{f}_last_cache.pkl")
    if has_v134_best:
        DET_CACHES[f]["sptal_best"] = load_oof_pkl(V134_CACHE_DIR / f"spine_small_tal_f{f}_best_cache.pkl")
    if has_v134_last:
        DET_CACHES[f]["sptal_last"] = load_oof_pkl(V134_CACHE_DIR / f"spine_small_tal_f{f}_last_cache.pkl")
    if has_v138_best:
        DET_CACHES[f]["sp_best"] = load_oof_pkl(V138_CACHE_DIR / f"spine_aux_f{f}_best_cache.pkl")
    if has_v138_last:
        DET_CACHES[f]["sp_last"] = load_oof_pkl(V138_CACHE_DIR / f"spine_aux_f{f}_last_cache.pkl")

print("ロード済み検出器キー:", list(DET_CACHES[0].keys()))
"""

    c4 = """# [4/4] 27倍高速 BBox-Cropped RLE 排他化 (consensus_fusion 同梱) & 2-Fold 交差盲検採点
# 全検出器・全画像の refined RLE を一度だけ軽量 BBox crop spec 化 (~6KB/個)
REF_SPECS = {0: {}, 2: {}}
for f in FOLDS:
    for dkey, cmap in DET_CACHES[f].items():
        REF_SPECS[f][dkey] = {
            name: [rle_to_crop_spec(r) for r in row[4]]
            for name, row in cmap.items()
        }

def score_obs_spec_predictions(specs_by_file, files_subset):
    s_iou = 0.0; tp = 0; fp = 0; fn = 0; obs = {}
    for name in files_subset:
        pr = fast_apply_own_specs(specs_by_file[name])
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
                si = float(mat[hits].sum()); ti = int(hits.sum())
                fpi = int((hits.sum(0) == 0).sum()); fni = int((hits.sum(1) == 0).sum())
                s_iou += si; tp += ti; fp += fpi; fn += fni
                os_s[0] += si; os_s[1] += ti; os_s[2] += fpi; os_s[3] += fni
        obs[name] = tuple(os_s)
    den = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": float(s_iou / den) if den > 0 else 0.0,
        "pq_sparse_r077": sparse_label_pq(s_iou, tp, fp, fn, 0.77),
        "sq": float(s_iou / tp) if tp else 0.0,
        "rq": float(tp / den) if den > 0 else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }, obs

def pool_metrics_from_obs(obs_dict, files_subset):
    arr = np.array([obs_dict[f] for f in files_subset], np.float64)
    s_iou, tp, fp, fn = arr.sum(0)
    tp, fp, fn = round(tp), round(fp), round(fn)
    den = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": float(s_iou / den) if den > 0 else 0.0,
        "pq_sparse_r077": sparse_label_pq(s_iou, tp, fp, fn, 0.77),
        "sq": float(s_iou / tp) if tp else 0.0,
        "rq": float(tp / den) if den > 0 else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }

def paired_boot(obs_base, obs_cand, files_subset, sparse=False):
    rng = np.random.default_rng(3407)
    ba = np.array([obs_base[f] for f in files_subset], np.float64)
    ca = np.array([obs_cand[f] for f in files_subset], np.float64)
    idx = rng.integers(0, len(files_subset), size=(2000, len(files_subset)))
    bs = ba[idx].sum(1); cs = ca[idx].sum(1)
    if sparse:
        r = 0.77
        bden = r * bs[:, 1] + 0.5 * bs[:, 2] + 0.5 * (1 - r) * bs[:, 1] + 0.5 * r * bs[:, 3]
        cden = r * cs[:, 1] + 0.5 * cs[:, 2] + 0.5 * (1 - r) * cs[:, 1] + 0.5 * r * cs[:, 3]
        d = (r * cs[:, 0]) / cden - (r * bs[:, 0]) / bden
    else:
        d = cs[:, 0] / (cs[:, 1] + 0.5 * cs[:, 2] + 0.5 * cs[:, 3]) - bs[:, 0] / (bs[:, 1] + 0.5 * bs[:, 2] + 0.5 * bs[:, 3])
    return {
        "n_obs": len(files_subset),
        "mean_delta": float(d.mean()),
        "ci95_low": float(np.percentile(d, 2.5)),
        "ci95_high": float(np.percentile(d, 97.5)),
        "prob_positive": float((d > 0).mean()),
    }

# 1対1 クラスタリングは active detector 集合ごとに1回だけ計算してメモリ保持
CLUSTER_CACHE = {}
def get_preclustered(fold, det_tuple):
    key = (fold, det_tuple)
    if key in CLUSTER_CACHE:
        return CLUSTER_CACHE[key]
    by_file = {}
    for name in eval_files[fold]:
        props = []
        for d in det_tuple:
            _, cf, areas, raw_rles, _ = DET_CACHES[fold][d][name]
            for i in range(len(cf)):
                if areas[i] >= 20.0:
                    props.append(DetectorProposal(d, float(cf[i]), raw_rles[i], i))
        by_file[name] = match_detector_proposals_1to1(props, iou_threshold=0.45)
    CLUSTER_CACHE[key] = by_file
    return by_file

# 評価する Arm 群を構築
ARMS = [
    ("prod", {"prod": 1.0}, 0.26, None),
    ("yolo26_026", {"yolo26": 1.0}, 0.26, None),
    ("tal_best_026", {"tal_best": 1.0}, 0.26, None),
    ("tal_best_028", {"tal_best": 1.0}, 0.28, None),
    ("pair_prod_tal_b_t21", {"prod": 0.50, "tal_best": 0.50}, 0.21, None),
    ("pair_prod_tal_b_t23", {"prod": 0.50, "tal_best": 0.50}, 0.23, None),
    ("trio_b_equal_t20", {"prod": 1/3, "yolo26": 1/3, "tal_best": 1/3}, 0.20, None),
    ("trio_b_equal_t22", {"prod": 1/3, "yolo26": 1/3, "tal_best": 1/3}, 0.22, None),
    ("trio_b_tal45_y30_z25_t21", {"tal_best": 0.45, "prod": 0.30, "yolo26": 0.25}, 0.21, None),
    ("trio_b_tal45_y30_z25_t23", {"tal_best": 0.45, "prod": 0.30, "yolo26": 0.25}, 0.23, None),
    ("cons2of3_trio_b_c26", {"prod": 1.0, "yolo26": 1.0, "tal_best": 1.0}, 0.26, 2),
    ("cons2of3_trio_b_c26_maj3", {"prod": 1.0, "yolo26": 1.0, "tal_best": 1.0}, 0.26, 2, "majority_3plus"),
]

if has_v133_last:
    ARMS.extend([
        ("tal_last_026", {"tal_last": 1.0}, 0.26, None),
        ("tal_last_028", {"tal_last": 1.0}, 0.28, None),
        ("pair_prod_tal_l_t21", {"prod": 0.50, "tal_last": 0.50}, 0.21, None),
        ("pair_prod_tal_l_t23", {"prod": 0.50, "tal_last": 0.50}, 0.23, None),
        ("trio_l_equal_t21", {"prod": 1/3, "yolo26": 1/3, "tal_last": 1/3}, 0.21, None),
        ("trio_l_tal45_y30_z25_t21", {"tal_last": 0.45, "prod": 0.30, "yolo26": 0.25}, 0.21, None),
        ("trio_l_tal45_y30_z25_t23", {"tal_last": 0.45, "prod": 0.30, "yolo26": 0.25}, 0.23, None),
        ("cons2of3_trio_l_c26", {"prod": 1.0, "yolo26": 1.0, "tal_last": 1.0}, 0.26, 2),
        ("cons2of3_trio_l_c26_maj3", {"prod": 1.0, "yolo26": 1.0, "tal_last": 1.0}, 0.26, 2, "majority_3plus"),
    ])

if has_v134_best:
    ARMS.extend([
        ("sptal_best_026", {"sptal_best": 1.0}, 0.26, None),
        ("sptal_best_028", {"sptal_best": 1.0}, 0.28, None),
        ("pair_tal_sptal_b_t22", {"tal_best": 0.50, "sptal_best": 0.50}, 0.22, None),
        ("pair_tal_sptal_b_t24", {"tal_best": 0.50, "sptal_best": 0.50}, 0.24, None),
        ("cons2of3_sptal_prod_z_c26_bm", {"sptal_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "best_member"),
        ("cons2of3_sptal_prod_z_c26_maj3", {"sptal_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
        ("quad_b_t20", {"sptal_best": 0.35, "tal_best": 0.30, "prod": 0.20, "yolo26": 0.15}, 0.20, None),
        ("quad_b_t22", {"sptal_best": 0.35, "tal_best": 0.30, "prod": 0.20, "yolo26": 0.15}, 0.22, None),
        ("quad_b_t24", {"sptal_best": 0.35, "tal_best": 0.30, "prod": 0.20, "yolo26": 0.15}, 0.24, None),
        ("cons2of4_quad_b_c26", {"sptal_best": 1.0, "tal_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2),
        ("cons2of4_quad_b_c26_maj3", {"sptal_best": 1.0, "tal_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
    ])

if has_v134_last:
    ARMS.extend([
        ("sptal_last_026", {"sptal_last": 1.0}, 0.26, None),
        ("pair_tal_sptal_l_t22", {"tal_last": 0.50, "sptal_last": 0.50}, 0.22, None),
        ("quad_l_t21", {"sptal_last": 0.35, "tal_last": 0.30, "prod": 0.20, "yolo26": 0.15}, 0.21, None),
        ("quad_l_t23", {"sptal_last": 0.35, "tal_last": 0.30, "prod": 0.20, "yolo26": 0.15}, 0.23, None),
    ])

if has_v138_best:
    ARMS.extend([
        ("sp_best_026", {"sp_best": 1.0}, 0.26, None),
        ("sp_best_028", {"sp_best": 1.0}, 0.28, None),
        ("pair_spb_prod_t23", {"sp_best": 0.60, "prod": 0.40}, 0.23, None),
        ("trio_spb50_y25_z25_t22", {"sp_best": 0.50, "prod": 0.25, "yolo26": 0.25}, 0.22, None),
        ("trio_spb50_y25_z25_t24", {"sp_best": 0.50, "prod": 0.25, "yolo26": 0.25}, 0.24, None),
        ("cons2of3_spb_prod_z_c26_maj3", {"sp_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
        ("cons2of4_spb_talb_prod_z_c26_maj3", {"sp_best": 1.0, "tal_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
    ])
    if has_v133_last:
        ARMS.extend([
            ("pair_spb55_tall45_t22", {"sp_best": 0.55, "tal_last": 0.45}, 0.22, None),
            ("pair_spb55_tall45_t24", {"sp_best": 0.55, "tal_last": 0.45}, 0.24, None),
            ("quad_spb40_tall25_y20_z15_t22", {"sp_best": 0.40, "tal_last": 0.25, "prod": 0.20, "yolo26": 0.15}, 0.22, None),
            ("quad_spb40_tall25_y20_z15_t24", {"sp_best": 0.40, "tal_last": 0.25, "prod": 0.20, "yolo26": 0.15}, 0.24, None),
            ("cons2of4_spb_tall_prod_z_c26_maj3", {"sp_best": 1.0, "tal_last": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
        ])
    if has_v134_best:
        ARMS.extend([
            ("pair_spb55_sptalb45_t24", {"sp_best": 0.55, "sptal_best": 0.45}, 0.24, None),
        ])

results = {}
obs_by_arm = {}

for arm_spec in tqdm(ARMS, desc="全 Arm 2-Fold 採点"):
    arm_name, wdict, thr, dvote = arm_spec[:4]
    mmode = arm_spec[4] if len(arm_spec) >= 5 else "best_member"
    det_tuple = tuple(sorted(wdict.keys()))
    obs_merged = {}
    fold_metrics = {}
    for f in FOLDS:
        if len(det_tuple) == 1:
            d0 = det_tuple[0]
            pm_specs = {}
            for name in eval_files[f]:
                _, cf, areas, _, _ = DET_CACHES[f][d0][name]
                sp_list = REF_SPECS[f][d0][name]
                pm_specs[name] = [
                    sp_list[i]
                    for i in range(len(cf))
                    if cf[i] >= thr and areas[i] >= 20.0
                ]
        else:
            cl_map = get_preclustered(f, det_tuple)
            pm_specs = {}
            for name in eval_files[f]:
                s_by_det = {d: REF_SPECS[f][d][name] for d in det_tuple}
                chosen = []
                for cl in cl_map[name]:
                    ok = (cl.count_above(thr) >= dvote) if dvote is not None else (cl.weighted_score(wdict) >= thr)
                    if ok:
                        sp = fuse_refined_crop_specs(cl, s_by_det, mode=mmode)
                        if sp[5] >= 20.0:
                            chosen.append(sp)
                pm_specs[name] = chosen
        fm, fo = score_obs_spec_predictions(pm_specs, eval_files[f])
        fold_metrics[f] = fm
        obs_merged.update(fo)
    obs_by_arm[arm_name] = obs_merged
    results[arm_name] = {
        "weights": wdict,
        "threshold": thr,
        "discrete_vote": dvote,
        "mask_mode": mmode,
        "fold0": fold_metrics[0],
        "fold2": fold_metrics[2],
        "pooled_471": pool_metrics_from_obs(obs_merged, all_files),
    }

prod_obs = obs_by_arm["prod"]
p0 = results["prod"]["fold0"]["pq"]
p2 = results["prod"]["fold2"]["pq"]
pp = results["prod"]["pooled_471"]["pq"]
ps = results["prod"]["pooled_471"]["pq_sparse_r077"]

print(f"\\n=== v136 2-FOLD HONEST SUMMARY (471 obs, v134_included={has_v134_best}) ===")
for arm_name, r in results.items():
    bt_pq = paired_boot(prod_obs, obs_by_arm[arm_name], all_files, sparse=False)
    bt_sp = paired_boot(prod_obs, obs_by_arm[arm_name], all_files, sparse=True)
    r["bootstrap_pq_vs_prod"] = bt_pq
    r["bootstrap_sparse_vs_prod"] = bt_sp
    f0d = r["fold0"]["pq"] - p0
    f2d = r["fold2"]["pq"] - p2
    ppd = r["pooled_471"]["pq"] - pp
    spd = r["pooled_471"]["pq_sparse_r077"] - ps
    print(
        f"{arm_name:25s} | f0={r['fold0']['pq']:.5f} ({f0d:+.5f}) | f2={r['fold2']['pq']:.5f} ({f2d:+.5f}) "
        f"| pool={r['pooled_471']['pq']:.5f} ({ppd:+.5f}) | sparse={r['pooled_471']['pq_sparse_r077']:.5f} ({spd:+.5f}) "
        f"| FP={r['pooled_471']['fp']:4d} | P(>0)={bt_pq['prob_positive']:.3f}"
    )

cand_names = [k for k in results if k != "prod"]
sel_f2_pq = max(cand_names, key=lambda k: results[k]["fold2"]["pq"])
sel_f0_pq = max(cand_names, key=lambda k: results[k]["fold0"]["pq"])
sel_f2_sp = max(cand_names, key=lambda k: results[k]["fold2"]["pq_sparse_r077"])
sel_f0_sp = max(cand_names, key=lambda k: results[k]["fold0"]["pq_sparse_r077"])

cross_selection = {
    "by_pq": {
        "selected_on_fold2": {
            "arm": sel_f2_pq,
            "fold2_delta": results[sel_f2_pq]["fold2"]["pq"] - p2,
            "blind_fold0_delta": results[sel_f2_pq]["fold0"]["pq"] - p0,
            "pooled_delta": results[sel_f2_pq]["pooled_471"]["pq"] - pp,
            "pooled_sparse_delta": results[sel_f2_pq]["pooled_471"]["pq_sparse_r077"] - ps,
            "bootstrap_pq": results[sel_f2_pq]["bootstrap_pq_vs_prod"],
        },
        "selected_on_fold0": {
            "arm": sel_f0_pq,
            "fold0_delta": results[sel_f0_pq]["fold0"]["pq"] - p0,
            "blind_fold2_delta": results[sel_f0_pq]["fold2"]["pq"] - p2,
            "pooled_delta": results[sel_f0_pq]["pooled_471"]["pq"] - pp,
            "pooled_sparse_delta": results[sel_f0_pq]["pooled_471"]["pq_sparse_r077"] - ps,
            "bootstrap_pq": results[sel_f0_pq]["bootstrap_pq_vs_prod"],
        },
    },
    "by_sparse_r077": {
        "selected_on_fold2": {
            "arm": sel_f2_sp,
            "fold2_sparse_delta": results[sel_f2_sp]["fold2"]["pq_sparse_r077"] - results["prod"]["fold2"]["pq_sparse_r077"],
            "blind_fold0_sparse_delta": results[sel_f2_sp]["fold0"]["pq_sparse_r077"] - results["prod"]["fold0"]["pq_sparse_r077"],
            "pooled_sparse_delta": results[sel_f2_sp]["pooled_471"]["pq_sparse_r077"] - ps,
            "pooled_pq_delta": results[sel_f2_sp]["pooled_471"]["pq"] - pp,
            "bootstrap_sparse": results[sel_f2_sp]["bootstrap_sparse_vs_prod"],
        },
        "selected_on_fold0": {
            "arm": sel_f0_sp,
            "fold0_sparse_delta": results[sel_f0_sp]["fold0"]["pq_sparse_r077"] - results["prod"]["fold0"]["pq_sparse_r077"],
            "blind_fold2_sparse_delta": results[sel_f0_sp]["fold2"]["pq_sparse_r077"] - results["prod"]["fold2"]["pq_sparse_r077"],
            "pooled_sparse_delta": results[sel_f0_sp]["pooled_471"]["pq_sparse_r077"] - ps,
            "pooled_pq_delta": results[sel_f0_sp]["pooled_471"]["pq"] - pp,
            "bootstrap_sparse": results[sel_f0_sp]["bootstrap_sparse_vs_prod"],
        },
    },
}

print("\\n=== CROSS-FOLD BLIND SELECTION ===")
print(json.dumps(cross_selection, indent=2, ensure_ascii=False))

manifest = {
    "experiment_id": "v136_strong_fusion_2fold",
    "has_v133_best": has_v133_best,
    "has_v133_last": has_v133_last,
    "has_v134_best": has_v134_best,
    "has_v134_last": has_v134_last,
    "has_v138_best": has_v138_best,
    "has_v138_last": has_v138_last,
    "results": results,
    "cross_selection": cross_selection,
}
mp = ARTIFACT_ROOT / "v136_strong_fusion_2fold_manifest.json"
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=ARTIFACT_ROOT, suffix=".tmp", delete=False) as h:
    h.write(json.dumps(manifest, indent=2, ensure_ascii=False))
    h.flush()
    os.fsync(h.fileno())
    tmp_p = h.name
os.replace(tmp_p, mp)
print("Saved manifest:", mp)
"""

    cells = [_cell("markdown", md), _cell("code", c1), _cell("code", c2), _cell("code", c3), _cell("code", c4)]
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            txt = "".join(ln for ln in c["source"] if not ln.lstrip().startswith("%"))
            ast.parse(txt, filename=f"v136_cell_{idx}.py")
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
    out = ROOT / "solar_filament_v136_strong_fusion_2fold_colab.ipynb"
    nb = build_notebook()
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {out.name} ({len(nb['cells'])} cells, AST verified)")


if __name__ == "__main__":
    main()
