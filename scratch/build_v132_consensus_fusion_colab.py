"""Build v132 Colab notebook: 2-Fold Honest 1-to-1 Multi-Detector Continuous Fusion (Fold 0 + Fold 2).

Implements Idea 8 (v132):
- Zero retraining (~12 minutes total on Colab GPU across 471 honest OOF observations = Fold 0 + Fold 2!).
- Uses the 100% leak-free 3-detector checkpoints already verified on Google Drive:
    * Fold 0 (236 obs): ``yolo_fold_0`` + ``v121_yolo26_f0`` + ``v121_croptrain_f0_honest`` + ``edgeatt_fold0``
    * Fold 2 (235 obs): ``yolo_fold_2`` + ``v117_yolo26`` + ``v124_croptrain_f2_honest`` + ``edgeatt_fold2``
- Fixes all three structural flaws in ``run_fold_oof_eval_consensus.py`` (v120/v121c/v124c):
    1. Strict 1-to-1 per-detector bipartite clustering (eliminating transitive same-detector chaining).
    2. Continuous calibrated cluster scoring ``sum_d w_d * s_d`` (eliminating ``cons2of3_c26``'s
       cliff that threw away ``(0.42, 0.24, 0)``, ``(0.24, 0.24, 0.24)``, and solo ``(0.75, 0, 0)``).
    3. Optional score-weighted proposal mask fusion (``soft_vote``) before ``MediSegEdgeAttUNet``.
- Performs strict out-of-fold cross-selection (pick best arm on Fold 2 -> blind evaluate on Fold 0,
  and pick best arm on Fold 0 -> blind evaluate on Fold 2) plus pooled 471-obs paired bootstrap CI
  against BOTH ``prod`` and ``cons2of3_c26_1to1``.

Usage:
    python scratch/build_v132_consensus_fusion_colab.py
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
    md_intro = """# v132 — 2-Fold Honest (Fold 0 + Fold 2, 471 観測) 1対1 連続スコア融合・軟投票実験 (Idea 8)

**学習ゼロ・Colab GPU 約 12 分で完結**（秘匿環境変数入力不要・1 クリック全セル実行）。

### なぜ `v121c` / `v124c` (`cons2of3_c26`) に大きな取りこぼしがあったか
`RESEARCH_LOG` §2.99 で `cons2of3_c26` は Fold 0 `+0.0030`・Fold 2 `+0.0086` と**両 fold で正**（平均 `+0.0058`）を達成した唯一の後段フィルタだったが、`run_fold_oof_eval_consensus.py` のソース監査で以下 3 点の構造的欠陥が判明した:
1. **推移的チェイニングのバグ**: union-find に「1 クラスタあたり同一 detector 高々 1 個」制約が無かったため、`yolo_A ↔ crop_bridge ↔ yolo_B` で隣接 2 個体が 1 クラスタに潰れていた。
2. **`0.26` 階段関数による TP 切り捨て**: `count(s >= 0.26) >= 2` のみが条件だったため、`(yolo11=0.45, yolo26=0.24, crop=0)`・3 モデル一致 `(0.24, 0.24, 0.24)`・超高確信単独検出 `(0.75, 0, 0)` を全て捨てていた（Fold 2 で `-15 TP`）。
3. **All-Data Trio 未評価**: `v126` 提出時は 2-fold 重み流用で、既に Drive 上に揃っている **All-Data 70ep 3 検出器（`yolo_full` + `v123 yolo26_all` + `v116 crop_all`）** での連続融合提出は一度も行われていない。

本ノートは **Fold 0 (236 obs) + Fold 2 (235 obs) = 471 観測** の漏洩なし OOF 重みを用い、**バッチ化 GPU Refiner キャッシュ** で全 7 Arm を一括評価し、Fold 2 選択 → Fold 0 盲検（および逆方向）の交差検証と 471 観測対応あり Bootstrap CI を算出する。
"""

    cell1_cfg = """# [1/5] 依存パッケージ・Drive マウント・実験設定 (完全分離 /content/solar_v132)
%pip install -q "ultralytics==8.4.120" pycocotools
import hashlib, json, math, os, pickle, shutil, sys, time
from collections import defaultdict
from dataclasses import dataclass
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
    SEED = 3407
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"

    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v132")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    CACHE_DIR = LOCAL_ROOT / "fold_caches"

    FOLDS = (0, 2)
    FOLD_CKPTS = {
        0: {
            "yolo": DRIVE_ROOT / "runs" / "yolo_fold_0" / "weights" / "best.pt",
            "yolo26": DRIVE_ROOT / "runs" / "v121_yolo26_f0" / "yolo_yolo26m_fold0" / "weights" / "best.pt",
            "crop": DRIVE_ROOT / "runs" / "v121_croptrain_f0_honest" / "yolo_croptrain_fold0" / "weights" / "best.pt",
            "refiner": ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold0_resume.pt",
        },
        2: {
            "yolo": DRIVE_ROOT / "runs" / "yolo_fold_2" / "weights" / "best.pt",
            "yolo26": DRIVE_ROOT / "runs" / "v117_yolo26" / "yolo_yolo26m_fold2" / "weights" / "best.pt",
            "crop": DRIVE_ROOT / "runs" / "v124_croptrain_f2_honest" / "yolo_croptrain_fold2" / "weights" / "best.pt",
            "refiner": ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold2_resume.pt",
        },
    }

    YOLO_IMGSZ = 1280
    DET_FLOOR = 0.10
    PRIMARY_CONF = 0.26
    PREDICT_IOU = 0.55
    MATCH_IOU = 0.45
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_THR = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    SPARSE_R = 0.77
    BOOTSTRAP_ITERS = 2000

for p in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)

for f, mp in CFG.FOLD_CKPTS.items():
    for k, pt in mp.items():
        assert pt.exists(), f"Missing fold {f} {k}: {pt}"

assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
RUNTIME_INFO = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "gpu": torch.cuda.get_device_name(0),
    "python": sys.version,
    "torch": torch.__version__,
    "ultralytics": version("ultralytics"),
}
print(json.dumps(RUNTIME_INFO, indent=2, ensure_ascii=False))
"""

    mediseg_src = _strip_module_for_inline(ROOT / "solar_filament" / "models" / "mediseg_models.py")
    fusion_src = _strip_module_for_inline(
        ROOT / "solar_filament" / "pipeline" / "consensus_fusion.py"
    )
    cell2_modules = (
        "# [2/5] MediSegEdgeAttUNet + 1対1 検出器マッチング & 連続スコア/マスク融合モジュール\n"
        + mediseg_src
        + "\n"
        + fusion_src
    )

    cell3_data = """# [3/5] COCO GT 読込・grouped_folds SHA256 検証・Fold 0 + Fold 2 (471 obs) ローカル SSD 配置
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
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)

folds_path = CFG.ARTIFACT_ROOT / "grouped_folds.csv"
folds_sha = file_sha256(folds_path)
if folds_sha != CFG.GROUPED_FOLDS_SHA256:
    raise ValueError(f"grouped_folds.csv SHA256 mismatch: {folds_sha}")
fold_df = pd.read_csv(folds_path)
fold_of_file = dict(zip(fold_df["file_name"], fold_df["fold"].astype(int)))

eval_files_by_fold = {f: [n for n in train_files if fold_of_file[n] == f] for f in CFG.FOLDS}
all_eval_files = eval_files_by_fold[0] + eval_files_by_fold[2]
assert len(eval_files_by_fold[0]) == 236 and len(eval_files_by_fold[2]) == 235

for name in tqdm(all_eval_files, desc="Fold 0 + Fold 2 画像をローカル SSD へコピー"):
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

gt_rles = {
    image_id: [to_rle(a, int(info["height"]), int(info["width"])) for a in annotations[image_id] if not a.get("iscrowd", 0)]
    for image_id, info in images.items()
}
print({"fold0_obs": len(eval_files_by_fold[0]), "fold2_obs": len(eval_files_by_fold[2]), "total_obs": len(all_eval_files)})
"""

    cell4_infer = """# [4/5] Fold 0 & Fold 2 の 3 検出器推論 + バッチ化 EdgeAtt Refiner 一括キャッシュ
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

def batch_refine_masks(refiner_model, image_bgr, bool_masks):
    if not bool_masks:
        return []
    boxes = [mask_square_box(pm) for pm in bool_masks]
    crops = np.stack([
        cv2.resize(image_bgr[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        for (x0, y0, x1, y1) in boxes
    ], axis=0)
    probs = []
    with torch.no_grad():
        for s in range(0, len(crops), 32):
            ten = torch.from_numpy(crops[s:s + 32]).float().cuda() / 255.0
            probs.append(refiner_model(ten).sigmoid()[:, 0].cpu().numpy())
    probs_320 = np.concatenate(probs, axis=0)
    out_rles = []
    for pm, (x0, y0, x1, y1), p320 in zip(bool_masks, boxes, probs_320):
        p_box = cv2.resize(p320, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        rm = np.zeros_like(pm, dtype=bool)
        rm[y0:y1, x0:x1] = p_box >= CFG.REFINER_THR
        rm &= pm
        out_rles.append(mask_utils.encode(np.asfortranarray(rm.astype(np.uint8))))
    return out_rles

CKPT_SHAS = {}
FOLD_DATA = {}

for f in CFG.FOLDS:
    ck_map = CFG.FOLD_CKPTS[f]
    CKPT_SHAS[f"fold_{f}"] = {k: file_sha256(v) for k, v in ck_map.items()}
    cache_pkl = CFG.CACHE_DIR / f"fold_{f}_trio_cache.pkl"
    if cache_pkl.exists():
        FOLD_DATA[f] = pickle.loads(cache_pkl.read_bytes())
        print(f"再利用: Fold {f} キャッシュ ({len(FOLD_DATA[f])} obs)")
        continue

    dets = {d: YOLO(str(ck_map[d])) for d in ("yolo", "yolo26", "crop")}
    refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
    rck = torch.load(ck_map["refiner"], map_location="cuda", weights_only=False)
    refiner.load_state_dict(rck.get("best_model") or rck["model"], strict=True)
    del rck

    rows = {}
    for name in tqdm(eval_files_by_fold[f], desc=f"Fold {f} 3-Det + Batch Refiner"):
        img_path = CFG.PREPROC_DIR / name
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        raw_bools = []
        meta = []
        for det_name, model in dets.items():
            res = model.predict(
                str(img_path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.DET_FLOOR,
                iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False,
            )[0]
            h, w = res.orig_shape
            if res.masks is None:
                continue
            confs = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
            for m, c in zip(res.masks.data.detach().cpu().numpy(), confs):
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                bm = m >= 0.5
                if float(c) >= CFG.DET_FLOOR and int(bm.sum()) >= CFG.MIN_AREA:
                    raw_bools.append(bm)
                    meta.append((det_name, float(c), mask_utils.encode(np.asfortranarray(bm.astype(np.uint8)))))

        ref_rles = batch_refine_masks(refiner, image, raw_bools)
        props = [
            DetectorProposal(det_name=dn, conf=cf, raw_rle=rr, prop_idx=idx)
            for idx, (dn, cf, rr) in enumerate(meta)
        ]
        clusters = match_detector_proposals_1to1(props, iou_threshold=CFG.MATCH_IOU)

        # 多検出器クラスタの soft_vote 提案マスクも一括でバッチ refine して保存
        multi_indices = [ci for ci, cl in enumerate(clusters) if len(cl.members) >= 2]
        soft_masks = [fuse_cluster_proposal_mask(clusters[ci], mode="soft_vote", vote_threshold=0.45) for ci in multi_indices]
        soft_ref_rles_sub = batch_refine_masks(refiner, image, soft_masks)
        soft_ref_map = dict(zip(multi_indices, soft_ref_rles_sub))

        cluster_records = []
        for ci, cl in enumerate(clusters):
            best_p = cl.best_member()
            max_ref_rle = ref_rles[best_p.prop_idx]
            soft_ref_rle = soft_ref_map.get(ci, max_ref_rle)
            cluster_records.append({
                "scores": {d: cl.score_of(d) for d in ("yolo", "yolo26", "crop")},
                "max_ref_rle": max_ref_rle,
                "soft_ref_rle": soft_ref_rle,
            })
        prod_ref_rles = [ref_rles[idx] for idx, (dn, cf, _) in enumerate(meta) if dn == "yolo" and cf >= CFG.PRIMARY_CONF]
        rows[name] = {"prod_ref_rles": prod_ref_rles, "clusters": cluster_records}

    cache_pkl.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    FOLD_DATA[f] = rows
    del dets, refiner; torch.cuda.empty_cache()
print("Cached both Fold 0 and Fold 2 successfully!")
"""

    cell5_eval = """# [5/5] Fold 0 + Fold 2 (471 obs) 採点・交差検証 (Fold2→Fold0 / Fold0→Fold2) & Bootstrap 判定
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

def paired_bootstrap_pq(per_obs_a, per_obs_b, obs_names, n_boot=2000, seed=3407):
    rng = np.random.default_rng(seed)
    n_u = len(obs_names)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_u, size=n_u)
        sa, sb = [0.0, 0, 0, 0], [0.0, 0, 0, 0]
        for ui in pick:
            nm = obs_names[ui]
            for k in range(4):
                sa[k] += per_obs_a[nm][k]
                sb[k] += per_obs_b[nm][k]
        den_a = sa[1] + 0.5 * sa[2] + 0.5 * sa[3]
        den_b = sb[1] + 0.5 * sb[2] + 0.5 * sb[3]
        deltas[b] = (sb[0] / den_b if den_b > 0 else 0.0) - (sa[0] / den_a if den_a > 0 else 0.0)
    return {
        "n_obs": n_u,
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "prob_positive": float((deltas > 0).mean()),
    }

# 事前固定 8 Arms (prod 再現確認 + 1対1 cons2of3_c26 + 連続スコア融合 + solo 救出 + soft_vote マスク)
ARMS = {
    "prod":               lambda row: row["prod_ref_rles"],
    "cons2of3_c26_1to1":  lambda row: [c["max_ref_rle"] for c in row["clusters"] if sum(v >= 0.26 for v in c["scores"].values()) >= 2],
    "solo_rescue_c26":    lambda row: [c["max_ref_rle"] for c in row["clusters"] if (sum(v >= 0.26 for v in c["scores"].values()) >= 2) or (max(c["scores"]["yolo"], c["scores"]["yolo26"]) >= 0.55)],
    "mean3_t19":          lambda row: [c["max_ref_rle"] for c in row["clusters"] if sum(c["scores"].values()) / 3.0 >= 0.19],
    "mean3_t21":          lambda row: [c["max_ref_rle"] for c in row["clusters"] if sum(c["scores"].values()) / 3.0 >= 0.21],
    "mean3_t23":          lambda row: [c["max_ref_rle"] for c in row["clusters"] if sum(c["scores"].values()) / 3.0 >= 0.23],
    "wmean_y40_z35_c25":  lambda row: [c["max_ref_rle"] for c in row["clusters"] if (0.40 * c["scores"]["yolo"] + 0.35 * c["scores"]["yolo26"] + 0.25 * c["scores"]["crop"]) >= 0.21],
    "mean3_t21_softmask": lambda row: [c["soft_ref_rle"] for c in row["clusters"] if sum(c["scores"].values()) / 3.0 >= 0.21],
}

def evaluate_arm(arm_fn):
    per_obs = {}
    fold_stats = {0: [0.0, 0, 0, 0], 2: [0.0, 0, 0, 0]}
    for f in CFG.FOLDS:
        for name in eval_files_by_fold[f]:
            rles_in = arm_fn(FOLD_DATA[f][name])
            masks_in = [mask_utils.decode(r) != 0 for r in rles_in]
            masks_in = [m for m in masks_in if int(m.sum()) >= CFG.MIN_AREA]
            final_masks = apply_ownership_prod(masks_in)
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(final_masks, -1).astype(np.uint8))) if final_masks else []
            obs_s = [0.0, 0, 0, 0]
            for iid in file_to_ids[name]:
                truth = gt_rles[iid]
                mat = iou_matrix(truth, pred_rles)
                if len(truth) == 0:
                    obs_s[2] += len(pred_rles)
                elif len(pred_rles) == 0:
                    obs_s[3] += len(truth)
                else:
                    hits = mat > 0.5
                    obs_s[0] += float(mat[hits].sum())
                    obs_s[1] += int(hits.sum())
                    obs_s[2] += int((hits.sum(axis=0) == 0).sum())
                    obs_s[3] += int((hits.sum(axis=1) == 0).sum())
            per_obs[name] = tuple(obs_s)
            for k in range(4):
                fold_stats[f][k] += obs_s[k]

    def _pack(st):
        si, tp, fp, fn = st
        den = tp + 0.5 * fp + 0.5 * fn
        return {
            "pq": float(si / den) if den > 0 else 0.0,
            "sq": float(si / tp) if tp > 0 else 0.0,
            "rq": float(tp / den) if den > 0 else 0.0,
            "pq_sparse_r077": sparse_label_pq(si, tp, fp, fn, CFG.SPARSE_R),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        }

    pooled = [fold_stats[0][k] + fold_stats[2][k] for k in range(4)]
    return {"fold_0": _pack(fold_stats[0]), "fold_2": _pack(fold_stats[2]), "pooled_471": _pack(pooled)}, per_obs

RESULTS, OBS_STATS = {}, {}
for arm_name, fn in tqdm(ARMS.items(), desc="全 8 Arm 採点"):
    RESULTS[arm_name], OBS_STATS[arm_name] = evaluate_arm(fn)

# Prod sanity check (v108/v121c Fold 0 = 0.444137, v76/v124c Fold 2 = 0.444238)
assert abs(RESULTS["prod"]["fold_0"]["pq"] - 0.444137) < 1e-3, RESULTS["prod"]["fold_0"]
assert abs(RESULTS["prod"]["fold_2"]["pq"] - 0.444238) < 1e-3, RESULTS["prod"]["fold_2"]

non_prod = [a for a in ARMS if a != "prod"]
best_on_f2 = max(non_prod, key=lambda a: RESULTS[a]["fold_2"]["pq"])
best_on_f0 = max(non_prod, key=lambda a: RESULTS[a]["fold_0"]["pq"])

BOOT_VS_PROD = {}
BOOT_VS_CONS2OF3 = {}
for a in non_prod:
    BOOT_VS_PROD[a] = paired_bootstrap_pq(OBS_STATS["prod"], OBS_STATS[a], all_eval_files, n_boot=CFG.BOOTSTRAP_ITERS, seed=CFG.SEED)
    if a != "cons2of3_c26_1to1":
        BOOT_VS_CONS2OF3[a] = paired_bootstrap_pq(OBS_STATS["cons2of3_c26_1to1"], OBS_STATS[a], all_eval_files, n_boot=CFG.BOOTSTRAP_ITERS, seed=CFG.SEED)

cross_val_summary = {
    "selected_on_fold2": {
        "arm": best_on_f2,
        "fold2_delta_vs_prod": RESULTS[best_on_f2]["fold_2"]["pq"] - RESULTS["prod"]["fold_2"]["pq"],
        "blind_fold0_delta_vs_prod": RESULTS[best_on_f2]["fold_0"]["pq"] - RESULTS["prod"]["fold_0"]["pq"],
        "pooled_471_delta_vs_prod": RESULTS[best_on_f2]["pooled_471"]["pq"] - RESULTS["prod"]["pooled_471"]["pq"],
        "pooled_bootstrap_vs_prod": BOOT_VS_PROD[best_on_f2],
    },
    "selected_on_fold0": {
        "arm": best_on_f0,
        "fold0_delta_vs_prod": RESULTS[best_on_f0]["fold_0"]["pq"] - RESULTS["prod"]["fold_0"]["pq"],
        "blind_fold2_delta_vs_prod": RESULTS[best_on_f0]["fold_2"]["pq"] - RESULTS["prod"]["fold_2"]["pq"],
        "pooled_471_delta_vs_prod": RESULTS[best_on_f0]["pooled_471"]["pq"] - RESULTS["prod"]["pooled_471"]["pq"],
        "pooled_bootstrap_vs_prod": BOOT_VS_PROD[best_on_f0],
    },
}

manifest_out = {
    "experiment_id": "v132_2fold_honest_continuous_fusion",
    "runtime": RUNTIME_INFO,
    "checkpoint_shas": CKPT_SHAS,
    "grouped_folds_sha256": folds_sha,
    "results": RESULTS,
    "cross_fold_blind_validation": cross_val_summary,
    "bootstrap_471_vs_prod": BOOT_VS_PROD,
    "bootstrap_471_vs_cons2of3_1to1": BOOT_VS_CONS2OF3,
}
manifest_path = CFG.ARTIFACT_ROOT / "v132_consensus_fusion_2fold_manifest.json"
manifest_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")

p0_base = RESULTS["prod"]["fold_0"]["pq"]
p2_base = RESULTS["prod"]["fold_2"]["pq"]
pp_base = RESULTS["prod"]["pooled_471"]["pq"]
print("=== v132 2-FOLD HONEST SUMMARY (Fold 0 [236] + Fold 2 [235] = 471 obs) ===")
for a in ARMS:
    r0, r2, rp = RESULTS[a]["fold_0"], RESULTS[a]["fold_2"], RESULTS[a]["pooled_471"]
    ci_str = ""
    if a != "prod":
        b = BOOT_VS_PROD[a]
        ci_str = f" | CI95=[{b['ci95_low']:+.5f}, {b['ci95_high']:+.5f}] P(>0)={b['prob_positive']:.3f}"
    print(
        f"{a:20s} | f0={r0['pq']:.5f} ({r0['pq']-p0_base:+.5f}) | f2={r2['pq']:.5f} ({r2['pq']-p2_base:+.5f}) | "
        f"pool471={rp['pq']:.5f} ({rp['pq']-pp_base:+.5f}) | TP={rp['tp']} FP={rp['fp']} FN={rp['fn']}{ci_str}"
    )
print("=== CROSS-FOLD BLIND SELECTION ===")
print(json.dumps(cross_val_summary, indent=2, ensure_ascii=False))
print("Saved manifest:", manifest_path)
"""

    cells = [
        _cell("markdown", md_intro),
        _cell("code", cell1_cfg),
        _cell("code", cell2_modules),
        _cell("code", cell3_data),
        _cell("code", cell4_infer),
        _cell("code", cell5_eval),
    ]
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            code_txt = "".join(ln for ln in c["source"] if not ln.lstrip().startswith("%"))
            ast.parse(code_txt, filename=f"v132_cell_{idx}.py")

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
    out_path = ROOT / "solar_filament_v132_consensus_fusion_colab.ipynb"
    nb = build_notebook()
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {out_path.name} ({len(nb['cells'])} cells, AST verified)")


if __name__ == "__main__":
    main()
