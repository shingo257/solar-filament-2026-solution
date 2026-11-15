"""Build v137 Colab notebook: All-Data Strong Detector & 1-to-1 Continuous Fusion Submission Generator.

Generates, validates (2048x2048 RLE round-trip, zero pixel overlap, non-empty masks),
and cross-audits (1-to-1 Hungarian IoU match against prod & v126 consensus) final Kaggle
submission CSV candidates on the 180 test images:

1. ``repro_v96full``: Sanity verification of ``yolo_full`` @ 0.26 + ``edgeatt_all`` refiner.
2. ``small_tal_all``: Standalone All-Data 70ep Small-TAL (``v133c``).
3. ``spine_small_tal_all``: Standalone All-Data 70ep Spine-Aux + Small-TAL (``v134c``, if ready).
4. ``strong_fusion_all``: 1-to-1 continuous multi-detector ensemble across All-Data checkpoints
   (``yolo_full`` + ``v123 yolo26_all`` + ``v133c Small-TAL`` [+ ``v134c Spine+Small-TAL``]),
   applying the exact weight/threshold rule selected out-of-fold by ``v136``.

Usage:
    python scratch/build_v137_final_submission_colab.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDISEG_PATH = ROOT / "solar_filament" / "models" / "mediseg_models.py"
CONSENSUS_PATH = ROOT / "solar_filament" / "pipeline" / "consensus_fusion.py"
VALIDATOR_PATH = ROOT / "solar_filament" / "pipeline" / "v32_overlap_cluster.py"


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
    val_src = _strip_package_imports(VALIDATOR_PATH.read_text(encoding="utf-8"))

    md = """# v137b — All-Data 強検出器 & 1対1 連続融合 + ピクセル多数決 最終提出 CSV 生成・厳密検証ノートブック

180 枚の公式テスト画像 (`2048x2048`) に対し、`prod` と一字一句同一の前処理 (`solar_three_channel` JPEG q96)・Refiner (`mediseg_edgeattnet_refiner_all_data_resume.pt`)・排他化 (`drop_contained@0.90` + 面積昇順 ownership + `min_area=20`) を適用し、以下を一括生成・厳密検証します：
1. `repro_v96full` (`yolo_full @ 0.26` 再現サニティチェック)
2. `v133c` 単体 (`Small-TAL` All-Data 70ep)
3. **471枚 2-Fold 盲検 100% 勝者**: `v137_cons2of3_trio_b_c26_maj3` (`prod + yolo26 + v133c Small-TAL`, 2-of-3 @ `0.26` + `majority_3plus`、`+0.810 pt` PQ / `P(>0)=100%`)
4. **Fold-2 直交 3 検出器勝者**: `v137_cons2of3_sptal_prod_z_c26_maj3` (`prod + yolo26 + v134c Spine+Small-TAL`、`v134c` が終盤進行中の場合は自動待機して一括生成、`+1.380 pt` PQ / Sparse PQ `0.4028`)

全出力 CSV は保存前に **`validate_submission`（2048×2048 RLE 復号・重複 ID ゼロ・ピクセル重複ゼロ検査）** と **`submission_prod.csv` との 1対1 Hungarian IoU 照合** を通過させます。
"""

    c1 = """# [1/4] 環境・Drive 強制再マウント・All-Data 重み診断 & v136 勝者設定の読込
%pip install -q "ultralytics==8.4.120" pycocotools scipy
import hashlib, json, math, os, pickle, shutil, tempfile, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from ultralytics import YOLO

drive.mount("/content/drive", force_remount=True)
assert torch.cuda.is_available(), "GPU ランタイムを使用してください"

DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
OUT_DIR = ARTIFACT_ROOT / "submissions_v137"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_CACHE_DIR = OUT_DIR / "test_det_caches"
TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEST_RAW_DIR = DRIVE_ROOT / "competition_data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
LOCAL_TEST_IMG = Path("/content/solar_v137/test_preproc")
LOCAL_WEIGHTS = Path("/content/solar_v137/weights")
LOCAL_TEST_IMG.mkdir(parents=True, exist_ok=True)
LOCAL_WEIGHTS.mkdir(parents=True, exist_ok=True)

test_files = sorted(p.name for p in TEST_RAW_DIR.glob("*.jpeg"))
assert len(test_files) == 180, f"Expected 180 test images, got {len(test_files)}"

REFINER_ALL = ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_all_data_resume.pt"
assert REFINER_ALL.exists(), f"Missing all-data refiner: {REFINER_ALL}"

INCLUDE_LAST_CKPTS = False

RUN_DIRS = {
    "prod": DRIVE_ROOT / "runs" / "v96_data_scaling" / "yolo_full_fold2",
    "yolo26": DRIVE_ROOT / "runs" / "v123_yolo26_all" / "yolo_yolo26m_all_fold0",
    "v133c": DRIVE_ROOT / "runs" / "v133_small_tal" / "yolo_small_tal_all_fold2",
    "v134c": DRIVE_ROOT / "runs" / "v134_spine_small_tal" / "yolo_spine_small_tal_all_fold2",
    "v138c": DRIVE_ROOT / "runs" / "v138_spine_aux" / "yolo_spine_aux_all_fold2",
}

CANDIDATE_PTS = {
    "prod": ("prod", RUN_DIRS["prod"] / "weights" / "best.pt"),
    "yolo26": ("yolo26", RUN_DIRS["yolo26"] / "weights" / "best.pt"),
    "tal_best": ("v133c", RUN_DIRS["v133c"] / "weights" / "best.pt"),
    "tal_last": ("v133c", RUN_DIRS["v133c"] / "weights" / "last.pt"),
    "sptal_best": ("v134c", RUN_DIRS["v134c"] / "weights" / "best.pt"),
    "sptal_last": ("v134c", RUN_DIRS["v134c"] / "weights" / "last.pt"),
    "sp_best": ("v138c", RUN_DIRS["v138c"] / "weights" / "best.pt"),
    "sp_last": ("v138c", RUN_DIRS["v138c"] / "weights" / "last.pt"),
}

def inspect_run_dir(run_key: str, run_dir: Path, target_epochs: int = 70):
    # Force FUSE directory cache refresh on Google Drive
    for d in [DRIVE_ROOT / "runs", run_dir.parent, run_dir, run_dir / "weights"]:
        try:
            if d.exists():
                os.listdir(str(d))
        except OSError:
            pass
    if not run_dir.exists():
        return {"exists": False, "complete": False, "epochs": 0, "best_mb": 0.0, "tc": False, "last_age_s": 1e9}
    tc = (run_dir / "TRAINING_COMPLETE").exists()
    rc = run_dir / "results.csv"
    epochs = 0
    if rc.exists():
        try:
            lines = [ln for ln in rc.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            epochs = max(0, len(lines) - 1)
        except OSError:
            pass
    bp = run_dir / "weights" / "best.pt"
    lp = run_dir / "weights" / "last.pt"
    best_mb = (bp.stat().st_size / (1024 * 1024)) if bp.exists() else 0.0
    last_age_s = (time.time() - lp.stat().st_mtime) if lp.exists() else 1e9
    if run_key in ("prod", "yolo26"):
        complete = bp.exists() and best_mb > 30.0
    else:
        # Completed Ultralytics 70ep run has TRAINING_COMPLETE or stripped best.pt (30-100 MB) after >=70 epochs
        complete = bp.exists() and (tc or (epochs >= target_epochs and 30.0 < best_mb < 100.0))
    return {
        "exists": True,
        "complete": complete,
        "epochs": epochs,
        "best_mb": round(best_mb, 2),
        "tc": tc,
        "last_age_s": round(last_age_s, 1),
    }

def refresh_available_pts(include_last: bool = False, verbose: bool = True):
    statuses = {rk: inspect_run_dir(rk, rd) for rk, rd in RUN_DIRS.items()}
    if verbose:
        print("=== All-Data Run 状態診断 ===")
        for rk, st in statuses.items():
            print(f"  {rk:8s} | exists={str(st['exists']):5s} | complete={str(st['complete']):5s} | tc={str(st['tc']):5s} | epochs={st['epochs']:2d} | best_mb={st['best_mb']:6.2f}")
    avail = {}
    for k, (rk, p) in CANDIDATE_PTS.items():
        if k.endswith("_last") and not include_last:
            continue
        if statuses[rk]["complete"] and p.exists():
            avail[k] = p
    return avail, statuses

AVAILABLE_PTS, RUN_STATUSES = refresh_available_pts(include_last=INCLUDE_LAST_CKPTS, verbose=True)

assert "prod" in AVAILABLE_PTS and "yolo26" in AVAILABLE_PTS and "tal_best" in AVAILABLE_PTS, (
    f"v133c (All-Data Small-TAL) が見つかりません。診断結果: {RUN_STATUSES}"
)
print("利用可能な All-Data 検出器:", {k: str(v) for k, v in AVAILABLE_PTS.items()})
"""

    c2 = (
        "# [2/4] 前処理・Refiner・1対1融合・2048x2048 厳密検証器の定義\n"
        + mediseg_src
        + "\n\n"
        + cons_src
        + "\n\n"
        + val_src
        + """

def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)

def solar_three_channel(gray):
    gray = gray.astype(np.float32)
    disk = gray > max(4.0, np.percentile(gray, 8))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=80, sigmaY=80)
    flattened = gray / np.maximum(background, 1.0)
    flat_u8 = robust_uint8(flattened, disk, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    raw_u8 = robust_uint8(gray, disk, 0.5, 99.5)
    dark_detail = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    dark_detail = robust_uint8(dark_detail, disk, 0.0, 99.5)
    out = np.dstack([raw_u8, clahe, dark_detail])
    out[~disk] = 0
    return out
"""
    )

    c3 = """# [3/4] Test 180枚の前処理ステージング & 各 All-Data 検出器 + EdgeAtt Refiner キャッシュ
def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

for name in tqdm(test_files, desc="Test 180枚 solar_three_channel 前処理"):
    dst = LOCAL_TEST_IMG / name
    if dst.exists():
        continue
    gray = None
    for attempt in range(4):
        try:
            raw = np.frombuffer((TEST_RAW_DIR / name).read_bytes(), np.uint8)
            gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                break
        except OSError:
            pass
        time.sleep(2.0 * (attempt + 1))
    assert gray is not None, f"Failed to decode test image: {name}"
    cv2.imwrite(str(dst), solar_three_channel(gray), [cv2.IMWRITE_JPEG_QUALITY, 96])

local_ref_pt = LOCAL_WEIGHTS / "refiner_all.pt"
shutil.copy2(REFINER_ALL, local_ref_pt)
refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
_ck = torch.load(local_ref_pt, map_location="cuda", weights_only=False)
refiner.load_state_dict(_ck.get("best_model") or _ck["model"], strict=True)
del _ck

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

def refine_batch(img_bgr, masks):
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

TEST_CACHES = {}
WEIGHT_SHAS = {"refiner_all": file_sha256(local_ref_pt)}

def ensure_test_cache(dkey: str, src_pt: Path):
    if dkey in TEST_CACHES:
        return
    sha = file_sha256(src_pt)
    WEIGHT_SHAS[dkey] = sha
    cpkl = TEST_CACHE_DIR / f"test_det_{sha[:16]}.pkl"
    if cpkl.exists():
        TEST_CACHES[dkey] = pickle.loads(cpkl.read_bytes())
        print(f"再利用 [{dkey}]: {cpkl.name} (sha={sha[:12]})")
        return
    lpt = LOCAL_WEIGHTS / f"{dkey}.pt"
    shutil.copy2(src_pt, lpt)
    model = YOLO(str(lpt))
    cdict = {}
    for name in tqdm(test_files, desc=f"test 推論+refine [{dkey}]"):
        ip = LOCAL_TEST_IMG / name
        r = model.predict(str(ip), imgsz=1280, conf=0.08, iou=0.55, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        h, w = r.orig_shape
        assert (h, w) == (2048, 2048), f"Unexpected test shape: {(h, w)}"
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
        rms = refine_batch(im, vms)
        raw_rles = [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in vms]
        ref_rles = [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in rms]
        cdict[name] = (vcf, areas, raw_rles, ref_rles)
    del model
    torch.cuda.empty_cache()
    cpkl.write_bytes(pickle.dumps(cdict))
    TEST_CACHES[dkey] = cdict
    print(f"保存完了 [{dkey}]: {cpkl.name} (sha={sha[:12]})")

for dkey, src_pt in list(AVAILABLE_PTS.items()):
    ensure_test_cache(dkey, src_pt)

# prod / yolo26 / tal_best の推論中 (~15分) に v134c (Spine+Small-TAL All-Data) が完走間近・完走済みなら自動回収
st_v134 = inspect_run_dir("v134c", RUN_DIRS["v134c"])
if not st_v134["complete"] and st_v134["epochs"] >= 62 and st_v134["last_age_s"] < 900:
    print(f"[v134c 自動待機] Epoch {st_v134['epochs']}/70 進行中 — 完走まで最大 25 分待機して sptal_best も一括生成します...")
    for _ in range(50):
        time.sleep(30)
        st_v134 = inspect_run_dir("v134c", RUN_DIRS["v134c"])
        print(f"  [v134c 監視] epochs={st_v134['epochs']}/70 | tc={st_v134['tc']} | best_mb={st_v134['best_mb']} | complete={st_v134['complete']}")
        if st_v134["complete"] or st_v134["last_age_s"] >= 900:
            break

AVAILABLE_PTS, RUN_STATUSES = refresh_available_pts(include_last=INCLUDE_LAST_CKPTS, verbose=True)
for dkey, src_pt in list(AVAILABLE_PTS.items()):
    ensure_test_cache(dkey, src_pt)
"""

    c4 = """# [4/4] 候補 CSV 生成・validate_submission 厳密検査・既存提出 CSV との Hungarian IoU 照合
SPECS_BY_DET = {
    dkey: {name: [rle_to_crop_spec(r) for r in row[3]] for name, row in cdict.items()}
    for dkey, cdict in TEST_CACHES.items()
}

def compare_with_reference(cand_df, ref_csv_path):
    if not ref_csv_path.exists():
        return None
    ref_df = pd.read_csv(ref_csv_path, keep_default_na=False)
    sub_a = defaultdict(list)
    sub_b = defaultdict(list)
    for row in cand_df.itertuples():
        sub_a[row.filament_id.rsplit("_", 1)[0]].append({"size": [2048, 2048], "counts": row.segmentation_rle.encode("utf-8")})
    for row in ref_df.itertuples():
        sub_b[row.filament_id.rsplit("_", 1)[0]].append({"size": [2048, 2048], "counts": row.segmentation_rle.encode("utf-8")})
    stems = sorted(set(sub_a) | set(sub_b))
    matched = 0
    ious = []
    for stem in stems:
        ra, rb = sub_a.get(stem, []), sub_b.get(stem, [])
        if not ra or not rb:
            continue
        mat = np.asarray(mask_utils.iou(ra, rb, [0] * len(rb)), np.float64)
        ious.extend(mat.max(axis=1).tolist())
        hits = mat > 0.5
        ri, ci = linear_sum_assignment(-hits.astype(np.int8))
        matched += int(hits[ri, ci].sum())
    return {
        "ref_rows": len(ref_df),
        "cand_rows": len(cand_df),
        "matched_1to1_iou05": matched,
        "match_rate_vs_ref": round(matched / max(len(ref_df), 1), 4),
        "mean_best_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
    }

SUBMISSION_SPECS = [
    ("repro_v96full", {"prod": 1.0}, 0.26, None, "best_member"),
    ("v137_small_tal_all_best_026", {"tal_best": 1.0}, 0.26, None, "best_member"),
    ("v137_cons2of3_trio_b_c26_maj3", {"prod": 1.0, "yolo26": 1.0, "tal_best": 1.0}, 0.26, 2, "majority_3plus"),
    ("v137_cons2of3_trio_b_c26_bm", {"prod": 1.0, "yolo26": 1.0, "tal_best": 1.0}, 0.26, 2, "best_member"),
    ("v137_trio_all_tal45_y30_z25_t23_maj3", {"tal_best": 0.45, "prod": 0.30, "yolo26": 0.25}, 0.23, None, "majority_3plus"),
]
if "tal_last" in TEST_CACHES:
    SUBMISSION_SPECS.extend([
        ("v137_small_tal_all_last_026", {"tal_last": 1.0}, 0.26, None, "best_member"),
        ("v137_cons2of3_trio_l_c26_maj3", {"prod": 1.0, "yolo26": 1.0, "tal_last": 1.0}, 0.26, 2, "majority_3plus"),
    ])
if "sptal_best" in TEST_CACHES:
    SUBMISSION_SPECS.extend([
        ("v137_spine_small_tal_all_best_026", {"sptal_best": 1.0}, 0.26, None, "best_member"),
        ("v137_cons2of3_sptal_prod_z_c26_maj3", {"sptal_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
        ("v137_cons2of4_quad_b_c26_maj3", {"sptal_best": 1.0, "tal_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
        ("v137_quad_all_best_t22_maj3", {"sptal_best": 0.35, "tal_best": 0.30, "prod": 0.20, "yolo26": 0.15}, 0.22, None, "majority_3plus"),
    ])
if "sptal_last" in TEST_CACHES:
    SUBMISSION_SPECS.extend([
        ("v137_spine_small_tal_all_last_026", {"sptal_last": 1.0}, 0.26, None, "best_member"),
        ("v137_quad_all_last_t22_maj3", {"sptal_last": 0.35, "tal_last": 0.30, "prod": 0.20, "yolo26": 0.15}, 0.22, None, "majority_3plus"),
    ])
if "sp_best" in TEST_CACHES:
    SUBMISSION_SPECS.extend([
        ("v137_spine_aux_all_best_026", {"sp_best": 1.0}, 0.26, None, "best_member"),
        ("v137_cons2of3_spb_prod_z_c26_maj3", {"sp_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
        ("v137_cons2of4_spb_talb_prod_z_c26_maj3", {"sp_best": 1.0, "tal_best": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
    ])
    if "tal_last" in TEST_CACHES:
        SUBMISSION_SPECS.extend([
            ("v137_cons2of4_spb_tall_prod_z_c26_maj3", {"sp_best": 1.0, "tal_last": 1.0, "prod": 1.0, "yolo26": 1.0}, 0.26, 2, "majority_3plus"),
            ("v137_quad_spb40_tall25_y20_z15_t23_maj3", {"sp_best": 0.40, "tal_last": 0.25, "prod": 0.20, "yolo26": 0.15}, 0.23, None, "majority_3plus"),
        ])

# v136 の 2-Fold 盲検勝者が存在し、全構成重みが揃っていれば自動追加
v136_mf = ARTIFACT_ROOT / "v136_strong_fusion_2fold_manifest.json"
if v136_mf.exists():
    v136_data = json.loads(v136_mf.read_text(encoding="utf-8"))
    seen_arms = set()
    for tag_sel, blk in [("v136_f2sel_pq", v136_data["cross_selection"]["by_pq"]["selected_on_fold2"]),
                         ("v136_f2sel_sparse", v136_data["cross_selection"]["by_sparse_r077"]["selected_on_fold2"])]:
        arm_id = blk["arm"]
        if arm_id in seen_arms:
            continue
        seen_arms.add(arm_id)
        arm_cfg = v136_data["results"][arm_id]
        if all(k in TEST_CACHES for k in arm_cfg["weights"]):
            SUBMISSION_SPECS.append((
                f"v137_{tag_sel}_{arm_id}",
                arm_cfg["weights"],
                arm_cfg["threshold"],
                arm_cfg["discrete_vote"],
                arm_cfg.get("mask_mode", "best_member"),
            ))

REF_PROD_CSV = ARTIFACT_ROOT / "repro_prod_colab" / "submission_prod.csv"
REF_CONS_CSV = ARTIFACT_ROOT / "consensus_v126" / "submission_consensus.csv"

audit_summary = {"weights_sha256": WEIGHT_SHAS, "candidates": {}}

for sub_name, wdict, thr, dvote, mmode in SUBMISSION_SPECS:
    det_tuple = tuple(sorted(wdict.keys()))
    rows = []
    empty_imgs = []
    for name in test_files:
        if len(det_tuple) == 1:
            d0 = det_tuple[0]
            cf, areas, _, _ = TEST_CACHES[d0][name]
            chosen_specs = [SPECS_BY_DET[d0][name][i] for i in range(len(cf)) if cf[i] >= thr and areas[i] >= 20.0]
        else:
            props = []
            for d in det_tuple:
                cf, areas, raw_rles, _ = TEST_CACHES[d][name]
                for i in range(len(cf)):
                    if areas[i] >= 20.0:
                        props.append(DetectorProposal(d, float(cf[i]), raw_rles[i], i))
            clusters = match_detector_proposals_1to1(props, iou_threshold=0.45)
            s_by_det = {d: SPECS_BY_DET[d][name] for d in det_tuple}
            chosen_specs = []
            for cl in clusters:
                ok = (cl.count_above(thr) >= dvote) if dvote is not None else (cl.weighted_score(wdict) >= thr)
                if ok:
                    sp = fuse_refined_crop_specs(cl, s_by_det, mode=mmode)
                    if sp[5] >= 20.0:
                        chosen_specs.append(sp)
        final_rles = fast_apply_own_specs(chosen_specs)
        if not final_rles:
            empty_imgs.append(name)
        stem = Path(name).stem
        for idx_m, rle in enumerate(final_rles, start=1):
            rows.append({
                "filament_id": f"{stem}_{idx_m}",
                "segmentation_rle": rle["counts"].decode("utf-8"),
            })

    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    val_rep = validate_submission(df)
    assert_codec_roundtrip(df["segmentation_rle"].iloc[:25].tolist())

    out_csv = OUT_DIR / f"submission_{sub_name}.csv"
    df.to_csv(out_csv, index=False, lineterminator="\\n")
    csv_sha = file_sha256(out_csv)

    cmp_prod = compare_with_reference(df, REF_PROD_CSV)
    cmp_cons = compare_with_reference(df, REF_CONS_CSV)
    audit_summary["candidates"][sub_name] = {
        "csv_path": str(out_csv),
        "sha256": csv_sha,
        "weights": wdict,
        "threshold": thr,
        "discrete_vote": dvote,
        "mask_mode": mmode,
        "validator": val_rep,
        "empty_test_images_count": len(empty_imgs),
        "empty_test_images": empty_imgs,
        "inst_per_180_img": round(len(df) / 180.0, 3),
        "vs_prod_csv": cmp_prod,
        "vs_consensus_v126_csv": cmp_cons,
    }
    print(
        f"{sub_name:38s} | rows={len(df):4d} ({len(df)/180:.2f}/img) | imgs={val_rep['images']}/180 "
        f"| overlap={val_rep['overlap_pixels']} | vs_prod_match={cmp_prod['match_rate_vs_ref'] if cmp_prod else 'N/A'}"
    )

audit_path = OUT_DIR / "v137_submission_audit_manifest.json"
audit_path.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("\\nSaved submission audit manifest:", audit_path)
"""

    cells = [_cell("markdown", md), _cell("code", c1), _cell("code", c2), _cell("code", c3), _cell("code", c4)]
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            txt = "".join(ln for ln in c["source"] if not ln.lstrip().startswith("%"))
            ast.parse(txt, filename=f"v137_cell_{idx}.py")
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
    out = ROOT / "solar_filament_v137_final_submission_colab.ipynb"
    nb = build_notebook()
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {out.name} ({len(nb['cells'])} cells, AST verified)")


if __name__ == "__main__":
    main()
