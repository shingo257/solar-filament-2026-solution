"""Embedded cell payloads for the build_v91_semsweep notebook builder.

v90 で学習・cache 済みの semantic 確率 map (runs/v90_semprop/probs_fold2)
を再利用し、提案化パラメータだけを動かす評価のみのグリッド実験。
Arm A = union 厳格化 grid (prop_thresh x min_area x dedup_iou)、
Arm B = det 低conf 提案の semantic 確率 rescue-scorer grid。
fold2 は検索フェーズ — argmax 設定は採用せず、fold0/1 確認用の
候補として記録するのみ (§2.46)。
"""


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


MEDISEG = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# v91 — semantic map 再利用の提案厳格化 grid + rescue-scorer (評価のみ)

v90 (§2.45, reject −0.0307) は blind-FN 救済機構の存在を実証したが
(TP +61)、sem 提案の粗さが FP +524 を量産した。本実験は v90 cache の
確率 map を再利用し、**学習なし**で提案化パラメータだけを変えて
2 つの機構を検索する fold2 検索フェーズ。

**Arm A — union 厳格化 grid**: `prop_thresh x min_area x dedup_iou` の
27 設定で union@0.26 を採点 (det 提案は cache、sem 提案は thresh ごとに
component+refine を 1 回だけ行い min_area/dedup だけ変える)。

**Arm B — rescue-scorer grid**: det の conf [0.08,0.26) 提案のうち
semantic 平均確率 >= theta のものを primary に昇格させる。theta で
grid。543 件の lowconf-covered FN 層を直接狙う機構。

**事前固定 (RESEARCH_LOG §2.46)**:

- fold2 は**検索**フェーズ: argmax 設定は採用候補にせず、fold0/1
  確認 (v92) 用の候補として `nominee` に記録するのみ
- v92 昇格ゲート (参考): `PQ(arm,0.26) − PQ(prod,0.26) >= +0.005` を
  fold0+fold1 両方で
- 全 grid 設定が prod を下回る場合、提案源ラインは negative として閉じる

GPU必須 (refiner)。学習なし ~30-45 分。
"""
    ),
    code(
        """# [1/6] 依存・Drive・設定
%pip install -q "ultralytics>=8.3,<9" pycocotools
import hashlib, json, math, os, random, shutil, sys, tempfile, time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
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

class CFG:
    SEED = 3407
    FOLD = 2
    # Drive artifacts/grouped_folds.csv (707 obs, file_name/fold) の凍結 SHA256
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v91_semsweep"
    YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v91_semsweep"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v91")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PROD_YOLO = DRIVE_ROOT / "runs" / f"yolo_fold_{FOLD}" / "weights" / "best.pt"
    REFINER_CKPT = ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{FOLD}_resume.pt"
    MANIFEST = ARTIFACT_ROOT / f"v91_semsweep_fold{FOLD}_manifest.json"
    # v90 学習済み確率 map を read-only で再利用 (v90 ノートを先に完走させること)
    SEM_PROBS = DRIVE_ROOT / "runs" / "v90_semprop" / f"probs_fold{FOLD}"
    # --- Arm A: union 厳格化 grid (提案化パラメータのみ) ---
    SEM_THRESH_GRID = (0.5, 0.65, 0.8)
    SEM_MINAREA_GRID = (20, 60, 120)
    SEM_DEDUP_GRID = (0.5, 0.25, 0.0)  # 0.0 = det と一切重ならない提案のみ
    # --- Arm B: det 低conf 提案の semantic rescue-scorer grid ---
    RESCUE_THETA_GRID = (0.3, 0.5, 0.7)   # det 生mask内の mean sem 確率閾値
    RESCUE_BAND = (0.08, 0.26)            # refine 済みの低conf帯
    # --- YOLO (prod det 評価のみ。再学習はしない) ---
    YOLO_IMGSZ = 1280
    # --- 評価 (run_fold_oof_eval.py / v87 と同一) ---
    PREDICT_CONF_FLOOR = 0.08   # refine 対象の下限 (prod 経路と同一)
    DIAG_CONF_FLOOR = 0.02      # 診断のみ: 生提案をこの下限まで記録
    PREDICT_IOU = 0.55
    CONF_SWEEP = (0.15, 0.2, 0.26, 0.32)
    PRIMARY_CONF = 0.26
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ = 0.444238
    GATE_DELTA = 0.005
    SPARSE_R = 0.77

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.RUNS_ROOT, CFG.YOLO_ROOT):
    path.mkdir(parents=True, exist_ok=True)

# fold 別 refiner ckpt 名が予定名と違う場合に備え、glob でフォールバック解決
if not CFG.REFINER_CKPT.exists():
    _cands = sorted(CFG.ARTIFACT_ROOT.glob(f"*refiner*fold{CFG.FOLD}*.pt"))
    if _cands:
        CFG.REFINER_CKPT = _cands[0]
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
required = {"competition data": CFG.DATA_CACHE, "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
            "前処理画像": CFG.PREPROC_CACHE, "prod YOLO": CFG.PROD_YOLO, "EdgeAtt refiner": CFG.REFINER_CKPT,
            "v90 確率map cache": CFG.SEM_PROBS}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提 artifact が見つからない (v90 ノートを先に完走させること): {missing}")
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
RUNTIME_INFO = {"started_utc": datetime.now(timezone.utc).isoformat(), "GPU": torch.cuda.get_device_name(0),
                "python": sys.version, "torch": torch.__version__, "ultralytics": version("ultralytics"),
                "pycocotools": version("pycocotools"), "opencv": cv2.__version__, "numpy": np.__version__}
print(RUNTIME_INFO)
"""
    ),
    code(
        """# [2/6] COCO・凍結 fold (SHA256 検証)・前処理画像
import csv, io, re

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

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
folds_sha256 = file_sha256(folds_path)
if folds_sha256 != CFG.GROUPED_FOLDS_SHA256:
    raise ValueError(f"grouped_folds.csv の SHA256 が不一致: {folds_sha256}")
fold_frame = pd.read_csv(folds_path)
if fold_frame["file_name"].duplicated().any() or set(fold_frame["file_name"]) != set(train_files):
    raise ValueError("fold CSV が全観測を一意にカバーしていない")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
valid_files = [n for n in train_files if fold_of_file[n] == CFG.FOLD]
fit_files = [n for n in train_files if fold_of_file[n] != CFG.FOLD]

for name in tqdm(train_files, desc="前処理画像をローカルへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)

def to_rle(annotation, height, width):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

gt_rles = {image_id: [to_rle(a, int(info["height"]), int(info["width"]))
                      for a in annotations[image_id] if not a.get("iscrowd", 0)]
           for image_id, info in images.items()}
print({"観測": len(train_files), "fit": len(fit_files), "val(OOF)": len(valid_files),
       "folds_sha256": folds_sha256[:16]})
"""
    ),
    code(
        """# [3/6] v90 cache 済み semantic 確率 map のロード (学習なし)
def load_sem_probs():
    \"\"\"v90 が Drive に保存した float16 npz を全 valid_files 分ロード。\"\"\"
    probs = {}
    missing = []
    for name in tqdm(valid_files, desc="semantic 確率 map (v90 cache)"):
        path = CFG.SEM_PROBS / (Path(name).stem + ".npz")
        if not path.exists():
            missing.append(name); continue
        probs[name] = np.load(path)["prob"].astype(np.float32)
    if missing:
        raise FileNotFoundError(f"v90 確率 map が不足: {len(missing)} 件 "
                                f"(v90 ノートを先に完走させること): {missing[:5]}")
    return probs
"""
    ),
    code(
        """# [4/6] refiner モデル定義 (solar_filament/models/mediseg_models.py と同一)
"""
        + MEDISEG
    ),
    code(
        """# [5/6] 評価関数 (run_fold_oof_eval.py / v87 と同一経路 + 低conf診断)
def sparse_label_pq(sum_iou, tp, fp, fn, r=0.77):
    \"\"\"dariushafshar の sparse-label 換算 (RESEARCH_LOG §2.28)。r=1 で通常の PQ。\"\"\"
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
def refine_proposal(image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal

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

class PanopticQuality:  # self-evaluation notebook 準拠 (閾値越えを全て数える)
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold, self.tp_ious, self.fp_count, self.fn_count = iou_threshold, [], 0, 0
    def add_matrix(self, matrix, n_gt, n_pred):
        if n_gt == 0:
            self.fp_count += n_pred; return
        if n_pred == 0:
            self.fn_count += n_gt; return
        hits = matrix > self.iou_threshold
        self.tp_ious.extend(matrix[hits].tolist())
        self.fp_count += int((hits.sum(axis=0) == 0).sum())
        self.fn_count += int((hits.sum(axis=1) == 0).sum())
    @property
    def score(self):
        d = len(self.tp_ious) + 0.5 * self.fp_count + 0.5 * self.fn_count
        return float(sum(self.tp_ious) / d) if d > 0 else 0.0
    def summary(self):
        tp, s = len(self.tp_ious), float(sum(self.tp_ious))
        return {"pq": self.score, "sq": s / tp if tp else 0.0, "tp": tp, "fp": self.fp_count,
                "fn": self.fn_count, "sum_iou": s,
                "pq_sparse_r077": sparse_label_pq(s, tp, self.fp_count, self.fn_count, CFG.SPARSE_R)}

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

def lowconf_coverage(raw_rles, raw_conf, truth_rles):
    \"\"\"primary 未満の生提案が truth を IoU>0.5 で覆う数 (診断のみ)。\"\"\"
    if not truth_rles:
        return 0
    band = [r for r, c in zip(raw_rles, raw_conf) if CFG.DIAG_CONF_FLOOR <= c < CFG.PRIMARY_CONF]
    if not band:
        return 0
    hits = np.asarray(mask_utils.iou(band, truth_rles, [0] * len(truth_rles)), dtype=np.float64).T
    return int((hits > 0.5).any(axis=1).sum())

def evaluate_yolo(weights_path, label):
    \"\"\"floor 推論 1 回 → conf>=0.08 の提案のみ refine (prod 経路と同一) →
    conf sweep はメモリ上で採点。conf [0.02,0.26) の生提案は診断用に保持。\"\"\"
    model = YOLO(str(weights_path))
    cache = []
    for name in tqdm(valid_files, desc=f"{label} 推論+refine"):
        path = CFG.PREPROC_DIR / name
        result = model.predict(str(path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.DIAG_CONF_FLOOR,
                               iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, conf = result_masks(result)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        raw_rles = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
        refined = [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8)))
                   if conf[i] >= CFG.PREDICT_CONF_FLOOR else None for i, m in enumerate(masks)]
        cache.append((name, conf, [float(m.sum()) for m in masks], raw_rles, refined))
    del model; torch.cuda.empty_cache()
    out, diag = {}, {"fn_at_primary": 0, "fn_covered_by_lowconf_prop": 0}
    for c in CFG.CONF_SWEEP:
        meter = PanopticQuality()
        n_pred = 0
        for name, conf, areas, raw_rles, refined in cache:
            keep = [i for i in range(len(conf)) if conf[i] >= c and areas[i] >= CFG.MIN_AREA]
            preds = apply_ownership([mask_utils.decode(refined[i]) != 0 for i in keep])
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
            n_pred += len(pred_rles)
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
        out[str(c)] = {**meter.summary(), "preds_per_image": n_pred / max(len(valid_files), 1)}
        print(label, c, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[str(c)].items()})
    # 診断: primary conf で FN となった GT のうち低conf生提案で覆える数
    for name, conf, areas, raw_rles, refined in cache:
        keep = [i for i in range(len(conf)) if conf[i] >= CFG.PRIMARY_CONF and areas[i] >= CFG.MIN_AREA]
        pred_masks = [mask_utils.decode(refined[i]) != 0 for i in keep]
        pred_rles = mask_utils.encode(np.asfortranarray(np.stack(pred_masks, -1).astype(np.uint8))) if pred_masks else []
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            if not truth:
                continue
            hits = iou_matrix(truth, pred_rles) > 0.5 if pred_rles else np.zeros((len(truth), 0))
            fn_idx = np.where(hits.sum(axis=1) == 0)[0] if pred_rles else np.arange(len(truth))
            if len(fn_idx):
                diag["fn_at_primary"] += len(fn_idx)
                diag["fn_covered_by_lowconf_prop"] += lowconf_coverage(
                    raw_rles, conf, [truth[i] for i in fn_idx])
    out["diag_lowconf"] = diag
    print(label, "diag:", diag)
    return out

# --- v91: det cache + semantic grid 評価 ---
def build_det_cache(weights_path):
    \"\"\"det 提案を 1 回推論+refine して cache。全 grid 設定で再利用。
    返り値: (name, conf, raw_rles, refined_rles_or_None) のリスト。\"\"\"
    model = YOLO(str(weights_path))
    cache = []
    for name in tqdm(valid_files, desc="det cache 推論+refine"):
        path = CFG.PREPROC_DIR / name
        result = model.predict(str(path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.DIAG_CONF_FLOOR,
                               iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, conf = result_masks(result)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        raw = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
        ref = [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8)))
               if conf[i] >= CFG.PREDICT_CONF_FLOOR else None for i, m in enumerate(masks)]
        cache.append((name, conf, [float(m.sum()) for m in masks], raw, ref))
    del model; torch.cuda.empty_cache()
    return cache

def sem_components(prob, threshold):
    \"\"\"fg 確率 → connected components。conf=成分内最大確率 (全成分返す)。\"\"\"
    binary = (prob >= threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for lab in range(1, n_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        m = labels == lab
        out.append({"area": area, "mask": m, "conf": float(prob[m].max())})
    return out

def prep_sem_per_thresh(probs):
    \"\"\"各 thresh の sem 成分を area>=min(MINAREA_GRID) で列挙し refine まで
    済ませて cache。grid 内の (min_area, dedup) 変化では再利用する。\"\"\"
    min_area_floor = min(CFG.SEM_MINAREA_GRID)
    per_thresh = {}
    for T in CFG.SEM_THRESH_GRID:
        items = {}
        for name in tqdm(valid_files, desc=f"sem 成分+refine thresh={T}"):
            image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
            comps = []
            for comp in sem_components(probs[name], T):
                if comp["area"] < min_area_floor:
                    continue
                comp["raw_rle"] = mask_utils.encode(np.asfortranarray(comp["mask"].astype(np.uint8)))
                comp["refined"] = refine_proposal(image, comp["mask"])
                comps.append(comp)
            items[name] = comps
        per_thresh[T] = items
    return per_thresh

def _union_score_at(det_cache, sem_items, c, min_area, dedup_iou):
    \"\"\"1 config × 1 conf: det(conf>=c) + sem(area>=min_area, dedup) →
    ownership → PQ 要素を返す。sem 提案は全 refine 済み mask を使う。\"\"\"
    meter = PanopticQuality()
    n_pred, n_sem = 0, 0
    for name, det_conf, areas, det_raw, det_ref in det_cache:
        keep_det = [i for i in range(len(det_conf))
                    if det_conf[i] >= c and areas[i] >= CFG.MIN_AREA]
        det_rles = [det_raw[i] for i in keep_det]
        pred_masks = [mask_utils.decode(det_ref[i]) != 0 for i in keep_det]
        for comp in sem_items.get(name, []):
            if comp["area"] < min_area:
                continue
            if det_rles and dedup_iou >= 0:
                ov = mask_utils.iou([comp["raw_rle"]], det_rles, [0] * len(det_rles))[0]
                if ov.max() > dedup_iou:
                    continue
            pred_masks.append(comp["refined"]); n_sem += 1
        preds = apply_ownership(pred_masks)
        pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
        n_pred += len(pred_rles)
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
    return meter, n_pred, n_sem

def union_grid(det_cache, per_thresh):
    \"\"\"Arm A: 27 config grid。@0.26 (gate point) と @0.20 を採点。\"\"\"
    rows = []
    det_tp = {}  # conf -> det-only TP (rescue gain の分母)
    for c in (CFG.PRIMARY_CONF, 0.2):
        det_meter, _, _ = _union_score_at(det_cache, {}, c, 10**9, -1)
        det_tp[c] = len(det_meter.tp_ious)
    for T in CFG.SEM_THRESH_GRID:
        for min_area in CFG.SEM_MINAREA_GRID:
            for dedup in CFG.SEM_DEDUP_GRID:
                meter, n_pred, n_sem = _union_score_at(
                    det_cache, per_thresh[T], CFG.PRIMARY_CONF, min_area, dedup)
                meter02, _, _ = _union_score_at(
                    det_cache, per_thresh[T], 0.2, min_area, dedup)
                s, s02 = meter.summary(), meter02.summary()
                row = {"thresh": T, "min_area": min_area, "dedup_iou": dedup,
                       "pq_026": s["pq"], "tp_026": s["tp"], "fp_026": s["fp"],
                       "fn_026": s["fn"], "rescued_tp_026": s["tp"] - det_tp[CFG.PRIMARY_CONF],
                       "pq_020": s02["pq"], "sem_kept_per_img": n_sem / len(valid_files),
                       "preds_per_image": n_pred / len(valid_files)}
                rows.append(row)
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    return rows

def rescue_grid(det_cache, probs):
    \"\"\"Arm B: det conf [0.08,0.26) の refine 済み提案で、raw mask 内の
    mean sem 確率 >= theta のものを primary に昇格。theta grid。\"\"\"
    lo, hi = CFG.RESCUE_BAND
    base_meter, _, _ = _union_score_at(det_cache, {}, CFG.PRIMARY_CONF, 10**9, -1)
    base_tp = len(base_meter.tp_ious)
    rows = []
    for theta in CFG.RESCUE_THETA_GRID:
        meter, n_resc = PanopticQuality(), 0
        n_pred = 0
        for name, det_conf, areas, det_raw, det_ref in det_cache:
            kept = [i for i in range(len(det_conf))
                    if det_conf[i] >= CFG.PRIMARY_CONF and areas[i] >= CFG.MIN_AREA]
            prob = probs[name]
            for i in range(len(det_conf)):
                if not (lo <= det_conf[i] < hi) or det_ref[i] is None or areas[i] < CFG.MIN_AREA:
                    continue
                m = mask_utils.decode(det_raw[i]) != 0  # 生 det mask 内の平均
                if float(prob[m].mean()) >= theta:
                    kept.append(i); n_resc += 1
            pred_masks = [mask_utils.decode(det_ref[i]) != 0 for i in kept]
            preds = apply_ownership(pred_masks)
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
            n_pred += len(pred_rles)
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
        s = meter.summary()
        row = {"theta": theta, "pq_026": s["pq"], "tp_026": s["tp"], "fp_026": s["fp"],
               "fn_026": s["fn"], "rescued_total": n_resc,
               "rescued_tp_026": s["tp"] - base_tp,
               "preds_per_image": n_pred / len(valid_files)}
        rows.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    return rows
"""
    ),
    code(
        """# [6/6] 実行: prod baseline → det cache → ArmA union grid / ArmB rescue grid → nominee 記録
def save_manifest(manifest):
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

CONTRACT = {"experiment_id": "v91_semsweep", "fold": CFG.FOLD,
            "protocol": ("RESEARCH_LOG §2.46 (v90 cache 確率 map の提案厳格化 grid "
                         "+ det 低conf rescue-scorer grid。fold2 は検索フェーズ、"
                         "argmax は採用せず fold0/1 確認候補として記録)"),
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO),
            "sem_probs_dir": str(CFG.SEM_PROBS),
            "grids": {"thresh": list(CFG.SEM_THRESH_GRID), "min_area": list(CFG.SEM_MINAREA_GRID),
                      "dedup_iou": list(CFG.SEM_DEDUP_GRID), "rescue_theta": list(CFG.RESCUE_THETA_GRID)},
            "primary_conf": CFG.PRIMARY_CONF,
            "selection_rule": "fold2 argmax = nominee (記録のみ)。昇格は別途 fold0/1 確認実験で両 fold Δ>=+0.005"}
manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8")) if CFG.MANIFEST.exists() else {}
if manifest and manifest.get("contract") != CONTRACT:
    raise ValueError("既存 manifest と入力契約が違う。履歴保持のため別パスを指定してください")
manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}

if "prod" not in manifest["results"]:
    manifest["results"]["prod"] = {"weights": str(CFG.PROD_YOLO), "by_conf": evaluate_yolo(CFG.PROD_YOLO, "prod")}
    save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]
print({"prod@primary": round(prod_pq, 6), "既知値": CFG.BASELINE_PQ,
       "差": round(prod_pq - CFG.BASELINE_PQ, 6)})
if abs(prod_pq - CFG.BASELINE_PQ) > 0.002:
    print("⚠️ prod の再現値が既知値から 0.002 以上ずれている — 配線/入力の差を確認 (判定は paired Δ で行う)")

probs = load_sem_probs()
det_cache = build_det_cache(CFG.PROD_YOLO)

if "armA_union_grid" not in manifest["results"]:
    per_thresh = prep_sem_per_thresh(probs)
    manifest["results"]["armA_union_grid"] = {"rows": union_grid(det_cache, per_thresh)}
    save_manifest(manifest)

if "armB_rescue_grid" not in manifest["results"]:
    manifest["results"]["armB_rescue_grid"] = {"rows": rescue_grid(det_cache, probs)}
    save_manifest(manifest)

# --- nominee 記録 (fold2 検索フェーズ。採用は fold0/1 確認で両 fold Δ>=+0.005) ---
rows_a = manifest["results"]["armA_union_grid"]["rows"]
rows_b = manifest["results"]["armB_rescue_grid"]["rows"]
nom_a = max(rows_a, key=lambda r: r["pq_026"])
nom_b = max(rows_b, key=lambda r: r["pq_026"])
manifest["nominees"] = {
    "armA_best": {**nom_a, "delta_026": nom_a["pq_026"] - prod_pq},
    "armB_best": {**nom_b, "delta_026": nom_b["pq_026"] - prod_pq},
    "prod_026": prod_pq,
    "note": "fold2 argmax は採用候補ではない。fold0/1 確認用候補として記録"}
manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
save_manifest(manifest)
print("ArmA best:", {k: round(v,4) if isinstance(v,float) else v for k,v in manifest["nominees"]["armA_best"].items()})
print("ArmB best:", {k: round(v,4) if isinstance(v,float) else v for k,v in manifest["nominees"]["armB_best"].items()})

print("prod sweep:", {c: round(v["pq"], 4) for c, v in manifest["results"]["prod"]["by_conf"].items()
                       if not c.startswith("diag")})
print("manifest:", CFG.MANIFEST)
"""
    ),
]
