"""Build exploratory, no-fit v62 diagnostics from existing checkpoints.

Recovered from PR244 da2d8d15; no historical GPU results are accepted here.
Counts and GT-informed interventions use separate (annotator, image) records.
Intervention deltas are oracle diagnostics, not proven bounds, deployable
predictions or automatic mechanism selection. Independent acceptance is required.

Before model loading supply artifacts/v62_expected_contract_fold{N}.json:
schema_version=1, fold, protocol_sha256, detector_sha256, refiner_sha256,
annotations_sha256, folds_sha256, preprocessed_sha256, baseline={pq,tolerance}.
Use independently frozen expected identities; this run never creates that file.
preprocessed_sha256 hashes UTF-8 json.dumps(mapping, sort_keys=True), where mapping
contains validation filenames and their SHA256. Explicit baseline pq/tolerance
are diagnostic reproduction guards, not scientific acceptance criteria.

Usage: python scratch/build_v62_error_anatomy.py --fold 2
Generation performs no inference, fitting, or downloads.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")

BASELINE_PQ_REFINED = {0: 0.4393, 1: 0.4415, 2: 0.4471}
DENSE_QUANTILE = 0.75
ORACLES = ("drop_orphan", "drop_fragment", "perfect_verifier", "merge")
PROTOCOL_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    + MODELS.replace("\r\n", "\n").encode("utf-8")
).hexdigest()

GUARDS = r'''def file_sha256(path):
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_checkpoint_snapshot(source, expected_sha256, destination):
    # A unique local copy isolates the loader from subsequent Drive syncs.
    # Verify the bytes actually copied, including any replacement during copy.
    import shutil
    import tempfile
    from pathlib import Path
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".pt", delete=False) as handle:
        local = Path(handle.name)
    try:
        shutil.copyfile(source, local)
        if file_sha256(local) != expected_sha256:
            raise ValueError("Copied checkpoint differs from frozen input identity")
        return local
    except BaseException:
        local.unlink(missing_ok=True)
        raise


def validate_fold_rows(rows, files, fold):
    from pathlib import PurePosixPath
    if type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("Invalid evaluation fold")
    mapping = {}
    for row in rows:
        name, value = row.get("file_name"), row.get("fold")
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).name != name or ":" in name or name in (".", "..")
                or type(value) is not int or value not in (0, 1, 2) or name in mapping):
            raise ValueError("Invalid or duplicate fold assignment")
        mapping[name] = value
    if set(mapping) != set(files):
        raise ValueError("Fold assignments must cover exactly the consumed files")
    if not any(v == fold for v in mapping.values()) or not any(v != fold for v in mapping.values()):
        raise ValueError("Training and validation partitions must be nonempty")
    return mapping


def validate_baseline(measured, expected, tolerance):
    import math
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (measured, expected, tolerance))
            or not 0 <= measured <= 1 or not 0 <= expected <= 1 or not 0 <= tolerance < 1
            or abs(measured - expected) > tolerance):
        raise ValueError("Baseline reproduction guard failed; no result accepted")


def require_expected_contract(path, actual):
    import json
    import re
    from pathlib import Path
    expected = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = {"schema_version", "fold", "protocol_sha256", "detector_sha256", "refiner_sha256",
            "annotations_sha256", "folds_sha256", "preprocessed_sha256", "baseline"}
    if not isinstance(expected, dict) or set(expected) != keys:
        raise ValueError("Expected frozen input contract is incomplete")
    baseline = expected["baseline"]
    if not isinstance(baseline, dict) or set(baseline) != {"pq", "tolerance"}:
        raise ValueError("Explicit baseline reproduction guard is required")
    validate_baseline(baseline["pq"], baseline["pq"], baseline["tolerance"])
    if type(expected["schema_version"]) is not int or type(expected["fold"]) is not int:
        raise ValueError("Invalid contract schema/fold")
    for key in keys - {"schema_version", "fold", "baseline"}:
        if not isinstance(expected[key], str) or re.fullmatch("[0-9a-f]{64}", expected[key]) is None:
            raise ValueError("Invalid expected SHA256")
    if {k:v for k,v in expected.items() if k != "baseline"} != actual:
        raise ValueError("Frozen input contract differs; do not infer")
    return expected


def validate_existing_manifest(path, expected):
    import json
    from pathlib import Path
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("input_contract") != expected:
            raise ValueError("Existing manifest contract differs; preserve it and choose a new path")


def atomic_json(path, payload):
    import json
    import os
    import tempfile
    from pathlib import Path
    path = Path(path)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
'''


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(fold):
    baseline_refined = BASELINE_PQ_REFINED[fold]

    cells = [
        markdown(
            f"""# v62 exploratory diagnostics — fold{fold}, no fitting

Counts and interventions use separate (annotator, image) records. Historical
reference PQ {baseline_refined} is context only. Independently freeze expected
input hashes and an explicit baseline reproduction guard before running; any
mismatch stops execution. See the builder docstring for the contract schema.

FN categories: blind (<0.1), weak (0.1–0.3), near-miss (0.3–0.5).
FP categories: orphan, fragment, misassigned, following the official all-hit rule.
Feature AUC labels mean refined-mask overlap >0.5 before ownership, not matched TP.
GT-informed interventions are oracle diagnostics: not proven bounds, deployable
predictions, automatic mechanism choices, or scientific acceptance.

Required: `artifacts/v62_expected_contract_fold{fold}.json`; generation of this
file from observed inputs is deliberately absent. Required keys: schema_version=1,
fold, protocol_sha256, detector_sha256, refiner_sha256, annotations_sha256,
folds_sha256, preprocessed_sha256, baseline={{pq,tolerance}}. All SHA values are
64 lowercase hex characters; preprocessed_sha256 hashes UTF-8 sorted JSON mapping
validation filenames to file SHA256. The diagnostic manifest uses
a new filename and atomic save after baseline reproduction. Each checkpoint is
copied to a unique local file and its copied bytes are checked immediately before
model loading, isolating subsequent Drive replacements (not an OS-level lock).
Annotation JSON and fold CSV are each captured once, parsed and hashed from the
same immutable byte buffers, so later Drive changes cannot relabel parsed data.
GPU execution and
independent acceptance remain unverified; no historical results are imported.
"""
        ),
        code(
            """# [1/6] Drive接続、設定、前提の確認
!pip -q install ultralytics pycocotools
import hashlib, json, math, random, shutil, sys, time, warnings
from datetime import datetime, timezone
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
drive.mount("/content/drive")

class CFG:
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v62_anatomy")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    FOLD = %d
    PROTOCOL_SHA256 = %r
    # --- OOF評価は確定構成(build_v58/v60/v61系と同一) ---
    YOLO_IMGSZ = 1280
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    CONF = 0.26
    PREDICT_CONF_FLOOR = 0.08
    PREDICT_IOU = 0.55
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ_REFINED = %r
    DENSE_QUANTILE = %r
    # --- 解剖パラメータ ---
    IOU_HIT = 0.5          # PQのマッチ閾値と同じ
    FN_WEAK_LO = 0.1       # blind/weak境界
    FN_NEAR_LO = 0.3       # weak/near-miss境界
    FP_ORPHAN_MAX = 0.1    # これ未満なら背景FP
    MERGE_MIN_IOU = 0.1    # merge oracleの断片判定IoU
    OVERMERGE_IOU = 0.3    # 1予測が複数GTにこのIoUで重なればover-merge
    ORACLES = %r
    MANIFEST = DRIVE_ROOT / "artifacts" / "v62_error_anatomy_fold%d_diagnostic_v1_manifest.json"
    EXPECTED_CONTRACT = ARTIFACT_ROOT / f"v62_expected_contract_fold{FOLD}.json"

def prod_checkpoint(fold):
    return CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{fold}" / "weights" / "best.pt"

for path in [CFG.LOCAL_ROOT, CFG.PREPROC_DIR]:
    path.mkdir(parents=True, exist_ok=True)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像キャッシュ": CFG.PREPROC_CACHE,
    f"fold{CFG.FOLD} prod検出器": prod_checkpoint(CFG.FOLD),
    f"fold{CFG.FOLD} 標準refiner": CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0), "torch": torch.__version__})
"""  # noqa: UP031 - preserve embedded dictionary braces
            % (fold, PROTOCOL_SHA256, baseline_refined, DENSE_QUANTILE, ORACLES, fold)
            + "\n" + GUARDS
        ),
        code(
            """# [2/6] COCOとfoldを読み、密度スライス用のcutを計算し、前処理画像をローカルSSDへ展開
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"学習annotation JSONが見つかりません: {base}")
    if len(matches) != 1:
        raise ValueError("Exactly one annotation JSON must be selected")
    return matches[0].parents[2]

TRAIN_JSON = next(find_dataset_root(CFG.DATA_CACHE).rglob(
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
# Capture each remote metadata file once: parse and fingerprint the same bytes.
annotation_bytes = TRAIN_JSON.read_bytes()
coco = json.loads(annotation_bytes.decode("utf-8"))

images = {item["id"]: item for item in coco["images"]}
if len(images) != len(coco["images"]):
    raise ValueError("Duplicate annotation image IDs")
annotations_by_image = defaultdict(list)
for ann in coco["annotations"]:
    if ann["image_id"] not in images:
        raise ValueError("Annotation references an unknown image")
    annotations_by_image[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)

from io import BytesIO
fold_bytes = (CFG.ARTIFACT_ROOT / "grouped_folds.csv").read_bytes()
fold_frame = pd.read_csv(BytesIO(fold_bytes))
fold_of_file = validate_fold_rows(fold_frame.to_dict("records"), train_files, CFG.FOLD)
train_files_fold = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]

def noncrowd_count(image_id):
    return sum(1 for a in annotations_by_image[image_id] if not a.get("iscrowd", 0))

# v57系と同じ定義: cutは学習サンプル(file,annotator)のGT数q75
train_sample_counts = [
    noncrowd_count(image_id)
    for name in train_files_fold
    for image_id in file_to_ids[name]
]
DENSE_CUT = float(np.quantile(train_sample_counts, CFG.DENSE_QUANTILE))
val_density = {
    name: float(np.mean([noncrowd_count(i) for i in file_to_ids[name]]))
    for name in valid_files_fold
}

for name in tqdm(valid_files_fold, desc="前処理画像をSSDへ"):
    local = CFG.PREPROC_DIR / name
    shutil.copy2(CFG.PREPROC_CACHE / name, local)  # replace stale local cache

PREPROC_HASHES = {name: file_sha256(CFG.PREPROC_DIR / name) for name in sorted(valid_files_fold)}
ACTUAL_INPUTS = {
    "schema_version": 1, "fold": CFG.FOLD, "protocol_sha256": CFG.PROTOCOL_SHA256,
    "detector_sha256": file_sha256(prod_checkpoint(CFG.FOLD)),
    "refiner_sha256": file_sha256(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt"),
    "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
    "folds_sha256": hashlib.sha256(fold_bytes).hexdigest(),
    "preprocessed_sha256": hashlib.sha256(json.dumps(PREPROC_HASHES, sort_keys=True).encode("utf-8")).hexdigest(),
}
INPUT_CONTRACT = require_expected_contract(CFG.EXPECTED_CONTRACT, ACTUAL_INPUTS)
validate_existing_manifest(CFG.MANIFEST, INPUT_CONTRACT)
print({"input_contract": "verified before model loading", "protocol_sha256": CFG.PROTOCOL_SHA256})

print({"検証(OOF)": len(valid_files_fold), "dense_cut": DENSE_CUT,
       "dense検証画像数": int(sum(v >= DENSE_CUT for v in val_density.values()))})
"""
        ),
        code(
            "# [3/6] MediSegUNet(solar_filament/models/mediseg_models.py と同一)+ refinerロード\n"
            + MODELS
            + """
refiner = MediSegUNet(3, 1, False).cuda().eval()
_verified_refiner = verified_checkpoint_snapshot(
    CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
    INPUT_CONTRACT["refiner_sha256"], CFG.LOCAL_ROOT / "verified_checkpoints")
_ckpt = torch.load(_verified_refiner,
                    map_location="cuda", weights_only=False)
refiner.load_state_dict(_ckpt.get("best_model") or _ckpt["model"], strict=True)
del _ckpt
"""
        ),
        code(
            """# [4/6] 推論・後処理・採点関数(build_v61系と同一経路+refiner内部統計の取得)
from pycocotools import mask as mask_utils
from ultralytics import YOLO

def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    return output, confidences

def mask_square_box(mask, context=1.65, minimum=96):
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
def refine_proposal_stats(image, proposal):
    \"\"\"確定構成のshrink maskと、verifier特徴用のrefiner prob統計を返す。\"\"\"
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    prob_full = np.zeros(proposal.shape, dtype=np.float32)
    prob_full[y0:y1, x0:x1] = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = (prob_full >= CFG.REFINER_THRESHOLD) & proposal
    inside = prob_full[proposal]
    probmass = float(inside.mean()) if inside.size else 0.0
    return refined, probmass

def encode_one(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))

def decode_one(rle):
    return mask_utils.decode(rle) != 0

def greedy_disjoint(masks, order, min_area=CFG.MIN_AREA):
    if not masks:
        return []
    occupied = np.zeros(masks[0].shape, dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept

def drop_contained_indices(masks, threshold=CFG.DROP_THRESHOLD):
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))
    areas = np.asarray(mask_utils.area(rles), dtype=np.float64)
    self_iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
    dropped = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped:
                continue
            value = self_iou[i, j]
            if value <= 0:
                continue
            intersection = value * (areas[i] + areas[j]) / (1.0 + value)
            smaller = i if areas[i] <= areas[j] else j
            if intersection / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]

def apply_ownership(masks, scores):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices(masks)
    return greedy_disjoint(masks, sorted(subset, key=lambda i: areas[i]))

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

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

def mask_iou(a, b):
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0

class PanopticQuality:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.tp_ious = []
        self.fp_count = 0
        self.fn_count = 0
    def add_matrix(self, matrix, n_gt, n_pred):
        if n_gt == 0:
            self.fp_count += n_pred; return
        if n_pred == 0:
            self.fn_count += n_gt; return
        hits = matrix > self.iou_threshold
        scores = matrix[hits].tolist()
        self.tp_ious.extend(scores)
        self.fp_count += int((hits.sum(axis=0) == 0).sum())
        self.fn_count += int((hits.sum(axis=1) == 0).sum())
    @property
    def denominator(self):
        return len(self.tp_ious) + 0.5 * self.fp_count + 0.5 * self.fn_count
    @property
    def score(self):
        d = self.denominator
        return float(sum(self.tp_ious) / d) if d > 0 else 0.0
    def sq_rq(self):
        tp = len(self.tp_ious)
        sq = float(sum(self.tp_ious) / tp) if tp > 0 else 0.0
        rq = float(tp / (tp + 0.5 * self.fp_count + 0.5 * self.fn_count)) if (tp + self.fp_count + self.fn_count) > 0 else 0.0
        return sq, rq
    def summary(self):
        sq, rq = self.sq_rq()
        return {"pq": self.score, "sq": sq, "rq": rq, "tp": len(self.tp_ious),
                "fp": self.fp_count, "fn": self.fn_count, "denominator": self.denominator}

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in annotations_by_image[image_id] if not ann.get("iscrowd", 0)]

print("後処理・採点関数の定義OK")
"""
        ),
        code(
            """# [5/6] prod検出器で1回推論 → ownership後の最終予測をRLE保持 + verifier特徴を記録
def collect_cache(best_path):
    verified_detector = verified_checkpoint_snapshot(
        best_path, INPUT_CONTRACT["detector_sha256"], CFG.LOCAL_ROOT / "verified_checkpoints")
    yolo_model = YOLO(str(verified_detector))
    cache = []
    for name in tqdm(valid_files_fold, desc=f"prod fold{CFG.FOLD} 推論+refine"):
        image_path = CFG.PREPROC_DIR / name
        if file_sha256(image_path) != PREPROC_HASHES[name]:
            raise ValueError(f"Verified image changed before inference: {name}")
        result = yolo_model.predict(str(image_path), imgsz=CFG.YOLO_IMGSZ,
                                     conf=CFG.PREDICT_CONF_FLOOR, iou=CFG.PREDICT_IOU,
                                     agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, confidences = result_masks(result)
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read verified image: {name}")
        if image.shape[:2] != (2048, 2048):
            raise ValueError("Production protocol expects 2048x2048 images")
        refined_masks = []
        feats = []
        for i in keep:
            rmask, probmass = refine_proposal_stats(image, masks[i])
            refined_masks.append(rmask)
            feats.append({
                "conf": float(confidences[i]),
                "raw_area": float(masks[i].sum()),
                "ref_area": float(rmask.sum()),
                "ref_raw_iou": mask_iou(rmask, masks[i]),
                "probmass": probmass,
            })
        scores = confidences[keep] if keep else np.zeros(0, np.float32)
        # 採点対象はownership後の最終予測(確定構成どおり)。
        # refined(pre-ownership)は feats とindex対応するproposal単位の
        # verifier分離度評価に使う。
        final = apply_ownership(refined_masks, scores)
        cache.append({
            "name": name,
            "refined": [encode_one(m) for m in refined_masks],
            "final": [encode_one(m) for m in final],
            "feats": feats,
        })
    del yolo_model
    torch.cuda.empty_cache()
    return cache

torch.cuda.reset_peak_memory_stats()
started = time.time()
cache = collect_cache(prod_checkpoint(CFG.FOLD))
print(f"推論+refine 完了({round((time.time() - started) / 60, 1)}min)、"
      f"画像 {len(cache)}、最終予測総数 {sum(len(e['final']) for e in cache)}")
"""
        ),
        code(
            """# [6/6] Per-annotator diagnostics; GT interventions are not acceptance evidence

def classify_matrix(matrix):
    n_gt, n_pred = matrix.shape
    gt_best = matrix.max(axis=1) if n_pred else np.zeros(n_gt)
    pred_best = matrix.max(axis=0) if n_gt else np.zeros(n_pred)
    gt_hit = gt_best > CFG.IOU_HIT
    labels = {"fn_blind": int((~gt_hit & (gt_best < CFG.FN_WEAK_LO)).sum()),
              "fn_weak": int((~gt_hit & (gt_best >= CFG.FN_WEAK_LO) & (gt_best < CFG.FN_NEAR_LO)).sum()),
              "fn_near": int((~gt_hit & (gt_best >= CFG.FN_NEAR_LO)).sum()),
              "tp_gts": int(gt_hit.sum()), "fp_orphan": 0, "fp_fragment": 0,
              "fp_misassigned": 0, "overmerge_preds": 0,
              "n_gt": n_gt, "n_pred": n_pred}
    for j, best in enumerate(pred_best):
        if best > CFG.IOU_HIT:
            pass
        elif best < CFG.FP_ORPHAN_MAX:
            labels["fp_orphan"] += 1
        elif gt_hit[int(matrix[:, j].argmax())]:
            labels["fp_fragment"] += 1
        else:
            labels["fp_misassigned"] += 1
        if n_gt and (matrix[:, j] > CFG.OVERMERGE_IOU).sum() >= 2:
            labels["overmerge_preds"] += 1
    return labels


def score_final_preds():
    meter = PanopticQuality()
    fn_dense = fn_sparse = 0
    anatomy = {k: 0 for k in classify_matrix(np.zeros((0, 0)))}
    for entry in tqdm(cache, desc="Per-annotator score + diagnostics"):
        for image_id in file_to_ids[entry["name"]]:
            truth, preds = gt_rles[image_id], entry["final"]
            matrix = iou_matrix(truth, preds)
            meter.add_matrix(matrix, len(truth), len(preds))
            labels = classify_matrix(matrix)
            for key, value in labels.items():
                anatomy[key] += value
            fn = labels["fn_blind"] + labels["fn_weak"] + labels["fn_near"]
            if val_density[entry["name"]] >= DENSE_CUT:
                fn_dense += fn
            else:
                fn_sparse += fn
    metrics = meter.summary()
    if (anatomy["fn_blind"] + anatomy["fn_weak"] + anatomy["fn_near"] != metrics["fn"]
            or anatomy["fp_orphan"] + anatomy["fp_fragment"] + anatomy["fp_misassigned"] != metrics["fp"]):
        raise ValueError("Diagnostic counts disagree with official all-hit scoring")
    return metrics, fn_dense, fn_sparse, anatomy


def score_oracle(name):
    if name not in ("drop_orphan", "drop_fragment", "perfect_verifier", "merge"):
        raise ValueError("Unknown oracle diagnostic")
    meter = PanopticQuality()
    for entry in cache:
        preds = [decode_one(r) for r in entry["final"]]
        # Each annotator supplies a separate GT-informed intervention. This is
        # optimistic information unavailable at deployment, not a bound/proposal.
        for image_id in file_to_ids[entry["name"]]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, entry["final"])
            kept = []
            if name == "merge":
                used = set()
                for gi in range(len(truth)):
                    members = [j for j in range(len(preds))
                               if j not in used and matrix[gi, j] > CFG.MERGE_MIN_IOU]
                    if len(members) >= 2:
                        used.update(members)
                        kept.append(np.logical_or.reduce([preds[j] for j in members]))
                kept.extend(preds[j] for j in range(len(preds)) if j not in used)
                kept = apply_ownership(kept, np.ones(len(kept), dtype=np.float32))
            else:
                gt_hit = ((matrix.max(axis=1) > CFG.IOU_HIT)
                          if len(preds) else np.zeros(len(truth), bool))
                for j in range(len(preds)):
                    best = float(matrix[:, j].max()) if len(truth) else 0.0
                    drop = best <= CFG.IOU_HIT if name == "perfect_verifier" else (
                        best < CFG.FP_ORPHAN_MAX if name == "drop_orphan" else (
                            best <= CFG.IOU_HIT and best >= CFG.FP_ORPHAN_MAX
                            and gt_hit[int(matrix[:, j].argmax())]))
                    if not drop:
                        kept.append(preds[j])
            pred_rles = [encode_one(m) for m in kept]
            meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
    return meter.summary()


def rank_auc(pos, neg):
    from scipy.stats import rankdata
    values = np.asarray(list(pos) + list(neg), dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("AUC requires finite scalar scores")
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = rankdata(values, method="average")
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def overlap_feature_auc():
    # Labels measure pre-ownership refined-mask overlap, not matched TP status.
    positive, negative = defaultdict(list), defaultdict(list)
    n_pos = n_neg = 0
    for entry in cache:
        for image_id in file_to_ids[entry["name"]]:
            matrix = iou_matrix(gt_rles[image_id], entry["refined"])
            for j, feat in enumerate(entry["feats"]):
                hit = bool(matrix[:, j].max() > CFG.IOU_HIT) if matrix.shape[0] else False
                if hit:
                    n_pos += 1
                else:
                    n_neg += 1
                bucket = positive if hit else negative
                for key, value in feat.items():
                    bucket[key].append(value)
    return ({k: rank_auc(positive[k], negative[k]) for k in sorted(set(positive) | set(negative))},
            {"overlap_positive": n_pos, "overlap_negative": n_neg})


baseline_metrics, fn_dense, fn_sparse, anatomy = score_final_preds()
validate_baseline(baseline_metrics["pq"], INPUT_CONTRACT["baseline"]["pq"],
                  INPUT_CONTRACT["baseline"]["tolerance"])
oracle_rows = []
for name in CFG.ORACLES:
    metrics = score_oracle(name)
    oracle_rows.append({"diagnostic": name, "metrics": metrics,
                        "delta_pq": metrics["pq"] - baseline_metrics["pq"]})
auc_rows, overlap_counts = overlap_feature_auc()
manifest = {
    "experiment_id": f"v62_error_anatomy_fold{CFG.FOLD}_diagnostic_v1",
    "input_contract": INPUT_CONTRACT,
    "scope": "exploratory GT-informed diagnostics; no mechanism selection or acceptance",
    "count_unit": "annotator_image", "matching": "official all IoU > 0.5 hits",
    "baseline": {"metrics": baseline_metrics, "fn_dense_val": fn_dense, "fn_sparse_val": fn_sparse},
    "anatomy": anatomy, "oracle_diagnostics": oracle_rows,
    "overlap_feature_auc": auc_rows, "overlap_counts": overlap_counts,
    "runtime": {"service": "Google Colab", "gpu": torch.cuda.get_device_name(0),
                "duration_seconds": time.time() - started,
                "measurement_scope": "inference/refinement and diagnostic scoring; setup excluded",
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "compute_units_used": None, "compute_units_note": "Record from Colab UI; not exposed by runtime API",
                "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": version("scipy"),
                             "opencv": cv2.__version__, "ultralytics": version("ultralytics"),
                             "pycocotools": version("pycocotools")}},
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
validate_existing_manifest(CFG.MANIFEST, INPUT_CONTRACT)
atomic_json(CFG.MANIFEST, manifest)
print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
print("Exploratory diagnostics only. No automatic mechanism adoption/rejection.")
"""
        ),
    ]
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["0", "1", "2"], help="評価対象fold")
    args = parser.parse_args()
    fold = int(args.fold)

    cells = build_cells(fold)
    for index, cell in enumerate(cells):
        cell["id"] = f"v62-anatomy-f{fold}-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output = ROOT / f"solar_filament_v62_error_anatomy_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
