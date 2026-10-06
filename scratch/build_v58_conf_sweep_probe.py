"""提案 conf スイープ プローブ notebook builder — v57(§6.16)の陰性結果を受けた検証。

## 目的

v57 で dense2x は dense FN を減らした(−49/−19)が FP が爆増(+229/+146)して
refined PQ が2seedとも悪化した。**提案品質を conf で絞れば FN 改善を残したまま
FP を削れる可能性**がある。また現行の操作点 CONF=0.26 自体が最適かどうかは
この評価経路では一度も掃引していない。

学習は一切なし。**既存の検出器 ckpt を使い回して推論+refiner+採点だけを行う**:

  - `prod`        — 現行提出と同じ `runs/yolo_fold_{N}/weights/best.pt`
  - `v57baseline` — v57 で再学習した V30 同等検出器(seed 3407/8407)
  - `v57dense2x`  — v57 の dense2x 検出器(seed 3407/8407)

計5本 × conf 候補スイープで、raw PQ / refined PQ / dense・sparse 別 FN の
曲線を fold{N} OOF 上に出す。

## 設計上の注意(選択バイアス対策)

このスイープは**評価foldそのもの上での探索**であり、採用閾値をここで決めると
「fold0選択バイアス」(free variant 0.86 の前科、README §2.1)を繰り返す。
出力は探索結果として扱い、候補点が見つかった場合のみ:

  1. 両config(baseline/dense2x)・両seedで同じ方向に改善する点を選び、
  2. fold1・fold2 でその点だけを独立確認する

までを採用条件とする。ここで決め打ちしない。

## 使い方

    python scratch/build_v58_conf_sweep_probe.py --fold 0

出力: solar_filament_v58_conf_sweep_fold{N}_colab.ipynb (Colab GPU推奨、学習なし)
前提: 対象foldの v57 ckpt(4本)と現行 `yolo_fold_{N}` ckpt が Drive にあること。
fold0 は過去の v57 から5本を想定するが、現存する重みは実行時に確認する。fold1/2 は v57 未実行なら prod のみで動く。
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")

# 標準refinerパイプラインの全fold OOF基準(README「現在地」)
BASELINE_PQ_REFINED = {0: 0.4393, 1: 0.4415, 2: 0.4471}
CONF_SWEEP = (0.05, 0.08, 0.12, 0.16, 0.20, 0.26, 0.32, 0.40)
PREDICT_FLOOR = 0.03  # スイープ下限より低く取り、あとからの絞り込みだけで評価する
DENSE_QUANTILE = 0.75

PROTOCOL_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    + MODELS.replace("\r\n", "\n").encode("utf-8")
).hexdigest()


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(fold):
    baseline_refined = BASELINE_PQ_REFINED[fold]

    cells = [
        markdown(
            f"""# 提案 conf スイープ プローブ — fold{fold}(学習なし、既存ckptの使い回し)

v57(§6.16)で dense2x は dense FN を減らしたが FP 爆増で refined PQ が悪化した。
**conf 閾値を動かして「FN改善を残しつつFPを削る」点がないか**を、学習済みckpt
5本(prod / v57baseline×2seed / v57dense2x×2seed)に対して評価する。
副次的に現行の操作点 CONF=0.26 が妥当かも確認できる。

| 列 | 内容 |
|---|---|
| raw PQ | 生YOLO proposals → ownership(refinerなし) |
| refined PQ | proposals → 既存標準refiner → shrink@0.45 → ownership(確定構成) |
| FN(dense/sparse) | 検証画像を train q75 cut でスライスした未マッチGT数 |

fold{fold}基準: refined PQ {baseline_refined}。**注意: これは評価fold上の探索であり、
ここで選んだ閾値をそのまま採用しない(fold0選択バイアスの前科あり)。有望点は
fold1/2での独立確認が必須。** CPU 契約を検証した補修版で、GPU受入・操作点採用は別途必要。
旧manifestは入力契約が無ければ再利用せず、履歴を保持して新しい保存先を選ぶ。
fold1/2 では CFG.SELECTED_CONF に fold0 で事前選択した値を指定する。未指定なら評価前に停止する。
GPU必須、学習なしで合計1〜1.5時間見込み。
"""
        ),
        code(
            """# [1/6] Drive接続、設定、前提の確認
!pip -q install ultralytics pycocotools
import hashlib, json, math, random, shutil, sys, time, warnings
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
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
    LOCAL_ROOT = Path("/content/solar_v58_confsweep")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    FOLD = %d
    PROTOCOL_SHA256 = %r
    # --- OOF評価は確定構成(build_v56/v57系と同一) ---
    YOLO_IMGSZ = 1280
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    PREDICT_CONF_FLOOR = %r
    PREDICT_IOU = 0.55
    CONF_SWEEP = %r
    SELECTED_CONF = None  # fold1/2: set the single threshold selected on fold0 before running
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ_REFINED = %r
    DENSE_QUANTILE = %r
    MANIFEST = DRIVE_ROOT / "artifacts" / "v58_conf_sweep_fold%d_manifest.json"

def confirmation_confidences():
    if CFG.FOLD == 0:
        return CFG.CONF_SWEEP
    value = CFG.SELECTED_CONF
    if type(value) not in (int, float) or not math.isfinite(value) or not CFG.PREDICT_CONF_FLOOR <= value <= 1:
        raise ValueError("Set SELECTED_CONF to the single threshold preselected on fold0; confirmation folds cannot sweep")
    return (value,)

CFG.CONF_SWEEP = confirmation_confidences()

def checkpoint_map(fold):
    \"\"\"label -> best.pt。存在しないものは呼び出し側でskipする。\"\"\"
    runs = CFG.DRIVE_ROOT / "runs"
    return {
        "prod": runs / f"yolo_fold_{fold}" / "weights" / "best.pt",
        "v57baseline_s3407": runs / "v57_dense_oversample" / f"yolo_v57_baseline_s3407_fold{fold}" / "weights" / "best.pt",
        "v57baseline_s8407": runs / "v57_dense_oversample" / f"yolo_v57_baseline_s8407_fold{fold}" / "weights" / "best.pt",
        "v57dense2x_s3407": runs / "v57_dense_oversample" / f"yolo_v57_dense2x_s3407_fold{fold}" / "weights" / "best.pt",
        "v57dense2x_s8407": runs / "v57_dense_oversample" / f"yolo_v57_dense2x_s8407_fold{fold}" / "weights" / "best.pt",
    }

for path in [CFG.LOCAL_ROOT, CFG.PREPROC_DIR]:
    path.mkdir(parents=True, exist_ok=True)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像キャッシュ": CFG.PREPROC_CACHE,
    f"fold{CFG.FOLD} 標準refiner": CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({k: "OK" for k in required})
RUNTIME_INFO = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "service": "Google Colab", "project": f"solar_v58_fold{CFG.FOLD}",
                "GPU": torch.cuda.get_device_name(0), "execution_scope": "inference_only",
                "compute_units": None,
                "compute_units_unavailable_reason": "Colab runtime API does not expose billing usage"}
print(RUNTIME_INFO)
"""  # noqa: UP031 - preserve braces in embedded Python source
            % (fold, PROTOCOL_SHA256, PREDICT_FLOOR, CONF_SWEEP if fold == 0 else (), baseline_refined, DENSE_QUANTILE, fold)
        ),
        code(
            """# [2/6] COCOとfoldを読み、密度スライス用のcutを計算し、前処理画像をローカルSSDへ展開
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"学習annotation JSONが見つかりません: {base}")
    return matches[0].parents[2]

TRAIN_JSON = next(find_dataset_root(CFG.DATA_CACHE).rglob(
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

images = {item["id"]: item for item in coco["images"]}
annotations_by_image = defaultdict(list)
for ann in coco["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)

fold_frame = pd.read_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv")
if fold_frame["file_name"].duplicated().any() or not fold_frame["fold"].isin([0, 1, 2]).all():
    raise ValueError("grouped folds must have unique filenames and integer folds 0/1/2")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
if any(name not in fold_of_file for name in train_files):
    raise ValueError("Observations missing from grouped folds")
train_files_fold = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]

def noncrowd_count(image_id):
    return sum(1 for a in annotations_by_image[image_id] if not a.get("iscrowd", 0))

# v57と同じ定義: cutは学習サンプル(file,annotator)のGT数q75
train_sample_counts = [
    noncrowd_count(image_id)
    for name in train_files_fold
    for image_id in file_to_ids[name]
]
if not train_sample_counts or not valid_files_fold:
    raise ValueError("Both training and validation observations are required")
DENSE_CUT = float(np.quantile(train_sample_counts, CFG.DENSE_QUANTILE))
val_density = {
    name: float(np.mean([noncrowd_count(i) for i in file_to_ids[name]]))
    for name in valid_files_fold
}

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

preprocessed_hashes = {}
for name in tqdm(valid_files_fold, desc="前処理画像をSSDへ・SHA256"):
    local = CFG.PREPROC_DIR / name
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG.PREPROC_CACHE / name, local)
    preprocessed_hashes[name] = file_sha256(local)

PILOT_CONTRACT = {
    "schema_version": 1, "fold": CFG.FOLD, "protocol_sha256": CFG.PROTOCOL_SHA256,
    "annotations_sha256": file_sha256(TRAIN_JSON),
    "folds_sha256": file_sha256(CFG.ARTIFACT_ROOT / "grouped_folds.csv"),
    "refiner_sha256": file_sha256(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt"),
    "checkpoint_sha256": {label: file_sha256(path.resolve(strict=True))
                          for label, path in checkpoint_map(CFG.FOLD).items() if path.is_file()},
    "preprocessed_sha256": hashlib.sha256(
        json.dumps(preprocessed_hashes, sort_keys=True).encode("utf-8")).hexdigest(),
    "dense_quantile": CFG.DENSE_QUANTILE, "dense_cut": DENSE_CUT,
    "conf_sweep": list(CFG.CONF_SWEEP),
    "recipe": {name: getattr(CFG, name) for name in (
        "YOLO_IMGSZ", "CROP_SIZE", "CROP_CONTEXT", "PREDICT_CONF_FLOOR", "PREDICT_IOU",
        "REFINER_THRESHOLD", "DROP_THRESHOLD", "MIN_AREA")},
    "runtime_versions": {"python": sys.version, "torch": torch.__version__,
                         "ultralytics": version("ultralytics"), "numpy": np.__version__,
                         "opencv": cv2.__version__, "pycocotools": version("pycocotools")},
}

def require_checkpoint(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"{path}: checkpoint changed after input contract; restart with fresh inputs")

def save_probe_manifest(manifest):
    import os
    import tempfile
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CFG.MANIFEST.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, CFG.MANIFEST)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def read_probe_manifest():
    if not CFG.MANIFEST.exists():
        return {"experiment_id": f"v58_conf_sweep_fold{CFG.FOLD}",
                "input_contract": PILOT_CONTRACT, "runs": []}
    manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("input_contract") != PILOT_CONTRACT
            or manifest.get("experiment_id") != f"v58_conf_sweep_fold{CFG.FOLD}"):
        raise ValueError("Manifest input contract differs or is absent; retain it and choose a fresh manifest path")
    known = set(PILOT_CONTRACT["checkpoint_sha256"])
    seen = set()
    if not isinstance(manifest.get("runs"), list):
        raise ValueError("Manifest rows must be a list")
    for row in manifest["runs"]:
        if not isinstance(row, dict):
            raise ValueError("Invalid manifest row")
        key = (row.get("ckpt"), row.get("conf"))
        if (key in seen or key[0] not in known or key[1] not in CFG.CONF_SWEEP
                or row.get("fold") != CFG.FOLD):
            raise ValueError("Invalid or duplicate manifest row")
        path = row.get("checkpoint")
        if not isinstance(path, str) or Path(path).resolve() != checkpoint_map(CFG.FOLD)[key[0]].resolve():
            raise ValueError("Manifest row checkpoint path differs from its label")
        for mode in ("raw", "refined"):
            if not isinstance(row.get(mode), dict):
                raise ValueError("Manifest row has invalid metrics")
            pq = row[mode].get("pq")
            if not isinstance(pq, (int, float)) or not math.isfinite(pq) or not 0 <= pq <= 1:
                raise ValueError("Manifest row has invalid PQ")
        counts = [row.get("fn_dense_val"), row.get("fn_sparse_val"),
                  row["refined"].get("fp"), row["refined"].get("fn")]
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Manifest row must contain nonnegative integer FN/FP counts")
        seen.add(key)
    return manifest

# Fail before loading/refining GPU models or changing the historical manifest.
read_probe_manifest()

print({"検証(OOF)": len(valid_files_fold), "dense_cut": DENSE_CUT,
       "dense検証画像数": int(sum(v >= DENSE_CUT for v in val_density.values()))})
"""
        ),
        code(
            "# [3/6] MediSegUNet (solar_filament/models/mediseg_models.py と同一。標準refinerの評価に使用)\n"
            + MODELS
        ),
        code(
            """# [4/6] 推論・後処理・採点関数(build_v57系と同一経路。提案ごとのrefinedは1回だけ計算してRLEで保持)
from pycocotools import mask as mask_utils
from ultralytics import YOLO

require_checkpoint(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
                   PILOT_CONTRACT["refiner_sha256"])
refiner = MediSegUNet(3, 1, False).cuda().eval()
_ckpt = torch.load(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
                    map_location="cuda", weights_only=False)
refiner.load_state_dict(_ckpt.get("best_model") or _ckpt["model"], strict=True)
del _ckpt

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
def refine_proposal(image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal  # shrink(確定構成)

def encode_one(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))

def decode_one(rle):
    return mask_utils.decode(rle) != 0

def greedy_disjoint(masks, order, min_area=CFG.MIN_AREA):
    occupied = np.zeros((2048, 2048), dtype=bool)
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
            """# [5/6] ckptごとに「floor推論→全提案をrefineしてRLE保持」→confスイープはメモリ上で評価
def collect_cache(best_path, label):
    \"\"\"floor(conf=%s)で全検証画像をpredictし、raw/refined両方のRLE+score+面積を保持。\"\"\"
    if label not in PILOT_CONTRACT["checkpoint_sha256"] or best_path.resolve() != checkpoint_map(CFG.FOLD)[label].resolve():
        raise ValueError("Evaluation checkpoint label/path differs from input contract")
    require_checkpoint(best_path, PILOT_CONTRACT["checkpoint_sha256"][label])
    yolo_model = YOLO(str(best_path))
    cache = []
    for name in tqdm(valid_files_fold, desc=f"{label} fold{CFG.FOLD} 推論+refine"):
        image_path = CFG.PREPROC_DIR / name
        result = yolo_model.predict(str(image_path), imgsz=CFG.YOLO_IMGSZ,
                                     conf=CFG.PREDICT_CONF_FLOOR, iou=CFG.PREDICT_IOU,
                                     agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, confidences = result_masks(result)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        entry = {"name": name, "scores": confidences, "areas": [], "raw": [], "refined": []}
        for mask in masks:
            entry["areas"].append(float(mask.sum()))
            entry["raw"].append(encode_one(mask))
            entry["refined"].append(encode_one(refine_proposal(image, mask)))
        entry["areas"] = np.asarray(entry["areas"], dtype=np.float64)
        cache.append(entry)
    del yolo_model
    torch.cuda.empty_cache()
    return cache

def score_conf(cache, conf):
    \"\"\"保持済みRLEを conf で絞り、ownership→PQ を raw/refined 両方で採点 + 密度別FN。\"\"\"
    meters = {"raw": PanopticQuality(), "refined": PanopticQuality()}
    fn_dense = fn_sparse = 0
    for entry in cache:
        keep = [i for i in range(len(entry["scores"]))
                if entry["scores"][i] >= conf and entry["areas"][i] >= CFG.MIN_AREA]
        scores = entry["scores"][keep] if keep else np.zeros(0, np.float32)
        name = entry["name"]
        pred_rles_by_mode = {}
        for mode in ("raw", "refined"):
            masks = [decode_one(entry[mode][i]) for i in keep]
            preds = apply_ownership(masks, scores)
            pred_rles = []
            if preds:
                encoded = mask_utils.encode(np.asfortranarray(np.stack(preds, axis=-1).astype(np.uint8)))
                pred_rles = encoded if isinstance(encoded, list) else [encoded]
            pred_rles_by_mode[mode] = pred_rles
        file_fn = 0
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            for mode in ("raw", "refined"):
                matrix = iou_matrix(truth, pred_rles_by_mode[mode])
                meters[mode].add_matrix(matrix, len(truth), len(pred_rles_by_mode[mode]))
                if mode == "refined":
                    if matrix.size:
                        file_fn += int((matrix.max(axis=1) <= 0.5).sum())
                    else:
                        file_fn += len(truth)
        if val_density[name] >= DENSE_CUT:
            fn_dense += file_fn
        else:
            fn_sparse += file_fn
    return {"raw": meters["raw"].summary(), "refined": meters["refined"].summary(),
            "fn_dense_val": fn_dense, "fn_sparse_val": fn_sparse}
"""  # noqa: UP031 - preserve braces in embedded Python source
            % PREDICT_FLOOR
        ),
        code(
            """# [6/6] ckpt×confを直列実行、runごとにmanifestへ逐次追記(途中再開対応)
manifest = read_probe_manifest()
rows = manifest["runs"]
done_keys = {(r["ckpt"], r["conf"]) for r in rows}
available = {label: p for label, p in checkpoint_map(CFG.FOLD).items() if p.is_file()}
if not available:
    raise FileNotFoundError("No evaluation checkpoint available")
missing_labels = sorted(set(checkpoint_map(CFG.FOLD)) - set(available))
manifest["missing_checkpoints"] = missing_labels
manifest["status"] = "partial_checkpoint_set" if missing_labels else "in_progress"
manifest["runtime"] = {**RUNTIME_INFO,
                       "execution_scope": "inference_only", "completion": "in_progress"}
session_started = time.time()
torch.cuda.reset_peak_memory_stats()
for label, path in checkpoint_map(CFG.FOLD).items():
    if not path.exists():
        print(f"{label}: ckptなし({path}) — skip")

try:
    for label, best_path in available.items():
        pending = [c for c in CFG.CONF_SWEEP if (label, c) not in done_keys]
        if not pending:
            print(f"{label}: manifest済み、skip")
            continue
        started = time.time()
        cache = collect_cache(best_path, label)  # 残っているconfがある限り再収集
        for conf in CFG.CONF_SWEEP:
            if (label, conf) in done_keys:
                continue
            metrics = score_conf(cache, conf)
            row = {"ckpt": label, "conf": conf, "fold": CFG.FOLD,
                   "checkpoint": str(best_path), **metrics}
            rows.append(row)
            done_keys.add((label, conf))
            manifest["runs"] = rows
            manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_probe_manifest(manifest)
            print(json.dumps(row, ensure_ascii=False))
        del cache
        torch.cuda.empty_cache()
        print(f"{label}: 完了({round((time.time() - started) / 60, 1)}min)")
except (Exception, KeyboardInterrupt) as error:
    manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    try:
        peak_vram = int(torch.cuda.max_memory_allocated())
        peak_note = None
    except Exception as stats_error:
        peak_vram = None
        peak_note = f"CUDA memory statistics unavailable: {type(stats_error).__name__}"
    manifest["runtime"].update({"duration_seconds": round(time.time() - session_started, 3),
                                "duration_scope": "current_inference_session",
                                "peak_vram_bytes": peak_vram,
                                "peak_vram_unavailable_reason": peak_note,
                                "completion": manifest["status"], "error_type": type(error).__name__})
    save_probe_manifest(manifest)
    raise
manifest["status"] = "partial_checkpoint_set" if missing_labels else "complete"
manifest["runtime"].update({"duration_seconds": round(time.time() - session_started, 3),
                            "duration_scope": "current_inference_session",
                            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                            "completion": "completed_available_checkpoints"})
save_probe_manifest(manifest)

print(f"\\n保存: {CFG.MANIFEST}")

# --- 集計: conf×refined PQ の曲線 + paired 比較 ---
print("\\n=== refined PQ 曲線(ckpt別) ===")
for label in available:
    series = sorted((r["conf"], r["refined"]["pq"], r["raw"]["pq"],
                     r["fn_dense_val"], r["fn_sparse_val"],
                     r["refined"]["fp"], r["refined"]["fn"])
                    for r in rows if r["ckpt"] == label)
    for conf, pq, raw_pq, fn_d, fn_s, fp, fn in series:
        print(f"{label} conf={conf:.2f}: refined {pq:.4f} | raw {raw_pq:.4f} | "
              f"FP {fp} FN {fn} | FN(dense/sparse) {fn_d}/{fn_s}")

print("\\n=== paired Δ(v57dense2x - v57baseline、同seed・同conf) ===")
for seed in (3407, 8407):
    d_label, b_label = f"v57dense2x_s{seed}", f"v57baseline_s{seed}"
    for conf in CFG.CONF_SWEEP:
        d = next((r for r in rows if r["ckpt"] == d_label and r["conf"] == conf), None)
        b = next((r for r in rows if r["ckpt"] == b_label and r["conf"] == conf), None)
        if d and b:
            print(f"seed {seed} conf={conf:.2f}: Δrefined {d['refined']['pq'] - b['refined']['pq']:+.4f} | "
                  f"ΔFN(dense) {d['fn_dense_val'] - b['fn_dense_val']:+d} | "
                  f"ΔFP {d['refined']['fp'] - b['refined']['fp']:+d}")

print(f"\\n採否の目安: fold0 のみ探索、fold1/2 は事前選択した単一confの確認。dense2xが「両seed・同じconfで」"
      "baselineを上回る点があれば、そのconfだけをfold1/2で独立確認(v57のfold1/2 ckptが必要)。"
      "prodの曲線は現行CONF=0.26が妥当かの健全性確認として使う。")
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
        cell["id"] = f"v58-confsweep-f{fold}-{index:02d}"

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

    output = ROOT / f"solar_filament_v58_conf_sweep_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
