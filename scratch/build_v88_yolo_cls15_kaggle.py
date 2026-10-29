"""V88 Kaggle 版ノートのビルダー: build_v88_yolo_cls15.py の Colab 版を
Kaggle dataset input で動くようにしたもの。評価経路・ゲートは完全同一。

入力 (kernel-metadata.json の sources):
- competition: filament-segmentation-2026 (COCO annotation json)
- dataset: shingo257/solar-filament-v35-artifacts (preprocessed_v30_q96/ +
  grouped_folds.csv)
- dataset: shingo257/solar-filament-2026-weights (yolo_fold2_best.pt = prod)
- dataset: shingo257/solar-v73-assets
  (mediseg_edgeattnet_refiner_fold2_resume.pt = fold2 refiner)

出力: /kaggle/working/v88_yolo_cls15_fold2_manifest.json (+ runs/ 学習済み重み)
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v88_yolo_cls15_fold2_kaggle.ipynb"
MEDISEG = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# v88 — YOLO cls=1.5 再学習 (fold2, Kaggle 版)

arm: `cls15` + paired baseline `prod` (solar-filament-2026-weights の
`yolo_fold2_best.pt`)。**変更は `cls=1.5` のみ**、他は prod と完全一致。

**事前固定ゲート** (RESEARCH_LOG §2.40):

- `PQ(cls15,0.26) − PQ(prod,0.26) >= +0.005`: fold0/1 に昇格検討
- `[0, +0.005)`: reject_noise
- `<0`: reject

**Add Input 必須**: competition `filament-segmentation-2026` +
dataset `shingo257/solar-filament-v35-artifacts` +
`shingo257/solar-filament-2026-weights` + `shingo257/solar-v73-assets`。
Internet ON (ultralytics の pip に必要)。GPU 推奨: L4 以上。
"""
    ),
    code(
        """# [1/6] 依存・input 解決・設定
%pip install -q "ultralytics>=8.3,<9"
import hashlib, json, math, os, random, shutil, sys, tempfile, time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO


def find_input(filename, contains=None):
    matches = [p for p in Path("/kaggle/input").rglob(filename)
               if contains is None or contains in p.parts]
    if not matches:
        raise FileNotFoundError(f"/kaggle/input に {filename} がない (contains={contains})")
    return matches[0]


class CFG:
    SEED = 3407
    FOLD = 2
    ARMS = ('cls15',)
    # Drive artifacts/grouped_folds.csv と同じ凍結分割 (sha256 は実行時検証)
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    FOLDS_CSV = find_input("grouped_folds.csv", "solar-filament-v35-artifacts")
    PREPROC_DIR = find_input("preprocessed_v30_q96", "solar-filament-v35-artifacts")
    WORK = Path("/kaggle/working/v88_yolo_cls15")
    RUNS_ROOT = WORK / "runs"
    YOLO_ROOT = WORK / "yolo_datasets"
    PROD_YOLO = find_input("yolo_fold2_best.pt", "solar-filament-2026-weights")
    REFINER_CKPT = find_input("mediseg_edgeattnet_refiner_fold2_resume.pt", "solar-v73-assets")
    MANIFEST = WORK / "v88_yolo_cls15_fold2_manifest.json"
    # --- YOLO (master §B と同一。cls 以外変更禁止) ---
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    YOLO_CLS = 1.5  # prod=0.5 (ultralytics default)。biconcavelens 検証値
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

for path in (CFG.WORK, CFG.RUNS_ROOT, CFG.YOLO_ROOT):
    path.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
RUNTIME_INFO = {"started_utc": datetime.now(timezone.utc).isoformat(), "GPU": torch.cuda.get_device_name(0),
                "python": sys.version, "torch": torch.__version__, "ultralytics": version("ultralytics"),
                "pycocotools": version("pycocotools"), "opencv": cv2.__version__, "numpy": np.__version__,
                "folds_csv": str(CFG.FOLDS_CSV), "preproc": str(CFG.PREPROC_DIR),
                "prod_yolo": str(CFG.PROD_YOLO), "refiner": str(CFG.REFINER_CKPT)}
print(RUNTIME_INFO)
"""
    ),
    code(
        """# [2/6] COCO・凍結 fold (SHA256 検証)
import re

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

TRAIN_JSON = next(Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
coco = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)

folds_sha256 = file_sha256(CFG.FOLDS_CSV)
if folds_sha256 != CFG.GROUPED_FOLDS_SHA256:
    raise ValueError(f"grouped_folds.csv の SHA256 が不一致: {folds_sha256}")
fold_frame = pd.read_csv(CFG.FOLDS_CSV)
if fold_frame["file_name"].duplicated().any() or set(fold_frame["file_name"]) != set(train_files):
    raise ValueError("fold CSV が全観測を一意にカバーしていない")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
valid_files = [n for n in train_files if fold_of_file[n] == CFG.FOLD]
fit_files = [n for n in train_files if fold_of_file[n] != CFG.FOLD]

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
       "folds_sha256": folds_sha256[:16], "preproc_images": len(list(CFG.PREPROC_DIR.glob('*.jpeg')))})
"""
    ),
    code(
        """# [3/6] prod all-labeler データセット構築 + YOLO 学習 (cls=1.5 のみ変更)
def polygon_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    polygon = max(polygons, key=len)  # MAGFiLOは1個体1polygon。複数返る不正データへの安全策
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.reshape(-1))

def write_variant(fold_root, split, file_name, image_id, suffix):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{stem}.jpeg"
    label_out = fold_root / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    if not image_out.exists():
        shutil.copy2((CFG.PREPROC_DIR / file_name).resolve(), image_out)
    lines = [polygon_line(ann, info["width"], info["height"])
             for ann in annotations[image_id] if not ann.get("iscrowd", 0)]
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(line for line in lines if line), encoding="utf-8")

def build_fold_dataset(fold):
    root = CFG.YOLO_ROOT / f"fold_{fold}"
    reference_id = {name: file_to_ids[name][0] for name in train_files}
    stats = {"train_samples": 0, "train_instances": 0, "val_samples": 0, "val_instances": 0}
    for file_name in tqdm(train_files, desc=f"fold {fold} データセット構築"):
        if fold_of_file[file_name] == fold:
            write_variant(root, "val", file_name, reference_id[file_name], "")
            stats["val_samples"] += 1
            stats["val_instances"] += len(annotations[reference_id[file_name]])
        else:
            for image_id in file_to_ids[file_name]:
                write_variant(root, "train", file_name, image_id, f"__labeler_{image_id}")
                stats["train_samples"] += 1
                stats["train_instances"] += len(annotations[image_id])
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
                         encoding="utf-8")
    return yaml_path, stats

def train_arm(arm):
    run_name = f"yolo_{arm}_fold{CFG.FOLD}"
    run_dir = CFG.RUNS_ROOT / run_name
    best_path, done_path = run_dir / "weights" / "best.pt", run_dir / "TRAINING_COMPLETE"
    yaml_path, stats = build_fold_dataset(CFG.FOLD)  # 既存ファイルは skip (冪等)
    if done_path.exists() and best_path.exists():
        print(f"{arm}: 学習済み checkpoint を再利用 {best_path}")
        return best_path, stats
    started = time.time()
    YOLO(CFG.YOLO_MODEL).train(
        data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
        batch=CFG.YOLO_BATCH, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
        cls=CFG.YOLO_CLS,  # ★ prod との唯一の差分 (default 0.5 → 1.5)
        optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(CFG.RUNS_ROOT), name=run_name, exist_ok=True,
    )
    stats["train_minutes"] = round((time.time() - started) / 60, 1)
    done_path.write_text("ok", encoding="utf-8")
    return best_path, stats
"""
    ),
    code(
        """# [4/6] refiner モデル定義 (solar_filament/models/mediseg_models.py と同一)
"""
        + MEDISEG
    ),
    code(
        """# [5/6] 評価関数 (v87 と同一経路 + 低conf診断)
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
"""
    ),
    code(
        """# [6/6] 実行: prod baseline → cls15 学習 → 評価 → ゲート判定、manifest を逐次保存
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

CONTRACT = {"experiment_id": "v88_yolo_cls15", "fold": CFG.FOLD, "arms": list(CFG.ARMS),
            "protocol": "RESEARCH_LOG §2.40 (cls=1.5 retrain, biconcavelens 診断の検証)",
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,
            "conf_sweep": list(CFG.CONF_SWEEP), "primary_conf": CFG.PRIMARY_CONF,
            "gate": f"PQ(arm,{CFG.PRIMARY_CONF}) - PQ(prod,{CFG.PRIMARY_CONF}) >= +{CFG.GATE_DELTA}"}
manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8")) if CFG.MANIFEST.exists() else {}
if manifest and manifest.get("contract") != CONTRACT:
    raise ValueError("既存 manifest と入力契約が違う。履歴保持のため別パスを指定してください")
manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}

if "prod" not in manifest["results"]:
    manifest["results"]["prod"] = {"weights": str(CFG.PROD_YOLO), "by_conf": evaluate_yolo(CFG.PROD_YOLO, "prod")}
    save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]
print({"prod@primary": round(prod_pq, 6), "既知値": CFG.BASELINE_PQ, "差": round(prod_pq - CFG.BASELINE_PQ, 6)})
if abs(prod_pq - CFG.BASELINE_PQ) > 0.002:
    print("⚠️ prod の再現値が既知値から 0.002 以上ずれている — 配線/入力の差を確認 (判定は paired Δ で行う)")

for arm in CFG.ARMS:
    if arm in manifest["results"]:
        continue
    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)
    delta = by_conf[str(CFG.PRIMARY_CONF)]["pq"] - prod_pq
    best_conf = max((c for c in by_conf if c != "diag_lowconf"), key=lambda c: by_conf[c]["pq"])
    verdict = "promote_to_fold01" if delta >= CFG.GATE_DELTA else ("reject_noise" if delta >= 0 else "reject")
    manifest["results"][arm] = {"weights": str(weights), "weights_sha256": file_sha256(weights),
                                "by_conf": by_conf, "delta_primary_vs_prod": delta,
                                "delta_sparse_primary_vs_prod": by_conf[str(CFG.PRIMARY_CONF)]["pq_sparse_r077"]
                                    - manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq_sparse_r077"],
                                "exploratory_best_conf": best_conf,
                                "exploratory_best_delta": by_conf[best_conf]["pq"] - prod_pq,
                                "verdict": verdict}
    manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    print(arm, {"Δ@primary": round(delta, 4), "verdict": verdict})

print(json.dumps({a: {c: round(r["by_conf"][c]["pq"], 4) for c in r["by_conf"] if c != "diag_lowconf"}
                  for a, r in manifest["results"].items()}, indent=2))
print("manifest:", CFG.MANIFEST)
"""
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": cells,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUTPUT}")
