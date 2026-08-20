"""Kaggleネイティブ・モデルサイズ拡大プローブ v2 のビルダー(fold0のみ)。

## 背景(V39からの引き継ぎ、改訂8)

V39(yolo11l-seg、batch4→2、lr0=7e-4据え置き)はfold0 OOF PQ 0.4328で不合格だった
(基準0.4393)。生YOLO(refinerなし)でもV30基準を下回っており、検出器そのものが
悪化していた。batchを半分にしたのにlr0を据え置いており、実効バッチサイズ減少で
勾配が不安定化した疑いが強い(線形スケーリング則ではbatchに比例してlrも下げる)。

## v2での変更点

- `lr0`: 7e-4 → **3.5e-4**(batch4→2の線形スケーリング則に合わせて半減)
- モデル(yolo11l-seg)・batch(2)・epoch(70)・augmentationはV39と同一

## なぜfold0だけか

3-fold全部を70 epochで回すのは高い。「単一foldのbestを一般化しない」という
このプロジェクトの方針どおり、**採用判定はしない**。fold0 OOF PQがV35の
fold0実測 0.4393 を上回ったときだけ fold1/2 に進む(P2-2/P2-3/V38と同じゲート)。

## Kaggleネイティブ(Drive不要)

前処理済み画像・fold分割・refiner checkpointは Kaggle Dataset
`shingo257/solar-filament-v35-artifacts` から読む。COCO annotationは
competition `filament-segmentation-2026` を直接アタッチする。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v41_kaggle_modelsize_v2.ipynb"
MEDISEG = (ROOT / "src" / "mediseg_models.py").read_text(encoding="utf-8")
METRICS = (ROOT / "src" / "metrics_official.py").read_text(encoding="utf-8")
METRICS = METRICS.replace("from __future__ import annotations\n", "", 1)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V41 Kaggle探針(v2: lr補正) — モデルサイズ拡大(yolo11m-seg → yolo11l-seg、fold0のみ)

V39(lr0=7e-4据え置き)は不合格(fold0 PQ 0.4328)で、生YOLOも悪化しており
batch4→2に対するlr未調整が疑われた。v2では線形スケーリング則で `lr0 7e-4→3.5e-4`
に補正し、モデル(`yolo11m-seg → yolo11l-seg`)・batch(2)・epoch(70)は
V39と同一のまま再測定する。

**採用判定はしない。** fold0 OOF PQ が V35 fold0 実測 **0.4393** を上回ったら
fold1/2 に進む。

Accelerator: **GPU T4 x2** を選択すること(P100は現行Kaggle PyTorchイメージで非対応)。
"""
    ),
    code(
        """# [1/6] 環境確認・依存関係
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3,<9", "pycocotools"])

import json, math, shutil, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from ultralytics import YOLO

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("gpu:", gpu_name)
if not torch.cuda.is_available():
    raise RuntimeError("GPUランタイムを選択してください(Settings > Accelerator > GPU T4 x2)")
if "P100" in gpu_name:
    raise RuntimeError(
        f"GPU {gpu_name} は現行Kaggle PyTorchイメージで問題が出る。"
        "Accelerator を GPU T4 x2 に変更して再実行してください。"
    )

def find_artifacts_root():
    matches = [p for p in Path("/kaggle/input").rglob("grouped_folds.csv")]
    if not matches:
        raise FileNotFoundError(
            "grouped_folds.csv が見つからない(Kaggleのinputマウントレイアウトが"
            "/kaggle/input/<slug>/ か /kaggle/input/datasets/<owner>/<slug>/ かは"
            "バージョンで変わるためglobで探す)。データセット "
            "shingo257/solar-filament-v35-artifacts をAdd Inputしたか確認"
        )
    return matches[0].parent

class CFG:
    SEED = 3407
    ARTIFACTS_ROOT = find_artifacts_root()
    PREPROC_DIR = ARTIFACTS_ROOT / "preprocessed_v30_q96"
    FOLDS_CSV = ARTIFACTS_ROOT / "grouped_folds.csv"
    REFINER_CKPT = ARTIFACTS_ROOT / "refiners" / "mediseg_refiner_fold0_resume.pt"
    WORK = Path("/kaggle/working/v41_modelsize_v2")
    YOLO_ROOT = WORK / "yolo_folds"
    RUNS_ROOT = WORK / "runs"
    CACHE_ROOT = WORK / "oof_cache"
    YOLO_MODEL = "yolo11l-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 2
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    FOLD = 0
    CONF = 0.26
    MIN_AREA = 20
    REFINER_THRESHOLD = 0.45
    CROP_SIZE = 320
    BASELINE_FOLD0_PQ = 0.4393

for required in (CFG.ARTIFACTS_ROOT, CFG.PREPROC_DIR, CFG.FOLDS_CSV, CFG.REFINER_CKPT):
    if not required.exists():
        raise FileNotFoundError(
            f"{required} が見つからない。データセット "
            "shingo257/solar-filament-v35-artifacts をAdd Inputしたか確認"
        )

CFG.WORK.mkdir(parents=True, exist_ok=True)
torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
print({"images": len(list(CFG.PREPROC_DIR.glob("*.jpeg"))), "gpu": gpu_name})
"""
    ),
    code(
        """# [2/6] competition dataset から COCO annotation・fold・前処理画像を揃える
def find_annotation_json():
    input_root = Path("/kaggle/input")
    matches = list(input_root.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    matches = [m for m in matches if "solar-filament-v35-artifacts" not in m.parts]
    if not matches:
        listed = sorted(p.as_posix() for p in input_root.rglob("*") if p.is_file())[:60]
        raise FileNotFoundError(
            "コンペのannotation JSONが見つからない。competition "
            "filament-segmentation-2026 をAdd Inputしたか確認。"
            f"sample_files={listed}"
        )
    return matches[0]

TRAIN_JSON = find_annotation_json()
print("TRAIN_JSON:", TRAIN_JSON)
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

images = {item["id"]: item for item in coco["images"]}
coco_anns = defaultdict(list)
for ann in coco["annotations"]:
    coco_anns[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

def annotation_rank(image_id):
    anns = coco_anns[image_id]
    return len(anns), sum(float(a.get("area", 0)) for a in anns)

file_to_reference_id = {name: max(ids, key=annotation_rank) for name, ids in file_to_ids.items()}
train_files = sorted(name for name in file_to_ids if (CFG.PREPROC_DIR / name).exists())

fold_frame = pd.read_csv(CFG.FOLDS_CSV)
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
train_files = [name for name in train_files if name in fold_of_file]

print({"観測": len(train_files),
       "fold0": sum(fold_of_file[n] == 0 for n in train_files),
       "annotation": len(coco["annotations"])})
"""
    ),
    code(
        """# [3/6] fold0 の YOLO dataset(V30/V35と同じ all-labeler 規則)を作って学習
def polygon_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    polygon = max(polygons, key=len)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.reshape(-1))

def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)

def write_variant(fold_root, split, file_name, image_id, suffix):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{stem}.jpeg"
    label_out = fold_root / "labels" / split / f"{stem}.txt"
    link_or_copy((CFG.PREPROC_DIR / file_name).resolve(), image_out)
    lines = []
    for ann in coco_anns[image_id]:
        if ann.get("iscrowd", 0):
            continue
        line = polygon_line(ann, info["width"], info["height"])
        if line:
            lines.append(line)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(lines), encoding="utf-8")

def build_fold_dataset(fold):
    root = CFG.YOLO_ROOT / f"fold_{fold}"
    if root.exists():
        shutil.rmtree(root)
    for file_name in train_files:
        if fold_of_file[file_name] == fold:
            write_variant(root, "val", file_name, file_to_reference_id[file_name], "")
        else:
            for image_id in file_to_ids[file_name]:
                write_variant(root, "train", file_name, image_id, f"__labeler_{image_id}")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
        encoding="utf-8")
    return yaml_path

yaml_path = build_fold_dataset(CFG.FOLD)
print(yaml_path)

def train_yolo_fold(fold, yaml_path):
    run_dir = CFG.RUNS_ROOT / f"yolo_modelsize_v2_fold_{fold}"
    best_path = run_dir / "weights" / "best.pt"
    batch = CFG.YOLO_BATCH
    last_error = None
    while batch >= 1:
        try:
            model = YOLO(CFG.YOLO_MODEL)
            model.train(
                data=str(yaml_path),
                epochs=CFG.YOLO_EPOCHS,
                imgsz=CFG.YOLO_IMGSZ,
                batch=batch,
                device=0,
                workers=4,
                patience=CFG.YOLO_PATIENCE,
                seed=CFG.SEED + fold,
                deterministic=True,
                single_cls=True,
                mask_ratio=CFG.YOLO_MASK_RATIO,
                overlap_mask=False,
                amp=True,
                cache="disk",
                optimizer="AdamW",
                lr0=3.5e-4,             # V39はbatch4→2なのにlr0据え置きで不安定化の疑い。線形則で半減
                lrf=0.05,
                weight_decay=5e-4,
                warmup_epochs=3,
                degrees=180,
                translate=0.04,
                scale=0.25,
                fliplr=0.5,
                flipud=0.5,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.18,
                mosaic=0.20,
                close_mosaic=10,
                project=str(CFG.RUNS_ROOT),
                name=f"yolo_modelsize_v2_fold_{fold}",
                exist_ok=True,
                plots=True,
                save=True,
                save_period=10,
                verbose=True,
            )
            last_error = None
            break
        except torch.cuda.OutOfMemoryError as error:
            last_error = error
            torch.cuda.empty_cache()
            batch -= 1
            print(f"OOM。batch を {batch} に下げて再試行")
    if last_error is not None:
        raise last_error
    if not best_path.exists():
        raise FileNotFoundError(best_path)
    return best_path

started = time.perf_counter()
best_path = train_yolo_fold(CFG.FOLD, yaml_path)
train_seconds = time.perf_counter() - started
print({"best": str(best_path), "size": best_path.stat().st_size, "train_seconds": round(train_seconds, 1)})
"""
    ),
    code(
        """# [4/6] fold0 OOF を npz キャッシュへ(V34/P2-2と同じ packed 形式)
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

def save_yolo_cache(path, masks, confidences):
    path.parent.mkdir(parents=True, exist_ok=True)
    if masks:
        packed = np.packbits(np.stack(masks).astype(np.uint8), axis=-1)
        shape = np.asarray(masks[0].shape, np.int32)
    else:
        packed = np.empty((0, 0, 0), np.uint8)
        shape = np.asarray([2048, 2048], np.int32)
    np.savez_compressed(path, packed=packed, shape=shape, confidences=confidences)

model = YOLO(str(best_path))
oof_files = [name for name in train_files if fold_of_file[name] == CFG.FOLD]
cache_dir = CFG.CACHE_ROOT / f"yolo_modelsize_v2_fold_{CFG.FOLD}"
for name in oof_files:
    path = cache_dir / f"{Path(name).stem}.npz"
    if path.exists():
        continue
    masks, confidences = result_masks(model.predict(
        str(CFG.PREPROC_DIR / name), imgsz=CFG.YOLO_IMGSZ, conf=0.08,
        iou=0.55, agnostic_nms=True, retina_masks=True, verbose=False,
    )[0])
    save_yolo_cache(path, masks, confidences)
print({"oof": len(oof_files), "cache": str(cache_dir), "npz": len(list(cache_dir.glob('*.npz')))})
"""
    ),
    code("# [5/6] MediSegUNet(src/mediseg_models.py と同一)\n" + MEDISEG),
    code(
        "# [5.5/6] 公式 PQ(src/metrics_official.py と同一)\n" + METRICS +
        """

# [6/6] fold0 OOF を V35 確定後処理で採点し、0.4393 と比較
def load_yolo_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))

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
    return refined & proposal

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

def drop_contained_indices(masks, threshold=0.90):
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

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

refiner = MediSegUNet(3, 1, False).cuda().eval()
try:
    ckpt = torch.load(CFG.REFINER_CKPT, map_location="cuda", weights_only=False)
except TypeError:
    ckpt = torch.load(CFG.REFINER_CKPT, map_location="cuda")
refiner.load_state_dict(ckpt.get("best_model") or ckpt["model"], strict=True)

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in coco_anns[image_id]
                         if not ann.get("iscrowd", 0)]

meter = PanopticQuality()
near_miss = 0
counts = []
cache_dir = CFG.CACHE_ROOT / f"yolo_modelsize_v2_fold_{CFG.FOLD}"
for name in oof_files:
    masks, confidences = load_yolo_cache(cache_dir / f"{Path(name).stem}.npz")
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
    proposals = [masks[i] for i in keep]
    scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
    image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
    refined = [refine_proposal(image, m) for m in proposals]
    predictions = apply_ownership(refined, scores)
    counts.append(len(predictions))
    pred_rles = []
    if predictions:
        encoded = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
        pred_rles = encoded if isinstance(encoded, list) else [encoded]
    for image_id in file_to_ids[name]:
        truth = gt_rles[image_id]
        matrix = iou_matrix(truth, pred_rles)
        meter.add_matrix(matrix, len(truth), len(pred_rles))
        if matrix.size:
            near_miss += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

summary = meter.summary()
delta = summary["pq"] - CFG.BASELINE_FOLD0_PQ
continue_folds = summary["pq"] > CFG.BASELINE_FOLD0_PQ
report = {
    "experiment": "v41_modelsize_v2",
    "yolo_model": CFG.YOLO_MODEL,
    "batch": CFG.YOLO_BATCH,
    "fold": CFG.FOLD,
    "gpu": gpu_name,
    "train_seconds": round(train_seconds, 1),
    "observations": len(oof_files),
    "pq": round(summary["pq"], 6),
    "sq": round(summary["sq"], 6),
    "rq": round(summary["rq"], 6),
    "near_miss": int(near_miss),
    "inst/img": float(np.mean(counts)) if counts else 0.0,
    "baseline_fold0_pq": CFG.BASELINE_FOLD0_PQ,
    "delta": round(delta, 6),
    "continue_fold12": bool(continue_folds),
    "decision": ("fold1/2 を同じモデルサイズで学習してよい"
                 if continue_folds else
                 "fold0 がベースラインを超えていない。全foldへ進まない"),
}
Path("/kaggle/working/v41_modelsize_v2_fold0.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"v41p-{index:02d}"
    if cell["cell_type"] != "code":
        continue
    stripped = re.sub(r"^[%!].*$", "", cell["source"], flags=re.M)
    ast.parse(stripped)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
generated = json.loads(OUTPUT.read_text(encoding="utf-8"))
text = "\n".join(
    "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    for cell in generated["cells"]
)
assert "class MediSegUNet" in text
assert "class PanopticQuality" in text
assert "yolo11l-seg" in text
assert "solar-filament-v35-artifacts" in text
assert "P100" in text
print(OUTPUT)
