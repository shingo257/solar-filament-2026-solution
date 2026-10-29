"""Embedded cell payloads for the build_v49_colab_pseudo_label_fold0_retrain notebook builder.

Verbatim extraction of the cell-source literals from
build_v49_colab_pseudo_label_fold0_retrain.py so the builder stays readable.
No logic; loaded by the builder via sibling import.
"""


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


MEDISEG = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")


METRICS = (ROOT / "solar_filament" / "metrics" / "metrics_official.py").read_text(encoding="utf-8")


METRICS = METRICS.replace("from __future__ import annotations\n", "", 1)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V49 探針 — pseudo-label混合でfold0 YOLOを再学習

707観測の年代分布偏り(2011-17=87% vs 2018-22=13%)を是正する仮説を検証する。
GONG追加画像150枚から生成したpseudo-label(35画像・72個体、confidence≥0.5+
2/3fold合意)を**fold0のtrain splitにのみ追加**し、V35と同一設定で再学習する。

| | V30 / V35 | この探針 |
|---|---|---|
| 学習fold | 0, 1, 2 | **0 のみ** |
| train画像 | 実データのみ | 実データ + pseudo-label 35枚 |
| val(評価)画像 | fold0実観測 | **fold0実観測のみ(pseudo-labelなし、リークなし)** |
| imgsz/epoch/augment | 1280/70/既定 | 同じ |
| 後処理 | shrink@0.45 + drop_contained | 同じ |
| 比較対象 | fold0 OOF PQ **0.4393** | これを上回れば fold1/2 へ |

**採用判定はしない。** fold0が良くても全fold OOFを測るまでtest推論しない。

GPU必須。A100推奨。
"""
    ),
    code(
        """# [1/8] 依存関係・Drive・前提artifactの確認
%pip install -q "ultralytics>=8.3,<9" pycocotools

import json, math, os, random, shutil, time
from collections import defaultdict
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
    V30_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    V35_ROOT = Path("/content/drive/MyDrive/solar_filament_v35")
    DATA_CACHE = V30_ROOT / "competition_data"
    ARTIFACT_ROOT = V30_ROOT / "artifacts"
    PREPROC_CACHE = V30_ROOT / "preprocessed_v30_q96"
    PSEUDO_IMAGE_DIR = V35_ROOT / "pseudo_label" / "preprocessed"
    PSEUDO_CACHE_DIR = V35_ROOT / "pseudo_label" / "cache"
    PSEUDO_MANIFEST = V35_ROOT / "pseudo_label" / "pseudo_label_manifest.json"
    LOCAL_ROOT = Path("/content/solar_v49")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    YOLO_ROOT = LOCAL_ROOT / "yolo_folds"
    RUNS_ROOT = V35_ROOT / "runs"
    CACHE_ROOT = V35_ROOT / "prediction_cache" / "oof"
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    FOLD = 0
    CONF = 0.26
    MIN_AREA = 20
    REFINER_THRESHOLD = 0.45   # 確定構成(V35と同一、free variantではない)
    DROP_THRESHOLD = 0.90
    CROP_SIZE = 320
    BASELINE_FOLD0_PQ = 0.4393

CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
CFG.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
CFG.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
CFG.CACHE_ROOT.mkdir(parents=True, exist_ok=True)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理(実データ)": CFG.PREPROC_CACHE,
    "fold0 refiner": CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt",
    "pseudo-label画像": CFG.PSEUDO_IMAGE_DIR,
    "pseudo-labelキャッシュ": CFG.PSEUDO_CACHE_DIR,
    "pseudo-label manifest": CFG.PSEUDO_MANIFEST,
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0), "runs": str(CFG.RUNS_ROOT)})"""
    ),
    code(
        """# [2/8] COCO・fold・前処理画像(実データ)をローカルSSDへ
TRAIN_JSON = next(CFG.DATA_CACHE.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

def annotation_rank(image_id):
    anns = annotations[image_id]
    return len(anns), sum(float(a.get("area", 0)) for a in anns)

file_to_reference_id = {name: max(ids, key=annotation_rank) for name, ids in file_to_ids.items()}
train_files = sorted(file_to_ids)
fold_frame = pd.read_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
assert all(name in fold_of_file for name in train_files)

for name in tqdm(train_files, desc="実データ前処理画像をSSDへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)

print({"実観測": len(train_files), "fold0": sum(v == 0 for v in fold_of_file.values()),
       "annotation": len(coco["annotations"])})"""
    ),
    code(
        """# [3/8] pseudo-labelをYOLO polygon形式へ変換(train splitにのみ追加)
def load_pseudo_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))

def mask_polygon_lines(mask, width, height):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        points = approx.astype(np.float32)
        points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
        points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in points.reshape(-1)))
    return lines

pseudo_manifest = json.loads(CFG.PSEUDO_MANIFEST.read_text(encoding="utf-8"))
pseudo_files = [m["file_name"] for m in pseudo_manifest["images"] if m["n_kept"] > 0]
print(f"pseudo-label採用画像: {len(pseudo_files)}枚, 合計{pseudo_manifest['total_kept']}個体")

pseudo_lines_by_file = {}
for name in pseudo_files:
    masks, confidences = load_pseudo_cache(CFG.PSEUDO_CACHE_DIR / f"{Path(name).stem}.npz")
    lines = []
    for mask in masks:
        lines.extend(mask_polygon_lines(mask, 2048, 2048))
    if lines:
        pseudo_lines_by_file[name] = lines

print(f"YOLO形式に変換できた画像: {len(pseudo_lines_by_file)}枚, "
      f"合計{sum(len(v) for v in pseudo_lines_by_file.values())}行")"""
    ),
    code(
        """# [4/8] fold0のYOLO dataset構築(実データ + pseudo-label、val=fold0実観測のみ)
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

def write_variant(fold_root, split, file_name, image_id, suffix, source_dir):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{stem}.jpeg"
    label_out = fold_root / "labels" / split / f"{stem}.txt"
    link_or_copy((source_dir / file_name).resolve(), image_out)
    lines = []
    for ann in annotations[image_id]:
        if ann.get("iscrowd", 0):
            continue
        line = polygon_line(ann, info["width"], info["height"])
        if line:
            lines.append(line)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(lines), encoding="utf-8")

def write_pseudo_variant(fold_root, file_name, lines):
    stem = Path(file_name).stem + "__pseudo"
    image_out = fold_root / "images" / "train" / f"{stem}.jpeg"
    label_out = fold_root / "labels" / "train" / f"{stem}.txt"
    link_or_copy((CFG.PSEUDO_IMAGE_DIR / file_name).resolve(), image_out)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(lines), encoding="utf-8")

def build_fold0_dataset_with_pseudo():
    root = CFG.YOLO_ROOT / "fold_0_pseudo"
    if root.exists():
        shutil.rmtree(root)
    for file_name in tqdm(train_files, desc="build fold0(実データ)"):
        if fold_of_file[file_name] == CFG.FOLD:
            write_variant(root, "val", file_name, file_to_reference_id[file_name], "", CFG.PREPROC_DIR)
        else:
            for image_id in file_to_ids[file_name]:
                write_variant(root, "train", file_name, image_id, f"__labeler_{image_id}", CFG.PREPROC_DIR)
    for file_name, lines in tqdm(pseudo_lines_by_file.items(), desc="build fold0(pseudo-label追加)"):
        write_pseudo_variant(root, file_name, lines)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
        encoding="utf-8")
    n_train = len(list((root / "images" / "train").glob("*.jpeg")))
    n_val = len(list((root / "images" / "val").glob("*.jpeg")))
    print({"train_images": n_train, "val_images": n_val, "yaml": str(yaml_path)})
    return yaml_path

yaml_path = build_fold0_dataset_with_pseudo()"""
    ),
    code(
        """# [5/8] fold0をpseudo-label混合で学習(V30のruns/fold0は上書きしない)
def train_yolo_fold0_pseudo(yaml_path):
    run_dir = CFG.RUNS_ROOT / "yolo_pseudolabel_fold_0"
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    done_path = run_dir / "TRAINING_COMPLETE"
    if done_path.exists() and best_path.exists():
        print(f"完了済み: {best_path}")
        return best_path
    if last_path.exists():
        print(f"resume: {last_path}")
        YOLO(str(last_path)).train(resume=True)
        if not best_path.exists():
            raise FileNotFoundError(best_path)
        done_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        return best_path

    model = YOLO(CFG.YOLO_MODEL)
    model.train(
        data=str(yaml_path),
        epochs=CFG.YOLO_EPOCHS,
        imgsz=CFG.YOLO_IMGSZ,
        batch=CFG.YOLO_BATCH,
        device=0,
        workers=4,
        patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED,
        deterministic=True,
        single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO,
        overlap_mask=False,
        amp=True,
        cache="disk",
        optimizer="AdamW",
        lr0=7e-4,
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
        name="yolo_pseudolabel_fold_0",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=5,
        verbose=True,
    )
    if not best_path.exists():
        raise FileNotFoundError(f"学習が{best_path}を作らずに終了")
    done_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return best_path

best_path = train_yolo_fold0_pseudo(yaml_path)
print({"best": str(best_path), "size": best_path.stat().st_size})"""
    ),
    code(
        """# [6/8] fold0実観測(val split=リークなし)へOOF推論
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
cache_dir = CFG.CACHE_ROOT / "yolo_pseudolabel_fold_0"
for name in tqdm(oof_files, desc="OOF推論(実観測のみ)"):
    path = cache_dir / f"{Path(name).stem}.npz"
    if path.exists():
        continue
    masks, confidences = result_masks(model.predict(
        str(CFG.PREPROC_DIR / name), imgsz=CFG.YOLO_IMGSZ, conf=0.08,
        iou=0.55, agnostic_nms=True, retina_masks=True, verbose=False,
    )[0])
    save_yolo_cache(path, masks, confidences)
print({"oof": len(oof_files), "cache": str(cache_dir), "npz": len(list(cache_dir.glob('*.npz')))})"""
    ),
    code("# [7/8] MediSegUNet・公式PQ(solar_filament/と同一)\n" + MEDISEG + "\n\n" + METRICS),
    code(
        """# [8/8] fold0 OOF(実観測のみ)をV35確定後処理で採点し、0.4393と比較
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
    return refined & proposal  # shrink(確定構成)

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

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

refiner = MediSegUNet(3, 1, False).cuda().eval()
ckpt = torch.load(CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt", map_location="cuda", weights_only=False)
refiner.load_state_dict(ckpt.get("best_model") or ckpt["model"], strict=True)

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in annotations[image_id] if not ann.get("iscrowd", 0)]

meter = PanopticQuality()
near_miss = 0
counts = []
cache_dir = CFG.CACHE_ROOT / "yolo_pseudolabel_fold_0"
for name in tqdm(oof_files, desc="PQ採点(実観測のみ)"):
    masks, confidences = load_yolo_cache(cache_dir / f"{Path(name).stem}.npz")
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences)) if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
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
    "scope": "fold0のみ、実観測OOFで採点(pseudo-labelは学習train splitのみ、評価に混入なし)",
    "fold": CFG.FOLD,
    "observations": len(oof_files),
    "pseudo_label_images_used_in_train": len(pseudo_lines_by_file),
    "pseudo_label_instances_used_in_train": sum(len(v) for v in pseudo_lines_by_file.values()),
    "pq": round(summary["pq"], 6),
    "sq": round(summary["sq"], 6),
    "rq": round(summary["rq"], 6),
    "near_miss": int(near_miss),
    "inst/img": float(np.mean(counts)) if counts else 0.0,
    "baseline_fold0_pq": CFG.BASELINE_FOLD0_PQ,
    "delta": round(delta, 6),
    "continue_fold12": bool(continue_folds),
    "decision": ("fold1/2でも同様にpseudo-labelを混ぜて学習してよい"
                 if continue_folds else
                 "fold0がベースラインを超えていない。全foldへ進まない"),
}
(CFG.V35_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
(CFG.V35_ROOT / "artifacts" / "v49_pseudolabel_fold0.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))"""
    ),
]
