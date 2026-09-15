"""Kaggleネイティブ・RF-DETR-Seg fold0検証ビルダー(高解像度版、V44の続き)。

## 背景(V44からの引き継ぎ)

V44(RFDETRSegMedium、既定解像度432px)はraw PQ 0.2069で生YOLO fold0基準
0.3977を大幅に下回り(Δ-0.190847)不合格だった。ただし432pxはYOLOの確定値
1280pxよりはるかに低く、2048px画像上の細長いfilament構造がほぼ潰れている
可能性が高い重大な交絡がある。「アーキテクチャが劣る」と断定するのは
時期尚早と判断し、解像度だけを変えて再検証する。

## このノートでV44と変えた点

`RFDETRSegMedium(resolution=768)` — 768はRF-DETR-Seg系で公式サポートされる
最大解像度(2XLargeの既定値、`patch_size×num_windows`制約により24の倍数に
限定)。モデルクラスはMediumのまま(パラメータ規模を揃え、変える変数を
解像度だけに絞る)。位置埋め込みサイズが変わるためV44のcheckpointからは
resumeできない(ゼロから再学習)。解像度3.16倍(768²/432²)でメモリ消費が
増えるため、batch_size 4→2、grad_accum 4→8(実効バッチ16のまま)、
gradient_checkpointing=Trueを追加。

## このノートでやること(V44と共通)

fold0のみ、COCO事前学習チェックポイントからfilament(単一クラス)へ
fine-tuningする。train=fold1+fold2の全ラベラーvariant、val=fold0の
参照variant(V35の学習データ規則と同一)。学習後、fold0の236観測に対し
raw推論(refiner・排他化なし)を行い、V38/V39で使った「生YOLO fold0基準
PQ 0.3977」と同じ条件で比較する。

## なぜfold0だけ・raw比較か

「単一foldのbestを一般化しない」という方針を踏襲。またrefiner/排他化は
YOLOのraw出力分布に合わせてチューニングされたものなので、まだ未調整の
RF-DETR出力に適用しても正しい比較にならない。まずアーキテクチャ単体の
検出品質(raw PQ)がYOLOのraw基準を超えるかだけを見る。超えた場合のみ、
refiner再学習・fold1/2展開を検討する。

## Kaggleネイティブ(Drive不要)

前処理済み画像・fold分割は Kaggle Dataset
`shingo257/solar-filament-v35-artifacts` から読む。COCO annotationは
competition `filament-segmentation-2026` を直接アタッチする。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v45_kaggle_rfdetr_hires_probe.ipynb"
METRICS = (ROOT / "solar_filament" / "metrics" / "metrics_official.py").read_text(encoding="utf-8")
METRICS = METRICS.replace("from __future__ import annotations\n", "", 1)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V45 Kaggle探針 — RF-DETR-Seg fold0検証(高解像度768px、V44の続き)

V44(RFDETRSegMedium、既定解像度432px)はraw PQ **0.2069**で生YOLO基準
0.3977を大幅に下回り(Δ-0.190847)不合格だった。ただし432pxはYOLO確定値
1280pxよりはるかに低く、細長いfilament構造が潰れている可能性が高い重大な
交絡がある。本ノートは**解像度だけ**を768px(RF-DETR-Seg系の公式最大値)へ
変更し、それ以外(モデルクラス・データセット・評価条件)はV44と揃えたまま
再検証する。位置埋め込みサイズが変わるためV44のcheckpointからはresumeせず
ゼロから学習する。

**採用判定はしない。** raw推論(refiner・排他化なし)のfold0 PQが
V38/V39で使った生YOLO fold0基準 **0.3977** を上回ったら、refiner再学習・
fold1/2展開を検討する。ここでも下回った場合は、解像度交絡ではなく
アーキテクチャそのものの不適合と判断し、RF-DETR方向を打ち切る。

Accelerator: **GPU T4 x2** を選択すること。768px学習は432px比で計算量が
約3.16倍のため、V44より大幅に時間がかかる見込み。
"""
    ),
    code(
        """# [1/6] 環境確認・依存関係
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "rfdetr[train]", "lightning", "faster-coco-eval>=1.7.2"])

import json, shutil, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

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
            "grouped_folds.csv が見つからない。データセット "
            "shingo257/solar-filament-v35-artifacts をAdd Inputしたか確認"
        )
    return matches[0].parent

class CFG:
    ARTIFACTS_ROOT = find_artifacts_root()
    PREPROC_DIR = ARTIFACTS_ROOT / "preprocessed_v30_q96"
    FOLDS_CSV = ARTIFACTS_ROOT / "grouped_folds.csv"
    WORK = Path("/kaggle/working/v45_rfdetr_hires")
    DATASET_ROOT = WORK / "fold0_dataset"
    TRAIN_OUT = WORK / "train_out"
    CACHE_ROOT = WORK / "oof_cache"
    FOLD = 0
    CATEGORY_ID = 1
    RAW_YOLO_FOLD0_PQ = 0.3977   # V38/V39で使ったrefinerなしYOLO fold0基準
    RESOLUTION = 768              # V44の432pxから引き上げ(RF-DETR-Seg系の公式最大値)
    EPOCHS = 10                   # Kaggle GPUセッション上限(約9時間)内に収める。1epoch~50分ペース想定
                                   # (必要ならcheckpointをKaggle Datasetへ再アップロードしてresume継続)
    BATCH_SIZE = 2
    GRAD_ACCUM = 8                # 実効バッチはV44と同じ16を維持
    LR = 5e-5   # 小規模データセット(<1000枚)向け

for required in (CFG.ARTIFACTS_ROOT, CFG.PREPROC_DIR, CFG.FOLDS_CSV):
    if not required.exists():
        raise FileNotFoundError(
            f"{required} が見つからない。データセット "
            "shingo257/solar-filament-v35-artifacts をAdd Inputしたか確認"
        )

CFG.WORK.mkdir(parents=True, exist_ok=True)
print({"images": len(list(CFG.PREPROC_DIR.glob("*.jpeg"))), "gpu": gpu_name})
"""
    ),
    code(
        """# [2/6] competition dataset から COCO annotation を揃える
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
anns_by_image = defaultdict(list)
for ann in coco["annotations"]:
    anns_by_image[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

def annotation_rank(anns):
    return len(anns), sum(float(a.get("area", 0)) for a in anns)

file_to_reference_id = {
    name: max(ids, key=lambda i: annotation_rank(anns_by_image[i]))
    for name, ids in file_to_ids.items()
}

fold_frame = pd.read_csv(CFG.FOLDS_CSV)
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
target_files = sorted(
    name for name in file_to_ids
    if name in fold_of_file and (CFG.PREPROC_DIR / name).exists()
)
print({"観測": len(target_files),
       "fold0": sum(fold_of_file[n] == 0 for n in target_files),
       "annotation": len(coco["annotations"])})
"""
    ),
    code(
        """# [3/6] fold0 holdout の RF-DETR COCO dataset を作る(YOLO学習と同じall-labeler規則)
def polygon_ok(ann):
    segs = [p for p in ann.get("segmentation", []) if isinstance(p, list) and len(p) >= 6]
    return bool(segs) and not ann.get("iscrowd", 0)

def make_annotation(ann, new_id, new_image_id):
    return {
        "id": new_id, "image_id": new_image_id, "category_id": CFG.CATEGORY_ID,
        "bbox": ann["bbox"], "area": ann["area"], "iscrowd": 0,
        "segmentation": [p for p in ann["segmentation"] if isinstance(p, list) and len(p) >= 6],
    }

def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)

def build_split(split, entries):
    split_dir = CFG.DATASET_ROOT / split
    out_images, out_annotations = [], []
    next_ann_id = 1
    for next_image_id, (file_name, image_id, out_stem) in enumerate(entries, start=1):
        info = images[image_id]
        out_name = f"{out_stem}.jpeg"
        link_or_copy((CFG.PREPROC_DIR / file_name).resolve(), split_dir / out_name)
        out_images.append({"id": next_image_id, "file_name": out_name,
                            "width": info["width"], "height": info["height"]})
        for ann in anns_by_image[image_id]:
            if not polygon_ok(ann):
                continue
            out_annotations.append(make_annotation(ann, next_ann_id, next_image_id))
            next_ann_id += 1
    payload = {
        "info": {"description": f"solar-filament fold{CFG.FOLD} {split}", "version": "1.0"},
        "licenses": [],
        "images": out_images,
        "categories": [{"id": CFG.CATEGORY_ID, "name": "filament", "supercategory": "filament"}],
        "annotations": out_annotations,
    }
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"{split}: images={len(out_images)} annotations={len(out_annotations)}")

if CFG.DATASET_ROOT.exists():
    shutil.rmtree(CFG.DATASET_ROOT)

train_entries, valid_entries = [], []
for file_name in target_files:
    stem = Path(file_name).stem
    if fold_of_file[file_name] == CFG.FOLD:
        valid_entries.append((file_name, file_to_reference_id[file_name], stem))
    else:
        for image_id in file_to_ids[file_name]:
            train_entries.append((file_name, image_id, f"{stem}__labeler_{image_id}"))

build_split("train", train_entries)
build_split("valid", valid_entries)
print(CFG.DATASET_ROOT)
"""
    ),
    code(
        """# [4/6] RFDETRSegMedium(解像度768px)を filament(単一クラス)へ fine-tuning
from rfdetr import RFDETRSegMedium

model = RFDETRSegMedium(resolution=CFG.RESOLUTION, gradient_checkpointing=True)

model.train(
    dataset_dir=str(CFG.DATASET_ROOT),
    epochs=CFG.EPOCHS,
    batch_size=CFG.BATCH_SIZE,
    grad_accum_steps=CFG.GRAD_ACCUM,
    lr=CFG.LR,
    output_dir=str(CFG.TRAIN_OUT),
    early_stopping=True,
    early_stopping_patience=10,
)

best_ckpt = CFG.TRAIN_OUT / "checkpoint_best_total.pth"
print(best_ckpt, best_ckpt.exists())
"""
    ),
    code(
        """# [5/6] fold0(236観測)へraw推論し、YOLO OOFキャッシュと同じnpz形式で保存
from rfdetr import RFDETRSegMedium

infer_model = RFDETRSegMedium(resolution=CFG.RESOLUTION, pretrain_weights=str(best_ckpt))

def save_cache(path, masks, confidences):
    path.parent.mkdir(parents=True, exist_ok=True)
    if masks:
        packed = np.packbits(np.stack(masks).astype(np.uint8), axis=-1)
        shape = np.asarray(masks[0].shape, np.int32)
    else:
        packed = np.empty((0, 0, 0), np.uint8)
        shape = np.asarray([2048, 2048], np.int32)
    np.savez_compressed(path, packed=packed, shape=shape, confidences=confidences)

fold0_files = [n for n in target_files if fold_of_file[n] == CFG.FOLD]
cache_dir = CFG.CACHE_ROOT / f"rfdetr_fold_{CFG.FOLD}"
started = time.perf_counter()
for position, name in enumerate(fold0_files, 1):
    out_path = cache_dir / f"{Path(name).stem}.npz"
    if out_path.exists():
        continue
    detections = infer_model.predict(str(CFG.PREPROC_DIR / name), threshold=0.05)
    masks = [m.astype(bool) for m in detections.mask] if detections.mask is not None else []
    confidences = (detections.confidence.astype(np.float32)
                   if detections.confidence is not None else np.zeros(0, np.float32))
    save_cache(out_path, masks, confidences)
    if position % 25 == 0 or position == len(fold0_files):
        print(f"  {position}/{len(fold0_files)}  {time.perf_counter() - started:.0f}s", flush=True)

print(cache_dir)
"""
    ),
    code(
        (
            "# [6/6] 公式PQ(src/metrics_official.py と同一)でraw fold0を採点し、生YOLO基準0.3977と比較\n"
            + METRICS
            + """

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

def masks_to_rles(masks):
    if not masks:
        return []
    return mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))

def iou_matrix(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    return np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64).T

def load_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))

CONF = 0.26
MIN_AREA = 20
meter = PanopticQuality()
for name in fold0_files:
    # annotator×image単位で評価する(公式PQ集計と同一。1画像に複数variantがあれば全variant分カウント)
    gt_by_variant = []
    for image_id in file_to_ids[name]:
        info = images[image_id]
        gt_by_variant.append([to_rle(a, info["height"], info["width"])
                               for a in anns_by_image[image_id] if polygon_ok(a)])

    cache_path = cache_dir / f"{Path(name).stem}.npz"
    if not cache_path.exists():
        continue
    masks, confidences = load_cache(cache_path)
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CONF and m.sum() >= MIN_AREA]
    kept_masks = [masks[i] for i in keep]
    pred_rles = masks_to_rles(kept_masks)

    for gt_rles in gt_by_variant:
        matrix = iou_matrix(gt_rles, pred_rles)
        meter.add_matrix(matrix, len(gt_rles), len(pred_rles))

summary = meter.summary()
report = {
    "scope": f"{len(fold0_files)}観測、fold0のみ、RFDETRSegMedium(resolution=768) raw推論(refiner/排他化なし)",
    **summary,
    "resolution": CFG.RESOLUTION,
    "raw_yolo_fold0_pq": CFG.RAW_YOLO_FOLD0_PQ,
    "delta_vs_raw_yolo": round(summary["pq"] - CFG.RAW_YOLO_FOLD0_PQ, 6),
    "beats_raw_yolo": bool(summary["pq"] > CFG.RAW_YOLO_FOLD0_PQ),
    "v44_raw_pq_432px": 0.20685291386467813,
    "delta_vs_v44": round(summary["pq"] - 0.20685291386467813, 6),
}
Path("/kaggle/working/v45_rfdetr_hires_fold0_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
"""
        )
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for cell in cells:
    if cell["cell_type"] == "code":
        ast.parse(cell["source"])
print("syntax OK")

text = json.dumps(notebook, ensure_ascii=False, indent=1)
assert "RFDETRSegMedium" in text
assert "RAW_YOLO_FOLD0_PQ = 0.3977" in text
assert "FOLD = 0" in text
assert "RESOLUTION = 768" in text
assert "resume=" not in text
OUTPUT.write_text(text, encoding="utf-8")
print(OUTPUT)
