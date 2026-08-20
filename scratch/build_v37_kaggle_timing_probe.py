"""Kaggle GPU 実測タイミングプローブのビルダー。

## 目的

このプロジェクトのYOLO学習は今までColab(Drive経由)でしか実測していない。
Kaggleの無料GPU枠(週30時間)がまだ余っているので、そちらでも学習できるように
したい。ただし本番の70 epoch学習をいきなり流す前に、
1. Kaggleネイティブ環境(competition dataset + 専用datasetのみ、Drive不要)で
   このプロジェクトのYOLO学習パイプラインが最初から最後まで通ることを確認する
2. 1epochあたりの実測時間を掴んで、70 epoch本番実験がKaggleの1セッション
   (12時間上限)に収まるか見積もる

の2つだけを目的とした最小プローブ。OOFスコアリングはしない(学習パイプライン
とタイミングだけが目的なので、V35確定後処理まで持ち出す必要はない)。

## 前提

- Kaggle Dataset `shingo257/solar-filament-v35-artifacts` に
  前処理済み画像(`preprocessed_v30_q96/`)と `grouped_folds.csv` を
  アップロード済み(Google Drive由来のものと同一内容)
- コンペ本体のデータセットは `competition_sources` で直接アタッチする
  (`/kaggle/input/competitions/filament-segmentation-2026/...`)
- 現行のKaggle PyTorchイメージはP100で問題が出るため、Accelerator は
  GPU T4 x2 を指定する(参考: `shingo257/e05a-yolo-train-t4` の実績)

## 変えるもの/変えないもの

V35確定学習(imgsz 1280, yolo11m-seg, batch 4)のハイパーパラメータは変えない。
epoch数だけ 70 → 3 に減らし、学習時間だけを計測する。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v37_kaggle_timing_probe.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# V37 Kaggle GPU タイミングプローブ

目的: (1) Kaggleネイティブ(Drive不要)でYOLO学習パイプラインが通るか確認、
(2) 1epochあたりの実測時間を掴む。**採用判定はしない・OOFスコアリングもしない。**

V35確定ハイパーパラメータ(yolo11m-seg, imgsz 1280, batch 4)のまま epoch数だけ 3 に落とす。

Accelerator: **GPU T4 x2** を選択すること(P100は現行Kaggle PyTorchイメージで非対応)。
"""
    ),
    code(
        """# [1/4] 環境確認・依存関係
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3,<9", "pycocotools"])

import json, shutil, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

input_root = Path("/kaggle/input")
print("input_root exists:", input_root.exists())
print("input_root top-level:", sorted(p.name for p in input_root.iterdir()) if input_root.exists() else None)
for top in (input_root.iterdir() if input_root.exists() else []):
    if top.is_dir():
        children = sorted(p.name for p in top.iterdir())[:20]
        print(f"  {top.name}/ -> {children}")

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
    WORK = Path("/kaggle/working/v37_probe")
    YOLO_ROOT = WORK / "yolo_folds"
    RUNS_ROOT = WORK / "runs"
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 3          # タイミング計測用。本番は70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    FOLD = 0

for required in (CFG.ARTIFACTS_ROOT, CFG.PREPROC_DIR, CFG.FOLDS_CSV):
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
        """# [2/4] competition dataset から COCO annotation を見つける
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
missing = len(file_to_ids) - len(train_files)

fold_frame = pd.read_csv(CFG.FOLDS_CSV)
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
train_files = [name for name in train_files if name in fold_of_file]

print({"観測": len(train_files), "前処理画像なしでskip": missing,
       "fold0": sum(fold_of_file[n] == 0 for n in train_files),
       "annotation": len(coco["annotations"])})
"""
    ),
    code(
        """# [3/4] fold0 の YOLO dataset(V30/V35と同じ all-labeler 規則)を作って学習
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
n_train = len(list((CFG.YOLO_ROOT / f"fold_{CFG.FOLD}" / "images" / "train").glob("*.jpeg")))
n_val = len(list((CFG.YOLO_ROOT / f"fold_{CFG.FOLD}" / "images" / "val").glob("*.jpeg")))
print({"yaml": str(yaml_path), "train_images": n_train, "val_images": n_val})

epoch_times = []

class EpochTimer:
    def __init__(self):
        self.t0 = None
    def on_train_epoch_start(self, trainer):
        self.t0 = time.perf_counter()
    def on_train_epoch_end(self, trainer):
        if self.t0 is not None:
            epoch_times.append(time.perf_counter() - self.t0)

timer = EpochTimer()
model = YOLO(CFG.YOLO_MODEL)
model.add_callback("on_train_epoch_start", timer.on_train_epoch_start)
model.add_callback("on_train_epoch_end", timer.on_train_epoch_end)

started = time.perf_counter()
model.train(
    data=str(yaml_path),
    epochs=CFG.YOLO_EPOCHS,
    imgsz=CFG.YOLO_IMGSZ,
    batch=CFG.YOLO_BATCH,
    device=0,
    workers=4,
    patience=CFG.YOLO_PATIENCE,
    seed=CFG.SEED + CFG.FOLD,
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
    warmup_epochs=1,      # 3epochしか回さないので warmup を1へ短縮(タイミング計測のみ、収束は見ない)
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
    name=f"yolo_probe_fold_{CFG.FOLD}",
    exist_ok=True,
    plots=False,
    save=True,
    verbose=True,
)
total_seconds = time.perf_counter() - started
print({"total_seconds": round(total_seconds, 1), "epoch_times": [round(t, 1) for t in epoch_times]})
"""
    ),
    code(
        """# [4/4] レポート保存
report = {
    "gpu": gpu_name,
    "yolo_model": CFG.YOLO_MODEL,
    "imgsz": CFG.YOLO_IMGSZ,
    "batch": CFG.YOLO_BATCH,
    "fold": CFG.FOLD,
    "train_images": n_train,
    "val_images": n_val,
    "probe_epochs": CFG.YOLO_EPOCHS,
    "epoch_seconds": [round(t, 1) for t in epoch_times],
    "mean_epoch_seconds_excl_first": (
        round(float(np.mean(epoch_times[1:])), 1) if len(epoch_times) > 1 else None
    ),
    "total_seconds": round(total_seconds, 1),
    "estimated_70epoch_hours": (
        round(float(np.mean(epoch_times[1:])) * 70 / 3600, 2) if len(epoch_times) > 1 else None
    ),
}
Path("/kaggle/working/v37_timing_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"v37p-{index:02d}"
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
assert "YOLO_EPOCHS = 3" in text
assert "solar-filament-v35-artifacts" in text
assert "P100" in text
print(OUTPUT)
