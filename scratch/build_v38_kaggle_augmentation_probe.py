"""Kaggleネイティブ・augmentation強化プローブのビルダー(fold0のみ)。

## 背景

2026-08-16夜のnear-miss被覆モード診断(§27.1、全707観測)で、
undercoverage(45%)とovercoverage(42%)がほぼ拮抗し、単一モードが支配的でない
ことが分かった。一方でIoU≤0.3の完全な取りこぼし(hard FN)がhard FPを37%上回り、
loss_unitsはGT個体数の多い密集シーン(worst15はいずれもgt_count_mean 10〜21)に
強く紐づく。imgsz変更(P2-2/P2-3)は既に不合格で解像度レバーは停止済み。

このプローブは「密集シーンでの検出漏れ」に直接効きうる augmentation 強化を試す。
V35確定ハイパーパラメータ(yolo11m-seg, imgsz 1280, batch 4, epoch 70)は変えず、
augmentationだけを強化する:

- mosaic: 0.20 → 0.75(密な合成シーンをより頻繁に学習させる)
- copy_paste: 0.0 → 0.3(YOLOセグメンテーションのインスタンス貼り付け拡張。
  重なり合う個体を人工的に増やす、near-miss/密集シーン対策として最も直接的)

他のaugmentationパラメータ(degrees/translate/scale/flip/hsv)はV35と同一。

## なぜfold0だけか

3-fold全部を70 epochで回すのは高い。「単一foldのbestを一般化しない」という
このプロジェクトの方針どおり、**採用判定はしない**。fold0 OOF PQがV35の
fold0実測 0.4393 を上回ったときだけ fold1/2 に進む(P2-2/P2-3と同じゲート)。

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
OUTPUT = ROOT / "solar_filament_v38_kaggle_augmentation_probe.ipynb"


try:
    from .build_v38_kaggle_augmentation_probe_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v38_kaggle_augmentation_probe_cells import cells

for index, cell in enumerate(cells):
    cell["id"] = f"v38p-{index:02d}"
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
assert "MOSAIC = 0.75" in text
assert "COPY_PASTE = 0.3" in text
assert "solar-filament-v35-artifacts" in text
assert "P100" in text
print(OUTPUT)
