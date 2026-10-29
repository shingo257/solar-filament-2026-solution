"""Kaggleネイティブ・モデルサイズ拡大プローブのビルダー(fold0のみ)。

## 背景

near-miss被覆モード診断(改訂7、§27.1)で、undercoverage(45%)とovercoverage(42%)
がほぼ拮抗し、hard FN(完全な取りこぼし)がhard FPを37%上回った。loss_unitsは
GT個体数の多い密集シーンに強く紐づく。imgsz変更(P2-2/P2-3)は不合格で解像度
レバーは停止済みだが、**imgszとモデル容量は別の軸**であり、
「1280px入力のままモデル容量だけ上げる」はまだ検証していない。

このプローブは `yolo11m-seg` → `yolo11l-seg` へモデルサイズだけを上げる。
imgsz(1280)・epoch(70)・augmentation(V35確定値のまま)は変えない。
VRAM消費が増えるため batch は 4 → 2 に下げ、OOM時は1へ自動フォールバックする。

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
OUTPUT = ROOT / "solar_filament_v39_kaggle_modelsize_probe.ipynb"


try:
    from .build_v39_kaggle_modelsize_probe_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v39_kaggle_modelsize_probe_cells import cells

for index, cell in enumerate(cells):
    cell["id"] = f"v39p-{index:02d}"
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
