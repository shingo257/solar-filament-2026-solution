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


try:
    from .build_v41_kaggle_modelsize_v2_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v41_kaggle_modelsize_v2_cells import cells

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
