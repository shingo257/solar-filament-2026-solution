"""Kaggleネイティブ・multi-seedアンサンブルプローブのビルダー(fold0のみ)。

## 背景(改訂9からの引き継ぎ)

imgsz(P2-2/P2-3)・augmentation(V38/V40)・モデルサイズ(V39/V41)の検出器強化系が
6本連続で不合格になった。V35確定パイプラインはこのデータ規模(707観測)に対して
かなり局所最適に近いと判断し、検出器のハイパーパラメータをいじる方向は一旦止める。

multi-seedアンサンブルは「検出器のランダムな取りこぼしを複数seedの平均で均す」
という別の軸で、検出器強化系とは独立な仮説。ハイパーパラメータはV35確定と
完全に同一のまま、`seed`だけを変えて2本目のfold0モデルを学習する
(GPU予算節約のため、V35確定のfold0モデル本体は既存のOOFキャッシュを再利用し、
新規学習はこのseed違いの1本だけ)。

## このノートでやること

fold0のYOLOをV35確定と同一設定(yolo11m-seg, imgsz1280, batch4, epoch70,
mosaic0.20)、seedだけ`3407`→`8407`に変えて学習し、OOF npzキャッシュを作る。
ローカルで既存のV35 fold0キャッシュ(seed3407)と合算し、refiner+確定後処理を
通してPQを測る(`scratch/run_multiseed_ensemble_probe.py`、別途ローカル実行)。

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
OUTPUT = ROOT / "solar_filament_v42_kaggle_multiseed.ipynb"


try:
    from .build_v42_kaggle_multiseed_probe_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v42_kaggle_multiseed_probe_cells import cells

for index, cell in enumerate(cells):
    cell["id"] = f"v42p-{index:02d}"
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
assert "SEED = 8407" in text
assert 'YOLO_MODEL = "yolo11m-seg.pt"' in text
assert "solar-filament-v35-artifacts" in text
assert "P100" in text
print(OUTPUT)
