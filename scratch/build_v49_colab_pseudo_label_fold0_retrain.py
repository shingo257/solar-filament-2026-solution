"""V49 探針ノートのビルダー: pseudo-label(GONG追加画像)をfold0学習データに
混ぜてYOLOを再学習し、リークなしで評価する。

## 仮説

707観測は2011-2017年が87%、2018-2022年がわずか13%と年代分布が偏っている
(README.md §2.1)。pseudo-labeling(3-fold YOLO WBF+refiner shrink@0.45で
2018-2022年のGONG追加画像150枚から35枚・72個体を生成、confidence≥0.5+
2/3fold合意の厳しいフィルタ)でこの偏りを是正できるか検証する。

## なぜfold0だけか

3-fold全部を70epoch回すのは高い。「単一foldのbestを一般化しない」という
本プロジェクトの一貫した教訓に従い、**採用判定はしない**。fold0 OOFが
V35のfold0基準0.4393を上回ったときだけfold1/2に進む。

## リーク防止

pseudo-label画像はどのfoldの評価にも使われない(全て新規画像、fold0の
val split=fold0の実観測のみ)。学習データ(train split)にのみ追加する。

## 変えるもの / 変えないもの

変える: fold0のtrain splitにpseudo-label 35画像・72個体を追加するだけ。
変えない: モデル(yolo11m-seg)、imgsz(1280)、epoch(70)、augment、
patience、mask_ratio、grouped folds、前処理JPEG q96、後処理(V35確定構成
shrink@0.45 + drop_contained_then_area_asc)。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v49_colab_pseudo_label_fold0_retrain.ipynb"


try:
    from .build_v49_colab_pseudo_label_fold0_retrain_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v49_colab_pseudo_label_fold0_retrain_cells import cells

for cell in cells:
    if cell["cell_type"] == "code":
        stripped = re.sub(r"^[%!].*$", "", cell["source"], flags=re.M)
        ast.parse(stripped)

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

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
generated = json.loads(OUTPUT.read_text(encoding="utf-8"))
text = "\n".join(
    "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    for cell in generated["cells"]
)
assert "class MediSegUNet" in text
assert "class PanopticQuality" in text
assert "yolo_pseudolabel_fold_0" in text
assert "REFINER_THRESHOLD = 0.45" in text  # 確定構成、free variantではない
print(OUTPUT)
