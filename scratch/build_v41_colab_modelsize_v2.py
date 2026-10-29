"""V41 Colab版ビルダー: モデルサイズ拡大(yolo11l-seg)v2をfold0だけで試す。

## 背景(KaggleのV39/V41からの引き継ぎ、改訂8)

near-miss被覆モード診断(改訂7、§27.1)で密集シーンでの検出漏れが損失の主因と
分かった。Kaggle無料GPU枠でV39(yolo11l-seg、batch4→2、lr0=7e-4据え置き)を
試したところfold0 OOF PQ 0.4328で不合格(基準0.4393)。生YOLOでもV30基準を
下回っており、batch半減に対してlrを調整していなかったことによる学習不安定化が
疑われた。v2ではlr0を線形スケーリング則で3.5e-4に補正して再測定する
(Kaggle版は`build_v41_kaggle_modelsize_v2.py`。GPU無料枠の都合でColabへ
持ち出すためのDrive版が本ビルダー)。

## なぜ fold0 だけか

3-fold 全部を70 epoch 回すのは高い。この文書は繰り返し
「単一foldのbestを一般化するな」と書いてきたので、**採用判定はしない**。
fold0 OOF が V35 の fold0 0.4393 を上回ったときだけ fold1/2 に進む。

## 上書き禁止

V30 の `solar_filament_v30/runs/yolo_fold_*` は触らない。
重みと OOF cache は `solar_filament_v35/` 以下へ書く。

## 変えるもの / 変えないもの

変える: モデル(`yolo11m-seg → yolo11l-seg`)、`lr0`(7e-4→3.5e-4)。
VRAM のため batch は 4 → 2（OOMならさらに1へ自動フォールバック）。
変えない: imgsz(1280)、epoch(70)、patience、mask_ratio、augment、
grouped folds、前処理 JPEG q96、後処理（V35 確定構成）。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v41_colab_modelsize_v2.ipynb"
# ノートに埋め込むと `from __future__ import annotations` が
# COCO の変数名 `annotations` を __future__._Feature で上書きする。


try:
    from .build_v41_colab_modelsize_v2_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v41_colab_modelsize_v2_cells import cells

for index, cell in enumerate(cells):
    cell["id"] = f"v41cp-{index:02d}"
    if cell["cell_type"] != "code":
        continue
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
assert "YOLO_IMGSZ = 1280" in text
assert 'YOLO_MODEL = "yolo11l-seg.pt"' in text
assert "lr0=3.5e-4" in text
assert "solar_filament_v35" in text
assert "yolo_modelsize_v2_fold_" in text
print(OUTPUT)
