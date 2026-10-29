"""P2-2 探針ノートのビルダー: YOLO imgsz 1536 を fold0 だけで試す。

V35 の確定後処理（shrink@0.45 + drop_contained_then_area_asc, conf 0.26）を
固定したまま、検出器の入力解像度だけを 1280 → 1536 に上げる。
YOLO11-seg の mask prototype は概ね imgsz/4 なので、320² → 384² になる。

## なぜ fold0 だけか

3-fold 全部を 1536 で 70 epoch 回すのは高い。この文書は繰り返し
「単一foldのbestを一般化するな」と書いてきたので、**採用判定はしない**。
fold0 OOF が V35 の fold0 0.4393 を上回ったときだけ fold1/2 に進む。

## 上書き禁止

V30 の `solar_filament_v30/runs/yolo_fold_*` は触らない。
重みと OOF cache は `solar_filament_v35/` 以下へ書く。

## 変えるもの / 変えないもの

変える: `imgsz` だけ（1280 → 1536）。VRAM のため batch は 4 → 2。
変えない: モデル (yolo11m-seg)、epoch、patience、mask_ratio、 augment、
grouped folds、前処理 JPEG q96、後処理（V35 確定構成）。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v36_yolo_imgsz1536_fold0_colab.ipynb"
# ノートに埋め込むと `from __future__ import annotations` が
# COCO の変数名 `annotations` を __future__._Feature で上書きする。


try:
    from .build_v36_yolo_imgsz_probe_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v36_yolo_imgsz_probe_cells import cells

for index, cell in enumerate(cells):
    cell["id"] = f"v36p-{index:02d}"
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
assert "YOLO_IMGSZ = 1536" in text
assert "solar_filament_v35" in text
assert "yolo_imgsz1536_fold_" in text
print(OUTPUT)
