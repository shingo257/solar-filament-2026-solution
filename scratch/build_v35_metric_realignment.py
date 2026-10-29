"""V35 診断ノートブックのビルダー。

目的は 4 つで、学習も test 推論も行わない。

  優先度1: 公式指標 Panoptic Quality で既存 OOF を再採点し、conf 動作点を引き直す
  優先度2: V31 の合格線 0.552553 を PQ の全 fold OOF で置き換える
  優先度3: duplicate 発生率をアノテータ数で層別し、主因が学習ラベル方針か判定する
  追加   : アノテータ不一致が課す PQ の天井を測り、0.35 と LB 1位 0.48 を相対化する

V34 の Drive には V30 YOLO の全 fold OOF cache が 2048x2048 のまま残っているので、
YOLO の再推論は不要。GPU なし・標準 RAM の Colab CPU で完走する想定。
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v35_metric_realignment_colab.ipynb"


try:
    from .build_v35_metric_realignment_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v35_metric_realignment_cells import cells

for index, cell in enumerate(cells):
    cell["id"] = f"v35-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
