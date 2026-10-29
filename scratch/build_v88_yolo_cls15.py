"""V88 探針ノートのビルダー: YOLO cls=1.5 (分類損失 reweight) で fold2 を再学習し、
リークなしで prod と paired 評価する。

## 仮説 (biconcavelens §2.39 の診断に基づく)

biconcavelens は「well-localized だが conf が著しく低い」missed filament を
発見し (bbox-IoU 0.70-0.87 が score 0.05)、ultralytics の cls 損失重みを
0.5→1.5 に上げることで confidence head の undertraining を修正、solo PQ
0.4171→0.4252 / LB 0.37→0.38 を得た。我々の pipeline でも同じ病態が
detector FN の一部を占める可能性がある。

## 変えるもの / 変えないもの

変える: `model.train(..., cls=1.5)` のみ (ultralytics default 0.5)。
変えない: モデル (yolo11m-seg)、imgsz (1280)、epoch (70)、patience (14)、
augment 一式、seed (3407+fold)、all-labeler データセット構築、fold2 のみ、
評価経路 (EdgeAtt fold2 refiner → drop_contained@0.90 → ownership
min_area=20 → 公式 PQ、conf sweep {0.15,0.20,0.26,0.32}、primary=0.26)。

## 追加の診断 (採用判定に使わない)

推論 floor を 0.02 に下げ、生提案 mask を全て保持。primary conf で FN となった
GT のうち「conf [0.02,0.26) に IoU>0.5 の生提案がある」割合を記録する。
detector-blind FN の中に低 score 候補が何割埋まっているかを直接測り、
biconcavelens 診断の我々への適用可能性を検証する (refine は conf>=0.08 の
提案のみ = prod 経路と同一)。
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v88_yolo_cls15_fold2_colab.ipynb"


try:
    from .build_v88_yolo_cls15_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v88_yolo_cls15_cells import cells

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"colab": {"provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "accelerator": "GPU"},
    "cells": cells,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUTPUT}")
