"""V89 探針ノートのビルダー: cls=1.5 の fold0/1 複製 + 移行 operating point 検証。

## 背景 (v88 結果に基づく)

v88 (fold2) で cls15 は paired Δ@0.26 = +0.0038 (事前ゲート +0.005 未達、
reject_noise) だったが、operating curve が右にシフトし conf 0.32 では
+0.0117 (0.4533 vs prod 0.4416)。fold2 の観察だけでは (a) cls15 の
改善が fold 間で一般化するか、(b) cls15 は prod と異なる operating
point を持つべきか、の 2 点が未検証。v89 は fold0 (CFG.FOLD を 1 に
変えれば fold1) で cls15 を同レシピ再学習し、paired prod と比較する。

## 変えるもの / 変えないもの

変える: fold (0 または 1)。`model.train(..., cls=1.5)` は v88 と同一。
変えない: モデル・imgsz・epoch・augment・seed (3407+fold)・
all-labeler データセット構築・評価経路は prod と完全一致。

## 事前固定ゲート (RESEARCH_LOG §2.41) — 2 つの判定を両方記録

- 同一 operating point: `PQ(cls15,0.26) − PQ(prod,0.26) >= +0.005`
- 移行 operating point (fold2 観察から 0.32 を fold0/1 に事前固定):
  `PQ(cls15,0.32) − PQ(prod,0.26) >= +0.005`
- 採用検討には fold0 + fold1 両方での合格が必要 (本ノートは fold 単体)
- いずれも未達なら reject_noise

## 診断 (採用判定に使わない)

v88 と同じ低 conf coverage 診断 (conf [0.02,0.26) の IoU>0.5 生提案で
覆える primary FN の数) を継続記録し、fold 間で病態構造が同じか見る。
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v89_cls15_fold01_colab.ipynb"


try:
    from .build_v89_cls15_fold01_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v89_cls15_fold01_cells import cells

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
