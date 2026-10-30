"""V91 探針ノートのビルダー: v90 cache 確率 map の提案厳格化 grid + rescue-scorer。

## 背景 (v90 §2.45 結果に基づく)

v90 は blind-FN 救済機構の存在を実証した (union TP +61) が、
sem 提案の粗さが FP +524 を量産し Δ = −0.0307 で reject。
本実験は v90 が Drive に cache した確率 map を再利用し、
**学習なし** で提案化パラメータだけを変えて 2 機構を検索する。

## Arm

- Arm A `union_grid`: prop_thresh x min_area x dedup_iou = 27 設定
- Arm B `rescue_grid`: det conf [0.08,0.26) の提案を mean sem 確率
  >= theta で昇格。theta grid (543 件 lowconf-covered FN 層を狙う)

## 事前固定ルール (RESEARCH_LOG §2.46)

- fold2 は検索フェーズ: argmax 設定は採用せず `nominees` に記録のみ
- 昇格は別途 fold0/1 確認実験で `PQ(arm,0.26)−PQ(prod,0.26)>=+0.005`
  を両 fold で満たす場合のみ
- 全 grid が prod を下回る場合、提案源ラインは negative で閉じる

## 実行

前提: v90 ノート完走 (runs/v90_semprop/probs_fold2/*.npz が必要)。
全実行 ~30-45 分 (det cache ~5 分 + thresh ごとの成分/refine + grid)。
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v91_semsweep_fold2_colab.ipynb"


try:
    from .build_v91_semsweep_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v91_semsweep_cells import cells

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
