"""V92 探針ノートのビルダー: Cellpose flow-field 提案源 (Biohub 1st 手法の移植)。

## 背景 (§2.50-2.51)

Biohub CTDD 上位手法の棚卸しで、Solar が未試行の本格機構は Cellpose
型 flow-field 個体分離のみと判明した。v90/v91 の free-blob pixel
提案は FP 爆発で死んだが、flow-field は各画素を必ず 1 つの
インスタンス中心に束縛するため blob より厳格な個体を生成する。
merged instance の分離を box-NMS と別原理で解ける可能性がある。

## 変えるもの / 変えないもの

変える: semantic U-Net を 3ch 化 (fg + flow_y + flow_x)。flow 教師は
各画素→所属 instance 重心への unit ベクトル。推論は fg 画素を
flow で Euler 追従 (~120 step) → 端点を DBSCAN でクラスタ →
クラスタ毎の画素集合を 1 instance 提案とする。
変えない: prod det 経路・refiner・ownership・公式PQ 採点・分割・
評価画像は prod と完全一致。union は det 提案に dedup 統合するだけ。

## 事前固定ゲート (RESEARCH_LOG §2.52)

- `PQ(union,0.26) − PQ(prod,0.26) >= +0.005` で fold0/1 昇格検討
- `[0,+0.005)` は reject_noise、`<0` は reject
- `flowprop` (flow 単独) は提案源性能の診断であり採用候補にしない
- 診断: prod の primary FN が flow 提案で IoU>0.5 救済された数
  (`fn_covered_by_flow_prop`) を `diag_flowrescue` に記録

## 実行

そのまま全実行で fold2 pilot (~1.2h 学習 + flow map ~10分 +
flow 追従 ~10分 + 評価)。checkpoint・map・manifest は Drive に
冪等保存され再実行で skip。
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v92_cellpose_flow_fold2_colab.ipynb"


try:
    from .build_v92_cellpose_flow_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v92_cellpose_flow_cells import cells

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
