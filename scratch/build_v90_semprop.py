"""V90 探針ノートのビルダー: semantic U-Net 提案源による blind-FN 救済。

## 背景 (v88/v89 診断に基づく)

v88 (fold2) の低 conf 診断で primary FN の 57% (543/953) は conf
[0.02,0.26) に IoU>0.5 の候補を持つことが分かった (fold0 でも 57.7%
と同型)。残り ~43% は detector が any-conf で提案を出さない
blind-FN で、conf 再配分やアンサンブルでは構造的に拾えない。

biconcavelens (§2.39) は semantic ConvNeXt-T U-Net (1024px crop、
(file,annotator) エントリ単位、4-flip TTA) を独立提案源として
detector-missed filament を ~30個追加獲得した (OOF 0.461, LB 0.40)。
v90 はこの機構を我々の評価経路に移植する fold2 pilot。

## 変えるもの / 変えないもの

変える: 提案源に semantic U-Net を追加 (学習は時間予算 ~1.2h)。
変えない: prod det 経路・refiner・ownership・公式PQ 採点・分割・
評価画像は prod と完全一致。union は det 提案に dedup 統合するだけ。

## 事前固定ゲート (RESEARCH_LOG §2.42)

- `PQ(union,0.26) − PQ(prod,0.26) >= +0.005` で fold0/1 昇格検討
- `[0,+0.005)` は reject_noise、`<0` は reject
- `semprop` (semantic 単独) は提案源性能の診断であり採用候補にしない
- 診断: prod の primary FN が semantic 提案で IoU>0.5 救済された数
  (`fn_covered_by_sem_prop`) を `diag_semrescue` に記録

## 実行

そのまま全実行で fold2 pilot (~1.5h 学習 + 確率map ~10分 + 評価)。
checkpoint・確率map・manifest は Drive に冪等保存され再実行で skip。
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v90_semprop_fold2_colab.ipynb"


try:
    from .build_v90_semprop_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v90_semprop_cells import cells

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
