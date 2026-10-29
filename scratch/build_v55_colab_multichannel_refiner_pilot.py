"""マルチチャンネルrefiner入力パイロット notebook builder — WINNING_STRATEGY_2026-08-29.md §6.12優先度2。

## 目的

2026-09-15の競合公開コード調査(`gopi470/VisionX`)で見つかった改善レバーの
優先度2: refinerの入力に「seedマスク」「正規化距離変換」の2チャンネルを追加する。
現状の`InstanceCropDataset`は色画像3chのみを入力しており、refinerは「クロップ内の
どの構造を残すべきか」を画像内容だけから暗黙的に判断しなければならない。
複数のfilamentが同じクロップ内に近接する場合、これは原理的に曖昧になりうる。

## 設計上の重要な注意(V46の教訓を踏まえる)

**学習時のseedチャンネルにGTマスクをそのまま使ってはいけない。** target(損失の
正解)とseedが同一だと、モデルは恒等写像を学習するだけで何も精密化を学ばない。
また、推論時のseedは実際のYOLO候補マスク(不完全)である一方、学習時にGTを
そのまま使うと訓練/推論分布の不一致が生じる——これはV46(README.md §2の
「refinerのfree-variant向け再学習」)が踏んだのと同種の罠であり、繰り返さない。

**対策: 学習時はGTマスクをランダムに侵食/膨張させた「疑似proposal」をseedとして
使う。** これはinteractive/promptable segmentation(DEXTR、RITM等)の標準的な
学習手法(GTをランダム摂動してプロンプトを模擬する)と同じ発想。侵食で細い
filament構造が消滅する場合は軽い膨張にフォールバックする安全策を入れている
(ローカルで2000試行・出現率0%を確認済み)。距離変換はクロップごとの最大値で
動的に正規化する(固定定数で正規化すると細い構造で常に極小値になり情報量が
失われるため、ローカルで実測して確認済み)。

## 設計

アーキテクチャは`MediSegUNet`(base)に固定し、fold0で2構成 × seed(3407, 8407) =
4本を学習・OOF評価する:

  1. `baseline`      — 現行と完全同一(3ch: 色画像のみ)
  2. `multichannel`   — 5ch(色画像3ch + seedマスク1ch + 正規化距離変換1ch)

Dataset自体は常に5chを生成し(augmentationのRNG消費を両構成で完全に揃えるため)、
`baseline`構成は先頭3chだけを切り出してモデルへ渡す(数値的には元の3ch入力と
完全に同一)。損失関数・学習率スケジュール・augmentation・後処理は
`build_v54_colab_ema_pilot.py`(EMAパイロット、優先度1、baseline結果が出次第
比較の起点にする)と同一。EMAは併用しない(1変数ずつ検証する規律を守る)。

## 使い方

    python scratch/build_v55_colab_multichannel_refiner_pilot.py

出力: solar_filament_v55_multichannel_refiner_pilot_fold0_colab.ipynb
(Colab A100/L4推奨、4本直列で1〜1.5時間程度の見込み)
"""

import json
from pathlib import Path

try:
    from .build_v55_colab_multichannel_refiner_pilot_cells import ROOT, build_cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v55_colab_multichannel_refiner_pilot_cells import ROOT, build_cells


def main():
    cells = build_cells()
    for index, cell in enumerate(cells):
        cell["id"] = f"v55-multichannel-{index:02d}"

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

    output = ROOT / "solar_filament_v55_multichannel_refiner_pilot_fold0_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
