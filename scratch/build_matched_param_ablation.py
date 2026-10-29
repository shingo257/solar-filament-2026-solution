"""Matched-parameter ablation notebook builder — report/main.tex Future Work item 1.

## 目的

report/main.tex(3-5回の外部査読で繰り返し指摘された未解決の交絡)を検証する:
「refiner(MediSegEdgeAttUNet)の改善は edge prior そのものの効果か、それとも
単にbottleneckのattention容量が増えただけの効果か」を切り分ける。

3構成を指定foldで比較する(いずれもそのfoldを検証専用に除外、他は全て同一レシピ):

  1. MediSegUNet          — プレーンU-Net baseline (31.04M params)
  2. MediSegPlainMHSAUNet — bottleneckに標準MHSA追加、edge priorなし (35.24M params)
  3. MediSegEdgeAttUNet   — bottleneckにEG-MHSA追加、edge prior あり (35.24M params)

2と3はパラメータ数がほぼ完全に一致する(差は edge_conv+edge_proj の約2,076個、
全体の0.006%)。2の存在により「1→2の差」(attention容量を足しただけの効果)と
「2→3の差」(edge priorを足した追加効果)を分離できる。これがEdgeAttNet論文
自身のablation("U-Net + MHSA, no positional encoding")と同じ切り分け。

各構成をseed 3407・8407の2本ずつ、計6本学習する(V42のmulti-seed実験と同じ
seed規約)。既存のfold別YOLO検出器・確定後処理(shrink@0.45 + drop_contained@0.90)
はそのまま流用し、refinerアーキテクチャとseedだけを変える。

## 背景(2026-08-31、fold0実測)

fold0(n=236)でこのablationを実行した結果、edge-attention refinerは
baselineを上回らず、3構成中最低だった(mean PQ: base 0.441412 >
plain_mhsa 0.441259 > edge_att 0.440474)。report/main.texはこの反転を
反映して全面改訂済み。旧報告の3-fold delta(fold0 +0.0062、fold1 +0.0009、
fold2 +0.0012)のうちfold1/fold2は今回実測したseed間ノイズ(std 6-8e-4)
より小さく、fold0限定では断定できない。fold1・fold2でも同じablationを
回し、この反転がfold0固有の現象か、プロジェクト全体に共通する現象かを
確認する必要がある(report Future Work item 1)。

## 使い方

    python scratch/build_matched_param_ablation.py --fold 0
    python scratch/build_matched_param_ablation.py --fold 1
    python scratch/build_matched_param_ablation.py --fold 2

出力: solar_filament_matched_param_ablation_fold{N}_colab.ipynb (Colab A100/L4推奨)
6本の学習(各14epoch)を直列に回すため、fold毎に2〜3時間かかる見込み。
"""

import argparse
import json
from pathlib import Path

try:
    from .build_matched_param_ablation_cells import ROOT, build_cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_matched_param_ablation_cells import ROOT, build_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["0", "1", "2"], help="学習対象fold")
    args = parser.parse_args()
    fold = int(args.fold)

    cells = build_cells(fold)
    for index, cell in enumerate(cells):
        cell["id"] = f"ablation-f{fold}-{index:02d}"

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

    output = ROOT / f"solar_filament_matched_param_ablation_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
