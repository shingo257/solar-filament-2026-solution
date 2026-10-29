"""V51 — EdgeAttNet風refiner(EG-MHSA)の学習ノートブックビルダー(全fold対応)。

## 目的

WINNING_STRATEGY_2026-08-29.md §1.4 / Phase 2。現行refiner(`MediSegUNet`、
プレーンU-Net、31M)を、bottleneckにEdge-Guided Multi-Head Self-Attention
(EG-MHSA)を挿入した`MediSegEdgeAttUNet`に置き換える効果を検証する。

参考文献: "EdgeAttNet: Towards Barb-Aware Filament Segmentation"
(arXiv:2509.02964)。MAGFiLO上でmIoU_pairwise 0.5724(U-Net)→0.6451
(EdgeAttNet)を報告。ただし著者の公開リポジトリには実際のアーキテクチャ
コードが存在しないことを確認済み(`solar_filament/models/mediseg_models.py`
のクラス定義コメント参照)。本ノートブックは論文の記述のみを根拠にした
独自の再実装を使う。

## 使い方

    python scratch/build_v51_edgeattnet_refiner.py --fold 0   # fold0を学習(他2foldはvalid)
    python scratch/build_v51_edgeattnet_refiner.py --fold 1
    python scratch/build_v51_edgeattnet_refiner.py --fold 2
    python scratch/build_v51_edgeattnet_refiner.py --fold all # 提出用: 全707画像でfine-tune

`--fold 0/1/2` はそれぞれのfoldを検証専用に除外して学習し(リークなし
OOF評価が可能)、`--fold all` はfold0のcheckpointをウォームスタートして
全707画像へ低LR・短epochで再学習する(V30本体のall-data refinerと
同じレシピ、提出モデル用)。

fold0/1/2の3本すべてが2026-08-29〜30の検証で「3-fold全結合(n=707)で
Δ+0.0028、95%CI[0.000915,0.004756]、ゼロを完全に除外」という結果を
再現しており(README §2.1.5)、このスクリプトはその結果を再現するための
正本である。

## 設計方針: 学習条件は`build_v35_refiner_folds12.py`と同一

crop size 320、context 1.65、loss = 0.35·BCE + 0.65·softDice + 0.15·edge、
14 epoch(all-dataは5 epoch・低LR)、LR 2e-4、CosineAnnealing、AMP、
crop Diceでbest選択——**モデルアーキテクチャ以外は一切変えない**。
"""

import argparse
import json
from pathlib import Path

try:
    from .build_v51_edgeattnet_refiner_cells import ROOT, build_cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v51_edgeattnet_refiner_cells import ROOT, build_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["0", "1", "2", "all"],
                         help="学習対象。0/1/2はその他foldで学習しそのfoldをOOF検証、"
                              "allはfold0からウォームスタートして全707画像で提出用モデルを再学習")
    args = parser.parse_args()

    cells = build_cells(args.fold)
    for index, cell in enumerate(cells):
        cell["id"] = f"v51-{args.fold}-{index:02d}"

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

    output = ROOT / f"solar_filament_v51_edgeattnet_refiner_{args.fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
