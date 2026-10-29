"""EMA(重み平均)パイロット notebook builder — WINNING_STRATEGY_2026-08-29.md §6.12優先度1。

## 目的

2026-09-15の競合公開コード調査(`gopi470/VisionX`)で見つかった、自チームが
一度も試していない改善レバーのうち最優先(低コスト・確認済みで未試行)の
EMA(Exponential Moving Average、重み平均)を検証する。

**実装前の確認事項(重要):** boundary loss(境界重視損失)は既にV30から
`refiner_loss`に組み込まれており未試行ではなかった(README.md §2.1.12参照)。
このnotebookは`refiner_loss`を一切変更せず、EMAの有無だけを変数にする。

## 設計

アーキテクチャはmatched-parameter ablationで最も長い実績を持つ`base`
(MediSegUNet、標準refiner)に固定し、fold0で2構成 × seed(3407, 8407) =
4本を学習・OOF評価する:

  1. `baseline` — 現行のV51/matched-parameter ablationと完全同一(EMAなし)
  2. `ema`      — 上記に加え、optimizer.step()ごとにEMA shadow重みを更新
                  (decay=0.999)。エポックごとの検証はEMA shadow重みで行い、
                  最良エポックのEMA shadow重みをcheckpointとして保存する

損失関数・学習率スケジュール・augmentation・batch size選択ロジックは
`build_matched_param_ablation.py`と完全同一(変数はEMAの有無のみ)。

fold別基準(標準refiner、matched-parameter ablation実測、2seed平均):
fold0 0.4414 / fold1 0.4405 / fold2 0.4434(README.md §2.1.10)。

## 使い方

    python scratch/build_v54_colab_ema_pilot.py --fold 0
    python scratch/build_v54_colab_ema_pilot.py --fold 1
    python scratch/build_v54_colab_ema_pilot.py --fold 2

出力: solar_filament_v54_ema_pilot_fold{N}_colab.ipynb (Colab A100/L4推奨)
4本の学習(各14epoch)を直列に回すため、1〜1.5時間程度の見込み。

複数foldを1セッションで直列に回す統合版(fold0の結果を受けたfold1/2の確認用):

    python scratch/build_v54_colab_ema_pilot.py --folds 1 2

出力: solar_filament_v54_ema_pilot_fold1_2_colab.ipynb。単独fold版と同じセルを
foldごとに並べ、(1) Driveのpartial JSONから完了済みrunを飛ばす途中再開、
(2) ピークVRAM・所要時間のmanifest記録、(3) Drive上の全foldのmanifestを
読む最終判定セル、を追加する。学習・評価のレシピは単独fold版と完全同一。
"""

import argparse
import json
from pathlib import Path

try:
    from .build_v54_colab_ema_pilot_cells import ROOT, build_cells, build_combined_cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v54_colab_ema_pilot_cells import ROOT, build_cells, build_combined_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", choices=["0", "1", "2"], help="評価対象fold")
    group.add_argument("--folds", nargs="+", choices=["0", "1", "2"],
                       help="1セッションで直列に回すfold(統合版notebookを生成)")
    args = parser.parse_args()

    if args.folds:
        folds = sorted({int(f) for f in args.folds})
        cells = build_combined_cells(folds)
        tag = "_".join(str(f) for f in folds)
        cell_prefix = f"v54-ema-f{tag}"
    else:
        fold = int(args.fold)
        cells = build_cells(fold)
        tag = str(fold)
        cell_prefix = f"v54-ema-f{fold}"
    for index, cell in enumerate(cells):
        cell["id"] = f"{cell_prefix}-{index:02d}"

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

    output = ROOT / f"solar_filament_v54_ema_pilot_fold{tag}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
