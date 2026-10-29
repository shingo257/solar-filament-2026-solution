"""V82 — soft annotator-mean 目標の refiner 学習ノートブックビルダー (C2, PLAN §7.2)。

## 目的

現行 refiner の学習単位は (file, image_id, ann) = **record 単位の硬いマスク**。
同一フィラメントを複数アノテータが描いていれば、中心・輪郭の違う 3 サンプル
として独立に学習される。V82 は GT インスタンスを IoU>0.3 でクラスタ化し、
`S_k = (1/R) Σ_r M_{k,r}` (R = 画像のアノテータ数)を soft 目標として与える
効果を検証する (PLAN_2026-10-09_FINAL_SPRINT.md §7.2)。モデルは現行 refiner
と同じ `MediSegUNet` のまま、目標の定義だけを変える。

arms:
  soft       : S_k ∈ [0,1] の soft 目標 (INTER_LINEAR リサイズ)
  hard_union : クラスタの union を 1/0 目標にする対照 (クラスタ化のみの効果を分離)

hard_record (現行) は既存 `mediseg_refiner_fold{f}_resume.pt` がそのまま対照。

## 使い方

    python scratch/build_v82_soft_target_refiner.py --fold 0 --arm soft --seed 3407
    python scratch/build_v82_soft_target_refiner.py --fold 0 --arm hard_union --seed 3407

GPU必須 (Colab L4 推奨、各run ≈1.5h)。fold 専用 OOF 評価は確定後処理
(shrink + drop_contained_then_area_asc)込みで、標準 refiner の基準 PQ と比較。
採用ゲート: soft − hard_record が 3fold・両seedで正、かつ pooled CI がゼロを除外。
"""

import argparse
import json
from pathlib import Path

try:
    from .build_v82_soft_target_refiner_cells import ROOT, build_cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v82_soft_target_refiner_cells import ROOT, build_cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--arm", type=str, required=True, choices=("soft", "hard_union"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cells = build_cells(args.fold, args.arm, args.seed)
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = args.out or ROOT / f"solar_filament_v82_{args.arm}_fold{args.fold}_s{args.seed}_colab.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
