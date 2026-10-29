"""v80 E1 — ハイブリッド候補集合 + 期待IoUスコアラーの全fold OOF 評価 (CPU)。

PLAN_2026-10-09_FINAL_SPRINT.md §7.1.4〜7.1.8。
`run_v80_hybrid_candidates.py` が保存した候補キャッシュ
(`solar_filament_v80/candidates/fold_{f}/<stem>.npz`) を読み、

  1. 教師 u(c) = (1/R) Σ_r max_g IoU(c, g_r) · 1[IoU > 0.5] を各候補に計算
     (h = 当たる割合、m = 当たった record の平均 IoU、u = h·m は厳密な分解)
  2. HistGradientBoosting (h 回帰 + m 回帰、ハイパラ固定、重み=record 数 R)
     を fold 外候補のみで学習し、fold 内候補に cross-fit で û を出す
  3. û > τ 選択 + tree 候補は Σ(û−τ) 最大の antichain DP → 排他化 → 公式 PQ

arms (同一候補・同一 refiner 出力・paired):
  prod          : yolo conf>=0.26 (現行ベースライン、τ に依らない)
  yolo_scorer   : yolo 候補 ∩ û>τ                (v64 相当の対照)
  tree_only     : tree 候補の antichain DP
  hybrid        : yolo û>τ ∪ tree antichain
  hybrid_oracle : 同上だが真値 u で選択 (機構の上界)

採用ゲート: hybrid − prod が 3fold すべて正 かつ pooled bootstrap 95%CI が
ゼロを除外。hybrid_oracle が +0.02 未満なら候補集合の上界が低いと判断する。
ただし現行キャッシュは上流モデルまでouter-fold別に再学習したnested構成ではない。
数値条件は探索的診断のみで、adoptは常にfalseとする。

使い方:
    python scratch/run_v80_scorer_eval.py [--limit N] [--workers 8] [--boot 2000]
"""


from __future__ import annotations


import sys
from pathlib import Path


try:
    from .run_v80_scorer_eval_impl import (
        CONTAINMENT,
        CONTEXT_FEATURES,
        EVAL_SIZE,
        FOLDS,
        NA_FEATURES,
        PROD_CONF,
        TAU_GRID,
        _evaluate_image,
        adoption_gate,
        context_features,
        load_candidates,
        rle_dicts,
        teachers_for,
    )
    from .run_v80_scorer_eval_main import main
except ImportError:  # Direct `python scratch/...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_v80_scorer_eval_impl import (
        CONTAINMENT,
        CONTEXT_FEATURES,
        EVAL_SIZE,
        FOLDS,
        NA_FEATURES,
        PROD_CONF,
        TAU_GRID,
        _evaluate_image,
        adoption_gate,
        context_features,
        load_candidates,
        rle_dicts,
        teachers_for,
    )
    from run_v80_scorer_eval_main import main



__all__ = [
    "CONTAINMENT",
    "CONTEXT_FEATURES",
    "EVAL_SIZE",
    "FOLDS",
    "NA_FEATURES",
    "PROD_CONF",
    "TAU_GRID",
    "_evaluate_image",
    "adoption_gate",
    "context_features",
    "load_candidates",
    "rle_dicts",
    "teachers_for",
]


if __name__ == "__main__":
    raise SystemExit(main())
