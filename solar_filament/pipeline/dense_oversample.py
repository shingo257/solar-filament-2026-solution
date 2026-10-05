"""Dense-scene replication for detector training — WINNING_STRATEGY_2026-08-29.md §6.12 優先度6。

診断済みの主要失敗モードは密集シーン(gt_count_mean 10-21)での検出漏れ
(hard FN 2138件、hard FP 1561件を37%上回る、2026-08-16診断)。YOLO検出器の
学習データにおいて、GTインスタンス数が上位に入る観測のサンプルを複製して
見かけ上の出現頻度を上げる(oversampling)、というまだ試していないデータ側
レバーの判定ロジック。

複製係数の計算だけを担い、データセットへの実際の書き込みは呼び出し側
(`scratch/build_v57_colab_dense_oversample_pilot.py` が生成するnotebook)が行う。
GPU学習を伴わない純粋な計算なのでローカルでテスト可能
(`tests/test_dense_oversample.py`)。
"""

from __future__ import annotations

import numpy as np


def dense_cut_value(gt_counts, quantile=0.75):
    """「密集」とみなすGTインスタンス数の分位点カット値を返す。空入力なら nan。"""
    counts = np.asarray(list(gt_counts), dtype=np.float64)
    if counts.size == 0:
        return float("nan")
    return float(np.quantile(counts, quantile))


def dense_replication_counts(gt_counts, quantile=0.75, extra_replicas=1):
    """各学習サンプルの複製係数を返す。

    Args:
        gt_counts: 学習サンプル(file, annotator)ごとの非crowd GTインスタンス数の列。
        quantile: 「密集」とみなす分位点。gt_countがこの分位数以上のサンプルを複製対象にする。
        extra_replicas: 複製対象に追加するコピー数。1なら対象サンプルは計2回出現する。

    Returns:
        (cut, reps): cut は適用した分位数(float)、reps は各サンプルの総出現回数の
        int リスト(非対象は1)。gt_counts が空なら (nan, [])。
    """
    counts = np.asarray(list(gt_counts), dtype=np.float64)
    if counts.size == 0:
        return float("nan"), []
    cut = dense_cut_value(counts, quantile)
    reps = [1 + int(extra_replicas) if c >= cut else 1 for c in counts]
    return cut, reps
