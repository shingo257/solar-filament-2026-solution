"""Characterization tests for dense-scene replication counts."""

from __future__ import annotations

import math

from solar_filament.pipeline.dense_oversample import (
    dense_cut_value,
    dense_replication_counts,
)


def test_empty_input_returns_nan_cut_and_no_reps() -> None:
    cut, reps = dense_replication_counts([])
    assert math.isnan(cut)
    assert reps == []


def test_known_quantile_marks_top_quartile() -> None:
    counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cut, reps = dense_replication_counts(counts, quantile=0.75)
    assert cut == 7.75
    assert reps == [1, 1, 1, 1, 1, 1, 1, 2, 2, 2]


def test_reps_are_one_plus_extra_for_dense_only() -> None:
    counts = [0, 5, 10, 15, 20]
    cut, reps = dense_replication_counts(counts, quantile=0.6, extra_replicas=2)
    assert reps == [1 + 2 if c >= cut else 1 for c in counts]
    assert all(isinstance(r, int) for r in reps)


def test_extra_replicas_zero_disables_oversampling() -> None:
    _cut, reps = dense_replication_counts([3, 9, 12], quantile=0.75, extra_replicas=0)
    assert reps == [1, 1, 1]


def test_flat_distribution_marks_everything_dense() -> None:
    # 全サンプル同値なら分位数もその値になり、全員が複製対象になる。
    # 「weightが全員2倍=実質変化なし」として無害なので、これを仕様として固定する。
    cut, reps = dense_replication_counts([4, 4, 4, 4])
    assert cut == 4.0
    assert reps == [2, 2, 2, 2]


def test_dense_cut_value_matches_replication_cut() -> None:
    counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert dense_cut_value(counts, 0.75) == dense_replication_counts(counts, 0.75)[0]


def test_dense_cut_value_empty_is_nan() -> None:
    assert math.isnan(dense_cut_value([]))


def test_accepts_numpy_array_and_returns_plain_ints() -> None:
    import numpy as np

    _, reps = dense_replication_counts(np.array([1, 2, 100]))
    assert reps[-1] == 2
    assert type(reps[-1]) is int
