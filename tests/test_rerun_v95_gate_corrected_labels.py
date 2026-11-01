"""Behavioral tests for the corrected v95 gate exact-PQ evaluation
(scratch/rerun_v95_gate_corrected_labels.py, §2.64).

`exact_pq_after_drop` computes the exact PQ after removing a set of
predictions: numerator loses the dropped preds' TP IoU, denominator
loses their TP/FP contributions, and GTs whose covering preds are ALL
dropped become new FNs (+0.5 each). Synthetic frames pin every branch:
empty drops, FP-only drops, unique-cover FN creation, and shared-cover
partial/full drops.
"""

import pandas as pd
import pytest

from scratch.rerun_v95_gate_corrected_labels import exact_pq_after_drop

BASE = {"tp": 10, "fp": 4, "fn": 2, "pq": 0.5}  # D = 10 + 2 + 1 = 13


def _feats(rows):
    """(file, pred_idx, tp_iou_sum, n_tp, n_fp) -> feats frame."""
    return pd.DataFrame(rows, columns=[
        "file_name", "pred_idx", "tp_iou_sum", "n_tp", "n_fp"])


def _tps(rows):
    """(file, record, gt_local, pred_idx) -> TP-pair frame."""
    return pd.DataFrame(rows, columns=[
        "file_name", "record_id", "gt_local", "pred_idx"])


def test_empty_drop_returns_base_pq():
    feats = _feats([("a.jpg", 0, 0.8, 1, 0)])
    tps = _tps([("a.jpg", 0, 0, 0)])
    assert exact_pq_after_drop(feats, tps, BASE, set()) == BASE["pq"]


def test_drop_fp_only_pred_raises_pq():
    # FP-only pred: numerator unchanged, denominator loses 0.5*n_fp.
    feats = _feats([
        ("a.jpg", 0, 0.8, 1, 0),   # TP pred, kept
        ("a.jpg", 1, 0.0, 0, 2),   # FP-in-2-records pred, dropped
    ])
    tps = _tps([("a.jpg", 0, 0, 0)])
    got = exact_pq_after_drop(
        feats, tps, BASE, {("a.jpg", 1)})
    # numerator = 0.5*13 - 0 = 6.5; D = 13 - 0 - 1 + 0 = 12
    assert got == pytest.approx(6.5 / 12)
    assert got > BASE["pq"]


def test_drop_unique_cover_pred_creates_fn():
    # p1 uniquely covers gt0 -> removing it makes gt0 a new FN:
    # numerator -0.8, denominator -1(TP) +0.5(new FN) = -0.5.
    feats = _feats([("a.jpg", 0, 0.8, 1, 0)])
    tps = _tps([("a.jpg", 0, 0, 0)])
    got = exact_pq_after_drop(
        feats, tps, BASE, {("a.jpg", 0)})
    # numerator = 6.5 - 0.8 = 5.7; D = 13 - 1 - 0 + 0.5 = 12.5
    assert got == pytest.approx(5.7 / 12.5)


def test_shared_cover_partial_drop_creates_no_fn():
    # gt0 covered by p0 AND p1: dropping only p0 leaves gt0 covered.
    feats = _feats([
        ("a.jpg", 0, 0.7, 1, 0),
        ("a.jpg", 1, 0.7, 1, 0),
    ])
    tps = _tps([
        ("a.jpg", 0, 0, 0),
        ("a.jpg", 0, 0, 1),
    ])
    got = exact_pq_after_drop(
        feats, tps, BASE, {("a.jpg", 0)})
    # numerator = 6.5 - 0.7 = 5.8; D = 13 - 1 - 0 + 0 = 12 (no new FN)
    assert got == pytest.approx(5.8 / 12)


def test_shared_cover_full_drop_creates_fn():
    feats = _feats([
        ("a.jpg", 0, 0.7, 1, 0),
        ("a.jpg", 1, 0.7, 1, 0),
    ])
    tps = _tps([
        ("a.jpg", 0, 0, 0),
        ("a.jpg", 0, 0, 1),
    ])
    got = exact_pq_after_drop(
        feats, tps, BASE, {("a.jpg", 0), ("a.jpg", 1)})
    # numerator = 6.5 - 1.4 = 5.1; D = 13 - 2 - 0 + 0.5 = 11.5
    assert got == pytest.approx(5.1 / 11.5)


def test_drop_is_scoped_per_file():
    # Same pred_idx in a different file must not be dropped.
    feats = _feats([
        ("a.jpg", 0, 0.8, 1, 0),
        ("b.jpg", 0, 0.9, 1, 0),
    ])
    tps = _tps([
        ("a.jpg", 0, 0, 0),
        ("b.jpg", 0, 0, 0),
    ])
    got = exact_pq_after_drop(
        feats, tps, BASE, {("a.jpg", 0)})
    # only a.jpg:0 removed: numerator = 6.5 - 0.8 = 5.7; D = 12.5
    assert got == pytest.approx(5.7 / 12.5)


def test_fp_and_tp_terms_combine_per_pred():
    # Pred with both TP and FP records: denominator delta is
    # -n_tp - 0.5*n_fp (+0.5*newly_fn).
    feats = _feats([("a.jpg", 0, 0.8, 1, 2)])
    tps = _tps([("a.jpg", 0, 0, 0)])
    got = exact_pq_after_drop(
        feats, tps, BASE, {("a.jpg", 0)})
    # numerator = 6.5 - 0.8 = 5.7; D = 13 - 1 - 1 + 0.5 = 11.5
    assert got == pytest.approx(5.7 / 11.5)
