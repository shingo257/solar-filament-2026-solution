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


# ------------------------------------------------- PanopticQuality oracle --
# The cases above pin the formula's arithmetic; this section additionally
# cross-checks exact_pq_after_drop against a real PanopticQuality re-score
# on a synthetic two-record image (the strongest available local oracle).

from solar_filament.metrics.metrics_official import (  # noqa: E402
    PanopticQuality)
import numpy as np  # noqa: E402

# File "img" scored under two annotator records. Final preds:
#   p0: TP in r1 (0.8, shares gt0 with p1), FP in r2
#   p1: TP in r1 (0.6, shares gt0) and r2 (0.7, unique cover), no FP
#   p2: FP in both records
#   p3: TP in r1 (0.7, covers gt1 alone) — survives every drop below, so
#       the orphaned-GT assertions retain a nonzero IoU numerator and can
#       distinguish exact FN accounting from a per-pred n_unique sum.
_ORACLE_R1 = np.array([
    [0.8, 0.6, 0.0, 0.0],   # gt0 vs [p0, p1, p2, p3]
    [0.0, 0.0, 0.0, 0.7],   # gt1 vs [p0, p1, p2, p3]
])
_ORACLE_R2 = np.array([[0.0, 0.7, 0.0, 0.0]])
_N_ORACLE_PREDS = 4


def _oracle_world():
    pq = PanopticQuality()
    pq.add_matrix(_ORACLE_R1, 2, _N_ORACLE_PREDS)
    pq.add_matrix(_ORACLE_R2, 1, _N_ORACLE_PREDS)
    base = pq.summary()
    feats = _feats([
        ("img", 0, 0.8, 1, 1),
        ("img", 1, 1.3, 2, 0),
        ("img", 2, 0.0, 0, 2),
        ("img", 3, 0.7, 1, 0),
    ])
    tps = _tps([
        ("img", "r1", 0, 0),
        ("img", "r1", 0, 1),
        ("img", "r1", 1, 3),
        ("img", "r2", 0, 1),
    ])
    return feats, tps, base


def _oracle_pq(drop_pred_idxs):
    """Ground truth: drop pred columns and re-score with PanopticQuality."""
    keep = [i for i in range(_N_ORACLE_PREDS) if i not in drop_pred_idxs]
    pq = PanopticQuality()
    pq.add_matrix(_ORACLE_R1[:, keep], 2, len(keep))
    pq.add_matrix(_ORACLE_R2[:, keep], 1, len(keep))
    return pq.summary()["pq"]


def _oracle_run(drop_pred_idxs):
    feats, tps, base = _oracle_world()
    keys = {("img", i) for i in drop_pred_idxs}
    return exact_pq_after_drop(feats, tps, base, keys)


def test_oracle_no_drops():
    feats, tps, base = _oracle_world()
    assert exact_pq_after_drop(feats, tps, base, set()) == pytest.approx(
        _oracle_pq(set()))


def test_oracle_fp_only_removal():
    # Dropping p2 removes its 2 FPs only; p0 and p3 stay FP in r2:
    # TP=4, FP=2, FN=0 -> D=5.0.
    assert _oracle_run({2}) == pytest.approx(_oracle_pq({2}))
    assert _oracle_pq({2}) == pytest.approx(2.8 / 5.0)


def test_oracle_unique_cover_removal():
    # Dropping p1 orphans r2's gt0 (r1's gt0 stays covered by p0):
    # TP=2 (p0, p3), FP=4, FN=1 -> D=4.5.
    assert _oracle_run({1}) == pytest.approx(_oracle_pq({1}))
    assert _oracle_pq({1}) == pytest.approx(1.5 / 4.5)


def test_oracle_partial_shared_cover_no_fn():
    # Dropping p0 leaves r1's gt0 covered by p1 — no new FN:
    # TP=3, FP=3, FN=0 -> D=4.5.
    assert _oracle_run({0}) == pytest.approx(_oracle_pq({0}))
    assert _oracle_pq({0}) == pytest.approx(2.0 / 4.5)


def test_oracle_remove_all_preds_sharing_gt():
    # Orphaned-GT case: dropping p0 and p1 removes every prediction
    # covering r1's gt0 AND p1's unique cover of r2's gt0 — two new FNs
    # that a per-pred n_unique sum (0 for p0 + 1 for p1) would undercount.
    # p3 survives as a TP so the numerator stays nonzero and the exact
    # FN count is what decides the score:
    # TP=1 (p3), FP=3 (p2 x2, p3 in r2), FN=2 -> D=3.5;
    # the buggy count would give 0.7/3.0.
    assert _oracle_run({0, 1}) == pytest.approx(_oracle_pq({0, 1}))
    assert _oracle_pq({0, 1}) == pytest.approx(0.7 / 3.5)
