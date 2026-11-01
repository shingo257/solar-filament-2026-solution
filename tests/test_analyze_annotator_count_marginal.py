"""Tests for the corrected marginal-value rule and ceiling (§2.66).

The rule k·s > PQ·R/2 is checked against a direct PQ recomputation: adding
one prediction that matches k of R records at IoU s converts k FNs to TPs
and adds R−k FPs.
"""

import pandas as pd
import pytest

from scratch.analyze_annotator_count_marginal import (
    ceiling_pq,
    marginal_gain_positive,
)


def _pq_after(tp_iou, tp, fp, fn, k, s, r):
    num = tp_iou + k * s
    den = (tp + k) + 0.5 * (fp + r - k) + 0.5 * (fn - k)
    return num / den


@pytest.mark.parametrize("k,r", [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)])
def test_rule_matches_direct_recompute(k, r):
    tp_iou, tp, fp, fn, s = 44.5, 65.0, 30.0, 40.0, 0.634
    base = tp_iou / (tp + 0.5 * fp + 0.5 * fn)
    after = _pq_after(tp_iou, tp, fp, fn, k, s, r)
    assert (after > base) == marginal_gain_positive(k, s, r, base)


def test_single_annotator_object_is_positive_on_r1_and_r2():
    # §2.54 claimed singleton predictions are negative; the corrected rule
    # says they are positive whenever R < 2s/PQ (≈2.85 at s=.634, PQ=.445).
    assert marginal_gain_positive(1, 0.634, 1, 0.445)
    assert marginal_gain_positive(1, 0.634, 2, 0.445)
    assert not marginal_gain_positive(1, 0.634, 3, 0.445)


def test_ceiling_predict_all_counts():
    # two clusters: (k=1,R=1) and (k=1,R=3); n_gt = 2.
    clusters = pd.DataFrame({"k": [1, 1], "R": [1, 3]})
    pq, kept = ceiling_pq(clusters, n_gt=2, s=0.6, select=False)
    # TP=2, FP=2, FN=0 -> 1.2 / 3
    assert pq == pytest.approx(1.2 / 3.0)
    assert kept == 1.0


def test_ceiling_selection_drops_negative_cluster():
    clusters = pd.DataFrame({"k": [1, 1], "R": [1, 3]})
    pq, kept = ceiling_pq(clusters, n_gt=2, s=0.6, select=True)
    # keep only R=1: TP=1, FP=0, FN=1 -> 0.6 / 1.5 = 0.4 (= predict-all 0.4,
    # but the R=3 singleton is break-even-negative at 0.6 < 0.4*3/2).
    assert pq == pytest.approx(0.4)
    assert kept == 0.5
