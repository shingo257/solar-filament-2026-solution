"""v99 contract tests: dedup acceptance rule and paired arm scoring on synthetic masks.

No Drive, no models. Verifies the halfcross split, cal-desc greedy dedup with conf
tie-break, and that ``evaluate_image`` scores prod / dedup arms with the counts the
§2.75 arm definitions imply.
"""

from __future__ import annotations

import numpy as np
import pytest

from solar_filament.pipeline.rle import encode_mask

pytest.importorskip("sklearn", reason="eval script needs scikit-learn")
pytest.importorskip("torch", reason="eval imports refiner module helpers")

from scratch.run_v80_scorer_eval_impl import EVAL_SIZE, rle_dicts  # noqa: E402
from scratch.run_v99_dedup_replication import (  # noqa: E402
    crossfit_halves,
    dedup_accept,
    evaluate_image,
)


def _rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((EVAL_SIZE, EVAL_SIZE), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _text(mask: np.ndarray) -> str:
    return encode_mask(mask)


def _gt(mask: np.ndarray) -> dict:
    return rle_dicts([_text(mask)])[0]


A = _rect(100, 200, 100, 200)
A_SHIFT = _rect(100, 200, 140, 240)  # IoU with A = 60/140 ≈ 0.43 > 0.30
B = _rect(600, 700, 600, 700)


def test_crossfit_halves_is_sorted_parity():
    assert crossfit_halves(["c", "a", "d", "b"]) == (["a", "c"], ["b", "d"])


def test_dedup_accept_cal_desc_with_conf_tiebreak():
    rles = rle_dicts([_text(A), _text(A_SHIFT), _text(B)])
    cal = np.array([0.30, 0.30, 0.10])
    conf = np.array([0.40, 0.90, 0.10])
    # equal cal → higher conf (A_SHIFT) wins; A is then suppressed; B is disjoint
    assert dedup_accept(rles, cal, conf, 0.30) == [1, 2]
    # a looser dedup threshold keeps both overlapping proposals
    assert sorted(dedup_accept(rles, cal, conf, 0.50)) == [0, 1, 2]


def test_evaluate_image_prod_vs_dedup_counts():
    bundle = {
        "rle_final": [_text(A), _text(A_SHIFT), _text(B)],
        "rle_prop": [_text(A), _text(A_SHIFT), _text(B)],
        "usable": np.array([True, True, True]),
        "conf": np.array([0.90, 0.50, 0.15]),
        "records": [[_gt(A), _gt(B)]],
        "cal": np.array([0.60, 0.40, 0.30]),
        "lam": 0.22,
    }
    out = evaluate_image(bundle)
    # prod: A + clipped A_SHIFT fragment (FP), B dropped by conf<0.26 (FN)
    assert out["prod"][1:] == (1, 1, 1)
    # dedup0.30: A_SHIFT suppressed, low-conf B rescued by cal>λ
    tp_sum, tp, fp, fn = out["dedup_0.30"]
    assert (tp, fp, fn) == (2, 0, 0)
    assert tp_sum == pytest.approx(2.0)
    # no dedup: same cal>λ set as rescue but the duplicate fragment survives
    assert out["nodedup"][1:] == (2, 1, 0)


def test_evaluate_image_respects_lambda_and_usable():
    bundle = {
        "rle_final": [_text(A), _text(B)],
        "rle_prop": [_text(A), _text(B)],
        "usable": np.array([True, False]),
        "conf": np.array([0.90, 0.95]),
        "records": [[_gt(A), _gt(B)]],
        "cal": np.array([0.10, 0.90]),
        "lam": 0.22,
    }
    out = evaluate_image(bundle)
    # A fails cal>λ, B is unusable (refiner killed it) → nothing emitted
    assert out["dedup_0.30"][1:] == (0, 0, 2)
