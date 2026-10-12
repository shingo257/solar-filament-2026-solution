"""v80 contract tests: scorer-eval worker on a synthetic candidate bundle.

No Drive, no models, no GPU. Builds three candidates (one YOLO proposal and a
two-node component tree) plus one annotator record with two GT filaments, and
verifies that ``_evaluate_image`` produces the exact TP/FP/FN counts the arm
definitions imply.
"""

from __future__ import annotations

import numpy as np
import pytest

from solar_filament.pipeline.rle import encode_mask

pytest.importorskip("sklearn", reason="eval script needs scikit-learn")
pytest.importorskip("torch", reason="eval imports refiner module helpers")

from scratch.run_v80_scorer_eval import (  # noqa: E402
    EVAL_SIZE,
    TAU_GRID,
    _evaluate_image,
    context_features,
    rle_dicts,
    teachers_for,
)


def _rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((EVAL_SIZE, EVAL_SIZE), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _rle(mask: np.ndarray) -> dict:
    return {
        "size": [EVAL_SIZE, EVAL_SIZE],
        "counts": encode_mask(mask).encode("utf-8"),
    }


@pytest.fixture(scope="module")
def bundle() -> dict:
    gt1 = _rect(100, 160, 100, 400)   # matched by the YOLO candidate
    gt2 = _rect(300, 380, 300, 700)   # matched by the deep tree node
    yolo = _rect(105, 158, 105, 395)  # conf 0.15 → below prod threshold
    tree_root = _rect(290, 430, 280, 900)   # contains gt2 + extra → IoU < 0.5
    tree_child = _rect(302, 378, 302, 698)  # tight → IoU > 0.5
    return {
        "rle_final": np.array(
            [encode_mask(m) for m in (yolo, tree_root, tree_child)], dtype=object
        ),
        "usable": np.array([True, True, True]),
        "source": np.array([0, 1, 1], dtype=np.int8),
        "conf": np.array([0.15, -1.0, -1.0], dtype=np.float32),
        "tree_offset": 1,
        "n_tree_nodes": 2,
        "tree_children": [[1], []],
        "tree_roots": [0],
        "records": [[_rle(gt1), _rle(gt2)]],
        # yolo û high enough to be taken by any τ in the grid; root vs child
        # forces the antichain DP to pick the child.
        "u_hat": np.array([0.60, 0.10, 0.55]),
        "u_true": np.array([0.80, 0.00, 0.70]),
    }


def test_evaluate_image_arm_counts(bundle: dict) -> None:
    out = _evaluate_image(bundle)
    tau = TAU_GRID[0]
    # prod: conf 0.15 < 0.26 → no predictions → both GT are FN
    assert out["prod"][tau] == (0, 0, 0, 2)
    # yolo_scorer: û=0.6 > τ → predicts gt1 only
    tp_sum, tp, fp, fn = out["yolo_scorer"][tau]
    assert (tp, fp, fn) == (1, 0, 1)
    assert 0.5 < tp_sum < 1.0
    # tree_only: antichain picks the child → predicts gt2 only
    tp_sum, tp, fp, fn = out["tree_only"][tau]
    assert (tp, fp, fn) == (1, 0, 1)
    # hybrid: both predictions survive ownership (disjoint masks)
    tp_sum, tp, fp, fn = out["hybrid"][tau]
    assert (tp, fp, fn) == (2, 0, 0)
    assert tp_sum == pytest.approx(out["yolo_scorer"][tau][0] + out["tree_only"][tau][0])
    # oracle uses u_true instead of u_hat — same picks here
    assert out["hybrid_oracle"][tau][1:] == out["hybrid"][tau][1:]


def test_teachers_for_hits_and_misses(bundle: dict) -> None:
    rles = rle_dicts(bundle["rle_final"])
    u, h = teachers_for(rles, bundle["records"])
    assert h.tolist() == pytest.approx([1.0, 0.0, 1.0])
    assert 0.5 < u[0] < 1.0 and 0.5 < u[2] < 1.0 and u[1] == 0.0


def test_context_features_yolo_support(bundle: dict) -> None:
    rles = rle_dicts(bundle["rle_final"])
    cand = {
        "source": bundle["source"],
        "conf": bundle["conf"],
    }
    ctx = context_features(cand, rles, n_annotators=1)
    assert ctx.shape == (3, 6)
    # the deep tree node sits inside the YOLO candidate's region? gt2 area is
    # separate from gt1 → no yolo overlap > 0.5 → NaN support
    assert np.isnan(ctx[2, 2])
    assert ctx[0, 3] == 3 and ctx[0, 4] == 1 and ctx[0, 5] == 1


def test_teachers_empty(bundle: dict) -> None:
    u, h = teachers_for([], bundle["records"])
    assert len(u) == 0 and len(h) == 0


def test_geometry_features_empty_mask() -> None:
    """refiner が全消しした候補 (usable=False で行は残す) の特徴抽出が落ちない。"""
    from scratch.run_v80_hybrid_candidates import _geometry_features

    empty = np.zeros((EVAL_SIZE, EVAL_SIZE), dtype=bool)
    assert _geometry_features(empty) == (0.0,) * 8
