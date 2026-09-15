"""Characterization tests for shared ownership / refine helpers."""

from __future__ import annotations

import numpy as np
import pytest

from solar_filament.pipeline.ownership import apply_ownership, drop_contained_indices, greedy_disjoint
from solar_filament.pipeline.rle import decode_mask, encode_mask
from solar_filament.pipeline.refine import shrink_mask


def _rect(y0: int, y1: int, x0: int, x1: int, size: int = 64) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_greedy_disjoint_removes_overlap() -> None:
    a = _rect(10, 30, 10, 30)
    b = _rect(20, 40, 20, 40)
    kept = greedy_disjoint([a, b], order=[0, 1], min_area=5, canvas=64)
    assert len(kept) == 2
    assert not np.logical_and(kept[0], kept[1]).any()


def test_drop_contained_requires_threshold() -> None:
    outer = _rect(5, 50, 5, 50)
    inner = _rect(15, 25, 15, 25)
    with pytest.raises(TypeError):
        drop_contained_indices([outer, inner])  # type: ignore[call-arg]
    kept = drop_contained_indices([outer, inner], threshold=0.90)
    assert kept == [0]


def test_apply_ownership_v35_rule_drops_contained() -> None:
    outer = _rect(5, 50, 5, 50)
    inner = _rect(15, 25, 15, 25)
    scores = np.array([0.9, 0.8], dtype=np.float64)
    kept = apply_ownership(
        [outer, inner],
        scores,
        rule="drop_contained_then_area_asc",
        containment_threshold=0.90,
        min_area=5,
        canvas=64,
    )
    assert len(kept) == 1
    assert kept[0].sum() == outer.sum()


def test_apply_ownership_threshold_matters() -> None:
    """Partial overlap may drop at 0.80 but not at a higher threshold."""
    a = _rect(10, 40, 10, 40)
    b = _rect(20, 45, 20, 45)
    scores = np.array([0.5, 0.4], dtype=np.float64)
    at_90 = apply_ownership(
        [a, b],
        scores,
        rule="drop_contained_then_area_asc",
        containment_threshold=0.90,
        min_area=5,
        canvas=64,
    )
    at_50 = apply_ownership(
        [a, b],
        scores,
        rule="drop_contained_then_area_asc",
        containment_threshold=0.50,
        min_area=5,
        canvas=64,
    )
    assert len(at_90) >= 1
    assert len(at_50) >= 1
    # Both must be disjoint.
    for masks in (at_90, at_50):
        if len(masks) == 2:
            assert not np.logical_and(masks[0], masks[1]).any()


def test_shrink_mask_intersects_proposal() -> None:
    proposal = _rect(10, 40, 10, 40)
    probability = np.zeros((30, 30), dtype=np.float32)
    probability[5:20, 5:20] = 0.9
    box = (10, 10, 40, 40)
    out = shrink_mask(proposal, probability, box, threshold=0.45)
    assert out.sum() < proposal.sum()
    assert (out <= proposal).all()


def test_rle_roundtrip_via_src_rle() -> None:
    mask = _rect(100, 200, 100, 250, size=2048)
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)
