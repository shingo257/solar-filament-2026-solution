"""Characterization tests for skeleton-based gap closure."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from solar_filament.pipeline.gap_close import skeleton_gap_close


def _line_mask(y: int, x0: int, x1: int, size: int = 64, thickness: int = 2) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y - thickness // 2 : y + thickness // 2 + 1, x0 : x1 + 1] = True
    return mask


def test_single_component_is_returned_unchanged() -> None:
    mask = _line_mask(y=32, x0=5, x1=58)
    out = skeleton_gap_close(mask)
    assert np.array_equal(out, mask)
    assert out is not mask  # a copy, not the same array


def test_empty_mask_is_returned_unchanged() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    out = skeleton_gap_close(mask)
    assert not out.any()


def test_small_gap_is_bridged_into_one_component() -> None:
    """A filament broken by a tiny 3px gap should become one connected mask."""
    left = _line_mask(y=32, x0=5, x1=27)
    right = _line_mask(y=32, x0=31, x1=58)  # gap of 3px between the two segments
    fragmented = left | right
    _, n_before = ndimage.label(fragmented, structure=np.ones((3, 3)))
    assert n_before == 2, "fixture must start fragmented into 2 pieces"

    bridged = skeleton_gap_close(fragmented, max_gap=8.0, bridge_width=3)
    _, n_after = ndimage.label(bridged, structure=np.ones((3, 3)))
    assert n_after == 1
    # bridging must only add pixels, never remove any of the original mask
    assert (fragmented <= bridged).all()


def test_large_gap_is_not_bridged() -> None:
    """Two fragments far apart should not be merged (they're likely separate objects)."""
    left = _line_mask(y=32, x0=5, x1=15)
    right = _line_mask(y=32, x0=45, x1=58)  # gap of ~29px, far beyond max_gap
    fragmented = left | right

    out = skeleton_gap_close(fragmented, max_gap=8.0, bridge_width=3)
    assert np.array_equal(out, fragmented)
    _, n_after = ndimage.label(out, structure=np.ones((3, 3)))
    assert n_after == 2


def test_gap_exactly_at_threshold_boundary_behaves_monotonically() -> None:
    """A gap that closes at a larger max_gap must also close at any larger value,
    and a gap that fails to close at a small max_gap must also fail at any smaller value."""
    left = _line_mask(y=32, x0=5, x1=27)
    right = _line_mask(y=32, x0=33, x1=58)  # gap of 5px
    fragmented = left | right

    not_bridged = skeleton_gap_close(fragmented, max_gap=2.0, bridge_width=3)
    _, n_small = ndimage.label(not_bridged, structure=np.ones((3, 3)))
    assert n_small == 2

    bridged = skeleton_gap_close(fragmented, max_gap=10.0, bridge_width=3)
    _, n_large = ndimage.label(bridged, structure=np.ones((3, 3)))
    assert n_large == 1


def test_tiny_blob_component_does_not_crash() -> None:
    """A degenerate single-pixel-ish component must not raise (skeleton has 0 neighbors)."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[10, 10] = True  # isolated single pixel
    mask[10, 40:45] = True  # a short separate segment
    out = skeleton_gap_close(mask, max_gap=50.0, bridge_width=3)
    assert out.any()
    assert (mask <= out).all()


def test_three_fragments_bridges_nearest_pairs_only() -> None:
    """With 3 fragments where only two pairs are within range, unrelated far fragments
    stay untouched while near ones merge."""
    a = _line_mask(y=32, x0=5, x1=20)
    b = _line_mask(y=32, x0=24, x1=40)  # 3px gap from a
    c = _line_mask(y=32, x0=60, x1=63, size=64)  # far from both a and b
    fragmented = a | b | c
    _, n_before = ndimage.label(fragmented, structure=np.ones((3, 3)))
    assert n_before == 3

    out = skeleton_gap_close(fragmented, max_gap=8.0, bridge_width=3)
    _, n_after = ndimage.label(out, structure=np.ones((3, 3)))
    assert n_after == 2  # {a, b} merged; c stays separate
