"""CPU unit tests for geodesic hysteresis expansion & disconnected merge splitting."""

from __future__ import annotations

import numpy as np

from solar_filament.pipeline.hysteresis_split import (
    hysteresis_expand_mask,
    split_disconnected_merges,
)


def test_hysteresis_recovers_connected_barb_without_neighbor_leak():
    canvas = (128, 128)
    proposal = np.zeros(canvas, dtype=bool)
    # YOLO clipped the filament right end & barb at x=60 (true filament goes to x=68)
    proposal[58:66, 20:60] = True

    box = (10, 40, 110, 90)
    prob = np.full((50, 100), 0.05, dtype=np.float32)
    # Connected filament body + 8px tip/barb extension (x=20..68 in global -> 10..58 in crop)
    prob[18:26, 10:58] = 0.85
    prob[14:18, 45:52] = 0.80  # connected barb protruding 4px above proposal
    # Separate neighboring filament 5px below (y=72..78 in global -> 32..38 in crop), ALSO high prob!
    prob[32:38, 15:55] = 0.92

    out = hysteresis_expand_mask(
        proposal, prob, box, low_thr=0.45, high_thr=0.70, max_expand_px=10
    )
    # Connected tip (x=60..68) and barb (y=54..58) recovered:
    assert out[60, 65]
    assert out[55, 57]
    # Disconnected neighbor within max_expand_px MUST NOT leak into mask:
    assert not out[74, 35]


def test_hysteresis_zero_expand_matches_shrink():
    proposal = np.zeros((64, 64), dtype=bool)
    proposal[20:30, 20:40] = True
    box = (10, 10, 50, 50)
    prob = np.full((40, 40), 0.80, dtype=np.float32)
    out = hysteresis_expand_mask(
        proposal, prob, box, low_thr=0.45, high_thr=0.70, max_expand_px=0
    )
    assert np.array_equal(out, proposal)


def test_split_separates_parallel_filaments_but_keeps_collinear_bridge():
    # Case 1: Two parallel horizontal filaments 24px apart vertically
    parallel = np.zeros((128, 128), dtype=bool)
    parallel[30:38, 20:80] = True  # area = 480, axis = horizontal
    parallel[60:66, 25:75] = True  # area = 300, perp_dist = ~29px
    parts = split_disconnected_merges(
        parallel, min_secondary_area=150, min_perp_offset_px=14.0
    )
    assert len(parts) == 2
    assert sum(int(p.sum()) for p in parts) == int(parallel.sum())
    assert not (parts[0] & parts[1]).any()

    # Case 2: Single horizontal filament broken by a 6px faint gap along the SAME axis
    collinear = np.zeros((128, 128), dtype=bool)
    collinear[50:56, 15:60] = True  # area = 270
    collinear[50:56, 66:105] = True  # area = 234, perp_dist = 0px
    kept = split_disconnected_merges(
        collinear, min_secondary_area=150, min_perp_offset_px=14.0
    )
    assert len(kept) == 1
    assert np.array_equal(kept[0], collinear)
