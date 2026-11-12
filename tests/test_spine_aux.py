"""CPU tests for spine_aux — prereg 2026-11-13 target path."""

from __future__ import annotations

import numpy as np
import pytest

from solar_filament.pipeline.spine_aux import (
    build_spine_target,
    cellize_mask_any,
    dilate_cells,
    flip_points_x,
    padding_exclusion_mask,
    rasterize_polyline_band,
    spine_aux_bce,
    spine_in_mask_fraction,
    transform_points_affine,
)

IMG = 2048
GRID = 160  # P3 at stride 8 for 1280 input


def test_horizontal_line_hits_cells():
    # horizontal spine across the middle row of the image
    pts = np.array([[0.0, 1024.0], [2047.0, 1024.0]])
    band = rasterize_polyline_band(pts, GRID, GRID, IMG, IMG, radius_cells=0)
    row = int(1024.0 * GRID / IMG)
    assert band[row].all()  # every column positive on that row
    assert band.sum() == GRID


def test_no_skipped_cells_on_diagonal():
    pts = np.array([[0.0, 0.0], [2047.0, 2047.0]])
    band = rasterize_polyline_band(pts, GRID, GRID, IMG, IMG, radius_cells=0)
    # diagonal must occupy all cells along the diagonal (supersampling)
    assert all(band[i, i] for i in range(GRID))


def test_radius_one_band_thickness():
    pts = np.array([[0.0, 1024.0], [2047.0, 1024.0]])
    band = rasterize_polyline_band(pts, GRID, GRID, IMG, IMG, radius_cells=1)
    row = int(1024.0 * GRID / IMG)
    for r in (row - 1, row, row + 1):
        assert band[r].all()
    assert not band[max(0, row - 2)].any()


def test_out_of_bounds_points_dropped_not_wrapped():
    pts = np.array([[-500.0, 1024.0], [2600.0, 1024.0]])
    band = rasterize_polyline_band(pts, GRID, GRID, IMG, IMG, radius_cells=0)
    row = int(1024.0 * GRID / IMG)
    # supersampled segment still crosses the whole image row
    assert band[row].all()
    # nothing lands outside the grid (shape itself guarantees this)
    assert band.shape == (GRID, GRID)


def test_dilate_cells_no_edge_wrap():
    m = np.zeros((8, 8), dtype=bool)
    m[0, 0] = True
    d = dilate_cells(m, 1)
    assert d[:2, :2].all() and d.sum() == 4  # no wrap-around growth


def test_flip_consistency():
    rng = np.random.default_rng(0)
    pts = np.sort(rng.uniform(0, IMG, (8, 2)), axis=0)
    band = rasterize_polyline_band(pts, GRID, GRID, IMG, IMG, radius_cells=1)
    flipped = rasterize_polyline_band(
        flip_points_x(pts, IMG), GRID, GRID, IMG, IMG, radius_cells=1)
    # allow a one-cell rounding tolerance at cell boundaries
    assert (band[:, ::-1] & ~dilate_cells(flipped, 1)).sum() == 0
    assert (flipped & ~dilate_cells(band[:, ::-1], 1)).sum() == 0


def test_affine_letterbox_consistency():
    # 2048 -> 1280 uniform scale; rasterize on the scaled image directly
    pts = np.array([[512.0, 512.0], [1536.0, 1536.0]])
    scaled = transform_points_affine(pts, (0.625, 0.625))
    a = rasterize_polyline_band(pts, GRID, GRID, IMG, IMG, radius_cells=0)
    b = rasterize_polyline_band(scaled, GRID, GRID, 1280, 1280, radius_cells=0)
    assert (a & ~dilate_cells(b, 1)).sum() == 0
    assert (b & ~dilate_cells(a, 1)).sum() == 0


def test_padding_exclusion_mask():
    v = padding_exclusion_mask(GRID, GRID, IMG, IMG, pad=(256, 0, 256, 0))
    # 256px bars on a 2048 canvas -> 12.5% of cells on each side invalid
    left_cols = int(np.ceil(256 / (IMG / GRID)))
    assert not v[:, :left_cols].any() and not v[:, GRID - left_cols:].any()
    assert v[:, left_cols:GRID - left_cols].all()


def test_build_target_missing_spine_excluded():
    good = np.array([[0.0, 100.0], [2047.0, 100.0]])
    missing_region = np.zeros((GRID, GRID), dtype=bool)
    missing_region[:, GRID // 2:] = True
    pos, valid = build_spine_target(
        [good, np.zeros((0, 2))], valid_spine=[True, False],
        out_h=GRID, out_w=GRID, img_h=IMG, img_w=IMG,
        radius_cells=0, missing_regions=[missing_region])
    row = int(100.0 * GRID / IMG)
    assert pos[row].all()          # good spine still rasterized
    assert not valid[:, GRID // 2:].any()  # missing region unsupervised
    assert (pos & ~valid).sum() > 0        # pos may exist but is unsupervised


def test_bce_perfect_prediction_zero():
    pos = np.zeros((4, 4), bool)
    pos[1, 1] = True
    valid = np.ones((4, 4), bool)
    logits = np.where(pos, 30.0, -30.0)
    loss, pt, nt, np_, nn_ = spine_aux_bce(logits, pos, valid)
    assert loss < 1e-4 and np_ == 1 and nn_ == 15


def test_bce_respects_valid_mask():
    pos = np.zeros((4, 4), bool)
    valid = np.ones((4, 4), bool)
    valid[0, :] = False
    logits = np.zeros((4, 4))
    _, _, _, np_, nn_ = spine_aux_bce(logits, pos, valid)
    assert np_ == 0 and nn_ == 12  # row 0 unsupervised


def test_bce_symmetric_terms():
    pos = np.zeros((2, 2), bool)
    pos[0, 0] = True
    valid = np.ones((2, 2), bool)
    logits = np.zeros((2, 2))
    loss, pt, nt, _, _ = spine_aux_bce(logits, pos, valid)
    assert pt == pytest.approx(nt) == pytest.approx(np.log(2.0))
    assert loss == pytest.approx(2 * np.log(2.0))


def test_spine_inside_union_mask():
    # GT union band rows 40..120; spine drawn inside -> fraction 1.0
    mask = np.zeros((GRID, GRID), bool)
    mask[40:120] = True
    spine = np.array([[512.0, 800.0], [1536.0, 1100.0]])
    band = rasterize_polyline_band(spine, GRID, GRID, IMG, IMG, radius_cells=0)
    assert spine_in_mask_fraction(band, mask, dilate=1) == 1.0


def test_spine_outside_mask_detected():
    mask = np.zeros((GRID, GRID), bool)
    mask[:40] = True
    spine = np.array([[0.0, 1800.0], [2047.0, 1800.0]])  # near bottom, far from mask
    band = rasterize_polyline_band(spine, GRID, GRID, IMG, IMG, radius_cells=0)
    assert spine_in_mask_fraction(band, mask, dilate=1) == 0.0


def test_cellize_mask_any_noninteger_ratio():
    # 2048 -> 160 is 12.8 px/cell; a 2-px thin line must survive
    m = np.zeros((IMG, IMG), bool)
    m[:, 1024:1026] = True  # vertical thin line
    g = cellize_mask_any(m, GRID, GRID)
    col = int(1024 * GRID / IMG)
    assert g[:, col].all() and g.sum() == GRID
    # single pixel marks exactly one cell
    m2 = np.zeros((IMG, IMG), bool)
    m2[1000, 1000] = True
    g2 = cellize_mask_any(m2, GRID, GRID)
    assert g2.sum() == 1


def test_cellize_empty_mask():
    assert not cellize_mask_any(np.zeros((IMG, IMG), bool), GRID, GRID).any()


def test_empty_spine_target():
    pos, valid = build_spine_target([], out_h=GRID, out_w=GRID, img_h=IMG, img_w=IMG)
    assert not pos.any() and valid.all()
    assert spine_in_mask_fraction(pos, np.zeros_like(pos)) == 1.0
