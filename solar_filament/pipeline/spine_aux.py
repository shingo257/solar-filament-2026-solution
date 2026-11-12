"""Spine auxiliary-teacher target construction (train-time only).

Pre-registered design (SPINE_AUX_TEACHER_PREREG_2026-11-13.md): a small
head predicts a spine heatmap from shared P3 features; the target is the
annotation spine polyline rasterized into a one-cell-radius positive
band on the P3 grid. Positives/negatives get separately-averaged BCE
over valid cells only (missing-spine regions and padding excluded).
Inference uses only the H-alpha image; the aux head is removed.

All functions are pure-numpy so the whole target path is CPU-testable.
"""

from __future__ import annotations

import numpy as np


def _cell_coords(points: np.ndarray, img_h: int, img_w: int,
                 out_h: int, out_w: int) -> np.ndarray:
    """Image-space (x,y) points -> integer (cy,cx) cell indices, in-bounds only."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    keep = (pts[:, 0] >= 0) & (pts[:, 0] < img_w) & (pts[:, 1] >= 0) & (pts[:, 1] < img_h)
    pts = pts[keep]
    cy = np.clip((pts[:, 1] * out_h / img_h).astype(np.int64), 0, out_h - 1)
    cx = np.clip((pts[:, 0] * out_w / img_w).astype(np.int64), 0, out_w - 1)
    return np.stack([cy, cx], axis=1)


def rasterize_polyline_band(points: np.ndarray, out_h: int, out_w: int,
                            img_h: int, img_w: int,
                            radius_cells: int = 1) -> np.ndarray:
    """Rasterize a spine polyline into a positive band on the feature grid.

    Segments are supersampled at <= half-cell spacing so no cell on the
    path is skipped; each stamped cell is then dilated by ``radius_cells``.
    Returns a bool array of shape (out_h, out_w).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    band = np.zeros((out_h, out_w), dtype=bool)
    if len(pts) == 0:
        return band
    cell_h = img_h / out_h
    cell_w = img_w / out_w
    step = 0.5 * min(cell_h, cell_w)

    samples = [pts[0]]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        dist = float(np.hypot(*(p1 - p0)))
        n = max(1, int(np.ceil(dist / step)))
        samples.extend(p0 + (p1 - p0) * t for t in np.linspace(0.0, 1.0, n + 1)[1:])
    cells = _cell_coords(np.asarray(samples), img_h, img_w, out_h, out_w)
    if len(cells):
        band[cells[:, 0], cells[:, 1]] = True
    return dilate_cells(band, radius_cells) if radius_cells > 0 else band


def dilate_cells(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Chebyshev dilation of a bool grid by ``radius_cells`` cells (no wrap)."""
    return _dilate_cells_no_wrap(mask, radius_cells) if radius_cells > 0 else np.asarray(mask, bool)


def _dilate_cells_no_wrap(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, constant_values=False)
        grown = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[dy:dy + h, dx:dx + w]
        out = grown
    return out


def transform_points_affine(points: np.ndarray, scale: tuple[float, float],
                            offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Apply scale+translate to (x,y) points (letterbox / mosaic placement)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    pts[:, 0] = pts[:, 0] * scale[0] + offset[0]
    pts[:, 1] = pts[:, 1] * scale[1] + offset[1]
    return pts


def flip_points_x(points: np.ndarray, img_w: int) -> np.ndarray:
    """Horizontal flip of (x,y) points, matching ``img[:, ::-1]``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    pts[:, 0] = img_w - 1.0 - pts[:, 0]
    return pts


def padding_exclusion_mask(out_h: int, out_w: int, img_h: int, img_w: int,
                           pad: tuple[int, int, int, int] = (0, 0, 0, 0)) -> np.ndarray:
    """Bool (out_h,out_w) validity mask; False over cells inside padding.

    ``pad`` = (left, top, right, bottom) pixels of the *canvas* (post-
    letterbox image size img_w×img_h) that carry no real image content.
    A cell is invalid when its centre falls in a padded band.
    """
    left, top, right, bottom = pad
    valid = np.ones((out_h, out_w), dtype=bool)
    cy = (np.arange(out_h) + 0.5) * img_h / out_h
    cx = (np.arange(out_w) + 0.5) * img_w / out_w
    valid &= (cy[:, None] >= top) & (cy[:, None] < img_h - bottom)
    valid &= (cx[None, :] >= left) & (cx[None, :] < img_w - right)
    return valid


def build_spine_target(spines: list[np.ndarray],
                       valid_spine: list[bool] | None = None,
                       out_h: int = 160, out_w: int = 160,
                       img_h: int = 2048, img_w: int = 2048,
                       radius_cells: int = 1,
                       missing_regions: list[np.ndarray] | None = None,
                       valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build (positive_band, valid_mask) for one image.

    ``spines``: list of (N,2) image-space polylines, one per annotation.
    ``valid_spine``: per-annotation flag; entries with False contribute no
    positive cells *and* (via ``missing_regions``) may carve invalid cells.
    ``missing_regions``: bool (out_h,out_w) masks marking cells that must
    be excluded from supervision (e.g. the GT region of a missing-spine
    annotation). ``valid_mask``: precomputed validity (e.g. padding).
    """
    pos = np.zeros((out_h, out_w), dtype=bool)
    if valid_spine is None:
        valid_spine = [True] * len(spines)
    for spine, ok in zip(spines, valid_spine):
        if ok and len(np.asarray(spine).reshape(-1)) >= 4:
            pos |= rasterize_polyline_band(spine, out_h, out_w, img_h, img_w,
                                           radius_cells=radius_cells)
    if valid_mask is None:
        valid_mask = np.ones((out_h, out_w), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool).copy()
    for region in missing_regions or []:
        valid_mask &= ~np.asarray(region, dtype=bool)
    return pos, valid_mask


def spine_aux_bce(logits: np.ndarray, pos: np.ndarray, valid: np.ndarray,
                  eps: float = 1e-7) -> tuple[float, float, float, int, int]:
    """Positive/negative separately-averaged BCE over valid cells.

    Returns (loss, pos_term, neg_term, n_pos, n_neg). Each term is the
    mean over its own class so the aux loss is decoupled from band
    sparsity; an absent class contributes 0.
    """
    logits = np.asarray(logits, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-logits))
    pos = np.asarray(pos, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    pos_sel = pos & valid
    neg_sel = (~pos) & valid
    pos_term = float(-np.log(np.clip(p[pos_sel], eps, 1.0)).mean()) if pos_sel.any() else 0.0
    neg_term = float(-np.log(np.clip(1.0 - p[neg_sel], eps, 1.0)).mean()) if neg_sel.any() else 0.0
    return pos_term + neg_term, pos_term, neg_term, int(pos_sel.sum()), int(neg_sel.sum())


def cellize_mask_any(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Downsample a bool image mask to a grid: cell positive iff any pixel.

    Uses an integral image so non-integer image/cell ratios (e.g. 2048 ->
    160 = 12.8 px/cell) are handled exactly; nearest-neighbour resize
    would erase thin structures and must not be used for this check.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    er = np.floor((np.arange(out_h) + 1) * h / out_h).astype(np.int64)
    ec = np.floor((np.arange(out_w) + 1) * w / out_w).astype(np.int64)
    sr = np.floor(np.arange(out_h) * h / out_h).astype(np.int64)
    sc = np.floor(np.arange(out_w) * w / out_w).astype(np.int64)
    sums = (ii[np.ix_(er, ec)] - ii[np.ix_(sr, ec)]
            - ii[np.ix_(er, sc)] + ii[np.ix_(sr, sc)])
    return sums > 0


def spine_in_mask_fraction(spine_band: np.ndarray, mask_band: np.ndarray,
                           dilate: int = 1) -> float:
    """Fraction of positive spine cells inside the (dilated) GT mask band.

    Pre-training consistency check: rasterize the GT union mask and the
    spine under the *same* geometric transform, then verify the spine
    band still lands inside the mask (audit-level expectation ~1.0).
    Returns 1.0 when there are no spine cells (nothing to contradict).
    """
    spine = np.asarray(spine_band, dtype=bool)
    mask = np.asarray(mask_band, dtype=bool)
    if dilate > 0:
        mask = _dilate_cells_no_wrap(mask, dilate)
    n = int(spine.sum())
    return float((spine & mask).sum() / n) if n else 1.0
