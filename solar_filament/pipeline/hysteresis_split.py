"""Geodesic hysteresis boundary expansion & parallel-component merge splitting.

Targets two orthogonal residual PQ loss modes identified in 3-fold error
decomposition (RESEARCH_LOG §2.60–2.66) and Organizer Topic 753665:

1. **Near-miss boundary & barb clipping** (817 FN + 588 FP at 0.3 <= IoU <= 0.5):
   Production ``shrink_mask`` clips ``refined & proposal``, cutting off thin
   filament barbs (explicitly included in GT per Topic 753665) and tapered tips
   whenever YOLO's 1/4-res prototype mask under-covers the extremities.
   :func:`hysteresis_expand_mask` keeps the core seed ``(prob >= low_thr) & proposal``
   and allows connected growth up to ``max_expand_px`` outside ``proposal`` ONLY
   where ``prob >= high_thr`` and connected to the core seed, preventing
   neighboring filaments from leaking in.

2. **Multi-filament proposal merges** (``merge_count = 0.217/record``, ~250 cases):
   When YOLO encloses two nearby filaments in a single proposal box and the
   crop refiner cleanly separates them into disconnected high-confidence masks,
   emitting them as a single disconnected RLE scores IoU <= 0.5 against both
   GTs (0 TP, 1 FP, 2 FN). :func:`split_disconnected_merges` separates large
   high-confidence secondary components ONLY when laterally offset from the
   primary component's principal axis (preserving collinear broken bridges of
   a single filament channel).
"""

from __future__ import annotations

import cv2
import numpy as np


def hysteresis_expand_mask(
    proposal: np.ndarray,
    probability: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    low_thr: float = 0.45,
    high_thr: float = 0.70,
    max_expand_px: int = 10,
) -> np.ndarray:
    """Grow ``(prob >= low_thr) & proposal`` along connected ``prob >= high_thr`` barbs/tips."""
    x0, y0, x1, y1 = box
    out = np.zeros_like(proposal, dtype=bool)
    prop_crop = proposal[y0:y1, x0:x1].astype(bool)
    core = (probability >= low_thr) & prop_crop
    if not core.any():
        return out
    if max_expand_px <= 0 or high_thr >= 1.0:
        out[y0:y1, x0:x1] = core
        return out

    ksize = 2 * int(max_expand_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    band = cv2.dilate(prop_crop.astype(np.uint8), kernel) != 0
    cand = core | ((probability >= high_thr) & band)
    num_labels, cc = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    if num_labels <= 2:
        out[y0:y1, x0:x1] = cand
        return out
    keep_ids = np.unique(cc[core])
    keep_ids = keep_ids[keep_ids > 0]
    out[y0:y1, x0:x1] = np.isin(cc, keep_ids)
    return out


def _principal_axis(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroid_xy, unit_principal_axis_xy) of pixel coordinates."""
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    if len(pts) < 3:
        return mean, np.array([1.0, 0.0], dtype=np.float64)
    centered = pts - mean
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    norm = float(np.linalg.norm(axis))
    return mean, (axis / norm if norm > 1e-12 else np.array([1.0, 0.0], dtype=np.float64))


def split_disconnected_merges(
    mask: np.ndarray,
    probability: np.ndarray | None = None,
    box: tuple[int, int, int, int] | None = None,
    *,
    min_secondary_area: int = 160,
    min_secondary_prob: float = 0.62,
    min_perp_offset_px: float = 14.0,
) -> list[np.ndarray]:
    """Split laterally offset high-confidence disconnected components while keeping collinear fragments."""
    ys_all, xs_all = np.nonzero(mask)
    if len(xs_all) < 2 * min_secondary_area:
        return [mask]

    bx0, bx1 = int(xs_all.min()), int(xs_all.max()) + 1
    by0, by1 = int(ys_all.min()), int(ys_all.max()) + 1
    sub = mask[by0:by1, bx0:bx1].astype(np.uint8)
    num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    if num_labels <= 2:
        return [mask]

    comp_ids = list(range(1, num_labels))
    comp_ids.sort(key=lambda cid: int(stats[cid, cv2.CC_STAT_AREA]), reverse=True)
    if int(stats[comp_ids[1], cv2.CC_STAT_AREA]) < min_secondary_area:
        return [mask]

    primary_id = comp_ids[0]
    py, px = np.nonzero(cc == primary_id)
    mu0, axis0 = _principal_axis(px + bx0, py + by0)

    prob_full = None
    if probability is not None and box is not None:
        px0, py0, px1, py1 = box
        prob_full = np.zeros(mask.shape, dtype=np.float32)
        prob_full[py0:py1, px0:px1] = probability

    split_ids: list[int] = []
    for cid in comp_ids[1:]:
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_secondary_area:
            continue
        cy, cx = np.nonzero(cc == cid)
        gx, gy = cx + bx0, cy + by0
        if prob_full is not None:
            mean_p = float(prob_full[gy, gx].mean())
            if mean_p < min_secondary_prob:
                continue
        centroid = np.array([float(gx.mean()), float(gy.mean())], dtype=np.float64)
        delta = centroid - mu0
        perp_dist = abs(delta[0] * axis0[1] - delta[1] * axis0[0])
        if perp_dist >= min_perp_offset_px:
            split_ids.append(cid)

    if not split_ids:
        return [mask]

    out_masks: list[np.ndarray] = []
    remaining = sub.astype(bool).copy()
    for cid in split_ids:
        comp_local = cc == cid
        remaining &= ~comp_local
        inst = np.zeros_like(mask, dtype=bool)
        inst[by0:by1, bx0:bx1] = comp_local
        out_masks.append(inst)

    primary_mask = np.zeros_like(mask, dtype=bool)
    primary_mask[by0:by1, bx0:bx1] = remaining
    out_masks.insert(0, primary_mask)
    return out_masks
