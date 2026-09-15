"""Exclusive pixel ownership for instance masks.

Containment threshold and ownership rule are **required arguments**. Do not
default them: V35 competition uses containment 0.90 while V42/S1 use 0.80.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pycocotools import mask as mask_util

EVAL_SIZE = 2048
DEFAULT_MIN_AREA = 20

OWNERSHIP_RULES = (
    "conf_first",
    "area_asc",
    "drop_contained_then_area_asc",
)


def masks_to_rles(masks: Sequence[np.ndarray]) -> list:
    if not masks:
        return []
    return mask_util.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))


def greedy_disjoint(
    masks: Sequence[np.ndarray],
    order: Sequence[int],
    min_area: int = DEFAULT_MIN_AREA,
    canvas: int = EVAL_SIZE,
) -> list[np.ndarray]:
    """Assign pixels in ``order`` so no two kept masks share a foreground pixel."""

    occupied = np.zeros((canvas, canvas), dtype=bool)
    kept: list[np.ndarray] = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept


def drop_contained_indices(
    masks: Sequence[np.ndarray],
    *,
    threshold: float,
    rles: list | None = None,
) -> list[int]:
    """Drop the smaller mask when it is contained in another at ``threshold``.

    Does not shrink masks; only removes whole candidates. ``threshold`` is
    required so V35 (0.90) and V42/S1 (0.80) cannot silently share a default.
    """

    if len(masks) < 2:
        return list(range(len(masks)))
    encoded = rles if rles is not None else masks_to_rles(masks)
    areas = np.asarray(mask_util.area(encoded), dtype=np.float64)
    self_iou = np.asarray(
        mask_util.iou(encoded, encoded, [0] * len(encoded)),
        dtype=np.float64,
    )
    dropped: set[int] = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped:
                continue
            value = self_iou[i, j]
            if value <= 0:
                continue
            intersection = value * (areas[i] + areas[j]) / (1.0 + value)
            smaller = i if areas[i] <= areas[j] else j
            if intersection / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]


def apply_ownership(
    masks: Sequence[np.ndarray],
    scores: np.ndarray | Sequence[float],
    *,
    rule: str,
    containment_threshold: float,
    min_area: int = DEFAULT_MIN_AREA,
    canvas: int = EVAL_SIZE,
) -> list[np.ndarray]:
    """Make masks pixel-disjoint according to ``rule``.

    Both ``rule`` and ``containment_threshold`` are keyword-only and required.
    """

    if rule not in OWNERSHIP_RULES:
        raise ValueError(f"unknown ownership rule: {rule!r}; expected one of {OWNERSHIP_RULES}")
    if not masks:
        return []
    scores_arr = np.asarray(scores, dtype=np.float64)
    if len(scores_arr) != len(masks):
        raise ValueError("scores must align with masks")
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    if rule == "conf_first":
        return greedy_disjoint(
            masks, list(np.argsort(-scores_arr)), min_area=min_area, canvas=canvas
        )
    if rule == "area_asc":
        return greedy_disjoint(
            masks, list(np.argsort(areas)), min_area=min_area, canvas=canvas
        )
    # drop_contained_then_area_asc
    subset = drop_contained_indices(masks, threshold=containment_threshold)
    return greedy_disjoint(
        masks, sorted(subset, key=lambda i: areas[i]), min_area=min_area, canvas=canvas
    )
