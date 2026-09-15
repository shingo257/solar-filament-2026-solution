"""Refiner post-process variants for MediSeg crop probabilities.

Competition V35 uses ``shrink`` only (``refined & proposal``). Guided variants
remain available for historical ablations but must not be used for submission
without a new all-fold gate.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def shrink_mask(
    proposal: np.ndarray,
    probability: np.ndarray,
    box: tuple[int, int, int, int],
    threshold: float,
) -> np.ndarray:
    """Keep only proposal pixels whose refined probability meets ``threshold``."""

    x0, y0, x1, y1 = box
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    return refined & proposal


def build_variant(
    proposal: np.ndarray,
    box: tuple[int, int, int, int],
    probability: np.ndarray,
    method: str,
    threshold: float,
) -> np.ndarray:
    """Apply shrink or guided refinements. Prefer :func:`shrink_mask` for V35."""

    x0, y0, x1, y1 = box
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    if method == "shrink":
        return refined & proposal
    kernel = 3 if method == "guided3" else 7
    support = cv2.dilate(
        proposal.astype(np.uint8),
        np.ones((kernel, kernel), np.uint8),
    ).astype(bool)
    return refined & support


def refine_proposal_shrink(
    proposal: np.ndarray,
    probability: np.ndarray,
    box: tuple[int, int, int, int],
    threshold: float,
) -> np.ndarray:
    """Alias used by submission builders; identical to :func:`shrink_mask`."""

    return shrink_mask(proposal, probability, box, threshold)


def apply_shrink_batch(
    proposals: Sequence[np.ndarray],
    boxes: Sequence[tuple[int, int, int, int]],
    probabilities: Sequence[np.ndarray],
    threshold: float,
) -> list[np.ndarray]:
    return [
        shrink_mask(proposal, probability, box, threshold)
        for proposal, box, probability in zip(proposals, boxes, probabilities, strict=True)
    ]
