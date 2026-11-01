"""Pairwise matching and statistical aggregates for filament instance masks.

Split from ``science_metrics.py`` under the 500-line guideline: this module
holds the dense IoU matching (Hungarian), fragmentation counting, paired
geometry errors and the observation-cluster bootstrap. Morphology primitives
and single-mask metrics live in ``science_metrics_morphology.py``.

These metrics complement, but never replace, the official Panoptic Quality
implementation in :mod:`solar_filament.metrics.metrics_official`.  They are intentionally small,
deterministic CPU functions so the S0 scientific audit can be tested without a
GPU or model checkpoint.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from solar_filament.metrics.science_metrics_morphology import (
    mask_geometry,
    skeletonize,
)


def iou_matrix(truths: list[np.ndarray], predictions: list[np.ndarray]) -> np.ndarray:
    """Dense pairwise mask IoU matrix."""

    matrix = np.zeros((len(truths), len(predictions)), dtype=np.float64)
    for row, truth in enumerate(truths):
        truth_mask = np.asarray(truth, dtype=bool)
        for column, prediction in enumerate(predictions):
            prediction_mask = np.asarray(prediction, dtype=bool)
            union = np.logical_or(truth_mask, prediction_mask).sum()
            if union:
                matrix[row, column] = np.logical_and(truth_mask, prediction_mask).sum() / union
    return matrix


def match_instances(
    truths: list[np.ndarray],
    predictions: list[np.ndarray],
    minimum_iou: float = 0.0,
) -> list[tuple[int, int, float]]:
    """One-to-one Hungarian matches, optionally discarding low-IoU pairs."""

    if not truths or not predictions:
        return []
    matrix = iou_matrix(truths, predictions)
    assignment_size = min(matrix.shape)
    eligible = matrix >= minimum_iou
    reward = matrix + eligible * (assignment_size + 1.0)
    rows, columns = linear_sum_assignment(-reward)
    return [
        (int(row), int(column), float(matrix[row, column]))
        for row, column in zip(rows, columns)
        if eligible[row, column]
    ]


def fragmentation_counts(
    truths: list[np.ndarray],
    predictions: list[np.ndarray],
    overlap_fraction: float = 0.10,
) -> tuple[int, int]:
    """Count excess truth-to-prediction splits and prediction-to-truth merges.

    The historical V34 audit used a 10% directional-overlap gate.  Keeping that
    contract makes the S0 comparison reproducible while leaving IoU matching as
    the independent TP/FP/FN definition.
    """

    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be between zero and one")
    truth_masks = [np.asarray(mask, dtype=bool) for mask in truths]
    prediction_masks = [np.asarray(mask, dtype=bool) for mask in predictions]
    splits = sum(
        max(sum(np.logical_and(truth, pred).sum() / max(truth.sum(), 1) >= overlap_fraction
                for pred in prediction_masks) - 1, 0)
        for truth in truth_masks
    )
    merges = sum(
        max(sum(np.logical_and(truth, pred).sum() / max(pred.sum(), 1) >= overlap_fraction
                for truth in truth_masks) - 1, 0)
        for pred in prediction_masks
    )
    return int(splits), int(merges)


def matched_geometry_errors(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Scale-aware morphology errors for a matched truth/prediction pair."""

    truth_geometry = mask_geometry(truth)
    prediction_geometry = mask_geometry(prediction)
    truth_area = max(truth_geometry["area"], 1.0)
    truth_perimeter = max(truth_geometry["perimeter"], 1.0)
    truth_axis = max(truth_geometry["major_axis_length"], 1.0)
    centroid_error = float(
        np.hypot(
            prediction_geometry["centroid_x"] - truth_geometry["centroid_x"],
            prediction_geometry["centroid_y"] - truth_geometry["centroid_y"],
        )
    )
    return {
        "area_relative_error": abs(
            prediction_geometry["area"] - truth_geometry["area"]
        ) / truth_area,
        "perimeter_relative_error": abs(
            prediction_geometry["perimeter"] - truth_geometry["perimeter"]
        ) / truth_perimeter,
        "major_axis_relative_error": abs(
            prediction_geometry["major_axis_length"]
            - truth_geometry["major_axis_length"]
        ) / truth_axis,
        "centroid_error_pixels": centroid_error,
    }


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    repeats: int = 2000,
    confidence: float = 0.95,
    seed: int = 3407,
) -> dict[str, float]:
    """Observation-cluster bootstrap CI for a scalar endpoint mean."""

    data = np.asarray(values, dtype=float).reshape(-1)
    cluster_ids = np.asarray(clusters).reshape(-1)
    if len(data) != len(cluster_ids):
        raise ValueError("values and clusters must have equal length")
    valid = np.isfinite(data)
    data = data[valid]
    cluster_ids = cluster_ids[valid]
    unique = np.unique(cluster_ids)
    if not len(unique):
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    _, inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_sums = np.bincount(inverse, weights=data, minlength=len(unique))
    cluster_counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(repeats, len(unique)))
    samples = cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(data.mean()),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
    }


def paired_centerline_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Compute clDice, centerline distance and topology with one skeletonisation."""

    truth_mask = np.asarray(truth, dtype=bool)
    prediction_mask = np.asarray(prediction, dtype=bool)
    union = truth_mask | prediction_mask
    if union.any():
        ys, xs = np.nonzero(union)
        y0, y1 = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 2, union.shape[0])
        x0, x1 = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 2, union.shape[1])
        truth_mask = truth_mask[y0:y1, x0:x1]
        prediction_mask = prediction_mask[y0:y1, x0:x1]
    truth_skeleton = skeletonize(truth_mask)
    prediction_skeleton = skeletonize(prediction_mask)

    if not truth_skeleton.any() and not prediction_skeleton.any():
        cldice = 1.0
        distance = 0.0
    elif not truth_skeleton.any() or not prediction_skeleton.any():
        cldice = 0.0
        distance = float("inf")
    else:
        topology_precision = float(truth_mask[prediction_skeleton].mean())
        topology_sensitivity = float(prediction_mask[truth_skeleton].mean())
        cldice = (
            2.0 * topology_precision * topology_sensitivity
            / (topology_precision + topology_sensitivity)
            if topology_precision + topology_sensitivity
            else 0.0
        )
        truth_points = np.argwhere(truth_skeleton)
        prediction_points = np.argwhere(prediction_skeleton)
        distance = 0.5 * (
            float(cKDTree(truth_points).query(prediction_points)[0].mean())
            + float(cKDTree(prediction_points).query(truth_points)[0].mean())
        )

    def counts(mask: np.ndarray, skeleton: np.ndarray) -> dict[str, int]:
        components = int(
            cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
        )
        neighbours = convolve(
            skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant"
        ) - skeleton.astype(np.uint8)
        return {
            "components": components,
            "endpoints": int(np.logical_and(skeleton, neighbours == 1).sum()),
            "branchpoints": int(np.logical_and(skeleton, neighbours >= 3).sum()),
        }

    truth_counts = counts(truth_mask, truth_skeleton)
    prediction_counts = counts(prediction_mask, prediction_skeleton)
    return {
        "cldice": float(cldice),
        "centerline_distance": float(distance),
        "truth_components": float(truth_counts["components"]),
        "prediction_components": float(prediction_counts["components"]),
        "truth_endpoints": float(truth_counts["endpoints"]),
        "prediction_endpoints": float(prediction_counts["endpoints"]),
        "truth_branchpoints": float(truth_counts["branchpoints"]),
        "prediction_branchpoints": float(prediction_counts["branchpoints"]),
    }
