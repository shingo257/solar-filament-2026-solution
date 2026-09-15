"""Scientific morphology metrics for solar-filament instance masks.

These metrics complement, but never replace, the official Panoptic Quality
implementation in :mod:`solar_filament.metrics.metrics_official`.  They are intentionally small,
deterministic CPU functions so the S0 scientific audit can be tested without a
GPU or model checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class SpineAudit:
    """Validation result for a flat COCO ``spine`` coordinate sequence."""

    points: np.ndarray
    coordinate_count: int
    point_count: int
    even_length: bool
    finite: bool
    in_bounds_count: int

    @property
    def in_bounds_fraction(self) -> float:
        if self.point_count == 0:
            return 0.0
        return self.in_bounds_count / self.point_count


def audit_spine_coordinates(
    coordinates: object,
    height: int,
    width: int,
) -> SpineAudit:
    """Parse and validate a flat ``[x0, y0, x1, y1, ...]`` spine sequence.

    Odd trailing values are reported and ignored when constructing ``points``.
    This makes the audit total over imperfect annotations while preserving an
    explicit ``even_length`` failure signal.
    """

    try:
        flat = np.asarray(coordinates if coordinates is not None else [], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        flat = np.empty(0, dtype=float)
    coordinate_count = int(flat.size)
    even_length = coordinate_count % 2 == 0
    usable = flat[: coordinate_count - (coordinate_count % 2)]
    points = usable.reshape(-1, 2) if usable.size else np.empty((0, 2), dtype=float)
    finite_rows = np.isfinite(points).all(axis=1) if points.size else np.zeros(0, dtype=bool)
    in_bounds = (
        finite_rows
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    ) if points.size else np.zeros(0, dtype=bool)
    return SpineAudit(
        points=points,
        coordinate_count=coordinate_count,
        point_count=int(len(points)),
        even_length=even_length,
        finite=bool(finite_rows.all()) if points.size else True,
        in_bounds_count=int(in_bounds.sum()),
    )


def spine_coverage(points: np.ndarray, mask: np.ndarray, radius: int = 0) -> float:
    """Fraction of valid spine points covered by a binary mask.

    ``radius`` optionally dilates the mask before sampling, allowing a tolerant
    measurement for sub-pixel annotation or one-pixel rasterisation offsets.
    Out-of-bounds and non-finite points are excluded from the denominator.
    """

    binary = np.asarray(mask, dtype=bool)
    coordinates = np.asarray(points, dtype=float).reshape(-1, 2)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        binary = cv2.dilate(binary.astype(np.uint8), kernel).astype(bool)
    finite_rows = np.isfinite(coordinates).all(axis=1)
    rounded = np.zeros_like(coordinates, dtype=np.int64)
    if finite_rows.any():
        rounded[finite_rows] = np.rint(coordinates[finite_rows]).astype(np.int64)
    xs, ys = rounded[:, 0], rounded[:, 1]
    valid = (
        finite_rows
        & (xs >= 0)
        & (xs < binary.shape[1])
        & (ys >= 0)
        & (ys < binary.shape[0])
    )
    if not valid.any():
        return 0.0
    return float(binary[ys[valid], xs[valid]].mean())


def boundary(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel inner boundary using 8-neighbour erosion."""

    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary)
    eroded = cv2.erode(binary.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return binary & ~eroded


def boundary_f1(truth: np.ndarray, prediction: np.ndarray, tolerance: int = 2) -> float:
    """Symmetric boundary F1 with a pixel-distance tolerance."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    truth_edge = boundary(truth)
    prediction_edge = boundary(prediction)
    if not truth_edge.any() and not prediction_edge.any():
        return 1.0
    if not truth_edge.any() or not prediction_edge.any():
        return 0.0
    truth_points = np.argwhere(truth_edge)
    prediction_points = np.argwhere(prediction_edge)
    precision = float(
        (cKDTree(truth_points).query(prediction_points)[0] <= tolerance).mean()
    )
    recall = float(
        (cKDTree(prediction_points).query(truth_points)[0] <= tolerance).mean()
    )
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton using only OpenCV core operations."""

    image = np.asarray(mask, dtype=bool).astype(np.uint8)
    skeleton = np.zeros_like(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while image.any():
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        skeleton |= image & ~opened
        image = cv2.erode(image, kernel)
    return skeleton.astype(bool)


def centerline_cldice(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Binary clDice measuring topology-preserving overlap."""

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
        return 1.0
    if not truth_skeleton.any() or not prediction_skeleton.any():
        return 0.0
    topology_precision = float(truth_mask[prediction_skeleton].mean())
    topology_sensitivity = float(prediction_mask[truth_skeleton].mean())
    if topology_precision + topology_sensitivity == 0:
        return 0.0
    return 2.0 * topology_precision * topology_sensitivity / (
        topology_precision + topology_sensitivity
    )


def symmetric_centerline_distance(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Mean bidirectional distance between the two morphological skeletons."""

    truth_skeleton = skeletonize(truth)
    prediction_skeleton = skeletonize(prediction)
    if not truth_skeleton.any() and not prediction_skeleton.any():
        return 0.0
    if not truth_skeleton.any() or not prediction_skeleton.any():
        return float("inf")
    to_truth = distance_transform_edt(~truth_skeleton)
    to_prediction = distance_transform_edt(~prediction_skeleton)
    return 0.5 * (
        float(to_truth[prediction_skeleton].mean())
        + float(to_prediction[truth_skeleton].mean())
    )


def topology_counts(mask: np.ndarray) -> dict[str, int]:
    """Connected components, skeleton pixels, endpoints and branch points."""

    binary = np.asarray(mask, dtype=bool)
    if binary.any():
        ys, xs = np.nonzero(binary)
        y0, y1 = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 2, binary.shape[0])
        x0, x1 = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 2, binary.shape[1])
        binary = binary[y0:y1, x0:x1]
    components = int(cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)[0] - 1)
    skeleton = skeletonize(binary)
    neighbours = convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    neighbours = neighbours - skeleton.astype(np.uint8)
    return {
        "components": components,
        "skeleton_pixels": int(skeleton.sum()),
        "endpoints": int(np.logical_and(skeleton, neighbours == 1).sum()),
        "branchpoints": int(np.logical_and(skeleton, neighbours >= 3).sum()),
    }


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


def rasterize_spine(
    points: np.ndarray,
    shape: tuple[int, int],
    thickness: int = 1,
) -> np.ndarray:
    """Rasterise a COCO spine polyline without extrapolating its endpoints."""

    if thickness < 1:
        raise ValueError("thickness must be positive")
    canvas = np.zeros(shape, dtype=np.uint8)
    coordinates = np.asarray(points, dtype=float).reshape(-1, 2)
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if not len(coordinates):
        return canvas.astype(bool)
    rounded = np.rint(coordinates).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, shape[0] - 1)
    if len(rounded) == 1:
        cv2.circle(canvas, tuple(rounded[0]), max(0, thickness // 2), 1, -1)
    else:
        cv2.polylines(canvas, [rounded.reshape(-1, 1, 2)], False, 1, thickness)
    return canvas.astype(bool)


def spine_centerline_metrics(
    points: np.ndarray,
    prediction: np.ndarray,
    tolerance: int = 2,
) -> dict[str, float]:
    """Compare an annotated spine polyline with a predicted mask skeleton."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    prediction_mask = np.asarray(prediction, dtype=bool)
    spine = rasterize_spine(points, prediction_mask.shape)
    prediction_skeleton = skeletonize(prediction_mask)
    if not spine.any() and not prediction_skeleton.any():
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "distance": 0.0}
    if not spine.any() or not prediction_skeleton.any():
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "distance": float("inf"),
        }
    to_spine = distance_transform_edt(~spine)
    to_prediction = distance_transform_edt(~prediction_skeleton)
    precision = float((to_spine[prediction_skeleton] <= tolerance).mean())
    recall = float((to_prediction[spine] <= tolerance).mean())
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    distance = 0.5 * (
        float(to_spine[prediction_skeleton].mean())
        + float(to_prediction[spine].mean())
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "distance": distance,
    }


def mask_geometry(mask: np.ndarray) -> dict[str, float]:
    """Area, perimeter, centroid and major-axis length for one binary mask."""

    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    area = float(binary.sum())
    if area == 0:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "centroid_x": float("nan"),
            "centroid_y": float("nan"),
            "major_axis_length": 0.0,
        }
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    ys, xs = np.nonzero(binary)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    if len(xs) < 2:
        major_axis = 0.0
    else:
        covariance = np.cov(np.stack([xs, ys]), bias=True)
        major_axis = float(
            4.0 * np.sqrt(max(float(np.linalg.eigvalsh(covariance)[-1]), 0.0))
        )
    return {
        "area": area,
        "perimeter": perimeter,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "major_axis_length": major_axis,
    }


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

