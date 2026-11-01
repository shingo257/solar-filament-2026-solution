import math

import numpy as np

import solar_filament.metrics.science_metrics as science_metrics
import solar_filament.metrics.science_metrics_matching as science_metrics_matching

from solar_filament.metrics.science_metrics import (
    audit_spine_coordinates,
    boundary_f1,
    centerline_cldice,
    cluster_bootstrap_mean,
    fragmentation_counts,
    match_instances,
    mask_geometry,
    matched_geometry_errors,
    paired_centerline_metrics,
    rasterize_spine,
    spine_centerline_metrics,
    spine_coverage,
    symmetric_centerline_distance,
    topology_counts,
)


def _line(size=64):
    mask = np.zeros((size, size), dtype=bool)
    mask[30:33, 8:56] = True
    return mask


def test_spine_audit_reports_odd_and_out_of_bounds_coordinates():
    result = audit_spine_coordinates([5, 6, 70, 10, 4], height=64, width=64)
    assert not result.even_length
    assert result.coordinate_count == 5
    assert result.point_count == 2
    assert result.in_bounds_count == 1
    assert result.in_bounds_fraction == 0.5


def test_spine_coverage_excludes_invalid_points_and_supports_tolerance():
    mask = np.zeros((16, 16), dtype=bool)
    mask[5, 5] = True
    points = np.array([[5, 5], [6, 5], [100, 100]], dtype=float)
    assert spine_coverage(points, mask) == 0.5
    assert spine_coverage(points, mask, radius=1) == 1.0


def test_exact_masks_have_perfect_science_metrics():
    truth = _line()
    assert boundary_f1(truth, truth) == 1.0
    assert centerline_cldice(truth, truth) == 1.0
    assert symmetric_centerline_distance(truth, truth) == 0.0


def test_broken_spine_has_more_components_and_worse_cldice():
    truth = _line()
    broken = truth.copy()
    broken[:, 30:35] = False
    assert topology_counts(truth)["components"] == 1
    assert topology_counts(broken)["components"] == 2
    assert centerline_cldice(truth, broken) < 1.0
    assert symmetric_centerline_distance(truth, broken) > 0.0


def test_empty_centerline_distance_contract():
    empty = np.zeros((16, 16), dtype=bool)
    nonempty = empty.copy()
    nonempty[4:8, 4:8] = True
    assert symmetric_centerline_distance(empty, empty) == 0.0
    assert math.isinf(symmetric_centerline_distance(empty, nonempty))


def test_hungarian_matching_is_one_to_one_and_thresholded():
    first = np.zeros((32, 32), dtype=bool)
    first[2:10, 2:10] = True
    second = np.zeros_like(first)
    second[20:28, 20:28] = True
    shifted = np.zeros_like(first)
    shifted[3:11, 3:11] = True
    matches = match_instances([first, second], [shifted, second], minimum_iou=0.5)
    assert [(row, column) for row, column, _ in matches] == [(0, 0), (1, 1)]
    assert matches[1][2] == 1.0


def test_matching_prioritizes_valid_pairs_before_subthreshold_iou(monkeypatch):
    matrix = np.array(
        [
            [0.51, 0.50],
            [0.50, 0.00],
        ]
    )
    # iou_matrix lives in the _matching impl module after the 500-line split;
    # match_instances resolves it through that module's globals.
    monkeypatch.setattr(science_metrics_matching, "iou_matrix", lambda truths, predictions: matrix)
    dummy = np.zeros((1, 1), dtype=bool)

    matches = match_instances(
        [dummy, dummy],
        [dummy, dummy],
        minimum_iou=float(np.nextafter(0.5, 1.0)),
    )

    assert matches == [(0, 0, 0.51)]


def test_spine_raster_geometry_and_matched_errors():
    points = np.array([[3, 8], [12, 8]], dtype=float)
    spine = rasterize_spine(points, (16, 16))
    assert spine[8, 3]
    assert spine[8, 12]
    assert mask_geometry(spine)["area"] == 10.0
    errors = matched_geometry_errors(spine, spine)
    assert all(value == 0.0 for value in errors.values())


def test_spine_centerline_metrics_use_distance_tolerance():
    points = np.array([[3, 8], [12, 8]], dtype=float)
    near = np.zeros((16, 16), dtype=bool)
    near[9, 3:13] = True
    metrics = spine_centerline_metrics(points, near, tolerance=1)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["distance"] == 1.0

    far = np.zeros((16, 16), dtype=bool)
    far[13, 3:13] = True
    assert spine_centerline_metrics(points, far, tolerance=1)["f1"] == 0.0


def test_cluster_bootstrap_resamples_observations_not_rows():
    values = np.array([0.0, 0.0, 1.0])
    clusters = np.array(["a", "a", "b"])
    result = cluster_bootstrap_mean(values, clusters, repeats=500, seed=1)
    assert result["mean"] == 1.0 / 3.0
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_fragmentation_counts_directional_split_and_merge():
    truth = np.zeros((24, 24), dtype=bool)
    truth[4:20, 4:20] = True
    left = np.zeros_like(truth)
    left[4:20, 4:12] = True
    right = np.zeros_like(truth)
    right[4:20, 12:20] = True
    assert fragmentation_counts([truth], [left, right]) == (1, 0)

    top = np.zeros_like(truth)
    top[2:10, 2:10] = True
    bottom = np.zeros_like(truth)
    bottom[14:22, 14:22] = True
    merged = np.logical_or(top, bottom)
    assert fragmentation_counts([top, bottom], [merged]) == (0, 1)
    assert fragmentation_counts([], []) == (0, 0)


def test_paired_centerline_metrics_match_individual_contracts():
    truth = _line()
    prediction = truth.copy()
    prediction[:, 30:35] = False
    result = paired_centerline_metrics(truth, prediction)
    assert result["cldice"] == centerline_cldice(truth, prediction)
    assert result["centerline_distance"] == symmetric_centerline_distance(truth, prediction)
    assert result["truth_components"] == topology_counts(truth)["components"]
    assert result["prediction_components"] == topology_counts(prediction)["components"]

