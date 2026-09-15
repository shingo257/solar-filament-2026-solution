import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from solar_filament.metrics.science_metrics import boundary, boundary_f1, paired_centerline_metrics, skeletonize


def reference_boundary_f1(truth, prediction, tolerance=2):
    truth_edge = boundary(truth)
    prediction_edge = boundary(prediction)
    truth_distance = distance_transform_edt(~truth_edge)
    prediction_distance = distance_transform_edt(~prediction_edge)
    precision = float((truth_distance[prediction_edge] <= tolerance).mean())
    recall = float((prediction_distance[truth_edge] <= tolerance).mean())
    return 2.0 * precision * recall / (precision + recall)


def reference_centerline_distance(truth, prediction):
    truth_skeleton = skeletonize(truth)
    prediction_skeleton = skeletonize(prediction)
    to_truth = distance_transform_edt(~truth_skeleton)
    to_prediction = distance_transform_edt(~prediction_skeleton)
    return 0.5 * (
        float(to_truth[prediction_skeleton].mean())
        + float(to_prediction[truth_skeleton].mean())
    )


def paired_masks():
    truth = np.zeros((128, 128), dtype=np.uint8)
    prediction = np.zeros_like(truth)
    cv2.line(truth, (12, 20), (102, 92), 1, 7)
    cv2.line(prediction, (14, 19), (104, 91), 1, 5)
    cv2.circle(truth, (80, 42), 10, 1, -1)
    cv2.circle(prediction, (82, 43), 8, 1, -1)
    return truth.astype(bool), prediction.astype(bool)


def test_kdtree_boundary_matches_full_distance_transform():
    truth, prediction = paired_masks()
    assert boundary_f1(truth, prediction, tolerance=2) == reference_boundary_f1(
        truth, prediction, tolerance=2
    )


def test_cropped_kdtree_centerline_matches_full_distance_transform():
    truth, prediction = paired_masks()
    measured = paired_centerline_metrics(truth, prediction)["centerline_distance"]
    assert np.isclose(measured, reference_centerline_distance(truth, prediction))
