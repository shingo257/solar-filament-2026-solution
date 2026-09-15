from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scratch.evaluate_s1_prediction_geometry_gate import (
    CACHE_VERSION,
    cross_model_features,
    load_cached_masks,
    mask_perimeter_4,
    mask_set_features,
    mask_shape_features,
)


def test_mask_perimeter_and_bbox_fill_for_rectangle():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 1:5] = True

    area, bbox_fill, log_elongation, log_thinness = mask_shape_features(mask)

    assert area == 12
    assert mask_perimeter_4(mask) == 14
    assert bbox_fill == 1.0
    assert log_elongation > 0
    assert log_thinness > 0


def test_mask_set_features_reports_overlap_and_union():
    first = np.zeros((4, 4), dtype=bool)
    second = np.zeros((4, 4), dtype=bool)
    first[0:2, 0:2] = True
    second[1:3, 1:3] = True

    result = mask_set_features([first, second], size=4)

    assert result["sum_area_fraction"] == pytest.approx(8 / 16)
    assert result["union_area_fraction"] == pytest.approx(7 / 16)
    assert result["overlap_area_fraction"] == pytest.approx(1 / 16)
    assert result["mean_area_fraction"] == pytest.approx(4 / 16)


def test_empty_mask_set_is_finite_and_zero():
    result = mask_set_features([], size=4)

    assert all(value == 0.0 for value in result.values())


def test_cross_model_features_measure_added_removed_and_agreement():
    v35 = np.zeros((4, 4), dtype=bool)
    v42 = np.zeros((4, 4), dtype=bool)
    v35[0:2, 0:2] = True
    v42[1:3, 1:3] = True

    result = cross_model_features([v35], [v42], size=4)

    assert result["union_iou"] == pytest.approx(1 / 7)
    assert result["v42_added_area_fraction"] == pytest.approx(3 / 16)
    assert result["v42_removed_area_fraction"] == pytest.approx(3 / 16)
    assert result["mean_v42_best_iou"] == pytest.approx(1 / 7)
    assert result["mean_v35_best_iou"] == pytest.approx(1 / 7)
    assert result["v42_unmatched_fraction_iou_lt_0_30"] == 1.0


def test_identical_empty_predictions_have_unit_union_iou():
    result = cross_model_features([], [], size=4)

    assert result["union_iou"] == 1.0
    assert result["mean_v42_best_iou"] == 0.0
    assert result["v42_unmatched_fraction_iou_lt_0_30"] == 0.0


def test_cached_masks_round_trip_and_identity_checks(tmp_path: Path):
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 3:7] = True
    path = tmp_path / "cache.npz"
    np.savez_compressed(
        path,
        packed=np.packbits(np.stack([mask]).astype(np.uint8), axis=-1),
        size=np.asarray(8, dtype=np.int32),
        model=np.asarray("v35"),
        version=np.asarray(CACHE_VERSION),
    )

    result = load_cached_masks(path, "v35", size=8)

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], mask)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_cached_masks(path, "v42", size=8)
