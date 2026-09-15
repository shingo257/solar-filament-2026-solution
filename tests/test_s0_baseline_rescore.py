from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scratch.run_s0_baseline_rescore import (
    SCIENCE_ENDPOINTS,
    evaluate_variant,
    load_prediction_cache,
    resolve_preprocessed_image_path,
    resolve_yolo_cache_path,
    save_prediction_cache,
    summarize_group,
)


def test_prediction_cache_round_trip(tmp_path):
    first = np.zeros((5, 5), dtype=bool)
    first[1:4, 2] = True
    second = np.eye(5, dtype=bool)
    path = tmp_path / "raw_fold_0" / "observation.npz"

    save_prediction_cache(path, [first, second], "raw", 5)

    loaded = load_prediction_cache(path, "raw", 5)
    assert loaded is not None
    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded[0], first)
    np.testing.assert_array_equal(loaded[1], second)
    assert load_prediction_cache(path, "v35", 5) is None


def test_partial_staging_falls_back_per_observation(tmp_path, monkeypatch):
    staged_oof = tmp_path / "staged_oof"
    canonical_oof = tmp_path / "canonical_oof"
    staged_images = tmp_path / "staged_images"
    canonical_images = tmp_path / "canonical_images"
    staged_oof.mkdir()
    staged_images.mkdir()
    (canonical_oof / "yolo_fold_1").mkdir(parents=True)
    canonical_images.mkdir()
    canonical_cache = canonical_oof / "yolo_fold_1" / "observation.npz"
    canonical_image = canonical_images / "observation.jpeg"
    canonical_cache.touch()
    canonical_image.touch()
    monkeypatch.setattr(
        "scratch.run_s0_baseline_rescore.OOF_CACHE", canonical_oof
    )
    monkeypatch.setattr(
        "scratch.run_s0_baseline_rescore.PREPROC", canonical_images
    )

    assert resolve_yolo_cache_path(staged_oof, 1, "observation.jpeg") == canonical_cache
    assert (
        resolve_preprocessed_image_path(staged_images, "observation.jpeg")
        == canonical_image
    )

    staged_cache = staged_oof / "yolo_fold_1" / "observation.npz"
    staged_cache.parent.mkdir()
    staged_cache.touch()
    staged_image = staged_images / "observation.jpeg"
    staged_image.touch()
    assert resolve_yolo_cache_path(staged_oof, 1, "observation.jpeg") == staged_cache
    assert resolve_preprocessed_image_path(staged_images, "observation.jpeg") == staged_image


def test_iou_equal_to_half_is_not_a_match():
    truth = np.zeros((4, 4), dtype=bool)
    truth[1, 1:3] = True
    prediction = np.zeros((4, 4), dtype=bool)
    prediction[1, 1] = True
    prepared = {
        "annotations": [{"category_id": 1}],
        "truths": [truth],
        "spines": [np.empty((0, 2), dtype=float)],
    }

    row = evaluate_variant(
        "example.png", 0, "labeler-example", prepared, [prediction], 4
    )

    assert row["tp"] == 0
    assert row["fp"] == 1
    assert row["fn"] == 1
    assert row["pq_512"] == 0.0


def test_summary_uses_micro_pq_not_mean_local_pq():
    rows = []
    for file_name, tp, fn, iou_sum, local_pq in (
        ("a.png", 1, 0, 1.0, 1.0),
        ("b.png", 0, 9, 0.0, 0.0),
    ):
        row = {
            "file_name": file_name,
            "tp": tp,
            "fp": 0,
            "fn": fn,
            "iou_sum": iou_sum,
            "pq_512": local_pq,
        }
        row.update({endpoint: 0.0 for endpoint in SCIENCE_ENDPOINTS})
        rows.append(row)

    summary = summarize_group(pd.DataFrame(rows), bootstrap_repeats=50)

    assert summary["pq_512"] == pytest.approx(1.0 / 5.5)
    assert summary["pq_512"] != pytest.approx(0.5)
