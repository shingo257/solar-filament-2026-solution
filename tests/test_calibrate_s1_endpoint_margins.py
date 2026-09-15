from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scratch.calibrate_s1_endpoint_margins import (
    ENDPOINTS,
    annotator_repeatability,
    bootstrap_mean_ci,
    descriptive,
    minimum_detectable_effect,
    normal_power,
    observation_metrics,
    operating_characteristics,
    paired_difference,
    required_sample_size,
)


def metric_row(file_name, model, value, labeler="labeler-1", block="development"):
    row = {
        "file_name": file_name,
        "model": model,
        "labeler": labeler,
        "fold": 0,
        "site_code": "Bh",
        "year": 2015,
        "time_block_role": block,
    }
    row.update({endpoint: value for endpoint in ENDPOINTS})
    return row


def test_descriptive_uses_sample_sd_and_reports_quantiles():
    result = descriptive(np.array([1.0, 2.0, 3.0]))

    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["sd"] == 1.0
    assert result["q10"] == pytest.approx(1.2)
    assert result["q90"] == pytest.approx(2.8)


def test_bootstrap_constant_has_exact_interval():
    result = bootstrap_mean_ci(np.array([0.25, 0.25, 0.25]), repeats=20)

    assert result == {"estimate": 0.25, "ci_low": 0.25, "ci_high": 0.25}


def test_observation_metrics_averages_annotators_before_comparison():
    frame = pd.DataFrame([
        metric_row("a.png", "v35", 0.2, "labeler-1"),
        metric_row("a.png", "v35", 0.6, "labeler-2"),
        metric_row("a.png", "v42", 0.7, "labeler-1"),
        metric_row("a.png", "v42", 0.9, "labeler-2"),
    ])

    result = observation_metrics(frame)

    assert len(result) == 2
    assert result.loc[result.model == "v35", "spine_coverage"].item() == pytest.approx(0.4)
    assert result.loc[result.model == "v42", "spine_coverage"].item() == pytest.approx(0.8)


def test_annotator_repeatability_uses_within_observation_pairs():
    frame = pd.DataFrame([
        metric_row("a.png", "v35", 0.2, "labeler-1"),
        metric_row("a.png", "v35", 0.6, "labeler-2"),
        metric_row("b.png", "v35", 0.5, "labeler-1"),
    ])

    result = annotator_repeatability(frame, "spine_coverage")

    assert result["multi_annotator_observations"] == 1
    assert result["pairwise_comparisons"] == 1
    assert result["pairwise_abs_difference"]["mean"] == pytest.approx(0.4)


def test_paired_difference_is_comparator_minus_v35():
    frame = pd.DataFrame([
        metric_row("a.png", "v35", 0.4),
        metric_row("a.png", "v42", 0.7),
        metric_row("b.png", "v35", 0.6, block="locked_evaluation"),
        metric_row("b.png", "v42", 0.5, block="locked_evaluation"),
    ])
    observations = observation_metrics(frame)

    all_values = paired_difference(observations, "spine_coverage", "v42")
    locked = paired_difference(
        observations, "spine_coverage", "v42", "locked_evaluation"
    )

    np.testing.assert_allclose(np.sort(all_values), [-0.1, 0.3])
    np.testing.assert_allclose(locked, [-0.1])


def test_power_increases_with_effect_and_mde_decreases_with_sample_size():
    assert normal_power(0.02, 0.1, 160) > normal_power(0.01, 0.1, 160)
    assert minimum_detectable_effect(0.1, 160, 0.8) < minimum_detectable_effect(
        0.1, 80, 0.8
    )


def test_operating_characteristics_records_beneficial_direction():
    higher = operating_characteristics(0.1, 160, "higher_is_better")
    lower = operating_characteristics(0.1, 160, "lower_is_better")

    assert higher["beneficial_paired_difference_direction"] == "positive"
    assert lower["beneficial_paired_difference_direction"] == "negative"
    assert (
        higher["mde_magnitude_beyond_registered_boundary"]["power_0.80"]
        == lower["mde_magnitude_beyond_registered_boundary"]["power_0.80"]
    )


def test_required_sample_size_decreases_for_larger_effect():
    small = required_sample_size(0.01, 0.1, 0.8)
    large = required_sample_size(0.02, 0.1, 0.8)

    assert small > large
    assert operating_characteristics(0.1, 160, "higher_is_better")[
        "required_sample_size_if_registered_margin_zero"
    ]["0.010"]["power_0.80"] == small
