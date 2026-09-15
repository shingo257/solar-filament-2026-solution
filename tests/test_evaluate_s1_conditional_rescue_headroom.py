from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scratch.evaluate_s1_conditional_rescue_headroom import (
    ENDPOINTS,
    FEATURE_SETS,
    aggregate_observations,
    audit_feature_set,
    bootstrap_indices,
    build_analysis_table,
    gate_target,
    oracle_masks,
    policy_summary,
    select_development_threshold,
)


def metric_row(
    file_name: str,
    model: str,
    value: float,
    *,
    labeler: str = "labeler-1",
    prediction_count: int = 3,
    prediction_components: int = 4,
    block: str = "development",
) -> dict[str, object]:
    row: dict[str, object] = {
        "file_name": file_name,
        "model": model,
        "labeler": labeler,
        "fold": 0,
        "site_code": "Bh",
        "year": 2015,
        "time_block_role": block,
        "prediction_count": prediction_count,
        "prediction_components": prediction_components,
    }
    row.update({endpoint: value for endpoint in ENDPOINTS})
    return row


def paired_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    return build_analysis_table(aggregate_observations(pd.DataFrame(rows)))


def test_aggregate_observations_averages_endpoints_but_preserves_predictions():
    frame = pd.DataFrame([
        metric_row("a.png", "v35", 0.2, labeler="one", prediction_count=3),
        metric_row("a.png", "v35", 0.6, labeler="two", prediction_count=3),
        metric_row("a.png", "v42", 0.7, labeler="one", prediction_count=5),
        metric_row("a.png", "v42", 0.9, labeler="two", prediction_count=5),
    ])

    result = aggregate_observations(frame)

    assert result.loc[result.model == "v35", "spine_coverage"].item() == pytest.approx(0.4)
    assert result.loc[result.model == "v42", "spine_coverage"].item() == pytest.approx(0.8)
    assert result.loc[result.model == "v35", "prediction_count"].item() == 3
    assert result.loc[result.model == "v42", "prediction_count"].item() == 5


def test_aggregate_observations_rejects_annotator_dependent_prediction_feature():
    frame = pd.DataFrame([
        metric_row("a.png", "v35", 0.2, labeler="one", prediction_count=3),
        metric_row("a.png", "v35", 0.6, labeler="two", prediction_count=4),
    ])

    with pytest.raises(ValueError, match="prediction_count varies"):
        aggregate_observations(frame)


def test_analysis_table_contains_prediction_deltas_and_unique_pairs():
    table = paired_table([
        metric_row("a.png", "v35", 0.2, prediction_count=3),
        metric_row("a.png", "v42", 0.7, prediction_count=5),
    ])

    assert len(table) == 1
    assert table.loc[0, "v35_prediction_count"] == 3
    assert table.loc[0, "v42_prediction_count"] == 5
    assert table.loc[0, "delta_prediction_count"] == 2


def test_feature_sets_exclude_ground_truth_at_inference():
    for specification in FEATURE_SETS.values():
        audit = audit_feature_set(specification)
        assert audit["forbidden_feature_intersection"] == []
        assert audit["ground_truth_features_used_at_inference"] is False
        assert audit["ground_truth_used_as_development_training_target"] is True

    with pytest.raises(ValueError, match="ground-truth-derived"):
        audit_feature_set({"numeric": ("gt_count",), "categorical": ()})


def test_oracle_catalog_safe_adds_catalog_constraints():
    rows = []
    for file_name, primary, precision, merge, count_error in (
        ("safe.png", 0.2, 0.1, -1.0, -1.0),
        ("unsafe.png", 0.3, -0.1, 0.0, -1.0),
        ("loss.png", -0.1, 0.2, -1.0, -1.0),
    ):
        base = metric_row(file_name, "v35", 0.5)
        candidate = metric_row(file_name, "v42", 0.5)
        candidate["spine_coverage"] = 0.5 + primary
        candidate["filament_precision"] = 0.5 + precision
        candidate["merge_count"] = 0.5 + merge
        candidate["count_abs_error"] = 0.5 + count_error
        rows.extend([base, candidate])
    table = paired_table(rows)

    masks = oracle_masks(table)

    by_file_primary = dict(zip(table.file_name, masks["oracle_primary"], strict=True))
    by_file_safe = dict(zip(table.file_name, masks["oracle_catalog_safe"], strict=True))
    assert by_file_primary == {"loss.png": False, "safe.png": True, "unsafe.png": True}
    assert by_file_safe == {"loss.png": False, "safe.png": True, "unsafe.png": False}


def test_threshold_selection_requires_development_and_zero_catalog_harm():
    rows = []
    # High-probability half improves every constrained endpoint; low half harms them.
    for index in range(20):
        base = metric_row(f"{index:02d}.png", "v35", 0.5)
        candidate = metric_row(f"{index:02d}.png", "v42", 0.5)
        if index >= 10:
            candidate["spine_coverage"] = 0.7
            candidate["filament_precision"] = 0.6
            candidate["merge_count"] = 0.4
            candidate["count_abs_error"] = 0.4
        else:
            candidate["spine_coverage"] = 0.3
            candidate["filament_precision"] = 0.3
            candidate["merge_count"] = 0.7
            candidate["count_abs_error"] = 0.7
        rows.extend([base, candidate])
    table = paired_table(rows)
    probabilities = np.linspace(0.01, 0.99, len(table))

    selected, _ = select_development_threshold(table, probabilities)

    assert selected["selection_status"] == "selected_on_development_oof"
    assert selected["selected_fraction"] >= 0.10
    assert selected["endpoint_mean_differences_vs_v35"]["filament_precision"] >= 0
    assert selected["endpoint_mean_differences_vs_v35"]["merge_count"] <= 0
    assert selected["endpoint_mean_differences_vs_v35"]["count_abs_error"] <= 0

    table["time_block_role"] = "locked_evaluation"
    with pytest.raises(ValueError, match="development rows only"):
        select_development_threshold(table, probabilities)


def test_policy_summary_uses_v42_only_where_selected():
    table = paired_table([
        metric_row("a.png", "v35", 0.2),
        metric_row("a.png", "v42", 0.8),
        metric_row("b.png", "v35", 0.4),
        metric_row("b.png", "v42", 0.0),
    ])
    indices = bootstrap_indices(len(table), repeats=20, seed=1)

    result = policy_summary(table, np.array([True, False]), indices)

    assert result["selected_v42"] == 1
    assert result["endpoints"]["spine_coverage"][
        "mean_difference_vs_v35"
    ] == pytest.approx(0.3)


def test_catalog_safe_target_encodes_primary_and_catalog_utility():
    base_safe = metric_row("safe.png", "v35", 0.5)
    candidate_safe = metric_row("safe.png", "v42", 0.5)
    candidate_safe["spine_coverage"] = 0.7
    candidate_safe["filament_precision"] = 0.6
    candidate_safe["merge_count"] = 0.4
    candidate_safe["count_abs_error"] = 0.4
    base_unsafe = metric_row("unsafe.png", "v35", 0.5)
    candidate_unsafe = metric_row("unsafe.png", "v42", 0.5)
    candidate_unsafe["spine_coverage"] = 0.7
    candidate_unsafe["filament_precision"] = 0.4
    table = paired_table([
        base_safe,
        candidate_safe,
        base_unsafe,
        candidate_unsafe,
    ])

    primary = dict(zip(table.file_name, gate_target(table, "primary_benefit"), strict=True))
    safe = dict(zip(table.file_name, gate_target(table, "catalog_safe_benefit"), strict=True))

    assert primary == {"safe.png": True, "unsafe.png": True}
    assert safe == {"safe.png": True, "unsafe.png": False}
    with pytest.raises(ValueError, match="unknown gate target"):
        gate_target(table, "not-a-target")
