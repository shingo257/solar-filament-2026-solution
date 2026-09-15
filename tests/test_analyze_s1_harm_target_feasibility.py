from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scratch" / "analyze_s1_harm_target_feasibility.py"
SPEC = importlib.util.spec_from_file_location("analyze_s1_harm_target_feasibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "file_name": "a.jpeg",
                "fold": 0,
                "candidate_index": 0,
                "time_block_role": "development",
                "site_code": "Ah",
                "year": 2011,
                "final_pq_positive_target": 1,
                "final_catalog_safe_target": 1,
                "delta_spine_coverage": 0.2,
                "delta_filament_precision": 0.0,
                "delta_merge_count": 0.0,
                "delta_count_abs_error": -1.0,
            },
            {
                "file_name": "a.jpeg",
                "fold": 0,
                "candidate_index": 1,
                "time_block_role": "development",
                "site_code": "Ah",
                "year": 2011,
                "final_pq_positive_target": 0,
                "final_catalog_safe_target": 0,
                "delta_spine_coverage": 0.1,
                "delta_filament_precision": -0.1,
                "delta_merge_count": 0.0,
                "delta_count_abs_error": 1.0,
            },
            {
                "file_name": "b.jpeg",
                "fold": 1,
                "candidate_index": 0,
                "time_block_role": "development",
                "site_code": "Bh",
                "year": 2012,
                "final_pq_positive_target": 1,
                "final_catalog_safe_target": 0,
                "delta_spine_coverage": 0.0,
                "delta_filament_precision": 0.0,
                "delta_merge_count": 1.0,
                "delta_count_abs_error": 0.0,
            },
        ]
    )


def test_derive_targets_matches_catalog_contract() -> None:
    labeled = MODULE.derive_targets(candidate_rows())
    assert labeled["final_pq_benefit"].tolist() == [True, False, True]
    assert labeled["count_harm"].tolist() == [False, True, False]
    assert labeled["precision_harm"].tolist() == [False, True, False]
    assert labeled["merge_harm"].tolist() == [False, False, True]
    assert labeled["any_catalog_harm"].tolist() == [False, True, True]
    assert labeled["catalog_safe"].tolist() == [True, False, False]


def test_derive_targets_rejects_locked_rows() -> None:
    rows = candidate_rows()
    rows.loc[0, "time_block_role"] = "locked"
    with pytest.raises(ValueError, match="development-only"):
        MODULE.derive_targets(rows)


def test_derive_targets_rejects_catalog_contract_mismatch() -> None:
    rows = candidate_rows()
    rows.loc[0, "final_catalog_safe_target"] = 0
    with pytest.raises(ValueError, match="catalog-safe contract mismatch"):
        MODULE.derive_targets(rows)


def test_build_observation_table_keeps_clusters() -> None:
    observations = MODULE.build_observation_table(MODULE.derive_targets(candidate_rows()))
    assert observations["file_name"].tolist() == ["a.jpeg", "b.jpeg"]
    assert observations["candidates"].tolist() == [2, 1]
    assert observations["count_harm_events"].tolist() == [1, 0]
    assert observations["merge_harm_events"].tolist() == [0, 1]


def test_wilson_interval_and_zero_event_sizes() -> None:
    low, high = MODULE.wilson_interval(5, 10)
    assert low == pytest.approx(0.236593, abs=1e-6)
    assert high == pytest.approx(0.763407, abs=1e-6)
    assert MODULE.zero_event_validation_size(0.05) == 59
    assert MODULE.zero_event_validation_size(0.02) == 149
    assert MODULE.zero_event_validation_size(0.01) == 299


def test_target_summary_counts_candidate_and_observation_events() -> None:
    labeled = MODULE.derive_targets(candidate_rows())
    observations = MODULE.build_observation_table(labeled)
    summary = MODULE.target_summary(labeled, observations, "precision_harm")
    assert summary["candidate_events"] == 1
    assert summary["observation_events"] == 1
    assert summary["folds_with_events"] == [0]
    assert summary["event_concentration"]["largest_observation_share"] == 1.0


def test_cluster_bootstrap_is_deterministic_and_cluster_based() -> None:
    labeled = MODULE.derive_targets(candidate_rows())
    observations = MODULE.build_observation_table(labeled)
    first = MODULE.cluster_bootstrap_projection(
        observations, "count_harm", 100, iterations=100, seed=7
    )
    second = MODULE.cluster_bootstrap_projection(
        observations, "count_harm", 100, iterations=100, seed=7
    )
    assert first == second
    assert first["point_from_pilot_rate"] == pytest.approx(100 / 3)
    assert np.isfinite(first["cluster_bootstrap95"]).all()


def test_build_report_has_no_model_or_threshold_output() -> None:
    labeled = MODULE.derive_targets(candidate_rows())
    observations = MODULE.build_observation_table(labeled)
    report = MODULE.build_report(
        labeled,
        observations,
        development_candidates=100,
        development_observations=20,
    )
    assert report["target_summaries"]["merge_harm"]["candidate_events"] == 1
    assert report["zero_observed_harm_validation_lower_bounds"][
        "independent_accepted_candidates_required"
    ]["harm_rate_below_0.02"] == 149
    assert "model" not in report
    assert "threshold" not in report
