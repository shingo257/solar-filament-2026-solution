from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scratch.run_s1_proposal_marginal_utility_pilot import (
    binary_diagnostics,
    final_catalog_safe_target,
    light_variant_metrics,
    linearized_pq_utility,
    safe_spearman,
    select_fixed_development_sample,
    stable_selection_hash,
)


def test_fixed_selection_is_seeded_balanced_and_development_only():
    rows = []
    candidates = []
    for fold in (0, 1, 2):
        for index in range(5):
            name = f"fold{fold}-{index}.jpeg"
            rows.append({
                "file_name": name,
                "fold": fold,
                "time_block_role": "development",
                "site_code": "Bh",
                "year": 2015,
            })
            candidates.append({
                "file_name": name,
                "fold": fold,
                "candidate_index": 0,
            })
        locked = f"fold{fold}-locked.jpeg"
        rows.append({
            "file_name": locked,
            "fold": fold,
            "time_block_role": "locked_evaluation",
            "site_code": "Bh",
            "year": 2020,
        })
        candidates.append({
            "file_name": locked,
            "fold": fold,
            "candidate_index": 0,
        })

    first = select_fixed_development_sample(
        pd.DataFrame(rows), pd.DataFrame(candidates), per_fold=2, seed=7
    )
    second = select_fixed_development_sample(
        pd.DataFrame(rows), pd.DataFrame(candidates), per_fold=2, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("fold").size().to_dict() == {0: 2, 1: 2, 2: 2}
    assert (first.time_block_role == "development").all()
    assert not first.file_name.str.contains("locked").any()


def test_stable_selection_hash_changes_with_seed():
    assert stable_selection_hash("a.jpeg", 1) == stable_selection_hash("a.jpeg", 1)
    assert stable_selection_hash("a.jpeg", 1) != stable_selection_hash("a.jpeg", 2)


def test_light_variant_metrics_exact_prediction():
    truth = np.zeros((16, 16), dtype=bool)
    truth[4:10, 3:12] = True
    spine = np.asarray([[4, 7], [7, 7], [10, 7]], dtype=float)
    prepared = {"truths": [truth], "spines": [spine]}

    exact = light_variant_metrics(prepared, [truth.copy()])
    missing = light_variant_metrics(prepared, [])

    assert exact["pq_512"] == 1.0
    assert exact["spine_coverage"] == 1.0
    assert exact["filament_recall"] == 1.0
    assert exact["filament_precision"] == 1.0
    assert exact["merge_count"] == 0.0
    assert exact["count_abs_error"] == 0.0
    assert missing["pq_512"] == 0.0
    assert missing["spine_coverage"] == 0.0
    assert missing["filament_recall"] == 0.0
    assert missing["count_abs_error"] == 1.0


def test_linearized_pq_utility_uses_fixed_reference_ratio():
    full = {"tp_iou_sum": 3.0, "denominator": 5.0}
    without = {"tp_iou_sum": 2.2, "denominator": 4.5}

    result = linearized_pq_utility(full, without, reference_pq=0.4)

    assert result == pytest.approx(0.8 - 0.4 * 0.5)


def test_catalog_safe_target_requires_all_endpoint_directions():
    safe = {
        "spine_coverage": 0.1,
        "filament_precision": 0.0,
        "merge_count": -1.0,
        "count_abs_error": 0.0,
    }

    assert final_catalog_safe_target(safe)
    for endpoint, bad_value in (
        ("spine_coverage", 0.0),
        ("filament_precision", -0.01),
        ("merge_count", 1.0),
        ("count_abs_error", 1.0),
    ):
        changed = dict(safe)
        changed[endpoint] = bad_value
        assert not final_catalog_safe_target(changed)


def test_binary_diagnostics_reports_reference_positive_performance():
    reference = np.asarray([True, True, False, False])
    predicted = np.asarray([True, False, True, False])

    result = binary_diagnostics(reference, predicted)

    assert result == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "agreement": 0.5,
        "precision": 0.5,
        "recall": 0.5,
    }


def test_safe_spearman_returns_null_for_constant_input():
    result = safe_spearman(
        pd.Series([1.0, 1.0, 1.0]),
        pd.Series([1.0, 2.0, 3.0]),
    )

    assert result == {"rho": None, "pvalue": None, "n": 3}
