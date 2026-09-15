from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import solar_filament.s1.set_context_features as MODULE


def mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    result = np.zeros(MODULE.IMAGE_SHAPE, dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def candidates(count: int) -> pd.DataFrame:
    records = []
    for index in range(count):
        record = {
            "file_name": "sample.jpeg",
            "fold": 0,
            "candidate_index": index,
        }
        for offset, column in enumerate(MODULE.BASE_FEATURE_COLUMNS, 1):
            record[column] = float(index + offset)
        records.append(record)
    return pd.DataFrame(records)


def metadata() -> dict[str, object]:
    return {
        "file_name": "sample.jpeg",
        "site_code": "Th",
        "year": 2012,
        "time_block_role": "development",
        "selection_hash": "abc",
    }


def test_ownership_trace_keeps_identity_and_drops_contained() -> None:
    large = mask(10, 40, 10, 40)
    small = mask(15, 25, 15, 25)
    other = mask(50, 60, 50, 60)
    masks = [large, small, other]
    intersections = MODULE.pairwise_intersections(masks)
    trace = MODULE.ownership_trace(masks, intersections)
    assert 0 in trace["kept_indices"]
    assert 1 in trace["dropped_by_containment"]
    assert 2 in trace["kept_indices"]
    assert len(trace["kept_indices"]) == 2


def test_build_observation_features_output_count_delta() -> None:
    large = mask(10, 40, 10, 40)
    small = mask(15, 25, 15, 25)
    other = mask(50, 60, 50, 60)
    table = candidates(3)
    features = MODULE.build_observation_features(
        table, [large, small, other], metadata()
    )
    by_index = features.set_index("candidate_index")
    assert by_index.loc[1, "dropped_by_containment"] == 1
    assert by_index.loc[1, "ownership_output_kept"] == 0
    assert by_index.loc[0, "ownership_output_kept"] == 1
    assert by_index.loc[2, "ownership_output_kept"] == 1
    assert int(features["full_output_count"].iloc[0]) == 2
    # Removing ``other`` drops output from 2 to 1 (large alone; small still contained).
    assert by_index.loc[2, "output_count_delta"] == 1
    # Removing the container keeps small+other = 2, same as full output.
    assert by_index.loc[0, "output_count_delta"] == 0


def test_output_count_delta_negative_when_container_displaces_peer() -> None:
    """Large candidate stays and displaces a peer that would otherwise survive alone."""
    a = mask(10, 40, 10, 40)
    b = mask(12, 42, 12, 42)  # heavy overlap; smaller area loses at containment 0.80
    c = mask(50, 60, 50, 60)
    # Force areas so b is smaller and highly contained in a.
    table = candidates(3)
    features = MODULE.build_observation_features(table, [a, b, c], metadata())
    by_index = features.set_index("candidate_index")
    assert int(features["full_output_count"].iloc[0]) >= 1
    assert set(by_index["ownership_output_kept"].tolist()) <= {0, 1}


def test_validate_output_rejects_forbidden_target_columns() -> None:
    table = candidates(1)
    frame = MODULE.build_observation_features(
        table, [mask(1, 5, 1, 5)], metadata()
    )
    frame["delta_pq_utility"] = 0.1
    with pytest.raises(ValueError, match="forbidden"):
        MODULE.validate_output(frame)


def test_feature_contract_columns_stable() -> None:
    assert len(MODULE.BASE_FEATURE_COLUMNS) == 11
    assert len(MODULE.CONTEXT_FEATURE_COLUMNS) == 29
    assert "site_code" not in MODULE.CONTEXT_FEATURE_COLUMNS
    assert "year" not in MODULE.CONTEXT_FEATURE_COLUMNS
    assert MODULE.OWNERSHIP_THRESHOLD == 0.80


def test_load_inputs_whitelist_uses_only_base_features(tmp_path) -> None:
    selection = pd.DataFrame([{
        "file_name": "sample.jpeg",
        "fold": 0,
        "site_code": "Th",
        "year": 2012,
        "time_block_role": "development",
        "selection_hash": "abc",
    }])
    record = {
        "file_name": "sample.jpeg",
        "fold": 0,
        "candidate_index": 0,
        "target": 1,
        "delta_pq_utility": 0.2,
    }
    for offset, column in enumerate(MODULE.BASE_FEATURE_COLUMNS, 1):
        record[column] = float(offset)
    candidates_frame = pd.DataFrame([record])
    selection_path = tmp_path / "selection.csv"
    candidates_path = tmp_path / "candidates.csv"
    selection.to_csv(selection_path, index=False)
    candidates_frame.to_csv(candidates_path, index=False)
    loaded_selection, loaded_candidates = MODULE.load_inputs(
        selection_path, candidates_path
    )
    assert "target" not in loaded_candidates.columns
    assert "delta_pq_utility" not in loaded_candidates.columns
    assert list(loaded_selection["file_name"]) == ["sample.jpeg"]
    assert len(loaded_candidates) == 1
