from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import solar_filament.s1.grouped_nested_split as MODULE


def observation_table() -> pd.DataFrame:
    records = []
    for fold in range(3):
        for index in range(12):
            records.append({
                "file_name": f"f{fold}_{index:02d}.jpeg",
                "fold": fold,
                "site_code": ("Ah", "Bh", "Ch")[index % 3],
                "year": 2011 + index % 3,
                "time_block_role": "development",
                "candidate_count": 1 + (fold + index) % 7,
            })
    return pd.DataFrame(records)


def test_stable_hash_is_deterministic_and_order_sensitive() -> None:
    assert MODULE.stable_hash(1, "a") == MODULE.stable_hash(1, "a")
    assert MODULE.stable_hash(1, "a") != MODULE.stable_hash("a", 1)


def test_registry_is_deterministic() -> None:
    observations = observation_table()
    first = MODULE.build_registry(observations, seed=7)
    second = MODULE.build_registry(observations, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_each_observation_is_outer_test_once() -> None:
    registry = MODULE.build_registry(observation_table())
    outer_test = registry.loc[registry["outer_role"] == "outer_test"]
    assert (outer_test["file_name"].value_counts() == 1).all()
    assert (outer_test["inner_fold"] == -1).all()
    assert (outer_test["source_fold"] == outer_test["outer_fold"]).all()


def test_outer_train_excludes_held_source_fold() -> None:
    registry = MODULE.build_registry(observation_table())
    outer_train = registry.loc[registry["outer_role"] == "outer_train"]
    assert (outer_train["source_fold"] != outer_train["outer_fold"]).all()
    for _, cycle in outer_train.groupby("outer_fold"):
        assert set(cycle["inner_fold"]) == {0, 1, 2}


def test_validate_registry_rejects_outer_leakage() -> None:
    observations = observation_table()
    registry = MODULE.build_registry(observations)
    row = registry.index[registry["outer_role"] == "outer_train"][0]
    registry.loc[row, "source_fold"] = registry.loc[row, "outer_fold"]
    with pytest.raises(ValueError, match="outer training leakage"):
        MODULE.validate_registry(registry, observations)


def test_inner_assignment_balances_composite_strata() -> None:
    observations = observation_table()
    training = observations.loc[observations["fold"] != 0]
    assigned = MODULE.assign_inner_folds(training, 0, seed=11)
    table = assigned.groupby(["balance_stratum", "inner_fold"]).size().unstack(fill_value=0)
    assert ((table.max(axis=1) - table.min(axis=1)) <= 1).all()


def test_balance_summary_covers_each_inner_fold() -> None:
    registry = MODULE.build_registry(observation_table())
    summary = MODULE.balance_summary(registry)
    assert set(summary) == {"0", "1", "2"}
    assert all(len(cycle["inner_validation"]) == 3 for cycle in summary.values())


def test_load_observation_table_excludes_locked_and_target_columns(tmp_path: Path) -> None:
    observations = observation_table()
    locked = observations.iloc[[0]].copy()
    locked["file_name"] = "locked.jpeg"
    locked["time_block_role"] = "locked_evaluation"
    strata = pd.concat([observations, locked], ignore_index=True)
    strata["instance_count_min"] = 999
    candidate_records = []
    for row in observations.itertuples(index=False):
        for candidate_index in range(row.candidate_count):
            candidate_records.append({
                "file_name": row.file_name,
                "fold": row.fold,
                "candidate_index": candidate_index,
                "target": 999,
            })
    strata_path = tmp_path / "strata.csv"
    candidates_path = tmp_path / "candidates.csv"
    strata.to_csv(strata_path, index=False)
    pd.DataFrame(candidate_records).to_csv(candidates_path, index=False)
    loaded, audit = MODULE.load_observation_table(strata_path, candidates_path)
    assert "locked.jpeg" not in set(loaded["file_name"])
    assert "instance_count_min" not in loaded.columns
    assert "target" not in loaded.columns
    assert audit["locked_observations_excluded"] == 1


def test_load_observation_table_rejects_noncontiguous_candidate_indices(
    tmp_path: Path,
) -> None:
    observations = observation_table().iloc[[0]]
    strata_path = tmp_path / "strata.csv"
    candidates_path = tmp_path / "candidates.csv"
    observations.to_csv(strata_path, index=False)
    pd.DataFrame([
        {"file_name": observations.iloc[0].file_name, "fold": 0, "candidate_index": 1}
    ]).to_csv(candidates_path, index=False)
    with pytest.raises(ValueError, match="not contiguous"):
        MODULE.load_observation_table(strata_path, candidates_path)
