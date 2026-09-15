"""Freeze a target-independent observation-grouped nested split registry for S1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATA = ROOT / "output" / "s0" / "observation_strata.csv"
DEFAULT_CANDIDATES = ROOT / "output" / "s0" / "v42_nested_candidate_predictions.csv"
DEFAULT_REGISTRY_OUT = ROOT / "output" / "s0" / "s1_grouped_nested_split_registry.csv"
DEFAULT_OUT = ROOT / "output" / "s0" / "s1_grouped_nested_split_registry.json"
SEED = 20_260_824
INNER_FOLDS = 3
DEVELOPMENT_ROLE = "development"
LOCKED_ROLE = "locked_evaluation"
STRATA_COLUMNS = ("file_name", "fold", "site_code", "year", "time_block_role")
CANDIDATE_COLUMNS = ("file_name", "fold", "candidate_index")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def load_observation_table(
    strata_path: Path, candidates_path: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load only split metadata and candidate identities; no target columns."""
    strata = pd.read_csv(strata_path, usecols=list(STRATA_COLUMNS))
    roles = strata["time_block_role"].astype(str).value_counts().to_dict()
    unexpected = sorted(set(roles).difference({DEVELOPMENT_ROLE, LOCKED_ROLE}))
    if unexpected:
        raise ValueError(f"unexpected time block roles: {unexpected}")
    development = strata.loc[
        strata["time_block_role"].astype(str) == DEVELOPMENT_ROLE
    ].copy()
    if development["file_name"].duplicated().any():
        raise ValueError("duplicate development observations")
    development["fold"] = pd.to_numeric(development["fold"], errors="raise").astype(int)
    development["year"] = pd.to_numeric(development["year"], errors="raise").astype(int)

    candidates = pd.read_csv(candidates_path, usecols=list(CANDIDATE_COLUMNS))
    candidates["fold"] = pd.to_numeric(candidates["fold"], errors="raise").astype(int)
    candidates["candidate_index"] = pd.to_numeric(
        candidates["candidate_index"], errors="raise"
    ).astype(int)
    candidate_groups = candidates.groupby(["file_name", "fold"], as_index=False).agg(
        candidate_count=("candidate_index", "count"),
        candidate_index_min=("candidate_index", "min"),
        candidate_index_max=("candidate_index", "max"),
    )
    expected_max = candidate_groups["candidate_count"] - 1
    if not (
        (candidate_groups["candidate_index_min"] == 0)
        & (candidate_groups["candidate_index_max"] == expected_max)
    ).all():
        raise ValueError("candidate indices are not contiguous")

    merged = development.merge(
        candidate_groups,
        on=["file_name", "fold"],
        how="left",
        validate="one_to_one",
    )
    if merged["candidate_count"].isna().any():
        missing = merged.loc[merged["candidate_count"].isna(), "file_name"].tolist()
        raise ValueError(f"development observations missing candidates: {missing[:5]}")
    merged["candidate_count"] = merged["candidate_count"].astype(int)
    return merged.drop(columns=["candidate_index_min", "candidate_index_max"]), {
        "development_observations": int(roles.get(DEVELOPMENT_ROLE, 0)),
        "locked_observations_excluded": int(roles.get(LOCKED_ROLE, 0)),
    }


def assign_inner_folds(
    outer_training: pd.DataFrame,
    outer_fold: int,
    *,
    inner_folds: int = INNER_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Greedily balance site×year cells, then candidate and observation totals."""
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least two")
    if outer_training.empty:
        raise ValueError("outer training table is empty")
    frame = outer_training.copy()
    frame["balance_stratum"] = (
        frame["site_code"].astype(str) + "|" + frame["year"].astype(str)
    )
    assignments: dict[str, int] = {}
    total_candidates = [0] * inner_folds
    total_observations = [0] * inner_folds
    stratum_counts: dict[str, list[int]] = {}

    for stratum, group in frame.groupby("balance_stratum", sort=True):
        stratum_counts[stratum] = [0] * inner_folds
        ordered = group.assign(
            assignment_hash=group["file_name"].map(
                lambda name: stable_hash(seed, outer_fold, stratum, name)
            )
        ).sort_values(["candidate_count", "assignment_hash"], ascending=[False, True])
        tie_priority = {
            fold: stable_hash(seed, outer_fold, stratum, "tie", fold)
            for fold in range(inner_folds)
        }
        for row in ordered.itertuples(index=False):
            chosen = min(
                range(inner_folds),
                key=lambda fold: (
                    stratum_counts[stratum][fold],
                    total_candidates[fold],
                    total_observations[fold],
                    tie_priority[fold],
                ),
            )
            assignments[str(row.file_name)] = chosen
            stratum_counts[stratum][chosen] += 1
            total_candidates[chosen] += int(row.candidate_count)
            total_observations[chosen] += 1

    frame["inner_fold"] = frame["file_name"].map(assignments).astype(int)
    frame["assignment_hash"] = frame["file_name"].map(
        lambda name: stable_hash(seed, outer_fold, name)
    )
    return frame


def build_registry(
    observations: pd.DataFrame,
    *,
    inner_folds: int = INNER_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    source_folds = sorted(observations["fold"].astype(int).unique().tolist())
    if source_folds != [0, 1, 2]:
        raise ValueError(f"expected frozen source folds [0, 1, 2], got {source_folds}")
    cycles = []
    for outer_fold in source_folds:
        outer_test = observations.loc[observations["fold"] == outer_fold].copy()
        outer_test["outer_fold"] = outer_fold
        outer_test["outer_role"] = "outer_test"
        outer_test["inner_fold"] = -1
        outer_test["balance_stratum"] = (
            outer_test["site_code"].astype(str) + "|" + outer_test["year"].astype(str)
        )
        outer_test["assignment_hash"] = outer_test["file_name"].map(
            lambda name: stable_hash(seed, outer_fold, name)
        )

        outer_training = observations.loc[observations["fold"] != outer_fold].copy()
        assigned = assign_inner_folds(
            outer_training, outer_fold, inner_folds=inner_folds, seed=seed
        )
        assigned["outer_fold"] = outer_fold
        assigned["outer_role"] = "outer_train"
        cycles.extend([assigned, outer_test])

    registry = pd.concat(cycles, ignore_index=True).rename(columns={"fold": "source_fold"})
    registry = registry[
        [
            "outer_fold",
            "outer_role",
            "inner_fold",
            "file_name",
            "source_fold",
            "site_code",
            "year",
            "time_block_role",
            "candidate_count",
            "balance_stratum",
            "assignment_hash",
        ]
    ].sort_values(["outer_fold", "outer_role", "inner_fold", "file_name"])
    return registry.reset_index(drop=True)


def validate_registry(
    registry: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    inner_folds: int = INNER_FOLDS,
) -> None:
    expected_observations = set(observations["file_name"].astype(str))
    outer_folds = sorted(observations["fold"].astype(int).unique().tolist())
    if len(registry) != len(observations) * len(outer_folds):
        raise ValueError("registry row count mismatch")
    if set(registry["time_block_role"].astype(str)) != {DEVELOPMENT_ROLE}:
        raise ValueError("non-development observation entered registry")
    if registry.duplicated(["outer_fold", "file_name"]).any():
        raise ValueError("observation duplicated within outer cycle")
    for outer_fold in outer_folds:
        cycle = registry.loc[registry["outer_fold"] == outer_fold]
        if set(cycle["file_name"].astype(str)) != expected_observations:
            raise ValueError(f"outer cycle {outer_fold} does not cover all observations")
        outer_test = cycle.loc[cycle["outer_role"] == "outer_test"]
        outer_train = cycle.loc[cycle["outer_role"] == "outer_train"]
        if not (outer_test["source_fold"].astype(int) == outer_fold).all():
            raise ValueError(f"outer test leakage in cycle {outer_fold}")
        if not (outer_train["source_fold"].astype(int) != outer_fold).all():
            raise ValueError(f"outer training leakage in cycle {outer_fold}")
        if set(outer_test["inner_fold"].astype(int)) != {-1}:
            raise ValueError("outer test observations received inner folds")
        if set(outer_train["inner_fold"].astype(int)) != set(range(inner_folds)):
            raise ValueError(f"inner fold coverage mismatch in cycle {outer_fold}")
    test_counts = (
        registry.loc[registry["outer_role"] == "outer_test", "file_name"].value_counts()
    )
    if not (test_counts == 1).all() or set(test_counts.index.astype(str)) != expected_observations:
        raise ValueError("each observation must be outer test exactly once")


def balance_summary(registry: pd.DataFrame, inner_folds: int = INNER_FOLDS) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for outer_fold, cycle in registry.groupby("outer_fold", sort=True):
        outer_test = cycle.loc[cycle["outer_role"] == "outer_test"]
        outer_train = cycle.loc[cycle["outer_role"] == "outer_train"]
        inner = []
        observation_totals = []
        candidate_totals = []
        all_strata = set(outer_train["balance_stratum"].astype(str))
        missing_cells = 0
        for inner_fold in range(inner_folds):
            validation = outer_train.loc[outer_train["inner_fold"] == inner_fold]
            strata = set(validation["balance_stratum"].astype(str))
            missing_cells += len(all_strata.difference(strata))
            observation_totals.append(len(validation))
            candidate_totals.append(int(validation["candidate_count"].sum()))
            inner.append({
                "inner_fold": inner_fold,
                "validation_observations": int(len(validation)),
                "validation_candidates": int(validation["candidate_count"].sum()),
                "site_counts": {
                    str(key): int(value)
                    for key, value in validation["site_code"].value_counts().sort_index().items()
                },
                "year_counts": {
                    str(int(key)): int(value)
                    for key, value in validation["year"].value_counts().sort_index().items()
                },
                "balance_strata_present": len(strata),
            })
        summary[str(int(outer_fold))] = {
            "outer_test_observations": int(len(outer_test)),
            "outer_test_candidates": int(outer_test["candidate_count"].sum()),
            "outer_train_observations": int(len(outer_train)),
            "outer_train_candidates": int(outer_train["candidate_count"].sum()),
            "inner_validation": inner,
            "inner_observation_max_minus_min": int(max(observation_totals) - min(observation_totals)),
            "inner_candidate_max_minus_min": int(max(candidate_totals) - min(candidate_totals)),
            "inner_candidate_relative_range": (
                (max(candidate_totals) - min(candidate_totals))
                / (sum(candidate_totals) / len(candidate_totals))
            ),
            "missing_site_year_cells_across_inner_folds": int(missing_cells),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", type=Path, default=DEFAULT_STRATA)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--registry-out", type=Path, default=DEFAULT_REGISTRY_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strata_path = args.strata.resolve()
    candidates_path = args.candidates.resolve()
    registry_out = args.registry_out.resolve()
    out_path = args.out.resolve()
    observations, role_audit = load_observation_table(strata_path, candidates_path)
    registry = build_registry(
        observations, inner_folds=args.inner_folds, seed=args.seed
    )
    validate_registry(registry, observations, inner_folds=args.inner_folds)
    registry_out.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(registry_out, index=False)
    payload = {
        "schema_version": 1,
        "status": "target_independent_grouped_nested_split_registry_frozen_not_model_not_s1",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed,
        "scope": {
            **role_audit,
            "development_candidates": int(observations["candidate_count"].sum()),
            "outer_folds": [0, 1, 2],
            "inner_folds_per_outer_cycle": args.inner_folds,
            "registry_rows": int(len(registry)),
            "locked_rows_in_registry": 0,
            "ground_truth_read": False,
            "target_columns_loaded": False,
            "feature_values_loaded": False,
            "s1_predictions_read": False,
            "review_responses_read": False,
            "private_key_read": False,
            "model_fitting_performed": False,
            "threshold_selection_performed": False,
        },
        "split_contract": {
            "outer_test": "frozen source fold; each observation is outer test exactly once",
            "outer_train": "the other two source folds",
            "inner_assignment": (
                "three target-independent folds greedily balanced by site×year cell, then "
                "prediction candidate count and observation count"
            ),
            "group_key": "file_name observation; all candidates remain together",
            "site_year_use": "split balance and reporting only; forbidden as prediction features",
            "inner_fit_fraction": "approximately two thirds of each outer-training cycle",
            "freeze_rule": "registry SHA-256 must not change after harm labels are joined",
        },
        "balance": balance_summary(registry, args.inner_folds),
        "inputs": {
            "observation_strata": str(strata_path),
            "observation_strata_sha256": sha256_file(strata_path),
            "strata_columns_loaded": list(STRATA_COLUMNS),
            "candidate_table": str(candidates_path),
            "candidate_table_sha256": sha256_file(candidates_path),
            "candidate_columns_loaded": list(CANDIDATE_COLUMNS),
        },
        "artifacts": {
            "registry": str(registry_out),
            "registry_sha256": sha256_file(registry_out),
        },
        "limitations": [
            "Target-independent balance does not guarantee equal harm prevalence.",
            "Source folds remain the historical grouped folds, not a new time/site holdout.",
            "Site and year are audit strata, never prediction features.",
            "No model, feature scaling, threshold, margin, locked evaluation, or review response was used.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "development_observations": role_audit["development_observations"],
                "development_candidates": payload["scope"]["development_candidates"],
                "registry_rows": len(registry),
                "output": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
