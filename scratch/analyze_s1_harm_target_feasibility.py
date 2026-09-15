"""Audit fixed-pilot harm labels without fitting a model or selecting a threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = ROOT / "output" / "s0" / "s1_proposal_marginal_utility_pilot.csv"
DEFAULT_OBSERVATIONS_OUT = (
    ROOT / "output" / "s0" / "s1_harm_target_feasibility_observations.csv"
)
DEFAULT_OUT = ROOT / "output" / "s0" / "s1_harm_target_feasibility.json"
DEFAULT_DEVELOPMENT_CANDIDATES = 4_899
DEFAULT_DEVELOPMENT_OBSERVATIONS = 547
SEED = 20_260_824
BOOTSTRAP_ITERATIONS = 10_000


@dataclass(frozen=True)
class TargetSpec:
    name: str
    role: str


TARGET_SPECS = (
    TargetSpec("final_pq_benefit", "benefit"),
    TargetSpec("spine_benefit", "benefit"),
    TargetSpec("count_harm", "harm"),
    TargetSpec("precision_harm", "harm"),
    TargetSpec("merge_harm", "harm"),
    TargetSpec("any_catalog_harm", "harm"),
    TargetSpec("catalog_safe", "joint"),
)
HARM_TARGETS = ("count_harm", "precision_harm", "merge_harm")
REQUIRED_COLUMNS = {
    "file_name",
    "fold",
    "candidate_index",
    "time_block_role",
    "site_code",
    "year",
    "final_pq_positive_target",
    "final_catalog_safe_target",
    "delta_spine_coverage",
    "delta_filament_precision",
    "delta_merge_count",
    "delta_count_abs_error",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def zero_event_validation_size(max_harm_rate: float, confidence: float = 0.95) -> int:
    """Independent zero-harm trials needed for an exact one-sided upper bound."""
    if not 0.0 < max_harm_rate < 1.0:
        raise ValueError("max_harm_rate must be in (0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    return math.ceil(math.log(alpha) / math.log(1.0 - max_harm_rate))


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(rows[column], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite values in {column}")
    return values


def derive_targets(rows: pd.DataFrame) -> pd.DataFrame:
    """Validate the pilot scope and derive fixed benefit/harm labels."""
    missing = sorted(REQUIRED_COLUMNS.difference(rows.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if rows.empty:
        raise ValueError("candidate rows are empty")
    roles = set(rows["time_block_role"].astype(str))
    if roles != {"development"}:
        raise ValueError(f"expected development-only rows, got {sorted(roles)}")
    if rows.duplicated(["file_name", "candidate_index"]).any():
        raise ValueError("duplicate file_name/candidate_index rows")

    labeled = rows.copy()
    labeled["final_pq_benefit"] = _numeric(labeled, "final_pq_positive_target") == 1
    labeled["spine_benefit"] = _numeric(labeled, "delta_spine_coverage") > 0.0
    labeled["count_harm"] = _numeric(labeled, "delta_count_abs_error") > 0.0
    labeled["precision_harm"] = _numeric(labeled, "delta_filament_precision") < 0.0
    labeled["merge_harm"] = _numeric(labeled, "delta_merge_count") > 0.0
    labeled["any_catalog_harm"] = labeled[list(HARM_TARGETS)].any(axis=1)
    labeled["catalog_safe"] = labeled["spine_benefit"] & ~labeled["any_catalog_harm"]

    archived_safe = _numeric(labeled, "final_catalog_safe_target") == 1
    mismatch = int((labeled["catalog_safe"] != archived_safe).sum())
    if mismatch:
        raise ValueError(f"catalog-safe contract mismatch in {mismatch} rows")
    return labeled


def build_observation_table(labeled: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for file_name, group in labeled.groupby("file_name", sort=True):
        metadata: dict[str, Any] = {}
        for column in ("fold", "site_code", "year", "time_block_role"):
            values = group[column].drop_duplicates().tolist()
            if len(values) != 1:
                raise ValueError(f"{file_name}: non-unique {column}: {values}")
            metadata[column] = values[0]
        record: dict[str, Any] = {
            "file_name": file_name,
            "fold": int(metadata["fold"]),
            "site_code": str(metadata["site_code"]),
            "year": int(metadata["year"]),
            "time_block_role": str(metadata["time_block_role"]),
            "candidates": int(len(group)),
        }
        for spec in TARGET_SPECS:
            events = int(group[spec.name].sum())
            record[f"{spec.name}_events"] = events
            record[f"{spec.name}_rate"] = events / len(group)
        records.append(record)
    return pd.DataFrame.from_records(records)


def target_summary(labeled: pd.DataFrame, observations: pd.DataFrame, name: str) -> dict[str, Any]:
    candidate_events = int(labeled[name].sum())
    candidate_total = int(len(labeled))
    event_counts = observations[f"{name}_events"].to_numpy(dtype=int)
    observation_events = int((event_counts > 0).sum())
    observation_total = int(len(observations))
    candidate_ci = wilson_interval(candidate_events, candidate_total)
    observation_ci = wilson_interval(observation_events, observation_total)

    positive_rows = labeled.loc[labeled[name]]
    shares = np.sort(event_counts[event_counts > 0].astype(float))[::-1]
    largest_share = float(shares[0] / candidate_events) if candidate_events else None
    top3_share = float(shares[:3].sum() / candidate_events) if candidate_events else None
    return {
        "candidate_events": candidate_events,
        "candidate_total": candidate_total,
        "candidate_rate": candidate_events / candidate_total,
        "candidate_rate_wilson95_descriptive": list(candidate_ci),
        "observation_events": observation_events,
        "observation_total": observation_total,
        "observation_event_rate": observation_events / observation_total,
        "observation_event_rate_wilson95": list(observation_ci),
        "folds_with_events": sorted(positive_rows["fold"].astype(int).unique().tolist()),
        "sites_with_events": sorted(positive_rows["site_code"].astype(str).unique().tolist()),
        "years_with_events": sorted(positive_rows["year"].astype(int).unique().tolist()),
        "events_per_positive_observation": {
            "min": int(shares.min()) if len(shares) else 0,
            "median": float(np.median(shares)) if len(shares) else 0.0,
            "max": int(shares.max()) if len(shares) else 0,
        },
        "event_concentration": {
            "largest_observation_share": largest_share,
            "top3_observations_share": top3_share,
        },
        "screening_flags": {
            "at_least_10_candidate_events": candidate_events >= 10,
            "at_least_10_event_observations": observation_events >= 10,
            "events_present_in_all_folds": len(set(positive_rows["fold"].astype(int))) == 3,
        },
    }


def cluster_bootstrap_projection(
    observations: pd.DataFrame,
    target: str,
    development_candidates: int,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Project event count by resampling whole observations, not candidate rows."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    candidates = observations["candidates"].to_numpy(dtype=int)
    events = observations[f"{target}_events"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(observations), size=(iterations, len(observations)))
    sampled_candidates = candidates[indices].sum(axis=1)
    sampled_events = events[indices].sum(axis=1)
    rates = sampled_events / sampled_candidates
    projected = rates * development_candidates
    return {
        "development_candidates": int(development_candidates),
        "point_from_pilot_rate": float(events.sum() / candidates.sum() * development_candidates),
        "cluster_bootstrap_median": float(np.median(projected)),
        "cluster_bootstrap95": [
            float(np.quantile(projected, 0.025)),
            float(np.quantile(projected, 0.975)),
        ],
        "iterations": int(iterations),
        "seed": int(seed),
        "interpretation": "planning range only; 12-observation pilot is not performance evidence",
    }


def harm_cooccurrence(labeled: pd.DataFrame) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    jaccard: dict[str, dict[str, float | None]] = {}
    for first in HARM_TARGETS:
        counts[first] = {}
        jaccard[first] = {}
        for second in HARM_TARGETS:
            both = int((labeled[first] & labeled[second]).sum())
            union = int((labeled[first] | labeled[second]).sum())
            counts[first][second] = both
            jaccard[first][second] = both / union if union else None
    return {"candidate_cooccurrence_counts": counts, "candidate_jaccard": jaccard}


def build_report(
    labeled: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    development_candidates: int,
    development_observations: int,
) -> dict[str, Any]:
    summaries = {
        spec.name: target_summary(labeled, observations, spec.name) for spec in TARGET_SPECS
    }
    projections = {
        target: cluster_bootstrap_projection(observations, target, development_candidates)
        for target in HARM_TARGETS
    }
    observation_projection: dict[str, Any] = {}
    for target in HARM_TARGETS:
        summary = summaries[target]
        low, high = summary["observation_event_rate_wilson95"]
        observation_projection[target] = {
            "point": summary["observation_event_rate"] * development_observations,
            "wilson95_planning_range": [low * development_observations, high * development_observations],
        }

    return {
        "target_summaries": summaries,
        "harm_cooccurrence": harm_cooccurrence(labeled),
        "full_development_planning_projection": {
            "candidate_events_cluster_bootstrap": projections,
            "observations_with_events_wilson": observation_projection,
        },
        "zero_observed_harm_validation_lower_bounds": {
            "confidence": 0.95,
            "independent_accepted_candidates_required": {
                "harm_rate_below_0.05": zero_event_validation_size(0.05),
                "harm_rate_below_0.02": zero_event_validation_size(0.02),
                "harm_rate_below_0.01": zero_event_validation_size(0.01),
            },
            "caveat": (
                "candidate counts are optimistic lower bounds; clustered observations and time/site "
                "coverage require more validation"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--observations-out", type=Path, default=DEFAULT_OBSERVATIONS_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--development-candidates", type=int, default=DEFAULT_DEVELOPMENT_CANDIDATES
    )
    parser.add_argument(
        "--development-observations", type=int, default=DEFAULT_DEVELOPMENT_OBSERVATIONS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows_path = args.rows.resolve()
    observations_out = args.observations_out.resolve()
    out_path = args.out.resolve()
    rows = pd.read_csv(rows_path)
    labeled = derive_targets(rows)
    observations = build_observation_table(labeled)
    report = build_report(
        labeled,
        observations,
        development_candidates=args.development_candidates,
        development_observations=args.development_observations,
    )

    observations_out.parent.mkdir(parents=True, exist_ok=True)
    observations.to_csv(observations_out, index=False)
    payload = {
        "schema_version": 1,
        "status": "fixed_pilot_harm_label_feasibility_not_model_not_threshold_not_registered",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "scope": {
            "candidate_rows": int(len(labeled)),
            "observations": int(len(observations)),
            "folds": sorted(observations["fold"].astype(int).unique().tolist()),
            "years": sorted(observations["year"].astype(int).unique().tolist()),
            "sites": sorted(observations["site_code"].astype(str).unique().tolist()),
            "time_block_role": "development only",
            "locked_rows_read": 0,
            "s1_predictions_read": False,
            "blinded_review_responses_read": False,
            "private_key_read": False,
            "model_fitting_performed": False,
            "threshold_selection_performed": False,
        },
        "target_contract": {
            "final_pq_benefit": "archived final_pq_positive_target == 1",
            "spine_benefit": "delta_spine_coverage > 0",
            "count_harm": "delta_count_abs_error > 0",
            "precision_harm": "delta_filament_precision < 0",
            "merge_harm": "delta_merge_count > 0",
            "any_catalog_harm": "count_harm or precision_harm or merge_harm",
            "catalog_safe": "spine_benefit and not any_catalog_harm",
        },
        "inputs": {"candidate_rows": str(rows_path), "candidate_rows_sha256": sha256_file(rows_path)},
        "artifacts": {
            "observation_table": str(observations_out),
            "observation_table_sha256": sha256_file(observations_out),
        },
        "analysis": report,
        "limitations": [
            "Fixed 12-observation pilot estimates label structure, not model performance.",
            "Candidate rows within an observation are dependent through ownership and count state.",
            "Bootstrap and Wilson ranges are planning aids, not registered confidence intervals.",
            "No full-development labels, model, threshold, locked rows, or review responses were used.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "observations": len(observations),
                "candidates": len(labeled),
                "output": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
