"""Calibrate S1 endpoint-margin operating characteristics on frozen S0 data.

This script does not choose margins and does not read any S1 predictions.  It
quantifies observation-level variability, annotator repeatability, empirical
paired-difference scales, and approximate locked-block power before S1 starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output/s0/science_case_metrics_v3.csv"
DEFAULT_OUTPUT = ROOT / "output/s0/s1_endpoint_margin_calibration.json"
SEED = 20260824
REPEATS = 5000
ENDPOINTS = {
    "spine_coverage": "higher_is_better",
    "filament_recall": "higher_is_better",
    "filament_precision": "higher_is_better",
    "merge_count": "lower_is_better",
    "spine_coverage_matched": "higher_is_better",
    "count_abs_error": "lower_is_better",
    "pq_512": "higher_is_better",
}
EFFECT_GRID = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def descriptive(values: np.ndarray) -> dict[str, Any]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"n": 0}
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "se_mean": float(clean.std(ddof=1) / np.sqrt(clean.size))
        if clean.size > 1
        else 0.0,
        "median": float(np.median(clean)),
        "q10": float(np.quantile(clean, 0.10)),
        "q90": float(np.quantile(clean, 0.90)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    repeats: int = REPEATS,
    seed: int = SEED,
) -> dict[str, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("bootstrap input is empty")
    if clean.size == 1 or np.all(clean == clean[0]):
        value = float(clean.mean())
        return {"estimate": value, "ci_low": value, "ci_high": value}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(clean, size=(repeats, clean.size), replace=True)
    means = sampled.mean(axis=1)
    return {
        "estimate": float(clean.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def observation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, str] = {endpoint: "mean" for endpoint in ENDPOINTS}
    aggregations.update({
        "fold": "first",
        "site_code": "first",
        "year": "first",
        "time_block_role": "first",
    })
    result = (
        frame.groupby(["file_name", "model"], as_index=False)
        .agg(aggregations)
        .sort_values(["file_name", "model"])
        .reset_index(drop=True)
    )
    if result.duplicated(["file_name", "model"]).any():
        raise AssertionError("observation aggregation is not unique")
    return result


def annotator_repeatability(
    frame: pd.DataFrame,
    endpoint: str,
) -> dict[str, Any]:
    per_observation = []
    pairwise = []
    for _, group in frame.loc[frame.model == "v35"].groupby("file_name"):
        values = group[endpoint].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        differences = [abs(float(a - b)) for a, b in combinations(values, 2)]
        pairwise.extend(differences)
        per_observation.append({
            "range": float(values.max() - values.min()),
            "sd": float(values.std(ddof=1)),
            "mean_pairwise_abs_diff": float(np.mean(differences)),
        })
    return {
        "multi_annotator_observations": len(per_observation),
        "pairwise_comparisons": len(pairwise),
        "pairwise_abs_difference": descriptive(np.asarray(pairwise)),
        "within_observation_range": descriptive(
            np.asarray([row["range"] for row in per_observation])
        ),
        "within_observation_sd": descriptive(
            np.asarray([row["sd"] for row in per_observation])
        ),
    }


def paired_difference(
    observations: pd.DataFrame,
    endpoint: str,
    comparator: str,
    block: str | None = None,
) -> np.ndarray:
    subset = observations
    if block is not None:
        subset = subset.loc[subset.time_block_role == block]
    pivot = subset.pivot(index="file_name", columns="model", values=endpoint)
    if comparator not in pivot or "v35" not in pivot:
        raise ValueError(f"missing model for paired comparison: {comparator}")
    values = (pivot[comparator] - pivot["v35"]).to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def normal_power(
    true_effect_above_margin: float,
    paired_sd: float,
    sample_size: int,
    alpha: float = 0.05,
) -> float:
    if paired_sd <= 0 or sample_size <= 0:
        raise ValueError("paired_sd and sample_size must be positive")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha)
    z_value = true_effect_above_margin * np.sqrt(sample_size) / paired_sd - z_alpha
    return float(normal.cdf(float(z_value)))


def minimum_detectable_effect(
    paired_sd: float,
    sample_size: int,
    power: float,
    alpha: float = 0.05,
) -> float:
    if not 0 < power < 1:
        raise ValueError("power must be between zero and one")
    normal = NormalDist()
    return float(
        (normal.inv_cdf(1 - alpha) + normal.inv_cdf(power))
        * paired_sd
        / np.sqrt(sample_size)
    )


def required_sample_size(
    true_effect_above_margin: float,
    paired_sd: float,
    power: float,
    alpha: float = 0.05,
) -> int:
    if true_effect_above_margin <= 0 or paired_sd <= 0:
        raise ValueError("effect and paired_sd must be positive")
    if not 0 < power < 1:
        raise ValueError("power must be between zero and one")
    normal = NormalDist()
    value = (
        (normal.inv_cdf(1 - alpha) + normal.inv_cdf(power))
        * paired_sd
        / true_effect_above_margin
    ) ** 2
    return int(np.ceil(value))


def operating_characteristics(
    paired_sd: float,
    sample_size: int,
    direction: str,
) -> dict[str, Any]:
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError(f"unknown endpoint direction: {direction}")
    return {
        "paired_sd": paired_sd,
        "sample_size": sample_size,
        "test": "one-sided normal approximation, alpha=0.05",
        "beneficial_paired_difference_direction": (
            "positive" if direction == "higher_is_better" else "negative"
        ),
        "mde_magnitude_beyond_registered_boundary": {
            "power_0.80": minimum_detectable_effect(paired_sd, sample_size, 0.80),
            "power_0.90": minimum_detectable_effect(paired_sd, sample_size, 0.90),
        },
        "power_if_registered_margin_zero_and_true_benefit_magnitude": {
            f"{effect:.3f}": normal_power(effect, paired_sd, sample_size)
            for effect in EFFECT_GRID
        },
        "required_sample_size_if_registered_margin_zero": {
            f"{effect:.3f}": {
                "power_0.80": required_sample_size(effect, paired_sd, 0.80),
                "power_0.90": required_sample_size(effect, paired_sd, 0.90),
            }
            for effect in EFFECT_GRID
        },
    }


def endpoint_calibration(
    raw_cases: pd.DataFrame,
    observations: pd.DataFrame,
    endpoint: str,
) -> dict[str, Any]:
    v35 = observations.loc[observations.model == "v35"]
    baseline = {}
    for label, subset in (
        ("all", v35),
        ("development", v35.loc[v35.time_block_role == "development"]),
        ("locked_evaluation", v35.loc[v35.time_block_role == "locked_evaluation"]),
    ):
        values = subset[endpoint].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        baseline[label] = {
            "descriptive": descriptive(values),
            "bootstrap_mean_95ci": bootstrap_mean_ci(values),
        }
    paired = {}
    scenarios = {}
    for comparator in ("raw", "v42"):
        for block in (None, "development", "locked_evaluation"):
            label = block or "all"
            values = paired_difference(observations, endpoint, comparator, block)
            key = f"{comparator}_minus_v35_{label}"
            paired[key] = {
                "descriptive": descriptive(values),
                "bootstrap_mean_95ci": bootstrap_mean_ci(values),
            }
            if block == "locked_evaluation" and values.size > 1:
                scenarios[key] = operating_characteristics(
                    float(values.std(ddof=1)), int(values.size), ENDPOINTS[endpoint]
                )
    locked_values = v35.loc[
        v35.time_block_role == "locked_evaluation", endpoint
    ].to_numpy(dtype=np.float64)
    locked_values = locked_values[np.isfinite(locked_values)]
    if locked_values.size > 1:
        scenarios["conservative_v35_locked_observation_sd"] = (
            operating_characteristics(
                float(locked_values.std(ddof=1)),
                int(locked_values.size),
                ENDPOINTS[endpoint],
            )
        )
    return {
        "direction": ENDPOINTS[endpoint],
        "v35_observation_level": baseline,
        "v35_annotator_repeatability": annotator_repeatability(raw_cases, endpoint),
        "empirical_paired_differences": paired,
        "locked_operating_characteristics": scenarios,
    }


def build_calibration(frame: pd.DataFrame, source_path: Path) -> dict[str, Any]:
    required = {
        "file_name",
        "model",
        "fold",
        "site_code",
        "year",
        "time_block_role",
        *ENDPOINTS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"science case metrics missing columns: {missing}")
    observations = observation_metrics(frame)
    unique_files = int(observations.file_name.nunique())
    locked_files = int(
        observations.loc[
            observations.time_block_role == "locked_evaluation", "file_name"
        ].nunique()
    )
    result = {
        "status": "calibration_only_margin_not_registered",
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "seed": SEED,
        "bootstrap_repeats": REPEATS,
        "aggregation_contract": (
            "Mean annotator variants within each observation/model first; then give each "
            "observation equal weight. Paired differences are comparator minus V35."
        ),
        "observations": unique_files,
        "locked_observations": locked_files,
        "s1_predictions_read": False,
        "endpoints": {
            endpoint: endpoint_calibration(frame, observations, endpoint)
            for endpoint in ENDPOINTS
        },
        "registration_constraints": [
            "Do not select a margin from statistical detectability alone.",
            "Use operational harm tolerance and measurement repeatability in addition.",
            "Keep official 2048-pixel PQ distinct from pq_512 in this calibration.",
            "Freeze margins after completed S0-4 review and before any S1 result is read.",
        ],
    }
    if unique_files != 707 or locked_files != 160:
        raise ValueError(
            f"unexpected observation counts: all={unique_files}, locked={locked_files}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    result = build_calibration(frame, args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "observations": result["observations"],
        "locked_observations": result["locked_observations"],
        "endpoints": list(result["endpoints"]),
        "output": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
