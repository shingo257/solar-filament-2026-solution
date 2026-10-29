"""Implementation helpers for `evaluate_s1_conditional_rescue_headroom` — extracted verbatim from `evaluate_s1_conditional_rescue_headroom.py`.

Holds the helper functions and module constants so `evaluate_s1_conditional_rescue_headroom.py` stays a readable
orchestrator. Loaded by the wrapper via sibling import.
"""


from __future__ import annotations


import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_INPUT = ROOT / "output/s0/science_case_metrics_v3.csv"


DEFAULT_CALIBRATION = ROOT / "output/s0/s1_endpoint_margin_calibration.json"


DEFAULT_OUTPUT = ROOT / "output/s0/s1_conditional_rescue_headroom.json"


SEED = 20260824


BOOTSTRAP_REPEATS = 5000


MIN_SELECTED_FRACTION = 0.10


ENDPOINTS = {
    "spine_coverage": "higher_is_better",
    "filament_recall": "higher_is_better",
    "filament_precision": "higher_is_better",
    "merge_count": "lower_is_better",
    "spine_coverage_matched": "higher_is_better",
    "count_abs_error": "lower_is_better",
    "pq_512": "higher_is_better",
}


PREDICTION_FEATURES = ("prediction_count", "prediction_components")


BASE_NUMERIC_FEATURES = (
    "v35_prediction_count",
    "v42_prediction_count",
    "delta_prediction_count",
    "v35_prediction_components",
    "v42_prediction_components",
    "delta_prediction_components",
)


FEATURE_SETS = {
    "mask_only": {
        "numeric": BASE_NUMERIC_FEATURES,
        "categorical": (),
    },
    "mask_plus_context": {
        "numeric": (*BASE_NUMERIC_FEATURES, "year"),
        "categorical": ("site_code",),
    },
}


FORBIDDEN_INFERENCE_FEATURES = {
    *ENDPOINTS,
    "labeler",
    "gt_count",
    "tp",
    "fp",
    "fn",
    "truth_components",
    "component_abs_error",
    "endpoint_abs_error",
    "branchpoint_abs_error",
    "gt_area_pixels_512",
    "annotation_area_median_2048",
    "bbox_center_radius_median",
    "bbox_center_radius_max",
    "area_group",
    "density_group",
    "radial_group",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def aggregate_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Average annotator endpoints while preserving prediction-only features."""
    required = {
        "file_name",
        "model",
        "fold",
        "site_code",
        "year",
        "time_block_role",
        *ENDPOINTS,
        *PREDICTION_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"science case metrics missing columns: {missing}")

    invariant = ("fold", "site_code", "year", "time_block_role", *PREDICTION_FEATURES)
    grouped = frame.groupby(["file_name", "model"], dropna=False)
    for column in invariant:
        maximum = int(grouped[column].nunique(dropna=False).max())
        if maximum != 1:
            raise ValueError(
                f"{column} varies across annotator rows within observation/model"
            )

    aggregations: dict[str, str] = {endpoint: "mean" for endpoint in ENDPOINTS}
    aggregations.update({column: "first" for column in invariant})
    result = (
        grouped.agg(aggregations)
        .reset_index()
        .sort_values(["file_name", "model"])
        .reset_index(drop=True)
    )
    return result


def build_analysis_table(observations: pd.DataFrame) -> pd.DataFrame:
    """Create one paired V35/V42 row per observation."""
    subset = observations.loc[observations.model.isin(["v35", "v42"])].copy()
    counts = subset.groupby("file_name").model.nunique()
    incomplete = counts.loc[counts != 2]
    if not incomplete.empty:
        raise ValueError(f"missing V35/V42 pairs for {len(incomplete)} observations")

    metadata_columns = ["fold", "site_code", "year", "time_block_role"]
    metadata = subset.loc[subset.model == "v35", ["file_name", *metadata_columns]].copy()
    other_metadata = subset.loc[
        subset.model == "v42", ["file_name", *metadata_columns]
    ].copy()
    checked = metadata.merge(other_metadata, on="file_name", suffixes=("_v35", "_v42"))
    for column in metadata_columns:
        if not (checked[f"{column}_v35"] == checked[f"{column}_v42"]).all():
            raise ValueError(f"V35/V42 metadata mismatch: {column}")

    table = metadata.reset_index(drop=True)
    for model in ("v35", "v42"):
        model_rows = subset.loc[subset.model == model].set_index("file_name")
        for endpoint in ENDPOINTS:
            table[f"{model}__{endpoint}"] = table.file_name.map(model_rows[endpoint])
        for feature in PREDICTION_FEATURES:
            table[f"{model}_{feature}"] = table.file_name.map(model_rows[feature])

    for feature in PREDICTION_FEATURES:
        table[f"delta_{feature}"] = (
            table[f"v42_{feature}"] - table[f"v35_{feature}"]
        )
    if table.file_name.nunique() != len(table):
        raise AssertionError("analysis table is not observation-unique")
    return table.sort_values("file_name").reset_index(drop=True)


def endpoint_delta(table: pd.DataFrame, endpoint: str) -> np.ndarray:
    return (
        table[f"v42__{endpoint}"] - table[f"v35__{endpoint}"]
    ).to_numpy(dtype=np.float64)


def oracle_masks(table: pd.DataFrame) -> dict[str, np.ndarray]:
    primary = endpoint_delta(table, "spine_coverage") > 0
    safe = (
        primary
        & (endpoint_delta(table, "filament_precision") >= 0)
        & (endpoint_delta(table, "merge_count") <= 0)
        & (endpoint_delta(table, "count_abs_error") <= 0)
    )
    return {"oracle_primary": primary, "oracle_catalog_safe": safe}


def audit_feature_set(specification: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    used = set(specification["numeric"]) | set(specification["categorical"])
    forbidden = sorted(used & FORBIDDEN_INFERENCE_FEATURES)
    if forbidden:
        raise ValueError(f"ground-truth-derived inference features requested: {forbidden}")
    return {
        "numeric": list(specification["numeric"]),
        "categorical": list(specification["categorical"]),
        "forbidden_feature_intersection": forbidden,
        "ground_truth_features_used_at_inference": False,
        "ground_truth_used_as_development_training_target": True,
    }


def make_estimator(specification: dict[str, tuple[str, ...]]) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    numeric = list(specification["numeric"])
    categorical = list(specification["categorical"])
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def fit_predict_probability(
    train: pd.DataFrame,
    train_target: np.ndarray,
    test: pd.DataFrame,
    specification: dict[str, tuple[str, ...]],
) -> np.ndarray:
    classes = np.unique(train_target)
    if classes.size == 1:
        return np.full(len(test), float(classes[0]), dtype=np.float64)
    estimator = make_estimator(specification)
    estimator.fit(train, train_target)
    return estimator.predict_proba(test)[:, 1].astype(np.float64)


def gate_target(table: pd.DataFrame, target_contract: str) -> np.ndarray:
    if target_contract == "primary_benefit":
        return endpoint_delta(table, "spine_coverage") > 0
    if target_contract == "catalog_safe_benefit":
        return oracle_masks(table)["oracle_catalog_safe"]
    raise ValueError(f"unknown gate target contract: {target_contract}")


def development_oof_and_locked_probabilities(
    table: pd.DataFrame,
    specification: dict[str, tuple[str, ...]],
    target_contract: str = "primary_benefit",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    development = table.loc[table.time_block_role == "development"].reset_index(drop=True)
    locked = table.loc[table.time_block_role == "locked_evaluation"].reset_index(drop=True)
    target = gate_target(development, target_contract)
    locked_target = gate_target(locked, target_contract)

    probabilities = np.full(len(development), np.nan, dtype=np.float64)
    for fold in sorted(development.fold.unique()):
        held_out = development.fold == fold
        training = ~held_out
        probabilities[held_out.to_numpy()] = fit_predict_probability(
            development.loc[training],
            target[training.to_numpy()],
            development.loc[held_out],
            specification,
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("development OOF probabilities are incomplete")
    locked_probabilities = fit_predict_probability(
        development,
        target,
        locked,
        specification,
    )
    return probabilities, locked_probabilities, target, locked_target


def classifier_diagnostics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": int(target.size),
        "positive": int(target.sum()),
        "positive_fraction": float(target.mean()),
        "probability_mean": float(probabilities.mean()),
        "probability_sd": float(probabilities.std(ddof=1)),
        "brier_score": float(brier_score_loss(target, probabilities)),
    }
    if np.unique(target).size == 2:
        result["roc_auc"] = float(roc_auc_score(target, probabilities))
        result["average_precision"] = float(
            average_precision_score(target, probabilities)
        )
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def threshold_row(
    table: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    selected = probabilities >= threshold
    fraction = float(selected.mean())
    deltas = {
        endpoint: float(np.nanmean(np.where(selected, endpoint_delta(table, endpoint), 0.0)))
        for endpoint in ENDPOINTS
    }
    feasible = bool(
        fraction >= MIN_SELECTED_FRACTION
        and deltas["filament_precision"] >= 0
        and deltas["merge_count"] <= 0
        and deltas["count_abs_error"] <= 0
    )
    return {
        "threshold": float(threshold),
        "selected": int(selected.sum()),
        "selected_fraction": fraction,
        "endpoint_mean_differences_vs_v35": deltas,
        "meets_development_catalog_zero_harm_constraints": feasible,
    }


def select_development_threshold(
    table: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select on development OOF outcomes only; locked outcomes are never passed."""
    if not (table.time_block_role == "development").all():
        raise ValueError("threshold selection accepts development rows only")
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scan = [threshold_row(table, probabilities, float(value)) for value in thresholds]
    feasible = [
        row
        for row in scan
        if row["meets_development_catalog_zero_harm_constraints"]
    ]
    if not feasible:
        selected = threshold_row(table, probabilities, 1.01)
        selected.update({
            "selection_status": "no_feasible_development_threshold_select_none",
            "selection_basis": "development_oof_only",
        })
        return selected, scan
    selected = max(
        feasible,
        key=lambda row: (
            row["endpoint_mean_differences_vs_v35"]["spine_coverage"],
            row["endpoint_mean_differences_vs_v35"]["filament_precision"],
            -row["endpoint_mean_differences_vs_v35"]["merge_count"],
            -row["endpoint_mean_differences_vs_v35"]["count_abs_error"],
            row["threshold"],
        ),
    ).copy()
    selected.update({
        "selection_status": "selected_on_development_oof",
        "selection_basis": "development_oof_only",
    })
    return selected, scan


def bootstrap_indices(n: int, repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(repeats, n), dtype=np.int32)


def policy_summary(
    table: pd.DataFrame,
    selected: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    if len(table) != len(selected):
        raise ValueError("policy mask length does not match table")
    endpoints = {}
    for endpoint, direction in ENDPOINTS.items():
        baseline = table[f"v35__{endpoint}"].to_numpy(dtype=np.float64)
        candidate = table[f"v42__{endpoint}"].to_numpy(dtype=np.float64)
        valid = np.isfinite(baseline) & (~selected | np.isfinite(candidate))
        delta = np.where(selected, candidate - baseline, 0.0)
        delta = delta.astype(np.float64)
        delta[~valid] = np.nan
        sampled = delta[indices].astype(np.float64)
        with np.errstate(invalid="ignore"):
            bootstrap_means = np.nanmean(sampled, axis=1)
        endpoints[endpoint] = {
            "direction": direction,
            "n_valid": int(valid.sum()),
            "mean_difference_vs_v35": float(np.nanmean(delta)),
            "bootstrap_95ci": {
                "low": float(np.nanquantile(bootstrap_means, 0.025)),
                "high": float(np.nanquantile(bootstrap_means, 0.975)),
            },
        }
    return {
        "observations": len(table),
        "selected_v42": int(selected.sum()),
        "selected_fraction": float(selected.mean()),
        "endpoints": endpoints,
    }


def calibration_mde(calibration: dict[str, Any]) -> float:
    return float(
        calibration["endpoints"]["spine_coverage"]
        ["locked_operating_characteristics"]
        ["v42_minus_v35_locked_evaluation"]
        ["mde_magnitude_beyond_registered_boundary"]["power_0.80"]
    )


def build_analysis(
    frame: pd.DataFrame,
    source_path: Path,
    calibration: dict[str, Any],
    calibration_path: Path,
) -> dict[str, Any]:
    observations = aggregate_observations(frame)
    table = build_analysis_table(observations)
    if len(table) != 707:
        raise ValueError(f"unexpected paired observation count: {len(table)}")

    development = table.loc[table.time_block_role == "development"].reset_index(drop=True)
    locked = table.loc[table.time_block_role == "locked_evaluation"].reset_index(drop=True)
    if len(development) != 547 or len(locked) != 160:
        raise ValueError(
            f"unexpected time blocks: development={len(development)}, locked={len(locked)}"
        )
    development_bootstrap = bootstrap_indices(
        len(development), BOOTSTRAP_REPEATS, SEED
    )
    locked_bootstrap = bootstrap_indices(
        len(locked), BOOTSTRAP_REPEATS, SEED + 1
    )

    oracle = oracle_masks(table)
    fixed_masks = {
        "v35_baseline": np.zeros(len(table), dtype=bool),
        "v42_all": np.ones(len(table), dtype=bool),
        **oracle,
    }
    policies: dict[str, Any] = {}
    for name, mask in fixed_masks.items():
        policies[name] = {
            "development": policy_summary(
                development,
                mask[table.time_block_role == "development"],
                development_bootstrap,
            ),
            "locked_evaluation": policy_summary(
                locked,
                mask[table.time_block_role == "locked_evaluation"],
                locked_bootstrap,
            ),
        }

    gates = {}
    for name, specification in FEATURE_SETS.items():
        feature_audit = audit_feature_set(specification)
        dev_probability, locked_probability, dev_target, locked_target = (
            development_oof_and_locked_probabilities(table, specification)
        )
        selected_threshold, scan = select_development_threshold(
            development, dev_probability
        )
        threshold = float(selected_threshold["threshold"])
        dev_selected = dev_probability >= threshold
        locked_selected = locked_probability >= threshold
        gates[name] = {
            "feature_audit": feature_audit,
            "model": (
                "class-weighted logistic regression; three-fold development OOF "
                "probabilities; full-development refit for locked prediction"
            ),
            "development_classifier_diagnostics": classifier_diagnostics(
                dev_target, dev_probability
            ),
            "locked_retrospective_classifier_diagnostics": classifier_diagnostics(
                locked_target, locked_probability
            ),
            "selected_threshold": selected_threshold,
            "development_threshold_scan": scan,
            "policy": {
                "development": policy_summary(
                    development, dev_selected, development_bootstrap
                ),
                "locked_evaluation_retrospective": policy_summary(
                    locked, locked_selected, locked_bootstrap
                ),
            },
        }

    mde = calibration_mde(calibration)
    comparison = {
        "locked_80pct_mde_beyond_unregistered_boundary": mde,
        "interpretation": (
            "A policy mean below this value is not automatically useless; this is an "
            "approximate detectability reference, not an operational margin."
        ),
        "policies": {},
    }
    for name, result in policies.items():
        delta = result["locked_evaluation"]["endpoints"]["spine_coverage"]
        comparison["policies"][name] = {
            "locked_primary_mean_difference": delta["mean_difference_vs_v35"],
            "locked_primary_bootstrap_95ci": delta["bootstrap_95ci"],
            "fraction_of_mde": delta["mean_difference_vs_v35"] / mde,
        }
    for name, result in gates.items():
        delta = result["policy"]["locked_evaluation_retrospective"]["endpoints"][
            "spine_coverage"
        ]
        comparison["policies"][f"observable_gate_{name}"] = {
            "locked_primary_mean_difference": delta["mean_difference_vs_v35"],
            "locked_primary_bootstrap_95ci": delta["bootstrap_95ci"],
            "fraction_of_mde": delta["mean_difference_vs_v35"] / mde,
        }

    return {
        "status": "retrospective_feasibility_only_not_registered",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "seed": SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "aggregation_contract": (
            "Average annotator variants within observation/model first; give each "
            "observation equal weight; all differences are selected-policy minus V35."
        ),
        "scope": {
            "observations": len(table),
            "development_observations": len(development),
            "locked_observations": len(locked),
            "development_years": sorted(development.year.unique().tolist()),
            "locked_years": sorted(locked.year.unique().tolist()),
            "unit_of_gating": "whole observation, not individual proposal",
            "s1_predictions_read": False,
            "blinded_review_responses_read": False,
            "locked_role": (
                "retrospective transfer diagnostic only; prior S0 analyses already "
                "inspected this block, so it is not an independent holdout"
            ),
        },
        "development_threshold_contract": {
            "minimum_selected_fraction": MIN_SELECTED_FRACTION,
            "constraints": [
                "filament_precision difference >= 0",
                "merge_count difference <= 0",
                "count_abs_error difference <= 0",
            ],
            "objective": "maximize spine_coverage mean difference vs V35",
            "data_used": "development OOF outcomes only",
        },
        "fixed_policy_ceilings": policies,
        "observable_gates": gates,
        "mde_reference": comparison,
        "limitations": [
            "Oracle policies use ground truth and are non-deployable ceilings.",
            "Observable gates require running both V35 and V42 before selection.",
            "The gate acts at observation level and cannot establish proposal-level S1 value.",
            "Threshold scanning makes development performance optimistic.",
            "Locked results are retrospective because the block was inspected previously.",
            "pq_512 is an observation-level proxy, not official 2048-pixel micro PQ.",
            "No S1 model, registration margin, GPU training, or review-derived label is used.",
        ],
    }
