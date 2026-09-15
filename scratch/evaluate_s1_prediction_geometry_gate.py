"""Evaluate a GT-free V35/V42 selector using cached prediction geometry.

The feature extractor reads only V35/V42 prediction masks and operational
metadata.  Development OOF outcomes fit/select the gate; the locked time block
is a retrospective transfer diagnostic, not a fresh holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.evaluate_s1_conditional_rescue_headroom import (
    BASE_NUMERIC_FEATURES,
    BOOTSTRAP_REPEATS,
    DEFAULT_CALIBRATION,
    DEFAULT_INPUT,
    ROOT,
    SEED,
    aggregate_observations,
    audit_feature_set,
    bootstrap_indices,
    build_analysis_table,
    calibration_mde,
    classifier_diagnostics,
    development_oof_and_locked_probabilities,
    policy_summary,
    select_development_threshold,
    sha256_file,
)

DEFAULT_CACHE_ROOT = ROOT / "output/s0/prediction_cache_512"
DEFAULT_FEATURES_OUTPUT = ROOT / "output/s0/s1_prediction_geometry_features.csv"
DEFAULT_OUTPUT = ROOT / "output/s0/s1_prediction_geometry_gate.json"
CACHE_VERSION = "s0-science-prediction-v2"
EVAL_SIZE = 512

PER_MODEL_METRICS = (
    "sum_area_fraction",
    "union_area_fraction",
    "overlap_area_fraction",
    "mean_area_fraction",
    "median_area_fraction",
    "max_area_fraction",
    "area_cv",
    "mean_bbox_fill",
    "mean_log_elongation",
    "mean_log_thinness",
)
CROSS_MODEL_METRICS = (
    "union_iou",
    "v42_added_area_fraction",
    "v42_removed_area_fraction",
    "mean_v42_best_iou",
    "mean_v35_best_iou",
    "v42_unmatched_fraction_iou_lt_0_30",
    "v35_unmatched_fraction_iou_lt_0_30",
)
GEOMETRY_FEATURES = (
    *BASE_NUMERIC_FEATURES,
    *(f"v35_{metric}" for metric in PER_MODEL_METRICS),
    *(f"v42_{metric}" for metric in PER_MODEL_METRICS),
    *CROSS_MODEL_METRICS,
)
FEATURE_SETS = {
    "prediction_counts": {
        "numeric": BASE_NUMERIC_FEATURES,
        "categorical": (),
    },
    "prediction_counts_plus_context": {
        "numeric": (*BASE_NUMERIC_FEATURES, "year"),
        "categorical": ("site_code",),
    },
    "prediction_geometry": {
        "numeric": GEOMETRY_FEATURES,
        "categorical": (),
    },
    "prediction_geometry_plus_context": {
        "numeric": (*GEOMETRY_FEATURES, "year"),
        "categorical": ("site_code",),
    },
}
TARGET_CONTRACTS = {
    "primary_benefit": "V42 spine_coverage minus V35 is positive",
    "catalog_safe_benefit": (
        "V42 spine_coverage minus V35 is positive, filament_precision is "
        "nonnegative, merge_count is nonpositive, and count_abs_error is nonpositive"
    ),
}




def load_cached_masks(path: Path, model: str, size: int = EVAL_SIZE) -> list[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        version = str(data["version"].item())
        stored_model = str(data["model"].item())
        stored_size = int(data["size"].item())
        if version != CACHE_VERSION or stored_model != model or stored_size != size:
            raise ValueError(
                f"cache identity mismatch: {path} "
                f"version={version} model={stored_model} size={stored_size}"
            )
        packed = data["packed"]
        if packed.size == 0:
            return []
        return [
            np.unpackbits(item, axis=-1)[:, :size].astype(bool)
            for item in packed
        ]


def mask_perimeter_4(mask: np.ndarray) -> int:
    horizontal = int(np.count_nonzero(mask[:, 1:] != mask[:, :-1]))
    vertical = int(np.count_nonzero(mask[1:, :] != mask[:-1, :]))
    edges = int(mask[:, 0].sum() + mask[:, -1].sum())
    edges += int(mask[0, :].sum() + mask[-1, :].sum())
    return horizontal + vertical + edges


def mask_shape_features(mask: np.ndarray) -> tuple[float, float, float, float]:
    y, x = np.nonzero(mask)
    area = float(x.size)
    if area == 0:
        return 0.0, 0.0, 0.0, 0.0
    width = int(x.max() - x.min() + 1)
    height = int(y.max() - y.min() + 1)
    bbox_fill = area / float(width * height)

    centered_x = x.astype(np.float64) - float(x.mean())
    centered_y = y.astype(np.float64) - float(y.mean())
    xx = float(np.mean(centered_x * centered_x))
    yy = float(np.mean(centered_y * centered_y))
    xy = float(np.mean(centered_x * centered_y))
    trace = xx + yy
    discriminant = max((xx - yy) ** 2 + 4.0 * xy * xy, 0.0) ** 0.5
    major = max((trace + discriminant) / 2.0, 0.0)
    minor = max((trace - discriminant) / 2.0, 0.0)
    elongation = np.sqrt((major + 1.0) / (minor + 1.0))

    perimeter = float(mask_perimeter_4(mask))
    thinness = perimeter * perimeter / max(4.0 * np.pi * area, 1.0)
    return area, bbox_fill, float(np.log1p(elongation)), float(np.log1p(thinness))


def mask_set_features(masks: list[np.ndarray], size: int = EVAL_SIZE) -> dict[str, float]:
    pixels = float(size * size)
    if not masks:
        return {metric: 0.0 for metric in PER_MODEL_METRICS}
    stacked = np.stack(masks).astype(bool)
    union = np.any(stacked, axis=0)
    shape_rows = [mask_shape_features(mask) for mask in masks]
    areas = np.asarray([row[0] for row in shape_rows], dtype=np.float64)
    bbox_fill = np.asarray([row[1] for row in shape_rows], dtype=np.float64)
    log_elongation = np.asarray([row[2] for row in shape_rows], dtype=np.float64)
    log_thinness = np.asarray([row[3] for row in shape_rows], dtype=np.float64)
    sum_area = float(areas.sum())
    union_area = float(union.sum())
    return {
        "sum_area_fraction": sum_area / pixels,
        "union_area_fraction": union_area / pixels,
        "overlap_area_fraction": max(sum_area - union_area, 0.0) / pixels,
        "mean_area_fraction": float(areas.mean()) / pixels,
        "median_area_fraction": float(np.median(areas)) / pixels,
        "max_area_fraction": float(areas.max()) / pixels,
        "area_cv": float(areas.std(ddof=0) / max(areas.mean(), 1.0)),
        "mean_bbox_fill": float(bbox_fill.mean()),
        "mean_log_elongation": float(log_elongation.mean()),
        "mean_log_thinness": float(log_thinness.mean()),
    }


def prediction_iou_matrix(
    first: list[np.ndarray], second: list[np.ndarray]
) -> np.ndarray:
    result = np.zeros((len(first), len(second)), dtype=np.float64)
    for row, left in enumerate(first):
        for column, right in enumerate(second):
            union = np.logical_or(left, right).sum()
            if union:
                result[row, column] = np.logical_and(left, right).sum() / union
    return result


def cross_model_features(
    v35_masks: list[np.ndarray],
    v42_masks: list[np.ndarray],
    size: int = EVAL_SIZE,
) -> dict[str, float]:
    shape = (size, size)
    v35_union = np.any(np.stack(v35_masks), axis=0) if v35_masks else np.zeros(shape, bool)
    v42_union = np.any(np.stack(v42_masks), axis=0) if v42_masks else np.zeros(shape, bool)
    union = np.logical_or(v35_union, v42_union).sum()
    intersection = np.logical_and(v35_union, v42_union).sum()
    pixels = float(size * size)
    pairwise = prediction_iou_matrix(v35_masks, v42_masks)
    v35_best = pairwise.max(axis=1) if len(v42_masks) else np.zeros(len(v35_masks))
    v42_best = pairwise.max(axis=0) if len(v35_masks) else np.zeros(len(v42_masks))
    return {
        "union_iou": float(intersection / union) if union else 1.0,
        "v42_added_area_fraction": float(np.logical_and(v42_union, ~v35_union).sum())
        / pixels,
        "v42_removed_area_fraction": float(np.logical_and(v35_union, ~v42_union).sum())
        / pixels,
        "mean_v42_best_iou": float(v42_best.mean()) if v42_best.size else 0.0,
        "mean_v35_best_iou": float(v35_best.mean()) if v35_best.size else 0.0,
        "v42_unmatched_fraction_iou_lt_0_30": float((v42_best < 0.30).mean())
        if v42_best.size
        else 0.0,
        "v35_unmatched_fraction_iou_lt_0_30": float((v35_best < 0.30).mean())
        if v35_best.size
        else 0.0,
    }


def cache_path(cache_root: Path, model: str, fold: int, file_name: str) -> Path:
    return cache_root / f"{model}_fold_{fold}" / f"{Path(file_name).stem}.npz"


def add_prediction_geometry(
    table: pd.DataFrame,
    cache_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    enriched = table.copy()
    rows: list[dict[str, float]] = []
    count_mismatches = []
    loaded = 0
    for record in enriched.itertuples(index=False):
        model_masks = {}
        for model in ("v35", "v42"):
            path = cache_path(cache_root, model, int(record.fold), record.file_name)
            masks = load_cached_masks(path, model)
            loaded += 1
            expected_count = int(getattr(record, f"{model}_prediction_count"))
            if len(masks) != expected_count:
                count_mismatches.append({
                    "file_name": record.file_name,
                    "model": model,
                    "cache_count": len(masks),
                    "metrics_count": expected_count,
                })
            model_masks[model] = masks
        features: dict[str, float] = {}
        for model in ("v35", "v42"):
            features.update({
                f"{model}_{name}": value
                for name, value in mask_set_features(model_masks[model]).items()
            })
        features.update(cross_model_features(model_masks["v35"], model_masks["v42"]))
        rows.append(features)
    if count_mismatches:
        raise ValueError(f"prediction cache count mismatches: {count_mismatches[:5]}")
    geometry = pd.DataFrame(rows)
    enriched = pd.concat([enriched.reset_index(drop=True), geometry], axis=1)
    if not np.isfinite(enriched[list(GEOMETRY_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("non-finite geometry feature")
    audit = {
        "cache_root": str(cache_root),
        "expected_files": len(enriched) * 2,
        "loaded_files": loaded,
        "count_mismatches": count_mismatches,
        "cache_version": CACHE_VERSION,
        "eval_size": EVAL_SIZE,
    }
    return enriched, audit


def constraint_diagnostics(scan: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in scan if row["selected_fraction"] >= 0.10]
    keys = {
        "precision_noninferior": lambda e: e["filament_precision"] >= 0,
        "merge_noninferior": lambda e: e["merge_count"] <= 0,
        "count_error_noninferior": lambda e: e["count_abs_error"] <= 0,
    }
    counts = {
        name: sum(predicate(row["endpoint_mean_differences_vs_v35"]) for row in eligible)
        for name, predicate in keys.items()
    }
    best = (
        max(
            eligible,
            key=lambda row: row["endpoint_mean_differences_vs_v35"]["spine_coverage"],
        )
        if eligible
        else None
    )
    return {
        "thresholds_with_minimum_selection": len(eligible),
        "thresholds_meeting_each_constraint": counts,
        "best_primary_without_catalog_constraints": best,
    }


def build_geometry_analysis(
    frame: pd.DataFrame,
    source_path: Path,
    calibration: dict[str, Any],
    calibration_path: Path,
    cache_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    table = build_analysis_table(aggregate_observations(frame))
    enriched, cache_audit = add_prediction_geometry(table, cache_root)
    development = enriched.loc[
        enriched.time_block_role == "development"
    ].reset_index(drop=True)
    locked = enriched.loc[
        enriched.time_block_role == "locked_evaluation"
    ].reset_index(drop=True)
    if len(enriched) != 707 or len(development) != 547 or len(locked) != 160:
        raise ValueError("unexpected observation/time-block counts")
    dev_bootstrap = bootstrap_indices(len(development), BOOTSTRAP_REPEATS, SEED)
    locked_bootstrap = bootstrap_indices(len(locked), BOOTSTRAP_REPEATS, SEED + 1)

    gates = {}
    for feature_name, specification in FEATURE_SETS.items():
        audit = audit_feature_set(specification)
        for target_name, target_definition in TARGET_CONTRACTS.items():
            name = f"{feature_name}__target_{target_name}"
            dev_probability, locked_probability, dev_target, locked_target = (
                development_oof_and_locked_probabilities(
                    enriched, specification, target_name
                )
            )
            selected_threshold, scan = select_development_threshold(
                development, dev_probability
            )
            threshold = float(selected_threshold["threshold"])
            gates[name] = {
                "feature_audit": audit,
                "target_contract": {
                    "name": target_name,
                    "positive_definition": target_definition,
                },
                "model": (
                    "class-weighted logistic regression; three-fold development OOF; "
                    "full-development refit for locked prediction"
                ),
                "development_classifier_diagnostics": classifier_diagnostics(
                    dev_target, dev_probability
                ),
                "locked_retrospective_classifier_diagnostics": classifier_diagnostics(
                    locked_target, locked_probability
                ),
                "selected_threshold": selected_threshold,
                "constraint_diagnostics": constraint_diagnostics(scan),
                "development_threshold_scan": scan,
                "policy": {
                    "development": policy_summary(
                        development, dev_probability >= threshold, dev_bootstrap
                    ),
                    "locked_evaluation_retrospective": policy_summary(
                        locked, locked_probability >= threshold, locked_bootstrap
                    ),
                },
            }

    mde = calibration_mde(calibration)
    mde_comparison = {}
    for name, gate in gates.items():
        primary = gate["policy"]["locked_evaluation_retrospective"]["endpoints"][
            "spine_coverage"
        ]
        mde_comparison[name] = {
            "mean_difference_vs_v35": primary["mean_difference_vs_v35"],
            "bootstrap_95ci": primary["bootstrap_95ci"],
            "fraction_of_locked_80pct_mde": primary["mean_difference_vs_v35"] / mde,
        }

    safe_feature_columns = [
        "file_name",
        "fold",
        "site_code",
        "year",
        "time_block_role",
        *GEOMETRY_FEATURES,
    ]
    result = {
        "status": "retrospective_geometry_gate_feasibility_only_not_registered",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "cache_audit": cache_audit,
        "observations": len(enriched),
        "development_observations": len(development),
        "locked_observations": len(locked),
        "feature_artifact_contract": (
            "The companion CSV contains only prediction-mask geometry and operational "
            "metadata; no ground truth endpoint or review response."
        ),
        "gates": gates,
        "mde_reference": {
            "locked_80pct_mde_beyond_unregistered_boundary": mde,
            "gates": mde_comparison,
        },
        "s1_predictions_read": False,
        "blinded_review_responses_read": False,
        "limitations": [
            "The gate requires both V35 and V42 mask generation before selection.",
            "Features are observation-level and do not localize proposal-level rescue.",
            "Development threshold scanning is optimistic even with OOF probabilities.",
            "Locked results are retrospective, not independent confirmation.",
            "No registration margin is selected by this experiment.",
        ],
    }
    return result, enriched[safe_feature_columns]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES_OUTPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    result, safe_features = build_geometry_analysis(
        frame, args.input, calibration, args.calibration, args.cache_root
    )
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    safe_features.to_csv(args.features_out, index=False)
    result["feature_artifact"] = str(args.features_out)
    result["feature_artifact_sha256"] = sha256_file(args.features_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "observations": result["observations"],
        "loaded_cache_files": result["cache_audit"]["loaded_files"],
        "gates": list(result["gates"]),
        "features": str(args.features_out),
        "output": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
