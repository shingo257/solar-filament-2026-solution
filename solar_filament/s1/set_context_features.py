"""Build GT-free observation/set-context features for the fixed S1 pilot.

The builder loads only inference-safe candidate columns, the public development
selection, and cached refined masks. It never loads target or endpoint values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solar_filament.s1.set_context_masks import (  # noqa: F401
    CACHE_SCHEMA,
    IMAGE_SHAPE,
    MIN_AREA,
    OWNERSHIP_THRESHOLD,
    _containment_subset,
    _zscore,
    cache_path,
    load_refined_cache,
    ownership_trace,
    pairwise_intersections,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION = ROOT / "output" / "s0" / "s1_proposal_marginal_utility_selection.csv"
DEFAULT_CANDIDATES = ROOT / "output" / "s0" / "v42_nested_candidate_predictions.csv"
DEFAULT_CACHE_ROOT = ROOT / "output" / "s0" / "s1_proposal_marginal_refined_cache"
DEFAULT_FEATURES_OUT = ROOT / "output" / "s0" / "s1_set_context_features.csv"
DEFAULT_OUT = ROOT / "output" / "s0" / "s1_set_context_features.json"

IDENTIFIER_COLUMNS = ("file_name", "fold", "candidate_index")
BASE_FEATURE_COLUMNS = (
    "detector_score",
    "source_count",
    "log_area",
    "bbox_extent",
    "log_aspect",
    "perimeter_over_sqrt_area",
    "solidity",
    "center_radius",
    "semantic_mean",
    "semantic_q10",
    "local_contrast",
)
SELECTION_COLUMNS = (
    "file_name",
    "fold",
    "site_code",
    "year",
    "time_block_role",
    "selection_hash",
)
CONTEXT_FEATURE_COLUMNS = (
    "proposal_count",
    "full_output_count",
    "without_output_count",
    "output_count_delta",
    "full_output_per_proposal_ratio",
    "detector_score_observation_mean",
    "detector_score_observation_std",
    "detector_score_observation_z",
    "detector_score_observation_percentile",
    "semantic_mean_observation_mean",
    "semantic_mean_observation_std",
    "semantic_mean_observation_z",
    "semantic_mean_observation_percentile",
    "source_count_observation_percentile",
    "refined_area_fraction",
    "refined_area_observation_percentile",
    "overlap_neighbor_count",
    "pairwise_iou_max",
    "pairwise_iou_mean_positive",
    "pairwise_iou_sum",
    "self_covered_by_other_max",
    "other_covered_by_self_max",
    "containment_conflict_count",
    "dropped_by_containment",
    "dropped_by_min_area_after_ownership",
    "ownership_output_kept",
    "ownership_exclusive_area_fraction",
    "ownership_retained_fraction",
    "ownership_order_percentile",
)
FORBIDDEN_OUTPUT_TOKENS = (
    "target",
    "hit_iou",
    "variant",
    "marginal",
    "verifier",
    "selected_l2",
    "threshold",
    "delta_pq",
    "delta_spine",
    "delta_filament",
    "delta_merge",
    "delta_count",
)


def build_observation_features(
    candidate_rows: pd.DataFrame,
    masks: list[np.ndarray],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    rows = candidate_rows.sort_values("candidate_index").reset_index(drop=True).copy()
    indices = rows["candidate_index"].to_numpy(dtype=int)
    if not np.array_equal(indices, np.arange(len(rows))):
        raise ValueError(f"non-contiguous candidate indices: {metadata['file_name']}")
    if len(rows) != len(masks):
        raise ValueError(
            f"candidate/cache count mismatch {metadata['file_name']}: {len(rows)} != {len(masks)}"
        )
    if not len(rows):
        raise ValueError(f"empty observation: {metadata['file_name']}")

    for column in BASE_FEATURE_COLUMNS:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        if not np.isfinite(rows[column].to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite {column}: {metadata['file_name']}")

    intersections = pairwise_intersections(masks)
    areas = np.diag(intersections).astype(float)
    union = areas[:, None] + areas[None, :] - intersections
    iou = np.divide(
        intersections,
        union,
        out=np.zeros_like(intersections, dtype=float),
        where=union > 0,
    )
    np.fill_diagonal(iou, 0.0)
    self_covered = np.divide(
        intersections,
        areas[:, None],
        out=np.zeros_like(intersections, dtype=float),
        where=areas[:, None] > 0,
    )
    other_covered = np.divide(
        intersections,
        areas[None, :],
        out=np.zeros_like(intersections, dtype=float),
        where=areas[None, :] > 0,
    )
    np.fill_diagonal(self_covered, 0.0)
    np.fill_diagonal(other_covered, 0.0)

    full = ownership_trace(masks, intersections)
    full_output_count = len(full["kept_indices"])
    without_counts = []
    for candidate_index in range(len(rows)):
        active = [index for index in range(len(rows)) if index != candidate_index]
        without_counts.append(len(ownership_trace(masks, intersections, active)["kept_indices"]))

    output = rows[list(IDENTIFIER_COLUMNS) + list(BASE_FEATURE_COLUMNS)].copy()
    output["site_code"] = str(metadata["site_code"])
    output["year"] = int(metadata["year"])
    output["time_block_role"] = str(metadata["time_block_role"])
    output["selection_hash"] = str(metadata["selection_hash"])
    output["proposal_count"] = len(rows)
    output["full_output_count"] = full_output_count
    output["without_output_count"] = without_counts
    output["output_count_delta"] = full_output_count - output["without_output_count"]
    output["full_output_per_proposal_ratio"] = full_output_count / len(rows)

    for column in ("detector_score", "semantic_mean"):
        mean, std, z = _zscore(rows[column])
        output[f"{column}_observation_mean"] = mean
        output[f"{column}_observation_std"] = std
        output[f"{column}_observation_z"] = z
        output[f"{column}_observation_percentile"] = rows[column].rank(
            method="average", pct=True
        )
    output["source_count_observation_percentile"] = rows["source_count"].rank(
        method="average", pct=True
    )
    output["refined_area_fraction"] = areas / float(IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    output["refined_area_observation_percentile"] = pd.Series(areas).rank(
        method="average", pct=True
    )
    output["overlap_neighbor_count"] = np.maximum(
        0, (intersections > 0).sum(axis=1) - 1
    )
    output["pairwise_iou_max"] = iou.max(axis=1)
    output["pairwise_iou_sum"] = iou.sum(axis=1)
    positive_counts = (iou > 0.0).sum(axis=1)
    output["pairwise_iou_mean_positive"] = np.divide(
        iou.sum(axis=1),
        positive_counts,
        out=np.zeros(len(rows), dtype=float),
        where=positive_counts > 0,
    )
    output["self_covered_by_other_max"] = self_covered.max(axis=1)
    output["other_covered_by_self_max"] = other_covered.max(axis=1)
    conflict = np.maximum(self_covered, other_covered) >= OWNERSHIP_THRESHOLD
    output["containment_conflict_count"] = conflict.sum(axis=1)
    output["dropped_by_containment"] = [
        int(index in full["dropped_by_containment"]) for index in range(len(rows))
    ]
    output["dropped_by_min_area_after_ownership"] = [
        int(index in full["dropped_by_min_area"]) for index in range(len(rows))
    ]
    output["ownership_output_kept"] = [
        int(index in full["kept_indices"]) for index in range(len(rows))
    ]
    output["ownership_exclusive_area_fraction"] = (
        full["exclusive_areas"] / float(IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    )
    output["ownership_retained_fraction"] = np.divide(
        full["exclusive_areas"],
        areas,
        out=np.zeros(len(rows), dtype=float),
        where=areas > 0,
    )
    output["ownership_order_percentile"] = np.where(
        full["output_order"] >= 0,
        (full["output_order"] + 1) / max(full_output_count, 1),
        0.0,
    )
    return output


def validate_output(frame: pd.DataFrame) -> None:
    expected = set(IDENTIFIER_COLUMNS) | set(BASE_FEATURE_COLUMNS) | set(
        CONTEXT_FEATURE_COLUMNS
    ) | {"site_code", "year", "time_block_role", "selection_hash"}
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(f"missing output columns: {missing}")
    forbidden = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden target-derived output columns: {forbidden}")
    if frame.duplicated(list(IDENTIFIER_COLUMNS)).any():
        raise ValueError("duplicate output candidate identities")
    if set(frame["time_block_role"].astype(str)) != {"development"}:
        raise ValueError("non-development row entered feature output")
    numeric = frame[list(BASE_FEATURE_COLUMNS) + list(CONTEXT_FEATURE_COLUMNS)]
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("missing or non-finite feature values")


def load_inputs(selection_path: Path, candidates_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = pd.read_csv(selection_path, usecols=list(SELECTION_COLUMNS))
    candidates = pd.read_csv(
        candidates_path,
        usecols=list(IDENTIFIER_COLUMNS) + list(BASE_FEATURE_COLUMNS),
    )
    if set(selection["time_block_role"].astype(str)) != {"development"}:
        raise ValueError("selection is not development-only")
    selected_names = set(selection["file_name"].astype(str))
    candidates = candidates.loc[candidates["file_name"].astype(str).isin(selected_names)].copy()
    if set(candidates["file_name"].astype(str)) != selected_names:
        missing = sorted(selected_names.difference(candidates["file_name"].astype(str)))
        raise ValueError(f"selected observations missing candidates: {missing}")
    return selection, candidates


def build_feature_table(
    selection: pd.DataFrame,
    candidates: pd.DataFrame,
    cache_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    cache_manifest = []
    for selected in selection.sort_values(["fold", "file_name"]).itertuples(index=False):
        file_name = str(selected.file_name)
        fold = int(selected.fold)
        path = cache_path(cache_root, fold, file_name)
        masks = load_refined_cache(path, file_name, fold)
        rows = candidates.loc[
            (candidates["file_name"].astype(str) == file_name)
            & (pd.to_numeric(candidates["fold"], errors="raise").astype(int) == fold)
        ]
        metadata = {
            "file_name": file_name,
            "site_code": selected.site_code,
            "year": selected.year,
            "time_block_role": selected.time_block_role,
            "selection_hash": selected.selection_hash,
        }
        frames.append(build_observation_features(rows, masks, metadata))
        cache_manifest.append({
            "file_name": file_name,
            "fold": fold,
            "candidate_count": len(masks),
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        })
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["fold", "file_name", "candidate_index"]
    ).reset_index(drop=True)
    validate_output(result)
    return result, cache_manifest


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "candidate_rows": int(len(frame)),
        "observations": int(frame["file_name"].nunique()),
        "base_feature_count": len(BASE_FEATURE_COLUMNS),
        "context_feature_count": len(CONTEXT_FEATURE_COLUMNS),
        "missing_feature_values": int(
            frame[list(BASE_FEATURE_COLUMNS) + list(CONTEXT_FEATURE_COLUMNS)].isna().sum().sum()
        ),
        "non_finite_feature_values": int(
            (~np.isfinite(
                frame[list(BASE_FEATURE_COLUMNS) + list(CONTEXT_FEATURE_COLUMNS)].to_numpy(
                    dtype=float
                )
            )).sum()
        ),
        "ownership": {
            "output_kept_candidates": int(frame["ownership_output_kept"].sum()),
            "dropped_by_containment": int(frame["dropped_by_containment"].sum()),
            "dropped_by_min_area_after_ownership": int(
                frame["dropped_by_min_area_after_ownership"].sum()
            ),
            "output_count_delta_distribution": {
                str(int(key)): int(value)
                for key, value in frame["output_count_delta"].value_counts().sort_index().items()
            },
        },
        "competition_geometry": {
            "candidates_with_overlap_neighbor": int((frame["overlap_neighbor_count"] > 0).sum()),
            "candidates_with_containment_conflict": int(
                (frame["containment_conflict_count"] > 0).sum()
            ),
            "ownership_retained_fraction_median": float(
                frame["ownership_retained_fraction"].median()
            ),
            "ownership_retained_fraction_min": float(
                frame["ownership_retained_fraction"].min()
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection_path = args.selection.resolve()
    candidates_path = args.candidates.resolve()
    cache_root = args.cache_root.resolve()
    features_out = args.features_out.resolve()
    out_path = args.out.resolve()
    selection, candidates = load_inputs(selection_path, candidates_path)
    features, cache_manifest = build_feature_table(selection, candidates, cache_root)

    features_out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_out, index=False)
    cache_manifest_hash = hashlib.sha256(
        "\n".join(
            f"{item['fold']}:{item['file_name']}:{item['sha256']}" for item in cache_manifest
        ).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "status": "fixed_gt_free_set_context_features_not_model_not_threshold_not_registered",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "scope": {
            "time_block_role": "development only",
            "locked_rows_read": 0,
            "ground_truth_read": False,
            "target_columns_loaded": False,
            "s1_predictions_read": False,
            "blinded_review_responses_read": False,
            "private_key_read": False,
            "model_fitting_performed": False,
            "threshold_selection_performed": False,
            "frozen_ownership_replayed": True,
        },
        "feature_contract": {
            "image_shape": list(IMAGE_SHAPE),
            "minimum_exclusive_area": MIN_AREA,
            "frozen_containment_threshold": OWNERSHIP_THRESHOLD,
            "identifiers": list(IDENTIFIER_COLUMNS),
            "base_inference_features": list(BASE_FEATURE_COLUMNS),
            "gt_free_set_context_features": list(CONTEXT_FEATURE_COLUMNS),
            "forbidden_output_tokens": list(FORBIDDEN_OUTPUT_TOKENS),
        },
        "summary": summarize(features),
        "inputs": {
            "selection": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "candidate_table": str(candidates_path),
            "candidate_table_sha256": sha256_file(candidates_path),
            "candidate_columns_loaded": list(IDENTIFIER_COLUMNS) + list(BASE_FEATURE_COLUMNS),
            "refined_cache_root": str(cache_root),
            "refined_cache_manifest": cache_manifest,
            "refined_cache_manifest_sha256": cache_manifest_hash,
        },
        "artifacts": {
            "features": str(features_out),
            "features_sha256": sha256_file(features_out),
        },
        "limitations": [
            "Features are fixed-pilot engineering evidence, not classifier performance.",
            "No feature-target association, model fit, threshold, or margin was evaluated.",
            "Full-development generation remains blocked on S0-4 review and registered margins.",
            "Set-context features must remain observation-grouped in all downstream splits.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "observations": payload["summary"]["observations"],
                "candidates": payload["summary"]["candidate_rows"],
                "features": len(BASE_FEATURE_COLUMNS) + len(CONTEXT_FEATURE_COLUMNS),
                "output": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
