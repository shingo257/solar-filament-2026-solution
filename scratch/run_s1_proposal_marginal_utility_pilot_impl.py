"""Implementation helpers for `run_s1_proposal_marginal_utility_pilot` — extracted verbatim from `run_s1_proposal_marginal_utility_pilot.py`.

Holds the helper functions and module constants so `run_s1_proposal_marginal_utility_pilot.py` stays a readable
orchestrator. Loaded by the wrapper via sibling import.
"""


from __future__ import annotations


import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))


from solar_filament.metrics.metrics_official import PanopticQuality  # noqa: E402
from solar_filament.metrics.science_metrics import (  # noqa: E402
    audit_spine_coordinates,
    fragmentation_counts,
    match_instances,
    spine_coverage,
)
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    PREPROC,
    TRAIN_JSON,
    build_variant,
    imread_unicode,
    iou_matrix as official_iou_matrix,
    masks_to_rles,
    refine_batch,
    to_rle,
)
from scratch.run_s0_baseline_rescore import (  # noqa: E402
    annotation_mask,
    resize_mask,
    resolve_preprocessed_image_path,
)
from scratch.run_v42_verifier_official_eval import (  # noqa: E402
    ARCHIVED_V35_PQ,
    OWNERSHIP_THRESHOLD,
    REFINER_METHOD,
    REFINER_THRESHOLD,
    fused_proposals,
    load_refiner,
)
from scratch.run_wbf_dropcontained_threshold_probe import (  # noqa: E402
    apply_ownership_with_threshold,
)


DEFAULT_STRATA = ROOT / "output/s0/observation_strata.csv"


DEFAULT_CANDIDATES = ROOT / "output/s0/v42_nested_candidate_predictions.csv"


DEFAULT_OUTPUT = ROOT / "output/s0/s1_proposal_marginal_utility_pilot.json"


DEFAULT_ROWS_OUTPUT = ROOT / "output/s0/s1_proposal_marginal_utility_pilot.csv"


DEFAULT_SELECTION_OUTPUT = ROOT / "output/s0/s1_proposal_marginal_utility_selection.csv"


DEFAULT_CACHE_ROOT = ROOT / "output/s0/s1_proposal_marginal_refined_cache"


SEED = 20260824


EVAL_SIZE = 512


CACHE_SCHEMA = "s1-proposal-marginal-refined-v1"


SCIENCE_ENDPOINTS = (
    "pq_512",
    "spine_coverage",
    "filament_recall",
    "filament_precision",
    "merge_count",
    "count_abs_error",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_selection_hash(file_name: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{file_name}".encode()).hexdigest()


def select_fixed_development_sample(
    strata: pd.DataFrame,
    candidates: pd.DataFrame,
    per_fold: int,
    seed: int = SEED,
) -> pd.DataFrame:
    if per_fold <= 0:
        raise ValueError("per_fold must be positive")
    required_strata = {"file_name", "fold", "time_block_role", "site_code", "year"}
    required_candidates = {"file_name", "fold", "candidate_index"}
    if missing := sorted(required_strata - set(strata.columns)):
        raise ValueError(f"strata missing columns: {missing}")
    if missing := sorted(required_candidates - set(candidates.columns)):
        raise ValueError(f"candidate table missing columns: {missing}")
    counts = (
        candidates.groupby(["file_name", "fold"], as_index=False)
        .candidate_index.count()
        .rename(columns={"candidate_index": "candidate_count"})
    )
    available = strata.merge(counts, on=["file_name", "fold"], how="inner")
    available = available.loc[available.time_block_role == "development"].copy()
    available["selection_hash"] = available.file_name.map(
        lambda name: stable_selection_hash(str(name), seed)
    )
    selected = []
    for fold in (0, 1, 2):
        group = available.loc[available.fold == fold].sort_values(
            ["selection_hash", "file_name"]
        )
        if len(group) < per_fold:
            raise ValueError(f"fold {fold} has only {len(group)} eligible observations")
        chosen = group.head(per_fold).copy()
        chosen["sample_rank_within_fold"] = np.arange(1, len(chosen) + 1)
        selected.append(chosen)
    result = pd.concat(selected, ignore_index=True).sort_values(
        ["fold", "sample_rank_within_fold"]
    )
    if not (result.time_block_role == "development").all():
        raise AssertionError("locked observation entered fixed sample")
    return result.reset_index(drop=True)


def refined_cache_path(cache_root: Path, fold: int, file_name: str) -> Path:
    return cache_root / f"fold_{fold}" / f"{Path(file_name).stem}.npz"


def save_refined_cache(
    path: Path,
    masks: list[np.ndarray],
    *,
    file_name: str,
    fold: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = (
        np.packbits(np.stack(masks).astype(np.uint8), axis=-1)
        if masks
        else np.empty((0, 0, 0), dtype=np.uint8)
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray(CACHE_SCHEMA),
        file_name=np.asarray(file_name),
        fold=np.asarray(fold, dtype=np.int16),
        shape=np.asarray([2048, 2048], dtype=np.int32),
        packed=packed,
        candidate_count=np.asarray(len(masks), dtype=np.int16),
    )
    temporary.replace(path)


def load_refined_cache(
    path: Path,
    file_name: str,
    fold: int,
) -> list[np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if (
                str(data["schema"].item()) != CACHE_SCHEMA
                or str(data["file_name"].item()) != file_name
                or int(data["fold"].item()) != fold
                or tuple(map(int, data["shape"])) != (2048, 2048)
            ):
                return None
            packed = data["packed"]
            masks = (
                [np.unpackbits(item, axis=-1)[:, :2048].astype(bool) for item in packed]
                if packed.size
                else []
            )
            if len(masks) != int(data["candidate_count"].item()):
                return None
            return masks
    except (KeyError, OSError, ValueError):
        return None


def refine_all_candidates(
    file_name: str,
    fold: int,
    refiner: torch.nn.Module,
) -> list[np.ndarray]:
    proposals, _ = fused_proposals(file_name, fold)
    if not proposals:
        return []
    image = imread_unicode(resolve_preprocessed_image_path(PREPROC, file_name))
    if image is None:
        raise FileNotFoundError(resolve_preprocessed_image_path(PREPROC, file_name))
    boxes, probabilities = refine_batch(refiner, image, proposals)
    return [
        build_variant(mask, box, probability, REFINER_METHOD, REFINER_THRESHOLD)
        for mask, box, probability in zip(proposals, boxes, probabilities)
    ]


def official_totals(
    truth_rles: list[list[dict[str, Any]]],
    predictions: list[np.ndarray],
) -> dict[str, float]:
    prediction_rles = masks_to_rles(predictions)
    accumulator = PanopticQuality()
    totals = {"tp": 0, "fp": 0, "fn": 0, "tp_iou_sum": 0.0}
    for truths in truth_rles:
        matrix = official_iou_matrix(truths, prediction_rles)
        row = accumulator.add_matrix(matrix, len(truths), len(prediction_rles))
        for key in totals:
            totals[key] += row[key]
    denominator = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
    return {
        **totals,
        "denominator": float(denominator),
        "pq": float(totals["tp_iou_sum"] / denominator) if denominator else 0.0,
    }


def prepare_light_variant(
    annotations: list[dict[str, Any]],
    height: int,
    width: int,
    size: int = EVAL_SIZE,
) -> dict[str, Any]:
    truths = [
        resize_mask(annotation_mask(annotation, height, width), size)
        for annotation in annotations
    ]
    spines = []
    for annotation in annotations:
        audit = audit_spine_coordinates(annotation.get("spine"), height, width)
        spines.append(audit.points * np.asarray([size / width, size / height]))
    return {"truths": truths, "spines": spines}


def light_variant_metrics(
    prepared: dict[str, Any],
    predictions: list[np.ndarray],
) -> dict[str, float]:
    truths = prepared["truths"]
    spines = prepared["spines"]
    matches = match_instances(
        truths,
        predictions,
        minimum_iou=float(np.nextafter(0.5, 1.0)),
    )
    covered = [0.0 if len(points) else np.nan for points in spines]
    for truth_index, prediction_index, _ in matches:
        if len(spines[truth_index]):
            covered[truth_index] = spine_coverage(
                spines[truth_index], predictions[prediction_index], radius=1
            )
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(truths) - tp
    iou_sum = float(sum(match[2] for match in matches))
    denominator = tp + 0.5 * fp + 0.5 * fn
    _, merge_count = fragmentation_counts(truths, predictions, overlap_fraction=0.10)
    finite_coverage = np.asarray(covered, dtype=np.float64)
    finite_coverage = finite_coverage[np.isfinite(finite_coverage)]
    return {
        "pq_512": iou_sum / denominator if denominator else 0.0,
        "spine_coverage": float(finite_coverage.mean())
        if finite_coverage.size
        else np.nan,
        "filament_recall": tp / len(truths) if truths else 0.0,
        "filament_precision": tp / len(predictions) if predictions else 0.0,
        "merge_count": float(merge_count),
        "count_abs_error": float(abs(len(predictions) - len(truths))),
    }


def observation_science_metrics(
    variants: list[dict[str, Any]],
    predictions_2048: list[np.ndarray],
    size: int = EVAL_SIZE,
) -> dict[str, float]:
    predictions = [resize_mask(mask, size) for mask in predictions_2048]
    rows = [light_variant_metrics(variant, predictions) for variant in variants]
    return {
        endpoint: float(np.nanmean([row[endpoint] for row in rows]))
        for endpoint in SCIENCE_ENDPOINTS
    }


def linearized_pq_utility(
    full: dict[str, float],
    without: dict[str, float],
    reference_pq: float = ARCHIVED_V35_PQ,
) -> float:
    delta_numerator = full["tp_iou_sum"] - without["tp_iou_sum"]
    delta_denominator = full["denominator"] - without["denominator"]
    return float(delta_numerator - reference_pq * delta_denominator)


def final_catalog_safe_target(deltas: dict[str, float]) -> bool:
    return bool(
        deltas["spine_coverage"] > 0
        and deltas["filament_precision"] >= 0
        and deltas["merge_count"] <= 0
        and deltas["count_abs_error"] <= 0
    )


def binary_diagnostics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(reference, dtype=bool)
    predicted = np.asarray(prediction, dtype=bool)
    tp = int(np.logical_and(truth, predicted).sum())
    fp = int(np.logical_and(~truth, predicted).sum())
    fn = int(np.logical_and(truth, ~predicted).sum())
    tn = int(np.logical_and(~truth, ~predicted).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "agreement": float((tp + tn) / len(truth)) if len(truth) else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def safe_spearman(first: pd.Series, second: pd.Series) -> dict[str, float | None]:
    first_values = first.to_numpy(float)
    second_values = second.to_numpy(float)
    valid = np.isfinite(first_values) & np.isfinite(second_values)
    if valid.sum() < 3:
        return {"rho": None, "pvalue": None, "n": int(valid.sum())}
    first_values = first_values[valid]
    second_values = second_values[valid]
    if np.ptp(first_values) == 0 or np.ptp(second_values) == 0:
        return {"rho": None, "pvalue": None, "n": int(valid.sum())}
    result = spearmanr(first_values, second_values)
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue), "n": int(valid.sum())}


def build_summary(rows: pd.DataFrame) -> dict[str, Any]:
    final_pq_positive = rows.final_pq_linearized_utility > 0
    final_safe = rows.final_catalog_safe_target.astype(bool).to_numpy()
    proxy_positive = rows.target.astype(bool).to_numpy()
    verifier_accept = rows.verifier_accept.astype(bool).to_numpy()
    neutral = np.isclose(rows.final_pq_linearized_utility.to_numpy(), 0.0)
    return {
        "candidates": len(rows),
        "observations": int(rows.file_name.nunique()),
        "final_pq_positive": int(final_pq_positive.sum()),
        "final_catalog_safe_positive": int(final_safe.sum()),
        "final_pq_neutral": int(neutral.sum()),
        "final_pq_neutral_fraction": float(neutral.mean()),
        "catalog_safety_failures_among_pre_refiner_proxy_positive": {
            "proxy_positive_candidates": int(proxy_positive.sum()),
            "nonpositive_spine_delta": int(
                np.logical_and(proxy_positive, rows.delta_spine_coverage <= 0).sum()
            ),
            "negative_precision_delta": int(
                np.logical_and(proxy_positive, rows.delta_filament_precision < 0).sum()
            ),
            "positive_merge_delta": int(
                np.logical_and(proxy_positive, rows.delta_merge_count > 0).sum()
            ),
            "positive_count_error_delta": int(
                np.logical_and(proxy_positive, rows.delta_count_abs_error > 0).sum()
            ),
            "failures_overlap": True,
        },
        "pre_refiner_proxy_vs_final_pq": binary_diagnostics(
            final_pq_positive.to_numpy(), proxy_positive
        ),
        "verifier_accept_vs_final_pq": binary_diagnostics(
            final_pq_positive.to_numpy(), verifier_accept
        ),
        "pre_refiner_proxy_vs_final_catalog_safe": binary_diagnostics(
            final_safe, proxy_positive
        ),
        "verifier_accept_vs_final_catalog_safe": binary_diagnostics(
            final_safe, verifier_accept
        ),
        "pre_refiner_utility_vs_final_pq_utility_spearman": safe_spearman(
            rows.marginal_utility, rows.final_pq_linearized_utility
        ),
        "pre_refiner_utility_vs_final_spine_delta_spearman": safe_spearman(
            rows.marginal_utility, rows.delta_spine_coverage
        ),
        "endpoint_delta_means": {
            endpoint: float(rows[f"delta_{endpoint}"].mean())
            for endpoint in SCIENCE_ENDPOINTS
        },
    }


def run_pilot(
    selection: pd.DataFrame,
    candidates: pd.DataFrame,
    cache_root: Path,
    threads: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with TRAIN_JSON.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[str(annotation["image_id"])].append(annotation)
    file_to_ids: dict[str, list[str]] = defaultdict(list)
    for image_id, image in images.items():
        file_to_ids[image["file_name"]].append(image_id)

    torch.set_num_threads(threads)
    refiners: dict[int, torch.nn.Module] = {}
    cache_hits = 0
    cache_builds = 0
    rows = []
    started = time.perf_counter()
    for position, selected in enumerate(selection.itertuples(index=False), 1):
        name = str(selected.file_name)
        fold = int(selected.fold)
        cache_path = refined_cache_path(cache_root, fold, name)
        refined = load_refined_cache(cache_path, name, fold)
        if refined is None:
            if fold not in refiners:
                refiners[fold] = load_refiner(fold)
            refined = refine_all_candidates(name, fold, refiners[fold])
            save_refined_cache(cache_path, refined, file_name=name, fold=fold)
            cache_builds += 1
        else:
            cache_hits += 1

        candidate_rows = candidates.loc[candidates.file_name == name].sort_values(
            "candidate_index"
        )
        indices = candidate_rows.candidate_index.to_numpy(dtype=int)
        if not np.array_equal(indices, np.arange(len(candidate_rows))):
            raise ValueError(f"candidate indices are not contiguous for {name}")
        if len(refined) != len(candidate_rows):
            raise ValueError(
                f"candidate count mismatch {name}: refined={len(refined)} "
                f"table={len(candidate_rows)}"
            )

        truth_rles = []
        science_variants = []
        for image_id in file_to_ids[name]:
            image_info = images[image_id]
            variant_annotations = annotations[image_id]
            truth_rles.append([
                to_rle(
                    annotation,
                    int(image_info["height"]),
                    int(image_info["width"]),
                )
                for annotation in variant_annotations
            ])
            science_variants.append(
                prepare_light_variant(
                    variant_annotations,
                    int(image_info["height"]),
                    int(image_info["width"]),
                )
            )

        full_prediction = apply_ownership_with_threshold(refined, OWNERSHIP_THRESHOLD)
        full_official = official_totals(truth_rles, full_prediction)
        full_science = observation_science_metrics(science_variants, full_prediction)
        for candidate in candidate_rows.itertuples(index=False):
            candidate_index = int(candidate.candidate_index)
            without_refined = [
                mask for index, mask in enumerate(refined) if index != candidate_index
            ]
            without_prediction = apply_ownership_with_threshold(
                without_refined, OWNERSHIP_THRESHOLD
            )
            without_official = official_totals(truth_rles, without_prediction)
            without_science = observation_science_metrics(
                science_variants, without_prediction
            )
            deltas = {
                endpoint: full_science[endpoint] - without_science[endpoint]
                for endpoint in SCIENCE_ENDPOINTS
            }
            pq_utility = linearized_pq_utility(full_official, without_official)
            rows.append({
                **candidate._asdict(),
                "time_block_role": "development",
                "site_code": selected.site_code,
                "year": int(selected.year),
                "selection_hash": selected.selection_hash,
                "full_output_count": len(full_prediction),
                "without_output_count": len(without_prediction),
                "delta_official_tp_iou_sum": (
                    full_official["tp_iou_sum"] - without_official["tp_iou_sum"]
                ),
                "delta_official_denominator": (
                    full_official["denominator"] - without_official["denominator"]
                ),
                "final_pq_linearized_utility": pq_utility,
                "final_pq_positive_target": int(pq_utility > 0),
                "final_catalog_safe_target": int(final_catalog_safe_target(deltas)),
                **{f"delta_{key}": value for key, value in deltas.items()},
            })
        print(
            f"proposal marginal {position}/{len(selection)} {name} "
            f"candidates={len(refined)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    audit = {
        "cache_hits": cache_hits,
        "cache_builds": cache_builds,
        "elapsed_seconds": elapsed,
        "seconds_per_observation": elapsed / len(selection),
        "seconds_per_candidate": elapsed / len(frame),
        "projected_seconds_for_547_development_observations_at_pilot_rate": (
            elapsed / len(selection) * 547
        ),
    }
    return frame, audit
