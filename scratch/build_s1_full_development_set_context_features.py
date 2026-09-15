"""Build full-development GT-free set-context features (no fit / no labels).

Creates the 547-observation selection, fills the refined-mask cache, then writes
``output/s1/full_development_set_context_features.csv`` for the harm-first
readiness gate. Target columns, private review material, and model fitting are
never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solar_filament.s1 import set_context_features as features  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    PREPROC,
    build_variant,
    imread_unicode,
    refine_batch,
)
from scratch.run_s0_baseline_rescore import resolve_preprocessed_image_path  # noqa: E402
from scratch.run_s1_proposal_marginal_utility_pilot import (  # noqa: E402
    CACHE_SCHEMA,
    SEED,
    load_refined_cache as load_refined_cache_optional,
    refined_cache_path,
    save_refined_cache,
    stable_selection_hash,
)
from scratch.run_v42_verifier_official_eval import (  # noqa: E402
    REFINER_METHOD,
    REFINER_THRESHOLD,
    fused_proposals,
    load_refiner,
)

DEFAULT_STRATA = ROOT / "output" / "s0" / "observation_strata.csv"
DEFAULT_CANDIDATES = ROOT / "output" / "s0" / "v42_nested_candidate_predictions.csv"
DEFAULT_CACHE_ROOT = ROOT / "output" / "s0" / "s1_proposal_marginal_refined_cache"
DEFAULT_SELECTION_OUT = ROOT / "output" / "s1" / "full_development_selection.csv"
DEFAULT_FEATURES_OUT = ROOT / "output" / "s1" / "full_development_set_context_features.csv"
DEFAULT_OUT = ROOT / "output" / "s1" / "full_development_set_context_features.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_all_development(
    strata: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    seed: int = SEED,
) -> pd.DataFrame:
    required_strata = {"file_name", "fold", "time_block_role", "site_code", "year"}
    if missing := sorted(required_strata - set(strata.columns)):
        raise ValueError(f"strata missing columns: {missing}")
    counts = (
        candidates.groupby(["file_name", "fold"], as_index=False)
        .candidate_index.count()
        .rename(columns={"candidate_index": "candidate_count"})
    )
    available = strata.merge(counts, on=["file_name", "fold"], how="inner")
    available = available.loc[available.time_block_role == "development"].copy()
    if available.empty:
        raise ValueError("no development observations with candidates")
    available["selection_hash"] = available.file_name.map(
        lambda name: stable_selection_hash(str(name), seed)
    )
    available["sample_rank_within_fold"] = (
        available.groupby("fold", sort=False).cumcount() + 1
    )
    result = available.sort_values(["fold", "file_name"]).reset_index(drop=True)
    if not (result.time_block_role == "development").all():
        raise AssertionError("locked observation entered full-development selection")
    if int(result.file_name.nunique()) != 547:
        raise ValueError(
            f"expected 547 development observations, got {result.file_name.nunique()}"
        )
    if int(result.candidate_count.sum()) != 4_899:
        raise ValueError(
            f"expected 4899 candidates, got {int(result.candidate_count.sum())}"
        )
    return result


def read_preprocessed_image(file_name: str, *, attempts: int = 5) -> np.ndarray:
    """Read a preprocessed JPEG with retries for flaky Google Drive paths."""
    path = resolve_preprocessed_image_path(PREPROC, file_name)
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            image = imread_unicode(path)
            if image is None:
                raise FileNotFoundError(path)
            return image
        except OSError as error:
            last_error = error
            print(
                f"[warn] image read failed attempt={attempt}/{attempts} "
                f"file={file_name} error={error}",
                flush=True,
            )
            time.sleep(min(2.0 * attempt, 8.0))
    assert last_error is not None
    raise OSError(f"failed to read {path} after {attempts} attempts") from last_error


def refine_all_candidates(
    file_name: str,
    fold: int,
    refiner: torch.nn.Module,
) -> list[np.ndarray]:
    proposals, _ = fused_proposals(file_name, fold)
    if not proposals:
        return []
    image = read_preprocessed_image(file_name)
    boxes, probabilities = refine_batch(refiner, image, proposals)
    return [
        build_variant(mask, box, probability, REFINER_METHOD, REFINER_THRESHOLD)
        for mask, box, probability in zip(proposals, boxes, probabilities)
    ]


def ensure_refined_cache(
    selection: pd.DataFrame,
    candidates: pd.DataFrame,
    cache_root: Path,
    *,
    threads: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    refiners: dict[int, torch.nn.Module] = {}
    cache_hits = 0
    cache_builds = 0
    started = time.perf_counter()
    total = len(selection)
    candidate_counts = (
        candidates.groupby("file_name").size().astype(int).to_dict()
    )
    for position, selected in enumerate(selection.itertuples(index=False), 1):
        name = str(selected.file_name)
        fold = int(selected.fold)
        path = refined_cache_path(cache_root, fold, name)
        candidate_count = int(candidate_counts[name])
        masks = load_refined_cache_optional(path, name, fold)
        if masks is not None and len(masks) == candidate_count:
            cache_hits += 1
        else:
            if fold not in refiners:
                refiners[fold] = load_refiner(fold)
            try:
                masks = refine_all_candidates(name, fold, refiners[fold])
            except Exception as error:  # noqa: BLE001 - resume-safe per-observation guard
                print(
                    f"[error] refine failed position={position}/{total} "
                    f"file={name} fold={fold}: {error}",
                    flush=True,
                )
                raise
            if len(masks) != candidate_count:
                raise ValueError(
                    f"candidate count mismatch {name}: refined={len(masks)} "
                    f"table={candidate_count}"
                )
            save_refined_cache(path, masks, file_name=name, fold=fold)
            cache_builds += 1

        if position == 1 or position % 25 == 0 or position == total:
            elapsed = time.perf_counter() - started
            rate = elapsed / position
            eta = rate * (total - position)
            print(
                f"[cache] {position}/{total} hits={cache_hits} builds={cache_builds} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    return {
        "cache_schema": CACHE_SCHEMA,
        "observations": total,
        "cache_hits": cache_hits,
        "cache_builds": cache_builds,
        "elapsed_seconds": float(time.perf_counter() - started),
        "cache_root": str(cache_root.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata", type=Path, default=DEFAULT_STRATA)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--selection-out", type=Path, default=DEFAULT_SELECTION_OUT)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--skip-cache-build",
        action="store_true",
        help="Assume refined caches already exist; fail if any are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strata = pd.read_csv(args.strata)
    candidates = pd.read_csv(args.candidates)
    selection = select_all_development(strata, candidates)
    args.selection_out.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.selection_out, index=False)
    print(
        f"selection: {len(selection)} observations, "
        f"{int(selection.candidate_count.sum())} candidates -> {args.selection_out}",
        flush=True,
    )

    if args.skip_cache_build:
        cache_summary = {
            "cache_schema": CACHE_SCHEMA,
            "skipped": True,
            "cache_root": str(args.cache_root.resolve()),
        }
    else:
        cache_summary = ensure_refined_cache(
            selection,
            candidates,
            args.cache_root,
            threads=args.threads,
        )
        print(f"cache: {json.dumps(cache_summary)}", flush=True)

    feature_started = time.perf_counter()
    loaded_selection, loaded_candidates = features.load_inputs(
        args.selection_out.resolve(), args.candidates.resolve()
    )
    feature_table, cache_manifest = features.build_feature_table(
        loaded_selection, loaded_candidates, args.cache_root.resolve()
    )
    if len(feature_table) != 4_899:
        raise ValueError(f"expected 4899 feature rows, got {len(feature_table)}")

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(args.features_out, index=False)
    cache_manifest_hash = hashlib.sha256(
        "\n".join(
            f"{item['fold']}:{item['file_name']}:{item['sha256']}"
            for item in cache_manifest
        ).encode()
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "full_development_gt_free_set_context_features_"
            "not_model_not_threshold_not_registered"
        ),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "scope": {
            "time_block_role": "development only",
            "observations": 547,
            "candidates": 4899,
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
            "image_shape": list(features.IMAGE_SHAPE),
            "minimum_exclusive_area": features.MIN_AREA,
            "frozen_containment_threshold": features.OWNERSHIP_THRESHOLD,
            "identifiers": list(features.IDENTIFIER_COLUMNS),
            "base_inference_features": list(features.BASE_FEATURE_COLUMNS),
            "gt_free_set_context_features": list(features.CONTEXT_FEATURE_COLUMNS),
            "forbidden_output_tokens": list(features.FORBIDDEN_OUTPUT_TOKENS),
        },
        "summary": features.summarize(feature_table),
        "cache_build": cache_summary,
        "feature_build_seconds": float(time.perf_counter() - feature_started),
        "inputs": {
            "strata": str(args.strata.resolve()),
            "strata_sha256": sha256_file(args.strata.resolve()),
            "selection": str(args.selection_out.resolve()),
            "selection_sha256": sha256_file(args.selection_out.resolve()),
            "candidate_table": str(args.candidates.resolve()),
            "candidate_table_sha256": sha256_file(args.candidates.resolve()),
            "refined_cache_root": str(args.cache_root.resolve()),
            "refined_cache_manifest_sha256": cache_manifest_hash,
            "refined_cache_entries": len(cache_manifest),
        },
        "artifacts": {
            "features": str(args.features_out.resolve()),
            "features_sha256": sha256_file(args.features_out.resolve()),
        },
        "limitations": [
            "GT-free features only; labels and nested OOF fitting remain blocked.",
            "No feature-target association, model fit, threshold, or margin evaluated.",
            "S0-4 response, endpoint margins, and decision policy remain separate gates.",
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"features: rows={len(feature_table)} -> {args.features_out}\n"
        f"manifest: {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
