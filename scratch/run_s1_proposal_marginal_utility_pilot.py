"""Measure post-refiner leave-one-proposal-out utility on development cases.

This is a fixed-sample CPU preflight, not S1 training or threshold selection.
It compares the existing pre-refiner candidate proxy with the actual local
effect of one fused proposal after the frozen refiner and ownership rule.
Locked observations and blinded-review artifacts are never read.
"""


from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path
import pandas as pd


try:
    from .run_s1_proposal_marginal_utility_pilot_impl import (
        CACHE_SCHEMA,
        DEFAULT_CACHE_ROOT,
        DEFAULT_CANDIDATES,
        DEFAULT_OUTPUT,
        DEFAULT_ROWS_OUTPUT,
        DEFAULT_SELECTION_OUTPUT,
        DEFAULT_STRATA,
        SEED,
        binary_diagnostics,
        build_summary,
        final_catalog_safe_target,
        light_variant_metrics,
        linearized_pq_utility,
        load_refined_cache,
        refined_cache_path,
        run_pilot,
        safe_spearman,
        save_refined_cache,
        select_fixed_development_sample,
        sha256_file,
        stable_selection_hash,
    )
except ImportError:  # Direct `python scratch/...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_s1_proposal_marginal_utility_pilot_impl import (
        CACHE_SCHEMA,
        DEFAULT_CACHE_ROOT,
        DEFAULT_CANDIDATES,
        DEFAULT_OUTPUT,
        DEFAULT_ROWS_OUTPUT,
        DEFAULT_SELECTION_OUTPUT,
        DEFAULT_STRATA,
        SEED,
        binary_diagnostics,
        build_summary,
        final_catalog_safe_target,
        light_variant_metrics,
        linearized_pq_utility,
        load_refined_cache,
        refined_cache_path,
        run_pilot,
        safe_spearman,
        save_refined_cache,
        select_fixed_development_sample,
        sha256_file,
        stable_selection_hash,
    )



__all__ = [
    "CACHE_SCHEMA",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_CANDIDATES",
    "DEFAULT_OUTPUT",
    "DEFAULT_ROWS_OUTPUT",
    "DEFAULT_SELECTION_OUTPUT",
    "DEFAULT_STRATA",
    "SEED",
    "binary_diagnostics",
    "build_summary",
    "final_catalog_safe_target",
    "light_variant_metrics",
    "linearized_pq_utility",
    "load_refined_cache",
    "refined_cache_path",
    "run_pilot",
    "safe_spearman",
    "save_refined_cache",
    "select_fixed_development_sample",
    "sha256_file",
    "stable_selection_hash",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", type=Path, default=DEFAULT_STRATA)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--per-fold", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--selection-out", type=Path, default=DEFAULT_SELECTION_OUTPUT)
    parser.add_argument("--rows-out", type=Path, default=DEFAULT_ROWS_OUTPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    strata = pd.read_csv(args.strata)
    candidates = pd.read_csv(args.candidates)
    selection = select_fixed_development_sample(
        strata, candidates, args.per_fold, SEED
    )
    args.selection_out.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.selection_out, index=False)
    if args.selection_only:
        print(selection[[
            "file_name", "fold", "site_code", "year", "candidate_count"
        ]].to_string(index=False))
        return 0

    if args.summarize_existing:
        if not args.rows_out.exists() or not args.out.exists():
            raise FileNotFoundError(
                "--summarize-existing requires existing row CSV and result JSON"
            )
        rows = pd.read_csv(args.rows_out)
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        runtime = previous["runtime"]
    else:
        rows, runtime = run_pilot(
            selection, candidates, args.cache_root, args.threads
        )
    elapsed = float(runtime["elapsed_seconds"])
    runtime.setdefault("seconds_per_observation", elapsed / len(selection))
    runtime.setdefault("seconds_per_candidate", elapsed / len(rows))
    runtime.setdefault(
        "projected_seconds_for_547_development_observations_at_pilot_rate",
        elapsed / len(selection) * 547,
    )
    development_names = set(
        strata.loc[strata.time_block_role == "development", "file_name"]
    )
    development_candidate_count = int(
        candidates.file_name.isin(development_names).sum()
    )
    runtime["development_candidates_available"] = development_candidate_count
    runtime["projected_seconds_for_development_at_pilot_candidate_rate"] = (
        elapsed / len(rows) * development_candidate_count
    )
    if not (rows.time_block_role == "development").all():
        raise AssertionError("pilot output contains non-development rows")
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.rows_out, index=False)
    result = {
        "schema_version": "s1-proposal-marginal-utility-pilot-v1",
        "status": "fixed_development_cpu_pilot_not_s1_not_registered",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seed": SEED,
        "selection_contract": (
            f"{args.per_fold} lowest seeded SHA-256 file names per fold among "
            "development observations with at least one V42 fused candidate; no outcome selection."
        ),
        "scope": {
            "observations": len(selection),
            "per_fold": args.per_fold,
            "candidates": len(rows),
            "time_block_role": "development only",
            "locked_rows_read": 0,
            "s1_predictions_read": False,
            "blinded_review_responses_read": False,
            "private_key_read": False,
        },
        "target_contract": {
            "context": (
                "Candidate addition to the frozen accept-all proposal set after "
                "shrink refiner and ownership; measured by leave-one-out."
            ),
            "official_pq_linearized_utility": (
                "delta TP-IoU numerator minus archived V35 PQ times delta denominator"
            ),
            "catalog_safe": (
                "positive spine coverage delta, nonnegative precision delta, "
                "nonpositive merge and count-error deltas"
            ),
        },
        "summary": build_summary(rows),
        "runtime": runtime,
        "inputs": {
            "strata": str(args.strata),
            "strata_sha256": sha256_file(args.strata),
            "candidates": str(args.candidates),
            "candidates_sha256": sha256_file(args.candidates),
        },
        "artifacts": {
            "selection": str(args.selection_out),
            "selection_sha256": sha256_file(args.selection_out),
            "candidate_rows": str(args.rows_out),
            "candidate_rows_sha256": sha256_file(args.rows_out),
            "refined_cache_root": str(args.cache_root),
        },
        "limitations": [
            (
                f"Fixed {len(selection)}-observation feasibility sample; not a "
                "performance estimate."
            ),
            "Leave-one-out effects are conditional on the accept-all proposal set.",
            "Candidate interactions make marginal utilities non-additive.",
            "No model fitting, threshold selection, locked evaluation, or GPU is used.",
            "A successful pilot would still require development nested OOF and a new holdout.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "observations": result["scope"]["observations"],
        "candidates": result["scope"]["candidates"],
        "elapsed_seconds": result["runtime"]["elapsed_seconds"],
        "output": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
