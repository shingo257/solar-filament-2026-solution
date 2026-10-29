"""Probe whether V35/V42 conditional rescue has measurable operational headroom.

This is a retrospective, observation-level feasibility experiment.  It uses
only development observations to fit an observable gate and to select its
threshold, then reports transfer to the already-inspected locked time block.
It does not train S1, register margins, or constitute an independent holdout.
"""


from __future__ import annotations


import argparse
import json
from pathlib import Path
import pandas as pd


try:
    from .evaluate_s1_conditional_rescue_headroom_impl import (
        BASE_NUMERIC_FEATURES,
        BOOTSTRAP_REPEATS,
        DEFAULT_CALIBRATION,
        DEFAULT_INPUT,
        DEFAULT_OUTPUT,
        ENDPOINTS,
        FEATURE_SETS,
        ROOT,
        SEED,
        aggregate_observations,
        audit_feature_set,
        bootstrap_indices,
        build_analysis,
        build_analysis_table,
        calibration_mde,
        classifier_diagnostics,
        development_oof_and_locked_probabilities,
        gate_target,
        oracle_masks,
        policy_summary,
        select_development_threshold,
        sha256_file,
    )
except ImportError:  # Direct `python scratch/...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_s1_conditional_rescue_headroom_impl import (
        BASE_NUMERIC_FEATURES,
        BOOTSTRAP_REPEATS,
        DEFAULT_CALIBRATION,
        DEFAULT_INPUT,
        DEFAULT_OUTPUT,
        ENDPOINTS,
        FEATURE_SETS,
        ROOT,
        SEED,
        aggregate_observations,
        audit_feature_set,
        bootstrap_indices,
        build_analysis,
        build_analysis_table,
        calibration_mde,
        classifier_diagnostics,
        development_oof_and_locked_probabilities,
        gate_target,
        oracle_masks,
        policy_summary,
        select_development_threshold,
        sha256_file,
    )



__all__ = [
    "BASE_NUMERIC_FEATURES",
    "BOOTSTRAP_REPEATS",
    "DEFAULT_CALIBRATION",
    "DEFAULT_INPUT",
    "DEFAULT_OUTPUT",
    "ENDPOINTS",
    "FEATURE_SETS",
    "ROOT",
    "SEED",
    "aggregate_observations",
    "audit_feature_set",
    "bootstrap_indices",
    "build_analysis",
    "build_analysis_table",
    "calibration_mde",
    "classifier_diagnostics",
    "development_oof_and_locked_probabilities",
    "gate_target",
    "oracle_masks",
    "policy_summary",
    "select_development_threshold",
    "sha256_file",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    result = build_analysis(frame, args.input, calibration, args.calibration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "observations": result["scope"]["observations"],
        "development_observations": result["scope"]["development_observations"],
        "locked_observations": result["scope"]["locked_observations"],
        "gates": list(result["observable_gates"]),
        "output": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
