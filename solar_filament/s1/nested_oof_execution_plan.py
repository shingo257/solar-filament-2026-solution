"""Freeze the S1 nested-OOF execution graph without loading features or targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json"
DEFAULT_READINESS = ROOT / "output" / "s0" / "s1_harm_first_model_spec_readiness.json"
DEFAULT_OUT = ROOT / "output" / "s0" / "s1_nested_oof_execution_plan.json"
REGISTRY_COLUMNS = (
    "outer_fold",
    "outer_role",
    "inner_fold",
    "file_name",
    "source_fold",
    "time_block_role",
    "candidate_count",
)
REQUIRED_READINESS_GATES = {
    "structure_valid",
    "feature_contract_hash_matches",
    "registry_hash_matches",
    "complete_s0_4_response",
    "all_endpoint_margins_registered",
    "decision_policy_registered_before_model_results",
    "full_development_features_complete",
    "full_development_labels_complete",
}
REGISTERED_SPEC_STATUS = "registered_structure_frozen_ready_for_execution"
REGISTERED_POLICY_STATUS = "registered_before_model_results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def observation_hash(frame: pd.DataFrame) -> str:
    lines = [
        f"{row.file_name}|{int(row.candidate_count)}"
        for row in frame.sort_values("file_name").itertuples(index=False)
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def split_stats(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "observations": int(len(frame)),
        "candidates": int(frame["candidate_count"].sum()),
        "observation_sha256": observation_hash(frame),
    }


def load_registry(path: Path, expected_sha256: str) -> pd.DataFrame:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"registry SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    frame = pd.read_csv(path, usecols=list(REGISTRY_COLUMNS))
    for column in ("outer_fold", "inner_fold", "source_fold", "candidate_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return frame


def validate_registry(registry: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    outer_count = int(spec["registry"]["outer_folds"])
    inner_count = int(spec["registry"]["inner_folds_per_outer"])
    expected_outer = set(range(outer_count))
    if set(registry["outer_fold"]) != expected_outer:
        raise ValueError("outer fold coverage does not match the model spec")
    if set(registry["time_block_role"].astype(str)) != {"development"}:
        raise ValueError("non-development row entered the execution registry")
    if (registry["candidate_count"] <= 0).any():
        raise ValueError("registry contains non-positive candidate counts")
    if registry.duplicated(["outer_fold", "file_name"]).any():
        raise ValueError("observation duplicated within an outer cycle")
    if (registry.groupby("file_name")["candidate_count"].nunique() != 1).any():
        raise ValueError("candidate counts changed across outer cycles")
    if (registry.groupby("file_name")["source_fold"].nunique() != 1).any():
        raise ValueError("source fold changed across outer cycles")

    all_files = set(registry["file_name"].astype(str))
    cycle_summary: dict[str, Any] = {}
    for outer_fold in sorted(expected_outer):
        cycle = registry.loc[registry["outer_fold"] == outer_fold]
        if set(cycle["file_name"].astype(str)) != all_files:
            raise ValueError(f"outer cycle {outer_fold} does not cover all observations")
        outer_test = cycle.loc[cycle["outer_role"] == "outer_test"]
        outer_train = cycle.loc[cycle["outer_role"] == "outer_train"]
        if len(outer_test) + len(outer_train) != len(cycle):
            raise ValueError(f"unexpected outer role in cycle {outer_fold}")
        if not (outer_test["source_fold"] == outer_fold).all():
            raise ValueError(f"outer test leakage in cycle {outer_fold}")
        if not (outer_train["source_fold"] != outer_fold).all():
            raise ValueError(f"outer training leakage in cycle {outer_fold}")
        if set(outer_test["inner_fold"]) != {-1}:
            raise ValueError("outer test rows received inner fold assignments")
        if set(outer_train["inner_fold"]) != set(range(inner_count)):
            raise ValueError(f"inner fold coverage mismatch in cycle {outer_fold}")
        if int(cycle["candidate_count"].sum()) != 4_899:
            raise ValueError(f"candidate coverage mismatch in cycle {outer_fold}")
        cycle_summary[str(outer_fold)] = {
            "outer_train": split_stats(outer_train),
            "outer_test": split_stats(outer_test),
        }

    test_counts = registry.loc[
        registry["outer_role"] == "outer_test", "file_name"
    ].value_counts()
    if not (test_counts == 1).all() or set(test_counts.index.astype(str)) != all_files:
        raise ValueError("each observation must be outer test exactly once")
    return {
        "observations": len(all_files),
        "candidates": int(
            registry.drop_duplicates("file_name")["candidate_count"].sum()
        ),
        "registry_rows": int(len(registry)),
        "outer_folds": outer_count,
        "inner_folds_per_outer": inner_count,
        "cycles": cycle_summary,
    }


def readiness_integrity_errors(
    spec: dict[str, Any], readiness: dict[str, Any], *, spec_sha256: str
) -> list[str]:
    errors: list[str] = []
    if readiness.get("spec", {}).get("sha256") != spec_sha256:
        errors.append("readiness references a different model spec hash")
    if readiness.get("structure_errors"):
        errors.append("readiness contains model structure errors")
    gates = readiness.get("gates")
    if not isinstance(gates, dict) or set(gates) != REQUIRED_READINESS_GATES:
        errors.append("readiness gate set does not match the execution contract")
        return errors
    expected_blockers = {key for key, passed in gates.items() if not passed}
    if set(readiness.get("blockers", [])) != expected_blockers:
        errors.append("readiness blockers do not match failed gates")
    if bool(readiness.get("ready_to_fit")) != all(bool(value) for value in gates.values()):
        errors.append("ready_to_fit is inconsistent with gate values")

    policy_registered = (
        spec.get("status") == REGISTERED_SPEC_STATUS
        and spec.get("decision_policy", {}).get("status") == REGISTERED_POLICY_STATUS
    )
    if bool(gates["decision_policy_registered_before_model_results"]) != policy_registered:
        errors.append("decision policy gate is inconsistent with the frozen spec")

    scope = readiness.get("scope", {})
    expected_scope = {
        "private_key_read": False,
        "locked_rows_read": 0,
        "model_fitting_performed": False,
        "threshold_selection_performed": False,
        "target_values_analyzed": False,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            errors.append(f"readiness scope violation: {key}")
    return errors


def c_slug(value: float) -> str:
    return format(float(value), "g").replace(".", "p")


def build_execution_plan(
    spec: dict[str, Any],
    readiness: dict[str, Any],
    registry: pd.DataFrame,
    *,
    spec_sha256: str,
    readiness_sha256: str,
    registry_sha256: str,
) -> dict[str, Any]:
    registry_summary = validate_registry(registry, spec)
    integrity_errors = readiness_integrity_errors(
        spec, readiness, spec_sha256=spec_sha256
    )
    fit_start_permitted = not integrity_errors and readiness.get("ready_to_fit") is True
    outer_folds = list(range(int(spec["registry"]["outer_folds"])))
    inner_folds = list(range(int(spec["registry"]["inner_folds_per_outer"])))
    heads = spec["heads"]
    c_grid = [float(value) for value in spec["common_model"]["inverse_regularization_grid"]]

    inner_jobs: list[dict[str, Any]] = []
    outer_refit_jobs: list[dict[str, Any]] = []
    replay_jobs: list[dict[str, Any]] = []
    for outer_fold in outer_folds:
        cycle = registry.loc[registry["outer_fold"] == outer_fold]
        outer_train = cycle.loc[cycle["outer_role"] == "outer_train"]
        outer_test = cycle.loc[cycle["outer_role"] == "outer_test"]
        inner_splits: dict[int, dict[str, Any]] = {}
        for validation_fold in inner_folds:
            validation = outer_train.loc[outer_train["inner_fold"] == validation_fold]
            training = outer_train.loc[outer_train["inner_fold"] != validation_fold]
            inner_splits[validation_fold] = {
                "training_inner_folds": [
                    fold for fold in inner_folds if fold != validation_fold
                ],
                "validation_inner_fold": validation_fold,
                "training": split_stats(training),
                "validation": split_stats(validation),
            }

        for head in heads:
            head_name = str(head["name"])
            for c_value in c_grid:
                for validation_fold in inner_folds:
                    split = inner_splits[validation_fold]
                    inner_jobs.append({
                        "job_id": (
                            f"inner-o{outer_fold}-{head_name}-"
                            f"c{c_slug(c_value)}-v{validation_fold}"
                        ),
                        "phase": "inner_model_selection",
                        "outer_fold": outer_fold,
                        "head": head_name,
                        "label_column": head["label_column"],
                        "feature_columns": head["features"],
                        "inverse_regularization_c": c_value,
                        **split,
                        "scaler_fit_scope": "training_inner_folds_only",
                        "metric": spec["inner_selection"]["primary_metric"],
                        "model_artifact_policy": "discard_after_validation_metrics",
                    })

            outer_refit_jobs.append({
                "job_id": f"outer-o{outer_fold}-{head_name}",
                "phase": "outer_refit_and_score_once",
                "outer_fold": outer_fold,
                "head": head_name,
                "label_column": head["label_column"],
                "feature_columns": head["features"],
                "selected_c_source": "minimum_inner_oof_weighted_log_loss",
                "training": split_stats(outer_train),
                "test": split_stats(outer_test),
                "scaler_fit_scope": "outer_training_only_after_c_selection",
                "outer_test_prediction_count": 1,
            })

        replay_jobs.append({
            "job_id": f"replay-o{outer_fold}",
            "phase": "decision_policy_and_frozen_ownership_replay",
            "outer_fold": outer_fold,
            "depends_on": [f"outer-o{outer_fold}-{head['name']}" for head in heads],
            "decision_policy_source": "frozen_registered_spec_only",
            "threshold_tuning_allowed": False,
            "test": split_stats(outer_test),
        })

    blockers = list(readiness.get("blockers", []))
    if integrity_errors:
        blockers.append("execution_plan_integrity")
    status = "execution_plan_invalid" if integrity_errors else (
        "execution_plan_ready" if fit_start_permitted else "execution_plan_frozen_fit_blocked"
    )
    return {
        "schema_version": 1,
        "status": status,
        "fit_start_permitted": fit_start_permitted,
        "scope": {
            "registry_identity_and_counts_loaded": True,
            "feature_values_loaded": False,
            "target_values_loaded": False,
            "review_response_loaded": False,
            "private_key_read": False,
            "locked_rows_read": 0,
            "model_fitting_performed": False,
            "threshold_selection_performed": False,
        },
        "input_hashes": {
            "model_spec": spec_sha256,
            "readiness": readiness_sha256,
            "split_registry": registry_sha256,
        },
        "integrity_errors": integrity_errors,
        "blockers": list(dict.fromkeys(blockers)),
        "registry_summary": registry_summary,
        "execution_contract": {
            "estimator": spec["common_model"],
            "head_count": len(heads),
            "c_grid": c_grid,
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "inner_fit_jobs": len(inner_jobs),
            "outer_refit_jobs": len(outer_refit_jobs),
            "total_fit_jobs": len(inner_jobs) + len(outer_refit_jobs),
            "ownership_replay_jobs": len(replay_jobs),
            "threshold_fit_jobs": 0,
        },
        "inner_selection_jobs": inner_jobs,
        "outer_refit_jobs": outer_refit_jobs,
        "ownership_replay_jobs": replay_jobs,
        "limitations": [
            "This plan enumerates compute units but contains no fitting implementation.",
            "Event sufficiency is checked only after registered full labels exist.",
            "Outer source folds are not a new independent time/site block.",
            "Candidate scores never replace frozen-ownership endpoint evaluation.",
        ],
    }


def enforce_fit_ready(plan: dict[str, Any]) -> None:
    if plan.get("fit_start_permitted") is not True:
        blockers = ", ".join(plan.get("blockers", [])) or "unknown"
        raise RuntimeError(f"S1 fitting is blocked: {blockers}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-fit-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = args.spec.resolve()
    readiness_path = args.readiness.resolve()
    out_path = args.out.resolve()
    spec = load_json(spec_path)
    readiness = load_json(readiness_path)
    registry_path = resolve_from_root(spec["registry"]["path"])
    registry = load_registry(registry_path, spec["registry"]["sha256"])
    plan = build_execution_plan(
        spec,
        readiness,
        registry,
        spec_sha256=sha256_file(spec_path),
        readiness_sha256=sha256_file(readiness_path),
        registry_sha256=sha256_file(registry_path),
    )
    plan["script"] = str(Path(__file__).resolve())
    plan["script_sha256"] = sha256_file(Path(__file__).resolve())
    plan["inputs"] = {
        "model_spec": str(spec_path),
        "readiness": str(readiness_path),
        "split_registry": str(registry_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": plan["status"],
        "fit_start_permitted": plan["fit_start_permitted"],
        "inner_fit_jobs": plan["execution_contract"]["inner_fit_jobs"],
        "outer_refit_jobs": plan["execution_contract"]["outer_refit_jobs"],
        "ownership_replay_jobs": plan["execution_contract"]["ownership_replay_jobs"],
        "blockers": plan["blockers"],
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.require_fit_ready and not plan["fit_start_permitted"]:
        return 2
    return 0 if not plan["integrity_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
