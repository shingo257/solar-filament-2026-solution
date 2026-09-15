"""Validate the frozen S1 model structure and fail closed on missing start gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json"
DEFAULT_OUT = ROOT / "output" / "s0" / "s1_harm_first_model_spec_readiness.json"
EXPECTED_HEADS = {
    "final_pq_benefit": "benefit",
    "spine_benefit": "benefit",
    "count_harm": "harm",
    "precision_harm": "harm",
    "merge_harm": "harm",
}
IDENTITY_COLUMNS = ("file_name", "fold", "candidate_index")
BLOCKED_SPEC_STATUS = "pre_registration_candidate_structure_frozen_execution_blocked"
REGISTERED_SPEC_STATUS = "registered_structure_frozen_ready_for_execution"
REGISTERED_POLICY_STATUS = "registered_before_model_results"
POLICY_THRESHOLD_KEYS = (
    "final_pq_benefit_min_probability",
    "spine_benefit_min_probability",
    "count_harm_max_probability",
    "precision_harm_max_probability",
    "merge_harm_max_probability",
    "minimum_selection_rate",
)


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


def allowed_features(feature_payload: dict[str, Any]) -> set[str]:
    contract = feature_payload.get("feature_contract", {})
    return set(contract.get("base_inference_features", [])) | set(
        contract.get("gt_free_set_context_features", [])
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def decision_policy_status(spec: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    spec_status = spec.get("status")
    policy = spec.get("decision_policy", {})
    provenance = policy.get("registration_provenance", {})
    values = {key: policy.get(key) for key in POLICY_THRESHOLD_KEYS}

    if spec_status == BLOCKED_SPEC_STATUS:
        for key, value in values.items():
            if not isinstance(value, str) or not value.startswith("TBD_"):
                errors.append(f"{key}: must remain TBD before policy registration")
        if provenance.get("registered_before_target_values_loaded") is not False:
            errors.append("blocked policy must not claim pre-target registration")
        return False, errors

    if spec_status != REGISTERED_SPEC_STATUS:
        return False, ["unexpected spec status"]
    if policy.get("status") != REGISTERED_POLICY_STATUS:
        errors.append("registered spec requires registered decision policy status")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{key}: registered threshold must be numeric")
            continue
        numeric = float(value)
        lower_inclusive = key != "minimum_selection_rate"
        lower_ok = numeric >= 0.0 if lower_inclusive else numeric > 0.0
        if not math.isfinite(numeric) or not lower_ok or numeric > 1.0:
            errors.append(f"{key}: registered threshold must be within probability bounds")
    if provenance.get("registered_before_target_values_loaded") is not True:
        errors.append("registered policy must prove registration before target loading")
    for key in ("registered_at", "registered_by", "rationale_document"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            errors.append(f"decision policy provenance missing {key}")
    for key in (
        "public_review_response_sha256",
        "endpoint_registration_sha256",
        "rationale_document_sha256",
    ):
        if not _is_sha256(provenance.get(key)):
            errors.append(f"decision policy provenance missing valid {key}")
    return not errors, errors


def validate_structure(
    spec: dict[str, Any],
    feature_payload: dict[str, Any],
    registry_payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if spec.get("status") not in {BLOCKED_SPEC_STATUS, REGISTERED_SPEC_STATUS}:
        errors.append("unexpected spec status")
    if feature_payload.get("scope", {}).get("target_columns_loaded") is not False:
        errors.append("feature contract did not prove target exclusion")
    if registry_payload.get("scope", {}).get("target_columns_loaded") is not False:
        errors.append("registry did not prove target exclusion")
    if registry_payload.get("scope", {}).get("locked_rows_in_registry") != 0:
        errors.append("registry includes locked rows")

    available = allowed_features(feature_payload)
    forbidden = set(spec.get("feature_contract", {}).get("forbidden_prediction_features", []))
    heads = spec.get("heads")
    if not isinstance(heads, list):
        return [*errors, "heads must be a list"]
    names = [head.get("name") for head in heads]
    if set(names) != set(EXPECTED_HEADS) or len(names) != len(set(names)):
        errors.append("head names do not match the five-head contract")
    label_columns = [head.get("label_column") for head in heads]
    if label_columns != names or len(label_columns) != len(set(label_columns)):
        errors.append("label columns must uniquely match the five head names")

    sufficiency = spec.get("head_sufficiency_contract", {})
    feature_count = int(sufficiency.get("features_per_primary_head", -1))
    intercept_df = int(sufficiency.get("intercept_degrees_of_freedom", -1))
    events_per_df = int(sufficiency.get("events_per_fitted_degree_of_freedom", -1))
    expected_events = (feature_count + intercept_df) * events_per_df
    if sufficiency.get("minimum_candidate_events_per_outer_training") != expected_events:
        errors.append("candidate event gate does not equal events-per-degree-of-freedom contract")
    if sufficiency.get("minimum_event_observations_per_outer_training", 0) < 30:
        errors.append("event-observation gate is below 30")

    for head in heads:
        name = str(head.get("name"))
        if head.get("role") != EXPECTED_HEADS.get(name):
            errors.append(f"{name}: role mismatch")
        features = head.get("features")
        if not isinstance(features, list) or len(features) != feature_count:
            errors.append(f"{name}: expected {feature_count} fixed features")
            continue
        if len(features) != len(set(features)):
            errors.append(f"{name}: duplicate features")
        unknown = sorted(set(features).difference(available))
        if unknown:
            errors.append(f"{name}: unknown features {unknown}")
        leaked = sorted(set(features).intersection(forbidden))
        if leaked:
            errors.append(f"{name}: forbidden features {leaked}")

    model = spec.get("common_model", {})
    if model.get("estimator") != "logistic_regression" or model.get("penalty") != "l2":
        errors.append("primary estimator must be L2 logistic regression")
    if model.get("selection_objective") != "observation_equal_weighted_binary_log_loss":
        errors.append("selection objective must be observation-equal weighted log loss")
    if spec.get("inner_selection", {}).get("threshold_selection_allowed") is not False:
        errors.append("threshold selection must remain disabled")

    _, policy_errors = decision_policy_status(spec)
    errors.extend(policy_errors)
    if spec.get("start_gates", {}).get(
        "decision_policy_registered_before_model_results_required"
    ) is not True:
        errors.append("decision policy registration must be a fit start gate")
    return errors


def review_response_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "complete_24": False, "response_count": 0}
    try:
        payload = load_json(path)
        responses = payload.get("responses")
        count = len(responses) if isinstance(responses, dict) else 0
        complete = count == 24 and all(
            isinstance(item, dict) and bool(item.get("choice"))
            for item in responses.values()
        )
        return {
            "path": str(path),
            "exists": True,
            "complete_24": complete,
            "response_count": count,
            "sha256": sha256_file(path),
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "path": str(path),
            "exists": True,
            "complete_24": False,
            "response_count": 0,
            "error": str(error),
        }


def table_status(
    path: Path,
    required_columns: set[str] | None = None,
    *,
    binary_columns: set[str] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": 0, "complete_4899": False}
    columns = list(IDENTITY_COLUMNS) + sorted(required_columns or set())
    try:
        frame = pd.read_csv(path, usecols=columns)
    except (OSError, ValueError) as error:
        return {
            "path": str(path),
            "exists": True,
            "rows": 0,
            "complete_4899": False,
            "error": str(error),
        }
    binary_valid = True
    for column in sorted(binary_columns or set()):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            binary_valid = False
    complete = bool(
        len(frame) == 4_899
        and not frame.duplicated(list(IDENTITY_COLUMNS)).any()
        and not frame.isna().any().any()
        and binary_valid
    )
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "complete_4899": complete,
        "required_columns": sorted(required_columns or set()),
        "binary_columns_valid": binary_valid,
        "sha256": sha256_file(path),
    }


def margins_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "tbd_occurrences": 0,
            "registered": False,
        }
    text = path.read_text(encoding="utf-8")
    count = text.count("TBD")
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "tbd_occurrences": count,
        "registered": count == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--review-response", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = args.spec.resolve()
    out_path = args.out.resolve()
    spec = load_json(spec_path)
    feature_path = resolve_from_root(spec["feature_contract"]["path"])
    registry_path = resolve_from_root(spec["registry"]["path"])
    feature_payload = load_json(feature_path)
    registry_json_path = registry_path.with_suffix(".json")
    registry_payload = load_json(registry_json_path)
    structure_errors = validate_structure(spec, feature_payload, registry_payload)
    policy_registered, _ = decision_policy_status(spec)

    feature_hash_matches = sha256_file(feature_path) == spec["feature_contract"]["sha256"]
    registry_hash_matches = sha256_file(registry_path) == spec["registry"]["sha256"]
    execution = spec["execution_inputs"]
    response_path = (
        args.review_response.resolve()
        if args.review_response is not None
        else resolve_from_root(execution["default_public_review_response"])
    )
    review = review_response_status(response_path)
    registration_path = resolve_from_root(execution["endpoint_registration"])
    margins = margins_status(registration_path)
    all_features = set().union(*(set(head["features"]) for head in spec["heads"]))
    label_columns = {str(head["label_column"]) for head in spec["heads"]}
    full_features = table_status(
        resolve_from_root(execution["full_development_features"]), all_features
    )
    full_labels = table_status(
        resolve_from_root(execution["full_development_labels"]),
        label_columns,
        binary_columns=label_columns,
    )
    gates = {
        "structure_valid": not structure_errors,
        "feature_contract_hash_matches": feature_hash_matches,
        "registry_hash_matches": registry_hash_matches,
        "complete_s0_4_response": review["complete_24"],
        "all_endpoint_margins_registered": margins["registered"],
        "decision_policy_registered_before_model_results": policy_registered,
        "full_development_features_complete": full_features["complete_4899"],
        "full_development_labels_complete": full_labels["complete_4899"],
    }
    ready_to_fit = all(gates.values())
    payload = {
        "schema_version": 2,
        "status": "ready_to_fit" if ready_to_fit else "structure_ready_execution_blocked",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "scope": {
            "private_key_read": False,
            "locked_rows_read": 0,
            "model_fitting_performed": False,
            "threshold_selection_performed": False,
            "target_values_analyzed": False,
        },
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "structure_errors": structure_errors,
        "gates": gates,
        "ready_to_fit": ready_to_fit,
        "review_response": review,
        "decision_policy": {
            "status": spec.get("decision_policy", {}).get("status"),
            "registered_before_model_results": policy_registered,
        },
        "endpoint_margins": margins,
        "full_development_features": full_features,
        "full_development_labels": full_labels,
        "blockers": [key for key, passed in gates.items() if not passed],
        "limitations": [
            "A valid model structure is not evidence of model performance.",
            "TBD probability thresholds are intentionally rejected for execution.",
            "A label table without all five binary head columns is incomplete.",
            "Private unblinding material is never read by this validator.",
            "A new independent time/site block remains required for confirmatory claims.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "structure_valid": gates["structure_valid"],
                "ready_to_fit": ready_to_fit,
                "blockers": payload["blockers"],
                "output": str(out_path),
            },
            indent=2,
        )
    )
    return 0 if not structure_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
