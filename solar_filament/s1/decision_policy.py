"""Create a registered S1 model-spec candidate only before target-label creation."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json"
DEFAULT_READINESS = ROOT / "output" / "s0" / "s1_harm_first_model_spec_readiness.json"
DEFAULT_REGISTRATION = ROOT / "S1_DECISION_POLICY_REGISTRATION_TEMPLATE_2026-08-24.json"
DEFAULT_AUDIT = ROOT / "output" / "s0" / "s1_decision_policy_registration_audit.json"
DEFAULT_CANDIDATE = ROOT / "output" / "s0" / "s1_registered_model_spec_candidate.json"
BLOCKED_SPEC_STATUS = "pre_registration_candidate_structure_frozen_execution_blocked"
REGISTERED_SPEC_STATUS = "registered_structure_frozen_ready_for_execution"
REGISTERED_POLICY_STATUS = "registered_before_model_results"
READY_REGISTRATION_STATUS = "ready_for_pre_target_registration"
POLICY_KEYS = (
    "final_pq_benefit_min_probability",
    "spine_benefit_min_probability",
    "count_harm_max_probability",
    "precision_harm_max_probability",
    "merge_harm_max_probability",
    "minimum_selection_rate",
)
ATTESTATION_KEYS = (
    "values_chosen_without_development_target_values",
    "review_response_used_only_for_qualitative_error_priorities",
    "endpoint_margins_are_primary_safety_contract",
)
PREREQUISITE_GATES = (
    "structure_valid",
    "feature_contract_hash_matches",
    "registry_hash_matches",
    "complete_s0_4_response",
    "all_endpoint_margins_registered",
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


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def validate_registration_payload(
    registration: dict[str, Any], *, source_spec_sha256: str
) -> list[str]:
    errors: list[str] = []
    if registration.get("schema_version") != 1:
        errors.append("registration schema_version must be 1")
    if registration.get("status") != READY_REGISTRATION_STATUS:
        errors.append(f"registration status must be {READY_REGISTRATION_STATUS}")
    if registration.get("source_spec_sha256") != source_spec_sha256:
        errors.append("registration source spec SHA-256 mismatch")

    registered_at = registration.get("registered_at")
    if not isinstance(registered_at, str):
        errors.append("registered_at must be a timezone-aware ISO-8601 string")
    else:
        try:
            parsed = datetime.fromisoformat(registered_at)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                errors.append("registered_at must include a timezone offset")
        except ValueError:
            errors.append("registered_at must be valid ISO-8601")
    registered_by = registration.get("registered_by")
    if not isinstance(registered_by, str) or not registered_by.strip():
        errors.append("registered_by must be non-empty")
    rationale = registration.get("rationale_document")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("rationale_document must be non-empty")

    thresholds = registration.get("decision_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(POLICY_KEYS):
        errors.append("decision_thresholds must contain exactly the six policy keys")
    else:
        for key in POLICY_KEYS:
            value = thresholds[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{key} must be numeric")
                continue
            numeric = float(value)
            lower_ok = numeric > 0.0 if key == "minimum_selection_rate" else numeric >= 0.0
            if not lower_ok or numeric > 1.0:
                errors.append(f"{key} must be within probability bounds")

    attestations = registration.get("attestations")
    if not isinstance(attestations, dict) or set(attestations) != set(ATTESTATION_KEYS):
        errors.append("attestations must contain exactly the three required statements")
    else:
        for key in ATTESTATION_KEYS:
            if attestations[key] is not True:
                errors.append(f"attestation must be true: {key}")
    return errors


def _hash_evidence(path_value: object, label: str, errors: list[str]) -> str | None:
    if not isinstance(path_value, str) or not path_value:
        errors.append(f"{label} path missing from readiness")
        return None
    path = Path(path_value).resolve()
    if not path.exists():
        errors.append(f"{label} file does not exist")
        return None
    return sha256_file(path)


def registration_prerequisite_errors(
    spec: dict[str, Any],
    readiness: dict[str, Any],
    registration: dict[str, Any],
    *,
    source_spec_sha256: str,
) -> tuple[list[str], dict[str, str]]:
    errors = validate_registration_payload(
        registration, source_spec_sha256=source_spec_sha256
    )
    evidence: dict[str, str] = {}
    if spec.get("status") != BLOCKED_SPEC_STATUS:
        errors.append("source spec is not the frozen blocked registration source")
    if readiness.get("spec", {}).get("sha256") != source_spec_sha256:
        errors.append("readiness references a different source spec SHA-256")

    gates = readiness.get("gates", {})
    for gate in PREREQUISITE_GATES:
        if gates.get(gate) is not True:
            errors.append(f"prerequisite gate not passed: {gate}")
    if gates.get("decision_policy_registered_before_model_results") is not False:
        errors.append("decision policy is already registered")

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

    label_path = resolve_from_root(spec["execution_inputs"]["full_development_labels"])
    if label_path.exists() or readiness.get("full_development_labels", {}).get("exists"):
        errors.append("full development target label file already exists")

    review = readiness.get("review_response", {})
    review_hash = _hash_evidence(review.get("path"), "public review response", errors)
    if review_hash is not None:
        evidence["public_review_response_sha256"] = review_hash
        if review.get("sha256") != review_hash:
            errors.append("public review response SHA-256 mismatch")
    if review.get("complete_24") is not True:
        errors.append("public review response is not complete for 24 cases")

    margins = readiness.get("endpoint_margins", {})
    endpoint_hash = _hash_evidence(margins.get("path"), "endpoint registration", errors)
    if endpoint_hash is not None:
        evidence["endpoint_registration_sha256"] = endpoint_hash
        if margins.get("sha256") != endpoint_hash:
            errors.append("endpoint registration SHA-256 mismatch")
    if margins.get("registered") is not True:
        errors.append("endpoint margins remain unregistered")

    rationale_value = registration.get("rationale_document")
    if isinstance(rationale_value, str) and rationale_value.strip():
        rationale_path = resolve_from_root(rationale_value)
        if "private" in {part.lower() for part in rationale_path.parts}:
            errors.append("rationale document must not be under a private path")
        elif not rationale_path.exists():
            errors.append("rationale document does not exist")
        else:
            evidence["rationale_document_sha256"] = sha256_file(rationale_path)
    return errors, evidence


def build_registered_candidate(
    spec: dict[str, Any],
    registration: dict[str, Any],
    evidence: dict[str, str],
) -> dict[str, Any]:
    candidate = copy.deepcopy(spec)
    candidate["status"] = REGISTERED_SPEC_STATUS
    policy = candidate["decision_policy"]
    policy["status"] = REGISTERED_POLICY_STATUS
    for key in POLICY_KEYS:
        policy[key] = float(registration["decision_thresholds"][key])
    policy["registration_provenance"] = {
        "registered_before_target_values_loaded": True,
        "registered_at": registration["registered_at"],
        "registered_by": registration["registered_by"],
        "public_review_response_sha256": evidence["public_review_response_sha256"],
        "endpoint_registration_sha256": evidence["endpoint_registration_sha256"],
        "rationale_document": registration["rationale_document"],
        "rationale_document_sha256": evidence["rationale_document_sha256"],
    }
    return candidate


def load_validator() -> Any:
    from solar_filament.s1 import harm_first_model_spec

    return harm_first_model_spec


def candidate_structure_errors(
    candidate: dict[str, Any], validator: Any | None = None
) -> list[str]:
    validator = load_validator() if validator is None else validator
    feature_path = resolve_from_root(candidate["feature_contract"]["path"])
    registry_path = resolve_from_root(candidate["registry"]["path"]).with_suffix(".json")
    return validator.validate_structure(
        candidate,
        load_json(feature_path),
        load_json(registry_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--candidate-out", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--require-registered", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = args.spec.resolve()
    readiness_path = args.readiness.resolve()
    registration_path = args.registration.resolve()
    audit_path = args.audit_out.resolve()
    candidate_path = args.candidate_out.resolve()
    if candidate_path == spec_path:
        raise ValueError("candidate output must not overwrite the canonical model spec")

    spec = load_json(spec_path)
    readiness = load_json(readiness_path)
    registration = load_json(registration_path)
    source_spec_sha256 = sha256_file(spec_path)
    errors, evidence = registration_prerequisite_errors(
        spec,
        readiness,
        registration,
        source_spec_sha256=source_spec_sha256,
    )
    candidate: dict[str, Any] | None = None
    candidate_errors: list[str] = []
    if not errors:
        candidate = build_registered_candidate(spec, registration, evidence)
        candidate_errors = candidate_structure_errors(candidate)
        errors.extend(candidate_errors)
    candidate_written = False
    candidate_sha256: str | None = None
    if not errors and candidate is not None:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        candidate_written = True
        candidate_sha256 = sha256_file(candidate_path)

    audit = {
        "schema_version": 1,
        "status": "registered_candidate_generated" if candidate_written else "registration_blocked",
        "registration_permitted": candidate_written,
        "scope": {
            "public_review_response_content_loaded": False,
            "private_key_read": False,
            "locked_rows_read": 0,
            "feature_values_loaded": False,
            "target_values_loaded": False,
            "model_fitting_performed": False,
            "threshold_selection_performed": False,
            "canonical_spec_overwritten": False,
        },
        "input_hashes": {
            "source_spec": source_spec_sha256,
            "readiness": sha256_file(readiness_path),
            "registration_payload": sha256_file(registration_path),
        },
        "evidence_hashes": evidence,
        "errors": errors,
        "candidate_structure_errors": candidate_errors,
        "candidate": {
            "path": str(candidate_path),
            "written": candidate_written,
            "sha256": candidate_sha256,
        },
        "limitations": [
            "This transaction never edits the canonical model spec.",
            "A generated candidate still requires explicit review and canonical promotion.",
            "Policy registration is not evidence of model performance.",
        ],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "registration_permitted": candidate_written,
        "error_count": len(errors),
        "errors": errors,
        "candidate_written": candidate_written,
        "audit": str(audit_path),
    }, indent=2, ensure_ascii=False))
    if args.require_registered and not candidate_written:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
