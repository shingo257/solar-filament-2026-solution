from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import solar_filament.s1.decision_policy as MODULE

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json"
READINESS_PATH = ROOT / "output" / "s0" / "s1_harm_first_model_spec_readiness.json"
TEMPLATE_PATH = ROOT / "S1_DECISION_POLICY_REGISTRATION_TEMPLATE_2026-08-24.json"


def real_payloads() -> tuple[dict, dict, dict]:
    return (
        json.loads(SPEC_PATH.read_text(encoding="utf-8")),
        json.loads(READINESS_PATH.read_text(encoding="utf-8")),
        json.loads(TEMPLATE_PATH.read_text(encoding="utf-8")),
    )


def complete_registration(tmp_path: Path) -> dict:
    rationale = tmp_path / "policy_rationale.md"
    rationale.write_text("# Frozen policy rationale\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "status": MODULE.READY_REGISTRATION_STATUS,
        "source_spec_sha256": MODULE.sha256_file(SPEC_PATH),
        "registered_at": "2026-08-25T09:00:00+09:00",
        "registered_by": "operations-owner",
        "rationale_document": str(rationale),
        "decision_thresholds": {
            "final_pq_benefit_min_probability": 0.6,
            "spine_benefit_min_probability": 0.6,
            "count_harm_max_probability": 0.2,
            "precision_harm_max_probability": 0.2,
            "merge_harm_max_probability": 0.1,
            "minimum_selection_rate": 0.1,
        },
        "attestations": {key: True for key in MODULE.ATTESTATION_KEYS},
    }


def ready_prerequisites(tmp_path: Path) -> tuple[dict, dict]:
    spec, readiness, _ = real_payloads()
    modified = copy.deepcopy(readiness)
    modified["spec"]["sha256"] = MODULE.sha256_file(SPEC_PATH)
    for gate in MODULE.PREREQUISITE_GATES:
        modified["gates"][gate] = True
    modified["gates"]["decision_policy_registered_before_model_results"] = False
    modified["ready_to_fit"] = False
    modified["full_development_labels"] = {
        "path": str(tmp_path / "missing_labels.csv"),
        "exists": False,
        "rows": 0,
        "complete_4899": False,
    }
    spec = copy.deepcopy(spec)
    spec["execution_inputs"]["full_development_labels"] = str(
        tmp_path / "missing_labels.csv"
    )

    response = tmp_path / "review_response.json"
    response.write_text('{"responses": "public evidence only"}\n', encoding="utf-8")
    modified["review_response"] = {
        "path": str(response),
        "exists": True,
        "complete_24": True,
        "response_count": 24,
        "sha256": MODULE.sha256_file(response),
    }
    endpoint = tmp_path / "registered_endpoint.md"
    endpoint.write_text("all numeric margins registered\n", encoding="utf-8")
    modified["endpoint_margins"] = {
        "path": str(endpoint),
        "exists": True,
        "registered": True,
        "tbd_occurrences": 0,
        "sha256": MODULE.sha256_file(endpoint),
    }
    return spec, modified


def test_real_template_is_incomplete_and_current_registration_is_blocked() -> None:
    spec, readiness, registration = real_payloads()
    errors, _ = MODULE.registration_prerequisite_errors(
        spec,
        readiness,
        registration,
        source_spec_sha256=MODULE.sha256_file(SPEC_PATH),
    )
    assert any("status must be" in error for error in errors)
    assert any("complete_s0_4_response" in error for error in errors)
    assert any("all_endpoint_margins_registered" in error for error in errors)


def test_complete_pre_target_registration_builds_valid_candidate(tmp_path: Path) -> None:
    spec, readiness = ready_prerequisites(tmp_path)
    registration = complete_registration(tmp_path)
    errors, evidence = MODULE.registration_prerequisite_errors(
        spec,
        readiness,
        registration,
        source_spec_sha256=MODULE.sha256_file(SPEC_PATH),
    )
    assert errors == []
    candidate = MODULE.build_registered_candidate(spec, registration, evidence)
    assert candidate["status"] == MODULE.REGISTERED_SPEC_STATUS
    assert candidate["decision_policy"]["status"] == MODULE.REGISTERED_POLICY_STATUS
    assert candidate["decision_policy"]["count_harm_max_probability"] == 0.2
    provenance = candidate["decision_policy"]["registration_provenance"]
    assert provenance["registered_before_target_values_loaded"] is True
    assert provenance["rationale_document_sha256"] == evidence[
        "rationale_document_sha256"
    ]
    assert MODULE.candidate_structure_errors(candidate) == []


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.5"])
def test_invalid_probability_value_is_rejected(tmp_path: Path, value: object) -> None:
    registration = complete_registration(tmp_path)
    registration["decision_thresholds"]["count_harm_max_probability"] = value
    errors = MODULE.validate_registration_payload(
        registration, source_spec_sha256=MODULE.sha256_file(SPEC_PATH)
    )
    assert any("count_harm_max_probability" in error for error in errors)


def test_minimum_selection_rate_cannot_be_zero(tmp_path: Path) -> None:
    registration = complete_registration(tmp_path)
    registration["decision_thresholds"]["minimum_selection_rate"] = 0.0
    errors = MODULE.validate_registration_payload(
        registration, source_spec_sha256=MODULE.sha256_file(SPEC_PATH)
    )
    assert any("minimum_selection_rate" in error for error in errors)


def test_registered_at_requires_timezone(tmp_path: Path) -> None:
    registration = complete_registration(tmp_path)
    registration["registered_at"] = "2026-08-25T09:00:00"
    errors = MODULE.validate_registration_payload(
        registration, source_spec_sha256=MODULE.sha256_file(SPEC_PATH)
    )
    assert any("timezone offset" in error for error in errors)


def test_existing_target_label_file_blocks_registration(tmp_path: Path) -> None:
    spec, readiness = ready_prerequisites(tmp_path)
    registration = complete_registration(tmp_path)
    label_path = Path(spec["execution_inputs"]["full_development_labels"])
    label_path.write_text("target values already exist\n", encoding="utf-8")
    errors, _ = MODULE.registration_prerequisite_errors(
        spec,
        readiness,
        registration,
        source_spec_sha256=MODULE.sha256_file(SPEC_PATH),
    )
    assert any("target label file already exists" in error for error in errors)


def test_public_response_hash_mismatch_blocks_registration(tmp_path: Path) -> None:
    spec, readiness = ready_prerequisites(tmp_path)
    registration = complete_registration(tmp_path)
    readiness["review_response"]["sha256"] = "0" * 64
    errors, _ = MODULE.registration_prerequisite_errors(
        spec,
        readiness,
        registration,
        source_spec_sha256=MODULE.sha256_file(SPEC_PATH),
    )
    assert any("public review response SHA-256 mismatch" in error for error in errors)


def test_private_rationale_path_is_rejected(tmp_path: Path) -> None:
    spec, readiness = ready_prerequisites(tmp_path)
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    rationale = private_dir / "rationale.md"
    rationale.write_text("must remain public\n", encoding="utf-8")
    registration = complete_registration(tmp_path)
    registration["rationale_document"] = str(rationale)
    errors, _ = MODULE.registration_prerequisite_errors(
        spec,
        readiness,
        registration,
        source_spec_sha256=MODULE.sha256_file(SPEC_PATH),
    )
    assert any("must not be under a private path" in error for error in errors)
