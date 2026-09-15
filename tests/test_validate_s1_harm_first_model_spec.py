from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import solar_filament.s1.harm_first_model_spec as MODULE

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json"
FEATURE_PATH = ROOT / "output" / "s0" / "s1_set_context_features.json"
REGISTRY_PATH = ROOT / "output" / "s0" / "s1_grouped_nested_split_registry.json"


def payloads() -> tuple[dict, dict, dict]:
    return (
        json.loads(SPEC_PATH.read_text(encoding="utf-8")),
        json.loads(FEATURE_PATH.read_text(encoding="utf-8")),
        json.loads(REGISTRY_PATH.read_text(encoding="utf-8")),
    )


def registered_spec() -> dict:
    spec, _, _ = payloads()
    modified = copy.deepcopy(spec)
    modified["status"] = MODULE.REGISTERED_SPEC_STATUS
    policy = modified["decision_policy"]
    policy["status"] = MODULE.REGISTERED_POLICY_STATUS
    for key in MODULE.POLICY_THRESHOLD_KEYS:
        policy[key] = 0.5
    policy["minimum_selection_rate"] = 0.1
    policy["registration_provenance"] = {
        "registered_before_target_values_loaded": True,
        "registered_at": "2026-08-25T00:00:00+09:00",
        "registered_by": "test_operator",
        "public_review_response_sha256": "a" * 64,
        "endpoint_registration_sha256": "b" * 64,
        "rationale_document": "S1_POLICY_REGISTRATION.md",
        "rationale_document_sha256": "c" * 64,
    }
    return modified


def test_real_spec_structure_is_valid() -> None:
    spec, feature, registry = payloads()
    assert MODULE.validate_structure(spec, feature, registry) == []


def test_unknown_feature_is_rejected() -> None:
    spec, feature, registry = payloads()
    modified = copy.deepcopy(spec)
    modified["heads"][0]["features"][0] = "unknown_feature"
    errors = MODULE.validate_structure(modified, feature, registry)
    assert any("unknown features" in error for error in errors)


def test_site_or_year_feature_is_rejected() -> None:
    spec, feature, registry = payloads()
    modified = copy.deepcopy(spec)
    modified["heads"][0]["features"][0] = "site_code"
    feature["feature_contract"]["gt_free_set_context_features"].append("site_code")
    errors = MODULE.validate_structure(modified, feature, registry)
    assert any("forbidden features" in error for error in errors)


def test_event_gate_below_contract_is_rejected() -> None:
    spec, feature, registry = payloads()
    modified = copy.deepcopy(spec)
    modified["head_sufficiency_contract"]["minimum_candidate_events_per_outer_training"] = 69
    errors = MODULE.validate_structure(modified, feature, registry)
    assert any("event gate" in error for error in errors)


def test_premature_numeric_threshold_is_rejected() -> None:
    spec, feature, registry = payloads()
    modified = copy.deepcopy(spec)
    modified["decision_policy"]["count_harm_max_probability"] = 0.1
    errors = MODULE.validate_structure(modified, feature, registry)
    assert any("must remain TBD" in error for error in errors)


def test_pending_decision_policy_is_an_explicit_fit_blocker() -> None:
    spec, _, _ = payloads()
    registered, errors = MODULE.decision_policy_status(spec)
    assert registered is False
    assert errors == []


def test_registered_decision_policy_requires_pre_target_provenance() -> None:
    modified = registered_spec()
    modified["decision_policy"]["registration_provenance"][
        "registered_before_target_values_loaded"
    ] = False
    registered, errors = MODULE.decision_policy_status(modified)
    assert registered is False
    assert any("before target loading" in error for error in errors)


def test_fully_registered_decision_policy_is_structurally_valid() -> None:
    modified = registered_spec()
    _, feature, registry = payloads()
    registered, errors = MODULE.decision_policy_status(modified)
    assert registered is True
    assert errors == []
    assert MODULE.validate_structure(modified, feature, registry) == []


def test_label_column_must_match_head_name() -> None:
    spec, feature, registry = payloads()
    spec["heads"][0]["label_column"] = "wrong_label"
    errors = MODULE.validate_structure(spec, feature, registry)
    assert any("label columns" in error for error in errors)


def test_missing_review_response_fails_closed(tmp_path: Path) -> None:
    status = MODULE.review_response_status(tmp_path / "missing.json")
    assert status == {
        "path": str(tmp_path / "missing.json"),
        "exists": False,
        "complete_24": False,
        "response_count": 0,
    }


def test_complete_public_review_response_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "responses.json"
    path.write_text(
        json.dumps({
            "responses": {
                f"case_{index:02d}": {"choice": "A"} for index in range(24)
            }
        }),
        encoding="utf-8",
    )
    status = MODULE.review_response_status(path)
    assert status["complete_24"] is True
    assert status["response_count"] == 24


def test_table_status_requires_4899_unique_complete_rows(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    pd.DataFrame([
        {"file_name": "a.jpeg", "fold": 0, "candidate_index": 0, "feature": 1.0}
    ]).to_csv(path, index=False)
    status = MODULE.table_status(path, {"feature"})
    assert status["exists"] is True
    assert status["rows"] == 1
    assert status["complete_4899"] is False


def test_label_table_without_required_head_columns_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "identities_only.csv"
    pd.DataFrame({
        "file_name": [f"case_{index}.jpeg" for index in range(4_899)],
        "fold": [0] * 4_899,
        "candidate_index": [0] * 4_899,
    }).to_csv(path, index=False)
    status = MODULE.table_status(path, {"count_harm"}, binary_columns={"count_harm"})
    assert status["complete_4899"] is False
    assert "error" in status


@pytest.mark.parametrize("bad_value", [-1, 0.5, 2])
def test_label_table_rejects_nonbinary_head_values(
    tmp_path: Path, bad_value: float
) -> None:
    path = tmp_path / f"bad_{bad_value}.csv"
    values = [0] * 4_899
    values[-1] = bad_value
    pd.DataFrame({
        "file_name": [f"case_{index}.jpeg" for index in range(4_899)],
        "fold": [0] * 4_899,
        "candidate_index": [0] * 4_899,
        "count_harm": values,
    }).to_csv(path, index=False)
    status = MODULE.table_status(path, {"count_harm"}, binary_columns={"count_harm"})
    assert status["binary_columns_valid"] is False
    assert status["complete_4899"] is False


def test_complete_binary_label_table_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "complete.csv"
    pd.DataFrame({
        "file_name": [f"case_{index}.jpeg" for index in range(4_899)],
        "fold": [0] * 4_899,
        "candidate_index": [0] * 4_899,
        "count_harm": [index % 2 for index in range(4_899)],
    }).to_csv(path, index=False)
    status = MODULE.table_status(path, {"count_harm"}, binary_columns={"count_harm"})
    assert status["binary_columns_valid"] is True
    assert status["complete_4899"] is True


def test_margins_status_detects_tbd(tmp_path: Path) -> None:
    path = tmp_path / "registration.md"
    path.write_text("primary: TBD\n", encoding="utf-8")
    status = MODULE.margins_status(path)
    assert status["registered"] is False
    assert status["tbd_occurrences"] == 1
