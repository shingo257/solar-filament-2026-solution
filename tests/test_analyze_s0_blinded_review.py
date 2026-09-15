from __future__ import annotations

import json

import pytest

import scratch.analyze_s0_blinded_review as analysis
from scratch.analyze_s0_blinded_review import (
    analyze_completed_review,
    build_unblinded_rows,
    summarize_rows,
    validate_blinded_payload,
    validate_private_key,
)


def response_payload(case_ids=("case_001", "case_002")):
    responses = {}
    for index, case_id in enumerate(case_ids):
        responses[case_id] = {
            "morphology": "A" if index == 0 else "tie",
            "catalog": "B" if index == 0 else "neither",
            "tags_a": ["spine_missing"] if index == 0 else [],
            "tags_b": ["false_positive"] if index == 0 else ["unclear"],
            "confidence": "4" if index == 0 else "2",
            "notes": "checked",
        }
    return {
        "package_id": analysis.PACKAGE_ID,
        "protocol_version": analysis.PROTOCOL_VERSION,
        "reviewer_id": "reviewer-1",
        "case_order": list(case_ids),
        "responses": responses,
    }


def private_key_payload():
    return {
        "package_id": analysis.PACKAGE_ID,
        "cases": [
            {
                "case_id": "case_001",
                "file_name": "a.png",
                "comparison": "comparison-1",
                "selection_stratum": "small",
                "output_a": "model-x",
                "output_b": "model-y",
            },
            {
                "case_id": "case_002",
                "file_name": "b.png",
                "comparison": "comparison-1",
                "selection_stratum": "dense",
                "output_a": "model-y",
                "output_b": "model-x",
            },
        ],
    }


def test_validate_blinded_payload_normalizes_confidence():
    normalized = validate_blinded_payload(
        response_payload(), ["case_001", "case_002"]
    )

    assert normalized["case_001"]["confidence"] == 4
    assert normalized["case_002"]["confidence"] == 2


def test_validate_blinded_payload_rejects_incomplete_case_set():
    payload = response_payload()
    del payload["responses"]["case_002"]

    with pytest.raises(ValueError, match="case set mismatch"):
        validate_blinded_payload(payload, ["case_001", "case_002"])


def test_validate_blinded_payload_rejects_unknown_tag():
    payload = response_payload()
    payload["responses"]["case_001"]["tags_a"] = ["training_label"]

    with pytest.raises(ValueError, match="unknown tags"):
        validate_blinded_payload(payload, ["case_001", "case_002"])


def test_validate_blinded_payload_requires_reviewer_id():
    payload = response_payload()
    payload["reviewer_id"] = " "

    with pytest.raises(ValueError, match="reviewer_id"):
        validate_blinded_payload(payload, ["case_001", "case_002"])


def test_unblind_and_summary_preserve_ties_and_map_output_tags():
    responses = validate_blinded_payload(
        response_payload(), ["case_001", "case_002"]
    )
    key = validate_private_key(
        private_key_payload(), ["case_001", "case_002"]
    )

    rows = build_unblinded_rows(responses, key)
    summary = summarize_rows(rows)["comparison-1"]

    assert rows[0]["morphology_preference"] == "model-x"
    assert rows[0]["catalog_preference"] == "model-y"
    assert rows[1]["morphology_preference"] == "tie"
    assert rows[1]["catalog_preference"] == "neither"
    assert summary["morphology"] == {
        "model_votes": {"model-x": 1},
        "tie": 1,
        "neither": 0,
        "decisive": 1,
    }
    assert summary["catalog"]["model_votes"] == {"model-y": 1}
    assert summary["error_tags"]["model-x"] == {
        "spine_missing": 1,
        "unclear": 1,
    }
    assert summary["confidence"]["low_confidence_case_ids"] == ["case_002"]


def test_incomplete_response_is_rejected_before_private_key_is_loaded(
    tmp_path, monkeypatch
):
    public_path = tmp_path / "selection_manifest.csv"
    public_path.write_text("case_id\ncase_001\n", encoding="utf-8")
    response_path = tmp_path / "responses.json"
    key_path = tmp_path / "private-key.json"
    calls = []

    def fake_load_json(path):
        calls.append(path)
        if path == response_path:
            return {
                "package_id": analysis.PACKAGE_ID,
                "protocol_version": analysis.PROTOCOL_VERSION,
                "reviewer_id": "",
                "responses": {},
            }
        raise AssertionError("private key must not be loaded")

    monkeypatch.setattr(analysis, "load_json", fake_load_json)

    with pytest.raises(ValueError, match="reviewer_id"):
        analyze_completed_review(response_path, public_path, key_path)
    assert calls == [response_path]


def test_completed_review_analysis_writes_source_hashes(tmp_path):
    public_path = tmp_path / "selection_manifest.csv"
    public_path.write_text(
        "case_id\ncase_001\ncase_002\n", encoding="utf-8"
    )
    response_path = tmp_path / "responses.json"
    response_path.write_text(
        json.dumps(response_payload()), encoding="utf-8"
    )
    key_path = tmp_path / "private-key.json"
    key_path.write_text(
        json.dumps(private_key_payload()), encoding="utf-8"
    )

    summary, rows = analyze_completed_review(
        response_path, public_path, key_path
    )

    assert summary["reviewed_cases"] == 2
    assert set(summary["source_sha256"]) == {
        "responses",
        "public_manifest",
        "private_key",
    }
    assert len(rows) == 2
    assert "training-label" in summary["interpretation_limit"]
