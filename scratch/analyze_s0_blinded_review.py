"""Validate and summarize a completed S0-4 blinded visual review.

The private key is loaded only after the response payload passes all public,
pre-unblinding completeness checks.  Outputs are descriptive review evidence;
they are never exported as training labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "output/s0/blinded_review_v1"
PACKAGE_ID = "s0-blinded-review-2026-08-24-v1"
PROTOCOL_VERSION = "s0-visual-review-v1"
PREFERENCES = {"A", "B", "tie", "neither"}
ERROR_TAGS = {
    "spine_missing",
    "fragmentation",
    "merge",
    "false_positive",
    "limb_artifact",
    "unclear",
}
REQUIRED_FIELDS = {"morphology", "catalog", "tags_a", "tags_b", "confidence", "notes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def public_case_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    case_ids = [str(row.get("case_id", "")).strip() for row in rows]
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError("public manifest has an empty or missing case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("public manifest contains duplicate case_ids")
    return case_ids


def _validate_tags(case_id: str, field: str, value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(tag, str) for tag in value):
        raise ValueError(f"{case_id}.{field} must be a list of strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{case_id}.{field} contains duplicate tags")
    unknown = sorted(set(value) - ERROR_TAGS)
    if unknown:
        raise ValueError(f"{case_id}.{field} has unknown tags: {unknown}")
    return value


def validate_blinded_payload(
    payload: dict[str, Any],
    expected_case_ids: list[str],
    package_id: str = PACKAGE_ID,
) -> dict[str, dict[str, Any]]:
    """Validate completeness without opening or depending on the private key."""
    if payload.get("package_id") != package_id:
        raise ValueError("response package_id does not match the public package")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("response protocol_version does not match")
    reviewer_id = payload.get("reviewer_id")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id must be non-empty")
    responses = payload.get("responses")
    if not isinstance(responses, dict):
        raise ValueError("responses must be an object")
    expected = set(expected_case_ids)
    actual = set(responses)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"response case set mismatch; missing={missing}, unknown={unknown}")
    case_order = payload.get("case_order")
    if case_order is not None:
        if not isinstance(case_order, list) or len(case_order) != len(set(case_order)):
            raise ValueError("case_order must be a duplicate-free list")
        if set(case_order) != expected:
            raise ValueError("case_order does not match the public manifest")

    normalized: dict[str, dict[str, Any]] = {}
    for case_id in expected_case_ids:
        response = responses[case_id]
        if not isinstance(response, dict):
            raise ValueError(f"{case_id} response must be an object")
        missing_fields = sorted(REQUIRED_FIELDS - set(response))
        if missing_fields:
            raise ValueError(f"{case_id} is missing fields: {missing_fields}")
        morphology = response.get("morphology")
        catalog = response.get("catalog")
        if morphology not in PREFERENCES:
            raise ValueError(f"{case_id}.morphology is incomplete or invalid")
        if catalog not in PREFERENCES:
            raise ValueError(f"{case_id}.catalog is incomplete or invalid")
        try:
            confidence = int(response.get("confidence"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{case_id}.confidence must be 1..5") from error
        if not 1 <= confidence <= 5:
            raise ValueError(f"{case_id}.confidence must be 1..5")
        notes = response.get("notes")
        if not isinstance(notes, str):
            raise ValueError(f"{case_id}.notes must be a string")
        normalized[case_id] = {
            "morphology": morphology,
            "catalog": catalog,
            "tags_a": _validate_tags(case_id, "tags_a", response.get("tags_a")),
            "tags_b": _validate_tags(case_id, "tags_b", response.get("tags_b")),
            "confidence": confidence,
            "notes": notes,
        }
    return normalized


def validate_private_key(
    payload: dict[str, Any],
    expected_case_ids: list[str],
    package_id: str = PACKAGE_ID,
) -> dict[str, dict[str, Any]]:
    if payload.get("package_id") != package_id:
        raise ValueError("private key package_id does not match")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("private key cases must be a list")
    keyed: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("private key case must be an object")
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in keyed:
            raise ValueError("private key has an empty or duplicate case_id")
        required = {"comparison", "output_a", "output_b", "file_name"}
        if not required.issubset(case):
            raise ValueError(f"private key case is incomplete: {case_id}")
        if case["output_a"] == case["output_b"]:
            raise ValueError(f"private key outputs are identical: {case_id}")
        keyed[case_id] = case
    if set(keyed) != set(expected_case_ids):
        raise ValueError("private key case set does not match the public manifest")
    return keyed


def unblind_preference(preference: str, key: dict[str, Any]) -> str:
    if preference == "A":
        return str(key["output_a"])
    if preference == "B":
        return str(key["output_b"])
    return preference


def build_unblinded_rows(
    responses: dict[str, dict[str, Any]],
    key_by_case: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case_id, response in responses.items():
        key = key_by_case[case_id]
        rows.append({
            "case_id": case_id,
            "file_name": key["file_name"],
            "comparison": key["comparison"],
            "selection_stratum": key.get("selection_stratum", ""),
            "output_a": key["output_a"],
            "output_b": key["output_b"],
            "morphology_preference": unblind_preference(response["morphology"], key),
            "catalog_preference": unblind_preference(response["catalog"], key),
            "tags_output_a": "|".join(response["tags_a"]),
            "tags_output_b": "|".join(response["tags_b"]),
            "confidence": response["confidence"],
            "notes": response["notes"],
        })
    return rows


def _question_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = Counter(str(row[field]) for row in rows)
    special = {name: int(counts.pop(name, 0)) for name in ("tie", "neither")}
    return {
        "model_votes": dict(sorted(counts.items())),
        **special,
        "decisive": int(sum(counts.values())),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["comparison"])].append(row)
    summaries = {}
    for comparison, group in sorted(grouped.items()):
        tag_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in group:
            for column, model_column in (
                ("tags_output_a", "output_a"),
                ("tags_output_b", "output_b"),
            ):
                tags = [tag for tag in str(row[column]).split("|") if tag]
                tag_counts[str(row[model_column])].update(tags)
        confidences = [int(row["confidence"]) for row in group]
        summaries[comparison] = {
            "cases": len(group),
            "morphology": _question_summary(group, "morphology_preference"),
            "catalog": _question_summary(group, "catalog_preference"),
            "error_tags": {
                model: dict(sorted(counts.items()))
                for model, counts in sorted(tag_counts.items())
            },
            "confidence": {
                "mean": sum(confidences) / len(confidences),
                "distribution": {
                    str(value): confidences.count(value) for value in range(1, 6)
                },
                "low_confidence_case_ids": [
                    str(row["case_id"]) for row in group if int(row["confidence"]) <= 2
                ],
            },
        }
    return summaries


def analyze_completed_review(
    response_path: Path,
    public_manifest_path: Path,
    private_key_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate publicly first; load the private key only after completion."""
    expected_case_ids = public_case_ids(public_manifest_path)
    response_payload = load_json(response_path)
    responses = validate_blinded_payload(response_payload, expected_case_ids)
    key_payload = load_json(private_key_path)
    key_by_case = validate_private_key(key_payload, expected_case_ids)
    rows = build_unblinded_rows(responses, key_by_case)
    summary = {
        "package_id": PACKAGE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed descriptive blinded-review analysis",
        "reviewer_id": response_payload["reviewer_id"].strip(),
        "reviewed_cases": len(rows),
        "interpretation_limit": (
            "Single-reviewer descriptive evidence; n=8 per comparison. "
            "Do not treat as a training-label export or independent performance estimate."
        ),
        "source_sha256": {
            "responses": sha256_file(response_path),
            "public_manifest": sha256_file(public_manifest_path),
            "private_key": sha256_file(private_key_path),
        },
        "comparisons": summarize_rows(rows),
    }
    return summary, rows


def write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty review table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=PACKAGE_ROOT / "selection_manifest.csv",
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        default=PACKAGE_ROOT / "private/blind_key.json",
    )
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_ROOT / "review_results")
    args = parser.parse_args()
    summary, rows = analyze_completed_review(
        args.responses,
        args.public_manifest,
        args.private_key,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "review_summary.json"
    cases_path = args.out_dir / "unblinded_case_review.csv"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_case_csv(cases_path, rows)
    print(json.dumps({
        "status": summary["status"],
        "reviewed_cases": summary["reviewed_cases"],
        "summary": str(summary_path),
        "cases": str(cases_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
