from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from solar_filament.s1 import nested_oof_execution_plan as MODULE


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json"
READINESS_PATH = ROOT / "output" / "s0" / "s1_harm_first_model_spec_readiness.json"


def real_inputs() -> tuple[dict, dict, pd.DataFrame]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    registry_path = MODULE.resolve_from_root(spec["registry"]["path"])
    registry = MODULE.load_registry(registry_path, spec["registry"]["sha256"])
    return spec, readiness, registry


def build_real_plan() -> dict:
    spec, readiness, registry = real_inputs()
    registry_path = MODULE.resolve_from_root(spec["registry"]["path"])
    return MODULE.build_execution_plan(
        spec,
        readiness,
        registry,
        spec_sha256=MODULE.sha256_file(SPEC_PATH),
        readiness_sha256=MODULE.sha256_file(READINESS_PATH),
        registry_sha256=MODULE.sha256_file(registry_path),
    )


def test_real_execution_plan_is_frozen_and_fit_blocked() -> None:
    plan = build_real_plan()
    assert plan["status"] == "execution_plan_frozen_fit_blocked"
    assert plan["fit_start_permitted"] is False
    assert plan["integrity_errors"] == []
    assert set(plan["blockers"]) == {
        "complete_s0_4_response",
        "all_endpoint_margins_registered",
        "decision_policy_registered_before_model_results",
        "full_development_labels_complete",
    }


def test_execution_plan_enumerates_exact_fit_cross_product() -> None:
    plan = build_real_plan()
    contract = plan["execution_contract"]
    assert contract["inner_fit_jobs"] == 3 * 5 * 4 * 3 == 180
    assert contract["outer_refit_jobs"] == 3 * 5 == 15
    assert contract["total_fit_jobs"] == 195
    assert contract["ownership_replay_jobs"] == 3
    assert contract["threshold_fit_jobs"] == 0
    job_ids = [job["job_id"] for job in plan["inner_selection_jobs"]]
    assert len(job_ids) == len(set(job_ids)) == 180


def test_every_inner_job_preserves_outer_training_accounting() -> None:
    plan = build_real_plan()
    cycles = plan["registry_summary"]["cycles"]
    for job in plan["inner_selection_jobs"]:
        outer_train = cycles[str(job["outer_fold"])]["outer_train"]
        assert job["validation_inner_fold"] not in job["training_inner_folds"]
        assert len(job["training_inner_folds"]) == 2
        assert (
            job["training"]["observations"] + job["validation"]["observations"]
            == outer_train["observations"]
        )
        assert (
            job["training"]["candidates"] + job["validation"]["candidates"]
            == outer_train["candidates"]
        )


def test_outer_refit_scores_each_test_once_per_head() -> None:
    plan = build_real_plan()
    cycles = plan["registry_summary"]["cycles"]
    assert len(plan["outer_refit_jobs"]) == 15
    for job in plan["outer_refit_jobs"]:
        cycle = cycles[str(job["outer_fold"])]
        assert job["training"] == cycle["outer_train"]
        assert job["test"] == cycle["outer_test"]
        assert job["outer_test_prediction_count"] == 1


def test_plan_never_loads_targets_or_private_material() -> None:
    plan = build_real_plan()
    assert plan["scope"] == {
        "registry_identity_and_counts_loaded": True,
        "feature_values_loaded": False,
        "target_values_loaded": False,
        "review_response_loaded": False,
        "private_key_read": False,
        "locked_rows_read": 0,
        "model_fitting_performed": False,
        "threshold_selection_performed": False,
    }


def test_fit_guard_raises_with_current_blockers() -> None:
    with pytest.raises(RuntimeError, match="decision_policy_registered"):
        MODULE.enforce_fit_ready(build_real_plan())


def test_readiness_spec_hash_mismatch_invalidates_plan() -> None:
    spec, readiness, registry = real_inputs()
    registry_path = MODULE.resolve_from_root(spec["registry"]["path"])
    plan = MODULE.build_execution_plan(
        spec,
        readiness,
        registry,
        spec_sha256="f" * 64,
        readiness_sha256=MODULE.sha256_file(READINESS_PATH),
        registry_sha256=MODULE.sha256_file(registry_path),
    )
    assert plan["status"] == "execution_plan_invalid"
    assert plan["fit_start_permitted"] is False
    assert "execution_plan_integrity" in plan["blockers"]


def test_registered_consistent_readiness_can_authorize_plan() -> None:
    spec, readiness, registry = real_inputs()
    modified_spec = copy.deepcopy(spec)
    modified_spec["status"] = MODULE.REGISTERED_SPEC_STATUS
    modified_spec["decision_policy"]["status"] = MODULE.REGISTERED_POLICY_STATUS
    modified_readiness = copy.deepcopy(readiness)
    for key in modified_readiness["gates"]:
        modified_readiness["gates"][key] = True
    modified_readiness["ready_to_fit"] = True
    modified_readiness["blockers"] = []
    modified_readiness["spec"]["sha256"] = "a" * 64
    plan = MODULE.build_execution_plan(
        modified_spec,
        modified_readiness,
        registry,
        spec_sha256="a" * 64,
        readiness_sha256="b" * 64,
        registry_sha256=spec["registry"]["sha256"],
    )
    assert plan["integrity_errors"] == []
    assert plan["fit_start_permitted"] is True
    assert plan["status"] == "execution_plan_ready"


def test_outer_training_leakage_is_rejected() -> None:
    spec, _, registry = real_inputs()
    modified = registry.copy()
    index = modified.index[
        (modified["outer_fold"] == 0) & (modified["outer_role"] == "outer_train")
    ][0]
    modified.loc[index, "source_fold"] = 0
    with pytest.raises(
        ValueError, match="source fold changed across outer cycles|outer training leakage"
    ):
        MODULE.validate_registry(modified, spec)


def test_registry_hash_mismatch_is_rejected() -> None:
    spec, _, _ = real_inputs()
    registry_path = MODULE.resolve_from_root(spec["registry"]["path"])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MODULE.load_registry(registry_path, "0" * 64)
