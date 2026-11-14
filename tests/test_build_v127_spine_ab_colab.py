"""v127 spine A/B builder contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "solar_filament_v127_spine_ab_colab.ipynb"


@pytest.fixture(scope="module")
def nb():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_v127", ROOT / "scratch/build_v127_spine_ab_colab.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    script = (ROOT / "scratch/run_fold_oof_eval_consensus.py").read_text(
        encoding="utf-8")
    return builder.build_notebook(script)


def _src(nb, i):
    return "".join(nb["cells"][i]["source"])


def test_notebook_builds_and_matches_disk(nb):
    built = nb
    disk = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert [c["source"] for c in built["cells"]] == [
        c["source"] for c in disk["cells"]]


def test_structure(nb):
    assert len(nb["cells"]) == 11
    assert nb["cells"][0]["cell_type"] == "markdown"


def test_split_cell_uses_pinned_ledgers(nb):
    s = _src(nb, 2)
    assert "time_block_assignment.csv" in s
    assert "spine_aux_prereg_exclusions_2026-11-13.csv" in s
    assert "a56372ea" in s
    assert "len(fit_files) == 365" in s
    assert "len(eval_files) == 180" in s


def test_locked_eval_isolated(nb):
    # locked_evaluation must never enter fit or eval
    assert "locked_evaluation" in _src(nb, 0)
    s2 = _src(nb, 2)
    assert "development" in s2


def test_pseudo_class_and_split_transform(nb):
    s = _src(nb, 4)
    assert "SplitSpineSegments" in s
    assert "nc=1" in s  # model stays single-class
    s3 = _src(nb, 3)
    assert "SPINE_CLS" in s3 and "nc: 2" in s3  # dataset yaml


def test_aux_weight_fixed_not_selected(nb):
    assert "AUX_WEIGHT = 0.5" in _src(nb, 4)


def test_eval_uses_extended_flags(nb):
    s = _src(nb, 8)
    assert "--prod-from" in s and "--eval-files" in s
    assert "trusted-sha256" in s
    # eval script is the embedded repo copy, not the stale dataset one
    assert "run_fold_oof_eval_consensus.py" in _src(nb, 1)
    assert "CONSENSUS_SCRIPT" in _src(nb, 1)


def test_gate_is_preregistered(nb):
    s = _src(nb, 10)
    assert "dpq >= 0.010" in s and "rec_gain >= 0.02" in s and "fp_rel <= 0.05" in s
    assert "pilot_positive" in s and "pilot_negative" in s
