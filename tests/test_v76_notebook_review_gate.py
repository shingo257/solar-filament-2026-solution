"""v76 notebook entry points must require independently reviewed provenance
before IO, and the embedded TTA eval script must match the repo copy."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "solar_filament_v76_ensemble_colab.ipynb"


def test_notebook_requires_reviewed_pin_before_input_io(monkeypatch):
    monkeypatch.delenv("V73_PROVENANCE_SHA256", raising=False)
    monkeypatch.setattr(Path, "rglob", lambda *args: pytest.fail("input IO before review pin"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code")
    with pytest.raises(RuntimeError, match="independent.*review"):
        exec(compile(source, "nb", "exec"), {})  # noqa: S102 - run repository preflight with IO blocked


def test_generated_notebook_matches_builder():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_v76", ROOT / "scratch/build_v76_ensemble_colab.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    script = (ROOT / "scratch/run_fold_oof_eval_tta.py").read_text(encoding="utf-8")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    built = builder.build_notebook(script)
    assert [c["source"] for c in built["cells"]] == [c["source"] for c in notebook["cells"]]


def test_embedded_tta_script_matches_repo_file():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    tree = ast.parse(source)
    embedded = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "TTA_SCRIPT" for t in node.targets):
            embedded = ast.literal_eval(node.value)
    assert embedded is not None, "TTA_SCRIPT assignment missing in setup cell"
    repo = (ROOT / "scratch/run_fold_oof_eval_tta.py").read_text(encoding="utf-8")
    assert embedded == repo
    ast.parse(embedded)


def test_v76_ensemble_arms_preregistered():
    """The v76 eval cell must keep the preregistered arm specs: baseline,
    ens3 (3-member prob average, single view), ens3f (same members, flip4)."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][2]["source"])
    assert "baseline={baseline_ckpt}:none" in source
    assert "ens3={ens_spec}:none" in source
    assert "ens3f={ens_spec}:flip4" in source
    summary = "".join(notebook["cells"][3]["source"])
    assert "d_f >= 0.005" in summary  # preregistered promotion threshold (§2.31)
