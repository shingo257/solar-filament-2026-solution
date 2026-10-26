"""v75 notebook entry points must require independently reviewed provenance
before IO, and the embedded TTA eval script must match the repo copy."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "solar_filament_v75_tta_guard_colab.ipynb"


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
        "build_v75", ROOT / "scratch/build_v75_tta_guard_colab.py")
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


def test_tta_guard_parameters_preregistered():
    """Guard constants must stay at the §2.30 preregistered values."""
    script = ROOT / "scratch/run_fold_oof_eval_tta.py"
    source = script.read_text(encoding="utf-8")
    ns = {"__file__": str(script)}
    exec(compile(source, "tta", "exec"), ns)  # noqa: S102
    assert ns["GUARD_AREA_RATIO"] == (0.85, 1.15)
    assert ns["GUARD_MAX_CC_DELTA"] == 1
    assert ns["GUARD_MIN_IOU"] == 0.85
    assert ns["GUARD_MIN_CC_AREA"] == 16
    assert set(ns["TTA_MODES"]) == {"none", "flip4", "flip4_guard"}


def test_tta_arm_inherits_reviewed_role_by_digest(tmp_path):
    """tta4/tta4g share the baseline ckpt: the arm name is not a manifest
    role, so binding must resolve through a byte-identical reviewed entry."""
    script = ROOT / "scratch/run_fold_oof_eval_tta.py"
    ns = {"__file__": str(script)}
    exec(compile(script.read_text(encoding="utf-8"), "tta", "exec"), ns)  # noqa: S102
    resolve = ns["resolve_binding_role"]
    reviewed = tmp_path / "refiner.pt"
    reviewed.write_bytes(b"reviewed-bytes")
    unreviewed = tmp_path / "other.pt"
    unreviewed.write_bytes(b"unreviewed-bytes")
    manifest = {"checkpoints": {
        "baseline": hashlib.sha256(b"reviewed-bytes").hexdigest(),
        "yolo": "0" * 64,
    }}
    assert resolve(reviewed, "baseline", manifest, {}) == "baseline"
    assert resolve(reviewed, "tta4", manifest, {}) == "baseline"
    assert resolve(reviewed, "tta4g", manifest, {}) == "baseline"
    # metadata-bearing ckpt (v73/v74-trained) keeps the arm name — the
    # embedded fold/folds_sha256 fields are checked downstream by
    # validate_checkpoint_binding's metadata branch.
    trained = {"fold": 2, "folds_sha256": "f" * 64}
    assert resolve(unreviewed, "ens3", manifest, trained) == "ens3"
    with pytest.raises(ValueError, match="provenance missing"):
        resolve(unreviewed, "tta4", manifest, {})


def test_parse_arm_accepts_ensemble_member_paths(tmp_path):
    script = ROOT / "scratch/run_fold_oof_eval_tta.py"
    ns = {"__file__": str(script)}
    exec(compile(script.read_text(encoding="utf-8"), "tta", "exec"), ns)  # noqa: S102
    parse_arm = ns["parse_arm"]
    a = tmp_path / "a.pt"; a.write_bytes(b"a")
    b = tmp_path / "b.pt"; b.write_bytes(b"b")
    name, paths, mode = parse_arm(f"ens={a}+{b}:flip4")
    assert name == "ens" and paths == [a, b] and mode == "flip4"
    name, paths, mode = parse_arm(f"single={a}")
    assert paths == [a] and mode == "none"
    with pytest.raises(ValueError):
        parse_arm(f"bad={a}+{tmp_path}/missing.pt:none")
