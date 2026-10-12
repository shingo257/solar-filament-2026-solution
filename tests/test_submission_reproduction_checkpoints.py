"""Exercise checkpoint dispatch without loading weights or training."""

import ast
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE = Path(__file__).resolve().parents[1]


def _dispatch(tmp_path, existing_epoch, external_epoch=None):
    notebook = json.loads(
        (MODULE / "solar_filament_submission_reproduction.ipynb").read_text(encoding="utf-8")
    )
    source = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if "def train_refiner_fold(" in "".join(cell.get("source", []))
    )
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "train_refiner_fold")
    # Stop exactly where the real training branch begins; no model code runs.
    boundary = next(index for index, node in enumerate(function.body)
                    if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "train_files_fold")
    function.body = function.body[:boundary] + ast.parse("return 'training'").body
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    checkpoint = tmp_path / "mediseg_edgeattnet_refiner_fold0_resume.pt"
    external = tmp_path / "external.pt"
    states = {}
    if existing_epoch is not None:
        checkpoint.write_text("partial", encoding="utf-8")
        states[checkpoint] = {"epoch": existing_epoch}
    if external_epoch is not None:
        external.write_text("attached", encoding="utf-8")
        states[external] = {"epoch": external_epoch}
    scope = {
        "CFG": SimpleNamespace(ARTIFACT_ROOT=tmp_path, REFINER_EPOCHS=14),
        "torch": SimpleNamespace(load=lambda path, **kwargs: states[path]),
        "shutil": shutil,
        "resolve_pretrained": lambda patterns: external if external_epoch is not None else checkpoint,
    }
    exec(compile(tree, "checkpoint-dispatch", "exec"), scope)
    return scope["train_refiner_fold"](0), checkpoint


def test_interrupted_local_checkpoint_falls_through_to_training(tmp_path):
    result, checkpoint = _dispatch(tmp_path, existing_epoch=2)
    assert result == "training"
    assert checkpoint.read_text(encoding="utf-8") == "partial"


def test_incomplete_attachment_is_not_accepted_as_complete(tmp_path):
    result, checkpoint = _dispatch(tmp_path, existing_epoch=None, external_epoch=2)
    assert result == "training"
    assert not checkpoint.exists()


@pytest.mark.parametrize("existing_epoch,external_epoch", [(13, None), (None, 13)])
def test_complete_checkpoint_is_reused(tmp_path, existing_epoch, external_epoch):
    result, checkpoint = _dispatch(tmp_path, existing_epoch, external_epoch)
    assert result == checkpoint
    assert checkpoint.exists()
