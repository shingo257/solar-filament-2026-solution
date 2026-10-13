"""Exercise per-invocation staging without loading models or running inference."""

import ast
import shutil
import tempfile
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1]


def _entrypoint(worker):
    source = (MODULE / "scratch" / "run_test_submission.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {"tempfile": tempfile, "Path": Path, "_run_submission": worker}
    exec(compile(tree, "submission-entrypoint", "exec"), scope)
    return scope["main"]


def test_changed_equal_size_inputs_receive_fresh_staging(tmp_path):
    source = tmp_path / "drive"
    source.mkdir()
    weight = source / "best.pt"
    image = source / "test.jpeg"
    seen = []

    def worker(staging):
        assert not list(staging.iterdir())
        for path in (weight, image):
            shutil.copy2(path, staging / path.name)
        seen.append((staging, (staging / weight.name).read_bytes(),
                     (staging / image.name).read_bytes()))
        return 0

    main = _entrypoint(worker)
    for data in (b"old", b"new"):
        weight.write_bytes(data)
        image.write_bytes(data)
        assert main() == 0
        assert not seen[-1][0].exists()
    assert seen[0][0] != seen[1][0]
    assert seen[0][1:] == (b"old", b"old")
    assert seen[1][1:] == (b"new", b"new")


def test_staging_is_removed_when_inference_fails():
    seen = []

    def worker(staging):
        seen.append(staging)
        (staging / "partial.jpeg").write_bytes(b"partial")
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        _entrypoint(worker)()
    assert not seen[0].exists()


def test_kaggle_package_matches_canonical_reproduction():
    canonical = MODULE / "solar_filament_submission_reproduction.ipynb"
    packaged = MODULE / "output" / "kaggle_kernel" / canonical.name
    assert packaged.read_bytes() == canonical.read_bytes()
