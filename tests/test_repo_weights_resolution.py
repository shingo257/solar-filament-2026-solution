"""repro notebook の repo weights 解決順序テスト (2026-10-23 要件対応).

repo 同梱 weights/ が dataset/Drive より優先されること、同梱 manifest の
SHA256 検証が不一致を拒否することを、生成 notebook の関数を AST 抽出して
合成ファイルだけで確認する。
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE = Path(__file__).resolve().parents[1]


def _extract_funcs(names):
    nb = json.loads((MODULE / "solar_filament_submission_reproduction.ipynb")
                    .read_text(encoding="utf-8"))
    found = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(line for line in "".join(cell["source"]).splitlines()
                           if not line.lstrip().startswith("%"))
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                found[node.name] = node
    missing = set(names) - set(found)
    assert not missing, f"notebook に関数なし: {missing}"
    return ast.Module(body=[found[n] for n in names], type_ignores=[])


def _scope(tmp_path, repo_dirs, dataset_root):
    scope = {
        "os": __import__("os"), "Path": Path, "json": json,
        "DATA_SEARCH_ROOTS": [dataset_root], "IS_KAGGLE": True,
        "CFG": SimpleNamespace(RUNS_ROOT=tmp_path / "nope_r",
                               ARTIFACT_ROOT=tmp_path / "nope_a"),
    }
    exec(compile(ast.fix_missing_locations(
        _extract_funcs(["repo_weights_roots", "verify_repo_weight",
                        "resolve_pretrained"])), "nb", "exec"), scope)
    scope["REPO_WEIGHTS"] = repo_dirs
    return scope


def _mk_weight(dir_path: Path, name: str, payload: bytes) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    f = dir_path / name
    f.write_bytes(payload * 1_200_000)  # >1MB サイズガードを通す
    return f


def test_repo_weights_take_priority_over_dataset(tmp_path):
    repo_w = tmp_path / "repo" / "weights"
    ds = tmp_path / "dataset"
    repo_f = _mk_weight(repo_w, "yolo_fold0_best.pt", b"R")
    ds.mkdir()
    (ds / "yolo_fold0_best.pt").write_bytes(b"D" * 1_200_000)
    (repo_w / "weights_manifest.json").write_text(json.dumps({
        "weights": [{"file_name": "yolo_fold0_best.pt",
                     "sha256": hashlib.sha256(repo_f.read_bytes()).hexdigest()}]}))
    scope = _scope(tmp_path, [repo_w], ds)
    assert scope["resolve_pretrained"](["yolo_fold0_best.pt"]) == repo_f


def test_repo_weight_sha_mismatch_raises(tmp_path):
    repo_w = tmp_path / "weights"
    bad = _mk_weight(repo_w, "yolo_fold0_best.pt", b"X")
    (repo_w / "weights_manifest.json").write_text(json.dumps({
        "weights": [{"file_name": "yolo_fold0_best.pt", "sha256": "0" * 64}]}))
    scope = _scope(tmp_path, [repo_w], tmp_path / "ds")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        scope["resolve_pretrained"](["yolo_fold0_best.pt"])


def test_falls_back_to_dataset_when_no_repo_weights(tmp_path):
    ds = tmp_path / "ds"
    ds_f = _mk_weight(ds, "yolo_fold0_best.pt", b"D")
    scope = _scope(tmp_path, [], tmp_path)
    assert scope["resolve_pretrained"](["yolo_fold0_best.pt"]) == ds_f


def test_returns_none_when_nothing_found(tmp_path):
    scope = _scope(tmp_path, [], tmp_path / "empty")
    assert scope["resolve_pretrained"](["nonexistent*.pt"]) is None


def test_weights_manifest_present_in_repo_dir_listing():
    """生成物の weights_manifest.json が repo 経路で参照可能な設計か。

    manifest は export スクリプトが weights/ に書き出す。notebook 側が
    manifest の隣に存在する file_name を検証することを確認する。
    """
    scope_names = _extract_funcs(["verify_repo_weight"])
    src = ast.unparse(scope_names)
    assert "weights_manifest.json" in src
    assert "sha256" in src
