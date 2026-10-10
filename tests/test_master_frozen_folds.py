"""Offline contracts for the actual master notebook's frozen observation folds."""

import ast
import hashlib
import json
import runpy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "solar_filament_master_pipeline_colab.ipynb"
BUILDER = ROOT / "scratch" / "build_master_pipeline_notebook.py"
NAMES = ["20260102000000Bh.jpeg", "20260102000100Bh.jpeg", "20260102000200Bh.jpeg"]
CSV = ("file_name,fold,reference_instances,labeler_count\n"
       + "\n".join(f"{name},{fold},4,2" for fold, name in enumerate(NAMES)) + "\n").encode()


def data_cell():
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][3]["source"]
    return "".join(raw) if isinstance(raw, list) else raw


def validator():
    tree = ast.parse(data_cell())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "load_frozen_folds")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["load_frozen_folds"]


def load(tmp_path, payload=CSV, names=NAMES, folds=3, digest=None):
    path = tmp_path / "grouped_folds.csv"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest() if digest is None else digest
    return validator()(path, names, folds, expected)


def test_frozen_fold_csv_keeps_extra_columns_and_exact_byte_identity(tmp_path):
    mapping, digest = load(tmp_path)
    assert mapping == dict(zip(NAMES, range(3)))
    assert digest == hashlib.sha256(CSV).hexdigest()


def test_missing_fold_file_is_never_generated(tmp_path):
    path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        validator()(path, NAMES, 3, hashlib.sha256(CSV).hexdigest())
    assert not path.exists()


@pytest.mark.parametrize("digest", ["", "0" * 64, "x" * 64, None, True])
def test_missing_or_mismatched_frozen_digest_is_rejected(tmp_path, digest):
    path = tmp_path / "grouped_folds.csv"
    path.write_bytes(CSV)
    with pytest.raises(ValueError):
        validator()(path, NAMES, 3, digest)


@pytest.mark.parametrize("payload", [
    b"file_name,file_name,fold\na,a,0\n",
    b"file_name\na.jpeg\n",
    CSV + f"{NAMES[0]},0,4,2\n".encode(),
    CSV.replace(f"{NAMES[2]},2,4,2\n".encode(), b""),
    CSV + b"extra.jpeg,0,4,2\n",
    CSV.replace(b",0,4,2", b",0,4,2,unexpected", 1),
    CSV.replace(b",0,4,2", b",0,4", 1),
    CSV.replace(b",2,4,2", b",1,4,2", 1),
    b"file_name,fold\n",
    CSV.replace(NAMES[0].encode(), b"../escape.jpeg", 1),
])
def test_malformed_or_incomplete_split_is_rejected(tmp_path, payload):
    with pytest.raises(ValueError):
        load(tmp_path, payload)


@pytest.mark.parametrize("fold", ["-1", "3", "0.5", "1.0", "NaN", "inf", "True", "", " 0", "０"])
def test_fold_values_are_exact_integers_in_range(tmp_path, fold):
    with pytest.raises(ValueError):
        load(tmp_path, CSV.replace(b",0,4,2", f",{fold},4,2".encode(), 1))


@pytest.mark.parametrize("names", [[], [NAMES[0], NAMES[0]], ["../escape.jpeg"], ["C:\\escape.jpeg"], [None]])
def test_expected_membership_is_nonempty_unique_safe_basenames(tmp_path, names):
    with pytest.raises(ValueError):
        load(tmp_path, names=names)


@pytest.mark.parametrize("folds", [True, 0, 1, 3.0])
def test_number_of_folds_is_valid(tmp_path, folds):
    with pytest.raises(ValueError):
        load(tmp_path, folds=folds)


def cell_environment(tmp_path, *, payload=CSV, digest=None, images=None, reject_preprocessing=False):
    root = tmp_path / "competition"
    annotation_dir = root / "nested" / "annotations"
    annotation_dir.mkdir(parents=True)
    images = images if images is not None else [{"id": i, "file_name": name} for i, name in enumerate(NAMES)]
    (annotation_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(
        json.dumps({"images": images, "annotations": []}), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    if payload is not None:
        (artifacts / "grouped_folds.csv").write_bytes(payload)
    calls = []

    def copy(*args):
        calls.append(args)
        if reject_preprocessing:
            raise AssertionError("preprocessing must not start for an invalid split")

    cfg = SimpleNamespace(DATA_CACHE=root, ARTIFACT_ROOT=artifacts,
                          PREPROC_DIR=tmp_path / "preprocessed", PREPROC_CACHE=tmp_path / "cache",
                          GROUPED_FOLDS_SHA256=digest or hashlib.sha256(payload or CSV).hexdigest(), N_FOLDS=3)
    namespace = {"Path": Path, "json": json, "defaultdict": defaultdict, "CFG": cfg,
                 "shutil": SimpleNamespace(copy2=copy), "tqdm": lambda values, **kwargs: values}
    return namespace, calls


@pytest.mark.parametrize("invalid", ["missing", "wrong_hash", "missing_member"])
def test_invalid_split_precedes_preprocessing(tmp_path, invalid):
    payload = None if invalid == "missing" else CSV
    if invalid == "missing_member":
        payload = CSV.replace(f"{NAMES[2]},2,4,2\n".encode(), b"")
    namespace, calls = cell_environment(tmp_path, payload=payload,
        digest="0" * 64 if invalid == "wrong_hash" else None, reject_preprocessing=True)
    with pytest.raises((FileNotFoundError, ValueError)):
        exec(compile(data_cell(), str(NOTEBOOK), "exec"), namespace)
    assert calls == []


def test_labeler_variants_share_one_observation_fold(tmp_path):
    images = [{"id": i, "file_name": name} for i, name in enumerate(NAMES)]
    images.append({"id": 10, "file_name": NAMES[0]})
    namespace, calls = cell_environment(tmp_path, images=images)
    exec(compile(data_cell(), str(NOTEBOOK), "exec"), namespace)
    assert len(namespace["images"]) == 4
    assert namespace["file_to_ids"][NAMES[0]] == [0, 10]
    assert namespace["fold_of_file"] == dict(zip(NAMES, range(3)))
    assert len(calls) == 3


def test_string_coco_ids_and_labeler_variants_remain_valid(tmp_path):
    images = [{"id": "040301-" + name, "file_name": name} for name in NAMES]
    images.append({"id": "040302-" + NAMES[0], "file_name": NAMES[0]})
    namespace, calls = cell_environment(tmp_path, images=images)
    exec(compile(data_cell(), str(NOTEBOOK), "exec"), namespace)
    assert namespace["file_to_ids"][NAMES[0]] == ["040301-" + NAMES[0], "040302-" + NAMES[0]]
    assert namespace["fold_of_file"] == dict(zip(NAMES, range(3)))
    assert len(calls) == 3


@pytest.mark.parametrize("image_id", [1, "040301-observation"])
def test_duplicate_coco_ids_fail_before_dict_overwrite(tmp_path, image_id):
    images = [{"id": image_id, "file_name": name} for name in NAMES]
    namespace, calls = cell_environment(tmp_path, images=images, reject_preprocessing=True)
    with pytest.raises(ValueError):
        exec(compile(data_cell(), str(NOTEBOOK), "exec"), namespace)
    assert calls == []


@pytest.mark.parametrize("image_id", [None, True, [], {}, "", " "])
def test_invalid_coco_id_fails_before_preprocessing(tmp_path, image_id):
    images = [{"id": image_id, "file_name": NAMES[0]}]
    namespace, calls = cell_environment(tmp_path, images=images, reject_preprocessing=True)
    with pytest.raises(ValueError):
        exec(compile(data_cell(), str(NOTEBOOK), "exec"), namespace)
    assert calls == []


def test_csv_is_read_once_and_same_bytes_are_validated(tmp_path, monkeypatch):
    path = tmp_path / "grouped_folds.csv"
    path.write_bytes(CSV)
    original = Path.read_bytes
    reads = []

    def read_once(candidate):
        if candidate == path:
            reads.append(candidate)
            assert len(reads) == 1
        return original(candidate)

    load_function = validator()
    monkeypatch.setattr(Path, "read_bytes", read_once)
    result, digest = load_function(path, NAMES, 3, hashlib.sha256(CSV).hexdigest())
    assert reads == [path]
    assert result == dict(zip(NAMES, range(3)))
    assert digest == hashlib.sha256(CSV).hexdigest()


def test_generated_notebook_matches_builder_without_writing_files(monkeypatch):
    captured = []

    def capture(path, text, **kwargs):
        assert path == NOTEBOOK
        captured.append(json.loads(text))
        return len(text)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "write_text", capture)
        runpy.run_path(str(BUILDER), run_name="__main__")
    assert captured == [json.loads(NOTEBOOK.read_text(encoding="utf-8"))]
    for cell in captured[0]["cells"]:
        if cell["cell_type"] == "code":
            raw = cell["source"]
            source = "".join(raw) if isinstance(raw, list) else raw
            ast.parse("\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%")))
