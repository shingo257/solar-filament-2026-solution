"""Exercise the actual generated v82 data cell before any model or GPU work."""

import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scratch" / "build_v82_soft_target_refiner.py"
NAMES = [f"20260102000{i}00Bh.jpeg" for i in range(3)]
CSV = ("file_name,fold\n" + "\n".join(f"{n},{i}" for i, n in enumerate(NAMES)) + "\n").encode()


def data_cell():
    spec = importlib.util.spec_from_file_location("v82_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.build_cells(0, "soft", 3407)[2]["source"]
    return source.split("\ntry:\n    from pycocotools", 1)[0]


def environment(tmp_path, payload=CSV, digest=None):
    annotations = tmp_path / "competition" / "nested" / "annotations"
    annotations.mkdir(parents=True)
    images = [{"id": i, "file_name": n} for i, n in enumerate(NAMES)]
    images.append({"id": 10, "file_name": NAMES[0]})
    (annotations / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(
        json.dumps({"images": images, "annotations": []}), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "grouped_folds.csv").write_bytes(payload)
    copies = []
    cfg = SimpleNamespace(DATA_CACHE=tmp_path / "competition", ARTIFACT_ROOT=artifacts,
        N_FOLDS=3, GROUPED_FOLDS_SHA256=digest if digest is not None else hashlib.sha256(payload).hexdigest(),
        PREPROC_DIR=tmp_path / "preprocessed", PREPROC_CACHE=tmp_path / "cache")
    namespace = {"Path": Path, "json": json, "pd": pd, "defaultdict": defaultdict, "CFG": cfg,
        "shutil": SimpleNamespace(copy2=lambda *args: copies.append(args)),
        "tqdm": lambda values, **kwargs: values}
    return namespace, copies


@pytest.mark.parametrize("invalid", ["wrong_hash", "unset_hash", "duplicate", "fractional", "missing", "range"])
def test_invalid_split_stops_before_preprocessing(tmp_path, invalid):
    payload, digest = CSV, None
    if invalid == "wrong_hash":
        digest = "0" * 64
    elif invalid == "unset_hash":
        digest = ""
    elif invalid == "duplicate":
        payload += f"{NAMES[0]},2\n".encode()
    elif invalid == "fractional":
        payload = CSV.replace(b",0\n", b",0.5\n")
    elif invalid == "missing":
        payload = CSV.replace(f"{NAMES[0]},0\n".encode(), b"")
    elif invalid == "range":
        payload = CSV.replace(b",0\n", b",3\n")
    namespace, copies = environment(tmp_path, payload, digest)
    with pytest.raises(ValueError):
        exec(compile(data_cell(), str(BUILDER), "exec"), namespace)
    assert copies == []


def test_frozen_split_preserves_annotator_groups(tmp_path):
    namespace, copies = environment(tmp_path)
    exec(compile(data_cell(), str(BUILDER), "exec"), namespace)
    assert namespace["fold_of_file"] == dict(zip(NAMES, range(3)))
    assert namespace["file_to_ids"][NAMES[0]] == [0, 10]
    assert namespace["frozen_fold_sha256"] == hashlib.sha256(CSV).hexdigest()
    assert len(copies) == 3


@pytest.mark.parametrize("invalid", ["missing_hash", "wrong_hash", "fold", "arm", "seed", "config"])
def test_checkpoint_contract_stops_before_model_loading(invalid):
    spec = importlib.util.spec_from_file_location("v82_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = next(c["source"] for c in module.build_cells(0, "soft", 3407)
        if c["cell_type"] == "code" and "expected_contract =" in c["source"])
    guard = source.split("\nfrom ultralytics import YOLO", 1)[0]
    guard = "\n".join(line for line in guard.splitlines() if not line.lstrip().startswith("!"))
    config = {"cluster_iou": 0.3}
    checkpoint = {"fold": 0, "arm": "soft", "seed": 3407,
        "folds_sha256": "a" * 64, "training_config": dict(config)}
    if invalid == "missing_hash":
        del checkpoint["folds_sha256"]
    elif invalid == "wrong_hash":
        checkpoint["folds_sha256"] = "b" * 64
    elif invalid == "config":
        checkpoint["training_config"] = {"cluster_iou": 0.5}
    else:
        checkpoint[invalid] = "different"
    namespace = {"CFG": SimpleNamespace(FOLD=0, ARM="soft", SEED=3407),
        "frozen_fold_sha256": "a" * 64, "training_config": config,
        "torch": SimpleNamespace(load=lambda *args, **kwargs: checkpoint),
        "checkpoint_path": "unused.pt"}
    with pytest.raises(ValueError, match="training contract"):
        exec(compile(guard, str(BUILDER), "exec"), namespace)
