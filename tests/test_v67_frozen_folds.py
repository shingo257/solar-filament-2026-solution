"""CPU-only checks of the actual generated temporal-pilot data cell."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "solar_filament_v67_temporal_aug_fold2_colab.ipynb"
NAMES = [f"20260102000{i}00Bh.jpeg" for i in range(3)]
CSV = ("file_name,fold\n" + "\n".join(f"{n},{i}" for i, n in enumerate(NAMES)) + "\n").encode()


def cell():
    source = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][2]["source"]
    return "".join(source) if isinstance(source, list) else source


def environment(tmp_path, payload=CSV, digest=None):
    data = tmp_path / "competition"
    annotations = data / "nested" / "annotations"
    annotations.mkdir(parents=True)
    images = [{"id": i, "file_name": n} for i, n in enumerate(NAMES)]
    images.append({"id": 10, "file_name": NAMES[0]})
    (annotations / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(
        json.dumps({"images": images, "annotations": []}), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "grouped_folds.csv").write_bytes(payload)
    copies = []
    cfg = SimpleNamespace(DATA_CACHE=data, ARTIFACT_ROOT=artifacts, FOLD=2,
        N_FOLDS=3, GROUPED_FOLDS_SHA256=digest if digest is not None else hashlib.sha256(payload).hexdigest(),
        PREPROC_DIR=tmp_path / "preprocessed", PREPROC_CACHE=tmp_path / "cache")
    namespace = {"Path": Path, "json": json, "pd": pd, "defaultdict": defaultdict, "CFG": cfg,
        "shutil": SimpleNamespace(copy2=lambda *args: copies.append(args)),
        "tqdm": lambda values, **kwargs: values, "write_status": lambda *args: None}
    return namespace, copies


@pytest.mark.parametrize("invalid", ["wrong_hash", "unset_hash", "duplicate", "fractional", "missing"])
def test_invalid_split_fails_before_preprocessing(tmp_path, invalid):
    payload = CSV
    digest = None
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
    namespace, copies = environment(tmp_path, payload, digest)
    with pytest.raises(ValueError):
        exec(compile(cell(), str(NOTEBOOK), "exec"), namespace)
    assert copies == []


def test_frozen_split_preserves_observation_groups(tmp_path):
    namespace, copies = environment(tmp_path)
    exec(compile(cell(), str(NOTEBOOK), "exec"), namespace)
    assert namespace["fold_of_file"] == dict(zip(NAMES, range(3)))
    assert namespace["file_to_ids"][NAMES[0]] == [0, 10]
    assert namespace["frozen_fold_sha256"] == hashlib.sha256(CSV).hexdigest()
    assert len(copies) == 1
