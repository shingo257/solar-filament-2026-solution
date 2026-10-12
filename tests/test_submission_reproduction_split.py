"""Reproduction must validate frozen observation folds before preprocessing."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd


MODULE = Path(__file__).resolve().parents[1]
VALID_CSV = "file_name,fold\nimage_0.jpeg,2\nimage_1.jpeg,0\nimage_2.jpeg,1\n"


def _run_preprocessing(tmp_path, csv_content, expected_hash):
    notebook = json.loads(
        (MODULE / "solar_filament_submission_reproduction.ipynb").read_text(encoding="utf-8")
    )
    source = "".join(notebook["cells"][3]["source"])
    files = ["image_0.jpeg", "image_1.jpeg", "image_2.jpeg"]
    annotation = tmp_path / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    annotation.write_text(json.dumps({"images": [
        {"id": index, "file_name": name} for index, name in enumerate(files)
    ], "annotations": []}), encoding="utf-8")
    (tmp_path / "train_images").mkdir()
    for name in files:
        (tmp_path / name).write_text("cached", encoding="utf-8")
    path = tmp_path / "grouped_folds.csv"
    if csv_content is not None:
        path.write_bytes(csv_content.encode("utf-8"))
    scope = {
        "Path": Path, "json": json, "defaultdict": defaultdict, "pd": pd,
        "CFG": SimpleNamespace(ARTIFACT_ROOT=tmp_path, PREPROC_DIR=tmp_path,
                               PREPROC_CACHE=None, N_FOLDS=3,
                               GROUPED_FOLDS_SHA256=expected_hash),
        "find_dataset_root": lambda: tmp_path,
        "tqdm": lambda values, **kwargs: values,
    }
    exec(compile(source, "reproduction-preprocessing", "exec"), scope)
    return scope["fold_of_file"], path


@pytest.mark.parametrize("expected_hash", ["", "0" * 64])
def test_unset_or_wrong_hash_stops_before_preprocessing(tmp_path, expected_hash):
    with pytest.raises(ValueError):
        _run_preprocessing(tmp_path, VALID_CSV, expected_hash)


def test_missing_frozen_csv_is_not_regenerated(tmp_path):
    with pytest.raises(FileNotFoundError):
        _run_preprocessing(tmp_path, None, "1" * 64)
    assert not (tmp_path / "grouped_folds.csv").exists()


def test_existing_frozen_split_is_preserved(tmp_path):
    digest = hashlib.sha256(VALID_CSV.encode("utf-8")).hexdigest()
    actual, path = _run_preprocessing(tmp_path, VALID_CSV, digest)
    assert actual == {"image_0.jpeg": 2, "image_1.jpeg": 0, "image_2.jpeg": 1}
    assert path.read_bytes() == VALID_CSV.encode("utf-8")


@pytest.mark.parametrize("invalid_csv", [
    VALID_CSV + "image_0.jpeg,1\n",
    VALID_CSV.replace("image_2.jpeg,1\n", ""),
    VALID_CSV.replace("image_2.jpeg,1", "image_2.jpeg,3"),
])
def test_matching_hash_does_not_override_observation_contract(tmp_path, invalid_csv):
    digest = hashlib.sha256(invalid_csv.encode("utf-8")).hexdigest()
    with pytest.raises(ValueError):
        _run_preprocessing(tmp_path, invalid_csv, digest)
