"""Contract tests for the v96 data-scaling notebooks (RESEARCH_LOG §2.67).

The builder must only change the training fold set, pin ultralytics to the
prod training version, restore cls=0.5 and redirect outputs; the frozen v88
evaluation cells (refiner, ownership, official PQ) must stay byte-identical.
"""

import json

import pytest

from scratch import build_v96_data_scaling_kaggle as builder


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def source_cells():
    return json.loads(builder.SOURCE.read_text(encoding="utf-8"))["cells"]


@pytest.mark.parametrize("arm", list(builder.ARMS))
def test_evaluation_cells_are_unchanged(arm, source_cells):
    cells = builder.build(arm)["cells"]
    assert len(cells) == len(source_cells)
    for index in (4, 5):  # refiner definition + evaluation path
        assert _src(cells[index]) == _src(source_cells[index])


@pytest.mark.parametrize("arm", list(builder.ARMS))
def test_recipe_matches_prod(arm):
    config = _src(builder.build(arm)["cells"][1])
    assert '"ultralytics==8.4.118"' in config
    assert "YOLO_CLS = 0.5" in config and "1.5" not in config.split("YOLO_CLS")[1].split("\n")[0]
    assert f"TRAIN_FOLDS = {builder.ARMS[arm]!r}" in config
    assert "FOLD = 2" in config
    for keep in ("YOLO_EPOCHS = 70", "YOLO_PATIENCE = 14", "YOLO_IMGSZ = 1280",
                 "YOLO_BATCH = 4", 'YOLO_MODEL = "yolo11m-seg.pt"'):
        assert keep in config


def test_dataset_uses_only_train_folds():
    cells = builder.build("third_f0")["cells"]
    assert "fold_of_file[n] in CFG.TRAIN_FOLDS" in _src(cells[2])
    dataset = _src(cells[3])
    assert "if fold_of_file[file_name] in CFG.TRAIN_FOLDS:" in dataset
    assert "        else:\n            for image_id" not in dataset


def test_full_arm_does_not_evaluate_on_training_fold():
    run = _src(builder.build("full")["cells"][6])
    assert "evaluate_yolo(" not in run
    assert "train_arm(arm)" in run


def test_gate_arms_record_last_pt_diagnostic():
    run = _src(builder.build("third_f1")["cells"][6])
    assert '"last_pt"' in run
    assert "promote_to_fold01" not in run


@pytest.mark.parametrize("arm", list(builder.ARMS))
def test_colab_variant_shares_kaggle_cells_and_recipe(arm, source_cells):
    colab = builder.build_colab(arm)["cells"]
    kaggle = builder.build(arm)["cells"]
    for index in (3, 4, 5, 6):
        assert _src(colab[index]) == _src(kaggle[index])
    assert _src(colab[4]) == _src(source_cells[4])
    config = _src(colab[1])
    assert '"ultralytics==8.4.118"' in config
    assert "    FOLD = 2" in config and f"TRAIN_FOLDS = {builder.ARMS[arm]!r}" in config
    assert "YOLO_CLS = 0.5" in config and "v89_yolo_cls15" not in config
    assert f'"v96_{arm}_colab_manifest.json"' in config
    assert "fold_of_file[n] in CFG.TRAIN_FOLDS" in _src(colab[2])


def test_replace_once_rejects_missing_or_duplicate():
    with pytest.raises(ValueError):
        builder.replace_once("abc", "x", "y")
    with pytest.raises(ValueError):
        builder.replace_once("aa", "a", "b")
