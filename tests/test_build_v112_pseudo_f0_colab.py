"""Contract tests for the v112 pseudotest fold-0 replication notebook (§2.90).

v112 is the honest fold-0 replication of v111's test-image pseudolabel
semi-supervised signal (fold-2 Δ=+0.0070, the largest positive in the
campaign but below the +0.010 promotion gate). Mirror design: fold-0
production pipeline labels the 180 test images; folds(1,2) + pseudo
labels retrain; fold-0 honest eval vs prod_f0 baseline 0.444137.
"""

import ast
import json

import pytest

from scratch import build_v112_pseudo_f0_colab as v112


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v112.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v112.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_fold0_mirror(nb):
    config = _src(nb["cells"][1])
    assert "FOLD = 0" in config
    assert "ARMS = ('pseudotest_f0',)" in config
    assert "TRAIN_FOLDS = (1, 2)" in config
    assert "PSEUDO_CONF = 0.26" in config
    assert "PSEUDO_DIR = None" in config
    assert "0.44413694566994816" in config  # v108 fold-0 prod 確定値
    assert "GATE_DELTA = 0.003" in config
    for path in ('"runs" / "v112_pseudo_f0"', '"yolo_datasets" / "v112_pseudo_f0"',
                 '"v112_pseudo_f0_colab_manifest.json"', 'Path("/content/solar_v112")'):
        assert path in config, path
    for kept in ("PRIMARY_CONF = 0.26", "CROP_SIZE = 320", "CROP_CONTEXT = 1.65",
                 "REFINER_THRESHOLD = 0.45", "MIN_AREA = 20",
                 "PREDICT_CONF_FLOOR = 0.08", "YOLO_MASK_RATIO = 2",
                 "YOLO_CLS = 0.5", "YOLO_EPOCHS = 70"):
        assert kept in config, kept


def test_dataset_cell_merges_pseudo_and_oom_fallback(nb):
    ds = _src(nb["cells"][3])
    assert '_pseudotest"' in ds
    assert 'if CFG.PSEUDO_DIR is not None:' in ds
    assert '(CFG.PSEUDO_DIR / "labels").glob("*.txt")' in ds
    assert '"pseudo_instances"' in ds
    assert '"pseudo_images"' in ds
    assert "for _batch in (CFG.YOLO_BATCH, 2):" in ds
    assert "torch.cuda.OutOfMemoryError" in ds
    assert '"batch_fallback"] = True' in ds
    for arg in ("mask_ratio=CFG.YOLO_MASK_RATIO", "cls=CFG.YOLO_CLS",
                'optimizer="AdamW"', "lr0=7e-4", "warmup_epochs=3",
                "mosaic=0.20", "close_mosaic=10"):
        assert arg in ds, arg


def test_pseudolabel_generation_fold0_teacher(nb):
    run = _src(nb["cells"][6])
    assert "conf=CFG.PREDICT_CONF_FLOOR" in run
    assert "iou=CFG.PREDICT_IOU" in run
    assert "retina_masks=True" in run
    assert "conf[i] >= CFG.PSEUDO_CONF and masks[i].sum() >= CFG.MIN_AREA" in run
    assert "refine_proposal(image, masks[i])" in run
    assert "apply_ownership(" in run
    assert 'rglob("test_images")' in run
    assert "assert len(test_files) == 180" in run
    assert "solar_three_channel" in run
    assert "fold0" in run  # fold-0 teacher naming
    assert "cv2.findContours" in run and "RETR_EXTERNAL" in run
    assert "PSEUDOLABELS_COMPLETE" in run
    assert 'CFG.PSEUDO_DIR = CFG.YOLO_ROOT / "pseudotest_teacher_f0"' in run
    assert 'manifest["datasets"]["pseudolabels"]' in run


def test_run_cell_contract_and_verdict(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v112_pseudo_f0"' in run
    assert "§2.90" in run
    assert "cross_fold_verified" in run
    assert "unverified_fold0" in run
    assert '"pseudo_conf": CFG.PSEUDO_CONF' in run


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_fixed_gate_and_baseline(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.90" in hdr
    assert "+0.003" in hdr
    assert "pseudotest_f0" in hdr
    assert "0.444137" in hdr
    assert "cross_fold_verified" in hdr
    assert "規約" in hdr
