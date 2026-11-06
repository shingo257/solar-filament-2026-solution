"""Contract tests for the v105 low-resolution arm expansion notebook (§2.83).

v105 = adaptive sequential follow-up to v103's imgsz960 positive direction
(+0.0023, sub-gate): extend the resolution dose-response below 960 with
train/infer imgsz{640,768} plus a prod_infer640 inference-only diagnostic
(the symmetric check — 960 inference-only was negative while 960 training
was positive). The tests pin that each arm changes only imgsz, the frozen
eval cells stay byte-identical to v96, and the preregistered +0.010 gate,
skip-train diagnostic, and OOM batch fallback are declared.
"""

import ast
import json

import pytest

from scratch import build_v105_imgszlow_colab as v105


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v105.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v105.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_all_arms_and_imgsz_only_changes(nb):
    config = _src(nb["cells"][1])
    for arm in ("prod_infer640", "imgsz640", "imgsz768"):
        assert f"'{arm}'" in config, arm
    assert "SKIP_TRAIN_ARMS = ('prod_infer640',)" in config
    assert "'prod_infer640': {'imgsz': 640}" in config
    assert "'imgsz640': {'imgsz': 640}" in config
    assert "'imgsz768': {'imgsz': 768}" in config
    assert "PROD_TRAIN_PARAMS" in config
    assert "YOLO_COPY_PASTE = 0.0" in config  # prod 既定は不変
    assert 'runs" / "v105_imgszlow"' in config
    assert "v105_imgszlow_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config
    for kept in ('YOLO_MODEL = "yolo11m-seg.pt"', "YOLO_IMGSZ = 1280",
                 "YOLO_BATCH = 4", "YOLO_EPOCHS = 70", "YOLO_PATIENCE = 14",
                 "YOLO_MASK_RATIO = 2", "PRIMARY_CONF = 0.26"):
        assert kept in config, kept


def test_train_cell_has_copy_paste_and_oom_fallback(nb):
    train = _src(nb["cells"][3])
    assert "YOLO(CFG.YOLO_MODEL).train(" in train
    assert "copy_paste=CFG.YOLO_COPY_PASTE" in train
    assert 'stats["batch_fallback"] = True' in train
    assert '"out of memory" not in str(exc).lower()' in train
    for arg in ("deterministic=True", "single_cls=True", 'cache="disk"',
                "cls=CFG.YOLO_CLS", "seed=CFG.SEED + CFG.FOLD"):
        assert arg in train, arg


def test_run_cell_applies_train_params_and_gate(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v105_lowres_residuals"' in run
    assert "§2.83" in run
    assert "promote_fold0_replication" in run
    assert "reject_recipe_fold2" in run
    assert "diagnostic_no_train" in run
    assert "CFG.YOLO_EPOCHS = p[\"epochs\"]" in run
    assert "CFG.YOLO_MASK_RATIO = p[\"mask_ratio\"]" in run
    assert "CFG.YOLO_COPY_PASTE = p[\"copy_paste\"]" in run
    assert '"train_params_of_arm"' in run
    assert "arm in CFG.SKIP_TRAIN_ARMS" in run
    assert "weights, stats = CFG.PROD_YOLO" in run
    assert "既知値から 0.002 以上" in run
    assert "delta = by_conf[str(CFG.PRIMARY_CONF)][\"pq\"] - prod_pq" in run


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_preregistered_gate(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.83" in hdr
    assert "+0.010" in hdr
    assert "fold0 honest 複製" in hdr
    assert "fold2 単独では提出を変更しない" in hdr
    assert "imgsz640" in hdr
    assert "imgsz768" in hdr
