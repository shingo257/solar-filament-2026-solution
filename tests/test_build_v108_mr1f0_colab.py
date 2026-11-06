"""Contract tests for the v108 mask_ratio1 fold-0 replication notebook (§2.86).

v108 = gate-extension replication disclosed in §2.86: fold-2's mask_ratio1
hit +0.0065 (largest ever, below the +0.010 gate), so a paired fold-0
replication is run with the decision criterion fixed before seeing any
fold-0 data. The tests pin: eval fold is 0, train folds are (1,2), the
baseline is the existing prod-recipe yolo_fold_0 (same seed), only
mask_ratio changes, the +0.003 criterion is declared, and frozen cells
stay byte-identical.
"""

import ast
import json

import pytest

from scratch import build_v108_mr1f0_colab as v108


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v108.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v108.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_fold0_replication(nb):
    config = _src(nb["cells"][1])
    assert "FOLD = 0" in config
    assert "TRAIN_FOLDS = (1, 2)" in config
    assert "'mask_ratio1_f0'" in config
    assert "'mask_ratio': 1" in config
    assert "yolo_fold_0" in config  # baseline は既存 prod-recipe fold-0 重み
    assert "BASELINE_PQ = 0.4443" in config  # fold-0 prod OOF 既知値
    assert 'runs" / "v108_mr1_f0"' in config
    assert "v108_mr1_f0_colab_manifest.json" in config
    assert "GATE_DELTA = 0.003" in config  # 複製検証幅 (fold0 取得前固定)
    for kept in ('YOLO_MODEL = "yolo11m-seg.pt"', "YOLO_IMGSZ = 1280",
                 "YOLO_BATCH = 4", "YOLO_EPOCHS = 70", "YOLO_PATIENCE = 14",
                 "PRIMARY_CONF = 0.26"):
        assert kept in config, kept


def test_run_cell_applies_mask_ratio_and_criterion(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v108_mr1_fold0"' in run
    assert "§2.86" in run
    assert "verified_cross_fold" in run
    assert "unverified_fold0" in run
    assert "CFG.YOLO_MASK_RATIO = p[\"mask_ratio\"]" in run
    assert '"train_params_of_arm"' in run
    assert "delta = by_conf[str(CFG.PRIMARY_CONF)][\"pq\"] - prod_pq" in run
    assert "既知値から 0.002 以上" in run


def test_train_cell_has_oom_fallback(nb):
    train = _src(nb["cells"][3])
    assert "YOLO(CFG.YOLO_MODEL).train(" in train
    assert 'stats["batch_fallback"] = True' in train
    assert '"out of memory" not in str(exc).lower()' in train
    for arg in ("deterministic=True", "single_cls=True", "mask_ratio=CFG.YOLO_MASK_RATIO",
                'cache="disk"', "cls=CFG.YOLO_CLS", "seed=CFG.SEED + CFG.FOLD"):
        assert arg in train, arg


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_fixed_criterion(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.86" in hdr
    assert "+0.003" in hdr
    assert "fold-0" in hdr
    assert "0.4443" in hdr
    assert "mask_ratio1_f0" in hdr
