"""Contract tests for the v100 capacity notebook (RESEARCH_LOG §2.76).

v100 = preregistered detector capacity gate: yolo11l-seg / yolo11x-seg trained
on the same folds (0,1) and evaluated on fold2 through the identical
prod/v96 path. The tests pin that only the model scale changes — recipe,
eval path, frozen cells and the preregistered gate stay byte-identical —
plus the recorded OOM batch fallback and the per-arm verdict strings.
"""

import ast
import json

import pytest

from scratch import build_v100_capacity_colab as v100


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v100.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v100.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_only_changes_capacity_and_paths(nb):
    config = _src(nb["cells"][1])
    assert "ARMS = ('yolo11l', 'yolo11x')" in config
    assert "TRAIN_FOLDS = (0, 1)" in config  # prod と同一の学習 fold 集合
    assert '"yolo11l": "yolo11l-seg.pt"' in config
    assert '"yolo11x": "yolo11x-seg.pt"' in config
    assert 'runs" / "v100_capacity"' in config
    assert 'yolo_datasets" / "v100_capacity"' in config
    assert "/content/solar_v100" in config
    assert "v100_capacity_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config
    # prod レシピは据え置き
    for kept in ("YOLO_IMGSZ = 1280", "YOLO_BATCH = 4", "YOLO_EPOCHS = 70",
                 "YOLO_PATIENCE = 14", "PRIMARY_CONF = 0.26",
                 "GROUPED_FOLDS_SHA256"):
        assert kept in config, kept


def test_train_cell_uses_per_arm_model_and_records_oom_fallback(nb):
    train = _src(nb["cells"][3])
    assert "YOLO(CFG.YOLO_MODELS[arm]).train(" in train
    assert "CFG.YOLO_MODEL)." not in train  # 単一モデル参照の残存禁止
    assert 'stats["batch_fallback"] = True' in train
    assert '"out of memory" not in str(exc).lower()' in train
    # レシピ差分なし (fallback 以外)
    for arg in ("deterministic=True", "single_cls=True", 'cache="disk"',
                "cls=CFG.YOLO_CLS", "seed=CFG.SEED + CFG.FOLD"):
        assert arg in train, arg


def test_run_cell_has_preregistered_gate_and_verdict(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v100_detector_capacity"' in run
    assert "§2.76" in run
    assert "promote_fold0_replication" in run
    assert "reject_capacity_fold2" in run
    # prod sanity 警告と paired Δ 評価は維持
    assert "既知値から 0.002 以上" in run
    assert "delta = by_conf[str(CFG.PRIMARY_CONF)][\"pq\"] - prod_pq" in run


def test_frozen_cells_unchanged(nb, nb_source):
    # 依存解決以降の評価機構は v96 と完全同一 (cell2 COCO/fold, cell4 refiner, cell5 eval)
    for idx in (2, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_preregistered_gate(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.76" in hdr
    assert "+0.010" in hdr
    assert "fold0 honest 複製" in hdr
    assert "fold2 単独では提出を変更しない" in hdr
