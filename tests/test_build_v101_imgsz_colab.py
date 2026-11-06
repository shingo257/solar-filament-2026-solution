"""Contract tests for the v101 imgsz notebook (RESEARCH_LOG §2.79).

v101 = preregistered input-resolution gate: yolo11m-seg trained at
imgsz {1536, 2048} on the same folds (0,1) and evaluated on fold2 through
the identical prod/v96/v100 path (train/infer at the arm's imgsz). Plus a
no-train diagnostic arm (`prod_infer1536`) that only re-runs prod weights
at imgsz 1536. The tests pin that only imgsz changes — recipe, eval path,
frozen cells and the preregistered gate stay fixed — plus the recorded
OOM batch fallback, the skip-train diagnostic arm, and the verdict strings.
"""

import ast
import json

import pytest

from scratch import build_v101_imgsz_colab as v101


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v101.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v101.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_only_changes_imgsz_arms_and_paths(nb):
    config = _src(nb["cells"][1])
    assert "ARMS = ('prod_infer1536', 'imgsz1536', 'imgsz2048')" in config
    assert "TRAIN_FOLDS = (0, 1)" in config  # prod と同一の学習 fold 集合
    assert "SKIP_TRAIN_ARMS = ('prod_infer1536',)" in config
    assert "'prod_infer1536': 1536" in config
    assert "'imgsz1536': 1536" in config
    assert "'imgsz2048': 2048" in config
    assert 'runs" / "v101_imgsz"' in config
    assert 'yolo_datasets" / "v101_imgsz"' in config
    assert "/content/solar_v101" in config
    assert "v101_imgsz_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config
    # モデルと prod レシピは据え置き
    for kept in ('YOLO_MODEL = "yolo11m-seg.pt"', "YOLO_IMGSZ = 1280",
                 "YOLO_BATCH = 4", "YOLO_EPOCHS = 70", "YOLO_PATIENCE = 14",
                 "PRIMARY_CONF = 0.26", "GROUPED_FOLDS_SHA256"):
        assert kept in config, kept


def test_train_cell_uses_same_model_arm_imgsz_and_records_oom_fallback(nb):
    train = _src(nb["cells"][3])
    assert "YOLO(CFG.YOLO_MODEL).train(" in train  # model は全 arm 同一
    assert "imgsz=CFG.YOLO_IMGSZ" in train  # arm 値は実行セルで差替
    assert 'stats["batch_fallback"] = True' in train
    assert '"out of memory" not in str(exc).lower()' in train
    # レシピ差分なし (fallback 以外)
    for arg in ("deterministic=True", "single_cls=True", 'cache="disk"',
                "cls=CFG.YOLO_CLS", "seed=CFG.SEED + CFG.FOLD"):
        assert arg in train, arg


def test_run_cell_has_preregistered_gate_verdict_and_imgsz_switch(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v101_imgsz_scaling"' in run
    assert "§2.79" in run
    assert "promote_fold0_replication" in run
    assert "reject_imgsz_fold2" in run
    assert "diagnostic_no_train" in run
    assert "CFG.YOLO_IMGSZ = CFG.IMGSZ_OF_ARM[arm]" in run
    assert "arm in CFG.SKIP_TRAIN_ARMS" in run
    assert '"imgsz_of_arm": dict(CFG.IMGSZ_OF_ARM)' in run
    # skip-train arm では prod 重みを使い last.pt 診断を回さない
    assert "weights, stats = CFG.PROD_YOLO" in run
    assert "arm not in CFG.SKIP_TRAIN_ARMS" in run
    # prod sanity 警告と paired Δ 評価は維持
    assert "既知値から 0.002 以上" in run
    assert "delta = by_conf[str(CFG.PRIMARY_CONF)][\"pq\"] - prod_pq" in run


def test_frozen_cells_unchanged(nb, nb_source):
    # 依存解決以降の評価機構は v96 と完全同一 (cell2 COCO/fold, cell4 refiner, cell5 eval)
    for idx in (2, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_preregistered_gate(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.79" in hdr
    assert "+0.010" in hdr
    assert "fold0 honest 複製" in hdr
    assert "fold2 単独では提出を変更しない" in hdr
    assert "prod_infer1536" in hdr
    assert "imgsz2048" in hdr
