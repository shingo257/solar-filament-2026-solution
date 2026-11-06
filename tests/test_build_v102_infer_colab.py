"""Contract tests for the v102 inference-parameter sweep notebook (§2.80).

v102 = preregistered sweep of the inference-time CFG params that were never
enumerated in the declared swept list (conf/閾値/drop/min_area/重み選択):
PREDICT_IOU, CROP_CONTEXT, CROP_SIZE and pre-refine proposal dilation.
All arms are inference-only (prod fold2 YOLO + all-data EdgeAtt refiner);
the frozen eval cells stay byte-identical to v96 and the per-arm changes are
applied through call-time CFG reads plus a documented wrapper for the
default-arg-bound mask_square_box and the proposal-dilation hook.
"""

import ast
import json

import pytest

from scratch import build_v102_infer_colab as v102


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v102.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v102.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_all_seven_infer_only_arms(nb):
    config = _src(nb["cells"][1])
    for arm in ("nms_iou_035", "nms_iou_075", "crop_ctx_130", "crop_ctx_200",
                "crop_size_448", "prop_dilate_2", "prop_dilate_4"):
        assert f"'{arm}'" in config, arm
    assert "'predict_iou': 0.35" in config and "'predict_iou': 0.75" in config
    assert "'crop_context': 1.30" in config and "'crop_context': 2.00" in config
    assert "'crop_size': 448" in config
    assert "'proposal_dilate_px': 2" in config and "'proposal_dilate_px': 4" in config
    assert "PROD_INFER_PARAMS" in config  # prod 既定値へ毎 arm リセット
    assert "PROPOSAL_DILATE_PX = 0" in config
    assert 'runs" / "v102_infer"' in config
    assert "v102_infer_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config
    # prod 既定値は据え置き
    for kept in ("PREDICT_IOU = 0.55", "CROP_SIZE = 320", "CROP_CONTEXT = 1.65",
                 "PRIMARY_CONF = 0.26", "GROUPED_FOLDS_SHA256"):
        assert kept in config, kept


def test_run_cell_applies_params_without_training(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v102_infer_params"' in run
    assert "§2.80" in run
    assert "promote_fold0_replication" in run
    assert "reject_param_fold2" in run
    assert "train_arm(" not in run  # 学習は一切呼ばない
    assert "weights = CFG.PROD_YOLO" in run
    assert "CFG.PREDICT_IOU" in run and "CFG.CROP_CONTEXT" in run
    assert "CFG.CROP_SIZE" in run and "CFG.PROPOSAL_DILATE_PX" in run
    assert '"params_of_arm"' in run
    # prod sanity 警告と paired Δ 評価は維持
    assert "既知値から 0.002 以上" in run
    assert "delta = by_conf[str(CFG.PRIMARY_CONF)][\"pq\"] - prod_pq" in run


def test_wrapper_preserves_prod_path_and_dilate_hook(nb):
    run = _src(nb["cells"][6])
    # default-arg binding を避ける call 時 CFG 読み wrapper
    assert "_orig_mask_square_box = mask_square_box" in run
    assert "CFG.CROP_CONTEXT if context is None else context" in run
    # proposal 膨張は prod では不活性 (PROPOSAL_DILATE_PX=0)
    assert "if CFG.PROPOSAL_DILATE_PX > 0" in run
    assert "cv2.dilate(proposal.astype(np.uint8), k) != 0" in run
    assert "cv2.MORPH_ELLIPSE" in run


def test_frozen_cells_unchanged(nb, nb_source):
    # cell2 COCO/fold, cell3 dataset/train 定義 (未使用), cell4 refiner, cell5 eval
    for idx in (2, 3, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_preregistered_gate(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.80" in hdr
    assert "+0.010" in hdr
    assert "fold0 honest 複製" in hdr
    assert "fold2 単独では提出を変更しない" in hdr
    assert "prop_dilate" in hdr
    assert "学習なし" in hdr
