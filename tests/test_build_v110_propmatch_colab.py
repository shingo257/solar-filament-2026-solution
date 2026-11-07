"""Contract tests for the v110 proposal-matched refiner notebook (§2.88).

v110 retrains the EdgeAtt refiner on detector-proposal crops (positives =
matched GT mask at IoU>=0.5, negatives = empty target) to close the
GT-crop ↔ proposal-crop distribution gap — the structural reason the
GT-trained refiner's mean probability could not discriminate existence
in v106. Tests pin the proposal generation (conf floor, pooled GT
matching), the dataset semantics, the refiner-swap arm logic, the
rescue re-test, and that frozen cells stay byte-identical.
"""

import ast
import json

import pytest

from scratch import build_v110_propmatch_colab as v110


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v110.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v110.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_v110_arms(nb):
    config = _src(nb["cells"][1])
    assert "'prop_matched'" in config and "'prop_rescue050'" in config
    for key in ("PROP_IOU_T = 0.5", "REFINER_EPOCHS = 14", "REFINER_SEED = 3407",
                "RESCUE_TAU = None", "'rescue_tau': 0.50",
                "TRAIN_FOLDS = (0, 1)"):
        assert key in config, key
    assert "v110_propmatch_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config


def test_cell3_proposal_matched_training(nb):
    cell3 = _src(nb["cells"][3])
    # 提案生成: folds(0,1)・prod 重み・conf=refine floor・pooled GT 照合
    assert "fold_of_file[n] != CFG.FOLD" in cell3
    assert "conf=CFG.PREDICT_CONF_FLOOR" in cell3
    assert "for gid in file_to_ids[name]" in cell3
    assert "ious[i, j] >= CFG.PROP_IOU_T" in cell3
    # dataset: 提案 mask → square box crop・正例=matched GT・負例=空マスク
    assert "prop = mask_utils.decode" in cell3
    assert "_prop_square_box(prop)" in cell3
    assert 'if s["gt"] is None' in cell3
    assert "np.zeros((CFG.CROP_SIZE, CFG.CROP_SIZE)" in cell3
    # refiner 学習: prod プロトコル同一 (edgeatt・AdamW 2e-4・cosine・ep14・AMP)
    assert "MediSegEdgeAttUNet(3, 1, False).cuda()" in cell3
    assert "AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)" in cell3
    assert "CosineAnnealingLR" in cell3
    assert "0.35 * F.binary_cross_entropy_with_logits" in cell3
    assert "_boundary_map" in cell3
    assert "PROPMATCH_CKPT.exists()" in cell3


def test_run_cell_refiner_swap_and_rescue(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v110_propmatch_refiner"' in run
    assert "§2.88" in run
    assert 'p["refiner"] == "propmatch"' in run
    assert "train_propmatched()" in run
    assert "refiner.load_state_dict" in run
    assert "refiner_sha256" in run
    # rescue wrap: conf∈[0.08,0.26) で refiner 平均確率>=τ → 0.2601
    assert "CFG.RESCUE_TAU = p[\"rescue_tau\"]" in run
    assert "_refiner_mean" in run
    assert "CFG.PREDICT_CONF_FLOOR <= conf[i] < CFG.PRIMARY_CONF" in run
    assert "conf[i] = 0.2601" in run
    assert "reject_v110_fold2" in run
    assert "promote_fold0_replication" in run


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"
    # cell3 は全面差替え (v110 は YOLO を学習しない)
    assert _src(nb["cells"][3]) != _src(nb_source["cells"][3])


def test_header_declares_fixed_gate(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.88" in hdr
    assert "prop_matched" in hdr
    assert "prop_rescue050" in hdr
    assert "+0.010" in hdr
