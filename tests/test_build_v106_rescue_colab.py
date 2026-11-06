"""Contract tests for the v106 existence-rescue + postproc notebook (§2.84).

v106 = inference-only arms: refiner-score-based rescue/demote of proposals
(the dominant existence bottleneck — 57% of primary-conf FNs are covered by
low-conf proposals), the never-swept postproc params (refiner threshold,
containment threshold, min area), and a combo of two sub-gate signals.
The tests pin that rescue/demote is implemented via result_masks wrapping
with the preregistered score formula, the default-arg binding of
drop_contained_indices is overridden correctly, all arms stay on the frozen
prod path, and the +0.010 gate is declared.
"""

import ast
import json

import pytest

from scratch import build_v106_rescue_colab as v106


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v106.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v106.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_all_arms_and_params(nb):
    config = _src(nb["cells"][1])
    for arm in ("rescue_mean050", "rescue_mean065", "demote_mean030",
                "refiner_th040", "refiner_th050", "drop080", "drop097",
                "minarea10", "minarea40", "combo_960_nms35"):
        assert f"'{arm}'" in config, arm
    assert "RESCUE_TAU = None" in config  # prod 既定は救済なし
    assert "DEMOTE_TAU = None" in config
    assert "RESCUE_BOOST = 0.2601" in config
    assert "'rescue_tau': 0.50" in config and "'rescue_tau': 0.65" in config
    assert "'demote_tau': 0.30" in config
    assert "'refiner_threshold': 0.40" in config and "'refiner_threshold': 0.50" in config
    assert "'drop_threshold': 0.80" in config and "'drop_threshold': 0.97" in config
    assert "'min_area': 10" in config and "'min_area': 40" in config
    assert "'predict_iou': 0.35" in config
    assert "v103_recipe/yolo_imgsz960_fold2/weights/best.pt" in config
    assert "v106_rescue_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config
    for kept in ("REFINER_THRESHOLD = 0.45", "DROP_THRESHOLD = 0.90",
                 "MIN_AREA = 20", "PREDICT_IOU = 0.55", "PRIMARY_CONF = 0.26"):
        assert kept in config, kept


def test_run_cell_has_rescue_wrapper_and_gate(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v106_rescue_postproc"' in run
    assert "§2.84" in run
    assert "promote_fold0_replication" in run
    assert "reject_v106_fold2" in run
    assert "train_arm(" not in run  # 学習は一切呼ばない
    # rescue/demote: result_masks ラップで refiner 平均確率 → conf 再スコア
    assert "_orig_result_masks = result_masks" in run
    assert "def _refiner_mean_prob" in run
    assert "probability[pcrop].mean()" in run  # 固定式: 提案画素上の平均確率
    assert "conf[i] = CFG.RESCUE_BOOST" in run
    assert "conf[i] = 0.0" in run
    # default-arg 束縛の回避: drop_contained_indices に CFG を渡す override
    assert "drop_contained_indices(masks, CFG.DROP_THRESHOLD)" in run
    assert "_orig_apply_ownership = apply_ownership" in run
    # per-arm CFG 差替え + weights override
    assert "CFG.REFINER_THRESHOLD = p[\"refiner_threshold\"]" in run
    assert "CFG.DROP_THRESHOLD = p[\"drop_threshold\"]" in run
    assert "CFG.MIN_AREA = p[\"min_area\"]" in run
    assert "CFG.PREDICT_IOU = p[\"predict_iou\"]" in run
    assert "CFG.WEIGHTS_OF_ARM[arm]" in run
    assert "delta = by_conf[str(CFG.PRIMARY_CONF)][\"pq\"] - prod_pq" in run


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_preregistered_gate(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.84" in hdr
    assert "+0.010" in hdr
    assert "fold0 honest 複製" in hdr
    assert "fold2 単独では提出を変更しない" in hdr
    assert "rescue_mean050" in hdr
    assert "combo_960_nms35" in hdr
