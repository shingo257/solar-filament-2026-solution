"""Contract tests for the v107 countfill + alt demote + ctx-ensemble notebook (§2.85).

v107 = inference-only follow-up to v106's failed score-based rescue:
rank/count correction (per-record GT ≈7.1 vs preds 6.38 under-count),
alternative demote score families (refined/proposal ratio, bbox aspect
as a falsifiable shape prior), and multi-context refine union/intersect.
The tests pin that conf rescoring stays inside a result_masks wrap with
the preregistered formulas, ctx ensemble wraps refine_proposal with an
explicit context (prod path identical at defaults), no training occurs,
and the +0.010 gate is declared.
"""

import ast
import json

import pytest

from scratch import build_v107_countfill_colab as v107


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v107.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v107.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_all_arms_and_params(nb):
    config = _src(nb["cells"][1])
    for arm in ("countfill_k7", "countfill_k9", "demote_ratio030",
                "demote_aspect15", "ctx_union", "ctx_intersect"):
        assert f"'{arm}'" in config, arm
    assert "COUNTFILL_K = None" in config  # prod 既定は rank 補正なし
    assert "DEMOTE_MODE = None" in config
    assert "CTX_ENSEMBLE = None" in config
    assert "'countfill_k': 7" in config and "'countfill_k': 9" in config
    assert "'demote_mode': 'ratio'" in config and "'demote_tau': 0.30" in config
    assert "'demote_mode': 'aspect'" in config and "'demote_tau': 1.5" in config
    assert "'ctx_ensemble': [[1.45, 1.85], 'union']" in config
    assert "'ctx_ensemble': [[1.45, 1.85], 'intersect']" in config
    assert "v107_countfill_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config
    for kept in ("REFINER_THRESHOLD = 0.45", "DROP_THRESHOLD = 0.90",
                 "MIN_AREA = 20", "PRIMARY_CONF = 0.26", "CROP_CONTEXT = 1.65"):
        assert kept in config, kept


def test_run_cell_has_rescore_and_ctx_wrappers(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v107_countfill_ctx"' in run
    assert "§2.85" in run
    assert "promote_fold0_replication" in run
    assert "reject_v107_fold2" in run
    assert "train_arm(" not in run  # 学習は一切呼ばない
    # countfill: primary 未満の上位を K までブースト (固定式)
    assert "_orig_result_masks = result_masks" in run
    assert "n_primary = int((conf >= CFG.PRIMARY_CONF).sum())" in run
    assert "conf[i] = CFG.RESCUE_BOOST" in run
    # demote: ratio (refined/proposal) と aspect (bbox 細長さ)
    assert 'CFG.DEMOTE_MODE == "ratio"' in run
    assert "refine_proposal(result.orig_img, m).sum()) / prop" in run
    assert "cv2.boundingRect" in run
    assert "conf[i] = 0.0" in run
    # ctx ensemble: refine_proposal ラップ + context 明示の同等関数
    assert "_orig_refine_proposal = refine_proposal" in run
    assert "mask_square_box(proposal, context=context)" in run
    assert "np.logical_or.reduce(masks)" in run
    assert "np.logical_and.reduce(masks)" in run
    assert "delta = by_conf[str(CFG.PRIMARY_CONF)][\"pq\"] - prod_pq" in run


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_preregistered_gate(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.85" in hdr
    assert "+0.010" in hdr
    assert "fold0 honest 複製" in hdr
    assert "fold2 単独では提出を変更しない" in hdr
    assert "countfill_k7" in hdr
    assert "ctx_union" in hdr


def test_params_of_arm_json_roundtrip_preserves_contract_equality(nb):
    """Ensure CONTRACT['params_of_arm'] survives JSON round-trip for manifest resume."""
    stripped = "\n".join(
        line for line in _src(nb["cells"][1]).splitlines() if not line.lstrip().startswith("%")
    )
    tree = ast.parse(stripped)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CFG")
    assign = next(
        n for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PARAMS_OF_ARM" for t in n.targets)
    )
    params_of_arm = ast.literal_eval(assign.value)
    contract_params = {k: dict(v) for k, v in params_of_arm.items()}
    assert json.loads(json.dumps(contract_params)) == contract_params
