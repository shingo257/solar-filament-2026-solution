"""Contract tests for the v109 residual inference arms notebook (§2.87).

v109 closes out the remaining untried inference-only mechanisms: refiner
dihedral TTA, same-weights multi-scale predict union, the never-swept
CROP_SIZE, single-context values, and a fine conf grid (diagnostic).
The tests pin the arm set, CFG-driven overrides (prod defaults = original
path), the ms_union SimpleNamespace merge, the TTA flip logic, and that
frozen cells stay byte-identical.
"""

import ast
import json

import pytest

from scratch import build_v109_tta_colab as v109


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v109.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v109.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_v109_arms(nb):
    config = _src(nb["cells"][1])
    for arm in ("'refine_tta4'", "'ms_union960'", "'crop384'", "'crop480'",
                "'ctx145'", "'ctx185'", "'conf_fine'"):
        assert arm in config, arm
    for key in ("MS_UNION_IMGSZ = None", "REFINE_TTA = 0", "CTX_VALUE = None",
                "'ms_union_imgsz': 960", "'refine_tta': 4",
                "'crop_size': 384", "'crop_size': 480",
                "'ctx_value': 1.45", "'ctx_value': 1.85",
                "0.22, 0.24", "0.28, 0.30"):
        assert key in config, key
    assert "v109_tta_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config
    for kept in ("PRIMARY_CONF = 0.26", "CROP_SIZE = 320", "CROP_CONTEXT = 1.65",
                 "REFINER_THRESHOLD = 0.45"):
        assert kept in config, kept


def test_run_cell_applies_params_and_verdict(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v109_tta_msunion_crop"' in run
    assert "§2.87" in run
    assert '"params_of_arm"' in run
    assert "diagnostic_conf_grid" in run
    assert "reject_v109_fold2" in run
    assert "promote_fold0_replication" in run
    # per-arm 既定は prod リテラル (前 arm の CFG 変更を持ち越さない回帰防止)
    assert '"crop_size": 320, "conf_sweep": (0.15, 0.2, 0.26, 0.32)' in run
    for line in ("CFG.MS_UNION_IMGSZ = p[\"ms_union_imgsz\"]",
                 "CFG.REFINE_TTA = p[\"refine_tta\"]",
                 "CFG.CTX_VALUE = p[\"ctx_value\"]",
                 "CFG.CROP_SIZE = p[\"crop_size\"]",
                 "CFG.CONF_SWEEP = tuple(p[\"conf_sweep\"])"):
        assert line in run, line


def test_wrappers_preserve_prod_path_and_implement_arms(nb):
    run = _src(nb["cells"][6])
    # TTA: 4 dihedral views averaged; prod (REFINE_TTA=0) → single view = 元関数経路
    assert 'flips = [None] if not CFG.REFINE_TTA else [None, "h", "v", "hv"]' in run
    assert "torch.flip" in run
    assert "torch.stack(probs).mean(0)" in run
    assert "mask_square_box(proposal, context=context)" in run
    # prod 既定では元関数呼び出し
    assert "if CFG.CTX_VALUE is None and not CFG.REFINE_TTA:" in run
    assert "return _orig_refine_proposal(image, proposal)" in run
    # ms-union: YOLO ラップ + 2スケール併合 (最小互換 namespace)
    assert "_orig_YOLO = YOLO" in run
    assert "if CFG.MS_UNION_IMGSZ is None:" in run
    assert 'kw2["imgsz"] = CFG.MS_UNION_IMGSZ' in run
    assert "SimpleNamespace" in run
    assert "torch.cat" in run


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 3, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_fixed_gate(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.87" in hdr
    assert "+0.010" in hdr
    assert "refine_tta4" in hdr
    assert "ms_union960" in hdr
    assert "diagnostic_conf_grid" in hdr
