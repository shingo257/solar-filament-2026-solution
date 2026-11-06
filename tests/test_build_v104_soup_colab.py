"""Contract tests for the v104 weight-soup + residual notebook (§2.82).

v104 = preregistered inference-only test of the last untested mechanisms:
checkpoint tail SWA (soup_tail3/7 from prod's periodic epochs), cross-run
weight soup (soup_res3: prod + v101 imgsz bests — weight-space ensemble,
distinct from the rejected prediction-space ensembles), ownership ordering
(own_desc), and a refiner second pass (refine_twice). Frozen eval cells stay
byte-identical to v96; soups and overrides live in the run cell.
"""

import ast
import json

import pytest

from scratch import build_v104_soup_colab as v104


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v104.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v104.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_all_arms_soup_members_and_params(nb):
    config = _src(nb["cells"][1])
    for arm in ("soup_tail3", "soup_tail7", "soup_res3", "own_desc", "refine_twice"):
        assert f"'{arm}'" in config, arm
    # soup メンバー (周期 ckpt + v101 解像度系統)
    assert "epoch60.pt" in config and "epoch65.pt" in config and "epoch40.pt" in config
    assert "yolo_imgsz1536_fold2" in config and "yolo_imgsz2048_fold2" in config
    assert "OWNERSHIP_ORDER = 'asc'" in config  # prod 既定
    assert "REFINE_TWICE = False" in config     # prod 既定
    assert "'ownership_order': 'desc'" in config
    assert "'refine_twice': True" in config
    assert 'runs" / "v104_soup"' in config
    assert "v104_soup_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config


def test_run_cell_builds_soup_and_overrides_without_training(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v104_weight_soup"' in run
    assert "§2.82" in run
    assert "promote_fold0_replication" in run
    assert "reject_v104_fold2" in run
    assert "train_arm(" not in run  # 学習は一切呼ばない
    # soup 組み立て: ema 優先で評価重みを拾い、soup 後 'ema' は外す
    assert "def build_soup" in run
    assert 'ckpt.get("ema")' in run
    # 周期 ckpt は model=None/ema に重み、best.pt は model に重み → ある方へ書き戻す
    assert 'base.get("model") if base.get("model") is not None else base.get("ema")' in run
    assert "target.load_state_dict(avg)" in run
    assert 'base["model"] = target' in run
    assert 'base.pop(k, None)' in run
    # ownership/refine の override (prod 既定では不活性)
    assert "_orig_apply_ownership = apply_ownership" in run
    assert 'CFG.OWNERSHIP_ORDER == "asc"' in run
    assert "_orig_refine_proposal = refine_proposal" in run
    assert "CFG.REFINE_TWICE" in run
    assert '"soup_of_arm"' in run
    assert "既知値から 0.002 以上" in run


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 3, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_preregistered_gate(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.82" in hdr
    assert "+0.010" in hdr
    assert "fold0 honest 複製" in hdr
    assert "fold2 単独では提出を変更しない" in hdr
    assert "soup_res3" in hdr
    assert "refine_twice" in hdr
