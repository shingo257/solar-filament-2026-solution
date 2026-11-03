"""Contract tests for the v97 family notebooks (RESEARCH_LOG §2.71/§2.72).

v97 = preregistered RT-DETR ensemble gate; v97b/v97c = post-gate exploratory
sweeps. All three share Drive-resident caches and per-file checkpoints, so the
tests pin the resume machinery, the mask-dtype crash fix, atomic cache writes,
and the exploratory manifest separation. They also pin the prod_dedup030
submission-variant wiring in the reproduction builder.
"""

import ast
import json

import pytest

from scratch import (
    build_submission_reproduction as repro,
    build_v97_rtdetr_ensemble_colab as v97,
    build_v97b_explore_colab as v97b,
    build_v97c_explore_colab as v97c,
    build_v98_foldensemble_colab as v98,
)


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _parseable(nb):
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = _src(cell)
        stripped = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


@pytest.fixture(scope="module")
def nb_v97():
    return json.loads(v97.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_v97c():
    return json.loads(v97c.DST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_repro():
    return json.loads(repro.OUTPUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_v98():
    return json.loads(v98.DST.read_text(encoding="utf-8"))


# ---- v97 main notebook: resume machinery + crash fix ----

def test_v97_all_cells_parse(nb_v97):
    _parseable(nb_v97)


def test_v97_caches_live_on_drive(nb_v97):
    config = _src(nb_v97["cells"][1])
    assert 'CAND_CACHE = ARTIFACT_ROOT / "v97_candidates.pt"' in config
    assert 'CAL_CACHE = ARTIFACT_ROOT / "v97_cal_maps.pt"' in config
    assert "CAND_CACHE = LOCAL_ROOT" not in config
    assert "CAL_CACHE = LOCAL_ROOT" not in config


def test_v97_eval_has_per_file_checkpoints(nb_v97):
    eval_src = _src(nb_v97["cells"][6])
    for name in ("_part_path", "_part_load", "_part_save",
                 "ensemble_eval_cached", "solo_eval_cached", "os.replace"):
        assert name in eval_src, name


def test_v97_decode_outputs_bool(nb_v97):
    # regression: mask_utils.decode は uint8 を返し apply_ownership の
    # `occupied |= exclusive` が UFuncTypeError で落ちた。4経路全てで .astype(bool)。
    eval_src = _src(nb_v97["cells"][6])
    assert eval_src.count("mask_utils.decode(") >= 4
    assert eval_src.count(".astype(bool)") >= 4


def test_v97_cache_writes_are_atomic_and_corrupt_tolerant(nb_v97):
    run_src = _src(nb_v97["cells"][7])
    # 非原子的な torch.save(..., CACHE) は残さない
    assert "torch.save(" not in run_src
    assert "_part_save(CFG.CAND_CACHE" in run_src
    assert "_part_save(CFG.CAL_CACHE" in run_src
    assert "unlink(missing_ok=True)" in run_src


def test_v97_preregistered_gate_is_recorded(nb_v97):
    run_src = _src(nb_v97["cells"][7])
    assert "reject_noise" in run_src and "delta" in run_src
    assert "+0.005" in run_src or "0.005" in run_src


# ---- v97c exploratory notebook: manifest separation + sweep coverage ----

def test_v97c_all_cells_parse(nb_v97c):
    _parseable(nb_v97c)


def test_v97c_uses_separate_manifest_and_contract(nb_v97c):
    config = _src(nb_v97c["cells"][1])
    assert "v97c_explore_colab_manifest.json" in config
    run_src = _src(nb_v97c["cells"][7])
    assert '"experiment_id": "v97c_explore"' in run_src


def test_v97c_exploration_covers_dedup_plateau(nb_v97c):
    run_src = _src(nb_v97c["cells"][7])
    for point in ("0.08", "0.12", "0.15", "0.20", "0.30", "0.50", "1.00"):
        assert point in run_src, point
    # λ×dedup 交互作用 (0.9x は v97b λグリッドのピーク位置)
    for m in ("0.8", "0.9", "1.2"):
        assert m in run_src, m
    assert "ablation" in run_src and "rescue_eval" in run_src
    assert "perfile_delta" in run_src


def test_v97c_tags_reuse_main_run_parts(nb_v97c):
    run_src = _src(nb_v97c["cells"][7])
    assert 'f"cross_d{dedup_iou}_l{accept_lam:.4f}"' in run_src


# ---- submission builder: prod_dedup030 wiring ----

def test_repro_setup_has_dedup030_variant():
    setup = repro.SETUP_CELL
    assert '"prod_dedup030"' in setup
    for const in ("DEDUP030_FLOOR", "DEDUP030_IOU", "DEDUP030_LAM",
                  "_DEDUP030_ISO_X", "_DEDUP030_ISO_Y"):
        assert const in setup, const


def test_repro_iso_arrays_match_fitted_map():
    import numpy as np

    npz = np.load(repro.MODULE / "artifacts" / "v97_global_iso_yolo.npz")
    ns = {"np": np}
    setup = repro.SETUP_CELL
    for name, key in (("_DEDUP030_ISO_X", "x"), ("_DEDUP030_ISO_Y", "y")):
        start = setup.index(f"{name} = np.array(") + len(f"{name} = np.array(")
        end = setup.index(")", start)
        arr = eval(setup[start:end], ns)  # 配列リテラルのみ (np.array(...) 内部)
        np.testing.assert_allclose(arr, npz[key], atol=1e-6, err_msg=name)
    assert float(npz["lam"]) == pytest.approx(0.22210895069309958)


def test_repro_cell12_dedup030_path(nb_repro):
    src12 = _src(nb_repro["cells"][12])
    assert "dedup030_accept" in src12
    assert "DEDUP030_FLOOR" in src12 and "DEDUP030_IOU" in src12
    assert "np.interp" in src12
    # emission は prod 経路 (ownership 適用) を維持
    assert "apply_ownership(refined, scores)" in src12


def test_repro_audit_keeps_disjoint_assert_for_dedup030(nb_repro):
    src14 = _src(nb_repro["cells"][14])
    assert '"prod_dedup030"' in src14
    assert 'assert overlap_n == 0' in src14


def test_repro_env_record_refiner_label(nb_repro):
    # regression: plain_ckpt は plain_mhsa_f2 専用。prod_v94/probe_refprop/
    # prod_dedup030 は ALLDATA_CKPT 参照でなければ NameError (Kaggle v1 実害)
    env_src = _src(nb_repro["cells"][16])
    assert 'SUBMISSION_VARIANT == "plain_mhsa_f2"' in env_src
    assert 'else ("refiner_" + CFG.SUBMISSION_VARIANT, plain_ckpt)' not in env_src


# ---- v98 fold-ensemble exploratory notebook ----

def test_v98_all_cells_parse(nb_v98):
    _parseable(nb_v98)


def test_v98_separate_manifest_and_cache(nb_v98):
    src1 = _src(nb_v98["cells"][1])
    assert "v98_foldensemble_colab_manifest.json" in src1
    src7 = _src(nb_v98["cells"][7])
    assert "v98_candidates.pt" in src7
    assert "v97_rtdetr_ensemble_colab_manifest" not in src7
    # resume 機構: 破損 cache 除去 + 原子保存
    assert "V98_CAND_CACHE.unlink(missing_ok=True)" in src7
    assert "_part_save(V98_CAND_CACHE" in src7


def test_v98_pool_coverage_and_honest_contract(nb_v98):
    src7 = _src(nb_v98["cells"][7])
    # fold0/1 重みの sha を contract に固定 (honest 評価の入力凍結)
    assert "yolo_f0_sha256" in src7 and "yolo_f1_sha256" in src7
    # プール組合せ + half-cross isotonic + dedup 2値
    for pool in ("f2f0", "f2f1", "f0f1", "f2f0f1"):
        assert f'"{pool}"' in src7
    assert "crossfit_halves" in src7 or "half_a" in src7
    assert "(0.10, 0.30)" in src7
    # eval parts の tag は v98 prefix で v97 parts と衝突しない
    assert 'tag=f"v98_' in src7


def test_v98_uses_cfg_drive_root(nb_v98):
    # regression: DRIVE_ROOT は CFG クラス属性。裸の DRIVE_ROOT は
    # cell7 スコープでは NameError (Colab 実害)
    src7 = _src(nb_v98["cells"][7])
    assert "CFG.DRIVE_ROOT" in src7
    import re as _re
    assert not _re.search(r"(?<!CFG\.)DRIVE_ROOT", src7)


@pytest.fixture(scope="module")
def nb_v98b():
    from scratch import build_v98b_foldonly_colab as v98b
    return json.loads(v98b.DST.read_text(encoding="utf-8"))


# ---- v98b honest fold-only ensemble ----

def test_v98b_all_cells_parse(nb_v98b):
    _parseable(nb_v98b)


def test_v98b_isolation_and_failclosed(nb_v98b):
    src7 = _src(nb_v98b["cells"][7])
    assert "v98b_foldonly_colab_manifest.json" in _src(nb_v98b["cells"][1])
    # fold2 非接触: val は fold 内 deterministic 20%
    assert "members[4::5]" in src7
    # 学習済み fold 重みではなく onlyfold 学習成果を使う
    # (runs/yolo_fold_0|1 へのパス参照禁止 — コメント中の言及は許容)
    assert '"runs" / "yolo_fold_0"' not in src7
    assert '"runs" / "yolo_fold_1"' not in src7
    # fail-closed: onlyfold が prod を不審に上回ったら中止
    assert "fold2 接触の兆候" in src7
    # 汚染警告が contract とヘッダに入っている
    assert "contaminated" in src7
    hdr = _src(nb_v98b["cells"][0])
    assert "汚染" in hdr and "証拠に使わない" in hdr


def test_v98b_resume_and_tags(nb_v98b):
    src7 = _src(nb_v98b["cells"][7])
    assert "TRAINING_COMPLETE" in src7
    assert "V98_CAND_CACHE.unlink(missing_ok=True)" in src7
    assert 'tag=f"v98b_' in src7


def test_v98_header_marks_contamination(nb_v98):
    # v98 pooled は汚染結果 — ヘッダで証拠不適格を宣告 (Codex P1 対応)
    hdr = _src(nb_v98["cells"][0])
    assert "汚染" in hdr and "証拠に使わない" in hdr and "§2.74" in hdr
