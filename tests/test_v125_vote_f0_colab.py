"""v125 — 検出集合固定の cross-proposal 輪郭投票 eval ノートブック検証 (P2)。

prod 検出集合 (yolo conf>=0.26) を leader として固定し、crop (v121a)・
yolo26 (v121b) の refined mask で等重み pixel 多数決 (>=0.5) を行う
開発側 eval。新規個体は出さない。ノートブックが honest det のみ・
fold-0・事前固定判定を守るかを検査する。
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "solar_filament_v125_vote_f0_colab.ipynb"
SCRIPT = ROOT / "scratch" / "run_fold_oof_eval_vote.py"


def _cell_src(idx: int) -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = nb["cells"][idx]["source"]
    return "".join(src) if isinstance(src, list) else src


def test_v125_requires_reviewed_pin_before_io(monkeypatch):
    """外部 pin 未設定では入力 I/O 前に停止 (v121c と同じ fail-closed)。"""
    monkeypatch.delenv("V73_FOLD0_PROVENANCE_SHA256", raising=False)
    monkeypatch.setattr(Path, "rglob", lambda *args: pytest.fail("input IO before review pin"))
    notebook = json.loads(NB.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code")
    with pytest.raises(RuntimeError, match="independent.*review"):
        exec(compile(source, "nb", "exec"), {})  # noqa: S102


def test_v125_embedded_script_matches_repo():
    """埋め込み eval script はリポジトリ版 vote script と byte 同一。"""
    setup = _cell_src(1)
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert repr(script_text) in setup or repr(script_text)[1:-1] in setup


def test_v125_honest_detector_paths():
    setup = _cell_src(1)
    assert "v121_croptrain_f0_honest" in setup
    assert "v121_yolo26_f0" in setup
    assert "v116_croptrain" not in setup and "v121_croptrain_f0/" not in setup.replace("honest", "")
    assert "v121a_croptrain_f0_colab_manifest.json" in setup
    assert "v121b_yolo26_f0_colab_manifest.json" in setup


def test_v125_eval_spec():
    ev = _cell_src(2)
    assert "'--fold', '0'" in ev
    assert "run_fold_oof_eval_vote.py" in ev
    for det in ("yolo=", "crop=", "yolo26="):
        assert det in ev
    assert "--trusted-sha256" in ev
    assert "v125_eval.json" in ev


def test_v125_gate_semantics():
    sm = _cell_src(3)
    assert "vote_eq05_delta_fold0" in sm
    assert "vote_positive_dev" in sm and "vote_negative" in sm
    assert "rep_delta >= 0.005" in sm
    assert "0.444137" in sm
    assert "v125_summary.json" in sm


def test_vote_script_fixed_detection():
    """vote script: leader は yolo conf>=--conf のみ・追加個体を出さない。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert 'd == "yolo" and s >= args.conf' in src
    assert "vote_eq05" in src and '"prod"' in src
    assert "args.vote_iou" in src
    # voter は非 yolo のみ・leader 未選定の提案は emitted しない
    assert 'if d != "yolo"' in src


def test_v125_cell_count():
    assert len(json.loads(NB.read_text(encoding="utf-8"))["cells"]) == 4
