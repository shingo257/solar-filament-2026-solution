"""v124c — cons2of3_c26 fold-2 honest 複製 eval ノートブックの検証 (§2.99 補完)。

v120 の fold-2 証拠は v116 crop det 漏洩で撤回済み。本ノートは v121c
(fold-0) ゲート通過時のみ実行する第二 fold 確認で、crop det を v124a
honest 版に差し替えて v120 と同一の 6 arm 評価を行う。trusted-pin は
各学習 manifest から実行時に読み取る (v121c と同じ方式)。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "solar_filament_v124c_consensus_f2_colab.ipynb"
SCRIPT = ROOT / "scratch" / "run_fold_oof_eval_consensus.py"


def _cell_src(idx: int) -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = nb["cells"][idx]["source"]
    return "".join(src) if isinstance(src, list) else src


def test_v124c_requires_reviewed_pin_before_io(monkeypatch):
    """外部 pin 未設定では入力 I/O 前に停止 (v121c と同じ fail-closed)。"""
    monkeypatch.delenv("V73_PROVENANCE_SHA256", raising=False)
    monkeypatch.setattr(Path, "rglob", lambda *args: pytest.fail("input IO before review pin"))
    notebook = json.loads(NB.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code")
    with pytest.raises(RuntimeError, match="independent provenance review"):
        exec(compile(source, "nb", "exec"), {})  # noqa: S102


def test_v124c_fold2_inputs_and_honest_crop():
    setup = _cell_src(1)
    assert "v124_croptrain_f2_honest" in setup
    assert "v116_croptrain" not in setup  # 漏洩 ckpt を参照しない
    assert "v124a_croptrain_f2_colab_manifest.json" in setup
    assert "v117_yolo26_colab_manifest.json" in setup
    assert "_v124a_man['datasets']['croptrain']['weights_sha256']" in setup
    assert "_v117_man['results']['yolo26m']['weights_sha256']" in setup
    assert "solar-v73-assets" in setup  # fold-2 reviewed manifest は assets から
    assert "mediseg_edgeattnet_refiner_fold2_resume.pt" in setup
    assert "yolo_fold_2" in setup


def test_v124c_eval_spec():
    ev = _cell_src(2)
    assert "'--fold', '2'" in ev
    for det in ("yolo=", "crop=", "yolo26="):
        assert f"'--detector', f'{det}" in ev or f"--detector', f'{det}" in ev
    assert "--trusted-sha256" in ev
    assert "v124_eval.json" in ev


def test_v124c_gate_semantics():
    sm = _cell_src(3)
    assert "cons2of3_c26_delta_fold2" in sm
    assert "second_fold_positive" in sm and "second_fold_negative" in sm
    assert "rep_delta >= 0.005" in sm
    assert "0.444238" in sm  # fold-2 prod sanity
    assert "v120_retracted_delta" in sm  # 撤回済み値は参照のみ
    assert "v124_summary.json" in sm


def test_v124c_embedded_script_matches_repo():
    """埋め込み eval script はリポジトリ版と byte 同一。"""
    setup = _cell_src(1)
    assert repr(SCRIPT.read_text(encoding="utf-8"))[1:-1] in setup or \
        SCRIPT.read_text(encoding="utf-8") in setup.replace("\\n", "\n").replace("\\'", "'")


def test_v124c_cell_count():
    assert len(json.loads(NB.read_text(encoding="utf-8"))["cells"]) == 4
