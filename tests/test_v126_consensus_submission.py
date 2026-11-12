"""v126 — exploratory consensus 提出ノートブック検証。

逐次ゲート (v121c) 未達で閉鎖済みだが、フロー外 fold-2 clean 複製
(v124c +0.008565) により 2-fold honest 証拠が両方陽性となった。
v126 は「両 summary を読み両方正 Δ を確認する」ガードを持つ
exploratory 提出経路で、v122 の cross_fold_verified ガード
(事前登録) とは別物であることを検査する。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "solar_filament_v126_consensus_submission_colab.ipynb"


def _cell_src(idx: int) -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = nb["cells"][idx]["source"]
    return "".join(src) if isinstance(src, list) else src


def test_v126_requires_both_fold_summaries():
    """cell 1 が v121/v124 両 summary を必須とし、両方正 Δ を検証する。"""
    c1 = _cell_src(1)
    assert "v121_summary.json" in c1 and "v124_summary.json" in c1
    assert "cons2of3_c26_delta_fold0" in c1
    assert "cons2of3_c26_delta_fold2" in c1
    assert "d_f0 > 0 and d_f2 > 0" in c1


def test_v126_not_cross_fold_guard():
    """v122 の cross_fold_verified 判定ガードは使わない (別経路)。

    コメント中の言及は許容するが、decision を gate 条件として
    比較するコードが無いことを確認する。
    """
    c1 = _cell_src(1)
    assert "decision') != 'cross_fold_verified'" not in c1
    assert "decision') == 'cross_fold_verified'" not in c1


def test_v126_predeclared_selection_rule():
    """選択ルールが事前申告されている: public > 0.39 → 差替。"""
    md = _cell_src(0)
    assert "0.39" in md
    assert "差替" in md or "維持" in md


def test_v126_detector_paths_measured():
    """v122 と同一の計測済み det 構成 (all-data trio 未計測のため不使用)。"""
    c1 = _cell_src(1)
    assert "yolo_fold_2" in c1 and "v116_croptrain" in c1 and "v117_yolo26" in c1
    assert "v123_yolo26_all" not in c1 and "yolo_full" not in c1


def test_v126_embeds_consensus_script():
    """cell 2 が run_test_submission_consensus.py を embed。"""
    c2 = _cell_src(2)
    assert "run_test_submission_consensus.py" in c2
    assert "cons2of3" in c2


def test_v126_output_path():
    assert "consensus_v126" in _cell_src(4)


def test_v126_cell_count():
    assert len(json.loads(NB.read_text(encoding="utf-8"))["cells"]) == 5
