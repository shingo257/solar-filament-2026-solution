"""v124a — crop detector fold-2 honest 学習ノートブックの検証 (§2.99 補完)。

v120 の fold-2 証拠撤回を受け、v121c (fold-0) ゲート通過時のみ投入する
第二 fold 複製。v116 漏洩バグ (`train_files` 全件) の修正版 `fit_files`
ループが fold-2 側にも効いていること、出力先が汚染 run と分離されている
ことを検査する。
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "solar_filament_v124a_croptrain_f2_colab.ipynb"


def _cell_src(idx: int) -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = nb["cells"][idx]["source"]
    return "".join(src) if isinstance(src, list) else src


def test_v124a_fold2_recipe():
    """FOLD=2・TRAIN_FOLDS=(0,1)・v116 crop レシピ・honest 専用パス。"""
    cfg = _cell_src(1)
    assert "FOLD = 2" in cfg
    assert "TRAIN_FOLDS = (0, 1)" in cfg
    for token in ("CROP_TILE = 896", "CROP_STRIDE = 448",
                  "CROP_STARTS = [0, 448, 1152]", "TRAIN_IMGSZ = 896"):
        assert token in cfg
    assert "v124_croptrain_f2_honest" in cfg
    assert "v124a_croptrain_f2_colab_manifest.json" in cfg
    assert "0.444238" in cfg  # fold-2 prod sanity (warn-only)
    # v121a・v116 系パスへの書き込みなし (汚染 artifact 温存)
    assert "v121_croptrain_f0" not in cfg
    assert "v116_croptrain" not in cfg


def test_v124a_uses_fit_files_only():
    """crop dataset は fit_files (folds 0,1) のみを回す — 漏洩修正の回帰。"""
    c3 = _cell_src(3)
    assert "for file_name in tqdm(fit_files" in c3
    assert "for file_name in tqdm(valid_files" in c3
    # train split ループに train_files 全件を使っていない
    assert not re.search(r"for file_name in tqdm\(train_files.*train", c3)


def test_v124a_run_cell():
    run = _cell_src(6)
    assert '"v124a_croptrain_f2"' in run
    assert "weights_sha256" in run
    assert "resumed_from_last_pt" in run or "resume" in run
    assert "v124c" in run  # 真の判定は fold-2 consensus eval

def test_v124a_cell_count():
    assert len(json.loads(NB.read_text(encoding="utf-8"))["cells"]) == 7
