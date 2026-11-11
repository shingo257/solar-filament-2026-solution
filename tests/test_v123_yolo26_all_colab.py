"""v123 — yolo26m all-data detector 学習ノートブックの検証 (§2.99 補助)。

all-data det trio (yolo11m-all=yolo_full_fold2・crop-all=v116・yolo26m-all=本稿)
の欠員を v121c ゲート判定と並行して投機的に学習するノートブック。
fold-0 を train に含むため fold-0 det eval は in-sample informational のみで、
OOF/ゲート証拠としては使用しない — ノートブックがそのことを明示するか検査する。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "solar_filament_v123_yolo26_all_colab.ipynb"


def _cell_src(nb_path: Path, idx: int) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    src = nb["cells"][idx]["source"]
    return "".join(src) if isinstance(src, list) else src


def test_v123_alldata_config():
    """all-data 学習: TRAIN_FOLDS は全 fold、パスは v123 専用。"""
    cfg = _cell_src(NB, 1)
    assert "FOLD = 0" in cfg
    assert "TRAIN_FOLDS = (0, 1, 2)" in cfg
    assert "'yolo26m_all': {'yolo_model': 'yolo26m-seg.pt'}" in cfg
    assert "v123_yolo26_all" in cfg
    assert "v123_yolo26_all_colab_manifest.json" in cfg
    assert "0.44413694566994816" in cfg  # fold-0 prod sanity (warn-only)
    # fold-1/2 専用パス・v121 系パスへの書き込みなし
    assert "v121_yolo26_f0" not in cfg
    assert "v96_data_scaling" not in cfg


def test_v123_insample_labeling():
    """fold-0 eval が in-sample informational である旨を manifest/title に残す。"""
    title = _cell_src(NB, 0)
    assert "in-sample" in title
    assert "投機" in title
    run = _cell_src(NB, 6)
    assert '"v123_yolo26_all"' in run
    assert "in-sample" in run
    assert "weights_sha256" in run  # trusted-pin 記録
    assert "resumed_from_last_pt" in run  # v121b と同じ resume 機構


def test_v123_cell_count():
    assert len(json.loads(NB.read_text(encoding="utf-8"))["cells"]) == 7
