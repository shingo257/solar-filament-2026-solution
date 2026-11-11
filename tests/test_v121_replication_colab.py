"""v121a/v121b/v121c — cons2of3_c26 fold-0 複製ノートブック群の検証 (§2.99)。

- v121a/v121b は v116/v117 レシピ verbatim の fold ミラー (FOLD=0,
  TRAIN_FOLDS=(1,2)) であり、学習 manifest に weights_sha256 を記録する
- v121c は reviewed provenance 必須・embedded consensus script が repo
  版と一致・fold-0 で 3 detector を評価・trusted-pin を学習 manifest
  から読み取る"""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_A = ROOT / "solar_filament_v121a_croptrain_f0_colab.ipynb"
NB_B = ROOT / "solar_filament_v121b_yolo26_f0_colab.ipynb"
NB_C = ROOT / "solar_filament_v121c_consensus_f0_colab.ipynb"


def _cell_src(nb_path, idx):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return "".join(nb["cells"][idx]["source"])


def test_v121a_fold0_recipe():
    """v121a: v116 crop レシピの fold-0 ミラー + 漏洩修正 (fit_files のみ)。"""
    cfg = _cell_src(NB_A, 1)
    assert "FOLD = 0" in cfg
    assert "TRAIN_FOLDS = (1, 2)" in cfg
    assert "CROP_TILE = 896" in cfg and "CROP_STARTS = [0, 448, 1152]" in cfg
    assert "TRAIN_IMGSZ = 896" in cfg
    assert "v121_croptrain_f0_honest" in cfg
    assert "v121a_croptrain_f0_colab_manifest.json" in cfg
    assert "0.44413694566994816" in cfg  # fold-0 prod 確定値
    # 漏洩修正: crop train は fit_files (folds(1,2)) のみ — 全 file は使わない
    ds = _cell_src(NB_A, 3)
    assert 'tqdm(fit_files, desc="crop dataset (train)")' in ds
    assert 'tqdm(train_files, desc="crop dataset (train)")' not in ds
    run = _cell_src(NB_A, 6)
    assert '"v121a_croptrain_f0"' in run
    assert "weights_sha256" in run  # v121c の trusted-pin 記録
    # 凍結セル (cell2 COCO/fold・cell5 評価関数) が残る
    assert len(json.loads(NB_A.read_text(encoding="utf-8"))["cells"]) == 7


def test_v121b_fold0_recipe():
    """v121b: v117 yolo26m レシピ verbatim の fold-0 ミラー。"""
    cfg = _cell_src(NB_B, 1)
    assert "FOLD = 0" in cfg
    assert "TRAIN_FOLDS = (1, 2)" in cfg
    assert "yolo26m" in cfg
    assert "v121_yolo26_f0" in cfg
    assert "v121b_yolo26_f0_colab_manifest.json" in cfg
    assert "0.44413694566994816" in cfg
    run = _cell_src(NB_B, 6)
    assert '"v121b_yolo26_f0"' in run
    # v117 と同じ resume 機構を保持
    assert "resumed_from_last_pt" in run


def test_v121c_requires_reviewed_pin_before_io(monkeypatch):
    monkeypatch.delenv("V73_FOLD0_PROVENANCE_SHA256", raising=False)
    monkeypatch.setattr(Path, "rglob", lambda *args: pytest.fail("input IO before review pin"))
    notebook = json.loads(NB_C.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code")
    with pytest.raises(RuntimeError, match="independent.*review"):
        exec(compile(source, "nb", "exec"), {})  # noqa: S102


def test_v121c_fold0_manifest_selfsha_reproduces_pin():
    """fold-0 manifest をセルの生成式から構築すると §2.99 記録の自己 sha に一致。

    ピンは実行時に導出せず、notebook セルの component sha (yolo/baseline/
    folds) から決定論的に構築する。V73_FOLD0_PROVENANCE_SHA256 =
    da3d53abc94023f61998ace4b54db98cfa1792a6b8660820583ec957f26e6a53"""
    import hashlib
    setup = _cell_src(NB_C, 1)
    assert "FOLD0_YOLO_SHA" in setup and "FOLD0_REFINER_SHA" in setup
    # セル中の pin 値と生成式を AST から抽出して manifest bytes を再現
    ns = {"json": json}
    for node in ast.walk(ast.parse(setup)):
        if not isinstance(node, ast.Assign):
            continue
        names = {getattr(t, "id", "") for t in node.targets}
        if names & {"FOLD0_YOLO_SHA", "FOLD0_REFINER_SHA",
                    "EXPECTED_FOLDS_SHA", "fold0_manifest"}:
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                       "nb", "exec"), ns)  # noqa: S102
    blob = json.dumps(ns["fold0_manifest"], sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == (
        "da3d53abc94023f61998ace4b54db98cfa1792a6b8660820583ec957f26e6a53")
    # fold-2 manifest は使わない (fold/binding 不一致のため download しない)
    assert "solar-v73-assets" not in setup
    assert "load_reviewed_manifest(manifest_path, EXPECTED_PROVENANCE_SHA, 0)" in setup


def test_v121c_embedded_script_matches_repo():
    notebook = json.loads(NB_C.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    tree = ast.parse(source)
    embedded = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "CONSENSUS_SCRIPT" for t in node.targets):
            embedded = ast.literal_eval(node.value)
    assert embedded is not None
    repo = (ROOT / "scratch/run_fold_oof_eval_consensus.py").read_text(encoding="utf-8")
    assert embedded == repo
    ast.parse(embedded)


def test_v121c_fold0_eval_spec():
    """v121c eval: fold-0・3 detector (v121a/v121b run dir)・trusted-pin は
    学習 manifest 由来・gate は cons2of3_c26 >=+0.005。"""
    setup = _cell_src(NB_C, 1)
    assert "yolo_fold_0" in setup
    assert "v121_croptrain_f0_honest/yolo_croptrain_fold0" in setup
    assert "v121_yolo26_f0/yolo_yolo26m_fold0" in setup
    assert "mediseg_edgeattnet_refiner_fold0_resume.pt" in setup
    assert "v121a_croptrain_f0_colab_manifest.json" in setup
    assert "v121b_yolo26_f0_colab_manifest.json" in setup
    # trusted-pin は manifest 記録値から構築 (ハードコード pin でない)
    assert "_v121a_man['datasets']['croptrain']['weights_sha256']" in setup
    assert "_v121b_man['results']['yolo26m']['weights_sha256']" in setup
    # fold-0 の fold マッピング検証
    assert "for i in coco['images']}, 0," in setup
    source = _cell_src(NB_C, 2)
    assert "'--fold', '0'" in source
    assert "--detector" in source and "--trusted-sha256" in source
    assert "v121_eval.json" in source
    summary = _cell_src(NB_C, 3)
    assert "rep_delta >= 0.005" in summary  # §2.99 複製ゲート
    assert "0.444137" in summary            # fold-0 prod sanity
    assert "cons2of3_c26" in summary and "cross_fold_verified" in summary
    assert "solar_filament_v121" in "".join(
        json.loads(NB_C.read_text(encoding="utf-8"))["cells"][1]["source"])
