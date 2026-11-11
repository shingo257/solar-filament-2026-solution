"""v120 notebook must require independently reviewed provenance before IO,
embed the consensus-derived eval script matching the repo copy, and keep the
preregistered detector-consensus arm spec (§2.98) with trusted-sha256 pins for
the manifest-undeclared crop/yolo26 detector checkpoints."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "solar_filament_v120_consensus_colab.ipynb"


def test_notebook_requires_reviewed_pin_before_input_io(monkeypatch):
    monkeypatch.delenv("V73_PROVENANCE_SHA256", raising=False)
    monkeypatch.setattr(Path, "rglob", lambda *args: pytest.fail("input IO before review pin"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code")
    with pytest.raises(RuntimeError, match="independent.*review"):
        exec(compile(source, "nb", "exec"), {})  # noqa: S102 - run repository preflight with IO blocked


def test_generated_notebook_matches_builder():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_v120", ROOT / "scratch/build_v120_consensus_colab.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    script = (ROOT / "scratch/run_fold_oof_eval_consensus.py").read_text(encoding="utf-8")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    built = builder.build_notebook(script)
    assert [c["source"] for c in built["cells"]] == [c["source"] for c in notebook["cells"]]


def test_embedded_consensus_script_matches_repo_file():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    tree = ast.parse(source)
    embedded = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "CONSENSUS_SCRIPT" for t in node.targets):
            embedded = ast.literal_eval(node.value)
    assert embedded is not None, "CONSENSUS_SCRIPT assignment missing in setup cell"
    repo = (ROOT / "scratch/run_fold_oof_eval_consensus.py").read_text(encoding="utf-8")
    assert embedded == repo
    ast.parse(embedded)


def test_v120_consensus_arms_preregistered():
    """v120 eval cell keeps the preregistered detector set (§2.98): prod yolo
    (manifest binding) + v116 crop + v117 yolo26 (trusted byte-pin)."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][2]["source"])
    assert "yolo={yolo_ckpt}" in source
    assert "crop={crop_ckpt}" in source
    assert "yolo26={yolo26_ckpt}" in source
    assert "--detector" in source and "--trusted-sha256" in source
    assert "baseline={baseline_ckpt}" in source
    setup = "".join(notebook["cells"][1]["source"])
    # detector byte-pin は §2.98 で事前固定
    assert "c23b1bb79620c62fd85984d032310bae26c240fd1be0aaf7df364fbc560fbcda" in setup
    assert "0c8bb71f6ab4398aaea2b729992a3b88f3f4d707e0cbcbe35f1b82a2533db7a5" in setup
    assert "TRUSTED_SHA256" in setup
    # 実行時に byte-pin 照合してから eval へ進む
    assert "trusted sha256 mismatch" in source and "_sha(p)" in source
    summary = "".join(notebook["cells"][3]["source"])
    assert "best_delta >= 0.010" in summary  # preregistered promotion threshold (§2.98)
    assert "0.444238" in summary  # prod arm sanity vs v76 baseline


def test_trusted_sha256_keyed_by_path_not_name():
    """回帰: crop/yolo26 は両方 best.pt 同名 — TRUSTED_SHA256 を name key に
    すると yolo26 の sha で crop が照合されて誤拒否する。dict 構築を
    実際に exec して path→sha の一意対応を検証する。"""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][1]["source"])
    tree = ast.parse(setup)
    assign = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.Assign) and any(
                      getattr(t, "id", "") == "TRUSTED_SHA256"
                      for t in n.targets))
    crop = Path("/d/runs/v116_croptrain/yolo_croptrain_fold2/weights/best.pt")
    yolo26 = Path("/d/runs/v117_yolo26/yolo_yolo26m_fold2/weights/best.pt")
    ns = {"inputs": {"crop_ckpt": crop, "yolo26_ckpt": yolo26}}
    exec(compile(ast.Module(body=[assign], type_ignores=[]), "<s>", "exec"), ns)
    trusted = ns["TRUSTED_SHA256"]
    assert len(trusted) == 2  # name key なら 1 に潰れる
    assert trusted[str(crop)].startswith("c23b1bb7")
    assert trusted[str(yolo26)].startswith("0c8bb71f")


def test_consensus_eval_script_scoring_path_unchanged():
    """派生 script の差分は multi-detector 提案 cluster のみ — 採点経路
    (refine → threshold → ownership → PQ) は v73 版と同一。"""
    cons = (ROOT / "scratch/run_fold_oof_eval_consensus.py").read_text(encoding="utf-8")
    orig = (ROOT / "scratch/run_fold_oof_eval.py").read_text(encoding="utf-8")
    for marker in (
        "rule=\"drop_contained_then_area_asc\"",
        "containment_threshold=DROP_THRESHOLD",
        "min_area=MIN_AREA, canvas=EVAL_SIZE",
        "meters[threshold].add_matrix(matrix, len(truth),",
        "def refine_probability(refiner, image, proposal):",
        "CROP_SIZE = 320",
        "DROP_THRESHOLD = 0.90",
        "EVAL_SIZE = 2048",
    ):
        assert marker in cons and marker in orig, marker
    # consensus 機構の marker
    for marker in ('"--detector"', '"--consensus-iou"', '"--det-floor"',
                   "def consensus_clusters(props, iou_thr):",
                   "props[i][0] == props[j][0]",
                   '"prod", "union3", "union3_c26", "cons2of3",'):
        assert marker in cons, marker


def _extract_consensus_clusters():
    """eval script 内の nested helper を exec して単体テスト可能にする。"""
    import numpy as np
    from collections import defaultdict
    src = (ROOT / "scratch/run_fold_oof_eval_consensus.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "consensus_clusters", "_mask_iou"):
            fns[node.name] = node
    assert set(fns) == {"consensus_clusters", "_mask_iou"}
    mod = ast.Module(body=list(fns.values()), type_ignores=[])
    ns = {"np": np, "defaultdict": defaultdict}
    exec(compile(mod, "<p>", "exec"), ns)
    return ns["consensus_clusters"], np


def test_consensus_clusters_merges_only_cross_detector_iou():
    """cluster semantics: cross-detector mask-IoU>=thr edge のみ結合、
    同 detector 内は edge なし、推移閉包で結合される。"""
    consensus_clusters, np = _extract_consensus_clusters()
    canvas = np.zeros((64, 64), dtype=bool)
    a = canvas.copy(); a[10:30, 10:30] = True
    b = canvas.copy(); b[11:31, 10:30] = True   # a と IoU~0.9 (cross det)
    c = canvas.copy(); c[40:55, 40:55] = True   # 非重複
    d = canvas.copy(); d[12:32, 10:30] = True   # a/b と IoU 高いが同 det
    props = [("yolo", a, 0.5), ("crop", b, 0.4), ("yolo26", c, 0.6),
             ("crop", d, 0.3)]
    clusters = consensus_clusters(props, 0.5)
    sets = [frozenset(m) for m in clusters]
    # yolo-crop edge で a,b が結合。同 det の b-d edge はないが d は
    # a (yolo) との cross-det edge でクラスタに入る → {0,1,3}
    assert frozenset({0, 1, 3}) in sets
    assert frozenset({2}) in sets             # c は孤立
    # 支持数は member 数ではなく distinct detector 数 ({yolo,crop}=2)
    sources = {props[i][0] for i in (0, 1, 3)}
    assert sources == {"yolo", "crop"}


def test_consensus_clusters_transitive_closure():
    """a-b, b-c が cross-detector edge を持てば a-c 非重複でも推移結合。"""
    consensus_clusters, np = _extract_consensus_clusters()
    canvas = np.zeros((64, 64), dtype=bool)
    a = canvas.copy(); a[10:30, 10:30] = True   # a-b IoU=0.538
    b = canvas.copy(); b[10:30, 16:36] = True   # b-c IoU=0.538
    c = canvas.copy(); c[10:30, 22:42] = True   # a-c IoU=0.25 (<0.5)
    props = [("yolo", a, 0.5), ("crop", b, 0.4), ("yolo26", c, 0.6)]
    clusters = consensus_clusters(props, 0.5)
    assert len(clusters) == 1 and len(clusters[0]) == 3
