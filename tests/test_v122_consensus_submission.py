"""v122 consensus 提出スクリプト/ノートブックの検証 (contingent、§2.99)。

- v121c `cross_fold_verified` なしに提出 CSV を生成しないガード
- emit_cons2of3_c26 が §2.98 eval と同じ semantics
  (conf>=0.26 の支持が >=2 detector のクラスタのみ放出)
- detector sha pin・prod 後処理経路の保持
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "solar_filament_v122_consensus_submission_colab.ipynb"
SCRIPT = ROOT / "scratch/run_test_submission_consensus.py"


def test_gate_guard_requires_cross_fold_verified():
    """v121c decision != cross_fold_verified ならセル1が RuntimeError で止まる。"""
    setup = "".join(json.loads(NOTEBOOK.read_text(encoding="utf-8"))
                    ["cells"][1]["source"])
    assert "v121_summary.json" in setup
    assert "cross_fold_verified" in setup
    assert "raise RuntimeError" in setup
    # gate 判定が先・パス検証が後
    assert setup.index("cross_fold_verified") < setup.index("test_images")


def test_embedded_submission_script_matches_repo():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][2]["source"])
    tree = ast.parse(source)
    embedded = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "CONSENSUS_SUB" for t in node.targets):
            embedded = ast.literal_eval(node.value)
    assert embedded is not None
    assert embedded == SCRIPT.read_text(encoding="utf-8")
    ast.parse(embedded)


def _extract(fns_names):
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name in fns_names]
    import numpy as np
    from collections import defaultdict
    ns = {"np": np, "defaultdict": defaultdict,
          "CONSENSUS_IOU": 0.5, "SUPPORT_CONF": 0.26}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "<s>", "exec"), ns)
    return ns


def test_emit_cons2of3_c26_semantics():
    """放出 = 支持 member (conf>=0.26) が >=2 detector のクラスタのみ。
    低conf の2-det 支持・高conf の1-det 単独は放出しない。"""
    import numpy as np
    ns = _extract({"emit_cons2of3_c26", "consensus_clusters", "_mask_iou"})
    canvas = np.zeros((64, 64), dtype=bool)
    a = canvas.copy(); a[10:30, 10:30] = True
    b = canvas.copy(); b[11:31, 10:30] = True
    c = canvas.copy(); c[40:55, 40:55] = True
    d = canvas.copy(); d[41:56, 41:55] = True   # c と IoU 高い (cross det)
    e = canvas.copy(); e[2:8, 50:58] = True    # crop 高conf singleton
    props = [
        ("yolo", a, 0.5), ("crop", b, 0.2),     # a-b cluster: 高conf は yolo のみ
        ("yolo", c, 0.5), ("yolo26", d, 0.5),   # c-d cluster: 2 det が高conf
        ("crop", e, 0.9),                       # singleton は放出しない
    ]
    emitted = ns["emit_cons2of3_c26"](props)
    masks = [m.tobytes() for m, _ in emitted]
    assert len(emitted) == 1                      # c-d cluster のみ
    assert c.tobytes() in masks or d.tobytes() in masks
    assert a.tobytes() not in masks               # 高conf 支持が1detのみ
    assert e.tobytes() not in masks               # singleton は放出しない


def test_prod_postprocess_and_detector_pins_preserved():
    src = SCRIPT.read_text(encoding="utf-8")
    # prod 後処理経路 (refine_crops → threshold → ownership → RLE)
    for marker in ("refine_crops(refiner, image, boxes)",
                   "rule=\"drop_contained_then_area_asc\"",
                   "containment_threshold=DROP_THRESHOLD",
                   "min_area=MIN_AREA",
                   "REFINER_THRESHOLD = 0.45",
                   "mediseg_edgeattnet_refiner_all_data_resume.pt"):
        assert marker in src, marker
    # §2.98 記録の detector sha pin (計測済み fold-2 構成)
    for marker in ("40ec31fd2c6aa8e3",
                   "c23b1bb79620c62fd85984d032310bae26c240fd1be0aaf7df364fbc560fbcda",
                   "0c8bb71f6ab4398aaea2b729992a3b88f3f4d707e0cbcbe35f1b82a2533db7a5",
                   "sha256 mismatch"):
        assert marker in src, marker
