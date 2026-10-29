"""v87 union ラベル純関数の単体テスト (合成マスク)。"""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BUILDER = Path(__file__).resolve().parents[1] / "scratch" / "build_v87_yolo_union_label.py"
spec = importlib.util.spec_from_file_location("b87", BUILDER)
b87 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b87)
NS = {}
exec(b87.UNION_SRC, NS)


def rect(y0, y1, x0, x1, size=64):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def iou(a, b):
    i = np.logical_and(a, b).sum()
    return i / max(np.logical_or(a, b).sum(), 1)


def mat(masks):
    return [[iou(a, b) for b in masks] for a in masks]


def test_three_annotators_one_filament_plus_singleton():
    masks = [rect(10, 20, 10, 40), rect(10, 21, 11, 40), rect(9, 20, 10, 39),  # 同一物体 r0,r1,r2
             rect(40, 50, 40, 60)]                                              # r1 だけ
    rec = [0, 1, 2, 1]
    clusters = NS["cluster_from_iou"](rec, mat(masks), 0.5)
    assert clusters == [[0, 1, 2], [3]]
    unions = NS["union_cluster_masks"](masks, rec, clusters)
    assert [a for _, a in unions] == [3, 1]
    assert unions[0][0].sum() == np.logical_or.reduce(masks[:3]).sum()
    hard = NS["copies_for_arm"](unions, 3, "union_hard")
    agree = NS["copies_for_arm"](unions, 3, "union_agree")
    assert hard == [[0, 1]] * 3
    assert agree == [[0, 1], [0], [0]]          # 合意3は3回、合意1は1回正例


def test_same_record_never_merged_even_indirectly():
    a, b = rect(10, 20, 10, 30), rect(10, 20, 14, 34)   # 同一 record 0 の重なる2個体
    c = rect(10, 20, 12, 32)                             # record 1、両方と IoU>0.5
    clusters = NS["cluster_from_iou"]([0, 0, 1], mat([a, b, c]), 0.5)
    assert sorted(len(x) for x in clusters) == [1, 2]
    for cl in clusters:
        assert len({[0, 0, 1][k] for k in cl}) == len(cl)


def test_low_iou_not_merged_and_yolo_line():
    a, b = rect(10, 20, 10, 30), rect(10, 20, 25, 45)  # IoU 0.2
    clusters = NS["cluster_from_iou"]([0, 1], mat([a, b]), 0.5)
    assert clusters == [[0], [1]]
    line, ncomp = NS["mask_to_yolo_line"](a)
    vals = [float(v) for v in line.split()[1:]]
    assert line.startswith("0 ") and ncomp == 1 and len(vals) % 2 == 0 and len(vals) >= 6
    assert all(0 <= v <= 1 for v in vals)
    assert NS["mask_to_yolo_line"](rect(0, 2, 0, 2))[0] is None   # min_area 未満


def test_multi_component_reports_count():
    m = rect(5, 15, 5, 15) | rect(40, 60, 40, 60)
    line, ncomp = NS["mask_to_yolo_line"](m)
    assert ncomp == 2 and line is not None


def test_sparse_pq_reduces_to_pq_at_r1():
    assert abs(NS["sparse_label_pq"](7.0, 10, 4, 6, r=1.0) - 7.0 / (10 + 2 + 3)) < 1e-12
    # 既知値チェック: README pooled counts TP5082/FP2286/FN3117/SQ0.678 → ≈0.388 (RESEARCH_LOG §2.28)
    v = NS["sparse_label_pq"](0.678 * 5082, 5082, 2286, 3117, r=0.77)
    assert abs(v - 0.388) < 0.002


def test_generated_notebook_cells_parse(tmp_path=None):
    import subprocess
    out = Path(tmp_path or "/tmp/v87nb")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(BUILDER), "--arm", "both", "--out-dir", str(out)], check=True)
    nb = json.loads(next(out.glob("*.ipynb")).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "\n".join(l for l in "".join(cell["source"]).splitlines() if not l.lstrip().startswith(("%", "!")))
            ast.parse(src)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
