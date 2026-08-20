"""P2-2 探針ノートが V30 重みを上書きせず、imgsz 1536 / fold0 に固定されていること。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOTEBOOK = ROOT / "solar_filament_v36_yolo_imgsz1536_fold0_colab.ipynb"


def _source(cell: dict[str, object]) -> str:
    raw = cell["source"]
    if isinstance(raw, list):
        return "".join(str(item) for item in raw)
    return str(raw)


def _notebook_text() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(_source(cell) for cell in notebook["cells"])


def test_probe_does_not_overwrite_v30_yolo_runs() -> None:
    text = _notebook_text()
    assert 'RUNS_ROOT = V35_ROOT / "runs"' in text
    assert "yolo_imgsz1536_fold_" in text
    assert "assert \"solar_filament_v30\" not in str(CFG.RUNS_ROOT)" in text


def test_probe_uses_imgsz_1536_and_fold0_only() -> None:
    text = _notebook_text()
    assert "YOLO_IMGSZ = 1536" in text
    assert "FOLD = 0" in text
    assert "BASELINE_FOLD0_PQ = 0.4393" in text
    assert "class MediSegUNet" in text
    assert "class PanopticQuality" in text


def test_metrics_cell_does_not_shadow_coco_annotations() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metrics = next(
        _source(cell)
        for cell in notebook["cells"]
        if _source(cell).startswith("# [7/8]")
    )
    assert "from __future__ import annotations" not in metrics
    score = next(
        _source(cell)
        for cell in notebook["cells"]
        if _source(cell).startswith("# [8/8]")
    )
    assert "if not isinstance(annotations, Mapping)" in score
    assert "mask_utils.iou(pred, truth, [0] * len(truth))" in score
    assert "mask_utils.iou(pred, truth, [0] * len(pred))" not in score


def test_probe_locks_v35_postprocess() -> None:
    text = _notebook_text()
    assert "CONF = 0.26" in text
    assert "REFINER_THRESHOLD = 0.45" in text
    assert "drop_contained" in text


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
