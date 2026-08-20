"""P2-3 推論解像度探針が学習せず、V30 重みと V35 cache だけを使うこと。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOTEBOOK = ROOT / "solar_filament_v36_infer_imgsz1536_fold0_colab.ipynb"


def _source(cell: dict[str, object]) -> str:
    raw = cell["source"]
    if isinstance(raw, list):
        return "".join(str(item) for item in raw)
    return str(raw)


def _notebook_text() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(_source(cell) for cell in notebook["cells"])


def test_infer_probe_does_not_train_or_overwrite_v30() -> None:
    text = _notebook_text()
    assert "model.train(" not in text
    assert 'WEIGHTS = V30_ROOT / "runs" / "yolo_fold_0" / "weights" / "best.pt"' in text
    assert "yolo_v30w_imgsz1536_fold_0" in text
    assert "CACHE_ROOT = V35_ROOT" in text or 'CACHE_ROOT = V35_ROOT / "prediction_cache"' in text
    assert "from __future__ import annotations" not in text
    assert "coco_anns" in text
    assert "mask_utils.iou(pred, truth, [0] * len(truth))" in text
    assert "BASELINE_FOLD0_PQ = 0.4393" in text
    assert "YOLO_IMGSZ = 1536" in text


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
