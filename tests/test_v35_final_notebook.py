"""V35 最終提出ノートが Colab で単体実行できることを固定する。

以前の骨格は `/content/mediseg_models.py` を exec しており、そのファイルが
無いと MediSegUNet 未定義で止まっていた。ビルダーは V30 と同じく
`src/mediseg_models.py` を埋め込むこと。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOTEBOOK = ROOT / "solar_filament_v35_final_submission_colab.ipynb"


def _source(cell: dict[str, object]) -> str:
    raw = cell["source"]
    if isinstance(raw, list):
        return "".join(str(item) for item in raw)
    return str(raw)


def _notebook_text() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(_source(cell) for cell in notebook["cells"])


def test_notebook_embeds_mediseg_unet() -> None:
    text = _notebook_text()
    assert "class MediSegUNet" in text
    assert "/content/mediseg_models.py" not in text
    assert "exec(open" not in text


def test_notebook_uses_confirmed_submission_config() -> None:
    text = _notebook_text()
    assert "SUBMISSION_CONF = 0.26" in text
    assert 'SUBMISSION_OWNERSHIP_RULE = "drop_contained_then_area_asc"' in text
    assert 'REFINER_METHOD = "shrink"' in text
    assert "REFINER_THRESHOLD = 0.45" in text


def test_notebook_keeps_v30_fold_ensemble() -> None:
    text = _notebook_text()
    assert "def ensemble_proposals" in text
    assert "votes >= 2" in text
    assert "def apply_ownership" in text
    assert "refined & proposal" in text


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
