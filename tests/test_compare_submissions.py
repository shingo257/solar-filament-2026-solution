"""Check CSV comparison against real compressed RLEs, without inference."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util


MODULE = Path(__file__).resolve().parents[1]


def _csv(path, positions):
    rows = []
    for index, left in enumerate(positions, 1):
        mask = np.zeros((2048, 2048), dtype=np.uint8, order="F")
        mask[10:20, left:left + 10] = 1
        rows.append({"filament_id": f"image_{index}",
                     "segmentation_rle": mask_util.encode(mask)["counts"].decode("utf-8")})
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.parametrize("a_positions,b_positions,expected_pairs,full", [
    ([10, 30], [10, 30], 2, 1),
    ([10, 30], [10], 1, 0),
    ([10], [10, 30], 1, 0),
    ([10], [10, 10], 1, 0),
])
def test_one_to_one_csv_comparison(tmp_path, monkeypatch, capsys,
                                   a_positions, b_positions, expected_pairs, full):
    script = MODULE / "scratch" / "compare_submissions.py"
    spec = importlib.util.spec_from_file_location("compare_submissions_tested", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _csv(a, a_positions)
    _csv(b, b_positions)
    monkeypatch.setattr(sys, "argv", [str(script), str(a), str(b)])
    assert module.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["b_matched_at_iou0.5"] == expected_pairs
    assert result["images_full_overlap"] == full
    assert result["match_rate_b"] == round(expected_pairs / len(b_positions), 4)
