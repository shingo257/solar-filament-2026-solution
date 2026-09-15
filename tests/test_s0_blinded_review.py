from __future__ import annotations

import re

import numpy as np
import pandas as pd

from scratch.run_s0_blinded_review import (
    COMPARISONS,
    apply_blinding,
    balanced_assignments,
    build_review_html,
    focus_box,
    top_unique,
)


def test_balanced_assignments_even_count_is_exactly_balanced():
    assignments = balanced_assignments(8, np.random.default_rng(7))

    assert assignments.dtype == bool
    assert assignments.sum() == 4


def test_balanced_assignments_odd_count_differs_by_at_most_one():
    assignments = balanced_assignments(7, np.random.default_rng(11))

    assert abs(int(assignments.sum()) - int((~assignments).sum())) == 1


def test_top_unique_respects_score_and_exclusions():
    candidates = pd.DataFrame({
        "file_name": ["used.png", "b.png", "a.png", "b.png"],
        "selection_score": [100.0, 3.0, 2.0, 1.0],
    })
    excluded = {"used.png"}

    selected = top_unique(candidates, 2, excluded)

    assert selected.file_name.tolist() == ["b.png", "a.png"]
    assert excluded == {"used.png", "a.png", "b.png"}


def test_apply_blinding_balances_each_comparison_and_uses_unique_cases():
    rows = []
    for comparison in COMPARISONS:
        for index in range(8):
            rows.append({
                "file_name": f"{comparison}-{index}.png",
                "comparison": comparison,
                "selection_stratum": "test",
                "selection_score": float(index),
            })

    blinded, key = apply_blinding(pd.DataFrame(rows), seed=20260824)

    assert len(blinded) == len(key) == 24
    assert blinded.case_id.nunique() == 24
    assert blinded.file_name.nunique() == 24
    for comparison, (first_model, _) in COMPARISONS.items():
        group = blinded.loc[blinded.comparison == comparison]
        assert (group.output_a == first_model).sum() == 4
        assert (group.output_b == first_model).sum() == 4


def test_focus_box_centers_largest_disagreement_and_stays_in_bounds():
    output_a = np.zeros((512, 512), dtype=bool)
    output_a[10:14, 10:14] = True
    output_a[400:430, 390:430] = True
    output_b = np.zeros_like(output_a)

    x0, y0, x1, y1 = focus_box([output_a], [output_b], [])

    assert 0 <= x0 < x1 <= 512
    assert 0 <= y0 < y1 <= 512
    assert x0 <= 410 < x1
    assert y0 <= 415 < y1
    assert not (x0 <= 10 < x1 and y0 <= 10 < y1)


def test_review_html_is_blinded_and_uses_independent_radio_groups():
    page = build_review_html([
        {"case_id": "case_001", "image": "case_001.png"},
        {"case_id": "case_002", "image": "case_002.png"},
    ])

    assert "localStorage" in page
    assert "判定JSONを書き出す" in page
    assert 'name="case_001-morphology"' in page
    assert 'name="case_001-catalog"' in page
    assert len(re.findall(r'name="case_001-morphology"', page)) == 4
    assert len(re.findall(r'name="case_001-catalog"', page)) == 4
    for model_identity in ("raw", "v35", "v42", "v42_nested_verifier"):
        assert re.search(rf"\b{re.escape(model_identity)}\b", page) is None
