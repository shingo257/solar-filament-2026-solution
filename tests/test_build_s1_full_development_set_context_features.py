from __future__ import annotations

import pandas as pd
import pytest

from scratch.build_s1_full_development_set_context_features import select_all_development


def _frame(role: str, fold: int, index: int) -> dict[str, object]:
    return {
        "file_name": f"{role}-{fold}-{index}.jpeg",
        "fold": fold,
        "time_block_role": role,
        "site_code": "Bh",
        "year": 2012 if role == "development" else 2020,
    }


def test_select_all_development_requires_exact_547_and_4899() -> None:
    strata_rows = [_frame("development", fold, index) for fold in (0, 1, 2) for index in range(10)]
    strata_rows.append(_frame("locked_evaluation", 0, 0))
    candidate_rows = [
        {"file_name": row["file_name"], "fold": row["fold"], "candidate_index": 0}
        for row in strata_rows
    ]
    with pytest.raises(ValueError, match="expected 547"):
        select_all_development(pd.DataFrame(strata_rows), pd.DataFrame(candidate_rows))
