import numpy as np
import pandas as pd
import pytest

from scratch.run_v42_verifier_official_eval import (
    boolean_values,
    cluster_bootstrap_ratio,
    decisions_by_observation,
)


def test_boolean_values_parses_false_strings_as_false():
    values = boolean_values(pd.Series(["True", "False", "1", "0"]))
    np.testing.assert_array_equal(values, [True, False, True, False])


def test_decisions_by_observation_requires_contiguous_candidate_indices():
    frame = pd.DataFrame({
        "file_name": ["a.png", "a.png"],
        "candidate_index": [0, 2],
        "verifier_accept": [True, False],
        "fold": [0, 0],
        "outer_fold": [0, 0],
    })
    with pytest.raises(ValueError, match="not contiguous"):
        decisions_by_observation(frame)


def test_cluster_bootstrap_ratio_preserves_constant_ratio():
    result = cluster_bootstrap_ratio(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        repeats=100,
    )
    assert result == {"estimate": 0.5, "ci_low": 0.5, "ci_high": 0.5}
