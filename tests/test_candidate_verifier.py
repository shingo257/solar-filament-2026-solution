import numpy as np
import pandas as pd

from solar_filament.candidate_verifier import (
    candidate_utility,
    fit_logistic_verifier,
    nested_oof_candidate_predictions,
    proposal_features,
)


def test_proposal_features_capture_consensus_and_shape():
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:15, 4:24] = True
    features = proposal_features(mask, score=0.8, source_count=2)
    assert features["detector_score"] == 0.8
    assert features["source_count"] == 2.0
    assert features["log_aspect"] > 0


def test_logistic_verifier_separates_simple_data():
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    targets = np.array([0, 0, 1, 1])
    model = fit_logistic_verifier(features, targets, l2=0.1)
    probabilities = model.predict_proba(features)
    assert probabilities[:2].max() < probabilities[2:].min()


def test_candidate_utility_uses_measured_marginal_value_of_accepted_candidates():
    utilities = np.array([0.7, -0.5])
    assert candidate_utility(utilities, [True, False]) == 0.7
    assert candidate_utility(utilities, [False, True]) == -0.5
    assert candidate_utility(utilities, [False, False]) == 0.0


def test_nested_predictions_cover_each_outer_fold_without_nulls():
    rows = []
    for fold in range(3):
        for index in range(20):
            target = int(index >= 10)
            rows.append({
                "fold": fold,
                "target": target,
                "hit_iou": 0.7 if target else 0.0,
                "marginal_utility": 0.7 if target else -0.5,
                "feature": float(index - 10) + fold * 0.01,
            })
    frame = pd.DataFrame(rows)
    result = nested_oof_candidate_predictions(frame, ["feature"])
    assert len(result) == len(frame)
    assert set(result.outer_fold) == {0, 1, 2}
    assert not result.verifier_probability.isna().any()
    assert (result.outer_fold == result.fold).all()
