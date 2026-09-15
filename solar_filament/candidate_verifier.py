"""Leakage-safe lightweight verifier for instance-proposal candidates.

This module produces outer-fold probabilities only.  Promotion still requires
reconstructing masks and measuring official all-fold PQ; candidate-level utility
is a screening objective, not a competition metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class LogisticVerifier:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = (np.asarray(features, dtype=float) - self.mean) / self.scale
        design = np.column_stack([np.ones(len(matrix)), matrix])
        logits = np.clip(design @ self.coefficients, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


def fit_logistic_verifier(
    features: np.ndarray,
    targets: np.ndarray,
    l2: float = 1.0,
) -> LogisticVerifier:
    """Fit a standardised L2 logistic model using SciPy only."""

    matrix = np.asarray(features, dtype=float)
    labels = np.asarray(targets, dtype=float).reshape(-1)
    if matrix.ndim != 2 or len(matrix) != len(labels):
        raise ValueError("features must be 2-D and align with targets")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])
    prior = np.clip(labels.mean(), 1e-4, 1.0 - 1e-4)
    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = np.log(prior / (1.0 - prior))

    def objective(coefficients):
        logits = np.clip(design @ coefficients, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        loss = np.logaddexp(0.0, logits).sum() - labels @ logits
        gradient = design.T @ (probabilities - labels)
        loss += 0.5 * l2 * float(coefficients[1:] @ coefficients[1:])
        gradient[1:] += l2 * coefficients[1:]
        return float(loss), gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"logistic verifier did not converge: {result.message}")
    return LogisticVerifier(mean=mean, scale=scale, coefficients=result.x)


def proposal_features(
    mask: np.ndarray,
    score: float,
    source_count: int,
    semantic_probability: np.ndarray | None = None,
    image: np.ndarray | None = None,
) -> dict[str, float]:
    """Shape, consensus, semantic and contrast features for one proposal."""

    binary = np.asarray(mask, dtype=bool)
    ys, xs = np.where(binary)
    if not len(xs):
        raise ValueError("empty proposals cannot be verified")
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    area = float(len(xs))
    perimeter = float(
        sum(
            cv2.arcLength(contour, True)
            for contour in cv2.findContours(
                binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )[0]
        )
    )
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    canvas_radius = max(0.5 * min(binary.shape), 1.0)
    features = {
        "detector_score": float(score),
        "source_count": float(source_count),
        "log_area": float(np.log1p(area)),
        "bbox_extent": area / max(float(height * width), 1.0),
        "log_aspect": float(np.log(max(width, 1) / max(height, 1))),
        "perimeter_over_sqrt_area": perimeter / max(np.sqrt(area), 1.0),
        "solidity": min(1.0, area / hull_area),
        "center_radius": float(
            np.hypot(center_x - binary.shape[1] / 2, center_y - binary.shape[0] / 2)
            / canvas_radius
        ),
    }
    if semantic_probability is not None:
        probability = np.asarray(semantic_probability, dtype=float)
        values = probability[binary]
        features["semantic_mean"] = float(values.mean())
        features["semantic_q10"] = float(np.quantile(values, 0.10))
    else:
        features["semantic_mean"] = 0.0
        features["semantic_q10"] = 0.0
    if image is not None:
        gray = np.asarray(image, dtype=float)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
        ring = cv2.dilate(binary.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~binary
        features["local_contrast"] = float(gray[ring].mean() - gray[binary].mean()) if ring.any() else 0.0
    else:
        features["local_contrast"] = 0.0
    return features


def candidate_utility(
    marginal_utilities: np.ndarray,
    decisions: np.ndarray,
) -> float:
    """Sum accepted candidates' measured marginal utility (not official PQ)."""

    utilities = np.asarray(marginal_utilities, dtype=float)
    accepted = np.asarray(decisions, dtype=bool)
    if utilities.shape != accepted.shape:
        raise ValueError("marginal_utilities and decisions must align")
    return float(utilities[accepted].sum())


def nested_oof_candidate_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    threshold_grid: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> pd.DataFrame:
    """Nested fold tuning; every returned probability is from an unseen outer fold."""

    required = {"fold", "target", "marginal_utility", *feature_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    outputs = []
    for outer_fold in sorted(frame.fold.unique()):
        training = frame[frame.fold != outer_fold]
        held_out = frame[frame.fold == outer_fold]
        inner_folds = sorted(training.fold.unique())
        best = None
        for l2 in l2_grid:
            inner_probabilities = np.zeros(len(training), dtype=float)
            for inner_fold in inner_folds:
                inner_train = training[training.fold != inner_fold]
                inner_valid = training[training.fold == inner_fold]
                model = fit_logistic_verifier(
                    inner_train[feature_columns].to_numpy(), inner_train.target.to_numpy(), l2=l2
                )
                positions = np.flatnonzero(training.fold.to_numpy() == inner_fold)
                inner_probabilities[positions] = model.predict_proba(
                    inner_valid[feature_columns].to_numpy()
                )
            for threshold in threshold_grid:
                utility = candidate_utility(
                    training.marginal_utility.to_numpy(),
                    inner_probabilities >= threshold,
                )
                candidate = (utility, threshold, l2)
                if best is None or candidate > best:
                    best = candidate
        _, threshold, l2 = best
        final_model = fit_logistic_verifier(
            training[feature_columns].to_numpy(), training.target.to_numpy(), l2=l2
        )
        probability = final_model.predict_proba(held_out[feature_columns].to_numpy())
        result = held_out.copy()
        result["outer_fold"] = outer_fold
        result["verifier_probability"] = probability
        result["verifier_accept"] = probability >= threshold
        result["selected_l2"] = l2
        result["selected_threshold"] = threshold
        outputs.append(result)
    return pd.concat(outputs).sort_index()
