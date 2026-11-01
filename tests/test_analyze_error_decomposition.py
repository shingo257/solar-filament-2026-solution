"""Behavioral tests for scratch/analyze_error_decomposition.py.

Covers the two pure aggregation helpers that classify cluster stage-loss
attribution and per-prediction marginal value. The module pulls in heavy
inference dependencies at import time (torch, ultralytics via
run_tile_infer_probe), so the tests skip on slim environments.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")
analysis = pytest.importorskip("scratch.analyze_error_decomposition")


# ---------------------------------------------------------------- stage_lost

def test_stage_lost_survived():
    assert analysis.stage_lost(True, True, True) == "survived_boundary"


def test_stage_lost_ownership_when_refiner_rescues():
    # Proposal coverage is below 0.5, refinement rescues it above 0.5,
    # then the disjoint ownership stage drops it -> ownership_lost, not
    # proposal_blind. Coverage is non-monotonic across stages.
    assert analysis.stage_lost(False, True, False) == "ownership_lost"


def test_stage_lost_refiner():
    assert analysis.stage_lost(True, False, False) == "refiner_lost"


def test_stage_lost_proposal_blind():
    assert analysis.stage_lost(False, False, False) == "proposal_blind"


def test_stage_lost_rescued_and_kept():
    assert analysis.stage_lost(False, True, True) == "survived_boundary"


# ------------------------------------------------------- marginal_pred_frame

def _tp(file_name="img", record_id="r1", pred_idx=0, gt_local=0, iou=0.8,
        conf=0.5):
    return {"file_name": file_name, "record_id": record_id,
            "pred_idx": pred_idx, "gt_local": gt_local, "conf": conf,
            "cluster_k": 1, "iou": iou}


def _fp(file_name="img", record_id="r2", pred_idx=0, conf=0.3, cls="free"):
    return {"file_name": file_name, "record_id": record_id,
            "pred_idx": pred_idx, "conf": conf, "cls": cls,
            "best_any_iou": 0.1, "pred_area": 100.0}


def test_marginal_all_fp_pred_keeps_conf():
    # A prediction that is FP in every record has no TP row; confidence
    # must still be preserved from the FP side rather than left missing.
    tp_df = pd.DataFrame([_tp(pred_idx=0)])
    fp_df = pd.DataFrame([_fp(pred_idx=1, conf=0.33),
                          _fp(pred_idx=1, conf=0.33, record_id="r3")])
    pred = analysis.marginal_pred_frame(tp_df, fp_df)
    row = pred[pred["pred_idx"] == 1].iloc[0]
    assert row["conf"] == pytest.approx(0.33)
    assert row["n_tp"] == 0
    assert row["n_fp"] == 2


def test_marginal_unique_cover_creates_fn():
    # Pred 0 uniquely covers gt0 (no other pred TP-matches it) and is FP
    # in the other record. Removing it drops the TP, drops the FP, and
    # creates one FN -> denom = 1 + 0.5 - 0.5 = 1.0, ratio = iou/denom.
    tp_df = pd.DataFrame([_tp(pred_idx=0, gt_local=0, iou=0.8)])
    fp_df = pd.DataFrame([_fp(pred_idx=0)])
    pred = analysis.marginal_pred_frame(tp_df, fp_df)
    row = pred.iloc[0]
    assert row["n_unique"] == 1
    assert row["marginal_denom"] == pytest.approx(1.0)
    assert row["ratio"] == pytest.approx(0.8)


def test_marginal_shared_cover_not_unique():
    # Two preds TP-match the same GT: neither uniquely covers it, so no
    # new FN is created on removal (n_unique = 0).
    tp_df = pd.DataFrame([_tp(pred_idx=0, gt_local=0, iou=0.7),
                          _tp(pred_idx=1, gt_local=0, iou=0.6)])
    fp_df = pd.DataFrame()
    pred = analysis.marginal_pred_frame(tp_df, fp_df)
    assert pred["n_unique"].tolist() == [0, 0]
    row0 = pred[pred["pred_idx"] == 0].iloc[0]
    assert row0["marginal_denom"] == pytest.approx(1.0)
    assert row0["ratio"] == pytest.approx(0.7)
