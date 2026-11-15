"""Unit tests for 1-to-1 cross-detector matching and continuous fusion (v132)."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from solar_filament.pipeline.consensus_fusion import (
    DetectorProposal,
    MatchedCluster,
    fast_apply_own_specs,
    fuse_cluster_proposal_mask,
    fuse_refined_crop_specs,
    match_detector_proposals_1to1,
    rle_to_crop_spec,
)
from solar_filament.pipeline.ownership import apply_ownership, masks_to_rles


def _rle_rect(y0: int, y1: int, x0: int, x1: int, h: int = 64, w: int = 64) -> dict:
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_utils.encode(np.asfortranarray(m))


def test_match_detector_proposals_prevents_transitive_same_detector_chaining() -> None:
    # Two adjacent yolo proposals bridged by one wide crop proposal:
    # Unconstrained union-find in v120 merged all 3 into one cluster; 1-to-1 prevents chaining!
    p_yolo_a = DetectorProposal("yolo", 0.70, _rle_rect(10, 30, 10, 30), 0)
    p_yolo_b = DetectorProposal("yolo", 0.60, _rle_rect(10, 30, 26, 46), 1)
    p_crop_bridge = DetectorProposal("crop", 0.50, _rle_rect(10, 30, 12, 44), 2)

    clusters = match_detector_proposals_1to1(
        [p_yolo_a, p_yolo_b, p_crop_bridge], iou_threshold=0.40
    )
    assert len(clusters) == 2
    for c in clusters:
        assert len(c.members) <= 2
        assert len(set(c.members.keys())) == len(c.members)


def test_continuous_score_and_soft_vote_mask_fusion() -> None:
    p1 = DetectorProposal("yolo", 0.42, _rle_rect(10, 30, 10, 30), 0)
    p2 = DetectorProposal("yolo26", 0.24, _rle_rect(10, 30, 12, 32), 1)
    clusters = match_detector_proposals_1to1([p1, p2], iou_threshold=0.45)
    assert len(clusters) == 1
    c = clusters[0]
    # Hard cons2of3_c26 drops (0.42, 0.24, 0.0) because count_above(0.26) == 1,
    # whereas continuous mean3 >= 0.20 keeps it (0.66 / 3 = 0.22)!
    assert c.count_above(0.26) == 1
    mean3 = c.weighted_score({"yolo": 1 / 3, "yolo26": 1 / 3, "crop": 1 / 3})
    assert abs(mean3 - 0.22) < 1e-6

    m_max = fuse_cluster_proposal_mask(c, mode="max_conf")
    m_soft = fuse_cluster_proposal_mask(c, mode="soft_vote", vote_threshold=0.45)
    assert m_max.shape == (64, 64)
    assert m_soft.shape == (64, 64)
    assert m_soft[20, 20]


def test_fast_apply_own_specs_exact_parity_with_apply_ownership() -> None:
    rng = np.random.default_rng(3407)
    assert fast_apply_own_specs([]) == []

    for trial in range(25):
        n = int(rng.integers(1, 14))
        masks: list[np.ndarray] = []
        for idx in range(n):
            m = np.zeros((128, 128), dtype=bool)
            if trial % 5 == 0 and idx > 0:
                # Include nested / identical-area edge cases
                m = masks[0].copy()
            else:
                y0 = int(rng.integers(0, 96))
                x0 = int(rng.integers(0, 96))
                hh = int(rng.integers(6, 30))
                ww = int(rng.integers(6, 30))
                m[y0 : y0 + hh, x0 : x0 + ww] = rng.random((hh, ww)) > 0.35
            masks.append(m)

        scores = rng.random(n)
        ref_kept = apply_ownership(
            masks,
            scores,
            rule="drop_contained_then_area_asc",
            containment_threshold=0.90,
            min_area=20,
            canvas=128,
        )
        ref_rles = masks_to_rles(ref_kept)

        specs = [
            rle_to_crop_spec(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))
            for m in masks
        ]
        fast_rles = fast_apply_own_specs(specs, thr=0.90, min_area=20)
        assert len(fast_rles) == len(ref_rles)
        for a, b in zip(fast_rles, ref_rles):
            assert a["counts"] == b["counts"]
            assert a["size"] == b["size"]


def test_fuse_refined_crop_specs_majority_and_fallback() -> None:
    sp_a = rle_to_crop_spec(_rle_rect(10, 20, 10, 22))
    sp_b = rle_to_crop_spec(_rle_rect(10, 20, 12, 24))
    sp_c = rle_to_crop_spec(_rle_rect(10, 20, 10, 24))
    specs_by_det = {"a": [sp_a], "b": [sp_b], "c": [sp_c]}

    # 2-member cluster: majority_3plus preserves best_member without 2-way tip clipping
    cl2 = MatchedCluster(
        members={
            "a": DetectorProposal("a", 0.70, _rle_rect(10, 20, 10, 22), 0),
            "b": DetectorProposal("b", 0.50, _rle_rect(10, 20, 12, 24), 0),
        }
    )
    out2 = fuse_refined_crop_specs(cl2, specs_by_det, mode="majority_3plus")
    assert out2[5] == sp_a[5]

    # 3-member cluster: 2-of-3 pixel majority covers [10:20, 10:24] (10 * 14 = 140 px)
    cl3 = MatchedCluster(
        members={
            "a": DetectorProposal("a", 0.70, _rle_rect(10, 20, 10, 22), 0),
            "b": DetectorProposal("b", 0.60, _rle_rect(10, 20, 12, 24), 0),
            "c": DetectorProposal("c", 0.50, _rle_rect(10, 20, 10, 24), 0),
        }
    )
    out3 = fuse_refined_crop_specs(cl3, specs_by_det, mode="majority_3plus")
    assert out3[0] == 10 and out3[1] == 20 and out3[2] == 10 and out3[3] == 24
    assert out3[5] == 140.0

    # 4-member cluster: confidence-weighted majority breaks 2-vs-2 ties
    sp_d = rle_to_crop_spec(_rle_rect(10, 20, 10, 20))
    specs4 = {**specs_by_det, "d": [sp_d]}
    cl4 = MatchedCluster(
        members={
            **cl3.members,
            "d": DetectorProposal("d", 0.40, _rle_rect(10, 20, 10, 20), 0),
        }
    )
    out4 = fuse_refined_crop_specs(cl4, specs4, mode="majority_3plus")
    assert out4[5] > 0.0
    out_soft = fuse_refined_crop_specs(cl3, specs_by_det, mode="soft_w50")
    assert out_soft[5] > 0.0
