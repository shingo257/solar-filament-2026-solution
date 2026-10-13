"""Tests for v82 soft annotator-mean targets (PLAN §7.2)."""

from __future__ import annotations

import numpy as np
import pytest

from solar_filament.pipeline.soft_target import (
    build_soft_targets,
    cluster_bbox,
    cluster_gt_instances,
    soft_target_crop,
)


def _rect(y0: int, y1: int, x0: int, x1: int, size: int = 96) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_soft_target_values_track_annotator_fraction() -> None:
    rec0 = [_rect(10, 20, 10, 60)]            # filament A
    rec1 = [_rect(11, 21, 11, 61)]            # filament A, shifted
    rec2 = [_rect(40, 50, 10, 30)]            # filament C only
    masks = [rec0, rec1, rec2]
    soft, clusters = build_soft_targets(masks)
    assert len(clusters) == 2
    # A: union of rec0/rec1 regions at value 2/3 (rec2 abstained)
    a = next(s for s, c in zip(soft, clusters) if len(c) == 2)
    assert a.max() == pytest.approx(2 / 3)
    assert (a > 0).sum() == (rec0[0] | rec1[0]).sum()
    assert a[12, 12] == pytest.approx(2 / 3)   # both drew here
    assert a[10, 10] == pytest.approx(1 / 3)   # only rec0
    # C: single-annotator filament → 1/3
    c = next(s for s, c in zip(soft, clusters) if len(c) == 1)
    assert c.max() == pytest.approx(1 / 3)
    assert c[45, 15] == pytest.approx(1 / 3)


def test_three_way_cluster_gradients() -> None:
    base = _rect(10, 30, 10, 30)
    wider = _rect(10, 32, 10, 32)   # covers base + margin
    widest = _rect(8, 34, 8, 34)    # covers both
    soft, clusters = build_soft_targets([[base], [wider], [widest]])
    assert len(clusters) == 1
    s = soft[0]
    assert s[20, 20] == pytest.approx(1.0)          # all three drew
    assert s[31, 31] == pytest.approx(2 / 3)        # wider + widest
    assert s[33, 33] == pytest.approx(1 / 3)        # widest only


def test_same_record_instances_never_merge() -> None:
    a1 = _rect(10, 30, 10, 30)      # rec0, filament 1
    a2 = _rect(5, 25, 3, 23)        # rec0, filament 2 (IoU(a1,a2)=0.32 but same record)
    b = _rect(10, 30, 12, 32)       # rec1: IoU(a1,b)=0.82 → merge; IoU(a2,b)=0.26 → no
    clusters = cluster_gt_instances([[a1, a2], [b]], iou_threshold=0.30)
    flat = sorted(sorted(c) for c in clusters)
    assert flat == [[(0, 0), (1, 0)], [(0, 1)]]


def test_transitive_clustering_across_records() -> None:
    a = _rect(10, 30, 10, 30)
    b = _rect(25, 45, 10, 30)       # IoU(a,b) = 5/35 = 0.14 < 0.3
    c = _rect(20, 40, 10, 30)       # overlaps both a and b strongly
    clusters = cluster_gt_instances([[a], [b], [c]], iou_threshold=0.30)
    assert len(clusters) == 1       # a–c and b–c links chain a and b together


def test_cross_record_bridge_does_not_merge_same_record_instances() -> None:
    left = _rect(10, 30, 10, 30)
    right = _rect(10, 30, 30, 50)
    bridge = _rect(10, 30, 15, 45)  # IoU > 0.3 with each distinct instance
    clusters = cluster_gt_instances([[left, right], [bridge]])
    assert len(clusters) == 2
    assert sorted(member for cluster in clusters for member in cluster) == [
        (0, 0), (0, 1), (1, 0),
    ]
    assert all(len({r for r, _ in cluster}) == len(cluster) for cluster in clusters)


def test_same_record_never_co_cluster_via_transitive_link() -> None:
    a = _rect(10, 30, 10, 40)       # rec0 filament 1
    b = _rect(15, 35, 25, 55)       # rec0 filament 2 (IoU(a,b)=0.23, same record anyway)
    c = _rect(12, 32, 15, 50)       # rec1: overlaps both a and b at IoU>0.3
    clusters = cluster_gt_instances([[a, b], [c]], iou_threshold=0.30)
    # a–c joins, but b must not join the a–c component (would put two rec0
    # instances in one cluster). {a,c} and {b} stay separate.
    assert len(clusters) == 2
    for cluster in clusters:
        recs = [r for r, _ in cluster]
        assert len(recs) == len(set(recs))


def test_first_record_empty_uses_available_shape() -> None:
    rec = _rect(10, 20, 10, 30)
    soft, clusters = build_soft_targets([[], [rec], [rec]])
    assert len(clusters) == 1
    assert soft[0].shape == rec.shape
    assert soft[0].max() == pytest.approx(2 / 3)   # 2 of 3 records drew


def test_crop_matches_full_frame() -> None:
    rec0 = [_rect(10, 20, 10, 60), _rect(50, 70, 50, 80)]
    rec1 = [_rect(11, 21, 11, 61), _rect(50, 71, 51, 81)]
    masks = [rec0, rec1]
    soft, clusters = build_soft_targets(masks)
    for s, cluster in zip(soft, clusters):
        box = cluster_bbox(masks, cluster)
        x0, y0, x1, y1 = box
        crop = soft_target_crop(masks, cluster, box)
        np.testing.assert_allclose(s[y0:y1, x0:x1], crop)
        outside = s.copy()
        outside[y0:y1, x0:x1] = 0
        assert outside.sum() == 0
        assert s.max() == crop.max()


def test_cluster_bbox_and_empty() -> None:
    masks = [[_rect(10, 20, 10, 30)], [_rect(11, 22, 12, 32)]]
    clusters = cluster_gt_instances(masks)
    assert len(clusters) == 1
    assert cluster_bbox(masks, clusters[0]) == (10, 10, 32, 22)
    assert build_soft_targets([[], []]) == ([], [])
