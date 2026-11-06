"""audit_annotator_agreement.py のコア判定ロジックを合成マスクで固定する。"""

from __future__ import annotations

import numpy as np

from scratch.audit_annotator_agreement import (
    cluster_coverage,
    mask_iou,
    pq_of,
)


def _rect(x0, y0, x1, y1, shape=(16, 16)):
    m = np.zeros(shape, dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


class TestMaskIou:
    def test_identical(self):
        m = _rect(0, 0, 4, 4)
        assert mask_iou(m, m) == 1.0

    def test_disjoint(self):
        assert mask_iou(_rect(0, 0, 4, 4), _rect(8, 8, 12, 12)) == 0.0

    def test_half(self):
        a = _rect(0, 0, 8, 8)
        b = _rect(4, 0, 12, 8)  # inter 4*8=32, union 8*8*2-32=96 -> 1/3
        assert abs(mask_iou(a, b) - 32 / 96) < 1e-9


class TestPqOf:
    def test_perfect(self):
        gts = [_rect(0, 0, 4, 4), _rect(8, 8, 12, 12)]
        assert pq_of(list(gts), gts) == 1.0

    def test_fp_penalty(self):
        gts = [_rect(0, 0, 8, 8)]
        preds = [_rect(0, 0, 8, 8), _rect(10, 10, 14, 14)]  # tp=1 iou=1, fp=1
        # 1 / (1 + 0.5) = 0.6667
        assert abs(pq_of(preds, gts) - 1 / 1.5) < 1e-9

    def test_fn_penalty(self):
        gts = [_rect(0, 0, 8, 8), _rect(10, 10, 14, 14)]
        preds = [_rect(0, 0, 8, 8)]
        assert abs(pq_of(preds, gts) - 1 / 1.5) < 1e-9

    def test_below_threshold_not_matched(self):
        gt = _rect(0, 0, 8, 8)
        pred = _rect(6, 6, 14, 14)  # inter 4, union 124 -> iou 0.032
        assert pq_of([pred], [gt]) == 0.0

    def test_empty(self):
        assert pq_of([], []) is None


class TestClusterCoverage:
    def test_shared_and_singleton(self):
        shared = _rect(0, 0, 8, 8)
        only_a = _rect(10, 0, 14, 4)
        only_b = _rect(10, 10, 14, 14)
        rec_masks = {"ra": [shared, only_a], "rb": [shared, only_b]}
        cov = cluster_coverage(rec_masks, ["ra", "rb"])
        assert cov[(2, 2)] == 1  # shared cluster marked by both
        assert cov[(2, 1)] == 2  # two singletons

    def test_chain_bridges_records(self):
        # a と c は直接重ならないが b を介して 3 record が繋がる
        a = _rect(0, 0, 8, 8)
        b = _rect(4, 0, 12, 8)   # overlaps a (1/3) and c
        c = _rect(8, 0, 16, 8)
        # b-a IoU 1/3 < 0.5 では繋がらないので、十分な重なりに作り直す
        a = _rect(0, 0, 10, 8)
        b = _rect(5, 0, 15, 8)   # a∩b=40, a∪b=120 -> 1/3 — still <0.5; widen
        a = _rect(0, 0, 12, 8)
        b = _rect(6, 0, 16, 8)   # a∩b=48, union 144 -> 1/3; need >=0.5
        # 0.5 を超えるには inter/union>=0.5 → inter >= 2*overlap差; 使いやすい配置
        a = _rect(0, 0, 10, 8)   # 80px
        b = _rect(4, 0, 14, 8)   # inter 48, union 112 -> 0.429; still <0.5
        a = _rect(0, 0, 11, 8)   # 88
        b = _rect(4, 0, 15, 8)   # inter 56, union 120 -> 0.467; closer
        a = _rect(0, 0, 11, 8)
        b = _rect(3, 0, 14, 8)   # inter 64, union 112 -> 0.571 OK
        c = _rect(8, 0, 16, 8)   # b∩c: b=3..14, c=8..16 inter 48, union 112 -> 0.429 NG
        c = _rect(8, 0, 15, 8)   # inter 48, union 104 -> 0.462 NG
        # c を b に寄せる
        c = _rect(9, 0, 15, 8)   # inter 40, union 88 -> 0.455 NG
        c = _rect(7, 0, 14, 8)   # c ⊆ b -> IoU = 56/88=0.636 OK; a∩c: a=0..11 c=7..14 → inter 32, union 120 →0.27 (橋渡し)
        rec_masks = {"ra": [a], "rb": [b], "rc": [c]}
        cov = cluster_coverage(rec_masks, ["ra", "rb", "rc"])
        assert cov[(3, 3)] == 1

    def test_no_cross_record_match(self):
        rec_masks = {"ra": [_rect(0, 0, 4, 4)], "rb": [_rect(10, 10, 14, 14)]}
        cov = cluster_coverage(rec_masks, ["ra", "rb"])
        assert cov[(2, 1)] == 2
