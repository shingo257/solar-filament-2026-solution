"""1-to-1 Multi-detector bipartite matching and continuous score/mask fusion (Idea 8, v132).

Fixes three structural issues in ``run_fold_oof_eval_consensus.py`` (v120/v121c/v124c):
1. Replaces unconstrained transitive union-find with greedy descending-IoU 1-to-1 bipartite
   clustering (at most 1 proposal per detector in every cluster), preventing adjacent-filament
   chaining across detectors.
2. Replaces hard step-function vote counting (``count(s >= 0.26) >= 2``, which discards
   ``(0.45, 0.24, 0)``, ``(0.24, 0.24, 0.24)``, and solo ``(0.75, 0, 0)``) with continuous
   calibrated cluster scoring ``sum_d w_d * s_d`` (missing detector contributes ``0.0``).
3. Supports score-weighted multi-detector proposal mask fusion before ``MediSegEdgeAttUNet``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_utils


@dataclass(frozen=True)
class DetectorProposal:
    det_name: str
    conf: float
    raw_rle: dict
    prop_idx: int


@dataclass
class MatchedCluster:
    members: dict[str, DetectorProposal]  # det_name -> DetectorProposal (strictly <=1 per det)

    def score_of(self, det_name: str) -> float:
        p = self.members.get(det_name)
        return float(p.conf) if p is not None else 0.0

    def weighted_score(self, weights: dict[str, float]) -> float:
        return float(sum(w * self.score_of(d) for d, w in weights.items()))

    def max_conf(self) -> float:
        return max((float(p.conf) for p in self.members.values()), default=0.0)

    def count_above(self, thr: float) -> int:
        return sum(1 for p in self.members.values() if float(p.conf) >= thr)

    def best_member(self) -> DetectorProposal:
        return max(self.members.values(), key=lambda p: (float(p.conf), p.det_name))


def match_detector_proposals_1to1(
    proposals: list[DetectorProposal],
    *,
    iou_threshold: float = 0.45,
) -> list[MatchedCluster]:
    """Group cross-detector proposals by descending mask IoU with <=1 member per detector."""
    n = len(proposals)
    if n == 0:
        return []
    rles = [p.raw_rle for p in proposals]
    iou_mat = np.asarray(mask_utils.iou(rles, rles, [0] * n), dtype=np.float64)

    parent = list(range(n))
    det_sets = [{proposals[i].det_name} for i in range(n)]

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        di = proposals[i].det_name
        for j in range(i + 1, n):
            if di == proposals[j].det_name:
                continue
            v = float(iou_mat[i, j])
            if v >= iou_threshold:
                pairs.append((v, i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    for _, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj and det_sets[ri].isdisjoint(det_sets[rj]):
            parent[rj] = ri
            det_sets[ri] |= det_sets[rj]

    grouped: dict[int, dict[str, DetectorProposal]] = {}
    for idx in range(n):
        r = find(idx)
        p = proposals[idx]
        grouped.setdefault(r, {})[p.det_name] = p

    clusters = [MatchedCluster(members=m) for m in grouped.values()]
    clusters.sort(key=lambda c: -c.max_conf())
    return clusters


def fuse_cluster_proposal_mask(
    cluster: MatchedCluster,
    *,
    mode: str = "max_conf",
    vote_threshold: float = 0.45,
) -> np.ndarray:
    """Build bool proposal mask for ``MediSegEdgeAttUNet`` from a matched cluster."""
    if mode == "max_conf" or len(cluster.members) == 1:
        best = cluster.best_member()
        return mask_utils.decode(best.raw_rle) != 0
    if mode != "soft_vote":
        raise ValueError(f"Unknown mask fusion mode: {mode}")

    acc: np.ndarray | None = None
    w_sum = 0.0
    for p in cluster.members.values():
        w = max(float(p.conf), 1e-4)
        m = (mask_utils.decode(p.raw_rle) != 0).astype(np.float32)
        acc = (w * m) if acc is None else (acc + w * m)
        w_sum += w
    assert acc is not None
    return (acc / max(w_sum, 1e-6)) >= float(vote_threshold)


CropSpec = tuple[int, int, int, int, np.ndarray, float, tuple[int, int]]


def rle_to_crop_spec(rle: dict) -> CropSpec:
    """Decode a mask RLE once into a compact bounding-box crop tuple (~6 KB vs 4.2 MB)."""
    m = mask_utils.decode(rle) != 0
    h, w = m.shape
    x, y, bw, bh = [round(float(v)) for v in mask_utils.toBbox(rle)]
    if bw <= 0 or bh <= 0:
        return (0, 0, 0, 0, np.zeros((0, 0), dtype=bool), 0.0, (h, w))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    sub = np.ascontiguousarray(m[y0:y1, x0:x1])
    return (y0, y1, x0, x1, sub, float(sub.sum()), (h, w))


def fuse_refined_crop_specs(
    cluster: MatchedCluster,
    specs_by_det: dict[str, list[CropSpec]],
    *,
    mode: str = "best_member",
) -> CropSpec:
    """Fuse already-refined ``CropSpec`` masks across members of a 1-to-1 matched cluster.

    Modes:
    - ``"best_member"``: return the refined ``CropSpec`` of ``cluster.best_member()``.
    - ``"majority_3plus"``: when ``>=3`` non-empty members agree on an instance, take pixelwise
      majority across their refined crops (unweighted ``>=(K+1)//2`` for odd ``K``, confidence-
      weighted ``>=0.5*sum(w)`` for even ``K`` to break 2-vs-2 ties cleanly); fall back to
      ``best_member`` when ``<=2`` members match so thin filament tips are never clipped by
      2-way strict intersection.
    - ``"soft_w50"``: confidence-weighted pixel average ``>= 0.5 * sum(w)`` for ``>=2`` members.
    """
    bm = cluster.best_member()
    bm_spec = specs_by_det[bm.det_name][bm.prop_idx]
    if mode == "best_member" or len(cluster.members) == 1:
        return bm_spec

    member_specs: list[CropSpec] = []
    weights: list[float] = []
    for dname, prop in cluster.members.items():
        sp = specs_by_det[dname][prop.prop_idx]
        if sp[5] > 0.0:
            member_specs.append(sp)
            weights.append(max(float(prop.conf), 1e-4))
    k = len(member_specs)
    if k == 0:
        return bm_spec
    if k == 1 or (mode == "majority_3plus" and k < 3):
        return bm_spec if bm_spec[5] > 0.0 else member_specs[0]

    hw = member_specs[0][6]
    y0 = min(sp[0] for sp in member_specs)
    y1 = max(sp[1] for sp in member_specs)
    x0 = min(sp[2] for sp in member_specs)
    x1 = max(sp[3] for sp in member_specs)
    acc = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)

    if mode == "majority_3plus" and (k % 2 == 1):
        for sy0, sy1, sx0, sx1, sub, _, _ in member_specs:
            acc[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] += sub.astype(np.float32)
        fused = acc >= float((k + 1) // 2)
    elif mode in ("majority_3plus", "soft_w50"):
        w_sum = float(sum(weights))
        for (sy0, sy1, sx0, sx1, sub, _, _), w in zip(member_specs, weights):
            acc[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] += w * sub.astype(np.float32)
        fused = acc >= (0.50 * w_sum)
    else:
        raise ValueError(f"Unknown refined crop fusion mode: {mode}")

    ys, xs = np.nonzero(fused)
    if len(xs) == 0:
        return (0, 0, 0, 0, np.zeros((0, 0), dtype=bool), 0.0, hw)
    ny0, ny1 = int(ys.min()), int(ys.max()) + 1
    nx0, nx1 = int(xs.min()), int(xs.max()) + 1
    sub_f = np.ascontiguousarray(fused[ny0:ny1, nx0:nx1])
    return (y0 + ny0, y0 + ny1, x0 + nx0, x0 + nx1, sub_f, float(len(xs)), hw)


def fast_apply_own_specs(
    specs: list[CropSpec],
    thr: float = 0.90,
    min_area: int = 20,
) -> list[dict]:
    """BBox-cropped ownership matching ``ownership.apply_ownership`` (drop_contained_then_area_asc)."""
    n = len(specs)
    if n == 0:
        return []
    h, w = specs[0][6]
    dropped: set[int] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if i in dropped or j in dropped:
                continue
            y0i, y1i, x0i, x1i, sub_i, ai, _ = specs[i]
            y0j, y1j, x0j, x1j, sub_j, aj, _ = specs[j]
            if ai <= 0.0 and aj <= 0.0:
                continue
            smaller = i if ai <= aj else j
            smaller_area = min(ai, aj)
            iy0, iy1 = max(y0i, y0j), min(y1i, y1j)
            ix0, ix1 = max(x0i, x0j), min(x1i, x1j)
            if iy1 <= iy0 or ix1 <= ix0 or (iy1 - iy0) * (ix1 - ix0) < thr * max(smaller_area, 1.0):
                continue
            inter = int(
                np.logical_and(
                    sub_i[iy0 - y0i : iy1 - y0i, ix0 - x0i : ix1 - x0i],
                    sub_j[iy0 - y0j : iy1 - y0j, ix0 - x0j : ix1 - x0j],
                ).sum()
            )
            if inter / max(smaller_area, 1.0) >= thr:
                dropped.add(smaller)

    keep = [i for i in range(n) if i not in dropped]
    if not keep:
        return []
    occ = np.zeros((h, w), dtype=bool)
    kept_subs: list[tuple[int, int, int, int, np.ndarray]] = []
    for idx in sorted(keep, key=lambda i: specs[i][5]):
        y0, y1, x0, x1, sub, _, _ = specs[idx]
        if sub.size == 0:
            continue
        ex = sub & ~occ[y0:y1, x0:x1]
        if int(ex.sum()) >= min_area:
            occ[y0:y1, x0:x1] |= ex
            kept_subs.append((y0, y1, x0, x1, ex))
    if not kept_subs:
        return []
    canvas = np.zeros((h, w, len(kept_subs)), dtype=np.uint8, order="F")
    for k, (y0, y1, x0, x1, ex) in enumerate(kept_subs):
        canvas[y0:y1, x0:x1, k] = ex
    return mask_utils.encode(canvas)
