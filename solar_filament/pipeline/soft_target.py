"""Soft annotator-mean targets for v82 refiner training (PLAN §7.2).

One physical filament appears as one instance in *each* annotator record, with
slightly different boundaries. Record-level hard targets teach the refiner to
chase individual annotator noise; the cluster-level soft target

    S_k(x) = (1/R) Σ_r union_i M_{k,r,i}(x)

is the probability that a random annotator marks pixel ``x`` (R = number of
annotator records for the image; records that did not draw the cluster count
zero). Predicting the distribution mean recovers the 0.5-IoU cliff on
near-miss candidates — the mechanism the near-miss analysis points at.

Clustering unions instances whose pairwise IoU exceeds ``iou_threshold``,
but never merges two instances from the *same* record: if one annotator drew
two overlapping masks they considered them distinct filaments.

Full-frame soft masks are expensive (float32, 16 MB each at 2048²), so the
training dataset should call :func:`soft_target_crop` with the cluster's crop
box and build the target only inside it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_CLUSTER_IOU = 0.30


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    return float(inter) / float(a.sum() + b.sum() - inter)


def cluster_gt_instances(
    masks_per_record: Sequence[Sequence[np.ndarray]],
    iou_threshold: float = DEFAULT_CLUSTER_IOU,
) -> list[list[tuple[int, int]]]:
    """Union-find clusters of GT instances across annotator records.

    ``masks_per_record[r]`` is the instance list of record ``r``. Returns a
    list of clusters; each cluster is a list of ``(record, instance)`` pairs.
    Instances sharing a record are never directly merged.
    """
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    instances = [
        (r, i)
        for r, masks in enumerate(masks_per_record)
        for i in range(len(masks))
    ]
    for inst in instances:
        parent[inst] = inst
    # 各 component が含む record 集合 (root で引く)。同一 record の2インスタンスが
    # 異なる成分経由で合流する transitive merge を防ぐ。
    records: dict[tuple[int, int], set[int]] = {
        inst: {inst[0]} for inst in instances
    }
    for pos, (r, i) in enumerate(instances):
        for s, j in instances[pos + 1 :]:
            if s == r:
                continue
            if _iou(masks_per_record[r][i], masks_per_record[s][j]) > iou_threshold:
                ra, rb = find((r, i)), find((s, j))
                if ra != rb and not records[ra] & records[rb]:
                    parent[rb] = ra
                    records[ra] |= records[rb]
    clusters: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for inst in instances:
        clusters.setdefault(find(inst), []).append(inst)
    return list(clusters.values())


def cluster_bbox(
    masks_per_record: Sequence[Sequence[np.ndarray]],
    cluster: Sequence[tuple[int, int]],
) -> tuple[int, int, int, int]:
    """Union bounding box ``(x0, y0, x1, y1)`` of a cluster's instances."""
    x0 = y0 = np.iinfo(np.int64).max
    x1 = y1 = 0
    for r, i in cluster:
        ys, xs = np.nonzero(masks_per_record[r][i])
        x0 = min(x0, int(xs.min()))
        x1 = max(x1, int(xs.max()) + 1)
        y0 = min(y0, int(ys.min()))
        y1 = max(y1, int(ys.max()) + 1)
    return x0, y0, x1, y1


def soft_target_crop(
    masks_per_record: Sequence[Sequence[np.ndarray]],
    cluster: Sequence[tuple[int, int]],
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Soft target ``S_k`` restricted to ``box`` = (x0, y0, x1, y1), float32.

    A record contributing several instances to the cluster contributes their
    union once (one annotator, one opinion).
    """
    x0, y0, x1, y1 = box
    acc = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    by_record: dict[int, list[int]] = {}
    for r, i in cluster:
        by_record.setdefault(r, []).append(i)
    for r, indices in by_record.items():
        union = np.logical_or.reduce(
            [masks_per_record[r][i][y0:y1, x0:x1] for i in indices]
        )
        acc += union.astype(np.float32)
    return acc / max(len(masks_per_record), 1)


def build_soft_targets(
    masks_per_record: Sequence[Sequence[np.ndarray]],
    iou_threshold: float = DEFAULT_CLUSTER_IOU,
) -> tuple[list[np.ndarray], list[list[tuple[int, int]]]]:
    """Full-frame soft targets + clusters. For tests and small canvases; the
    training path should prefer :func:`soft_target_crop` to avoid holding many
    float32 canvases."""
    clusters = cluster_gt_instances(masks_per_record, iou_threshold)
    if not clusters:
        return [], []
    height, width = next(
        inst.shape for record in masks_per_record for inst in record
    )
    soft = []
    for cluster in clusters:
        target = np.zeros((height, width), dtype=np.float32)
        box = cluster_bbox(masks_per_record, cluster)
        x0, y0, x1, y1 = box
        target[y0:y1, x0:x1] = soft_target_crop(masks_per_record, cluster, box)
        soft.append(target)
    return soft, clusters
