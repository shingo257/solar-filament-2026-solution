"""Component tree over multi-threshold connected components (v80 candidate source).

A dense semantic probability map ``p`` is binarised at several ascending
thresholds. Because ``p >= t_hi`` implies ``p >= t_lo`` for ``t_hi > t_lo``,
every connected component at a higher threshold is nested inside exactly one
component at the next lower threshold, so the components form a tree
("max-tree" over the level sets). Two nodes either nest or are pixel-disjoint —
never partially overlap — which makes any antichain (no ancestor/descendant
pair) a valid set of non-overlapping instance candidates.

This is the candidate-expansion mechanism for v80: YOLO proposals cover the
confident filaments while the tree surfaces components the detector never
proposed (the "blind FN" that dominate the residual error).

Memory note: node masks are stored as a bounding box plus a local boolean
patch, never as full-canvas arrays (a naive bool-per-node layout costs ~100 GB
across the OOF set — the v64 lesson). Use :meth:`ComponentTree.mask_of` or
:func:`iter_full_masks` only at point of use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

# 0.10 … 0.80 inclusive, step 0.05. Low enough to surface weak regions,
# high enough that the deepest level still corresponds to "confident core".
DEFAULT_LEVELS: tuple[float, ...] = tuple(round(0.10 + 0.05 * i, 2) for i in range(15))
DEFAULT_MIN_AREA = 20
# A node whose mask is ≥ this fraction of its parent's area adds no new
# candidate geometry — it is the same region persisting across the threshold
# boundary. Dropped to keep the candidate count down.
DEFAULT_MERGE_RATIO = 0.98


@dataclass
class TreeNode:
    """One connected component at one threshold level.

    ``mask`` is a local boolean patch covering ``bbox`` =
    ``(x0, y0, x1, y1)``; ``area`` counts foreground pixels.
    """

    index: int
    level_index: int
    threshold: float
    parent: int
    children: tuple[int, ...]
    area: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


class ComponentTree:
    """Nodes + level definition for one image."""

    def __init__(self, nodes: list[TreeNode], levels: Sequence[float], shape: tuple[int, int]):
        self.nodes = nodes
        self.levels = tuple(float(v) for v in levels)
        self.shape = shape

    def __len__(self) -> int:
        return len(self.nodes)

    def mask_of(self, node_index: int) -> np.ndarray:
        """Materialise the node on the full canvas."""
        node = self.nodes[node_index]
        x0, y0, x1, y1 = node.bbox
        full = np.zeros(self.shape, dtype=bool)
        full[y0:y1, x0:x1] = node.mask
        return full

    def ancestors(self, node_index: int) -> list[int]:
        """Parent chain from ``node_index`` up to (excluding) the root."""
        chain: list[int] = []
        cursor = self.nodes[node_index].parent
        while cursor >= 0:
            chain.append(cursor)
            cursor = self.nodes[cursor].parent
        return chain


def _components(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected components (8-connectivity) → (label map, stats)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    return labels, stats[:count]


def build_component_tree(
    probability: np.ndarray,
    levels: Sequence[float] = DEFAULT_LEVELS,
    min_area: int = DEFAULT_MIN_AREA,
    merge_ratio: float = DEFAULT_MERGE_RATIO,
) -> ComponentTree:
    """Build the max-tree of connected components over ``levels``.

    ``levels`` must be ascending thresholds. Components smaller than
    ``min_area`` are skipped. A component whose area is at least
    ``merge_ratio`` of its containing parent node's area is dropped as a
    duplicate (its deeper children re-attach to the parent representative).

    Because threshold supersets always exist at lower levels, every node at
    level ``k > 0`` has a parent; nodes at level 0 are roots (``parent = -1``).
    """
    levels = tuple(float(v) for v in levels)
    if not levels or any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("levels must be strictly ascending")
    probability = np.asarray(probability, dtype=np.float32)

    nodes: list[TreeNode] = []
    pending_children: list[list[int]] = []
    prev_labels: np.ndarray | None = None
    # raw CC label at the previous level -> surviving representative node index
    prev_representative: dict[int, int] = {}

    for level_index, threshold in enumerate(levels):
        labels, stats = _components(probability >= threshold)
        cur_representative: dict[int, int] = {}
        for label in range(1, stats.shape[0]):
            x0, y0, w, h, area = (int(v) for v in stats[label])
            if area < min_area:
                continue
            local = labels[y0 : y0 + h, x0 : x0 + w] == label

            parent = -1
            if level_index > 0 and prev_labels is not None:
                ys, xs = np.nonzero(local)
                raw_parent = int(prev_labels[y0 + ys[0], x0 + xs[0]])
                parent = prev_representative.get(raw_parent, -1)
                if parent < 0:
                    raise RuntimeError(
                        f"level {level_index} component has no surviving parent "
                        f"(raw label {raw_parent}); tree invariant broken"
                    )
                # node ⊆ parent by threshold nesting, so IoU = area / parent.area
                if area >= merge_ratio * nodes[parent].area:
                    # duplicate geometry: adopt the parent's representative
                    cur_representative[label] = parent
                    continue

            node = TreeNode(
                index=len(nodes),
                level_index=level_index,
                threshold=threshold,
                parent=parent,
                children=(),
                area=area,
                bbox=(x0, y0, x0 + w, y0 + h),
                mask=local,
            )
            nodes.append(node)
            pending_children.append([])
            if parent >= 0:
                pending_children[parent].append(node.index)
            cur_representative[label] = node.index

        prev_labels = labels
        prev_representative = cur_representative

    frozen = [
        TreeNode(
            index=n.index,
            level_index=n.level_index,
            threshold=n.threshold,
            parent=n.parent,
            children=tuple(sorted(pending_children[n.index])),
            area=n.area,
            bbox=n.bbox,
            mask=n.mask,
        )
        for n in nodes
    ]
    return ComponentTree(frozen, levels, probability.shape)


def is_ancestor(tree: ComponentTree, ancestor: int, node: int) -> bool:
    """True when ``ancestor`` is a strict ancestor of ``node``."""
    cursor = tree.nodes[node].parent
    while cursor >= 0:
        if cursor == ancestor:
            return True
        cursor = tree.nodes[cursor].parent
    return False


def select_antichain_forest(
    children: Sequence[Sequence[int]],
    roots: Sequence[int],
    utilities: Sequence[float],
    tau: float,
) -> list[int]:
    """Exact maximum of ``Σ (u_i − tau)`` over all antichains of a forest.

    Selecting a node yields ``u_i − tau`` and blocks its whole subtree;
    skipping it leaves each child's subtree to be optimised independently.
    That is a tree DP — ``f(i) = max(u_i − tau, Σ_children f(c))`` — which a
    descending-utility greedy does *not* reproduce (a 0.6 root vs two 0.5
    children: greedy keeps the root, the optimum keeps both children).

    ``children[i]`` lists the children of node ``i``; every child must have a
    larger index than its parent (guaranteed when nodes were emitted level by
    level), so reversed index order is a valid post-order. Utilities of
    ``-inf`` mark unusable nodes: never selected, children unaffected.
    """
    scores = np.asarray(utilities, dtype=np.float64)
    count = len(children)
    if len(scores) != count:
        raise ValueError("utilities must align with nodes")
    for index, kids in enumerate(children):
        if any(child <= index for child in kids):
            raise ValueError("children must have larger indices than parents")

    best = np.zeros(count, dtype=np.float64)
    took_self = np.zeros(count, dtype=bool)
    for index in range(count - 1, -1, -1):
        gain_select = scores[index] - tau if scores[index] > tau else -np.inf
        gain_skip = float(sum(best[child] for child in children[index]))
        if gain_select > gain_skip:
            best[index] = gain_select
            took_self[index] = True
        else:
            best[index] = gain_skip

    selected: list[int] = []

    def visit(index: int) -> None:
        if took_self[index]:
            selected.append(index)
            return
        for child in children[index]:
            visit(child)

    for index in roots:
        visit(index)
    return sorted(selected)


def select_antichain(
    tree: ComponentTree,
    utilities: Sequence[float],
    tau: float,
) -> list[int]:
    """Exact antichain selection over a built :class:`ComponentTree`."""
    children = [list(node.children) for node in tree.nodes]
    roots = [node.index for node in tree.nodes if node.parent < 0]
    return select_antichain_forest(children, roots, utilities, tau)
