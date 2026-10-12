"""Tests for the v80 component-tree candidate source.

Synthetic probability maps with nested blobs exercise the two structural
guarantees the scorer relies on: every node nests or is disjoint (the tree
invariant), and ``select_antichain`` keeps the best node per root-to-leaf
path.
"""

from __future__ import annotations

import numpy as np
import pytest

from solar_filament.pipeline.component_tree import (
    ComponentTree,
    build_component_tree,
    is_ancestor,
    select_antichain,
)


def _blob_field(shapes: list[tuple[int, int, int, int, float]], size: int = 256) -> np.ndarray:
    """Max-combine rectangular blobs: (y0, y1, x0, x1, height)."""
    field = np.zeros((size, size), dtype=np.float32)
    for y0, y1, x0, x1, value in shapes:
        field[y0:y1, x0:x1] = np.maximum(field[y0:y1, x0:x1], value)
    return field


def _masks_disjoint(tree: ComponentTree, i: int, j: int) -> bool:
    a = tree.mask_of(i)
    b = tree.mask_of(j)
    if not np.logical_and(a, b).any():
        return True
    return bool((a <= b).all() or (b <= a).all())


def test_nested_blobs_form_chain() -> None:
    # A bright core inside a dim halo: thresholding high isolates the core.
    prob = _blob_field([(40, 120, 40, 120, 0.30), (60, 90, 60, 90, 0.70)])
    tree = build_component_tree(prob, levels=(0.10, 0.30, 0.50, 0.70), min_area=20)
    assert len(tree) >= 2
    levels = {n.index: n.level_index for n in tree.nodes}
    deepest = max(levels, key=levels.get)
    chain = tree.ancestors(deepest)
    assert chain, "deep core must have ancestors"
    root = chain[-1]
    assert tree.nodes[root].level_index == 0
    # containment: child mask ⊆ parent mask
    child_mask = tree.mask_of(deepest)
    parent_mask = tree.mask_of(tree.nodes[deepest].parent)
    assert (child_mask <= parent_mask).all()


def test_separate_regions_are_siblings_not_nested() -> None:
    prob = _blob_field([(20, 80, 20, 80, 0.50), (150, 210, 150, 210, 0.50)])
    tree = build_component_tree(prob, levels=(0.10, 0.30), min_area=20)
    roots = [n.index for n in tree.nodes if n.parent < 0]
    assert len(roots) == 2
    assert not np.logical_and(tree.mask_of(roots[0]), tree.mask_of(roots[1])).any()


def test_nesting_invariant_holds_for_random_fields() -> None:
    rng = np.random.default_rng(3407)
    for _ in range(10):
        prob = rng.random((192, 192), dtype=np.float32) ** 3
        tree = build_component_tree(
            prob, levels=(0.05, 0.15, 0.30, 0.50), min_area=15
        )
        for i in range(len(tree)):
            for j in range(i + 1, len(tree)):
                assert _masks_disjoint(tree, i, j), f"nodes {i},{j} partially overlap"


def test_merge_ratio_drops_persistent_region() -> None:
    # One blob far above all thresholds → only one node survives dedup.
    prob = _blob_field([(30, 100, 30, 100, 0.95)])
    tree = build_component_tree(
        prob, levels=(0.10, 0.30, 0.50, 0.70), min_area=20, merge_ratio=0.98
    )
    assert len(tree) == 1
    assert tree.nodes[0].level_index == 0


def test_min_area_filters_small_components() -> None:
    prob = _blob_field([(10, 14, 10, 13, 0.90), (40, 120, 40, 120, 0.90)])
    tree = build_component_tree(prob, levels=(0.10,), min_area=20)
    assert len(tree) == 1
    assert tree.nodes[0].area == 80 * 80


def test_antichain_selects_child_sum_when_it_beats_parent() -> None:
    """Greedy "best per path" fails here: root 0.6 vs children 0.5 + 0.5.
    The DP must take both children (sum 1.0 > 0.6)."""
    prob = _blob_field([(20, 140, 20, 140, 0.35), (40, 70, 40, 70, 0.75),
                        (90, 120, 90, 120, 0.75)])
    tree = build_component_tree(prob, levels=(0.10, 0.30, 0.50, 0.70), min_area=20)
    # halo root + two distinct cores (the two bright blobs are disjoint)
    roots = [n for n in tree.nodes if n.parent < 0]
    assert len(roots) == 1 and len(roots[0].children) == 2
    root = roots[0].index
    utilities = np.zeros(len(tree))
    utilities[root] = 0.60
    for child in roots[0].children:
        utilities[child] = 0.50
    keep = select_antichain(tree, utilities, tau=0.20)
    assert set(keep) == set(roots[0].children)


def test_antichain_keeps_best_per_path() -> None:
    prob = _blob_field([(20, 140, 20, 140, 0.35), (50, 100, 50, 100, 0.75)])
    tree = build_component_tree(prob, levels=(0.10, 0.30, 0.50, 0.70), min_area=20)
    utilities = np.array([0.4 + 0.1 * n.level_index for n in tree.nodes])
    keep = select_antichain(tree, utilities, tau=0.3)
    # selected set must be an antichain
    for i in keep:
        for j in keep:
            if i != j:
                assert not is_ancestor(tree, i, j)
                assert not is_ancestor(tree, j, i)


def test_antichain_descendant_beats_ancestor_when_higher_utility() -> None:
    prob = _blob_field([(20, 140, 20, 140, 0.35), (50, 100, 50, 100, 0.75)])
    tree = build_component_tree(prob, levels=(0.10, 0.30, 0.50, 0.70), min_area=20)
    assert len(tree) == 2  # halo root + core child (dedup leaves these two)
    root, core = (0, 1) if tree.nodes[0].level_index == 0 else (1, 0)
    assert tree.nodes[core].parent == root
    utilities = np.zeros(len(tree))
    utilities[root] = 0.20  # below tau → never selected
    utilities[core] = 0.60
    keep = select_antichain(tree, utilities, tau=0.25)
    assert keep == [core]


def _brute_force_best(tree: ComponentTree, utilities: np.ndarray, tau: float) -> float:
    """Exhaustive max of Σ(u − tau) over antichains. For small trees only."""
    best = 0.0
    n = len(tree)
    for bits in range(1 << n):
        chosen = [i for i in range(n) if bits >> i & 1]
        ok = all(
            not is_ancestor(tree, i, j) and not is_ancestor(tree, j, i)
            for i in chosen for j in chosen if i != j
        )
        if ok:
            best = max(best, sum(utilities[i] - tau for i in chosen))
    return best


def test_antichain_exhaustive_random_trees() -> None:
    """Property tests: (a) result is always an antichain, (b) on small trees
    its utility equals the brute-force optimum."""
    rng = np.random.default_rng(8407)
    for _ in range(150):
        prob = rng.random((160, 160), dtype=np.float32) ** 2
        tree = build_component_tree(prob, levels=(0.10, 0.25, 0.45, 0.70), min_area=10)
        utilities = rng.random(len(tree))
        tau = float(rng.uniform(0.1, 0.9))
        keep = select_antichain(tree, utilities, tau)
        for i in keep:
            for j in keep:
                if i != j:
                    assert not is_ancestor(tree, i, j) and not is_ancestor(tree, j, i)
        if len(tree) <= 14:
            optimum = _brute_force_best(tree, utilities, tau)
            assert abs(sum(utilities[i] - tau for i in keep) - optimum) < 1e-9


def test_empty_and_flat_probability() -> None:
    tree = build_component_tree(np.zeros((64, 64), np.float32))
    assert len(tree) == 0
    assert select_antichain(tree, np.zeros(0), tau=0.2) == []
    flat = np.full((64, 64), 0.95, np.float32)
    tree = build_component_tree(flat, levels=(0.10, 0.50), min_area=20)
    assert len(tree) == 1  # whole canvas is one node; level-1 duplicate dropped


def test_levels_must_be_ascending() -> None:
    prob = np.zeros((64, 64), np.float32)
    with pytest.raises(ValueError):
        build_component_tree(prob, levels=(0.5, 0.3))
