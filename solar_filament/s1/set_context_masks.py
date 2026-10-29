"""Mask cache loading and ownership/area helpers for the S1 context builder.

Split out of ``set_context_features.py``; that module keeps feature assembly and CLI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CACHE_SCHEMA = "s1-proposal-marginal-refined-v1"
IMAGE_SHAPE = (2048, 2048)
MIN_AREA = 20
OWNERSHIP_THRESHOLD = 0.80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(cache_root: Path, fold: int, file_name: str) -> Path:
    return cache_root / f"fold_{fold}" / f"{Path(file_name).stem}.npz"


def load_refined_cache(path: Path, file_name: str, fold: int) -> list[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"].item()) != CACHE_SCHEMA:
            raise ValueError(f"unexpected cache schema: {path}")
        if str(data["file_name"].item()) != file_name or int(data["fold"].item()) != fold:
            raise ValueError(f"cache identity mismatch: {path}")
        if tuple(map(int, data["shape"])) != IMAGE_SHAPE:
            raise ValueError(f"cache shape mismatch: {path}")
        packed = data["packed"]
        expected = int(data["candidate_count"].item())
        masks = (
            [np.unpackbits(item, axis=-1)[:, : IMAGE_SHAPE[1]].astype(bool) for item in packed]
            if packed.size
            else []
        )
    if len(masks) != expected:
        raise ValueError(f"cache candidate count mismatch: {path}")
    return masks


def pairwise_intersections(masks: list[np.ndarray]) -> np.ndarray:
    count = len(masks)
    intersections = np.zeros((count, count), dtype=np.int64)
    for first in range(count):
        intersections[first, first] = int(np.count_nonzero(masks[first]))
        for second in range(first + 1, count):
            value = int(np.count_nonzero(masks[first] & masks[second]))
            intersections[first, second] = value
            intersections[second, first] = value
    return intersections


def _containment_subset(
    active: list[int], areas: np.ndarray, intersections: np.ndarray
) -> tuple[list[int], set[int]]:
    dropped: set[int] = set()
    for position, first in enumerate(active):
        for second in active[position + 1 :]:
            if first in dropped or second in dropped:
                continue
            intersection = float(intersections[first, second])
            if intersection <= 0.0:
                continue
            smaller = first if areas[first] <= areas[second] else second
            if intersection / max(float(areas[smaller]), 1.0) >= OWNERSHIP_THRESHOLD:
                dropped.add(smaller)
    return [index for index in active if index not in dropped], dropped


def ownership_trace(
    masks: list[np.ndarray],
    intersections: np.ndarray,
    active: list[int] | None = None,
) -> dict[str, Any]:
    """Apply the frozen containment+area-ascending ownership while retaining identity."""
    active_indices = list(range(len(masks))) if active is None else list(active)
    areas = np.diag(intersections).astype(np.int64)
    subset, dropped_by_containment = _containment_subset(
        active_indices, areas, intersections
    )
    order = sorted(subset, key=lambda index: areas[index])
    occupied = np.zeros(IMAGE_SHAPE, dtype=bool)
    exclusive_areas = np.zeros(len(masks), dtype=np.int64)
    output_order = np.full(len(masks), -1, dtype=np.int64)
    dropped_by_min_area: set[int] = set()
    kept_masks: list[np.ndarray] = []
    kept_indices: list[int] = []
    for index in order:
        exclusive = masks[index] & ~occupied
        exclusive_area = int(np.count_nonzero(exclusive))
        exclusive_areas[index] = exclusive_area
        if exclusive_area < MIN_AREA:
            dropped_by_min_area.add(index)
            continue
        output_order[index] = len(kept_indices)
        kept_indices.append(index)
        kept_masks.append(exclusive)
        occupied |= exclusive
    return {
        "kept_indices": kept_indices,
        "kept_masks": kept_masks,
        "output_order": output_order,
        "exclusive_areas": exclusive_areas,
        "dropped_by_containment": dropped_by_containment,
        "dropped_by_min_area": dropped_by_min_area,
    }


def _zscore(values: pd.Series) -> tuple[float, float, np.ndarray]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=0))
    z = (array - mean) / std if std > 0.0 else np.zeros_like(array)
    return mean, std, z
