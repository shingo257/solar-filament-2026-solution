"""V32 overlap-aware repair for Solar Filament Kaggle submissions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


HEIGHT = 2048
WIDTH = 2048


@dataclass(frozen=True)
class MergeConfig:
    name: str
    min_dice: float
    min_containment: float = 0.0
    min_area: int = 20


def decode_compressed_counts(text: str) -> list[int]:
    counts: list[int] = []
    position = 0
    while position < len(text):
        value = 0
        shift = 0
        while True:
            code = ord(text[position]) - 48
            position += 1
            value |= (code & 0x1F) << (5 * shift)
            shift += 1
            if not code & 0x20:
                if code & 0x10:
                    value |= -1 << (5 * shift)
                break
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def decode_mask(text: str, height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    flat = np.zeros(height * width, dtype=bool)
    offset = 0
    foreground = False
    for run in decode_compressed_counts(text):
        if run < 0 or offset + run > flat.size:
            raise ValueError("Invalid compressed RLE run")
        if foreground and run:
            flat[offset : offset + run] = True
        offset += run
        foreground = not foreground
    if offset != flat.size:
        raise ValueError(f"RLE size mismatch: {offset} != {flat.size}")
    return flat.reshape((height, width), order="F")


def encode_compressed_counts(counts: Iterable[int]) -> str:
    counts = [int(value) for value in counts]
    output: list[str] = []
    for index, original in enumerate(counts):
        value = original - counts[index - 2] if index > 2 else original
        while True:
            code = value & 0x1F
            value >>= 5
            more = (code & 0x10 and value != -1) or (not code & 0x10 and value != 0)
            if more:
                code |= 0x20
            output.append(chr(code + 48))
            if not more:
                break
    return "".join(output)


def encode_mask(mask: np.ndarray) -> str:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    padded = np.concatenate((np.zeros(1, dtype=np.uint8), flat, np.zeros(1, dtype=np.uint8)))
    changes = np.flatnonzero(np.diff(padded))
    boundaries = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(boundaries).astype(np.int64).tolist()
    if counts and counts[-1] == 0:
        counts.pop()
    return encode_compressed_counts(counts)


def overlap_metrics(first: np.ndarray, second: np.ndarray, first_area: int, second_area: int) -> tuple[float, float, int]:
    intersection = int(np.logical_and(first, second).sum())
    if intersection == 0:
        return 0.0, 0.0, 0
    containment = intersection / min(first_area, second_area)
    dice = 2.0 * intersection / (first_area + second_area)
    return containment, dice, intersection


def _components(size: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(size))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in edges:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(size):
        grouped[find(index)].append(index)
    return list(grouped.values())


def cluster_masks(
    masks: list[np.ndarray],
    config: MergeConfig,
) -> tuple[list[list[int]], list[tuple[int, int, float, float, int]]]:
    areas = [int(mask.sum()) for mask in masks]
    accepted: list[tuple[int, int]] = []
    audit: list[tuple[int, int, float, float, int]] = []
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            containment, dice, intersection = overlap_metrics(
                masks[first], masks[second], areas[first], areas[second]
            )
            if intersection == 0:
                continue
            audit.append((first, second, containment, dice, intersection))
            if containment >= config.min_containment and dice >= config.min_dice:
                accepted.append((first, second))
    return _components(len(masks), accepted), audit


def build_candidate(
    source: pd.DataFrame,
    config: MergeConfig,
    progress: Callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in source.itertuples(index=False):
        image_id, suffix = str(row.filament_id).rsplit("_", 1)
        grouped[image_id].append((int(suffix), str(row.segmentation_rle)))

    iterator = grouped.items()
    if progress is not None:
        iterator = progress(iterator, total=len(grouped), desc=config.name)

    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for image_id, records in iterator:
        records = sorted(records)
        masks = [decode_mask(counts) for _, counts in records]
        components, overlap_audit = cluster_masks(masks, config)
        clusters: list[tuple[int, np.ndarray, int]] = []
        for component in components:
            merged = np.logical_or.reduce([masks[index] for index in component])
            priority = min(records[index][0] for index in component)
            clusters.append((priority, merged, len(component)))

        occupied = np.zeros((HEIGHT, WIDTH), dtype=bool)
        kept: list[tuple[np.ndarray, int]] = []
        removed_pixels = 0
        dropped = 0
        for _, merged, component_size in sorted(clusters, key=lambda item: item[0]):
            removed_pixels += int(np.logical_and(merged, occupied).sum())
            exclusive = merged & ~occupied
            if exclusive.sum() < config.min_area:
                dropped += 1
                continue
            occupied |= exclusive
            kept.append((exclusive, component_size))

        for counter, (mask, _) in enumerate(kept, start=1):
            rows.append(
                {
                    "filament_id": f"{image_id}_{counter}",
                    "segmentation_rle": encode_mask(mask),
                }
            )
        audits.append(
            {
                "image_id": image_id,
                "source_masks": len(masks),
                "overlapping_pairs": len(overlap_audit),
                "clusters": len(components),
                "merged_groups": sum(len(component) > 1 for component in components),
                "submitted": len(kept),
                "removed_cross_cluster_pixels": removed_pixels,
                "dropped_masks": dropped,
            }
        )

    candidate = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    return candidate, pd.DataFrame(audits)


def validate_submission(frame: pd.DataFrame) -> dict[str, object]:
    if frame.columns.tolist() != ["filament_id", "segmentation_rle"]:
        raise AssertionError(frame.columns.tolist())
    if frame.empty or frame.isna().any().any():
        raise AssertionError("Submission is empty or contains null values")
    if frame["filament_id"].duplicated().any():
        raise AssertionError("Duplicate filament_id")

    occupied_by_image: dict[str, np.ndarray] = {}
    areas: list[int] = []
    for row in frame.itertuples(index=False):
        mask = decode_mask(str(row.segmentation_rle))
        if not mask.any():
            raise AssertionError(f"Empty mask: {row.filament_id}")
        image_id = str(row.filament_id).rsplit("_", 1)[0]
        occupied = occupied_by_image.setdefault(image_id, np.zeros((HEIGHT, WIDTH), dtype=bool))
        if np.logical_and(occupied, mask).any():
            raise AssertionError(f"Overlapping masks: {image_id}")
        occupied |= mask
        areas.append(int(mask.sum()))

    return {
        "rows": len(frame),
        "images": len(occupied_by_image),
        "overlap_pixels": 0,
        "area_min": min(areas),
        "area_median": int(np.median(areas)),
        "area_max": max(areas),
    }


def assert_codec_roundtrip(samples: Iterable[str]) -> None:
    for text in samples:
        decoded = decode_mask(text)
        encoded = encode_mask(decoded)
        if not np.array_equal(decoded, decode_mask(encoded)):
            raise AssertionError("Compressed RLE round-trip failed")


def load_submission(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.columns.tolist() != ["filament_id", "segmentation_rle"]:
        raise ValueError(f"Unexpected columns: {frame.columns.tolist()}")
    return frame
