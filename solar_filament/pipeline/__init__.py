"""V35 pipeline primitives: ownership, refine, RLE, OOF cache, paths."""

from solar_filament.pipeline.component_tree import (
    ComponentTree,
    build_component_tree,
    select_antichain,
    select_antichain_forest,
)
from solar_filament.pipeline.gap_close import skeleton_gap_close
from solar_filament.pipeline.io_images import imread_unicode, imread_unicode_with_retry
from solar_filament.pipeline.oof_cache import load_yolo_cache, load_yolo_cache_with_retry
from solar_filament.pipeline.ownership import apply_ownership, drop_contained_indices, greedy_disjoint
from solar_filament.pipeline.paths import resolve_drive, v30_root
from solar_filament.pipeline.refine import build_variant, shrink_mask
from solar_filament.pipeline.rle import decode_mask, encode_mask
from solar_filament.pipeline.soft_target import (
    DEFAULT_CLUSTER_IOU,
    build_soft_targets,
    cluster_bbox,
    cluster_gt_instances,
    soft_target_crop,
)

__all__ = [
    "DEFAULT_CLUSTER_IOU",
    "ComponentTree",
    "apply_ownership",
    "build_component_tree",
    "build_soft_targets",
    "build_variant",
    "cluster_bbox",
    "cluster_gt_instances",
    "decode_mask",
    "drop_contained_indices",
    "encode_mask",
    "greedy_disjoint",
    "imread_unicode",
    "imread_unicode_with_retry",
    "load_yolo_cache",
    "load_yolo_cache_with_retry",
    "resolve_drive",
    "select_antichain",
    "select_antichain_forest",
    "shrink_mask",
    "skeleton_gap_close",
    "soft_target_crop",
    "v30_root",
]
