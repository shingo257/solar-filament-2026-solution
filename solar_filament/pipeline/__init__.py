"""V35 pipeline primitives: ownership, refine, RLE, OOF cache, paths."""

from solar_filament.pipeline.component_tree import (
    ComponentTree,
    build_component_tree,
    select_antichain,
    select_antichain_forest,
)
from solar_filament.pipeline.consensus_fusion import (
    DetectorProposal,
    MatchedCluster,
    fuse_cluster_proposal_mask,
    match_detector_proposals_1to1,
)
from solar_filament.pipeline.gap_close import skeleton_gap_close
from solar_filament.pipeline.hysteresis_split import (
    hysteresis_expand_mask,
    split_disconnected_merges,
)
from solar_filament.pipeline.io_images import imread_unicode, imread_unicode_with_retry
from solar_filament.pipeline.oof_cache import (
    load_yolo_cache,
    load_yolo_cache_with_retry,
)
from solar_filament.pipeline.ownership import (
    apply_ownership,
    drop_contained_indices,
    greedy_disjoint,
)
from solar_filament.pipeline.paths import resolve_drive, v30_root
from solar_filament.pipeline.refine import build_variant, shrink_mask
from solar_filament.pipeline.rle import decode_mask, encode_mask
from solar_filament.pipeline.sibling_shield_yolo import (
    SplitSiblingPolygons,
    build_sibling_shield_trainer_classes,
    compute_sibling_bce_weights,
    emit_sibling_shield_label_txt,
    rasterize_sibling_grid,
)
from solar_filament.pipeline.small_tal_yolo import (
    build_pure_small_tal_trainer_classes,
    build_small_tal_trainer_classes,
    compute_anchor_equalization_weights,
)
from solar_filament.pipeline.soft_target import (
    DEFAULT_CLUSTER_IOU,
    build_soft_targets,
    cluster_bbox,
    cluster_gt_instances,
    soft_target_crop,
)
from solar_filament.pipeline.spine_refiner import (
    SpineGuidedEdgeAttUNet,
    crop_spine_target,
    mass_preserving_spine_weights,
    spine_refiner_loss,
)

__all__ = [
    "DEFAULT_CLUSTER_IOU",
    "ComponentTree",
    "DetectorProposal",
    "MatchedCluster",
    "SpineGuidedEdgeAttUNet",
    "SplitSiblingPolygons",
    "apply_ownership",
    "build_component_tree",
    "build_pure_small_tal_trainer_classes",
    "build_sibling_shield_trainer_classes",
    "build_small_tal_trainer_classes",
    "build_soft_targets",
    "build_variant",
    "cluster_bbox",
    "cluster_gt_instances",
    "compute_anchor_equalization_weights",
    "compute_sibling_bce_weights",
    "crop_spine_target",
    "decode_mask",
    "drop_contained_indices",
    "emit_sibling_shield_label_txt",
    "encode_mask",
    "fuse_cluster_proposal_mask",
    "greedy_disjoint",
    "hysteresis_expand_mask",
    "imread_unicode",
    "imread_unicode_with_retry",
    "load_yolo_cache",
    "load_yolo_cache_with_retry",
    "mass_preserving_spine_weights",
    "match_detector_proposals_1to1",
    "rasterize_sibling_grid",
    "resolve_drive",
    "select_antichain",
    "select_antichain_forest",
    "shrink_mask",
    "skeleton_gap_close",
    "soft_target_crop",
    "spine_refiner_loss",
    "split_disconnected_merges",
    "v30_root",
]
