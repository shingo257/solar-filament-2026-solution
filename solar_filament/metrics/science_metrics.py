"""Scientific morphology metrics for solar-filament instance masks.

Compatibility wrapper: the implementation now lives in
``science_metrics_morphology.py`` (spine/boundary/skeleton primitives and
single-mask metrics) and ``science_metrics_matching.py`` (IoU matching,
fragmentation, paired errors and the cluster bootstrap). Public names are
re-exported unchanged so existing imports keep working.

These metrics complement, but never replace, the official Panoptic Quality
implementation in :mod:`solar_filament.metrics.metrics_official`.  They are intentionally small,
deterministic CPU functions so the S0 scientific audit can be tested without a
GPU or model checkpoint.
"""

from solar_filament.metrics.science_metrics_morphology import (
    SpineAudit,
    audit_spine_coordinates,
    boundary,
    boundary_f1,
    centerline_cldice,
    mask_geometry,
    rasterize_spine,
    skeletonize,
    spine_centerline_metrics,
    spine_coverage,
    symmetric_centerline_distance,
    topology_counts,
)
from solar_filament.metrics.science_metrics_matching import (
    cluster_bootstrap_mean,
    fragmentation_counts,
    iou_matrix,
    match_instances,
    matched_geometry_errors,
    paired_centerline_metrics,
)

__all__ = [
    "SpineAudit",
    "audit_spine_coordinates",
    "boundary",
    "boundary_f1",
    "centerline_cldice",
    "cluster_bootstrap_mean",
    "fragmentation_counts",
    "iou_matrix",
    "mask_geometry",
    "match_instances",
    "matched_geometry_errors",
    "paired_centerline_metrics",
    "rasterize_spine",
    "skeletonize",
    "spine_centerline_metrics",
    "spine_coverage",
    "symmetric_centerline_distance",
    "topology_counts",
]
