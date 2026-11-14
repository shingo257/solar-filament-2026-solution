"""Spine aux-teacher — ultralytics integration glue (train-time only).

Design (prereg 2026-11-13): spine polylines travel through the standard
YOLO-seg label/transform pipeline as **pseudo-instances of a reserved
class** so every geometric transform (mosaic concat, RandomPerspective
``apply_segments`` affine, flips, letterbox) applies identical math to
spine and mask. Immediately before the final ``Format`` transform a
``SplitSpineSegments`` step removes the pseudo-instances and stores the
transformed polylines on ``labels['spine_segments_px']``; the detection
pipeline then sees only real instances (model stays nc=1).

Constraints verified against ultralytics 8.4.120 source:
- ``verify_image_label`` asserts ``max_cls < num_cls`` -> the dataset
  yaml for spine training must declare ``nc: 2`` ({0: filament,
  1: spine_aux}); the trainer forces model nc=1 anyway.
- ``Instances.segments`` is one row per instance, so the pseudo-class
  must ride as its own instance row (not a second segment).
- Split must run *before* ``Format`` rasterizes masks, else spine rows
  would corrupt GT masks.
"""

from __future__ import annotations

import numpy as np

SPINE_CLS = 1  # reserved pseudo-class in the spine-augmented dataset yaml


def spine_label_row(spine_xy: np.ndarray, img_h: int, img_w: int) -> str:
    """Format one annotation spine as a YOLO-seg label row of SPINE_CLS.

    ``spine_xy``: (N,2) image-space polyline. Output is the normalized
    ``cls x1 y1 x2 y2 ...`` polygon-format row; a polyline is a valid
    (degenerate) polygon for the transform math we rely on.
    """
    pts = np.asarray(spine_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError("spine polyline needs >= 2 points")
    norm = pts.copy()
    norm[:, 0] /= img_w
    norm[:, 1] /= img_h
    flat = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
    return f"{SPINE_CLS} {flat}"


def emit_label_txt(instances: list[np.ndarray], spines: list[np.ndarray],
                   img_h: int, img_w: int,
                   real_cls: int = 0) -> str:
    """Emit a full label .txt: real instance polygons + spine pseudo-rows.

    ``instances``: list of (M,2) polygon point arrays (image coords).
    ``spines``: list of (N,2) spine polylines, same order as instances.
    """
    rows = []
    for poly in instances:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        norm = pts.copy()
        norm[:, 0] /= img_w
        norm[:, 1] /= img_h
        rows.append(f"{real_cls} " + " ".join(f"{v:.6f}" for v in norm.reshape(-1)))
    for spine in spines:
        if len(np.asarray(spine).reshape(-1, 2)) >= 2:
            rows.append(spine_label_row(spine, img_h, img_w))
    return "\n".join(rows) + ("\n" if rows else "")


class SplitSpineSegments:
    """Pre-Format transform: move SPINE_CLS instances to spine_segments_px.

    Sits last in the geometric Compose (inserted just before ``Format``).
    At this point ``labels['instances'].segments`` are pixel-coordinate
    polylines in the post-mosaic/post-perspective canvas (size
    ``imgsz``); we keep them in pixel coords so the aux-target
    rasterizer can map directly onto the P3 grid.
    """

    def __init__(self, spine_cls: int = SPINE_CLS):
        self.spine_cls = spine_cls

    def __call__(self, labels: dict) -> dict:
        instances = labels.get("instances")
        cls = labels.get("cls")
        if instances is None or cls is None:
            labels["spine_segments_px"] = []
            return labels
        cls = np.asarray(cls)
        is_spine = cls.reshape(-1) == self.spine_cls
        segs = np.asarray(instances.segments)
        labels["spine_segments_px"] = [
            segs[i] for i in np.flatnonzero(is_spine)
        ] if segs.ndim == 3 else []
        keep = ~is_spine
        labels["instances"] = instances[keep]
        labels["cls"] = cls[keep]
        return labels


def split_spine_batch(batch: dict, spine_key: str = "spine_segments_px") -> list[list[np.ndarray]]:
    """Collect per-image spine polyline lists from a collated batch.

    Default ``collate_fn`` keeps unknown keys as a per-sample list; this
    helper normalizes that into ``list[image] -> list[(N,2) arrays]``.
    """
    value = batch.get(spine_key, [])
    if isinstance(value, list):
        return [v if isinstance(v, list) else [v] for v in value]
    return [[value]] if len(value) else [[]]


def build_p3_targets(spine_lists: list[list[np.ndarray]], batch_shape_hw: tuple[int, int],
                     grid: int = 160, radius_cells: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize each batch image's spine polylines to (pos, valid) grids."""
    from solar_filament.pipeline.spine_aux import build_spine_target
    h, w = batch_shape_hw
    pos = np.zeros((len(spine_lists), grid, grid), dtype=bool)
    valid = np.ones_like(pos)
    for i, spines in enumerate(spine_lists):
        p, v = build_spine_target(spines, out_h=grid, out_w=grid,
                                  img_h=h, img_w=w, radius_cells=radius_cells)
        pos[i], valid[i] = p, v
    return pos, valid


class SpineAuxLoss:
    """Reference torch implementation of the separately-averaged BCE.

    The Colab builder attaches a 1x1 conv head to the P3 feature map,
    obtains per-image logits of shape (B,1,grid,grid) and calls
    ``loss = SpineAuxLoss.call(logits, pos, valid) * aux_weight``.
    """

    @staticmethod
    def call(logits, pos, valid, eps: float = 1e-7):
        import torch
        pos = torch.as_tensor(pos, dtype=torch.bool, device=logits.device)
        valid = torch.as_tensor(valid, dtype=torch.bool, device=logits.device)
        prob = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
        pos_sel = pos & valid
        neg_sel = (~pos) & valid
        pos_term = (-torch.log(prob)[pos_sel]).mean() if pos_sel.any() else logits.new_zeros(())
        neg_term = (-torch.log(1.0 - prob)[neg_sel]).mean() if neg_sel.any() else logits.new_zeros(())
        return pos_term + neg_term
