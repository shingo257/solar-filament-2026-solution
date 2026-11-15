"""Spine aux-teacher — ultralytics integration glue & trainer (train-time + Arm B+ rescue).

Design (prereg 2026-11-13): spine polylines travel through the standard
YOLO-seg label/transform pipeline as **pseudo-instances of a reserved
class** so every geometric transform (mosaic concat, RandomPerspective
``apply_segments`` affine, flips, letterbox) applies identical math to
spine and mask. Immediately before the final ``Format`` transform a
``SplitSpineSegments`` step removes the pseudo-instances and stores the
transformed polylines on ``labels['spine_segments_px']``; the detection
pipeline then sees only real instances (model stays nc=1).

Constraints verified against ultralytics 8.4.118–8.4.120 source:
- ``verify_image_label`` asserts ``max_cls < num_cls`` -> the dataset
  yaml for spine training must declare ``nc: 2`` ({0: filament,
  1: spine_aux}); ``single_cls`` MUST be ``False`` (else ``BaseDataset``
  overwrites ``cls=1`` to ``0`` before transforms), and the custom trainer
  forces ``model.nc = 1`` and ``model.names = {0: 'filament'}``.
- ``Format.apply_instances`` calls ``cls.squeeze(1)`` -> ``labels['cls']``
  after ``SplitSpineSegments`` must have 2D shape ``(N_keep, 1)``.
- ``YOLODataset.collate_fn`` uses ``zip(*values)``, producing a ``tuple``
  of length ``B`` for custom keys -> ``split_spine_batch`` accepts both
  ``list`` and ``tuple``.
- ``SpineAuxLoss.call`` computes BCE in ``float32`` via log-sigmoid so
  AMP ``float16`` never underflows ``1 - 1e-7 -> 1.0`` into ``NaN`` grads.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    norm[:, 0] = np.clip(norm[:, 0] / img_w, 0.0, 1.0)
    norm[:, 1] = np.clip(norm[:, 1] / img_h, 0.0, 1.0)
    flat = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
    return f"{SPINE_CLS} {flat}"


def _ensure_min_three_points(pts: np.ndarray) -> np.ndarray:
    """Subdivide a 2-point segment by its midpoint so YOLO-seg sees >=7 tokens (not a 5-token bbox)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 2:
        mid = 0.5 * (pts[0] + pts[1])
        return np.vstack([pts[0], mid, pts[1]])
    return pts


def emit_label_txt(
    instances: list[np.ndarray],
    spines: list[np.ndarray],
    img_h: int,
    img_w: int,
    real_cls: int = 0,
) -> str:
    """Emit a full label .txt: real instance polygons + spine pseudo-rows.

    ``instances``: list of (M,2) polygon point arrays (image coords).
    ``spines``: list of (N,2) spine polylines, same order as instances.
    2-point spines are midpoint-subdivided to 3 collinear points so
    Ultralytics ``verify_image_label`` never rejects them as 5-column boxes.
    """
    rows = []
    for poly in instances:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        norm = pts.copy()
        norm[:, 0] = np.clip(norm[:, 0] / img_w, 0.0, 1.0)
        norm[:, 1] = np.clip(norm[:, 1] / img_h, 0.0, 1.0)
        rows.append(f"{real_cls} " + " ".join(f"{v:.6f}" for v in norm.reshape(-1)))
    for spine in spines:
        pts = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 2:
            rows.append(spine_label_row(_ensure_min_three_points(pts), img_h, img_w))
    return "\n".join(rows) + ("\n" if rows else "")


def _pad_lrtb_from_labels(labels: dict) -> tuple[int, int, int, int]:
    """Extract (left, top, right, bottom) canvas padding pixels from LetterBox metadata."""
    ratio_pad = labels.get("ratio_pad")
    if not isinstance(ratio_pad, (list, tuple)) or len(ratio_pad) < 2:
        return (0, 0, 0, 0)
    pad_wh = ratio_pad[1]
    if not isinstance(pad_wh, (list, tuple)) or len(pad_wh) < 2:
        return (0, 0, 0, 0)
    dw, dh = float(pad_wh[0]), float(pad_wh[1])
    left = max(0, round(dw - 0.1))
    right = max(0, round(dw + 0.1))
    top = max(0, round(dh - 0.1))
    bottom = max(0, round(dh + 0.1))
    return (left, top, right, bottom)


class SplitSpineSegments:
    """Pre-Format transform: move SPINE_CLS instances to spine_segments_px.

    Sits last in the geometric Compose (inserted just before ``Format``).
    Preserves ``cls`` as 2D ``(N_keep, 1)`` when input is 2D or ``img`` is
    present (required by ``Format.apply_instances`` ``cls.squeeze(1)``), and
    converts normalized segments to canvas pixel coordinates when
    ``augment=False`` (e.g. validation or ``close_mosaic`` epochs).
    """

    def __init__(self, spine_cls: int = SPINE_CLS):
        self.spine_cls = spine_cls

    def __call__(self, labels: dict) -> dict:
        instances = labels.get("instances")
        cls = labels.get("cls")
        labels["spine_pad_lrtb"] = _pad_lrtb_from_labels(labels)
        if instances is None or cls is None:
            labels["spine_segments_px"] = []
            return labels
        cls_arr = np.asarray(cls)
        keep_2d = cls_arr.ndim == 2 or "img" in labels
        flat_cls = cls_arr.reshape(-1)
        is_spine = flat_cls == self.spine_cls
        segs = np.asarray(instances.segments)
        if segs.ndim == 3 and "img" in labels and getattr(instances, "normalized", False):
            h, w = labels["img"].shape[:2]
            scale = np.array([float(w), float(h)], dtype=segs.dtype)
            segs = segs * scale
        labels["spine_segments_px"] = (
            [segs[i] for i in np.flatnonzero(is_spine)] if segs.ndim == 3 else []
        )
        keep = ~is_spine
        labels["instances"] = instances[keep]
        kept_cls = flat_cls[keep]
        labels["cls"] = kept_cls.reshape(-1, 1) if keep_2d else kept_cls
        return labels


def split_spine_batch(batch: dict, spine_key: str = "spine_segments_px") -> list[list[np.ndarray]]:
    """Collect per-image spine polyline lists from a collated batch.

    ``YOLODataset.collate_fn`` aggregates unknown keys via ``zip(*values)``,
    producing a ``tuple`` of length ``B``; direct callers may pass a ``list``.
    """
    value = batch.get(spine_key, ())
    if isinstance(value, (list, tuple)):
        return [list(v) if isinstance(v, (list, tuple)) else [v] for v in value]
    return [[value]] if len(value) else [[]]


def split_pad_batch(
    batch: dict,
    batch_size: int,
    pad_key: str = "spine_pad_lrtb",
) -> list[tuple[int, int, int, int]]:
    """Extract per-image (left, top, right, bottom) padding tuples from a batch."""
    raw = batch.get(pad_key)
    if isinstance(raw, (list, tuple)) and len(raw) == batch_size:
        out: list[tuple[int, int, int, int]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                out.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
            else:
                out.append((0, 0, 0, 0))
        return out
    return [(0, 0, 0, 0)] * batch_size


def build_p3_targets(
    spine_lists: list[list[np.ndarray]],
    batch_shape_hw: tuple[int, int],
    grid: int | tuple[int, int] = 160,
    radius_cells: int = 1,
    pads: Sequence[tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize each batch image's spine polylines to (pos, valid) grids."""
    from solar_filament.pipeline.spine_aux import (
        build_spine_target,
        padding_exclusion_mask,
    )

    out_h, out_w = (grid, grid) if isinstance(grid, int) else (int(grid[0]), int(grid[1]))
    h, w = int(batch_shape_hw[0]), int(batch_shape_hw[1])
    bsz = len(spine_lists)
    pos = np.zeros((bsz, out_h, out_w), dtype=bool)
    valid = np.ones_like(pos)
    for i, spines in enumerate(spine_lists):
        pad = pads[i] if pads is not None and i < len(pads) else (0, 0, 0, 0)
        vmask = (
            padding_exclusion_mask(out_h, out_w, h, w, pad=pad)
            if any(pad)
            else None
        )
        p, v = build_spine_target(
            spines,
            out_h=out_h,
            out_w=out_w,
            img_h=h,
            img_w=w,
            radius_cells=radius_cells,
            valid_mask=vmask,
        )
        pos[i], valid[i] = p, v
    return pos, valid


class SpineAuxLoss:
    """AMP-safe torch implementation of positive/negative separately-averaged BCE."""

    @staticmethod
    def call(logits, pos, valid, eps: float = 1e-7):
        import torch
        import torch.nn.functional as F

        del eps  # log-sigmoid BCE is exact and avoids float16 clamp rounding to 1.0
        logits32 = logits.float()
        pos_t = torch.as_tensor(pos, dtype=torch.bool, device=logits.device)
        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=logits.device)
        if pos_t.ndim == 3 and logits32.ndim == 4:
            pos_t = pos_t.unsqueeze(1)
            valid_t = valid_t.unsqueeze(1)
        pos_sel = pos_t & valid_t
        neg_sel = (~pos_t) & valid_t
        zero = logits32.sum() * 0.0
        pos_term = (
            F.binary_cross_entropy_with_logits(
                logits32[pos_sel], torch.ones_like(logits32[pos_sel]), reduction="mean"
            )
            if pos_sel.any()
            else zero
        )
        neg_term = (
            F.binary_cross_entropy_with_logits(
                logits32[neg_sel], torch.zeros_like(logits32[neg_sel]), reduction="mean"
            )
            if neg_sel.any()
            else zero
        )
        return pos_term + neg_term


def compute_spine_ridge_score(
    mask: np.ndarray,
    spine_prob: np.ndarray,
    *,
    top_quantile: float = 0.75,
) -> float:
    """Mean spine heatmap probability along the top-quantile ridge inside ``mask``.

    Maps pixel coordinates directly onto the P3 ``spine_prob`` grid ``(gh, gw)``
    without upsampling ``spine_prob`` to ``2048x2048``.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    h, w = mask.shape
    gh, gw = spine_prob.shape
    cy = np.clip((ys.astype(np.int64) * gh) // h, 0, gh - 1)
    cx = np.clip((xs.astype(np.int64) * gw) // w, 0, gw - 1)
    vals = spine_prob[cy, cx].astype(np.float64)
    if len(vals) <= 4 or top_quantile <= 0.0:
        return float(vals.mean())
    thr = float(np.quantile(vals, top_quantile))
    top_vals = vals[vals >= thr]
    return float(top_vals.mean()) if len(top_vals) else float(vals.mean())


def selective_spine_rescue(
    conf: Sequence[float] | np.ndarray,
    areas: Sequence[float] | np.ndarray,
    spine_scores: Sequence[float] | np.ndarray,
    *,
    primary_conf: float = 0.26,
    rescue_floor: float = 0.14,
    min_spine: float = 0.65,
    min_area: int = 20,
) -> list[int]:
    """Keep all primary proposals plus low-conf proposals with strong P3 spine ridge support."""
    conf_arr = np.asarray(conf, dtype=np.float64)
    area_arr = np.asarray(areas, dtype=np.float64)
    spine_arr = np.asarray(spine_scores, dtype=np.float64)
    if not (len(conf_arr) == len(area_arr) == len(spine_arr)):
        raise ValueError("conf, areas, and spine_scores must have equal length")
    kept: list[int] = []
    for i in range(len(conf_arr)):
        if area_arr[i] < min_area:
            continue
        if conf_arr[i] >= primary_conf or (
            conf_arr[i] >= rescue_floor and spine_arr[i] >= min_spine
        ):
            kept.append(i)
    return kept


def build_spine_trainer_classes(aux_weight: float = 0.0, radius_cells: int = 1):
    """Build Ultralytics Dataset, Loss, and SegmentationTrainer classes for Spine A/B.

    Both Arm A (``aux_weight=0.0``) and Arm B (``aux_weight>0.0``) can share the
    exact same dataset/transform/trainer path so the ONLY difference between
    Arm A and Arm B is the P3 ``1x1 conv`` spine auxiliary supervision.
    """
    from copy import copy

    from torch import nn
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import DEFAULT_CFG, RANK
    from ultralytics.utils.loss import v8SegmentationLoss
    from ultralytics.utils.torch_utils import unwrap_model

    class SpineYOLODataset(YOLODataset):
        def build_transforms(self, hyp=None):
            transforms = super().build_transforms(hyp)
            transforms.insert(-1, SplitSpineSegments(spine_cls=SPINE_CLS))
            return transforms

    class SpineSegmentationLoss(v8SegmentationLoss):
        def __init__(self, model: nn.Module, tal_topk: int = 10):
            super().__init__(model, tal_topk=tal_topk)
            self.spine_head = getattr(model, "spine_head", None)
            self.aux_weight = float(aux_weight)
            self.radius_cells = int(radius_cells)

        def loss(self, preds: dict, batch: dict):
            loss_vec, loss_items = super().loss(preds, batch)
            if (
                self.aux_weight > 0.0
                and self.spine_head is not None
                and "spine_segments_px" in batch
            ):
                p3_feat = preds["feats"][0]
                aux_logits = self.spine_head(p3_feat)
                bsz = int(aux_logits.shape[0])
                spine_lists = split_spine_batch(batch)
                if len(spine_lists) == bsz:
                    pads = split_pad_batch(batch, bsz)
                    pos, valid = build_p3_targets(
                        spine_lists,
                        batch["img"].shape[-2:],
                        grid=aux_logits.shape[-2:],
                        radius_cells=self.radius_cells,
                        pads=pads,
                    )
                    aux_raw = SpineAuxLoss.call(aux_logits, pos, valid)
                    weighted = self.aux_weight * aux_raw
                    loss_vec[4] = loss_vec[4] + weighted * bsz
                    loss_items["sem_loss"] = (loss_items["sem_loss"] + weighted).detach()
            return loss_vec, loss_items

    class SpineSegmentationTrainer(SegmentationTrainer):
        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            overrides = dict(overrides or {})
            # Never allow BaseDataset.update_labels to overwrite SPINE_CLS=1 -> 0
            overrides["single_cls"] = False
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            # Force single real output class (0: filament) regardless of data.yaml nc=2
            orig_names = self.data.get("names")
            self.data = {**self.data, "names": {0: "filament"}}
            try:
                model = self.set_model_names_for_load(
                    SegmentationModel(
                        cfg,
                        nc=1,
                        ch=self.data.get("channels", 3),
                        verbose=verbose and RANK == -1,
                    )
                )
                if weights:
                    model.load(weights)
            finally:
                if orig_names is not None:
                    self.data["names"] = orig_names
            model.nc = 1
            model.names = {0: "filament"}
            if aux_weight > 0.0:
                seg_head = model.model[-1]
                p3_ch = int(seg_head.cv2[0][0].conv.in_channels)
                model.spine_head = nn.Conv2d(p3_ch, 1, kernel_size=1)
            return model

        def set_model_attributes(self):
            super().set_model_attributes()
            umodel = unwrap_model(self.model)
            umodel.nc = 1
            umodel.names = {0: "filament"}
            umodel.criterion = SpineSegmentationLoss(umodel)

        def get_class_counts(self):
            classes = np.concatenate(
                [lb["cls"].reshape(-1) for lb in self.train_loader.dataset.labels], 0
            )
            real = classes[classes == 0].astype(int)
            return np.bincount(real, minlength=1).astype(np.float32)

        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            return SpineYOLODataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect or mode == "val",
                cache=self.args.cache or None,
                single_cls=False,
                stride=gs,
                pad=0.0 if mode == "train" else 0.5,
                prefix=f"{mode}: ",
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction if mode == "train" else 1.0,
            )

        def get_validator(self):
            val_args = copy(self.args)
            val_args.single_cls = True
            from ultralytics.models import yolo
            return yolo.segment.SegmentationValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=val_args,
                _callbacks=self.callbacks,
            )

    return SpineYOLODataset, SpineSegmentationLoss, SpineSegmentationTrainer


def extract_p3_spine_prob(yolo_obj, image_path: str, *, imgsz: int = 1280, conf: float = 0.02, iou: float = 0.55):
    """Run standard ``YOLO.predict`` while capturing P3 ``spine_head`` probability map ``(H_p3, W_p3)``.

    Returns ``(result, spine_prob_or_none)``. If ``yolo_obj.model`` has no ``spine_head``
    (e.g. Arm A), ``spine_prob_or_none`` is ``None`` and prediction is identical to standard YOLO.
    """
    import torch

    net = yolo_obj.model
    spine_head = getattr(net, "spine_head", None)
    if spine_head is None:
        res = yolo_obj.predict(
            str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            agnostic_nms=True,
            retina_masks=True,
            verbose=False,
        )[0]
        return res, None

    p3_layer_idx = int(net.model[-1].f[0])
    captured: list[torch.Tensor] = []

    def _hook(_module, _inp, out):
        captured.append(out.detach())

    handle = net.model[p3_layer_idx].register_forward_hook(_hook)
    try:
        res = yolo_obj.predict(
            str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            agnostic_nms=True,
            retina_masks=True,
            verbose=False,
        )[0]
        if not captured:
            return res, None
        with torch.no_grad():
            feat = captured[-1]
            spine_head = spine_head.to(device=feat.device, dtype=feat.dtype).eval()
            prob = torch.sigmoid(spine_head(feat).float())[0, 0].cpu().numpy()
        return res, prob
    finally:
        handle.remove()
