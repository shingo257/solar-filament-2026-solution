"""Cross-annotator unlabeled-sibling negative shielding for YOLO11m-seg (Idea 7, v131).

Mathematical motivation:
In MAGFiLO, 296/707 images have R in {2, 3} annotators, and 57.9% of physical filaments
are singletons (marked by k=1 of R annotators). Under per-record replication (prod), a
k-of-R filament receives k positive updates with soft TAL target t_+ (~0.50) and (R - k)
hard background negative updates (t = 0, weight = 1.0), driving its stationary confidence
to p* = (k * t_+) / (k + (R - k) * w_sib):
  - With w_sib = 1.0 (prod): R=2, k=1 -> p* ~ 0.25 < 0.26 (dropped right at the 0.26 cliff,
    even though RESEARCH_LOG §2.66 proved R=2, k=1 predictions have strongly positive
    marginal PQ = 0.65 >> 0.445!).
  - With w_sib = 0.20 (80% sibling negative shielding):
      * R=1, k=1 -> p* ~ 0.50
      * R=2, k=1 -> p* ~ 0.42 >= 0.26 (recovered!)
      * R=3, k=2 -> p* ~ 0.45 >= 0.26
      * R=3, k=1 -> p* ~ 0.36 (strictly preserving consensus order 3/3 > 2/3 > 1/2 > 1/3)
      * True non-filament solar disk background (k=0) keeps w_bce = 1.0 across all R copies!

Unlike v87 (union_hard / union_agree), positive polygons remain 100% crisp single-annotator
masks (no union bloat), and pseudo-class cls=1 polygons pass through Ultralytics's exact
Mosaic + RandomPerspective + Flip pipeline before SplitSiblingPolygons strips them prior to
Format so the model stays strictly nc=1.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


def _format_poly_line(cls_id: int, poly_xy: np.ndarray, img_h: int, img_w: int) -> str | None:
    pts = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return None
    if len(pts) == 2:
        mid = 0.5 * (pts[0:1] + pts[1:2])
        pts = np.vstack([pts[0:1], mid, pts[1:2]])
    xs = np.clip(pts[:, 0] / float(img_w), 0.0, 1.0)
    ys = np.clip(pts[:, 1] / float(img_h), 0.0, 1.0)
    interleaved = np.empty(2 * len(pts), dtype=np.float64)
    interleaved[0::2] = xs
    interleaved[1::2] = ys
    coords = " ".join(f"{v:.6f}" for v in interleaved)
    return f"{cls_id} {coords}"


def emit_sibling_shield_label_txt(
    pos_polys_xy: list[np.ndarray],
    sibling_polys_xy: list[np.ndarray],
    img_h: int,
    img_w: int,
) -> str:
    """Emit Ultralytics seg .txt with cls=0 (annotator GT) and cls=1 (sibling ignore)."""
    lines: list[str] = []
    for poly in pos_polys_xy:
        ln = _format_poly_line(0, poly, img_h, img_w)
        if ln is not None:
            lines.append(ln)
    for sib in sibling_polys_xy:
        ln = _format_poly_line(1, sib, img_h, img_w)
        if ln is not None:
            lines.append(ln)
    return "\n".join(lines) + ("\n" if lines else "")


class SplitSiblingPolygons:
    """Transform inserted right before Format to strip cls=1 sibling-ignore polygons."""

    def __init__(self, sibling_cls: int = 1):
        self.sibling_cls = int(sibling_cls)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        cls_arr = labels.get("cls")
        instances = labels.get("instances")
        if cls_arr is None or instances is None or len(cls_arr) == 0:
            labels["sibling_polys_px"] = []
            return labels

        flat_cls = np.asarray(cls_arr).reshape(-1)
        is_sib = flat_cls == self.sibling_cls
        if not np.any(is_sib):
            labels["sibling_polys_px"] = []
            return labels

        h, w = labels["img"].shape[:2]
        if getattr(instances, "normalized", False):
            instances.denormalize(w, h)
        segs = instances.segments
        sib_polys = [np.asarray(segs[i], dtype=np.float32).copy() for i in np.flatnonzero(is_sib)]

        keep = ~is_sib
        labels["instances"] = instances[keep]
        kept_cls = flat_cls[keep].astype(np.float32)
        labels["cls"] = kept_cls.reshape(-1, 1) if cls_arr.ndim == 2 else kept_cls
        labels["sibling_polys_px"] = sib_polys
        return labels


def rasterize_sibling_grid(
    sib_polys_px: list[np.ndarray],
    grid_h: int,
    grid_w: int,
    img_h: int,
    img_w: int,
    *,
    dilate_cells: int = 1,
) -> np.ndarray:
    """Rasterize pixel-space sibling polygons onto a ``(grid_h, grid_w)`` bool mask."""
    canvas = np.zeros((grid_h, grid_w), dtype=np.uint8)
    if not sib_polys_px:
        return canvas.astype(bool)
    sx = float(grid_w) / max(float(img_w), 1.0)
    sy = float(grid_h) / max(float(img_h), 1.0)
    valid_ipts = []
    for poly in sib_polys_px:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        mapped = np.empty_like(pts)
        mapped[:, 0] = pts[:, 0] * sx
        mapped[:, 1] = pts[:, 1] * sy
        valid_ipts.append(np.round(mapped).astype(np.int32).reshape(-1, 1, 2))
    if valid_ipts:
        cv2.fillPoly(canvas, valid_ipts, color=1)
        if dilate_cells > 0:
            k = 2 * int(dilate_cells) + 1
            canvas = cv2.dilate(canvas, np.ones((k, k), dtype=np.uint8), iterations=1)
    return canvas > 0


def compute_sibling_bce_weights(
    sibling_batch: list[list[np.ndarray]] | tuple[list[np.ndarray], ...],
    anchor_points_px: torch.Tensor,
    fg_mask: torch.Tensor,
    imgsz_hw: tuple[int, int],
    *,
    grid_hw: tuple[int, int] | None = None,
    sib_neg_weight: float = 0.20,
    dilate_cells: int = 1,
) -> torch.Tensor:
    """Compute ``(B, N_anchors)`` BCE weights shielding ``~fg_mask`` anchors inside sibling masks.

    - Positive anchors (``fg_mask == True``) ALWAYS receive weight ``1.0``.
    - True background anchors outside ``sibling_batch`` ALWAYS receive weight ``1.0``.
    - ``~fg_mask`` anchors whose center falls inside a sibling annotator's polygon receive
      ``sib_neg_weight`` (e.g. ``0.20``).
    - When ``sib_neg_weight == 1.0`` or ``sibling_batch`` has no polygons, returns all ones.
    """
    bsz, n_anc = fg_mask.shape
    weights = torch.ones((bsz, n_anc), dtype=torch.float32, device=fg_mask.device)
    if abs(float(sib_neg_weight) - 1.0) < 1e-7 or not sibling_batch:
        return weights
    if not any(len(p) > 0 for p in sibling_batch):
        return weights

    img_h, img_w = int(imgsz_hw[0]), int(imgsz_hw[1])
    if grid_hw is None:
        gh, gw = max(1, img_h // 8), max(1, img_w // 8)
    else:
        gh, gw = int(grid_hw[0]), int(grid_hw[1])

    grids = np.stack(
        [
            rasterize_sibling_grid(
                sibling_batch[b] if b < len(sibling_batch) else [],
                gh,
                gw,
                img_h,
                img_w,
                dilate_cells=dilate_cells,
            )
            for b in range(bsz)
        ],
        axis=0,
    )
    grid_t = torch.from_numpy(grids).to(device=fg_mask.device, dtype=torch.bool)

    # anchor_points_px: (N_anchors, 2) in (x_px, y_px)
    sx = float(gw) / max(float(img_w), 1.0)
    sy = float(gh) / max(float(img_h), 1.0)
    cx = (anchor_points_px[:, 0] * sx).long().clamp(0, gw - 1)
    cy = (anchor_points_px[:, 1] * sy).long().clamp(0, gh - 1)

    in_sib = grid_t[:, cy, cx]  # (B, N_anchors)
    shield_mask = (~fg_mask) & in_sib
    weights[shield_mask] = float(sib_neg_weight)
    return weights


def build_sibling_shield_trainer_classes(
    *,
    sib_neg_weight: float = 0.20,
    dilate_cells: int = 1,
):
    """Build Ultralytics Dataset, Loss, and SegmentationTrainer with sibling negative shielding."""
    from torch import nn
    from ultralytics.data.augment import Format
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils.loss import v8SegmentationLoss
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.torch_utils import unwrap_model

    class SiblingShieldYOLODataset(YOLODataset):
        def build_transforms(self, hyp=None):
            transforms = super().build_transforms(hyp=hyp)
            inserted = False
            for idx, t in enumerate(transforms.transforms):
                if isinstance(t, Format):
                    transforms.transforms.insert(idx, SplitSiblingPolygons(sibling_cls=1))
                    inserted = True
                    break
            if not inserted:
                transforms.transforms.append(SplitSiblingPolygons(sibling_cls=1))
            return transforms

        @staticmethod
        def collate_fn(batch: list[dict]) -> dict:
            sib_list = [sample.pop("sibling_polys_px", []) for sample in batch]
            collated = YOLODataset.collate_fn(batch)
            collated["sibling_polys_px"] = sib_list
            return collated

    class SiblingShieldSegmentationLoss(v8SegmentationLoss):
        def __init__(self, model: nn.Module, tal_topk: int = 10):
            super().__init__(model, tal_topk=tal_topk)
            self.sib_neg_weight = float(sib_neg_weight)
            self.dilate_cells = int(dilate_cells)

        def get_assigned_targets_and_loss(self, preds: dict, batch: dict) -> tuple:
            loss = torch.zeros(3, device=self.device)  # box, cls, dfl
            pred_distri, pred_scores = (
                preds["boxes"].permute(0, 2, 1).contiguous(),
                preds["scores"].permute(0, 2, 1).contiguous(),
            )
            anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
                * self.stride[0]
            )

            targets = torch.cat(
                (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
            )
            targets = self.preprocess(
                targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
            anchor_points_px = anchor_points * stride_tensor

            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points_px,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

            p3_h, p3_w = int(preds["feats"][0].shape[2]), int(preds["feats"][0].shape[3])
            img_h, img_w = round(float(imgsz[0].item())), round(float(imgsz[1].item()))
            w_bce = compute_sibling_bce_weights(
                batch.get("sibling_polys_px", ()),
                anchor_points_px.detach(),
                fg_mask,
                (img_h, img_w),
                grid_hw=(p3_h, p3_w),
                sib_neg_weight=self.sib_neg_weight,
                dilate_cells=self.dilate_cells,
            )

            target_scores_sum = max(target_scores.sum(), 1)
            bce_loss = self.bce(pred_scores, target_scores.to(dtype)) * w_bce.to(dtype).unsqueeze(
                -1
            )
            if self.class_weights is not None:
                bce_loss = bce_loss * self.class_weights
            loss[1] = bce_loss.sum() / target_scores_sum

            if fg_mask.sum():
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes / stride_tensor,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                    imgsz,
                    stride_tensor,
                )

            loss[0] *= self.hyp.box
            loss[1] *= self.hyp.cls
            loss[2] *= self.hyp.dfl
            return (
                (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
                loss,
                dict(zip(self.loss_names, loss.detach())),
            )

    class SiblingShieldSegmentationTrainer(SegmentationTrainer):
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            self.data["nc"] = 1
            self.data["names"] = {0: "filament"}
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            return model

        def set_model_attributes(self):
            super().set_model_attributes()
            umodel = unwrap_model(self.model)
            umodel.nc = 1
            umodel.names = {0: "filament"}
            umodel.criterion = SiblingShieldSegmentationLoss(umodel)

        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
            data_for_loader = dict(self.data)
            data_for_loader["nc"] = 2
            data_for_loader["names"] = {0: "filament", 1: "sibling_ignore"}
            return SiblingShieldYOLODataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect or mode == "val",
                cache=self.args.cache or None,
                single_cls=False,
                stride=int(gs),
                pad=0.0 if mode == "train" else 0.5,
                prefix=f"{mode}: ",
                task=self.args.task,
                classes=self.args.classes,
                data=data_for_loader,
                fraction=self.args.fraction if mode == "train" else 1.0,
            )

    return (
        SiblingShieldYOLODataset,
        SiblingShieldSegmentationLoss,
        SiblingShieldSegmentationTrainer,
    )
