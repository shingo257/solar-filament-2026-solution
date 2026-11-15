"""Small-instance Task-Aligned Assigner anchor equalization for YOLO11m-seg.

Implements Idea 6 (v130):
In Ultralytics ``v8SegmentationLoss``, ``TaskAlignedAssigner(topk=10)`` assigns 10
positive anchors to large filaments, whereas small filaments (``area <= 1024 px^2``
at ``2048``, ``<= 400 px^2`` at ``1280``, 42.4% of MAGFiLO GT) physically cover only
1–3 stride-8 anchor centers inside their bounding box. Summing over positive anchors
starves small filaments by 5x–10x in ``cls_loss``, ``box_loss``, ``dfl_loss``, and
``seg_loss`` simultaneously.

``compute_anchor_equalization_weights`` equalizes positive anchor weights by
``clamp(topk / n_assigned_to_gt, 1.0, max_boost)`` and normalizes across the batch's
positive anchors so:
  1. Total positive ``target_scores`` weight sum is 100% preserved (negative anchor
     calibration at ``conf = 0.26`` is invariant).
  2. Total positive ``seg_loss`` weight sum ``== fg_mask.sum()`` (effective mask LR
     is invariant).
  3. When ``max_boost == 1.0``, all weights equal ``1.0`` (exact identity with vanilla
     Ultralytics ``v8SegmentationLoss``).
"""

from __future__ import annotations

import torch

from solar_filament.pipeline.spine_yolo import build_spine_trainer_classes


def compute_anchor_equalization_weights(
    fg_mask: torch.Tensor,
    target_gt_idx: torch.Tensor,
    target_scores: torch.Tensor | None = None,
    *,
    topk: int = 10,
    max_boost: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mass-preserving ``(w_cls_box, w_seg)`` of shape ``(B, N_anchors)``.

    - Background anchors (``~fg_mask``) always receive weight ``1.0``.
    - Positive anchors assigned to a GT instance that received ``c`` anchors get raw
      boost ``clamp(topk / c, 1.0, max_boost)``.
    - ``w_seg[fg_mask]`` is normalized so ``w_seg[fg_mask].sum() == fg_mask.sum()``.
    - ``w_cls_box[fg_mask]`` is normalized so
      ``(target_scores * w_cls_box[..., None]).sum() == target_scores.sum()``,
      keeping the positive/negative classification balance strictly invariant.
    """
    bsz, _ = fg_mask.shape
    w_seg = torch.ones_like(fg_mask, dtype=torch.float32)
    w_cls = torch.ones_like(fg_mask, dtype=torch.float32)
    if max_boost <= 1.0 or not fg_mask.any():
        return w_cls, w_seg

    raw_fg_list: list[torch.Tensor] = []
    for b in range(bsz):
        fg_b = fg_mask[b]
        if not fg_b.any():
            continue
        gt_idx_b = target_gt_idx[b, fg_b].long()
        counts = torch.bincount(gt_idx_b, minlength=int(gt_idx_b.max().item()) + 1).float()
        per_anchor_cnt = counts[gt_idx_b].clamp(min=1.0)
        raw_b = (float(topk) / per_anchor_cnt).clamp(min=1.0, max=float(max_boost))
        raw_fg_list.append(raw_b)

    raw_fg = torch.cat(raw_fg_list, dim=0)
    fg_count = float(fg_mask.sum().item())
    norm_seg = raw_fg * (fg_count / raw_fg.sum().clamp(min=1e-6))
    w_seg[fg_mask] = norm_seg

    if target_scores is not None:
        ts_fg = target_scores[fg_mask].float().sum(dim=-1)  # (N_fg,)
        orig_ts_sum = ts_fg.sum()
        if orig_ts_sum > 1e-6:
            weighted_ts_sum = (ts_fg * raw_fg).sum().clamp(min=1e-6)
            norm_cls = raw_fg * (orig_ts_sum / weighted_ts_sum)
        else:
            norm_cls = norm_seg
        w_cls[fg_mask] = norm_cls
    else:
        w_cls[fg_mask] = norm_seg

    return w_cls, w_seg


def build_small_tal_trainer_classes(
    *,
    max_boost: float = 4.0,
    aux_weight: float = 0.0,
    radius_cells: int = 1,
):
    """Build Ultralytics Dataset, Anchor-Equalized Loss, and SegmentationTrainer classes."""
    from torch import nn
    from ultralytics.utils.ops import xyxy2xywh
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.torch_utils import unwrap_model

    base_ds_cls, base_loss_cls, base_trainer_cls = build_spine_trainer_classes(
        aux_weight=aux_weight,
        radius_cells=radius_cells,
    )

    class SmallTALSegmentationLoss(base_loss_cls):
        def __init__(self, model: nn.Module, tal_topk: int = 10):
            super().__init__(model, tal_topk=tal_topk)
            self.tal_topk = int(tal_topk)
            self.max_boost = float(max_boost)
            self._last_w_seg: torch.Tensor | None = None

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

            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

            w_cls, w_seg = compute_anchor_equalization_weights(
                fg_mask,
                target_gt_idx,
                target_scores,
                topk=self.tal_topk,
                max_boost=self.max_boost,
            )
            self._last_w_seg = w_seg

            target_scores_sum = max(target_scores.sum(), 1)

            bce_loss = self.bce(pred_scores, target_scores.to(dtype)) * w_cls.to(dtype).unsqueeze(
                -1
            )
            if self.class_weights is not None:
                bce_loss = bce_loss * self.class_weights
            loss[1] = bce_loss.sum() / target_scores_sum

            if fg_mask.sum():
                weighted_target_scores = target_scores * w_cls.to(target_scores.dtype).unsqueeze(-1)
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes / stride_tensor,
                    weighted_target_scores,
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

        def calculate_segmentation_loss(
            self,
            fg_mask: torch.Tensor,
            masks: torch.Tensor,
            target_gt_idx: torch.Tensor,
            target_bboxes: torch.Tensor,
            batch_idx: torch.Tensor,
            proto: torch.Tensor,
            pred_masks: torch.Tensor,
            imgsz: torch.Tensor,
        ) -> torch.Tensor:
            _, _, mask_h, mask_w = proto.shape
            loss = 0
            target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
            marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
            mxyxy = target_bboxes_normalized * torch.tensor(
                [mask_w, mask_h, mask_w, mask_h], device=proto.device
            )
            w_seg = (
                self._last_w_seg
                if self._last_w_seg is not None and self._last_w_seg.shape == fg_mask.shape
                else torch.ones_like(fg_mask, dtype=torch.float32)
            )

            for i, single_i in enumerate(
                zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, w_seg)
            ):
                fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, w_seg_i = (
                    single_i
                )
                if fg_mask_i.any():
                    mask_idx = target_gt_idx_i[fg_mask_i]
                    if self.overlap:
                        gt_mask = (masks[i] == (mask_idx + 1).view(-1, 1, 1)).float()
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                    eff_area = marea_i[fg_mask_i] / w_seg_i[fg_mask_i].to(marea_i.dtype).clamp(
                        min=1e-6
                    )
                    loss += self.single_mask_loss(
                        gt_mask,
                        pred_masks_i[fg_mask_i],
                        proto_i,
                        mxyxy_i[fg_mask_i],
                        eff_area,
                    )
                else:
                    loss += (proto * 0).sum() + (pred_masks * 0).sum()

            return loss / fg_mask.sum()

    class SmallTALSegmentationTrainer(base_trainer_cls):
        def set_model_attributes(self):
            super().set_model_attributes()
            umodel = unwrap_model(self.model)
            umodel.nc = 1
            umodel.names = {0: "filament"}
            umodel.criterion = SmallTALSegmentationLoss(umodel)

    return base_ds_cls, SmallTALSegmentationLoss, SmallTALSegmentationTrainer


def build_pure_small_tal_trainer_classes(*, max_boost: float = 4.0):
    """Build standard ``single_cls=True`` Ultralytics Loss & Trainer with Small-TAL equalization.

    Leaves ``YOLODataset``, ``data.yaml`` (``nc=1``), and RNG/transform streams 100% identical
    to ``prod`` (``yolo_fold_{0,1,2}`` and ``v96`` ``yolo_full_fold2``), changing only
    ``umodel.criterion``.
    """
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils.loss import v8SegmentationLoss
    from ultralytics.utils.torch_utils import unwrap_model

    _, spine_tal_loss_cls, _ = build_small_tal_trainer_classes(
        max_boost=max_boost,
        aux_weight=0.0,
    )

    class PureSmallTALSegmentationLoss(v8SegmentationLoss):
        def __init__(self, model, tal_topk: int = 10):
            super().__init__(model, tal_topk=tal_topk)
            self.tal_topk = int(tal_topk)
            self.max_boost = float(max_boost)
            self._last_w_seg: torch.Tensor | None = None

        get_assigned_targets_and_loss = spine_tal_loss_cls.get_assigned_targets_and_loss
        calculate_segmentation_loss = spine_tal_loss_cls.calculate_segmentation_loss

    class PureSmallTALSegmentationTrainer(SegmentationTrainer):
        def set_model_attributes(self):
            super().set_model_attributes()
            umodel = unwrap_model(self.model)
            umodel.criterion = PureSmallTALSegmentationLoss(umodel)

    return PureSmallTALSegmentationLoss, PureSmallTALSegmentationTrainer
