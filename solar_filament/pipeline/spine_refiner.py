"""320x320 Spine-guided EdgeAttUNet refiner target, mass-preserving loss, and model wrapper.

Implements Idea 5 (v129): applies organizer-approved (Topic 755241) human ``spine``
centerline supervision at full ``320x320`` crop resolution inside ``MediSegEdgeAttUNet``
while keeping test-time inference 100% architecture-identical (vanilla ``3 -> 1``
``MediSegEdgeAttUNet`` + ``shrink_mask`` inside the YOLO proposal).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from solar_filament.models.mediseg_models import MediSegEdgeAttUNet
from solar_filament.pipeline.spine_yolo import SpineAuxLoss


def crop_spine_target(
    spine_xy: np.ndarray | list,
    box: tuple[int, int, int, int],
    *,
    crop_size: int = 320,
    radius_px: int = 2,
    gt_crop_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize an image-space ``spine`` polyline onto a ``(crop_size, crop_size)`` bool mask.

    If ``gt_crop_mask`` is provided, the rasterized spine band is intersected with
    ``gt_crop_mask > 0`` so spine positives never spill outside the ground-truth mask.
    """
    out = np.zeros((crop_size, crop_size), dtype=np.uint8)
    pts = np.asarray(spine_xy, dtype=np.float64).reshape(-1, 2)
    x0, y0, x1, y1 = map(float, box)
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    if len(pts) >= 2:
        mapped = np.empty_like(pts)
        mapped[:, 0] = (pts[:, 0] - x0) * (crop_size / bw)
        mapped[:, 1] = (pts[:, 1] - y0) * (crop_size / bh)
        ipts = np.round(mapped).astype(np.int32).reshape(-1, 1, 2)
        thickness = max(1, 2 * int(radius_px) + 1)
        cv2.polylines(out, [ipts], isClosed=False, color=1, thickness=thickness, lineType=cv2.LINE_8)
    band = out > 0
    if gt_crop_mask is not None:
        gt_bool = np.asarray(gt_crop_mask) > 0
        clipped = band & gt_bool
        if clipped.any():
            band = clipped
        else:
            band = gt_bool & band
    return band


def mass_preserving_spine_weights(
    targets: torch.Tensor,
    spine_targets: torch.Tensor,
    *,
    gamma: float = 1.5,
) -> torch.Tensor:
    """Compute per-pixel BCE weights that upweight spine pixels while preserving foreground mass.

    For each sample ``b`` with foreground mask ``M = (targets > 0.5)`` and spine band
    ``S = (spine_targets > 0.5) & M``:
      - Background pixels ``~M`` get weight ``1.0``.
      - Foreground pixels ``M`` get ``(|M| / (|M| + gamma * |S|)) * (1 + gamma * S)`` so
        ``sum_{x in M} w(x) == |M|`` identically, preventing any shift of the ``0.45``
        inference probability threshold.
    """
    fg = targets > 0.5
    if gamma <= 0.0:
        return torch.ones_like(targets, dtype=torch.float32)
    sp = (spine_targets > 0.5) & fg
    fg_f = fg.float()
    sp_f = sp.float()
    fg_sum = fg_f.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
    sp_sum = sp_f.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
    scale = torch.where(
        fg_sum > 0.0,
        fg_sum / (fg_sum + float(gamma) * sp_sum).clamp(min=1e-6),
        torch.ones_like(fg_sum),
    )
    fg_weights = scale * (1.0 + float(gamma) * sp_f)
    return torch.where(fg, fg_weights, torch.ones_like(fg_weights))


def _boundary_map(t: torch.Tensor) -> torch.Tensor:
    dil = F.max_pool2d(t, 5, stride=1, padding=2)
    ero = -F.max_pool2d(-t, 5, stride=1, padding=2)
    return torch.clamp(dil - ero, 0.0, 1.0)


def spine_refiner_loss(
    mask_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    spine_logits: torch.Tensor | None = None,
    spine_targets: torch.Tensor | None = None,
    aux_weight: float = 0.0,
    spine_gamma: float = 0.0,
) -> torch.Tensor:
    """AMP-safe EdgeAttUNet loss with optional decoder spine aux head and mass-preserving BCE."""
    logits32 = mask_logits.float()
    targets32 = targets.float()
    probs = torch.sigmoid(logits32)
    inter = (probs * targets32).sum((1, 2, 3))
    denom = probs.sum((1, 2, 3)) + targets32.sum((1, 2, 3))
    sdice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

    if spine_gamma > 0.0 and spine_targets is not None:
        pix_w = mass_preserving_spine_weights(targets32, spine_targets.float(), gamma=spine_gamma)
        bce_map = F.binary_cross_entropy_with_logits(logits32, targets32, reduction="none")
        bce = (bce_map * pix_w).mean()
    else:
        bce = F.binary_cross_entropy_with_logits(logits32, targets32)

    region = 0.35 * bce + 0.65 * sdice
    edge = F.l1_loss(_boundary_map(probs), _boundary_map(targets32))
    total = region + 0.15 * edge

    if aux_weight > 0.0 and spine_logits is not None and spine_targets is not None:
        sp_bool = spine_targets > 0.5
        valid = torch.ones_like(sp_bool, dtype=torch.bool)
        aux_loss = SpineAuxLoss.call(spine_logits, sp_bool, valid)
        total = total + float(aux_weight) * aux_loss
    return total


class SpineGuidedEdgeAttUNet(MediSegEdgeAttUNet):
    """MediSegEdgeAttUNet with a train-time-only 1x1 ``spine_head`` on decoder ``up4`` features.

    ``export_base_state_dict()`` strips ``spine_head.*`` keys so checkpoints load with
    ``strict=True`` into vanilla ``MediSegEdgeAttUNet(3, 1, False)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        bilinear: bool = False,
        num_heads: int = 8,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bilinear=bilinear,
            num_heads=num_heads,
        )
        self.spine_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.edge_attn(x5, x)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        dec = self.up4(out, x1)
        return self.outc(dec), self.spine_head(dec)

    def export_base_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if not k.startswith("spine_head.")
        }
