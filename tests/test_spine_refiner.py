"""Unit and E2E tests for 320x320 Spine-guided EdgeAttUNet refiner (v129)."""

from __future__ import annotations

import numpy as np
import torch

from solar_filament.models.mediseg_models import MediSegEdgeAttUNet
from solar_filament.pipeline.spine_refiner import (
    SpineGuidedEdgeAttUNet,
    crop_spine_target,
    mass_preserving_spine_weights,
    spine_refiner_loss,
)


def test_crop_spine_target_maps_and_clips_to_gt_mask() -> None:
    gt = np.zeros((160, 160), dtype=np.uint8)
    gt[70:90, 20:140] = 1
    # Box in 2048 space: (100, 200, 420, 520) -> width=320, height=320 -> scale 0.5 to 160
    spine_xy = np.array([[120.0, 360.0], [260.0, 360.0], [400.0, 360.0]], dtype=np.float64)
    band = crop_spine_target(
        spine_xy,
        (100, 200, 420, 520),
        crop_size=160,
        radius_px=2,
        gt_crop_mask=gt,
    )
    assert band.shape == (160, 160)
    assert band.dtype == bool
    assert band.any()
    assert np.all(band <= (gt > 0))
    # Spine runs along y = (360 - 200) * 0.5 = 80
    assert band[80, 80]


def test_mass_preserving_spine_weights_conserves_foreground_mass() -> None:
    targets = torch.zeros((3, 1, 64, 64), dtype=torch.float32)
    spines = torch.zeros((3, 1, 64, 64), dtype=torch.float32)
    targets[0, 0, 20:44, 10:54] = 1.0
    spines[0, 0, 31:33, 12:52] = 1.0
    targets[1, 0, 15:20, 15:45] = 1.0
    spines[1, 0, 17:18, 15:45] = 1.0
    # Sample 2: empty background crop (edge case)

    w = mass_preserving_spine_weights(targets, spines, gamma=1.5)
    assert w.shape == targets.shape
    # Background weights must be strictly 1.0
    assert torch.allclose(w[targets < 0.5], torch.ones_like(w[targets < 0.5]))
    # Foreground sum per sample must equal |M| identically
    for b in range(3):
        fg_mask = targets[b] > 0.5
        expected = fg_mask.float().sum()
        actual = w[b][fg_mask].sum()
        assert torch.isclose(actual, expected, rtol=1e-5, atol=1e-4)
    # Spine pixels get strictly larger weight than off-spine foreground pixels
    sp0 = (spines[0] > 0.5) & (targets[0] > 0.5)
    off0 = (~(spines[0] > 0.5)) & (targets[0] > 0.5)
    assert float(w[0][sp0].mean()) > float(w[0][off0].mean())


def test_spine_guided_edgeatt_exports_strict_vanilla_state_dict_and_backward() -> None:
    torch.manual_seed(0)
    model = SpineGuidedEdgeAttUNet(3, 1, False, num_heads=4)
    x = torch.randn(2, 3, 64, 64)
    gt = (torch.rand(2, 1, 64, 64) > 0.7).float()
    sp = ((torch.rand(2, 1, 64, 64) > 0.85) & (gt > 0.5)).float()

    mask_logits, spine_logits = model.forward_with_aux(x)
    loss = spine_refiner_loss(
        mask_logits.half(),
        gt,
        spine_logits=spine_logits.half(),
        spine_targets=sp,
        aux_weight=0.25,
        spine_gamma=1.5,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.spine_head.weight.grad is not None
    assert torch.isfinite(model.spine_head.weight.grad).all()

    # Exported state_dict must load with strict=True into vanilla MediSegEdgeAttUNet
    base = MediSegEdgeAttUNet(3, 1, False, num_heads=4).eval()
    model.eval()
    base.load_state_dict(model.export_base_state_dict(), strict=True)
    with torch.no_grad():
        y_guided, _ = model.forward_with_aux(x)
        y_base = base(x)
    assert torch.allclose(y_guided, y_base, atol=1e-6)
