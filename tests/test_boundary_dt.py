"""Characterization tests for the distance-transform boundary loss helpers."""

from __future__ import annotations

import numpy as np
import pytest
from solar_filament.pipeline.boundary_dt import (
    batch_signed_distance_maps,
    boundary_term,
    signed_distance_map,
)


def _disk(shape=(64, 64), center=(32, 32), radius=10):
    ys, xs = np.ogrid[: shape[0], : shape[1]]
    return ((ys - center[0]) ** 2 + (xs - center[1]) ** 2 <= radius**2).astype(np.uint8)


def test_empty_and_full_masks_have_no_boundary_signal() -> None:
    assert np.all(signed_distance_map(np.zeros((32, 32), np.uint8)) == 0)
    assert np.all(signed_distance_map(np.ones((32, 32), np.uint8)) == 0)


def test_sign_convention_inside_negative_outside_positive() -> None:
    phi = signed_distance_map(_disk())
    assert phi[32, 32] < 0  # GT の最深部
    assert phi[0, 0] > 0  # 最も遠い外部
    assert abs(phi[32, 32]) == pytest.approx(1.0) or abs(phi[0, 0]) == pytest.approx(1.0)


def test_range_is_normalized_to_unit() -> None:
    phi = signed_distance_map(_disk())
    assert phi.max() <= 1.0 and phi.min() >= -1.0
    assert abs(np.abs(phi).max() - 1.0) < 1e-6


def test_boundary_is_approximately_zero() -> None:
    phi = signed_distance_map(_disk())
    # ディスク縁(半径10)上の画素は 0 に近い
    assert abs(phi[22, 32]) < 0.2


def test_inverting_mask_flips_the_sign_pattern() -> None:
    mask = _disk()
    phi = signed_distance_map(mask)
    phi_inv = signed_distance_map(1 - mask)
    assert np.sign(phi[32, 32]) == -np.sign(phi_inv[32, 32])
    assert np.sign(phi[0, 0]) == -np.sign(phi_inv[0, 0])


def test_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        signed_distance_map(np.zeros((4, 4, 4)))


def test_batch_helper_stacks_and_drops_channel() -> None:
    masks = np.stack([_disk(), np.zeros((64, 64), np.uint8)])[:, None]
    phis = batch_signed_distance_maps(masks)
    assert phis.shape == (2, 64, 64)
    assert np.all(phis[1] == 0)


def test_boundary_term_orders_predictions_correctly() -> None:
    mask = _disk()
    phi = signed_distance_map(mask)
    perfect = mask.astype(np.float64)
    empty = np.zeros_like(mask, dtype=np.float64)
    everything = np.ones_like(mask, dtype=np.float64)
    shifted = np.roll(mask.astype(np.float64), 15, axis=0)  # ずれた予測
    assert boundary_term(perfect, phi) < boundary_term(empty, phi)
    assert boundary_term(empty, phi) == pytest.approx(0.0)
    assert boundary_term(everything, phi) > 0
    assert boundary_term(shifted, phi) > boundary_term(perfect, phi)


def test_boundary_term_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        boundary_term(np.zeros((8, 8)), np.zeros((4, 4)))


@pytest.mark.parametrize("mask", [np.full((3, 3), np.nan), np.full((3, 3), np.inf), np.full((3, 3), -1)])
def test_mask_rejects_invalid_values(mask):
    with pytest.raises(ValueError):
        signed_distance_map(mask)


def test_batch_rejects_multichannel_targets():
    with pytest.raises(ValueError):
        batch_signed_distance_maps(np.zeros((2, 3, 4, 4)))


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_boundary_term_rejects_invalid_probability(value):
    with pytest.raises(ValueError):
        boundary_term(np.full((4, 4), value), np.zeros((4, 4)))


def test_boundary_term_rejects_nonfinite_distance():
    with pytest.raises(ValueError):
        boundary_term(np.zeros((4, 4)), np.full((4, 4), np.nan))


def test_torch_boundary_gradient_pushes_inside_up_and_outside_down():
    import torch
    phi = signed_distance_map(_disk())
    logits = torch.zeros(phi.shape, requires_grad=True)
    (logits.sigmoid() * torch.from_numpy(phi)).mean().backward()
    assert logits.grad[32, 32] < 0
    assert logits.grad[0, 0] > 0
