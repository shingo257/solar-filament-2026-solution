"""Unit and Ultralytics E2E tests for Unlabeled-Sibling Negative Shielding (v131)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from solar_filament.pipeline.sibling_shield_yolo import (
    build_sibling_shield_trainer_classes,
    compute_sibling_bce_weights,
    emit_sibling_shield_label_txt,
)
from solar_filament.pipeline.spine_yolo import build_spine_trainer_classes


def test_compute_sibling_bce_weights_shields_only_background_inside_sibling() -> None:
    sib_poly = np.array([[16, 16], [48, 16], [48, 48], [16, 48]], dtype=np.float32)
    # 3 anchors: (32, 32) inside sib as background; (32, 32) as positive; (96, 96) outside sib
    anchors_px = torch.tensor([[32.0, 32.0], [32.0, 32.0], [96.0, 96.0]])
    fg_mask = torch.tensor(
        [
            [False, True, False],  # image 0 has sibling poly
            [False, False, False],  # image 1 has NO sibling poly
        ]
    )
    w = compute_sibling_bce_weights(
        [[sib_poly], []],
        anchors_px,
        fg_mask,
        (128, 128),
        sib_neg_weight=0.20,
        dilate_cells=1,
    )
    assert abs(float(w[0, 0]) - 0.20) < 1e-6  # shielded unlabeled sibling!
    assert abs(float(w[0, 1]) - 1.00) < 1e-6  # positive anchor NEVER downweighted!
    assert abs(float(w[0, 2]) - 1.00) < 1e-6  # true solar disk background stays 1.0!
    assert torch.allclose(w[1], torch.ones_like(w[1]))


def test_sibling_shield_loss_weight_one_matches_base_loss_exactly(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(7)
    model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=1, verbose=False)
    model.args = get_cfg(overrides={"task": "segment", "overlap_mask": False})

    _, base_loss_cls, _ = build_spine_trainer_classes(aux_weight=0.0)
    _, sib1_cls, _ = build_sibling_shield_trainer_classes(sib_neg_weight=1.0)
    _, sib02_cls, trainer_cls = build_sibling_shield_trainer_classes(sib_neg_weight=0.20)

    lb = base_loss_cls(model)
    l1 = sib1_cls(model)
    l02 = sib02_cls(model)

    x = torch.randn(2, 3, 128, 128)
    preds = model(x)
    masks = torch.zeros((2, 32, 32), dtype=torch.float32)
    masks[0, 4:12, 4:12] = 1.0
    masks[1, 4:12, 4:12] = 1.0
    sib_poly = np.array([[64, 64], [104, 64], [104, 104], [64, 104]], dtype=np.float32)
    batch = {
        "img": x,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]),
        "masks": masks,
        "spine_segments_px": ([], []),
        "sibling_polys_px": [[sib_poly], []],
    }
    tot_b, _ = lb(preds, batch)
    tot_1, _ = l1(preds, batch)
    tot_02, _ = l02(preds, batch)
    assert torch.allclose(tot_b, tot_1, rtol=1e-6, atol=1e-6)
    # cls loss component (index 2) is strictly smaller when shielding negative sibling anchors
    assert float(tot_02[2].detach().item()) < float(tot_1[2].detach().item())

    # 1-epoch E2E training & inference check
    ds = tmp_path / "ds_sib"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for i in range(2):
            img = np.full((128, 128, 3), 120, dtype=np.uint8)
            cv2.rectangle(img, (20, 20), (60, 40), (30, 30, 30), -1)
            cv2.imwrite(str(ds / "images" / split / f"im_{i}.jpg"), img)
            pos = [np.array([[20, 20], [60, 20], [60, 40], [20, 40]], dtype=np.float64)]
            sib = (
                [np.array([[72, 72], [108, 72], [108, 96], [72, 96]], dtype=np.float64)]
                if split == "train"
                else []
            )
            (ds / "labels" / split / f"im_{i}.txt").write_text(
                emit_sibling_shield_label_txt(pos, sib, 128, 128), encoding="utf-8"
            )
    yaml_p = ds / "data.yaml"
    yaml_p.write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\nnc: 2\n"
        "names:\n  0: filament\n  1: sibling_ignore\n",
        encoding="utf-8",
    )
    trainer = trainer_cls(
        overrides={
            "model": "yolo11n-seg.yaml",
            "data": str(yaml_p),
            "epochs": 1,
            "imgsz": 128,
            "batch": 2,
            "device": "cpu",
            "workers": 0,
            "project": str(tmp_path / "runs"),
            "name": "sib_e2e",
            "plots": False,
            "val": True,
            "mosaic": 0.5,
        }
    )
    trainer.train()
    last_pt = tmp_path / "runs" / "sib_e2e" / "weights" / "last.pt"
    assert last_pt.exists()
    res = YOLO(str(last_pt)).predict(
        str(ds / "images" / "val" / "im_0.jpg"), imgsz=128, verbose=False
    )
    assert len(res) == 1
