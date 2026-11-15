"""Unit and Ultralytics E2E tests for Small-Instance TAL Anchor Equalization (v130)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from solar_filament.pipeline.small_tal_yolo import (
    build_pure_small_tal_trainer_classes,
    build_small_tal_trainer_classes,
    compute_anchor_equalization_weights,
)
from solar_filament.pipeline.spine_yolo import (
    build_spine_trainer_classes,
    emit_label_txt,
)


def test_anchor_equalization_boosts_small_and_conserves_mass() -> None:
    fg_mask = torch.zeros((2, 20), dtype=torch.bool)
    target_gt_idx = torch.zeros((2, 20), dtype=torch.long)
    target_scores = torch.zeros((2, 20, 1), dtype=torch.float32)

    # Image 0: GT 0 is large (10 anchors, score 0.8), GT 1 is tiny (1 anchor, score 0.4)
    fg_mask[0, 0:10] = True
    target_gt_idx[0, 0:10] = 0
    target_scores[0, 0:10, 0] = 0.8

    fg_mask[0, 12] = True
    target_gt_idx[0, 12] = 1
    target_scores[0, 12, 0] = 0.4

    # Image 1: GT 0 has 2 anchors (score 0.5)
    fg_mask[1, 3:5] = True
    target_gt_idx[1, 3:5] = 0
    target_scores[1, 3:5, 0] = 0.5

    w_cls, w_seg = compute_anchor_equalization_weights(
        fg_mask, target_gt_idx, target_scores, topk=10, max_boost=4.0
    )
    # Background anchors strictly 1.0
    assert torch.allclose(w_cls[~fg_mask], torch.ones_like(w_cls[~fg_mask]))
    assert torch.allclose(w_seg[~fg_mask], torch.ones_like(w_seg[~fg_mask]))

    # Mass conservation for seg and cls/box
    assert torch.isclose(w_seg[fg_mask].sum(), fg_mask.float().sum(), rtol=1e-5)
    weighted_ts = (target_scores * w_cls.unsqueeze(-1)).sum()
    assert torch.isclose(weighted_ts, target_scores.sum(), rtol=1e-5)

    # 1-anchor tiny GT gets strictly 4x the weight of 10-anchor large GT
    ratio_cls = float(w_cls[0, 12] / w_cls[0, 0])
    ratio_seg = float(w_seg[0, 12] / w_seg[0, 0])
    assert abs(ratio_cls - 4.0) < 1e-5
    assert abs(ratio_seg - 4.0) < 1e-5


def test_anchor_equalization_max_boost_one_is_identity() -> None:
    fg_mask = torch.tensor([[True, True, False, True]])
    target_gt_idx = torch.tensor([[0, 0, 0, 1]])
    target_scores = torch.tensor([[[0.7], [0.6], [0.0], [0.4]]])
    w_cls, w_seg = compute_anchor_equalization_weights(
        fg_mask, target_gt_idx, target_scores, topk=10, max_boost=1.0
    )
    assert torch.allclose(w_cls, torch.ones_like(w_cls))
    assert torch.allclose(w_seg, torch.ones_like(w_seg))


def test_small_tal_trainer_one_epoch_e2e_and_max_boost_one_parity(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    ds = tmp_path / "ds"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for i in range(2):
            img = np.full((128, 128, 3), 120, dtype=np.uint8)
            cv2.rectangle(img, (20, 24), (80, 44), (30, 30, 30), -1)
            cv2.rectangle(img, (92, 92), (100, 100), (25, 25, 25), -1)
            cv2.imwrite(str(ds / "images" / split / f"im_{i}.jpg"), img)
            poly_large = np.array([[20, 24], [80, 24], [80, 44], [20, 44]], dtype=np.float64)
            poly_small = np.array([[92, 92], [100, 92], [100, 100], [92, 100]], dtype=np.float64)
            sp_large = np.array([[22, 34], [78, 34]], dtype=np.float64)
            sp_small = np.array([[93, 96], [99, 96]], dtype=np.float64)
            txt = emit_label_txt([poly_large, poly_small], [sp_large, sp_small], 128, 128)
            (ds / "labels" / split / f"im_{i}.txt").write_text(txt, encoding="utf-8")

    yaml_p = ds / "data.yaml"
    yaml_p.write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\nnc: 2\n"
        "names:\n  0: filament\n  1: spine_aux\n",
        encoding="utf-8",
    )

    _, _, trainer_cls = build_small_tal_trainer_classes(max_boost=4.0, aux_weight=0.0)
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
            "name": "small_tal_e2e",
            "plots": False,
            "val": True,
            "mosaic": 0.0,
        }
    )
    trainer.train()
    last_pt = tmp_path / "runs" / "small_tal_e2e" / "weights" / "last.pt"
    assert last_pt.exists()
    loaded = YOLO(str(last_pt))
    res = loaded.predict(str(ds / "images" / "val" / "im_0.jpg"), imgsz=128, verbose=False)
    assert len(res) == 1


def test_small_tal_loss_max_boost_one_matches_base_loss_exactly() -> None:
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(42)
    model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=1, verbose=False)
    model.args = get_cfg(overrides={"task": "segment", "overlap_mask": False})

    _, base_loss_cls, _ = build_spine_trainer_classes(aux_weight=0.0)
    _, tal_loss_one_cls, _ = build_small_tal_trainer_classes(max_boost=1.0, aux_weight=0.0)
    _, tal_loss_four_cls, _ = build_small_tal_trainer_classes(max_boost=4.0, aux_weight=0.0)

    loss_base = base_loss_cls(model)
    loss_one = tal_loss_one_cls(model)
    loss_four = tal_loss_four_cls(model)

    x = torch.randn(2, 3, 128, 128)
    preds = model(x)
    masks = torch.zeros((2, 32, 32), dtype=torch.float32)
    masks[0, 4:20, 4:24] = 1.0  # large object
    masks[1, 24:26, 24:26] = 1.0  # tiny object
    batch = {
        "img": x,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor(
            [
                [0.4375, 0.3750, 0.6250, 0.5000],
                [0.78125, 0.78125, 0.0625, 0.0625],
            ],
            dtype=torch.float32,
        ),
        "masks": masks,
        "spine_segments_px": ([], []),
    }

    tot_b, items_b = loss_base(preds, batch)
    tot_1, items_1 = loss_one(preds, batch)
    tot_4, _items_4 = loss_four(preds, batch)

    assert torch.allclose(tot_b, tot_1, rtol=1e-6, atol=1e-6)
    for k in items_b:
        assert torch.allclose(items_b[k], items_1[k], rtol=1e-6, atol=1e-6)
    assert torch.isfinite(tot_4).all()
    assert not torch.allclose(tot_b, tot_4)


def test_pure_small_tal_single_cls_parity_and_train(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.loss import v8SegmentationLoss

    torch.manual_seed(99)
    model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=1, verbose=False)
    model.args = get_cfg(overrides={"task": "segment", "overlap_mask": False})
    loss_v8 = v8SegmentationLoss(model)
    loss_p1_cls, _ = build_pure_small_tal_trainer_classes(max_boost=1.0)
    loss_p4_cls, trainer_cls = build_pure_small_tal_trainer_classes(max_boost=4.0)

    x = torch.randn(2, 3, 128, 128)
    preds = model(x)
    masks = torch.zeros((2, 32, 32), dtype=torch.float32)
    masks[0, 4:20, 4:24] = 1.0
    masks[1, 24:26, 24:26] = 1.0
    batch = {
        "img": x,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor(
            [[0.4375, 0.3750, 0.6250, 0.5000], [0.78125, 0.78125, 0.0625, 0.0625]],
            dtype=torch.float32,
        ),
        "masks": masks,
    }
    tb, _ = loss_v8(preds, batch)
    t1, _ = loss_p1_cls(model)(preds, batch)
    t4, _ = loss_p4_cls(model)(preds, batch)
    assert torch.allclose(tb, t1, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(tb, t4)

    ds = tmp_path / "ds_pure"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for i in range(2):
            img = np.full((128, 128, 3), 120, dtype=np.uint8)
            cv2.rectangle(img, (20, 24), (80, 44), (30, 30, 30), -1)
            cv2.imwrite(str(ds / "images" / split / f"im_{i}.jpg"), img)
            (ds / "labels" / split / f"im_{i}.txt").write_text(
                "0 0.156250 0.187500 0.625000 0.187500 0.625000 0.343750 0.156250 0.343750\n",
                encoding="utf-8",
            )
    yaml_p = ds / "data.yaml"
    yaml_p.write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n",
        encoding="utf-8",
    )
    YOLO("yolo11n-seg.yaml").train(
        trainer=trainer_cls,
        data=str(yaml_p),
        epochs=1,
        imgsz=128,
        batch=2,
        device="cpu",
        workers=0,
        single_cls=True,
        project=str(tmp_path / "runs"),
        name="pure_tal",
        plots=False,
        mosaic=0.0,
    )
    last_pt = tmp_path / "runs" / "pure_tal" / "weights" / "last.pt"
    assert last_pt.exists()
    res = YOLO(str(last_pt)).predict(str(ds / "images" / "val" / "im_0.jpg"), imgsz=128, verbose=False)
    assert len(res) == 1


def test_combined_spine_aux_and_small_tal_e2e(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    ds = tmp_path / "ds_combo"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for i in range(2):
            img = np.full((128, 128, 3), 120, dtype=np.uint8)
            cv2.rectangle(img, (20, 24), (80, 44), (30, 30, 30), -1)
            cv2.rectangle(img, (92, 92), (100, 100), (25, 25, 25), -1)
            cv2.imwrite(str(ds / "images" / split / f"im_{i}.jpg"), img)
            poly_large = np.array([[20, 24], [80, 24], [80, 44], [20, 44]], dtype=np.float64)
            poly_small = np.array([[92, 92], [100, 92], [100, 100], [92, 100]], dtype=np.float64)
            sp_large = np.array([[22, 34], [78, 34]], dtype=np.float64)
            sp_small = np.array([[93, 96], [99, 96]], dtype=np.float64)
            txt = emit_label_txt([poly_large, poly_small], [sp_large, sp_small], 128, 128)
            (ds / "labels" / split / f"im_{i}.txt").write_text(txt, encoding="utf-8")

    yaml_p = ds / "data.yaml"
    yaml_p.write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\nnc: 2\n"
        "names:\n  0: filament\n  1: spine_aux\n",
        encoding="utf-8",
    )

    _, _, trainer_cls = build_small_tal_trainer_classes(max_boost=4.0, aux_weight=0.5)
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
            "name": "combo_e2e",
            "plots": False,
            "val": True,
            "mosaic": 0.0,
        }
    )
    trainer.train()
    assert hasattr(trainer.model, "spine_head")
    last_pt = tmp_path / "runs" / "combo_e2e" / "weights" / "last.pt"
    assert last_pt.exists()
    res = YOLO(str(last_pt)).predict(str(ds / "images" / "val" / "im_0.jpg"), imgsz=128, verbose=False)
    assert len(res) == 1
