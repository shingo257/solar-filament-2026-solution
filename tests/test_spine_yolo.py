"""CPU tests for spine_yolo integration glue."""

from __future__ import annotations

import numpy as np
import pytest

from solar_filament.pipeline.spine_yolo import (
    SPINE_CLS,
    SpineAuxLoss,
    SplitSpineSegments,
    build_p3_targets,
    build_spine_trainer_classes,
    compute_spine_ridge_score,
    emit_label_txt,
    extract_p3_spine_prob,
    selective_spine_rescue,
    spine_label_row,
    split_spine_batch,
)

IMG = 2048


def _instances(bboxes, segments, normalized: bool = True):
    ultralytics = pytest.importorskip("ultralytics")
    from ultralytics.utils.instance import Instances
    assert ultralytics is not None
    return Instances(np.asarray(bboxes, dtype=np.float32),
                     segments=np.asarray(segments, dtype=np.float32),
                     bbox_format="xywh", normalized=normalized)


def test_spine_label_row_normalizes():
    row = spine_label_row(np.array([[1024.0, 1024.0], [2048.0, 512.0]]), IMG, IMG)
    parts = row.split()
    assert parts[0] == str(SPINE_CLS)
    vals = [float(v) for v in parts[1:]]
    assert vals == pytest.approx([0.5, 0.5, 1.0, 0.25])


def test_spine_label_row_requires_two_points():
    with pytest.raises(ValueError):
        spine_label_row(np.array([[1.0, 2.0]]), IMG, IMG)


def test_emit_label_txt_mixes_classes():
    poly = np.array([[0, 0], [1024, 0], [1024, 1024]], dtype=float)
    spine = np.array([[0, 512], [1024, 512]], dtype=float)
    txt = emit_label_txt([poly], [spine], IMG, IMG)
    lines = txt.strip().split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith(f"{SPINE_CLS} ")


def test_emit_label_txt_skips_degenerate_spine():
    txt = emit_label_txt([], [np.array([[5.0, 5.0]])], IMG, IMG)
    assert txt == ""


def test_two_point_spine_passes_ultralytics_verify_image_label(tmp_path):
    """191 real MAGFiLO annotations have 2-point spines; must not emit 5-token rows."""
    pytest.importorskip("ultralytics")
    import cv2
    from ultralytics.data.utils import verify_image_label

    img_path = tmp_path / "obs.jpg"
    lb_path = tmp_path / "obs.txt"
    cv2.imwrite(str(img_path), np.zeros((128, 128, 3), dtype=np.uint8))
    poly = np.array([[20.0, 40.0], [80.0, 40.0], [80.0, 60.0], [20.0, 60.0]])
    two_pt_spine = np.array([[20.0, 50.0], [80.0, 50.0]])  # 2 points!
    lb_path.write_text(emit_label_txt([poly], [two_pt_spine], 128, 128), encoding="utf-8")

    res = verify_image_label((str(img_path), str(lb_path), "", False, 2, 0, 0, False))
    im_file, lb, _shape, segments, _kp, _nm, nf, _ne, nc_corrupt, msg = res
    assert nc_corrupt == 0, f"corrupt label: {msg}"
    assert nf == 1 and im_file is not None
    assert len(lb) == 2 and len(segments) == 2


def test_split_removes_spine_instances():
    instances = _instances(
        bboxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.5, 0.01], [0.1, 0.1, 0.1, 0.1]],
        segments=[[[0.4, 0.4], [0.6, 0.6]], [[0.1, 0.5], [0.9, 0.5]], [[0.05, 0.05], [0.15, 0.15]]],
    )
    labels = {"instances": instances, "cls": np.array([0, SPINE_CLS, 0], dtype=np.float32)}
    out = SplitSpineSegments()(labels)
    assert len(out["instances"].bboxes) == 2
    assert list(out["cls"]) == [0, 0]
    assert len(out["spine_segments_px"]) == 1
    np.testing.assert_allclose(out["spine_segments_px"][0],
                               [[0.1, 0.5], [0.9, 0.5]], rtol=1e-5)


def test_split_handles_no_spine():
    instances = _instances(bboxes=[[0.5, 0.5, 0.2, 0.2]],
                           segments=[[[0.4, 0.4], [0.6, 0.6]]])
    labels = {"instances": instances, "cls": np.array([0], dtype=np.float32)}
    out = SplitSpineSegments()(labels)
    assert len(out["instances"].bboxes) == 1
    assert out["spine_segments_px"] == []


def test_split_spine_batch_shapes():
    a = np.array([[10.0, 20.0], [30.0, 40.0]])
    batch = {"spine_segments_px": [[a], [], a]}
    got = split_spine_batch(batch)
    assert len(got) == 3 and got[0][0] is a and got[1] == [] and got[2] == [a]


def test_build_p3_targets_grid():
    spine = np.array([[0.0, 640.0], [1279.0, 640.0]])
    pos, valid = build_p3_targets([[spine]], (1280, 1280), grid=160, radius_cells=0)
    row = int(640.0 * 160 / 1280)
    assert pos.shape == (1, 160, 160) and valid.shape == pos.shape
    assert pos[0, row].all() and valid.all()


def test_spine_aux_loss_torch():
    import torch
    logits = torch.tensor([[[[30.0, -30.0], [-30.0, 30.0]]]])
    pos = np.array([[[[True, False], [False, True]]]])
    valid = np.ones_like(pos)
    loss = SpineAuxLoss.call(logits, pos, valid)
    assert float(loss) < 1e-4


def test_spine_aux_loss_masked():
    import torch
    logits = torch.zeros((1, 1, 2, 2))
    valid = np.array([[[[True, False], [True, True]]]])
    pos = np.zeros((1, 1, 2, 2), dtype=bool)
    loss = SpineAuxLoss.call(logits, pos, valid)
    assert float(loss) == pytest.approx(np.log(2.0), rel=1e-5)


def test_split_spine_batch_handles_collate_tuple():
    """YOLODataset.collate_fn uses zip(*values), producing a tuple of length B."""
    a = np.array([[10.0, 20.0], [30.0, 40.0]])
    b = np.array([[50.0, 60.0], [70.0, 80.0]])
    batch = {"spine_segments_px": ([a], [], [b])}
    got = split_spine_batch(batch)
    assert len(got) == 3
    assert got[0][0] is a and got[1] == [] and got[2][0] is b


def test_split_format_contract_2d_cls_and_denormalize():
    """Real Ultralytics Format requires cls shape (N, 1) and pixel-coord spine segments."""
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import Format

    instances = _instances(
        bboxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.8, 0.02]],
        segments=[
            [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
            [[0.1, 0.5], [0.9, 0.5], [0.9, 0.5], [0.1, 0.5]],
        ],
        normalized=True,
    )
    labels = {
        "img": np.zeros((128, 128, 3), dtype=np.uint8),
        "instances": instances,
        "cls": np.array([[0.0], [float(SPINE_CLS)]], dtype=np.float32),
        "ori_shape": (128, 128),
        "resized_shape": (128, 128),
    }
    split_out = SplitSpineSegments()(labels)
    assert split_out["cls"].shape == (1, 1)
    assert len(split_out["spine_segments_px"]) == 1
    # Even when instances entered normalized (augment=False / close_mosaic), spine_segments_px must be pixel coords
    np.testing.assert_allclose(
        split_out["spine_segments_px"][0][:2],
        [[12.8, 64.0], [115.2, 64.0]],
        rtol=1e-4,
    )
    formatted = Format(return_mask=True, mask_ratio=2, mask_overlap=False)(split_out)
    assert formatted["cls"].shape == (1, 1)
    assert formatted["masks"].shape == (1, 64, 64)


def test_spine_aux_loss_fp16_amp_finite():
    """Under AMP float16, clamp(1-1e-7) rounds to 1.0 causing log(0)=-inf unless computed in fp32."""
    import torch

    logits = torch.tensor([[[[30.0, -30.0], [-30.0, 30.0]]]], dtype=torch.float16, requires_grad=True)
    pos = np.array([[[[True, False], [False, True]]]])
    valid = np.ones_like(pos)
    loss = SpineAuxLoss.call(logits, pos, valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_spine_ridge_score_and_selective_rescue():
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[1016:1032, 400:800] = True
    spine_prob = np.full((160, 160), 0.05, dtype=np.float32)
    spine_prob[80, 30:63] = 0.90  # centerline ridge at y ~ 1024
    ridge = compute_spine_ridge_score(mask, spine_prob, top_quantile=0.75)
    assert ridge > 0.70

    kept = selective_spine_rescue(
        conf=[0.42, 0.19, 0.21, 0.10],
        areas=[250, 120, 120, 150],
        spine_scores=[0.20, 0.82, 0.25, 0.95],
        primary_conf=0.26,
        rescue_floor=0.14,
        min_spine=0.65,
        min_area=20,
    )
    assert kept == [0, 1]


def test_end_to_end_ultralytics_spine_trainer(tmp_path):
    """Real 1-epoch Ultralytics train + val + checkpoint reload + P3 spine heatmap extraction."""
    pytest.importorskip("ultralytics")
    import cv2
    from ultralytics import YOLO

    ds = tmp_path / "ds"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for idx in range(2):
            img = np.full((128, 128, 3), 120, dtype=np.uint8)
            cv2.rectangle(img, (24, 56), (104, 72), (30, 30, 30), -1)
            img_path = ds / "images" / split / f"s{idx}.jpg"
            cv2.imwrite(str(img_path), img)
            poly = np.array([[24, 56], [104, 56], [104, 72], [24, 72]], dtype=float)
            spine = np.array([[24, 64], [64, 64], [104, 64]], dtype=float)
            txt = (
                emit_label_txt([poly], [spine], 128, 128)
                if split == "train"
                else emit_label_txt([poly], [], 128, 128)
            )
            (ds / "labels" / split / f"s{idx}.txt").write_text(txt, encoding="utf-8")

    yaml_path = ds / "data.yaml"
    yaml_path.write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\nnc: 2\n"
        "names:\n  0: filament\n  1: spine_aux\n",
        encoding="utf-8",
    )

    _, _, trainer_cls = build_spine_trainer_classes(aux_weight=0.25, radius_cells=1)
    trainer = trainer_cls(
        overrides={
            "model": "yolo11n-seg.yaml",
            "data": str(yaml_path),
            "epochs": 1,
            "imgsz": 128,
            "batch": 2,
            "device": "cpu",
            "workers": 0,
            "project": str(tmp_path / "runs"),
            "name": "smoke_b",
            "plots": False,
            "amp": False,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "exist_ok": True,
        }
    )
    trainer.train()
    last_pt = tmp_path / "runs" / "smoke_b" / "weights" / "last.pt"
    assert last_pt.exists()

    reloaded = YOLO(str(last_pt))
    assert reloaded.model.nc == 1
    assert hasattr(reloaded.model, "spine_head")
    res, spine_prob = extract_p3_spine_prob(
        reloaded, str(ds / "images" / "val" / "s0.jpg"), imgsz=128, conf=0.01
    )
    assert res.orig_shape == (128, 128)
    assert spine_prob is not None and spine_prob.shape == (16, 16)
    assert np.isfinite(spine_prob).all()


