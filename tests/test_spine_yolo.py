"""CPU tests for spine_yolo integration glue."""

from __future__ import annotations

import numpy as np
import pytest

from solar_filament.pipeline.spine_yolo import (
    SPINE_CLS,
    SpineAuxLoss,
    SplitSpineSegments,
    build_p3_targets,
    emit_label_txt,
    spine_label_row,
    split_spine_batch,
)

IMG = 2048


def _instances(bboxes, segments):
    from ultralytics.utils.instance import Instances
    return Instances(np.asarray(bboxes, dtype=np.float32),
                     segments=np.asarray(segments, dtype=np.float32),
                     bbox_format="xywh", normalized=True)


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
