"""Contract and behavioral tests for the v115 Mask R-CNN family notebook (§2.93).

v115 introduces a genuinely different detector family (torchvision
Mask R-CNN ResNet50-FPN-v2, COCO-pretrained) to answer two questions the
YOLO-only campaign could not: (1) is a different family stronger, and
(2) does a cross-detector dedup pool (the lever other teams reported)
help our accepted pipeline. Frozen eval cells stay verbatim from the v96
third_f0 notebook; the run cell mirrors evaluate_yolo's cache/scoring.
"""

import ast
import json
import random
import re
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

from scratch import build_v115_mrcnn_colab as v115


def test_cfg_provides_all_attrs_consumed_by_frozen_train_and_run_cells(nb):
    """cells (2/3/4/5/6) reference CFG.* — every attribute they need must be
    assigned in the CFG class in cell 1."""
    config = _src(nb["cells"][1])
    assigned = set(re.findall(r"^\s+(\w+)\s*=", config, re.MULTILINE))
    needed = set()
    for idx in (2, 3, 4, 5, 6):
        needed |= set(re.findall(r"CFG\.(\w+)", _src(nb["cells"][idx])))
    missing = needed - assigned
    assert not missing, f"CFG attributes missing: {missing}"


def _src(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(v115.OUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nb_source():
    return json.loads(v115.SOURCE.read_text(encoding="utf-8"))


def test_all_cells_parse(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        stripped = "\n".join(
            line for line in _src(cell).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(stripped)


def test_config_declares_v115_arms_and_recipe(nb):
    config = _src(nb["cells"][1])
    for arm in ("'mrcnn_raw'", "'mrcnn_ref'", "'union_yolo_mrcnn'"):
        assert arm in config, arm
    for key in ("MRCNN_EPOCHS = 10", "MRCNN_BATCH = 2", "MRCNN_LR = 1e-4",
                "MRCNN_MIN_SIZE = 1280", "UNION_MERGE_IOU = 0.30",
                "TRAIN_FOLDS = (0, 1)", "FOLD = 2"):
        assert key in config, key
    assert "v115_mrcnn_colab_manifest.json" in config
    assert "GATE_DELTA = 0.010" in config
    for kept in ("PRIMARY_CONF = 0.26", "CROP_SIZE = 320",
                 "REFINER_THRESHOLD = 0.45", "PREDICT_CONF_FLOOR = 0.08",
                 "DIAG_CONF_FLOOR = 0.02", "MIN_AREA = 20"):
        assert kept in config, kept


def test_mrcnn_training_cell(nb):
    cell = _src(nb["cells"][3])
    assert "maskrcnn_resnet50_fpn_v2" in cell
    assert 'weights="DEFAULT"' in cell
    assert "num_classes = CFG.MRCNN_NCLASS + 1" in cell
    assert "FastRCNNPredictor(in_features, num_classes)" in cell
    assert "MaskRCNNPredictor(" in cell and "256, num_classes)" in cell
    assert "model.roi_heads.score_thresh = CFG.DIAG_CONF_FLOOR" in cell
    assert "model.transform.min_size" in cell
    # aug: dihedral only; boxes & masks filtered in lockstep
    assert "_boxes_of" in cell
    assert "boxes, masks = _boxes_of(masks, iw, ih)" in cell
    # inclusive boxes clamped inside image bounds (device-side assert guard)
    assert "float(width) - 1.0" in cell and "float(height) - 1.0" in cell
    assert "np.rot90" in cell
    # failing-batch diagnostics for CUDA assert localization
    assert '"failed_batch"' in cell
    # OOM fallback + resume guard + ckpt provenance
    assert "torch.cuda.OutOfMemoryError" in cell
    assert "TRAINING_COMPLETE" in cell
    assert "mrcnn_fpnv2_fold" in _src(nb["cells"][1])


def test_mrcnn_predictor_classes_empty_mask_filter_and_oom_fallback(tmp_path):
    """Behavioral guards without requiring torchvision in CPU CI:
    1) Both FastRCNNPredictor & MaskRCNNPredictor receive num_classes = MRCNN_NCLASS + 1 = 2
       (preventing IndexError when indexing label=1 into mask_logits[:, label]),
       and roi_heads.score_thresh == DIAG_CONF_FLOOR (0.02).
    2) _boxes_of / mrcnn_sample drops empty masks in lockstep with boxes.
    3) train_mrcnn falls back once from batch=2 to batch=1 on CUDA OOM and
       preserves prior stats on resume.
    """
    calls = {}

    class StubFastPredictor(torch.nn.Module):
        def __init__(self, in_features, num_classes):
            super().__init__()
            calls["fast_num_classes"] = num_classes
            self.cls_score = torch.nn.Linear(in_features, num_classes)

    class StubMaskPredictor(torch.nn.Module):
        def __init__(self, in_channels, dim_reduced, num_classes):
            super().__init__()
            calls["mask_num_classes"] = num_classes
            self.conv5_mask = SimpleNamespace(in_channels=in_channels)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1)

    class StubMRCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1))
            self.roi_heads = SimpleNamespace(
                box_predictor=SimpleNamespace(cls_score=SimpleNamespace(in_features=16)),
                mask_predictor=SimpleNamespace(conv5_mask=SimpleNamespace(in_channels=16)),
                score_thresh=0.05,
            )
            self.transform = SimpleNamespace(min_size=(800,), max_size=1333)

        def cuda(self):
            return self

        def forward(self, imgs, tgts=None):
            for t in tgts or []:
                max_lbl = int(t["labels"].max())
                assert max_lbl < calls["mask_num_classes"], (
                    f"label {max_lbl} out of bounds for mask_num_classes={calls['mask_num_classes']}"
                )
                assert len(t["boxes"]) == len(t["labels"]) == t["masks"].shape[0]
            if len(imgs) > 1 and not calls.get("oom_raised"):
                calls["oom_raised"] = True
                raise torch.cuda.OutOfMemoryError("synthetic OOM on batch=2")
            return {"loss_box": self.param * 0.5, "loss_mask": self.param * 0.5}

    img_dir = tmp_path / "preproc"
    img_dir.mkdir()
    for fname in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(img_dir / fname), np.zeros((64, 64, 3), dtype=np.uint8))

    empty_rle = mask_utils.encode(np.zeros((64, 64), dtype=np.uint8, order="F"))
    fg = np.zeros((64, 64), dtype=np.uint8, order="F")
    fg[10:22, 14:30] = 1
    fg_rle = mask_utils.encode(fg)

    cfg = SimpleNamespace(
        SEED=3407,
        FOLD=2,
        MRCNN_NCLASS=1,
        MRCNN_MIN_SIZE=1280,
        DIAG_CONF_FLOOR=0.02,
        PREPROC_DIR=img_dir,
        MRCNN_CKPT=tmp_path / "runs" / "mrcnn_last.pt",
        MRCNN_EPOCHS=1,
        MRCNN_BATCH=2,
        MRCNN_LR=1e-4,
        MRCNN_WD=1e-4,
    )

    orig_to = torch.Tensor.to

    def _monkey_to(self, dev, *a, **kw):
        return self if dev == "cuda" else orig_to(self, dev, *a, **kw)

    torch.Tensor.to = _monkey_to
    try:
        src = "\n".join(
            line for line in v115.CELL3.splitlines()
            if not line.startswith("from torchvision.")
        )
        ns = {
            "CFG": cfg,
            "np": np,
            "cv2": cv2,
            "torch": torch,
            "random": random,
            "time": time,
            "tqdm": lambda it, **kw: it,
            "mask_utils": mask_utils,
            "to_rle": lambda ann, h, w: ann["rle"],
            "images": {1: {"height": 64, "width": 64}, 2: {"height": 64, "width": 64}},
            "annotations": {
                1: [{"rle": empty_rle, "iscrowd": 0}, {"rle": fg_rle, "iscrowd": 0}],
                2: [{"rle": fg_rle, "iscrowd": 0}],
            },
            "fit_files": ["a.jpg", "b.jpg"],
            "file_to_ids": {"a.jpg": [1], "b.jpg": [2]},
            "maskrcnn_resnet50_fpn_v2": lambda weights=None: StubMRCNN(),
            "FastRCNNPredictor": StubFastPredictor,
            "MaskRCNNPredictor": StubMaskPredictor,
        }
        exec(compile(src, "<v115_cell3>", "exec"), ns)  # noqa: S102

        m = ns["build_mrcnn"]()
        assert calls["fast_num_classes"] == 2
        assert calls["mask_num_classes"] == 2
        assert m.roi_heads.score_thresh == pytest.approx(0.02)

        # Empty mask in image 1 must be dropped together with its box
        _, tgt = ns["mrcnn_sample"]("a.jpg", 1)
        assert tgt["boxes"].shape == (1, 4)
        assert tgt["labels"].tolist() == [1]
        assert tgt["masks"].shape == (1, 64, 64)
        assert int(tgt["masks"][0].sum()) == 12 * 16

        # OOM fallback batch=2 -> batch=1, then resume preserves stats
        ckpt, stats = ns["train_mrcnn"]()
        assert ckpt.exists()
        assert stats["batch_used"] == 1
        assert stats["oom_fallback"] is True
        assert stats["train_samples"] == 2
        _, resumed_stats = ns["train_mrcnn"](stats)
        assert resumed_stats["reused"] is True
        assert resumed_stats["train_samples"] == 2
        assert resumed_stats["batch_used"] == 1
    finally:
        torch.Tensor.to = orig_to


def test_merge_union_caches_rle_dedup_and_refined_reuse():
    # Extract merge_union_caches from CELL6 and verify exact RLE dedup + refined reuse
    tree = ast.parse(v115.CELL6)
    fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "merge_union_caches")
    mod = ast.Module(body=[fn_node], type_ignores=[])
    ns = {
        "CFG": SimpleNamespace(UNION_MERGE_IOU=0.30),
        "np": np,
        "mask_utils": mask_utils,
        "tqdm": lambda it, **kw: it,
    }
    exec(compile(mod, "<merge_union>", "exec"), ns)  # noqa: S102
    merge_union = ns["merge_union_caches"]

    m1 = np.zeros((64, 64), dtype=np.uint8, order="F")
    m1[5:25, 5:25] = 1
    m2 = np.zeros((64, 64), dtype=np.uint8, order="F")
    m2[6:25, 6:25] = 1  # IoU > 0.30 with m1 -> should be dropped
    m3 = np.zeros((64, 64), dtype=np.uint8, order="F")
    m3[40:55, 40:55] = 1  # disjoint -> should be kept

    r1, r2, r3 = mask_utils.encode(m1), mask_utils.encode(m2), mask_utils.encode(m3)
    cache_a = [("img0.jpg", np.array([0.80], np.float32), [float(m1.sum())], [r1], ["ref_m1"])]
    cache_b = [
        ("img0.jpg", np.array([0.65, 0.50], np.float32),
         [float(m2.sum()), float(m3.sum())], [r2, r3], ["ref_m2", "ref_m3"])
    ]
    merged = merge_union(cache_a, cache_b, "union")
    name, conf, areas, raw_rles, refined = merged[0]
    assert name == "img0.jpg"
    assert np.allclose(conf, [0.80, 0.50])
    assert areas == [float(m1.sum()), float(m3.sum())]
    assert raw_rles == [r1, r3]
    assert refined == ["ref_m1", "ref_m3"]


def test_eval_mirrors_frozen_scoring(nb):
    run = _src(nb["cells"][6])
    # cache structure identical to evaluate_yolo: (name, conf, areas, raw_rles, refined)
    assert "refine_proposal(image, m)" in run
    assert "CFG.PREDICT_CONF_FLOOR" in run
    assert "CFG.DIAG_CONF_FLOOR" in run
    # scoring path: keep conf>=c & MIN_AREA → ownership → pooled PQ
    assert "areas[i] >= CFG.MIN_AREA" in run
    assert "apply_ownership" in run
    assert "file_to_ids[name]" in run
    assert "PanopticQuality" in run
    # arm wiring: raw uses raw_rles, ref/union use refined
    assert '"mrcnn_raw": False' in run
    assert '"mrcnn_ref": True' in run
    # union merge: conf-desc greedy IoU>0.30 via RLE IoU + refined reuse
    assert "CFG.UNION_MERGE_IOU" in run
    assert "pool.sort(key=lambda t: -t[0])" in run
    assert "mask_utils.iou" in run
    # caches must NOT be in manifest (non-JSON types)
    assert 'manifest["caches"]' not in run
    assert "CACHES = {}" in run


def test_run_cell_contract_and_verdict(nb):
    run = _src(nb["cells"][6])
    assert '"experiment_id": "v115_mrcnn"' in run
    assert "§2.93" in run
    assert "maskrcnn_resnet50_fpn_v2" in run
    assert "fold0 honest replication" in run
    assert "promote_fold0_replication" in run
    assert "reject_v115_fold2" in run
    # prod baseline still measured by the FROZEN evaluate_yolo (paired)
    assert 'evaluate_yolo(CFG.PROD_YOLO, "prod")' in run


def test_frozen_cells_unchanged(nb, nb_source):
    for idx in (2, 4, 5):
        assert _src(nb["cells"][idx]) == _src(nb_source["cells"][idx]), f"cell {idx} drifted"


def test_header_declares_fixed_gate_and_caveats(nb):
    hdr = _src(nb["cells"][0])
    assert "§2.93" in hdr
    assert "+0.010" in hdr
    assert "mrcnn_raw" in hdr and "union_yolo_mrcnn" in hdr
    assert "未校正" in hdr or "caveat" in hdr.lower()
    assert "folds(1,2)" in hdr
