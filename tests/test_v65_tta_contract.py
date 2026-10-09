"""CPU contracts for exploratory v65; no detector/GPU or scientific acceptance."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v65", ROOT / "scratch/build_v65_tta_consensus.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def execute_generated(source, env):
    # Only repository-owned notebook source is executed in these CPU tests.
    exec(source, env)  # noqa: S102


def helpers():
    env = {"np": np}
    execute_generated(builder.GUARDS, env)
    return env


@pytest.mark.parametrize("labels,reason", [([], "empty"), ([False] * 3, "negative"),
                                        ([True] * 3, "positive")])
def test_degenerate_training_completes_with_explicit_fallback(labels, reason):
    env = helpers()
    x = np.zeros((len(labels), 15))
    x[:, 0] = .26
    model, status = env["fit_tta_selector"](x, np.array(labels, dtype=bool))
    assert reason in status["reason"].lower()
    probability = model.predict_proba(x)
    assert probability.shape == (len(labels), 2)
    assert np.allclose(probability[:, 1], x[:, 0])
    importance = env["selector_importance"](model, np.zeros((2, 15)), np.array([False, True]))
    assert importance["values"] is None
    assert importance["reason"]


def test_feature_matrix_preserves_empty_dimensions_and_feature_order():
    env = helpers()
    x, y = env["feature_matrix"]([])
    assert x.shape == (0, 15)
    assert y.shape == (0,) and y.dtype == np.bool_
    keys = env["FEATURE_KEYS"]
    assert len(keys) == len(set(keys)) == 15
    features = dict(zip(keys, range(15)))
    x, y = env["feature_matrix"]([{"props": [{"feat": features, "label_hit": True}]}])
    assert x.tolist() == [list(range(15))]
    assert y.tolist() == [True]


@pytest.mark.parametrize("shape", [(1, 14), (1, 16), (15,)])
def test_selector_rejects_wrong_feature_dimensions(shape):
    with pytest.raises(ValueError):
        helpers()["fit_tta_selector"](np.zeros(shape), np.zeros(1, dtype=bool))


def test_selector_rejects_nonfinite_features():
    x = np.zeros((2, 15))
    x[0, 2] = np.nan
    with pytest.raises(ValueError):
        helpers()["fit_tta_selector"](x, np.array([False, True]))


def test_real_cpu_selector_and_degenerate_importance():
    from threadpoolctl import threadpool_limits

    env = helpers()
    x = np.zeros((40, 15))
    x[20:, 0] = 1
    y = np.array([False] * 20 + [True] * 20)
    with threadpool_limits(limits=1):
        model, status = env["fit_tta_selector"](x, y)
        assert status["reason"] is None
        p = model.predict_proba(x)
        assert p[20, 1] > p[0, 1]
        for xb, yb in [(x[:0], y[:0]), (x[:3], y[:3]), (x[20:23], y[20:23])]:
            importance = env["selector_importance"](model, xb, yb)
            assert importance["values"] is None and importance["reason"]
        importance = env["selector_importance"](model, x, y)
    assert importance["reason"] is None
    assert set(importance["values"]) == set(env["FEATURE_KEYS"])
    assert np.isfinite(list(importance["values"].values())).all()


def test_generated_cells_match_saved_output_and_remove_stale_reuse():
    for fold in range(3):
        for cell in builder.build_cells(fold):
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
    saved = json.loads((ROOT / "solar_filament_v65_tta_consensus_fold2_colab.ipynb").read_text(encoding="utf-8"))
    assert saved["cells"] == builder.notebook(2)["cells"]
    assert f"PROTOCOL_SHA256 = '{builder.PROTOCOL_SHA256}'" in saved["cells"][1]["source"]
    source = "\n".join(c["source"] for c in saved["cells"])
    assert "pickle" not in source
    assert "GRID_FILE" not in source
    assert "!pip" not in source
    assert "val_grid" not in source
    assert "INPUT_CONTRACT = require_selector_contract" in saved["cells"][2]["source"]
    assert "torch.load(" not in saved["cells"][1]["source"] + saved["cells"][2]["source"]
    assert "torch.load(" in saved["cells"][3]["source"]
    assert all(not c.get("outputs") for c in saved["cells"])


def test_baseline_drift_prevents_fitting_and_manifest_publication():
    from types import SimpleNamespace

    env = helpers()
    published = []
    env.update({"val_rows": [], "CFG": SimpleNamespace(CONF=.26),
                "INPUT_CONTRACT": {"baseline": {"pq": .4, "tolerance": .001}},
                "publish_manifest": lambda *args: published.append(args),
                "PanopticQuality": type("Metric", (), {"summary": lambda self: {
                    "pq": 0., "sq": 0., "rq": 0., "tp": 0, "fp": 0, "fn": 0, "denominator": 0.}})})
    with pytest.raises(ValueError, match="Baseline reproduction"):
        execute_generated(builder.EVALUATE, env)
    assert not published
    assert "selector" not in env


@pytest.fixture
def input_environment(tmp_path):
    import hashlib
    import shutil
    import sys
    from collections import defaultdict
    from datetime import datetime, timezone
    from types import SimpleNamespace

    import cv2
    import pandas as pd
    from pycocotools import mask as mask_utils

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    def mapping_digest(data):
        return digest(json.dumps(data, sort_keys=True).encode())

    names = ["a.png", "b.png", "held.png"]
    annotation = json.dumps({"images": [{"id": i, "file_name": name, "height": 2048, "width": 2048}
        for i, name in enumerate(names)] + [{"id": 9, "file_name": "held.png", "height": 2048, "width": 2048}],
        "annotations": []}).encode()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_bytes(annotation)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    fold_bytes = b"file_name,fold\na.png,0\nb.png,1\nheld.png,2\n"
    (artifacts / "grouped_folds.csv").write_bytes(fold_bytes)
    detector = tmp_path / "detector.pt"
    detector.write_bytes(b"detector fixture")
    (artifacts / "mediseg_refiner_fold2_resume.pt").write_bytes(b"refiner fixture")
    cache, staged = tmp_path / "cache", tmp_path / "staged"
    cache.mkdir()
    staged.mkdir()
    image = np.zeros((2048, 2048, 3), np.uint8)
    image[5, 7] = [10, 20, 30]
    image_bytes = cv2.imencode(".png", image)[1].tobytes()
    for name in names:
        (cache / name).write_bytes(image_bytes)
        (staged / name).write_bytes(b"stale local image")
    config = {"YOLO_IMGSZ": 1280, "CROP_SIZE": 320, "CROP_CONTEXT": 1.65,
        "CONF": .26, "PREDICT_CONF_FLOOR": .05, "PREDICT_IOU": .55,
        "REFINER_THRESHOLD": .45, "DROP_THRESHOLD": .90, "MIN_AREA": 1,
        "VIEWS": [("id", 1280, False), ("hflip", 1280, True), ("big", 1536, False)],
        "TTA_MATCH_IOU": .3, "CONF_GRID": (.05, .08, .12, .20, .26, .35),
        "PKEEP_GRID": (0., .3, .5, .7, .85, .93), "SUPPORT_GRID": (0, 1, 2), "SPLIT_SEED": 3407}
    partitions = {"train_a": ["a.png"], "train_b": ["b.png"], "heldout": ["held.png"]}
    versions = dict.fromkeys(("numpy", "pandas", "scipy", "torch", "opencv-python",
        "ultralytics", "pycocotools", "scikit-learn"), "fixture")
    expected = {"schema_version": 1, "fold": 2, "protocol_sha256": builder.PROTOCOL_SHA256,
        "detector_sha256": digest(b"detector fixture"), "refiner_sha256": digest(b"refiner fixture"),
        "annotations_sha256": digest(annotation), "folds_sha256": digest(fold_bytes),
        "preprocessed_sha256": mapping_digest(dict.fromkeys(names, digest(image_bytes))),
        "config_sha256": mapping_digest(config), "selector_split_sha256": mapping_digest(partitions),
        "runtime_versions": {"python": sys.version.split()[0], **versions},
        "baseline": {"pq": 0., "tolerance": 0.}}
    contract = artifacts / "expected.json"
    contract.write_text(json.dumps(expected), encoding="utf-8")
    cfg = SimpleNamespace(**config, FOLD=2, PROTOCOL_SHA256=builder.PROTOCOL_SHA256,
        DATA_CACHE=dataset, ARTIFACT_ROOT=artifacts, PREPROC_CACHE=cache, PREPROC_DIR=staged,
        LOCAL_ROOT=tmp_path / "local", MANIFEST=tmp_path / "manifest.json", EXPECTED_CONTRACT=contract)
    env = helpers()
    guard = env["require_local_publication"]
    env["require_local_publication"] = lambda path: guard(path,
        f"1 0 0:1 / {tmp_path.as_posix()} rw - overlay overlay rw")
    cuda = SimpleNamespace(empty_cache=lambda: None, get_device_name=lambda index: "CPU stub",
        max_memory_allocated=lambda: 0)
    import time
    env.update(CFG=cfg, Path=Path, json=json, hashlib=hashlib, defaultdict=defaultdict,
        pd=pd, shutil=shutil, sys=sys, version=versions.__getitem__, prod_checkpoint=lambda fold: detector,
        tqdm=lambda values, **kwargs: values, cv2=cv2, mask_utils=mask_utils,
        torch=SimpleNamespace(cuda=cuda, no_grad=lambda: lambda fn: fn), time=time,
        started=time.time(), started_utc="CPU test", datetime=datetime, timezone=timezone)
    return env, expected, image


def test_generated_inputs_match_independently_frozen_identity(input_environment):
    env, expected, _ = input_environment
    execute_generated(builder.build_cells(2)[2]["source"], env)
    assert env["ACTUAL_INPUTS"] == {k: v for k, v in expected.items() if k != "baseline"}
    assert env["INPUT_CONTRACT"] == expected
    assert env["PARTITIONS"] == {"train_a": ["a.png"], "train_b": ["b.png"], "heldout": ["held.png"]}
    assert env["file_to_ids"]["held.png"] == [2, 9]
    assert env["TRAIN_JSON"] == env["CFG"].DATA_CACHE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


@pytest.mark.parametrize("key", ["config_sha256", "detector_sha256", "refiner_sha256",
    "folds_sha256", "preprocessed_sha256", "selector_split_sha256", "protocol_sha256"])
def test_changed_identity_stops_before_model_load(input_environment, key):
    env, expected, _ = input_environment
    env["CFG"].EXPECTED_CONTRACT.write_text(json.dumps({**expected, key: "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen selector inputs differ"):
        execute_generated(builder.build_cells(2)[2]["source"], env)
    assert "refiner" not in env and not env["CFG"].MANIFEST.exists()


def test_missing_dependency_precedes_input_discovery(tmp_path):
    import sys
    from importlib.metadata import PackageNotFoundError
    from types import SimpleNamespace

    def missing(package):
        raise PackageNotFoundError(package)

    env = {"CFG": SimpleNamespace(MANIFEST=tmp_path / "result.json", SPLIT_SEED=3407), "sys": sys,
        "require_local_publication": lambda path: None, "version": missing}
    with pytest.raises(PackageNotFoundError):
        execute_generated(builder.build_cells(2)[2]["source"], env)
    assert "coco" not in env


@pytest.mark.parametrize("empty_split", [None, "train_a", "train_b"])
def test_three_view_collection_and_generated_evaluation_complete(input_environment, empty_split):
    from types import SimpleNamespace

    env, expected, original_image = input_environment
    expected["baseline"] = {"pq": 1., "tolerance": 0.}
    env["CFG"].EXPECTED_CONTRACT.write_text(json.dumps(expected), encoding="utf-8")
    execute_generated(builder.build_cells(2)[2]["source"], env)
    execute_generated(builder.build_cells(2)[4]["source"].replace("from ultralytics import YOLO\n", ""), env)
    mask = np.zeros((2048, 2048), bool)
    mask[20:24, 30:35] = True
    supported = np.zeros_like(mask)
    supported[20:22, 30:33] = True  # Exact 6/20=.3 support boundary.
    truth = env["encode_one"](mask)
    env["gt_rles"] = {0: [truth], 1: [], 2: [truth], 9: [truth]}
    env["refine_proposal_stats"] = lambda image, proposal: (proposal, .8)
    calls = []

    class Detector:
        def predict(self, image, **kwargs):
            flipped = bool(image[5, 2040].any())
            assert np.array_equal(image, np.fliplr(original_image) if flipped else original_image)
            calls.append((kwargs, flipped))
            if ((empty_split == "train_a" and len(calls) <= 3)
                    or (empty_split == "train_b" and 4 <= len(calls) <= 6)):
                masks, scores = [], np.zeros(0, np.float32)
            elif flipped:
                masks, scores = [np.fliplr(supported).copy()], np.array([.77], np.float32)
            elif kwargs["imgsz"] == 1536:
                masks, scores = [np.zeros_like(mask), mask.copy()], np.array([.99, .88], np.float32)
            else:
                masks, scores = [mask.copy()], np.array([.35], np.float32)
            image[:] = 255  # No detector mutation may leak into another view/refinement.
            return [SimpleNamespace(masks=masks, scores=scores)]

    env["YOLO"] = lambda path: Detector()
    env["result_masks"] = lambda result: (result.masks, result.scores)
    execute_generated(builder.COLLECT, env)
    assert len(calls) == 9
    assert [(c[0]["imgsz"], c[1]) for c in calls] == [(1280, False), (1280, True), (1536, False)] * 3
    assert all(c[0]["conf"] == .05 and c[0]["iou"] == .55 and c[0]["retina_masks"] for c in calls)
    features = env["val_rows"][0]["props"][0]["feat"]
    assert features["tta_hflip_iou"] == pytest.approx(.3)
    assert features["tta_hflip_conf"] == pytest.approx(.77)
    assert features["tta_big_iou"] == 1.
    assert features["tta_big_conf"] == pytest.approx(.88)
    assert features["tta_support"] == 2.
    execute_generated(builder.EVALUATE, env)
    result = json.loads(env["CFG"].MANIFEST.read_text(encoding="utf-8"))
    assert result["baseline"]["tp"] == result["selected"]["tp"] == 2
    assert result["baseline"]["pq"] == result["selected"]["pq"] == 1.
    assert len(result["train_b_grid"]) == 54
    expected_points = [(mode, cf, param) for cf in (.05, .08, .12, .20, .26, .35)
        for mode, params in (("support", (0, 1, 2)), ("pk", (0., .3, .5, .7, .85, .93)))
        for param in params]
    assert [(p["mode"], p["conf_floor"], p["param"]) for p in result["train_b_grid"]] == expected_points
    assert result["frozen_operation_point"]["mode"] == "support"
    assert result["frozen_operation_point"]["conf_floor"] == .05
    assert result["frozen_operation_point"]["param"] == 0
    assert result["feature_importance"]["values"] is None
    assert ("empty" if empty_split == "train_a" else "positive") in result["selector_status"]["reason"]
    if empty_split == "train_b":
        assert result["train_b_selection_status"] == "no_proposals_deterministic_first_tie"
    else:
        assert all(point["metrics"]["pq"] == 0. for point in result["train_b_grid"])
        assert result["train_b_selection_status"] == "tie_first_grid_point"


def test_tta_best_iou_pairs_confidence_and_deterministic_ties():
    env = helpers()
    env["encode_one"] = lambda mask: mask
    env["iou_matrix"] = lambda truth, prediction: np.array([[.4, .8, .8]])
    assert env["tta_features"](None, [1, 2, 3], [.9, .2, .7]) == (.8, .2)
    assert env["tta_features"](None, [], []) == (0., 0.)
    with pytest.raises(ValueError, match="coverage"):
        env["tta_features"](None, [1], [])
    env["iou_matrix"] = lambda truth, prediction: np.zeros((1, 3))
    assert env["tta_features"](None, [1, 2, 3], [.9, .2, .7]) == (0., 0.)


def test_duplicate_annotations_stop_before_model_loading(input_environment):
    env, _, _ = input_environment
    original = env["CFG"].DATA_CACHE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    nested = original.parent / "duplicate"
    nested.mkdir()
    (nested / original.name).write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="Exactly one annotation JSON"):
        execute_generated(builder.build_cells(2)[2]["source"], env)
    assert "refiner" not in env and not env["CFG"].MANIFEST.exists()


def test_generated_refinement_uses_nondefault_crop_context():
    from types import SimpleNamespace

    env = helpers()
    env.update(CFG=SimpleNamespace(MIN_AREA=1, DROP_THRESHOLD=.9, CROP_CONTEXT=2.4),
        torch=SimpleNamespace(no_grad=lambda: lambda fn: fn), images={}, annotations_by_image={})
    execute_generated(builder.build_cells(2)[4]["source"].replace("from ultralytics import YOLO\n", ""), env)
    observed = []

    def stop_at_crop(mask, context=1.65):
        observed.append(context)
        raise RuntimeError("crop boundary reached")

    env["mask_square_box"] = stop_at_crop
    with pytest.raises(RuntimeError, match="crop boundary"):
        env["refine_proposal_stats"](None, np.ones((2, 2), bool))
    assert observed == [2.4]
