"""CPU contracts for the exploratory selector notebook; no GPU acceptance."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v64", ROOT / "scratch/build_v64_selector_pilot.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def helpers():
    env = {"np": np}
    exec(builder.BASE.GUARDS, env)
    exec(builder.SELECTOR_HELPERS, env)
    return env


def publication_helpers(tmp_path, kind="overlay"):
    env = helpers()
    guard = env["require_local_publication"]
    mounts = f"1 0 0:1 / {tmp_path.as_posix()} rw - {kind} source rw"
    env["require_local_publication"] = lambda path: guard(path, mounts)
    return env


def test_publication_preserves_existing_manifest(tmp_path):
    path = tmp_path / "complete.json"
    path.write_bytes(b"earlier complete result")
    with pytest.raises(ValueError, match="exists"):
        publication_helpers(tmp_path)["publish_manifest"](path, {"replacement": True})
    assert path.read_bytes() == b"earlier complete result"
    assert not list(tmp_path.glob("*.publish-lock"))


def test_publication_rejects_fuse_mount_before_reservation(tmp_path):
    env = helpers()
    mounts = (f"1 0 0:1 / {Path(tmp_path.anchor).as_posix()} rw - overlay overlay rw\n"
              f"2 1 0:2 / {tmp_path.as_posix()} rw - fuse.drive drive rw\n")
    with pytest.raises(ValueError, match="local filesystem"):
        env["require_local_publication"](tmp_path / "manifest.json", mounts)
    env["require_local_publication"](tmp_path / "manifest.json", mounts.splitlines()[0])
    with pytest.raises(ValueError, match="local filesystem"):
        env["require_local_publication"](tmp_path / "manifest.json", "")
    with pytest.raises(ValueError, match="local filesystem"):
        publication_helpers(tmp_path, "fuse.drive")["publish_manifest"](
            tmp_path / "manifest.json", {"value": 1})
    assert not list(tmp_path.glob("*.publish-lock"))
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("types", [("overlay", "fuse.drive"), ("fuse.drive", "overlay"),
                                    ("overlay", "ext4")])
def test_publication_refuses_ambiguous_stacked_mounts(tmp_path, types):
    mounts = "\n".join(f"{index} 0 0:1 / {tmp_path.as_posix()} rw - {kind} source rw"
                       for index, kind in enumerate(types, 1))
    with pytest.raises(ValueError, match="local filesystem"):
        helpers()["require_local_publication"](tmp_path / "manifest.json", mounts)


def test_overlapping_publications_accept_exactly_one_result(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "complete.json"
    barrier = Barrier(2)
    publish = publication_helpers(tmp_path)["publish_manifest"]

    def run(value):
        barrier.wait()
        try:
            publish(path, {"value": value})
            return value
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, (1, 2)))
    accepted = [value for value in results if value is not None]
    assert len(accepted) == 1
    assert json.loads(path.read_text()) == {"value": accepted[0]}
    assert not list(tmp_path.glob("*.publish-lock"))


def test_publication_refuses_an_active_or_interrupted_reservation(tmp_path):
    path = tmp_path / "complete.json"
    reservation = tmp_path / "complete.json.publish-lock"
    reservation.mkdir()
    with pytest.raises(ValueError, match="reserved"):
        publication_helpers(tmp_path)["publish_manifest"](path, {"value": 1})
    assert reservation.is_dir()
    assert not path.exists()


def test_failed_publication_releases_only_its_own_reservation(tmp_path):
    path = tmp_path / "complete.json"
    env = publication_helpers(tmp_path)

    def fail(*args):
        raise OSError("interrupted write")

    env["atomic_json"] = fail
    with pytest.raises(OSError, match="interrupted"):
        env["publish_manifest"](path, {"value": 1})
    assert not path.exists()
    assert not list(tmp_path.glob("*.publish-lock"))


def test_partitions_are_complete_disjoint_and_deterministic():
    partition = helpers()["selector_partitions"]
    mapping = {"a": 0, "b": 0, "c": 1, "d": 2}
    a, b, val = partition(mapping, 2)
    assert set(a) | set(b) == {"a", "b", "c"}
    assert not set(a) & set(b)
    assert val == ["d"]
    assert partition(mapping, 2) == (a, b, val)
    with pytest.raises(ValueError):
        partition({"a": 0, "b": 2}, 2)


@pytest.mark.parametrize("labels", [[], [True], [False, False]])
def test_classifier_refuses_degenerate_training(labels):
    with pytest.raises(ValueError, match="two classes"):
        helpers()["fit_selector"](np.zeros((len(labels), 10)), np.array(labels))


def test_generated_cells_parse_and_saved_notebook_matches():
    for fold in range(3):
        for cell in builder.build_cells(fold):
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
    notebook = json.loads((ROOT / "solar_filament_v64_selector_fold2_colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["cells"] == builder.notebook(2)["cells"]
    sources = "\n".join(c["source"] for c in notebook["cells"])
    assert "pickle" not in sources
    assert "val_grid" not in sources
    assert "!pip" not in sources
    assert "image.copy()" in sources
    setup = notebook["cells"][1]["source"]
    assert 'MANIFEST = LOCAL_ROOT /' in setup
    assert "PREDICT_CONF_FLOOR = 0.05" in setup
    assert f"PROTOCOL_SHA256 = '{builder.PROTOCOL_SHA256}'" in setup
    for constant in ("BASELINE_PQ_REFINED", "ORACLES", "IOU_HIT", "FN_WEAK_LO", "DENSE_QUANTILE"):
        assert constant not in setup


def test_frozen_contract_requires_baseline_identity_and_runtime(tmp_path):
    env = helpers()
    actual = {"schema_version": 1, "fold": 2, "protocol_sha256": "a" * 64,
              "runtime_versions": {"python": "3.11"}, "config_sha256": "b" * 64}
    path = tmp_path / "contract.json"
    with pytest.raises(FileNotFoundError):
        env["require_selector_contract"](path, actual)
    expected = {**actual, "baseline": {"pq": .4, "tolerance": .001}}
    path.write_text(json.dumps(expected))
    assert env["require_selector_contract"](path, actual) == expected
    for patch in [{"runtime_versions": {"python": "3.12"}}, {"protocol_sha256": "c" * 64},
                  {"baseline": {"pq": .4}}, {"config_sha256": True}]:
        path.write_text(json.dumps({**expected, **patch}))
        with pytest.raises(ValueError):
            env["require_selector_contract"](path, actual)


def test_real_cpu_classifier_handles_both_classes():
    from threadpoolctl import threadpool_limits

    env = helpers()
    x = np.zeros((40, 10))
    x[20:, 0] = 1
    with threadpool_limits(limits=1):
        model = env["fit_selector"](x, np.array([False] * 20 + [True] * 20))
        predictions = model.predict_proba(x)
    assert predictions.shape == (40, 2)
    assert np.isfinite(predictions).all()
    assert predictions[20, 1] > predictions[0, 1]


def test_real_official_evaluator_uses_all_hits_and_strict_threshold():
    import cv2
    from pycocotools import mask as mask_utils

    env = helpers()
    env.update({"cv2": cv2, "mask_utils": mask_utils, "images": {}, "annotations_by_image": {},
                "torch": type("Torch", (), {"no_grad": staticmethod(lambda: lambda f: f)}),
                "CFG": type("Config", (), {"MIN_AREA": 1, "DROP_THRESHOLD": .9})})
    # Execute actual generated evaluator definitions, omitting optional YOLO import.
    source = builder.build_cells(2)[4]["source"].replace("from ultralytics import YOLO\n", "")
    exec(source, env)
    metric = env["PanopticQuality"]()
    metric.add_matrix(np.array([[.6, .7], [.5, .2]]), 2, 2)
    assert metric.summary() == pytest.approx({"pq": 1.3 / 2.5, "sq": .65,
        "rq": .8, "tp": 2, "fp": 0, "fn": 1, "denominator": 2.5})
    env["validate_metrics"](metric.summary())
    mask = np.zeros((8, 8), bool)
    mask[1:4, 1:4] = True
    feat = env["proposal_features"](np.zeros((8, 8, 3), np.uint8), mask, .7, mask, .8)
    assert feat["ref_raw_iou"] == 1
    assert feat["darkness"] == 0
    assert set(feat) == set(env["FEATURE_KEYS"])


@pytest.mark.parametrize("context", [.8, 2.4])
def test_generated_refinement_consumes_frozen_crop_context(context):
    from types import SimpleNamespace

    env = helpers()
    env.update({"CFG": SimpleNamespace(MIN_AREA=1, DROP_THRESHOLD=.9, CROP_CONTEXT=context),
                "torch": SimpleNamespace(no_grad=lambda: lambda f: f),
                "images": {}, "annotations_by_image": {}})
    exec(builder.build_cells(2)[4]["source"].replace("from ultralytics import YOLO\n", ""), env)
    observed = []

    def stop_at_crop(mask, context=1.65):
        observed.append(context)
        raise RuntimeError("crop boundary reached")

    env["mask_square_box"] = stop_at_crop
    with pytest.raises(RuntimeError, match="crop boundary"):
        env["refine_proposal_stats"](None, np.ones((2, 2), bool))
    assert observed == [context]


@pytest.mark.parametrize("changed", [{"tp": True}, {"denominator": 9}, {"rq": .1}, {"sq": float("nan")}])
def test_output_metrics_reject_inconsistent_arithmetic(changed):
    env = helpers()
    with pytest.raises(ValueError):
        env["validate_metrics"]({"tp": 1, "fp": 0, "fn": 0,
                                  "denominator": 1, "pq": .7, "sq": .7, "rq": 1, **changed})


def test_captured_image_replacement_is_rejected_and_original_bytes_decoded():
    import hashlib

    import cv2

    env = helpers()
    original = np.zeros((2048, 2048, 3), np.uint8)
    original[5, 7] = [10, 20, 30]
    captured = cv2.imencode(".png", original)[1].tobytes()
    expected = hashlib.sha256(captured).hexdigest()
    assert np.array_equal(env["decode_verified_image"](captured, expected), original)
    replaced = cv2.imencode(".png", np.ones_like(original))[1].tobytes()
    with pytest.raises(ValueError, match="bytes changed"):
        env["decode_verified_image"](replaced, expected)


def test_baseline_drift_stops_before_fit_and_atomic_publication():
    env = helpers()
    published = []
    env.update({"val_rows": [], "CFG": type("Cfg", (), {"CONF": .26}),
                "INPUT_CONTRACT": {"baseline": {"pq": .4, "tolerance": .001}},
                "atomic_json": lambda *args: published.append(args),
                "PanopticQuality": type("Metric", (), {"summary": lambda self: {
                    "pq": 0., "sq": 0., "rq": 0., "tp": 0, "fp": 0, "fn": 0, "denominator": 0.}})})
    with pytest.raises(ValueError, match="Baseline reproduction"):
        exec(builder.EVALUATE, env)
    assert not published
    assert "selector" not in env


def test_contract_gate_precedes_any_model_loading():
    cells = builder.build_cells(2)
    assert "INPUT_CONTRACT = require_selector_contract" in cells[2]["source"]
    assert "YOLO(" not in cells[1]["source"] + cells[2]["source"]
    assert "torch.load(" not in cells[1]["source"] + cells[2]["source"]
    assert "torch.load(" in cells[3]["source"]


def test_generated_input_cell_matches_independently_frozen_contract(tmp_path):
    import hashlib
    import shutil
    import sys
    from collections import defaultdict
    from types import SimpleNamespace

    import pandas as pd

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    names = ["a.png", "b.png", "held.png"]
    annotation = json.dumps({"images": [{"id": i, "file_name": name}
        for i, name in enumerate(names)], "annotations": []}).encode()
    dataset = tmp_path / "dataset"
    annotation_dir = dataset / "nested" / "annotations"
    annotation_dir.mkdir(parents=True)
    (annotation_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_bytes(annotation)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    folds = b"file_name,fold\na.png,0\nb.png,1\nheld.png,2\n"
    (artifacts / "grouped_folds.csv").write_bytes(folds)
    detector = tmp_path / "detector.pt"
    detector.write_bytes(b"detector fixture")
    (artifacts / "mediseg_refiner_fold2_resume.pt").write_bytes(b"refiner fixture")
    cache, staged = tmp_path / "cache", tmp_path / "staged"
    cache.mkdir()
    staged.mkdir()
    for name in names:
        (cache / name).write_bytes(name.encode())
        (staged / name).write_bytes(b"stale local bytes")
    config = {"YOLO_IMGSZ": 1280, "CROP_SIZE": 320, "CROP_CONTEXT": 1.65,
              "CONF": .26, "PREDICT_CONF_FLOOR": .05, "PREDICT_IOU": .55,
              "REFINER_THRESHOLD": .45, "DROP_THRESHOLD": .90, "MIN_AREA": 20}
    partitions = {"train_a": ["a.png"], "train_b": ["b.png"], "heldout": ["held.png"]}
    packages = ("numpy", "pandas", "scipy", "torch", "opencv-python",
                "ultralytics", "pycocotools", "scikit-learn")
    versions = {package: f"fixture-{package}" for package in packages}
    expected = {"schema_version": 1, "fold": 2, "protocol_sha256": builder.PROTOCOL_SHA256,
        "detector_sha256": digest(b"detector fixture"),
        "refiner_sha256": digest(b"refiner fixture"), "annotations_sha256": digest(annotation),
        "folds_sha256": digest(folds), "preprocessed_sha256": digest(json.dumps(
            {name: digest(name.encode()) for name in names}, sort_keys=True).encode()),
        "config_sha256": digest(json.dumps(config, sort_keys=True).encode()),
        "selector_split_sha256": digest(json.dumps(partitions, sort_keys=True).encode()),
        "runtime_versions": {"python": sys.version.split()[0], **versions},
        "baseline": {"pq": .4, "tolerance": .001}}
    contract = artifacts / "expected.json"
    contract.write_text(json.dumps(expected))
    cfg = SimpleNamespace(**config, FOLD=2, PROTOCOL_SHA256=builder.PROTOCOL_SHA256,
        DATA_CACHE=dataset, ARTIFACT_ROOT=artifacts, PREPROC_CACHE=cache, PREPROC_DIR=staged,
        MANIFEST=tmp_path / "manifest.json", EXPECTED_CONTRACT=contract)

    def environment():
        env = publication_helpers(tmp_path)
        env.update({"CFG": cfg, "Path": Path, "json": json, "hashlib": hashlib,
            "defaultdict": defaultdict, "pd": pd, "shutil": shutil, "sys": sys,
            "version": versions.__getitem__, "prod_checkpoint": lambda fold: detector,
            "tqdm": lambda values, **kwargs: values})
        return env

    source = builder.build_cells(2)[2]["source"]
    env = environment()
    exec(source, env)
    assert env["ACTUAL_INPUTS"] == {key: value for key, value in expected.items() if key != "baseline"}
    assert env["INPUT_CONTRACT"] == expected
    assert env["PARTITIONS"] == partitions
    assert "noncrowd_count" not in env
    assert "DENSE_CUT" not in env
    assert {path.name: path.read_bytes() for path in staged.iterdir()} == {
        name: name.encode() for name in names}
    for changed in ({"config_sha256": "0" * 64}, {"selector_split_sha256": "0" * 64},
                    {"runtime_versions": {**expected["runtime_versions"], "numpy": "wrong"}}):
        contract.write_text(json.dumps({**expected, **changed}))
        with pytest.raises(ValueError, match="Frozen selector inputs differ"):
            exec(source, environment())


def test_generated_input_heading_describes_selector_staging():
    source = builder.build_cells(2)[2]["source"]
    assert "# Stage all selector inputs and verify the frozen contract." in source
    assert "密度スライス" not in source
    assert "解剖パラメータ" not in builder.build_cells(2)[1]["source"]
    assert "noncrowd_count" not in source
    assert "DENSE_CUT" not in source


def test_unsupported_publication_stops_before_input_staging(tmp_path):
    from types import SimpleNamespace

    checked = []

    def refuse(path):
        checked.append(path)
        raise ValueError("Unsupported local filesystem")

    path = tmp_path / "manifest.json"
    env = {"CFG": SimpleNamespace(MANIFEST=path), "require_local_publication": refuse}
    with pytest.raises(ValueError, match="local filesystem"):
        exec(builder.build_cells(2)[2]["source"], env)
    assert checked == [path]
    assert "coco" not in env


def test_missing_runtime_dependency_stops_before_input_staging(tmp_path):
    import sys
    from importlib.metadata import PackageNotFoundError
    from types import SimpleNamespace

    checked = []

    def missing(package):
        checked.append(package)
        raise PackageNotFoundError(package)

    env = {"CFG": SimpleNamespace(MANIFEST=tmp_path / "manifest.json"),
           "require_local_publication": lambda path: None, "sys": sys, "version": missing}
    with pytest.raises(PackageNotFoundError):
        exec(builder.build_cells(2)[2]["source"], env)
    assert checked
    assert "coco" not in env


def test_generated_collection_selection_and_atomic_manifest_with_cpu_stubs(tmp_path):
    import hashlib
    import time
    from datetime import datetime, timezone
    from types import SimpleNamespace

    import cv2
    from pycocotools import mask as mask_utils

    names = ["a.png", "b.png", "held.png"]
    image = np.zeros((2048, 2048, 3), np.uint8)
    captured = cv2.imencode(".png", image)[1].tobytes()
    for name in names:
        (tmp_path / name).write_bytes(captured)
    checkpoint = tmp_path / "det.pt"
    checkpoint.write_bytes(b"CPU stub checkpoint")
    masks = [np.zeros((2048, 2048), bool), np.zeros((2048, 2048), bool)]
    masks[0][2:6, 2:6] = True
    masks[1][12:16, 12:16] = True
    prediction_inputs = []

    class FakeDetector:
        def __init__(self, path):
            assert Path(path).read_bytes() == checkpoint.read_bytes()

        def predict(self, consumed, **kwargs):
            prediction_inputs.append(consumed)
            assert np.array_equal(consumed, image)
            return [None]

    cuda = SimpleNamespace(empty_cache=lambda: None, get_device_name=lambda _: "CPU stub",
                           max_memory_allocated=lambda: 0)
    cfg = SimpleNamespace(MIN_AREA=1, DROP_THRESHOLD=.9, FOLD=2, LOCAL_ROOT=tmp_path,
                          PREPROC_DIR=tmp_path, YOLO_IMGSZ=1280, PREDICT_CONF_FLOOR=.05,
                          PREDICT_IOU=.55, CONF=.26, MANIFEST=tmp_path / "complete.json")
    env = publication_helpers(tmp_path)
    env.update({"cv2": cv2, "mask_utils": mask_utils, "CFG": cfg,
                "torch": SimpleNamespace(no_grad=lambda: lambda f: f, cuda=cuda),
                "images": {}, "annotations_by_image": {}})
    exec(builder.build_cells(2)[4]["source"].replace("from ultralytics import YOLO\n", ""), env)
    truth = env["encode_one"](masks[0])
    env.update({"YOLO": FakeDetector, "prod_checkpoint": lambda _: checkpoint,
                "result_masks": lambda _: (masks, np.array([.8, .8])),
                "refine_proposal_stats": lambda consumed, mask: (mask, .8),
                "PREPROC_HASHES": dict.fromkeys(names, hashlib.sha256(captured).hexdigest()),
                "file_to_ids": {"a.png": [0], "b.png": [1], "held.png": [2, 3]},
                "images": {index: {"height": 2048, "width": 2048} for index in range(4)},
                "gt_rles": {**dict.fromkeys(range(3), [truth]), 3: [env["encode_one"](masks[1])]},
                "tqdm": lambda value, **kwargs: value,
                "train_A_files": names[:1], "train_B_files": names[1:2], "valid_files_fold": names[2:],
                "INPUT_CONTRACT": {"detector_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                   "baseline": {"pq": 2 / 3, "tolerance": 0}},
                "hashlib": hashlib, "time": time, "datetime": datetime, "timezone": timezone,
                "started": time.time(), "started_utc": datetime.now(timezone.utc).isoformat(),
                "json": json, "PARTITIONS": {"train_a": names[:1], "train_b": names[1:2], "heldout": names[2:]}})
    exec(builder.COLLECT, env)
    assert len(prediction_inputs) == 3
    # Each proposal overlaps one annotator: the training target intentionally unions them.
    assert [p["label_hit"] for p in env["val_rows"][0]["props"]] == [True, True]
    assert all(isinstance(p["refined_rle"], dict) for row in env["val_rows"] for p in row["props"])

    class Selector:
        def predict_proba(self, x):
            return np.array([[.1, .9], [.9, .1]] * (len(x) // 2))

    env["fit_selector"] = lambda x, y: Selector()
    exec(builder.EVALUATE, env)
    result = json.loads(cfg.MANIFEST.read_text())
    assert result["baseline"]["pq"] == pytest.approx(2 / 3)
    assert result["selected"]["pq"] == .5
    # The same predictions are scored separately against each annotator, then pooled.
    assert {key: result["baseline"][key] for key in ("tp", "fp", "fn")} == {"tp": 2, "fp": 2, "fn": 0}
    assert {key: result["selected"][key] for key in ("tp", "fp", "fn")} == {"tp": 1, "fp": 1, "fn": 1}
    assert len(result["train_b_grid"]) == 36
    assert result["frozen_operation_point"]["p_keep"] == .3
    assert result["proposal_counts"] == {"train_a": 2, "train_b": 2, "heldout": 2}
    assert not list(tmp_path.glob("*.tmp"))
    published_bytes = cfg.MANIFEST.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        exec(builder.EVALUATE, env)
    assert cfg.MANIFEST.read_bytes() == published_bytes
