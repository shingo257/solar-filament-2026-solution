"""CPU contracts for cached-proposal diagnostics, not independent GPU acceptance."""

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location("v66", ROOT / "scratch/build_v66_rfdetr_union.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    env = {"np": np, "Path": Path, "json": json, "hashlib": hashlib}
    exec(builder().GUARDS, env)
    return env


def packed_bytes(packed=None, shape=None, confidences=None, **extra):
    if shape is None:
        shape = np.array([4, 9], dtype=np.int32)
    if packed is None:
        packed = np.zeros((2, 4, 2), dtype=np.uint8)
    if confidences is None:
        confidences = np.array([.8, .2], dtype=np.float32)
    stream = io.BytesIO()
    np.savez_compressed(stream, packed=packed, shape=shape, confidences=confidences, **extra)
    return stream.getvalue()


def test_real_producer_nonempty_and_nonbyte_width_roundtrip():
    masks = np.zeros((2, 4, 9), dtype=bool)
    masks[0, 0, [0, 8]] = True
    masks[1, 2, 3:7] = True
    decoded, confidences = helpers()["load_npz_masks"](
        packed_bytes(packed=np.packbits(masks, axis=-1)), (4, 9))
    assert np.array_equal(decoded, masks)
    assert confidences == pytest.approx([.8, .2])


def test_real_producer_empty_sentinel_is_valid():
    masks, confidences = helpers()["load_npz_masks"](packed_bytes(
        packed=np.empty((0, 0, 0), np.uint8), shape=np.array([2048, 2048], np.int32),
        confidences=np.empty(0, np.float32)), (2048, 2048))
    assert len(masks) == len(confidences) == 0


def test_loader_preserves_producer_float32_confidence_threshold_semantics():
    original = np.array([.26, .26], dtype=np.float32)
    _, decoded = helpers()["load_npz_masks"](packed_bytes(confidences=original), (4, 9))
    assert decoded.dtype == original.dtype
    assert [score >= .26 for score in decoded] == [score >= .26 for score in original]


@pytest.mark.parametrize("changed", [
    {"packed": np.zeros((2, 3, 2), np.uint8)},
    {"packed": np.zeros((2, 4, 1), np.uint8)},
    {"packed": np.zeros((2, 4, 2), np.int32)},
    {"packed": np.zeros((2, 4), np.uint8)},
    {"shape": np.array([4., 9.])},
    {"shape": np.array([0, 9])},
    {"shape": np.array([4, 9, 2])},
    {"confidences": np.empty(0, np.float32)},
    {"confidences": np.array([[.8, .2]])},
    {"confidences": np.array([np.nan, .2])},
    {"confidences": np.array([np.inf, .2])},
    {"confidences": np.array([-.1, .2])},
    {"confidences": np.array([1.1, .2])},
    {"confidences": np.array([".8", ".2"], dtype=object)},
    {"unexpected": np.array([1])},
])
def test_malformed_cache_is_rejected_instead_of_silently_truncated(changed):
    with pytest.raises(ValueError):
        helpers()["load_npz_masks"](packed_bytes(**changed), (4, 9))


def test_cache_frame_must_match_consuming_image():
    with pytest.raises(ValueError):
        helpers()["load_npz_masks"](packed_bytes(), (9, 4))


def test_cache_coverage_is_exact_and_filename_keyed(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"a")
    (tmp_path / "b.npz").write_bytes(b"b")
    fn = helpers()["validate_cache_coverage"]
    assert fn(tmp_path, ["a.png", "b.jpg"]) == {
        "a.png": tmp_path / "a.npz", "b.jpg": tmp_path / "b.npz"}
    with pytest.raises(ValueError):
        fn(tmp_path, ["a.png", "b.jpg", "missing.png"])
    with pytest.raises(ValueError):
        fn(tmp_path, ["a.png"])
    with pytest.raises(ValueError):
        fn(tmp_path, ["a.png", "a.jpg"])
    with pytest.raises(ValueError):
        fn(tmp_path, ["../a.png"])


def test_blind_denominator_includes_no_rf_proposals_and_all_annotators():
    result = helpers()["rescue_counts"](
        [np.array([0., .05, .1]), np.array([0.]), np.array([])],
        [np.empty((3, 0)), np.empty((1, 0)), np.empty((0, 0))])
    assert result == {"n_blind": 3, "rescued_03": 0, "rescued_05": 0}


def test_rescue_thresholds_are_strict_and_baseline_point_one_is_not_blind():
    result = helpers()["rescue_counts"](
        [np.array([0., 0., .09, .1])],
        [np.array([[.3, .2], [.5, .4], [.6, 0.], [.99, .99]])])
    assert result == {"n_blind": 3, "rescued_03": 2, "rescued_05": 1}


@pytest.mark.parametrize("matrices", [[], [np.zeros((2, 1))], [np.full((1, 1), np.nan)]])
def test_rescue_requires_aligned_finite_matrices(matrices):
    with pytest.raises(ValueError):
        helpers()["rescue_counts"]([np.array([0.])], matrices)


def producer_manifest(names):
    return {"schema_version": 1, "fold": 0, "producer_protocol_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64, "inference_floor": .05, "training_folds": [1, 2],
            "prediction_unit": "filename", "preprocessing_sha256": "3" * 64,
            "frame": [2048, 2048], "files": dict.fromkeys(names, "4" * 64)}


def test_cache_producer_identity_is_bound_but_preprocessing_may_differ():
    manifest = producer_manifest(["a.png"])
    validate = helpers()["validate_cache_manifest"]
    assert validate(manifest, ["a.png"], manifest["files"], .05) == manifest
    different = {**manifest, "preprocessing_sha256": "5" * 64}
    assert validate(different, ["a.png"], manifest["files"], .05) == different


@pytest.mark.parametrize("changed", [{"fold": True}, {"fold": 1},
    {"training_folds": [0, 1, 2]}, {"training_folds": [True, 2]},
    {"frame": [2048., 2048]}, {"inference_floor": .1}, {"inference_floor": True},
    {"inference_floor": float("nan")}, {"checkpoint_sha256": "unknown"},
    {"files": {"a.png": "5" * 64}}, {"files": {}}, {"prediction_unit": "image_id"}])
def test_cache_producer_mismatch_is_rejected(changed):
    manifest = producer_manifest(["a.png"])
    with pytest.raises(ValueError):
        helpers()["validate_cache_manifest"]({**manifest, **changed}, ["a.png"], manifest["files"], .05)


@pytest.fixture
def input_environment(tmp_path, request):
    import sys
    from collections import defaultdict
    from datetime import datetime, timezone
    from types import SimpleNamespace

    import cv2
    import pandas as pd
    from pycocotools import mask as mask_utils

    module = builder()
    names = ["blind.png", "rescue.png"]
    dataset, artifacts, preproc = (tmp_path / name for name in ("dataset", "artifacts", "preproc"))
    for directory in (dataset, artifacts, preproc):
        directory.mkdir()
    coco = {"images": [{"id": i, "file_name": name, "height": 2048, "width": 2048}
        for i, name in enumerate(["blind.png", "rescue.png", "train1.png", "train2.png", "blind.png"])],
        "annotations": []}
    annotation_bytes = json.dumps(coco).encode()
    (dataset / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_bytes(annotation_bytes)
    fold_bytes = b"file_name,fold\nblind.png,0\nrescue.png,0\ntrain1.png,1\ntrain2.png,2\n"
    (artifacts / "grouped_folds.csv").write_bytes(fold_bytes)
    refiner_bytes = b"refiner fixture; never loaded"
    (artifacts / "mediseg_refiner_fold0_resume.pt").write_bytes(refiner_bytes)
    image_bytes = cv2.imencode(".png", np.zeros((2048, 2048, 3), np.uint8))[1].tobytes()
    for name in names:
        (preproc / name).write_bytes(image_bytes)
    mask = np.zeros((1, 2048, 2048), np.uint8)
    mask[0, 20:40, 30:60] = 1
    empty = packed_bytes(packed=np.empty((0, 0, 0), np.uint8),
                         shape=np.array([2048, 2048]), confidences=np.empty(0))
    rf_conf = getattr(request, "param", .10)
    full = packed_bytes(packed=np.packbits(mask, axis=-1), shape=np.array([2048, 2048]),
                        confidences=np.array([rf_conf]))
    directories, cache_digests, manifest_digests = {}, {}, {}
    def digest(data):
        return hashlib.sha256(data).hexdigest()

    def mapping_digest(data):
        return digest(json.dumps(data, sort_keys=True).encode())
    for kind in ("yolo", "rfdetr"):
        directory = tmp_path / kind
        directory.mkdir()
        directories[kind] = directory
        payloads = {name: full if kind == "rfdetr" and name == "rescue.png" else empty for name in names}
        hashes = {name: digest(data) for name, data in payloads.items()}
        for name, data in payloads.items():
            (directory / (Path(name).stem + ".npz")).write_bytes(data)
        manifest = {**producer_manifest(names), "files": hashes,
                    "preprocessing_sha256": ("6" if kind == "yolo" else "7") * 64}
        manifest_bytes = json.dumps(manifest).encode()
        (directory / "producer.json").write_bytes(manifest_bytes)
        cache_digests[kind], manifest_digests[kind] = mapping_digest(hashes), digest(manifest_bytes)
    config = {"CROP_SIZE": 320, "CROP_CONTEXT": 2., "YOLO_CONF": .26,
              "RFDETR_GRID": module.RFDETR_GRID, "REFINER_THRESHOLD": .45,
              "DROP_THRESHOLD": .9, "MIN_AREA": 20}
    versions = dict.fromkeys(("numpy", "pandas", "scipy", "torch", "opencv-python", "pycocotools"), "fixture")
    expected = {"schema_version": 1, "fold": 0, "protocol_sha256": module.PROTOCOL_SHA256,
        "config_sha256": mapping_digest(config), "runtime_versions": {"python": sys.version.split()[0], **versions},
        "annotations_sha256": digest(annotation_bytes), "folds_sha256": digest(fold_bytes),
        "preprocessed_sha256": mapping_digest(dict.fromkeys(names, digest(image_bytes))),
        "refiner_sha256": digest(refiner_bytes), "yolo_manifest_sha256": manifest_digests["yolo"],
        "rfdetr_manifest_sha256": manifest_digests["rfdetr"], "yolo_cache_sha256": cache_digests["yolo"],
        "rfdetr_cache_sha256": cache_digests["rfdetr"], "baseline": {"pq": 0., "tolerance": 0.}}
    contract = artifacts / "expected.json"
    contract.write_text(json.dumps(expected), encoding="utf-8")
    cfg = SimpleNamespace(**config, DATA_CACHE=dataset, ARTIFACT_ROOT=artifacts, PREPROC_CACHE=preproc,
        YOLO_OOF=directories["yolo"], RFDETR_OOF=directories["rfdetr"], FOLD=0,
        LOCAL_ROOT=tmp_path / "local", MANIFEST=tmp_path / "local/result.json",
        EXPECTED_CONTRACT=contract, PROTOCOL_SHA256=module.PROTOCOL_SHA256)
    env = helpers()
    guard = env["require_local_publication"]
    env["require_local_publication"] = lambda path: guard(path,
        f"1 0 0:1 / {tmp_path.as_posix()} rw - overlay overlay rw")
    cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: None,
        get_device_name=lambda index: "CPU stub", max_memory_allocated=lambda: 0)
    import time
    env.update(CFG=cfg, CONFIG_KEYS=module.CONFIG_KEYS, sys=sys, version=versions.__getitem__,
        defaultdict=defaultdict, pd=pd, BytesIO=io.BytesIO, cv2=cv2, mask_utils=mask_utils,
        torch=SimpleNamespace(cuda=cuda, no_grad=lambda: lambda function: function),
        tqdm=lambda values, **kwargs: values, time=time, datetime=datetime, timezone=timezone)
    env["TEST_RF_CONF"] = rf_conf
    return module, env, expected, mask[0].astype(bool)


def test_actual_generated_input_cell_matches_independently_frozen_contract(input_environment):
    module, env, expected, _ = input_environment
    exec(module.INPUTS, env)
    assert env["ACTUAL_INPUTS"] == {key: value for key, value in expected.items() if key != "baseline"}
    assert env["INPUT_CONTRACT"] == expected
    assert env["valid_files"] == ["blind.png", "rescue.png"]
    assert env["file_to_ids"]["blind.png"] == [0, 4]


@pytest.mark.parametrize("input_environment", [.05, .10], indirect=True)
def test_generated_evaluation_counts_absent_rf_and_pools_annotators(input_environment):
    module, env, _, mask = input_environment
    exec(module.INPUTS, env)
    exec(module.build_cells()[4]["source"], env)
    truth = env["encode_one"](mask)
    env["gt_rles"] = {0: [truth], 1: [truth], 4: [truth]}
    env["refine_proposal_stats"] = lambda image, proposal: (proposal, .8)
    exec(module.EVALUATE, env)
    result = json.loads(env["CFG"].MANIFEST.read_text(encoding="utf-8"))
    eligible = int(env["TEST_RF_CONF"] >= .10)
    assert result["rescue"] == {"n_blind": 3, "rescued_03": eligible, "rescued_05": eligible}
    assert result["baseline"]["fn"] == 3
    assert [arm["pq"] for arm in result["arms"]] == [.5, .5 * eligible, 0., 0., 0.]
    assert result["runtime"]["gpu"] == "CPU stub"
    assert "best_arm" not in result


def test_rescue_excludes_refined_masks_that_union_discards_for_area(input_environment):
    module, env, _, mask = input_environment
    exec(module.INPUTS, env)
    exec(module.build_cells()[4]["source"], env)
    tiny = np.zeros_like(mask)
    tiny[20, 30] = True
    truth = env["encode_one"](tiny)
    env["gt_rles"] = {0: [truth], 1: [truth], 4: [truth]}
    env["refine_proposal_stats"] = lambda image, proposal: (tiny, .8)
    exec(module.EVALUATE, env)
    result = json.loads(env["CFG"].MANIFEST.read_text(encoding="utf-8"))
    assert result["rescue"] == {"n_blind": 3, "rescued_03": 0, "rescued_05": 0}
    assert all(arm["pq"] == 0. for arm in result["arms"])


def test_union_publication_refuses_fuse_before_reservation(tmp_path):
    env = helpers()
    guard = env["require_local_publication"]
    env["require_local_publication"] = lambda path: guard(path,
        f"1 0 0:1 / {tmp_path.as_posix()} rw - fuse.drive drive rw")
    with pytest.raises(ValueError, match="local filesystem"):
        env["publish_manifest"](tmp_path / "result.json", {})
    assert not list(tmp_path.iterdir())


def test_union_publication_preserves_an_existing_reservation(tmp_path):
    env = helpers()
    guard = env["require_local_publication"]
    env["require_local_publication"] = lambda path: guard(path,
        f"1 0 0:1 / {tmp_path.as_posix()} rw - overlay overlay rw")
    reservation = tmp_path / "result.json.publish-lock"
    reservation.mkdir()
    with pytest.raises(ValueError, match="reserved"):
        env["publish_manifest"](tmp_path / "result.json", {})
    assert reservation.is_dir() and not (tmp_path / "result.json").exists()


def test_changed_cache_bytes_stop_before_result_publication(input_environment):
    module, env, _, _ = input_environment
    exec(module.INPUTS, env)
    exec(module.build_cells()[4]["source"], env)
    env["CACHE_PATHS"]["yolo"]["blind.png"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Cache bytes changed"):
        exec(module.EVALUATE, env)
    assert not env["CFG"].MANIFEST.exists()


@pytest.mark.parametrize("key", ["config_sha256", "rfdetr_manifest_sha256", "folds_sha256"])
def test_frozen_input_identity_mismatch_stops_before_refiner_loading(input_environment, key):
    module, env, expected, _ = input_environment
    env["CFG"].EXPECTED_CONTRACT.write_text(json.dumps({**expected, key: "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen union inputs differ"):
        exec(module.INPUTS, env)
    assert "refiner" not in env and not env["CFG"].MANIFEST.exists()


def test_baseline_reproduction_failure_cannot_publish(input_environment):
    module, env, expected, mask = input_environment
    env["CFG"].EXPECTED_CONTRACT.write_text(json.dumps({**expected,
        "baseline": {"pq": 1., "tolerance": 0.}}), encoding="utf-8")
    exec(module.INPUTS, env)
    exec(module.build_cells()[4]["source"], env)
    truth = env["encode_one"](mask)
    env["gt_rles"] = {0: [truth], 1: [truth], 4: [truth]}
    env["refine_proposal_stats"] = lambda image, proposal: (proposal, .8)
    with pytest.raises(ValueError):
        exec(module.EVALUATE, env)
    assert not env["CFG"].MANIFEST.exists()


def test_dependency_preflight_precedes_metadata_discovery(tmp_path):
    import sys
    from importlib.metadata import PackageNotFoundError
    from types import SimpleNamespace

    def missing(package):
        raise PackageNotFoundError(package)

    env = {"CFG": SimpleNamespace(MANIFEST=tmp_path / "manifest.json"), "sys": sys,
           "require_local_publication": lambda path: None, "version": missing}
    with pytest.raises(PackageNotFoundError):
        exec(builder().INPUTS, env)
    assert "annotation_paths" not in env


def test_generated_refiner_consumes_nondefault_contracted_crop_context():
    from types import SimpleNamespace

    env = helpers()
    env.update(CFG=SimpleNamespace(MIN_AREA=1, DROP_THRESHOLD=.9, CROP_CONTEXT=2.4),
               torch=SimpleNamespace(no_grad=lambda: lambda function: function),
               images={}, annotations_by_image={})
    exec(builder().build_cells()[4]["source"], env)
    observed = []

    def stop(mask, context=1.65):
        observed.append(context)
        raise RuntimeError("crop reached")

    env["mask_square_box"] = stop
    with pytest.raises(RuntimeError, match="crop reached"):
        env["refine_proposal_stats"](None, np.ones((2, 2), bool))
    assert observed == [2.4]


def test_generated_notebook_is_unexecuted_and_matches_builder():
    module = builder()
    notebook = json.loads((ROOT / "solar_filament_v66_rfdetr_union_fold0_colab.ipynb").read_text(encoding="utf-8"))
    assert notebook == module.notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("\n".join(line for line in cell["source"].splitlines()
                                if not line.startswith("!")))
