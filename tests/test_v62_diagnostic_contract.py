"""CPU-only diagnostic recovery: no model loading, fitting or download."""
import ast
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v62", ROOT / "scratch/build_v62_error_anatomy.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def functions():
    source = "\n".join(c["source"] for c in BUILDER.build_cells(2) if c["cell_type"] == "code")
    return [n for n in ast.parse("\n".join(s for s in source.splitlines() if not s.lstrip().startswith("!"))).body
            if isinstance(n, ast.FunctionDef) or (isinstance(n, ast.ClassDef) and n.name == "PanopticQuality")]


def helpers():
    ns = {"np":np, "mask_utils":mask_utils, "defaultdict":defaultdict,
              "CFG":SimpleNamespace(MIN_AREA=1, DROP_THRESHOLD=.9, IOU_HIT=.5,
                  FP_ORPHAN_MAX=.1, FN_WEAK_LO=.1, FN_NEAR_LO=.3, MERGE_MIN_IOU=.1, OVERMERGE_IOU=.3),
              "torch":SimpleNamespace(no_grad=lambda: lambda f: f), "tqdm":lambda items, **kw: items}
    exec(compile(ast.Module(body=functions(), type_ignores=[]), "v62-cpu", "exec"), ns)  # noqa: S102 - trusted generated CPU helpers, no notebook side effects
    return ns


@pytest.mark.parametrize("pos,neg,expected", [([1],[1],.5),([1,1],[1,1],.5),([0,1],[1,2],.125),([2],[1],1),([], [1], None)])
def test_rank_auc_ties(pos, neg, expected):
    assert helpers()["rank_auc"](pos, neg) == expected


def scenario(predictions):
    ns = helpers()
    truth = np.ones((4,4), bool)
    rle = ns["encode_one"](truth)
    ns.update(cache=[{"name":"a", "final":[ns["encode_one"](p) for p in predictions]}],
              file_to_ids={"a":[1,2]}, gt_rles={1:[rle],2:[rle]}, val_density={"a":2}, DENSE_CUT=1)
    return ns


def test_all_missed_truths_are_classified_as_blind():
    ns = scenario([])
    baseline, dense, sparse, anatomy = ns["score_final_preds"]()
    assert baseline["fn"] == dense == anatomy["fn_blind"] == 2
    assert sparse == 0


def test_annotator_duplicates_are_not_overmerges():
    ns = scenario([np.ones((4,4),bool)])
    _,_,_, anatomy = ns["score_final_preds"]()
    assert anatomy["overmerge_preds"] == 0
    assert anatomy["n_pred"] == 2  # one prediction scored against each annotator


def test_fold_contract_rejects_duplicate_and_missing_files():
    ns = helpers()
    for rows in ([{"file_name":"a","fold":0},{"file_name":"a","fold":1}],
                 [{"file_name":"b","fold":0}]):
        with pytest.raises(ValueError):
            ns["validate_fold_rows"](rows, ["a"], 2)


def test_baseline_reproduction_is_fail_closed():
    fn = helpers()["validate_baseline"]
    fn(.4471, .4471, .0005)
    for measured in (.4432, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            fn(measured, .4471, .0005)


def test_contract_mismatch_preserves_original_manifest(tmp_path):
    ns = helpers()
    path = tmp_path / "result.json"
    path.write_text('{"legacy": true}')
    with pytest.raises(ValueError):
        ns["validate_existing_manifest"](path, {"fold":2})
    assert json.loads(path.read_text()) == {"legacy":True}


def test_atomic_output_survives_serialization_failure(tmp_path):
    ns = helpers()
    ns.update(json=json, Path=Path, os=__import__("os"), tempfile=__import__("tempfile"), math=__import__("math"))
    path = tmp_path / "result.json"
    path.write_text('{"old": true}')
    with pytest.raises(ValueError):
        ns["atomic_json"](path, {"pq":float("nan")})
    assert json.loads(path.read_text()) == {"old":True}


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [[1,2]]])
def test_auc_rejects_invalid_values(scores):
    with pytest.raises(ValueError):
        helpers()["rank_auc"](scores, [0])


def test_auc_matches_pairwise_reference_with_mixed_ties():
    positive, negative = [0,1,1,2,4], [0,1,2,2,3]
    expected = sum((p > n) + .5 * (p == n) for p in positive for n in negative) / 25
    assert helpers()["rank_auc"](positive, negative) == expected


def test_unknown_oracle_is_refused():
    with pytest.raises(ValueError, match="Unknown"):
        helpers()["score_oracle"]("typo")


def test_empty_truth_records_count_predictions_as_orphans():
    ns = scenario([np.ones((4,4),bool)])
    ns["gt_rles"] = {1:[],2:[]}
    baseline,_,_, anatomy = ns["score_final_preds"]()
    assert baseline["fp"] == anatomy["fp_orphan"] == anatomy["n_pred"] == 2
    assert baseline["fn"] == anatomy["fn_blind"] == 0


def test_duplicate_predictions_keep_organizer_all_hit_semantics():
    ns = scenario([np.ones((4,4),bool),np.ones((4,4),bool)])
    baseline,_,_, anatomy = ns["score_final_preds"]()
    assert baseline["tp"] == 4  # both IoU hits in both annotator records
    assert baseline["fp"] == baseline["fn"] == 0
    assert anatomy["tp_gts"] == 2
    assert anatomy["n_pred"] == 4


def test_oracle_interventions_use_each_annotator_separately():
    ns = scenario([np.ones((4,4),bool)])
    ns["gt_rles"][2] = []
    baseline,_,_,_ = ns["score_final_preds"]()
    diagnostic = ns["score_oracle"]("perfect_verifier")
    assert baseline["fp"] == 1
    assert diagnostic["fp"] == 0
    assert diagnostic["tp"] == 1
    assert diagnostic["pq"] == 1


def test_feature_labels_include_empty_gt_negatives():
    ns = scenario([np.ones((4,4),bool)])
    ns["gt_rles"][2] = []
    ns["cache"][0].update(refined=ns["cache"][0]["final"], feats=[{"conf":.6}])
    auc, counts = ns["overlap_feature_auc"]()
    assert counts == {"overlap_positive":1,"overlap_negative":1}
    assert auc == {"conf":.5}


def frozen_inputs():
    return {"schema_version":1,"fold":2, **{key:"a"*64 for key in (
        "protocol_sha256","detector_sha256","refiner_sha256","annotations_sha256",
        "folds_sha256","preprocessed_sha256")}}


def test_only_independently_expected_contract_is_accepted(tmp_path):
    ns = helpers()
    path = tmp_path / "expected.json"
    actual = frozen_inputs()
    expected = {**actual,"baseline":{"pq":.4471,"tolerance":.0005}}
    with pytest.raises(FileNotFoundError):
        ns["require_expected_contract"](path, actual)
    path.write_text(json.dumps(expected))
    assert ns["require_expected_contract"](path, actual) == expected
    assert json.loads(path.read_text()) == expected


@pytest.mark.parametrize("key", list(frozen_inputs()))
def test_any_frozen_identity_mismatch_is_refused(tmp_path, key):
    ns = helpers()
    path = tmp_path / "expected.json"
    actual = frozen_inputs()
    expected = {**actual,"baseline":{"pq":.4471,"tolerance":.0005}}
    expected[key] = 0 if key in ("schema_version","fold") else "b"*64
    path.write_text(json.dumps(expected))
    with pytest.raises(ValueError):
        ns["require_expected_contract"](path, actual)


@pytest.mark.parametrize("name,fold", [("../x",0),("a\\b",0),("/a",0),("C:a",0),("a",True),("a",1.5),("a",3)])
def test_invalid_fold_records_are_refused(name, fold):
    with pytest.raises(ValueError):
        helpers()["validate_fold_rows"]([{"file_name":name,"fold":fold}], [name], 2)


def test_valid_folds_cover_all_inputs_and_nonempty_partitions():
    rows = [{"file_name":"a","fold":2},{"file_name":"b","fold":0}]
    assert helpers()["validate_fold_rows"](rows, ["a","b"], 2) == {"a":2,"b":0}
    with pytest.raises(ValueError):
        helpers()["validate_fold_rows"](rows[:1], ["a"], 2)


def test_atomic_replace_failure_preserves_output_and_cleans_temp(tmp_path, monkeypatch):
    import os
    path = tmp_path / "result.json"
    path.write_text('{"old": true}')
    def fail_replace(*args):
        raise OSError("interrupted")
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        helpers()["atomic_json"](path, {"new":1})
    assert json.loads(path.read_text()) == {"old":True}
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_output_roundtrip(tmp_path):
    path = tmp_path / "result.json"
    helpers()["atomic_json"](path, {"pq":.4})
    assert json.loads(path.read_text()) == {"pq":.4}


@pytest.mark.parametrize("fold", [0,1,2])
def test_all_generated_cells_parse_without_gpu_or_download(fold):
    cells = BUILDER.build_cells(fold)
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("\n".join(line for line in cell["source"].splitlines() if not line.lstrip().startswith("!")))
    assert "require_expected_contract" in cells[2]["source"]
    assert "torch.load" not in cells[2]["source"]


def test_committed_fold2_notebook_matches_builder():
    saved = json.loads((ROOT / "solar_filament_v62_error_anatomy_fold2_colab.ipynb").read_text(encoding="utf-8"))
    assert [(c["cell_type"],c["source"]) for c in saved["cells"]] == [
        (c["cell_type"],c["source"]) for c in BUILDER.build_cells(2)]
    assert all(c.get("outputs",[]) == [] for c in saved["cells"])


@pytest.mark.parametrize("pos,neg", [([], [float("nan")]), ([float("inf")], [])])
def test_single_class_auc_still_rejects_invalid_values(pos, neg):
    with pytest.raises(ValueError):
        helpers()["rank_auc"](pos, neg)


@pytest.mark.parametrize("replacement", [{}, {"pq":float("nan"),"tolerance":.001},
    {"pq":.4,"tolerance":True}, {"pq":.4,"tolerance":-1}])
def test_invalid_reproduction_guard_is_refused(tmp_path, replacement):
    path = tmp_path / "expected.json"
    actual = frozen_inputs()
    path.write_text(json.dumps({**actual,"baseline":replacement}))
    with pytest.raises(ValueError):
        helpers()["require_expected_contract"](path, actual)


@pytest.mark.parametrize("model", ["detector", "refiner"])
def test_checkpoint_changed_after_preflight_is_refused_before_load(tmp_path, model):
    ns = helpers()
    ns["CFG"].LOCAL_ROOT = tmp_path / "local"
    ns["CFG"].ARTIFACT_ROOT = tmp_path
    ns["CFG"].FOLD = 2
    checkpoint = tmp_path / ("detector.pt" if model == "detector" else "mediseg_refiner_fold2_resume.pt")
    checkpoint.write_bytes(b"preflight identity")
    expected = ns["file_sha256"](checkpoint)
    ns["INPUT_CONTRACT"] = {model + "_sha256":expected}
    checkpoint.write_bytes(b"Drive replaced these bytes")
    def forbidden_load(*args, **kwargs):
        pytest.fail("changed remote checkpoint reached model loader")
    if model == "detector":
        ns["YOLO"] = forbidden_load
        with pytest.raises(ValueError, match="checkpoint"):
            ns["collect_cache"](checkpoint)
    else:
        ns["MediSegUNet"] = lambda *args: SimpleNamespace(cuda=lambda:SimpleNamespace(eval=lambda:object()))
        ns["torch"].load = forbidden_load
        source = BUILDER.build_cells(2)[3]["source"].split("refiner = MediSegUNet",1)[1]
        with pytest.raises(ValueError, match="checkpoint"):
            exec("refiner = MediSegUNet" + source, ns)  # noqa: S102 - CPU load guard regression


def test_verified_snapshot_preserves_bytes_after_remote_replacement(tmp_path):
    ns = helpers()
    remote = tmp_path / "drive.pt"
    remote.write_bytes(b"expected weights")
    expected = ns["file_sha256"](remote)
    local = ns["verified_checkpoint_snapshot"](remote, expected, tmp_path / "local")
    assert local != remote
    assert local.suffix == ".pt"
    assert ns["file_sha256"](local) == expected
    remote.write_bytes(b"new Drive weights")
    assert local.read_bytes() == b"expected weights"


def test_bad_snapshot_is_removed_without_model_load(tmp_path):
    ns = helpers()
    remote = tmp_path / "drive.pt"
    remote.write_bytes(b"changed weights")
    destination = tmp_path / "local"
    with pytest.raises(ValueError, match="checkpoint"):
        ns["verified_checkpoint_snapshot"](remote, "a"*64, destination)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("model", ["detector", "refiner"])
def test_both_load_sites_use_verified_local_bytes(tmp_path, model):
    ns = helpers()
    ns["CFG"].LOCAL_ROOT = tmp_path / "local"
    ns["CFG"].ARTIFACT_ROOT = tmp_path
    ns["CFG"].FOLD = 2
    checkpoint = tmp_path / ("detector.pt" if model == "detector" else "mediseg_refiner_fold2_resume.pt")
    checkpoint.write_bytes(b"expected weights")
    expected = ns["file_sha256"](checkpoint)
    ns["INPUT_CONTRACT"] = {model + "_sha256":expected}
    def loader(path, **kwargs):
        local = Path(path)
        assert local != checkpoint
        assert local.parent == ns["CFG"].LOCAL_ROOT / "verified_checkpoints"
        checkpoint.write_bytes(b"Drive replacement at load time")
        assert ns["file_sha256"](local) == expected
        raise RuntimeError("verified loader reached")
    if model == "detector":
        ns["YOLO"] = loader
        with pytest.raises(RuntimeError, match="verified loader reached"):
            ns["collect_cache"](checkpoint)
    else:
        ns["MediSegUNet"] = lambda *args: SimpleNamespace(cuda=lambda:SimpleNamespace(eval=lambda:object()))
        ns["torch"].load = loader
        source = BUILDER.build_cells(2)[3]["source"].split("refiner = MediSegUNet",1)[1]
        with pytest.raises(RuntimeError, match="verified loader reached"):
            exec("refiner = MediSegUNet" + source, ns)  # noqa: S102 - CPU local load sentinel


def test_snapshot_copy_failure_cleans_partial_local_file(tmp_path, monkeypatch):
    import shutil
    def interrupted(source, target):
        Path(target).write_bytes(b"partial")
        raise OSError("copy interrupted")
    monkeypatch.setattr(shutil, "copyfile", interrupted)
    destination = tmp_path / "local"
    with pytest.raises(OSError, match="interrupted"):
        helpers()["verified_checkpoint_snapshot"](tmp_path / "drive.pt", "a"*64, destination)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("changed_input", ["annotations", "folds"])
def test_metadata_digest_identifies_bytes_actually_parsed(tmp_path, monkeypatch, changed_input):
    import hashlib
    import shutil

    import pandas as pd
    dataset = tmp_path / "data" / "sub" / "annotations"
    dataset.mkdir(parents=True)
    annotation_path = dataset / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    original_coco = {"images":[{"id":1,"file_name":"train.png","width":2048,"height":2048},
                               {"id":2,"file_name":"val.png","width":2048,"height":2048}],
                     "annotations":[{"image_id":1},{"image_id":2}]}
    annotation_path.write_text(json.dumps(original_coco), encoding="utf-8")
    fold_path = tmp_path / "grouped_folds.csv"
    fold_path.write_text("file_name,fold\ntrain.png,0\nval.png,2\n", encoding="utf-8")
    original_digests = {"annotations":hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                        "folds":hashlib.sha256(fold_path.read_bytes()).hexdigest()}
    preproc = tmp_path / "cache"
    preproc.mkdir()
    (preproc / "val.png").write_bytes(b"validation image bytes")
    detector = tmp_path / "detector.pt"
    detector.write_bytes(b"detector")
    (tmp_path / "mediseg_refiner_fold2_resume.pt").write_bytes(b"refiner")
    local = tmp_path / "local"
    local.mkdir()
    ns = helpers()
    ns.update(Path=Path, json=json, pd=pd, hashlib=hashlib, shutil=shutil,
              prod_checkpoint=lambda fold:detector,
              require_expected_contract=lambda path, actual:actual,
              validate_existing_manifest=lambda path, actual:None)
    ns["CFG"] = SimpleNamespace(DATA_CACHE=tmp_path / "data", ARTIFACT_ROOT=tmp_path,
        FOLD=2, DENSE_QUANTILE=.75, PREPROC_DIR=local, PREPROC_CACHE=preproc,
        PROTOCOL_SHA256="a"*64, EXPECTED_CONTRACT=tmp_path / "expected.json",
        MANIFEST=tmp_path / "result.json")
    old_load, old_loads, old_csv = json.load, json.loads, pd.read_csv
    def after_json_parse(data):
        if changed_input == "annotations" and isinstance(data,dict) and "images" in data:
            annotation_path.write_text('{"replacement": true}', encoding="utf-8")
        return data
    def load(*args, **kwargs):
        return after_json_parse(old_load(*args, **kwargs))
    def loads(*args, **kwargs):
        return after_json_parse(old_loads(*args, **kwargs))
    def read_csv(*args, **kwargs):
        frame = old_csv(*args, **kwargs)
        if changed_input == "folds":
            fold_path.write_text("file_name,fold\ntrain.png,2\nval.png,0\n", encoding="utf-8")
        return frame
    monkeypatch.setattr(json, "load", load)
    monkeypatch.setattr(json, "loads", loads)
    monkeypatch.setattr(pd, "read_csv", read_csv)
    exec(BUILDER.build_cells(2)[2]["source"], ns)  # noqa: S102 - metadata cell only, no model loading
    assert ns["ACTUAL_INPUTS"][changed_input + "_sha256"] == original_digests[changed_input]
    assert ns["coco"] == original_coco
    assert ns["fold_of_file"] == {"train.png":0,"val.png":2}
