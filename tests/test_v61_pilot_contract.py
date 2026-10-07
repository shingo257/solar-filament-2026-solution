"""No-fit SAM2 diagnostic CPU contracts; no real models, downloads or GPU."""
import ast
import importlib.util
import json
from collections import defaultdict
from itertools import combinations_with_replacement
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v61", ROOT / "scratch/build_v61_sam2_refiner_pilot.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def helpers():
    source = "\n".join(c["source"] for c in BUILDER.build_cells(2) if c["cell_type"] == "code")
    tree = ast.parse("\n".join(l for l in source.splitlines() if not l.lstrip().startswith("!")))
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) or
             isinstance(n,ast.ClassDef) and n.name == "PanopticQuality"]
    ns = {"np":np,"json":json,"Path":Path,"mask_utils":mask_utils,"defaultdict":defaultdict,
          "CFG":SimpleNamespace(MIN_AREA=1,DROP_THRESHOLD=.9,FOLD=2,CONF=.26,
              ARMS=("raw","refined","sam2_free","sam2_shrink")),
          "torch":SimpleNamespace(no_grad=lambda:lambda f:f),"tqdm":lambda seq,**kw:seq}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"v61-cpu","exec"),ns)  # noqa: S102 - generated CPU functions only
    return ns


def metrics(pq=.4471):
    meter = helpers()["PanopticQuality"]()
    meter.add_matrix(np.array([[2*pq,0],[0,0]]),2,2)
    return {"metrics":meter.summary(),"fn_dense_val":1,"fn_sparse_val":0}


def test_baseline_drift_stops_before_any_manifest_write(tmp_path):
    ns = helpers()
    ns["CFG"].MANIFEST = tmp_path / "result.json"
    ns["CFG"].BASELINE_PQ_REFINED = .4471
    ns["CFG"].ADOPT_DELTA = .005
    ns["CFG"].NOISE_BAND = .002
    ns.update(cache=[],valid_files_fold=[],sam2_diag={},time=__import__("time"),
              INPUT_CONTRACT={"baseline":{"pq":.4471,"tolerance":.0005}},
              score_arm=lambda cache,arm:metrics(.4432), prod_checkpoint=lambda fold:"detector.pt")
    with pytest.raises(ValueError,match="Baseline"):
        exec(BUILDER.build_cells(2)[6]["source"],ns)  # noqa: S102 - guarded scoring cell with CPU sentinels
    assert not ns["CFG"].MANIFEST.exists()


def test_legacy_partial_results_are_refused_without_overwriting(tmp_path):
    ns = helpers()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"runs":[{"arm":"raw","fold":1,**metrics()}]}))
    old = path.read_bytes()
    with pytest.raises(ValueError,match="manifest"):
        ns["validate_existing_manifest"](path,{"fold":2})
    assert path.read_bytes() == old


def test_paired_cache_rejects_missing_arm_predictions():
    ns = helpers()
    row = {"name":"a","scores":[.8],"raw":[{}],"refined":[],"sam2_free":[{}],"sam2_shrink":[{}]}
    with pytest.raises(ValueError,match="paired"):
        ns["validate_paired_cache"]([row],["a"])


@pytest.mark.parametrize("measured", [.4432,float("nan"),float("inf")])
def test_explicit_baseline_guard_rejects_drift_and_nonfinite(measured):
    with pytest.raises(ValueError):
        helpers()["validate_baseline"](measured,.4471,.0005)


def test_sam2_revision_requires_exact_commit_and_config_sha(tmp_path):
    with pytest.raises(ValueError,match="commit"):
        helpers()["snapshot_sam2_source"](tmp_path,"main","a"*64,tmp_path / "local")


def test_existing_sam2_import_cannot_shadow_verified_source(tmp_path):
    with pytest.raises(ValueError,match="import"):
        helpers()["require_fresh_sam2_import"]({"sam2.build_sam":object()})


@pytest.fixture
def source_repository(tmp_path):
    import hashlib
    import subprocess
    repository = tmp_path / "source"
    repository.mkdir()
    def git(*args):
        return subprocess.check_output(["git","-C",str(repository),*args],text=True).strip()
    git("init","-q")
    config = repository / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"model: synthetic-test-only\n")
    (repository / "sam2/__init__.py").write_bytes(b"# synthetic source fixture\n")
    (repository / "LICENSE").write_bytes(b"synthetic test fixture\n")
    git("add","--","sam2","LICENSE")
    git("-c","user.name=CPU contract","-c","user.email=cpu@example.invalid","-c","commit.gpgsign=false",
        "commit","-qm","synthetic source fixture")
    return repository, git("rev-parse","HEAD"), hashlib.sha256(config.read_bytes()).hexdigest()


def test_source_snapshot_uses_commit_objects_not_dirty_checkout(tmp_path, source_repository):
    repository, commit, digest = source_repository
    (repository / "sam2/__init__.py").write_bytes(b"unapproved dirty source")
    local = helpers()["snapshot_sam2_source"](repository,commit,digest,tmp_path / "local")
    assert (local / "sam2/__init__.py").read_bytes() == b"# synthetic source fixture\n"
    assert local != repository


def test_source_config_mismatch_removes_snapshot(tmp_path, source_repository):
    repository, commit, _ = source_repository
    destination = tmp_path / "local"
    with pytest.raises(ValueError,match="config"):
        helpers()["snapshot_sam2_source"](repository,commit,"a"*64,destination)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("member_name,is_link", [("sam2/../../escape",False),("/escape",False),("sam2/link",True)])
def test_unsafe_archive_members_are_rejected_and_cleaned(tmp_path,monkeypatch,member_name,is_link):
    import io
    import subprocess
    import tarfile
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload,mode="w") as archive:
        item = tarfile.TarInfo(member_name)
        if is_link:
            item.type = tarfile.SYMTYPE
            item.linkname = "/outside"
        else:
            item.size = 1
        archive.addfile(item,None if is_link else io.BytesIO(b"x"))
    commit = "a"*40
    def output(args,**kwargs):
        return commit if "rev-parse" in args else payload.getvalue()
    monkeypatch.setattr(subprocess,"check_output",output)
    destination = tmp_path / "local"
    with pytest.raises(ValueError):
        helpers()["snapshot_sam2_source"](tmp_path,commit,"a"*64,destination)
    assert list(destination.iterdir()) == []


def frozen_contract():
    digest_keys = ("detector_sha256","refiner_sha256","annotations_sha256","folds_sha256",
                   "preprocessed_sha256","sam2_config_sha256","sam2_checkpoint_sha256")
    return {"schema_version":1,"fold":2,"protocol_sha256":BUILDER.PROTOCOL_SHA256,
            **{k:"a"*64 for k in digest_keys},"sam2_commit":"b"*40,
            "sam2_config":"configs/sam2.1/sam2.1_hiera_l.yaml",
            "baseline":{"pq":.4471,"tolerance":.0005},
            "runtime_versions":{key:"synthetic-test-only" for key in (
                "python","torch","torchvision","numpy","pandas","opencv","ultralytics",
                "pycocotools","hydra-core","omegaconf","iopath","Pillow")}}


def test_expected_contract_is_required_and_independent(tmp_path):
    ns = helpers()
    path = tmp_path / "expected.json"
    with pytest.raises(FileNotFoundError):
        ns["read_expected_contract"](path,2,BUILDER.PROTOCOL_SHA256)
    expected = frozen_contract()
    path.write_text(json.dumps(expected))
    assert ns["read_expected_contract"](path,2,BUILDER.PROTOCOL_SHA256) == expected
    ns["check_actual_inputs"](expected,{k:v for k,v in expected.items() if k != "baseline"})
    assert json.loads(path.read_text()) == expected


@pytest.mark.parametrize("key", [k for k in frozen_contract() if k != "baseline"])
def test_any_frozen_input_config_or_runtime_mismatch_is_refused(key):
    expected = frozen_contract()
    actual = {k:v for k,v in expected.items() if k != "baseline"}
    actual[key] = "different"
    with pytest.raises(ValueError):
        helpers()["check_actual_inputs"](expected,actual)


@pytest.mark.parametrize("key,value", [("sam2_commit","main"),("sam2_commit","a; echo injected"),
    ("sam2_config","../other.yaml"),("sam2_config_sha256","NOT_A_HASH"),
    ("baseline",{"pq":.4,"tolerance":True}),("runtime_versions",{}),("fold",True),
    ("protocol_sha256","b"*64),("schema_version",True)])
def test_invalid_expected_contract_is_refused(tmp_path,key,value):
    expected = frozen_contract()
    expected[key] = value
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(expected))
    with pytest.raises(ValueError):
        helpers()["read_expected_contract"](path,2,BUILDER.PROTOCOL_SHA256)


@pytest.mark.parametrize("name", ["sam2","sam2.build_sam","sam2.modeling.foo"])
def test_any_preloaded_sam2_module_requires_fresh_runtime(name):
    with pytest.raises(ValueError,match="import"):
        helpers()["require_fresh_sam2_import"]({name:object()})
    helpers()["require_fresh_sam2_import"]({"other_sam2_name":object()})


def paired_cache(ns):
    mask = np.zeros((4,4),bool)
    mask[:2,:2] = True
    rle = ns["encode_one"](mask)
    return [{"name":"a","scores":[.8],**{a:[dict(rle)] for a in ns["CFG"].ARMS}}]


def test_valid_paired_cache_and_digest_are_stable():
    ns = helpers()
    cache = paired_cache(ns)
    ns["validate_paired_cache"](cache,["a"])
    digest = ns["paired_cache_digest"](cache)
    assert len(digest) == 64
    assert ns["paired_cache_digest"](cache) == digest
    cache[0]["scores"] = [.7]
    assert ns["paired_cache_digest"](cache) != digest


@pytest.mark.parametrize("case", ["duplicate","missing","nan","shape"])
def test_invalid_paired_cache_is_refused(case):
    ns = helpers()
    cache = paired_cache(ns)
    if case == "duplicate":
        cache.append(cache[0])
    elif case == "missing":
        cache[0]["sam2_free"] = []
    elif case == "nan":
        cache[0]["scores"] = [float("nan")]
    else:
        cache[0]["sam2_free"] = [ns["encode_one"](np.ones((2,2),bool))]
    with pytest.raises(ValueError):
        ns["validate_paired_cache"](cache,["a"])


def result_rows(ns):
    return [{"arm":a,"fold":2,"conf":.26,**metrics()} for a in ns["CFG"].ARMS]


def test_complete_manifest_validates_without_reusing_arm_results(tmp_path):
    ns = helpers()
    contract = frozen_contract()
    path = tmp_path / "result.json"
    payload = complete_manifest(ns,contract)
    ns["atomic_json"](path,payload)
    assert ns["validate_existing_manifest"](path,contract) is None
    assert json.loads(path.read_text()) == payload


@pytest.mark.parametrize("case", ["partial","duplicate","wrongfold","nan","fncounts"])
def test_invalid_paired_results_are_refused(case):
    ns = helpers()
    rows = result_rows(ns)
    if case == "partial":
        rows.pop()
    elif case == "duplicate":
        rows[1]["arm"] = "raw"
    elif case == "wrongfold":
        rows[0]["fold"] = 1
    elif case == "nan":
        rows[0]["metrics"]["pq"] = float("nan")
    else:
        rows[0]["fn_dense_val"] = 3
    with pytest.raises(ValueError):
        ns["validate_run_rows"](rows)


def test_atomic_manifest_failure_keeps_prior_complete_unit(tmp_path,monkeypatch):
    import os
    path = tmp_path / "report.json"
    path.write_text('{"old": true}')
    def interrupted(*args):
        raise OSError("interrupted")
    monkeypatch.setattr(os,"replace",interrupted)
    with pytest.raises(OSError):
        helpers()["atomic_json"](path,{"new":1})
    assert json.loads(path.read_text()) == {"old":True}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("fold", [0,1,2])
def test_all_generated_cells_parse_with_no_fit_or_download(fold):
    for cell in BUILDER.build_cells(fold):
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
            assert "urlretrieve" not in cell["source"]
            assert "git+https" not in cell["source"]
            assert ".fit(" not in cell["source"]


def test_committed_notebook_matches_generated_source():
    path = ROOT / "solar_filament_v61_sam2_refiner_fold2_colab.ipynb"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [(c["cell_type"],c["source"]) for c in saved["cells"]] == [
        (c["cell_type"],c["source"]) for c in BUILDER.build_cells(2)]
    assert all(c.get("outputs",[]) == [] for c in saved["cells"])


def test_cleanup_refuses_outside_snapshot_destination(tmp_path, source_repository, monkeypatch):
    import tempfile
    repository, commit, _ = source_repository
    destination = tmp_path / "local"
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "preserve.txt"
    marker.write_text("preserve")
    monkeypatch.setattr(tempfile,"mkdtemp",lambda **kwargs:str(outside))
    with pytest.raises(ValueError,match="cleanup target"):
        helpers()["snapshot_sam2_source"](repository,commit,"a"*64,destination)
    assert marker.read_text() == "preserve"


def complete_manifest(ns,contract):
    return {"experiment_id":"v61_sam2_refiner_fold2_diagnostic_v1",
        "status":"complete_diagnostic","input_contract":contract,"paired_cache_sha256":"a"*64,
        "scope":"exploratory paired diagnostic; no mechanism adoption or scientific acceptance",
        "count_unit":"annotator_image","matching":"official all IoU > 0.5 hits",
        "runs":result_rows(ns),"paired_delta_pq":{a:0.0 for a in ns["CFG"].ARMS},
        "sam2_diagnostics":{"n_proposals":0,"sam2_pred_iou_mean":None,
            "iou_sam2_vs_proposal_mean":None,"iou_sam2_vs_proposal_median":None,
            "sam2_area_over_proposal_mean":None,"sam2_empty_fraction":0.0},
        "runtime":{"service":"Google Colab","project":"synthetic-test-only",
            "notebook":"solar_filament_v61_sam2_refiner_fold2_colab.ipynb","gpu":"synthetic GPU",
            "duration_seconds":0.0,"peak_vram_bytes":0,
            "measurement_scope":"detector checkpoint snapshot/loading, inference/refinement and paired scoring; earlier SAM2/refiner setup excluded",
            "compute_units_used":None,"compute_units_note":"Record from Colab UI; runtime API unavailable",
            "versions":contract["runtime_versions"]},"completed_at_utc":"2026-10-07T00:00:00+00:00"}


@pytest.mark.parametrize("field",list(complete_manifest(helpers(),frozen_contract())))
def test_partial_top_level_manifest_is_preserved(tmp_path,field):
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    del payload[field]
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload))
    old = path.read_bytes()
    with pytest.raises(ValueError,match="manifest"):
        ns["validate_existing_manifest"](path,contract)
    assert path.read_bytes() == old


@pytest.mark.parametrize("name",["hydra","hydra.core.global_hydra"])
def test_cached_uninitialized_hydra_is_refused_before_import(tmp_path,monkeypatch,name):
    import importlib
    import sys
    from types import ModuleType
    global_module = ModuleType("hydra.core.global_hydra")
    global_module.GlobalHydra = SimpleNamespace(instance=lambda:SimpleNamespace(is_initialized=lambda:False))
    monkeypatch.setitem(sys.modules,"hydra.core.global_hydra",global_module)
    monkeypatch.setitem(sys.modules,name,global_module)
    monkeypatch.setattr(importlib,"import_module",lambda name:SimpleNamespace(__file__=tmp_path / "sam2/__init__.py"))
    with pytest.raises(ValueError,match="import"):
        helpers()["activate_sam2_snapshot"](tmp_path)


def test_generated_scoring_cell_emits_complete_roundtrip_without_reusing_arms(tmp_path):
    from datetime import datetime, timezone
    ns = helpers()
    contract = frozen_contract()
    contract["baseline"] = {"pq":.4,"tolerance":0.0}
    meter = ns["PanopticQuality"]()
    meter.add_matrix(np.array([[.8,0],[0,0]]),2,2)
    scored = {"metrics":meter.summary(),"fn_dense_val":1,"fn_sparse_val":0}
    ns["CFG"].MANIFEST = tmp_path / "result.json"
    ns["CFG"].DRIVE_ROOT = tmp_path
    called = []
    def score(cache,arm):
        called.append(arm)
        return scored
    ns.update(INPUT_CONTRACT=contract,cache=[],valid_files_fold=[],
        sam2_diag=complete_manifest(ns,contract)["sam2_diagnostics"],score_arm=score,
        RUNTIME_VERSIONS=contract["runtime_versions"],datetime=datetime,timezone=timezone,
        started=0,time=SimpleNamespace(time=lambda:1.0))
    ns["torch"].cuda = SimpleNamespace(get_device_name=lambda _:"synthetic GPU",max_memory_allocated=lambda:0)
    cell = BUILDER.build_cells(2)[6]["source"]
    exec(cell,ns)  # noqa: S102 - generated scoring with CPU-only sentinels
    ns["validate_existing_manifest"](ns["CFG"].MANIFEST,contract)
    emitted = json.loads(ns["CFG"].MANIFEST.read_text())
    assert emitted["runtime"]["measurement_scope"] == (
        "detector checkpoint snapshot/loading, inference/refinement and paired scoring; earlier SAM2/refiner setup excluded")
    exec(cell,ns)  # noqa: S102 - complete prior unit cannot skip freshly scoring any arm
    assert called == ["refined","raw","sam2_free","sam2_shrink"]*2
    ns["validate_existing_manifest"](ns["CFG"].MANIFEST,contract)


@pytest.mark.parametrize("case",["diag_missing","diag_nan","diag_count","delta_mismatch","runtime_missing",
    "runtime_nan","runtime_versions","timestamp","scope","runtime_type","row_type","metrics_type"])
def test_malformed_manifest_provenance_is_preserved(tmp_path,case):
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    if case == "diag_missing":
        del payload["sam2_diagnostics"]["n_proposals"]
    elif case == "diag_nan":
        payload["sam2_diagnostics"].update(n_proposals=1,sam2_pred_iou_mean=float("nan"))
    elif case == "diag_count":
        payload["sam2_diagnostics"]["n_proposals"] = True
    elif case == "delta_mismatch":
        payload["paired_delta_pq"]["raw"] = .1
    elif case == "runtime_missing":
        del payload["runtime"]["gpu"]
    elif case == "runtime_nan":
        payload["runtime"]["duration_seconds"] = float("nan")
    elif case == "runtime_versions":
        payload["runtime"]["versions"] = {}
    elif case == "timestamp":
        payload["completed_at_utc"] = "2026-10-07T00:00:00"
    elif case == "scope":
        payload["scope"] = ""
    elif case == "row_type":
        payload["runs"][0] = None
    elif case == "metrics_type":
        payload["runs"][0]["metrics"] = []
    else:
        payload["runtime"] = []
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    old = path.read_bytes()
    with pytest.raises(ValueError):
        ns["validate_existing_manifest"](path,contract)
    assert path.read_bytes() == old


def test_predicted_iou_estimate_can_exceed_probability_range(tmp_path):
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    payload["sam2_diagnostics"].update(n_proposals=1,sam2_pred_iou_mean=1.2,
        iou_sam2_vs_proposal_mean=.3,iou_sam2_vs_proposal_median=.3,sam2_area_over_proposal_mean=2.0)
    ns["validate_manifest"](payload,contract)


@pytest.mark.parametrize("foreign_sam2",[False,True])
def test_generated_postimport_loop_allows_external_fresh_hydra_only(tmp_path,foreign_sam2):
    snapshot = (tmp_path / "verified").resolve()
    modules = {"sam2":SimpleNamespace(__file__=snapshot / "sam2/__init__.py"),
        "sam2.build_sam":SimpleNamespace(__file__=snapshot / "sam2/build_sam.py"),
        "hydra":SimpleNamespace(__file__=tmp_path / "site-packages/hydra/__init__.py"),
        "hydra.core.global_hydra":SimpleNamespace(__file__=tmp_path / "site-packages/hydra/core/global_hydra.py")}
    if foreign_sam2:
        modules["sam2.build_sam"].__file__ = tmp_path / "foreign/sam2/build_sam.py"
    tree = ast.parse(BUILDER.build_cells(2)[3]["source"])
    loop = next(n for n in tree.body if isinstance(n,ast.For) and "sys.modules.items" in ast.unparse(n.iter))
    code = compile(ast.Module(body=[loop],type_ignores=[]),"generated-import-loop","exec")
    ns = {"Path":Path,"sys":SimpleNamespace(modules=modules),"SAM2_SOURCE_SNAPSHOT":snapshot}
    if foreign_sam2:
        with pytest.raises(ValueError,match="outside verified snapshot"):
            exec(code,ns)  # noqa: S102 - actual generated import guard with synthetic module paths
    else:
        exec(code,ns)  # noqa: S102 - fresh external Hydra is a runtime dependency, not SAM2 source


@pytest.mark.parametrize("case",["missing","nan","inf","bool","string","negative","wrong",
    "extra_metric","extra_run","missing_run_key"])
def test_nested_run_schema_corruption_is_preserved(tmp_path,case):
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    for row in payload["runs"]:
        if case == "missing":
            del row["metrics"]["denominator"]
        elif case == "extra_metric":
            row["metrics"]["unsupported"] = 1
        elif case == "extra_run":
            row["unsupported"] = 1
        elif case == "missing_run_key":
            del row["fn_sparse_val"]
        else:
            row["metrics"]["denominator"] = {"nan":float("nan"),"inf":float("inf"),
                "bool":True,"string":"2","negative":-1,"wrong":2.5}[case]
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload))
    old = path.read_bytes()
    with pytest.raises(ValueError):
        ns["validate_existing_manifest"](path,contract)
    assert path.read_bytes() == old


@pytest.mark.parametrize("matrix,n_gt,n_pred,denominator",[
    (np.empty((0,0)),0,0,0.0),(np.empty((0,1)),0,1,.5),
    (np.array([[.8,.8]]),1,2,2.0)])
def test_scorer_denominator_empty_half_and_all_hit_cases(matrix,n_gt,n_pred,denominator):
    ns = helpers()
    meter = ns["PanopticQuality"]()
    meter.add_matrix(matrix,n_gt,n_pred)
    measured = meter.summary()
    assert measured["denominator"] == denominator
    rows = [{"arm":arm,"fold":2,"conf":.26,"metrics":dict(measured),
        "fn_dense_val":measured["fn"],"fn_sparse_val":0} for arm in ns["CFG"].ARMS]
    ns["validate_run_rows"](rows)


@pytest.mark.parametrize("case",["rq","pq","sq","zero_tp_sq"])
def test_corrupted_quality_components_are_preserved(tmp_path,case):
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    # Use actual consistent components, then corrupt exactly the stated identity.
    for row in payload["runs"]:
        row["metrics"].update(pq=.4471,sq=.8942,rq=.5)
        if case == "zero_tp_sq":
            row["metrics"].update(tp=0,denominator=1.0,pq=0.0,rq=0.0,sq=.7)
        else:
            row["metrics"][case] += .0001
    # The synthetic expected PQ follows the corrupted value so baseline guards
    # cannot accidentally mask missing within-result consistency validation.
    contract["baseline"]["pq"] = payload["runs"][0]["metrics"]["pq"]
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(payload))
    old = path.read_bytes()
    with pytest.raises(ValueError):
        ns["validate_existing_manifest"](path,contract)
    assert path.read_bytes() == old


def test_float64_reassociation_tolerance_accepts_small_component_roundoff():
    ns = helpers()
    rows = result_rows(ns)
    for row in rows:
        row["metrics"]["rq"] += 5e-13
        row["metrics"]["pq"] += 5e-13
    ns["validate_run_rows"](rows)


@pytest.mark.parametrize("n,fraction,valid",[(3,.5,False),(3,1/3,True),(3,2/3,True),
    (3,0.0,True),(3,1.0,True),(7,3/7,True),(0,.5,False),(0,0.0,True)])
def test_empty_fraction_matches_integer_proposal_count(tmp_path,n,fraction,valid):
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    if n:
        payload["sam2_diagnostics"].update(n_proposals=n,sam2_pred_iou_mean=.7,
            iou_sam2_vs_proposal_mean=.3,iou_sam2_vs_proposal_median=.3,sam2_area_over_proposal_mean=2.0)
    payload["sam2_diagnostics"]["sam2_empty_fraction"] = fraction
    if n and fraction >= .5:
        payload["sam2_diagnostics"]["iou_sam2_vs_proposal_median"] = 0.0
    if n and fraction == 1.0:
        payload["sam2_diagnostics"].update(iou_sam2_vs_proposal_mean=0.0,sam2_area_over_proposal_mean=0.0)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    old = path.read_bytes()
    if valid:
        ns["validate_existing_manifest"](path,contract)
    else:
        with pytest.raises(ValueError):
            ns["validate_existing_manifest"](path,contract)
    assert path.read_bytes() == old


@pytest.mark.parametrize("field",["iou_sam2_vs_proposal_mean","iou_sam2_vs_proposal_median",
    "sam2_area_over_proposal_mean"])
def test_all_empty_nonzero_aggregates_are_preserved(tmp_path,field):
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    payload["sam2_diagnostics"].update(n_proposals=3,sam2_empty_fraction=1.0,
        sam2_pred_iou_mean=.7,iou_sam2_vs_proposal_mean=0.0,
        iou_sam2_vs_proposal_median=0.0,sam2_area_over_proposal_mean=0.0)
    payload["sam2_diagnostics"][field] = .1
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(payload))
    old = path.read_bytes()
    with pytest.raises(ValueError):
        ns["validate_existing_manifest"](path,contract)
    assert path.read_bytes() == old


def mask_diagnostics(kinds):
    proposal = np.array([[True,False],[True,False]])
    masks = {"empty":np.zeros_like(proposal),"full":proposal,
        "half":np.array([[True,False],[False,False]]),"disjoint":~proposal}
    sam2_masks = [masks[kind] for kind in kinds]
    overlaps = [float((mask & proposal).sum()) / float((mask | proposal).sum())
        for mask in sam2_masks]
    areas = [float(mask.sum()) / max(float(proposal.sum()),1.0) for mask in sam2_masks]
    return {"n_proposals":len(kinds),"sam2_pred_iou_mean":None,
        "iou_sam2_vs_proposal_mean":float(np.mean(overlaps)),
        "iou_sam2_vs_proposal_median":float(np.median(overlaps)),
        "sam2_area_over_proposal_mean":float(np.mean(areas)),
        "sam2_empty_fraction":sum(not mask.any() for mask in sam2_masks) / len(kinds)}


@pytest.mark.parametrize("kinds,corruption",[
    (["full","empty","empty","empty"],
        {"iou_sam2_vs_proposal_mean":.9,"iou_sam2_vs_proposal_median":.1}),
    (["half"],{"iou_sam2_vs_proposal_median":.7}),
    (["full"],{"iou_sam2_vs_proposal_mean":.9,"iou_sam2_vs_proposal_median":.9,
        "sam2_area_over_proposal_mean":.1}),
    (["full","empty"],{"iou_sam2_vs_proposal_median":.4}),
    (["full","empty","empty"],{"iou_sam2_vs_proposal_median":.3}),
    (["full","empty","empty"],{"iou_sam2_vs_proposal_mean":.8}),
    (["full","empty"],{"sam2_area_over_proposal_mean":0.0}),
    (["disjoint","disjoint","disjoint"],{"iou_sam2_vs_proposal_median":.1}),
    (["full"]*3,{"iou_sam2_vs_proposal_mean":.1,"iou_sam2_vs_proposal_median":.9}),
    (["full"]*3,{"iou_sam2_vs_proposal_mean":.9,"iou_sam2_vs_proposal_median":.1}),
    (["full","full","empty"],
        {"iou_sam2_vs_proposal_mean":.6,"iou_sam2_vs_proposal_median":.2}),
    (["full","full","empty","empty"],
        {"iou_sam2_vs_proposal_mean":.4,"iou_sam2_vs_proposal_median":.6}),
])
def test_impossible_mask_diagnostics_are_preserved(tmp_path,kinds,corruption):
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    payload["sam2_diagnostics"] = mask_diagnostics(kinds)
    payload["sam2_diagnostics"].update(corruption)
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(payload))
    old = path.read_bytes()
    with pytest.raises(ValueError):
        ns["validate_existing_manifest"](path,contract)
    assert path.read_bytes() == old


@pytest.mark.parametrize("kinds",[["empty"],["half"],["disjoint"],
    ["full","empty"],["full","empty","empty"],
    ["full","full","empty","empty"],["disjoint","empty"]])
def test_concrete_mask_diagnostic_boundaries_are_valid(tmp_path,kinds):
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    payload["sam2_diagnostics"] = mask_diagnostics(kinds)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    old = path.read_bytes()
    ns["validate_existing_manifest"](path,contract)
    assert path.read_bytes() == old


def test_all_small_concrete_mask_multisets_have_valid_diagnostics():
    ns = helpers()
    contract = frozen_contract()
    payload = complete_manifest(ns,contract)
    checked = 0
    for n in range(1,7):
        for kinds in combinations_with_replacement(("empty","full","half","disjoint"),n):
            payload["sam2_diagnostics"] = mask_diagnostics(kinds)
            ns["validate_manifest"](payload,contract)
            checked += 1
    assert checked == 209
