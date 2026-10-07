"""Build exploratory no-fit SAM2 box-prompt comparisons from frozen local inputs.

Recovered code only from PR244 da2d8d15: raw/refined/sam2_free/sam2_shrink use
identical detector proposal indices and the official per-annotator all-hit PQ.
Mask changes can alter SQ, FP and FN; no fixed-FN or mechanism-adoption claim is
made. Historical rejected/STOP decisions and GPU result documents are unchanged.

An independently frozen artifacts/v61_expected_contract_fold{N}.json is required
before any model loading. Required keys: schema_version=1, fold, protocol_sha256,
detector_sha256, refiner_sha256, annotations_sha256, folds_sha256,
preprocessed_sha256, sam2_commit (full40hex), sam2_config (canonical SAM2.1 large),
sam2_config_sha256, sam2_checkpoint_sha256, runtime_versions and
baseline={pq,tolerance}. No canonical hashes/tolerance are generated here.
preprocessed_sha256 uses UTF-8 json.dumps(validation_filename_to_SHA256,
sort_keys=True). Runtime version keys are documented in the generated notebook.

Pre-provision dependencies, the SAM2 checkpoint and a local Git repository
containing the approved commit at artifacts/sam2_source. No source/checkpoint
download or unpinned SAM2 installation occurs. Exact committed source objects are
archived into a unique local snapshot, including the expected config. Fresh SAM2
and Hydra imports are mandatory; restart the runtime to repeat model setup.
Models consume hash-verified local checkpoint snapshots, metadata consumes one
captured byte buffer per file, and all arms share the same image bytes/proposals.

Results are one complete atomic paired diagnostic unit after baseline reproduction.
Partial/legacy manifests are rejected; no per-arm result resume or reuse occurs.
Interrupted evaluation reruns all arms. Independent GPU/scientific acceptance is
unverified and no adoption/rejection is inferred from the exploratory differences.

Usage: python scratch/build_v61_sam2_refiner_pilot.py --fold 2
Generation alone performs no model loading, fitting, GPU/cloud launch or download.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")

# 標準refinerパイプラインの全fold OOF基準(README「現在地」)
BASELINE_PQ_REFINED = {0: 0.4393, 1: 0.4415, 2: 0.4471}
DENSE_QUANTILE = 0.75
ARMS = ("raw", "refined", "sam2_free", "sam2_shrink")
PROTOCOL_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes().replace(b"\r\n",b"\n")
    + MODELS.replace("\r\n","\n").encode("utf-8")
).hexdigest()

# Standalone notebook helpers: audited v62 byte/snapshot guards plus v61 pairing.
GUARDS = r"""def file_sha256(path):
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def verified_checkpoint_snapshot(source, expected_sha256, destination):
    # A unique local copy isolates the loader from subsequent Drive syncs.
    # Verify the bytes actually copied, including any replacement during copy.
    import shutil
    import tempfile
    from pathlib import Path
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".pt", delete=False) as handle:
        local = Path(handle.name)
    try:
        shutil.copyfile(source, local)
        if file_sha256(local) != expected_sha256:
            raise ValueError("Copied checkpoint differs from frozen input identity")
        return local
    except BaseException:
        local.unlink(missing_ok=True)
        raise

def validate_fold_rows(rows, files, fold):
    from pathlib import PurePosixPath
    if type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("Invalid evaluation fold")
    mapping = {}
    for row in rows:
        name, value = row.get("file_name"), row.get("fold")
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).name != name or ":" in name or name in (".", "..")
                or type(value) is not int or value not in (0, 1, 2) or name in mapping):
            raise ValueError("Invalid or duplicate fold assignment")
        mapping[name] = value
    if set(mapping) != set(files):
        raise ValueError("Fold assignments must cover exactly the consumed files")
    if not any(v == fold for v in mapping.values()) or not any(v != fold for v in mapping.values()):
        raise ValueError("Training and validation partitions must be nonempty")
    return mapping

def validate_baseline(measured, expected, tolerance):
    import math
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (measured, expected, tolerance))
            or not 0 <= measured <= 1 or not 0 <= expected <= 1 or not 0 <= tolerance < 1
            or abs(measured - expected) > tolerance):
        raise ValueError("Baseline reproduction guard failed; no result accepted")

def atomic_json(path, payload):
    import json
    import os
    import tempfile
    from pathlib import Path
    path = Path(path)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def read_expected_contract(path, fold, protocol_sha256):
    import json
    import re
    from pathlib import Path
    expected = json.loads(Path(path).read_bytes().decode("utf-8"))
    keys = {"schema_version", "fold", "protocol_sha256", "detector_sha256", "refiner_sha256",
            "annotations_sha256", "folds_sha256", "preprocessed_sha256", "baseline",
            "sam2_commit", "sam2_config", "sam2_config_sha256", "sam2_checkpoint_sha256", "runtime_versions"}
    if not isinstance(expected, dict) or set(expected) != keys:
        raise ValueError("Independently frozen expected contract is incomplete")
    if (type(expected["schema_version"]) is not int or expected["schema_version"] != 1
            or type(expected["fold"]) is not int or expected["fold"] != fold
            or expected["protocol_sha256"] != protocol_sha256):
        raise ValueError("Frozen schema/fold/protocol differs")
    for key in (k for k in keys if k.endswith("_sha256")):
        if not isinstance(expected[key], str) or re.fullmatch("[0-9a-f]{64}", expected[key]) is None:
            raise ValueError("Invalid expected SHA256")
    if not isinstance(expected["sam2_commit"],str) or re.fullmatch("[0-9a-f]{40}",expected["sam2_commit"]) is None:
        raise ValueError("SAM2 requires an independently approved full commit")
    if expected["sam2_config"] != "configs/sam2.1/sam2.1_hiera_l.yaml":
        raise ValueError("This diagnostic supports only the source SAM2.1 large configuration")
    baseline = expected["baseline"]
    if not isinstance(baseline,dict) or set(baseline) != {"pq","tolerance"}:
        raise ValueError("Explicit baseline reproduction guard is required")
    validate_baseline(baseline["pq"],baseline["pq"],baseline["tolerance"])
    versions = expected["runtime_versions"]
    version_keys = {"python","torch","torchvision","numpy","pandas","opencv","ultralytics",
                    "pycocotools","hydra-core","omegaconf","iopath","Pillow"}
    if not isinstance(versions,dict) or set(versions) != version_keys or any(
            not isinstance(v,str) or not v for v in versions.values()):
        raise ValueError("Frozen runtime versions are required")
    return expected


def check_actual_inputs(expected, actual):
    if {k:v for k,v in expected.items() if k != "baseline"} != actual:
        raise ValueError("Consumed inputs/config/runtime differ from frozen expected contract")


def require_fresh_sam2_import(modules):
    if any(name in ("sam2", "hydra") or name.startswith(("sam2.", "hydra.")) for name in modules):
        raise ValueError("SAM2/Hydra import already cached; restart the runtime before using the verified snapshot")


def snapshot_sam2_source(repository, commit, config_sha256, destination):
    import io
    import os
    import re
    import shutil
    import subprocess
    import tarfile
    import tempfile
    from pathlib import Path, PurePosixPath
    if not isinstance(commit,str) or re.fullmatch("[0-9a-f]{40}",commit) is None:
        raise ValueError("SAM2 source must use an exact approved commit")
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    git_environment = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1")
    resolved = subprocess.check_output(["git","-C",str(repository),"rev-parse","--verify",commit+"^{commit}"],
                                       env=git_environment,text=True).strip()
    if resolved != commit:
        raise ValueError("SAM2 commit resolution differs")
    payload = subprocess.check_output(["git","-C",str(repository),"-c","core.autocrlf=false","-c","core.eol=lf","archive","--format=tar",commit,"sam2","LICENSE"],
                                      env=git_environment)
    local = Path(tempfile.mkdtemp(prefix="sam2-",dir=destination))
    try:
        with tarfile.open(fileobj=io.BytesIO(payload),mode="r:") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if (path.is_absolute() or ".." in path.parts or "\\" in member.name or ":" in member.name
                        or not path.parts or path.parts[0] not in ("sam2","LICENSE")):
                    raise ValueError("Unsafe SAM2 archive member")
                target = local.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as handle:
                        target.write_bytes(handle.read())
                else:
                    raise ValueError("SAM2 archive links/special entries are unsupported")
        config = local / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        if file_sha256(config) != config_sha256:
            raise ValueError("SAM2 snapshot config differs from frozen identity")
        if not (local / "sam2/__init__.py").is_file():
            raise ValueError("SAM2 source package is missing")
        return local
    except BaseException:
        if local.resolve().parent != destination.resolve() or not local.name.startswith("sam2-"):
            raise ValueError("source cleanup target outside snapshot destination")
        shutil.rmtree(local)
        raise


def activate_sam2_snapshot(snapshot):
    import importlib
    import sys
    from pathlib import Path
    require_fresh_sam2_import(sys.modules)
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        raise ValueError("Hydra configuration already initialized; restart before SAM2 import")
    snapshot = Path(snapshot).resolve()
    sys.path.insert(0,str(snapshot))
    module = importlib.import_module("sam2")
    if Path(module.__file__).resolve() != snapshot / "sam2/__init__.py":
        raise ValueError("SAM2 import did not use the verified source snapshot")


def validate_paired_cache(cache, expected_files):
    names = [row["name"] for row in cache]
    if len(names) != len(set(names)) or set(names) != set(expected_files):
        raise ValueError("paired cache must cover exactly the validation files")
    for row in cache:
        scores = np.asarray(row["scores"],dtype=np.float64)
        if scores.ndim != 1 or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
            raise ValueError("Invalid paired confidence scores")
        if any(arm not in row or len(row[arm]) != len(scores) for arm in CFG.ARMS):
            raise ValueError("All paired arms must retain identical proposal indices")
        for j in range(len(scores)):
            shape = decode_one(row["raw"][j]).shape
            if any(decode_one(row[arm][j]).shape != shape for arm in CFG.ARMS):
                raise ValueError("Paired arm mask shapes differ")


def paired_cache_digest(cache):
    import hashlib
    import json
    payload = [{"name":r["name"],"scores":[float(s) for s in r["scores"]],
                "arms":{arm:[{"size":v["size"],"counts":v["counts"].decode("ascii")}
                              for v in r[arm]] for arm in CFG.ARMS}} for r in cache]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode("utf-8")).hexdigest()


def validate_run_rows(rows):
    import math
    if (not isinstance(rows,list) or len(rows) != len(CFG.ARMS)
            or any(not isinstance(r,dict) for r in rows) or {r.get("arm") for r in rows} != set(CFG.ARMS)):
        raise ValueError("Complete paired four-arm results are required")
    for row in rows:
        if set(row) != {"arm","fold","conf","metrics","fn_dense_val","fn_sparse_val"}:
            raise ValueError("Incomplete or unsupported arm result schema")
        if row.get("fold") != CFG.FOLD or row.get("conf") != CFG.CONF:
            raise ValueError("Result fold/conf differs")
        metrics = row.get("metrics",{})
        if not isinstance(metrics,dict) or set(metrics) != {"pq","sq","rq","tp","fp","fn","denominator"}:
            raise ValueError("Incomplete or unsupported arm metrics schema")
        for key in ("pq","sq","rq"):
            value = metrics.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError("Invalid arm quality metric")
        for key in ("tp","fp","fn"):
            if type(metrics.get(key)) is not int or metrics[key] < 0:
                raise ValueError("Invalid arm count")
        denominator = metrics["denominator"]
        if (type(denominator) not in (int,float) or not math.isfinite(denominator)
                or denominator != metrics["tp"] + 0.5 * metrics["fp"] + 0.5 * metrics["fn"]):
            raise ValueError("Arm denominator disagrees with scorer counts")
        expected_rq = metrics["tp"] / denominator if denominator > 0 else 0.0
        # Only float64 arithmetic reassociation tolerance; not baseline acceptance.
        if (not math.isclose(metrics["rq"],expected_rq,rel_tol=1e-12,abs_tol=1e-12)
                or not math.isclose(metrics["pq"],metrics["sq"]*metrics["rq"],rel_tol=1e-12,abs_tol=1e-12)
                or metrics["tp"] == 0 and not math.isclose(metrics["sq"],0.0,rel_tol=1e-12,abs_tol=1e-12)):
            raise ValueError("Arm quality components disagree with scorer identities")
        for key in ("fn_dense_val","fn_sparse_val"):
            if type(row.get(key)) is not int or row[key]<0:
                raise ValueError("Invalid density FN count")
        if row["fn_dense_val"] + row["fn_sparse_val"] != metrics["fn"]:
            raise ValueError("Arm density counts disagree")


def validate_manifest(manifest, expected):
    import math
    import re
    from datetime import datetime, timedelta
    keys = {"experiment_id", "status", "input_contract", "scope", "count_unit", "matching",
            "paired_cache_sha256", "runs", "sam2_diagnostics", "paired_delta_pq", "runtime", "completed_at_utc"}
    def invalid():
        raise ValueError("Existing manifest is legacy/partial/mismatched; preserve it and choose a new path")
    def finite(value):
        return type(value) in (int,float) and math.isfinite(value)
    if (not isinstance(manifest,dict) or set(manifest) != keys
            or manifest["experiment_id"] != f"v61_sam2_refiner_fold{CFG.FOLD}_diagnostic_v1"
            or manifest["input_contract"] != expected or manifest["status"] != "complete_diagnostic"
            or manifest["scope"] != "exploratory paired diagnostic; no mechanism adoption or scientific acceptance"
            or manifest["count_unit"] != "annotator_image" or manifest["matching"] != "official all IoU > 0.5 hits"
            or not isinstance(manifest["paired_cache_sha256"],str)
            or re.fullmatch("[0-9a-f]{64}",manifest["paired_cache_sha256"]) is None):
        invalid()
    validate_run_rows(manifest["runs"])
    ref = next(row for row in manifest["runs"] if row["arm"] == "refined")
    validate_baseline(ref["metrics"]["pq"],expected["baseline"]["pq"],expected["baseline"]["tolerance"])
    deltas = manifest["paired_delta_pq"]
    if (not isinstance(deltas,dict) or set(deltas) != set(CFG.ARMS)
            or any(not finite(deltas[r["arm"]]) or deltas[r["arm"]] != r["metrics"]["pq"]-ref["metrics"]["pq"]
                   for r in manifest["runs"])):
        invalid()
    diag = manifest["sam2_diagnostics"]
    stats = {"sam2_pred_iou_mean", "iou_sam2_vs_proposal_mean", "iou_sam2_vs_proposal_median",
             "sam2_area_over_proposal_mean"}
    if (not isinstance(diag,dict) or set(diag) != stats | {"n_proposals","sam2_empty_fraction"}
            or type(diag["n_proposals"]) is not int or diag["n_proposals"] < 0
            or not finite(diag["sam2_empty_fraction"]) or not 0 <= diag["sam2_empty_fraction"] <= 1):
        invalid()
    for key in stats:
        value = diag[key]
        if value is None:
            if diag["n_proposals"] != 0 and key != "sam2_pred_iou_mean":
                invalid()
        elif (not finite(value) or diag["n_proposals"] == 0
                or key.startswith("iou_") and not 0 <= value <= 1
                or key == "sam2_area_over_proposal_mean" and value < 0):
            invalid()
    if diag["n_proposals"] == 0 and diag["sam2_empty_fraction"] != 0:
        invalid()
    empty_count = diag["sam2_empty_fraction"] * diag["n_proposals"]
    # Integer-count reconstruction allows only float64 arithmetic roundoff.
    if not math.isclose(empty_count,round(empty_count),rel_tol=1e-12,abs_tol=1e-12):
        invalid()
    # Every empty mask contributes exactly zero overlap and area ratio.
    if (diag["n_proposals"] > 0 and diag["sam2_empty_fraction"] == 1.0
            and any(diag[key] != 0.0 for key in ("iou_sam2_vs_proposal_mean",
                "iou_sam2_vs_proposal_median","sam2_area_over_proposal_mean"))):
        invalid()
    if diag["n_proposals"] > 0:
        n = diag["n_proposals"]
        mean = diag["iou_sam2_vs_proposal_mean"]
        median = diag["iou_sam2_vs_proposal_median"]
        n_empty = round(empty_count)
        # IoU <= SAM2/proposal area; bounds allow 1e-12 float64 arithmetic roundoff.
        # Nonnegative overlaps are at most one, and empty masks add exact zeros.
        if (n <= 2 and mean != median
                or 2*n_empty > n and median != 0.0
                or mean > (n-n_empty)/n + 1e-12
                or mean > diag["sam2_area_over_proposal_mean"] + 1e-12
                or (diag["sam2_area_over_proposal_mean"] > 0) != (n_empty < n)
                or mean == 0.0 and median != 0.0):
            invalid()
        # Sorted overlaps in [0,1] with at least n_empty zeros bound their sum
        # around the middle value(s); use the same arithmetic-only tolerance.
        k = n//2
        if n % 2:
            lower_sum, upper_sum = (k+1)*median, k+max(k+1-n_empty,0)*median
        elif n_empty < k:
            lower_sum, upper_sum = (k+1)*median, (k-n_empty+1)*median+k-1
        elif n_empty == k:
            lower_sum, upper_sum = n*median, 2*median+k-1
        else:
            lower_sum, upper_sum = 0.0, n-n_empty
        if mean < lower_sum/n - 1e-12 or mean > min(upper_sum,n-n_empty)/n + 1e-12:
            invalid()
    runtime = manifest["runtime"]
    runtime_keys = {"service","project","notebook","gpu","duration_seconds","peak_vram_bytes",
                    "measurement_scope","compute_units_used","compute_units_note","versions"}
    if (not isinstance(runtime,dict) or set(runtime) != runtime_keys
            or any(not isinstance(runtime[k],str) or not runtime[k].strip()
                   for k in ("service","project","notebook","gpu","measurement_scope","compute_units_note"))
            or runtime["service"] != "Google Colab"
            or runtime["notebook"] != f"solar_filament_v61_sam2_refiner_fold{CFG.FOLD}_colab.ipynb"
            or runtime["measurement_scope"] != "detector checkpoint snapshot/loading, inference/refinement and paired scoring; earlier SAM2/refiner setup excluded"
            or runtime["versions"] != expected["runtime_versions"]
            or not finite(runtime["duration_seconds"]) or runtime["duration_seconds"] < 0
            or type(runtime["peak_vram_bytes"]) is not int or runtime["peak_vram_bytes"] < 0
            or runtime["compute_units_used"] is not None and (
                not finite(runtime["compute_units_used"]) or runtime["compute_units_used"] < 0)):
        invalid()
    try:
        timestamp = datetime.fromisoformat(manifest["completed_at_utc"])
        if timestamp.utcoffset() != timedelta(0):
            invalid()
    except (TypeError,ValueError):
        invalid()


def validate_existing_manifest(path, expected):
    import json
    from pathlib import Path
    path = Path(path)
    if path.exists():
        validate_manifest(json.loads(path.read_bytes().decode("utf-8")),expected)

"""

def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(fold):
    baseline_refined = BASELINE_PQ_REFINED[fold]

    cells = [
        markdown(
            f"""# v61 exploratory SAM2 box-prompt diagnostic — fold{fold}, no fitting

Compare raw, refined, sam2_free and sam2_shrink from the same detector proposals.
Mask changes can alter FP and FN as well as SQ; low agreement with a proposal
alone does not establish a domain gap. No historical GPU results, automatic
mechanism choices or adoption/rejection claims are imported.

Historical reference PQ {baseline_refined} is context only. Before models load,
provide independently frozen `artifacts/v61_expected_contract_fold{fold}.json`:
schema_version=1, fold, protocol_sha256, detector_sha256, refiner_sha256,
annotations_sha256, folds_sha256, preprocessed_sha256, sam2_commit (40hex),
sam2_config=`configs/sam2.1/sam2.1_hiera_l.yaml`, sam2_config_sha256,
sam2_checkpoint_sha256, runtime_versions, baseline={{pq,tolerance}}.
SHA256 values are 64 lowercase hex. preprocessed_sha256 hashes the UTF-8 sorted
JSON map of validation filenames to file SHA256. Runtime versions must include
python, torch, torchvision, numpy, pandas, opencv, ultralytics, pycocotools,
hydra-core, omegaconf, iopath and Pillow. Explicit baseline pq/tolerance are
reproduction guards only, not scientific acceptance thresholds.

Pre-provision dependencies, SAM2 checkpoint and approved source Git repository at
`artifacts/sam2_source`; no remote source/checkpoint download occurs. A unique
local source snapshot is generated from the approved commit, excluding dirty
working files. Config SHA and imported package paths are checked. Existing SAM2
modules or initialized Hydra require a fresh runtime, not module reload guessing.

One complete four-arm unit is saved atomically only after baseline reproduction.
Legacy/partial/mismatched manifests are refused. Existing results never skip arms;
interrupted evaluation reruns the whole paired unit. GPU and independent
scientific acceptance remain unverified; previous rejected/STOP decisions stand.
"""
        ),
        code(
            r"""# [1/6] Pre-provisioned runtime and independently frozen expected inputs
import hashlib, json, math, shutil, sys, time, warnings
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from tqdm.auto import tqdm

drive.mount("/content/drive")

class CFG:
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v61_diagnostic")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    FOLD = %d
    PROTOCOL_SHA256 = %r
    YOLO_IMGSZ = 1280
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    CONF = 0.26
    PREDICT_CONF_FLOOR = 0.08
    PREDICT_IOU = 0.55
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    DENSE_QUANTILE = %r
    ARMS = %r
    SAM2_CKPT = ARTIFACT_ROOT / "sam2.1_hiera_large.pt"
    SAM2_SOURCE = ARTIFACT_ROOT / "sam2_source"
    MANIFEST = ARTIFACT_ROOT / f"v61_sam2_refiner_fold{FOLD}_diagnostic_v1_manifest.json"
    EXPECTED_CONTRACT = ARTIFACT_ROOT / f"v61_expected_contract_fold{FOLD}.json"

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR):
    path.mkdir(parents=True,exist_ok=True)

def prod_checkpoint(fold):
    return CFG.DRIVE_ROOT / "runs" / f"yolo_fold_{fold}" / "weights" / "best.pt"
"""  # noqa: UP031 - preserve embedded dictionary braces
            % (fold, PROTOCOL_SHA256, DENSE_QUANTILE, ARMS)
            + "\n" + GUARDS
            + "\nINPUT_CONTRACT = read_expected_contract(CFG.EXPECTED_CONTRACT, CFG.FOLD, CFG.PROTOCOL_SHA256)\n"
        ),
        code(
            r"""# [2/6] COCOとfoldを読み、密度スライス用のcutを計算し、前処理画像をローカルSSDへ展開
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"学習annotation JSONが見つかりません: {base}")
    if len(matches) != 1:
        raise ValueError("Exactly one annotation JSON must be selected")
    return matches[0].parents[2]

TRAIN_JSON = next(find_dataset_root(CFG.DATA_CACHE).rglob(
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
# Capture each remote metadata file once: parse and fingerprint the same bytes.
annotation_bytes = TRAIN_JSON.read_bytes()
coco = json.loads(annotation_bytes.decode("utf-8"))

images = {item["id"]: item for item in coco["images"]}
if len(images) != len(coco["images"]):
    raise ValueError("Duplicate annotation image IDs")
annotations_by_image = defaultdict(list)
for ann in coco["annotations"]:
    if ann["image_id"] not in images:
        raise ValueError("Annotation references an unknown image")
    annotations_by_image[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)

from io import BytesIO
fold_bytes = (CFG.ARTIFACT_ROOT / "grouped_folds.csv").read_bytes()
fold_frame = pd.read_csv(BytesIO(fold_bytes))
fold_of_file = validate_fold_rows(fold_frame.to_dict("records"), train_files, CFG.FOLD)
train_files_fold = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]

def noncrowd_count(image_id):
    return sum(1 for a in annotations_by_image[image_id] if not a.get("iscrowd", 0))

# v57系と同じ定義: cutは学習サンプル(file,annotator)のGT数q75
train_sample_counts = [
    noncrowd_count(image_id)
    for name in train_files_fold
    for image_id in file_to_ids[name]
]
DENSE_CUT = float(np.quantile(train_sample_counts, CFG.DENSE_QUANTILE))
val_density = {
    name: float(np.mean([noncrowd_count(i) for i in file_to_ids[name]]))
    for name in valid_files_fold
}

for name in tqdm(valid_files_fold, desc="前処理画像をSSDへ"):
    local = CFG.PREPROC_DIR / name
    shutil.copy2(CFG.PREPROC_CACHE / name, local)  # replace stale local cache

PREPROC_HASHES = {name: file_sha256(CFG.PREPROC_DIR / name) for name in sorted(valid_files_fold)}
SAM2_SOURCE_SNAPSHOT = snapshot_sam2_source(CFG.SAM2_SOURCE, INPUT_CONTRACT["sam2_commit"],
    INPUT_CONTRACT["sam2_config_sha256"], CFG.LOCAL_ROOT / "verified_sources")
RUNTIME_VERSIONS = {"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
                    "pandas":pd.__version__,"opencv":cv2.__version__,
                    **{key:version(key) for key in ("torchvision","ultralytics","pycocotools","hydra-core",
                                                   "omegaconf","iopath","Pillow")}}
ACTUAL_INPUTS = {
    "sam2_commit": INPUT_CONTRACT["sam2_commit"], "sam2_config":INPUT_CONTRACT["sam2_config"],
    "sam2_config_sha256": file_sha256(SAM2_SOURCE_SNAPSHOT / "sam2" / INPUT_CONTRACT["sam2_config"]),
    "sam2_checkpoint_sha256": file_sha256(CFG.SAM2_CKPT), "runtime_versions":RUNTIME_VERSIONS,
    "schema_version": 1, "fold": CFG.FOLD, "protocol_sha256": CFG.PROTOCOL_SHA256,
    "detector_sha256": file_sha256(prod_checkpoint(CFG.FOLD)),
    "refiner_sha256": file_sha256(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt"),
    "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
    "folds_sha256": hashlib.sha256(fold_bytes).hexdigest(),
    "preprocessed_sha256": hashlib.sha256(json.dumps(PREPROC_HASHES, sort_keys=True).encode("utf-8")).hexdigest(),
}
check_actual_inputs(INPUT_CONTRACT, ACTUAL_INPUTS)
validate_existing_manifest(CFG.MANIFEST, INPUT_CONTRACT)
print({"input_contract": "verified before model loading", "protocol_sha256": CFG.PROTOCOL_SHA256})

print({"検証(OOF)": len(valid_files_fold), "dense_cut": DENSE_CUT,
       "dense検証画像数": int(sum(v >= DENSE_CUT for v in val_density.values()))})
"""
        ),
        code(
            "# [3/6] MediSegUNet(solar_filament/models/mediseg_models.py と同一)+ SAM2 predictor のロード\n"
            + MODELS
            + """
activate_sam2_snapshot(SAM2_SOURCE_SNAPSHOT)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Reject mixed cached child modules even when the top-level package is correct.
for name, module in list(sys.modules.items()):
    if name == "sam2" or name.startswith("sam2."):
        if not getattr(module,"__file__",None) or not Path(module.__file__).resolve().is_relative_to(SAM2_SOURCE_SNAPSHOT):
            raise ValueError("SAM2 child import outside verified snapshot")

refiner = MediSegUNet(3, 1, False).cuda().eval()
_verified_refiner = verified_checkpoint_snapshot(
    CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
    INPUT_CONTRACT["refiner_sha256"], CFG.LOCAL_ROOT / "verified_checkpoints")
_ckpt = torch.load(_verified_refiner, map_location="cuda", weights_only=False)
refiner.load_state_dict(_ckpt.get("best_model") or _ckpt["model"], strict=True)
del _ckpt

_verified_sam2 = verified_checkpoint_snapshot(CFG.SAM2_CKPT,
    INPUT_CONTRACT["sam2_checkpoint_sha256"],CFG.LOCAL_ROOT / "verified_checkpoints")
if file_sha256(SAM2_SOURCE_SNAPSHOT / "sam2" / INPUT_CONTRACT["sam2_config"]) != INPUT_CONTRACT["sam2_config_sha256"]:
    raise ValueError("Verified SAM2 config changed before model load")
sam2_model = build_sam2(INPUT_CONTRACT["sam2_config"],str(_verified_sam2),device="cuda",mode="eval")
sam2_predictor = SAM2ImagePredictor(sam2_model)
"""
        ),
        code(
            """# [4/6] 推論・後処理・採点関数(build_v58系と同一経路。armごとのRLEを保持)
from pycocotools import mask as mask_utils
from ultralytics import YOLO

def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    return output, confidences

def mask_square_box(mask, context=1.65, minimum=96):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, width, height
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > width: x0 -= x1 - width; x1 = width
    if y1 > height: y0 -= y1 - height; y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

def mask_tight_box(mask):
    \"\"\"提案maskの外接矩形 xyxy。SAM2 の box prompt に使う。\"\"\"
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

@torch.no_grad()
def refine_proposal(image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal  # shrink(確定構成)

@torch.no_grad()
def sam2_predict_box(image_rgb, proposal):
    \"\"\"proposalの外接矩形をbox promptとしてSAM2で1mask推論。
    戻り値: (mask bool, predicted_iou float or None)。空boxなら空mask。\"\"\"
    box = mask_tight_box(proposal)
    if box is None:
        return np.zeros_like(proposal), None
    masks, iou_preds, _ = sam2_predictor.predict(
        box=box[None, :], multimask_output=False)
    # predict は binarized を返す設計だが、logits が返る版でも >0 で安全に閾値化できる
    values = np.asarray(masks[0])
    if values.shape != proposal.shape or not np.isfinite(values).all():
        raise ValueError("Invalid SAM2 predicted mask")
    mask = values > 0
    score = float(iou_preds[0]) if iou_preds is not None and len(iou_preds) else None
    if score is not None and not math.isfinite(score):
        raise ValueError("Nonfinite SAM2 predicted IoU")
    return mask, score

def encode_one(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))

def decode_one(rle):
    return mask_utils.decode(rle) != 0

def greedy_disjoint(masks, order, min_area=CFG.MIN_AREA):
    if not masks:
        return []
    occupied = np.zeros(masks[0].shape,dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept

def drop_contained_indices(masks, threshold=CFG.DROP_THRESHOLD):
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))
    areas = np.asarray(mask_utils.area(rles), dtype=np.float64)
    self_iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
    dropped = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped:
                continue
            value = self_iou[i, j]
            if value <= 0:
                continue
            intersection = value * (areas[i] + areas[j]) / (1.0 + value)
            smaller = i if areas[i] <= areas[j] else j
            if intersection / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]

def apply_ownership(masks, scores):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices(masks)
    return greedy_disjoint(masks, sorted(subset, key=lambda i: areas[i]))

def to_rle(annotation, height, width):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

class PanopticQuality:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.tp_ious = []
        self.fp_count = 0
        self.fn_count = 0
    def add_matrix(self, matrix, n_gt, n_pred):
        if n_gt == 0:
            self.fp_count += n_pred; return
        if n_pred == 0:
            self.fn_count += n_gt; return
        hits = matrix > self.iou_threshold
        scores = matrix[hits].tolist()
        self.tp_ious.extend(scores)
        self.fp_count += int((hits.sum(axis=0) == 0).sum())
        self.fn_count += int((hits.sum(axis=1) == 0).sum())
    @property
    def denominator(self):
        return len(self.tp_ious) + 0.5 * self.fp_count + 0.5 * self.fn_count
    @property
    def score(self):
        d = self.denominator
        return float(sum(self.tp_ious) / d) if d > 0 else 0.0
    def sq_rq(self):
        tp = len(self.tp_ious)
        sq = float(sum(self.tp_ious) / tp) if tp > 0 else 0.0
        rq = float(tp / (tp + 0.5 * self.fp_count + 0.5 * self.fn_count)) if (tp + self.fp_count + self.fn_count) > 0 else 0.0
        return sq, rq
    def summary(self):
        sq, rq = self.sq_rq()
        return {"pq": self.score, "sq": sq, "rq": rq, "tp": len(self.tp_ious),
                "fp": self.fp_count, "fn": self.fn_count, "denominator": self.denominator}

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in annotations_by_image[image_id] if not ann.get("iscrowd", 0)]

print("後処理・採点関数の定義OK")
"""
        ),
        code(
            """# [5/6] prod検出器で1回推論 → 全armのmaskをRLE保持 → armごとに採点
def collect_cache(best_path):
    \"\"\"floor(conf=floor)でpredictし、conf>=CONF・area>=MIN_AREAの提案に対して
    raw/refined/sam2_free/sam2_shrink のRLEを同一proposal indexで保持。\"\"\"
    verified_detector = verified_checkpoint_snapshot(best_path, INPUT_CONTRACT["detector_sha256"],
        CFG.LOCAL_ROOT / "verified_checkpoints")
    yolo_model = YOLO(str(verified_detector))
    cache = []
    diag = {"pred_iou": [], "iou_vs_proposal": [], "area_ratio": [], "n_empty_sam2": 0, "n_proposals": 0}
    for name in tqdm(valid_files_fold, desc=f"prod fold{CFG.FOLD} 推論+refine+SAM2"):
        image_path = CFG.PREPROC_DIR / name
        image_bytes = image_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != PREPROC_HASHES[name]:
            raise ValueError(f"Preprocessed image differs: {name}")
        image_bgr = cv2.imdecode(np.frombuffer(image_bytes,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.shape[:2] != (2048,2048):
            raise ValueError(f"Invalid production image: {name}")
        result = yolo_model.predict(image_bgr.copy(), imgsz=CFG.YOLO_IMGSZ,
                                     conf=CFG.PREDICT_CONF_FLOOR, iou=CFG.PREDICT_IOU,
                                     agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, confidences = result_masks(result)
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(image_rgb)  # 画像ごとに1回だけ特徴を計算
        entry = {"name": name, "scores": confidences[keep] if keep else np.zeros(0, np.float32)}
        for arm in CFG.ARMS:
            entry[arm] = []
        for i in keep:
            proposal = masks[i]
            sam2_mask, pred_iou = sam2_predict_box(image_rgb, proposal)
            entry["raw"].append(encode_one(proposal))
            entry["refined"].append(encode_one(refine_proposal(image_bgr, proposal)))
            entry["sam2_free"].append(encode_one(sam2_mask))
            entry["sam2_shrink"].append(encode_one(sam2_mask & proposal))
            # --- ドメインギャップ診断 ---
            diag["n_proposals"] += 1
            if pred_iou is not None:
                diag["pred_iou"].append(pred_iou)
            inter = float((sam2_mask & proposal).sum())
            union = float((sam2_mask | proposal).sum())
            diag["iou_vs_proposal"].append(inter / union if union > 0 else 0.0)
            area_prop = float(proposal.sum())
            diag["area_ratio"].append(float(sam2_mask.sum()) / max(area_prop, 1.0))
            if sam2_mask.sum() == 0:
                diag["n_empty_sam2"] += 1
        cache.append(entry)
    del yolo_model
    torch.cuda.empty_cache()
    summary_diag = {
        "n_proposals": diag["n_proposals"],
        "sam2_pred_iou_mean": float(np.mean(diag["pred_iou"])) if diag["pred_iou"] else None,
        "iou_sam2_vs_proposal_mean": float(np.mean(diag["iou_vs_proposal"])) if diag["iou_vs_proposal"] else None,
        "iou_sam2_vs_proposal_median": float(np.median(diag["iou_vs_proposal"])) if diag["iou_vs_proposal"] else None,
        "sam2_area_over_proposal_mean": float(np.mean(diag["area_ratio"])) if diag["area_ratio"] else None,
        "sam2_empty_fraction": diag["n_empty_sam2"] / max(diag["n_proposals"], 1),
    }
    return cache, summary_diag

def score_arm(cache, arm):
    \"\"\"保持済みRLEをownership→PQ採点 + 密度別FN(arm間で同一proposal index)。\"\"\"
    if arm not in CFG.ARMS:
        raise ValueError("Unknown diagnostic arm")
    meter = PanopticQuality()
    fn_dense = fn_sparse = 0
    for entry in cache:
        name = entry["name"]
        masks = [decode_one(rle) for rle in entry[arm]]
        preds = apply_ownership(masks, entry["scores"])
        pred_rles = []
        if preds:
            encoded = mask_utils.encode(np.asfortranarray(np.stack(preds, axis=-1).astype(np.uint8)))
            pred_rles = encoded if isinstance(encoded, list) else [encoded]
        file_fn = 0
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, pred_rles)
            meter.add_matrix(matrix, len(truth), len(pred_rles))
            if matrix.size:
                file_fn += int((matrix.max(axis=1) <= 0.5).sum())
            else:
                file_fn += len(truth)
        if val_density[name] >= DENSE_CUT:
            fn_dense += file_fn
        else:
            fn_sparse += file_fn
    return {"metrics": meter.summary(),
            "fn_dense_val": fn_dense, "fn_sparse_val": fn_sparse}

torch.cuda.reset_peak_memory_stats()
started = time.time()
cache, sam2_diag = collect_cache(prod_checkpoint(CFG.FOLD))
print(f"推論+4arm生成 完了({round((time.time() - started) / 60, 1)}min)")
print(json.dumps({"sam2_diagnostics": sam2_diag}, ensure_ascii=False, indent=2))
"""
        ),
        code(
            """# [6/6] A complete paired diagnostic unit; no old arm metrics are reused
validate_existing_manifest(CFG.MANIFEST,INPUT_CONTRACT)
validate_paired_cache(cache,valid_files_fold)
refined = score_arm(cache,"refined")
validate_baseline(refined["metrics"]["pq"],INPUT_CONTRACT["baseline"]["pq"],
                  INPUT_CONTRACT["baseline"]["tolerance"])
rows = []
for arm in CFG.ARMS:
    result = refined if arm == "refined" else score_arm(cache,arm)
    rows.append({"arm":arm,"fold":CFG.FOLD,"conf":CFG.CONF,**result})
validate_run_rows(rows)
manifest = {
    "experiment_id":f"v61_sam2_refiner_fold{CFG.FOLD}_diagnostic_v1",
    "status":"complete_diagnostic", "input_contract":INPUT_CONTRACT,
    "scope":"exploratory paired diagnostic; no mechanism adoption or scientific acceptance",
    "count_unit":"annotator_image", "matching":"official all IoU > 0.5 hits",
    "paired_cache_sha256":paired_cache_digest(cache),"runs":rows,"sam2_diagnostics":sam2_diag,
    "paired_delta_pq":{r["arm"]:r["metrics"]["pq"]-refined["metrics"]["pq"] for r in rows},
    "runtime":{"service":"Google Colab","project":str(CFG.DRIVE_ROOT),
               "notebook":f"solar_filament_v61_sam2_refiner_fold{CFG.FOLD}_colab.ipynb",
               "gpu":torch.cuda.get_device_name(0),"duration_seconds":time.time()-started,
               "peak_vram_bytes":int(torch.cuda.max_memory_allocated()),
               "measurement_scope":"detector checkpoint snapshot/loading, inference/refinement and paired scoring; earlier SAM2/refiner setup excluded",
               "compute_units_used":None,"compute_units_note":"Record from Colab UI; runtime API unavailable",
               "versions":RUNTIME_VERSIONS},
    "completed_at_utc":datetime.now(timezone.utc).isoformat(),
}
validate_existing_manifest(CFG.MANIFEST,INPUT_CONTRACT)
validate_manifest(manifest,INPUT_CONTRACT)
atomic_json(CFG.MANIFEST,manifest)
print(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
print("Exploratory differences only. No adoption/rejection; independent confirmation required.")
"""
        ),
    ]
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["0", "1", "2"], help="評価対象fold")
    args = parser.parse_args()
    fold = int(args.fold)

    cells = build_cells(fold)
    for index, cell in enumerate(cells):
        cell["id"] = f"v61-sam2-f{fold}-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output = ROOT / f"solar_filament_v61_sam2_refiner_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
