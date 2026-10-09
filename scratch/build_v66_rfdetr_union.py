"""Recover PR244's exploratory fold0 RF-DETR union-of-cached-proposals pilot.

No inference, fitting, downloads, or scientific acceptance occurs at generation.
Runtime applies the existing refiner to pre-provisioned cached masks. The five
RF floors are a historically exposed discovery grid, not independent validation
or an adoption decision. Official PQ scores each annotator separately and pools
all strict IoU>0.5 hits. Rescue counts all baseline-blind GT (IoU<0.1), including
images with no eligible RF proposals; rescue thresholds are strict >0.3/>0.5.
RF rescue candidates satisfy the union's refined-mask minimum area before ownership.

Independently freeze artifacts/v66_expected_contract_fold0.json before execution.
It contains schema_version=1, fold=0, protocol_sha256, config_sha256,
runtime_versions, annotations_sha256, folds_sha256, preprocessed_sha256,
refiner_sha256, yolo_manifest_sha256, rfdetr_manifest_sha256, yolo_cache_sha256,
rfdetr_cache_sha256, baseline={pq,tolerance}. Hash dictionaries use UTF-8
json.dumps(mapping, sort_keys=True). No canonical fingerprints or tolerances
are invented. Each cache directory needs a separately supplied producer.json:
schema_version=1, fold=0, producer_protocol_sha256, checkpoint_sha256,
inference_floor, training_folds=[1,2], prediction_unit="filename",
preprocessing_sha256, frame=[2048,2048], files={filename:npz_sha256}.
These are supplier provenance claims; their truth/OOF independence and the real
Colab runtime remain unverified. Cache membership must exactly cover fold0.
The original YOLO and RF producers use different preprocessing (q97/q96).
Their fingerprints are separate from the refiner's consumed image fingerprint;
coordinate registration/common-observation compatibility needs external acceptance.

Preinstall numpy,pandas,scipy,torch,opencv-python,pycocotools. Linux local-only
publication is checked before staging and again at completion. No resume reuse.
Download the completed local manifest before ending the runtime. Local capacity,
copy time, GPU execution, and independent official acceptance are unverified.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_builder("build_v62_error_anatomy.py", "v66_base")
PUBLICATION = load_builder("build_v64_selector_pilot.py", "v66_publication")
PROTOCOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    + Path(BASE.__file__).read_bytes().replace(b"\r\n", b"\n")
    + Path(PUBLICATION.__file__).read_bytes().replace(b"\r\n", b"\n")
    + BASE.MODELS.replace("\r\n", "\n").encode()).hexdigest()
RFDETR_GRID = (.05, .10, .20, .30, .40)
CONFIG_KEYS = ("CROP_SIZE", "CROP_CONTEXT", "YOLO_CONF", "RFDETR_GRID",
               "REFINER_THRESHOLD", "DROP_THRESHOLD", "MIN_AREA")

GUARDS = BASE.GUARDS + "\n" + PUBLICATION.SELECTOR_HELPERS.split("def selector_partitions")[0] + r'''
def digest_mapping(mapping):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()

def validate_names(names):
    from pathlib import PurePosixPath
    if (not names or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n or "\\" in n or ":" in n
                or PurePosixPath(n).name != n or n in (".", "..") for n in names)
            or len({PurePosixPath(n).stem for n in names}) != len(names)):
        raise ValueError("Invalid, duplicate or colliding cache filenames")

def validate_cache_coverage(directory, names):
    from pathlib import Path
    validate_names(names)
    directory = Path(directory)
    expected = {Path(name).stem + ".npz" for name in names}
    actual = {path.name for path in directory.iterdir() if path.suffix == ".npz"}
    if actual != expected:
        raise ValueError(f"Cache coverage differs: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    paths = {name: directory / (Path(name).stem + ".npz") for name in names}
    if any(not p.is_file() or p.is_symlink() for p in paths.values()):
        raise ValueError("Cache entries must be regular files")
    return paths

def load_npz_masks(data, expected_shape):
    from io import BytesIO
    import numpy as np
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != {"packed", "shape", "confidences"}:
            raise ValueError("Cache keys differ")
        packed, shape, confs = (archive[k] for k in ("packed", "shape", "confidences"))
    if (shape.shape != (2,) or shape.dtype.kind not in "iu" or (shape <= 0).any()
            or tuple(shape.tolist()) != tuple(expected_shape)):
        raise ValueError("Cache image shape differs")
    h, w = (int(v) for v in shape)
    if (packed.dtype != np.uint8 or packed.ndim != 3 or confs.ndim != 1
            or confs.dtype.kind not in "fiu" or len(confs) != len(packed)
            or not np.isfinite(confs).all() or ((confs < 0) | (confs > 1)).any()):
        raise ValueError("Cache mask/confidence schema differs")
    # Both producers use (0,0,0) for a legitimate no-detection cache.
    if len(packed) == 0 and packed.shape == (0, 0, 0):
        return [], confs.astype(np.float32)
    if packed.shape[1:] != (h, (w + 7) // 8):
        raise ValueError("Cache packed shape differs")
    bits = np.unpackbits(packed, axis=2)
    if bits[:, :, w:].any():
        raise ValueError("Cache padding bits are not zero")
    return [mask != 0 for mask in bits[:, :, :w]], confs.astype(np.float32)

def validate_cache_manifest(manifest, names, cache_hashes, required_floor):
    import math
    import re
    validate_names(names)
    keys = {"schema_version", "fold", "producer_protocol_sha256", "checkpoint_sha256",
            "inference_floor", "training_folds", "prediction_unit", "preprocessing_sha256", "frame", "files"}
    if not isinstance(manifest, dict) or set(manifest) != keys:
        raise ValueError("Incomplete cache producer manifest")
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or type(manifest["fold"]) is not int or manifest["fold"] != 0
            or manifest["training_folds"] != [1, 2]
            or any(type(f) is not int for f in manifest["training_folds"])
            or manifest["frame"] != [2048, 2048]
            or any(type(v) is not int for v in manifest["frame"])
            or manifest["prediction_unit"] != "filename"):
        raise ValueError("Cache producer split/frame claims differ")
    floor = manifest["inference_floor"]
    if type(floor) not in (int, float) or not math.isfinite(floor) or not 0 <= floor <= required_floor:
        raise ValueError("Cache producer inference floor differs")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(names) or files != cache_hashes:
        raise ValueError("Cache producer file identities differ")
    hashes = [manifest[k] for k in ("producer_protocol_sha256", "checkpoint_sha256", "preprocessing_sha256")]
    hashes += list(files.values())
    if any(not isinstance(v, str) or re.fullmatch("[0-9a-f]{64}", v) is None for v in hashes):
        raise ValueError("Invalid cache SHA256 identity")
    return manifest

def require_union_contract(path, actual):
    import json
    import re
    from pathlib import Path
    expected = json.loads(Path(path).read_bytes())
    if not isinstance(expected, dict) or set(expected) != set(actual) | {"baseline"}:
        raise ValueError("Incomplete independently frozen union contract")
    baseline = expected["baseline"]
    if not isinstance(baseline, dict) or set(baseline) != {"pq", "tolerance"}:
        raise ValueError("Explicit frozen baseline required")
    validate_baseline(baseline["pq"], baseline["pq"], baseline["tolerance"])
    if (type(expected["schema_version"]) is not int or expected["schema_version"] != 1
            or type(expected["fold"]) is not int or expected["fold"] != 0):
        raise ValueError("Invalid union schema/fold")
    for key in actual:
        if key.endswith("sha256") and (not isinstance(expected[key], str)
                or re.fullmatch("[0-9a-f]{64}", expected[key]) is None):
            raise ValueError("Invalid union SHA256 identity")
    if {k: v for k, v in expected.items() if k != "baseline"} != actual:
        raise ValueError("Frozen union inputs differ; do not load refiner")
    return expected

def rescue_counts(base_best, rf_matrices):
    import numpy as np
    if len(base_best) != len(rf_matrices):
        raise ValueError("Rescue annotator coverage differs")
    result = {"n_blind": 0, "rescued_03": 0, "rescued_05": 0}
    for best, matrix in zip(base_best, rf_matrices):
        best, matrix = np.asarray(best), np.asarray(matrix)
        if (best.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != len(best)
                or not np.isfinite(best).all() or not np.isfinite(matrix).all()
                or ((best < 0) | (best > 1)).any() or ((matrix < 0) | (matrix > 1)).any()):
            raise ValueError("Invalid rescue IoU arrays")
        blind = best < .1
        result["n_blind"] += int(blind.sum())
        rescued = matrix.max(axis=1) if matrix.shape[1] else np.zeros(len(best))
        result["rescued_03"] += int((blind & (rescued > .3)).sum())
        result["rescued_05"] += int((blind & (rescued > .5)).sum())
    return result
'''

SETUP = '''# Pre-provision dependencies/artifacts; no installation or inference here.
import hashlib, json, math, shutil, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from tqdm.auto import tqdm
drive.mount("/content/drive")

class CFG:
    V30 = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = V30 / "competition_data"
    ARTIFACT_ROOT = V30 / "artifacts"
    PREPROC_CACHE = V30 / "preprocessed_v30_q96"
    YOLO_OOF = Path("/content/drive/MyDrive/solar_filament_v34/prediction_cache/oof/yolo_fold_0")
    RFDETR_OOF = Path("/content/drive/MyDrive/solar_filament_v35/v45_rfdetr_oof_cache_fold0")
    LOCAL_ROOT = Path("/content/solar_v66_union")
    FOLD = 0
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    YOLO_CONF = .26
    RFDETR_GRID = (.05, .10, .20, .30, .40)
    REFINER_THRESHOLD = .45
    DROP_THRESHOLD = .90
    MIN_AREA = 20
    MANIFEST = LOCAL_ROOT / "v66_rfdetr_union_fold0_manifest.json"
    EXPECTED_CONTRACT = ARTIFACT_ROOT / "v66_expected_contract_fold0.json"
'''

INPUTS = '''require_local_publication(CFG.MANIFEST)
RUNTIME_VERSIONS = {"python": sys.version.split()[0], **{p: version(p) for p in (
    "numpy", "pandas", "scipy", "torch", "opencv-python", "pycocotools")}}
if CFG.MANIFEST.exists():
    raise ValueError("Manifest already exists; preserve it and choose a new path")
annotation_paths = list(CFG.DATA_CACHE.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
if len(annotation_paths) != 1:
    raise ValueError("Exactly one annotation JSON required")
annotation_bytes = annotation_paths[0].read_bytes()
coco = json.loads(annotation_bytes)
images = {item["id"]: item for item in coco["images"]}
if len(images) != len(coco["images"]):
    raise ValueError("Duplicate annotation image IDs")
annotations_by_image = defaultdict(list)
for ann in coco["annotations"]:
    if ann["image_id"] not in images:
        raise ValueError("Unknown annotation image ID")
    annotations_by_image[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
fold_bytes = (CFG.ARTIFACT_ROOT / "grouped_folds.csv").read_bytes()
fold_of_file = validate_fold_rows(pd.read_csv(BytesIO(fold_bytes)).to_dict("records"), sorted(file_to_ids), 0)
valid_files = sorted(name for name, fold in fold_of_file.items() if fold == 0)
validate_names(valid_files)
if any((images[i]["height"], images[i]["width"]) != (2048, 2048)
       for name in valid_files for i in file_to_ids[name]):
    raise ValueError("Annotation frame differs")
PREPROC_HASHES = {name: file_sha256(CFG.PREPROC_CACHE / name) for name in valid_files}
preprocessing_sha256 = digest_mapping(PREPROC_HASHES)
CACHE_PATHS, CACHE_HASHES, CACHE_MANIFESTS, manifest_hashes = {}, {}, {}, {}
for kind, directory, floor in (("yolo", CFG.YOLO_OOF, CFG.YOLO_CONF),
                                ("rfdetr", CFG.RFDETR_OOF, min(CFG.RFDETR_GRID))):
    paths = validate_cache_coverage(directory, valid_files)
    CACHE_PATHS[kind] = paths
    hashes = {}
    for name in tqdm(valid_files, desc=f"Validate {kind} cache"):
        captured = paths[name].read_bytes()
        hashes[name] = hashlib.sha256(captured).hexdigest()
        load_npz_masks(captured, (2048, 2048))
    producer_bytes = (directory / "producer.json").read_bytes()
    CACHE_MANIFESTS[kind] = validate_cache_manifest(json.loads(producer_bytes), valid_files,
        hashes, floor)
    CACHE_HASHES[kind] = hashes
    manifest_hashes[kind] = hashlib.sha256(producer_bytes).hexdigest()
ACTUAL_INPUTS = {"schema_version": 1, "fold": 0, "protocol_sha256": CFG.PROTOCOL_SHA256,
    "config_sha256": digest_mapping({key: getattr(CFG, key) for key in CONFIG_KEYS}),
    "runtime_versions": RUNTIME_VERSIONS, "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
    "folds_sha256": hashlib.sha256(fold_bytes).hexdigest(), "preprocessed_sha256": preprocessing_sha256,
    "refiner_sha256": file_sha256(CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt"),
    "yolo_manifest_sha256": manifest_hashes["yolo"], "rfdetr_manifest_sha256": manifest_hashes["rfdetr"],
    "yolo_cache_sha256": digest_mapping(CACHE_HASHES["yolo"]),
    "rfdetr_cache_sha256": digest_mapping(CACHE_HASHES["rfdetr"])}
INPUT_CONTRACT = require_union_contract(CFG.EXPECTED_CONTRACT, ACTUAL_INPUTS)
CFG.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
if not torch.cuda.is_available():
    raise ValueError("Pre-provision a CUDA runtime for refiner execution")
started, started_utc = time.time(), datetime.now(timezone.utc).isoformat()
torch.cuda.reset_peak_memory_stats()
print({"input_contract": "verified before refiner loading", "n_files": len(valid_files)})
'''

EVALUATE = '''def score_preds(prediction, name, meter):
    best = []
    for image_id in file_to_ids[name]:
        truth = gt_rles[image_id]
        matrix = iou_matrix(truth, prediction)
        meter.add_matrix(matrix, len(truth), len(prediction))
        best.append(matrix.max(axis=1) if matrix.shape[1] else np.zeros(len(truth)))
    return best

meters = {"baseline": PanopticQuality(), **{f"union@{f}": PanopticQuality() for f in CFG.RFDETR_GRID}}
rescue = {"n_blind": 0, "rescued_03": 0, "rescued_05": 0}
for name in tqdm(valid_files, desc="Paired cached union evaluation"):
    captured = (CFG.PREPROC_CACHE / name).read_bytes()
    if hashlib.sha256(captured).hexdigest() != PREPROC_HASHES[name]:
        raise ValueError("Preprocessed bytes changed")
    image = cv2.imdecode(np.frombuffer(captured, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (2048, 2048, 3):
        raise ValueError("Preprocessed image frame differs")
    refined, confidences = {}, {}
    for kind, floor in (("yolo", CFG.YOLO_CONF), ("rfdetr", min(CFG.RFDETR_GRID))):
        cache_bytes = CACHE_PATHS[kind][name].read_bytes()
        if hashlib.sha256(cache_bytes).hexdigest() != CACHE_HASHES[kind][name]:
            raise ValueError("Cache bytes changed after contract verification")
        masks, scores = load_npz_masks(cache_bytes, (2048, 2048))
        selected = [(m, c) for m, c in zip(masks, scores) if c >= floor and m.sum() >= CFG.MIN_AREA]
        refined[kind] = [refine_proposal_stats(image, m)[0] for m, _ in selected]
        confidences[kind] = np.asarray([c for _, c in selected])
    base_masks = [m for m in refined["yolo"] if m.sum() >= CFG.MIN_AREA]
    base_prediction = [encode_one(m) for m in apply_ownership(base_masks, np.ones(len(base_masks)))]
    base_best = score_preds(base_prediction, name, meters["baseline"])
    for floor in CFG.RFDETR_GRID:
        masks = refined["yolo"] + [m for m, c in zip(refined["rfdetr"], confidences["rfdetr"]) if c >= floor]
        masks = [m for m in masks if m.sum() >= CFG.MIN_AREA]
        prediction = [encode_one(m) for m in apply_ownership(masks, np.ones(len(masks)))]
        score_preds(prediction, name, meters[f"union@{floor}"])
    rf_prediction = [encode_one(m) for m, c in zip(refined["rfdetr"], confidences["rfdetr"])
                     if c >= .10 and m.sum() >= CFG.MIN_AREA]
    matrices = [iou_matrix(gt_rles[i], rf_prediction) for i in file_to_ids[name]]
    counts = rescue_counts(base_best, matrices)
    for key in rescue:
        rescue[key] += counts[key]
out = {key: meter.summary() for key, meter in meters.items()}
baseline = out["baseline"]
validate_baseline(baseline["pq"], INPUT_CONTRACT["baseline"]["pq"], INPUT_CONTRACT["baseline"]["tolerance"])
arms = [{"rfdetr_floor": floor, **out[f"union@{floor}"],
         "delta_pq": out[f"union@{floor}"]["pq"] - baseline["pq"]} for floor in CFG.RFDETR_GRID]
manifest = {"schema_version": 1, "kind": "exploratory_v66_union_complete",
    "scientific_acceptance": "unverified supplier claims; historically exposed discovery grid; independent confirmation required",
    "input_contract": INPUT_CONTRACT, "supplier_claims": CACHE_MANIFESTS,
    "filenames": valid_files, "baseline": baseline, "arms": arms, "rescue": rescue,
    "runtime": {"service": "Google Colab", "gpu": torch.cuda.get_device_name(0),
        "started_at_utc": started_utc, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "refiner loading and paired cache evaluation; excludes input validation/staging",
        "elapsed_seconds": time.time() - started, "gpu_peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "compute_units_used": None, "compute_units_note": "Record separately from Colab UI; runtime API unavailable"}}
publish_manifest(CFG.MANIFEST, manifest)
print(json.dumps({"baseline": baseline, "arms": arms, "manifest": str(CFG.MANIFEST)}))
'''


def build_cells():
    setup = SETUP + f"\nCFG.PROTOCOL_SHA256 = {PROTOCOL_SHA256!r}\nCONFIG_KEYS = {CONFIG_KEYS!r}\n" + GUARDS
    model = BASE.build_cells(0)[3]["source"]
    functions = BASE.build_cells(0)[4]["source"].replace("from ultralytics import YOLO\n", "")
    functions = functions.replace("mask_square_box(proposal)", "mask_square_box(proposal, context=CFG.CROP_CONTEXT)")
    cells = [BASE.markdown(__doc__), BASE.code(setup), BASE.code(INPUTS), BASE.code(model),
             BASE.code(functions), BASE.code(EVALUATE), BASE.markdown(
                 "Download the completed local manifest before ending the runtime. No automatic adoption decision or resume reuse. "
                 "If interrupted, preserve any existing result/reservation and choose a new MANIFEST path; "
                 "do not automatically remove a potentially active reservation.")]
    for index, cell in enumerate(cells):
        cell["id"] = f"v66-union-{index:02d}"
    return cells


def notebook():
    return {"cells": build_cells(), "metadata": {"accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    output = ROOT / "solar_filament_v66_rfdetr_union_fold0_colab.ipynb"
    output.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
