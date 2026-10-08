"""Build an exploratory learned component-selector pilot recovered from PR244.

CPU HistGBM fitting is performed in the generated notebook after GPU proposal
collection. A/B are filename partitions of detector training data, not independent
base-model OOF or guaranteed group-disjoint samples. Held-out results were already
exposed historically; independent confirmation is required for scientific use.
Selector training labels use overlap with any annotator as an exploratory target.
PQ instead scores the same predictions separately per annotator and pools counts;
the union training label is not a claim of an official matched true positive.

Pre-provision dependencies and artifacts. Independently freeze
artifacts/v64_expected_contract_foldN.json before execution. Its schema extends
the v62 input contract with runtime_versions, config_sha256 and selector_split_sha256. The latter
hashes UTF-8 sorted JSON {train_a:[...],train_b:[...],heldout:[...]} using the
deterministic split below. preprocessed_sha256 covers ALL consumed filenames.
protocol_sha256 binds this builder, the v62 builder and embedded model source.
Full keys: schema_version=1, fold, protocol_sha256, detector_sha256,
refiner_sha256, annotations_sha256, folds_sha256, preprocessed_sha256,
config_sha256, selector_split_sha256, runtime_versions, baseline.
config_sha256 hashes UTF-8 json.dumps(mapping, sort_keys=True) for CFG keys
YOLO_IMGSZ, CROP_SIZE, CROP_CONTEXT, CONF, PREDICT_CONF_FLOOR, PREDICT_IOU,
REFINER_THRESHOLD, DROP_THRESHOLD, MIN_AREA, preserving their int/float types.
baseline={pq,tolerance} must be separately frozen for the matching protocol.
No canonical identity or tolerance is manufactured by this builder.

No cached proposals, classifier, grid, or partial manifest are reused. Only a
complete baseline-gated exploratory manifest is published atomically. Publication
reserves a sibling .publish-lock directory exclusively and checks the result path
inside that reservation. Concurrent publishers and cell reruns cannot replace a
completed result. A reservation left by a killed runtime fails closed: preserve
it and select a new output path rather than automatically reclaiming it.
Linux publication requires an unambiguous local filesystem; stacked mount points and
FUSE/network mount targets are refused without relying on mount listing order.
The default manifest lives under /content/solar_v64_selector. Download the
completed file before ending the Colab runtime; Drive durability is not assumed.
The actual Colab filesystem/runtime remains unverified. A preflight runs before
input staging/model loading, with another check at publication after evaluation.
Preinstall the listed runtime packages before execution; dependency versions are
checked before staging. All training files are staged; local capacity and copy
time have not been measured in Colab.
Generation
alone performs no inference, fitting, downloads or external execution.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = Path(__file__).with_name("build_v62_error_anatomy.py")
_SPEC = importlib.util.spec_from_file_location("v64_base", BASE_PATH)
BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BASE)
PROTOCOL_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    + BASE_PATH.read_bytes().replace(b"\r\n", b"\n")
    + BASE.MODELS.replace("\r\n", "\n").encode()
).hexdigest()

SELECTOR_HELPERS = r'''def require_local_publication(path, mountinfo=None):
    import re
    import sys
    from pathlib import Path
    path = Path(path).resolve()
    if mountinfo is None:
        if sys.platform == "win32":
            if path.drive.startswith("\\\\"):
                raise ValueError("Manifest publication requires a local filesystem")
            return  # CPU-test support only; mapped Windows drives are not classified.
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    matches = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields = before.split()
        if not separator or len(fields) < 5 or not after.split():
            continue
        point = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])).resolve()
        if point == path or point in path.parents:
            matches.append((len(point.parts), after.split()[0]))
    deepest = max((depth for depth, _ in matches), default=-1)
    candidates = [kind for depth, kind in matches if depth == deepest]
    if len(candidates) != 1 or candidates[0] not in {
            "overlay", "ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "ramfs", "rootfs", "zfs"}:
        raise ValueError("Manifest publication requires a verified local filesystem")


def publish_manifest(path, payload):
    from pathlib import Path
    path = Path(path)
    require_local_publication(path)
    reservation = path.with_name(path.name + ".publish-lock")
    try:
        reservation.mkdir()
    except FileExistsError as error:
        raise ValueError("Manifest publication is reserved; choose a new output path") from error
    try:
        if path.exists():
            raise ValueError("Manifest already exists; preserve it and choose a new output path")
        atomic_json(path, payload)
    finally:
        reservation.rmdir()


def selector_partitions(mapping, fold):
    import random
    training = sorted(n for n, f in mapping.items() if f != fold)
    heldout = sorted(n for n, f in mapping.items() if f == fold)
    rng = random.Random(3407)
    training = sorted(training, key=lambda name: rng.random())
    middle = len(training) // 2
    a, b = training[:middle], training[middle:]
    if not a or not b or not heldout:
        raise ValueError("Selector partitions must all be nonempty")
    return a, b, heldout


def fit_selector(x, y):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    if (x.ndim != 2 or x.shape[1] != 10 or len(x) != len(y)
            or not np.isfinite(x).all() or set(np.unique(y)) != {False, True}):
        raise ValueError("Finite training features and two classes are required")
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=.08,
                                          max_depth=6, min_samples_leaf=20, random_state=3407)
    model.fit(x, y)
    return model


def require_selector_contract(path, actual):
    import json
    import re
    from pathlib import Path
    expected = json.loads(Path(path).read_bytes())
    if not isinstance(expected, dict) or set(expected) != set(actual) | {"baseline"}:
        raise ValueError("Incomplete independently frozen selector contract")
    baseline = expected["baseline"]
    if not isinstance(baseline, dict) or set(baseline) != {"pq", "tolerance"}:
        raise ValueError("Explicit frozen baseline is required")
    # Schema/range validation only; EVALUATE compares the measured baseline.
    validate_baseline(baseline["pq"], baseline["pq"], baseline["tolerance"])
    if type(expected["schema_version"]) is not int or type(expected["fold"]) is not int:
        raise ValueError("Invalid schema/fold")
    for key in actual:
        if key.endswith("sha256") and (not isinstance(expected[key], str)
                or re.fullmatch("[0-9a-f]{64}", expected[key]) is None):
            raise ValueError("Invalid SHA256 identity")
    if {k: v for k, v in expected.items() if k != "baseline"} != actual:
        raise ValueError("Frozen selector inputs differ; do not load models")
    return expected


def validate_rows(rows, names):
    import numpy as np
    if len(rows) != len(names) or {r["name"] for r in rows} != set(names):
        raise ValueError("Missing or duplicate proposal rows")
    for row in rows:
        for p in row["props"]:
            if (set(p["feat"]) != set(FEATURE_KEYS)
                    or not np.isfinite([p["feat"][k] for k in FEATURE_KEYS]).all()
                    or not 0 <= p["conf"] <= 1 or type(p["label_hit"]) is not bool):
                raise ValueError("Invalid proposal features/label")


def validate_metrics(metrics):
    import math
    for key in ("tp", "fp", "fn"):
        if type(metrics[key]) is not int or metrics[key] < 0:
            raise ValueError("Invalid official counts")
    tp, fp, fn = (metrics[k] for k in ("tp", "fp", "fn"))
    denominator = tp + .5 * fp + .5 * fn
    if metrics["denominator"] != denominator:
        raise ValueError("Invalid official denominator")
    if any(not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1
           for k in ("pq", "sq", "rq")):
        raise ValueError("Invalid metric range")
    rq = tp / denominator if denominator else 0.0
    if (not math.isclose(metrics["rq"], rq, abs_tol=1e-12)
            or not math.isclose(metrics["pq"], metrics["sq"] * rq, abs_tol=1e-12)
            or (tp == 0 and metrics["sq"] != 0)):
        raise ValueError("Inconsistent official metrics")
    return metrics


def decode_verified_image(captured, expected_hash):
    import hashlib
    import cv2
    import numpy as np
    if hashlib.sha256(captured).hexdigest() != expected_hash:
        raise ValueError("Image bytes changed after contract verification")
    image = cv2.imdecode(np.frombuffer(captured, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (2048, 2048):
        raise ValueError("Production protocol requires decodable 2048x2048 images")
    return image
'''

FEATURES = r'''FEATURE_KEYS = ["conf", "probmass", "ref_raw_iou", "raw_area", "ref_area",
                "area_ratio", "aspect", "solidity", "darkness", "limb"]

def proposal_features(image, proposal, conf, refined, probmass):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ys, xs = np.where(proposal)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    contours, _ = cv2.findContours(proposal.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in contours) or 1.0
    ring = (cv2.dilate(proposal.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0) & ~proposal
    inside = float(gray[proposal].mean())
    darkness = (float(gray[ring].mean()) if ring.any() else inside) - inside
    disk = gray > max(20.0, float(gray.max()) * .15)
    if disk.sum() > 1000:
        dys, dxs = np.where(disk)
        cx, cy, radius = dxs.mean(), dys.mean(), np.sqrt(disk.sum() / np.pi)
    else:
        cy, cx = np.array(proposal.shape) / 2
        radius = min(proposal.shape) * 900 / 2048
    raw_area, ref_area = float(proposal.sum()), float(refined.sum())
    return dict(zip(FEATURE_KEYS, [float(conf), float(probmass), mask_iou(refined, proposal),
        raw_area, ref_area, ref_area / raw_area, float(max(w, h) / min(w, h)),
        raw_area / hull_area, darkness,
        float(np.hypot(xs.mean() - cx, ys.mean() - cy) / max(radius, 1.0))]))
'''

COLLECT = r'''# Collect fresh in-memory RLE proposals; no legacy/partial cache reuse.
verified_detector = verified_checkpoint_snapshot(prod_checkpoint(CFG.FOLD),
    INPUT_CONTRACT["detector_sha256"], CFG.LOCAL_ROOT / "verified_checkpoints")
yolo_model = YOLO(str(verified_detector))

def collect(names, label):
    rows = []
    for name in tqdm(names, desc=label):
        captured = (CFG.PREPROC_DIR / name).read_bytes()
        image = decode_verified_image(captured, PREPROC_HASHES[name])
        if any((images[i]["height"], images[i]["width"]) != image.shape[:2]
               for i in file_to_ids[name]):
            raise ValueError("Annotation/image dimensions differ")
        result = yolo_model.predict(image.copy(), imgsz=CFG.YOLO_IMGSZ,
            conf=CFG.PREDICT_CONF_FLOOR, iou=CFG.PREDICT_IOU,
            agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, confidences = result_masks(result)
        if len(masks) != len(confidences):
            raise ValueError("Detector mask/confidence count differs")
        props = []
        for mask, conf in zip(masks, confidences):
            if mask.sum() < CFG.MIN_AREA:
                continue
            refined, probmass = refine_proposal_stats(image, mask)
            props.append({"conf": float(conf), "refined_rle": encode_one(refined),
                          "feat": proposal_features(image, mask, conf, refined, probmass)})
        # Union-annotator overlap is a training target, not a matched-TP claim.
        truth = [r for image_id in file_to_ids[name] for r in gt_rles[image_id]]
        matrix = iou_matrix(truth, [p["refined_rle"] for p in props])
        for j, p in enumerate(props):
            p["label_hit"] = bool(len(truth) and matrix[:, j].max() > .5)
        rows.append({"name": name, "props": props})
    validate_rows(rows, names)
    return rows

trainA_rows = collect(train_A_files, "train-A discovery proposals")
trainB_rows = collect(train_B_files, "train-B discovery proposals")
val_rows = collect(valid_files_fold, "historically exposed heldout proposals")
del yolo_model
torch.cuda.empty_cache()
'''

EVALUATE = r'''def feature_matrix(rows):
    return np.asarray([[p["feat"][k] for k in FEATURE_KEYS]
        for row in rows for p in row["props"]], dtype=float).reshape(-1, len(FEATURE_KEYS))

def score_rows(rows, conf_floor, probabilities=None, p_keep=None):
    if probabilities is not None and (len(probabilities) != sum(len(r["props"]) for r in rows)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Classifier probability coverage/range differs")
    metric = PanopticQuality()
    offset = 0
    for row in rows:
        keep = []
        for p in row["props"]:
            probability = probabilities[offset] if probabilities is not None else 1.0
            offset += 1
            if p["conf"] >= conf_floor and (p_keep is None or probability >= p_keep):
                keep.append(p)
        final = apply_ownership([decode_one(p["refined_rle"]) for p in keep],
                                [p["conf"] for p in keep])
        prediction = [encode_one(mask) for mask in final]
        for image_id in file_to_ids[row["name"]]:
            truth = gt_rles[image_id]
            metric.add_matrix(iou_matrix(truth, prediction), len(truth), len(prediction))
    return validate_metrics(metric.summary())

# Reproduce separately frozen baseline before selector diagnostics or fitting.
baseline = score_rows(val_rows, CFG.CONF)
validate_baseline(baseline["pq"], INPUT_CONTRACT["baseline"]["pq"],
                  INPUT_CONTRACT["baseline"]["tolerance"])
x_a = feature_matrix(trainA_rows)
y_a = np.asarray([p["label_hit"] for row in trainA_rows for p in row["props"]])
selector = fit_selector(x_a, y_a)
x_b = feature_matrix(trainB_rows)
if not len(x_b):
    raise ValueError("Train-B has no proposals; operation point cannot be selected")
prob_b = selector.predict_proba(x_b)[:, 1]
grid = []
for conf_floor in tqdm([.05, .08, .12, .20, .26, .35], desc="Train-B operation grid"):
    for p_keep in [0, .3, .5, .7, .85, .93]:
        metrics = score_rows(trainB_rows, conf_floor, prob_b, p_keep)
        grid.append({"conf_floor": conf_floor, "p_keep": p_keep, "metrics": metrics})
# Deterministic ties prefer the earliest predeclared point. No heldout grid.
chosen = max(grid, key=lambda item: item["metrics"]["pq"])
x_val = feature_matrix(val_rows)
prob_val = selector.predict_proba(x_val)[:, 1] if len(x_val) else np.zeros(0)
selected = score_rows(val_rows, chosen["conf_floor"], prob_val, chosen["p_keep"])
manifest = {
    "schema_version": 1, "kind": "exploratory_v64_selector_complete",
    "scientific_acceptance": "unverified; historically exposed heldout; independent confirmation required",
    "input_contract": INPUT_CONTRACT, "partitions": PARTITIONS,
    "feature_keys": FEATURE_KEYS, "union_annotator_overlap_target": True,
    "proposal_counts": {"train_a": len(x_a), "train_b": len(x_b), "heldout": len(x_val)},
    "train_b_grid": grid, "frozen_operation_point": chosen,
    "baseline": baseline, "selected": selected, "delta_pq": selected["pq"] - baseline["pq"],
    "runtime": {"service": "Google Colab", "gpu": torch.cuda.get_device_name(0),
        "started_at_utc": started_utc, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "refiner loading, proposal collection, CPU selector fitting and scoring; excludes input staging",
        "gpu_peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.time() - started, "compute_units_used": None,
        "compute_units_note": "Record separately from Colab UI; runtime API unavailable"},
}
publish_manifest(CFG.MANIFEST, manifest)
print(json.dumps({"baseline": baseline, "selected": selected, "manifest": str(CFG.MANIFEST)}))
'''


def build_cells(fold):
    if type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("Invalid fold")
    cells = BASE.build_cells(fold)[:5]
    cells[0] = BASE.markdown(__doc__ + "\n\nRuntime identity keys: python, numpy, pandas, scipy, "
                             "torch, opencv-python, ultralytics, pycocotools, scikit-learn. "
                             "Restart runtime and rerun all cells after interruption.")
    setup = cells[1]["source"].replace("!pip -q install ultralytics pycocotools\n", "")
    setup = setup.replace("v62_", "v64_").replace("v62-", "v64-")
    setup = setup.replace("solar_v64_anatomy", "solar_v64_selector")
    setup = setup.replace("v64_error_anatomy_fold", "v64_selector_fold")
    setup = setup.replace('MANIFEST = DRIVE_ROOT / "artifacts" /', 'MANIFEST = LOCAL_ROOT /')
    setup = setup.replace("# --- 解剖パラメータ ---", "# Exploratory selector output and frozen input contract")
    unused = {"BASELINE_PQ_REFINED", "DENSE_QUANTILE", "IOU_HIT", "FN_WEAK_LO", "FN_NEAR_LO",
              "FP_ORPHAN_MAX", "MERGE_MIN_IOU", "OVERMERGE_IOU", "ORACLES"}
    setup = "\n".join(line for line in setup.splitlines()
                      if line.strip().partition("=")[0].strip() not in unused) + "\n"
    setup = setup.replace(repr(BASE.PROTOCOL_SHA256), repr(PROTOCOL_SHA256))
    setup = setup.replace("PREDICT_CONF_FLOOR = 0.08", "PREDICT_CONF_FLOOR = 0.05")
    setup = setup.replace("# --- OOF評価は確定構成(build_v58/v60/v61系と同一) ---",
                          "# Shared production postprocessing; exploratory selector discovery")
    setup += "\n" + SELECTOR_HELPERS
    cells[1]["source"] = setup
    inputs = cells[2]["source"].replace("for name in tqdm(valid_files_fold,", "for name in tqdm(train_files,")
    inputs = inputs.replace("for name in sorted(valid_files_fold)", "for name in sorted(train_files)")
    inputs = inputs[:inputs.index("def noncrowd_count")] + inputs[inputs.index("for name in tqdm"):]
    inputs = inputs[:inputs.index('print({"検証(OOF)')]
    inputs = "# Stage all selector inputs and verify the frozen contract.\n" + inputs.split("\n", 1)[1]
    inputs = inputs.replace('INPUT_CONTRACT = require_expected_contract', '''train_A_files, train_B_files, valid_files_fold = selector_partitions(fold_of_file, CFG.FOLD)
PARTITIONS = {"train_a": train_A_files, "train_b": train_B_files, "heldout": valid_files_fold}
ACTUAL_INPUTS["selector_split_sha256"] = hashlib.sha256(json.dumps(PARTITIONS, sort_keys=True).encode()).hexdigest()
ACTUAL_INPUTS["config_sha256"] = hashlib.sha256(json.dumps({
    key: getattr(CFG, key) for key in ("YOLO_IMGSZ", "CROP_SIZE", "CROP_CONTEXT", "CONF",
        "PREDICT_CONF_FLOOR", "PREDICT_IOU", "REFINER_THRESHOLD", "DROP_THRESHOLD", "MIN_AREA")
}, sort_keys=True).encode()).hexdigest()
ACTUAL_INPUTS["runtime_versions"] = RUNTIME_VERSIONS
if CFG.MANIFEST.exists():
    raise ValueError("Complete manifest already exists; preserve it and choose a new output path")
INPUT_CONTRACT = require_selector_contract''')
    cells[2]["source"] = '''require_local_publication(CFG.MANIFEST)
# Dependencies must be preinstalled; refuse missing packages before any staging.
RUNTIME_VERSIONS = {"python": sys.version.split()[0], **{
    package: version(package) for package in ("numpy", "pandas", "scipy", "torch",
        "opencv-python", "ultralytics", "pycocotools", "scikit-learn")}}
''' + inputs
    cells[2]["source"] = cells[2]["source"].replace("検証(OOF)", "historically exposed heldout")
    cells[3]["source"] = ("started = time.time()\nstarted_utc = datetime.now(timezone.utc).isoformat()\n"
                          "torch.cuda.reset_peak_memory_stats()\n" + cells[3]["source"])
    cells[4]["source"] += "\n" + FEATURES
    cells[4]["source"] = cells[4]["source"].replace(
        "mask_square_box(proposal)", "mask_square_box(proposal, context=CFG.CROP_CONTEXT)")
    cells.extend([BASE.code(COLLECT), BASE.code(EVALUATE)])
    cells.append(BASE.markdown("Download the completed local manifest from the Colab Files pane "
        "before ending the runtime. Its path is printed above. Do not redirect MANIFEST to Drive/FUSE: "
        "publication rejects nonlocal mounts. A copied file is a backup; its durability is not validated here."))
    for index, cell in enumerate(cells):
        cell["id"] = f"v64-selector-f{fold}-{index:02d}"
    return cells


def notebook(fold):
    return {"cells": build_cells(fold), "metadata": {
        "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    fold = parser.parse_args().fold
    output = ROOT / f"solar_filament_v64_selector_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook(fold), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
