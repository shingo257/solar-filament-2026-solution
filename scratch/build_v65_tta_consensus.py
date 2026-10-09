"""Recover PR244's exploratory three-view TTA component-selector pilot.

Generation performs no inference, fitting, downloads, or external execution.
Runtime collects fresh id(1280), horizontal-flip(1280), and big(1536) proposals;
only identity-view proposals are refined/scored, with the original 15 features.
Support means other-view IoU >=0.3. Train-A fits the CPU selector; train-B selects
one of the original 54 support/probability operating points, then that point is
frozen for historically exposed heldout evaluation. There is no heldout oracle
grid or automatic adoption/rejection claim. A/B are filename partitions of
detector training data, not independent OOF or guaranteed group-disjoint data.
Union-annotator labels are an exploratory target; official strict-IoU>0.5 PQ
instead scores predictions separately per annotator and pools TP/FP/FN.

Preinstall all v64 runtime packages (including scikit-learn). Independently freeze
artifacts/v65_expected_contract_foldN.json with the v64 schema: schema_version=1,
fold, protocol_sha256, detector_sha256, refiner_sha256, annotations_sha256,
folds_sha256, preprocessed_sha256, config_sha256, selector_split_sha256,
runtime_versions, baseline={pq,tolerance}. No canonical identities/tolerance are
manufactured. preprocessed_sha256 covers ALL consumed filenames; split digest is
UTF-8 json.dumps({train_a:[...],train_b:[...],heldout:[...]},sort_keys=True).
Annotation discovery selects the unique exact JSON path below DATA_CACHE;
it never ascends into neighboring datasets or repeats a broader search.
config digest covers v64 configuration keys plus VIEWS,TTA_MATCH_IOU,CONF_GRID,
PKEEP_GRID,SUPPORT_GRID,SPLIT_SEED using sorted JSON, preserving numeric types.
Protocol binds this builder, reviewed v64/v62 builders and embedded model source.

No external serialized rows, partial results, or grid scores are reused. Restart
all cells with a fresh unique output after interruption. Completed manifests are
published through v64's reserved atomic local-file publication. Download the
completed /content/solar_v65_selector manifest before ending Colab. Actual Colab
execution, disk capacity, GPU costs, and independent acceptance are unverified.
Degenerate train-A uses confidence fallback with an explicit reason; importance
is unavailable for fallback or empty/one-class train-B, never fabricated.
An other-view proposal with zero overlap is not a correspondence: both matching
IoU and confidence are zero, correcting the source pilot's unrelated confidence.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v65_base", Path(__file__).with_name("build_v64_selector_pilot.py"))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
PROTOCOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    + Path(BASE.__file__).read_bytes().replace(b"\r\n", b"\n")
    + Path(BASE.BASE.__file__).read_bytes().replace(b"\r\n", b"\n")
    + BASE.BASE.MODELS.replace("\r\n", "\n").encode()).hexdigest()
EXTRA_CONFIG_KEYS = ("VIEWS", "TTA_MATCH_IOU", "CONF_GRID", "PKEEP_GRID", "SUPPORT_GRID", "SPLIT_SEED")

TTA_HELPERS = r'''FEATURE_KEYS = ["conf", "probmass", "ref_raw_iou", "raw_area", "ref_area",
    "area_ratio", "aspect", "solidity", "darkness", "limb", "tta_hflip_iou",
    "tta_hflip_conf", "tta_big_iou", "tta_big_conf", "tta_support"]

def feature_matrix(rows):
    import numpy as np
    x = np.asarray([[p["feat"][k] for k in FEATURE_KEYS] for r in rows for p in r["props"]], dtype=float)
    y = np.asarray([p["label_hit"] for r in rows for p in r["props"]], dtype=bool)
    return x.reshape(-1, len(FEATURE_KEYS)), y

class ConfidenceFallback:
    def predict_proba(self, x):
        import numpy as np
        p = np.asarray(x[:, 0], dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Fallback confidence outside [0,1]")
        return np.stack([1-p, p], axis=1)

def fit_tta_selector(x, y):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_KEYS) or y.ndim != 1 or len(x) != len(y)
            or y.dtype != np.bool_ or not np.isfinite(x).all()):
        raise ValueError("Finite 15-feature matrix and boolean labels required")
    classes = np.unique(y)
    if len(classes) < 2:
        reason = "empty_train_a" if not len(y) else ("all_positive_train_a" if y[0] else "all_negative_train_a")
        return ConfidenceFallback(), {"kind": "confidence_fallback", "reason": reason}
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=.08, max_depth=6,
                                          min_samples_leaf=20, random_state=3407)
    model.fit(x, y)
    return model, {"kind": "hist_gradient_boosting", "reason": None}

def selector_importance(model, x, y):
    import numpy as np
    from sklearn.inspection import permutation_importance
    if isinstance(model, ConfidenceFallback):
        return {"values": None, "reason": "confidence_fallback_has_no_fitted_importance"}
    if not len(y):
        return {"values": None, "reason": "empty_train_b"}
    if len(np.unique(y)) < 2:
        return {"values": None, "reason": "one_class_train_b_roc_auc_undefined"}
    importance = permutation_importance(model, x, y, n_repeats=3, random_state=3407, scoring="roc_auc")
    values = importance.importances_mean
    if len(values) != len(FEATURE_KEYS) or not np.isfinite(values).all():
        raise ValueError("Invalid fitted feature importance")
    return {"values": dict(zip(FEATURE_KEYS, [float(v) for v in values])), "reason": None}

def tta_features(base_mask, other_rles, other_confs):
    import numpy as np
    confs = np.asarray(other_confs)
    if (confs.ndim != 1 or len(other_rles) != len(confs) or not np.isfinite(confs).all()
            or ((confs < 0) | (confs > 1)).any()):
        raise ValueError("TTA proposal confidence coverage/range differs")
    if not other_rles:
        return 0., 0.
    matrix = iou_matrix([encode_one(base_mask)], other_rles)
    index = int(matrix[0].argmax())
    if matrix[0, index] <= 0:
        return 0., 0.
    return float(matrix[0, index]), float(confs[index])
'''
GUARDS = BASE.BASE.GUARDS + "\n" + BASE.SELECTOR_HELPERS + "\n" + TTA_HELPERS
FEATURES = BASE.FEATURES + '\nFEATURE_KEYS += ["tta_hflip_iou", "tta_hflip_conf", "tta_big_iou", "tta_big_conf", "tta_support"]\n'

COLLECT = r'''# Fresh collection only; no serialized row/grid resume.
verified_detector = verified_checkpoint_snapshot(prod_checkpoint(CFG.FOLD),
    INPUT_CONTRACT["detector_sha256"], CFG.LOCAL_ROOT / "verified_checkpoints")
yolo_model = YOLO(str(verified_detector))

def predict_view(model, image, imgsz, hflip):
    # Independent array per view isolates consumed pixels from detector mutation.
    src = cv2.flip(image, 1) if hflip else image.copy()
    result = model.predict(src, imgsz=imgsz, conf=CFG.PREDICT_CONF_FLOOR, iou=CFG.PREDICT_IOU,
        agnostic_nms=True, retina_masks=True, verbose=False)[0]
    masks, confs = result_masks(result)
    confs = np.asarray(confs, dtype=np.float32)
    if (len(masks) != len(confs) or confs.ndim != 1 or not np.isfinite(confs).all()
            or ((confs < 0) | (confs > 1)).any()
            or any(m.shape != image.shape[:2] for m in masks)):
        raise ValueError("Detector view proposal schema differs")
    if hflip:
        masks = [np.fliplr(mask).copy() for mask in masks]
    return masks, confs

def collect(names, label):
    rows = []
    for name in tqdm(names, desc=label):
        captured = (CFG.PREPROC_DIR / name).read_bytes()
        image = decode_verified_image(captured, PREPROC_HASHES[name])
        if any((images[i]["height"], images[i]["width"]) != image.shape[:2] for i in file_to_ids[name]):
            raise ValueError("Annotation/image dimensions differ")
        view_masks, view_rles, view_confs = {}, {}, {}
        for vname, imgsz, hflip in CFG.VIEWS:
            masks, confs = predict_view(yolo_model, image, imgsz, hflip)
            kept = [(m, c) for m, c in zip(masks, confs) if m.sum() >= CFG.MIN_AREA]
            view_masks[vname] = [m for m, _ in kept]
            view_rles[vname] = [encode_one(m) for m, _ in kept]
            view_confs[vname] = np.asarray([c for _, c in kept], dtype=np.float32)
        props = []
        for mask, conf in zip(view_masks["id"], view_confs["id"]):
            refined, probmass = refine_proposal_stats(image, mask)
            feat = proposal_features(image, mask, conf, refined, probmass)
            for vname in ("hflip", "big"):
                best_iou, best_conf = tta_features(mask, view_rles[vname], view_confs[vname])
                feat[f"tta_{vname}_iou"], feat[f"tta_{vname}_conf"] = best_iou, best_conf
            feat["tta_support"] = float(sum(feat[f"tta_{v}_iou"] >= CFG.TTA_MATCH_IOU for v in ("hflip", "big")))
            props.append({"conf": float(conf), "feat": feat, "refined_rle": encode_one(refined)})
        truth = [r for i in file_to_ids[name] for r in gt_rles[i]]
        matrix = iou_matrix(truth, [p["refined_rle"] for p in props])
        for j, prop in enumerate(props):
            prop["label_hit"] = bool(len(truth) and matrix[:, j].max() > .5)
        rows.append({"name": name, "props": props})
    validate_rows(rows, names)
    return rows

trainA_rows = collect(train_A_files, "train-A 3-view discovery")
trainB_rows = collect(train_B_files, "train-B 3-view discovery")
val_rows = collect(valid_files_fold, "historically exposed heldout 3-view")
del yolo_model
torch.cuda.empty_cache()
'''

EVALUATE = r'''def score_rows(rows, conf_floor, keep_fn):
    meter = PanopticQuality()
    for row in rows:
        kept = [p for p in row["props"] if p["conf"] >= conf_floor and keep_fn(p)]
        masks = apply_ownership([decode_one(p["refined_rle"]) for p in kept], np.ones(len(kept), np.float32))
        prediction = [encode_one(mask) for mask in masks]
        for image_id in file_to_ids[row["name"]]:
            truth = gt_rles[image_id]
            meter.add_matrix(iou_matrix(truth, prediction), len(truth), len(prediction))
    return validate_metrics(meter.summary())

baseline = score_rows(val_rows, CFG.CONF, lambda p: True)
validate_baseline(baseline["pq"], INPUT_CONTRACT["baseline"]["pq"], INPUT_CONTRACT["baseline"]["tolerance"])
x_a, y_a = feature_matrix(trainA_rows)
selector, selector_status = fit_tta_selector(x_a, y_a)
for row in trainB_rows + val_rows:
    x, _ = feature_matrix([row])
    probabilities = selector.predict_proba(x)[:, 1] if len(x) else np.zeros(0)
    if (len(probabilities) != len(row["props"]) or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Selector probability coverage/range differs")
    for p, probability in zip(row["props"], probabilities):
        p["p"] = float(probability)

def keep_point(mode, param):
    return (lambda p: p["feat"]["tta_support"] >= param) if mode == "support" else (lambda p: p["p"] >= param)

grid = []
for conf_floor in tqdm(CFG.CONF_GRID, desc="Train-B operating grid"):
    for mode, params in (("support", CFG.SUPPORT_GRID), ("pk", CFG.PKEEP_GRID)):
        for param in params:
            metrics = score_rows(trainB_rows, conf_floor, keep_point(mode, param))
            grid.append({"mode": mode, "conf_floor": conf_floor, "param": param, "metrics": metrics})
chosen = max(grid, key=lambda point: point["metrics"]["pq"])
selected = score_rows(val_rows, chosen["conf_floor"], keep_point(chosen["mode"], chosen["param"]))
x_b, y_b = feature_matrix(trainB_rows)
importance = selector_importance(selector, x_b, y_b)
manifest = {"schema_version": 1, "kind": "exploratory_v65_tta_complete",
    "scientific_acceptance": "unverified; historically exposed heldout; independent confirmation required",
    "input_contract": INPUT_CONTRACT, "partitions": PARTITIONS, "views": CFG.VIEWS,
    "feature_keys": FEATURE_KEYS, "union_annotator_overlap_target": True,
    "selector_status": selector_status, "feature_importance": importance,
    "train_b_selection_status": ("no_proposals_deterministic_first_tie" if not len(y_b)
        else "tie_first_grid_point" if len({point["metrics"]["pq"] for point in grid}) == 1
        else "exploratory_train_b_selection"),
    "proposal_counts": {"train_a": len(y_a), "train_b": len(y_b), "heldout": sum(len(r["props"]) for r in val_rows)},
    "train_b_grid": grid, "frozen_operation_point": chosen, "baseline": baseline, "selected": selected,
    "delta_pq": selected["pq"] - baseline["pq"],
    "runtime": {"service": "Google Colab", "gpu": torch.cuda.get_device_name(0),
        "started_at_utc": started_utc, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "refiner loading, three-view collection, CPU selector fitting/importance/scoring; excludes input staging",
        "gpu_peak_vram_bytes": int(torch.cuda.max_memory_allocated()), "elapsed_seconds": time.time()-started,
        "compute_units_used": None, "compute_units_note": "Record separately from Colab UI; runtime API unavailable"}}
publish_manifest(CFG.MANIFEST, manifest)
print(json.dumps({"baseline": baseline, "selected": selected, "manifest": str(CFG.MANIFEST)}))
'''


def build_cells(fold):
    cells = BASE.build_cells(fold)[:5]
    cells[0] = BASE.BASE.markdown(__doc__)
    setup = cells[1]["source"].replace("v64_", "v65_").replace("v64-", "v65-")
    setup = setup.replace(repr(BASE.PROTOCOL_SHA256), repr(PROTOCOL_SHA256))
    setup += '''
CFG.VIEWS = [("id", 1280, False), ("hflip", 1280, True), ("big", 1536, False)]
CFG.TTA_MATCH_IOU = .3
CFG.CONF_GRID = (.05, .08, .12, .20, .26, .35)
CFG.PKEEP_GRID = (0., .3, .5, .7, .85, .93)
CFG.SUPPORT_GRID = (0, 1, 2)
CFG.SPLIT_SEED = 3407
''' + TTA_HELPERS
    cells[1]["source"] = setup
    inputs = cells[2]["source"].replace('"MIN_AREA")', '"MIN_AREA", ' + ', '.join(repr(k) for k in EXTRA_CONFIG_KEYS) + ')')
    discovery_start = inputs.index("def find_dataset_root(base):")
    discovery_end = inputs.index("# Capture each remote metadata file once:")
    inputs = inputs[:discovery_start] + '''annotation_paths = list(CFG.DATA_CACHE.rglob(
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
if not annotation_paths:
    raise FileNotFoundError("Annotation JSON missing below DATA_CACHE")
if len(annotation_paths) != 1:
    raise ValueError("Exactly one annotation JSON must be selected below DATA_CACHE")
TRAIN_JSON = annotation_paths[0]
''' + inputs[discovery_end:]
    inputs = inputs.replace('require_local_publication(CFG.MANIFEST)\n', '''require_local_publication(CFG.MANIFEST)
# Shared v64 partitions and selector/importance randomness use this fixed seed.
if type(CFG.SPLIT_SEED) is not int or CFG.SPLIT_SEED != 3407:
    raise ValueError("This protocol requires SPLIT_SEED=3407")
''', 1)
    cells[2]["source"] = inputs
    cells[4]["source"] = cells[4]["source"].replace(BASE.FEATURES, FEATURES)
    cells.extend([BASE.BASE.code(COLLECT), BASE.BASE.code(EVALUATE), BASE.BASE.markdown(
        "Download the completed local manifest before ending the runtime. No stale row/grid reuse or automatic adoption decision.")])
    for index, cell in enumerate(cells):
        cell["id"] = f"v65-tta-f{fold}-{index:02d}"
    return cells


def notebook(fold):
    notebook = BASE.notebook(fold)
    notebook["cells"] = build_cells(fold)
    return notebook


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    fold = parser.parse_args().fold
    output = ROOT / f"solar_filament_v65_tta_consensus_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook(fold), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
