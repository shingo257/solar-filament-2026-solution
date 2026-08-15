import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = Path(__file__).with_name("build_v30_prize_colab.py")
OUTPUT = ROOT / "solar_filament_v30_prize_colab_instance_ensemble.ipynb"

# Keep the large notebook source in one readable base builder, then apply the
# small reliability/performance amendments here.
runpy.run_path(str(BASE_BUILDER), run_name="__main__")
notebook = json.loads(OUTPUT.read_text(encoding="utf-8"))


def replace(cell_index, old, new):
    source = notebook["cells"][cell_index]["source"]
    if old not in source:
        raise RuntimeError(f"Cell {cell_index}: expected fragment not found")
    notebook["cells"][cell_index]["source"] = source.replace(old, new, 1)


replace(
    2,
    "from sklearn.model_selection import StratifiedKFold",
    "from sklearn.model_selection import KFold, StratifiedKFold",
)

replace(
    3,
    '''if not CFG.LOCAL_DATA.exists():
    print("Copying cached data from Drive to the Colab SSD...")
    shutil.copytree(CFG.DATA_CACHE, CFG.LOCAL_DATA)

DATA_ROOT = find_dataset_root(CFG.LOCAL_DATA)''',
    '''try:
    DATA_ROOT = find_dataset_root(CFG.LOCAL_DATA)
except FileNotFoundError:
    if CFG.LOCAL_DATA.exists():
        shutil.rmtree(CFG.LOCAL_DATA)
    print("Copying cached data from Drive to the Colab SSD...")
    shutil.copytree(CFG.DATA_CACHE, CFG.LOCAL_DATA)
    DATA_ROOT = find_dataset_root(CFG.LOCAL_DATA)''',
)

replace(
    4,
    '''fold_of_file = {}
splitter = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
for fold, (_, valid_idx) in enumerate(splitter.split(train_files, strata)):
    for idx in valid_idx:
        fold_of_file[train_files[idx]] = fold''',
    '''fold_of_file = {}
strata_array = np.asarray(strata)
_, stratum_counts = np.unique(strata_array, return_counts=True)
if len(stratum_counts) and stratum_counts.min() >= CFG.N_FOLDS:
    splitter = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    splits = splitter.split(train_files, strata_array)
else:
    print("Rare count stratum detected; falling back to shuffled KFold.")
    splitter = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    splits = splitter.split(train_files)
for fold, (_, valid_idx) in enumerate(splits):
    for idx in valid_idx:
        fold_of_file[train_files[idx]] = fold''',
)

replace(
    10,
    '''def gt_variants(file_name):
    variants = []
    for image_id in file_to_ids[file_name]:
        variants.append([ann_mask(image_id, ann).astype(bool) for ann in annotations[image_id]])
    return variants''',
    '''CALIBRATION_SIZE = 256
CALIBRATION_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]

def compact_mask(mask):
    return cv2.resize(
        mask.astype(np.uint8), (CALIBRATION_SIZE, CALIBRATION_SIZE),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

def gt_variants(file_name):
    variants = []
    for image_id in file_to_ids[file_name]:
        variants.append([compact_mask(ann_mask(image_id, ann)) for ann in annotations[image_id]])
    return variants''',
)

replace(
    10,
    '''    candidates = []
    for confidence, proposal in zip(confidences, masks):
        box, probability = refine_candidate(image, proposal, valid_refiner)
        candidates.append((float(confidence), proposal, box, probability))
    calibration_cache[file_name] = candidates

def materialize_candidates(candidates, conf, threshold, min_area):
    predictions = []
    for confidence, proposal, (x0, y0, x1, y1), probability in candidates:
        if confidence < conf:
            continue
        mask = np.zeros_like(proposal, dtype=bool)
        mask[y0:y1, x0:x1] = probability >= threshold
        if mask.sum() >= min_area:
            predictions.append(mask)
    return predictions

grid = []
for conf in [0.12, 0.16, 0.20, 0.24, 0.28]:
    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for min_area in [20, 35, 60, 100]:
            file_scores = []
            for file_name, candidates in calibration_cache.items():
                predictions = materialize_candidates(candidates, conf, threshold, min_area)
                labeler_scores = [instance_dice_score(gt, predictions) for gt in gt_variants(file_name)]
                file_scores.append(float(np.mean(labeler_scores)))
            grid.append({"conf": conf, "threshold": threshold, "min_area": min_area, "instance_dice": float(np.mean(file_scores))})

calibration = pd.DataFrame(grid).sort_values("instance_dice", ascending=False).reset_index(drop=True)''',
    '''    candidates = []
    for confidence, proposal in zip(confidences, masks):
        (x0, y0, x1, y1), probability = refine_candidate(image, proposal, valid_refiner)
        threshold_masks = {}
        full_areas = {}
        for threshold in CALIBRATION_THRESHOLDS:
            full = np.zeros_like(proposal, dtype=bool)
            full[y0:y1, x0:x1] = probability >= threshold
            threshold_masks[threshold] = compact_mask(full)
            full_areas[threshold] = int(full.sum())
        candidates.append({
            "confidence": float(confidence),
            "masks": threshold_masks,
            "areas": full_areas,
        })
    calibration_cache[file_name] = candidates

gt_cache = {file_name: gt_variants(file_name) for file_name in calibration_cache}

def materialize_candidates(candidates, conf, threshold, min_area):
    return [
        candidate["masks"][threshold]
        for candidate in candidates
        if candidate["confidence"] >= conf and candidate["areas"][threshold] >= min_area
    ]

def evaluate_params(conf, threshold, min_area):
    file_scores = []
    for file_name, candidates in calibration_cache.items():
        predictions = materialize_candidates(candidates, conf, threshold, min_area)
        labeler_scores = [instance_dice_score(gt, predictions) for gt in gt_cache[file_name]]
        file_scores.append(float(np.mean(labeler_scores)))
    return float(np.mean(file_scores))

# Sequential search avoids repeating expensive mask matching for all 100
# combinations while retaining the parameters that materially affect count.
grid = []
for conf in [0.12, 0.16, 0.20, 0.24, 0.28]:
    for min_area in [20, 35, 60, 100]:
        grid.append({
            "conf": conf, "threshold": 0.50, "min_area": min_area,
            "instance_dice": evaluate_params(conf, 0.50, min_area),
        })
coarse = max(grid, key=lambda row: row["instance_dice"])
for threshold in CALIBRATION_THRESHOLDS:
    grid.append({
        "conf": coarse["conf"], "threshold": threshold,
        "min_area": coarse["min_area"],
        "instance_dice": evaluate_params(coarse["conf"], threshold, coarse["min_area"]),
    })
threshold_best = max(grid, key=lambda row: row["instance_dice"])
for conf in sorted(set([max(0.08, threshold_best["conf"] - 0.02), threshold_best["conf"], threshold_best["conf"] + 0.02])):
    for min_area in sorted(set([max(10, int(threshold_best["min_area"] * 0.7)), int(threshold_best["min_area"]), int(threshold_best["min_area"] * 1.4)])):
        grid.append({
            "conf": conf, "threshold": threshold_best["threshold"], "min_area": min_area,
            "instance_dice": evaluate_params(conf, threshold_best["threshold"], min_area),
        })

calibration = pd.DataFrame(grid).drop_duplicates(
    ["conf", "threshold", "min_area"]
).sort_values("instance_dice", ascending=False).reset_index(drop=True)''',
)

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
