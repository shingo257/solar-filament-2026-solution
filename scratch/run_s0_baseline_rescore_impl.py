"""Implementation helpers for `run_s0_baseline_rescore` — extracted verbatim from `run_s0_baseline_rescore.py`.

Holds the helper functions and module constants so `run_s0_baseline_rescore.py` stays a readable
orchestrator. Loaded by the wrapper via sibling import.
"""


from __future__ import annotations


import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from skimage.segmentation import watershed


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from solar_filament.models.mediseg_models import MediSegUNet  # noqa: E402
from solar_filament.metrics.science_metrics import (  # noqa: E402
    audit_spine_coordinates,
    boundary_f1,
    cluster_bootstrap_mean,
    fragmentation_counts,
    match_instances,
    matched_geometry_errors,
    paired_centerline_metrics,
    spine_centerline_metrics,
    spine_coverage,
    topology_counts,
)
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS,
    MIN_AREA,
    OOF_CACHE,
    PREPROC,
)


V34_SEMANTIC = Path("G:/マイドライブ/solar_filament_v34/prediction_cache/oof")


SEED8407 = {
    0: ROOT / "kaggle_downloads/v42/fold0_kernel_output/v42_multiseed/oof_cache/yolo_seed8407_fold_0",
    1: ROOT / "kaggle_downloads/v42/fold1_kernel_output/v42_multiseed/oof_cache/yolo_seed8407_fold_1",
    2: ROOT / "kaggle_downloads/v42/fold2_kernel_output/v42_multiseed/oof_cache/yolo_seed8407_fold_2",
}


MODELS = ("raw", "v35", "v34", "v42")


CACHE_VERSION = "s0-science-prediction-v2"


SCORING_VERSION = "s0-science-scoring-v3"


SCIENCE_ENDPOINTS = (
    "spine_coverage",
    "spine_coverage_strict",
    "spine_coverage_matched",
    "spine_coverage_matched_strict",
    "spine_centerline_precision",
    "spine_centerline_recall",
    "spine_centerline_f1",
    "spine_centerline_distance",
    "boundary_f1",
    "cldice",
    "centerline_distance",
    "area_relative_error",
    "perimeter_relative_error",
    "major_axis_relative_error",
    "centroid_error_pixels",
    "component_abs_error",
    "endpoint_abs_error",
    "branchpoint_abs_error",
    "filament_recall",
    "filament_precision",
    "filament_f1",
    "count_abs_error",
    "split_count",
    "merge_count",
)


LOWER_IS_BETTER = frozenset({
    "centerline_distance", "area_relative_error", "perimeter_relative_error",
    "major_axis_relative_error", "centroid_error_pixels", "component_abs_error",
    "endpoint_abs_error", "branchpoint_abs_error", "count_abs_error", "split_count",
    "merge_count", "spine_centerline_distance", "fp", "fn",
})


def annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rle = mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get("counts"), list):
        rle = mask_util.frPyObjects(segmentation, height, width)
    else:
        rle = dict(segmentation)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
    return mask_util.decode(rle).astype(bool)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def prediction_cache_path(root: Path, model: str, fold: int, name: str) -> Path:
    return root / f"{model}_fold_{fold}" / f"{Path(name).stem}.npz"


def save_prediction_cache(
    path: Path,
    predictions: list[np.ndarray],
    model: str,
    size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if predictions:
        packed = np.packbits(np.stack(predictions).astype(np.uint8), axis=-1)
    else:
        packed = np.empty((0, 0, 0), dtype=np.uint8)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        packed=packed,
        size=np.asarray(size, dtype=np.int32),
        model=np.asarray(model),
        version=np.asarray(CACHE_VERSION),
    )
    temporary.replace(path)


def load_prediction_cache(path: Path, model: str, size: int) -> list[np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if (
                str(data["version"].item()) != CACHE_VERSION
                or str(data["model"].item()) != model
                or int(data["size"].item()) != size
            ):
                return None
            packed = data["packed"]
            if packed.size == 0:
                return []
            return [
                np.unpackbits(item, axis=-1)[:, :size].astype(bool)
                for item in packed
            ]
    except (KeyError, OSError, ValueError):
        return None


def preferred_oof_root() -> Path:
    local = Path(os.environ.get("TEMP", "C:/Temp")) / "solar_diag_oof"
    return local if local.exists() and any(local.iterdir()) else OOF_CACHE


def preferred_image_root() -> Path:
    local = Path(os.environ.get("TEMP", "C:/Temp")) / "solar_diag_images"
    return local if local.exists() and any(local.iterdir()) else PREPROC


def resolve_yolo_cache_path(oof_root: Path, fold: int, name: str) -> Path:
    """Use the staged cache when present, otherwise fall back per observation."""
    relative = Path(f"yolo_fold_{fold}") / f"{Path(name).stem}.npz"
    preferred = oof_root / relative
    fallback = OOF_CACHE / relative
    return preferred if preferred.exists() or not fallback.exists() else fallback


def resolve_preprocessed_image_path(image_root: Path, name: str) -> Path:
    """Use the staged image when present, otherwise fall back per observation."""
    preferred = image_root / name
    fallback = PREPROC / name
    return preferred if preferred.exists() or not fallback.exists() else fallback


def refiner_checkpoint_path(fold: int) -> Path:
    local = (
        Path(os.environ.get("TEMP", "C:/Temp"))
        / "solar_diag_ckpts"
        / f"mediseg_refiner_fold{fold}_resume.pt"
    )
    return local if local.exists() else ARTIFACTS / f"mediseg_refiner_fold{fold}_resume.pt"


def filter_native(masks: list[np.ndarray], scores: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    keep = [index for index, (mask, score) in enumerate(zip(masks, scores))
            if score >= 0.26 and mask.sum() >= MIN_AREA]
    return [masks[index] for index in keep], scores[keep] if keep else np.zeros(0, np.float32)


def nms_masks(masks: list[np.ndarray], scores: np.ndarray, threshold: float = 0.48):
    kept_masks, kept_scores = [], []
    for index in np.argsort(-scores):
        candidate = masks[index]
        if any(
            np.logical_and(candidate, other).sum() / max(np.logical_or(candidate, other).sum(), 1)
            >= threshold
            for other in kept_masks
        ):
            continue
        kept_masks.append(candidate)
        kept_scores.append(float(scores[index]))
    return kept_masks, np.asarray(kept_scores, np.float32)


def semantic_to_instances(probability, seed_masks, scores, size: int):
    threshold, seed_conf = 0.52, 0.22
    min_area = max(2, int(round(60 * (size / 2048) ** 2)))
    foreground = probability >= threshold
    selected = [(mask, score) for mask, score in zip(seed_masks, scores) if score >= seed_conf]
    seeds, _ = nms_masks(
        [item[0] for item in selected],
        np.asarray([item[1] for item in selected], np.float32),
    ) if selected else ([], np.zeros(0, np.float32))
    markers = np.zeros(foreground.shape, np.int32)
    kernel = np.ones((3, 3), np.uint8)
    next_label = 1
    for mask in seeds:
        core = mask & (probability >= max(0.20, threshold * 0.65))
        core = cv2.erode(core.astype(np.uint8), kernel).astype(bool) & (markers == 0)
        if core.any():
            markers[core] = next_label
            next_label += 1
    components, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    for component in range(1, components):
        region = labels == component
        if stats[component, cv2.CC_STAT_AREA] < max(min_area * 2, 40) or np.any(markers[region]):
            continue
        values = np.where(region, probability, -1)
        y, x = np.unravel_index(np.argmax(values), values.shape)
        if probability[y, x] >= min(0.98, threshold + 0.10):
            markers[y, x] = next_label
            next_label += 1
    if next_label == 1:
        return []
    partition = watershed(
        -probability, markers=markers, mask=foreground, watershed_line=True
    )
    return [(partition == label) for label in range(1, next_label)
            if (partition == label).sum() >= min_area]


def mean_or_nan(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def prepare_variant(annotations, size):
    height = int(annotations[0]["_height"]) if annotations else 2048
    width = int(annotations[0]["_width"]) if annotations else 2048
    truths = [resize_mask(annotation_mask(annotation, height, width), size) for annotation in annotations]
    spines = []
    for annotation in annotations:
        audit = audit_spine_coordinates(annotation.get("spine"), height, width)
        spines.append(audit.points * np.asarray([size / width, size / height]))
    return {
        "annotations": annotations,
        "truths": truths,
        "spines": spines,
    }


def evaluate_variant(name, fold, image_id, prepared, predictions, size, meta=None):
    meta = meta or {}
    truths = prepared["truths"]
    spines = prepared["spines"]
    annotations = prepared["annotations"]
    strict_iou_threshold = float(np.nextafter(0.5, 1.0))
    matches = match_instances(truths, predictions, minimum_iou=strict_iou_threshold)
    metric_values = defaultdict(list)
    tolerant_spine_by_truth = [
        0.0 if len(points) else float("nan") for points in spines
    ]
    strict_spine_by_truth = tolerant_spine_by_truth.copy()
    for truth_index, prediction_index, iou in matches:
        truth = truths[truth_index]
        prediction = predictions[prediction_index]
        metric_values["iou"].append(iou)
        if len(spines[truth_index]):
            coverage = spine_coverage(spines[truth_index], prediction, radius=1)
            strict_coverage = spine_coverage(
                spines[truth_index], prediction, radius=0
            )
            tolerant_spine_by_truth[truth_index] = coverage
            strict_spine_by_truth[truth_index] = strict_coverage
            metric_values["spine_coverage_matched"].append(coverage)
            metric_values["spine_coverage_matched_strict"].append(strict_coverage)
            spine_line = spine_centerline_metrics(
                spines[truth_index], prediction, tolerance=2
            )
            for key, value in spine_line.items():
                metric_values[f"spine_centerline_{key}"].append(value)
        metric_values["boundary_f1"].append(boundary_f1(truth, prediction, tolerance=2))
        centerline = paired_centerline_metrics(truth, prediction)
        metric_values["cldice"].append(centerline["cldice"])
        metric_values["centerline_distance"].append(centerline["centerline_distance"])
        for key, value in matched_geometry_errors(truth, prediction).items():
            metric_values[key].append(value)
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(truths) - tp
    iou_sum = float(sum(match[2] for match in matches))
    denominator = tp + 0.5 * fp + 0.5 * fn
    recall = tp / len(truths) if truths else 0.0
    precision = tp / len(predictions) if predictions else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    split_count, merge_count = fragmentation_counts(truths, predictions, overlap_fraction=0.10)
    empty = np.zeros((size, size), dtype=bool)
    truth_union = np.logical_or.reduce(truths) if truths else empty
    prediction_union = np.logical_or.reduce(predictions) if predictions else empty
    truth_topology = topology_counts(truth_union)
    prediction_topology = topology_counts(prediction_union)
    category_counts = defaultdict(int)
    for annotation in annotations:
        category_counts[int(annotation.get("category_id", -1))] += 1
    row = {
        "file_name": name,
        "fold": fold,
        "site_code": meta.get("site_code", ""),
        "year": meta.get("year", ""),
        "time_block_role": meta.get("time_block_role", ""),
        "labeler_count": meta.get("labeler_count", ""),
        "instance_count_disagreement": (
            float(meta.get("instance_count_max", 0)) - float(meta.get("instance_count_min", 0))
        ),
        "image_id": image_id,
        "labeler": str(image_id).split("-", 1)[0],
        "gt_count": len(truths),
        "prediction_count": len(predictions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_sum": iou_sum,
        "pq_512": iou_sum / denominator if denominator else 0.0,
        "filament_recall": recall,
        "filament_precision": precision,
        "filament_f1": f1,
        "spine_coverage": mean_or_nan(tolerant_spine_by_truth),
        "spine_coverage_strict": mean_or_nan(strict_spine_by_truth),
        "count_abs_error": abs(len(predictions) - len(truths)),
        "split_count": split_count,
        "merge_count": merge_count,
        "truth_components": truth_topology["components"],
        "prediction_components": prediction_topology["components"],
        "component_abs_error": abs(
            prediction_topology["components"] - truth_topology["components"]
        ),
        "endpoint_abs_error": abs(
            prediction_topology["endpoints"] - truth_topology["endpoints"]
        ),
        "branchpoint_abs_error": abs(
            prediction_topology["branchpoints"] - truth_topology["branchpoints"]
        ),
        "category_1_gt_count": category_counts[1],
        "category_2_gt_count": category_counts[2],
        "gt_area_pixels_512": int(sum(mask.sum() for mask in truths)),
    }
    for key in (
        "iou", "spine_coverage_matched", "spine_coverage_matched_strict",
        "spine_centerline_precision", "spine_centerline_recall",
        "spine_centerline_f1", "spine_centerline_distance", "boundary_f1", "cldice",
        "centerline_distance", "area_relative_error", "perimeter_relative_error",
        "major_axis_relative_error", "centroid_error_pixels",
    ):
        row[key] = mean_or_nan(metric_values[key])
    return row


def load_refiner(fold, threads):
    torch.set_num_threads(threads)
    checkpoint = torch.load(
        refiner_checkpoint_path(fold), map_location="cpu", weights_only=False
    )
    model = MediSegUNet(3, 1, False).eval()
    model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)
    return model


def summarize_group(group, bootstrap_repeats):
    tp = int(group["tp"].sum())
    fp = int(group["fp"].sum())
    fn = int(group["fn"].sum())
    iou_sum = float(group["iou_sum"].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    row = {
        "variants": len(group),
        "observations": group["file_name"].nunique(),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pq_512": iou_sum / denominator if denominator else 0.0,
        "sq_512": iou_sum / tp if tp else 0.0,
        "rq_512": tp / denominator if denominator else 0.0,
    }
    for endpoint in SCIENCE_ENDPOINTS:
        ci = cluster_bootstrap_mean(
            group[endpoint].to_numpy(),
            group["file_name"].to_numpy(),
            repeats=bootstrap_repeats,
        )
        row[endpoint] = ci["mean"]
        row[f"{endpoint}_ci_low"] = ci["ci_low"]
        row[f"{endpoint}_ci_high"] = ci["ci_high"]
    return row


def paired_delta_rows(frame, bootstrap_repeats, reference="v35"):
    keys = ["file_name", "image_id"]
    endpoints = ("pq_512",) + SCIENCE_ENDPOINTS
    reference_frame = frame[frame.model == reference].set_index(keys)
    rows = []
    for candidate in sorted(set(frame.model) - {reference}):
        candidate_frame = frame[frame.model == candidate].set_index(keys)
        common = reference_frame.index.intersection(candidate_frame.index)
        for endpoint in endpoints:
            candidate_values = candidate_frame.loc[common, endpoint].to_numpy(dtype=float)
            reference_values = reference_frame.loc[common, endpoint].to_numpy(dtype=float)
            raw_delta = candidate_values - reference_values
            improvement = -raw_delta if endpoint in LOWER_IS_BETTER else raw_delta
            ci = cluster_bootstrap_mean(
                improvement,
                np.asarray([index[0] for index in common]),
                repeats=bootstrap_repeats,
            )
            rows.append({
                "reference": reference,
                "candidate": candidate,
                "endpoint": endpoint,
                "paired_variants": len(common),
                "candidate_mean": mean_or_nan(candidate_values),
                "reference_mean": mean_or_nan(reference_values),
                "raw_delta_candidate_minus_reference": mean_or_nan(raw_delta),
                "improvement": ci["mean"],
                "improvement_ci_low": ci["ci_low"],
                "improvement_ci_high": ci["ci_high"],
            })
    return rows
