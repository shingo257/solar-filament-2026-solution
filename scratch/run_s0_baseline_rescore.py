"""S0-3: V35/raw YOLO/V34/V42を同じ科学endpointで再評価する。

既存のリークなしOOF cacheとfold別refinerだけを使うCPU評価。科学指標は
``--eval-size``（既定512）へ統一して計算し、公式2048 PQと混同しない。
V42は採用候補ではなくRecall-FP trade-offを測るnegative controlである。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util
from skimage.segmentation import watershed


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solar_filament.pipeline.mask_wbf import weighted_mask_fusion  # noqa: E402
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
from scratch.run_pipeline_stack import apply_ownership  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS,
    FOLDS_CSV,
    MIN_AREA,
    OOF_CACHE,
    PREPROC,
    TRAIN_JSON,
    build_variant,
    imread_unicode,
    load_yolo_cache,
    refine_batch,
)
from scratch.run_wbf_dropcontained_threshold_probe import (  # noqa: E402
    apply_ownership_with_threshold,
    filter_proposals,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--folds", nargs="+", type=int, choices=(0, 1, 2))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prediction-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--prediction-cache", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "output/s0")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    prediction_root = args.prediction_cache or args.out / f"prediction_cache_{args.eval_size}"

    with TRAIN_JSON.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        image = images[str(annotation["image_id"])]
        copy = dict(annotation)
        copy["_height"], copy["_width"] = image["height"], image["width"]
        annotations_by_image[str(annotation["image_id"])].append(copy)
    file_to_ids = defaultdict(list)
    for image_id, image in images.items():
        file_to_ids[image["file_name"]].append(image_id)
    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of = dict(zip(fold_frame.file_name, fold_frame.fold.astype(int)))
    strata = pd.read_csv(args.out / "observation_strata.csv")
    stratum = strata.set_index("file_name").to_dict(orient="index")
    targets = [name for name in sorted(file_to_ids) if name in fold_of]
    if args.folds:
        targets = [name for name in targets if fold_of[name] in set(args.folds)]
    if args.limit:
        targets = targets[: args.limit]

    suffix = "_v3"
    suffix += f"_limit{args.limit}" if args.limit else ""
    if args.folds:
        suffix += "_folds" + "-".join(str(fold) for fold in sorted(args.folds))
    model_tag = "-".join(args.models)
    checkpoint_path = (
        args.out / f"science_case_metrics_checkpoint_v2_{model_tag}_{args.eval_size}{suffix}.csv"
    )
    refiners = {}
    cache_stats = {model: {"hits": 0, "built": 0} for model in args.models}
    oof_root = preferred_oof_root()
    image_root = preferred_image_root()
    rows = []
    completed = set()
    if not args.prediction_only and args.resume and checkpoint_path.exists():
        checkpoint_frame = pd.read_csv(checkpoint_path)
        required = {"file_name", "model", "iou_sum", "spine_coverage_matched"}
        if required <= set(checkpoint_frame.columns):
            for file_name, group in checkpoint_frame.groupby("file_name"):
                if set(group.model.unique()) == set(args.models):
                    completed.add(file_name)
            checkpoint_frame = checkpoint_frame[checkpoint_frame.file_name.isin(completed)]
            rows = checkpoint_frame.to_dict(orient="records")
            print(
                f"checkpointから再開: {len(completed)}/{len(targets)}観測完了済み",
                flush=True,
            )
        else:
            print("旧形式checkpointは無視します", flush=True)
    pending = [name for name in targets if name not in completed]
    started = time.perf_counter()
    for position, name in enumerate(pending, len(completed) + 1):
        fold = fold_of[name]
        model_predictions = {}
        missing = []
        for model in args.models:
            cache_path = prediction_cache_path(prediction_root, model, fold, name)
            cached = None if args.rebuild_cache else load_prediction_cache(
                cache_path, model, args.eval_size
            )
            if cached is None:
                missing.append(model)
            else:
                model_predictions[model] = cached
                cache_stats[model]["hits"] += 1

        if missing:
            yolo_path = resolve_yolo_cache_path(oof_root, fold, name)
            masks, scores = load_yolo_cache(yolo_path)
            native, native_scores = filter_native(masks, scores)
            image = None
            if {"v35", "v42"} & set(missing):
                image_path = resolve_preprocessed_image_path(image_root, name)
                image = imread_unicode(image_path)
                if image is None:
                    raise FileNotFoundError(image_path)
                if fold not in refiners:
                    refiners[fold] = load_refiner(fold, args.threads)
            if "raw" in missing:
                model_predictions["raw"] = apply_ownership(
                    native, native_scores, "conf_first"
                )
            if "v35" in missing:
                boxes, probabilities = (
                    refine_batch(refiners[fold], image, native) if native else ([], [])
                )
                refined = [
                    build_variant(
                        native[index], boxes[index], probabilities[index], "shrink", 0.45
                    )
                    for index in range(len(native))
                ]
                model_predictions["v35"] = apply_ownership(
                    refined, native_scores, "drop_contained_then_area_asc"
                )
            if "v42" in missing:
                second_masks, second_scores = load_yolo_cache(
                    SEED8407[fold] / f"{Path(name).stem}.npz"
                )
                second, second_scores = filter_proposals(second_masks, second_scores)
                combined = native + second
                combined_scores = (
                    np.concatenate([native_scores, second_scores])
                    if combined else np.zeros(0, np.float32)
                )
                sources = [0] * len(native) + [1] * len(second)
                fused, _, _ = weighted_mask_fusion(
                    combined,
                    combined_scores,
                    sources,
                    iou_threshold=0.5,
                    vote_threshold=0.6,
                )
                boxes, probabilities = (
                    refine_batch(refiners[fold], image, fused) if fused else ([], [])
                )
                built = [
                    build_variant(
                        fused[index], boxes[index], probabilities[index], "shrink", 0.45
                    )
                    for index in range(len(fused))
                ]
                model_predictions["v42"] = apply_ownership_with_threshold(built, 0.80)
            if "v34" in missing:
                probability_path = (
                    V34_SEMANTIC / f"semantic_fold_{fold}" / f"{Path(name).stem}.png"
                )
                probability = cv2.imdecode(
                    np.fromfile(probability_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                if probability is None:
                    raise FileNotFoundError(probability_path)
                probability = cv2.resize(
                    probability.astype(np.float32) / 65535.0,
                    (args.eval_size, args.eval_size),
                    interpolation=cv2.INTER_AREA,
                )
                resized_seeds = [resize_mask(mask, args.eval_size) for mask in masks]
                model_predictions["v34"] = semantic_to_instances(
                    probability, resized_seeds, scores, args.eval_size
                )
            for model in missing:
                predictions = model_predictions[model]
                if model != "v34":
                    predictions = [resize_mask(mask, args.eval_size) for mask in predictions]
                    model_predictions[model] = predictions
                save_prediction_cache(
                    prediction_cache_path(prediction_root, model, fold, name),
                    predictions,
                    model,
                    args.eval_size,
                )
                cache_stats[model]["built"] += 1

        if args.prediction_only:
            if position % 10 == 0 or position == len(targets):
                print(
                    f"{position}/{len(targets)} prediction cache {time.perf_counter() - started:.0f}s",
                    flush=True,
                )
            continue

        meta = stratum.get(name, {})
        jobs = []
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            for image_id in file_to_ids[name]:
                prepared = prepare_variant(annotations_by_image[image_id], args.eval_size)
                for model in args.models:
                    future = executor.submit(
                        evaluate_variant,
                        name,
                        fold,
                        image_id,
                        prepared,
                        model_predictions[model],
                        args.eval_size,
                        meta,
                    )
                    jobs.append((model, future))
            for model, future in jobs:
                row = future.result()
                row["model"] = model
                rows.append(row)
        if position % 10 == 0 or position == len(targets):
            checkpoint_frame = pd.DataFrame(rows)
            temporary = checkpoint_path.with_suffix(".tmp")
            checkpoint_frame.to_csv(temporary, index=False)
            temporary.replace(checkpoint_path)
            print(f"{position}/{len(targets)} {time.perf_counter() - started:.0f}s checkpoint保存", flush=True)

    if args.prediction_only:
        cache_report = {
            "schema_version": "s0-science-prediction-cache-v2",
            "cache_version": CACHE_VERSION,
            "eval_size": args.eval_size,
            "folds": sorted(args.folds) if args.folds else [0, 1, 2],
            "models": args.models,
            "observations": len(targets),
            "elapsed_seconds": time.perf_counter() - started,
            "statistics": cache_stats,
            "prediction_cache": str(prediction_root),
        }
        fold_tag = "-".join(str(fold) for fold in cache_report["folds"])
        cache_manifest = args.out / f"prediction_cache_manifest_folds{fold_tag}.json"
        cache_manifest.write_text(
            json.dumps(cache_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(cache_report, ensure_ascii=False, indent=2))
        return 0

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["file_name", "image_id", "model"]).reset_index(drop=True)
    annotation_audit_path = args.out / "annotation_spine_audit.csv"
    annotation_audit = pd.read_csv(annotation_audit_path, dtype={"image_id": str})
    image_features = (
        annotation_audit.groupby("image_id", as_index=False)
        .agg(
            annotation_area_median_2048=("area", "median"),
            bbox_center_radius_median=("bbox_center_radius_normalized", "median"),
            bbox_center_radius_max=("bbox_center_radius_normalized", "max"),
        )
    )
    frame["image_id"] = frame["image_id"].astype(str)
    frame = frame.merge(image_features, on="image_id", how="left", validate="many_to_one")
    frame["area_group"] = pd.cut(
        frame["annotation_area_median_2048"],
        bins=[-np.inf, 1000.0, 5000.0, np.inf],
        labels=["small_le_1000", "medium_1000_5000", "large_gt_5000"],
    ).astype(str)
    frame["density_group"] = pd.cut(
        frame["gt_count"],
        bins=[-np.inf, 5, 10, np.inf],
        labels=["sparse_le_5", "medium_6_10", "dense_gt_10"],
    ).astype(str)
    frame["radial_group"] = pd.cut(
        frame["bbox_center_radius_max"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["central_le_0.33", "mid_0.33_0.66", "limb_proxy_gt_0.66"],
    ).astype(str)
    case_path = args.out / f"science_case_metrics{suffix}.csv"
    frame.to_csv(case_path, index=False)

    summary_rows = []
    for model, group in frame.groupby("model"):
        row = {"model": model}
        row.update(summarize_group(group, args.bootstrap_repeats))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("model")
    summary_path = args.out / f"science_model_summary{suffix}.csv"
    summary.to_csv(summary_path, index=False)

    stratum_frame = frame.copy()
    stratum_frame["disagreement_group"] = np.where(
        pd.to_numeric(stratum_frame["instance_count_disagreement"], errors="coerce").fillna(0) > 0,
        "count_disagreement",
        "count_agreement",
    )
    stratum_rows = []
    for model, group in stratum_frame.groupby("model"):
        row = {"model": model, "stratum_type": "all", "stratum_value": "all"}
        row.update(summarize_group(group, args.bootstrap_repeats))
        stratum_rows.append(row)
    for stratum_type, column in (
        ("time_block", "time_block_role"),
        ("site", "site_code"),
        ("count_disagreement", "disagreement_group"),
        ("labeler_count", "labeler_count"),
        ("fold", "fold"),
        ("area", "area_group"),
        ("density", "density_group"),
        ("radial_proxy", "radial_group"),
    ):
        for (model, value), group in stratum_frame.groupby(["model", column], dropna=False):
            row = {
                "model": model,
                "stratum_type": stratum_type,
                "stratum_value": str(value),
            }
            row.update(summarize_group(group, args.bootstrap_repeats))
            stratum_rows.append(row)
    stratum_summary = pd.DataFrame(stratum_rows).sort_values(
        ["stratum_type", "stratum_value", "model"]
    )
    stratum_path = args.out / f"science_stratum_summary{suffix}.csv"
    stratum_summary.to_csv(stratum_path, index=False)

    paired_rows = (
        paired_delta_rows(frame, args.bootstrap_repeats) if "v35" in args.models else []
    )
    paired = pd.DataFrame(paired_rows)
    paired_path = args.out / f"science_paired_deltas{suffix}.csv"
    paired.to_csv(paired_path, index=False)

    output_paths = {
        "case_csv": case_path,
        "summary_csv": summary_path,
        "stratum_csv": stratum_path,
        "paired_delta_csv": paired_path,
    }
    provenance_candidates = sorted(args.out.glob("*provenance*manifest*.json"))
    report = {
        "schema_version": "s0-science-rescore-v3",
        "scoring_version": SCORING_VERSION,
        "eval_size": args.eval_size,
        "matching": "maximum-cardinality Hungarian IoU > 0.5 at eval_size; pq_512 is diagnostic, not Kaggle PQ",
        "models": args.models,
        "observations": len(targets),
        "annotator_variants": int(frame[["file_name", "image_id"]].drop_duplicates().shape[0]),
        "elapsed_seconds": time.perf_counter() - started,
        "bootstrap_repeats": args.bootstrap_repeats,
        "cluster_unit": "file_name observation",
        "prediction_cache": {
            "version": CACHE_VERSION,
            "root": str(prediction_root),
            "statistics": cache_stats,
        },
        "data_roots": {
            "yolo_oof": str(oof_root),
            "preprocessed_images": str(image_root),
            "v34_semantic": str(V34_SEMANTIC),
        },
        "v34_instance_builder": {
            "engine": "skimage.segmentation.watershed",
            "threshold": 0.52,
            "seed_confidence": 0.22,
            "minimum_area_at_2048": 60,
            "recover_unseeded_components": True,
            "watershed_line": True,
        },
        "strata_definitions": {
            "area": {
                "source": "median annotation area per annotator variant at native 2048 scale",
                "bins": ["<=1000", "1000<area<=5000", ">5000"],
            },
            "density": {
                "source": "ground-truth instance count per annotator variant",
                "bins": ["<=5", "6-10", ">10"],
            },
            "radial_proxy": {
                "source": "maximum normalized bbox-center image radius per annotator variant",
                "bins": ["<=0.33", "0.33<radius<=0.66", ">0.66"],
                "warning": "image-center proxy; not calibrated heliographic distance",
            },
        },
        "provenance_manifest": (
            str(provenance_candidates[0]) if provenance_candidates else None
        ),
        "outputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in output_paths.items()
        },
    }
    manifest_path = args.out / f"science_rescore_manifest{suffix}.json"
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        summary[
            [
                "model", "pq_512", "spine_coverage", "boundary_f1", "cldice",
                "filament_recall", "filament_precision", "fp", "fn",
            ]
        ]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
