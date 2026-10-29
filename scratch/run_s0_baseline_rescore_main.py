"""Entry-point `main()` for `run_s0_baseline_rescore` — extracted verbatim from `run_s0_baseline_rescore.py`.

Split out so both the wrapper and `run_s0_baseline_rescore_impl.py` stay readable.
"""


from __future__ import annotations


import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from solar_filament.pipeline.mask_wbf import weighted_mask_fusion  # noqa: E402
from scratch.run_pipeline_stack import apply_ownership  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    FOLDS_CSV,
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


try:
    from .run_s0_baseline_rescore_impl import (
        CACHE_VERSION,
        MODELS,
        SCORING_VERSION,
        SEED8407,
        V34_SEMANTIC,
        evaluate_variant,
        filter_native,
        load_prediction_cache,
        load_refiner,
        paired_delta_rows,
        prediction_cache_path,
        preferred_image_root,
        preferred_oof_root,
        prepare_variant,
        resize_mask,
        resolve_preprocessed_image_path,
        resolve_yolo_cache_path,
        save_prediction_cache,
        semantic_to_instances,
        sha256_file,
        summarize_group,
    )
except ImportError:  # Direct `python scratch/...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_s0_baseline_rescore_impl import (
        CACHE_VERSION,
        MODELS,
        SCORING_VERSION,
        SEED8407,
        V34_SEMANTIC,
        evaluate_variant,
        filter_native,
        load_prediction_cache,
        load_refiner,
        paired_delta_rows,
        prediction_cache_path,
        preferred_image_root,
        preferred_oof_root,
        prepare_variant,
        resize_mask,
        resolve_preprocessed_image_path,
        resolve_yolo_cache_path,
        save_prediction_cache,
        semantic_to_instances,
        sha256_file,
        summarize_group,
    )


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
