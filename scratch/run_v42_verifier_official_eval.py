"""Evaluate the nested-OOF V42 candidate verifier with official 2048 PQ.

The candidate verifier is trained and tuned on the two non-held-out folds.  For
each outer-fold observation this script reconstructs the exact V42 WBF proposal
order, applies that observation's nested-OOF decisions *before* the refiner,
then runs the frozen fold-specific refiner and fixed ownership rule.  The output
is therefore both leakage-safe and representative of production inference.

Prediction caches make the CPU experiment resumable and allow folds to run in
separate processes::

    python scratch/run_v42_verifier_official_eval.py --folds 0 --prediction-only --threads 4
    python scratch/run_v42_verifier_official_eval.py --folds 1 --prediction-only --threads 4
    python scratch/run_v42_verifier_official_eval.py --folds 2 --prediction-only --threads 4
    python scratch/run_v42_verifier_official_eval.py --aggregate-only --threads 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solar_filament.pipeline.mask_wbf import weighted_mask_fusion  # noqa: E402
from solar_filament.models.mediseg_models import MediSegUNet  # noqa: E402
from solar_filament.metrics.metrics_official import PanopticQuality  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    FOLDS_CSV,
    OOF_CACHE,
    PREPROC,
    TRAIN_JSON,
    build_variant,
    imread_unicode,
    iou_matrix,
    load_yolo_cache,
    masks_to_rles,
    refine_batch,
    to_rle,
)
from scratch.run_s0_baseline_rescore import (  # noqa: E402
    SEED8407,
    filter_native,
    refiner_checkpoint_path,
    resolve_preprocessed_image_path,
)
from scratch.run_wbf_dropcontained_threshold_probe import (  # noqa: E402
    apply_ownership_with_threshold,
    filter_proposals,
)

CACHE_SCHEMA = "v42-verifier-official-prediction-v1"
WBF_IOU = 0.5
WBF_VOTE = 0.6
REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45
OWNERSHIP_THRESHOLD = 0.80
ARCHIVED_V35_PQ = 0.4426332944385452
ARCHIVED_V42_PQ = 0.4399323740073615
FORMAL_GATE_PQ = 0.452633


def boolean_values(values: pd.Series) -> np.ndarray:
    """Parse CSV booleans without treating the string ``False`` as truthy."""
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = set(normalized.unique()) - allowed
    if unexpected:
        raise ValueError(f"unexpected verifier_accept values: {sorted(unexpected)}")
    return normalized.isin({"true", "1"}).to_numpy(dtype=bool)


def decisions_by_observation(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return candidate-index ordered outer-OOF decisions with integrity checks."""
    required = {"file_name", "candidate_index", "verifier_accept", "outer_fold", "fold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"nested prediction columns missing: {sorted(missing)}")
    if not (frame["outer_fold"].astype(int) == frame["fold"].astype(int)).all():
        raise ValueError("outer_fold does not match the observation fold")
    output: dict[str, np.ndarray] = {}
    for name, group in frame.groupby("file_name", sort=False):
        ordered = group.sort_values("candidate_index")
        indices = ordered["candidate_index"].to_numpy(dtype=int)
        if not np.array_equal(indices, np.arange(len(ordered))):
            raise ValueError(f"candidate indices are not contiguous for {name}")
        output[str(name)] = boolean_values(ordered["verifier_accept"])
    return output


def save_masks(
    path: Path,
    masks: list[np.ndarray],
    *,
    file_name: str,
    fold: int,
    fused_count: int,
    accepted_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if masks:
        height, width = masks[0].shape
        packed = np.packbits(np.stack(masks).astype(np.uint8), axis=-1)
    else:
        height = width = 2048
        packed = np.empty((0, 0, 0), dtype=np.uint8)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray(CACHE_SCHEMA),
        file_name=np.asarray(file_name),
        fold=np.asarray(fold, dtype=np.int16),
        shape=np.asarray([height, width], dtype=np.int32),
        packed=packed,
        fused_count=np.asarray(fused_count, dtype=np.int16),
        accepted_count=np.asarray(accepted_count, dtype=np.int16),
        output_count=np.asarray(len(masks), dtype=np.int16),
    )
    temporary.replace(path)


def load_masks(path: Path, expected_name: str, expected_fold: int) -> tuple[list[np.ndarray], dict] | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if (
                str(data["schema"].item()) != CACHE_SCHEMA
                or str(data["file_name"].item()) != expected_name
                or int(data["fold"].item()) != expected_fold
            ):
                return None
            height, width = map(int, data["shape"])
            packed = data["packed"]
            masks = (
                [np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed]
                if packed.size
                else []
            )
            if masks and any(mask.shape != (height, width) for mask in masks):
                return None
            metadata = {
                "fused_count": int(data["fused_count"].item()),
                "accepted_count": int(data["accepted_count"].item()),
                "output_count": int(data["output_count"].item()),
            }
            return masks, metadata
    except (KeyError, OSError, ValueError):
        return None


def cluster_bootstrap_ratio(
    numerators: np.ndarray,
    denominators: np.ndarray,
    repeats: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Observation-cluster bootstrap for a corpus-level ratio such as PQ."""
    numerator_values = np.asarray(numerators, dtype=np.float64)
    denominator_values = np.asarray(denominators, dtype=np.float64)
    if numerator_values.shape != denominator_values.shape or numerator_values.ndim != 1:
        raise ValueError("numerators and denominators must be aligned 1-D arrays")
    if len(numerator_values) == 0 or denominator_values.sum() <= 0:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        indices = rng.integers(0, len(numerator_values), len(numerator_values))
        denominator = denominator_values[indices].sum()
        samples[repeat] = numerator_values[indices].sum() / denominator if denominator else 0.0
    return {
        "estimate": float(numerator_values.sum() / denominator_values.sum()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def load_refiner(fold: int) -> MediSegUNet:
    path = refiner_checkpoint_path(fold)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = MediSegUNet(3, 1, False).eval()
    model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)
    print(f"fold{fold} refiner: {path} (best Dice {checkpoint.get('best_dice'):.6f})")
    return model


def fused_proposals(name: str, fold: int) -> tuple[list[np.ndarray], np.ndarray]:
    stem = Path(name).stem
    masks1, scores1 = load_yolo_cache(OOF_CACHE / f"yolo_fold_{fold}" / f"{stem}.npz")
    masks2, scores2 = load_yolo_cache(SEED8407[fold] / f"{stem}.npz")
    first, scores1 = filter_native(masks1, scores1)
    second, scores2 = filter_proposals(masks2, scores2)
    combined = first + second
    combined_scores = (
        np.concatenate([scores1, scores2]) if combined else np.zeros(0, np.float32)
    )
    sources = [0] * len(first) + [1] * len(second)
    fused, scores, _ = weighted_mask_fusion(
        combined,
        combined_scores,
        sources,
        iou_threshold=WBF_IOU,
        vote_threshold=WBF_VOTE,
    )
    return fused, scores


def build_prediction(
    name: str,
    fold: int,
    decisions: np.ndarray,
    refiner: MediSegUNet,
) -> tuple[list[np.ndarray], dict]:
    fused, _ = fused_proposals(name, fold)
    if len(fused) != len(decisions):
        raise ValueError(
            f"candidate count mismatch for {name}: reconstructed={len(fused)}, nested={len(decisions)}"
        )
    accepted = [mask for mask, keep in zip(fused, decisions) if keep]
    if accepted:
        image = imread_unicode(resolve_preprocessed_image_path(PREPROC, name))
        if image is None:
            raise FileNotFoundError(PREPROC / name)
        boxes, probabilities = refine_batch(refiner, image, accepted)
        refined = [
            build_variant(mask, box, probability, REFINER_METHOD, REFINER_THRESHOLD)
            for mask, box, probability in zip(accepted, boxes, probabilities)
        ]
        final = apply_ownership_with_threshold(refined, OWNERSHIP_THRESHOLD)
    else:
        final = []
    return final, {
        "fused_count": len(fused),
        "accepted_count": len(accepted),
        "output_count": len(final),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--prediction-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=ROOT / "output/s0")
    args = parser.parse_args()
    folds = sorted(set(args.folds))
    if not folds or set(folds) - {0, 1, 2}:
        raise ValueError(f"folds must be drawn from 0,1,2: {folds}")
    if args.prediction_only and args.aggregate_only:
        raise ValueError("--prediction-only and --aggregate-only are mutually exclusive")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    cache_root = args.out / "v42_verifier_official_cache"

    nested_path = args.out / "v42_nested_candidate_predictions.csv"
    nested = pd.read_csv(nested_path)
    decisions = decisions_by_observation(nested)

    with TRAIN_JSON.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    file_to_ids = defaultdict(list)
    for image_id, image in images.items():
        file_to_ids[image["file_name"]].append(image_id)
    gt_rles = {
        image_id: [
            to_rle(annotation, int(images[image_id]["height"]), int(images[image_id]["width"]))
            for annotation in image_annotations
        ]
        for image_id, image_annotations in annotations.items()
    }
    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of = dict(zip(fold_frame.file_name, fold_frame.fold.astype(int)))
    targets = [name for name in sorted(file_to_ids) if fold_of[name] in folds]
    if args.limit:
        targets = targets[: args.limit]

    refiners: dict[int, MediSegUNet] = {}
    cache_hits = 0
    cache_builds = 0
    metadata_rows = []
    started = time.perf_counter()
    for position, name in enumerate(targets, 1):
        fold = fold_of[name]
        cache_path = cache_root / f"fold_{fold}" / f"{Path(name).stem}.npz"
        cached = load_masks(cache_path, name, fold)
        if cached is None:
            if args.aggregate_only:
                raise FileNotFoundError(f"official verifier cache missing: {cache_path}")
            if fold not in refiners:
                refiners[fold] = load_refiner(fold)
            observation_decisions = decisions.get(name, np.zeros(0, dtype=bool))
            masks, metadata = build_prediction(name, fold, observation_decisions, refiners[fold])
            save_masks(
                cache_path,
                masks,
                file_name=name,
                fold=fold,
                fused_count=metadata["fused_count"],
                accepted_count=metadata["accepted_count"],
            )
            cache_builds += 1
        else:
            masks, metadata = cached
            cache_hits += 1
        metadata_rows.append({"file_name": name, "fold": fold, **metadata})
        if position % 25 == 0 or position == len(targets):
            elapsed = time.perf_counter() - started
            print(
                f"prediction {position}/{len(targets)} hits={cache_hits} builds={cache_builds} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    metadata_frame = pd.DataFrame(metadata_rows)
    suffix = "_" + "-".join(map(str, folds))
    if args.limit:
        suffix += f"_limit{args.limit}"
    metadata_path = args.out / f"v42_verifier_official_prediction_manifest{suffix}.csv"
    metadata_frame.to_csv(metadata_path, index=False)
    if args.prediction_only:
        print(json.dumps({
            "status": "prediction cache complete",
            "folds": folds,
            "observations": len(targets),
            "cache_hits": cache_hits,
            "cache_builds": cache_builds,
            "fused_candidates": int(metadata_frame.fused_count.sum()),
            "accepted_candidates": int(metadata_frame.accepted_count.sum()),
            "final_instances": int(metadata_frame.output_count.sum()),
            "manifest": str(metadata_path),
        }, ensure_ascii=False, indent=2))
        return 0

    overall = PanopticQuality()
    by_fold = {fold: PanopticQuality() for fold in folds}
    case_rows = []
    for position, name in enumerate(targets, 1):
        fold = fold_of[name]
        cache_path = cache_root / f"fold_{fold}" / f"{Path(name).stem}.npz"
        loaded = load_masks(cache_path, name, fold)
        if loaded is None:
            raise RuntimeError(f"cache disappeared during evaluation: {cache_path}")
        masks, metadata = loaded
        prediction_rles = masks_to_rles(masks)
        totals = {"tp": 0, "fp": 0, "fn": 0, "tp_iou_sum": 0.0}
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, prediction_rles)
            row = overall.add_matrix(matrix, len(truth), len(prediction_rles))
            by_fold[fold].add_matrix(matrix, len(truth), len(prediction_rles))
            for key in totals:
                totals[key] += row[key]
        denominator = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
        case_rows.append({
            "file_name": name,
            "fold": fold,
            **metadata,
            **totals,
            "denominator": denominator,
            "observation_pq": totals["tp_iou_sum"] / denominator if denominator else 0.0,
        })
        if position % 50 == 0 or position == len(targets):
            print(f"official PQ {position}/{len(targets)} pq={overall.score:.6f}", flush=True)

    case_frame = pd.DataFrame(case_rows)
    case_path = args.out / f"v42_verifier_official_case_metrics{suffix}.csv"
    case_frame.to_csv(case_path, index=False)
    strata_source = pd.read_csv(args.out / "observation_strata.csv")
    enriched = case_frame.merge(
        strata_source[["file_name", "site_code", "time_block_role"]],
        on="file_name",
        how="left",
        validate="one_to_one",
    )
    stratum_rows = []
    for stratum_type, column in (
        ("fold", "fold"),
        ("site", "site_code"),
        ("time_block", "time_block_role"),
    ):
        for value, group in enriched.groupby(column, dropna=False):
            numerator = float(group.tp_iou_sum.sum())
            denominator = float(group.denominator.sum())
            stratum_rows.append({
                "stratum_type": stratum_type,
                "stratum_value": str(value),
                "observations": len(group),
                "pq": numerator / denominator if denominator else 0.0,
                "tp": int(group.tp.sum()),
                "fp": int(group.fp.sum()),
                "fn": int(group.fn.sum()),
                "tp_iou_sum": numerator,
                "denominator": denominator,
                "fused_candidates": int(group.fused_count.sum()),
                "accepted_candidates": int(group.accepted_count.sum()),
                "final_instances": int(group.output_count.sum()),
            })
    stratum_frame = pd.DataFrame(stratum_rows)
    stratum_path = args.out / f"v42_verifier_official_stratum_metrics{suffix}.csv"
    stratum_frame.to_csv(stratum_path, index=False)
    summary = overall.summary()
    pq_interval = cluster_bootstrap_ratio(
        case_frame.tp_iou_sum.to_numpy(),
        case_frame.denominator.to_numpy(),
        repeats=args.bootstrap_repeats,
    )
    full_allfold = folds == [0, 1, 2] and not args.limit and len(targets) == 707
    report = {
        "schema_version": "v42-verifier-official-eval-v1",
        "status": (
            "retrospective nested-OOF complete; not independent holdout"
            if full_allfold
            else "diagnostic subset"
        ),
        "scope": f"{len(targets)} observations, folds {folds}, outer-OOF verifier + fold-specific refiner",
        "settings": {
            "wbf_iou_threshold": WBF_IOU,
            "wbf_vote_threshold": WBF_VOTE,
            "refiner": f"{REFINER_METHOD}@{REFINER_THRESHOLD:.2f}",
            "ownership_drop_contained_threshold": OWNERSHIP_THRESHOLD,
            "verifier_application": "before refiner",
        },
        **summary,
        "pq_observation_cluster_bootstrap": pq_interval,
        "bootstrap_repeats": args.bootstrap_repeats,
        "by_fold": {str(fold): by_fold[fold].summary() for fold in folds},
        "proposal_counts": {
            "fused": int(metadata_frame.fused_count.sum()),
            "accepted": int(metadata_frame.accepted_count.sum()),
            "rejected": int((metadata_frame.fused_count - metadata_frame.accepted_count).sum()),
            "final_instances": int(metadata_frame.output_count.sum()),
            "accepted_fraction": float(
                metadata_frame.accepted_count.sum() / metadata_frame.fused_count.sum()
            ),
        },
        "archived_v42_accept_all_pq": ARCHIVED_V42_PQ,
        "delta_vs_v42_accept_all": summary["pq"] - ARCHIVED_V42_PQ if full_allfold else None,
        "archived_v35_pq": ARCHIVED_V35_PQ,
        "delta_vs_v35": summary["pq"] - ARCHIVED_V35_PQ if full_allfold else None,
        "formal_gate_pq": FORMAL_GATE_PQ,
        "passes_v35": bool(summary["pq"] > ARCHIVED_V35_PQ) if full_allfold else None,
        "passes_formal_gate": bool(summary["pq"] >= FORMAL_GATE_PQ) if full_allfold else None,
        "decision": (
            "reject as V35 replacement; retain as negative result and S1 hypothesis evidence"
            if full_allfold and summary["pq"] < ARCHIVED_V35_PQ
            else "not decided on diagnostic subset"
        ),
        "limitation": "same 707 grouped OOF observations used for nested model selection; no time/site holdout claim",
        "case_metrics": str(case_path),
        "stratum_metrics": str(stratum_path),
        "prediction_manifest": str(metadata_path),
        "nested_predictions": str(nested_path),
    }
    report_path = args.out / f"v42_verifier_official_eval{suffix}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"elapsed={time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
