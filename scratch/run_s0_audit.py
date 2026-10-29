"""Run S0-0 provenance freeze and S0-1 MAGFiLO annotation audit.

The default paths target the Drive layout used by V30/V34.  This script never
downloads data or runs a model.  Outputs are written under ``output/s0``.

Usage:
    python scratch/run_s0_audit.py
    python scratch/run_s0_audit.py --quick-hash
"""


from __future__ import annotations


import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from solar_filament.metrics.metrics_official import METRIC_VERSION  # noqa: E402


try:
    from .run_s0_audit_impl import (
        DEFAULT_V30,
        DEFAULT_V34,
        PROJECT_ROOT,
        audit_annotations,
        directory_manifest,
        file_manifest,
        parse_labeler,
        parse_site,
        point_distance_to_annotation,
        write_csv,
    )
except ImportError:  # Direct `python scratch/...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_s0_audit_impl import (
        DEFAULT_V30,
        DEFAULT_V34,
        PROJECT_ROOT,
        audit_annotations,
        directory_manifest,
        file_manifest,
        parse_labeler,
        parse_site,
        point_distance_to_annotation,
        write_csv,
    )



__all__ = [
    "DEFAULT_V30",
    "DEFAULT_V34",
    "PROJECT_ROOT",
    "audit_annotations",
    "directory_manifest",
    "file_manifest",
    "parse_labeler",
    "parse_site",
    "point_distance_to_annotation",
    "write_csv",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v30", type=Path, default=DEFAULT_V30)
    parser.add_argument("--v34", type=Path, default=DEFAULT_V34)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/s0")
    parser.add_argument(
        "--quick-hash",
        action="store_true",
        help="Hash cache inventory only. Critical JSON/CSV/checkpoints are still content-hashed.",
    )
    args = parser.parse_args()
    train_json = args.v30 / (
        "competition_data/MAGFiLO_1.0_Kaggle_2026/train/"
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    folds_csv = args.v30 / "artifacts/grouped_folds.csv"
    required = [train_json, folds_csv]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Required S0 input is missing:")
        for path in missing:
            print(f"  {path}")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    print("[S0-0] provenance freeze", flush=True)
    checkpoint_entries = {
        f"refiner_fold_{fold}": file_manifest(
            args.v30 / f"artifacts/mediseg_refiner_fold{fold}_resume.pt"
        )
        for fold in range(3)
    }
    cache_entries = {}
    for kind in ("yolo", "semantic"):
        for fold in range(3):
            name = f"{kind}_fold_{fold}"
            cache_entries[name] = directory_manifest(
                args.v34 / f"prediction_cache/oof/{name}",
                content_hash=not args.quick_hash,
            )
    seed8407_roots = {
        0: PROJECT_ROOT / (
            "kaggle_downloads/v42/fold0_kernel_output/v42_multiseed/"
            "oof_cache/yolo_seed8407_fold_0"
        ),
        1: PROJECT_ROOT / (
            "kaggle_downloads/v42/fold1_kernel_output/v42_multiseed/"
            "oof_cache/yolo_seed8407_fold_1"
        ),
        2: PROJECT_ROOT / (
            "kaggle_downloads/v42/fold2_kernel_output/v42_multiseed/"
            "oof_cache/yolo_seed8407_fold_2"
        ),
    }
    for fold, root in seed8407_roots.items():
        cache_entries[f"yolo_seed8407_fold_{fold}"] = directory_manifest(
            root, content_hash=not args.quick_hash
        )
    expected_cache_counts = {
        "yolo_fold_0": 236,
        "yolo_fold_1": 236,
        "yolo_fold_2": 235,
        "semantic_fold_0": 236,
        "semantic_fold_1": 236,
        "semantic_fold_2": 235,
        "yolo_seed8407_fold_0": 236,
        "yolo_seed8407_fold_1": 236,
        "yolo_seed8407_fold_2": 235,
    }
    for name, expected_count in expected_cache_counts.items():
        cache_entries[name]["expected_file_count"] = expected_count
        cache_entries[name]["complete"] = (
            cache_entries[name]["exists"]
            and cache_entries[name]["file_count"] == expected_count
        )

    critical_inputs = {
        "train_annotations": file_manifest(train_json),
        "grouped_folds": file_manifest(folds_csv),
        "v35_pipeline_stack": file_manifest(PROJECT_ROOT / "output/v35/pipeline_stack.json"),
        "v35_case_diagnostics": file_manifest(
            PROJECT_ROOT / "output/v35/case_diagnostics.csv"
        ),
        "v42_fold0": file_manifest(
            PROJECT_ROOT / "output/v42/wbf_dropcontained_threshold0.80.json"
        ),
        "v42_fold1": file_manifest(
            PROJECT_ROOT / "output/v42/wbf_dropcontained_threshold0.80_fold1.json"
        ),
        "v42_fold2": file_manifest(
            PROJECT_ROOT / "output/v42/wbf_dropcontained_threshold0.80_fold2.json"
        ),
        "v42_allfold": file_manifest(
            PROJECT_ROOT / "output/v42/wbf_dropcontained_threshold0.80_allfold.json"
        ),
    }
    source_files = {
        "s0_audit": file_manifest(PROJECT_ROOT / "scratch/run_s0_audit.py"),
        "official_metrics": file_manifest(PROJECT_ROOT / "src/metrics_official.py"),
        "science_metrics": file_manifest(PROJECT_ROOT / "src/science_metrics.py"),
        "v35_rescore": file_manifest(PROJECT_ROOT / "scratch/run_pq_oof_rescore.py"),
        "v35_refiner_allfold": file_manifest(
            PROJECT_ROOT / "scratch/run_refiner_pq_allfold.py"
        ),
        "v42_wbf_dropcontained": file_manifest(
            PROJECT_ROOT / "scratch/run_wbf_dropcontained_threshold_probe.py"
        ),
    }
    missing_required = [
        name for name, entry in {**critical_inputs, **checkpoint_entries}.items()
        if not entry["exists"]
    ] + [name for name, entry in cache_entries.items() if not entry["complete"]]
    completion_status = (
        "complete-content-hash"
        if not missing_required and not args.quick_hash
        else "complete-inventory-hash" if not missing_required else "incomplete-missing-inputs"
    )
    provenance = {
        "schema_version": "s0-provenance-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "completion_status": completion_status,
        "missing_required": missing_required,
        "metric_version": METRIC_VERSION,
        "hash_policy": "critical files always content-hashed; cache directories follow hash_mode",
        "critical_inputs": critical_inputs,
        "source_files": source_files,
        "checkpoints": checkpoint_entries,
        "oof_caches": cache_entries,
    }
    (args.output / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if missing_required:
        print(f"S0-0 stopped: missing/incomplete inputs: {missing_required}", flush=True)
        return 3

    print("[S0-1] annotation/time/site audit", flush=True)
    with train_json.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    with folds_csv.open(encoding="utf-8-sig", newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
    report, observation_rows, annotation_rows = audit_annotations(coco, fold_rows)
    report["provenance_manifest"] = "output/s0/provenance_manifest.json"
    (args.output / "annotation_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output / "observation_strata.csv", observation_rows)
    write_csv(
        args.output / "time_block_assignment.csv",
        [
            {
                "file_name": row["file_name"],
                "fold": row["fold"],
                "site_code": row["site_code"],
                "year": row["year"],
                "time_block_role": row["time_block_role"],
            }
            for row in observation_rows
        ],
    )
    write_csv(args.output / "annotation_spine_audit.csv", annotation_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
