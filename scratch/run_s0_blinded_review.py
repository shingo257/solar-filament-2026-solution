"""Build the S0-4 blinded visual-review package from frozen CPU caches.

The package contains 24 unique observations by default:

* matched-spine trade-off cases,
* small/dense/limb rescue cases,
* late-time candidate-rejection cases.

The review page never contains model identities.  Model mapping and selection
scores live under ``private/`` and must remain unopened until review export.
"""


from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from scratch.run_refiner_pq_allfold import (  # noqa: E402
    FOLDS_CSV,
    TRAIN_JSON,
)
from scratch.run_s0_baseline_rescore import (  # noqa: E402
    preferred_image_root,
)


try:
    from .run_s0_blinded_review_impl import (
        COMPARISONS,
        DEFAULT_SEED,
        PACKAGE_ID,
        PROTOCOL_VERSION,
        apply_blinding,
        balanced_assignments,
        build_review_html,
        focus_box,
        load_references,
        protocol_markdown,
        render_case_panel,
        select_cases,
        sha256_file,
        top_unique,
    )
except ImportError:  # Direct `python scratch/...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_s0_blinded_review_impl import (
        COMPARISONS,
        DEFAULT_SEED,
        PACKAGE_ID,
        PROTOCOL_VERSION,
        apply_blinding,
        balanced_assignments,
        build_review_html,
        focus_box,
        load_references,
        protocol_markdown,
        render_case_panel,
        select_cases,
        sha256_file,
        top_unique,
    )



__all__ = [
    "COMPARISONS",
    "DEFAULT_SEED",
    "PACKAGE_ID",
    "PROTOCOL_VERSION",
    "apply_blinding",
    "balanced_assignments",
    "build_review_html",
    "focus_box",
    "load_references",
    "protocol_markdown",
    "render_case_panel",
    "select_cases",
    "sha256_file",
    "top_unique",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-comparison", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "output/s0/blinded_review_v1"
    )
    args = parser.parse_args()
    if args.per_comparison < 3:
        raise ValueError("--per-comparison must be at least 3")
    assets = args.out / "assets"
    private = args.out / "private"
    assets.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    science_path = ROOT / "output/s0/science_case_metrics_v3.csv"
    official_path = ROOT / "output/s0/v42_verifier_official_case_metrics_0-1-2.csv"
    science = pd.read_csv(science_path)
    official = pd.read_csv(official_path)
    selected = select_cases(science, official, args.per_comparison)
    blinded, key = apply_blinding(selected, args.seed)

    with TRAIN_JSON.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    images, annotations, file_to_ids = load_references(coco)
    image_root = preferred_image_root()
    print(f"images: {image_root}")
    panel_rows = []
    for position, row in enumerate(blinded.itertuples(index=False), 1):
        output_path = assets / f"{row.case_id}.png"
        panel_rows.append(
            render_case_panel(
                row, images, annotations, file_to_ids, image_root, output_path
            )
        )
        print(f"panel {position}/{len(blinded)}: {row.case_id}", flush=True)

    public = blinded[
        [
            "case_id",
            "file_name",
            "fold",
            "site_code",
            "year",
            "time_block_role",
        ]
    ].copy()
    public["panel"] = public.case_id.map(lambda value: f"assets/{value}.png")
    public_path = args.out / "selection_manifest.csv"
    public.to_csv(public_path, index=False)
    audit_path = private / "selection_audit.csv"
    blinded.to_csv(audit_path, index=False)
    key_path = private / "blind_key.json"
    key_path.write_text(
        json.dumps(
            {
                "package_id": PACKAGE_ID,
                "warning": "Open only after exporting completed blinded responses.",
                "seed": args.seed,
                "cases": key,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    review_cases = [
        {"case_id": row["case_id"], "image": Path(row["image"]).name}
        for row in panel_rows
    ]
    html_path = args.out / "review.html"
    html_path.write_text(build_review_html(review_cases), encoding="utf-8")
    protocol_path = args.out / "REVIEW_PROTOCOL.md"
    protocol_path.write_text(protocol_markdown(), encoding="utf-8")

    generated = [
        public_path,
        audit_path,
        key_path,
        html_path,
        protocol_path,
        *sorted(assets.glob("case_*.png")),
    ]
    comparison_counts = {
        key_name: int((blinded.comparison == key_name).sum())
        for key_name in COMPARISONS
    }
    position_balance = {}
    for comparison, group in blinded.groupby("comparison"):
        first_model = COMPARISONS[comparison][0]
        position_balance[comparison] = {
            "first_model_as_a": int((group.output_a == first_model).sum()),
            "first_model_as_b": int((group.output_b == first_model).sum()),
        }
    manifest = {
        "package_id": PACKAGE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "awaiting independent blinded human review",
        "selection_seed": args.seed,
        "observations": len(blinded),
        "unique_observations": int(blinded.file_name.nunique()),
        "comparison_counts": comparison_counts,
        "position_balance": position_balance,
        "blind_integrity": {
            "review_html_contains_model_mapping": False,
            "private_key": str(key_path),
            "instruction": "do not open private/ until response JSON is exported",
        },
        "source_files": {
            str(science_path): sha256_file(science_path),
            str(official_path): sha256_file(official_path),
            str(TRAIN_JSON): sha256_file(TRAIN_JSON),
            str(FOLDS_CSV): sha256_file(FOLDS_CSV),
        },
        "generated_files": {
            str(path.relative_to(args.out)): sha256_file(path) for path in generated
        },
    }
    manifest_path = args.out / "package_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": manifest["status"],
        "observations": manifest["observations"],
        "comparison_counts": comparison_counts,
        "position_balance": position_balance,
        "review_html": str(html_path),
        "private_key": str(key_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
