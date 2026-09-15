"""Public-only readiness audit for the S0-4 blinded review package.

This audit intentionally never reads ``private/``.  It checks that public
panels are decodable and non-empty, verifies their hashes, and quantifies how
the deliberately enriched review sample differs from the 707-observation
corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "output/s0/blinded_review_v1"
MODEL_IDENTITIES = ("raw", "v35", "v42", "v42_nested_verifier")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def categorical_distribution(
    selected: pd.Series,
    corpus: pd.Series,
) -> dict[str, Any]:
    selected_values = selected.fillna("missing").astype(str)
    corpus_values = corpus.fillna("missing").astype(str)
    categories = sorted(set(selected_values) | set(corpus_values))
    selected_counts = Counter(selected_values)
    corpus_counts = Counter(corpus_values)
    rows = {}
    for category in categories:
        selected_share = selected_counts[category] / len(selected_values)
        corpus_share = corpus_counts[category] / len(corpus_values)
        rows[category] = {
            "selected_count": int(selected_counts[category]),
            "selected_share": selected_share,
            "corpus_count": int(corpus_counts[category]),
            "corpus_share": corpus_share,
            "representation_ratio": (
                selected_share / corpus_share if corpus_share else None
            ),
        }
    return rows


def model_identity_leaks(page: str) -> list[str]:
    return [
        identity
        for identity in MODEL_IDENTITIES
        if re.search(
            rf"(?i)(?<![A-Za-z0-9_]){re.escape(identity)}(?![A-Za-z0-9_])",
            page,
        )
    ]


def audit_panels(
    package_root: Path,
    public: pd.DataFrame,
    generated_hashes: dict[str, str],
) -> dict[str, Any]:
    failures = []
    shapes = Counter()
    pixel_stds = []
    byte_sizes = []
    for row in public.itertuples(index=False):
        relative = Path(str(row.panel))
        path = package_root / relative
        if not path.is_file():
            failures.append({"case_id": row.case_id, "reason": "missing"})
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append({"case_id": row.case_id, "reason": "decode_failed"})
            continue
        shapes[str(tuple(int(value) for value in image.shape))] += 1
        pixel_std = float(image.std())
        pixel_stds.append(pixel_std)
        byte_sizes.append(path.stat().st_size)
        if pixel_std < 5.0 or int(image.max()) - int(image.min()) < 40:
            failures.append({"case_id": row.case_id, "reason": "near_blank"})
        expected = generated_hashes.get(str(relative))
        if expected is None:
            failures.append({"case_id": row.case_id, "reason": "hash_not_registered"})
        elif sha256_file(path) != expected:
            failures.append({"case_id": row.case_id, "reason": "hash_mismatch"})
    return {
        "expected": len(public),
        "decoded": len(pixel_stds),
        "shape_counts": dict(sorted(shapes.items())),
        "pixel_std_min": min(pixel_stds) if pixel_stds else None,
        "pixel_std_max": max(pixel_stds) if pixel_stds else None,
        "file_bytes_min": min(byte_sizes) if byte_sizes else None,
        "file_bytes_max": max(byte_sizes) if byte_sizes else None,
        "failures": failures,
    }


def build_readiness_audit(
    package_root: Path,
    public: pd.DataFrame,
    corpus: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    page = (package_root / "review.html").read_text(encoding="utf-8")
    panels = audit_panels(
        package_root,
        public,
        manifest["generated_files"],
    )
    unique_observations = int(public.file_name.nunique())
    distributions = {
        column: categorical_distribution(public[column], corpus[column])
        for column in ("fold", "site_code", "time_block_role")
    }
    selected_locked = distributions["time_block_role"].get("locked_evaluation", {})
    warnings = []
    if selected_locked.get("representation_ratio", 0) > 1.5:
        warnings.append(
            "locked_evaluation is intentionally enriched; do not interpret the 24-case "
            "preference rate as a corpus prevalence estimate"
        )
    if public.fold.value_counts().max() / len(public) > 0.45:
        warnings.append(
            "one fold exceeds 45% of the review sample; report fold counts with results"
        )
    corpus_sites = set(corpus.site_code.astype(str))
    selected_sites = set(public.site_code.astype(str))
    missing_sites = sorted(corpus_sites - selected_sites)
    if missing_sites:
        warnings.append(f"review sample omits corpus sites: {missing_sites}")
    leaks = model_identity_leaks(page)
    ready = (
        len(public) == 24
        and unique_observations == 24
        and panels["decoded"] == 24
        and not panels["failures"]
        and len(panels["shape_counts"]) == 1
        and not leaks
    )
    return {
        "package_id": manifest["package_id"],
        "status": (
            "ready_for_independent_review_with_sampling_caveats"
            if ready
            else "not_ready"
        ),
        "private_files_read": False,
        "selected_observations": len(public),
        "unique_observations": unique_observations,
        "corpus_observations": len(corpus),
        "year_range": [int(public.year.min()), int(public.year.max())],
        "distinct_years": int(public.year.nunique()),
        "html_model_identity_leaks": leaks,
        "panel_audit": panels,
        "public_distributions": distributions,
        "warnings": warnings,
        "interpretation_limit": (
            "The review set is hypothesis-enriched and supports error-mode adjudication, "
            "not population prevalence or an independent performance estimate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument(
        "--science-cases",
        type=Path,
        default=ROOT / "output/s0/science_case_metrics_v3.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output/s0/s0_blinded_review_readiness_audit_public.json",
    )
    args = parser.parse_args()
    public = pd.read_csv(args.package_root / "selection_manifest.csv")
    science = pd.read_csv(args.science_cases)
    corpus = (
        science.loc[science.model == "v35"]
        .drop_duplicates("file_name")
        [["file_name", "fold", "site_code", "year", "time_block_role"]]
    )
    manifest = json.loads(
        (args.package_root / "package_manifest.json").read_text(encoding="utf-8")
    )
    audit = build_readiness_audit(args.package_root, public, corpus, manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"].startswith("ready_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
