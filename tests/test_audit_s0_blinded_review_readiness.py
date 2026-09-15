from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from scratch.audit_s0_blinded_review_readiness import (
    audit_panels,
    build_readiness_audit,
    categorical_distribution,
    model_identity_leaks,
    sha256_file,
)


def test_categorical_distribution_reports_enrichment_ratio():
    result = categorical_distribution(
        pd.Series(["locked", "locked", "development"]),
        pd.Series(["locked", "development", "development", "development"]),
    )

    assert result["locked"]["selected_count"] == 2
    assert result["locked"]["corpus_count"] == 1
    assert result["locked"]["representation_ratio"] == 8 / 3


def test_model_identity_leaks_uses_token_boundaries():
    assert model_identity_leaks("reviewer and raw output") == ["raw"]
    assert model_identity_leaks("draw and reviewer") == []


def test_audit_panels_decodes_and_verifies_hash(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    image = np.zeros((100, 150, 3), dtype=np.uint8)
    image[:, :75] = (255, 80, 10)
    path = assets / "case_001.png"
    assert cv2.imwrite(str(path), image)
    public = pd.DataFrame({
        "case_id": ["case_001"],
        "panel": ["assets/case_001.png"],
    })

    result = audit_panels(
        tmp_path,
        public,
        {str(tmp_path.joinpath("assets/case_001.png").relative_to(tmp_path)): sha256_file(path)},
    )

    assert result["decoded"] == 1
    assert result["shape_counts"] == {"(100, 150, 3)": 1}
    assert result["failures"] == []


def test_build_readiness_audit_flags_intentional_locked_enrichment(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    public_rows = []
    hashes = {}
    sites = ["Bh", "Ch", "Lh", "Mh", "Th", "Uh"]
    for index in range(24):
        case_id = f"case_{index + 1:03d}"
        panel = f"assets/{case_id}.png"
        image = np.zeros((100, 150, 3), dtype=np.uint8)
        image[:, : 20 + index] = (255, 100, 20)
        path = tmp_path / panel
        assert cv2.imwrite(str(path), image)
        hashes[str(path.relative_to(tmp_path))] = sha256_file(path)
        public_rows.append({
            "case_id": case_id,
            "file_name": f"observation-{index}.png",
            "panel": panel,
            "fold": index % 3,
            "site_code": sites[index % len(sites)],
            "year": 2011 + index % 12,
            "time_block_role": (
                "locked_evaluation" if index < 14 else "development"
            ),
        })
    (tmp_path / "review.html").write_text("reviewer only", encoding="utf-8")
    public = pd.DataFrame(public_rows)
    corpus = pd.DataFrame({
        "file_name": [f"corpus-{index}.png" for index in range(100)],
        "fold": [index % 3 for index in range(100)],
        "site_code": [sites[index % len(sites)] for index in range(100)],
        "year": [2011 + index % 12 for index in range(100)],
        "time_block_role": [
            "locked_evaluation" if index < 20 else "development"
            for index in range(100)
        ],
    })

    result = build_readiness_audit(
        tmp_path,
        public,
        corpus,
        {"package_id": "test-package", "generated_files": hashes},
    )

    assert result["status"] == "ready_for_independent_review_with_sampling_caveats"
    assert result["private_files_read"] is False
    assert result["html_model_identity_leaks"] == []
    assert any("intentionally enriched" in warning for warning in result["warnings"])
