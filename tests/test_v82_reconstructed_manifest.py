"""A recovered printed summary must not masquerade as acceptance evidence."""

import hashlib
import json
import sys

from scratch import v82_manifest_from_printed


def test_printed_manifest_is_ineligible_even_with_a_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "mediseg_soft_refiner_fold0_s3407_resume.pt"
    checkpoint.write_bytes(b"checkpoint bytes; never loaded by this test")
    printed = {
        "arm": "soft", "fold": 0, "seed": 3407,
        "best_crop_dice": 0.5, "baseline_pq": 0.3,
        "by_threshold": {"0.45": 0.4},
    }
    monkeypatch.setattr(sys, "argv", [
        "v82_manifest_from_printed.py", "--artifact-root", str(tmp_path),
        "--printed-json", json.dumps(printed),
    ])
    assert v82_manifest_from_printed.main() == 0
    result = json.loads((tmp_path / "v82_soft_fold0_s3407_manifest.json").read_text(encoding="utf-8"))
    assert result["adoption_eligible"] is False
    assert result["result_kind"] == "reconstructed_printed_summary"
    assert result["provenance_limitations"]
    assert result["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result["oof_pq_by_threshold"] == printed["by_threshold"]
