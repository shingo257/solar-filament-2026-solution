"""V35 最終提出CSVをローカル検証器で全件確認し、過去提出と比較する。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v32_overlap_cluster import load_submission, validate_submission  # noqa: E402

OUT = ROOT / "output" / "v35"
CSV = OUT / "submission_v35_final_pq_aligned.csv"
AUDIT = OUT / "v35_final_pq_aligned_audit.csv"
EXPECTED_SHA256 = "10cd5f79000861b0c2fcdc103ad085efeb6f44589590bbc2c353b24c01402162"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    digest = sha256_of(CSV)
    assert digest == EXPECTED_SHA256, f"SHA256 mismatch: {digest}"

    frame = load_submission(CSV)
    report = validate_submission(frame)
    stems = {value.rsplit("_", 1)[0] for value in frame["filament_id"]}

    audit = pd.read_csv(AUDIT)
    empty = audit.loc[audit["submitted"] == 0, "file_name"].tolist()
    zero_proposal = audit.loc[audit["proposals"] == 0, "file_name"].tolist()

    previous = {}
    for name, path in (
        ("V30 overlapping", Path(r"G:\マイドライブ\solar_filament_v30\artifacts\submission_v30_prize_instance_ensemble.csv")),
        ("V30b disjoint", Path(r"G:\マイドライブ\solar_filament_v30\artifacts\submission_v30b_disjoint_instance_ensemble.csv")),
        ("V32b", ROOT / "output" / "v32" / "submission_v32b_containment099_dice050.csv"),
    ):
        if path.exists():
            previous_frame = pd.read_csv(path, keep_default_na=False)
            previous[name] = {
                "rows": len(previous_frame),
                "images": previous_frame["filament_id"].str.rsplit("_", n=1).str[0].nunique(),
            }

    v30_audit_path = Path(r"G:\マイドライブ\solar_filament_v30\artifacts\test_instance_audit.csv")
    v30_empty: list[str] = []
    if v30_audit_path.exists():
        v30_audit = pd.read_csv(v30_audit_path)
        if "submitted" in v30_audit.columns:
            v30_empty = v30_audit.loc[v30_audit["submitted"] == 0, "file_name"].tolist()

    summary = {
        "sha256": digest,
        "validator": report,
        "inst_per_predicted_image": round(report["rows"] / max(report["images"], 1), 3),
        "empty_images": empty,
        "zero_proposal_images": zero_proposal,
        "v30_empty_images": v30_empty,
        "previous": previous,
        "audit_rows": int(len(audit)),
        "audit_submitted_sum": int(audit["submitted"].sum()),
    }
    (OUT / "v35_final_pq_aligned_local_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
