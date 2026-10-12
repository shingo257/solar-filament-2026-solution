"""Colab カーネルが死んで manifest 未保存の場合のローカル再構築。

notebook の [6/7] 出力 JSON ({"fold":..,"arm":..,"seed":..,"baseline_pq":..,
"by_threshold":{...},"best_crop_dice":..}) を --printed-json に貼るだけで、
Drive 上の checkpoint から sha256 を計算し、CFG 定数と併せて
v82_{arm}_fold{f}_s{seed}_manifest.json を artifacts に生成する。

使い方:
    python scratch/v82_manifest_from_printed.py \
        --printed-json '{"fold":2,"arm":"soft","seed":3407,...}'
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ARTIFACT_ROOT = Path(r"G:/マイドライブ/solar_filament_v30/artifacts")

CFG = {
    "cluster_iou": 0.30,
    "repeat_factor": 1.5,
    "model": "MediSegUNet(3,1) — 現行 refiner と同一",
    "epochs": 14,
    "lr": 2e-4,
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--printed-json", required=True,
                        help="notebook [6/7] の print された JSON 文字列")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()

    printed = json.loads(args.printed_json)
    arm, fold, seed = printed["arm"], printed["fold"], printed["seed"]

    checkpoint = args.artifact_root / f"mediseg_{arm}_refiner_fold{fold}_s{seed}_resume.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 未着: {checkpoint}")

    manifest = {
        "experiment_id": "v82_soft_target_refiner",
        "arm": arm, "fold": fold, "seed": seed,
        **CFG,
        "best_crop_dice": printed["best_crop_dice"],
        "baseline_pq": printed["baseline_pq"],
        "oof_pq_by_threshold": printed["by_threshold"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_of(checkpoint),
        "gate": "soft - hard_record が 3fold・両seedで正、pooled CI がゼロを除外",
        "reconstructed_locally": True,
    }
    out = args.artifact_root / f"v82_{arm}_fold{fold}_s{seed}_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")
    print(json.dumps(manifest, ensure_ascii=True, indent=2)[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
