"""Migrate only independently bound legacy frames into a new, validated directory.

A trusted preexisting source record must bind the raw original manifest, ordered
decoded-frame hashes, and sampling config. Its SHA256 must be pinned independently;
creating it from the current legacy inputs does not establish provenance. Without
such a record, rebuild with build_gong_ssl_corpus.py instead of certifying old bytes.

Record schema: schema_version=1, manifest_sha256, corpus_config containing year,
days_per_month and frames_per_day, frames=[{file, frame_sha256}, ...] in original
manifest order. This command never modifies the legacy corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.build_gong_ssl_corpus import (
    GUARD_POLICY_VERSION,
    IMG_SIZE,
    corpus_config,
    parse_stamp,
)
from scratch.pretrain_unet_encoder_ssl import load_corpus


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("full lowercase SHA256 required")
    return value


def upgrade_corpus(*, corpus: Path, out: Path, train_json: Path, test_dir: Path,
                   source_record: Path, expected_source_record_sha256: str,
                   exclusion_minutes: int = 360, batch_size: int = 250) -> dict:
    """Verify trusted source bindings before staging a separate v2 corpus."""
    corpus, out = corpus.resolve(), out.resolve()
    if (corpus == out or corpus in out.parents or out in corpus.parents
            or out.exists() or not out.parent.is_dir()):
        raise ValueError("output must be a new directory separate from the source tree")
    if (type(batch_size) is not int or batch_size <= 0
            or type(exclusion_minutes) is not int or exclusion_minutes < 360):
        raise ValueError("positive batch size and exclusion window >=360 required")
    record_bytes = source_record.read_bytes()
    if _digest(record_bytes) != _sha256(expected_source_record_sha256):
        raise ValueError("trusted source record SHA256 mismatch")
    record = json.loads(record_bytes)
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("invalid trusted source record schema")
    config = record.get("corpus_config")
    if not isinstance(config, dict):
        raise TypeError("trusted source sampling config required")
    limits = {"year": 9999, "days_per_month": 31, "frames_per_day": 1440}
    for key, maximum in limits.items():
        value = config.get(key)
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError(f"invalid trusted sampling config: {key}")
    cfg = SimpleNamespace(**{key: config[key] for key in limits},
                          exclusion_minutes=exclusion_minutes, batch_size=batch_size)

    manifest_bytes = (corpus / "manifest.jsonl").read_bytes()
    if _digest(manifest_bytes) != _sha256(record.get("manifest_sha256")):
        raise ValueError("original manifest SHA256 mismatch")
    rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines()]
    bindings = record.get("frames")
    if not rows or not isinstance(bindings, list) or len(bindings) != len(rows):
        raise ValueError("ordered trusted decoded-frame bindings required")
    arrays = []
    for path in sorted(corpus.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            candidates = [data[key] for key in data.files if data[key].ndim >= 1]
            if len(candidates) != 1:
                raise ValueError("each legacy batch must contain exactly one frame array")
            arr = candidates[0]
            if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[1:] != (IMG_SIZE, IMG_SIZE):
                raise ValueError("invalid legacy frame shape/type")
            arrays.append(arr)
        print(f"verified source batch {path.name}", flush=True)
    if not arrays:
        raise ValueError("legacy batches missing")
    frames = np.concatenate(arrays)
    if len(frames) != len(rows):
        raise ValueError("legacy manifest/frame count mismatch")
    for frame, row, binding in zip(frames, rows, bindings):
        if (not isinstance(row, dict) or not isinstance(binding, dict)
                or row.get("file") != binding.get("file")):
            raise ValueError("trusted frame ordering differs from source manifest")
        expected = _sha256(binding.get("frame_sha256"))
        if _digest(frame.tobytes()) != expected:
            raise ValueError("trusted decoded-frame SHA256 mismatch")
        if "frame_sha256" in row and row["frame_sha256"] != expected:
            raise ValueError("original manifest frame SHA256 contradicts trusted record")
        row["frame_sha256"] = expected

    coco = json.loads(train_json.read_text(encoding="utf-8"))
    images = coco.get("images") if isinstance(coco, dict) else None
    if not isinstance(images, list) or not images:
        raise ValueError("actual train observations required")
    train_stamps = {parse_stamp(image["file_name"].replace(".jpeg", ".fits.fz"))
                    for image in images}
    test_stamps = {parse_stamp(path.name.replace(".jpeg", ".fits.fz"))
                   for path in test_dir.iterdir()
                   if path.is_file() and path.suffix == ".jpeg"}
    if not train_stamps or not test_stamps:
        raise ValueError("nonempty actual train and test anchors required")
    keep, dropped = [], []
    for index, row in enumerate(rows):
        stamp = parse_stamp(row["file"])
        if stamp.year != cfg.year:
            raise ValueError("manifest observation differs from trusted year")
        if any(abs(stamp - anchor) <= timedelta(minutes=exclusion_minutes)
               for anchor in train_stamps | test_stamps):
            dropped.append(row["file"])
        else:
            keep.append(index)
    if not keep:
        raise ValueError("no frames remain after observation exclusion")
    kept_frames, kept_rows = frames[keep], [rows[index] for index in keep]
    summary = {"kept": len(keep), "skipped_observation_window": len(dropped),
               "img_size": IMG_SIZE, "dry_run": False,
               "exclusion_minutes": exclusion_minutes,
               "guard_policy": GUARD_POLICY_VERSION,
               "migration_source_record_sha256": expected_source_record_sha256,
               "migration_manifest_sha256": record["manifest_sha256"]}
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-upgrade-", dir=out.parent))
    try:
        for index, start in enumerate(range(0, len(kept_frames), batch_size)):
            np.savez_compressed(staging / f"batch_{index:03d}.npz",
                                kept_frames[start:start + batch_size],
                                batch_size=batch_size)
            print(f"staged batch {index} ({min(start + batch_size, len(keep))}/{len(keep)})",
                  flush=True)
        (staging / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in kept_rows), encoding="utf-8")
        for name, value in (("corpus_config.json", corpus_config(test_stamps, train_stamps, cfg)),
                            ("corpus_summary.json", summary),
                            ("upgrade_dropped_files.json", dropped)):
            (staging / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        load_corpus(staging, train_json, test_dir)
        if out.exists():
            raise ValueError("output appeared during migration")
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("corpus", "out", "train-json", "test-dir", "source-record"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--expected-source-record-sha256", required=True)
    parser.add_argument("--exclusion-minutes", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=250)
    summary = upgrade_corpus(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
