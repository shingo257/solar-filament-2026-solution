"""Fail-closed v73 split and checkpoint identity checks (standard library only)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a full lowercase SHA256")
    return value


def load_reviewed_manifest(path: Path, expected_sha256: str, fold: int) -> dict:
    """The expected hash must come from independent review, not the file itself."""
    if sha256_file(path) != require_sha256(expected_sha256):
        raise ValueError("reviewed provenance manifest SHA256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or type(manifest.get("fold")) is not int or manifest["fold"] != fold):
        raise ValueError("provenance manifest schema/fold mismatch")
    require_sha256(manifest.get("folds_sha256"))
    bindings = manifest.get("checkpoints")
    if not isinstance(bindings, dict) or "yolo" not in bindings:
        raise ValueError("reviewed YOLO checkpoint binding is required")
    for role, digest in bindings.items():
        if not isinstance(role, str) or not role:
            raise ValueError("checkpoint role must be non-empty")
        require_sha256(digest)
    return manifest


def load_fold_mapping(path: Path, file_names, fold: int,
                      expected_sha256: str) -> dict[str, int]:
    if type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("invalid requested fold")
    if sha256_file(path) != require_sha256(expected_sha256):
        raise ValueError("fold CSV SHA256 mismatch")
    mapping = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not {"file_name", "fold"} <= set(reader.fieldnames)):
            raise ValueError("invalid fold CSV columns")
        for row in reader:
            name, value = row["file_name"], row["fold"]
            if not name or name in mapping or value not in ("0", "1", "2"):
                raise ValueError("duplicate observation or invalid fold assignment")
            mapping[name] = int(value)
    if set(mapping) != set(file_names):
        raise ValueError("fold mapping must cover exactly the train observations")
    if not any(value == fold for value in mapping.values()):
        raise ValueError("empty evaluated fold")
    if not any(value != fold for value in mapping.values()):
        raise ValueError("empty training partition")
    return mapping


def validate_checkpoint_metadata(checkpoint: dict, fold: int,
                                 folds_sha256: str) -> None:
    if (not isinstance(checkpoint, dict)
            or type(checkpoint.get("fold")) is not int
            or checkpoint["fold"] != fold
            or checkpoint.get("folds_sha256") != require_sha256(folds_sha256)):
        raise ValueError("checkpoint fold/split provenance mismatch or missing")


def validate_checkpoint_binding(path: Path, role: str, manifest: dict,
                                checkpoint: dict | None = None) -> str:
    """Legacy files require an explicit reviewed byte-to-fold/split binding."""
    digest = sha256_file(path)
    expected = manifest["checkpoints"].get(role)
    if expected is not None and digest != expected:
        raise ValueError(f"checkpoint SHA256 mismatch: {role}")
    if checkpoint is None:
        if expected is None:
            raise ValueError(f"reviewed checkpoint binding missing: {role}")
    elif "fold" in checkpoint or "folds_sha256" in checkpoint:
        validate_checkpoint_metadata(checkpoint, manifest["fold"],
                                     manifest["folds_sha256"])
    elif expected is None:
        raise ValueError(f"checkpoint provenance missing: {role}")
    return digest

def validate_corpus_provenance(provenance) -> dict:
    if (not isinstance(provenance, dict)
            or provenance.get("guard_policy") != "train-test-temporal-v2"
            or not re.fullmatch(r"[0-9a-f]{16}", str(provenance.get("config_fingerprint", "")))
            or type(provenance.get("frames")) is not int or provenance["frames"] <= 0
            or type(provenance.get("exclusion_minutes")) is not int
            or provenance["exclusion_minutes"] < 360):
        raise ValueError("verified v2 corpus provenance required")
    for key in ("corpus_config_sha256", "manifest_sha256",
                "train_observations_sha256", "test_observations_sha256"):
        require_sha256(provenance.get(key))
    return dict(provenance)

def observation_digest(names) -> str:
    stamps = set()
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(
                r"[0-9]{14}[BCLMTUbclmtu]h\.jpeg", name):
            raise ValueError("invalid observation timestamp")
        stamps.add(datetime.strptime(name[:14], "%Y%m%d%H%M%S")
                   .replace(tzinfo=timezone.utc).isoformat())
    if not stamps:
        raise ValueError("missing protected observations")
    return hashlib.sha256(json.dumps(sorted(stamps), separators=(",", ":")).encode()).hexdigest()


def validate_training_corpus_provenance(provenance, coco: dict, test_dir: Path) -> dict:
    provenance = validate_corpus_provenance(provenance)
    images = coco.get("images") if isinstance(coco, dict) else None
    if not isinstance(images, list) or not images:
        raise ValueError("missing actual train observations")
    train_digest = observation_digest(image.get("file_name") for image in images)
    test_digest = observation_digest(
        path.name for path in test_dir.iterdir() if path.is_file() and path.suffix == ".jpeg")
    if (provenance["train_observations_sha256"] != train_digest
            or provenance["test_observations_sha256"] != test_digest):
        raise ValueError("SSL corpus protected observations differ from this run")
    return provenance
