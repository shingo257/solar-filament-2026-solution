"""Synthetic split/checkpoint negatives; no ML imports or real model artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

from scratch.v73_split_provenance import (
    load_fold_mapping, load_reviewed_manifest, sha256_file,
    validate_checkpoint_binding, validate_checkpoint_metadata,
    observation_digest, validate_training_corpus_provenance,
)


class TestV73SplitProvenance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / "folds.csv"
        self.csv.write_text("file_name,fold\na.jpeg,0\nb.jpeg,2\n", encoding="utf-8")
        self.checkpoint = self.root / "synthetic.bin"
        self.checkpoint.write_bytes(b"synthetic model bytes, not a real checkpoint")
        self.manifest = {
            "schema_version": 1, "fold": 2, "folds_sha256": sha256_file(self.csv),
            "checkpoints": {"yolo": sha256_file(self.checkpoint)},
        }

    def mapping(self, names=("a.jpeg", "b.jpeg"), fold=2):
        return load_fold_mapping(self.csv, names, fold, sha256_file(self.csv))

    def test_valid_mapping(self):
        self.assertEqual(self.mapping(), {"a.jpeg": 0, "b.jpeg": 2})

    def test_duplicate_observation_rejected(self):
        self.csv.write_text("file_name,fold\na.jpeg,0\na.jpeg,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.mapping()

    def test_missing_and_extra_observations_rejected(self):
        for names in [("a.jpeg",), ("a.jpeg", "b.jpeg", "c.jpeg")]:
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "exactly"):
                self.mapping(names)

    def test_invalid_fold_values_rejected(self):
        for value in ["2.0", "-1", "3", "", "NaN"]:
            self.csv.write_text(f"file_name,fold\na.jpeg,0\nb.jpeg,{value}\n",
                                encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.mapping()

    def test_empty_partitions_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty evaluated"):
            self.mapping(fold=1)
        self.csv.write_text("file_name,fold\na.jpeg,2\nb.jpeg,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty training"):
            self.mapping()

    def test_csv_hash_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_fold_mapping(self.csv, ("a.jpeg", "b.jpeg"), 2, "0" * 64)

    def test_reviewed_manifest_hash_and_schema(self):
        path = self.root / "reviewed.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.assertEqual(load_reviewed_manifest(path, sha256_file(path), 2), self.manifest)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_reviewed_manifest(path, "0" * 64, 2)
        with self.assertRaisesRegex(ValueError, "schema/fold"):
            load_reviewed_manifest(path, sha256_file(path), 0)

    def test_metadata_fold_hash_missing_mismatches_rejected(self):
        valid = {"fold": 2, "folds_sha256": self.manifest["folds_sha256"]}
        validate_checkpoint_metadata(valid, 2, self.manifest["folds_sha256"])
        for checkpoint in [{}, {**valid, "fold": 0},
                           {**valid, "fold": True},
                           {**valid, "folds_sha256": "0" * 64},
                           {"fold": 2}]:
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                validate_checkpoint_metadata(checkpoint, 2, self.manifest["folds_sha256"])

    def test_legacy_checkpoint_needs_reviewed_binding(self):
        self.assertEqual(validate_checkpoint_binding(
            self.checkpoint, "yolo", self.manifest), sha256_file(self.checkpoint))
        for role in ["baseline", "ssl"]:
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "missing"):
                validate_checkpoint_binding(self.checkpoint, role, self.manifest, {})

    def test_manifest_yolo_and_baseline_bytes_are_enforced(self):
        self.manifest["checkpoints"]["baseline"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_checkpoint_binding(self.checkpoint, "baseline", self.manifest, {})
        self.checkpoint.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_checkpoint_binding(self.checkpoint, "yolo", self.manifest)

    def test_manifest_does_not_override_contradictory_metadata(self):
        with self.assertRaisesRegex(ValueError, "fold/split"):
            validate_checkpoint_binding(self.checkpoint, "yolo", self.manifest,
                                        {"fold": 0, "folds_sha256": self.manifest["folds_sha256"]})

    def test_ssl_protected_observations_must_match_current_run(self):
        test_dir = self.root / "test"
        test_dir.mkdir()
        test_name = "20110201000000Bh.jpeg"
        (test_dir / test_name).touch()
        coco = {"images": [{"file_name": "20110101000000Bh.jpeg"}]}
        provenance = {
            "guard_policy": "train-test-temporal-v2", "config_fingerprint": "a" * 16,
            "corpus_config_sha256": "b" * 64, "manifest_sha256": "c" * 64,
            "frames": 1, "exclusion_minutes": 360,
            "train_observations_sha256": observation_digest([coco["images"][0]["file_name"]]),
            "test_observations_sha256": observation_digest([test_name]),
        }
        self.assertEqual(validate_training_corpus_provenance(
            provenance, coco, test_dir), provenance)
        for key in ("train_observations_sha256", "test_observations_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "observations differ"):
                validate_training_corpus_provenance(
                    {**provenance, key: "0" * 64}, coco, test_dir)


if __name__ == "__main__":
    unittest.main()
