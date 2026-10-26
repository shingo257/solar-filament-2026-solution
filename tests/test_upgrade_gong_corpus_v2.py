"""Fail-closed, non-destructive migration of independently bound legacy frames."""
import hashlib
import json

import numpy as np
import pytest

from scratch import upgrade_gong_corpus_v2 as upgrade
from scratch.build_gong_ssl_corpus import BASE, IMG_SIZE


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    frames = np.stack([np.full((IMG_SIZE, IMG_SIZE), i, np.uint8) for i in (1, 2)])
    names = ["20110101000000Bh.fits.fz", "20110110000000Bh.fits.fz"]
    rows = [{"file": name, "utc": f"2011-01-{day}T00:00:00+00:00",
             "url": f"{BASE}/201101/201101{day}/{name}",
             "sha256": "a" * 64, "bytes": 10}
            for name, day in zip(names, ("01", "10"))]
    manifest = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (source / "manifest.jsonl").write_bytes(manifest)
    np.savez_compressed(source / "batch_000.npz", frames)
    train = tmp_path / "train.json"
    train.write_text(json.dumps({"images": [{"file_name": "20110101000000Bh.jpeg"}]}))
    test = tmp_path / "test_images"
    test.mkdir()
    (test / "20110201000000Bh.jpeg").touch()
    record = {"schema_version": 1, "manifest_sha256": digest(manifest),
              "corpus_config": {"year": 2011, "days_per_month": 4, "frames_per_day": 12},
              "frames": [{"file": name, "frame_sha256": digest(frame.tobytes())}
                         for name, frame in zip(names, frames)]}
    record_path = tmp_path / "trusted.json"
    record_path.write_text(json.dumps(record))
    return {"corpus": source, "out": tmp_path / "v2", "train_json": train,
            "test_dir": test, "source_record": record_path,
            "expected_source_record_sha256": digest(record_path.read_bytes()),
            "exclusion_minutes": 360, "batch_size": 2}


def snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir()}


def run(inputs):
    return upgrade.upgrade_corpus(**inputs)


def test_missing_trusted_record_is_rejected_without_source_changes(inputs):
    before = snapshot(inputs["corpus"])
    inputs["source_record"] = inputs["source_record"].with_name("missing.json")
    with pytest.raises((ValueError, OSError)):
        run(inputs)
    assert snapshot(inputs["corpus"]) == before
    assert not inputs["out"].exists()


@pytest.mark.parametrize("mutation", ["corrupt", "reorder", "manifest", "record_pin"])
def test_unbound_input_is_rejected(inputs, mutation):
    source = inputs["corpus"]
    if mutation in ("corrupt", "reorder"):
        with np.load(source / "batch_000.npz") as data:
            arr = data["arr_0"].copy()
        if mutation == "corrupt":
            arr[0, 0, 0] = 99
        else:
            arr = arr[::-1]
        np.savez_compressed(source / "batch_000.npz", arr)
    elif mutation == "manifest":
        with (source / "manifest.jsonl").open("ab") as handle:
            handle.write(b"\n")
    else:
        inputs["expected_source_record_sha256"] = "b" * 64
    before = snapshot(source)
    with pytest.raises(ValueError):
        run(inputs)
    assert snapshot(source) == before
    assert not inputs["out"].exists()


def test_valid_migration_filters_overlap_and_preserves_source(inputs):
    before = snapshot(inputs["corpus"])
    run(inputs)
    assert snapshot(inputs["corpus"]) == before
    out = inputs["out"]
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert [r["file"] for r in rows] == ["20110110000000Bh.fits.fz"]
    assert json.loads((out / "upgrade_dropped_files.json").read_text()) == [
        "20110101000000Bh.fits.fz"]
    with np.load(out / "batch_000.npz") as data:
        assert int(data["batch_size"]) == 2
        assert np.all(data["arr_0"] == 2)


def test_failed_write_cleans_only_staging(inputs, monkeypatch):
    before = snapshot(inputs["corpus"])
    unrelated = inputs["out"].parent / ".v2-unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("keep")

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(upgrade.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="full disk"):
        run(inputs)
    assert snapshot(inputs["corpus"]) == before
    assert not inputs["out"].exists()
    assert (unrelated / "keep").read_text() == "keep"
    assert not list(inputs["out"].parent.glob(".v2-upgrade-*"))


@pytest.mark.parametrize("mode", ["source", "inside_source", "ancestor", "exists"])
def test_unsafe_output_is_rejected(inputs, mode):
    source = inputs["corpus"]
    inputs["out"] = {"source": source, "inside_source": source / "new",
                     "ancestor": source.parent, "exists": source.parent / "existing"}[mode]
    if mode == "exists":
        inputs["out"].mkdir()
    before = snapshot(source)
    with pytest.raises(ValueError):
        run(inputs)
    assert snapshot(source) == before


@pytest.mark.parametrize("key,value", [("batch_size", 0), ("exclusion_minutes", 359)])
def test_invalid_output_parameters_fail(inputs, key, value):
    inputs[key] = value
    with pytest.raises(ValueError):
        run(inputs)
    assert not inputs["out"].exists()


def test_empty_test_anchors_fail(inputs):
    for path in inputs["test_dir"].iterdir():
        path.unlink()
    with pytest.raises(ValueError):
        run(inputs)
    assert not inputs["out"].exists()


def test_invalid_record_config_is_not_guessed(inputs):
    record = json.loads(inputs["source_record"].read_text())
    record["corpus_config"].pop("year")
    inputs["source_record"].write_text(json.dumps(record))
    inputs["expected_source_record_sha256"] = digest(inputs["source_record"].read_bytes())
    with pytest.raises(ValueError):
        run(inputs)
    assert not inputs["out"].exists()
