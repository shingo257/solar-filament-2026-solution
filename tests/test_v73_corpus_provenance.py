"""Reject legacy, inconsistent, or temporally overlapping SSL corpora."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scratch.build_gong_ssl_corpus import corpus_config_fingerprint, parse_stamp
from scratch.pretrain_unet_encoder_ssl import load_corpus


def corpus(tmp_path):
    train = tmp_path / "train.json"
    train.write_text(json.dumps({"images": [{"file_name": "20110101000000Bh.jpeg"}]}))
    test = tmp_path / "test"
    test.mkdir()
    (test / "20110102000000Ch.jpeg").touch()
    out = tmp_path / "corpus"
    out.mkdir()
    config = dict(guard_policy="train-test-temporal-v2", year=2011,
                  days_per_month=4, frames_per_day=12, exclusion_minutes=360,
                  batch_size=250, img_size=512,
                  train_stamps=[parse_stamp("20110101000000Bh.fits.fz").isoformat()],
                  test_stamps=[parse_stamp("20110102000000Ch.fits.fz").isoformat()])
    config["fingerprint"] = corpus_config_fingerprint(
        {parse_stamp("20110102000000Ch.fits.fz")},
        {parse_stamp("20110101000000Bh.fits.fz")}, SimpleNamespace(**config))
    (out / "corpus_config.json").write_text(json.dumps(config))
    (out / "corpus_summary.json").write_text(json.dumps(dict(
        guard_policy=config["guard_policy"], dry_run=False, kept=1,
        exclusion_minutes=360, img_size=512)))
    frame = np.zeros((512, 512), np.uint8)
    np.savez(out / "batch_000.npz", frame[None], batch_size=250)
    name = "20110201000000Bh.fits.fz"
    row = dict(file=name, url=f"https://gong2.nso.edu/ftp/HA/haf/201102/20110201/{name}",
               utc=parse_stamp(name).isoformat(), sha256="a" * 64, bytes=1,
               frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
    (out / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    return out, train, test, config, row


def test_valid_corpus_returns_bound_provenance(tmp_path):
    out, train, test, _, _ = corpus(tmp_path)
    frames, provenance = load_corpus(out, train, test)
    assert frames.shape == (1, 512, 512)
    assert provenance["guard_policy"] == "train-test-temporal-v2"
    assert provenance["frames"] == 1
    for key, name in (("train", "20110101000000Bh.fits.fz"),
                      ("test", "20110102000000Ch.fits.fz")):
        expected = hashlib.sha256(json.dumps(
            [parse_stamp(name).isoformat()], separators=(",", ":")).encode()).hexdigest()
        assert provenance[f"{key}_observations_sha256"] == expected


@pytest.mark.parametrize("invalid", ["legacy", "dry_run", "short_window", "count",
    "frame_hash", "duplicate", "anchor", "boundary", "wrong_url", "source_hash"])
def test_invalid_corpus_is_rejected(tmp_path, invalid):
    out, train, test, _, row = corpus(tmp_path)
    if invalid == "legacy":
        (out / "corpus_config.json").write_text('{"fingerprint":"legacy"}')
    elif invalid in ("dry_run", "short_window", "count"):
        path = out / "corpus_summary.json"
        summary = json.loads(path.read_text())
        summary[{"dry_run": "dry_run", "short_window": "exclusion_minutes", "count": "kept"}[invalid]] = {
            "dry_run": True, "short_window": 0, "count": 2}[invalid]
        path.write_text(json.dumps(summary))
    elif invalid == "anchor":
        train.write_text(json.dumps({"images": [{"file_name": "20110103000000Bh.jpeg"}]}))
    else:
        if invalid == "frame_hash":
            row["frame_sha256"] = "b" * 64
        elif invalid == "boundary":
            row["file"] = "20110101060000Ch.fits.fz"
            row["utc"] = parse_stamp(row["file"]).isoformat()
            row["url"] = f"https://gong2.nso.edu/ftp/HA/haf/201101/20110101/{row['file']}"
        elif invalid == "wrong_url":
            row["url"] = "https://example.com/frame"
        elif invalid == "source_hash":
            row["sha256"] = "invalid"
        (out / "manifest.jsonl").write_text((json.dumps(row) + "\n") * (2 if invalid == "duplicate" else 1))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_corpus(out, train, test)
