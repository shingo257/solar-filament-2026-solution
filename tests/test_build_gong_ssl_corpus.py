"""build_gong_ssl_corpus.prepare_output_dir の resume 契約テスト。

中断/完了済みの出力 dir への追い越し実行で batch npz と manifest が
不整合にならないことを保証する (Codex P1 指摘への回帰)。
"""

import json

import numpy as np
import pytest

from scratch.build_gong_ssl_corpus import (
    batch_frame_count, corpus_config_fingerprint, prepare_output_dir)
from scratch.pretrain_gong_ssl import load_corpus


def _row(name):
    return json.dumps({"file": name, "url": "u", "utc": "t",
                       "sha256": "x", "bytes": 1})


def _make_corpus(out, n_batches=2, batch_size=3, manifest_rows=5):
    """2 complete batch (6行分のはず) + manifest 5行 = 中断状態を模擬。"""
    for i in range(n_batches):
        np.savez_compressed(out / f"batch_{i:03d}.npz",
                            np.zeros((batch_size, 4, 4), np.uint8),
                            batch_size=batch_size)
    rows = [_row(f"f{i:014d}bh.fits.fz") for i in range(manifest_rows)]
    (out / "manifest.jsonl").write_text(
        "".join(r + "\n" for r in rows), encoding="utf-8")
    return rows


def test_clean_dir_passes_without_resume(tmp_path):
    done, batch_idx, kept = prepare_output_dir(tmp_path, 250, resume=False)
    assert done == set() and batch_idx == 0 and kept == 0


def test_nonempty_dir_refuses_without_resume(tmp_path):
    _make_corpus(tmp_path)
    with pytest.raises(SystemExit):
        prepare_output_dir(tmp_path, 250, resume=False)


def test_resume_truncates_manifest_to_batch_boundary(tmp_path):
    # 中断の典型形: manifest 5行書いた後 batch 未保存で停止 → batch 1個
    # (3フレーム分) のみ存在。manifest は npz 保存済みの3行へ切り詰める。
    out = tmp_path
    np.savez_compressed(out / "batch_000.npz",
                        np.zeros((3, 4, 4), np.uint8), batch_size=3)
    rows = [_row(f"f{i:014d}bh.fits.fz") for i in range(5)]
    (out / "manifest.jsonl").write_text(
        "".join(r + "\n" for r in rows), encoding="utf-8")

    done, batch_idx, kept = prepare_output_dir(out, 3, resume=True)

    # 1 batch = 3行だけ残し、4-5行目 (npz 未保存分) は切り捨て
    assert batch_idx == 1 and kept == 3
    assert len(done) == 3
    remaining = (out / "manifest.jsonl").read_text().splitlines()
    assert remaining == rows[:3]


def test_resume_accepts_completed_corpus_with_partial_last_batch(tmp_path):
    # 完了 run: 2 batch (3 + 2 = 5フレーム、最終は部分) + manifest 5行。
    # 容量ではなく実フレーム数で検証し、retry が拒否されないことを保証。
    np.savez_compressed(tmp_path / "batch_000.npz",
                        np.zeros((3, 4, 4), np.uint8), batch_size=3)
    np.savez_compressed(tmp_path / "batch_001.npz",
                        np.zeros((2, 4, 4), np.uint8), batch_size=3)
    rows = [_row(f"f{i:014d}bh.fits.fz") for i in range(5)]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(r + "\n" for r in rows), encoding="utf-8")

    done, batch_idx, kept = prepare_output_dir(tmp_path, 3, resume=True)

    assert (batch_idx, kept) == (2, 5)
    assert len(done) == 5


def test_resume_rejects_manifest_below_batch_capacity(tmp_path):
    # 2 batch (容量6行) なのに manifest 5行 = npz が manifest 未記載の
    # フレームを保持する破損状態 → resume 不可
    _make_corpus(tmp_path, n_batches=2, batch_size=3, manifest_rows=5)
    with pytest.raises(SystemExit):
        prepare_output_dir(tmp_path, 3, resume=True)


def test_resume_rejects_batch_size_mismatch(tmp_path):
    _make_corpus(tmp_path, n_batches=1, batch_size=3, manifest_rows=3)
    with pytest.raises(SystemExit):
        prepare_output_dir(tmp_path, 250, resume=True)


def test_resume_binds_to_corpus_config_fingerprint(tmp_path):
    # 設定指紋が一致すれば resume 可、corpus_config.json を保持する
    np.savez_compressed(tmp_path / "batch_000.npz",
                        np.zeros((3, 4, 4), np.uint8), batch_size=3)
    rows = [_row(f"f{i:014d}bh.fits.fz") for i in range(3)]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(r + "\n" for r in rows), encoding="utf-8")
    (tmp_path / "corpus_config.json").write_text(
        json.dumps({"fingerprint": "cfg-a"}), encoding="utf-8")

    done, _, kept = prepare_output_dir(
        tmp_path, 3, resume=True, config_fp="cfg-a")
    assert kept == 3
    assert json.loads((tmp_path / "corpus_config.json").read_text())[
        "fingerprint"] == "cfg-a"

    # 除外境界・サンプリング規則が変わった resume は拒否
    with pytest.raises(SystemExit):
        prepare_output_dir(tmp_path, 3, resume=True, config_fp="cfg-b")


def test_resume_rejects_missing_config_file(tmp_path):
    # corpus_config.json が無い旧 corpus は provenance 検証不能で拒否
    np.savez_compressed(tmp_path / "batch_000.npz",
                        np.zeros((3, 4, 4), np.uint8), batch_size=3)
    (tmp_path / "manifest.jsonl").write_text(
        "".join(_row(f"f{i:014d}bh.fits.fz") + "\n" for i in range(3)),
        encoding="utf-8")
    with pytest.raises(SystemExit):
        prepare_output_dir(tmp_path, 3, resume=True, config_fp="cfg-a")


def test_clean_dir_writes_config_fingerprint(tmp_path):
    prepare_output_dir(tmp_path, 250, resume=False, config_fp="cfg-x")
    assert json.loads(
        (tmp_path / "corpus_config.json").read_text())["fingerprint"] == "cfg-x"


def test_config_fingerprint_tracks_leakage_inputs():
    from datetime import datetime, timezone
    from types import SimpleNamespace
    stamps = [datetime(2020, 1, 1, tzinfo=timezone.utc)]
    args = SimpleNamespace(year=2011, days_per_month=4, frames_per_day=12,
                           exclusion_minutes=360, batch_size=250)
    fp = corpus_config_fingerprint(stamps, set(), args)
    assert fp == corpus_config_fingerprint(stamps, set(), args)
    other = SimpleNamespace(**{**vars(args), "exclusion_minutes": 120})
    assert corpus_config_fingerprint(stamps, set(), other) != fp
    more = stamps + [datetime(2020, 1, 2, tzinfo=timezone.utc)]
    assert corpus_config_fingerprint(more, set(), args) != fp


def test_batch_frame_count_and_load_corpus_handle_aux_keys(tmp_path):
    # npz の key 順は保証されず batch_size (0-d) が先頭になりうる。
    # フレーム配列は ndim>=1 で識別しなければならない。
    np.savez_compressed(tmp_path / "batch_000.npz",
                        np.zeros((3, 4, 4), np.uint8), batch_size=3)
    assert batch_frame_count(tmp_path / "batch_000.npz") == 3
    corpus = load_corpus(tmp_path)
    assert corpus.shape == (3, 4, 4) and corpus.dtype == np.uint8


def test_manifest_only_dir_is_also_dirty(tmp_path):
    (tmp_path / "manifest.jsonl").write_text(_row("x") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        prepare_output_dir(tmp_path, 250, resume=False)
    # resume なら batch 0個なので manifest は空に切り詰められる
    done, batch_idx, kept = prepare_output_dir(tmp_path, 250, resume=True)
    assert (done, batch_idx, kept) == (set(), 0, 0)
    assert (tmp_path / "manifest.jsonl").read_text() == ""


@pytest.mark.parametrize("anchor_kind", ["train", "test"])
def test_main_excludes_train_and_test_windows_across_sites(
        tmp_path, monkeypatch, anchor_kind):
    """Only candidates outside the inclusive window may enter the corpus."""
    from datetime import timedelta
    import sys

    from scratch import build_gong_ssl_corpus as corpus

    anchor = corpus.parse_stamp("20200101120000Mh.fits.fz")
    other = "20200301120000Bh.jpeg"
    anchor_name = "20200101120000Mh.jpeg"
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / (anchor_name if anchor_kind == "test" else other)).touch()
    train_json = tmp_path / "train.json"
    train_json.write_text(json.dumps({"images": [
        {"file_name": anchor_name if anchor_kind == "train" else other}]}),
        encoding="utf-8")
    # Every candidate uses a different site from the anchor.
    names = [(anchor + timedelta(seconds=seconds)).strftime("%Y%m%d%H%M%S")
             + "Ch.fits.fz"
             for seconds in (-21601, -21600, -60, 0, 60, 21600, 21601)]
    monkeypatch.setattr(corpus, "list_day", lambda day:
                        names if (day.month, day.day) == (1, 1) else [])
    out = tmp_path / "corpus"
    monkeypatch.setattr(sys, "argv", ["build_gong_ssl_corpus",
        "--test-dir", str(test_dir), "--train-json", str(train_json),
        "--year", "2020", "--days-per-month", "1", "--frames-per-day", "20",
        "--out", str(out), "--dry-run"])

    assert corpus.main() == 0

    summary = json.loads((out / "corpus_summary.json").read_text())
    assert summary["kept"] == 2  # +/-6h plus one second only
    assert summary["skipped_observation_window"] == 5


@pytest.mark.parametrize("images", [
    [], [{"file_name": "unparseable.jpeg"}],
    [{"file_name": "20200101120000Mh.jpeg"}, {"file_name": "bad.jpeg"}],
    [{}], [{"file_name": "20200230120000Mh.jpeg"}],
])
def test_main_rejects_incomplete_train_anchors_before_listing(
        tmp_path, monkeypatch, images):
    import sys

    from scratch import build_gong_ssl_corpus as corpus

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "20200301120000Bh.jpeg").touch()
    train_json = tmp_path / "train.json"
    train_json.write_text(json.dumps({"images": images}), encoding="utf-8")
    listed = []
    monkeypatch.setattr(corpus, "list_day", lambda day: listed.append(day) or [])
    monkeypatch.setattr(sys, "argv", ["build_gong_ssl_corpus",
        "--test-dir", str(test_dir), "--train-json", str(train_json),
        "--out", str(tmp_path / "corpus"), "--dry-run"])

    with pytest.raises(SystemExit, match="train"):
        corpus.main()
    assert listed == []


def test_main_rejects_negative_exclusion_window(tmp_path, monkeypatch):
    import sys

    from scratch import build_gong_ssl_corpus as corpus

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "20200301120000Bh.jpeg").touch()
    train_json = tmp_path / "train.json"
    train_json.write_text(json.dumps({"images": [
        {"file_name": "20200101120000Mh.jpeg"}]}), encoding="utf-8")
    monkeypatch.setattr(corpus, "list_day", lambda day: [])
    monkeypatch.setattr(sys, "argv", ["build_gong_ssl_corpus",
        "--test-dir", str(test_dir), "--train-json", str(train_json),
        "--exclusion-minutes", "-1", "--out", str(tmp_path / "corpus"),
        "--dry-run"])
    with pytest.raises(SystemExit, match="exclusion-minutes"):
        corpus.main()


def test_resume_rejects_legacy_exact_train_policy(tmp_path):
    """Identical anchors/settings must not resume a pre-fix corpus."""
    import hashlib
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from scratch.build_gong_ssl_corpus import IMG_SIZE

    test = [datetime(2020, 3, 1, tzinfo=timezone.utc)]
    train = {datetime(2020, 1, 1, tzinfo=timezone.utc)}
    args = SimpleNamespace(year=2011, days_per_month=4, frames_per_day=12,
                           exclusion_minutes=360, batch_size=3)
    # Actual pre-fix fingerprint algorithm, intentionally without policy version.
    legacy = hashlib.sha256()
    for stamp in sorted(s.isoformat() for s in test):
        legacy.update(stamp.encode())
    for stamp in sorted(s.isoformat() for s in train):
        legacy.update(b"train:" + stamp.encode())
    legacy.update(json.dumps({
        "year": args.year, "days_per_month": args.days_per_month,
        "frames_per_day": args.frames_per_day,
        "exclusion_minutes": args.exclusion_minutes,
        "batch_size": args.batch_size, "img_size": IMG_SIZE},
        sort_keys=True).encode())
    old_fp = legacy.hexdigest()[:16]
    _make_corpus(tmp_path, n_batches=1, batch_size=3, manifest_rows=3)
    (tmp_path / "corpus_config.json").write_text(
        json.dumps({"fingerprint": old_fp}), encoding="utf-8")
    current_fp = corpus_config_fingerprint(test, train, args)

    assert current_fp != old_fp
    with pytest.raises(SystemExit):
        prepare_output_dir(tmp_path, 3, resume=True, config_fp=current_fp)
