"""Offline contracts for explicit, fail-closed temporal frame acquisition."""

import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from scratch import fetch_temporal_neighbors as cli


ANCHOR = "20260102000000Bh.jpeg"
DOWNLOAD_URL = f"{cli.BASE}/202601/20260102/20260102000100Bh.fits.fz"


def instant(value):
    return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must explicitly mock all HTTP")

    monkeypatch.setattr(cli.urllib.request, "urlopen", forbidden)


def inputs(tmp_path, *, dry_run=True, names=None):
    names = names or [ANCHOR]
    source = tmp_path / "files.txt"
    source.write_text("\n".join(names) + "\n", encoding="utf-8")
    contract = {
        "schema_version": 1,
        "protocol_sha256": cli.PROTOCOL_SHA256,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "training_filenames": sorted(set(names)),
        "config": {"minutes": 30, "per_obs": 2, "count": 20, "dry_run": dry_run},
        "excluded_observations": ["20260103120000Ch.jpeg"],
        "excluded_utc_dates": ["2026-01-03"],
        "exclusion_minutes": 30,
        "policy": "all_sites_time_and_utc_date",
        "scope": "availability_only",
    }
    policy = tmp_path / "contract.json"
    policy.write_text(json.dumps(contract), encoding="utf-8")
    out = tmp_path / "acquisition"
    argv = ["--files", str(source), "--contract", str(policy), "--out", str(out)]
    if dry_run:
        argv.append("--dry-run")
    return source, policy, contract, out, argv


@pytest.mark.parametrize("name", [
    "../20260102000000Bh.jpeg", "x20260102000000Bh.jpeg",
    "20260102000000Bh.jpeg.tmp", "20260230000000Bh.jpeg",
    "20260102000000Zh.jpeg", "２０２６０１０２００００００Bh.jpeg",
])
def test_observation_names_are_full_valid_basenames(name):
    with pytest.raises(ValueError):
        cli.parse_obs(name)


def test_observation_and_archive_share_utc_identity():
    expected = (instant("20260102000000"), "B")
    assert cli.parse_obs(ANCHOR) == expected
    assert cli.parse_obs("20260102000000Bh.fits.fz") == expected


def test_coco_labeler_duplicates_are_valid_but_image_ids_are_unique(tmp_path):
    source = tmp_path / "train.json"
    source.write_text(json.dumps({"images": [
        {"id": 1, "file_name": ANCHOR}, {"id": 2, "file_name": ANCHOR},
    ]}), encoding="utf-8")
    captured, names = cli.load_input(source, True)
    assert captured == source.read_bytes()
    assert names == [ANCHOR]
    source.write_text(json.dumps({"images": [
        {"id": 1, "file_name": ANCHOR}, {"id": 1, "file_name": ANCHOR},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError):
        cli.load_input(source, True)


def test_text_duplicates_rejected(tmp_path):
    source = tmp_path / "files.txt"
    source.write_text(f"{ANCHOR}\n{ANCHOR}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cli.load_input(source, False)


def test_candidates_use_actual_seconds_dedup_and_window_days(monkeypatch):
    days = []

    def listing(day):
        days.append(day.strftime("%Y%m%d"))
        if day.strftime("%Y%m%d") == "20260101":
            return [("20260101235959Bh.fits.fz", "B")]
        return [("20260102000002Bh.fits.fz", "B"),
                ("20260102000001Bh.fits.fz", "B"),
                ("20260102000001Bh.fits.fz", "B"),
                ("20260102000000Bh.fits.fz", "B"),
                ("20260102000001Ch.fits.fz", "C")]

    monkeypatch.setattr(cli, "list_day", listing)
    found = cli.neighbors_for(instant("20260102000000"), "B", 30, {})
    assert [item["file"] for item in found] == [
        "20260101235959Bh.fits.fz", "20260102000001Bh.fits.fz",
        "20260102000002Bh.fits.fz",
    ]
    assert days == ["20260101", "20260102"]


def test_listing_only_accepts_valid_files_from_requested_day(monkeypatch):
    html = """<a href="20260102000100Bh.fits.fz">valid</a>
    <a href="20260102000100Bh.fits.fz">duplicate</a>
    <a href="20260101000100Bh.fits.fz">wrong day</a>
    <a href="20260230000100Bh.fits.fz">bad calendar</a>
    <a href="../20260102000200Bh.fits.fz">traversal</a>"""
    monkeypatch.setattr(cli, "fetch_url", lambda url: html)
    assert cli.list_day(instant("20260102000000")) == [
        ("20260102000100Bh.fits.fz", "B"),
    ]


def test_excluded_candidate_date_is_removed_before_cap(monkeypatch):
    monkeypatch.setattr(cli, "list_day", lambda day: [
        ("20260101235959Bh.fits.fz", "B"),
        ("20260102000001Bh.fits.fz", "B"),
    ])
    found = cli.neighbors_for(
        instant("20260102000000"), "B", 30, {},
        excluded_utc_dates=["2026-01-01"],
    )
    assert found[0]["file"] == "20260102000001Bh.fits.fz"
    assert len(found) == 1


def test_protected_times_apply_across_sites_and_dates(monkeypatch):
    monkeypatch.setattr(cli, "list_day", lambda day: [
        ("20260102120100Bh.fits.fz", "B"),
        ("20260102122500Bh.fits.fz", "B"),
        ("20260102122000Bh.fits.fz", "B"),
    ])
    found = cli.neighbors_for(
        instant("20260102120000"), "B", 30, {},
        excluded_observations=["20260102125000Ch.jpeg"],
        excluded_utc_dates=[], exclusion_minutes=30,
    )
    assert [item["file"] for item in found] == [
        "20260102120100Bh.fits.fz",
    ]
    with pytest.raises(ValueError):
        cli.neighbors_for(
            instant("20260102120000"), "B", 30, {},
            excluded_utc_dates=["2026-01-02"],
        )


@pytest.mark.parametrize("key,value", [
    ("protocol_sha256", "0" * 64), ("input_sha256", "0" * 64),
    ("training_filenames", []), ("scope", "training_approved"),
    ("policy", "same_site_only"), ("exclusion_minutes", 29),
    ("schema_version", True),
])
def test_invalid_contract_rejected_before_any_network(tmp_path, key, value):
    _, policy, contract, out, argv = inputs(tmp_path)
    contract[key] = value
    policy.write_text(json.dumps(contract), encoding="utf-8")
    assert cli.main(argv) != 0
    assert not out.exists()


def test_training_anchor_cannot_overlap_cross_site_protection(tmp_path):
    _, policy, contract, out, argv = inputs(tmp_path)
    contract["excluded_observations"] = ["20260102000100Ch.jpeg"]
    policy.write_text(json.dumps(contract), encoding="utf-8")
    assert cli.main(argv) != 0
    assert not out.exists()


@pytest.mark.parametrize("option,value", [
    ("--count", "0"), ("--per-obs", "-1"), ("--minutes", "1441"),
])
def test_invalid_config_never_acquires(tmp_path, option, value):
    _, _, _, out, argv = inputs(tmp_path)
    try:
        result = cli.main(argv + [option, value])
    except SystemExit as error:
        result = error.code
    assert result != 0
    assert not out.exists()


def test_existing_directory_is_untouched(tmp_path):
    _, _, _, out, argv = inputs(tmp_path)
    out.mkdir()
    marker = out / "manifest.json"
    marker.write_bytes(b"existing immutable evidence")
    assert cli.main(argv) != 0
    assert marker.read_bytes() == b"existing immutable evidence"


def test_listing_failure_has_failed_manifest_and_nonzero_exit(tmp_path, monkeypatch):
    _, _, _, out, argv = inputs(tmp_path)

    def failed(day, **kwargs):
        raise OSError("offline fixture failure")

    monkeypatch.setattr(cli, "list_day", failed)
    assert cli.main(argv) != 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "failed"


def test_empty_listing_is_successful_availability_only(tmp_path, monkeypatch):
    _, _, _, out, argv = inputs(tmp_path)
    monkeypatch.setattr(cli, "list_day", lambda day, **kwargs: [])
    assert cli.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "complete"
    assert manifest["scope"] == "availability_only"


class Response:
    def __init__(self, chunks, *, url=DOWNLOAD_URL, status=200):
        self.chunks = iter(chunks)
        self.url = url
        self.status = status
        self.headers = {}

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        value = next(self.chunks, b"")
        if isinstance(value, BaseException):
            raise value
        return value


def test_declared_truncated_download_is_not_published(tmp_path, monkeypatch):
    response = Response([b"short", b""])
    response.headers = {"Content-Length": "100"}
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: response)
    destination = tmp_path / "frame.fits.fz"
    with pytest.raises(ValueError, match="length"):
        cli.download(DOWNLOAD_URL, destination)
    assert list(tmp_path.iterdir()) == []


def test_download_records_declared_length_verification(tmp_path, monkeypatch):
    response = Response([b"fits", b""])
    response.headers = {"Content-Length": "4"}
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: response)
    result = cli.download(DOWNLOAD_URL, tmp_path / "frame.fits.fz")
    assert result["content_length_verified"] is True
    assert result["declared_content_length"] == 4


def test_download_never_reports_after_publication(tmp_path, monkeypatch):
    import builtins

    destination = tmp_path / "frame.fits.fz"
    reported_after_publication = []

    def record_progress(message, **kwargs):
        reported_after_publication.append(destination.exists())

    monkeypatch.setattr(builtins, "print", record_progress)
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: Response([b"fits", b""]))
    result = cli.download(DOWNLOAD_URL, destination)
    assert destination.read_bytes() == b"fits"
    assert result["bytes"] == 4
    assert result["sha256"] == hashlib.sha256(b"fits").hexdigest()
    assert len(reported_after_publication) >= 2
    assert not any(reported_after_publication)


@pytest.mark.parametrize("error", [BrokenPipeError, ValueError])
def test_download_progress_output_failure_is_exercised(tmp_path, monkeypatch, error):
    import builtins

    calls = []
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    def broken_progress(message, **kwargs):
        calls.append(message)
        raise error("consumer closed stdout")

    monkeypatch.setattr(builtins, "print", broken_progress)
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: Response([b"fits", b""]))
    destination = tmp_path / "frame.fits.fz"
    result = cli.download(DOWNLOAD_URL, destination)
    assert len(calls) >= 2
    assert destination.read_bytes() == b"fits"
    assert result["bytes"] == 4


def test_closed_console_does_not_prevent_manifest_publication(tmp_path, monkeypatch):
    import builtins

    _, _, _, out, argv = inputs(tmp_path, dry_run=False)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    candidate = "20260102000100Bh.fits.fz"
    monkeypatch.setattr(cli, "list_day", lambda day, **kwargs: [(candidate, "B")])
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: Response([b"fits", b""]))

    def closed(*args, **kwargs):
        raise BrokenPipeError("consumer closed stdout")

    monkeypatch.setattr(builtins, "print", closed)
    assert cli.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "complete"
    assert manifest["stats"]["downloaded"] == 1
    assert manifest["entries"][0]["neighbors"][0]["bytes"] == 4
    assert (out / "fits" / candidate).read_bytes() == b"fits"


def test_real_closed_pipe_retains_successful_process_exit():
    process = subprocess.Popen(
        [sys.executable, "-c", "from scratch import fetch_temporal_neighbors as c; c.report_progress('progress'); raise SystemExit(0)"],
        cwd=cli.Path(cli.__file__).resolve().parents[1],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    process.stdout.close()
    try:
        assert process.wait(timeout=30) == 0
        assert b"BrokenPipeError" not in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stderr.close()


def test_download_never_replaces_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "frame.fits.fz"
    destination.write_bytes(b"original")
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: Response([b"new", b""]))
    with pytest.raises(FileExistsError):
        cli.download(DOWNLOAD_URL, destination)
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(3)])
def test_second_download_interruption_records_failure(tmp_path, monkeypatch, interruption):
    _, _, _, out, argv = inputs(tmp_path, dry_run=False)
    first = "20260102000100Bh.fits.fz"
    second = "20260102000200Bh.fits.fz"
    monkeypatch.setattr(cli, "list_day", lambda day, **kwargs: [(first, "B"), (second, "B")])
    responses = iter([
        Response([b"complete", b""]),
        Response([b"partial", interruption], url=f"{cli.BASE}/202601/20260102/{second}"),
    ])
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: next(responses))
    try:
        result = cli.main(argv)
    except BaseException as error:
        pytest.fail(f"Interruption escaped before recording failure: {type(error).__name__}")
    assert result != 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "failed"
    assert manifest["error"]["type"] == type(interruption).__name__
    assert manifest["stats"]["downloaded"] == 1
    assert (out / "fits" / first).read_bytes() == b"complete"
    assert list((out / "fits").iterdir()) == [out / "fits" / first]


@pytest.mark.parametrize("status,content_range,final_url", [
    (206, None, DOWNLOAD_URL),
    (200, "bytes 0-3/100", DOWNLOAD_URL),
    (200, None, DOWNLOAD_URL.replace("000100", "000200")),
    (200, None, "https://fixture.invalid/frame"),
])
def test_download_rejects_partial_or_wrong_final_object(
    tmp_path, monkeypatch, status, content_range, final_url,
):
    response = Response([b"fits", b""], url=final_url, status=status)
    response.headers = {"Content-Length": "4"}
    if content_range is not None:
        response.headers["Content-Range"] = content_range
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: response)
    with pytest.raises(ValueError):
        cli.download(DOWNLOAD_URL, tmp_path / "frame.fits.fz")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status,final_url", [
    (206, f"{cli.BASE}/202601/20260102/"),
    (200, f"{cli.BASE}/202601/20260101/"),
    (200, "https://fixture.invalid/listing"),
])
def test_listing_rejects_partial_or_wrong_final_url(monkeypatch, status, final_url):
    response = Response([b"<html></html>"], url=final_url, status=status)
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: response)
    with pytest.raises(ValueError):
        cli.fetch_url(f"{cli.BASE}/202601/20260102/")


def test_listing_records_resolved_response_and_body_identity(monkeypatch):
    url = f"{cli.BASE}/202601/20260102/"
    body = b"<html>archive listing</html>"
    response = Response([body], url=url)
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: response)
    provenance = []
    assert cli.fetch_url(url, provenance=provenance) == body.decode()
    assert provenance == [{"requested_url": url, "resolved_url": url,
                           "http_status": 200, "content_length_header": None,
                           "sha256": hashlib.sha256(body).hexdigest()}]


@pytest.mark.parametrize("key,value", [
    ("minutes", 31), ("per_obs", 3), ("count", 1), ("dry_run", False),
])
def test_frozen_config_must_equal_actual_arguments(tmp_path, key, value):
    _, policy, contract, out, argv = inputs(tmp_path)
    contract["config"][key] = value
    policy.write_text(json.dumps(contract), encoding="utf-8")
    assert cli.main(argv) != 0
    assert not out.exists()


def test_dry_run_with_candidates_never_calls_download(tmp_path, monkeypatch):
    _, _, _, out, argv = inputs(tmp_path)
    candidate = "20260102000100Bh.fits.fz"
    monkeypatch.setattr(cli, "list_day", lambda day, **kwargs: [(candidate, "B")])
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("dry-run must never download")

    monkeypatch.setattr(cli, "download", forbidden)
    assert cli.main(argv) == 0
    assert calls == []
    assert not (out / "fits").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["downloaded"] == 0
    assert manifest["entries"][0]["neighbors"][0]["file"] == candidate


def test_http200_without_matching_links_is_unverified(monkeypatch, capsys):
    url = f"{cli.BASE}/202601/20260102/"
    response = Response([b"<html>maintenance</html>"], url=url)
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: response)
    provenance = []
    assert cli.list_day(instant("20260102000000"), provenance=provenance) == []
    assert provenance[0]["matching_files"] == 0
    assert provenance[0]["listing_interpretation"] == "unverified_no_matching_files"
    assert provenance[0]["content_length_header"] is None
    assert "archive availability is unverified" in capsys.readouterr().out


def test_manifest_publication_failure_reports_processed_acquisition(tmp_path, monkeypatch, capsys):
    _, _, _, out, argv = inputs(tmp_path, dry_run=False)
    candidate = "20260102000100Bh.fits.fz"
    monkeypatch.setattr(cli, "list_day", lambda day, **kwargs: [(candidate, "B")])
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: Response([b"fits", b""]))

    def failed_publish(*args):
        raise OSError("fixture publication failure")

    monkeypatch.setattr(cli, "atomic_publish", failed_publish)
    assert cli.main(argv) != 0
    assert (out / "fits" / candidate).read_bytes() == b"fits"
    assert not (out / "manifest.json").exists()
    assert "manifest publication failed" in capsys.readouterr().out


def test_two_anchors_share_one_download_and_manifest_identity(tmp_path, monkeypatch, capsys):
    _, _, _, out, argv = inputs(tmp_path, dry_run=False, names=[ANCHOR, "20260102000200Bh.jpeg"])
    candidate = "20260102000100Bh.fits.fz"
    monkeypatch.setattr(cli, "list_day", lambda day, **kwargs: [(candidate, "B")])
    requests = []

    def transport(url, **kwargs):
        assert f"Downloading {candidate}" in capsys.readouterr().out
        requests.append(url)
        return Response([b"shared frame", b""])

    monkeypatch.setattr(cli.urllib.request, "urlopen", transport)
    assert cli.main(argv) == 0
    assert requests == [DOWNLOAD_URL]
    progress = capsys.readouterr().out
    assert f"Downloading {candidate}: 12 bytes" in progress
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["downloaded"] == 1
    assert manifest["stats"]["obs"] == 2
    first, second = [entry["neighbors"][0] for entry in manifest["entries"]]
    for key in ("sha256", "bytes", "resolved_url", "http_status"):
        assert first[key] == second[key]
    assert first["sha256"] == hashlib.sha256(b"shared frame").hexdigest()


@pytest.mark.parametrize("broken", [False, True])
def test_download_publication_is_atomic_and_hashed(tmp_path, monkeypatch, broken):
    _, _, _, out, argv = inputs(tmp_path, dry_run=False)
    candidate = "20260102000100Bh.fits.fz"
    monkeypatch.setattr(cli, "list_day", lambda day, **kwargs: [(candidate, "B")])
    payload = b"mock FITS bytes"
    chunks = [payload, OSError("interrupted")] if broken else [payload, b""]
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: Response(chunks))
    result = cli.main(argv)
    manifest_text = (out / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    if broken:
        assert result != 0
        assert manifest["state"] == "failed"
        assert not (out / "fits" / candidate).exists()
        assert list((out / "fits").iterdir()) == []
    else:
        assert result == 0
        assert manifest["state"] == "complete"
        assert (out / "fits" / candidate).read_bytes() == payload
        assert hashlib.sha256(payload).hexdigest() in manifest_text
        item = manifest["entries"][0]["neighbors"][0]
        assert item["content_length_verified"] is False
        assert item["declared_content_length"] is None
