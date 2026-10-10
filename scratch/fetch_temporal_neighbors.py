"""Acquire GONG haf temporal-neighbor files for availability inspection only.

Provide exactly one --train-json or --files plus an independently frozen
--contract JSON. Dry-run still makes archive-listing HTTP requests, but downloads
no frames. No training, label transfer, preprocessing, WCS registration, license
decision, or scientific acceptance is performed. This is not a v67 input adapter.

Contract exact keys: schema_version=1, protocol_sha256 (this source normalized to
LF), input_sha256 (exact input bytes), training_filenames (full sorted membership
before --count), config={minutes,per_obs,count,dry_run}, excluded_observations
(strict observation filenames), excluded_utc_dates (YYYY-MM-DD), exclusion_minutes
(minutes <= value <=1440), policy="all_sites_time_and_utc_date",
scope="availability_only". Membership/exclusions are operator-supplied claims;
their independence and actual heldout completeness are not verified by this tool.
All-site exclusion windows and whole UTC dates apply to anchors and candidates
before caps. Fresh output directories only; no resume or existing-file overwrite.
Output must be on a filesystem permitting hard links; publication fails closed
otherwise. Published files retain owner-only temporary-file permissions. A
missing Content-Length is recorded as unverified; hashes do not prove FITS
integrity. Catchable acquisition interruptions attempt a failed manifest; forced
process termination or filesystem failure can prevent manifest publication.
Archive responses must be HTTP 200 without Content-Range and have exactly the
requested final GONG URL. Successful listing bodies and resolved response URLs
are fingerprinted in the manifest; this does not attest intermediate redirects.
Complete means requested processing finished, not exhaustive archive availability.
HTTP 200 without matching links (including an error/maintenance page) is recorded
as unverified_no_matching_files; no directory-index format is independently
verified here. An empty observed candidate set does not prove an empty archive.
Such pages always emit a warning. Raw listing Content-Length is recorded;
close-delimited bodies without it cannot be independently checked for truncation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://gong2.nso.edu/ftp/HA/haf"
TIMEOUT = 20
PROTOCOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
NAME_RE = re.compile(r"([0-9]{14})([BCLMTUbclmtu])h\.(?:jpeg|fits\.fz)")
FILE_RE = re.compile(r'''href=["']([0-9]{14}[BCLMTUbclmtu]h\.fits\.fz)["']''')


def report_progress(message):
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        stream = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            stream.close()
        except (OSError, ValueError):
            pass  # Prevent shutdown from flushing the broken stream again.


def parse_obs(file_name: str) -> tuple[datetime, str]:
    match = NAME_RE.fullmatch(file_name) if isinstance(file_name, str) else None
    if not match:
        raise ValueError(f"Invalid observation basename: {file_name!r}")
    return datetime.strptime(match[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc), match[2].upper()


def validate_response(response, url):
    resolved = response.geturl()
    status = response.status
    if not url.startswith(BASE + "/") or resolved != url:
        raise ValueError("Archive final URL differs from the requested GONG object")
    if status != 200 or response.headers.get("Content-Range") is not None:
        raise ValueError("Archive response must be full HTTP 200 without Content-Range")
    return {"requested_url": url, "resolved_url": resolved, "http_status": status}


def fetch_url(url: str, provenance=None) -> str:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        info = validate_response(response, url)
        payload = response.read()
        html = payload.decode("utf-8")
        if provenance is not None:
            provenance.append({**info, "content_length_header": response.headers.get("Content-Length"),
                               "sha256": hashlib.sha256(payload).hexdigest()})
        return html


def list_day(day: datetime, provenance=None) -> list[tuple[str, str]]:
    url = f"{BASE}/{day:%Y%m}/{day:%Y%m%d}/"
    html = fetch_url(url) if provenance is None else fetch_url(url, provenance=provenance)
    entries = set()
    for match in FILE_RE.finditer(html):
        name = match[1]
        try:
            stamp, site = parse_obs(name)
        except ValueError:
            continue
        if stamp.date() == day.date():
            entries.add((name, site))
    if not entries:
        report_progress(f"Warning: no matching files observed at {url}; archive availability is unverified.")
    if provenance is not None:
        provenance[-1].update({"matching_files": len(entries), "listing_interpretation":
                              "matching_files_observed" if entries else "unverified_no_matching_files"})
    return sorted(entries)


def excluded(stamp, excluded_observations, excluded_utc_dates, exclusion_minutes):
    return (stamp.strftime("%Y-%m-%d") in excluded_utc_dates
            or any(abs((stamp - parse_obs(name)[0]).total_seconds()) <= exclusion_minutes * 60
                   for name in excluded_observations))


def neighbors_for(obs_time, site, minutes, day_cache, excluded_observations=(),
                  excluded_utc_dates=(), exclusion_minutes=None, listing_provenance=None):
    if type(minutes) is not int or not 0 < minutes <= 1440:
        raise ValueError("minutes must be between 1 and 1440")
    if obs_time.tzinfo is None or obs_time.utcoffset() != timedelta(0):
        raise ValueError("Observation time must be UTC")
    margin = minutes if exclusion_minutes is None else exclusion_minutes
    if type(margin) is not int or not minutes <= margin <= 1440:
        raise ValueError("Invalid exclusion_minutes")
    if excluded(obs_time, excluded_observations, excluded_utc_dates, margin):
        raise ValueError("Anchor overlaps an excluded observation/time/date")
    start, stop = obs_time - timedelta(minutes=minutes), obs_time + timedelta(minutes=minutes)
    day, last = start.date(), stop.date()
    candidates = {}
    while day <= last:
        key = day.strftime("%Y%m%d")
        if key not in day_cache:
            stamp_day = datetime.combine(day, datetime.min.time(), timezone.utc)
            day_cache[key] = (list_day(stamp_day) if listing_provenance is None else
                             list_day(stamp_day, provenance=listing_provenance))
        for name, listed_site in day_cache[key]:
            stamp, candidate_site = parse_obs(name)
            if not name.endswith(".fits.fz") or stamp.date() != day or listed_site.upper() != candidate_site:
                continue
            seconds = (stamp - obs_time).total_seconds()
            if (candidate_site != site.upper() or seconds == 0 or abs(seconds) > minutes * 60
                    or excluded(stamp, excluded_observations, excluded_utc_dates, margin)):
                continue
            candidates[name] = {"file": name, "url_day": key,
                                "dt_seconds": seconds, "dt_min": seconds / 60}
        day += timedelta(days=1)
    return sorted(candidates.values(), key=lambda item: (abs(item["dt_seconds"]), item["file"]))


def load_input(path, is_coco):
    data = Path(path).read_bytes()
    if is_coco:
        coco = json.loads(data)
        images = coco["images"]
        ids = [image["id"] for image in images]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate COCO image IDs")
        names = [image["file_name"] for image in images]
    else:
        names = [line.strip() for line in data.decode("utf-8").splitlines() if line.strip()]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate text observation filenames")
    if not names:
        raise ValueError("Input membership is empty")
    for name in names:
        parse_obs(name)
    return data, sorted(set(names))


def validate_contract(path, input_bytes, names, config):
    contract = json.loads(Path(path).read_bytes())
    keys = {"schema_version", "protocol_sha256", "input_sha256", "training_filenames", "config",
            "excluded_observations", "excluded_utc_dates", "exclusion_minutes", "policy", "scope"}
    if not isinstance(contract, dict) or set(contract) != keys:
        raise ValueError("Incomplete independently frozen acquisition contract")
    if (type(contract["schema_version"]) is not int or contract["schema_version"] != 1
            or contract["protocol_sha256"] != PROTOCOL_SHA256
            or contract["input_sha256"] != hashlib.sha256(input_bytes).hexdigest()
            or contract["training_filenames"] != names
            or contract["policy"] != "all_sites_time_and_utc_date"
            or contract["scope"] != "availability_only"):
        raise ValueError("Acquisition contract identity/policy differs")
    frozen = contract["config"]
    if (not isinstance(frozen, dict) or set(frozen) != {"minutes", "per_obs", "count", "dry_run"}
            or frozen != config or type(frozen["dry_run"]) is not bool
            or any(type(frozen[k]) is not int or frozen[k] <= 0 for k in ("minutes", "per_obs", "count"))
            or frozen["minutes"] > 1440):
        raise ValueError("Acquisition configuration differs")
    margin = contract["exclusion_minutes"]
    if type(margin) is not int or not frozen["minutes"] <= margin <= 1440:
        raise ValueError("Invalid exclusion_minutes")
    observations, dates = contract["excluded_observations"], contract["excluded_utc_dates"]
    for values in (observations, dates):
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(set(values)) != len(values):
            raise ValueError("Exclusions must be unique string lists")
    for name in observations:
        parse_obs(name)
    for day in dates:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day):
            raise ValueError("Invalid excluded UTC date")
        datetime.strptime(day, "%Y-%m-%d")
    for name in names:
        if excluded(parse_obs(name)[0], observations, dates, margin):
            raise ValueError("Anchor overlaps an excluded observation/time/date")
    return contract


def atomic_publish(path, data):
    """Publish a complete sibling temporary file exclusively; never replace."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download(url, destination):
    destination = Path(destination)
    temporary, digest, size = None, hashlib.sha256(), 0
    report_progress(f"Downloading {destination.name}")
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(url, timeout=TIMEOUT * 3) as response:
                response_info = validate_response(response, url)
                declared_length = getattr(response, "headers", {}).get("Content-Length")
                if declared_length is not None:
                    if not re.fullmatch(r"[0-9]+", declared_length):
                        raise ValueError("Invalid archive Content-Length")
                    declared_length = int(declared_length)
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    report_progress(f"Downloading {destination.name}: {size} bytes")
                if declared_length is not None and size != declared_length:
                    raise ValueError("Archive content length differs from Content-Length")
            if not size:
                raise ValueError("Empty archive download")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        return {**response_info, "sha256": digest.hexdigest(), "bytes": size,
                "declared_content_length": declared_length,
                "content_length_verified": declared_length is not None}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args):
    if (args.train_json is None) == (args.files is None):
        raise ValueError("Provide exactly one input source")
    for key in ("minutes", "per_obs", "count"):
        if type(getattr(args, key)) is not int or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.minutes > 1440:
        raise ValueError("minutes must be <=1440")
    source = args.train_json if args.train_json is not None else args.files
    input_bytes, names = load_input(source, args.train_json is not None)
    config = {k: getattr(args, k) for k in ("minutes", "per_obs", "count", "dry_run")}
    contract = validate_contract(args.contract, input_bytes, names, config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "state": "failed", "scope": "availability_only",
                "contract": contract, "operator_claims_independently_verified": False,
                "entries": [], "listings": [],
                "stats": {"obs": 0, "with_neighbor": 0, "downloaded": 0}}
    day_cache, downloads = {}, {}
    try:
        if not args.dry_run:
            (out / "fits").mkdir()
        report_progress("Availability only; dry-run still requests archive listings." if args.dry_run else "Availability acquisition only.")
        for index, name in enumerate(names[:args.count], 1):
            stamp, site = parse_obs(name)
            entry = {"file_name": name, "obs_utc": stamp.isoformat(), "site": site, "neighbors": []}
            manifest["entries"].append(entry)
            candidates = neighbors_for(stamp, site, args.minutes, day_cache,
                contract["excluded_observations"], contract["excluded_utc_dates"], contract["exclusion_minutes"],
                listing_provenance=manifest["listings"])[:args.per_obs]
            for candidate in candidates:
                filename, day = candidate["file"], candidate["url_day"]
                url = f"{BASE}/{day[:6]}/{day}/{filename}"
                item = {**candidate, "url": url}
                if not args.dry_run:
                    if filename not in downloads:
                        downloads[filename] = download(url, out / "fits" / filename)
                        manifest["stats"]["downloaded"] += 1
                    item.update(downloads[filename])
                entry["neighbors"].append(item)
            manifest["stats"]["obs"] += 1
            manifest["stats"]["with_neighbor"] += bool(candidates)
            report_progress(f"[{index}/{min(args.count, len(names))}] {name}: {len(candidates)} neighbors")
        manifest["state"] = "complete"
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        report_progress(f"Acquisition failed: {error}")
    try:
        atomic_publish(out / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode())
    except Exception as error:
        report_progress(f"Acquisition processed but manifest publication failed: {error}")
        return 1
    return 0 if manifest["state"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--train-json", type=Path)
    inputs.add_argument("--files", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--minutes", type=int, default=30, help="Window radius in minutes, 1..1440")
    parser.add_argument("--per-obs", type=int, default=2)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("outputs/temporal_neighbors"))
    parser.add_argument("--dry-run", action="store_true", help="List archive via HTTP; do not download frames")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        report_progress(f"Acquisition refused: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
