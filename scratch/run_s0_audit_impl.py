"""Implementation helpers for `run_s0_audit` — extracted verbatim from `run_s0_audit.py`.

Holds the helper functions and module constants so `run_s0_audit.py` stays a readable
orchestrator. Loaded by the wrapper via sibling import.
"""


from __future__ import annotations


import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from solar_filament.metrics.science_metrics import audit_spine_coordinates  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DRIVE = Path("G:/マイドライブ")


DEFAULT_V30 = DEFAULT_DRIVE / "solar_filament_v30"


DEFAULT_V34 = DEFAULT_DRIVE / "solar_filament_v34"


SITE_PATTERN = re.compile(r"([A-Za-z]h)$")


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(path: Path, content_hash: bool) -> dict:
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.exists() else []
    digest = hashlib.sha256()
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "suffix_counts": {},
            "hash_mode": "content" if content_hash else "path-size-inventory",
            "sha256": None,
        }
    total_bytes = 0
    suffixes = Counter()
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        total_bytes += size
        suffixes[item.suffix.lower() or "<none>"] += 1
        file_digest = sha256_file(item) if content_hash else "not-computed"
        digest.update(f"{relative}\0{size}\0{file_digest}\n".encode("utf-8"))
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "suffix_counts": dict(sorted(suffixes.items())),
        "hash_mode": "content" if content_hash else "path-size-inventory",
        "sha256": digest.hexdigest(),
    }


def file_manifest(path: Path, content_hash: bool = True) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "sha256": sha256_file(path) if content_hash else "not-computed",
    }


def point_distance_to_annotation(point: tuple[float, float], segmentation: object) -> float:
    """Signed distance to the closest polygon; positive values are inside."""

    x, y = point
    distances: list[float] = []
    if isinstance(segmentation, list):
        for polygon in segmentation:
            values = np.asarray(polygon, dtype=np.float32)
            if values.size < 6 or values.size % 2:
                continue
            contour = values.reshape(-1, 1, 2)
            distances.append(
                float(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
            )
    return max(distances, default=float("-inf"))


def point_inside_annotation(point: tuple[float, float], segmentation: object) -> bool:
    return point_distance_to_annotation(point, segmentation) >= 0.0


def segmentation_kind(segmentation: object) -> str:
    if isinstance(segmentation, list):
        return "polygon"
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        return "rle_uncompressed" if isinstance(counts, list) else "rle_compressed"
    return "missing_or_invalid"


def parse_site(file_name: str) -> str:
    match = SITE_PATTERN.search(Path(file_name).stem)
    if not match:
        return "unknown"
    code = match.group(1)
    return code[0].upper() + code[1:].lower()


def parse_labeler(image_id: object) -> str:
    value = str(image_id)
    return value.split("-", 1)[0] if "-" in value else "unknown"


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def audit_annotations(coco: dict, fold_rows: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    categories = {int(item["id"]): item["name"] for item in coco["categories"]}
    image_ids = [str(item["id"]) for item in coco["images"]]
    annotation_ids = [str(item["id"]) for item in coco["annotations"]]
    images = {str(item["id"]): item for item in coco["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[str(annotation["image_id"])].append(annotation)

    fold_of_file = {row["file_name"]: int(row["fold"]) for row in fold_rows}
    observation_images = defaultdict(list)
    for image in coco["images"]:
        observation_images[image["file_name"]].append(image)
    observation_names = set(observation_images)
    fold_names = [row["file_name"] for row in fold_rows]
    unknown_annotation_images = sum(
        str(annotation["image_id"]) not in images for annotation in coco["annotations"]
    )

    annotation_rows = []
    spine_lengths: list[float] = []
    areas: list[float] = []
    radial_distances: list[float] = []
    point_counts: list[float] = []
    all_inside_fractions: list[float] = []
    all_within_1px_fractions: list[float] = []
    all_within_2px_fractions: list[float] = []
    chirality_counts = Counter({name: 0 for name in categories.values()})
    labeler_counts = Counter()
    bad_spines = Counter()
    segmentation_counts = Counter()

    for annotation in coco["annotations"]:
        image = images.get(str(annotation["image_id"]))
        if image is None:
            continue
        spine = audit_spine_coordinates(annotation.get("spine"), image["height"], image["width"])
        if not spine.even_length:
            bad_spines["odd_coordinate_count"] += 1
        if not spine.finite:
            bad_spines["non_finite"] += 1
        if spine.in_bounds_count != spine.point_count:
            bad_spines["out_of_bounds"] += 1
        distances = [
            point_distance_to_annotation((float(x), float(y)), annotation.get("segmentation"))
            for x, y in spine.points
            if np.isfinite([x, y]).all()
            and 0 <= x < image["width"]
            and 0 <= y < image["height"]
        ]
        inside_fraction = float(np.mean(np.asarray(distances) >= 0.0)) if distances else 0.0
        within_1px_fraction = float(np.mean(np.asarray(distances) >= -1.0)) if distances else 0.0
        within_2px_fraction = float(np.mean(np.asarray(distances) >= -2.0)) if distances else 0.0
        category = categories.get(int(annotation["category_id"]), "unknown")
        labeler = parse_labeler(annotation["image_id"])
        chirality_counts[category] += 1
        labeler_counts[labeler] += 1
        segmentation_counts[segmentation_kind(annotation.get("segmentation"))] += 1
        point_counts.append(spine.point_count)
        all_inside_fractions.append(inside_fraction)
        all_within_1px_fractions.append(within_1px_fraction)
        all_within_2px_fractions.append(within_2px_fraction)
        area = float(annotation.get("area", 0.0))
        areas.append(area)
        bbox = np.asarray(annotation.get("bbox", []), dtype=float).reshape(-1)
        radial_distance = float("nan")
        if bbox.size == 4 and np.isfinite(bbox).all():
            center_x = bbox[0] + 0.5 * bbox[2]
            center_y = bbox[1] + 0.5 * bbox[3]
            radius = 0.5 * min(float(image["width"]), float(image["height"]))
            radial_distance = float(
                np.hypot(center_x - 0.5 * image["width"], center_y - 0.5 * image["height"])
                / max(radius, 1.0)
            )
            radial_distances.append(radial_distance)
        if spine.point_count >= 2:
            spine_lengths.append(float(np.linalg.norm(np.diff(spine.points, axis=0), axis=1).sum()))
        annotation_rows.append({
            "annotation_id": annotation["id"],
            "image_id": annotation["image_id"],
            "file_name": image["file_name"],
            "fold": fold_of_file.get(image["file_name"], -1),
            "labeler": labeler,
            "chirality": category,
            "area": area,
            "bbox_center_radius_normalized": radial_distance,
            "segmentation_kind": segmentation_kind(annotation.get("segmentation")),
            "spine_points": spine.point_count,
            "spine_even": spine.even_length,
            "spine_finite": spine.finite,
            "spine_in_bounds_fraction": spine.in_bounds_fraction,
            "spine_inside_polygon_fraction": inside_fraction,
            "spine_within_1px_polygon_fraction": within_1px_fraction,
            "spine_within_2px_polygon_fraction": within_2px_fraction,
        })

    observation_rows = []
    years = Counter()
    sites = Counter()
    folds = Counter()
    parse_failures = 0
    observation_timestamps = Counter()
    observation_dates: list[datetime] = []
    site_year_counts = Counter()
    fold_site_counts = Counter()
    labeler_count_distribution = Counter()
    chirality_disagreement = 0
    instance_count_disagreement = 0
    date_disagreement = 0
    observations_without_date = 0
    for file_name, variants in sorted(observation_images.items()):
        parsed_dates = []
        variant_chiralities = set()
        variant_chirality_counters = []
        instance_counts = []
        for image in variants:
            try:
                parsed = datetime.fromisoformat(image["date_captured"])
                parsed_dates.append(parsed)
            except (KeyError, TypeError, ValueError):
                parse_failures += 1
            anns = annotations_by_image[str(image["id"])]
            instance_counts.append(len(anns))
            chirality_counter = Counter(
                categories.get(int(a["category_id"]), "unknown") for a in anns
            )
            variant_chirality_counters.append(chirality_counter)
            variant_chiralities.update(chirality_counter)
        site = parse_site(file_name)
        fold = fold_of_file.get(file_name, -1)
        date_value = min(parsed_dates) if parsed_dates else None
        if len({value.isoformat() for value in parsed_dates}) > 1:
            date_disagreement += 1
        if date_value is None:
            observations_without_date += 1
        if date_value:
            observation_dates.append(date_value)
            years[date_value.year] += 1
            observation_timestamps[date_value.isoformat()] += 1
            site_year_counts[(site, date_value.year)] += 1
        sites[site] += 1
        folds[fold] += 1
        fold_site_counts[(fold, site)] += 1
        labeler_count_distribution[len(variants)] += 1
        chirality_signatures = {
            tuple(sorted(counter.items())) for counter in variant_chirality_counters
        }
        if len(chirality_signatures) > 1:
            chirality_disagreement += 1
        if instance_counts and min(instance_counts) != max(instance_counts):
            instance_count_disagreement += 1
        observation_rows.append({
            "file_name": file_name,
            "fold": fold,
            "site_code": site,
            "date_captured": date_value.isoformat(sep=" ") if date_value else "",
            "year": date_value.year if date_value else "",
            "labeler_count": len(variants),
            "instance_count_min": min(instance_counts) if instance_counts else 0,
            "instance_count_max": max(instance_counts) if instance_counts else 0,
            "chirality_classes": "|".join(sorted(variant_chiralities)),
        })

    years_sorted = sorted(years)
    locked_years: list[int] = []
    locked_count = 0
    target = max(1, round(len(observation_rows) * 0.20))
    for year in reversed(years_sorted):
        locked_years.append(year)
        locked_count += years[year]
        if locked_count >= target:
            break
    locked_years.sort()
    locked_sites = Counter()
    for row in observation_rows:
        is_locked = row["year"] in locked_years
        row["time_block_role"] = "locked_evaluation" if is_locked else "development"
        if is_locked:
            locked_sites[row["site_code"]] += 1

    ordered_dates = sorted(observation_dates)
    minute_gaps = [
        (later - earlier).total_seconds() / 60.0
        for earlier, later in zip(ordered_dates, ordered_dates[1:])
    ]
    proximity_counts = {
        "within_5_minutes": sum(0.0 < gap <= 5.0 for gap in minute_gaps),
        "within_30_minutes": sum(0.0 < gap <= 30.0 for gap in minute_gaps),
        "within_60_minutes": sum(0.0 < gap <= 60.0 for gap in minute_gaps),
    }

    report = {
        "schema_version": "s0-annotation-audit-v1",
        "scope": {
            "image_records": len(coco["images"]),
            "unique_observations": len(observation_rows),
            "annotations": len(coco["annotations"]),
            "fold_counts": {str(key): value for key, value in sorted(folds.items())},
        },
        "integrity": {
            "duplicate_image_ids": len(image_ids) - len(set(image_ids)),
            "duplicate_annotation_ids": len(annotation_ids) - len(set(annotation_ids)),
            "duplicate_fold_rows": len(fold_names) - len(set(fold_names)),
            "annotations_with_unknown_image_id": unknown_annotation_images,
            "observations_missing_fold": sorted(observation_names - set(fold_names)),
            "fold_rows_without_observation": sorted(set(fold_names) - observation_names),
        },
        "segmentation": {
            "type_counts": dict(sorted(segmentation_counts.items())),
            "polygon_distance_note": (
                "Spine containment is measured against vector polygons; RLE annotations "
                "would be reported but are not decoded by this audit."
            ),
        },
        "spine": {
            "bad_annotation_counts": dict(bad_spines),
            "point_count": quantiles(point_counts),
            "polyline_length_pixels": quantiles(spine_lengths),
            "inside_polygon_fraction": quantiles(all_inside_fractions),
            "all_points_inside_fraction": float(np.mean(np.asarray(all_inside_fractions) == 1.0)),
            "within_1px_polygon_fraction": quantiles(all_within_1px_fractions),
            "within_2px_polygon_fraction": quantiles(all_within_2px_fractions),
            "all_points_within_1px_fraction": float(
                np.mean(np.asarray(all_within_1px_fractions) == 1.0)
            ),
            "all_points_within_2px_fraction": float(
                np.mean(np.asarray(all_within_2px_fractions) == 1.0)
            ),
        },
        "chirality": {
            "instance_counts": dict(sorted(chirality_counts.items())),
            "observation_disagreement_count": chirality_disagreement,
            "observation_disagreement_fraction": chirality_disagreement / max(len(observation_rows), 1),
        },
        "labelers": dict(sorted(labeler_counts.items())),
        "labeler_count_per_observation": {
            str(key): value for key, value in sorted(labeler_count_distribution.items())
        },
        "time": {
            "image_record_parse_failures": parse_failures,
            "observations_without_date": observations_without_date,
            "observation_variant_date_disagreement": date_disagreement,
            "observation_year_counts": {str(key): value for key, value in sorted(years.items())},
            "exact_timestamp_collision_groups": sum(
                value > 1 for value in observation_timestamps.values()
            ),
            "exact_timestamp_duplicate_observations": sum(
                max(0, value - 1) for value in observation_timestamps.values()
            ),
            "consecutive_observation_proximity": proximity_counts,
            "proximity_note": "Counts include different site codes and are diagnostic only.",
            "locked_latest_years": locked_years,
            "locked_observations": locked_count,
            "locked_fraction": locked_count / max(len(observation_rows), 1),
            "locked_site_counts": dict(sorted(locked_sites.items())),
            "status": (
                "locked for future S1 evaluation; existing OOF comparisons are retrospective"
            ),
        },
        "site": {
            "code_counts": dict(sorted(sites.items())),
            "site_by_year": {
                f"{site}:{year}": value
                for (site, year), value in sorted(site_year_counts.items())
            },
            "fold_by_site": {
                f"fold{fold}:{site}": value
                for (fold, site), value in sorted(fold_site_counts.items())
            },
            "mapping_status": "codes only; physical observatory names not inferred",
        },
        "area_pixels": quantiles(areas),
        "bbox_center_radius_normalized": {
            **quantiles(radial_distances),
            "definition": (
                "bbox center distance from image center divided by half the shorter image side; "
                "this is an image-center proxy, not a calibrated heliographic distance"
            ),
        },
        "instance_count_disagreement": {
            "observations": instance_count_disagreement,
            "fraction": instance_count_disagreement / max(len(observation_rows), 1),
        },
    }
    return report, observation_rows, annotation_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
