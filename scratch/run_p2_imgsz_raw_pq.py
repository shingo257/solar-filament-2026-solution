"""P2-2 の 1536 OOF と V30 の 1280 OOF を、refiner なしで同じ規則で採点する。

目的: 学習解像度の差が検出器そのもの（生YOLO）で出ているのか、
refiner との相性なのかを切り分ける。GPU 不要。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics_official import PanopticQuality  # noqa: E402
from scratch.run_pipeline_stack import apply_ownership  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    FOLDS_CSV,
    MIN_AREA,
    TRAIN_JSON,
    iou_matrix,
    load_yolo_cache,
    masks_to_rles,
    to_rle,
)

DRIVE = Path("G:/マイドライブ")
CACHE_1280 = DRIVE / "solar_filament_v34/prediction_cache/oof/yolo_fold_0"
CACHE_1536 = DRIVE / "solar_filament_v35/prediction_cache/oof/yolo_imgsz1536_fold_0"
OUT = Path(__file__).resolve().parents[1] / "output" / "v36"
CONF = 0.26
FOLD = 0
RULES = ("conf_first", "drop_contained_then_area_asc")
BASELINE_FOLD0_RAW = 0.397686


def load_gt() -> tuple[dict[str, list[int]], dict[int, list[Any]], dict[str, int]]:
    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
    annotations: dict[int, list[Any]] = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    file_to_ids: dict[str, list[int]] = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)
    gt_rles = {
        i: [to_rle(a, int(images[i]["height"]), int(images[i]["width"]))
            for a in anns if not a.get("iscrowd", 0)]
        for i, anns in annotations.items()
    }
    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    return file_to_ids, gt_rles, fold_of_file


def score_cache(
    cache: Path,
    file_to_ids: dict[str, list[int]],
    gt_rles: dict[int, list[Any]],
    fold_of_file: dict[str, int],
    rule: str,
) -> dict[str, Any]:
    targets = [
        name for name in sorted(file_to_ids)
        if fold_of_file[name] == FOLD and (cache / f"{Path(name).stem}.npz").exists()
    ]
    meter = PanopticQuality()
    near_miss = 0
    counts: list[int] = []
    for name in targets:
        masks, confidences = load_yolo_cache(cache / f"{Path(name).stem}.npz")
        keep = [
            i for i, (mask, score) in enumerate(zip(masks, confidences))
            if score >= CONF and mask.sum() >= MIN_AREA
        ]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if keep else np.zeros(0, np.float32)
        predictions = apply_ownership(proposals, scores, rule)
        counts.append(len(predictions))
        rles = masks_to_rles(predictions)
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, rles)
            meter.add_matrix(matrix, len(truth), len(rles))
            if matrix.size:
                near_miss += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())
    summary = meter.summary()
    return {
        "cache": str(cache),
        "rule": rule,
        "conf": CONF,
        "observations": len(targets),
        "pq": round(float(summary["pq"]), 6),
        "sq": round(float(summary["sq"]), 6),
        "rq": round(float(summary["rq"]), 6),
        "tp": int(summary["tp"]),
        "fp": int(summary["fp"]),
        "fn": int(summary["fn"]),
        "near_miss": int(near_miss),
        "inst/img": float(np.mean(counts)) if counts else 0.0,
    }


def main() -> int:
    missing = [path for path in (CACHE_1280, CACHE_1536) if not path.exists()]
    if missing:
        print(f"OOF cache が無い: {missing}")
        return 1
    file_to_ids, gt_rles, fold_of_file = load_gt()
    rows: list[dict[str, Any]] = []
    for label, cache in (("imgsz1280_v30", CACHE_1280), ("imgsz1536_trained", CACHE_1536)):
        for rule in RULES:
            print(f"scoring {label} {rule}", flush=True)
            row = score_cache(cache, file_to_ids, gt_rles, fold_of_file, rule)
            row["detector"] = label
            row["baseline_fold0_raw_conf_first"] = BASELINE_FOLD0_RAW
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "scope": "fold0 236観測、refiner なし。同一 conf 0.26 と同一排他化。",
        "purpose": "P2-2 の PQ 低下が検出器本体か refiner 相性かを切り分ける",
        "table": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p2_2_raw_yolo_fold0.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
