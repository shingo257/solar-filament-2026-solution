"""2つの submission CSV を per-image IoU で照合する (レプリカ精度の事前検証)。

提出スロットを使わずに「ローカル生成 CSV が過去の公式提出とどれだけ一致するか」
を測る。片方を pseudo-GT として PQ 風の指標(TP数・一致率)を出し、加えて
filament 単位の best-IoU 分布を報告する。

使い方:
    python scratch/compare_submissions.py <a.csv> <b.csv>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

SIZE = 2048


def load_submission(path: Path) -> dict[str, list[dict]]:
    per_image = defaultdict(list)
    for row in pd.read_csv(path).itertuples():
        stem = row.filament_id.rsplit("_", 1)[0]
        per_image[stem].append(
            {"size": [SIZE, SIZE],
             "counts": row.segmentation_rle.encode("utf-8")})
    return per_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_a", type=Path)
    parser.add_argument("csv_b", type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sub_a = load_submission(args.csv_a)
    sub_b = load_submission(args.csv_b)
    stems = sorted(set(sub_a) | set(sub_b))

    matched = 0
    total_a = total_b = 0
    iou_values = []
    images_full_overlap = 0
    for stem in stems:
        rles_a, rles_b = sub_a.get(stem, []), sub_b.get(stem, [])
        total_a += len(rles_a)
        total_b += len(rles_b)
        if not rles_a or not rles_b:
            continue
        matrix = np.asarray(
            mask_util.iou(rles_a, rles_b, [0] * len(rles_b)), dtype=np.float64)
        iou_values.extend(matrix.max(axis=1).tolist())
        hits = matrix > args.iou
        matched += int((hits.any(axis=0)).sum())
        if (len(rles_a) == len(rles_b)
                and hits.any(axis=0).all() and hits.any(axis=1).all()):
            images_full_overlap += 1

    ious = np.asarray(iou_values)
    report = {
        "a": str(args.csv_a), "b": str(args.csv_b),
        "images": len(stems),
        "filaments_a": total_a, "filaments_b": total_b,
        f"b_matched_at_iou{args.iou}": matched,
        "match_rate_b": round(matched / max(total_b, 1), 4),
        "best_iou_per_a": {
            "mean": round(float(ious.mean()), 4) if ious.size else 0.0,
            "p50": round(float(np.median(ious)), 4) if ious.size else 0.0,
            "frac_iou_1.0": round(float((ious == 1.0).mean()), 4) if ious.size else 0.0,
            "frac_ge_0.9": round(float((ious >= 0.9).mean()), 4) if ious.size else 0.0,
        },
        "images_full_overlap": images_full_overlap,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
