# -*- coding: utf-8 -*-
"""near-miss ペアの pred/GT 面積バイアス診断 (morphology arm の事前診断)。

prod arm (=conf>=0.26 保持 → drop_contained → greedy_disjoint → min_area) の
最終予測と GT を 1:1 近傍照合し、IoU 帯ごとの pred_area/gt_area 中央値を出す。
pred が系統的に小さければ dilation arm、大きければ erosion arm が示唆される。
結果は観察のみ — 採用判断は別途事前登録した morphology sweep で行う。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratch"))

from run_refiner_pq_allfold import MIN_AREA, to_rle  # noqa: E402
from run_v80_scorer_eval import (  # noqa: E402
    CONTAINMENT, PROD_CONF, rle_dicts,
)
from solar_filament.pipeline.ownership import (  # noqa: E402
    drop_contained_indices, greedy_disjoint,
)
from solar_filament.pipeline.paths import (  # noqa: E402
    grouped_folds_csv, resolve_drive, train_annotations_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drive", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    drive = resolve_drive(args.drive)
    cache_root = drive / "solar_filament_v80" / "candidates"

    with open(train_annotations_json(drive), encoding="utf-8") as h:
        coco = json.load(h)
    images = {i["id"]: i for i in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    file_to_ids = defaultdict(list)
    for iid, info in images.items():
        file_to_ids[info["file_name"]].append(iid)
    gt_rles = {iid: [to_rle(a, int(images[iid]["height"]), int(images[iid]["width"]))
                     for a in anns_by_img[iid]] for iid in images}
    fold_frame = pd.read_csv(grouped_folds_csv(drive))
    fold_of = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    stem_to_file = {Path(n).stem: n for n in file_to_ids}

    rows = []
    npz_files = sorted((cache_root / f"fold_{args.fold}").glob("*.npz"))
    if args.limit:
        npz_files = npz_files[: args.limit]
    for k, npz in enumerate(npz_files):
        bundle = dict(np.load(npz, allow_pickle=True))
        stem = npz.stem
        fname = stem_to_file.get(stem)
        if fname is None:
            continue
        iid = file_to_ids[fname][0]
        gts = gt_rles[iid]
        if not gts:
            continue
        masks = [mask_util.decode(r).astype(bool) for r in rle_dicts(bundle["rle_final"])]
        rles = rle_dicts(bundle["rle_final"])
        usable = np.asarray(bundle["usable"], dtype=bool)
        source = np.asarray(bundle["source"], dtype=int)
        conf = np.asarray(bundle["conf"])
        keep = list(np.where((source == 0) & usable & (conf >= PROD_CONF))[0])
        subset = [masks[i] for i in keep]
        subset_rles = [rles[i] for i in keep]
        alive = drop_contained_indices(subset, threshold=CONTAINMENT, rles=subset_rles)
        areas = np.array([m.sum() for m in subset])
        preds = greedy_disjoint(subset, sorted(alive, key=lambda i: areas[i]),
                                min_area=MIN_AREA)
        if not preds:
            continue
        pred_rles = [mask_util.encode(np.asfortranarray(p.astype(np.uint8))) for p in preds]
        ious = mask_util.iou(pred_rles, gts, [0] * len(gts))  # (n_pred, n_gt)
        for pi, pred in enumerate(preds):
            j = int(np.argmax(ious[pi]))
            iou = float(ious[pi, j])
            if iou < 0.1:
                continue
            gt_area = float(mask_util.area(gts[j]))
            rows.append({
                "stem": stem, "iou": iou,
                "pred_area": float(pred.sum()), "gt_area": gt_area,
                "area_ratio": float(pred.sum()) / max(gt_area, 1.0),
            })
        if (k + 1) % 20 == 0:
            print(f"{k+1}/{len(npz_files)} pairs={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    bands = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    summary = {}
    for lo, hi in bands:
        sub = df[(df.iou >= lo) & (df.iou < hi)]
        if len(sub):
            summary[f"iou_{lo}_{hi}"] = {
                "n": len(sub),
                "area_ratio_median": round(float(sub.area_ratio.median()), 3),
                "area_ratio_p25": round(float(sub.area_ratio.quantile(.25)), 3),
                "area_ratio_p75": round(float(sub.area_ratio.quantile(.75)), 3),
            }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out / f"nearmiss_area_bias_fold{args.fold}.csv", index=False)
        (args.out / f"nearmiss_area_bias_fold{args.fold}.json").write_text(
            json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
