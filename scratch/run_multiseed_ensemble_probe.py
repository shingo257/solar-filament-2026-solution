"""V35確定のfold0(seed3407)とV42(seed8407)のOOFを合算し、アンサンブルPQを測る。

検出器強化系(imgsz/augmentation/モデルサイズ)が6本連続不合格になったため、
検出器のハイパーパラメータは一切変えず、独立に学習した2本のseedの検出結果を
足し合わせるだけで精度が上がるか(ランダムな取りこぼしを均せるか)を見る。

やり方: 画像ごとに両seedのproposal(mask+confidence)を単純に連結し、
V35確定後処理(refiner shrink@0.45 + drop_contained_then_area_asc)にそのまま通す。
drop_containedがIoU>=0.90の重複を吸収するので、2seedがほぼ同じ個体を検出した
場合は自然に1個体分にまとまる。

使い方:
    python scratch/run_multiseed_ensemble_probe.py [--limit N] [--threads 14]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS, FOLDS_CSV, MIN_AREA, OOF_CACHE, PREPROC, TRAIN_JSON,
    build_variant, imread_unicode, iou_matrix, load_yolo_cache,
    masks_to_rles, refine_batch, to_rle,
)
from scratch.run_pipeline_stack import apply_ownership  # noqa: E402

CONF = 0.26
REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45
FOLD = 0
BASELINE_FOLD0_PQ = 0.4393
SEED2_CACHE_ENV = "SEED2_OOF_CACHE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument(
        "--seed2-cache", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "kaggle_downloads" / "v42" / "oof_cache" / "yolo_seed8407_fold_0",
        help="V42(seed8407)のOOF npzキャッシュディレクトリ",
    )
    parser.add_argument("--out", type=Path, default=Path("output/v42"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.seed2_cache.exists():
        print(f"seed2キャッシュが見つからない: {args.seed2_cache}")
        return 1

    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {i["id"]: i for i in coco["images"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    file_to_ids = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)
    gt_rles = {
        image_id: [to_rle(a, int(images[image_id]["height"]), int(images[image_id]["width"]))
                   for a in anns]
        for image_id, anns in annotations.items()
    }

    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    targets = [
        n for n in sorted(file_to_ids)
        if fold_of_file[n] == FOLD
        and (OOF_CACHE / f"yolo_fold_{FOLD}" / f"{Path(n).stem}.npz").exists()
        and (args.seed2_cache / f"{Path(n).stem}.npz").exists()
    ]
    if args.limit:
        targets = targets[:args.limit]
    print(f"観測数: {len(targets)}", flush=True)

    checkpoint = torch.load(
        ARTIFACTS / "mediseg_refiner_fold0_resume.pt", map_location="cpu", weights_only=False)
    refiner = MediSegUNet(3, 1, False).eval()
    refiner.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    meter_single = PanopticQuality()
    meter_ensemble = PanopticQuality()
    near_miss_single = 0
    near_miss_ensemble = 0
    started = time.perf_counter()

    for position, name in enumerate(targets, 1):
        variants = [gt_rles[i] for i in file_to_ids[name]]
        image = imread_unicode(PREPROC / name)
        if image is None:
            continue

        masks1, conf1 = load_yolo_cache(OOF_CACHE / f"yolo_fold_{FOLD}" / f"{Path(name).stem}.npz")
        masks2, conf2 = load_yolo_cache(args.seed2_cache / f"{Path(name).stem}.npz")

        def build_final(masks, confidences):
            keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                    if s >= CONF and m.sum() >= MIN_AREA]
            proposals = [masks[i] for i in keep]
            scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
            if not proposals:
                return []
            boxes, probabilities = refine_batch(refiner, image, proposals)
            refined = [build_variant(proposals[i], boxes[i], probabilities[i],
                                     REFINER_METHOD, REFINER_THRESHOLD)
                       for i in range(len(proposals))]
            return apply_ownership(refined, scores, "drop_contained_then_area_asc")

        single_final = build_final(masks1, conf1)
        single_rles = masks_to_rles(single_final)

        combined_masks = list(masks1) + list(masks2)
        combined_conf = np.concatenate([conf1, conf2]) if len(masks1) or len(masks2) else np.zeros(0, np.float32)
        ensemble_final = build_final(combined_masks, combined_conf)
        ensemble_rles = masks_to_rles(ensemble_final)

        for truth in variants:
            matrix_s = iou_matrix(truth, single_rles)
            meter_single.add_matrix(matrix_s, len(truth), len(single_rles))
            if matrix_s.size:
                near_miss_single += int(((matrix_s > 0.3) & (matrix_s <= 0.5)).any(axis=0).sum())

            matrix_e = iou_matrix(truth, ensemble_rles)
            meter_ensemble.add_matrix(matrix_e, len(truth), len(ensemble_rles))
            if matrix_e.size:
                near_miss_ensemble += int(((matrix_e > 0.3) & (matrix_e <= 0.5)).any(axis=0).sum())

        if position % 25 == 0 or position == len(targets):
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                  f"single={meter_single.score:.4f}  ensemble={meter_ensemble.score:.4f}", flush=True)

    summary_single = meter_single.summary()
    summary_ensemble = meter_ensemble.summary()
    report = {
        "scope": f"{len(targets)}観測、fold0のみ。V35確定(seed3407) vs seed3407+8407アンサンブル",
        "single_seed3407": {**summary_single, "near_miss": near_miss_single},
        "ensemble_seed3407_8407": {**summary_ensemble, "near_miss": near_miss_ensemble},
        "baseline_fold0_pq": BASELINE_FOLD0_PQ,
        "delta_ensemble_vs_baseline": round(summary_ensemble["pq"] - BASELINE_FOLD0_PQ, 6),
        "delta_ensemble_vs_single_local": round(summary_ensemble["pq"] - summary_single["pq"], 6),
        "continue_fold12": bool(summary_ensemble["pq"] > BASELINE_FOLD0_PQ),
    }
    (args.out / "multiseed_ensemble_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
