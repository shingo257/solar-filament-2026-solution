"""アノテータ間の不一致が課す Panoptic Quality の天井を測る (GPU不要・モデル不要)。

公式PQは (annotator, image) ごとに同じ予測集合を突き合わせ、TP/FP/FN を全体へ
足し込む。したがって「あるアノテータの GT をそのまま提出した」場合ですら満点には
ならない。その値がこのデータで到達しうる PQ の実質的な上限になる。

競技のCOCO annotation だけを読む。学習も推論もしない。マスクは decode せず
pycocotools の RLE 同士の IoU を使うので、2048x2048 を配列展開しない。

使い方:
    python scratch/run_annotator_ceiling.py [--json PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics_official import PanopticQuality  # noqa: E402

DEFAULT_JSON = Path(
    "G:/マイドライブ/solar_filament_v30/competition_data/MAGFiLO_1.0_Kaggle_2026"
    "/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)


def to_rle(annotation: dict, height: int, width: int) -> dict:
    """COCO の segmentation を compressed RLE へ正規化する。"""
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_util.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def iou_matrix(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    """(n_gt, n_pred) の IoU 行列。RLE のまま計算するので展開コストがない。"""
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    # pycocotools: iou(dt, gt, iscrowd) -> (len(dt), len(gt))
    matrix = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    return np.asarray(matrix, dtype=np.float64).T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=Path("output/v35/annotator_ceiling.json"))
    args = parser.parse_args()

    print(f"読み込み: {args.json}")
    with open(args.json, encoding="utf-8") as handle:
        coco = json.load(handle)

    images = {item["id"]: item for item in coco["images"]}
    annotations: dict = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    file_to_ids: dict = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)

    observations = sorted(file_to_ids)
    multi = [name for name in observations if len(file_to_ids[name]) > 1]
    print({
        "観測画像": len(observations),
        "複数アノテータ観測": len(multi),
        "annotation総数": len(coco["annotations"]),
    })

    # 各 annotator-image の RLE を一度だけ作る
    print("RLE正規化中...")
    rles_by_id: dict = {}
    for index, (image_id, anns) in enumerate(annotations.items(), 1):
        info = images[image_id]
        height, width = int(info["height"]), int(info["width"])
        rles_by_id[image_id] = [to_rle(ann, height, width) for ann in anns]
        if index % 200 == 0 or index == len(annotations):
            print(f"  {index}/{len(annotations)} annotator-image", flush=True)

    # 参照アノテータの選び方ごとに天井を測る
    max_annotators = max(len(ids) for ids in file_to_ids.values())
    results = {}
    cross_ious: list[float] = []
    agreement_hits = agreement_pairs = 0

    for reference_index in range(max_annotators):
        accumulator = PanopticQuality()
        used = 0
        for position, name in enumerate(observations, 1):
            ids = file_to_ids[name]
            if reference_index >= len(ids):
                continue  # この観測には参照にできるアノテータがいない
            used += 1
            prediction = rles_by_id[ids[reference_index]]
            for image_id in ids:
                truth = rles_by_id[image_id]
                matrix = iou_matrix(truth, prediction)
                accumulator.add_matrix(matrix, len(truth), len(prediction))
                if reference_index == 0 and image_id != ids[0] and matrix.size:
                    best = matrix.max(axis=1)
                    cross_ious.extend(best.tolist())
                    agreement_hits += int((best > 0.5).sum())
                    agreement_pairs += int(best.size)
            if position % 200 == 0:
                print(f"  ref={reference_index} {position}/{len(observations)}"
                      f" PQ={accumulator.score:.4f}", flush=True)
        summary = accumulator.summary()
        summary["observations_used"] = used
        results[f"reference_annotator_{reference_index}"] = summary
        print(f"参照アノテータ{reference_index}: PQ={summary['pq']:.4f} "
              f"(SQ={summary['sq']:.4f} RQ={summary['rq']:.4f}, {used}観測)")

    ceiling = results["reference_annotator_0"]["pq"]
    report = {
        "purpose": "アノテータ不一致が課すPQの天井 (GTのみ、モデル非依存)",
        "metric": "Panoptic Quality (公式self-evaluation notebook準拠)",
        "source_json": str(args.json),
        "observations": len(observations),
        "multi_annotator_observations": len(multi),
        "total_annotations": len(coco["annotations"]),
        "ceilings": results,
        "primary_ceiling_pq": ceiling,
        "inter_annotator": {
            "compared_gt_instances": agreement_pairs,
            "agree_at_iou_gt_0.5": agreement_hits,
            "agreement_rate": (agreement_hits / agreement_pairs) if agreement_pairs else None,
            "best_iou_mean": float(np.mean(cross_ious)) if cross_ious else None,
            "best_iou_median": float(np.median(cross_ious)) if cross_ious else None,
        },
        "context": {
            "own_best_public_pq": 0.35,
            "leaderboard_top_snapshot": 0.48,
            "note": "LBスナップショットは2026-08-12の再採点以前の可能性あり",
        },
    }
    if ceiling > 0:
        report["context"]["own_best_as_fraction_of_ceiling"] = round(0.35 / ceiling, 4)
        report["context"]["leaderboard_top_as_fraction_of_ceiling"] = round(0.48 / ceiling, 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
