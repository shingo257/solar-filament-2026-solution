"""V30 YOLO の全fold OOF を公式 Panoptic Quality で採点し直す (GPU不要・学習なし)。

優先度1: PQ で既存 OOF を再採点し、conf 動作点を引き直す
優先度2: 廃止した Dice ゲート `0.552553` を PQ 基準へ置き換える
優先度3: duplicate 発生率をアノテータ数で層別し、主因が学習ラベル方針か判定する

V34 Drive の OOF cache (2048x2048 のまま packbits 保存) をそのまま読む。
npz は画像あたり1回だけ読み、全 conf 候補を同一ロードから評価する。
IoU は mask を配列展開せず pycocotools の RLE 経路で計算する。

使い方:
    python scratch/run_pq_oof_rescore.py [--limit N] [--conf 0.08 0.12 ...]
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
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics_official import PanopticQuality, marginal_prediction_threshold  # noqa: E402

DRIVE = Path("G:/マイドライブ")
V30 = DRIVE / "solar_filament_v30"
V34 = DRIVE / "solar_filament_v34"
TRAIN_JSON = V30 / "competition_data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FOLDS_CSV = V30 / "artifacts/grouped_folds.csv"
OOF_CACHE = V34 / "prediction_cache/oof"

EVAL_SIZE = 2048
MIN_AREA = 20
DUP_CONF = 0.26


def to_rle(annotation: dict, height: int, width: int) -> dict:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_util.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def masks_to_rles(masks: list[np.ndarray]) -> list[dict]:
    if not masks:
        return []
    stacked = np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8))
    return mask_util.encode(stacked)


def iou_matrix(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    matrix = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    return np.asarray(matrix, dtype=np.float64).T


def load_yolo_cache(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    masks = [np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed]
    return masks, data["confidences"].astype(np.float32)


def make_disjoint(masks: list[np.ndarray], confidences: np.ndarray,
                  conf: float, min_area: int) -> list[np.ndarray]:
    """confidence 降順の先着排他化。V30b と同じ方式。"""
    if len(confidences) == 0:
        return []
    occupied = np.zeros((EVAL_SIZE, EVAL_SIZE), dtype=bool)
    kept = []
    for index in np.argsort(-confidences):
        if confidences[index] < conf:
            continue
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="先頭N観測だけ処理 (動作確認用)")
    parser.add_argument("--conf", type=float, nargs="+",
                        default=[0.08, 0.12, 0.16, 0.20, 0.26, 0.35, 0.45])
    parser.add_argument("--out", type=Path, default=Path("output/v35"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"読み込み: {TRAIN_JSON.name}")
    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
    annotations: dict = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    file_to_ids: dict = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)

    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    folds = sorted(set(fold_of_file.values()))

    observations = sorted(file_to_ids)
    missing_fold = [n for n in observations if n not in fold_of_file]
    assert not missing_fold, f"fold未割当が{len(missing_fold)}件: {missing_fold[:5]}"

    gt_rles: dict = {}
    for image_id, anns in annotations.items():
        info = images[image_id]
        gt_rles[image_id] = [to_rle(a, int(info["height"]), int(info["width"])) for a in anns]

    available = [n for n in observations
                 if (OOF_CACHE / f"yolo_fold_{fold_of_file[n]}" / f"{Path(n).stem}.npz").exists()]
    if args.limit:
        available = available[:args.limit]
    print({"観測": len(observations), "OOF cacheあり": len(available),
           "複数アノテータ": sum(len(file_to_ids[n]) > 1 for n in available)})

    overall = {conf: PanopticQuality() for conf in args.conf}
    by_fold = {conf: {f: PanopticQuality() for f in folds} for conf in args.conf}
    near_miss = {conf: 0 for conf in args.conf}
    pred_counts = {conf: [] for conf in args.conf}
    duplicate_rows = []

    started = time.perf_counter()
    for position, name in enumerate(available, 1):
        fold = fold_of_file[name]
        masks, confidences = load_yolo_cache(
            OOF_CACHE / f"yolo_fold_{fold}" / f"{Path(name).stem}.npz")
        variants = [gt_rles[i] for i in file_to_ids[name]]

        for conf in args.conf:
            predictions = make_disjoint(masks, confidences, conf, MIN_AREA)
            pred_rles = masks_to_rles(predictions)
            pred_counts[conf].append(len(predictions))
            for truth in variants:
                matrix = iou_matrix(truth, pred_rles)
                overall[conf].add_matrix(matrix, len(truth), len(pred_rles))
                by_fold[conf][fold].add_matrix(matrix, len(truth), len(pred_rles))
                if matrix.size:
                    near_miss[conf] += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

        # 優先度3: 排他化する前の生候補で重複構造を測る
        raw = [m for m, s in zip(masks, confidences) if s >= DUP_CONF and m.sum() >= MIN_AREA]
        raw_rles = masks_to_rles(raw)
        contained = overlapping = 0
        if len(raw_rles) > 1:
            self_iou = iou_matrix(raw_rles, raw_rles)
            areas = np.asarray(mask_util.area(raw_rles), dtype=np.float64)
            for i in range(len(raw_rles)):
                for j in range(i + 1, len(raw_rles)):
                    value = self_iou[i, j]
                    if value <= 0:
                        continue
                    overlapping += 1
                    intersection = value * (areas[i] + areas[j]) / (1.0 + value)
                    if intersection / max(min(areas[i], areas[j]), 1.0) >= 0.90:
                        contained += 1
        duplicate_rows.append({
            "file_name": name, "fold": fold,
            "n_annotators": len(file_to_ids[name]),
            "raw_proposals": len(raw),
            "overlapping_pairs": overlapping,
            "contained_pairs": contained,
            "gt_count_mean": float(np.mean([len(v) for v in variants])),
        })

        if position % 50 == 0 or position == len(available):
            elapsed = time.perf_counter() - started
            reference = args.conf[len(args.conf) // 2]
            print(f"  {position}/{len(available)}  {elapsed:.0f}s  "
                  f"PQ(conf={reference})={overall[reference].score:.4f}", flush=True)

    summary = pd.DataFrame({
        conf: {
            "PQ": overall[conf].summary()["pq"],
            "SQ": overall[conf].summary()["sq"],
            "RQ": overall[conf].summary()["rq"],
            "TP": overall[conf].summary()["tp"],
            "FP": overall[conf].summary()["fp"],
            "FN": overall[conf].summary()["fn"],
            "near_miss": near_miss[conf],
            "pred/img": float(np.mean(pred_counts[conf])),
        } for conf in args.conf
    }).T
    summary.index.name = "conf"
    print("\n=== 全fold OOF Panoptic Quality ===")
    print(summary.round(4).to_string())

    fold_table = pd.DataFrame({
        conf: {f"fold{f}": by_fold[conf][f].score for f in folds} for conf in args.conf
    }).T
    fold_table.index.name = "conf"
    print("\n=== fold別 PQ ===")
    print(fold_table.round(4).to_string())

    best_conf = max(args.conf, key=lambda c: overall[c].score)
    best = overall[best_conf].summary()
    worst_fold = min(by_fold[best_conf][f].score for f in folds)

    duplicates = pd.DataFrame(duplicate_rows)
    duplicates["contained_ratio"] = (
        duplicates["contained_pairs"] / duplicates["overlapping_pairs"].clip(lower=1))
    duplicates.to_csv(args.out / "duplicate_stratification.csv", index=False)
    grouped = duplicates.assign(multi=duplicates["n_annotators"] > 1).groupby("multi")[
        ["raw_proposals", "overlapping_pairs", "contained_pairs", "contained_ratio"]]
    print("\n=== 優先度3: duplicate層別 (False=単一アノテータ, True=複数) ===")
    print(grouped.agg(["mean", "count"]).round(4).to_string())

    verdict = "検定不能"
    p_value = None
    try:
        from scipy.stats import mannwhitneyu
        single = duplicates.loc[duplicates["n_annotators"] == 1, "overlapping_pairs"]
        multi = duplicates.loc[duplicates["n_annotators"] > 1, "overlapping_pairs"]
        statistic, p_value = mannwhitneyu(multi, single, alternative="greater")
        verdict = "支持" if p_value < 0.01 else "不支持"
        print(f"\nMann-Whitney U (複数 > 単一): U={statistic:.0f}, p={p_value:.3e} → 仮説を{verdict}")
    except Exception as error:  # noqa: BLE001
        print(f"検定失敗: {error}")

    gate = {
        "metric": "Panoptic Quality (公式self-evaluation notebook準拠)",
        "baseline_source": "V30 YOLO 3-fold OOF (V34 cache再利用, 学習なし)",
        "observations_scored": len(available),
        "best_conf": best_conf,
        "measured_all_fold_pq": round(best["pq"], 6),
        "measured_sq": round(best["sq"], 6),
        "measured_rq": round(best["rq"], 6),
        "counts": {"tp": best["tp"], "fp": best["fp"], "fn": best["fn"]},
        "per_fold_pq": {int(f): round(by_fold[best_conf][f].score, 6) for f in folds},
        "new_gate_pq": round(best["pq"] + 0.010, 6),
        "new_gate_worst_fold_pq": round(worst_fold, 6),
        "marginal_tp_probability": round(
            marginal_prediction_threshold(best["pq"], best["sq"] or 0.7), 4),
        "retired_gate": {"value": 0.5525530052850139,
                         "reason": "公式でないHungarian Dice、かつfold0のみの探索best"},
        "duplicate_hypothesis": {
            "statement": "複数アノテータを独立教師にしたことが系統的duplicateの主因",
            "p_value_multi_greater_than_single": None if p_value is None else float(p_value),
            "verdict": verdict,
        },
    }
    summary.to_csv(args.out / "pq_conf_sweep.csv")
    fold_table.to_csv(args.out / "pq_by_fold.csv")
    (args.out / "pq_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"\n所要 {time.perf_counter() - started:.0f}s / 保存先 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
