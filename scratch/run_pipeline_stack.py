"""refiner と排他化方式の組み合わせを全fold OOF の PQ で確定する (CPU、GPU不要)。

単独計測の利得は足し算になるとは限らない。

  - ownership の +0.0069 は **生の YOLO マスク** で測った値。refine 後のマスクは
    縮んで重なりが減るので、同じ利得が残るとは限らない。
  - conf の最適点 0.26 も **refiner なし** で測った値。refiner は FP を減らすので、
    最適な動作点が下へ動く可能性がある (候補を増やす余裕が生まれる)。

したがって提出構成を決める前に、refiner × ownership × conf を同時に測る。

リーク管理 (§18.2): 各観測へ、その fold を学習から除いた refiner だけを適用する。

使い方:
    python scratch/run_pipeline_stack.py [--threads 14] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

CHECKPOINT_EVERY = 25   # 観測25件ごとに保存。PC再起動時の損失をこの件数分に抑える

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS, FOLDS, FOLDS_CSV, MIN_AREA, OOF_CACHE, PREPROC, TRAIN_JSON,
    build_variant, greedy_disjoint, imread_unicode, iou_matrix,
    masks_to_rles, refine_batch, to_rle,
)


def load_yolo_cache_with_retry(path: Path, attempts: int = 4, backoff: float = 2.0):
    """Google Drive のネットワークマウントは瞬断することがある (OSError errno 22)。

    707観測の全件走査を1件のI/Oエラーで失うのは高くつくので、待機を挟んで再試行する。
    """
    last_error = None
    for attempt in range(attempts):
        try:
            data = np.load(path)
            _, width = map(int, data["shape"])
            packed = data["packed"]
            if packed.size == 0:
                return [], np.zeros(0, np.float32)
            return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
                    data["confidences"].astype(np.float32))
        except OSError as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"{path} の読み込みに{attempts}回失敗した: {last_error}")

REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45          # 全fold OOF での最良 (0.45〜0.65 は実質フラット)
CONF_GRID = (0.16, 0.20, 0.26, 0.32)
OWNERSHIPS = ("conf_first", "area_asc", "drop_contained_then_area_asc")
CONTAINMENT = 0.90


def drop_contained_indices(masks, rles=None, threshold=CONTAINMENT):
    """他のmaskへ threshold 以上包含された小さい方を落とす。縮めない。"""
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = rles if rles is not None else masks_to_rles(masks)
    areas = np.asarray(mask_util.area(rles), dtype=np.float64)
    self_iou = iou_matrix(rles, rles)
    dropped: set = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped:
                continue
            value = self_iou[i, j]
            if value <= 0:
                continue
            intersection = value * (areas[i] + areas[j]) / (1.0 + value)
            smaller = i if areas[i] <= areas[j] else j
            if intersection / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]


def imread_unicode_with_retry(path: Path, attempts: int = 4, backoff: float = 2.0):
    last_error = None
    for attempt in range(attempts):
        try:
            return imread_unicode(path)
        except OSError as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"{path} の読み込みに{attempts}回失敗した: {last_error}")


def save_checkpoint(path: Path, state: dict) -> None:
    """一時ファイル→アトミックリネームで保存する。

    保存の途中でPCが落ちても、リネーム前の一時ファイルが壊れているだけで
    直前の正常なチェックポイントは無傷のまま残る。
    """
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def apply_ownership(masks, scores, rule):
    """排他化の規則ごとに occupancy 順序 (と drop) を決めて disjoint 化する。"""
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    if rule == "conf_first":
        return greedy_disjoint(masks, list(np.argsort(-scores)))
    if rule == "area_asc":
        return greedy_disjoint(masks, list(np.argsort(areas)))
    if rule == "drop_contained_then_area_asc":
        subset = drop_contained_indices(masks)
        return greedy_disjoint(masks, sorted(subset, key=lambda i: areas[i]))
    raise ValueError(rule)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("output/v35"))
    parser.add_argument("--fresh", action="store_true",
                        help="既存チェックポイントを無視して最初からやり直す")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    checkpoint_path = args.out / "pipeline_stack_checkpoint.pkl"

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
    targets = [n for n in sorted(file_to_ids)
               if (OOF_CACHE / f"yolo_fold_{fold_of_file[n]}" / f"{Path(n).stem}.npz").exists()]
    if args.limit:
        targets = targets[:args.limit]

    refiners = {}
    for fold in FOLDS:
        checkpoint = torch.load(ARTIFACTS / f"mediseg_refiner_fold{fold}_resume.pt",
                                map_location="cpu")
        model = MediSegUNet(3, 1, False).eval()
        model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)
        refiners[fold] = model

    combos = [(c, o) for c in CONF_GRID for o in OWNERSHIPS]
    keys = [f"conf{c:.2f}|{o}" for c, o in combos]
    # 参照: refiner なし・現行構成 (conf 0.26 + conf_first)
    keys.append("BASELINE_yolo_conf0.26_conf_first")
    conf_floor = min(CONF_GRID)

    # --- チェックポイントからの再開 ---
    # PC再起動でジョブが落ちても、直近 CHECKPOINT_EVERY 件分の損失だけで再開できる。
    checkpoint = None if args.fresh else load_checkpoint(checkpoint_path)
    if checkpoint is not None and checkpoint.get("targets_signature") == (len(targets), keys):
        overall = checkpoint["overall"]
        by_fold = checkpoint["by_fold"]
        near_miss = checkpoint["near_miss"]
        counts = checkpoint["counts"]
        skipped = checkpoint["skipped"]
        start_position = checkpoint["position"]
        elapsed_before = checkpoint["elapsed_seconds"]
        print(f"チェックポイントから再開: {start_position}/{len(targets)}件 完了済み "
              f"(これまでの所要 {elapsed_before/60:.1f}分)")
    else:
        if checkpoint is not None:
            print("チェックポイントの設定 (観測数・構成) が現在の設定と一致しないため、"
                  "最初からやり直す")
        overall = {k: PanopticQuality() for k in keys}
        by_fold = {k: {f: PanopticQuality() for f in FOLDS} for k in keys}
        near_miss = {k: 0 for k in keys}
        counts = {k: [] for k in keys}
        skipped = []
        start_position = 0
        elapsed_before = 0.0

    print({"観測": len(targets), "refiner": f"{REFINER_METHOD}@{REFINER_THRESHOLD}",
           "conf掃引": CONF_GRID, "排他化": OWNERSHIPS, "threads": args.threads,
           "再開位置": start_position})

    started = time.perf_counter()
    for position, name in enumerate(targets, 1):
        if position <= start_position:
            continue
        fold = fold_of_file[name]
        masks, confidences = load_yolo_cache_with_retry(
            OOF_CACHE / f"yolo_fold_{fold}" / f"{Path(name).stem}.npz")
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= conf_floor and m.sum() >= MIN_AREA]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
        variants_gt = [gt_rles[i] for i in file_to_ids[name]]

        image = imread_unicode_with_retry(PREPROC / name)
        if image is None:
            skipped.append(name)
            if len(skipped) > max(3, len(targets) // 20):
                raise RuntimeError(f"前処理画像を読めない観測が{len(skipped)}件: {skipped[:3]}")
            continue

        # refiner確率は conf_floor 以上の候補について1度だけ計算して使い回す
        if proposals:
            boxes, probabilities = refine_batch(refiners[fold], image, proposals)
            refined = [build_variant(proposals[i], boxes[i], probabilities[i],
                                     REFINER_METHOD, REFINER_THRESHOLD)
                       for i in range(len(proposals))]
        else:
            refined = []

        def score_set(key, predictions):
            rles = masks_to_rles(predictions)
            counts[key].append(len(predictions))
            for truth in variants_gt:
                matrix = iou_matrix(truth, rles)
                overall[key].add_matrix(matrix, len(truth), len(rles))
                by_fold[key][fold].add_matrix(matrix, len(truth), len(rles))
                if matrix.size:
                    near_miss[key] += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

        for conf, rule in combos:
            index = [i for i in range(len(refined))
                     if scores[i] >= conf and refined[i].sum() >= MIN_AREA]
            score_set(f"conf{conf:.2f}|{rule}",
                      apply_ownership([refined[i] for i in index], scores[index], rule))

        index = [i for i in range(len(proposals)) if scores[i] >= 0.26]
        score_set("BASELINE_yolo_conf0.26_conf_first",
                  apply_ownership([proposals[i] for i in index], scores[index], "conf_first"))

        if position % 25 == 0 or position == len(targets):
            best_now = max(keys, key=lambda k: overall[k].score)
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                  f"best={best_now} {overall[best_now].score:.4f}", flush=True)

        if position % CHECKPOINT_EVERY == 0 or position == len(targets):
            save_checkpoint(checkpoint_path, {
                "position": position,
                "targets_signature": (len(targets), keys),
                "overall": overall,
                "by_fold": by_fold,
                "near_miss": near_miss,
                "counts": counts,
                "skipped": skipped,
                "elapsed_seconds": elapsed_before + (time.perf_counter() - started),
            })

    table = pd.DataFrame({
        k: {**{n: overall[k].summary()[n] for n in ("pq", "sq", "rq", "tp", "fp", "fn")},
            **{f"pq_fold{f}": by_fold[k][f].score for f in FOLDS},
            "near_miss": near_miss[k], "inst/img": float(np.mean(counts[k]))}
        for k in keys
    }).T
    table.index.name = "config"
    baseline = table.loc["BASELINE_yolo_conf0.26_conf_first", "pq"]
    table["ΔPQ_vs_baseline"] = table["pq"] - baseline
    table["worst_fold"] = table[[f"pq_fold{f}" for f in FOLDS]].min(axis=1)
    table = table.sort_values("pq", ascending=False)
    print(f"\n=== 全fold OOF: refiner x 排他化 x conf ({len(targets) - len(skipped)}観測) ===")
    print(table.round(4).to_string())

    best = table.index[0]
    table.to_csv(args.out / "pipeline_stack.csv")
    report = {
        "scope": f"全fold OOF ({len(targets) - len(skipped)}観測)。リークなし。",
        "refiner": f"{REFINER_METHOD}@{REFINER_THRESHOLD}",
        "baseline_config": "生YOLO + conf0.26 + conf_first (V30b相当)",
        "baseline_pq": float(baseline),
        "best_config": str(best),
        "best_pq": float(table.loc[best, "pq"]),
        "best_worst_fold_pq": float(table.loc[best, "worst_fold"]),
        "total_gain": float(table.loc[best, "ΔPQ_vs_baseline"]),
        "refiner_only_gain": float(
            table.loc["conf0.26|conf_first", "pq"] - baseline),
        "ownership_gain_on_top_of_refiner": float(
            table.loc["conf0.26|drop_contained_then_area_asc", "pq"]
            - table.loc["conf0.26|conf_first", "pq"]),
        "ownership_gain_on_raw_yolo_was": 0.0069,
        "new_gate_pq": round(float(table.loc[best, "pq"]) + 0.010, 6),
        "annotator_ceiling_pq": 0.7587,
        "table": json.loads(table.round(6).to_json(orient="index")),
    }
    (args.out / "pipeline_stack.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "table"},
                            ensure_ascii=False, indent=2))
    total_elapsed = elapsed_before + (time.perf_counter() - started)
    print(f"\n所要 {total_elapsed:.0f}s (今回のみ {time.perf_counter() - started:.0f}s)")
    print(f"完了。チェックポイント {checkpoint_path} は残してある"
          f"（再実行時は --fresh を付けない限り自動的にスキップされる）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
