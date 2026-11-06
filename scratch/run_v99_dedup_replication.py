"""v99 — dedup030 受理規則の fold0/fold1 独立再現 (RESEARCH_LOG §2.75, 事前登録)。

v80 candidates cache (`solar_filament_v80/candidates/fold_{f}/<stem>.npz`) の
yolo 候補だけを使い、v97 の受理規則 (fold 内 halfcross isotonic → cal 降順の
raw 提案 mask-IoU dedup → cal>λ) を prod と paired で採点する。学習なし・CPU のみ。

使い方:
    python scratch/run_v99_dedup_replication.py [--folds 0 1 2] [--workers 8] [--boot 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))

from solar_filament.pipeline.ownership import (  # noqa: E402
    drop_contained_indices,
    greedy_disjoint,
)
from solar_filament.pipeline.paths import (  # noqa: E402
    grouped_folds_csv,
    resolve_drive,
    train_annotations_json,
)
from run_refiner_pq_allfold import MIN_AREA, to_rle  # noqa: E402
from run_v80_scorer_eval_impl import (  # noqa: E402
    CONTAINMENT,
    PROD_CONF,
    _encode,
    load_candidates,
    pq_of,
    rle_dicts,
    teachers_for,
)
from src.metrics_official import PQ_IOU_THRESHOLD  # noqa: E402

PRIMARY_DEDUP = 0.30
DIAG_DEDUPS = (0.10, 0.20, 0.50)
GATE_FOLDS = (0, 1)


def crossfit_halves(files: list[str]) -> tuple[list[str], list[str]]:
    """v97 と同一の deterministic 2 分割 (sorted index parity)。"""
    srt = sorted(files)
    return srt[0::2], srt[1::2]


def dedup_accept(prop_rles: list[dict], cal: np.ndarray, conf: np.ndarray,
                 dedup_iou: float) -> list[int]:
    """cal 降順 (同値は conf 降順) に貪欲受理し、受理済みと raw 提案 mask-IoU が
    dedup_iou を超える候補を破棄する (v97 `dedup_accept` と同じ規則)。"""
    order = sorted(range(len(prop_rles)), key=lambda i: (-cal[i], -conf[i]))
    accepted: list[int] = []
    for i in order:
        if accepted:
            ious = np.asarray(mask_util.iou(
                [prop_rles[i]], [prop_rles[a] for a in accepted], [0] * len(accepted)))
            if (ious > dedup_iou).any():
                continue
        accepted.append(i)
    return accepted


def score_keep(final_rles: list[dict], keep: list[int],
               records: list[list[dict]]) -> tuple[float, int, int, int]:
    """keep の refined mask を prod と同一 ownership に通し公式 PQ 寄与を返す。"""
    subset_rles = [final_rles[i] for i in keep]
    subset = [mask_util.decode(r).astype(bool) for r in subset_rles]
    alive = drop_contained_indices(subset, threshold=CONTAINMENT, rles=subset_rles)
    areas = np.array([m.sum() for m in subset])
    preds = greedy_disjoint(subset, sorted(alive, key=lambda i: areas[i]), min_area=MIN_AREA)
    pred_rles = _encode(preds)
    tp_sum = 0.0
    tp_c = fp_c = fn_c = 0
    for gt in records:
        if not pred_rles:
            fn_c += len(gt)
            continue
        if not gt:
            fp_c += len(pred_rles)
            continue
        matrix = np.asarray(mask_util.iou(pred_rles, gt, [0] * len(gt)), dtype=np.float64).T
        hits = matrix > PQ_IOU_THRESHOLD
        tp_sum += float(matrix[hits].sum())
        tp_c += int(hits.sum())
        fp_c += int((hits.sum(axis=0) == 0).sum())
        fn_c += int((hits.sum(axis=1) == 0).sum())
    return tp_sum, tp_c, fp_c, fn_c


def evaluate_image(bundle: dict) -> dict:
    """1観測の arm 別 PQ 寄与。bundle['cal'] が無ければ prod のみ。"""
    final_rles = rle_dicts(bundle["rle_final"])
    usable = np.asarray(bundle["usable"], dtype=bool)
    conf = np.asarray(bundle["conf"], dtype=np.float64)
    out = {"prod": score_keep(
        final_rles, list(np.where(usable & (conf >= PROD_CONF))[0]), bundle["records"])}
    if "cal" not in bundle:
        return out
    cal = np.asarray(bundle["cal"], dtype=np.float64)
    lam = float(bundle["lam"])
    prop_rles = rle_dicts(bundle["rle_prop"])
    out["nodedup"] = score_keep(
        final_rles, [i for i in range(len(conf)) if cal[i] > lam and usable[i]],
        bundle["records"])
    for d in (PRIMARY_DEDUP, *DIAG_DEDUPS):
        keep = [i for i in dedup_accept(prop_rles, cal, conf, d) if cal[i] > lam and usable[i]]
        out[f"dedup_{d:.2f}"] = score_keep(final_rles, keep, bundle["records"])
    return out


def run_pool(bundles: list[dict], workers: int, desc: str) -> list[dict]:
    if workers <= 1:
        return [evaluate_image(b) for b in tqdm(bundles, desc=desc)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(evaluate_image, bundles, chunksize=4),
                         total=len(bundles), desc=desc))


def paired_boot(counts: dict, arm: str, names: list[str], boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(boot):
        s = rng.choice(names, size=len(names), replace=True)
        deltas.append(pq_of([counts[n][arm] for n in s]) - pq_of([counts[n]["prod"] for n in s]))
    return np.percentile(deltas, [2.5, 97.5]).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("output/v99_dedup_replication"))
    parser.add_argument("--fixed-conf", type=float, default=None,
                        help="事後診断: isotonic の代わりに cal=conf・受理 conf>=値 で評価")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    drive = resolve_drive(args.drive)
    cache_root = args.cache or drive / "solar_filament_v80" / "candidates"
    with open(train_annotations_json(drive), encoding="utf-8") as handle:
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
                   for a in annotations.get(image_id, [])]
        for image_id in images
    }
    fold_csv = grouped_folds_csv(drive)
    fold_frame = pd.read_csv(fold_csv)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    stem_to_file = {Path(n).stem: n for n in file_to_ids}

    counts: dict[str, dict] = {}
    fold_files: dict[int, list[str]] = {}
    lambdas: dict[int, float] = {}
    effective_conf: dict[int, float | None] = {}
    for fold in args.folds:
        paths = sorted((cache_root / f"fold_{fold}").glob("*.npz"))
        paths = [p for p in paths
                 if p.stem in stem_to_file and fold_of_file.get(stem_to_file[p.stem]) == fold]
        if args.limit:
            paths = paths[: args.limit]
        bundles = {}
        for p in tqdm(paths, desc=f"load fold{fold}"):
            name = stem_to_file[p.stem]
            cand = load_candidates(p)
            yolo = np.where(np.asarray(cand["source"]).astype(int) == 0)[0]
            records = [gt_rles[i] for i in file_to_ids[name]]
            prop = [cand["rle_prop"][i] for i in yolo]
            hit_iou, _ = teachers_for(rle_dicts(prop), records)
            bundles[name] = {
                "rle_final": [cand["rle_final"][i] for i in yolo], "rle_prop": prop,
                "usable": np.asarray(cand["usable"])[yolo],
                "conf": np.asarray(cand["conf"], dtype=np.float64)[yolo],
                "hit_iou": hit_iou, "records": records,
            }
        names = sorted(bundles)
        fold_files[fold] = names

        prod_only = run_pool([{k: v for k, v in bundles[n].items() if k != "hit_iou"}
                              for n in names], args.workers, f"prod fold{fold}")
        lam = 0.5 * pq_of([r["prod"] for r in prod_only])
        lambdas[fold] = lam

        half_a, half_b = crossfit_halves(names)
        maps = {}
        for fit_half, apply_half in ((half_a, half_b), (half_b, half_a)):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(np.concatenate([bundles[n]["conf"] for n in fit_half]),
                    np.concatenate([bundles[n]["hit_iou"] for n in fit_half]))
            maps.update({n: iso for n in apply_half})
        work = []
        accepted_conf = []
        for n in names:
            b = {k: v for k, v in bundles[n].items() if k != "hit_iou"}
            if args.fixed_conf is not None:
                b["cal"], b["lam"] = b["conf"], np.nextafter(args.fixed_conf, -np.inf)
            else:
                b["cal"] = maps[n].predict(b["conf"]) if len(b["conf"]) else np.zeros(0)
                b["lam"] = lam
            accepted_conf += list(b["conf"][b["cal"] > b["lam"]])
            work.append(b)
        effective_conf[fold] = float(min(accepted_conf)) if accepted_conf else None
        for n, res in zip(names, run_pool(work, args.workers, f"arms fold{fold}")):
            counts[n] = res

    arms = list(next(iter(counts.values())).keys())
    table = {}
    for fold in args.folds:
        names = fold_files[fold]
        prod_pq = pq_of([counts[n]["prod"] for n in names])
        table[f"fold{fold}"] = {
            "n_obs": len(names), "lambda": lambdas[fold],
            "effective_min_accepted_conf": effective_conf[fold],
            **{arm: pq_of([counts[n][arm] for n in names]) for arm in arms},
            **{f"delta_{arm}": pq_of([counts[n][arm] for n in names]) - prod_pq
               for arm in arms if arm != "prod"},
            "tp_fp_fn": {arm: [sum(counts[n][arm][j] for n in names) for j in (1, 2, 3)]
                         for arm in arms},
        }
    primary = f"dedup_{PRIMARY_DEDUP:.2f}"
    gate_names = [n for f in GATE_FOLDS if f in fold_files for n in fold_files[f]]
    verdict = None
    if all(f in fold_files for f in GATE_FOLDS):
        ci = paired_boot(counts, primary, gate_names, args.boot, args.seed)
        deltas = [table[f"fold{f}"][f"delta_{primary}"] for f in GATE_FOLDS]
        replicated = all(d > 0 for d in deltas) and ci[0] > 0
        verdict = {
            "primary_arm": primary, "gate_folds": list(GATE_FOLDS), "fold_deltas": deltas,
            "pooled_delta": pq_of([counts[n][primary] for n in gate_names])
            - pq_of([counts[n]["prod"] for n in gate_names]),
            "pooled_ci95": ci, "verdict": "replicated" if replicated else "not_replicated",
        }
    report = {
        "experiment_id": "v99_dedup_replication", "protocol": "RESEARCH_LOG §2.75",
        "cache_root": str(cache_root), "folds_csv": str(fold_csv),
        "limit": args.limit, "boot": args.boot, "seed": args.seed,
        "fixed_conf": args.fixed_conf,
        "result_kind": "posthoc_diagnostic_not_gate" if args.fixed_conf is not None
        else "preregistered_gate",
        "per_fold": table, "gate": verdict,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    stem = "summary" if args.fixed_conf is None else f"posthoc_fixedconf_{args.fixed_conf:.3f}"
    out_path = args.out / (f"{stem}_smoke.json" if args.limit else f"{stem}.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
