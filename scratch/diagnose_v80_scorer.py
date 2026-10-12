"""v80 スコアラー診断 — û と真 u のソース別ランク品質を測る。

背景: run_v80_scorer_eval.py の全fold評価で hybrid_oracle=0.5493 vs
hybrid=0.4284 と 0.12 の差があり、ボトルネックは候補集合ではなく
û 予測にあることが示唆された。このスクリプトは同じ教師・cross-fit
HistGBM を再現し、yolo / tree ソース別に以下を報告する:

- Spearman ρ(û, u): 全候補 / u>0 正例内
- ROC-AUC(ĥ 相当 = û, ラベル h>0)
- u 上位候補の û ランク分布(mean percentile)

使い方:
    python scratch/diagnose_v80_scorer.py --drive "G:/マイドライブ" [--limit 200]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_v80_scorer_eval import (  # noqa: E402
    CONTEXT_FEATURES,
    FOLDS,
    NA_FEATURES,
    context_features,
    load_candidates,
    rle_dicts,
    teachers_for,
)
from solar_filament.pipeline.paths import (  # noqa: E402
    grouped_folds_csv,
    resolve_drive,
    train_annotations_json,
)
from run_refiner_pq_allfold import to_rle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    drive = resolve_drive(args.drive)
    cache_root = args.cache or drive / "solar_filament_v80" / "candidates"

    with open(train_annotations_json(drive), encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
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
    fold_frame = pd.read_csv(grouped_folds_csv(drive))
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    stem_to_file = {Path(n).stem: n for n in file_to_ids}

    gen_src = (Path(__file__).parent / "run_v80_hybrid_candidates.py").read_text(
        encoding="utf-8")
    feature_names = next(
        ast.literal_eval(node.value)
        for node in ast.parse(gen_src).body
        if isinstance(node, ast.Assign) and node.targets[0].id == "FEATURE_NAMES"
    )
    all_features = list(feature_names) + list(CONTEXT_FEATURES)
    na_index = [all_features.index(n) for n in NA_FEATURES if n in all_features]

    targets = []
    for fold in FOLDS:
        fold_dir = cache_root / f"fold_{fold}"
        if not fold_dir.exists():
            continue
        targets += [
            (fold, p.stem) for p in sorted(fold_dir.glob("*.npz"))
            if p.stem in stem_to_file and fold_of_file.get(stem_to_file[p.stem]) == fold
        ]
    if args.limit:
        # fold ごとに均等に間引く(先頭 N 件が全て fold0 になると cross-fit の
        # held-out 予測が得られず診断不能になる)
        per_fold = max(1, args.limit // len(FOLDS))
        by_fold = defaultdict(list)
        for t in targets:
            by_fold[t[0]].append(t)
        targets = [t for fold in FOLDS for t in by_fold[fold][:per_fold]]
    print({"観測": len(targets), "特徴量": len(all_features)}, flush=True)

    rows = []
    started = time.perf_counter()
    for position, (fold, stem) in enumerate(targets, 1):
        name = stem_to_file[stem]
        cand = load_candidates(cache_root / f"fold_{fold}" / f"{stem}.npz")
        pred_rles = rle_dicts(cand["rle_final"])
        records = [gt_rles[i] for i in file_to_ids[name]]
        u, h = teachers_for(pred_rles, records)
        ctx = context_features(cand, pred_rles, len(records))
        X = np.concatenate(
            [cand["feats"].astype(np.float64), ctx.astype(np.float64)], axis=1
        )
        X[:, na_index] = np.where(X[:, na_index] < 0, np.nan, X[:, na_index])
        for i in range(len(pred_rles)):
            rows.append((u[i], h[i], float(len(records)), X[i],
                         int(cand["source"][i]), fold))
        if position % 50 == 0 or position == len(targets):
            print(f"  load {position}/{len(targets)} {time.perf_counter()-started:.0f}s",
                  flush=True)

    u_all = np.asarray([r[0] for r in rows])
    h_all = np.asarray([r[1] for r in rows])
    w_all = np.asarray([r[2] for r in rows])
    X_all = np.stack([r[3] for r in rows])
    src_all = np.asarray([r[4] for r in rows])
    fold_rows = np.asarray([r[5] for r in rows])
    positives = h_all > 0

    u_hat = np.zeros(len(rows))
    params = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                  random_state=args.seed)
    for fold in FOLDS:
        train = fold_rows != fold
        test = fold_rows == fold
        if not train.any() or not test.any():
            continue
        model_h = HistGradientBoostingRegressor(**params).fit(
            X_all[train], h_all[train], sample_weight=w_all[train])
        pos = train & positives
        if not pos.any():
            continue
        model_m = HistGradientBoostingRegressor(**params).fit(
            X_all[pos], u_all[pos] / np.maximum(h_all[pos], 1e-6),
            sample_weight=w_all[pos])
        u_hat[test] = model_h.predict(X_all[test]) * model_m.predict(X_all[test])

    report = {"observations": len(targets), "rows": len(rows)}
    for src, label in ((0, "yolo"), (1, "tree"), (-1, "all")):
        mask = np.ones(len(rows), bool) if src == -1 else src_all == src
        uu, pp = u_hat[mask], u_all[mask]
        hh = h_all[mask]
        entry = {
            "n": int(mask.sum()),
            "u>0率": float((uu.size and (u_all[mask] > 0).mean()) or 0),
            "spearman_all": float(spearmanr(uu, pp).statistic) if mask.sum() > 2 else None,
            "roc_auc_h>0": float(roc_auc_score(hh > 0, uu))
            if 0 < (hh > 0).sum() < mask.sum() else None,
        }
        pos = pp > 0
        if pos.sum() > 2:
            entry["spearman_pos"] = float(spearmanr(uu[pos], pp[pos]).statistic)
            # 真の良候補(u>0.3)が û ランクの何パーセンタイルに来るか
            good = pp >= 0.3
            if good.sum() > 0:
                ranks = rankdata(uu) / len(uu)
                entry["mean_pct_rank_of_u>=0.3"] = float(ranks[good].mean())
        report[label] = entry

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
