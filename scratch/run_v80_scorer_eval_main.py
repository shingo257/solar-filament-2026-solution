"""Entry-point `main()` for `run_v80_scorer_eval` — extracted verbatim from `run_v80_scorer_eval.py`.

Split out so both the wrapper and `run_v80_scorer_eval_impl.py` stay readable.
"""


from __future__ import annotations


import argparse
import ast
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))


from solar_filament.pipeline.paths import (  # noqa: E402
    grouped_folds_csv,
    resolve_drive,
    train_annotations_json,
)
from run_refiner_pq_allfold import (  # noqa: E402
    to_rle,
)


try:
    from .run_v80_scorer_eval_impl import (
        ARMS,
        CONTEXT_FEATURES,
        FOLDS,
        NA_FEATURES,
        TAU_GRID,
        YOLO_TAU_FIXED,
        _evaluate_image,
        adoption_gate,
        context_features,
        load_candidates,
        pq_of,
        rle_dicts,
        teachers_for,
    )
except ImportError:  # Direct `python scratch/...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_v80_scorer_eval_impl import (
        ARMS,
        CONTEXT_FEATURES,
        FOLDS,
        NA_FEATURES,
        TAU_GRID,
        YOLO_TAU_FIXED,
        _evaluate_image,
        adoption_gate,
        context_features,
        load_candidates,
        pq_of,
        rle_dicts,
        teachers_for,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=None,
                        help="既定: <drive>/solar_filament_v80/candidates")
    parser.add_argument("--drive", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", type=Path, default=Path("output/v80"))
    parser.add_argument("--drop-features", type=str, default="",
                        help="カンマ区切りの除外特徴名(test 推論で計算不能な"
                             " prob_* を落とした提出経路の評価用)")
    parser.add_argument("--arms", type=str, default="",
                        help="評価する arm のカンマ区切り部分集合"
                             " (既定: 全 arm)。B1 rescue の決定には"
                             " 'prod,rescue' で十分 (mask decode が主コスト)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    active_arms = tuple(a for a in ARMS
                        if a in set(args.arms.split(","))) if args.arms else ARMS
    unknown_arms = set(args.arms.split(",")) - set(ARMS) if args.arms else set()
    if unknown_arms:
        raise ValueError(f"未知の arm: {sorted(unknown_arms)}")
    if args.arms:
        print({"active_arms": list(active_arms)}, flush=True)

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
                   for a in anns]
        for image_id, anns in annotations.items()
    }
    fold_frame = pd.read_csv(grouped_folds_csv(drive))
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    stem_to_file = {Path(n).stem: n for n in file_to_ids}

    manifest_path = cache_root / "manifest_fold0.json"
    if manifest_path.exists():
        feature_names = json.loads(
            manifest_path.read_text(encoding="utf-8"))["feature_names"]
    else:
        # DriveFS 同期遅延等で manifest が未着でも、特徴名は生成スクリプトの
        # FEATURE_NAMES 定数が正本なのでそれを AST から読んで継続する。
        gen_src = (Path(__file__).parent / "run_v80_hybrid_candidates.py").read_text(
            encoding="utf-8")
        feature_names = next(
            ast.literal_eval(node.value)
            for node in ast.parse(gen_src).body
            if isinstance(node, ast.Assign) and node.targets[0].id == "FEATURE_NAMES"
        )
        print(f"警告: {manifest_path} 未着のため FEATURE_NAMES を生成スクリプトから取得",
              flush=True)
    all_features = list(feature_names) + list(CONTEXT_FEATURES)
    drop = set(args.drop_features.split(",")) if args.drop_features else set()
    unknown = drop - set(all_features)
    if unknown:
        raise ValueError(f"未知の特徴名を drop 指定: {sorted(unknown)}")
    keep_idx = [i for i, n in enumerate(all_features) if n not in drop]
    all_features = [n for n in all_features if n not in drop]
    na_index = [all_features.index(n) for n in NA_FEATURES if n in all_features]
    if drop:
        print({"drop_features": sorted(drop), "残特徴": len(all_features)},
              flush=True)

    # ---- 1. 候補の読み込み、教師 u/h、文脈特徴 ------------------------------
    rows = []
    bundles = {}
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
        targets = targets[: args.limit]
    print({"観測": len(targets), "特徴量": len(all_features)}, flush=True)

    started_load = time.perf_counter()
    for position, (fold, stem) in enumerate(targets, 1):
        name = stem_to_file[stem]
        cand = load_candidates(cache_root / f"fold_{fold}" / f"{stem}.npz")
        pred_rles = rle_dicts(cand["rle_final"])
        records = [gt_rles[i] for i in file_to_ids[name]]
        u, h = teachers_for(pred_rles, records)
        ctx = context_features(cand, pred_rles, len(records))
        X = np.concatenate(
            [cand["feats"].astype(np.float64), ctx.astype(np.float64)], axis=1
        )[:, keep_idx]
        X[:, na_index] = np.where(X[:, na_index] < 0, np.nan, X[:, na_index])
        offset = int(cand["tree_offset"])
        n_tree = int(cand["n_tree_nodes"])
        parent = cand["parent"].astype(int)
        children = [[] for _ in range(n_tree)]
        roots = []
        for node in range(n_tree):
            if parent[offset + node] < 0:
                roots.append(node)
            else:
                children[parent[offset + node]].append(node)
        base = len(rows)
        n_cand = len(pred_rles)
        bundles[name] = {
            "fold": fold,
            "rle_final": cand["rle_final"],
            "usable": cand["usable"],
            "source": cand["source"],
            "conf": cand["conf"],
            "tree_offset": offset,
            "n_tree_nodes": n_tree,
            "tree_children": children,
            "tree_roots": roots,
            "records": records,
            "row_slice": (base, base + n_cand),
        }
        for i in range(n_cand):
            rows.append((u[i], h[i], float(len(records)), X[i]))
        if position % 50 == 0 or position == len(targets):
            print(f"  load+teacher {position}/{len(targets)}  "
                  f"{time.perf_counter() - started_load:.0f}s", flush=True)

    u_all = np.asarray([r[0] for r in rows])
    h_all = np.asarray([r[1] for r in rows])
    w_all = np.asarray([r[2] for r in rows])
    X_all = np.stack([r[3] for r in rows])
    fold_rows = np.asarray([
        bundles[name]["fold"]
        for name, bundle in bundles.items()
        for _ in range(bundle["row_slice"][1] - bundle["row_slice"][0])
    ])
    positives = h_all > 0
    print({
        "候補数": len(rows), "h>0": int(positives.sum()),
        "u>0 割合": float((u_all > 0).mean()),
        "load+teacher 所要": round(time.perf_counter() - started_load, 1),
    }, flush=True)

    # ---- 2. cross-fit hurdle (h 回帰 × m 回帰、ハイパラ固定) -----------------
    u_hat = np.zeros(len(rows), dtype=np.float64)
    params = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                  random_state=args.seed)
    for fold in FOLDS:
        train = fold_rows != fold
        if not train.any() or not (fold_rows == fold).any():
            # --limit smoke test 等で当該 fold の学習/評価行が無い場合はスキップ
            print(f"fold{fold}: 学習{int(train.sum())}/評価{int((fold_rows == fold).sum())}行のため skip",
                  flush=True)
            continue
        model_h = HistGradientBoostingRegressor(**params).fit(
            X_all[train], h_all[train], sample_weight=w_all[train]
        )
        pos = train & positives
        test = fold_rows == fold
        if not pos.any():
            u_hat[test] = 0.0
            continue
        model_m = HistGradientBoostingRegressor(**params).fit(
            X_all[pos],
            u_all[pos] / np.maximum(h_all[pos], 1e-6),
            sample_weight=w_all[pos],
        )
        u_hat[test] = model_h.predict(X_all[test]) * model_m.predict(X_all[test])

    for name, bundle in bundles.items():
        lo, hi = bundle["row_slice"]
        bundle["u_hat"] = u_hat[lo:hi]
        bundle["u_true"] = u_all[lo:hi]
        bundle["arms"] = active_arms

    # ---- 3. 画像並列で (arm, τ) の PQ 寄与を集計 ------------------------------
    counts = {arm: defaultdict(dict) for arm in active_arms}
    names = sorted(bundles)
    started_eval = time.perf_counter()
    work = [bundles[n] for n in names]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_evaluate_image, work, chunksize=4))
    else:
        results = [_evaluate_image(b) for b in work]
    for name, per_arm in zip(names, results):
        for arm in active_arms:
            for t, c in per_arm[arm].items():
                counts[arm][t][name] = c
    print(f"evaluate {len(names)}観測: {time.perf_counter() - started_eval:.0f}s", flush=True)

    # ---- 4. 集計: arm ごとの τ 掃引 → 固定点 τ* → fold 別・bootstrap ----------
    arm_pq = {
        arm: {t: pq_of(c.values()) for t, c in counts[arm].items()}
        for arm in active_arms
    }
    for arm in active_arms:
        print(f"{arm}: " + "  ".join(
            f"τ={t:.2f}:{v:.4f}" for t, v in sorted(arm_pq[arm].items())))
    # fixed/best/split tau は関連 arm が active の時だけ意味を持つ。
    # --arms prod,rescue の B1 実行では rescue 系だけを確実に算出する。
    if "hybrid" in active_arms:
        fixed = min(TAU_GRID, key=lambda t: abs(t - arm_pq["hybrid"][t] / 2))
        best_tau = max(TAU_GRID, key=lambda t: arm_pq["hybrid"][t])
    else:
        fixed = best_tau = YOLO_TAU_FIXED
    if "hybrid_split" in active_arms:
        split_tau = max(counts["hybrid_split"],
                        key=lambda t: arm_pq["hybrid_split"][t])
    else:
        split_tau = 0.0
    # B1 rescue: τ 掃引の max Δ を事前登録ゲートに対応させる
    rescue_best_tau = (max(counts["rescue"],
                           key=lambda t: arm_pq["rescue"][t])
                       if "rescue" in active_arms else None)
    # B2 morph: 半径掃引の max Δ を事前登録ゲートに対応させる
    morph_best_r = (max(counts["morph"],
                        key=lambda t: arm_pq["morph"][t])
                    if "morph" in active_arms else None)

    fold_names = {f: [n for n in names if bundles[n]["fold"] == f] for f in FOLDS}
    star_tau = {arm: fixed for arm in active_arms}
    if "hybrid_split" in active_arms:
        star_tau["hybrid_split"] = split_tau
    if rescue_best_tau is not None:
        star_tau["rescue"] = rescue_best_tau
    if morph_best_r is not None:
        star_tau["morph"] = morph_best_r
    table = {}
    for arm in active_arms:
        star = counts[arm][star_tau[arm]]
        row = {"pq": pq_of(star.values())}
        tp = sum(c[1] for c in star.values())
        fp = sum(c[2] for c in star.values())
        fn = sum(c[3] for c in star.values())
        row.update(tp=tp, fp=fp, fn=fn,
                   sq=sum(c[0] for c in star.values()) / max(tp, 1),
                   rq=tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0)
        for f in FOLDS:
            row[f"pq_fold{f}"] = pq_of([star[n] for n in fold_names[f] if n in star])
        table[arm] = row
    table = pd.DataFrame(table).T
    table["ΔPQ_vs_prod"] = table["pq"] - table.loc["prod", "pq"]
    table["worst_fold"] = table[[f"pq_fold{f}" for f in FOLDS]].min(axis=1)

    rng = np.random.default_rng(args.seed)
    samples = [
        rng.choice(names, size=len(names), replace=True) for _ in range(args.boot)
    ]

    def boot_ci(arm: str, tau: float) -> np.ndarray:
        deltas = [
            pq_of([counts[arm][tau][n] for n in s])
            - pq_of([counts["prod"][fixed][n] for n in s])
            for s in samples
        ]
        return np.percentile(deltas, [2.5, 97.5])

    def ci_if(arm: str, tau: float):
        return boot_ci(arm, tau) if arm in active_arms and "prod" in counts \
            else (float("nan"), float("nan"))

    ci = ci_if("hybrid", fixed)
    ci_split = ci_if("hybrid_split", split_tau)
    ci_yolo = ci_if("yolo_scorer", fixed)
    ci_rescue = ci_if("rescue", rescue_best_tau) \
        if rescue_best_tau is not None else (float("nan"), float("nan"))
    ci_morph = ci_if("morph", morph_best_r) \
        if morph_best_r is not None else (float("nan"), float("nan"))

    # per-image counts を保存 — arm/τ の再集計や別 bootstrap に再 decode 不要
    (args.out / "v80_counts.json").write_text(json.dumps(
        {arm: {str(t): counts[arm][t] for t in counts[arm]}
         for arm in active_arms},
        ensure_ascii=False), encoding="utf-8")

    print(f"\n=== v80 hybrid candidates + expected-IoU scorer (τ*={fixed:.2f}, "
          f"hybrid_split tree τ={split_tau:.2f}) ===")
    print(table.round(4).to_string())
    table.to_csv(args.out / "v80_scorer_eval.csv")

    fold_cols = [f"pq_fold{f}" for f in FOLDS]
    report = {
        "scope": f"探索的scorer cross-fit ({len(names)}観測)、上流モデルはnon-nested",
        "result_kind": "exploratory_non_nested_scorer_crossfit",
        "adoption_eligible": False,
        "provenance_limitations": [
            "Scorer training-fold upstream models include outer held-out-fold labels",
            "Frozen split identity and complete candidate-cache coverage are not verified",
            "Nested outer-fold upstream/cache refits or independent evaluation are required",
        ],
        "active_arms": list(active_arms),
        "dropped_features": sorted(drop),
        "n_features": len(all_features),
        "arm_tau_sweep": {a: {str(t): round(v, 6) for t, v in sorted(p.items())}
                          for a, p in arm_pq.items()},
        "table": json.loads(table.round(6).to_json(orient="index")),
        "runtime_sec": round(time.perf_counter() - started, 1),
        "utc_completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if "hybrid" in active_arms:
        report.update({
            "tau_fixed_point": float(fixed),
            "tau_best_report_only": float(best_tau),
            "delta_hybrid_vs_prod": float(table.loc["hybrid", "ΔPQ_vs_prod"]),
            "bootstrap_ci95_hybrid_minus_prod": [float(ci[0]), float(ci[1])],
        })
    if "hybrid_oracle" in active_arms:
        report["delta_oracle_vs_prod"] = float(
            table.loc["hybrid_oracle", "ΔPQ_vs_prod"])
    if "yolo_scorer" in active_arms:
        report.update({
            "delta_yolo_scorer_vs_prod": float(
                table.loc["yolo_scorer", "ΔPQ_vs_prod"]),
            "bootstrap_ci95_yolo_scorer_minus_prod":
                [float(ci_yolo[0]), float(ci_yolo[1])],
        })
    if "hybrid_split" in active_arms:
        report["hybrid_split"] = {
            "tree_tau_used": float(split_tau),
            "yolo_tau_fixed": YOLO_TAU_FIXED,
            "delta_vs_prod": float(table.loc["hybrid_split", "ΔPQ_vs_prod"]),
            "delta_by_fold": {f"fold{f}": float(
                table.loc["hybrid_split", f"pq_fold{f}"]
                - table.loc["prod", f"pq_fold{f}"]) for f in FOLDS},
            "bootstrap_ci95_vs_prod": [float(ci_split[0]), float(ci_split[1])],
            "note": "tree tau は pooled PQ 上の事後最適(探索的 arm であり採否判定はしない)",
        }
    if "rescue" in active_arms:
        report["rescue"] = {
            "definition": "keep conf>=PROD_CONF or (conf in [SCORER_MIN_CONF,PROD_CONF) and u_hat>tau)",
            "tau_sweep": {str(t): round(v, 6)
                          for t, v in sorted(arm_pq["rescue"].items())},
            "best_tau": float(rescue_best_tau),
            "delta_vs_prod": float(
                arm_pq["rescue"][rescue_best_tau] - table.loc["prod", "pq"]),
            "bootstrap_ci95_at_best_tau": [float(ci_rescue[0]), float(ci_rescue[1])],
            "b1_gate": ("promote_candidate_requires_nested_reverification"
                        if arm_pq["rescue"][rescue_best_tau] - table.loc["prod", "pq"]
                        >= 0.005 else
                        "reject_noise" if arm_pq["rescue"][rescue_best_tau]
                        - table.loc["prod", "pq"] >= 0 else "reject"),
        }
    if "morph" in active_arms:
        report["morph"] = {
            "definition": "prod keep masks に disk dilation(r>0)/erosion(r<0) を適用",
            "radius_sweep": {str(t): round(v, 6)
                             for t, v in sorted(arm_pq["morph"].items())},
            "best_radius": float(morph_best_r),
            "delta_vs_prod": float(
                arm_pq["morph"][morph_best_r] - table.loc["prod", "pq"]),
            "bootstrap_ci95_at_best_radius": [float(ci_morph[0]),
                                              float(ci_morph[1])],
            "b2_gate": ("promote_candidate_requires_nested_reverification"
                        if arm_pq["morph"][morph_best_r] - table.loc["prod", "pq"]
                        >= 0.005 else
                        "reject_noise" if arm_pq["morph"][morph_best_r]
                        - table.loc["prod", "pq"] >= 0 else "reject"),
        }
    if "hybrid" in active_arms and "hybrid_oracle" in active_arms:
        report["gate"] = adoption_gate(
            [float(table.loc["hybrid", col] - table.loc["prod", col])
             for col in fold_cols],
            float(ci[0]), float(table.loc["hybrid_oracle", "ΔPQ_vs_prod"]))
    (args.out / "v80_scorer_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("table", "tau_sweep")},
                     ensure_ascii=False, indent=2))
    return 0
