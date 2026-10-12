"""v80 E1 — ハイブリッド候補集合 + 期待IoUスコアラーの全fold OOF 評価 (CPU)。

PLAN_2026-10-09_FINAL_SPRINT.md §7.1.4〜7.1.8。
`run_v80_hybrid_candidates.py` が保存した候補キャッシュ
(`solar_filament_v80/candidates/fold_{f}/<stem>.npz`) を読み、

  1. 教師 u(c) = (1/R) Σ_r max_g IoU(c, g_r) · 1[IoU > 0.5] を各候補に計算
     (h = 当たる割合、m = 当たった record の平均 IoU、u = h·m は厳密な分解)
  2. HistGradientBoosting (h 回帰 + m 回帰、ハイパラ固定、重み=record 数 R)
     を fold 外候補のみで学習し、fold 内候補に cross-fit で û を出す
  3. û > τ 選択 + tree 候補は Σ(û−τ) 最大の antichain DP → 排他化 → 公式 PQ

arms (同一候補・同一 refiner 出力・paired):
  prod          : yolo conf>=0.26 (現行ベースライン、τ に依らない)
  yolo_scorer   : yolo 候補 ∩ û>τ                (v64 相当の対照)
  tree_only     : tree 候補の antichain DP
  hybrid        : yolo û>τ ∪ tree antichain
  hybrid_oracle : 同上だが真値 u で選択 (機構の上界)

採用ゲート: hybrid − prod が 3fold すべて正 かつ pooled bootstrap 95%CI が
ゼロを除外。hybrid_oracle が +0.02 未満なら候補集合の上界が低いと判断する。

使い方:
    python scratch/run_v80_scorer_eval.py [--limit N] [--workers 8] [--boot 2000]
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
from pycocotools import mask as mask_util
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))

from solar_filament.pipeline.component_tree import select_antichain_forest  # noqa: E402
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
from src.metrics_official import PQ_IOU_THRESHOLD  # noqa: E402

EVAL_SIZE = 2048
PROD_CONF = 0.26
SCORER_MIN_CONF = 0.08
CONTAINMENT = 0.90
FOLDS = (0, 1, 2)
# τ = PQ/2 の固定点は PQ≈0.44 なら ≈0.22。0.16〜0.32 で掃引し、
# 掃引で最良の τ は参考報告のみ (主結果は固定点)。
TAU_GRID = tuple(round(0.16 + 0.02 * i, 2) for i in range(9))  # 0.16 … 0.32
# v81: hybrid の tree 側を独立 τ で掃引する。診断で û の絶対校正が tree で
# 過大(中位候補の累積選択で FP 爆発)と判明したため、yolo は固定点 0.22 に
# 固定し、tree 側だけ 0.20–0.68 の広めグリッドで探す。
TREE_TAU_GRID = tuple(round(0.20 + 0.04 * i, 2) for i in range(13))  # 0.20 … 0.68
YOLO_TAU_FIXED = 0.22  # hybrid の固定点 τ*≈PQ/2
ARMS = ("prod", "yolo_scorer", "tree_only", "hybrid", "hybrid_oracle",
        "hybrid_split")


def arm_tau_grid(arm: str) -> tuple[float, ...]:
    if arm == "prod":
        return (0.0,)
    if arm == "hybrid_split":
        return TREE_TAU_GRID
    return TAU_GRID




CONTEXT_FEATURES = (
    "max_iou_any", "n_overlap_10", "yolo_support_conf",
    "n_cand_img", "n_yolo_img", "n_annotators",
)

# -1 センチネルを NaN に戻す列 (HistGBM は NaN を欠損として扱う)
NA_FEATURES = ("yolo_conf", "level_index", "n_children", "n_siblings",
               "parent_area_ratio", "yolo_support_conf")


def load_candidates(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def rle_dicts(rle_texts) -> list[dict]:
    return [
        {"size": [EVAL_SIZE, EVAL_SIZE], "counts": str(t).encode("utf-8")}
        for t in rle_texts
    ]


def teachers_for(pred_rles: list[dict], gt_per_record: list[list[dict]]):
    """(u, h): record 平均の maxIoU·1[>0.5] と、record 平均の命中割合。"""
    n = len(pred_rles)
    if n == 0 or not gt_per_record:
        return np.zeros(n), np.zeros(n)
    u = np.zeros(n, dtype=np.float64)
    h = np.zeros(n, dtype=np.float64)
    for gt_rles in gt_per_record:
        matrix = np.asarray(
            mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64
        )
        best = matrix.max(axis=1) if matrix.size else np.zeros(n)
        u += np.where(best > PQ_IOU_THRESHOLD, best, 0.0)
        h += (best > PQ_IOU_THRESHOLD)
    return u / len(gt_per_record), h / len(gt_per_record)


def context_features(cand: dict, pred_rles: list[dict], n_annotators: int) -> np.ndarray:
    n = len(pred_rles)
    out = np.zeros((n, len(CONTEXT_FEATURES)), dtype=np.float32)
    if n == 0:
        return out
    self_iou = np.asarray(
        mask_util.iou(pred_rles, pred_rles, [0] * n), dtype=np.float64
    )
    np.fill_diagonal(self_iou, 0.0)
    source = cand["source"].astype(int)
    conf = cand["conf"]
    is_yolo = source == 0
    yolo_support = np.full(n, np.nan, dtype=np.float64)
    for i in np.where(~is_yolo)[0]:
        hits = np.where((self_iou[i] > PQ_IOU_THRESHOLD) & is_yolo)[0]
        if hits.size:
            yolo_support[i] = conf[hits].max()
    out[:, 0] = self_iou.max(axis=1)
    out[:, 1] = (self_iou > 0.10).sum(axis=1)
    out[:, 2] = yolo_support
    out[:, 3] = n
    out[:, 4] = float(is_yolo.sum())
    out[:, 5] = n_annotators
    return out


def _encode(masks: list[np.ndarray]) -> list[dict]:
    if not masks:
        return []
    return mask_util.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))


def _evaluate_image(bundle: dict) -> dict:
    """1観測分の (arm, τ) ごとの PQ 寄与 (tp_iou_sum, tp, fp, fn)。

    ワーカープロセスで実行される。mask の decode は画像につき1回だけ行い、
    排他化は既存ルール (drop_contained@0.90 → area 昇順 greedy) を使う。
    """
    masks = [
        mask_util.decode(r).astype(bool) for r in rle_dicts(bundle["rle_final"])
    ]
    rles = rle_dicts(bundle["rle_final"])
    usable = np.asarray(bundle["usable"], dtype=bool)
    source = np.asarray(bundle["source"], dtype=int)
    conf = np.asarray(bundle["conf"])
    offset = int(bundle["tree_offset"])
    n_tree = int(bundle["n_tree_nodes"])
    children = bundle["tree_children"]
    roots = bundle["tree_roots"]
    out = {arm: {} for arm in ARMS}

    for arm in ARMS:
        util = np.asarray(bundle["u_true"] if arm == "hybrid_oracle" else bundle["u_hat"])
        yolo_ok = (source == 0) & usable & (conf >= SCORER_MIN_CONF)
        for tau in arm_tau_grid(arm):
            keep: list[int] = []
            if arm == "prod":
                keep += list(np.where((source == 0) & usable & (conf >= PROD_CONF))[0])
            if arm in ("yolo_scorer", "hybrid", "hybrid_oracle"):
                keep += list(np.where(yolo_ok & (util > tau))[0])
            if arm == "hybrid_split":
                # yolo 側は固定点 τ* で固定、tree 側だけ tau を掃引 (v81)
                keep += list(np.where(yolo_ok & (util > YOLO_TAU_FIXED))[0])
            if arm in ("tree_only", "hybrid", "hybrid_oracle", "hybrid_split"):
                tree_util = np.full(n_tree, -np.inf)
                for node in range(n_tree):
                    row = offset + node
                    if usable[row]:
                        tree_util[node] = util[row]
                keep += [
                    offset + node
                    for node in select_antichain_forest(children, roots, tree_util, tau)
                ]
            subset = [masks[i] for i in keep]
            subset_rles = [rles[i] for i in keep]
            alive = drop_contained_indices(
                subset, threshold=CONTAINMENT, rles=subset_rles
            )
            areas = np.array([m.sum() for m in subset])
            preds = greedy_disjoint(
                subset, sorted(alive, key=lambda i: areas[i]), min_area=MIN_AREA
            )
            pred_rles = _encode(preds)
            tp_sum = tp_c = fp_c = fn_c = 0
            for gt in bundle["records"]:
                if not pred_rles:
                    fn_c += len(gt)
                    continue
                matrix = np.asarray(
                    mask_util.iou(pred_rles, gt, [0] * len(gt)), dtype=np.float64
                ).T
                hits = matrix > PQ_IOU_THRESHOLD
                tp_sum += float(matrix[hits].sum())
                tp_c += int(hits.sum())
                fp_c += int((hits.sum(axis=0) == 0).sum())
                fn_c += int((hits.sum(axis=1) == 0).sum())
            # prod は τ 不変 — 全 τ に同じ値を入れて後段を単純化する
            for t in (TAU_GRID if arm == "prod" else (tau,)):
                out[arm][t] = (tp_sum, tp_c, fp_c, fn_c)
    return out


def pq_of(counts) -> float:
    tp_sum = sum(c[0] for c in counts)
    tp = sum(c[1] for c in counts)
    fp = sum(c[2] for c in counts)
    fn = sum(c[3] for c in counts)
    den = tp + 0.5 * fp + 0.5 * fn
    return tp_sum / den if den > 0 else 0.0


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

    # ---- 3. 画像並列で (arm, τ) の PQ 寄与を集計 ------------------------------
    counts = {arm: defaultdict(dict) for arm in ARMS}
    names = sorted(bundles)
    started_eval = time.perf_counter()
    work = [bundles[n] for n in names]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_evaluate_image, work, chunksize=4))
    else:
        results = [_evaluate_image(b) for b in work]
    for name, per_arm in zip(names, results):
        for arm in ARMS:
            for t, c in per_arm[arm].items():
                counts[arm][t][name] = c
    print(f"evaluate {len(names)}観測: {time.perf_counter() - started_eval:.0f}s", flush=True)

    # ---- 4. 集計: arm ごとの τ 掃引 → 固定点 τ* → fold 別・bootstrap ----------
    arm_pq = {
        arm: {t: pq_of(c.values()) for t, c in counts[arm].items()}
        for arm in ARMS
    }
    for arm in ARMS:
        print(f"{arm}: " + "  ".join(
            f"τ={t:.2f}:{v:.4f}" for t, v in sorted(arm_pq[arm].items())))
    fixed = min(TAU_GRID, key=lambda t: abs(t - arm_pq["hybrid"][t] / 2))
    best_tau = max(TAU_GRID, key=lambda t: arm_pq["hybrid"][t])
    split_tau = max(counts["hybrid_split"], key=lambda t: arm_pq["hybrid_split"][t])

    fold_names = {f: [n for n in names if bundles[n]["fold"] == f] for f in FOLDS}
    star_tau = {arm: fixed for arm in ARMS}
    star_tau["hybrid_split"] = split_tau
    table = {}
    for arm in ARMS:
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

    ci = boot_ci("hybrid", fixed)
    ci_split = boot_ci("hybrid_split", split_tau)
    ci_yolo = boot_ci("yolo_scorer", fixed)

    # per-image counts を保存 — arm/τ の再集計や別 bootstrap に再 decode 不要
    (args.out / "v80_counts.json").write_text(json.dumps(
        {arm: {str(t): counts[arm][t] for t in counts[arm]} for arm in ARMS},
        ensure_ascii=False), encoding="utf-8")

    print(f"\n=== v80 hybrid candidates + expected-IoU scorer (τ*={fixed:.2f}, "
          f"hybrid_split tree τ={split_tau:.2f}) ===")
    print(table.round(4).to_string())
    table.to_csv(args.out / "v80_scorer_eval.csv")

    fold_cols = [f"pq_fold{f}" for f in FOLDS]
    report = {
        "scope": f"全fold OOF ({len(names)}観測)、候補=YOLO∪component-tree、cross-fit HistGBM",
        "dropped_features": sorted(drop),
        "n_features": len(all_features),
        "tau_fixed_point": float(fixed),
        "tau_best_report_only": float(best_tau),
        "arm_tau_sweep": {a: {str(t): round(v, 6) for t, v in sorted(p.items())}
                          for a, p in arm_pq.items()},
        "hybrid_split": {
            "tree_tau_used": float(split_tau),
            "yolo_tau_fixed": YOLO_TAU_FIXED,
            "delta_vs_prod": float(table.loc["hybrid_split", "ΔPQ_vs_prod"]),
            "delta_by_fold": {f"fold{f}": float(
                table.loc["hybrid_split", f"pq_fold{f}"]
                - table.loc["prod", f"pq_fold{f}"]) for f in FOLDS},
            "bootstrap_ci95_vs_prod": [float(ci_split[0]), float(ci_split[1])],
            "note": "tree tau は pooled PQ 上の事後最適(探索的 arm であり採否判定はしない)",
        },
        "table": json.loads(table.round(6).to_json(orient="index")),
        "delta_hybrid_vs_prod": float(table.loc["hybrid", "ΔPQ_vs_prod"]),
        "delta_oracle_vs_prod": float(table.loc["hybrid_oracle", "ΔPQ_vs_prod"]),
        "delta_yolo_scorer_vs_prod": float(table.loc["yolo_scorer", "ΔPQ_vs_prod"]),
        "bootstrap_ci95_hybrid_minus_prod": [float(ci[0]), float(ci[1])],
        "bootstrap_ci95_yolo_scorer_minus_prod": [float(ci_yolo[0]), float(ci_yolo[1])],
        "gate": {
            "adopt": bool(
                (table.loc["hybrid", fold_cols] > table.loc["prod", fold_cols]).all()
                and ci[0] > 0
            ),
            "oracle_headroom_ok": bool(table.loc["hybrid_oracle", "ΔPQ_vs_prod"] >= 0.02),
        },
        "runtime_sec": round(time.perf_counter() - started, 1),
        "utc_completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "v80_scorer_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("table", "tau_sweep")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
