"""A1(yolo_scorer) 提出用 scorer を全 OOF 行で学習して pkl に保存する。

cross-fit は「評価の正直性」のためのもの。test 配置は全707観測分
(yolo+tree 全候補、eval と同一の学習集合)で h/m モデルを再学習する
(標準的な全データ再学習)。特徴構築・NaN 化・ハイパラは
`run_v80_scorer_eval.py` と完全に同一経路を再利用する。

成果物 (重み dataset に同梱する):
    artifacts_scorer/a1_scorer_h.pkl        — 命中確率 ĥ の HistGBM
    artifacts_scorer/a1_scorer_m.pkl        — 正例条件付き mean-IoU m̂ の HistGBM
    artifacts_scorer/a1_scorer_meta.json    — 特徴名・τ・選択ルール・学習行数・sha

使い方:
    python scratch/train_a1_scorer.py [--drive "G:/マイドライブ"]
        [--drop-features "prob_mean_in,prob_max_in,prob_frac50_in,prob_mean_ring"]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from run_v80_scorer_eval import (  # noqa: E402
    CONTEXT_FEATURES,
    FOLDS,
    NA_FEATURES,
    context_features,
    load_candidates,
    rle_dicts,
    teachers_for,
)
from run_refiner_pq_allfold import to_rle  # noqa: E402
from solar_filament.pipeline.paths import (  # noqa: E402
    grouped_folds_csv,
    resolve_drive,
    train_annotations_json,
)

# 提出固定値 — v80 eval で採用ゲートを通過した構成 (PLAN §7.4)
TAU_STAR = 0.22
SCORER_MIN_CONF = 0.08
HGBM_PARAMS = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None,
                        help="既定: <drive>/solar_filament_v80/candidates")
    parser.add_argument("--drop-features", type=str,
                        default="prob_mean_in,prob_max_in,prob_frac50_in,"
                                "prob_mean_ring")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts_scorer"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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
        image_id: [to_rle(a, int(images[image_id]["height"]),
                          int(images[image_id]["width"]))
                   for a in anns]
        for image_id, anns in annotations.items()
    }
    fold_frame = pd.read_csv(grouped_folds_csv(drive))
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    stem_to_file = {Path(n).stem: n for n in file_to_ids}

    # 特徴名: manifest を正本、未着なら候補生成スクリプトの FEATURE_NAMES
    manifest_path = cache_root / "manifest_fold0.json"
    if manifest_path.exists():
        feature_names = json.loads(
            manifest_path.read_text(encoding="utf-8"))["feature_names"]
    else:
        import ast
        gen_src = (Path(__file__).parent / "run_v80_hybrid_candidates.py"
                   ).read_text(encoding="utf-8")
        feature_names = next(
            ast.literal_eval(node.value)
            for node in ast.parse(gen_src).body
            if isinstance(node, ast.Assign) and node.targets[0].id == "FEATURE_NAMES"
        )

    all_features = list(feature_names) + list(CONTEXT_FEATURES)
    drop = set(args.drop_features.split(",")) if args.drop_features else set()
    unknown = drop - set(all_features)
    if unknown:
        raise ValueError(f"未知の特徴名を drop 指定: {sorted(unknown)}")
    keep_idx = [i for i, n in enumerate(all_features) if n not in drop]
    features_kept = [n for n in all_features if n not in drop]
    na_index = [features_kept.index(n) for n in NA_FEATURES if n in features_kept]
    print({"観測": "全fold", "特徴": len(features_kept), "drop": sorted(drop)},
          flush=True)

    # ---- 行構築 (run_v80_scorer_eval と同一) ----------------------------------
    rows_X, rows_u, rows_h, rows_w = [], [], [], []
    n_obs = 0
    started = time.perf_counter()
    for fold in FOLDS:
        fold_dir = cache_root / f"fold_{fold}"
        for path in sorted(fold_dir.glob("*.npz")):
            name = stem_to_file.get(path.stem)
            if name is None or fold_of_file.get(name) != fold:
                continue
            cand = load_candidates(path)
            pred_rles = rle_dicts(cand["rle_final"])
            records = [gt_rles[i] for i in file_to_ids[name]]
            u, h = teachers_for(pred_rles, records)
            ctx = context_features(cand, pred_rles, len(records))
            X = np.concatenate(
                [cand["feats"].astype(np.float64), ctx.astype(np.float64)],
                axis=1)[:, keep_idx]
            X[:, na_index] = np.where(X[:, na_index] < 0, np.nan, X[:, na_index])
            rows_X.append(X)
            rows_u.append(u)
            rows_h.append(h)
            rows_w.append(np.full(len(u), float(len(records))))
            n_obs += 1
        print(f"  fold{fold} 読了 ({n_obs}観測)  {time.perf_counter()-started:.0f}s",
              flush=True)

    X_all = np.concatenate(rows_X)
    u_all = np.concatenate(rows_u)
    h_all = np.concatenate(rows_h)
    w_all = np.concatenate(rows_w)
    print({"行数": len(u_all), "h>0": int((h_all > 0).sum())}, flush=True)

    # ---- 学習 (cross-fit なし、全行) ------------------------------------------
    params = dict(HGBM_PARAMS, random_state=args.seed)
    model_h = HistGradientBoostingRegressor(**params).fit(
        X_all, h_all, sample_weight=w_all)
    positives = h_all > 0
    model_m = HistGradientBoostingRegressor(**params).fit(
        X_all[positives],
        u_all[positives] / np.maximum(h_all[positives], 1e-6),
        sample_weight=w_all[positives])

    h_path = args.out / "a1_scorer_h.pkl"
    m_path = args.out / "a1_scorer_m.pkl"
    h_path.write_bytes(pickle.dumps(model_h))
    m_path.write_bytes(pickle.dumps(model_m))
    meta = {
        "experiment": "A1 yolo_scorer 提出用 (v80 eval で採用ゲート通過)",
        "feature_names": features_kept,
        "dropped_features": sorted(drop),
        "tau_star": TAU_STAR,
        "scorer_min_conf": SCORER_MIN_CONF,
        "selection": "u_hat > tau_star → drop_contained@0.90 → "
                     "greedy_disjoint(面積昇順, min_area=20)",
        "u_hat": "model_h.predict(X) * model_m.predict(X)",
        "hgbm_params": {k: v for k, v in params.items() if k != "random_state"},
        "seed": args.seed,
        "train_rows": int(len(u_all)),
        "train_observations": int(n_obs),
        "sample_weight": "観測あたり annotation record 数",
        "test_imputation": {"n_annotators": "学習時中央値≈2 (PLAN §7.4)",
                            "yolo_support_conf": "yolo 候補は常に NaN"},
        "scorer_h_sha256": sha256_of(h_path),
        "scorer_m_sha256": sha256_of(m_path),
    }
    (args.out / "a1_scorer_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"h": str(h_path), "m": str(m_path),
                      "rows": len(u_all)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
