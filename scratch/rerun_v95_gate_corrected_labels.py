"""v95 emission gate を修正後 marginal ラベルで再評価する (§2.64)。

初版 v95 (output/v95_features/gate_results.json) は marginal 分母が
「unique 被覆 GT の新規 FN 化」を反映していない旧ラベルで学習・
評価された。本スクリプトは cache 済み特徴量 (fold{0,1,2}.csv) と
修正済み per_pred_marginal.csv / tp_decomp.csv から、旧ラベル
(ratio_old = tp_iou_sum/(n_tp+0.5·n_fp)) と新ラベル (厳密
marginal_denom) の両定義で cross-fit ゲート評価を再実行する。

exact PQ は線形除去デルタで厳密に計算する:
  PQ' = (ΣIoU_TP − Σ_drop tp_iou) /
        (D − Σ_drop n_tp − 0.5·Σ_drop n_fp + 0.5·newly_fn)
  newly_fn = covering preds が全て drop 集合に含まれる
        (file_name, record_id, gt_local) の数
(n_unique の和と一致することを実データでも確認済み。)

モデル: HistGradientBoostingClassifier (max_iter=200, seed 0) と
LogisticRegression (StandardScaler, max_iter=5000)。tau は
train 2 fold の合計 ΔPQ が最大になる閾値を 0.3–0.9 で選択し、
held-out fold に適用 (リークなし)。

使い方:
    python scratch/rerun_v95_gate_corrected_labels.py \
        --out output/v95_features/gate_results_corrected.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_COLS = [
    "conf", "fin_area", "prop_area", "ref_area", "n_prop_photo",
    "n_fin_photo", "n_records", "cx", "cy", "aspect", "extent",
    "prop_overlap_max", "prop_overlap_cnt", "shrink_ref_prop",
    "iou_prop_ref", "radial", "raw_in", "raw_contrast", "clahe_in",
    "clahe_contrast", "blackhat_in", "blackhat_contrast",
]
OUT_ROOT = Path(__file__).resolve().parents[1] / "output"


def load_fold(f: int):
    x = pd.read_csv(OUT_ROOT / "v95_features" / f"fold{f}.csv")
    m = pd.read_csv(
        OUT_ROOT / f"error_decomp_fold{f}" / "per_pred_marginal.csv")
    m["ratio_old"] = (
        m["tp_iou_sum"] / (m["n_tp"] + 0.5 * m["n_fp"]))
    x = x.merge(
        m[["file_name", "pred_idx", "ratio", "ratio_old",
           "tp_iou_sum", "n_tp", "n_fp"]],
        left_on=["file_name", "fin_idx"],
        right_on=["file_name", "pred_idx"])
    tp = pd.read_csv(
        OUT_ROOT / f"error_decomp_fold{f}" / "tp_decomp.csv")
    base = json.loads(
        (OUT_ROOT / f"error_decomp_fold{f}" / "summary.json")
        .read_text())["official_reproduction"]
    return x, tp, base


def exact_pq_after_drop(feats, tps, base, dropped_keys):
    """drop 集合を除去した後の厳密 PQ (線形デルタ・FN 化を厳密集計)。"""
    D = base["tp"] + 0.5 * base["fp"] + 0.5 * base["fn"]
    iou_tot = base["pq"] * D
    dr = feats[
        [(a, b) in dropped_keys
         for a, b in zip(feats["file_name"], feats["pred_idx"])]]
    if len(dr) == 0:
        return base["pq"]
    cov = (tps.groupby(["file_name", "record_id", "gt_local"])
           ["pred_idx"].apply(set))
    nnew = sum(
        1 for (fn_, _r, _g), ps in cov.items()
        if all((fn_, p) in dropped_keys for p in ps))
    return ((iou_tot - dr["tp_iou_sum"].sum())
            / (D - dr["n_tp"].sum() - 0.5 * dr["n_fp"].sum()
               + 0.5 * nnew))


def fit_model(name, Xtr, ytr):
    if name == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=200, random_state=0).fit(Xtr, ytr)
    pipe_sc = StandardScaler()
    lr = LogisticRegression(max_iter=5000, C=1.0)
    lr.fit(pipe_sc.fit_transform(Xtr), ytr)
    return (pipe_sc, lr)


def predict(model, name, X):
    if name == "gbm":
        return model.predict_proba(X)[:, 1]
    sc, lr = model
    return lr.predict_proba(sc.transform(X))[:, 1]


def run(feats, tps, bases, label_col, model_name):
    """cross-fit: held-out fold ごとに train 2 fold で学習し評価。"""
    res = {}
    for held in (0, 1, 2):
        trn = [f for f in (0, 1, 2) if f != held]
        tr = pd.concat([feats[f] for f in trn])
        te = feats[held]
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(tr[X_COLS])
        Xte = imp.transform(te[X_COLS])
        model = fit_model(model_name, Xtr, tr[label_col])
        ptr = predict(model, model_name, Xtr)
        pte = predict(model, model_name, Xte)
        trp = tr.assign(p=ptr)
        # tau sweep: train 2 fold の合計 ΔPQ を最大化
        best_tau, best_gain = None, -9.0
        for tau in np.arange(0.3, 0.91, 0.05):
            g = 0.0
            for f in trn:
                sub = feats[f].merge(
                    trp[["file_name", "pred_idx", "p"]],
                    on=["file_name", "pred_idx"])
                drop = set(zip(sub.loc[sub.p > tau, "file_name"],
                               sub.loc[sub.p > tau, "pred_idx"]))
                g += (exact_pq_after_drop(feats[f], tps[f], bases[f],
                                          drop)
                      - bases[f]["pq"])
            if g > best_gain:
                best_gain, best_tau = g, tau
        tep = te.assign(p=pte)
        drop = set(zip(tep.loc[tep.p > best_tau, "file_name"],
                       tep.loc[tep.p > best_tau, "pred_idx"]))
        pq = exact_pq_after_drop(feats[held], tps[held],
                                 bases[held], drop)
        res[str(held)] = {
            "tau": round(float(best_tau), 2),
            "train_gain": round(float(best_gain), 4),
            "dropped": len(drop),
            "exact_pq": round(float(pq), 6),
            "delta": round(float(pq - bases[held]["pq"]), 6),
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    feats, tps, bases = {}, {}, {}
    for f in (0, 1, 2):
        feats[f], tps[f], bases[f] = load_fold(f)
        feats[f]["y_old"] = (
            feats[f]["ratio_old"] < bases[f]["pq"]).astype(int)
        feats[f]["y_new"] = (
            feats[f]["ratio"] < bases[f]["pq"]).astype(int)

    out = {}
    for label in ("y_old", "y_new"):
        for model in ("gbm", "lr"):
            key = f"{model}_{label}"
            out[key] = run(feats, tps, bases, label, model)
            print(f"--- {key} ---")
            print(json.dumps(out[key], indent=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
