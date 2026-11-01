"""§2.66: annotator 数 R 別の予測限界価値と、訂正版の採点上限推定。

§2.54 は「単独 annotator 物体を予測すると 1TP vs ~1.44FP で限界価値が負」
「完璧な union 予測でも PQ≈0.49」と結論したが、予測しない場合に残る FN
(分母 +0.5) を差し引いていなかった。per-(annotator, image) 採点で k 人が
描いた物体を IoU s で当てる予測の限界価値は

    Δnum = k·s,  Δden = k·1 + (R−k)·0.5 − k·0.5 = R/2

なので、PQ を上げる条件は k·s > PQ·R/2。本スクリプトは error_decomp
artifact (3 fold) から (1) R×k 別の実測限界比、(2) 単一マスク/クラスタ
近似での上限推定を再計算する。推論・GPU 不要 (コミット済み CSV のみ)。

    python scratch/analyze_annotator_count_marginal.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def marginal_gain_positive(k: float, s: float, r: float, pq: float) -> bool:
    """k 人が描いた物体を IoU s で当てる予測が PQ を上げるか (k·s > PQ·R/2)。"""
    return k * s > pq * r / 2.0


def ceiling_pq(clusters: pd.DataFrame, n_gt: int, s: float,
               select: bool = True, iters: int = 100) -> tuple[float, float]:
    """クラスタ毎に 1 マスクを出し全メンバーと IoU s で一致する近似上限。

    clusters は列 ``k`` (描いた annotator 数) と ``R`` (画像の annotator 数)。
    TP = k、FP = R−k、FN = n_gt − ΣTP。select=True なら限界価値が正の
    クラスタだけを出す不動点を反復で求める。戻り値は (PQ, 出力クラスタ率)。
    """
    k = clusters["k"].clip(upper=clusters["R"]).astype(float)
    r = clusters["R"].astype(float)

    def score(mask: pd.Series) -> float:
        tp = k[mask].sum()
        fp = (r - k)[mask].sum()
        den = tp + 0.5 * fp + 0.5 * (n_gt - tp)
        return float(s * tp / den) if den else 0.0

    keep = pd.Series(True, index=clusters.index)
    pq = score(keep)
    if select:
        for _ in range(iters):
            nxt = k * s > pq * r / 2.0
            new_pq = score(nxt)
            if nxt.equals(keep) and abs(new_pq - pq) < 1e-12:
                break
            keep, pq = nxt, new_pq
    return pq, float(keep.mean())


def load_fold(decomp_dir: Path) -> dict:
    photo = pd.read_csv(decomp_dir / "per_photo.csv")
    n_records = photo.set_index("file_name")["n_records"]
    marg = pd.read_csv(decomp_dir / "per_pred_marginal.csv")
    clusters = pd.read_csv(decomp_dir / "cluster_coverage.csv")
    fn = pd.read_csv(decomp_dir / "fn_decomp.csv")
    summary = json.loads((decomp_dir / "summary.json").read_text(encoding="utf-8"))
    marg["R"] = marg["file_name"].map(n_records)
    clusters = clusters.assign(R=clusters["file_name"].map(n_records),
                               k=clusters["cluster_k"])
    fn["R"] = fn["file_name"].map(n_records)
    return {"photo": photo, "marg": marg, "clusters": clusters, "fn": fn,
            "pq": float(summary["official_reproduction"]["pq"])}


def fold_report(fold: dict) -> dict:
    photo, marg, fn, pq = fold["photo"], fold["marg"], fold["fn"], fold["pq"]
    gt_by_r = photo.groupby("n_records")["n_gt_instances"].sum()
    table = {}
    for (r, k), g in marg.groupby(["R", "n_tp"]):
        table[f"R{int(r)}_k{int(k)}"] = {
            "n": int(len(g)),
            "mean_ratio": round(float(g["ratio"].mean()), 4),
            "share_value_positive": round(float((g["ratio"] >= pq).mean()), 4),
        }
    return {
        "pq": pq,
        "files_by_R": {int(r): int(n) for r, n in
                       photo["n_records"].value_counts().sort_index().items()},
        "gt_share_by_R": {int(r): round(float(v / gt_by_r.sum()), 4)
                          for r, v in gt_by_r.items()},
        "fn_by_R": {int(r): int(n) for r, n in
                    fn["R"].value_counts().sort_index().items()},
        "marginal_by_R_k": table,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=ROOT / "output")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "output" / "annotator_count_marginal.json")
    args = ap.parse_args()

    folds = [load_fold(args.output_root / f"error_decomp_fold{f}")
             for f in range(3)]
    clusters = pd.concat([f["clusters"] for f in folds], ignore_index=True)
    n_gt = int(sum(f["photo"]["n_gt_instances"].sum() for f in folds))
    ceiling = {}
    for s in (0.634, 0.68):
        all_pq, _ = ceiling_pq(clusters, n_gt, s, select=False)
        sel_pq, kept = ceiling_pq(clusters, n_gt, s, select=True)
        ceiling[str(s)] = {"predict_all_pq": round(all_pq, 4),
                           "optimal_subset_pq": round(sel_pq, 4),
                           "kept_cluster_share": round(kept, 4)}
    report = {
        "rule": "predict iff k*s > PQ*R/2 (per-(annotator,image) pooling)",
        "folds": {f"fold{i}": fold_report(f) for i, f in enumerate(folds)},
        "ceiling_single_mask_per_cluster": {
            "n_clusters": int(len(clusters)), "n_gt": n_gt, "by_iou": ceiling,
            "assumption": ("one mask per cluster matches every member at "
                           "IoU s (>0.5); detection is perfect"),
        },
    }
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    for name, rep in report["folds"].items():
        print(name, "PQ", round(rep["pq"], 4), "files_by_R", rep["files_by_R"])
        for key, v in rep["marginal_by_R_k"].items():
            print(f"  {key}: n={v['n']} mean_ratio={v['mean_ratio']}"
                  f" pos={v['share_value_positive']}")
    print("ceiling", json.dumps(ceiling))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
