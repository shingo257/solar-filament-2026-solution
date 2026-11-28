# 反実仮想実験スクリプト (2026-11-28) — C:/work/solar_counterfactual から保全。
# データセットキャッシュ (selection_dataset*.pkl) はサイズのためコミット対象外、
# Drive の v140 oof cache から本スクリプトで再生成可能。詳細は RESEARCH_LOG 2026-11-28 項。
"""Evaluate learned vs rule-based candidate selection on the ERS union3 pool.

Cross-fitted LightGBM: train on one fold's clusters, select on the other.
Honest held-out official pooled PQ, compared against in-experiment baselines
computed on identical images and identical disjointness rules.
"""
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

V30 = Path(r"G:/マイドライブ/solar_filament_v30")
OUT = Path(r"C:/work/solar_counterfactual")

# ---- GT (same loader as dataset builder) ----
coco = json.load(open(V30 / "competition_data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
images = {im["id"]: im for im in coco["images"]}
file_to_ids: dict[str, list[int]] = {}
for iid, im in images.items():
    file_to_ids.setdefault(im["file_name"], []).append(iid)
annotations: dict[int, list] = {}
for a in coco["annotations"]:
    annotations.setdefault(a["image_id"], []).append(a)


def to_rle(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    if isinstance(seg.get("counts"), list):
        return mask_utils.frPyObjects(seg, h, w)
    rle = dict(seg)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


gt_rles = {
    iid: [to_rle(a, int(im["height"]), int(im["width"])) for a in annotations.get(iid, []) if not a.get("iscrowd", 0)]
    for iid, im in images.items()
}

FEATURES = [
    "n_members", "max_conf", "mean_conf", "min_conf_present",
    "conf_prod", "conf_yolo26", "conf_sptal", "n_det_above_026",
    "mask_stability", "log_area", "aspect", "extent", "cent_r",
    "rank_in_image", "n_clusters_image",
    "int_inside", "int_ring", "int_contrast", "int_diff",
]


def official_score(preds_per_file):
    """pooled multi-annotator PQ. preds_per_file: {fn: [rle,...]}"""
    si = tp = fp = fn_ = 0.0
    for fn, pred in preds_per_file.items():
        for iid in file_to_ids[fn]:
            tr = gt_rles[iid]
            if not tr:
                fp += len(pred)
                continue
            if not pred:
                fn_ += len(tr)
                continue
            mat = np.asarray(mask_utils.iou(pred, tr, [0] * len(tr)), np.float64).T
            hits = mat > 0.5
            si += float(mat[hits].sum())
            tp += int(hits.sum())
            fp += int((hits.sum(0) == 0).sum())
            fn_ += int((hits.sum(1) == 0).sum())
    tp, fp, fn_ = int(round(tp)), int(round(fp)), int(round(fn_))
    den = tp + 0.5 * fp + 0.5 * fn_
    return dict(pq=si / den, sq=si / max(tp, 1), rq=tp / den, tp=tp, fp=fp, fn=fn_)


def greedy_disjoint(items):
    """items: list of (score, rle) sorted desc by score -> non-overlapping subset.

    Compressed-domain greedy: skip any candidate overlapping an already-kept one
    (stricter than the pipeline's pixel clipping, but identical rule across all
    compared selection methods, so the comparison is fair).
    """
    kept = []
    union = None
    for score, rle in items:
        if union is not None:
            inter = mask_utils.merge([union, rle], intersect=True)
            if mask_utils.area(inter) > 0:
                continue
        if mask_utils.area(rle) < 20:
            continue
        kept.append(rle)
        union = rle if union is None else mask_utils.merge([union, rle], intersect=False)
    return kept


def main():
    rows = pickle.load(open(OUT / "selection_dataset.pkl", "rb"))
    df = pd.DataFrame([r[0] for r in rows])
    rle_of = {(r[0]["file"], i): r[1] for r in rows for i in [0]}  # not needed; keep parallel arrays instead
    # parallel arrays: rebuild mapping file -> list of (feat, rle)
    by_file: dict[str, list] = {}
    for feats, rle in rows:
        by_file.setdefault(feats["file"], []).append((feats, rle))

    print(f"clusters: {len(df)}, positive(target>0): {(df.target > 0).sum()}, mean target: {df.target.mean():.4f}")
    print(f"target>0 rate by n_members: {df.groupby('n_members').target.apply(lambda s: (s > 0).mean()).round(3).to_dict()}")

    results = {}
    for test_fold, train_fold in ((2, 0), (0, 2)):
        tr = df[df.fold == train_fold]
        te = df[df.fold == test_fold]
        model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=3407,
        )
        model.fit(tr[FEATURES], tr.target)
        df.loc[te.index, "score"] = model.predict(te[FEATURES])
        # correlation on held-out
        pred = df.loc[te.index, "score"].values
        corr = np.corrcoef(pred, te.target.values)[0, 1]
        print(f"\n[train f{train_fold} -> test f{test_fold}] heldout score-target corr: {corr:.3f}")
        imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:8]
        print("  top features:", [(n, int(v)) for n, v in imp])

    # write cross-fitted scores back into the original feats dicts (shared refs)
    for i, (feats, _rle) in enumerate(rows):
        feats["score"] = float(df.iloc[i]["score"])

    # now df.score is cross-fitted for every cluster
    # --- selection variants, evaluated per fold then pooled ---
    def build_preds(fold, mode, tau=None):
        preds = {}
        for feats_list_file, items in ((fn, items) for fn, items in by_file.items() if items[0][0]["fold"] == fold):
            sel = []
            for feats, rle in items:
                if mode == "rule_cons2of3":
                    keep = feats["n_det_above_026"] >= 2
                    sc = feats["max_conf"]
                elif mode == "rule_any_conf026":
                    keep = feats["max_conf"] >= 0.26
                    sc = feats["max_conf"]
                elif mode == "learned":
                    keep = feats["score"] >= tau
                    sc = feats["score"]
                else:
                    raise ValueError(mode)
                if keep:
                    sel.append((sc, rle))
            sel.sort(key=lambda x: -x[0])
            preds[feats_list_file] = greedy_disjoint(sel)
        return preds

    # tune tau on train fold per direction, plus fixed 0.25
    for tau_mode in ("fixed025", "crossfit_tuned"):
        per_fold = {}
        for test_fold, train_fold in ((2, 0), (0, 2)):
            if tau_mode == "fixed025":
                tau = 0.25
            else:
                # tune tau on train fold: maximize train PQ
                best_tau, best_pq = 0.25, -1
                for cand_tau in np.arange(0.10, 0.45, 0.05):
                    m = official_score(build_preds(train_fold, "learned", cand_tau))
                    if m["pq"] > best_pq:
                        best_pq, best_tau = m["pq"], float(cand_tau)
                tau = best_tau
            m = official_score(build_preds(test_fold, "learned", tau))
            per_fold[test_fold] = (tau, m)
            print(f"learned[{tau_mode}] fold{test_fold}: tau={tau:.2f} PQ {m['pq']:.4f} SQ {m['sq']:.4f} RQ {m['rq']:.4f} TP {m['tp']} FP {m['fp']} FN {m['fn']}")
        results[f"learned_{tau_mode}"] = per_fold

    for mode, tag in (("rule_cons2of3", "consensus rule (>=2 det @0.26)"), ("rule_any_conf026", "any detector @0.26")):
        for fold in (0, 2):
            m = official_score(build_preds(fold, mode))
            print(f"{tag} fold{fold}: PQ {m['pq']:.4f} SQ {m['sq']:.4f} RQ {m['rq']:.4f} TP {m['tp']} FP {m['fp']} FN {m['fn']}")

    json.dump({"note": "see stdout"}, open(OUT / "eval_done.json", "w"))


if __name__ == "__main__":
    main()
