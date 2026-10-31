"""v94/v95 実験ハーネス — emission variant 評価 + pred 特徴量抽出。

analyze_error_decomposition.py が出力した stage キャッシュ
(<out>/cache/*.npz に prop/ref/fin packed mask + confs) から、

  (a) emission arm の公式 PQ を fold 単位で再評価する
      - final       : prod そのまま (baseline)
      - refined     : ownership 撤廃、refined mask をそのまま emit (v94)
      - ref_prop    : refined + proposals の 2 仮説 emit (採点面探索)
      - all3        : refined + proposals + final
  (b) final pred 1 件ごとの推論時特徴量を CSV 化する (v95 ゲート用)
      conf / 面積系 / prop↔ref 一致 / pred 間重なり / 形状 /
      photometric contrast / disk 内位置

使い方:
    python scratch/run_v94_v95_emission.py \
        --ann <train.json> --imgdir <train_images/> \
        --val-list <foldK.txt> --cache output/error_decomp_foldK/cache \
        --out output/v95_features/foldK.csv --arms-out <json>

GT 側の情報は ratio ラベル計算と arm 評価のみに使い、特徴量には
一切混入しない (v95 のリーク防止)。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tile_infer_probe import to_rle  # noqa: E402
from solar_filament.metrics.metrics_official import PanopticQuality  # noqa: E402
from analyze_error_decomposition import load_preprocessed  # noqa: E402

H, W = 2048, 2048
CLUSTER_IOU = 0.5


def unpack(packed, count, h=H, w=W):
    if count == 0:
        return []
    flat = np.unpackbits(packed, axis=1)[:, : h * w]
    return [r.reshape(h, w).astype(bool) for r in flat]


def rles(masks):
    if not masks:
        return []
    return mask_util.encode(
        np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))


def iou_matrix(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)))
    return np.asarray(
        mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)),
        dtype=np.float64).T


def mask_props(mask):
    """bbox 形状特徴 (extent/aspect) と重心。"""
    ys, xs = np.where(mask)
    if not len(xs):
        return dict(cx=np.nan, cy=np.nan, bw=np.nan, bh=np.nan,
                    aspect=np.nan, extent=np.nan)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    return dict(cx=float(xs.mean()), cy=float(ys.mean()),
                bw=float(bw), bh=float(bh),
                aspect=float(max(bw, bh) / max(min(bw, bh), 1)),
                extent=float(mask.sum() / max(bw * bh, 1)))


def contrast_features(gray3ch, mask):
    """前処理 3ch 画像の mask 内部 vs 周辺リングの強度差。"""
    out = {}
    inner = mask.astype(np.uint8)
    ring = cv2.dilate(inner, np.ones((15, 15), np.uint8)).astype(bool) & ~mask
    for ch, name in enumerate(("raw", "clahe", "blackhat")):
        chan = gray3ch[..., ch].astype(np.float32)
        m_in = float(chan[mask].mean()) if mask.any() else np.nan
        m_out = float(chan[ring].mean()) if ring.any() else np.nan
        out[f"{name}_in"] = m_in
        out[f"{name}_contrast"] = m_in - m_out
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ann", type=Path, required=True)
    p.add_argument("--imgdir", type=Path, required=True)
    p.add_argument("--val-list", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="per-final-pred 特徴量 CSV")
    p.add_argument("--arms-out", type=Path, default=None,
                   help="emission arm 別 PQ JSON")
    p.add_argument("--skip-features", action="store_true")
    args = p.parse_args()

    val_names = [ln.strip() for ln in args.val_list.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
    with open(args.ann, encoding="utf-8") as fh:
        coco = json.load(fh)
    images = {i["id"]: i for i in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    f2r = defaultdict(list)
    for iid, info in images.items():
        f2r[info["file_name"]].append(iid)

    arm_pq = {a: PanopticQuality()
              for a in ("final", "refined", "ref_prop", "all3")}
    feat_rows = []

    for idx, name in enumerate(val_names, 1):
        stem = Path(name).stem
        saved = np.load(args.cache / f"{stem}.npz")
        prop = unpack(saved["prop_packed"], int(saved["prop_n"]))
        ref = unpack(saved["ref_packed"], int(saved["ref_n"]))
        fin = unpack(saved["fin_packed"], int(saved["fin_n"]))
        confs = saved["confs"].astype(np.float32)

        pr, rr, fr = rles(prop), rles(ref), rles(fin)

        # final -> 親 proposal (IoU argmax) で conf・面積比を復元
        n_fin = len(fin)
        parent = np.full(n_fin, -1, np.int64)
        if n_fin and pr:
            m_fp = iou_matrix(fr, pr)            # (n_fin, n_prop)
            parent = m_fp.argmax(axis=1)
            ok = m_fp[np.arange(n_fin), parent] > 0.0
            parent[~ok] = -1

        # prop 同士の重なり構造
        n_prop = len(prop)
        prop_overlap_max = np.zeros(n_prop)
        prop_overlap_cnt = np.zeros(n_prop)
        if n_prop > 1:
            m_pp = iou_matrix(pr, pr)
            np.fill_diagonal(m_pp, 0.0)
            prop_overlap_max = m_pp.max(axis=1)
            prop_overlap_cnt = (m_pp > 0.3).sum(axis=1)

        # GT (評価・ラベルのみ、特徴に混入しない)
        rids = sorted(f2r[name])
        gt_by_rec = {}
        for rid in rids:
            info = images[rid]
            gt_by_rec[rid] = [to_rle(a, int(info["height"]),
                                     int(info["width"]))
                              for a in anns[rid]]

        # --- emission arms ---
        arm_sets = {"final": fr, "refined": rr,
                    "ref_prop": rr + pr, "all3": rr + pr + fr}
        for rid in rids:
            g = gt_by_rec[rid]
            mat_fin = iou_matrix(g, fr)
            for arm, rset in arm_sets.items():
                if arm == "final":
                    mat = mat_fin
                else:
                    mat = iou_matrix(g, rset)
                arm_pq[arm].add_matrix(mat, len(g), len(rset))

        # --- per-final-pred 特徴 (GT を使わない部分) ---
        if not args.skip_features:
            image = load_preprocessed(args.imgdir / name)
            disk = image[..., 0] > 8
            dys, dxs = np.where(disk)
            dcx = float(dxs.mean()) if len(dxs) else W / 2
            dcy = float(dys.mean()) if len(dys) else H / 2
            drad = float(np.sqrt(disk.sum() / math.pi))
            for j in range(n_fin):
                f = fin[j]
                pj = parent[j]
                pm = mask_props(f)
                feats = {
                    "file_name": name, "fin_idx": j,
                    "conf": float(confs[pj]) if pj >= 0 else np.nan,
                    "fin_area": float(f.sum()),
                    "prop_area": float(prop[pj].sum()) if pj >= 0 else np.nan,
                    "ref_area": float(ref[pj].sum()) if pj >= 0 else np.nan,
                    "n_prop_photo": n_prop, "n_fin_photo": n_fin,
                    "n_records": len(rids),
                    "cx": pm["cx"], "cy": pm["cy"],
                    "aspect": pm["aspect"], "extent": pm["extent"],
                    "prop_overlap_max": float(prop_overlap_max[pj])
                    if pj >= 0 else np.nan,
                    "prop_overlap_cnt": float(prop_overlap_cnt[pj])
                    if pj >= 0 else np.nan,
                }
                if pj >= 0 and prop[pj].any():
                    feats["shrink_ref_prop"] = float(
                        ref[pj].sum() / max(prop[pj].sum(), 1))
                    inter = float((prop[pj] & ref[pj]).sum())
                    union = float((prop[pj] | ref[pj]).sum())
                    feats["iou_prop_ref"] = inter / max(union, 1.0)
                else:
                    feats["shrink_ref_prop"] = np.nan
                    feats["iou_prop_ref"] = np.nan
                feats["radial"] = float(
                    math.hypot(pm["cx"] - dcx, pm["cy"] - dcy)
                    / max(drad, 1.0)) if not np.isnan(pm["cx"]) else np.nan
                feats.update(contrast_features(image, f))
                feat_rows.append(feats)
        if idx % 20 == 0 or idx == len(val_names):
            print(f"{idx}/{len(val_names)}", flush=True)

    arm_summary = {a: dict(pq=q.summary()["pq"], tp=q.summary()["tp"],
                           fp=q.summary()["fp"], fn=q.summary()["fn"])
                   for a, q in arm_pq.items()}
    print(json.dumps(arm_summary, indent=2))
    if args.arms_out:
        args.arms_out.parent.mkdir(parents=True, exist_ok=True)
        args.arms_out.write_text(json.dumps(arm_summary, indent=2))
    if not args.skip_features:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(feat_rows).to_csv(args.out, index=False)
        print(f"features -> {args.out} ({len(feat_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
