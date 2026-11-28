# 反実仮想実験スクリプト (2026-11-28) — C:/work/solar_counterfactual から保全。
# データセットキャッシュ (selection_dataset*.pkl) はサイズのためコミット対象外、
# Drive の v140 oof cache から本スクリプトで再生成可能。詳細は RESEARCH_LOG 2026-11-28 項。
"""Counterfactual experiment: learned candidate selection on the ERS union3 pool.

Builds the per-cluster dataset (features + PQ-contribution target) from the
v140 OOF caches (fold0/fold2 eval images, hflip view unified), then trains a
cross-fitted LightGBM scorer and evaluates honest held-out PQ vs rule baselines.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, r"C:/work/code/ER-FlowScan/Solar_Filament_Segmentation")
from solar_filament.pipeline.consensus_fusion import (  # noqa: E402
    DetectorProposal,
    match_detector_proposals_1to1,
)

V30 = Path(r"G:/マイドライブ/solar_filament_v30")
PREP = V30 / "preprocessed_v30_q96"
OUT = Path(r"C:/work/solar_counterfactual")

# ---------- GT ----------
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


def load_cache(p):
    return {r[0]: r for r in pickle.load(open(p, "rb"))}


BASE = V30 / "artifacts/v140_final_insurance/oof_caches"
DETS = ("prod", "yolo26", "sptal")


def fold_caches(fold):
    suffix = {
        ("prod", 0): "prod_f0_hflip_ab8810916e34.pkl",
        ("prod", 2): "prod_f2_hflip_40ec31fd2c6a.pkl",
        ("yolo26", 0): "yolo26_f0_hflip_81981abf9b36.pkl",
        ("yolo26", 2): "yolo26_f2_hflip_0c8bb71f6ab4.pkl",
        ("sptal", 0): "sptal_best_f0_hflip_cce140bf6842.pkl",
        ("sptal", 2): "sptal_best_f2_hflip_d79a163925fa.pkl",
    }
    return {d: load_cache(BASE / suffix[(d, fold)]) for d in DETS}


def ring_stats(gray, x0, y0, x1, y1, sub_mask, ring_width=12):
    """mean intensity inside mask and in surrounding ring, computed on the
    cropped bbox+margin patch only (fast)."""
    m = sub_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_width + 1, 2 * ring_width + 1))
    dil = cv2.dilate(m, kernel).astype(bool)
    ero = cv2.erode(m, np.ones((3, 3), np.uint8)).astype(bool) if m.sum() > 40 else sub_mask
    ring = dil & ~m
    inner = ero if ero.sum() >= 10 else sub_mask
    mi = float(gray[inner].mean())
    mr = float(gray[ring].mean()) if ring.sum() >= 10 else mi
    return mi, mr


def extract_fold(fold):
    caches = fold_caches(fold)
    files = sorted(set.intersection(*[set(c) for c in caches.values()]))
    rows = []
    t0 = time.time()
    for fi, fn in enumerate(files):
        proposals = []
        decoded: dict[tuple[str, int], np.ndarray] = {}
        rle_of: dict[tuple[str, int], dict] = {}
        for d in DETS:
            row = caches[d][fn]
            for i in range(len(row[1])):
                proposals.append(DetectorProposal(d, float(row[1][i]), row[4][i], i))
                rle_of[(d, i)] = row[4][i]
                decoded[(d, i)] = mask_utils.decode(row[4][i]).astype(bool)
        clusters = match_detector_proposals_1to1(proposals, iou_threshold=0.45)
        if not clusters:
            continue
        gray = cv2.imdecode(np.fromfile(PREP / fn, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        cx, cy = w / 2, h / 2
        istd = float(gray.std()) + 1e-6

        for rank, cl in enumerate(sorted(clusters, key=lambda c: -c.max_conf())):
            best = cl.best_member()
            repr_rle = rle_of[(best.det_name, best.prop_idx)]
            repr_mask = decoded[(best.det_name, best.prop_idx)]
            area = float(repr_mask.sum())
            if area < 20:
                continue
            ys, xs = np.where(repr_mask)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            bw, bh = x1 - x0, y1 - y0
            aspect = max(bw, bh) / max(1, min(bw, bh))
            extent = area / float(bw * bh)
            cent_r = float(np.hypot(xs.mean() - cx, ys.mean() - cy)) / (w / 2)
            # stability: mean pairwise IoU of member refined masks
            memb_masks = [decoded[(p.det_name, p.prop_idx)] for p in cl.members.values()]
            if len(memb_masks) >= 2:
                ious = []
                for i in range(len(memb_masks)):
                    for j in range(i + 1, len(memb_masks)):
                        inter = float((memb_masks[i] & memb_masks[j]).sum())
                        union = float((memb_masks[i] | memb_masks[j]).sum())
                        ious.append(inter / union if union else 0.0)
                stability = float(np.mean(ious))
            else:
                stability = 0.0
            # cropped intensity stats
            rw = 14
            cx0, cy0 = max(0, x0 - rw), max(0, y0 - rw)
            cx1, cy1 = min(w, x1 + rw), min(h, y1 + rw)
            sub_gray = gray[cy0:cy1, cx0:cx1]
            sub_mask = repr_mask[cy0:cy1, cx0:cx1]
            mi, mr = ring_stats(sub_gray, cx0, cy0, cx1, cy1, sub_mask)
            confs = {d: cl.score_of(d) for d in DETS}
            feats = dict(
                n_members=len(cl.members),
                max_conf=cl.max_conf(),
                mean_conf=float(np.mean(list(cl.score_of(d) for d in cl.members))),
                min_conf_present=min((float(p.conf) for p in cl.members.values())),
                conf_prod=confs["prod"], conf_yolo26=confs["yolo26"], conf_sptal=confs["sptal"],
                n_det_above_026=cl.count_above(0.26),
                mask_stability=stability,
                area=area, log_area=float(np.log1p(area)),
                aspect=aspect, extent=extent, cent_r=cent_r,
                bbox_w=bw, bbox_h=bh,
                rank_in_image=rank,
                n_clusters_image=len(clusters),
                int_inside=mi, int_ring=mr, int_contrast=(mr - mi) / istd, int_diff=mr - mi,
            )
            # target: mean over annotator records of best-match IoU counted 0 when <= 0.5
            contribs = []
            for iid in file_to_ids[fn]:
                tr = gt_rles[iid]
                if not tr:
                    contribs.append(0.0)
                    continue
                v = np.asarray(mask_utils.iou([repr_rle], tr, [0] * len(tr)), np.float64).ravel()
                best_iou = float(v.max()) if v.size else 0.0
                contribs.append(best_iou if best_iou > 0.5 else 0.0)
            feats["target"] = float(np.mean(contribs))
            feats["file"] = fn
            feats["fold"] = fold
            rows.append((feats, repr_rle))
        if (fi + 1) % 50 == 0:
            print(f"  fold{fold}: {fi+1}/{len(files)} images, {len(rows)} clusters, {time.time()-t0:.0f}s", flush=True)
    print(f"fold{fold} done: {len(rows)} clusters in {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    all_rows = []
    for fold in (0, 2):
        all_rows.extend(extract_fold(fold))
    with open(OUT / "selection_dataset.pkl", "wb") as f:
        pickle.dump(all_rows, f)
    print(f"total clusters: {len(all_rows)}")


if __name__ == "__main__":
    main()
