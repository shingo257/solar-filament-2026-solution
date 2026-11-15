"""Generate Figure 4 (consensus_breakthrough.png) from Fold-2 honest OOF caches.

Reproduces exact Fold-2 OOF predictions for single-detector ``prod`` (0.44423 PQ)
vs. 3-detector 1-to-1 ``cons2of3_sptal_maj3`` (0.45803 PQ: ``prod`` + ``yolo26`` +
``spine_small_tal``, 2-of-3 @ 0.26 + ``majority_3plus`` refined crop vote +
area-ascending disjoint ownership), selects genuine representative crops for:
  (a) Small/thin filament TP recovery (missed by single ``prod``, recovered by consensus)
  (b) Spurious false-positive elimination (emitted by ``prod`` with 0 IoU across all
      annotators, rejected by 2-of-3 consensus while preserving adjacent true filament)
and renders ``report/figures/consensus_breakthrough.png``.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]


def _poly_to_rle(seg_list: list, h: int = 2048, w: int = 2048) -> dict:
    rles = mask_utils.frPyObjects(seg_list, h, w)
    return mask_utils.merge(rles)


def _score_pair(pred_rles: list[dict], gt_rles: list[dict]) -> tuple[float, int, int, int]:
    np_p, ng = len(pred_rles), len(gt_rles)
    if np_p == 0 and ng == 0:
        return 0.0, 0, 0, 0
    if np_p == 0:
        return 0.0, 0, 0, ng
    if ng == 0:
        return 0.0, 0, np_p, 0
    ious = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * ng), dtype=np.float64)
    hits = ious > 0.5
    tp = int(hits.sum())
    iou_sum = float(ious[hits].sum())
    pred_matched = int(hits.any(axis=1).sum())
    gt_matched = int(hits.any(axis=0).sum())
    fp = np_p - pred_matched
    fn = ng - gt_matched
    return iou_sum, tp, fp, fn


def _draw_contours(img_rgb: np.ndarray, mask_bool: np.ndarray, color_rgb: tuple[int, int, int], thickness: int = 2, dashed: bool = False) -> None:
    cnts, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not dashed:
        cv2.drawContours(img_rgb, cnts, -1, color_rgb, thickness, lineType=cv2.LINE_AA)
        return
    for cnt in cnts:
        pts = cnt[:, 0, :]
        n = len(pts)
        step = 10
        for s in range(0, n, step * 2):
            sub = pts[s : min(s + step, n)]
            if len(sub) >= 2:
                cv2.polylines(img_rgb, [sub.reshape(-1, 1, 2)], False, color_rgb, thickness, lineType=cv2.LINE_AA)


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from solar_filament.pipeline.consensus_fusion import (
        DetectorProposal,
        fast_apply_own_specs,
        fuse_refined_crop_specs,
        match_detector_proposals_1to1,
        rle_to_crop_spec,
    )

    drive = Path("G:/\u30de\u30a4\u30c9\u30e9\u30a4\u30d6/solar_filament_v30")
    art = drive / "artifacts"
    coco_path = drive / "competition_data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    prep_dir = drive / "preprocessed_v30_q96"

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    anns_by_img: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("segmentation"):
            anns_by_img[str(ann["image_id"])].append(_poly_to_rle(ann["segmentation"]))

    gt_by_file: dict[str, list[list[dict]]] = defaultdict(list)
    for img in coco["images"]:
        gt_by_file[img["file_name"]].append(anns_by_img[str(img["id"])])

    with (art / "v136_oof_caches/prod_f2_best_cache.pkl").open("rb") as f:
        c_prod = {row[0]: row for row in pickle.load(f)}
    with (art / "v136_oof_caches/yolo26_f2_best_cache.pkl").open("rb") as f:
        c_y26 = {row[0]: row for row in pickle.load(f)}
    with (art / "v134_oof_caches/spine_small_tal_f2_best_cache.pkl").open("rb") as f:
        c_sptal = {row[0]: row for row in pickle.load(f)}

    files = sorted(c_sptal.keys())
    tot_prod = [0.0, 0, 0, 0]
    tot_cons = [0.0, 0, 0, 0]

    recovery_cands = []
    fp_clean_cands = []

    for fn in files:
        # 1. Single prod @ 0.26
        _, p_confs, p_areas, _p_raw, p_ref = c_prod[fn]
        p_specs = [
            rle_to_crop_spec(p_ref[i])
            for i in range(len(p_confs))
            if float(p_confs[i]) >= 0.26 and float(p_areas[i]) >= 20.0
        ]
        prod_rles = fast_apply_own_specs(p_specs)

        # 2. 3-det 1-to-1 cons2of3_sptal_maj3 @ 0.26
        props: list[DetectorProposal] = []
        specs_by_det: dict[str, list] = {}
        for dname, cache in (("prod", c_prod), ("yolo26", c_y26), ("sptal", c_sptal)):
            _, confs, areas, raw_rles, ref_rles = cache[fn]
            d_specs = [rle_to_crop_spec(r) for r in ref_rles]
            specs_by_det[dname] = d_specs
            for idx in range(len(confs)):
                if float(confs[idx]) >= 0.08 and float(areas[idx]) >= 20.0:
                    props.append(
                        DetectorProposal(
                            det_name=dname,
                            conf=float(confs[idx]),
                            raw_rle=raw_rles[idx],
                            prop_idx=idx,
                        )
                    )
        clusters = match_detector_proposals_1to1(props, iou_threshold=0.45)
        fused_specs = [
            fuse_refined_crop_specs(cl, specs_by_det, mode="majority_3plus")
            for cl in clusters
            if cl.count_above(0.26) >= 2
        ]
        cons_rles = fast_apply_own_specs(fused_specs)

        annotator_sets = gt_by_file[fn]
        for gt_list in annotator_sets:
            s_p, tp_p, fp_p, fn_p = _score_pair(prod_rles, gt_list)
            s_c, tp_c, fp_c, fn_c = _score_pair(cons_rles, gt_list)
            tot_prod[0] += s_p
            tot_prod[1] += tp_p
            tot_prod[2] += fp_p
            tot_prod[3] += fn_p
            tot_cons[0] += s_c
            tot_cons[1] += tp_c
            tot_cons[2] += fp_c
            tot_cons[3] += fn_c

        # Pool all annotators on this image to find unequivocal recoveries and unequivocal FPs
        all_gt_with_set = [(g, gl) for gl in annotator_sets for g in gl]
        all_gt = [g for g, _ in all_gt_with_set]
        if not all_gt:
            continue

        # Check unequivocal TP recovery (missed by prod with IoU == 0.0, recovered by cons with IoU >= 0.74)
        if cons_rles:
            iou_c_gt = np.asarray(mask_utils.iou(cons_rles, all_gt, [0] * len(all_gt)), dtype=np.float64)
            iou_p_gt = (
                np.asarray(mask_utils.iou(prod_rles, all_gt, [0] * len(all_gt)), dtype=np.float64)
                if prod_rles
                else np.zeros((0, len(all_gt)), dtype=np.float64)
            )
            for gi, (g_rle, g_set) in enumerate(all_gt_with_set):
                best_p = float(iou_p_gt[:, gi].max()) if len(prod_rles) else 0.0
                best_c = float(iou_c_gt[:, gi].max())
                g_area = float(mask_utils.area(g_rle))
                if best_p == 0.0 and best_c >= 0.74 and 450 <= g_area <= 2600:
                    ci = int(iou_c_gt[:, gi].argmax())
                    recovery_cands.append((best_c, g_area, fn, g_rle, cons_rles[ci], prod_rles, cons_rles, g_set))

        # Check unequivocal FP suppression (prod emits an instance with 0 IoU vs ALL annotators,
        # near a real GT filament >= 600px that BOTH detect cleanly, and cons has 0 FP in crop!)
        if prod_rles and cons_rles:
            iou_p_all = np.asarray(mask_utils.iou(prod_rles, all_gt, [0] * len(all_gt)), dtype=np.float64)
            for pi, p_rle in enumerate(prod_rles):
                p_area = float(mask_utils.area(p_rle))
                if float(iou_p_all[pi].max()) == 0.0 and 500 <= p_area <= 3500:
                    px, py, pw, ph = mask_utils.toBbox(p_rle)
                    pcx, pcy = px + pw / 2, py + ph / 2
                    # Find nearby true GT within ~260px so crop shows BOTH true filament + suppressed FP!
                    for g_rle in annotator_sets[0]:
                        gx, gy, gw, gh = mask_utils.toBbox(g_rle)
                        gcx, gcy = gx + gw / 2, gy + gh / 2
                        dist = float(np.hypot(pcx - gcx, pcy - gcy))
                        g_area = float(mask_utils.area(g_rle))
                        if 90 <= dist <= 240 and g_area >= 700:
                            # Ensure cons recovers this GT cleanly and has no overlap with the FP
                            iou_c_g = float(np.asarray(mask_utils.iou(cons_rles, [g_rle], [0])).max())
                            iou_c_fp = float(np.asarray(mask_utils.iou(cons_rles, [p_rle], [0])).max())
                            if iou_c_g >= 0.72 and iou_c_fp == 0.0:
                                fp_clean_cands.append((iou_c_g, p_area, dist, fn, p_rle, g_rle, prod_rles, cons_rles, annotator_sets[0]))

    pq_p = tot_prod[0] / (tot_prod[1] + 0.5 * tot_prod[2] + 0.5 * tot_prod[3])
    pq_c = tot_cons[0] / (tot_cons[1] + 0.5 * tot_cons[2] + 0.5 * tot_cons[3])
    print(f"Verified Fold-2 prod PQ={pq_p:.5f} (TP={tot_prod[1]}, FP={tot_prod[2]}, FN={tot_prod[3]})")
    print(f"Verified Fold-2 cons PQ={pq_c:.5f} (TP={tot_cons[1]}, FP={tot_cons[2]}, FN={tot_cons[3]})")
    assert abs(pq_c - 0.45803) < 1e-4, pq_c

    recovery_cands.sort(key=lambda t: -t[0])
    fp_clean_cands.sort(key=lambda t: (-t[0], t[2]))
    assert recovery_cands and fp_clean_cands, (len(recovery_cands), len(fp_clean_cands))

    rec = recovery_cands[0]
    fpc = fp_clean_cands[0]
    print(f"Selected (a) recovery: file={rec[2]}, IoU_cons={rec[0]:.3f}, area={rec[1]:.0f}px")
    print(f"Selected (b) FP suppression: file={fpc[3]}, nearby_GT_IoU={fpc[0]:.3f}, FP_area={fpc[1]:.0f}px, dist={fpc[2]:.0f}px")

    raw_train_dir = drive / "competition_data" / "MAGFiLO_1.0_Kaggle_2026" / "train"

    def _load_prep_rgb(fn: str) -> np.ndarray:
        p = prep_dir / fn
        if p.exists():
            bgr = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from src.preprocess import solar_three_channel
        rp = raw_train_dir / fn
        gray = cv2.imdecode(np.frombuffer(rp.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        assert gray is not None, rp
        return solar_three_channel(gray)

    def render_crop(fn: str, focus_rles: list[dict], gt_rles: list[dict], prod_rles: list[dict], cons_rles: list[dict], pad: int = 95) -> np.ndarray:
        rgb = _load_prep_rgb(fn)
        h, w, _ = rgb.shape

        boxes = [mask_utils.toBbox(r) for r in focus_rles]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        side = max(x1 - x0, y1 - y0) + 2 * pad
        side = max(side, 280)
        half = round(side / 2)
        ix0 = max(0, min(w - 2 * half, round(cx) - half))
        iy0 = max(0, min(h - 2 * half, round(cy) - half))
        ix1, iy1 = ix0 + 2 * half, iy0 + 2 * half

        canvas = rgb[iy0:iy1, ix0:ix1].copy()

        # Draw GT (lime green), prod (red dashed), cons (gold/yellow solid)
        for r in gt_rles:
            m = (mask_utils.decode(r) != 0)[iy0:iy1, ix0:ix1]
            if m.any():
                _draw_contours(canvas, m, (60, 255, 80), thickness=3, dashed=False)
        for r in prod_rles:
            m = (mask_utils.decode(r) != 0)[iy0:iy1, ix0:ix1]
            if m.any():
                _draw_contours(canvas, m, (255, 65, 65), thickness=2, dashed=True)
        for r in cons_rles:
            m = (mask_utils.decode(r) != 0)[iy0:iy1, ix0:ix1]
            if m.any():
                _draw_contours(canvas, m, (255, 225, 40), thickness=2, dashed=False)
        return canvas

    crop_a = render_crop(rec[2], [rec[3], rec[4]], rec[7], rec[5], rec[6], pad=90)
    crop_b = render_crop(fpc[3], [fpc[4], fpc[5]], fpc[8], fpc[6], fpc[7], pad=85)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.35), dpi=240)
    axes[0].imshow(crop_a)
    axes[0].set_title(f"(a) Small-filament recovery (IoU 0→{rec[0]:.2f})", fontsize=9.5, fontweight="bold", pad=5)
    axes[0].axis("off")

    axes[1].imshow(crop_b)
    axes[1].set_title(f"(b) Solo-FP removal ({int(fpc[1])} px FP suppressed)", fontsize=9.5, fontweight="bold", pad=5)
    axes[1].axis("off")

    legend_handles = [
        Line2D([0], [0], color=(60 / 255, 1.0, 80 / 255), lw=2.4, label="Ground truth"),
        Line2D([0], [0], color=(1.0, 65 / 255, 65 / 255), lw=2.0, ls="--", label="Single prod (LB 0.39)"),
        Line2D([0], [0], color=(1.0, 225 / 255, 40 / 255), lw=2.2, label="3-det. 1-to-1 + maj3 (LB 0.40)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.2,
        columnspacing=1.2,
    )
    plt.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.11, wspace=0.06)

    out_wt = ROOT / "report" / "figures" / "consensus_breakthrough.png"
    fig.savefig(out_wt, dpi=240)
    plt.close(fig)
    print("Wrote:", out_wt, "size:", out_wt.stat().st_size)


if __name__ == "__main__":
    main()
