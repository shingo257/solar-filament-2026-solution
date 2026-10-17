"""v72 決定評価: SSL verifier スコアで提案を閾値フィルタし fold2 OOF PQ を測る。

RESEARCH_LOG §2.22 (v70 棄却) → §2.25 (v72 SimCLR 事前学習):
GONG 事前学習で verifier fold2 AUC が 0.644→0.699 に改善し、blend AUC
0.802 > conf 0.797 を初めて超えた。ただし AUC の改善が PQ に繋がるかは別問題。
本スクリプトは prod と同一 operating point で「verifier スコアが低い提案を
refine 前に落とす」効果を fold2 OOF で直接測定する。

設計 (min_area sweep §2.23 と同型):
  - 提案源: prediction_cache/oof/yolo_fold_2 の npz (conf>=0.08 全提案)。
    verifier の prop_idx は npz index に一致 (build_verifier_dataset.py)。
  - prod operating point 再現: conf>=0.26 かつ prop.sum()>=20 のみ残す。
    sanity gate: verifier フィルタ無し (t=0) で PQ ≈ 0.4442 を再現するはず。
  - verifier arm: 上記に加えて verifier_prob >= t でフィルタ
    (t は score 分布の分位から選ぶ)。
  - refine は t=0 の提案集合 (全 conf>=0.26) に1回だけ実行し npz cache。
    閾値 sweep は ownership+PQ のみなので高速。

使い方:
    python scratch/run_verifier_filter_eval.py --fold 2 \
        --verifier-scores output/verifier_v1/scores_ssl_fold2.npy \
        --verifier-labels output/verifier_v1/labels_fold2.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_refiner_pq_allfold import refine_crops, square_box  # noqa: E402
from run_tile_infer_probe import PREPROC, TRAIN_JSON, to_rle  # noqa: E402
from solar_filament.pipeline.ownership import apply_ownership  # noqa: E402
from src.mediseg_models import MediSegEdgeAttUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402

DRIVE = Path("G:/マイドライブ")
V30 = DRIVE / "solar_filament_v30"
ARTIFACTS = V30 / "artifacts"
FOLDS_CSV = V30 / "artifacts/grouped_folds.csv"
OOF = DRIVE / "solar_filament_v34/prediction_cache/oof"

DETECT_CONF_KEEP = 0.26       # prod 同一の conf floor
MIN_AREA = 20                 # prod 同一
REFINER_THRESHOLD = 0.45      # prod 同一
DROP_THRESHOLD = 0.90         # prod 同一 (containment)

# sweep する verifier スコア閾値 (0 はフィルタ無し = prod 再現 gate)
VTHR = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

CACHE_VERSION = 1


def config_fingerprint(scores_path, labels_path, refiner_ckpt, fold) -> str:
    """cache npz に埋め込む入力指紋。

    verifier スコア/ラベル・refiner 重み・operating point のいずれかが
    変わると指紋が変化し、古い cache が誤って再利用されるのを防ぐ。
    """
    h = hashlib.sha256()
    for path in (Path(scores_path), Path(labels_path), Path(refiner_ckpt)):
        h.update(Path(path).name.encode())
        h.update(hashlib.sha256(Path(path).read_bytes()).digest())
    h.update(json.dumps({
        "fold": fold, "conf": DETECT_CONF_KEEP, "min_area": MIN_AREA,
        "ref_thr": REFINER_THRESHOLD, "drop": DROP_THRESHOLD,
        "v": CACHE_VERSION}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def obs_fingerprint(base_fp: str, *paths: Path) -> str:
    """観測毎の cache 指紋: グローバル指紋 + OOF 提案 npz + 前処理画像。

    cache は OOF npz 由来の keep_idx/confs/vscores と前処理画像由来の
    refined mask を保持するため、これらが再生成されると古い cache を
    再利用してはいけない。照合のたびに入力バイトを hash する。
    """
    h = hashlib.sha256(base_fp.encode())
    for path in paths:
        path = Path(path)
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def load_cache(cache_file: Path, fingerprint: str):
    """指紋一致時のみ npz を返す。非存在・指紋不一致は None (再計算)。"""
    if not cache_file.exists():
        return None
    saved = np.load(cache_file)
    stored = saved["fingerprint"].item() if "fingerprint" in saved.files else None
    return saved if stored == fingerprint else None


def load_proposals(npz_path):
    saved = np.load(npz_path)
    packed, (h, w), confs = saved["packed"], saved["shape"], saved["confidences"]
    n = packed.shape[0]
    if n == 0:
        return [], confs.astype(np.float32)
    flat = np.unpackbits(packed.reshape(n, -1), axis=1)[:, : h * w]
    return [row.reshape(h, w).astype(bool) for row in flat], confs.astype(
        np.float32)


def pack_masks(masks):
    if not masks:
        return np.zeros((0, 0), np.uint8), 0
    flat = np.stack([m.astype(np.uint8).reshape(-1) for m in masks])
    return np.packbits(flat, axis=1), len(masks)


def unpack_masks(packed, count):
    if count == 0:
        return []
    flat = np.unpackbits(packed, axis=1)[:, :2048 * 2048]
    return [row.reshape(2048, 2048).astype(bool) for row in flat]


def refine_proposals(refiner, image, proposals):
    boxes = [square_box(p) for p in proposals]
    probabilities = refine_crops(refiner, image, boxes) if boxes else []
    refined = []
    for prop, (x0, y0, x1, y1), prob in zip(proposals, boxes, probabilities):
        mask = np.zeros_like(prop, dtype=bool)
        mask[y0:y1, x0:x1] = prob >= REFINER_THRESHOLD
        refined.append(mask & prop)
    return refined


def masks_to_rles(masks):
    if not masks:
        return []
    return mask_util.encode(
        np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))


def iou_matrix(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    return np.asarray(
        mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64).T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verifier-scores", type=Path, required=True,
                        help="scores_fold{K}.npy (labels CSV と同順)")
    parser.add_argument("--verifier-labels", type=Path, required=True,
                        help="labels_fold{K}.csv (file_name,prop_idx,conf,...)")
    parser.add_argument("--cache", type=Path,
                        default=Path("output/verifier_filter/cache"))
    parser.add_argument("--out", type=Path, default=Path("output/verifier_filter"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    # verifier スコア: (file_name, prop_idx) -> prob
    labels = pd.read_csv(args.verifier_labels)
    labels = labels[labels["fold"] == args.fold]
    scores = np.load(args.verifier_scores)
    if len(labels) != len(scores):
        raise ValueError(f"labels {len(labels)} != scores {len(scores)}")
    vmap: dict[str, dict[int, float]] = defaultdict(dict)
    for (_, row), score in zip(labels.iterrows(), scores):
        vmap[row.file_name][int(row.prop_idx)] = float(score)

    ckpt = ARTIFACTS / f"mediseg_edgeattnet_refiner_fold{args.fold}_resume.pt"
    assert ckpt.exists(), f"refiner 重みなし: {ckpt}"
    fingerprint = config_fingerprint(
        args.verifier_scores, args.verifier_labels, ckpt, args.fold)
    refiner = MediSegEdgeAttUNet(3, 1, False).to(args.device).eval()
    checkpoint = torch.load(ckpt, map_location="cpu")
    refiner.load_state_dict(
        checkpoint.get("best_model") or checkpoint["model"], strict=True)

    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
    annotations: dict = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    file_to_ids: dict = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)

    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    observations = sorted(n for n, f in fold_of_file.items()
                          if f == args.fold and n in file_to_ids)
    if args.limit:
        observations = observations[:args.limit]
    print({"fold": args.fold, "観測": len(observations)})

    pq = {t: PanopticQuality() for t in VTHR}
    counts = {t: 0 for t in VTHR}
    rows = []
    unmatched = matched = 0
    started = time.perf_counter()
    for index, name in enumerate(observations, 1):
        stem = Path(name).stem
        cache_file = args.cache / f"{stem}_fold{args.fold}.npz"
        npz_path = OOF / f"yolo_fold_{args.fold}" / f"{stem}.npz"
        obs_fp = obs_fingerprint(fingerprint, npz_path, PREPROC / name)
        saved = load_cache(cache_file, obs_fp)
        if saved is not None:
            keep_idx = saved["keep_idx"].astype(int).tolist()
            confs = saved["confs"].astype(np.float32)
            vscores = saved["vscores"].astype(np.float32)
            refined = unpack_masks(saved["ref_packed"], int(saved["ref_n"]))
        else:
            # npz 全提案から prod operating point のサブセットを抽出
            proposals, npz_confs = load_proposals(npz_path)
            keep_idx = [i for i, (p, c) in enumerate(zip(proposals, npz_confs))
                        if c >= DETECT_CONF_KEEP and p.sum() >= MIN_AREA]
            obs_vmap = vmap.get(name, {})
            vscores = np.array([obs_vmap.get(i, np.nan) for i in keep_idx],
                               dtype=np.float32)
            confs = npz_confs[keep_idx].astype(np.float32)
            kept = [proposals[i] for i in keep_idx]
            image = cv2.imdecode(
                np.fromfile(str(PREPROC / name), dtype=np.uint8),
                cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(PREPROC / name)
            refined = refine_proposals(refiner, image, kept)
            ref_packed, ref_n = pack_masks(refined)
            np.savez_compressed(
                cache_file, keep_idx=np.array(keep_idx, dtype=np.int32),
                confs=confs, vscores=vscores,
                ref_packed=ref_packed, ref_n=ref_n,
                fingerprint=obs_fp)
        matched += int(np.sum(~np.isnan(vscores)))
        unmatched += int(np.sum(np.isnan(vscores)))

        arm_final = {}
        for t in VTHR:
            keep = np.isnan(vscores) | (vscores >= t) if t > 0 else \
                np.ones(len(vscores), bool)
            final = apply_ownership(
                [m for m, k in zip(refined, keep) if k],
                confs[keep],
                rule="drop_contained_then_area_asc",
                containment_threshold=DROP_THRESHOLD, min_area=MIN_AREA)
            arm_final[t] = masks_to_rles(final)
            counts[t] += len(final)

        for image_id in file_to_ids[name]:
            info = images[image_id]
            gt = [to_rle(a, int(info["height"]), int(info["width"]))
                  for a in annotations[image_id]]
            for t, rles in arm_final.items():
                pq[t].add_matrix(iou_matrix(gt, rles), len(gt), len(rles))
        rows.append({"file_name": name, "proposals": len(refined),
                     "vscore_missing": int(np.sum(np.isnan(vscores)))})
        if index % 10 == 0 or index == len(observations):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(observations)}  {elapsed:.0f}s  "
                  f"残見積 {elapsed / index * (len(observations) - index):.0f}s",
                  flush=True)

    pd.DataFrame(rows).to_csv(args.out / f"per_obs_fold{args.fold}.csv",
                              index=False)
    base_pq = float(pq[0.0].score)
    summary = {"fold": args.fold, "observations": len(observations),
               "vscore_match": {"matched": int(matched),
                                "unmatched": int(unmatched)},
               "prod_gate_pq": base_pq,
               "thresholds": {},
               "wall_s": round(time.perf_counter() - started, 1)}
    for t in VTHR:
        arm_pq = float(pq[t].score)
        summary["thresholds"][str(t)] = {
            "pq": arm_pq, "delta_vs_gate": arm_pq - base_pq,
            "final_count": counts[t], **pq[t].summary()}
    (args.out / f"summary_fold{args.fold}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
