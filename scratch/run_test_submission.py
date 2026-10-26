"""test 180画像の推論 → submission.csv をローカル生成する(ファイル提出、GPU不要)。

本番構成 (master pipeline cell12 と同一):
  solar_three_channel → YOLO fold2 @1280 → EdgeAtt all-data refiner
  → drop_contained@0.90 → 面積昇順 ownership → RLE

--mode a1 では本番の conf>=0.26 静的フィルタを、conf>=0.08 候補の
scorer 選択 (û = ĥ*m̂ > 0.22、29特徴) に置き換える (PLAN §7.4 案(a))。
特徴は `run_v80_hybrid_candidates.py` の式を test 用にミラーしたもので、
列順は `a1_scorer_meta.json` の feature_names が正本。

使い方:
    python scratch/run_test_submission.py --mode prod
    python scratch/run_test_submission.py --mode a1
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))

from preprocess_pseudo_label_images import solar_three_channel  # noqa: E402
from run_refiner_pq_allfold import refine_crops, square_box  # noqa: E402
from run_v80_hybrid_candidates import (  # noqa: E402
    _geometry_features,
    _refiner_stats,
    _region_stats,
    _ring,
)
from solar_filament.pipeline.io_images import (  # noqa: E402
    imread_unicode_with_retry,
)
from solar_filament.pipeline.ownership import apply_ownership  # noqa: E402
from solar_filament.pipeline.paths import (  # noqa: E402
    artifacts_dir,
    resolve_drive,
)
from src.mediseg_models import MediSegEdgeAttUNet, MediSegPlainMHSAUNet  # noqa: E402

# --- 本番確定値 (cell12 / V52 提出と同一) --------------------------------------
YOLO_IMGSZ = 1280
PREDICT_IOU = 0.55
SUBMISSION_CONF = 0.26          # prod の静的 conf フィルタ
PREDICT_CONF = max(0.08, SUBMISSION_CONF * 0.75)  # = 0.195 (cell12 の式そのまま)
SCORER_MIN_CONF = 0.08          # A1 の候補フロア
REFINER_THRESHOLD = 0.45
DROP_THRESHOLD = 0.90
MIN_AREA = 20
TAU_STAR = 0.22                 # A1 scorer 固定点
N_ANNOTATORS_IMPUTE = 2.0       # test 補完: 学習時中央値≈2 (PLAN §7.4, a1_scorer_meta.json)
SINGLE_FOLD = 2
CROP_SIZE = 320
CROP_CONTEXT = 1.65

# refiner バリアント: ファイル名 → モデル種別
REFINER_VARIANTS = {
    "edgeatt_all": ("mediseg_edgeattnet_refiner_all_data_resume.pt", "edgeatt"),
    # ablation 最良 plain-MHSA (fold2 seed8407 OOF PQ 0.44630、edge prior なし)。
    # matched-param ablation で edge prior 固有効果が未確認のため提出候補。
    "plain_mhsa_f2": ("ablation_plain_mhsa_seed8407_fold2.pt", "plain_mhsa"),
    # ablation 単一セル最良 edge-att (fold2 seed8407 OOF PQ 0.44749)。
    # all-data 版とのデータ量差の対照としての提出候補。
    "edgeatt_f2": ("ablation_edge_att_seed8407_fold2.pt", "edgeatt"),
}

# Kaggle 重み dataset (shingo257/solar-filament-2026-weights) 側のファイル名。
# --weights-dir にダウンロードした dataset を渡すと Drive 無しで再現できる。
DATASET_FILENAMES = {
    "yolo": f"yolo_fold{SINGLE_FOLD}_best.pt",
    "edgeatt_all": "refiner_all_data.pt",
    "plain_mhsa_f2": "ablation_plain_mhsa_seed8407_fold2.pt",
    "edgeatt_f2": "ablation_edge_att_seed8407_fold2.pt",
}

# repo 同梱 weights/ (best_model 抽出名) と Kaggle dataset 名の両方を受理
PLAIN_MHSA_ALT_NAME = "refiner_plain_mhsa_fold2_s8407.pt"


def resolve_weights_dir_paths(weights_dir: Path, refiner: str):
    """--weights-dir 内の yolo/refiner 重みパスを解決して返す。"""
    if not weights_dir.is_dir():
        raise FileNotFoundError(
            f"--weights-dir が存在しません: {weights_dir}")
    yolo_src = weights_dir / DATASET_FILENAMES["yolo"]
    refiner_src = weights_dir / DATASET_FILENAMES[refiner]
    if not refiner_src.exists() and refiner == "plain_mhsa_f2":
        alt = weights_dir / PLAIN_MHSA_ALT_NAME
        if alt.exists():
            refiner_src = alt
    missing = [p.name for p in (yolo_src, refiner_src) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"--weights-dir に重みがありません: {missing} "
            f"({weights_dir})")
    return yolo_src, refiner_src


# 29特徴ベース列 (FEATURE_NAMES から prob_* 4列を除いた順序をここでも固定)
BASE_FEATURES_NO_PROB = (
    "log_area", "log_area_prop", "shrink_ratio", "aspect", "elongation",
    "solidity", "bbox_w_norm", "bbox_h_norm", "rho_centroid", "rho_edge",
    "img_mean_in", "img_p05_in", "img_mean_ring",
    "ref_mean_in", "ref_max_in", "ref_frac45_in", "ref_ncomp",
    "yolo_conf", "level_index", "n_children", "n_siblings",
    "parent_area_ratio", "source_tree",
)
CONTEXT_FEATURES = (
    "max_iou_any", "n_overlap_10", "yolo_support_conf",
    "n_cand_img", "n_yolo_img", "n_annotators",
)
NA_FEATURES = ("level_index", "n_children", "n_siblings",
               "parent_area_ratio", "yolo_support_conf")


def result_masks(result) -> tuple[list[np.ndarray], np.ndarray]:
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = (
        result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        if result.boxes is not None else np.zeros(0, np.float32)
    )
    return output, confidences


def candidate_base_features(proposal, final, prob_crop, box, gray, conf):
    """run_v80_hybrid_candidates.py の yolo 候補 feats 行 (prob_* を除く)。"""
    geo = _geometry_features(final)
    img_in, img_p05, _, _, img_ring = _region_stats(gray, final, _ring(final))
    x0, y0, x1, y1 = box
    prop_crop = proposal[y0:y1, x0:x1]
    ref_in = _refiner_stats(prop_crop, prob_crop, REFINER_THRESHOLD)
    n_comp = float(cv2.connectedComponents(
        final.astype(np.uint8), connectivity=8)[0] - 1)
    return (
        geo[0], float(np.log1p(proposal.sum())),
        final.sum() / max(float(proposal.sum()), 1.0),
        *geo[1:], img_in, img_p05, img_ring,
        *ref_in, n_comp,
        float(conf), -1.0, -1.0, -1.0, -1.0, 0.0,  # 木由来列は -1 → NaN 化
    )


def context_columns(final_rles, n_cand):
    """CONTEXT_FEATURES の値 (yolo のみの画像で source=0 前提)。"""
    if n_cand == 0:
        return np.zeros((0, len(CONTEXT_FEATURES)), np.float32)
    self_iou = np.asarray(
        mask_util.iou(final_rles, final_rles, [0] * n_cand), dtype=np.float64)
    np.fill_diagonal(self_iou, 0.0)
    ctx = np.zeros((n_cand, len(CONTEXT_FEATURES)), np.float32)
    ctx[:, 0] = self_iou.max(axis=1)
    ctx[:, 1] = (self_iou > 0.10).sum(axis=1)
    ctx[:, 2] = np.nan          # yolo_support_conf: yolo 候補は常に NaN
    ctx[:, 3] = n_cand
    ctx[:, 4] = float(n_cand)   # n_yolo_img = n_cand (tree 候補なし)
    ctx[:, 5] = N_ANNOTATORS_IMPUTE
    return ctx


def _run_submission(local: Path) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prod", "a1"), default="prod")
    parser.add_argument("--refiner", choices=tuple(REFINER_VARIANTS),
                        default="edgeatt_all",
                        help="prod モードで使う refiner バリアント")
    parser.add_argument("--drive", type=Path, default=None)
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="Kaggle 重み dataset のダウンロード先。指定時は"
                             "Drive の checkpoint ではなくこのディレクトリから"
                             "重みを解決する(test画像は引き続き Drive)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--device", type=str, default="cpu",
                        help="refiner/YOLO の推論デバイス (例: cuda)")
    parser.add_argument("--out", type=Path, default=Path("output/test_submission"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    drive = resolve_drive(args.drive)
    artifacts = artifacts_dir(drive)
    test_dir = (drive / "solar_filament_v30" / "competition_data"
                / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images")
    test_files = sorted(p.name for p in test_dir.glob("*.jpeg"))
    assert len(test_files) == 180, f"test画像が180枚ではありません: {len(test_files)}"
    if args.limit:
        test_files = test_files[:args.limit]

    # --- ローカル ASCII ステージング (非ASCII Drive パスを YOLO に渡さない) ----
    local_img = local / "images"
    local_w = local / "weights"
    for d in (local_img, local_w):
        d.mkdir(parents=True, exist_ok=True)

    refiner_filename, refiner_kind = REFINER_VARIANTS[args.refiner]
    if args.weights_dir is not None:
        yolo_src, refiner_src = resolve_weights_dir_paths(
            args.weights_dir, args.refiner)
    else:
        yolo_src = (drive / "solar_filament_v30" / "runs"
                    / f"yolo_fold_{SINGLE_FOLD}" / "weights" / "best.pt")
        refiner_src = artifacts / refiner_filename
    if args.mode == "a1" and args.refiner != "edgeatt_all":
        raise ValueError("--refiner バリアントは prod モード専用です")
    yolo_path = local_w / "yolo_fold2.pt"
    refiner_path = local_w / f"refiner_{args.refiner}.pt"
    for src, dst in ((yolo_src, yolo_path), (refiner_src, refiner_path)):
        print(f"copy {src.name} -> {dst}", flush=True)
        shutil.copy2(src, dst)

    scorer_h = scorer_m = None
    if args.mode == "a1":
        scorer_base = args.weights_dir if args.weights_dir is not None else artifacts
        with open(scorer_base / "a1_scorer_h.pkl", "rb") as f:
            scorer_h = pickle.load(f)
        with open(scorer_base / "a1_scorer_m.pkl", "rb") as f:
            scorer_m = pickle.load(f)
        meta = json.loads((scorer_base / "a1_scorer_meta.json").read_text(encoding="utf-8"))
        expected = list(BASE_FEATURES_NO_PROB) + list(CONTEXT_FEATURES)
        assert meta["feature_names"] == expected, (
            f"meta の特徴順と本スクリプトが不一致: {meta['feature_names']}")
        print({"a1": "scorer読込", "特徴": len(expected), "τ": meta["tau_star"]},
              flush=True)

    # --- 前処理 (solar_three_channel を raw test JPEG に適用、q96) ------------
    for name in test_files:
        dst = local_img / name
        gray = None
        for attempt in range(4):
            raw = None
            try:
                raw = np.frombuffer((test_dir / name).read_bytes(), np.uint8)
            except OSError:
                time.sleep(2.0 * (attempt + 1))
                continue
            gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                break
            time.sleep(2.0 * (attempt + 1))
        if gray is None:
            raise FileNotFoundError(f"test画像を読めません: {name}")
        cv2.imwrite(str(dst), solar_three_channel(gray),
                    [cv2.IMWRITE_JPEG_QUALITY, 96])
    print({"test画像": len(test_files), "stage": str(local_img)}, flush=True)

    from ultralytics import YOLO
    yolo_model = YOLO(str(yolo_path))
    refiner_cls = (MediSegPlainMHSAUNet if refiner_kind == "plain_mhsa"
                   else MediSegEdgeAttUNet)
    refiner = refiner_cls(3, 1, False).eval().to(args.device)
    checkpoint = torch.load(refiner_path, map_location="cpu")
    refiner.load_state_dict(
        checkpoint.get("best_model") or checkpoint["model"], strict=True)

    predict_conf = SCORER_MIN_CONF if args.mode == "a1" else PREDICT_CONF
    rows, audit = [], []
    started = time.perf_counter()
    for index, name in enumerate(test_files, 1):
        image_path = local_img / name
        image = imread_unicode_with_retry(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        result = yolo_model.predict(
            str(image_path), imgsz=YOLO_IMGSZ, conf=predict_conf,
            iou=PREDICT_IOU, agnostic_nms=True, retina_masks=True,
            device=args.device, verbose=False)[0]
        masks, confidences = result_masks(result)

        if args.mode == "prod":
            keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                    if s >= SUBMISSION_CONF and m.sum() >= MIN_AREA]
            proposals = [masks[i] for i in keep]
            boxes = [square_box(p) for p in proposals]
            probabilities = refine_crops(refiner, image, boxes) if boxes else []
            refined = []
            for i, (prop, (x0, y0, x1, y1)) in enumerate(zip(proposals, boxes)):
                mask = np.zeros_like(prop, dtype=bool)
                mask[y0:y1, x0:x1] = probabilities[i] >= REFINER_THRESHOLD
                refined.append(mask & prop)
            final_masks = apply_ownership(
                refined, confidences[keep] if keep else np.zeros(0),
                rule="drop_contained_then_area_asc",
                containment_threshold=DROP_THRESHOLD, min_area=MIN_AREA)
            audit.append({"file_name": name, "proposals": len(proposals),
                          "submitted": len(final_masks)})
        else:  # a1
            keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                    if s >= SCORER_MIN_CONF and m.sum() >= MIN_AREA]
            proposals = [masks[i] for i in keep]
            kept_conf = confidences[keep] if keep else np.zeros(0)
            boxes = [square_box(p) for p in proposals]
            probabilities = refine_crops(refiner, image, boxes) if boxes else []
            finals = []
            for i, (prop, (x0, y0, x1, y1)) in enumerate(zip(proposals, boxes)):
                mask = np.zeros_like(prop, dtype=bool)
                mask[y0:y1, x0:x1] = probabilities[i] >= REFINER_THRESHOLD
                finals.append(mask & prop)
            usable = [f.sum() >= MIN_AREA for f in finals]

            if finals:
                base = np.stack([
                    candidate_base_features(proposals[i], finals[i],
                                            probabilities[i], boxes[i], gray,
                                            kept_conf[i])
                    for i in range(len(finals))
                ])
                final_rles = mask_util.encode(np.asfortranarray(
                    np.stack(finals, axis=-1).astype(np.uint8)))
                ctx = context_columns(final_rles, len(finals))
                X = np.concatenate([base.astype(np.float64),
                                    ctx.astype(np.float64)], axis=1)
                for j, name_ in enumerate(expected):
                    if name_ in NA_FEATURES:
                        col = X[:, j]
                        X[:, j] = np.where(col < 0, np.nan, col)
                u_hat = scorer_h.predict(X) * scorer_m.predict(X)
                selected = [i for i in range(len(finals))
                            if usable[i] and u_hat[i] > TAU_STAR]
            else:
                selected = []
            final_masks = apply_ownership(
                [finals[i] for i in selected],
                kept_conf[selected] if selected else np.zeros(0),
                rule="drop_contained_then_area_asc",
                containment_threshold=DROP_THRESHOLD, min_area=MIN_AREA)
            audit.append({"file_name": name, "proposals": len(proposals),
                          "scorer_kept": len(selected),
                          "submitted": len(final_masks)})

        stem = Path(name).stem
        for i_m, mask in enumerate(final_masks, start=1):
            encoded = mask_util.encode(
                np.asfortranarray(mask.astype(np.uint8)))
            rows.append({"filament_id": f"{stem}_{i_m}",
                         "segmentation_rle": encoded["counts"].decode("utf-8")})
        if index % 10 == 0 or index == len(test_files):
            elapsed = time.perf_counter() - started
            print(f"  {index}/{len(test_files)}  {elapsed:.0f}s  "
                  f"残見積 {elapsed / index * (len(test_files) - index):.0f}s",
                  flush=True)

    total_s = time.perf_counter() - started
    suffix = "" if args.refiner == "edgeatt_all" else f"_{args.refiner}"
    tag = f"{args.mode}{suffix}"
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_csv = args.out / f"submission_{tag}.csv"
    submission.to_csv(out_csv, index=False, lineterminator="\n")
    efficiency = {"mode": args.mode, "refiner": args.refiner,
                  "test_images": len(test_files),
                  "total_wall_s": round(total_s, 1),
                  "s_per_image": round(total_s / len(test_files), 3),
                  "device": args.device, "threads": args.threads}
    (args.out / f"efficiency_{tag}.json").write_text(
        json.dumps(efficiency, indent=2), encoding="utf-8")
    (args.out / f"audit_{tag}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    print(json.dumps({"submission": str(out_csv), "rows": len(submission),
                      "images_with_predictions": sum(
                          1 for a in audit if a["submitted"] > 0),
                      "efficiency": efficiency, "code_sha": sha},
                     ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    # Each invocation stages its own inputs; no previous Drive/code generation
    # can leave same-sized weights or same-named preprocessed images behind.
    with tempfile.TemporaryDirectory(prefix="solar_test_") as staging:
        return _run_submission(Path(staging))


if __name__ == "__main__":
    raise SystemExit(main())
