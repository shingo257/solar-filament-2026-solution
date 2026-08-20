"""画像別診断CSVと worst-case contact sheet を作る (HANDOVER §27)。

なぜ必要か:
  審査は定量70% / 定性30%で、定性側は「予測マスクの形態的妥当性」を明示的に採点する。
  現状の成果物は数値表だけで、**何がどう失敗しているのかを示す図が1枚もない**。
  技術報告書にも図が要る。

  加えて、PQ は全画像を通した micro 集計なので、集計値だけ見ていると
  「どの画像がどれだけ損をさせているか」が見えない。

出力:
  1. `output/v35/case_diagnostics.csv`  — 全707観測の1行診断
  2. `output/v35/contact_sheets/*.png`  — 損失の大きい順に N 枚

per-image PQ について:
  公式 PQ は分母を全画像で共有する micro 集計なので、厳密な「画像ごとの PQ」は
  定義できない。ここでは当該画像の annotator-image だけで同じ式を計算した
  **局所 PQ** を出す。順位付けの道具であって公式値ではない。
  併せて、分母だけを増やして分子に寄与しない量 `loss_units = 0.5*(FP+FN)` も出す。
  こちらの方が「全体スコアをどれだけ押し下げているか」に忠実である。

使い方:
    python scratch/run_case_diagnostics.py [--sheets 30] [--threads 14]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS, FOLDS, FOLDS_CSV, MIN_AREA, OOF_CACHE, PREPROC, TRAIN_JSON,
    build_variant, greedy_disjoint, imread_unicode, iou_matrix, load_yolo_cache,
    masks_to_rles, refine_batch, to_rle,
)

CONF = 0.26
REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45
DISPLAY = 640            # 表示用の縮小サイズ。2048 のまま描くと図が重い
EVAL_SIZE = 2048


def refiner_checkpoint_path(fold: int) -> Path:
    """Drive 直読みが OSError になることがあるので TEMP コピーを優先する。"""
    local = Path(os.environ.get("TEMP", "C:/Temp")) / "solar_diag_ckpts" / f"mediseg_refiner_fold{fold}_resume.pt"
    if local.exists():
        return local
    return ARTIFACTS / f"mediseg_refiner_fold{fold}_resume.pt"


def image_dir() -> Path:
    """707枚を毎回 Drive から読むと同期不良で無言クラッシュすることがある。TEMP コピーを優先する。"""
    local = Path(os.environ.get("TEMP", "C:/Temp")) / "solar_diag_images"
    if local.exists() and any(local.iterdir()):
        return local
    return PREPROC


def oof_cache_dir() -> Path:
    """OOF npz キャッシュも同じ理由で Drive 直読みが途中で OSError になる。TEMP コピーを優先する。"""
    local = Path(os.environ.get("TEMP", "C:/Temp")) / "solar_diag_oof"
    if local.exists() and any(local.iterdir()):
        return local
    return OOF_CACHE


def compute_case(name, fold, refiners, oof_root, image_root, gt_rles, file_to_ids):
    """1画像分の推論と局所PQを計算する。呼び出し側で row と可視化データを使い分ける。"""
    masks, confidences = load_yolo_cache(
        oof_root / f"yolo_fold_{fold}" / f"{Path(name).stem}.npz")
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
            if s >= CONF and m.sum() >= MIN_AREA]
    proposals = [masks[i] for i in keep]
    scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
    order = list(np.argsort(-scores))
    variants = [gt_rles[i] for i in file_to_ids[name]]

    image = imread_unicode(image_root / name)
    if image is None:
        return None, None

    if proposals:
        boxes, probabilities = refine_batch(refiners[fold], image, proposals)
        refined = [build_variant(proposals[i], boxes[i], probabilities[i],
                                 REFINER_METHOD, REFINER_THRESHOLD)
                   for i in range(len(proposals))]
    else:
        refined = []
    final = greedy_disjoint(refined, order)
    final_rles = masks_to_rles(final)

    local = PanopticQuality()
    matched_flags = np.zeros(len(final), dtype=bool)
    gt_hit_total = gt_total = 0
    near = 0
    for truth in variants:
        matrix = iou_matrix(truth, final_rles)
        local.add_matrix(matrix, len(truth), len(final_rles))
        if matrix.size:
            hits = matrix > 0.5
            matched_flags |= hits.any(axis=0)
            gt_hit_total += int(hits.any(axis=1).sum())
            near += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())
        gt_total += len(truth)

    summary = local.summary()
    row = {
        "file_name": name,
        "fold": fold,
        "n_annotators": len(variants),
        "gt_count_mean": float(np.mean([len(v) for v in variants])),
        "gt_count_min": int(min(len(v) for v in variants)),
        "gt_count_max": int(max(len(v) for v in variants)),
        "proposal_count": len(proposals),
        "final_count": len(final),
        "tp": summary["tp"], "fp": summary["fp"], "fn": summary["fn"],
        "local_pq": summary["pq"], "local_sq": summary["sq"], "local_rq": summary["rq"],
        "loss_units": 0.5 * (summary["fp"] + summary["fn"]),
        "near_miss": near,
        "gt_hit_rate": gt_hit_total / max(gt_total, 1),
        "foreground_fraction": float(np.mean([
            mask_util.area(v).sum() for v in variants]) / (EVAL_SIZE ** 2)) if variants else 0.0,
    }
    visuals = (image, variants, proposals, final, matched_flags)
    return row, visuals


def shrink_for_display(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (DISPLAY, DISPLAY),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """表示用の輪郭。塗り潰すと重なりが見えないので輪郭で描く。"""
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    return (dilated.astype(bool)) & ~mask


def draw_panel(axis, base: np.ndarray, layers, title: str):
    """base(グレースケール)の上に (mask, color) を輪郭で重ねる。"""
    axis.imshow(base, cmap="gray", vmin=0, vmax=255)
    canvas = np.zeros((*base.shape, 4), dtype=float)
    for mask, color in layers:
        edge = outline(mask)
        canvas[edge] = color
    axis.imshow(canvas)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheets", type=int, default=30, help="contact sheet を出す枚数")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("output/v35"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sheet_dir = args.out / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(TRAIN_JSON, encoding="utf-8") as handle:
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

    oof_root = oof_cache_dir()
    print(f"reading OOF cache from {oof_root}", flush=True)

    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    targets = [n for n in sorted(file_to_ids)
               if (oof_root / f"yolo_fold_{fold_of_file[n]}" / f"{Path(n).stem}.npz").exists()]
    if args.limit:
        targets = targets[:args.limit]

    image_root = image_dir()
    print(f"reading images from {image_root}", flush=True)

    refiners = {}
    for fold in FOLDS:
        path = refiner_checkpoint_path(fold)
        print(f"loading refiner fold{fold} from {path}", flush=True)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = MediSegUNet(3, 1, False).eval()
        model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)
        refiners[fold] = model

    # 1パス目: 全707観測の診断行だけを作る。画像/マスクは各回で捨てる
    # (全件分を保持すると 2048x2048 のマスクが積み上がってメモリ枯渇するため — worst30 は2パス目で再計算する)
    rows = []
    started = time.perf_counter()
    for position, name in enumerate(targets, 1):
        fold = fold_of_file[name]
        image_started = time.perf_counter()
        row, _visuals = compute_case(name, fold, refiners, oof_root, image_root, gt_rles, file_to_ids)
        if row is None:
            continue
        row["seconds"] = round(time.perf_counter() - image_started, 3)
        rows.append(row)

        if position % 10 == 0 or position == len(targets):
            print(f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s", flush=True)
        if position % 50 == 0:
            pd.DataFrame(rows).to_csv(args.out / "case_diagnostics.csv", index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "case_diagnostics.csv", index=False)
    print(f"\n診断CSV: {args.out / 'case_diagnostics.csv'} ({len(frame)}行)")

    print("\n=== 全体を最も押し下げている観測 (loss_units 降順) ===")
    worst = frame.sort_values("loss_units", ascending=False).head(15)
    print(worst[["file_name", "fold", "gt_count_mean", "final_count",
                 "tp", "fp", "fn", "local_pq", "near_miss"]].to_string(index=False))

    print("\n=== 相関 (何が損失と結びついているか) ===")
    numeric = frame[["loss_units", "local_pq", "gt_count_mean", "proposal_count",
                     "near_miss", "foreground_fraction", "n_annotators"]]
    print(numeric.corr()["loss_units"].round(3).to_string())

    # contact sheet: 損失の大きい順。1パス目では画像を保持していないので worst30 だけ再計算する
    selected = frame.sort_values("loss_units", ascending=False).head(args.sheets)
    print(f"\ncontact sheet 用に worst{len(selected)} を再計算します", flush=True)
    for rank, row in enumerate(selected.itertuples(index=False), 1):
        name = row.file_name
        fold = fold_of_file[name]
        _row, visuals = compute_case(name, fold, refiners, oof_root, image_root, gt_rles, file_to_ids)
        if visuals is None:
            continue
        image, variants, proposals, final, matched_flags = visuals
        base = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (DISPLAY, DISPLAY),
                          interpolation=cv2.INTER_AREA)

        figure, axes = plt.subplots(1, 4, figsize=(18, 5))
        gt0 = [shrink_for_display(mask_util.decode(r).astype(bool)) for r in variants[0]]
        draw_panel(axes[0], base, [(m, (0.2, 1.0, 0.2, 1.0)) for m in gt0],
                   f"GT (annotator 0): {len(gt0)}個")

        palette = [(0.2, 1.0, 0.2, 1.0), (1.0, 0.8, 0.0, 1.0), (0.4, 0.7, 1.0, 1.0)]
        layers = []
        for index, variant in enumerate(variants):
            color = palette[index % len(palette)]
            layers += [(shrink_for_display(mask_util.decode(r).astype(bool)), color)
                       for r in variant]
        draw_panel(axes[1], base, layers,
                   f"GT 全variant ({len(variants)}名, {[len(v) for v in variants]})")

        draw_panel(axes[2], base,
                   [(shrink_for_display(p), (1.0, 0.4, 0.9, 1.0)) for p in proposals],
                   f"YOLO proposals: {len(proposals)}個")

        # TP=緑 / FP=赤。GT側の未検出は panel0/1 と見比べる
        coded = [(shrink_for_display(m),
                  (0.2, 1.0, 0.2, 1.0) if matched_flags[i] else (1.0, 0.25, 0.25, 1.0))
                 for i, m in enumerate(final)]
        draw_panel(axes[3], base, coded,
                   f"最終 (緑=TP {int(row.tp)} / 赤=FP {int(row.fp)}, FN={int(row.fn)})")

        figure.suptitle(
            f"#{rank} {name}  fold{row.fold}  局所PQ={row.local_pq:.3f}  "
            f"loss_units={row.loss_units:.1f}  near-miss={int(row.near_miss)}",
            fontsize=11)
        figure.tight_layout()
        figure.savefig(sheet_dir / f"worst{rank:02d}_{Path(name).stem}.png", dpi=90)
        plt.close(figure)

    print(f"\ncontact sheet: {sheet_dir} に {len(selected)} 枚")
    print(f"所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
