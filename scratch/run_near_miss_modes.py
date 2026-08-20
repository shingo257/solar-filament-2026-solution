"""V35 確定パイプライン上の near-miss を被覆モードで分類する。

run_case_diagnostics.py と同じ設定（conf 0.26 + shrink@0.45 + greedy_disjoint）で
最終 mask を作り、IoU 0.3〜0.5 のペアを次に分ける。

- overcoverage: pred が GT をほぼ覆いつつ面積が明らかに大きい（削り余地）
- undercoverage: GT が pred をほぼ覆いつつ面積が明らかに大きい（伸ばし／検出器）
- shape_mismatch: 面積は近いが形が違う（ずれ・ねじれ）
- ambiguous: 上記以外

使い方:
    python scratch/run_near_miss_modes.py [--limit N] [--threads 14]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS,
    FOLDS,
    FOLDS_CSV,
    MIN_AREA,
    OOF_CACHE,
    PREPROC,
    TRAIN_JSON,
    build_variant,
    greedy_disjoint,
    imread_unicode,
    iou_matrix,
    load_yolo_cache,
    masks_to_rles,
    refine_batch,
    to_rle,
)

CONF = 0.26
REFINER_METHOD = "shrink"
REFINER_THRESHOLD = 0.45
AREA_RATIO = 1.15
COVER_RATIO = 0.70


def refiner_checkpoint_path(fold: int) -> Path:
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


def classify_pair(pred: np.ndarray, gt: np.ndarray) -> str:
    pred_area = float(pred.sum())
    gt_area = float(gt.sum())
    inter = float(np.logical_and(pred, gt).sum())
    if pred_area <= 0 or gt_area <= 0:
        return "ambiguous"
    cover_gt = inter / gt_area
    cover_pred = inter / pred_area
    if cover_gt >= COVER_RATIO and pred_area >= gt_area * AREA_RATIO:
        return "overcoverage"
    if cover_pred >= COVER_RATIO and gt_area >= pred_area * AREA_RATIO:
        return "undercoverage"
    if abs(pred_area - gt_area) / max(pred_area, gt_area) <= 0.20:
        return "shape_mismatch"
    return "ambiguous"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("output/v35"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

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
        image_id: [
            to_rle(a, int(images[image_id]["height"]), int(images[image_id]["width"]))
            for a in anns
        ]
        for image_id, anns in annotations.items()
    }
    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))

    oof_root = oof_cache_dir()
    print(f"reading OOF cache from {oof_root}", flush=True)
    image_root = image_dir()
    print(f"reading images from {image_root}", flush=True)

    targets = [
        name
        for name in sorted(file_to_ids)
        if (oof_root / f"yolo_fold_{fold_of_file[name]}" / f"{Path(name).stem}.npz").exists()
    ]
    if args.limit:
        targets = targets[: args.limit]

    refiners = {}
    for fold in FOLDS:
        path = refiner_checkpoint_path(fold)
        print(f"loading refiner fold{fold} from {path}", flush=True)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = MediSegUNet(3, 1, False).eval()
        model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)
        refiners[fold] = model

    mode_counts: Counter[str] = Counter()
    pair_rows: list[dict] = []
    hard_fn = hard_fp = 0
    started = time.perf_counter()

    for position, name in enumerate(targets, 1):
        fold = fold_of_file[name]
        masks, confidences = load_yolo_cache(
            oof_root / f"yolo_fold_{fold}" / f"{Path(name).stem}.npz"
        )
        keep = [
            i
            for i, (mask, score) in enumerate(zip(masks, confidences))
            if score >= CONF and mask.sum() >= MIN_AREA
        ]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if keep else np.zeros(0, np.float32)
        order = list(np.argsort(-scores))
        variants = [gt_rles[i] for i in file_to_ids[name]]

        image = imread_unicode(image_root / name)
        if image is None:
            continue

        if proposals:
            boxes, probabilities = refine_batch(refiners[fold], image, proposals)
            refined = [
                build_variant(
                    proposals[i],
                    boxes[i],
                    probabilities[i],
                    REFINER_METHOD,
                    REFINER_THRESHOLD,
                )
                for i in range(len(proposals))
            ]
        else:
            refined = []
        final = greedy_disjoint(refined, order)
        final_rles = masks_to_rles(final)

        for truth in variants:
            matrix = iou_matrix(truth, final_rles)
            if matrix.size == 0:
                hard_fn += len(truth)
                hard_fp += len(final_rles)
                continue
            for gt_index in range(matrix.shape[0]):
                best_pred = int(matrix[gt_index].argmax()) if matrix.shape[1] else -1
                best_iou = float(matrix[gt_index, best_pred]) if best_pred >= 0 else 0.0
                if best_iou <= 0.3:
                    hard_fn += 1
                    continue
                if best_iou > 0.5:
                    continue
                pred_mask = final[best_pred]
                gt_mask = mask_util.decode(truth[gt_index]).astype(bool)
                mode = classify_pair(pred_mask, gt_mask)
                mode_counts[mode] += 1
                pair_rows.append(
                    {
                        "file_name": name,
                        "fold": fold,
                        "n_annotators": len(variants),
                        "iou": best_iou,
                        "mode": mode,
                        "pred_area": int(pred_mask.sum()),
                        "gt_area": int(gt_mask.sum()),
                    }
                )
            for pred_index in range(matrix.shape[1]):
                if float(matrix[:, pred_index].max()) <= 0.3:
                    hard_fp += 1

        if position % 50 == 0 or position == len(targets):
            print(
                f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                f"near-miss={sum(mode_counts.values())}",
                flush=True,
            )

    report = {
        "scope": "全fold OOF、V35 確定後処理、near-miss(IoU 0.3-0.5) の被覆モード",
        "observations": len(targets),
        "near_miss_pairs": int(sum(mode_counts.values())),
        "mode_counts": dict(mode_counts),
        "hard_fn_iou_le_0_3": hard_fn,
        "hard_fp_iou_le_0_3": hard_fp,
        "decision_hint": {
            "overcoverage_dominant": "削る系 refiner / 形態学を優先",
            "undercoverage_dominant": "検出器または伸ばし系（ただし guided は既に劣位）",
            "shape_mismatch_dominant": "検出器の原型形状・位置ずれ",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "near_miss_modes.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(pair_rows).to_csv(args.out / "near_miss_modes.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"所要 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
