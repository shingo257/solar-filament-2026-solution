"""P2-3 をローカル CPU で実行する: V30 fold0 重みを学習せず、推論 imgsz 1536。

Google Drive の日本語パスを YOLO に渡さない。重みと fold0 画像は ASCII の
ローカルキャッシュへコピーしてから推論する。
"""

from __future__ import annotations

import os
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402
from scratch.run_pipeline_stack import apply_ownership  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS,
    FOLDS_CSV,
    MIN_AREA,
    PREPROC,
    TRAIN_JSON,
    build_variant,
    imread_unicode,
    iou_matrix,
    load_yolo_cache,
    masks_to_rles,
    refine_batch,
    to_rle,
)

DRIVE = Path("G:/マイドライブ")
WEIGHTS_SRC = DRIVE / "solar_filament_v30/runs/yolo_fold_0/weights/best.pt"
LOCAL = Path(os.environ.get("TEMP", "C:/Temp")) / "solar_p2_3"
WEIGHTS_LOCAL = LOCAL / "v30_fold0_best.pt"
IMAGES_LOCAL = LOCAL / "images"
CACHE = LOCAL / "oof_npz"
DRIVE_CACHE = DRIVE / "solar_filament_v35/prediction_cache/oof/yolo_v30w_imgsz1536_fold_0"
OUT = Path(__file__).resolve().parents[1] / "output" / "v36"
IMGSZ = 1536
CONF_KEEP = 0.26
PRED_CONF = 0.08
FOLD = 0
BASELINE = 0.4393


def result_masks(result: Any) -> tuple[list[np.ndarray], np.ndarray]:
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output: list[np.ndarray] = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = (
        result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        if result.boxes is not None else np.zeros(0, np.float32)
    )
    return output, confidences


def save_yolo_cache(path: Path, masks: list[np.ndarray], confidences: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if masks:
        packed = np.packbits(np.stack(masks).astype(np.uint8), axis=-1)
        shape = np.asarray(masks[0].shape, np.int32)
    else:
        packed = np.empty((0, 0, 0), np.uint8)
        shape = np.asarray([2048, 2048], np.int32)
    np.savez_compressed(path, packed=packed, shape=shape, confidences=confidences)


def load_fold0_names() -> list[str]:
    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    file_to_ids: dict[str, list[int]] = defaultdict(list)
    for item in coco["images"]:
        file_to_ids[item["file_name"]].append(item["id"])
    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    return [name for name in sorted(file_to_ids) if fold_of_file[name] == FOLD]


def infer(limit: int) -> list[str]:
    from ultralytics import YOLO

    LOCAL.mkdir(parents=True, exist_ok=True)
    IMAGES_LOCAL.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS_LOCAL.exists() or WEIGHTS_LOCAL.stat().st_size != WEIGHTS_SRC.stat().st_size:
        print(f"copy weights -> {WEIGHTS_LOCAL}", flush=True)
        shutil.copy2(WEIGHTS_SRC, WEIGHTS_LOCAL)

    names = load_fold0_names()
    if limit:
        names = names[:limit]
    for name in names:
        dest = IMAGES_LOCAL / name
        if not dest.exists():
            shutil.copy2(PREPROC / name, dest)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS_LOCAL))
    started = time.time()
    for index, name in enumerate(names, start=1):
        path = CACHE / f"{Path(name).stem}.npz"
        if path.exists():
            print(f"[{index}/{len(names)}] skip {name}", flush=True)
            continue
        image_path = IMAGES_LOCAL / name
        predict_started = time.time()
        masks, confidences = result_masks(model.predict(
            str(image_path), imgsz=IMGSZ, conf=PRED_CONF, iou=0.55,
            agnostic_nms=True, retina_masks=True, verbose=False, device=device,
        )[0])
        save_yolo_cache(path, masks, confidences)
        elapsed = time.time() - predict_started
        print(
            f"[{index}/{len(names)}] {name} inst={len(masks)} {elapsed:.1f}s",
            flush=True,
        )
    print(f"infer done in {time.time() - started:.1f}s", flush=True)
    return names


def score(names: list[str]) -> dict[str, Any]:
    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
    annotations: dict[int, list[Any]] = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    file_to_ids: dict[str, list[int]] = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)
    gt_rles = {
        i: [to_rle(a, int(images[i]["height"]), int(images[i]["width"]))
            for a in anns if not a.get("iscrowd", 0)]
        for i, anns in annotations.items()
    }

    checkpoint = torch.load(ARTIFACTS / "mediseg_refiner_fold0_resume.pt", map_location="cpu")
    model = MediSegUNet(3, 1, False).eval()
    model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    meter = PanopticQuality()
    near_miss = 0
    counts: list[int] = []
    for index, name in enumerate(names, start=1):
        masks, confidences = load_yolo_cache(CACHE / f"{Path(name).stem}.npz")
        keep = [
            i for i, (mask, score) in enumerate(zip(masks, confidences))
            if score >= CONF_KEEP and mask.sum() >= MIN_AREA
        ]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if keep else np.zeros(0, np.float32)
        image = imread_unicode(PREPROC / name)
        if image is None:
            raise FileNotFoundError(PREPROC / name)
        if proposals:
            boxes, probabilities = refine_batch(model, image, proposals)
            refined = [
                build_variant(proposals[i], boxes[i], probabilities[i], "shrink", 0.45)
                for i in range(len(proposals))
            ]
        else:
            refined = []
        predictions = apply_ownership(refined, scores, "drop_contained_then_area_asc")
        counts.append(len(predictions))
        rles = masks_to_rles(predictions)
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, rles)
            meter.add_matrix(matrix, len(truth), len(rles))
            if matrix.size:
                near_miss += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())
        if index % 20 == 0 or index == len(names):
            print(f"score {index}/{len(names)} pq={meter.summary()['pq']:.4f}", flush=True)

    summary = meter.summary()
    report = {
        "probe": "P2-3 infer-only imgsz1536 on V30 fold0 weights",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_imgsz": 1280,
        "infer_imgsz": IMGSZ,
        "fold": FOLD,
        "observations": len(names),
        "pq": round(float(summary["pq"]), 6),
        "sq": round(float(summary["sq"]), 6),
        "rq": round(float(summary["rq"]), 6),
        "tp": int(summary["tp"]),
        "fp": int(summary["fp"]),
        "fn": int(summary["fn"]),
        "near_miss": int(near_miss),
        "inst/img": float(np.mean(counts)) if counts else 0.0,
        "baseline_fold0_pq": BASELINE,
        "delta": round(float(summary["pq"] - BASELINE), 6),
        "continue_fold12": bool(summary["pq"] > BASELINE),
        "decision": (
            "fold1/2 を同じ推論 imgsz で回してよい"
            if summary["pq"] > BASELINE
            else "推論 1536 もベースラインを超えていない。解像度レバーはここで止める"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p2_3_infer_imgsz1536_fold0.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    drive_art = DRIVE / "solar_filament_v35/artifacts"
    drive_art.mkdir(parents=True, exist_ok=True)
    (drive_art / "p2_3_infer_imgsz1536_fold0.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
    for npz in CACHE.glob("*.npz"):
        dest = DRIVE_CACHE / npz.name
        if not dest.exists():
            shutil.copy2(npz, dest)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if args.score_only:
        names = [f"{path.stem}.jpeg" for path in sorted(CACHE.glob("*.npz"))]
        if args.limit:
            names = names[:args.limit]
        score(names)
        return 0
    names = infer(args.limit)
    if not args.infer_only:
        score(names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
