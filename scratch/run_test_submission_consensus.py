"""test 180画像の推論 → consensus (cons2of3_c26) submission.csv (contingent v122)。

**実行条件**: v121c fold-0 eval が `cross_fold_verified`
(cons2of3_c26 Δ@0.45 >= +0.005, §2.99) を出した場合のみ実行する。
unverified ならこのスクリプトは提出物を作らない (検証前の提出は
事前登録プロトコルに反する)。

`run_test_submission.py --mode prod` からの派生。差分は提案生成のみ:
prod の単一 YOLO を、§2.98 で検証した 3 detector コンセンサスに
置き換える。後続 (refine_crops → threshold → ownership → RLE → CSV)
は一切変更しない。

提案生成 (§2.98 で計測したものと同一 semantics):
  - detector: prod yolo11m fold_2 / v116 croptrain fold_2 /
    v117 yolo26m fold_2 (全て folds(0,1) 学習 — test は完全 disjoint
    なので fold-2 学習済みの使用は正当。all-data 版は未計測構成の
    ため、計測済み構成のまま提出する)
  - 各 det を conf floor 0.08・imgsz=1280・iou=0.55・agnostic_nms・
    retina_masks で推論 (prod 推論パラメータと同一)
  - mask-IoU>=0.5 の推移クラスタ (union-find、cross-detector edge のみ)
  - 放出 = 支持 member のうち conf>=0.26 が >=2 detector のクラスタ、
    max-conf member の mask + conf

使い方:
    python scratch/run_test_submission_consensus.py --device cuda \
        --out output/test_submission_consensus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
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
from solar_filament.pipeline.io_images import (  # noqa: E402
    imread_unicode_with_retry,
)
from solar_filament.pipeline.ownership import apply_ownership  # noqa: E402
from solar_filament.pipeline.paths import (  # noqa: E402
    artifacts_dir,
    resolve_drive,
)
from src.mediseg_models import MediSegEdgeAttUNet  # noqa: E402

# --- 本番確定値 (run_test_submission.py prod と同一) -------------------------
YOLO_IMGSZ = 1280
PREDICT_IOU = 0.55
REFINER_THRESHOLD = 0.45
DROP_THRESHOLD = 0.90
MIN_AREA = 20

# --- §2.98 固定パラメータ -----------------------------------------------------
DET_FLOOR = 0.08          # 全 detector の提案保持 floor
CONSENSUS_IOU = 0.5       # cross-detector cluster の mask-IoU 閾値
SUPPORT_CONF = 0.26       # cons2of3_c26 の「支持」とみなす member conf

REFINER_FILENAME = "mediseg_edgeattnet_refiner_all_data_resume.pt"  # prod 同一

# 計測済み fold-2 detector セット + sha256 pin (§2.98 記録値)
CONSENSUS_DETECTORS = {
    "yolo": ("runs/yolo_fold_2/weights/best.pt",
             "40ec31fd2c6aa8e32c952e64be453f4351ace339d31ef747b4f28624eafe7e8c"),
    "crop": ("runs/v116_croptrain/yolo_croptrain_fold2/weights/best.pt",
             "c23b1bb79620c62fd85984d032310bae26c240fd1be0aaf7df364fbc560fbcda"),
    "yolo26": ("runs/v117_yolo26/yolo_yolo26m_fold2/weights/best.pt",
               "0c8bb71f6ab4398aaea2b729992a3b88f3f4d707e0cbcbe35f1b82a2533db7a5"),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_masks(result) -> tuple[list[np.ndarray], np.ndarray]:
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    output = [m >= 0.5 for m in result.masks.data.detach().cpu().numpy()]
    confidences = (
        result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        if result.boxes is not None else np.zeros(0, np.float32))
    return output, confidences


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    return inter / int(np.logical_or(a, b).sum()) if inter else 0.0


def consensus_clusters(props, iou_thr):
    """run_fold_oof_eval_consensus.py と同一: cross-detector pair のみ
    mask-IoU>=iou_thr で結合する union-find (同 detector 内 edge なし)。"""
    n = len(props)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    boxes = []
    for _, m, _ in props:
        ys, xs = np.nonzero(m)
        boxes.append((int(xs.min()), int(ys.min()),
                      int(xs.max()) + 1, int(ys.max()) + 1) if xs.size
                     else (0, 0, 0, 0))
    areas = [int(m.sum()) for _, m, _ in props]
    for i in range(n):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, n):
            if props[i][0] == props[j][0]:
                continue
            bx0, by0, bx1, by1 = boxes[j]
            inter = (max(0, min(ax1, bx1) - max(ax0, bx0))
                     * max(0, min(ay1, by1) - max(ay0, by0)))
            if inter < iou_thr * max(areas[i], areas[j]):
                continue
            if _mask_iou(props[i][1], props[j][1]) >= iou_thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def emit_cons2of3_c26(props):
    """§2.98 arm と同一: conf>=SUPPORT_CONF の支持が >=2 detector の
    クラスタから max-conf member を放出 → (mask, conf) リスト。"""
    emitted = []
    for mem in consensus_clusters(props, CONSENSUS_IOU):
        strong = [i for i in mem if props[i][2] >= SUPPORT_CONF]
        if len({props[i][0] for i in strong}) >= 2:
            best = max(mem, key=lambda i: props[i][2])
            emitted.append((props[best][1], props[best][2]))
    return emitted


def _run_submission(local: Path) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path,
                        default=Path("output/test_submission_consensus"))
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

    local_img = local / "images"
    local_w = local / "weights"
    for d in (local_img, local_w):
        d.mkdir(parents=True, exist_ok=True)

    # --- detector staging + sha256 pin 検証 (§2.98 記録値) --------------------
    det_paths = {}
    for det_name, (rel, expected_sha) in CONSENSUS_DETECTORS.items():
        src = drive / "solar_filament_v30" / rel
        if not src.exists():
            raise FileNotFoundError(f"detector weights missing: {src}")
        actual = sha256_of(src)
        if actual != expected_sha:
            raise RuntimeError(
                f"detector {det_name} sha256 mismatch: {actual[:16]} != "
                f"{expected_sha[:16]} (§2.98 pin)")
        dst = local_w / f"{det_name}.pt"
        print(f"copy {src.name} ({det_name}) -> {dst}  sha={actual[:12]}",
              flush=True)
        shutil.copy2(src, dst)
        det_paths[det_name] = dst
    refiner_src = artifacts / REFINER_FILENAME
    refiner_path = local_w / "refiner_edgeatt_all.pt"
    print(f"copy {refiner_src.name} -> {refiner_path}", flush=True)
    shutil.copy2(refiner_src, refiner_path)

    # --- 前処理 (solar_three_channel、prod と同一) ----------------------------
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
    det_models = {n: YOLO(str(p)) for n, p in det_paths.items()}
    refiner = MediSegEdgeAttUNet(3, 1, False).eval().to(args.device)
    checkpoint = torch.load(refiner_path, map_location="cpu")
    refiner.load_state_dict(
        checkpoint.get("best_model") or checkpoint["model"], strict=True)

    rows, audit = [], []
    started = time.perf_counter()
    for index, name in enumerate(test_files, 1):
        image_path = local_img / name
        image = imread_unicode_with_retry(image_path)
        if image is None:
            raise FileNotFoundError(image_path)

        # 3 detector 提案 (floor 0.08) → cons2of3_c26 放出
        props = []
        for det_name, model in det_models.items():
            result = model.predict(
                str(image_path), imgsz=YOLO_IMGSZ, conf=DET_FLOOR,
                iou=PREDICT_IOU, agnostic_nms=True, retina_masks=True,
                device=args.device, verbose=False)[0]
            masks, confidences = result_masks(result)
            for m, s in zip(masks, confidences):
                if s >= DET_FLOOR and m.sum() >= MIN_AREA:
                    props.append((det_name, m, float(s)))
        emitted = emit_cons2of3_c26(props)
        proposals = [m for m, _ in emitted]
        emitted_conf = np.asarray([s for _, s in emitted], dtype=np.float32)

        # 以降 prod 経路 verbatim: refine → threshold → ownership → RLE
        boxes = [square_box(p) for p in proposals]
        probabilities = refine_crops(refiner, image, boxes) if boxes else []
        refined = []
        for i, (prop, (x0, y0, x1, y1)) in enumerate(zip(proposals, boxes)):
            mask = np.zeros_like(prop, dtype=bool)
            mask[y0:y1, x0:x1] = probabilities[i] >= REFINER_THRESHOLD
            refined.append(mask & prop)
        final_masks = apply_ownership(
            refined, emitted_conf,
            rule="drop_contained_then_area_asc",
            containment_threshold=DROP_THRESHOLD, min_area=MIN_AREA)
        audit.append({"file_name": name, "proposals_all_dets": len(props),
                      "consensus_emitted": len(emitted),
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
    tag = "consensus"
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_csv = args.out / f"submission_{tag}.csv"
    submission.to_csv(out_csv, index=False, lineterminator="\n")
    efficiency = {"mode": "consensus_cons2of3_c26", "detectors": list(CONSENSUS_DETECTORS),
                  "refiner": "edgeatt_all", "test_images": len(test_files),
                  "total_wall_s": round(total_s, 1),
                  "s_per_image": round(total_s / len(test_files), 3),
                  "device": args.device, "threads": args.threads,
                  "consensus": {"det_floor": DET_FLOOR, "iou": CONSENSUS_IOU,
                                "support_conf": SUPPORT_CONF}}
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
    with tempfile.TemporaryDirectory(prefix="solar_consensus_") as staging:
        return _run_submission(Path(staging))


if __name__ == "__main__":
    raise SystemExit(main())
