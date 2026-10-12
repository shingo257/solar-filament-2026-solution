"""v80 E1 — YOLO ∪ component-tree 候補を生成し、fold 別 refiner を適用してキャッシュする。

PLAN_2026-10-09_FINAL_SPRINT.md §7.1。候補源は2系統:

  yolo : 既存 OOF cache (`solar_filament_v34/.../yolo_fold_f/<stem>.npz`) の
         conf>=--yolo-conf 提案 (既定 0.08)。
  tree : V34 semantic OOF 確率マップ (`semantic_fold_f/<stem>.png`) を
         15 段階の閾値で連結成分化した component tree の全ノード。

両系統に「その観測の fold を学習から除いた」refiner
(`mediseg_refiner_fold{f}_resume.pt`) を適用し、shrink@--shrink の最終候補
マスクを RLE で `solar_filament_v80/candidates/fold_{f}/<stem>.npz` に保存する。
教師・採点は行わない (run_v80_scorer_eval.py の仕事)。CPU のみで動くが遅い;
Colab T4 なら fold あたり約 1.5〜2h の見込み。

使い方:
    python scratch/run_v80_hybrid_candidates.py --fold 0 [--limit N] [--threads 14]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))

import torch  # noqa: E402

from solar_filament.pipeline.component_tree import (  # noqa: E402
    DEFAULT_LEVELS,
    build_component_tree,
)
from solar_filament.pipeline.paths import (  # noqa: E402
    artifacts_dir,
    grouped_folds_csv,
    oof_cache_root,
    preprocessed_dir,
    resolve_drive,
    train_annotations_json,
)
from solar_filament.pipeline.rle import encode_mask  # noqa: E402
from run_refiner_pq_allfold import (  # noqa: E402
    MIN_AREA,
    refine_crops,
    square_box,
    square_box_from_extent,
)
from solar_filament.pipeline.io_images import (  # noqa: E402
    imread_unicode_with_retry,
)
from solar_filament.pipeline.oof_cache import (  # noqa: E402
    load_yolo_cache_with_retry,
)
from src.mediseg_models import MediSegUNet  # noqa: E402

DISK_CENTER = (1024.0, 1024.0)
DISK_RADIUS = 899.7

FEATURE_NAMES = (
    # geometry (final mask / proposal / bbox)
    "log_area", "log_area_prop", "shrink_ratio", "aspect", "elongation",
    "solidity", "bbox_w_norm", "bbox_h_norm", "rho_centroid", "rho_edge",
    # intensity of the preprocessed image
    "img_mean_in", "img_p05_in", "img_mean_ring",
    # dense probability map statistics
    "prob_mean_in", "prob_max_in", "prob_frac50_in", "prob_mean_ring",
    # refiner probability statistics
    "ref_mean_in", "ref_max_in", "ref_frac45_in", "ref_ncomp",
    # candidate bookkeeping
    "yolo_conf", "level_index", "n_children", "n_siblings",
    "parent_area_ratio", "source_tree",
)


def _read_bytes_retry(path: Path, attempts: int = 4, backoff: float = 2.0) -> bytes:
    """DriveFS の一時的な OSError(errno 22) を backoff 付きでやり直す。"""
    for attempt in range(attempts):
        try:
            return path.read_bytes()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff * (attempt + 1))
    raise AssertionError("unreachable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for attempt in range(4):
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            check=True, text=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - manifest must not crash the pipeline
        return "unknown"


def _ring(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask


def _geometry_features(mask: np.ndarray) -> tuple[float, ...]:
    ys, xs = np.nonzero(mask)
    area = float(len(ys))
    if area == 0:
        # refiner が全消しした候補 (usable=False で行は残す) の退化特徴
        return (0.0,) * 8
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    bbox_w = float(x1 - x0)
    bbox_h = float(y1 - y0)
    aspect = max(bbox_w, bbox_h) / max(1.0, min(bbox_w, bbox_h))
    coords = np.stack([ys - ys.mean(), xs - xs.mean()])
    eig = np.linalg.eigvalsh(coords @ coords.T / max(area, 1.0))
    elongation = float(np.sqrt(max(eig[-1], 0.0) / max(eig[0], 1e-6)))
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull_area = (
        sum(float(cv2.contourArea(cv2.convexHull(c))) for c in contours) or 1.0
    )
    solidity = float(area / hull_area)
    rho_c = float(
        np.hypot(ys.mean() - DISK_CENTER[0], xs.mean() - DISK_CENTER[1]) / DISK_RADIUS
    )
    rho_edge = float(
        np.sqrt((ys - DISK_CENTER[0]) ** 2 + (xs - DISK_CENTER[1]) ** 2).max()
        / DISK_RADIUS
    )
    return (
        float(np.log1p(area)), aspect, elongation, solidity,
        bbox_w / mask.shape[1], bbox_h / mask.shape[0], rho_c, rho_edge,
    )


def _region_stats(values: np.ndarray, mask: np.ndarray, ring: np.ndarray):
    inside = values[mask]
    band = values[ring] if ring.any() else np.zeros(1, values.dtype)
    return (
        float(inside.mean()) if inside.size else 0.0,
        float(np.percentile(inside, 5)) if inside.size else 0.0,
        float(inside.max()) if inside.size else 0.0,
        float((inside >= 0.5).mean()) if inside.size else 0.0,
        float(band.mean()),
    )


def _refiner_stats(
    proposal: np.ndarray, probability: np.ndarray, threshold: float
) -> tuple[float, ...]:
    inside = probability[proposal]
    if inside.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(inside.mean()),
        float(inside.max()),
        float((inside >= threshold).mean()),
    )


def _n_components(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return count - 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--yolo-conf", type=float, default=0.08)
    parser.add_argument("--shrink", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--min-area", type=int, default=MIN_AREA)
    parser.add_argument("--drive", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="既定: <drive>/solar_filament_v80/candidates")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    drive = resolve_drive(args.drive)
    out_root = (args.out or drive / "solar_filament_v80" / "candidates")
    out_dir = out_root / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(train_annotations_json(drive), encoding="utf-8") as handle:
        coco = json.load(handle)
    files = {info["file_name"] for info in coco["images"]}
    fold_frame = pd.read_csv(grouped_folds_csv(drive))
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    oof = oof_cache_root(drive)
    preproc = preprocessed_dir(drive)

    targets = sorted(
        n for n in files
        if fold_of_file.get(n) == args.fold
        and (oof / f"yolo_fold_{args.fold}" / f"{Path(n).stem}.npz").exists()
        and (oof / f"semantic_fold_{args.fold}" / f"{Path(n).stem}.png").exists()
    )
    if args.limit:
        targets = targets[: args.limit]

    checkpoint_path = artifacts_dir(drive) / f"mediseg_refiner_fold{args.fold}_resume.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"fold{args.fold} の refiner がありません: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MediSegUNet(3, 1, False).to(device).eval()
    model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    print({
        "fold": args.fold, "観測": len(targets), "yolo_conf": args.yolo_conf,
        "shrink": args.shrink, "out": str(out_dir),
        "リーク": "なし (fold 専用 refiner・OOF マップのみ使用)",
    }, flush=True)

    started = time.perf_counter()
    counts, fingerprints = [], {}
    skipped = []
    for position, name in enumerate(targets, 1):
        stem = Path(name).stem
        out_path = out_dir / f"{stem}.npz"
        if args.skip_existing and out_path.exists():
            counts.append(-1)
            continue
        yolo_path = oof / f"yolo_fold_{args.fold}" / f"{stem}.npz"
        prob_path = oof / f"semantic_fold_{args.fold}" / f"{stem}.png"
        try:
            image = imread_unicode_with_retry(preproc / name)
        except RuntimeError:
            image = None  # DriveFS の一時的な読み込み失敗を skip に丸める
        if image is None:
            skipped.append(name)
            if len(skipped) > max(3, len(targets) // 20):
                raise RuntimeError(f"前処理画像を読めない観測が{len(skipped)}件: {skipped[:3]}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        raw = np.frombuffer(_read_bytes_retry(prob_path), np.uint8)
        prob = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        fingerprints[name] = {
            "yolo_npz": _sha256(yolo_path),
            "semantic_png": _sha256(prob_path),
        }

        yolo_masks, yolo_conf = load_yolo_cache_with_retry(yolo_path)
        yolo_keep = [
            i for i, (m, s) in enumerate(zip(yolo_masks, yolo_conf))
            if s >= args.yolo_conf and m.sum() >= args.min_area
        ]
        tree = build_component_tree(prob, min_area=args.min_area)

        # meta tuple: (kind, conf, level, n_children, n_siblings, parent_area_ratio, parent)
        # proposals は全面マスクを保持しない — 全候補の 2048² bool は ~450MB/画像
        # で OOM した (v64 教訓)。box は extent から直接計算し、proposal 本体は
        # ループ内で必要な1枚だけ materialize する。
        sources = [("yolo", i) for i in yolo_keep]
        meta = [
            ("yolo", float(yolo_conf[i]), -1, -1, -1, -1.0, -1) for i in yolo_keep
        ]
        for index in range(len(tree)):
            sources.append(("tree", index))
            node = tree.nodes[index]
            n_siblings = (
                len(tree.nodes[node.parent].children) - 1 if node.parent >= 0 else 0
            )
            parent_ratio = (
                node.area / tree.nodes[node.parent].area if node.parent >= 0 else -1.0
            )
            meta.append((
                "tree", -1.0, node.level_index, len(node.children),
                n_siblings, parent_ratio, node.parent,
            ))

        def proposal_of(i):
            kind, ref = sources[i]
            return yolo_masks[ref] if kind == "yolo" else tree.mask_of(ref)

        height, width = image.shape[:2]
        boxes = []
        for i, (kind, ref) in enumerate(sources):
            if kind == "yolo":
                boxes.append(square_box(yolo_masks[ref]))
            else:
                x0, y0, x1, y1 = tree.nodes[ref].bbox
                boxes.append(square_box_from_extent(
                    x0, y0, x1, y1, height, width))
        probabilities = refine_crops(model, image, boxes) if boxes else []

        rle_final, rle_prop, feats = [], [], []
        source, conf_arr, level_arr, parent_arr = [], [], [], []
        children_arr, siblings_arr, usable_arr = [], [], []
        for i in range(len(sources)):
            proposal = proposal_of(i)
            refined = np.zeros_like(proposal, dtype=bool)
            x0, y0, x1, y1 = boxes[i]
            refined[y0:y1, x0:x1] = probabilities[i] >= args.shrink
            final = refined & proposal
            # Keep the row even when the refined mask dies: tree node indices
            # must stay aligned so eval-time antichain DP can block it.
            usable = bool(final.sum() >= args.min_area)
            usable_arr.append(usable)
            ring = _ring(proposal)
            geo = _geometry_features(final)
            img_in, img_p05, _, _, img_ring = _region_stats(gray, final, _ring(final))
            prob_in = _region_stats(prob, proposal, ring)
            # probabilities[i] はクロップ座標 (boxes[i] サイズ) — proposal を
            # 同じ窓に切ってから統計を取る
            prop_crop = proposal[y0:y1, x0:x1]
            ref_in = _refiner_stats(prop_crop, probabilities[i], args.shrink)
            kind, conf, level, n_children, n_siblings, parent_ratio, parent = meta[i]
            feats.append((
                geo[0], float(np.log1p(proposal.sum())),
                final.sum() / max(float(proposal.sum()), 1.0),
                *geo[1:], img_in, img_p05, img_ring,
                prob_in[0], prob_in[2], prob_in[3], prob_in[4],
                *ref_in, float(_n_components(final)),
                conf, float(level), float(n_children), float(n_siblings),
                float(parent_ratio), 1.0 if kind == "tree" else 0.0,
            ))
            rle_final.append(encode_mask(final))
            rle_prop.append(encode_mask(proposal))
            source.append(1 if kind == "tree" else 0)
            conf_arr.append(conf)
            level_arr.append(level)
            children_arr.append(n_children)
            siblings_arr.append(n_siblings)
            parent_arr.append(parent)

        # tmp に書いて rename — --skip-existing 併用の並走実行で
        # 半書き込み npz を拾わないための公開用アトミック化。
        # (file object を渡す: Path だと np.savez が .npz を補完する)
        tmp_path = out_path.with_suffix(".npz.tmp")
        with open(tmp_path, "wb") as handle:
            np.savez_compressed(
                handle,
                rle_final=np.asarray(rle_final, dtype=object),
                rle_prop=np.asarray(rle_prop, dtype=object),
                feats=np.asarray(feats, dtype=np.float32).reshape(len(rle_final), -1),
                source=np.asarray(source, dtype=np.int8),
                conf=np.asarray(conf_arr, dtype=np.float32),
                level=np.asarray(level_arr, dtype=np.int16),
                parent=np.asarray(parent_arr, dtype=np.int32),
                n_children=np.asarray(children_arr, dtype=np.int16),
                n_siblings=np.asarray(siblings_arr, dtype=np.int16),
                usable=np.asarray(usable_arr, dtype=bool),
                tree_offset=np.int64(len(yolo_keep)),
                n_tree_nodes=np.int64(len(tree)),
            )
        os.replace(tmp_path, out_path)
        counts.append((len(yolo_keep), len(tree), int(sum(usable_arr))))
        if position % 10 == 0 or position == len(targets):
            done = [c for c in counts if c != -1]
            print(
                f"  {position}/{len(targets)}  {time.perf_counter() - started:.0f}s  "
                f"yolo={np.mean([c[0] for c in done]):.1f} "
                f"tree={np.mean([c[1] for c in done]):.1f} "
                f"kept={np.mean([c[2] for c in done]):.1f}",
                flush=True,
            )

    kept = [c for c in counts if c != -1]
    manifest = {
        "experiment": "v80 candidates",
        "fold": args.fold,
        "code_sha": _git_sha(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "threads": args.threads,
        "yolo_conf": args.yolo_conf,
        "shrink_threshold": args.shrink,
        "tree_levels": list(DEFAULT_LEVELS),
        "min_area": args.min_area,
        "feature_names": list(FEATURE_NAMES),
        "images": len(kept),
        "skipped_missing_image": skipped,
        "candidates_per_image": {
            "yolo": float(np.mean([c[0] for c in kept])) if kept else 0.0,
            "tree": float(np.mean([c[1] for c in kept])) if kept else 0.0,
            "kept_after_refiner": float(np.mean([c[2] for c in kept])) if kept else 0.0,
        },
        "input_fingerprints": fingerprints,
        "runtime_sec": round(time.perf_counter() - started, 1),
        "utc_completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_kind": "fresh candidate generation (no resume of training)",
    }
    manifest_path = out_root / f"manifest_fold{args.fold}.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print("\n" + json.dumps({k: v for k, v in manifest.items() if k != "input_fingerprints"},
                            ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
