"""v82 refiner arms の paired OOF 評価 — per-image counts を保存し bootstrap CI を取る。

manifest (`v82_*_manifest.json`) は fold 集計 PQ しか持たないため、採用ゲートの
pooled CI を厳密に取るには checkpoint をこのスクリプトで再評価する。
**v82 ノートブックの OOF プロトコルと完全に揃える**:

  YOLO OOF cache (conf>=0.08) → keep conf>=0.26 & area>=20
  → refiner forward → shrink@τ (mask[y0:y1,x0:x1]=prob>=τ & proposal)
  → drop_contained@0.90 → greedy_disjoint (面積昇順, min_area 20)
  → record 単位の公式 PQ 寄与を per-image で保存

評価対象 (fold ごとに5モデル、held-out fold にのみ適用):
  hard_record = mediseg_refiner_fold{f}_resume.pt        (現行対照)
  soft        = mediseg_soft_refiner_fold{f}_s{seed}_resume.pt
  hard_union  = mediseg_hard_union_refiner_fold{f}_s{seed}_resume.pt

使い方:
    python scratch/run_v82_refiner_eval.py [--drive "G:/マイドライブ"]
        [--folds 0,1,2] [--seeds 3407,8407] [--limit N] [--threads 14]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))

from solar_filament.pipeline.ownership import apply_ownership  # noqa: E402
from solar_filament.pipeline.paths import (  # noqa: E402
    artifacts_dir,
    grouped_folds_csv,
    oof_cache_root,
    preprocessed_dir,
    resolve_drive,
    train_annotations_json,
)
from run_refiner_pq_allfold import (  # noqa: E402
    MIN_AREA,
    imread_unicode,
    iou_matrix,
    masks_to_rles,
    refine_batch,
    to_rle,
)
from src.mediseg_models import MediSegUNet  # noqa: E402
from src.metrics_official import PQ_IOU_THRESHOLD  # noqa: E402

CONF_KEEP = 0.26          # v82 notebook の CFG.CONF
CONTAINMENT = 0.90
THRESHOLD_SWEEP = (0.35, 0.40, 0.45, 0.50)  # notebook の THRESHOLD_SWEEP と同一
FOLDS = (0, 1, 2)
SEEDS = (3407, 8407)
# arm -> checkpoint ファイル名テンプレート ({f}=fold, {s}=seed)
CKPT_PATTERNS = {
    "hard_record": "mediseg_refiner_fold{f}_resume.pt",
    "soft_s{s}": "mediseg_soft_refiner_fold{f}_s{s}_resume.pt",
    "hard_union_s{s}": "mediseg_hard_union_refiner_fold{f}_s{s}_resume.pt",
}


def arm_keys(seeds: tuple[int, ...]) -> list[str]:
    keys = ["hard_record"]
    for arm in ("soft", "hard_union"):
        keys += [f"{arm}_s{s}" for s in seeds]
    return keys


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


def load_model(path: Path) -> MediSegUNet:
    # torch.load(path) は内部の zipfile read が DriveFS の一時 errno 22 で
    # 落ちる。retry 付き bytes 読み込み → BytesIO 経由にする。
    checkpoint = torch.load(io.BytesIO(_read_bytes_retry(path)),
                            map_location="cpu")
    model = MediSegUNet(3, 1, False).eval()
    model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"],
                          strict=True)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=Path, default=None)
    parser.add_argument("--folds", type=str, default="0,1,2")
    parser.add_argument("--seeds", type=str, default="3407,8407")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("output/v82"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    drive = resolve_drive(args.drive)
    artifacts = artifacts_dir(drive)
    oof_cache = oof_cache_root(drive)
    preproc = preprocessed_dir(drive)
    folds = tuple(int(x) for x in args.folds.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))

    with open(train_annotations_json(drive), encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {i["id"]: i for i in coco["images"]}
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    file_to_ids = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)
    gt_rles = {
        image_id: [to_rle(a, int(images[image_id]["height"]),
                          int(images[image_id]["width"]))
                   for a in anns if not a.get("iscrowd", 0)]
        for image_id, anns in annotations.items()
    }

    fold_frame = pd.read_csv(grouped_folds_csv(drive))
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))

    keys = arm_keys(seeds)
    # (arm_key, tau, image) -> (tp_iou_sum, tp, fp, fn)
    counts = {key: {t: {} for t in THRESHOLD_SWEEP} for key in keys}
    missing_ckpt = []
    started = time.perf_counter()

    for fold in folds:
        names = [n for n in sorted(file_to_ids) if fold_of_file[n] == fold]
        cache_dir = oof_cache / f"yolo_fold_{fold}"
        names = [n for n in names if (cache_dir / f"{Path(n).stem}.npz").exists()]
        if args.limit:
            names = names[:args.limit]

        models = {}
        for key in keys:
            if key == "hard_record":
                path = artifacts / CKPT_PATTERNS["hard_record"].format(f=fold)
            else:
                arm, seed = key.rsplit("_s", 1)
                path = artifacts / f"mediseg_{arm}_refiner_fold{fold}_s{seed}_resume.pt"
            if not path.exists():
                missing_ckpt.append(str(path))
                continue
            models[key] = load_model(path)
        print(f"fold{fold}: {len(names)}観測, モデル={list(models)}", flush=True)
        if not models:
            continue

        for position, name in enumerate(names, 1):
            cache = np.load(cache_dir / f"{Path(name).stem}.npz")
            packed, confs = cache["packed"], cache["confidences"].astype(np.float32)
            _, width = map(int, cache["shape"])
            proposals, keeps = [], []
            for i in range(len(confs)):
                mask = np.unpackbits(packed[i], axis=-1)[:, :width].astype(bool)
                if confs[i] >= CONF_KEEP and mask.sum() >= MIN_AREA:
                    proposals.append(mask)
                    keeps.append(i)

            image = imread_unicode(preproc / name)
            if image is None:
                continue
            records = [gt_rles[i] for i in file_to_ids[name]]

            for key, model in models.items():
                if not proposals:
                    built_per_tau = {t: [] for t in THRESHOLD_SWEEP}
                else:
                    boxes, probabilities = refine_batch(model, image, proposals)
                    built_per_tau = {}
                    for t in THRESHOLD_SWEEP:
                        built = []
                        for proposal, (x0, y0, x1, y1), prob in zip(
                                proposals, boxes, probabilities):
                            mask = np.zeros_like(proposal)
                            mask[y0:y1, x0:x1] = prob >= t
                            built.append(mask & proposal)  # shrink
                        built_per_tau[t] = built

                for t in THRESHOLD_SWEEP:
                    built = built_per_tau[t]
                    preds = apply_ownership(
                        built, np.ones(len(built)),
                        rule="drop_contained_then_area_asc",
                        containment_threshold=CONTAINMENT,
                        min_area=MIN_AREA)
                    rles = masks_to_rles(preds)
                    tp_sum = tp = fp = fn = 0
                    for truth in records:
                        matrix = iou_matrix(truth, rles)
                        hits = matrix > PQ_IOU_THRESHOLD
                        tp_sum += float(matrix[hits].sum())
                        tp += int(hits.sum())
                        fp += len(rles) - int(hits.sum())
                        fn += len(truth) - int(hits.sum())
                    counts[key][t][name] = (tp_sum, tp, fp, fn)

            if position % 25 == 0 or position == len(names):
                print(f"  fold{fold} {position}/{len(names)}  "
                      f"{time.perf_counter() - started:.0f}s", flush=True)

    out_path = args.out / "v82_refiner_counts.json"
    out_path.write_text(json.dumps(
        {key: {str(t): c for t, c in per_tau.items()}
         for key, per_tau in counts.items()},
        ensure_ascii=False), encoding="utf-8")
    (args.out / "v82_refiner_counts_provenance.json").write_text(json.dumps({
        "result_kind": "exploratory_paired_counts",
        "adoption_eligible": False,
        "observed_folds_sha256": hashlib.sha256(grouped_folds_csv(drive).read_bytes()).hexdigest(),
        "annotations_sha256": hashlib.sha256(train_annotations_json(drive).read_bytes()).hexdigest(),
        "counts_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "missing_checkpoints": missing_ckpt,
        "folds_requested": list(folds), "seeds_requested": list(seeds),
        "limit": args.limit,
        "limitations": [
            "Observed hashes are not independently pinned training-split provenance",
            "Checkpoint/cache training membership and complete input coverage are not verified",
            "Paired counts alone do not establish independent adoption acceptance",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}  missing_ckpt={len(missing_ckpt)}")
    for p in missing_ckpt:
        print("  missing:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
