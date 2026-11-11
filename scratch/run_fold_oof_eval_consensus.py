"""fold N OOF で複数検出器の提案をクラスタ化し、合意ゲートで採点 (v120)。

`run_fold_oof_eval.py` (v73) からの派生。postprocess は同一:
detector fold-N 推論 (imgsz=1280, conf=0.08, iou=0.55, agnostic_nms,
retina_masks) → arm 別の提案集合 → refiner で 320px crop の
確率マップ → threshold で shrink → drop_contained(0.90) →
greedy_disjoint(min_area=20) → 公式 PQ (annotator x image group 毎、
IoU>0.5 厳密より大きい)。

差分は提案生成だけ: 複数 --detector の提案 (conf floor=--det-floor) を
mask-IoU >= --consensus-iou の推移クラスタ (union-find) にまとめ、
arm ごとの合意ルールで放出する:
  prod        : `yolo` という名の detector の提案のみ conf>=--conf
  union3      : 全クラスタ (>=1 detector) — max-conf member の mask
  union3_c26  : max member conf>=--conf のクラスタ (v115 型 union) — 同上
  cons2of3    : >=2 detector が支持するクラスタ — max-conf member の mask
  cons2of3_c26: 支持 member のうち conf>=--conf のものが >=2 detector — 同上
  cons3of3    : 3 detector 全員が支持 — 同上

`yolo` という名の detector は manifest の `yolo` role で binding 検証し、
他は --trusted-sha256 の byte-pin で受理する (§2.98)。

使い方:
    python scratch/run_fold_oof_eval_consensus.py \
        --train-json train/..._train.json --preproc-dir preprocessed_v30_q96 \
        --folds-csv grouped_folds.csv --fold 2 \
        --detector yolo=yolo_fold2_best.pt \
        --detector crop=v116_crop_best.pt --detector yolo26=v117_best.pt \
        --trusted-sha256 <crop_sha> --trusted-sha256 <yolo26_sha> \
        --refiner baseline=mediseg_edgeattnet_refiner_fold2_resume.pt \
        --out output/v120_eval.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scratch.v73_split_provenance import (
        load_fold_mapping, load_reviewed_manifest, validate_checkpoint_binding,
        validate_training_corpus_provenance)
except ImportError:
    from v73_split_provenance import (
        load_fold_mapping, load_reviewed_manifest, validate_checkpoint_binding,
        validate_training_corpus_provenance)
try:
    from src.metrics_official import PanopticQuality  # noqa: E402
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics_official import PanopticQuality  # noqa: E402

CROP_SIZE = 320
CROP_CONTEXT = 1.65
CONF = 0.26
MIN_AREA = 20
DROP_THRESHOLD = 0.90
EVAL_SIZE = 2048


def load_models():
    try:
        from solar_filament.models.mediseg_models import (
            MediSegEdgeAttUNet, MediSegUNet)
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mediseg_models import MediSegEdgeAttUNet, MediSegUNet
    return MediSegUNet, MediSegEdgeAttUNet


def load_ownership():
    """canonical apply_ownership を repo / flat dataset 両配置で解決。"""
    try:
        from solar_filament.pipeline.ownership import apply_ownership
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ownership import apply_ownership
    return apply_ownership


def to_rle(annotation, height, width, mask_utils):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def iou_matrix(truth, pred, mask_utils):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)),
                      dtype=np.float64).T


def mask_square_box(mask, context=CROP_CONTEXT, minimum=96):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, width, height
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1,
                             ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


@torch.no_grad()
def refine_probability(refiner, image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE),
                      interpolation=cv2.INTER_LINEAR)
    tensor = (torch.from_numpy(crop.transpose(2, 0, 1)[None].copy())
              .float().cuda() / 255.0)
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    return (cv2.resize(probability, (x1 - x0, y1 - y0),
                       interpolation=cv2.INTER_LINEAR), (x0, y0, x1, y1))


def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height),
                              interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = (result.boxes.conf.detach().cpu().numpy().astype(np.float32)
                   if result.boxes is not None else np.zeros(0, np.float32))
    return output, confidences


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--preproc-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--folds-csv", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True,
                        help="reviewed JSON binding YOLO/legacy weights to fold/split")
    parser.add_argument("--provenance-sha256", required=True,
                        help="independently reviewed manifest SHA256; never derive at runtime")
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--detector", action="append", required=True,
                        help="name=ckpt_path (複数指定可。`yolo` は manifest binding、"
                             "他は --trusted-sha256 pin で受理)")
    parser.add_argument("--refiner", action="append", required=True,
                        help="name=ckpt_path (複数指定可、同一提案で採点)")
    parser.add_argument("--trusted-sha256", action="append", default=[],
                        help="外部検証済み ckpt の sha256 pin (repeatable)。"
                             "一致バイトは manifest/metadata binding を代替受理")
    parser.add_argument("--model", type=str, default="edgeatt",
                        choices=("unet", "edgeatt"))
    parser.add_argument("--thresholds", type=str, default="0.35,0.40,0.45,0.50")
    parser.add_argument("--conf", type=float, default=CONF,
                        help="prod arm / cons2of3_c26 の operating conf")
    parser.add_argument("--det-floor", type=float, default=0.08,
                        help="全 detector の提案保持 floor (cluster 入力)")
    parser.add_argument("--consensus-iou", type=float, default=0.5,
                        help="cross-detector cluster の mask-IoU 閾値")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trusted = set(args.trusted_sha256)
    manifest = load_reviewed_manifest(
        args.provenance_manifest, args.provenance_sha256, args.fold)
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    thresholds = [float(t) for t in args.thresholds.split(",")]
    arms = []
    for spec in args.refiner:
        name, _, path = spec.partition("=")
        if (not name or name == "yolo" or name in {n for n, _ in arms}
                or not path or not Path(path).exists()):
            raise ValueError(f"--refiner name=path が不正: {spec}")
        arms.append((name, Path(path)))

    detectors = []
    for spec in args.detector:
        name, _, path = spec.partition("=")
        if (not name or not path or not Path(path).exists()
                or name in {n for n, _ in detectors}):
            raise ValueError(f"--detector name=path が不正: {spec}")
        detectors.append((name, Path(path)))
    if "yolo" not in {n for n, _ in detectors}:
        raise ValueError("--detector yolo=... は必須 (prod arm/binding の基準)")

    folds_sha256 = hashlib.sha256(args.folds_csv.read_bytes()).hexdigest()
    with open(args.train_json, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {i["id"]: i for i in coco["images"]}
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    file_to_ids = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)
    fold_of_file = load_fold_mapping(
        args.folds_csv, file_to_ids, args.fold, manifest["folds_sha256"])
    detector_shas = {}
    for name, path in detectors:
        digest = sha256_of(path)
        if name == "yolo":
            validate_checkpoint_binding(path, "yolo", manifest)
        elif digest not in trusted:
            raise ValueError(
                f"detector {name} bytes match no reviewed binding and is not "
                f"in --trusted-sha256: {digest[:16]}")
        detector_shas[name] = digest
        print(f"detector {name}: sha256={digest[:16]}", flush=True)
    yolo_sha256 = detector_shas["yolo"]
    checked_arms = {}
    for name, path in arms:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        digest = validate_checkpoint_binding(path, name, manifest, checkpoint)
        provenance = checkpoint.get("init_report", {}).get("corpus_provenance")
        if str(checkpoint.get("init", "")).startswith("ssl:") or provenance is not None:
            validate_training_corpus_provenance(provenance, coco, args.test_dir)
        checked_arms[name] = (checkpoint, digest)
    # All fold/split and legacy byte bindings pass before any CUDA/model work.
    oof_files = [n for n in sorted(file_to_ids)
                 if fold_of_file.get(n) == args.fold]
    if args.limit:
        oof_files = oof_files[:args.limit]
    print({"fold": args.fold, "oof_files": len(oof_files),
           "arms": [n for n, _ in arms], "folds_sha256": folds_sha256[:16]},
          flush=True)

    MediSegUNet, MediSegEdgeAttUNet = load_models()
    apply_ownership = load_ownership()
    model_cls = MediSegEdgeAttUNet if args.model == "edgeatt" else MediSegUNet
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO

    gt_rles = {image_id: [to_rle(a, int(info["height"]), int(info["width"]),
                                 mask_utils)
                          for a in annotations[image_id]
                          if not a.get("iscrowd", 0)]
               for image_id, info in images.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    refiners = {}
    for name, path in arms:
        ckpt, digest = checked_arms.pop(name)
        model = model_cls(3, 1, False).cuda().eval()
        model.load_state_dict(ckpt.get("best_model") or ckpt["model"],
                              strict=True)
        refiners[name] = {"model": model,
                          "sha256": digest,
                          "corpus_provenance": ckpt.get("init_report", {}).get("corpus_provenance"),
                          "best_dice": ckpt.get("best_dice"),
                          "init": ckpt.get("init", "pretrained")}
        print(f"arm {name}: best_dice={refiners[name]['best_dice']}",
              flush=True)

    det_models = {n: YOLO(str(p)) for n, p in detectors}
    started = time.time()

    # 1) 全 detector の提案を det-floor で保持 (cluster 入力は arm 共通)
    props_by_file = {}
    for name in oof_files:
        image_path = args.preproc_dir / name
        entries = []
        for det_name, model in det_models.items():
            result = model.predict(
                str(image_path), imgsz=1280, conf=args.det_floor, iou=0.55,
                agnostic_nms=True, retina_masks=True, verbose=False)[0]
            masks, confidences = result_masks(result)
            for m, s in zip(masks, confidences):
                if s >= args.det_floor and m.sum() >= MIN_AREA:
                    entries.append((det_name, m, float(s)))
        props_by_file[name] = entries
    print(f"proposals done: {time.time() - started:.0f}s", flush=True)

    # 2) cross-detector cluster → arm 別の放出提案集合
    def _mask_iou(a, b):
        inter = int(np.logical_and(a, b).sum())
        return inter / int(np.logical_or(a, b).sum()) if inter else 0.0

    def consensus_clusters(props, iou_thr):
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
                # mask IoU <= box_inter / max(mask_area) — 上界未満なら skip
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

    ARM_NAMES = ("prod", "union3", "union3_c26", "cons2of3",
                 "cons2of3_c26", "cons3of3")
    emitted_by_file = {a: {n: [] for n in oof_files} for a in ARM_NAMES}
    cluster_diag = {"clusters_per_image": [],
                    "sources_per_cluster": []}
    for name in oof_files:
        props = props_by_file[name]
        emitted_by_file["prod"][name] = [
            m for d, m, s in props if d == "yolo" and s >= args.conf]
        clusters = consensus_clusters(props, args.consensus_iou)
        cluster_diag["clusters_per_image"].append(len(clusters))
        for mem in clusters:
            sources = {props[i][0] for i in mem}
            strong = {props[i][0] for i in mem
                      if props[i][2] >= args.conf}
            mask = props[max(mem, key=lambda i: props[i][2])][1]
            cluster_diag["sources_per_cluster"].append(len(sources))
            emitted_by_file["union3"][name].append(mask)
            if max(props[i][2] for i in mem) >= args.conf:
                emitted_by_file["union3_c26"][name].append(mask)
            if len(sources) >= 2:
                emitted_by_file["cons2of3"][name].append(mask)
            if len(strong) >= 2:
                emitted_by_file["cons2of3_c26"][name].append(mask)
            if len(sources) >= 3:
                emitted_by_file["cons3of3"][name].append(mask)
    print(json.dumps({
        "emitted_per_arm": {a: int(np.mean([len(v[n]) for n in oof_files]))
                            for a, v in emitted_by_file.items()},
        "clusters_per_image": float(np.mean(cluster_diag["clusters_per_image"])),
        "sources_per_cluster": float(np.mean(cluster_diag["sources_per_cluster"]))},
        indent=1), flush=True)

    # 3) refiner 確率マップを (refiner × proposal arm) 毎に計算 → threshold sweep → PQ
    report = {"fold": args.fold, "folds_sha256": folds_sha256,
              "provenance_manifest_sha256": args.provenance_sha256,
              "detector_sha256": detector_shas,
              "yolo_sha256": yolo_sha256,
              "conf": args.conf, "det_floor": args.det_floor,
              "consensus_iou": args.consensus_iou, "min_area": MIN_AREA,
              "drop_threshold": DROP_THRESHOLD, "thresholds": thresholds,
              "emitted_per_arm": {a: int(np.mean([len(v[n]) for n in oof_files]))
                                for a, v in emitted_by_file.items()},
              "oof_files": len(oof_files), "arms": {}}
    for arm_name, arm in refiners.items():
        for prop_arm in ARM_NAMES:
            meters = {t: PanopticQuality() for t in thresholds}
            key = f"{arm_name}::{prop_arm}"
            for name in oof_files:
                image_path = args.preproc_dir / name
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                proposals = emitted_by_file[prop_arm][name]
                prob_maps = [refine_probability(arm["model"], image, m)
                             for m in proposals]
                for threshold in thresholds:
                    refined = []
                    for (probability, box), proposal in zip(
                            prob_maps, proposals):
                        x0, y0, x1, y1 = box
                        mask = np.zeros_like(proposal, dtype=bool)
                        mask[y0:y1, x0:x1] = probability >= threshold
                        refined.append(mask & proposal)
                    predictions = apply_ownership(
                        refined, np.ones(len(refined)),
                        rule="drop_contained_then_area_asc",
                        containment_threshold=DROP_THRESHOLD,
                        min_area=MIN_AREA, canvas=EVAL_SIZE)
                    pred_rles = (mask_utils.encode(np.asfortranarray(
                        np.stack(predictions, axis=-1).astype(np.uint8)))
                        if predictions else [])
                    for image_id in file_to_ids[name]:
                        truth = gt_rles[image_id]
                        matrix = iou_matrix(truth, pred_rles, mask_utils)
                        meters[threshold].add_matrix(matrix, len(truth),
                                                     len(pred_rles))
            report["arms"][key] = {
                "refiner": arm_name, "proposal_arm": prop_arm,
                "ckpt_sha256": arm["sha256"], "init": arm["init"],
                "corpus_provenance": arm["corpus_provenance"],
                "best_crop_dice": arm["best_dice"],
                "emitted": int(np.mean([len(emitted_by_file[prop_arm][n])
                                        for n in oof_files])),
                "by_threshold": {str(t): meters[t].summary()
                                 for t in thresholds},
            }
            report["arms"][key]["pq_by_threshold"] = {
                str(t): round(meters[t].score, 6) for t in thresholds}
            print({key: report["arms"][key]["pq_by_threshold"]},
                  flush=True)

    report["minutes"] = round((time.time() - started) / 60, 1)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps({a: r["pq_by_threshold"]
                      for a, r in report["arms"].items()},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
