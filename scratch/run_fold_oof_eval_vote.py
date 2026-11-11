"""fold N OOF で prod 検出集合を固定した cross-proposal 輪郭投票を採点 (P2/v125)。

`run_fold_oof_eval_consensus.py` (v120/§2.98) からの派生。postprocess は同一:
detector fold-N 推論 (imgsz=1280, conf=0.08, iou=0.55, agnostic_nms,
retina_masks) → refiner で 320px crop の確率マップ → threshold で shrink →
drop_contained(0.90) → greedy_disjoint(min_area=20) → 公式 PQ
(annotator x image group 毎、IoU>0.5 厳密より大きい)。

差分は「何を mask にするか」だけ。検出集合は **prod (yolo detector の
conf>=--conf 提案) で完全固定** — leader の追加・削除はしない
(consensus 系の存在フィルタとは別の実験、調査計画 2026-11-10 P2)。

arm (事前固定・2 つのみ):
  prod      : leader の refined mask のみ (対照 = prod pipeline と同一)
  vote_eq05 : leader refined + voter refined の等重み pixel 多数決
              (score(p) = 平均 >= 0.5 を保持。N=2 なら intersection、
              N=3 なら >=2 mask が支持 — voter 2 枚が一致する領域のみ
              leader 範囲外へ拡張できる。閾値の後付け調整はしない)

voter 対応付け (事前固定):
  - 非 `yolo` detector (crop, yolo26) の det-floor 提案のうち、leader の
    det mask と mask-IoU >= --vote-iou のもの
  - 各 voter 提案は IoU 最大の leader 1 つにのみ所属 (曖昧 chain で
    別個体を融合しない)
  - voter は raw conf を比較しない (detector 間の校正なし・等重み)

`yolo` detector は manifest の `yolo` role で binding、他は
--trusted-sha256 の byte-pin で受理 (§2.99)。

使い方:
    python scratch/run_fold_oof_eval_vote.py \
        --train-json train/..._train.json --preproc-dir preprocessed_v30_q96 \
        --folds-csv grouped_folds.csv --fold 0 \
        --provenance-manifest reviewed.json --provenance-sha256 <pin> \
        --detector yolo=yolo_fold0_best.pt \
        --detector crop=v121a_crop_best.pt --detector yolo26=v121b_best.pt \
        --trusted-sha256 <crop_sha> --trusted-sha256 <yolo26_sha> \
        --refiner baseline=mediseg_edgeattnet_refiner_fold0_resume.pt \
        --out output/v125_eval.json
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
                        help="prod leader 提案の operating conf")
    parser.add_argument("--det-floor", type=float, default=0.08,
                        help="voter detector の提案保持 floor")
    parser.add_argument("--vote-iou", type=float, default=0.5,
                        help="voter → leader 対応付けの det-mask IoU 閾値")
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
        raise ValueError("--detector yolo=... は必須 (prod leader/binding の基準)")

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

    # 1) 全 detector の提案を det-floor で保持 (voter 入力は arm 共通)
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

    # 2) leader = yolo conf>=--conf のみ (検出集合固定)。voter は非 yolo 提案を
    #    最良 IoU leader 1 つに割当 (mask-IoU >= --vote-iou)
    def _mask_iou(a, b):
        inter = int(np.logical_and(a, b).sum())
        return inter / int(np.logical_or(a, b).sum()) if inter else 0.0

    leaders_by_file = {}      # name -> [mask]
    voters_by_file = {}       # name -> [[voter masks per leader idx]]
    vote_diag = {"leaders_per_image": [], "voters_per_leader": [],
                 "unassigned_voters": 0, "total_voters": 0}
    for name in oof_files:
        props = props_by_file[name]
        leaders = [m for d, m, s in props
                   if d == "yolo" and s >= args.conf]
        candidates = [(d, m, s) for d, m, s in props if d != "yolo"]
        per_leader = [[] for _ in leaders]
        for d, m, s in candidates:
            vote_diag["total_voters"] += 1
            if not leaders:
                vote_diag["unassigned_voters"] += 1
                continue
            ious = [_mask_iou(m, lm) for lm in leaders]
            best = int(np.argmax(ious))
            if ious[best] >= args.vote_iou:
                per_leader[best].append(m)
            else:
                vote_diag["unassigned_voters"] += 1
        leaders_by_file[name] = leaders
        voters_by_file[name] = per_leader
        vote_diag["leaders_per_image"].append(len(leaders))
        vote_diag["voters_per_leader"].extend(len(v) for v in per_leader)
    print(json.dumps({
        "leaders_per_image": float(np.mean(vote_diag["leaders_per_image"]))
            if vote_diag["leaders_per_image"] else 0.0,
        "voters_per_leader": float(np.mean(vote_diag["voters_per_leader"]))
            if vote_diag["voters_per_leader"] else 0.0,
        "unassigned_voters": vote_diag["unassigned_voters"],
        "total_voter_props": vote_diag["total_voters"]},
        indent=1), flush=True)

    # 3) leader + voter を同一 refiner で refine → arm 別 mask 合成 → PQ
    ARM_NAMES = ("prod", "vote_eq05")
    report = {"fold": args.fold, "folds_sha256": folds_sha256,
              "provenance_manifest_sha256": args.provenance_sha256,
              "detector_sha256": detector_shas,
              "yolo_sha256": yolo_sha256,
              "conf": args.conf, "det_floor": args.det_floor,
              "vote_iou": args.vote_iou, "vote_rule": "equal-weight pixel majority >=0.5",
              "min_area": MIN_AREA, "drop_threshold": DROP_THRESHOLD,
              "thresholds": thresholds,
              "leaders_per_image": float(np.mean(vote_diag["leaders_per_image"]))
                  if vote_diag["leaders_per_image"] else 0.0,
              "voters_per_leader": float(np.mean(vote_diag["voters_per_leader"]))
                  if vote_diag["voters_per_leader"] else 0.0,
              "unassigned_voters": vote_diag["unassigned_voters"],
              "oof_files": len(oof_files), "arms": {}}
    for arm_name, arm in refiners.items():
        for prop_arm in ARM_NAMES:
            meters = {t: PanopticQuality() for t in thresholds}
            key = f"{arm_name}::{prop_arm}"
            area_deltas = []
            for name in oof_files:
                image_path = args.preproc_dir / name
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                leaders = leaders_by_file[name]
                voters = voters_by_file[name]
                # leader/voter をまとめて refine (proposal 数を最小化)
                all_props, index = [], []
                for li, lm in enumerate(leaders):
                    index.append((li, -1)); all_props.append(lm)
                    for vi, vm in enumerate(voters[li]):
                        index.append((li, vi)); all_props.append(vm)
                prob_maps = [refine_probability(arm["model"], image, m)
                             for m in all_props]
                for threshold in thresholds:
                    refined_idx = {}
                    for (li, vi), (probability, box), proposal in zip(
                            index, prob_maps, all_props):
                        x0, y0, x1, y1 = box
                        mask = np.zeros_like(proposal, dtype=bool)
                        mask[y0:y1, x0:x1] = probability >= threshold
                        refined_idx[(li, vi)] = mask & proposal
                    predictions = []
                    for li in range(len(leaders)):
                        if prop_arm == "prod":
                            predictions.append(refined_idx[(li, -1)])
                            continue
                        members = [refined_idx[(li, -1)]]
                        members.extend(refined_idx[(li, vi)]
                                       for vi in range(len(voters[li])))
                        stack = np.stack(members).astype(np.float32)
                        voted = stack.mean(axis=0) >= 0.5
                        area_deltas.append(int(voted.sum())
                                           - int(members[0].sum()))
                        predictions.append(voted)
                    predictions = apply_ownership(
                        predictions, np.ones(len(predictions)),
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
                "emitted": int(np.mean([len(leaders_by_file[n])
                                        for n in oof_files])),
                "mean_area_delta_px": (float(np.mean(area_deltas))
                                       if prop_arm == "vote_eq05" and area_deltas
                                       else 0.0),
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
                      for a, r in report["arms"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
