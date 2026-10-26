"""fold N OOF eval with refiner flip-4 TTA + topology guard (v75).

run_fold_oof_eval.py と同一の proposals/postprocess/PQ 計算を共有し、
refiner 確率マップの計算だけを variant にする paired comparison。
arm 指定は `name=ckpt_path[:mode]` (mode: none|flip4|flip4_guard,
default none)。

flip4: 320px crop を identity/hflip/vflip/hvflip の4view で推論し、
逆変換して平均した確率マップを使う (Fususu handoff 由来)。

flip4_guard: flip4 平均マップが single-view の topology を保存する時だけ
採用し、違反時は single-view にフォールバック。guard は crop 解像度の
0.5 二値マップで判定 (事前固定, RESEARCH_LOG §2.30):
  - 面積比 tta/single が [0.85, 1.15] 内
  - 連結成分数 (>=16px) の差が <=1
  - IoU(single_bin, tta_bin) >= 0.85
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from scratch.run_fold_oof_eval import (  # noqa: E402
        CROP_SIZE, CONF, MIN_AREA, DROP_THRESHOLD, EVAL_SIZE,
        load_models, load_ownership, to_rle, iou_matrix, mask_square_box,
        result_masks)
    from scratch.v73_split_provenance import (  # noqa: E402
        load_fold_mapping, load_reviewed_manifest, sha256_file,
        validate_checkpoint_binding, validate_training_corpus_provenance)
except ImportError:
    from run_fold_oof_eval import (  # noqa: E402
        CROP_SIZE, CONF, MIN_AREA, DROP_THRESHOLD, EVAL_SIZE,
        load_models, load_ownership, to_rle, iou_matrix, mask_square_box,
        result_masks)
    from v73_split_provenance import (  # noqa: E402
        load_fold_mapping, load_reviewed_manifest, sha256_file,
        validate_checkpoint_binding, validate_training_corpus_provenance)

try:
    from src.metrics_official import PanopticQuality  # noqa: E402
except ImportError:
    from metrics_official import PanopticQuality  # noqa: E402

TTA_MODES = ("none", "flip4", "flip4_guard")
GUARD_AREA_RATIO = (0.85, 1.15)
GUARD_MAX_CC_DELTA = 1
GUARD_MIN_IOU = 0.85
GUARD_MIN_CC_AREA = 16


@torch.no_grad()
def _crop_prob(refiner, crop):
    tensor = (torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))[None])
              .float().cuda() / 255.0)
    return refiner(tensor).sigmoid()[0, 0].cpu().numpy()


def _flip(arr, hf, vf):
    out = arr[:, ::-1] if hf else arr
    return out[::-1] if vf else out


def _cc_count(binary):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    return sum(1 for i in range(1, count)
               if stats[i, cv2.CC_STAT_AREA] >= GUARD_MIN_CC_AREA)


def topology_ok(single_prob, tta_prob):
    """事前固定 guard: tta 二値マップが single-view topology を保存するか。"""
    single = single_prob >= 0.5
    tta = tta_prob >= 0.5
    s_area, t_area = int(single.sum()), int(tta.sum())
    if s_area == 0 or t_area == 0:
        return s_area == t_area
    if not GUARD_AREA_RATIO[0] <= t_area / s_area <= GUARD_AREA_RATIO[1]:
        return False
    if abs(_cc_count(single) - _cc_count(tta)) > GUARD_MAX_CC_DELTA:
        return False
    union = int((single | tta).sum())
    return int((single & tta).sum()) / union >= GUARD_MIN_IOU


@torch.no_grad()
def refine_probability_tta(refiner, image, proposal, mode):
    """(prob_map, box, fell_back) を返す。

    mode=none は run_fold_oof_eval.refine_probability と同一経路。
    fell_back は flip4_guard で guard 違反により single-view を採用した時のみ
    True。
    """
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE),
                      interpolation=cv2.INTER_LINEAR)
    single = _crop_prob(refiner, crop)
    fell_back = False
    if mode == "none":
        use = single
    else:
        acc = single.astype(np.float64)
        for hf, vf in ((1, 0), (0, 1), (1, 1)):
            acc += _flip(_crop_prob(refiner, _flip(crop, hf, vf)), hf, vf)
        tta = acc / 4.0
        fell_back = mode == "flip4_guard" and not topology_ok(single, tta)
        use = single if fell_back else tta
    return (cv2.resize(use.astype(np.float32), (x1 - x0, y1 - y0),
                       interpolation=cv2.INTER_LINEAR), (x0, y0, x1, y1),
            fell_back)


def arm_probability(models, image, proposal, mode):
    """ensemble arm の確率マップ: 各 member の prob map (mode 適用済) を
    平均する。box は proposal 由来で全 member 共通。fell_back は
    member 全体が single-view にフォールバックした時のみ True。"""
    probs, box, fbs = [], None, []
    for model in models:
        prob, box, fb = refine_probability_tta(model, image, proposal, mode)
        probs.append(prob.astype(np.float64))
        fbs.append(fb)
    return (np.mean(probs, axis=0).astype(np.float32), box,
            all(fbs) if fbs else False)


def resolve_binding_role(path, name, manifest, checkpoint):
    """Arm 名が manifest の role ならそのまま使う。無ければ、ファイルが
    reviewed checkpoint のどれかとバイト一致する時だけその role を継承する
    (tta4/tta4g・ensemble member が baseline ckpt を共有するケース)。
    ckpt 自体が fold/folds_sha256 metadata を持つ場合は arm 名のまま
    validate_checkpoint_binding の metadata 検証に委ねる
    (v73/v74 で学習された ckpt)。どれにも該当しない未知バイトは拒否する。"""
    declared = manifest["checkpoints"]
    if name in declared:
        return name
    digest = sha256_file(path)
    matched = sorted(role for role, expected in declared.items()
                     if expected == digest)
    if matched:
        return matched[0]
    if (isinstance(checkpoint, dict)
            and ("fold" in checkpoint or "folds_sha256" in checkpoint)):
        return name
    raise ValueError(f"checkpoint provenance missing: {name} "
                     "(bytes match no reviewed checkpoint binding)")


def parse_arm(spec):
    """name=ckpt[:mode] または name=ckpt1+ckpt2+...[:mode] (ensemble arm)。"""
    name, sep, rest = spec.partition("=")
    path, _, mode = rest.rpartition(":")
    if mode not in TTA_MODES:
        path, mode = rest, "none"
    paths = [Path(p) for p in path.split("+") if p]
    if (not name or name == "yolo" or not paths
            or not all(p.exists() for p in paths)):
        raise ValueError(f"--refiner name=path[:mode] が不正: {spec}")
    return name, paths, mode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--preproc-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--folds-csv", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--provenance-sha256", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--refiner", action="append", required=True,
                        help="name=ckpt_path[:none|flip4|flip4_guard]")
    parser.add_argument("--model", type=str, default="edgeatt",
                        choices=("unet", "edgeatt"))
    parser.add_argument("--thresholds", type=str, default="0.35,0.40,0.45,0.50")
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_reviewed_manifest(
        args.provenance_manifest, args.provenance_sha256, args.fold)
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    thresholds = [float(t) for t in args.thresholds.split(",")]
    seen = set()
    arms = []
    for spec in args.refiner:
        name, path, mode = parse_arm(spec)
        if name in seen:
            raise ValueError(f"duplicate arm: {name}")
        seen.add(name)
        arms.append((name, path, mode))

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
    yolo_sha256 = validate_checkpoint_binding(args.yolo, "yolo", manifest)
    checked_arms = {}
    for name, paths, mode in arms:
        members = []
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu",
                                    weights_only=True)
            role = resolve_binding_role(path, name, manifest, checkpoint)
            digest = validate_checkpoint_binding(
                path, role, manifest, checkpoint)
            provenance = checkpoint.get(
                "init_report", {}).get("corpus_provenance")
            if (str(checkpoint.get("init", "")).startswith("ssl:")
                    or provenance is not None):
                validate_training_corpus_provenance(
                    provenance, coco, args.test_dir)
            members.append((checkpoint, digest))
        checked_arms[name] = (members, mode)
    oof_files = [n for n in sorted(file_to_ids)
                 if fold_of_file.get(n) == args.fold]
    if args.limit:
        oof_files = oof_files[:args.limit]
    print({"fold": args.fold, "oof_files": len(oof_files),
           "arms": [(n, m) for n, _, m in arms],
           "folds_sha256": folds_sha256[:16]}, flush=True)

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
    for name, _, mode in arms:
        members, _ = checked_arms.pop(name)
        models = []
        for ckpt, _digest in members:
            model = model_cls(3, 1, False).cuda().eval()
            model.load_state_dict(ckpt.get("best_model") or ckpt["model"],
                                  strict=True)
            models.append(model)
        refiners[name] = {"models": models,
                          "sha256": [d for _, d in members],
                          "tta_mode": mode,
                          "best_dice": [c.get("best_dice")
                                        for c, _ in members],
                          "init": [c.get("init", "pretrained")
                                   for c, _ in members]}
        print(f"arm {name}: members={len(models)} tta={mode} "
              f"best_dice={refiners[name]['best_dice']}", flush=True)

    yolo_model = YOLO(str(args.yolo))
    started = time.time()

    proposals_by_file = {}
    for name in oof_files:
        image_path = args.preproc_dir / name
        result = yolo_model.predict(
            str(image_path), imgsz=1280, conf=0.08, iou=0.55,
            agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, confidences = result_masks(result)
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= args.conf and m.sum() >= MIN_AREA]
        proposals_by_file[name] = [masks[i] for i in keep]
    print(f"proposals done: {time.time() - started:.0f}s", flush=True)

    report = {"fold": args.fold, "folds_sha256": folds_sha256,
              "provenance_manifest_sha256": args.provenance_sha256,
              "yolo_sha256": yolo_sha256,
              "conf": args.conf, "min_area": MIN_AREA,
              "drop_threshold": DROP_THRESHOLD, "thresholds": thresholds,
              "tta_guard": {"area_ratio": list(GUARD_AREA_RATIO),
                            "max_cc_delta": GUARD_MAX_CC_DELTA,
                            "min_iou": GUARD_MIN_IOU,
                            "min_cc_area": GUARD_MIN_CC_AREA},
              "oof_files": len(oof_files), "arms": {}}
    for arm_name, arm in refiners.items():
        meters = {t: PanopticQuality() for t in thresholds}
        fallback = {"applied": 0, "fallback": 0}
        for name in oof_files:
            image_path = args.preproc_dir / name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mode = arm["tta_mode"]
            results = [arm_probability(arm["models"], image, m, mode)
                       for m in proposals_by_file[name]]
            prob_maps = [(prob, box) for prob, box, _ in results]
            fallback["applied"] += sum(1 for *_, fb in results if not fb)
            fallback["fallback"] += sum(1 for *_, fb in results if fb)
            for threshold in thresholds:
                refined = []
                for (probability, box), proposal in zip(
                        prob_maps, proposals_by_file[name]):
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
        report["arms"][arm_name] = {
            "ckpt_sha256": (arm["sha256"][0]
                            if len(arm["sha256"]) == 1 else arm["sha256"]),
            "member_sha256": arm["sha256"], "init": arm["init"],
            "tta_mode": arm["tta_mode"],
            "tta_fallback": fallback if arm["tta_mode"] == "flip4_guard" else None,
            "best_crop_dice": arm["best_dice"],
            "by_threshold": {str(t): meters[t].summary() for t in thresholds},
        }
        report["arms"][arm_name]["pq_by_threshold"] = {
            str(t): round(meters[t].score, 6) for t in thresholds}
        print({arm_name: report["arms"][arm_name]["pq_by_threshold"]},
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
