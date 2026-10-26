"""refiner (MediSegUNet / MediSegEdgeAttUNet) を指定 fold 構成で学習する (v73)。

build_v35_refiner_folds12.py / build_v51_edgeattnet_refiner.py が notebook に
埋め込んでいた学習ループを、そのまま CLI 化したもの。**レシピは一字一句
同じ** (record 単位 InstanceCropDataset、crop 320 / context 1.65、aug は
rot90+h/v flip+ch0/1 gamma、loss = 0.35 BCE + 0.65 soft Dice + 0.15 edge L1、
AdamW lr 2e-4 wd 1e-4、CosineAnnealing、14 epoch、best-Dice 保存)。

違いは `--init` だけ:
  - `scratch`      : ランダム初期化 (現行 recipe)
  - `ssl:<path>`   : encoder (inc/down1-4) だけ SSL ckpt で初期化
                     (pretrain_unet_encoder_ssl.py の出力。decoder/edge_attn
                     は従来どおりランダム)
  - `ckpt:<path>`  : resume 用に full state_dict を読む (既存互換)

リーク契約: `--fold N` は fold N の観測を **学習から完全に除外** し
fold N を valid に使う。grouped_folds.csv の SHA256 を ckpt に記録する。

使い方:
    python scratch/train_refiner_fold.py \
        --train-json train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --preproc-dir preprocessed_v30_q96 --folds-csv grouped_folds.csv \
        --fold 2 --model edgeatt --init ssl:unet_encoder_ssl.pt \
        --out edgeatt_ssl_fold2.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scratch.v73_split_provenance import (
        load_fold_mapping, validate_checkpoint_metadata, validate_corpus_provenance,
        validate_training_corpus_provenance)
except ImportError:
    from v73_split_provenance import (
        load_fold_mapping, validate_checkpoint_metadata, validate_corpus_provenance,
        validate_training_corpus_provenance)


def load_models():
    """code dataset の flat 配置 (mediseg_models.py 同梱) と repo 配置の両対応。"""
    try:
        from solar_filament.models.mediseg_models import (
            MediSegEdgeAttUNet, MediSegUNet)
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mediseg_models import MediSegEdgeAttUNet, MediSegUNet
    return MediSegUNet, MediSegEdgeAttUNet


# --- 以下、build_v35_refiner_folds12.py と同一ロジック ---

CROP_SIZE = 320
CROP_CONTEXT = 1.65
REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)


def square_box_from_bbox(bbox, height, width, context=CROP_CONTEXT, minimum=96):
    x, y, w, h = map(float, bbox)
    cx, cy = x + w / 2, y + h / 2
    side = int(math.ceil(max(w, h) * context))
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


def ann_mask(image_id, ann, images, coco_mask):
    info = images[image_id]
    height, width = int(info["height"]), int(info["width"])
    segmentation = ann["segmentation"]
    if isinstance(segmentation, list):
        rle = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get("counts"), list):
        rle = coco_mask.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return coco_mask.decode(rle).astype(np.uint8)


def soft_dice_loss(logits, targets):
    probs = logits.sigmoid()
    intersection = (probs * targets).sum((1, 2, 3))
    denominator = probs.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def boundary_map(tensor):
    dilation = F.max_pool2d(tensor, 5, stride=1, padding=2)
    erosion = -F.max_pool2d(-tensor, 5, stride=1, padding=2)
    return torch.clamp(dilation - erosion, 0, 1)


def refiner_loss(logits, targets):
    region = (0.35 * F.binary_cross_entropy_with_logits(logits, targets)
              + 0.65 * soft_dice_loss(logits, targets))
    edge = F.l1_loss(boundary_map(logits.sigmoid()), boundary_map(targets))
    return region + 0.15 * edge


def crop_dice(model, loader, threshold=0.5):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for image, target in loader:
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True).bool()
            pred = model(image).sigmoid() >= threshold
            intersection = (pred & target).sum((1, 2, 3)).float()
            denominator = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
            total += ((2 * intersection + 1) / (denominator + 1)).sum().item()
            count += len(image)
    return total / max(count, 1)


def apply_encoder_init(model: torch.nn.Module, ssl_ckpt: Path) -> dict:
    """encoder (inc/down1-4) だけ SSL state_dict で初期化する。

    戻り値は load_state_dict(strict=False) の結果 (missing/unexpected)。
    encoder キー以外を含む ckpt は拒否する (誤って別モデルを読まないため)。
    """
    state = torch.load(ssl_ckpt, map_location="cpu", weights_only=True)
    provenance = validate_corpus_provenance(state.get("corpus_provenance"))
    state = state.get("encoder")
    if not state:
        raise ValueError(f"SSL ckpt が空です: {ssl_ckpt}")
    encoder_prefix = ("inc.", "down1.", "down2.", "down3.", "down4.")
    bad = [k for k in state if not k.startswith(encoder_prefix)]
    if bad:
        raise ValueError(
            f"SSL ckpt に encoder 以外のキーが含まれます: {bad[:5]}")
    result = model.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys]
    if unexpected:
        raise ValueError(f"SSL ckpt のキーがモデルと不一致: {unexpected[:5]}")
    partial = [k for k in result.missing_keys if k.startswith(encoder_prefix)]
    if partial:
        raise ValueError(
            f"SSL ckpt に encoder キーの一部が欠落: {partial[:5]}")
    return {"loaded": sorted(state), "missing": sorted(result.missing_keys),
            "corpus_provenance": provenance}


class InstanceCropDataset(Dataset):
    """(file, image_id, ann) record 単位の crop 学習セット (v35 完全同一)。"""

    def __init__(self, file_names, file_to_ids, annotations, images,
                 preproc_dir, coco_mask, all_labelers=True, augment=False):
        self.samples, self.augment = [], augment
        self.file_to_ids, self.annotations, self.images = (
            file_to_ids, annotations, images)
        self.preproc_dir, self.coco_mask = preproc_dir, coco_mask
        for file_name in file_names:
            for image_id in (file_to_ids[file_name] if all_labelers else []):
                for ann in annotations[image_id]:
                    if not ann.get("iscrowd", 0) and ann.get("area", 0) > 0:
                        self.samples.append((file_name, image_id, ann))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, image_id, ann = self.samples[index]
        info = self.images[image_id]
        image = cv2.imread(str(self.preproc_dir / file_name), cv2.IMREAD_COLOR)
        target = ann_mask(image_id, ann, self.images, self.coco_mask)
        x0, y0, x1, y1 = square_box_from_bbox(
            ann["bbox"], info["height"], info["width"])
        image = cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE),
                            interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        target = torch.from_numpy(target[None].copy()).float()
        if self.augment:
            k = random.randrange(4)
            image = torch.rot90(image, k, (1, 2))
            target = torch.rot90(target, k, (1, 2))
            if random.random() < 0.5:
                image, target = torch.flip(image, (2,)), torch.flip(target, (2,))
            if random.random() < 0.5:
                image, target = torch.flip(image, (1,)), torch.flip(target, (1,))
            if random.random() < 0.5:
                image[:2] = torch.clamp(image[:2] ** random.uniform(0.85, 1.20), 0, 1)
        return image, target


def choose_batch_size(model_fn, dataset):
    for candidate in REFINER_BATCH_CANDIDATES:
        try:
            loader = DataLoader(dataset, batch_size=candidate, shuffle=True,
                                num_workers=0)
            image, target = next(iter(loader))
            probe = model_fn().cuda()
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(probe(image.cuda()), target.cuda())
            loss.backward()
            del probe, image, target, loss, loader
            torch.cuda.empty_cache()
            return candidate
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("どのbatch sizeもGPUメモリに収まらない")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--preproc-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--folds-csv", type=Path, required=True)
    parser.add_argument("--expected-folds-sha256", required=True,
                        help="independently reviewed frozen split SHA256")
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--model", type=str, default="edgeatt",
                        choices=("unet", "edgeatt"))
    parser.add_argument("--init", type=str, default="scratch",
                        help="scratch | ssl:<ckpt> | ckpt:<ckpt(resume)>")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    MediSegUNet, MediSegEdgeAttUNet = load_models()
    model_cls = MediSegEdgeAttUNet if args.model == "edgeatt" else MediSegUNet
    from pycocotools import mask as coco_mask

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
    train_files = sorted(file_to_ids)
    fold_of_file = load_fold_mapping(
        args.folds_csv, train_files, args.fold, args.expected_folds_sha256)
    resume_state = None
    if args.init.startswith("ckpt:"):
        resume_state = torch.load(args.init[5:], map_location="cpu",
                                  weights_only=True)
        validate_checkpoint_metadata(resume_state, args.fold, folds_sha256)
        provenance = resume_state.get("init_report", {}).get("corpus_provenance")
        if str(resume_state.get("init", "")).startswith("ssl:") or provenance is not None:
            validate_training_corpus_provenance(provenance, coco, args.test_dir)
    elif args.init.startswith("ssl:"):
        encoder = torch.load(args.init[4:], map_location="cpu", weights_only=True)
        validate_training_corpus_provenance(
            encoder.get("corpus_provenance"), coco, args.test_dir)
        del encoder
    elif args.init != "scratch":
        raise ValueError("unknown --init; use scratch, ssl:<path> or ckpt:<path>")

    train_names = [n for n in train_files if fold_of_file[n] != args.fold]
    valid_names = [n for n in train_files if fold_of_file[n] == args.fold]
    assert not set(train_names) & set(valid_names), "fold分割が壊れている"
    missing = [n for n in train_names + valid_names
               if not (args.preproc_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"前処理画像が {len(missing)} 件不足: {missing[:3]}")

    train_ds = InstanceCropDataset(train_names, file_to_ids, annotations,
                                   images, args.preproc_dir, coco_mask,
                                   augment=True)
    valid_ds = InstanceCropDataset(valid_names, file_to_ids, annotations,
                                   images, args.preproc_dir, coco_mask,
                                   augment=False)
    print({"fold": args.fold, "model": args.model, "init": args.init,
           "seed": args.seed, "train_crops": len(train_ds),
           "valid_crops": len(valid_ds), "folds_sha256": folds_sha256[:16]},
          flush=True)

    model = model_cls(3, 1, False)
    init_report = {"init": args.init}
    if args.init.startswith("ssl:"):
        init_report.update(
            apply_encoder_init(model, Path(args.init[4:])))
        init_report["encoder_keys"] = len(init_report["loaded"])
        print({"ssl_init": {"loaded": init_report["encoder_keys"],
                          "missing": len(init_report["missing"])}}, flush=True)
    elif args.init.startswith("ckpt:"):
        state = resume_state
        model.load_state_dict(state.get("best_model") or state["model"],
                              strict=True)
        init_report["resumed_from"] = args.init[5:]
        if "corpus_provenance" in state.get("init_report", {}):
            init_report["corpus_provenance"] = state["init_report"]["corpus_provenance"]

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for refiner training")
    model = model.cuda()
    batch_size = choose_batch_size(model_cls, train_ds)
    print("batch_size =", batch_size, flush=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda")
    training_config = {"crop_size": CROP_SIZE, "crop_context": CROP_CONTEXT,
                       "epochs": args.epochs, "lr": args.lr,
                       "model": args.model, "init": args.init,
                       "batch_size": batch_size}

    started = time.time()
    best_dice, best_state = -1.0, None
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        progress = tqdm(train_loader,
                        desc=f"fold{args.fold} {args.model} {args.init.split(':')[0]} "
                             f"{epoch + 1}/{args.epochs}")
        for image, target in progress:
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = refiner_loss(model(image), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{running / (progress.n + 1):.4f}")
        scheduler.step()
        dice = crop_dice(model, valid_loader)
        if dice > best_dice:
            best_dice = dice
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "best_model": best_state, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_dice": best_dice,
                    "fold": args.fold, "model_class": args.model,
                    "seed": args.seed, "init": args.init,
                    "init_report": init_report,
                    "folds_sha256": folds_sha256,
                    "training_config": training_config}, args.out)
        history.append({"epoch": epoch + 1, "loss": round(running, 4),
                        "valid_dice": round(dice, 6)})
        print(f"fold{args.fold} {args.model} {args.init.split(':')[0]} "
              f"epoch {epoch + 1}: crop Dice {dice:.6f} (best {best_dice:.6f})",
              flush=True)

    result = {"checkpoint": str(args.out), "best_crop_dice": best_dice,
              "fold": args.fold, "model": args.model, "seed": args.seed,
              "init": args.init, "folds_sha256": folds_sha256,
              "epochs": args.epochs, "history": history,
              "minutes": round((time.time() - started) / 60, 1)}
    (args.out.parent / f"train_result_{args.model}_{args.init.split(':')[0]}"
                       f"_fold{args.fold}_s{args.seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
