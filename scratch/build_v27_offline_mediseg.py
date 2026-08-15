import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_kaggle_ensemble_mediseg_v27_offline_realdata.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip()}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip(),
    }


model_source = (ROOT / "src" / "mediseg_models.py").read_text(encoding="utf-8")

cells = [
    md(
        """
        # Solar Filament Segmentation - V27 offline real-data MediSeg

        This is the MediSeg-based correction of V26. It uses the repository's
        `MediSegUNet` and `CombinedLoss`, trains them on the real MAGFiLO COCO
        annotations, separates the semantic prediction into individual filament
        instances, and writes official compressed COCO RLE counts.

        - No `pip install`
        - No Internet access required
        - No YOLO/Ultralytics dependency
        - The synthetic `best_solar_unet` checkpoint is intentionally not used
        - Kaggle accelerator: GPU
        """
    ),
    code(
        """
        import gc
        import glob
        import json
        import os
        import random
        from collections import defaultdict
        from pathlib import Path

        import cv2
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        try:
            from pycocotools import mask as mask_utils
            HAS_PYCOCOTOOLS = True
        except ImportError:
            mask_utils = None
            HAS_PYCOCOTOOLS = False

        SEED = 42
        INPUT_SIZE = (512, 512)
        OUTPUT_SIZE = (2048, 2048)
        BATCH_SIZE = 1
        ACCUMULATION_STEPS = 4
        EPOCHS = 20
        PATIENCE = 5
        LEARNING_RATE = 3e-4
        WEIGHT_DECAY = 1e-4
        NUM_WORKERS = 2
        MIN_COMPONENT_AREA = 20
        OUTPUT_PATH = "/kaggle/working/submission_v27_offline_realdata_mediseg.csv"
        MODEL_PATH = "/kaggle/working/best_v27_mediseg_unet.pth"

        def seed_everything(seed=SEED):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        seed_everything()
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print({"device": str(DEVICE), "pycocotools": HAS_PYCOCOTOOLS})
        """
    ),
    code(
        """
        ROOT_CANDIDATES = [
            "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
            "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
        ]
        ROOT_DIR = next((p for p in ROOT_CANDIDATES if os.path.isdir(p)), None)
        if ROOT_DIR is None:
            matches = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026", recursive=True)
            ROOT_DIR = matches[0] if matches else None
        if ROOT_DIR is None:
            raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 was not found under /kaggle/input")

        TRAIN_DIR = os.path.join(ROOT_DIR, "train", "train_images")
        TEST_DIR = os.path.join(ROOT_DIR, "test", "test_images")
        TRAIN_JSON = os.path.join(
            ROOT_DIR, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        )
        required = [TRAIN_DIR, TEST_DIR, TRAIN_JSON]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing competition inputs: {missing}")

        with open(TRAIN_JSON, "r", encoding="utf-8") as handle:
            coco = json.load(handle)

        images_by_id = {item["id"]: item for item in coco["images"]}
        annotations_by_id = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_id[annotation["image_id"]].append(annotation)

        # Some observations were annotated by multiple labelers. Select one complete
        # annotation set per physical image to avoid train/validation leakage.
        file_to_image_ids = defaultdict(list)
        for image_id, info in images_by_id.items():
            file_to_image_ids[info["file_name"]].append(image_id)

        def completeness(image_id):
            anns = annotations_by_id.get(image_id, [])
            return len(anns), sum(float(ann.get("area", 0)) for ann in anns)

        selected_id_by_file = {
            name: max(ids, key=completeness) for name, ids in file_to_image_ids.items()
        }
        filenames = sorted(selected_id_by_file)
        rng = random.Random(SEED)
        rng.shuffle(filenames)
        split = max(1, int(round(0.2 * len(filenames))))
        valid_files = filenames[:split]
        train_files = filenames[split:]
        test_files = sorted(Path(TEST_DIR).glob("*.jpeg"))
        print({
            "root": ROOT_DIR,
            "train": len(train_files),
            "valid": len(valid_files),
            "test": len(test_files),
            "multi_labeler_images": sum(len(ids) > 1 for ids in file_to_image_ids.values()),
        })
        """
    ),
    code(
        """
        def annotation_union_mask(filename):
            image_id = selected_id_by_file[filename]
            info = images_by_id[image_id]
            mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
            for annotation in annotations_by_id.get(image_id, []):
                segmentation = annotation.get("segmentation", [])
                if isinstance(segmentation, list):
                    for polygon in segmentation:
                        if len(polygon) < 6:
                            continue
                        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
                elif HAS_PYCOCOTOOLS and isinstance(segmentation, dict):
                    decoded = mask_utils.decode(segmentation)
                    mask = np.maximum(mask, decoded.astype(np.uint8))
            return mask

        def normalize_halpha(image):
            image = image.astype(np.float32)
            nonzero = image[image > 0]
            if nonzero.size:
                low, high = np.percentile(nonzero, [1.0, 99.5])
                image = np.clip((image - low) / max(high - low, 1.0), 0.0, 1.0)
            else:
                image.fill(0)
            return image

        class MAGFiLODataset(Dataset):
            def __init__(self, names, augment=False):
                self.names = list(names)
                self.augment = augment

            def __len__(self):
                return len(self.names)

            def __getitem__(self, index):
                filename = self.names[index]
                image = cv2.imread(os.path.join(TRAIN_DIR, filename), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(filename)
                mask = annotation_union_mask(filename)
                image = cv2.resize(image, INPUT_SIZE[::-1], interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, INPUT_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
                image = normalize_halpha(image)

                if self.augment:
                    k = random.randrange(4)
                    image = np.rot90(image, k).copy()
                    mask = np.rot90(mask, k).copy()
                    if random.random() < 0.5:
                        image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
                    if random.random() < 0.5:
                        image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
                    if random.random() < 0.5:
                        image = np.clip(image ** random.uniform(0.85, 1.20), 0, 1)

                image_tensor = torch.from_numpy(image).float().unsqueeze(0)
                mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
                return image_tensor, mask_tensor, filename

        train_loader = DataLoader(
            MAGFiLODataset(train_files, augment=True), batch_size=BATCH_SIZE,
            shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
        )
        valid_loader = DataLoader(
            MAGFiLODataset(valid_files, augment=False), batch_size=1,
            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
        )
        images_batch, masks_batch, names_batch = next(iter(train_loader))
        print({
            "image_shape": tuple(images_batch.shape),
            "mask_shape": tuple(masks_batch.shape),
            "foreground_fraction": float(masks_batch.mean()),
            "example": names_batch[0],
        })
        """
    ),
    code(model_source),
    code(
        """
        # The repository implementation is reused directly, with one H-alpha input
        # channel instead of repeating the same grayscale channel three times.
        model = MediSegUNet(in_channels=1, out_channels=1, bilinear=False).to(DEVICE)
        criterion = CombinedLoss(bce_weight=0.35, dice_weight=0.65)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        print({"parameters": sum(p.numel() for p in model.parameters())})

        def batch_dice(logits, targets, threshold=0.5):
            predictions = torch.sigmoid(logits) >= threshold
            targets = targets >= 0.5
            intersection = (predictions & targets).sum((1, 2, 3)).float()
            denominator = predictions.sum((1, 2, 3)) + targets.sum((1, 2, 3))
            return ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()

        @torch.no_grad()
        def validate_epoch():
            model.eval()
            losses, dices = [], []
            for images, masks, _ in valid_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = criterion(logits, masks)
                losses.append(float(loss))
                dices.append(float(batch_dice(logits, masks)))
            return float(np.mean(losses)), float(np.mean(dices))

        best_dice = -1.0
        stale_epochs = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = []
            progress = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
            for step, (images, masks, _) in enumerate(progress, 1):
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(images)
                    raw_loss = criterion(logits, masks)
                    loss = raw_loss / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                if step % ACCUMULATION_STEPS == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running.append(float(raw_loss.detach()))
                progress.set_postfix(loss=f"{np.mean(running[-20:]):.4f}")

            val_loss, val_dice = validate_epoch()
            scheduler.step()
            print({
                "epoch": epoch, "train_loss": float(np.mean(running)),
                "val_loss": val_loss, "val_dice_0.5": val_dice,
            })
            if val_dice > best_dice:
                best_dice = val_dice
                stale_epochs = 0
                torch.save(model.state_dict(), MODEL_PATH)
            else:
                stale_epochs += 1
                if stale_epochs >= PATIENCE:
                    print("Early stopping")
                    break

        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print({"best_val_dice_0.5": best_dice, "model": MODEL_PATH})
        """
    ),
    code(
        """
        # Select the probability threshold only from the held-out validation set.
        thresholds = np.arange(0.25, 0.76, 0.05)
        scores = {float(t): [] for t in thresholds}
        with torch.no_grad():
            for images, masks, _ in tqdm(valid_loader, desc="threshold calibration"):
                images = images.to(DEVICE)
                probabilities = torch.sigmoid(model(images))[0, 0].cpu().numpy()
                target = masks[0, 0].numpy().astype(bool)
                for threshold in thresholds:
                    prediction = probabilities >= threshold
                    intersection = np.logical_and(prediction, target).sum()
                    dice = (2 * intersection + 1e-6) / (prediction.sum() + target.sum() + 1e-6)
                    scores[float(threshold)].append(float(dice))
        mean_scores = {threshold: float(np.mean(values)) for threshold, values in scores.items()}
        BEST_THRESHOLD = max(mean_scores, key=mean_scores.get)
        print({"threshold_scores": mean_scores, "best_threshold": BEST_THRESHOLD})
        """
    ),
    code(
        """
        @torch.no_grad()
        def predict_probability(image_path):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_path)
            resized = cv2.resize(image, INPUT_SIZE[::-1], interpolation=cv2.INTER_AREA)
            normalized = normalize_halpha(resized)
            tensor = torch.from_numpy(normalized).float()[None, None].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                base = torch.sigmoid(model(tensor))
                horizontal = torch.flip(
                    torch.sigmoid(model(torch.flip(tensor, dims=[3]))), dims=[3]
                )
                vertical = torch.flip(
                    torch.sigmoid(model(torch.flip(tensor, dims=[2]))), dims=[2]
                )
                probability = (base + horizontal + vertical) / 3.0
            probability = probability[0, 0].float().cpu().numpy()
            return cv2.resize(
                probability, OUTPUT_SIZE[::-1], interpolation=cv2.INTER_LINEAR
            )

        def split_instances(probability, threshold):
            binary = (probability >= threshold).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
            instances = []
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < MIN_COMPONENT_AREA:
                    continue
                instance = (labels == label).astype(np.uint8)
                instances.append(instance)
            instances.sort(key=lambda mask: int(mask.sum()), reverse=True)
            return instances

        def compress_coco_counts(counts):
            chars = []
            for index, count in enumerate(counts):
                value = int(count)
                if index > 2:
                    value -= int(counts[index - 2])
                more = True
                while more:
                    char = value & 0x1F
                    value >>= 5
                    more = value != -1 if (char & 0x10) else value != 0
                    if more:
                        char |= 0x20
                    chars.append(chr(char + 48))
            return "".join(chars)

        def fallback_coco_encode(mask):
            pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
            changes = np.flatnonzero(np.diff(np.concatenate(([0], pixels, [0]))))
            counts, previous = [], 0
            for start, end in zip(changes[0::2], changes[1::2]):
                counts.extend([int(start - previous), int(end - start)])
                previous = int(end)
            if previous < pixels.size:
                counts.append(int(pixels.size - previous))
            return compress_coco_counts(counts)

        def encode_instance(mask):
            mask = np.asfortranarray(mask.astype(np.uint8))
            if HAS_PYCOCOTOOLS:
                return mask_utils.encode(mask)["counts"].decode("utf-8")
            return fallback_coco_encode(mask)
        """
    ),
    code(
        """
        rows = []
        instance_counts = {}
        for image_path in tqdm(test_files, desc="MediSeg inference"):
            probability = predict_probability(image_path)
            instances = split_instances(probability, BEST_THRESHOLD)
            image_id = image_path.stem
            instance_counts[image_id] = len(instances)
            for index, instance in enumerate(instances, 1):
                rows.append({
                    "filament_id": f"{image_id}_{index}",
                    "segmentation_rle": encode_instance(instance),
                })

        submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        if submission.empty:
            raise RuntimeError("No filament instances were predicted")
        submission.to_csv(OUTPUT_PATH, index=False, lineterminator="\\n")
        print({
            "output": OUTPUT_PATH,
            "rows": len(submission),
            "images_with_predictions": sum(value > 0 for value in instance_counts.values()),
            "test_images": len(test_files),
            "instances_per_image_median": float(np.median(list(instance_counts.values()))),
        })
        display(submission.head())
        """
    ),
    code(
        """
        checked = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
        assert checked.columns.tolist() == ["filament_id", "segmentation_rle"]
        assert not checked.empty
        assert not checked.isna().any().any()
        assert not checked["filament_id"].duplicated().any()
        assert not checked["segmentation_rle"].str.strip().eq("").any()
        test_ids = {path.stem for path in test_files}
        submitted_ids = {value.rsplit("_", 1)[0] for value in checked["filament_id"]}
        assert submitted_ids <= test_ids

        if HAS_PYCOCOTOOLS:
            areas = []
            for counts in tqdm(checked["segmentation_rle"], desc="official RLE validation"):
                decoded = mask_utils.decode({
                    "size": list(OUTPUT_SIZE), "counts": counts.encode("utf-8")
                })
                assert decoded.shape == OUTPUT_SIZE
                areas.append(int(decoded.sum()))
            assert min(areas) > 0
            print({
                "official_rle_validation": "passed",
                "area_min": min(areas),
                "area_median": int(np.median(areas)),
                "area_max": max(areas),
            })
        else:
            print("pycocotools unavailable: used embedded compatible COCO encoder")
        print("READY TO SUBMIT:", OUTPUT_PATH)
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(notebook, handle, ensure_ascii=False, indent=1)
    handle.write("\n")

print(OUTPUT)
