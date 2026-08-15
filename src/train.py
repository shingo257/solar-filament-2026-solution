import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import Config
from src.utils import seed_everything, compute_dice_coeff, compute_iou
from src.dataset import SolarFilamentDataset
from src.model import build_model
import numpy as np

class BCEDiceLoss(nn.Module):
    """Combination of BCE Loss and Dice Loss for Segmentation."""
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (pred_sigmoid.sum() + target.sum() + 1e-6)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        if use_amp and scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores, iou_scores = [], []

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()

        preds = torch.sigmoid(outputs)
        dice = compute_dice_coeff(preds, masks)
        iou = compute_iou(preds, masks)

        dice_scores.append(dice)
        iou_scores.append(iou)

    return running_loss / len(dataloader), float(np.mean(dice_scores)), float(np.mean(iou_scores))

def main():
    config = Config()
    seed_everything(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"=== Starting Solar Filament Training on Device: {config.DEVICE} ===")
    print(f"Model Architecture: {config.ARCHITECTURES} | Backbone: {config.BACKBONE}")

    # Generate Synthetic Sample Data for Local Test
    dummy_data = []
    for _ in range(16):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:300, 100:300] = 255
        dummy_data.append({"image": img, "mask": mask})

    train_loader = DataLoader(SolarFilamentDataset(dummy_data[:12], img_size=config.IMAGE_SIZE, is_train=True), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SolarFilamentDataset(dummy_data[12:], img_size=config.IMAGE_SIZE, is_train=False), batch_size=config.BATCH_SIZE, shuffle=False)

    model = build_model(config).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = BCEDiceLoss()
    scaler = torch.amp.GradScaler('cuda') if config.USE_AMP and torch.cuda.is_available() else None

    best_val_loss = float("inf")
    best_model_path = os.path.join(config.OUTPUT_DIR, "best_solar_unet.pth")

    for epoch in range(1, config.EPOCHS + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, config.DEVICE, config.USE_AMP)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, config.DEVICE)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{config.EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | Time: {elapsed:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved Best Model checkpoint to {best_model_path}")

    print("=== Training Complete! ===")

if __name__ == "__main__":
    main()
