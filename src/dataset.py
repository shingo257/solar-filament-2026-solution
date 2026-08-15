import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

def get_transforms(img_size=(512, 512), is_train=True):
    """Albumentations 2D Segmentation Transformations with fallback."""
    if HAS_ALBUMENTATIONS:
        if is_train:
            return A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    return None

class SolarFilamentDataset(Dataset):
    """
    2D Dataset for Solar Filament Segmentation.
    Handles image paths / mask paths or inline numpy arrays.
    """
    def __init__(self, data_list, img_size=(512, 512), is_train=True):
        self.data_list = data_list
        self.transform = get_transforms(img_size, is_train=is_train)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        if isinstance(item, dict):
            img_path = item.get("image_path", None)
            mask_path = item.get("mask_path", None)
            
            if img_path and os.path.exists(img_path):
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = item.get("image", np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = item.get("mask", np.zeros((512, 512), dtype=np.uint8))
        else:
            image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            mask = np.zeros((512, 512), dtype=np.uint8)

        # Apply Albumentations Transformation or Pure Numpy Fallback
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].unsqueeze(0).float() / 255.0
        else:
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return image_tensor, mask_tensor
