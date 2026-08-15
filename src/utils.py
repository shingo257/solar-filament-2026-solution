import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def rle_encode(mask: np.ndarray) -> str:
    """
    Convert a binary mask array to Kaggle Run-Length Encoding (RLE) string.
    mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle: str, shape: tuple) -> np.ndarray:
    """
    Decode Run-Length Encoding (RLE) string to binary mask.
    shape: (height, width)
    """
    if not isinstance(mask_rle, str) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def compute_dice_coeff(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6):
    """Compute 2D Dice Coefficient."""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    dice = (2.0 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)
    return dice.item()

def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6):
    """Compute 2D Intersection over Union (IoU / Jaccard)."""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    total = pred_bin.sum() + target.sum()
    union = total - intersection
    return ((intersection + smooth) / (union + smooth)).item()
