# [2/5] MediSeg UNetアーキテクチャ・WBF(solar_filament/と同一)
"""
MediSeg-AI Knowledge Port for Solar Filament Segmentation
Includes:
1. MediSeg UNet
2. MediSeg DeepLabV3Plus with ASPP (Atrous Spatial Pyramid Pooling)
3. CombinedLoss (BCE + Dice Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ============================================================
# 1. Combined Loss (BCE + Dice Loss)
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# ============================================================
# 2. MediSeg UNet Architecture
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class MediSegUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, bilinear: bool = False):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# ============================================================
# 3. MediSeg DeepLabV3Plus with ASPP Module
# ============================================================
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn3x3_1 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn3x3_2 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn3x3_3 = nn.BatchNorm2d(out_channels)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_global = nn.BatchNorm2d(out_channels)
        
        self.conv_final = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x1 = self.relu(self.bn1x1(self.conv1x1(x)))
        x2 = self.relu(self.bn3x3_1(self.conv3x3_1(x)))
        x3 = self.relu(self.bn3x3_2(self.conv3x3_2(x)))
        x4 = self.relu(self.bn3x3_3(self.conv3x3_3(x)))
        
        x5 = self.global_avg_pool(x)
        x5 = self.relu(self.bn_global(self.conv_global(x5)))
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.relu(self.bn_final(self.conv_final(x_cat)))

class MediSegDeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Feature Extractor
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(512, 256)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        feat = self.stem(x)
        aspp_feat = self.aspp(feat)
        out = self.classifier(aspp_feat)
        return F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)


"""複数seed(独立に学習した検出器)のmask提案を統合するWeighted Mask Fusion。

## 動機

multi-seedアンサンブル(V42)は単純連結(全proposalをそのままconcat)で
TP+53よりFP+190の方が大きく、正味で悪化した(改訂9 §27.5)。それぞれの
seedに固有の誤検出が、確定後処理のdrop_contained(IoU≥0.90の重複除去のみ)
では吸収しきれなかったため。

WBF(Weighted Boxes Fusion、本来はbox用)のmask版。IoUでクラスタ化し、
クラスタ内のmaskを confidence 加重平均→0.5で二値化して1個の統合maskにする。
2つのseedが同じ個体を検出した場合はconfidence加重平均で滑らかに統合され、
片方のseedだけの誤検出は単独クラスタのまま残る(スコアで後段のconf閾値に
掛けられる)ので、単純連結よりFPを抑えられるという仮説を検証する。
"""


import numpy as np
from pycocotools import mask as mask_util


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union else 0.0


def weighted_mask_fusion(
    masks: list[np.ndarray],
    scores: np.ndarray,
    source_ids: list[int] | None = None,
    iou_threshold: float = 0.5,
    vote_threshold: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    """複数seed分のmask提案をIoUクラスタ化し、加重平均で統合する。

    Args:
        masks: 統合対象のboolマスク(全seed分をあらかじめ結合したもの)。
        scores: 各マスクのconfidence。
        source_ids: 各マスクの由来seed(任意)。同一クラスタ内の由来seed数を
            数えるためだけに使う(合意度の記録用、フュージョン計算自体には未使用)。
        iou_threshold: この値以上のIoUを同一個体とみなしてクラスタ化する。
            drop_containedの0.90より緩い値を想定(異なるseedの検出は
            境界が微妙にずれるため、厳しすぎると同一個体でも別クラスタになる)。
        vote_threshold: confidence加重平均マスクをこの閾値で二値化する。
            0.5より低いと統合maskが大きくなる(overcoverage方向)、
            高いと小さくなる(undercoverage方向)。

    Returns:
        (統合後のmaskリスト, 統合後のscore配列, 各クラスタの由来seed数リスト)
    """
    if not masks:
        return [], np.zeros(0, np.float32), []
    if source_ids is None:
        source_ids = [0] * len(masks)

    order = list(np.argsort(-scores))
    clusters: list[dict] = []  # {"rep": mask, "members": [idx,...]}

    for index in order:
        mask = masks[index]
        best_cluster = None
        best_iou = iou_threshold
        for cluster in clusters:
            iou = _mask_iou(mask, cluster["rep"])
            if iou >= best_iou:
                best_cluster = cluster
                best_iou = iou
        if best_cluster is None:
            clusters.append({"rep": mask, "members": [index]})
        else:
            best_cluster["members"].append(index)
            # 代表maskは最高スコアのまま固定(orderがスコア降順なので先頭が最高)

    fused_masks: list[np.ndarray] = []
    fused_scores: list[float] = []
    fused_source_counts: list[int] = []
    for cluster in clusters:
        members = cluster["members"]
        member_scores = np.array([scores[i] for i in members], dtype=np.float64)
        weights = member_scores / member_scores.sum()
        accum = np.zeros(masks[members[0]].shape, dtype=np.float64)
        for member_index, weight in zip(members, weights):
            accum += masks[member_index].astype(np.float64) * weight
        fused_mask = accum >= vote_threshold
        if fused_mask.sum() == 0:
            continue
        fused_masks.append(fused_mask)
        fused_scores.append(float(member_scores.mean()))
        fused_source_counts.append(len({source_ids[i] for i in members}))

    return fused_masks, np.asarray(fused_scores, dtype=np.float32), fused_source_counts

def soft_dice_loss(logits, targets):
    probs = logits.sigmoid()
    intersection = (probs * targets).sum((1, 2, 3))
    denominator = probs.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()

def imread_unicode(path):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
