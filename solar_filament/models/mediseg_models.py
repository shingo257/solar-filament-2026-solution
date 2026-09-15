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

# ============================================================
# 4. Edge-Guided Attention U-Net (EG-MHSA)
# ============================================================
# 2026-08-29、WINNING_STRATEGY_2026-08-29.md §1.4のPhase 2実験用に追加。
# 参考文献: "EdgeAttNet: Towards Barb-Aware Filament Segmentation"
# (arXiv:2509.02964)。MAGFiLO上でU-Net(mIoU_pairwise 0.5724)から
# EdgeAttNet(0.6451)への改善を報告しており、barb(フィラメントの微細な
# 枝構造)の認識精度向上を狙ったアーキテクチャ。本プロジェクトの
# refiner(MediSegUNet)が抱える主要な弱点(near-miss境界品質)と直接
# 対応するため、置き換え候補として追試する。
#
# 【重要な注意】著者の公開リポジトリ(github.com/dasjar/EdgeAttNet)を
# 直接確認したところ、`model/edgeattnet_model.py`(本来UNetEdgeTransformer
# クラスが定義されているべきファイル)の中身が誤って`main.py`と同一に
# なっており、実際のアーキテクチャコードはリポジトリ上に存在しない
# (2026-08-29、`gh api`でリポジトリ全ファイルを走査して確認)。学習済み
# 重みの配布も確認できなかった。**以下は論文本文のアーキテクチャ記述
# (edge prior E=σ(Conv3x3(入力画像))をbottleneckへbilinear補間し
# 1x1convで射影、Q・Kへ加算、位置エンコーディングなし)のみを根拠にした
# 独自の再実装であり、著者オリジナル実装とのコード上の一致は未検証。**
# ベースにするU-Net本体はMediSegUNetをそのまま流用しているため、
# パラメータ総数は論文記載の値(22.6M)とは一致しない。


class EdgeGuidedMHSA(nn.Module):
    """Edge-Guided Multi-Head Self-Attention (EG-MHSA)。

    U-NetのbottleneckでSelf-Attentionを行う際、入力画像から抽出した
    エッジ事前分布をQuery/Keyに加算し、境界・barb構造への感度を高める。
    Value・出力チャネル数はbottleneckの特徴マップと同一に保つ
    (MediSegUNetへdrop-inできるよう設計)。
    """

    def __init__(self, in_channels: int, bottleneck_channels: int, num_heads: int = 8):
        super().__init__()
        if bottleneck_channels % num_heads != 0:
            raise ValueError(
                f"bottleneck_channels({bottleneck_channels}) must be divisible "
                f"by num_heads({num_heads})"
            )
        # E = sigmoid(Conv3x3(入力画像)): 単一チャンネルのエッジ事前分布
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        # E' = Conv1x1(interpolate(E)): bottleneckのチャンネル数へ射影
        self.edge_proj = nn.Conv2d(1, bottleneck_channels, kernel_size=1)
        self.mhsa = nn.MultiheadAttention(
            bottleneck_channels, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(bottleneck_channels)

    def forward(self, bottleneck: torch.Tensor, original_image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = bottleneck.shape
        edge_map = torch.sigmoid(self.edge_conv(original_image))
        edge_map = F.interpolate(edge_map, size=(h, w), mode="bilinear", align_corners=True)
        edge_prior = self.edge_proj(edge_map)  # (B, C, H, W)

        x_flat = bottleneck.flatten(2).transpose(1, 2)  # (B, N, C)
        edge_flat = edge_prior.flatten(2).transpose(1, 2)  # (B, N, C)
        qk = x_flat + edge_flat  # Q = K = x + E'。位置エンコーディングは使わない(論文通り)

        attn_out, _ = self.mhsa(qk, qk, x_flat)  # V = x(エッジ由来の加算なし)
        attn_out = self.norm(attn_out + x_flat)  # 残差接続 + LayerNorm(論文に明記はないが標準的なTransformerブロックとして採用)
        return attn_out.transpose(1, 2).reshape(b, c, h, w)


class MediSegEdgeAttUNet(nn.Module):
    """MediSegUNetのbottleneckにEdgeGuidedMHSAを挿入したdrop-in代替。

    入出力インターフェースはMediSegUNetと完全に同一(in_channels=3の
    crop、out_channels=1の確率マップ)。既存のrefiner学習・推論コードを
    変更せず、モデルクラスの差し替えだけで比較できるように設計している。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        bilinear: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        bottleneck_channels = 1024 // factor
        self.down4 = Down(512, bottleneck_channels)
        self.edge_attn = EdgeGuidedMHSA(in_channels, bottleneck_channels, num_heads=num_heads)
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
        x5 = self.edge_attn(x5, x)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        return self.outc(out)


# ============================================================
# 5. Plain MHSA U-Net (matched-parameter ablation control, no edge prior)
# ============================================================
# 2026-08-31、report/main.tex Future Work item 1(matched-parameter
# ablation)のために追加。EdgeGuidedMHSAからedge_conv/edge_projを除いた
# だけの構成——Q=K=V=bottleneck特徴量の標準的なself-attentionで、
# edge priorをQ/Kへ加算する項だけがない。edge_conv/edge_projは
# 3(in_channels)*1*3*3+1 + 1*1024*1*1+1024 ≈ 2,076パラメータしかなく
# 全体(約35.2M)の0.006%程度なので、このクラスはMediSegEdgeAttUNetと
# ほぼ完全にパラメータ数が一致する("U-Net + MHSA, no positional
# encoding"に相当、EdgeAttNet論文のablationと同じ切り分け)。
# report/main.texが指摘してきた「edge priorそのものの効果か、単に
# attention容量が増えただけか」という交絡を分離するための唯一の差分。


class PlainMHSA(nn.Module):
    """edge priorなしの標準Multi-Head Self-Attention(bottleneck上)。

    EdgeGuidedMHSAと同じchannel数・head数・残差接続+LayerNormを保ちつつ、
    Q・Kへの edge_prior 加算だけを行わない(Q=K=V=bottleneck特徴量)。
    """

    def __init__(self, bottleneck_channels: int, num_heads: int = 8):
        super().__init__()
        if bottleneck_channels % num_heads != 0:
            raise ValueError(
                f"bottleneck_channels({bottleneck_channels}) must be divisible "
                f"by num_heads({num_heads})"
            )
        self.mhsa = nn.MultiheadAttention(
            bottleneck_channels, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(bottleneck_channels)

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        b, c, h, w = bottleneck.shape
        x_flat = bottleneck.flatten(2).transpose(1, 2)  # (B, N, C)
        attn_out, _ = self.mhsa(x_flat, x_flat, x_flat)  # Q=K=V=x(edge priorなし)
        attn_out = self.norm(attn_out + x_flat)
        return attn_out.transpose(1, 2).reshape(b, c, h, w)


class MediSegPlainMHSAUNet(nn.Module):
    """MediSegEdgeAttUNetからedge priorだけを除いた matched-parameter 対照群。

    入出力インターフェースはMediSegUNet/MediSegEdgeAttUNetと完全に同一。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        bilinear: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        bottleneck_channels = 1024 // factor
        self.down4 = Down(512, bottleneck_channels)
        self.plain_attn = PlainMHSA(bottleneck_channels, num_heads=num_heads)
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
        x5 = self.plain_attn(x5)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        return self.outc(out)
