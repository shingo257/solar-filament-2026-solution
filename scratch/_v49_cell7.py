# [7/8] MediSegUNet・公式PQ(solar_filament/と同一)
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


"""公式評価指標のローカル実装 (Solar Filament Segmentation Challenge 2026).

## リーダーボード指標 = Panoptic Quality (PQ) 【確認済み】

2026-08-07 に評価方法が変更され、2026-08-12 にリーダーボードが再採点された。
それ以前は Dice (torchmetrics.segmentation.DiceScore) だった。現行は PQ。

    PQ = sum(IoU of TP pairs) / (|TP| + 0.5*|FP| + 0.5*|FN|),  TP は IoU > 0.5

出典は主催者が公開した自己評価ノートブック (Apache-2.0):
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
本モジュールの PQ 実装は、その `get_pq_score` / `fp_count_hit` / `fn_count_hit`
の挙動をそのまま再現する。原典は Kirillov et al., CVPR 2019 (10.1109/CVPR.2019.00963)
だが、**公式実装は原典そのままではない**ので、原典ではなく公式実装に合わせること。

公式実装で特に効く仕様:

  1. **IoU > 0.5 は硬い崖。** IoU 0.5 は Dice 0.667 に相当する。これを下回る予測は
     得点に一切寄与せず、FP として分母を 0.5 増やし、対応する GT も FN として
     さらに 0.5 増やす。惜しい予測は「無得点」ではなく「二重の減点」になる。
  2. **評価単位は (annotator, image) ペア。** GT の filament_id は
     `<annotator_id>-<image_id>_<n>` 形式で、`annotator_image` ごとに 1 行を作り、
     同一の予測集合を各アノテータの GT と個別に突き合わせて、TP/FP/FN を
     全体へ足し込む。複数アノテータ画像は事実上その人数ぶん重み付けされる。
     variant ごとにスコアを出して平均するのではない (V30/V31 の扱いとは別物)。
  3. **分母は全画像を通した micro 集計。** 画像単位のマクロ平均ではないので、
     個体数の多い画像が支配的になる。
  4. **FP は「どの GT とも IoU > 0.5 にならない予測」。** アノテータごとに数える。

## 副次指標

MIoU (Multiscale IoU) も実装してあるが、**リーダーボード指標ではない**。
IEEE Big Data Cup の紹介ページには MIoU が挙がっているものの、Kaggle の
Evaluation は PQ と Dice/IoU 分布・一対多/多対一分布で構成される。
MIoU は細構造 (barb) の再現性を測る補助指標、および技術報告書 (評価の 30% を
占める定性評価) の裏付けとして使う位置づけに留める。

  - MIoU: Ahmadzadeh, Kempton, Chen, Angryk, ICIP 2021 (arXiv:2105.14572)
    参照実装 https://bitbucket.org/gsudmlab/multiscale_iou/
  - EdgeAttNet (arXiv:2509.02964) は MAGFiLO test split で
    mIoU_pairwise 0.6451 / mIoU_multiscale 0.7032 を報告

## 過去数値との接続

V19〜V34 が使っていた Hungarian matched Dice は PQ ではない。normalization も
(max(|GT|,|pred|) vs TP+0.5FP+0.5FN)、通貨も (Dice vs IoU)、閾値の有無も違う。
過去の OOF 数値と本モジュールの PQ を直接比較しないこと。接続用に
hungarian_dice() を残してあるので、必ず指標名を明示して併記する。
"""


import numpy as np
from scipy.optimize import linear_sum_assignment

# Δ = {2^n : n < 10} = {1, 2, 4, ..., 512} (原論文 4章の実験設定)
DEFAULT_DELTAS: tuple[int, ...] = tuple(2 ** n for n in range(10))

# 原論文は contour を推奨。EdgeAttNet は mask と記述。既定は原論文に合わせる。
USE_CONTOUR_DEFAULT = True

METRIC_VERSION = "official-v1-2026-08-15"


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def contour_of(mask: np.ndarray) -> np.ndarray:
    """8近傍 erosion との差分で 1px の輪郭を取り出す。

    cv2/scipy に依存せず numpy のシフトだけで計算する。空 mask は空 contour を返す。
    """
    mask = _as_bool(mask)
    if not mask.any():
        return mask
    eroded = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
            shifted[yd, xd] = mask[ys, xs]
            eroded &= shifted
    return mask & ~eroded


def cell_reduce(mask: np.ndarray, delta: int) -> np.ndarray:
    """s(o, delta): delta x delta のセルに 1 つでも前景があれば 1 とする粗視化。"""
    mask = _as_bool(mask)
    if delta <= 1:
        return mask
    height, width = mask.shape
    padded_h = -(-height // delta) * delta
    padded_w = -(-width // delta) * delta
    if padded_h != height or padded_w != width:
        padded = np.zeros((padded_h, padded_w), dtype=bool)
        padded[:height, :width] = mask
        mask = padded
    return mask.reshape(padded_h // delta, delta, padded_w // delta, delta).any(axis=(1, 3))


def cell_keys(
    mask: np.ndarray,
    deltas: tuple[int, ...] = DEFAULT_DELTAS,
    use_contour: bool = USE_CONTOUR_DEFAULT,
) -> list[np.ndarray]:
    """mask を「各 delta で占有するセルの一意キー配列」へ前計算する。

    s(o, delta) を密な 2 次元配列ではなく座標集合として持つ。輪郭画素は画像
    全体に比べて極端に疎なので、計算量が画像面積 O(H*W) から輪郭長 O(P) に落ちる。
    2048x2048 の総当たり評価が現実的な時間で終わるのはこの前計算のおかげであり、
    V34 の OOF 評価が 524/707 で停止した経路を避けるための中核。

    グリッドは画像原点にそろえたままなので、密な cell_reduce と厳密に一致する。
    """
    mask = _as_bool(mask)
    if use_contour:
        mask = contour_of(mask)
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    keys = []
    for delta in deltas:
        if ys.size == 0:
            keys.append(np.empty(0, dtype=np.int64))
            continue
        cells_per_row = -(-width // delta)
        flat = (ys // delta).astype(np.int64) * cells_per_row + (xs // delta)
        keys.append(np.unique(flat))
    return keys


def multiscale_iou_from_keys(
    truth_keys: list[np.ndarray],
    prediction_keys: list[np.ndarray],
) -> float:
    """cell_keys() の出力どうしで MIoU を計算する。"""
    ratios = []
    for truth_cells, prediction_cells in zip(truth_keys, prediction_keys):
        if truth_cells.size == 0:
            ratios.append(0.0)
            continue
        shared = np.intersect1d(truth_cells, prediction_cells, assume_unique=True).size
        ratios.append(float(shared) / truth_cells.size)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def multiscale_iou(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    deltas: tuple[int, ...] = DEFAULT_DELTAS,
    use_contour: bool = USE_CONTOUR_DEFAULT,
) -> float:
    """MIoU(o, o~) = mean_delta r(o, o~, delta).

    r(o, o~, delta) = n(s(o, delta) & s(o~, delta)) / n(s(o, delta))

    非対称であることに注意: 分母は ground truth 側だけで決まる。予測が過剰に
    大きいことは r では直接罰せられない。precision 側は別指標で見ること。

    多数のペアを評価するときは cell_keys() で前計算して
    multiscale_iou_from_keys() を使うこと。こちらは輪郭を毎回作り直す。
    """
    truth = _as_bool(ground_truth)
    if use_contour:
        if not contour_of(truth).any():
            return 0.0
    elif not truth.any():
        return 0.0
    return multiscale_iou_from_keys(
        cell_keys(ground_truth, deltas, use_contour),
        cell_keys(prediction, deltas, use_contour),
    )


def iou(first: np.ndarray, second: np.ndarray) -> float:
    first = _as_bool(first)
    second = _as_bool(second)
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(first, second).sum()) / union


def dice(first: np.ndarray, second: np.ndarray) -> float:
    first = _as_bool(first)
    second = _as_bool(second)
    denominator = int(first.sum()) + int(second.sum())
    if denominator == 0:
        return 0.0
    return 2.0 * float(np.logical_and(first, second).sum()) / denominator


def pairwise_iou(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    zero_intersection: str = "exclude",
) -> float:
    """交差を持つ (gt, pred) ペアの IoU 平均 (EdgeAttNet の IoU_pairwise)。

    zero_intersection="exclude" は本文記述 (交差のあるペアのみ評価) に、
    "zero" は式の otherwise=0 に対応する。検出漏れ・過剰検出は
    この指標では罰せられないので、必ず hit/miss rate と併記すること。
    """
    if not gt_masks or not pred_masks:
        return 0.0
    scores = []
    for truth in gt_masks:
        truth = _as_bool(truth)
        for prediction in pred_masks:
            prediction = _as_bool(prediction)
            if np.logical_and(truth, prediction).any():
                scores.append(iou(truth, prediction))
            elif zero_intersection == "zero":
                scores.append(0.0)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def matched_multiscale_iou(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    deltas: tuple[int, ...] = DEFAULT_DELTAS,
    use_contour: bool = USE_CONTOUR_DEFAULT,
    unmatched_penalty: bool = True,
) -> float:
    """個体を Hungarian で対応付けてから MIoU を平均する。

    unmatched_penalty=True では max(len(gt), len(pred)) で割るため、検出漏れと
    過剰検出の両方が罰せられる。公式が対応付けをどう行うかは未確認なので、
    これは「MIoU を instance 指標へ持ち上げた自前の変種」であることを忘れないこと。
    """
    if not gt_masks and not pred_masks:
        return 1.0
    if not gt_masks or not pred_masks:
        return 0.0
    truth_keys = [cell_keys(m, deltas, use_contour) for m in gt_masks]
    prediction_keys = [cell_keys(m, deltas, use_contour) for m in pred_masks]
    matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, truth in enumerate(truth_keys):
        for j, prediction in enumerate(prediction_keys):
            matrix[i, j] = multiscale_iou_from_keys(truth, prediction)
    rows, columns = linear_sum_assignment(-matrix)
    total = float(matrix[rows, columns].sum())
    divisor = max(len(gt_masks), len(pred_masks)) if unmatched_penalty else len(rows)
    return total / max(divisor, 1)


def hit_miss_rate(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """(hit_rate, miss_rate, false_positive_rate) を IoU 閾値ベースで返す。

    hit  : 閾値以上の予測を 1 つ以上持つ GT の割合 (recall)
    miss : 1 - hit
    fp   : どの GT とも閾値以上で対応しない予測の割合
    """
    if not gt_masks:
        return 0.0, 0.0, 1.0 if pred_masks else 0.0
    if not pred_masks:
        return 0.0, 1.0, 0.0
    matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, truth in enumerate(gt_masks):
        for j, prediction in enumerate(pred_masks):
            matrix[i, j] = iou(truth, prediction)
    hits = float((matrix.max(axis=1) >= threshold).mean())
    false_positives = float((matrix.max(axis=0) < threshold).mean())
    return hits, 1.0 - hits, false_positives


def hungarian_dice(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> float:
    """V31/V34 と同一の strict instance Dice。過去数値との接続用にのみ使う。"""
    if not gt_masks and not pred_masks:
        return 1.0
    if not gt_masks or not pred_masks:
        return 0.0
    matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, truth in enumerate(gt_masks):
        for j, prediction in enumerate(pred_masks):
            matrix[i, j] = dice(truth, prediction)
    rows, columns = linear_sum_assignment(-matrix)
    return float(matrix[rows, columns].sum() / max(len(gt_masks), len(pred_masks)))


def fragmentation_counts(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    overlap_fraction: float = 0.10,
) -> tuple[int, int]:
    """(splits, merges)。V34 の実装と同一定義。"""
    splits = 0
    for truth in gt_masks:
        truth = _as_bool(truth)
        area = max(int(truth.sum()), 1)
        matches = sum(
            int(np.logical_and(truth, _as_bool(p)).sum()) / area >= overlap_fraction
            for p in pred_masks
        )
        splits += max(matches - 1, 0)
    merges = 0
    for prediction in pred_masks:
        prediction = _as_bool(prediction)
        area = max(int(prediction.sum()), 1)
        matches = sum(
            int(np.logical_and(_as_bool(t), prediction).sum()) / area >= overlap_fraction
            for t in gt_masks
        )
        merges += max(matches - 1, 0)
    return splits, merges


# --------------------------------------------------------------------------
# Panoptic Quality — リーダーボード指標
# --------------------------------------------------------------------------

PQ_IOU_THRESHOLD = 0.5


def iou_matrix_of(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    """公式 get_overlap_matrices と同じ (n_gt, n_pred) の IoU 行列。"""
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float64)
    gt_flat = np.stack([_as_bool(m).ravel() for m in gt_masks]).astype(np.float64)
    pred_flat = np.stack([_as_bool(m).ravel() for m in pred_masks]).astype(np.float64)
    intersection = gt_flat @ pred_flat.T
    union = gt_flat.sum(1)[:, None] + pred_flat.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class PanopticQuality:
    """公式 get_pq_score をそのまま再現する累積器。

    公式実装は (annotator, image) ごとに 1 回 add() され、TP/FP/FN が全体へ
    足し込まれる。画像単位のマクロ平均ではないので、必ず全 annotator-image を
    同じ累積器へ入れてから score を読むこと。
    """

    def __init__(self, iou_threshold: float = PQ_IOU_THRESHOLD):
        self.iou_threshold = iou_threshold
        self.tp_ious: list[float] = []
        self.fp_count = 0
        self.fn_count = 0

    def add(self, gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> dict:
        """1 つの annotator-image を加算し、その行の内訳を返す。"""
        n_gt, n_pred = len(gt_masks), len(pred_masks)
        if n_gt == 0:
            self.fp_count += n_pred
            return {"n_gt": 0, "n_pred": n_pred, "tp": 0, "fp": n_pred, "fn": 0, "tp_iou_sum": 0.0}
        if n_pred == 0:
            self.fn_count += n_gt
            return {"n_gt": n_gt, "n_pred": 0, "tp": 0, "fp": 0, "fn": n_gt, "tp_iou_sum": 0.0}
        return self.add_matrix(iou_matrix_of(gt_masks, pred_masks), n_gt, n_pred)

    def add_matrix(self, matrix: np.ndarray, n_gt: int, n_pred: int) -> dict:
        """IoU 行列を計算済みの場合の加算経路。"""
        if n_gt == 0:
            self.fp_count += n_pred
            return {"n_gt": 0, "n_pred": n_pred, "tp": 0, "fp": n_pred, "fn": 0, "tp_iou_sum": 0.0}
        if n_pred == 0:
            self.fn_count += n_gt
            return {"n_gt": n_gt, "n_pred": 0, "tp": 0, "fp": 0, "fn": n_gt, "tp_iou_sum": 0.0}
        hits = matrix > self.iou_threshold
        scores = matrix[hits].tolist()
        false_positives = int((hits.sum(axis=0) == 0).sum())
        false_negatives = int((hits.sum(axis=1) == 0).sum())
        self.tp_ious.extend(scores)
        self.fp_count += false_positives
        self.fn_count += false_negatives
        return {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "tp": len(scores),
            "fp": false_positives,
            "fn": false_negatives,
            "tp_iou_sum": float(sum(scores)),
        }

    @property
    def denominator(self) -> float:
        return len(self.tp_ious) + 0.5 * self.fp_count + 0.5 * self.fn_count

    @property
    def score(self) -> float:
        denominator = self.denominator
        return float(sum(self.tp_ious) / denominator) if denominator > 0 else 0.0

    def summary(self) -> dict:
        """PQ と、その内訳 (SQ x RQ)。どこが効いているかを分離して見るため。

        SQ = TP の平均 IoU (形の質)、RQ = TP/(TP+0.5FP+0.5FN) (検出の質)。
        PQ = SQ * RQ。0.35 が「形が悪い」のか「取りこぼしている」のかを区別する。
        """
        tp = len(self.tp_ious)
        segmentation_quality = float(np.mean(self.tp_ious)) if tp else 0.0
        denominator = self.denominator
        recognition_quality = float(tp / denominator) if denominator > 0 else 0.0
        return {
            "pq": self.score,
            "sq": segmentation_quality,
            "rq": recognition_quality,
            "tp": tp,
            "fp": self.fp_count,
            "fn": self.fn_count,
            "denominator": denominator,
            "iou_threshold": self.iou_threshold,
            "metric_version": METRIC_VERSION,
        }


def marginal_prediction_threshold(current_pq: float, expected_tp_iou: float = 0.7) -> float:
    """候補を 1 つ追加してよい最小の TP 確率。

    PQ = N/D のとき、TP 確率 p・想定 IoU u の候補を足すと期待 PQ は
    (N + p*u)/(D + 0.5)。これが N/D を上回る条件は p*u > 0.5*PQ。
    つまり **p > 0.5*PQ/u**。PQ=0.35, u=0.7 なら p > 0.25 で、
    「4 回に 1 回しか当たらない候補でも出した方が得」になる。

    Hungarian Dice を基準に conf を絞ってきた運用とは最適点が大きく違うので、
    conf 閾値は必ずこの式を目安に PQ 上で測り直すこと。
    """
    if expected_tp_iou <= 0:
        return 1.0
    return float(min(1.0, 0.5 * current_pq / expected_tp_iou))


def evaluate_image(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    deltas: tuple[int, ...] = DEFAULT_DELTAS,
    use_contour: bool = USE_CONTOUR_DEFAULT,
    iou_threshold: float = PQ_IOU_THRESHOLD,
    with_miou: bool = False,
) -> dict:
    """1 annotator-image 分の診断値。PQ 集計は別途 PanopticQuality で行うこと。

    with_miou=False が既定。MIoU はリーダーボード指標ではなく、全 OOF で回すと
    高価なので、必要なときだけ有効にする。
    """
    hit, miss, false_positive = hit_miss_rate(gt_masks, pred_masks, iou_threshold)
    splits, merges = fragmentation_counts(gt_masks, pred_masks)
    matrix = iou_matrix_of(gt_masks, pred_masks)
    near_miss = int(((matrix > 0.3) & (matrix <= iou_threshold)).any(axis=0).sum()) if matrix.size else 0
    record = {
        "metric_version": METRIC_VERSION,
        "gt_count": len(gt_masks),
        "pred_count": len(pred_masks),
        "count_error": abs(len(gt_masks) - len(pred_masks)),
        "best_iou_mean": float(matrix.max(axis=1).mean()) if matrix.size else 0.0,
        # IoU 0.3〜0.5 の「あと少しで TP だった」予測。ここが厚いなら
        # 検出は当たっていて境界品質が崖を越えられていない、という診断になる。
        "near_miss_predictions": near_miss,
        "pairwise_iou": pairwise_iou(gt_masks, pred_masks),
        "hungarian_dice": hungarian_dice(gt_masks, pred_masks),
        "hit_rate": hit,
        "miss_rate": miss,
        "false_positive_rate": false_positive,
        "splits": splits,
        "merges": merges,
    }
    if with_miou:
        record["matched_miou"] = matched_multiscale_iou(gt_masks, pred_masks, deltas, use_contour)
    return record
