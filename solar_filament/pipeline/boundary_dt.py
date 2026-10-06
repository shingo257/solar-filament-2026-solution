"""Distance-transform boundary loss — WINNING_STRATEGY_2026-08-29.md §6.12 優先度4。

現行 `refiner_loss` の境界項は morphological-gradient(dilate−erode)の L1 で、
V30 から不変の粗い定式化(README.md §2.1.12)。このモジュールはその代替として
Kervadec らの距離変換ベース境界損失(boundary loss)を提供する:

    L_boundary = mean( sigmoid(logits) * phi_G )

ここで phi_G は GT マスクの正規化済み符号付き距離写像で、GT 内部で負、
外部で正となる離散画素上の近似。損失を下げる方向は「内部で p→1、外部で p→0」で、
勾配の大きさが境界からの距離に比例する = 境界から遠い誤りほど強く罰する。

計算は numpy/cv2 のみ(GPU不要)なのでローカルでテスト可能
(`tests/test_boundary_dt.py`)。notebook 側(`build_v59_boundary_dt_pilot.py`
が生成)ではバッチ内各サンプルの augmented target から phi を作り、
既存の region 損失(0.35*BCE + 0.65*Dice)に `0.15 * boundary` として足す
= 既存 edge 項の重み0.15をそのまま引き継ぐ単変数置換。
"""

from __future__ import annotations

import cv2
import numpy as np


def signed_distance_map(mask):
    """二値マスクの正規化済み符号付き距離写像 phi_G を返す。

    Args:
        mask: (H, W) の二値マスク。有限な 0/1 を受け付ける。

    Returns:
        float32 の (H, W) 配列。GT 内部で負(最深部で最小)、外部で正
        (最遠部で最大)。離散画素の境界近傍では小さい非ゼロ値を持つ。
        絶対値の最大が 1 になるよう正規化する。
        空マスク・全面マスク(境界が存在しない)は全ゼロを返す
        (境界信号が定義できないため、損失への寄与をゼロにする)。
    """
    array = np.asarray(mask)
    if not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError("mask must contain finite binary values")
    binary = array.astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {binary.shape}")
    area = int(binary.sum())
    if area == 0 or area == binary.size:
        return np.zeros(binary.shape, dtype=np.float32)
    # cv2.distanceTransform は各非ゼロ画素について最近傍ゼロ画素への距離を返す。
    #   inside  = GT 内部で「境界までの距離」、外部で 0
    #   outside = GT 外部で「境界までの距離」、内部で 0
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    phi = outside - inside  # 内部で負、外部で正
    peak = float(np.abs(phi).max())
    if peak <= 0:
        return np.zeros(binary.shape, dtype=np.float32)
    return (phi / peak).astype(np.float32)


def batch_signed_distance_maps(targets):
    """(N, H, W) または (N, 1, H, W) のターゲット列から phi の列を作る。

    Returns:
        float32 の (N, H, W) 配列(targets が4次元なら (N, 1, H, W) の次元は落とす)。
    """
    array = np.asarray(targets)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError("targets must have exactly one channel")
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"targets must be (N,H,W) or (N,1,H,W), got shape {array.shape}")
    return np.stack([signed_distance_map(t) for t in array])


def boundary_term(probabilities, phi):
    """mean(p * phi_G) — 距離変換境界損失の本体(numpy評価用・テスト用)。

    学習時は torch 側で `(logits.sigmoid() * phi).mean()` として同じ式を評価する。
    完全な予測(p=GT)なら負値(最小方向)、空予測なら 0、GT外を立てるほど正に増える。
    """
    p = np.asarray(probabilities, dtype=np.float64)
    g = np.asarray(phi, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: probabilities {p.shape} vs phi {g.shape}")
    if not p.size or not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("probabilities and distances must be nonempty and finite")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be in [0, 1]")
    return float((p * g).mean())
