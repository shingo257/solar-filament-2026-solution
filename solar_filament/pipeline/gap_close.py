"""Skeleton-based gap closure — WINNING_STRATEGY_2026-08-29.md §6.12 優先度3。

watershed(`blob_split.py`、過剰結合したマスクの分離、不採用)とは逆方向の
後処理: refiner適用後に一つの候補マスクが小さな途切れで複数の連結成分へ
分断されている場合、スケルトンの端点同士が近ければ橋渡しして単一マスクに
再結合する。診断済みの主要失敗モード(fragmentation、細い構造の途切れ)に
対する、まだ試していない軸。

GPU学習を伴わない古典的画像処理なので、ローカルでテスト可能
(`tests/test_gap_close.py`)。実際のPQへの影響はColab上のOOF評価で確認する
(`scratch/build_v56_colab_gap_closure_probe.py`)。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _skeleton_endpoints(component_mask: np.ndarray) -> list[tuple[int, int]]:
    """スケルトンの端点(近傍スケルトン画素が1個以下)の座標一覧を返す。

    近傍0個(孤立点、小さな塊がそのままスケルトン化された場合)も端点として扱う。
    """
    skeleton = skeletonize(component_mask)
    if not skeleton.any():
        return []
    skeleton_u8 = skeleton.astype(np.uint8)
    neighbor_count = (
        ndimage.convolve(skeleton_u8, np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0)
        - skeleton_u8
    )
    ys, xs = np.where(skeleton)
    degrees = neighbor_count[ys, xs]
    endpoint = degrees <= 1
    return list(zip(ys[endpoint].tolist(), xs[endpoint].tolist()))


def _draw_bridge(mask: np.ndarray, point_a: tuple[int, int], point_b: tuple[int, int], width: int) -> np.ndarray:
    y0, x0 = point_a
    y1, x1 = point_b
    length = int(round(float(np.hypot(y1 - y0, x1 - x0)))) + 1
    ys = np.linspace(y0, y1, length).round().astype(int)
    xs = np.linspace(x0, x1, length).round().astype(int)
    bridged = mask.copy()
    radius = width // 2
    height, mask_width = mask.shape
    for y, x in zip(ys, xs):
        y_lo, y_hi = max(0, y - radius), min(height, y + radius + 1)
        x_lo, x_hi = max(0, x - radius), min(mask_width, x + radius + 1)
        bridged[y_lo:y_hi, x_lo:x_hi] = True
    return bridged


def skeleton_gap_close(mask: np.ndarray, max_gap: float = 8.0, bridge_width: int = 3) -> np.ndarray:
    """1つの候補マスク内で分断された連結成分を、スケルトン端点の近接ペアごとに橋渡しする。

    Args:
        mask: 2値マスク(bool)。1つのYOLO候補proposalに対するrefiner適用後の結果を想定。
        max_gap: 橋渡しを許す最大距離(画素)。これを超えるペアは別の実体とみなし結合しない。
        bridge_width: 橋渡しする線の太さ(画素)。

    Returns:
        橋渡し後のマスク(元のmaskとの論理和、追加分のみ増える)。連結成分が1個以下なら
        元のmaskをそのまま返す(コピー)。
    """
    labeled, n_components = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n_components <= 1:
        return mask.copy()

    endpoints_by_component = [
        _skeleton_endpoints(labeled == label_id) for label_id in range(1, n_components + 1)
    ]

    bridged = mask.copy()
    for i in range(len(endpoints_by_component)):
        for j in range(i + 1, len(endpoints_by_component)):
            best_pair = None
            best_distance = max_gap
            for point_i in endpoints_by_component[i]:
                for point_j in endpoints_by_component[j]:
                    distance = float(np.hypot(point_i[0] - point_j[0], point_i[1] - point_j[1]))
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (point_i, point_j)
            if best_pair is not None:
                bridged = _draw_bridge(bridged, best_pair[0], best_pair[1], bridge_width)
    return bridged
