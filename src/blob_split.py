"""距離変換+watershedで、癒着した複数個体のmaskを分離する。

近接する2本のフィラメントをYOLOが1つの塊として検出すると、その塊は
一方のGTに対してovercoverage(面積が大きすぎる)、もう一方のGTに対しては
検出漏れ(undercoverage/hard FN)になる。near-miss被覆モード診断(改訂7)で
under/overがほぼ拮抗していたのはこの「癒着」が主因という仮説を検証する。

GTを使わず、mask形状だけから分離候補を作る。過剰分割を避けるため、
ピークの分離度(距離変換値の谷/山比)が十分に低いときだけ分割する。
"""

from __future__ import annotations

import cv2
import numpy as np


def split_touching_blob(
    mask: np.ndarray,
    core_fraction: float = 0.8,
    min_component_area: int = 80,
    min_core_area: int = 60,
) -> list[np.ndarray]:
    """1つのboolマスクを、癒着が疑われる場合に複数のboolマスクへ分割する。

    距離変換を `core_fraction * 局所最大` で大域的に閾値化し、太い「胴体」だけを
    core領域として残す。フィラメントのように幅がほぼ一定な細長い形状は、
    core領域も1本の連結成分のままになるため分割されない。2個体が細い橋で
    繋がっている場合だけ、橋の部分がcore閾値を割り込み、core領域が2つに
    分かれる。この2つをwatershedのマーカーにして元のmask全体へ広げる。

    局所最大点をそのままマーカーにする単純なpeak-pickingは、湾曲した
    細長い形状の稜線上に複数の極大点を誤検出し過分割する(実測で確認済み)。
    大域閾値によるcore連結成分ベースの本実装はこれを避ける。

    分割不要(単一の塊、あるいは既に閾値以下)なら [mask] をそのまま返す。
    """
    if mask.sum() < min_component_area * 2:
        return [mask]

    mask_u8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if dist.max() < 3.0:
        return [mask]

    # ローカル最大値ではなく大域閾値でcore領域を作る(過分割防止の要)
    core = (dist >= core_fraction * dist.max()).astype(np.uint8)
    n_cores, core_labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    valid_cores = [i for i in range(1, n_cores) if stats[i, cv2.CC_STAT_AREA] >= min_core_area]
    if len(valid_cores) < 2:
        return [mask]

    # watershed: 背景=1, 各coreの塊=2..n+1, 未確定領域=0
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[mask_u8 == 0] = 1
    for rank, core_id in enumerate(valid_cores, start=2):
        markers[core_labels == core_id] = rank

    color = cv2.cvtColor(mask_u8 * 255, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)

    pieces = []
    for rank in range(2, len(valid_cores) + 2):
        piece = (markers == rank) & mask
        if int(piece.sum()) >= min_component_area:
            pieces.append(piece)
    if len(pieces) < 2:
        return [mask]
    return pieces


def split_proposals(
    masks: list[np.ndarray],
    scores: np.ndarray,
    **split_kwargs,
) -> tuple[list[np.ndarray], np.ndarray]:
    """複数proposalへsplit_touching_blobを適用し、スコアは分割元をそのまま複製する。"""
    out_masks: list[np.ndarray] = []
    out_scores: list[float] = []
    for mask, score in zip(masks, scores):
        pieces = split_touching_blob(mask, **split_kwargs)
        out_masks.extend(pieces)
        out_scores.extend([float(score)] * len(pieces))
    return out_masks, np.asarray(out_scores, dtype=np.float32)
