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

from __future__ import annotations

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
