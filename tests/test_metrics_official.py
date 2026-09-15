"""metrics_official の性質テスト。

pytest でも `python tests/test_metrics_official.py` でも走る。
「Dice では見えないが MIoU では罰せられる」ことを実測で固定するのが目的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_filament.metrics.metrics_official import (  # noqa: E402
    DEFAULT_DELTAS,
    PanopticQuality,
    cell_reduce,
    contour_of,
    dice,
    hit_miss_rate,
    hungarian_dice,
    iou_matrix_of,
    marginal_prediction_threshold,
    matched_multiscale_iou,
    multiscale_iou,
    pairwise_iou,
)

SIZE = 2048


def _shifted(mask: np.ndarray, dy: int) -> np.ndarray:
    return np.roll(mask, dy, axis=0)


def _spine(with_barbs: bool) -> np.ndarray:
    """細長い spine と、そこから出る barb を持つフィラメント様の mask。"""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[1000:1005, 800:1200] = True  # spine: 400 x 5
    if with_barbs:
        for i, x in enumerate(range(820, 1180, 45)):
            if i % 2 == 0:
                mask[980:1000, x:x + 3] = True  # 上向き barb
            else:
                mask[1005:1025, x:x + 3] = True  # 下向き barb
    return mask


def test_contour_is_hollow():
    solid = np.zeros((20, 20), dtype=bool)
    solid[5:15, 5:15] = True
    edge = contour_of(solid)
    assert edge.sum() == 36, edge.sum()          # 10x10 の外周 = 36 px
    assert not edge[9, 9]                         # 内部は輪郭ではない
    assert contour_of(np.zeros((8, 8), bool)).sum() == 0


def test_cell_reduce_is_any_pooling():
    mask = np.zeros((8, 8), dtype=bool)
    mask[3, 3] = True
    assert cell_reduce(mask, 1).sum() == 1
    assert cell_reduce(mask, 4).shape == (2, 2)
    assert cell_reduce(mask, 4).sum() == 1
    # 割り切れない形状はパディングされる
    assert cell_reduce(np.ones((5, 5), bool), 4).shape == (2, 2)


def test_miou_is_asymmetric():
    """分母が GT 側だけなので MIoU(a,b) != MIoU(b,a)。

    r の非対称性そのものを見るため use_contour=False で入れ子領域を比較する。
    小さい方を GT に置くと「予測が GT を完全に覆っている」ため満点になり、
    大きい方を GT に置くと取りこぼしとして減点される。
    """
    big = np.zeros((256, 256), dtype=bool)
    big[50:200, 50:200] = True
    small = np.zeros((256, 256), dtype=bool)
    small[100:150, 100:150] = True
    truth_is_big = multiscale_iou(big, small, use_contour=False)
    truth_is_small = multiscale_iou(small, big, use_contour=False)
    assert truth_is_small == 1.0, truth_is_small
    assert truth_is_big < 0.75, truth_is_big
    print(f"  非対称: MIoU(GT=大,pred=小)={truth_is_big:.4f}  逆={truth_is_small:.4f}")


def test_miou_ignores_interior_but_dice_does_not():
    """輪郭ベースなので、中身をくり抜いても MIoU は変わらない。Dice は激変する。"""
    solid = np.zeros((256, 256), dtype=bool)
    solid[50:200, 50:200] = True
    hollow = solid & ~np.pad(
        np.ones((146, 146), bool), ((52, 58), (52, 58)), constant_values=False
    )[:256, :256]
    assert multiscale_iou(solid, solid) == 1.0
    assert abs(multiscale_iou(solid, hollow) - 1.0) < 1e-9
    assert dice(solid, hollow) < 0.2


def test_barb_loss_is_invisible_to_dice_but_costs_miou():
    """本命。barb を落とした予測は Dice ではほぼ無傷、MIoU では明確に減点される。"""
    truth = _spine(with_barbs=True)
    without_barbs = _spine(with_barbs=False)

    dice_score = dice(truth, without_barbs)
    miou_score = multiscale_iou(truth, without_barbs)

    assert dice_score > 0.85, dice_score          # Dice は「ほぼ正解」と判定する
    assert miou_score < dice_score - 0.15, (dice_score, miou_score)
    print(f"  barb欠落: Dice={dice_score:.4f}  MIoU={miou_score:.4f}")


def test_evaluation_resolution_changes_the_score():
    """評価解像度は中立ではない (V34 の CALIB_SIZE=512 問題)。

    同じ予測でも 2048 と 512 で MIoU の値が変わる。したがって 512 で測った
    OOF 値を 2048 前提のリーダーボードと同じ土俵に置いてはいけない。
    測定された向きと大きさはテスト出力に出す (推測ではなく実測を記録する)。
    """
    truth = _spine(with_barbs=True)
    prediction = _spine(with_barbs=False)
    full = multiscale_iou(truth, prediction)
    reduced = multiscale_iou(cell_reduce(truth, 4), cell_reduce(prediction, 4))
    assert abs(full - reduced) > 0.005, (full, reduced)
    direction = "低解像度の方が甘い" if reduced > full else "低解像度の方が辛い"
    print(f"  2048評価={full:.4f}  512評価={reduced:.4f}  ({direction})")


def test_pairwise_iou_ignores_missed_detections():
    """検出漏れは pairwise IoU では罰せられない。hit/miss rate が必要な理由。"""
    a = np.zeros((64, 64), dtype=bool)
    a[10:20, 10:20] = True
    b = np.zeros((64, 64), dtype=bool)
    b[40:50, 40:50] = True
    both, only_a = [a, b], [a]
    assert pairwise_iou(both, only_a) == 1.0      # b を見逃しても満点
    hit, miss, _ = hit_miss_rate(both, only_a)
    assert hit == 0.5 and miss == 0.5             # hit rate は正しく半減する


def test_matched_miou_penalises_count_error():
    a = np.zeros((64, 64), dtype=bool)
    a[10:20, 10:20] = True
    b = np.zeros((64, 64), dtype=bool)
    b[40:50, 40:50] = True
    assert matched_multiscale_iou([a, b], [a]) < matched_multiscale_iou([a, b], [a, b])


def test_hungarian_dice_matches_v31_implementation():
    """過去数値との接続を保証する回帰テスト。V31/V34 の式をそのまま再現する。"""

    def v31_instance_dice(gt_masks, pred_masks):
        if not gt_masks and not pred_masks:
            return 1.0
        if not gt_masks or not pred_masks:
            return 0.0
        matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
        for i, gt in enumerate(gt_masks):
            for j, pred in enumerate(pred_masks):
                intersection = np.logical_and(gt, pred).sum()
                matrix[i, j] = 2 * intersection / max(gt.sum() + pred.sum(), 1)
        rows, cols = linear_sum_assignment(-matrix)
        return float(matrix[rows, cols].sum() / max(len(gt_masks), len(pred_masks)))

    rng = np.random.default_rng(0)
    for _ in range(20):
        truths = [rng.random((48, 48)) > 0.7 for _ in range(rng.integers(1, 5))]
        preds = [rng.random((48, 48)) > 0.7 for _ in range(rng.integers(1, 5))]
        assert abs(hungarian_dice(truths, preds) - v31_instance_dice(truths, preds)) < 1e-5


def test_delta_set_matches_paper():
    assert DEFAULT_DELTAS == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def test_cell_keys_fast_path_equals_dense_definition():
    """座標集合による高速経路が、密な cell_reduce の定義と厳密に一致すること。"""
    rng = np.random.default_rng(7)
    for _ in range(15):
        shape = (int(rng.integers(30, 70)), int(rng.integers(30, 70)))
        truth = rng.random(shape) > 0.6
        prediction = rng.random(shape) > 0.6
        for use_contour in (True, False):
            fast = multiscale_iou(truth, prediction, use_contour=use_contour)

            dense_truth = contour_of(truth) if use_contour else truth
            dense_prediction = contour_of(prediction) if use_contour else prediction
            ratios = []
            for delta in DEFAULT_DELTAS:
                truth_cells = cell_reduce(dense_truth, delta)
                prediction_cells = cell_reduce(dense_prediction, delta)
                denominator = int(truth_cells.sum())
                if denominator == 0:
                    ratios.append(0.0)
                    continue
                shared = int(np.logical_and(truth_cells, prediction_cells).sum())
                ratios.append(shared / denominator)
            dense = float(np.mean(ratios)) if dense_truth.any() else 0.0

            assert abs(fast - dense) < 1e-9, (fast, dense, use_contour, shape)


def test_full_resolution_matching_is_fast_enough():
    """2048^2 で 8x8 個体の総当たりが実用時間で終わること (V34 の停止事故の再発防止)。"""
    import time

    truths = [_shifted(_spine(True), dy=i * 60) for i in range(8)]
    predictions = [_shifted(_spine(False), dy=i * 60 + 3) for i in range(8)]
    started = time.perf_counter()
    score = matched_multiscale_iou(truths, predictions)
    elapsed = time.perf_counter() - started
    assert 0.0 < score < 1.0, score
    assert elapsed < 5.0, f"8x8 の 2048^2 総当たりに {elapsed:.1f}s かかった"
    print(f"  2048^2 8x8 総当たり: {elapsed:.2f}s  matched_MIoU={score:.4f}")


# ---------------------------------------------------------------------------
# Panoptic Quality — リーダーボード指標
# ---------------------------------------------------------------------------


def _box(y0, y1, x0, x1, size=64):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _official_get_pq_score(rows, iou_threshold=0.5):
    """主催者の self-evaluation notebook の get_pq_score を逐語移植したもの。

    rows は (iou_matrix, n_gt, n_pred) のリスト。torch を numpy へ置き換えた以外は
    制御フローを変えていない。本体実装がここから乖離したら気づけるようにする。
    """
    tp_iou_scores, fp_count, fn_count = [], 0, 0
    for iou_matrix, n_gt, n_pred in rows:
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        hit_matrix = iou_matrix > iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += int((hit_matrix.sum(axis=0) == 0).sum())
        fn_count += int((hit_matrix.sum(axis=1) == 0).sum())
    denominator = len(tp_iou_scores) + 0.5 * fp_count + 0.5 * fn_count
    return (sum(tp_iou_scores) / denominator) if denominator > 0 else 0.0


def test_pq_matches_official_implementation():
    """本体の PanopticQuality が公式実装と一致すること (差分テスト)。"""
    rng = np.random.default_rng(11)
    for _ in range(200):
        n_gt = int(rng.integers(0, 5))
        n_pred = int(rng.integers(0, 5))
        matrix = rng.random((n_gt, n_pred)) if n_gt and n_pred else np.zeros((n_gt, n_pred))
        accumulator = PanopticQuality()
        accumulator.add_matrix(matrix, n_gt, n_pred)
        expected = _official_get_pq_score([(matrix, n_gt, n_pred)])
        assert abs(accumulator.score - expected) < 1e-12, (accumulator.score, expected)


def test_pq_hand_computed():
    """手計算で追える最小例。TP2件(IoU 0.8, 0.6) + FP1 + FN1。"""
    matrix = np.array([[0.8, 0.0, 0.0],
                       [0.0, 0.6, 0.0],
                       [0.0, 0.0, 0.0]])
    accumulator = PanopticQuality()
    accumulator.add_matrix(matrix, 3, 3)
    # TP=2 (0.8, 0.6), pred2 はどのGTとも当たらずFP, gt2 は当たらずFN
    assert accumulator.fp_count == 1 and accumulator.fn_count == 1
    expected = (0.8 + 0.6) / (2 + 0.5 * 1 + 0.5 * 1)
    assert abs(accumulator.score - expected) < 1e-12
    assert abs(accumulator.score - 0.4666666666666667) < 1e-12


def test_iou_050_is_a_hard_cliff():
    """IoU 0.5 ちょうどは TP ではない。惜しい予測は無得点ではなく二重減点になる。"""
    just_under = PanopticQuality()
    just_under.add_matrix(np.array([[0.50]]), 1, 1)
    assert just_under.score == 0.0
    assert just_under.fp_count == 1 and just_under.fn_count == 1   # FP と FN の両方
    assert just_under.denominator == 1.0

    just_over = PanopticQuality()
    just_over.add_matrix(np.array([[0.51]]), 1, 1)
    assert abs(just_over.score - 0.51) < 1e-12

    # 何も出さない方がマシ、ということはない: 出さなければ FN だけで分母 0.5
    silent = PanopticQuality()
    silent.add_matrix(np.zeros((1, 0)), 1, 0)
    assert silent.denominator == 0.5 and silent.score == 0.0


def test_pq_decomposes_into_sq_times_rq():
    accumulator = PanopticQuality()
    accumulator.add_matrix(np.array([[0.9, 0.0], [0.0, 0.7]]), 2, 2)
    accumulator.add_matrix(np.array([[0.0, 0.0]]), 1, 2)
    summary = accumulator.summary()
    assert abs(summary["sq"] * summary["rq"] - summary["pq"]) < 1e-12
    assert summary["tp"] == 2 and summary["fp"] == 2 and summary["fn"] == 1


def test_multiple_annotators_are_pooled_not_averaged():
    """同一予測が各アノテータの GT と別々に突き合わされ、全体へ足し込まれること。

    アノテータ A には当たり、B には外れる予測は TP 1 件と FP 1 件の両方を生む。
    variant ごとにスコアを出して平均する V30/V31 の扱いとは結果が変わる。
    """
    prediction = [_box(10, 30, 10, 30)]
    annotator_a = [_box(10, 30, 10, 30)]            # 完全一致 -> TP (IoU 1.0)
    annotator_b = [_box(40, 60, 40, 60)]            # 重なりなし -> FP + FN

    pooled = PanopticQuality()
    pooled.add(annotator_a, prediction)
    pooled.add(annotator_b, prediction)
    assert pooled.tp_ious == [1.0]
    assert pooled.fp_count == 1 and pooled.fn_count == 1
    assert abs(pooled.score - 1.0 / (1 + 0.5 + 0.5)) < 1e-12

    # variant 平均だと (1.0 + 0.0)/2 = 0.5 になり、公式の 0.5 とは一致しても
    # 一般には食い違う。3アノテータで確認する。
    three = PanopticQuality()
    for gt in (annotator_a, annotator_b, annotator_b):
        three.add(gt, prediction)
    assert abs(three.score - 1.0 / (1 + 0.5 * 2 + 0.5 * 2)) < 1e-12
    assert abs(three.score - 1.0 / 3.0) < 1e-12     # variant平均なら 1/3 ではなく 0.333.. 別経路


def test_iou_matrix_matches_pairwise_definition():
    gt = [_box(10, 30, 10, 30), _box(40, 60, 40, 60)]
    pred = [_box(10, 30, 10, 30), _box(20, 40, 20, 40)]
    matrix = iou_matrix_of(gt, pred)
    assert matrix.shape == (2, 2)
    assert abs(matrix[0, 0] - 1.0) < 1e-12
    manual = np.logical_and(gt[0], pred[1]).sum() / np.logical_or(gt[0], pred[1]).sum()
    assert abs(matrix[0, 1] - manual) < 1e-12


def test_marginal_threshold_math():
    """PQ 0.35・想定 IoU 0.7 なら、当たる確率 25% の候補でも出した方が得。"""
    assert abs(marginal_prediction_threshold(0.35, 0.7) - 0.25) < 1e-12
    # PQ が上がるほど、追加候補に要求される確度も上がる
    assert marginal_prediction_threshold(0.48, 0.7) > marginal_prediction_threshold(0.35, 0.7)
    print(f"  PQ=0.35で追加すべき最小TP確率: {marginal_prediction_threshold(0.35, 0.7):.3f}")


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
