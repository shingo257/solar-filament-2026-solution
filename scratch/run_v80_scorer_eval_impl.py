"""Implementation helpers for `run_v80_scorer_eval` — extracted verbatim from `run_v80_scorer_eval.py`.

Holds the helper functions and module constants so `run_v80_scorer_eval.py` stays a readable
orchestrator. Loaded by the wrapper via sibling import.
"""


from __future__ import annotations


import sys
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mask_util


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))


from solar_filament.pipeline.component_tree import select_antichain_forest  # noqa: E402
from solar_filament.pipeline.ownership import (  # noqa: E402
    drop_contained_indices,
    greedy_disjoint,
)
from run_refiner_pq_allfold import (  # noqa: E402
    MIN_AREA,
)
from src.metrics_official import PQ_IOU_THRESHOLD  # noqa: E402


EVAL_SIZE = 2048


PROD_CONF = 0.26


SCORER_MIN_CONF = 0.08


CONTAINMENT = 0.90


FOLDS = (0, 1, 2)


# τ = PQ/2 の固定点は PQ≈0.44 なら ≈0.22。0.16〜0.32 で掃引し、
# 掃引で最良の τ は参考報告のみ (主結果は固定点)。
TAU_GRID = tuple(round(0.16 + 0.02 * i, 2) for i in range(9))  # 0.16 … 0.32


# v81: hybrid の tree 側を独立 τ で掃引する。診断で û の絶対校正が tree で
# 過大(中位候補の累積選択で FP 爆発)と判明したため、yolo は固定点 0.22 に
# 固定し、tree 側だけ 0.20–0.68 の広めグリッドで探す。
TREE_TAU_GRID = tuple(round(0.20 + 0.04 * i, 2) for i in range(13))  # 0.20 … 0.68


YOLO_TAU_FIXED = 0.22  # hybrid の固定点 τ*≈PQ/2


# B2 morphology arm: prod 保持集合の候補マスクに半径 r px の disk
# dilation (r>0) / erosion (r<0) を適用してから通常経路へ通す。
MORPH_GRID = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)


ARMS = ("prod", "yolo_scorer", "tree_only", "hybrid", "hybrid_oracle",
        "hybrid_split", "rescue", "morph")


def arm_tau_grid(arm: str) -> tuple[float, ...]:
    if arm == "morph":
        return MORPH_GRID
    if arm == "prod":
        return (0.0,)
    if arm == "hybrid_split":
        return TREE_TAU_GRID
    return TAU_GRID


CONTEXT_FEATURES = (
    "max_iou_any", "n_overlap_10", "yolo_support_conf",
    "n_cand_img", "n_yolo_img", "n_annotators",
)


# -1 センチネルを NaN に戻す列 (HistGBM は NaN を欠損として扱う)
NA_FEATURES = ("yolo_conf", "level_index", "n_children", "n_siblings",
               "parent_area_ratio", "yolo_support_conf")


def load_candidates(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def rle_dicts(rle_texts) -> list[dict]:
    return [
        {"size": [EVAL_SIZE, EVAL_SIZE], "counts": str(t).encode("utf-8")}
        for t in rle_texts
    ]


def teachers_for(pred_rles: list[dict], gt_per_record: list[list[dict]]):
    """(u, h): record 平均の maxIoU·1[>0.5] と、record 平均の命中割合。"""
    n = len(pred_rles)
    if n == 0 or not gt_per_record:
        return np.zeros(n), np.zeros(n)
    u = np.zeros(n, dtype=np.float64)
    h = np.zeros(n, dtype=np.float64)
    for gt_rles in gt_per_record:
        matrix = np.asarray(
            mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64
        )
        best = matrix.max(axis=1) if matrix.size else np.zeros(n)
        u += np.where(best > PQ_IOU_THRESHOLD, best, 0.0)
        h += (best > PQ_IOU_THRESHOLD)
    return u / len(gt_per_record), h / len(gt_per_record)


def context_features(cand: dict, pred_rles: list[dict], n_annotators: int) -> np.ndarray:
    n = len(pred_rles)
    out = np.zeros((n, len(CONTEXT_FEATURES)), dtype=np.float32)
    if n == 0:
        return out
    self_iou = np.asarray(
        mask_util.iou(pred_rles, pred_rles, [0] * n), dtype=np.float64
    )
    np.fill_diagonal(self_iou, 0.0)
    source = cand["source"].astype(int)
    conf = cand["conf"]
    is_yolo = source == 0
    yolo_support = np.full(n, np.nan, dtype=np.float64)
    for i in np.where(~is_yolo)[0]:
        hits = np.where((self_iou[i] > PQ_IOU_THRESHOLD) & is_yolo)[0]
        if hits.size:
            yolo_support[i] = conf[hits].max()
    out[:, 0] = self_iou.max(axis=1)
    out[:, 1] = (self_iou > 0.10).sum(axis=1)
    out[:, 2] = yolo_support
    out[:, 3] = n
    out[:, 4] = float(is_yolo.sum())
    out[:, 5] = n_annotators
    return out


def _encode(masks: list[np.ndarray]) -> list[dict]:
    if not masks:
        return []
    return mask_util.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))


def _evaluate_image(bundle: dict) -> dict:
    """1観測分の (arm, τ) ごとの PQ 寄与 (tp_iou_sum, tp, fp, fn)。

    ワーカープロセスで実行される。mask の decode は画像につき1回だけ行い、
    排他化は既存ルール (drop_contained@0.90 → area 昇順 greedy) を使う。
    """
    masks = [
        mask_util.decode(r).astype(bool) for r in rle_dicts(bundle["rle_final"])
    ]
    rles = rle_dicts(bundle["rle_final"])
    usable = np.asarray(bundle["usable"], dtype=bool)
    source = np.asarray(bundle["source"], dtype=int)
    conf = np.asarray(bundle["conf"])
    offset = int(bundle["tree_offset"])
    n_tree = int(bundle["n_tree_nodes"])
    children = bundle["tree_children"]
    roots = bundle["tree_roots"]
    active = bundle.get("arms", ARMS)
    out = {arm: {} for arm in active}

    for arm in active:
        util = np.asarray(bundle["u_true"] if arm == "hybrid_oracle" else bundle["u_hat"])
        yolo_ok = (source == 0) & usable & (conf >= SCORER_MIN_CONF)
        for tau in arm_tau_grid(arm):
            keep: list[int] = []
            if arm in ("prod", "morph"):
                # morph (B2, §2.37): keep 集合は prod と同一 — 違いは
                # マスクへの disk dilation/erosion 適用のみ。
                keep += list(np.where((source == 0) & usable & (conf >= PROD_CONF))[0])
            if arm == "rescue":
                # B1 (§2.36): conf>=0.26 は無条件維持し、conf∈[0.08,0.26) の
                # 候補を û>τ で追加採用する strictly-additive rescue。
                keep += list(np.where(
                    (source == 0) & usable &
                    ((conf >= PROD_CONF) |
                     ((conf >= SCORER_MIN_CONF) & (util > tau))))[0])
            if arm in ("yolo_scorer", "hybrid", "hybrid_oracle"):
                keep += list(np.where(yolo_ok & (util > tau))[0])
            if arm == "hybrid_split":
                # yolo 側は固定点 τ* で固定、tree 側だけ tau を掃引 (v81)
                keep += list(np.where(yolo_ok & (util > YOLO_TAU_FIXED))[0])
            if arm in ("tree_only", "hybrid", "hybrid_oracle", "hybrid_split"):
                tree_util = np.full(n_tree, -np.inf)
                for node in range(n_tree):
                    row = offset + node
                    if usable[row]:
                        tree_util[node] = util[row]
                keep += [
                    offset + node
                    for node in select_antichain_forest(children, roots, tree_util, tau)
                ]
            subset = [masks[i] for i in keep]
            if arm == "morph":
                r = int(round(tau))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * abs(r) + 1, 2 * abs(r) + 1))
                op = cv2.dilate if r > 0 else cv2.erode
                subset = [
                    op(m.astype(np.uint8), kernel).astype(bool) for m in subset
                ]
                subset_rles = _encode(
                    [m.astype(np.uint8) for m in subset])
            else:
                subset_rles = [rles[i] for i in keep]
            alive = drop_contained_indices(
                subset, threshold=CONTAINMENT, rles=subset_rles
            )
            areas = np.array([m.sum() for m in subset])
            preds = greedy_disjoint(
                subset, sorted(alive, key=lambda i: areas[i]), min_area=MIN_AREA
            )
            pred_rles = _encode(preds)
            tp_sum = tp_c = fp_c = fn_c = 0
            for gt in bundle["records"]:
                if not pred_rles:
                    fn_c += len(gt)
                    continue
                matrix = np.asarray(
                    mask_util.iou(pred_rles, gt, [0] * len(gt)), dtype=np.float64
                ).T
                hits = matrix > PQ_IOU_THRESHOLD
                tp_sum += float(matrix[hits].sum())
                tp_c += int(hits.sum())
                fp_c += int((hits.sum(axis=0) == 0).sum())
                fn_c += int((hits.sum(axis=1) == 0).sum())
            # prod は τ 不変 — 全 τ に同じ値を入れて後段を単純化する
            for t in (TAU_GRID if arm == "prod" else (tau,)):
                out[arm][t] = (tp_sum, tp_c, fp_c, fn_c)
    return out


def pq_of(counts) -> float:
    tp_sum = sum(c[0] for c in counts)
    tp = sum(c[1] for c in counts)
    fp = sum(c[2] for c in counts)
    fn = sum(c[3] for c in counts)
    den = tp + 0.5 * fp + 0.5 * fn
    return tp_sum / den if den > 0 else 0.0


def adoption_gate(fold_deltas, ci_lower: float, oracle_delta: float) -> dict:
    """Keep the numeric diagnostic separate from independent acceptance.

    Rows in scorer-training fold g use an upstream model trained on all folds
    except g, including scorer-evaluation fold f. Excluding f only in the
    scorer fit therefore does not establish nested out-of-fold validation.
    """
    return {
        "adopt": False,
        "exploratory_numeric_gate_passed": bool(
            len(fold_deltas) == len(FOLDS)
            and all(delta > 0 for delta in fold_deltas) and ci_lower > 0),
        "oracle_headroom_ok": bool(oracle_delta >= 0.02),
        "reason": "Non-nested upstream fits and unverified frozen split/cache coverage",
    }
