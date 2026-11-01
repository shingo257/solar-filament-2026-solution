"""analyze_error_decomposition の前処理・refiner・マスクキャッシュ・分析層。

prod 前処理 (robust_uint8/solar_three_channel/load_preprocessed) は
build_submission_reproduction.py PREPROC_CELL と同一式、refiner 補助は
run_refiner_pq_allfold と同一式。被覆段階帰属 (stage_lost) と per-pred
marginal 価値 (marginal_pred_frame) もここに住み、wrapper から
re-export される。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

YOLO_IMGSZ = 1280
DETECT_CONF = 0.26
REFINER_THRESHOLD = 0.45
DROP_THRESHOLD = 0.90
MIN_AREA = 20
CROP_SIZE = 320
CROP_CONTEXT = 1.65
REFINER_BATCH = 8

CLUSTER_IOU = 0.5      # annotator 間で同一物体とみなすリンク閾値
NEAR_MISS_LO = 0.3     # near-miss 帯の下限 (<= CLUSTER_IOU)


# --- prod 前処理 (build_submission_reproduction.py PREPROC_CELL と同一式) ---

def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)


def solar_three_channel(gray):
    gray = gray.astype(np.float32)
    disk = gray > max(4.0, np.percentile(gray, 8))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=80, sigmaY=80)
    flattened = gray / np.maximum(background, 1.0)
    flat_u8 = robust_uint8(flattened, disk, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    raw_u8 = robust_uint8(gray, disk, 0.5, 99.5)
    dark_detail = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    dark_detail = robust_uint8(dark_detail, disk, 0.0, 99.5)
    out = np.dstack([raw_u8, clahe, dark_detail])
    out[~disk] = 0
    return out


def load_preprocessed(path: Path) -> np.ndarray:
    """raw JPEG を prod と同じ経路 (3ch -> q96 JPEG -> decode) で読み込む。"""
    gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    encoded = cv2.imencode(".jpg", solar_three_channel(gray),
                           [cv2.IMWRITE_JPEG_QUALITY, 96])[1]
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# --- refiner (run_refiner_pq_allfold と同一式) ---

def square_box_from_extent(ex0, ey0, ex1, ey1, height, width,
                           context=CROP_CONTEXT, minimum=96):
    cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
    side = int(math.ceil(max(ex1 - ex0, ey1 - ey0) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def square_box(mask, context=CROP_CONTEXT, minimum=96):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, width, height
    return square_box_from_extent(
        xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, height, width,
        context=context, minimum=minimum)


def refine_crops(model, image, boxes):
    crops = []
    for x0, y0, x1, y1 in boxes:
        crop = cv2.resize(image[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE),
                          interpolation=cv2.INTER_LINEAR)
        crops.append(crop.transpose(2, 0, 1))
    probabilities = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(crops), REFINER_BATCH):
            chunk = np.stack(crops[start:start + REFINER_BATCH])
            tensor = torch.from_numpy(chunk.copy()).float().to(device) / 255.0
            output = model(tensor).sigmoid()[:, 0].cpu().numpy()
            for offset in range(output.shape[0]):
                x0, y0, x1, y1 = boxes[start + offset]
                probabilities.append(cv2.resize(
                    output[offset], (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_LINEAR))
    return probabilities


def refine_proposals(refiner, image, proposals):
    boxes = [square_box(p) for p in proposals]
    probabilities = refine_crops(refiner, image, boxes) if boxes else []
    refined = []
    for prop, (x0, y0, x1, y1), prob in zip(proposals, boxes, probabilities):
        mask = np.zeros_like(prop, dtype=bool)
        mask[y0:y1, x0:x1] = prob >= REFINER_THRESHOLD
        refined.append(mask & prop)
    return refined


# --- mask キャッシュ ---

def pack_masks(masks):
    if not masks:
        return np.zeros((0, 0), np.uint8), 0
    flat = np.stack([m.astype(np.uint8).reshape(-1) for m in masks])
    return np.packbits(flat, axis=1), len(masks)


def unpack_masks(packed, count, height=2048, width=2048):
    if count == 0:
        return []
    flat = np.unpackbits(packed, axis=1)[:, : height * width]
    return [row.reshape(height, width).astype(bool) for row in flat]


def rles_of(masks):
    """mask list -> RLE list (cache 向け)。空 list は []。"""
    if not masks:
        return []
    return mask_util.encode(
        np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))


def iou_matrix(gt_rles, pred_rles):
    """(n_gt, n_pred) の IoU 行列。pycocotools RLE-IoU で高速計算。

    mask 配列ではなく RLE (mask_util.encode 済み) を受け取る。
    """
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    return np.asarray(
        mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)),
        dtype=np.float64).T


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def stage_lost(prop_cov: bool, ref_cov: bool, fin_cov: bool) -> str:
    """クラスタ被覆 (>0.5) を失った段階の帰属。

    被覆は単調ではない: refined は proposal の縮小なので IoU を
    0.5 以上に引き上げることがある (prop_uncovered → ref_covered)。
    「最初に被覆の無い段階」で break すると救済後に ownership で
    失われた物体を proposal_blind と誤帰属するため、被覆が存在した
    最後の段階で判定する。
    """
    if fin_cov:
        return "survived_boundary"
    if ref_cov:
        return "ownership_lost"
    if prop_cov:
        return "refiner_lost"
    return "proposal_blind"


def marginal_pred_frame(tp_df: pd.DataFrame,
                        fp_df: pd.DataFrame) -> pd.DataFrame:
    """per-(file, pred) の marginal 価値フレーム。

    pred を除去した場合の分母変化は厳密には
    -n_tp (TP 喪失) -0.5·n_fp (FP 喪失) +0.5·n_unique (その pred が
    唯一被覆していた GT が新たに FN 化)。よって marginal 分母は
    n_tp + 0.5·n_fp - 0.5·n_unique。unique 被覆とは、同一
    (file_name, record_id, gt) に IoU>0.5 で届く pred がこの pred
    だけの TP ペア。
    """
    if len(tp_df):
        tp_g = tp_df.groupby(["file_name", "pred_idx"]).agg(
            tp_iou_sum=("iou", "sum"), n_tp=("iou", "size"),
            conf=("conf", "first"))
    else:
        tp_g = pd.DataFrame(
            columns=["file_name", "pred_idx", "tp_iou_sum", "n_tp",
                     "conf"]).set_index(["file_name", "pred_idx"])
    if len(tp_df) and "gt_local" in tp_df.columns:
        cover = (tp_df.groupby(["file_name", "record_id", "gt_local"])
                 ["pred_idx"].nunique())
        uq = tp_df.merge(cover.rename("cover_n").reset_index(),
                         on=["file_name", "record_id", "gt_local"])
        uniq = (uq[uq["cover_n"] == 1]
                .groupby(["file_name", "pred_idx"]).size()
                .rename("n_unique"))
        tp_g = tp_g.join(uniq)
    if len(fp_df):
        fp_g = fp_df.groupby(["file_name", "pred_idx"]).agg(
            n_fp=("cls", "size"),
            fp_conf=("conf", "first"),
            fp_cls=("cls", lambda s: s.value_counts().to_dict()))
    else:
        fp_g = pd.DataFrame(
            columns=["file_name", "pred_idx", "n_fp", "fp_conf",
                     "fp_cls"]).set_index(["file_name", "pred_idx"])
    pred = tp_g.join(fp_g, how="outer").reset_index()
    for col, fill in (("tp_iou_sum", 0.0), ("n_tp", 0),
                      ("n_fp", 0), ("n_unique", 0)):
        if col not in pred.columns:
            pred[col] = fill
        pred[col] = pred[col].fillna(fill)
    if "fp_conf" in pred.columns:
        pred["conf"] = pred["conf"].fillna(pred["fp_conf"])
    pred["marginal_denom"] = (pred["n_tp"] + 0.5 * pred["n_fp"]
                              - 0.5 * pred["n_unique"])
    pred["ratio"] = pred["tp_iou_sum"] / pred["marginal_denom"]
    return pred
