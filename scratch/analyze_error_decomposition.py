"""fold2 OOF の誤差を annotator 層別に分解する根本原因分析。

prod 構成 (yolo_fold2 + refiner_all_data, conf 0.26, refiner thr 0.45,
drop 0.90, min_area 20) をローカル CPU で再推論し、各予測/GT を
「レコードをまたいだ物体クラスタ」単位で分解する。公式評価は
(annotator, image) ごとの micro 集計なので、同一物体が他アノテータの
レコードでは GT・このレコードでは FP になる「annotator-gap 損失」と、
本当の検出失敗を分離するのが目的。

分類:
  FN GT (record R で未検出):
    - proposal_blind : どの raw 提案も g のクラスタ (他 annotator 版含む) と
                       IoU<=0.5 にならない → 検出器の盲点
    - refiner_lost   : proposal はクラスタをカバーするが refined/final が落とす
    - near_miss      : final pred が g 自身と 0.3<IoU<=0.5 (境界品質の崖)
    - cross_covered  : final pred が他レコードのクラスタ仲間と >0.5 だが
                       g 自身とは <=0.5 (annotator 不一致損失)
  FP pred (record R で未ヒット):
    - annotator_gap  : 他レコードのクラスタに >0.5 でヒット (実在物体)
    - near_miss      : どこかのクラスタと 0.3<IoU<=0.5
    - free           : どのクラスタにも >0.3 で届かない (真の幻覚)

反実仮想:
  c1: IoU 閾値を 0.3 に下げた場合の PQ (near-miss 帯の救済上限)
  c2: cluster-coverage transfer — クラスタ仲間のどれかがヒットすれば
      全 member を救済した場合の PQ (annotator 不一致の回収上限)
  c3: 全 GT 完全検出 oracle の PQ (=1.0、分母規模の確認用)
  c4: free FP を全部消せた場合の PQ (幻覚抑制の上限)
  c5: union dedup 評価 — クラスタ代表 1 個ずつで採点した場合の PQ

使い方:
    python scratch/analyze_error_decomposition.py \
        --ann <train.json> --imgdir <train_images/> --val-list <fold2.txt> \
        --yolo <yolo_fold2_best.pt> --refiner <refiner_all_data.pt> \
        --out output/error_decomp_fold2 --device cpu [--skip-infer]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tile_infer_probe import predict_masks, to_rle  # noqa: E402
from solar_filament.metrics.metrics_official import PanopticQuality  # noqa: E402
from solar_filament.models.mediseg_models import MediSegEdgeAttUNet  # noqa: E402
from solar_filament.pipeline.ownership import apply_ownership  # noqa: E402

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--imgdir", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True,
                        help="fold2 val file_name の改行区切りリスト")
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None,
                        help="既定: <out>/cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-infer", action="store_true")
    args = parser.parse_args()
    cache = args.cache or (args.out / "cache")
    cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    val_names = [ln.strip() for ln in args.val_list.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
    print({"val_images": len(val_names)})

    with open(args.ann, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
    annotations: dict = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    file_to_ids: dict = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)

    # --- Phase 1: 推論 + キャッシュ -------------------------------------
    yolo = None
    refiner = None
    if not args.skip_infer:
        from ultralytics import YOLO
        yolo = YOLO(str(args.yolo))
        refiner = MediSegEdgeAttUNet(3, 1, False).to(args.device).eval()
        checkpoint = torch.load(args.refiner, map_location="cpu")
        state = (checkpoint.get("best_model") or checkpoint.get("model")
                 or checkpoint.get("state_dict") or checkpoint)
        refiner.load_state_dict(state, strict=True)

    started = time.perf_counter()
    for index, name in enumerate(val_names, 1):
        stem = Path(name).stem
        cache_file = cache / f"{stem}.npz"
        if cache_file.exists():
            continue
        image = load_preprocessed(args.imgdir / name)
        proposals, confs = predict_masks(
            yolo, image, YOLO_IMGSZ, DETECT_CONF, args.device)
        refined = refine_proposals(refiner, image, proposals)
        final = apply_ownership(
            refined, confs,
            rule="drop_contained_then_area_asc",
            containment_threshold=DROP_THRESHOLD, min_area=MIN_AREA)
        prop_packed, prop_n = pack_masks(proposals)
        ref_packed, ref_n = pack_masks(refined)
        fin_packed, fin_n = pack_masks(final)
        np.savez_compressed(
            cache_file, prop_packed=prop_packed, prop_n=prop_n,
            ref_packed=ref_packed, ref_n=ref_n,
            fin_packed=fin_packed, fin_n=fin_n, confs=confs)
        if index % 5 == 0 or index == len(val_names):
            elapsed = time.perf_counter() - started
            remaining = elapsed / index * (len(val_names) - index)
            print(f"infer {index}/{len(val_names)} {elapsed:.0f}s "
                  f"残見積 {remaining:.0f}s", flush=True)

    # --- Phase 2: annotator 層別分解 ------------------------------------
    pq = PanopticQuality()                    # 公式経路の再現
    pq_thr03 = PanopticQuality(iou_threshold=NEAR_MISS_LO)   # c1: 帯域救済
    oracle_pq = PanopticQuality()             # c3: 全 GT 完全検出
    freefp_pq = PanopticQuality()             # c4: free FP 除去
    union_pq = PanopticQuality()              # c5: クラスタ代表 dedup
    cluster_cov_pq = PanopticQuality()        # c2: クラスタ仲間ヒットで救済

    fn_rows = []   # 未検出 GT 1 行ずつ
    fp_rows = []   # FP 予測 1 行ずつ
    tp_rows = []
    cluster_rows = []  # 物体 (クラスタ) 単位の被覆段階
    per_photo = []

    for name in val_names:
        stem = Path(name).stem
        saved = np.load(cache / f"{stem}.npz")
        proposals = unpack_masks(saved["prop_packed"], int(saved["prop_n"]))
        refined = unpack_masks(saved["ref_packed"], int(saved["ref_n"]))
        final = unpack_masks(saved["fin_packed"], int(saved["fin_n"]))
        prop_rles = rles_of(proposals)
        ref_rles = rles_of(refined)
        fin_rles = rles_of(final)
        fin_area = [float(mask_util.area(r)) for r in fin_rles]

        record_ids = sorted(file_to_ids[name])
        # 全レコードの GT を RLE で集める (dense mask へは decode しない)
        gt_all = []          # (record_id, rle, area)
        for rid in record_ids:
            info = images[rid]
            for ann in annotations[rid]:
                rle = to_rle(ann, int(info["height"]), int(info["width"]))
                gt_all.append((rid, rle, float(mask_util.area(rle))))
        gt_rles = [r for _, r, _ in gt_all]
        uf = UnionFind(len(gt_all))
        n_gt = len(gt_all)
        for i in range(n_gt):
            for j in range(i + 1, n_gt):
                if gt_all[i][0] == gt_all[j][0]:
                    continue  # 同レコード内はクラスタ化しない
                if mask_util.iou([gt_rles[i]], [gt_rles[j]], [0])[0] > CLUSTER_IOU:
                    uf.union(i, j)
        clusters = defaultdict(list)
        for idx in range(n_gt):
            clusters[uf.find(idx)].append(idx)
        cluster_of = {idx: uf.find(idx) for idx in range(n_gt)}
        # クラスタの annotator 数 (ユニーク record_id)
        cluster_k = {root: len({gt_all[i][0] for i in members})
                     for root, members in clusters.items()}
        # クラスタ代表 = 最大面積 member (union dedup 評価用)
        cluster_rep = {root: max(members, key=lambda i: gt_all[i][2])
                       for root, members in clusters.items()}
        rep_idx = sorted(cluster_rep.values())
        rep_rles = [gt_rles[i] for i in rep_idx]

        # 大きな IoU 行列は写真あたり 3 回だけ (n_gt_all × 各 pred 集合)
        m_prop = iou_matrix(gt_rles, prop_rles)
        m_ref = iou_matrix(gt_rles, ref_rles)
        m_fin = iou_matrix(gt_rles, fin_rles)

        # final pred -> 元 proposal の conf 復元 (ownership で並びが変わるため、
        # final ⊆ parent proposal で IoU 最大の prop が親)
        confs = saved["confs"].astype(np.float32)
        pred_conf = np.full(len(final), np.nan, np.float32)
        if fin_rles and prop_rles:
            m_fp = iou_matrix(fin_rles, prop_rles)  # (n_fin, n_prop)
            own = m_fp.argmax(axis=1)
            ok = m_fp[np.arange(len(fin_rles)), own] > 0.0
            pred_conf[ok] = confs[own[ok]]

        # --- クラスタ (物体) 単位の段階被覆 (行列スライスのみ) ------------
        for root, members in clusters.items():
            cluster_rows.append({
                "file_name": name, "cluster_k": cluster_k[root],
                "members": len(members),
                "rep_area": gt_all[cluster_rep[root]][2],
                "best_proposal": float(m_prop[members].max())
                if m_prop.size else 0.0,
                "best_refined": float(m_ref[members].max())
                if m_ref.size else 0.0,
                "best_final": float(m_fin[members].max())
                if m_fin.size else 0.0,
            })

        # c5 union dedup: 写真 1 回だけ評価 (レコード重複を除去した物体数)
        union_pq.add_matrix(m_fin[rep_idx], len(rep_idx), len(fin_rles))

        # レコード -> 行 index の対応表
        rec_rows_map = defaultdict(list)
        for idx, (rid, _, _) in enumerate(gt_all):
            rec_rows_map[rid].append(idx)

        for rid in record_ids:
            rec_idx = rec_rows_map[rid]
            rec_rles = [gt_rles[i] for i in rec_idx]
            mat = m_fin[rec_idx]                     # (n_rec_gt, n_pred)
            hits = mat > CLUSTER_IOU
            pq.add_matrix(mat, len(rec_idx), len(fin_rles))
            pq_thr03.add_matrix(mat, len(rec_idx), len(fin_rles))

            # c3 oracle: pred=GT自身
            oracle_pq.add_matrix(iou_matrix(rec_rles, rec_rles),
                                 len(rec_idx), len(rec_idx))

            # c2 cluster-coverage: g の行をクラスタ仲間を含めた max に置換
            # →「どれかの annotator 版がヒットすれば救済」の反実仮想
            mat_cov = mat.copy()
            for local_i, gidx in enumerate(rec_idx):
                members = clusters[cluster_of[gidx]]
                if len(members) > 1:
                    mat_cov[local_i] = m_fin[members].max(axis=0)
            cluster_cov_pq.add_matrix(mat_cov, len(rec_idx), len(fin_rles))

            # --- FN 分解 (真の FN のみ) ---
            fn_mask = hits.sum(axis=1) == 0
            for local_i in np.flatnonzero(fn_mask):
                gidx = rec_idx[int(local_i)]
                root = cluster_of[gidx]
                k = cluster_k[root]
                best_final = float(mat[local_i].max()) if fin_rles else 0.0
                mates = [i for i in clusters[root] if i != gidx]
                mate_hit = (float(m_fin[mates].max())
                            if mates and m_fin.size else 0.0)
                # クラスタ被覆を失った段階 (被覆は単調でない:
                # refined が proposal の IoU を 0.5 超に救済し得るため
                # 最後に被覆が残っていた段階で帰属する)
                cov = {}
                for stage_name, m_all in (("proposal", m_prop),
                                          ("refined", m_ref),
                                          ("final", m_fin)):
                    cov[stage_name] = bool(
                        m_all.size and float(
                            m_all[clusters[root]].max()) > CLUSTER_IOU)
                stage = stage_lost(cov["proposal"], cov["refined"],
                                   cov["final"])
                cls = ("cross_covered" if mate_hit > CLUSTER_IOU
                       else ("near_miss" if best_final > NEAR_MISS_LO
                             else stage))
                fn_rows.append({
                    "file_name": name, "record_id": rid, "cluster_k": k,
                    "cls": cls, "stage_lost": stage,
                    "best_final_iou": best_final, "mate_hit_iou": mate_hit,
                    "gt_area": gt_all[gidx][2],
                })

            # --- FP 分解 ---
            fp_mask = hits.sum(axis=0) == 0
            for pi in np.flatnonzero(fp_mask):
                best_any = float(m_fin[:, pi].max()) if m_fin.size else 0.0
                cls = ("annotator_gap" if best_any > CLUSTER_IOU
                       else ("near_miss" if best_any > NEAR_MISS_LO
                             else "free"))
                fp_rows.append({
                    "file_name": name, "record_id": rid, "pred_idx": int(pi),
                    "conf": float(pred_conf[pi]),
                    "cls": cls, "best_any_iou": best_any,
                    "pred_area": fin_area[pi],
                })

            # --- TP ペア (層別) ---
            for gi, pi in np.argwhere(hits):
                gidx = rec_idx[int(gi)]
                tp_rows.append({
                    "file_name": name, "record_id": rid, "pred_idx": int(pi),
                    "gt_local": int(gi),
                    "conf": float(pred_conf[int(pi)]),
                    "cluster_k": cluster_k[cluster_of[gidx]],
                    "iou": float(mat[gi, pi]),
                })

            # --- c4: free FP 除去版 (列スライスのみ) ----------------------
            free_flags = []
            for pi in range(len(fin_rles)):
                if hits.sum(axis=0)[pi] > 0:
                    free_flags.append(False)
                    continue
                free_flags.append(
                    float(m_fin[:, pi].max()) <= NEAR_MISS_LO
                    if m_fin.size else True)
            kept_idx = [i for i, f in enumerate(free_flags) if not f]
            freefp_pq.add_matrix(mat[:, kept_idx] if kept_idx else
                                 np.zeros((len(rec_idx), 0)),
                                 len(rec_idx), len(kept_idx))

        per_photo.append({
            "file_name": name, "n_records": len(record_ids),
            "n_gt_instances": n_gt, "n_clusters": len(clusters),
            "n_singleton": sum(1 for k in cluster_k.values() if k == 1),
            "n_preds": len(fin_rles), "n_proposals": len(prop_rles),
        })

    # --- 集計 -----------------------------------------------------------
    base = pq.summary()
    fn_df = pd.DataFrame(fn_rows)
    fp_df = pd.DataFrame(fp_rows)
    tp_df = pd.DataFrame(tp_rows)
    cl_df = pd.DataFrame(cluster_rows)
    photo_df = pd.DataFrame(per_photo)

    def cls_counts(df):
        return df["cls"].value_counts().to_dict() if len(df) else {}

    # 物体 (クラスタ) レベルの被覆 funnel: proposal→refined→final で
    # どの段階がクラスタ被覆 (>0.5) を失うか
    funnel = {}
    if len(cl_df):
        for st in ("proposal", "refined", "final"):
            funnel[st] = int((cl_df[f"best_{st}"] > CLUSTER_IOU).sum())
        funnel["n_clusters"] = len(cl_df)

    summary = {
        "config": {"variant": "prod", "conf": DETECT_CONF,
                   "refiner_thr": REFINER_THRESHOLD, "drop": DROP_THRESHOLD,
                   "min_area": MIN_AREA, "imgsz": YOLO_IMGSZ,
                   "cluster_iou": CLUSTER_IOU, "near_miss_lo": NEAR_MISS_LO},
        "official_reproduction": base,
        "photos": {
            "n": len(photo_df),
            "records": int(photo_df.n_records.sum()),
            "gt_instances": int(photo_df.n_gt_instances.sum()),
            "clusters": int(photo_df.n_clusters.sum()),
            "singleton_clusters": int(photo_df.n_singleton.sum()),
            "singleton_share": float(photo_df.n_singleton.sum()
                                     / photo_df.n_clusters.sum()),
            "preds_final": int(photo_df.n_preds.sum()),
            "proposals": int(photo_df.n_proposals.sum()),
        },
        "cluster_coverage_funnel": funnel,
        "fn_by_class": cls_counts(fn_df),
        "fp_by_class": cls_counts(fp_df),
        "fn_by_cluster_k": fn_df.groupby("cluster_k").size().to_dict()
        if len(fn_df) else {},
        "fn_area_median_by_cls": fn_df.groupby("cls")["gt_area"]
        .median().to_dict() if len(fn_df) else {},
        "fp_area_median_by_cls": fp_df.groupby("cls")["pred_area"]
        .median().to_dict() if len(fp_df) else {},
        "recall_by_cluster_k": {},
        "tp_iou_by_cluster_k": tp_df.groupby("cluster_k")["iou"]
        .agg(["count", "mean"]).to_dict() if len(tp_df) else {},
        "counterfactuals": {
            "c1_iou_thr_0.3": pq_thr03.summary(),
            "c2_cluster_coverage_transfer": cluster_cov_pq.summary(),
            "c3_oracle_all_gt": oracle_pq.summary(),
            "c4_drop_free_fp": freefp_pq.summary(),
            "c5_union_dedup_eval": union_pq.summary(),
        },
    }
    if len(fn_df) or len(tp_df):
        all_gt_k = (pd.concat([tp_df[["cluster_k"]],
                               fn_df[["cluster_k"]]])
                    .groupby("cluster_k").size())
        fn_k = fn_df.groupby("cluster_k").size() if len(fn_df) \
            else pd.Series(dtype=int)
        for k, tot in all_gt_k.items():
            missed = int(fn_k.get(int(k), 0)) if len(fn_k) else 0
            summary["recall_by_cluster_k"][str(k)] = {
                "gt": int(tot), "fn": missed,
                "recall": 1.0 - missed / int(tot)}

    # --- per-pred marginal 価値 -----------------------------------------
    # 各 final pred の真の PQ 寄与: Σ hit IoU / marginal 分母。
    # pred は写真の全 record で採点されるので、ratio > 現 PQ なら正味プラス。

    if len(tp_df) or len(fp_df):
        pred = marginal_pred_frame(tp_df, fp_df)
        cur_pq = base["pq"]
        neg = pred[pred["ratio"] < cur_pq]
        summary_pred = {
            "n_preds": len(pred),
            "value_negative_n": int(len(neg)),
            "value_negative_share": float(len(neg) / len(pred)),
            "neg_conf_median": float(neg["conf"].median())
            if len(neg) else None,
            "pos_conf_median": float(
                pred[pred["ratio"] >= cur_pq]["conf"].median())
            if (pred["ratio"] >= cur_pq).any() else None,
            "conf_band_value": {
                str(b): {
                    "n": int(len(g)),
                    "ratio_mean": float(g["ratio"].mean()),
                    "neg_share": float((g["ratio"] < cur_pq).mean()),
                }
                for b, g in pred.groupby(pd.cut(
                    pred["conf"],
                    [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]))
            },
        }
        summary["per_pred_marginal"] = summary_pred
        pred.to_csv(args.out / "per_pred_marginal.csv", index=False)

    fn_df.to_csv(args.out / "fn_decomp.csv", index=False)
    fp_df.to_csv(args.out / "fp_decomp.csv", index=False)
    tp_df.to_csv(args.out / "tp_decomp.csv", index=False)
    cl_df.to_csv(args.out / "cluster_coverage.csv", index=False)
    photo_df.to_csv(args.out / "per_photo.csv", index=False)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
