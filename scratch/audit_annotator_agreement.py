"""Annotator-agreement audit: 「天井は annotator 不一致」の一次データ直接検証。

MAGFiLO の train annotations は 707 物理画像 × 1154 annotator record
(`image_id` = `<annotator_prefix>-<file stem>`) から成り、公式 PQ は
record 単位で採点される。つまり「annotator A が描き annotator B が
描かなかったフィラメント」は、A record では GT、B record では FP に
なる — 天井の構造的原因。このスクリプトは COCO JSON から

  1. inter-annotator PQ (record A のマスクを予測として record B に採点)
  2. 物理フィラメント cluster の annotator 被覆率
     (IoU>=0.5 で record 横断マッチ → 何人中何人がマークしたか)
  3. record 内 GT 同士の重複 (disjoint 制約下で構造的に失点不可避か)
  4. GT 面積分布 (min_area=20 との関係)

を計測する。使い方:

    python scratch/audit_annotator_agreement.py <annotations.json>

RESEARCH_LOG §2.78 の一次証拠。判定ロジック・閾値は本スクリプト内に
留め、notebook には依存しない。
"""

from __future__ import annotations

import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

MATCH_IOU = 0.5  # 公式 PQ と同じ IoU 判定


def decode_ann(ann: dict, height: int, width: int) -> np.ndarray:
    """COCO polygon segmentation -> uint8 mask。"""
    rle = mask_utils.frPyObjects(ann["segmentation"], height, width)
    rle = mask_utils.merge(rle) if isinstance(rle, list) else rle
    return mask_utils.decode(rle)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    return inter / int((a | b).sum()) if inter else 0.0


def pq_of(preds: list[np.ndarray], gts: list[np.ndarray]) -> float | None:
    """簡易 PQ: IoU>=0.5 で greedy 1-to-1 マッチ → sum_iou/(tp+0.5fp+0.5fn)。

    公式 metric と同じ判定構造 (disjoint 前提の greedy 割当)。マスクが
    両方空なら None を返す。
    """
    tp = 0
    sum_iou = 0.0
    used: set[int] = set()
    for pm in preds:
        best_iou, best_gi = 0.0, -1
        for gi, gm in enumerate(gts):
            if gi in used:
                continue
            value = mask_iou(pm, gm)
            if value > best_iou:
                best_iou, best_gi = value, gi
        if best_iou >= MATCH_IOU:
            tp += 1
            sum_iou += best_iou
            used.add(best_gi)
    fp = len(preds) - tp
    fn = len(gts) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return (sum_iou / denom) if denom > 0 else None


def cluster_coverage(rec_masks: dict[str, list[np.ndarray]], recs: list[str]) -> collections.Counter:
    """record 横断クラスタ: IoU>=0.5 で連結する mask を同一物理フィラメントと
    みなし、(file の record 数, その cluster をマークした annotator 数) を返す。"""
    allm = [(r, mi) for r in recs for mi in range(len(rec_masks[r]))]
    seen: set[tuple[str, int]] = set()
    coverage: collections.Counter = collections.Counter()
    for r, mi in allm:
        if (r, mi) in seen:
            continue
        inrecs = {r}
        stack = [(r, mi)]
        seen.add((r, mi))
        while stack:
            cr, cm = stack.pop()
            for r2, m2 in allm:
                if r2 == cr or (r2, m2) in seen:
                    continue
                if mask_iou(rec_masks[cr][cm], rec_masks[r2][m2]) >= MATCH_IOU:
                    seen.add((r2, m2))
                    stack.append((r2, m2))
                    inrecs.add(r2)
        coverage[(len(recs), len(inrecs))] += 1
    return coverage


def audit(annotations_json: Path) -> dict:
    doc = json.loads(annotations_json.read_text(encoding="utf-8"))
    height = doc["images"][0]["height"]
    width = doc["images"][0]["width"]

    file2recs: dict[str, list[str]] = collections.defaultdict(list)
    for image in doc["images"]:
        file2recs[image["file_name"]].append(image["id"])
    anns_by_rec: dict[str, list[dict]] = collections.defaultdict(list)
    for ann in doc["annotations"]:
        anns_by_rec[ann["image_id"]].append(ann)

    # --- record 内の GT 重複 (disjoint 制約との矛盾がないか) ---
    within_overlap_px = 0
    within_overlap_pairs = 0
    total_ann_px = 0
    for anns in anns_by_rec.values():
        masks = [decode_ann(a, height, width) for a in anns]
        union = np.zeros((height, width), dtype=np.int32)
        for m in masks:
            union += m
            total_ann_px += int(m.sum())
        within_overlap_px += int((union >= 2).sum())
        for x, y in itertools.combinations(range(len(masks)), 2):
            if int((masks[x] & masks[y]).sum()) > 0:
                within_overlap_pairs += 1

    # --- multi-record file: inter-annotator PQ + cluster coverage ---
    pq_pairs: list[float] = []
    coverage: collections.Counter = collections.Counter()
    matched_ious: list[float] = []
    for recs in file2recs.values():
        if len(recs) < 2:
            continue
        rec_masks = {r: [decode_ann(a, height, width) for a in anns_by_rec[r]] for r in recs}
        for i, ri in enumerate(recs):
            for k, rk in enumerate(recs):
                if i == k:
                    continue
                pq = pq_of(rec_masks[ri], rec_masks[rk])
                if pq is not None:
                    pq_pairs.append(pq)
        coverage.update(cluster_coverage(rec_masks, recs))
        # matched pair IoU (record[0] を基準に貪欲マッチ)
        for ri in recs[1:]:
            used: set[int] = set()
            for bm in rec_masks[recs[0]]:
                best_iou, best_gi = 0.0, -1
                for gi, gm in enumerate(rec_masks[ri]):
                    if gi in used:
                        continue
                    value = mask_iou(bm, gm)
                    if value > best_iou:
                        best_iou, best_gi = value, gi
                if best_iou >= MATCH_IOU:
                    used.add(best_gi)
                    matched_ious.append(best_iou)

    areas = sorted(a["area"] for a in doc["annotations"])
    clusters = sum(coverage.values())
    singletons = sum(v for (nrecs, nmarked), v in coverage.items() if nmarked == 1)
    return {
        "images": len(doc["images"]),
        "unique_file_names": len(file2recs),
        "multi_record_files": sum(1 for r in file2recs.values() if len(r) >= 2),
        "records_per_file": dict(collections.Counter(len(r) for r in file2recs.values())),
        "annotations": len(doc["annotations"]),
        "category_counts": dict(collections.Counter(a["category_id"] for a in doc["annotations"])),
        "area_min": areas[0],
        "area_lt20": sum(1 for a in areas if a < 20),
        "within_record_overlap_px": within_overlap_px,
        "within_record_overlap_frac": within_overlap_px / max(total_ann_px, 1),
        "within_record_overlap_pairs": within_overlap_pairs,
        "ordered_annotator_pairs": len(pq_pairs),
        "inter_annotator_pq_mean": float(np.mean(pq_pairs)),
        "inter_annotator_pq_median": float(np.median(pq_pairs)),
        "cluster_coverage": {f"recs={k[0]}_marked={k[1]}": v for k, v in sorted(coverage.items())},
        "clusters_total": clusters,
        "singleton_frac": singletons / max(clusters, 1),
        "matched_pair_iou_mean": float(np.mean(matched_ious)),
        "matched_pair_iou_median": float(np.median(matched_ious)),
    }


def main() -> None:
    path = Path(sys.argv[1])
    result = audit(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
