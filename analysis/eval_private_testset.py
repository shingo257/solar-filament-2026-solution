#!/usr/bin/env python
"""Private testset 完全無偏評価ハーネス (公開待ち・前倒し準備).

主催者が private testset (画像 + ラベル) を公開した時点で、凍結済みの提出
CSV 群を公式 PQ で一括評価する。提出選択 58363293 (v137 cons2of3_sptal)
が本当に最良だったかを、候補全 CSV で検証できる。

使い方 (testset 公開後):

    python analysis/eval_private_testset.py         --annotations <testset_annotation.json>         [--submissions-dir "G:/マイドライブ/solar_filament_v30/artifacts/submissions_v137"]         [--out analysis/private_testset_eval.md]

前提:
  - GT は train と同じ COCO 形式 JSON (複数アノテータ record は file_name
    重複の別 image エントリ)。**公開された形式が違う場合は load_gt() を
    合わせること** — そこ以外は形式非依存。
  - 提出 CSV は competition 形式: `filament_id,segmentation_rle` で
    filament_id = `<image_stem>_<n>`、RLE は圧縮文字列。画像サイズ 2048x2048。

検証済みの構成要素:
  - solar_filament.metrics.metrics_official.PanopticQuality — 公式
    self-evaluation notebook の挙動再現 (V35 で LB との整合を確認済み)
  - CSV パーサは v137 提出ファイルで smoke test 済み (下記 --selftest)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solar_filament.metrics.metrics_official import PanopticQuality

IMAGE_SIZE = (2048, 2048)

# LB 既知値との照合サニティチェック (private、2桁丸め)
KNOWN_LB = {
    "submission_v137_cons2of3_sptal_prod_z_c26_maj3.csv": ("58363293 (選択)", 0.37),
    "submission_v137_cons2of3_trio_b_c26_maj3.csv": ("trio_b", 0.36),
}


def parse_submission_csv(path: Path) -> dict[str, list[np.ndarray]]:
    """提出 CSV を image stem -> bool mask list にデコードする。"""
    by_image: dict[str, list[np.ndarray]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = row["filament_id"]
            stem = fid.rsplit("_", 1)[0]
            rle = {"size": list(IMAGE_SIZE), "counts": row["segmentation_rle"].encode("ascii")}
            by_image[stem].append(mask_utils.decode(rle).astype(bool))
    return dict(by_image)


def load_gt(json_path: Path) -> dict[str, list[tuple[str, list[np.ndarray]]]]:
    """COCO 形式 GT を file_stem -> [(record_id, gt masks)] に読む。

    train 形式では複数アノテータ record が file_name 重複の別 image エントリ
    として入っている。各 image エントリを 1 アノテータ record として扱い、
    公式の (annotator, image) 評価単位を再現する。
    """
    coco = json.load(open(json_path, encoding="utf-8"))
    images = {im["id"]: im for im in coco["images"]}
    anns: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    out = defaultdict(list)  # type: dict[str, list[tuple[str, list[np.ndarray]]]]
    for iid, im in images.items():
        stem = Path(im["file_name"]).stem
        h, w = int(im["height"]), int(im["width"])
        masks = []
        for a in anns.get(iid, []):
            if a.get("iscrowd", 0):
                continue
            seg = a["segmentation"]
            if isinstance(seg, list):
                rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
            elif isinstance(seg.get("counts"), list):
                rle = mask_utils.frPyObjects(seg, h, w)
            else:
                rle = dict(seg)
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode("ascii")
            masks.append(mask_utils.decode(rle).astype(bool))
        out[stem].append((str(iid), masks))
    return dict(out)


def evaluate(preds: dict[str, list[np.ndarray]], gt: dict[str, list[tuple[str, list[np.ndarray]]]]) -> dict:
    pq = PanopticQuality()
    n_images = 0
    missing = []
    for stem, records in gt.items():
        pred_masks = preds.get(stem)
        if pred_masks is None:
            missing.append(stem)
            pred_masks = []
        n_images += 1
        for _rec, gt_masks in records:
            pq.add(gt_masks, pred_masks)
    s = pq.summary()
    s["images_evaluated"] = n_images
    s["images_missing_pred"] = len(missing)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, help="private testset COCO annotation JSON")
    ap.add_argument("--submissions-dir", type=Path,
                    default=Path(r"G:/マイドライブ/solar_filament_v30/artifacts/submissions_v137"))
    ap.add_argument("--out", type=Path, default=Path("analysis/private_testset_eval.md"))
    ap.add_argument("--selftest", action="store_true", help="CSV パーサの smoke test のみ")
    args = ap.parse_args()

    if args.selftest:
        for csv_path in sorted(args.submissions_dir.glob("*.csv"))[:2]:
            preds = parse_submission_csv(csv_path)
            counts = [len(v) for v in preds.values()]
            print(f"{csv_path.name}: {len(preds)} images, {sum(counts)} instances, "
                  f"mean {np.mean(counts):.1f}/img")
        print("selftest OK")
        return

    if not args.annotations or not args.annotations.exists():
        sys.exit("--annotations <testset json> が必要 (testset 公開後に実行)")

    gt = load_gt(args.annotations)
    print(f"GT: {len(gt)} images, {sum(len(r) for r in gt.values())} annotator records")

    rows = []
    for csv_path in sorted(args.submissions_dir.glob("*.csv")):
        preds = parse_submission_csv(csv_path)
        s = evaluate(preds, gt)
        label, lb = KNOWN_LB.get(csv_path.name, ("", None))
        rows.append((csv_path.name, label, lb, s))
        print(f"{csv_path.name}: PQ {s['pq']:.4f} SQ {s['sq']:.4f} RQ {s['rq']:.4f} "
              f"TP {s['tp']} FP {s['fp']} FN {s['fn']} missing {s['images_missing_pred']}"
              + (f"  [LB private {lb}]" if lb else ""))

    rows.sort(key=lambda r: -r[3]["pq"])
    lines = ["# Private testset 完全無偏評価", "",
             f"annotations: `{args.annotations}`", "",
             "| submission | note | LB private | PQ | SQ | RQ | TP | FP | FN |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, label, lb, s in rows:
        lines.append(f"| {name} | {label} | {lb if lb else '-'} | {s['pq']:.4f} | "
                     f"{s['sq']:.4f} | {s['rq']:.4f} | {s['tp']} | {s['fp']} | {s['fn']} |")
    args.out.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
