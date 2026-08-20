"""P2-2 の fold0 OOF cache を、V35 確定後処理でローカル採点する。

Colab ノートが Drive へ npz を書いたあと、GPU なしでも同じ数値を再現できる。
キャッシュがまだ無いときは、先に `solar_filament_v36_yolo_imgsz1536_fold0_colab.ipynb`
を実行すること。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mediseg_models import MediSegUNet  # noqa: E402
from src.metrics_official import PanopticQuality  # noqa: E402
from scratch.run_pipeline_stack import apply_ownership  # noqa: E402
from scratch.run_refiner_pq_allfold import (  # noqa: E402
    ARTIFACTS, FOLDS_CSV, MIN_AREA, PREPROC, TRAIN_JSON,
    build_variant, imread_unicode, iou_matrix, load_yolo_cache,
    masks_to_rles, refine_batch, to_rle,
)

DRIVE = Path("G:/マイドライブ")
CACHE = DRIVE / "solar_filament_v35/prediction_cache/oof/yolo_imgsz1536_fold_0"
OUT = Path(__file__).resolve().parents[1] / "output" / "v35"
BASELINE_FOLD0_PQ = 0.4393
CONF = 0.26
FOLD = 0


def main() -> int:
    if not CACHE.exists():
        print(f"OOF cache がまだない: {CACHE}")
        print("先に Colab で solar_filament_v36_yolo_imgsz1536_fold0_colab.ipynb を実行する。")
        return 1

    with open(TRAIN_JSON, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {item["id"]: item for item in coco["images"]}
    annotations: dict = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    file_to_ids: dict = defaultdict(list)
    for image_id, info in images.items():
        file_to_ids[info["file_name"]].append(image_id)
    gt_rles = {
        i: [to_rle(a, int(images[i]["height"]), int(images[i]["width"]))
            for a in anns if not a.get("iscrowd", 0)]
        for i, anns in annotations.items()
    }
    fold_frame = pd.read_csv(FOLDS_CSV)
    fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
    targets = [n for n in sorted(file_to_ids)
               if fold_of_file[n] == FOLD and (CACHE / f"{Path(n).stem}.npz").exists()]
    if not targets:
        print(f"fold{FOLD} の npz が {CACHE} に無い")
        return 1

    checkpoint = torch.load(ARTIFACTS / "mediseg_refiner_fold0_resume.pt", map_location="cpu")
    model = MediSegUNet(3, 1, False).eval()
    model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)

    meter = PanopticQuality()
    near_miss = 0
    counts: list[int] = []
    for name in targets:
        masks, confidences = load_yolo_cache(CACHE / f"{Path(name).stem}.npz")
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences))
                if s >= CONF and m.sum() >= MIN_AREA]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
        image = imread_unicode(PREPROC / name)
        if image is None:
            raise FileNotFoundError(PREPROC / name)
        if proposals:
            boxes, probabilities = refine_batch(model, image, proposals)
            refined = [build_variant(proposals[i], boxes[i], probabilities[i], "shrink", 0.45)
                       for i in range(len(proposals))]
        else:
            refined = []
        predictions = apply_ownership(refined, scores, "drop_contained_then_area_asc")
        counts.append(len(predictions))
        rles = masks_to_rles(predictions)
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            matrix = iou_matrix(truth, rles)
            meter.add_matrix(matrix, len(truth), len(rles))
            if matrix.size:
                near_miss += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

    summary = meter.summary()
    report = {
        "imgsz": 1536,
        "fold": FOLD,
        "observations": len(targets),
        **{k: summary[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")},
        "near_miss": near_miss,
        "inst/img": float(np.mean(counts)) if counts else 0.0,
        "baseline_fold0_pq": BASELINE_FOLD0_PQ,
        "delta": float(summary["pq"] - BASELINE_FOLD0_PQ),
        "continue_fold12": bool(summary["pq"] > BASELINE_FOLD0_PQ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p2_2_imgsz1536_fold0.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
