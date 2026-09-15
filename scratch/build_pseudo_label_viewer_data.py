"""pseudo-label(3-fold YOLO WBF+refiner shrink@0.45で生成)を人間が目視確認する
ためのビューア用データを作る。near_miss_viewerと同じ構成のアーティファクトへ渡す。

使い方:
    python scratch/build_pseudo_label_viewer_data.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

MANIFEST = Path("output/pseudo_label_raw/pseudo_label_manifest.json")
CACHE_DIR = Path("output/pseudo_label_raw/cache")
IMAGE_DIR = Path("output/pseudo_label_raw/preprocessed")
OUT_JSON = Path("output/pseudo_label_viewer/pseudo_label_cases.json")
THUMB_SIZE = 640


def load_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32), np.zeros(0, np.int32)
    masks = [np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed]
    return masks, data["confidences"].astype(np.float32), data["agreement"].astype(np.int32)


def mask_to_polygons(mask, scale):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        # 100x100の相対座標(SVG viewBoxで使う)
        points = (approx.astype(np.float32) * scale).tolist()
        polygons.append(points)
    return polygons


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    for entry in manifest["images"]:
        if entry["n_kept"] == 0:
            continue
        name = entry["file_name"]
        cache_path = CACHE_DIR / f"{Path(name).stem}.npz"
        image_path = IMAGE_DIR / name
        if not cache_path.exists() or not image_path.exists():
            continue
        masks, confidences, agreement = load_cache(cache_path)
        if not masks:
            continue

        # 3チャンネル前処理画像から先頭チャンネル(raw強調)をグレースケール表示に使う
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        thumb = cv2.resize(image, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumb_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

        scale = 100.0 / width  # viewBox 0..100 相対座標
        instances = []
        for mask, conf, agree in zip(masks, confidences, agreement):
            polygons = mask_to_polygons(mask, scale)
            if not polygons:
                continue
            instances.append({
                "confidence": round(float(conf), 3),
                "agreement": int(agree),
                "area_px": int(mask.sum()),
                "polygons": polygons,
            })
        if not instances:
            continue

        cases.append({
            "file_name": name,
            "thumb_b64": thumb_b64,
            "n_instances": len(instances),
            "instances": instances,
        })

    OUT_JSON.write_text(json.dumps({
        "scope": "3-fold YOLO WBF(2/3合意)+refiner(shrink@0.45確定構成)で生成したpseudo-label",
        "n_images_with_labels": len(cases),
        "n_total_instances": sum(c["n_instances"] for c in cases),
        "cases": cases,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"cases: {len(cases)}, total instances: {sum(c['n_instances'] for c in cases)}")
    print(f"saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
