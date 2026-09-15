"""ホームページ掲載用のショーケースデータを作る。

各画像について、生データ(グレースケール)と技術強調表示(3チャンネル、
カラー)の両方を用意し、ページ側でトグル切り替えできるようにする。

安全性の配慮:
- 画像はGONGアーカイブの生データ(NSO/AURA提供、公開データ)由来のグレー
  スケール画像、および自社パイプラインの技術的前処理(3チャンネル強調表示)
  のみを使う。コンペ提供のMAGFiLO正解ラベル(CC BY-NC 4.0、非商用)は
  一切使用しない。
- 表示する検出結果は自社モデル(YOLO+refiner)が独自に生成したものであり、
  コンペのground truthではない。
- 性能数値(PQ)は「公開ベンチマークでの自社アルゴリズムの評価結果」という
  事実の記述であり、ベンチマークのデータ自体を再配布するものではない。

使い方:
    python scratch/build_showcase_data.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

MANIFEST = Path("output/pseudo_label_raw/pseudo_label_manifest.json")
CACHE_DIR = Path("output/pseudo_label_raw/cache")
IMAGE_DIR_GRAY = Path("output/pseudo_label_raw/images")          # 自然なグレースケール
IMAGE_DIR_ENHANCED = Path("output/pseudo_label_raw/preprocessed")  # 3ch技術強調(モデル入力そのもの)
OUT_JSON = Path("output/showcase/showcase_cases.json")
THUMB_SIZE = 720
TOP_N = 10


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
        polygons.append((approx.astype(np.float32) * scale).tolist())
    return polygons


def encode_thumb(path, color):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)
    thumb = cv2.resize(image, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer.tobytes()).decode("ascii"), image.shape[1]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    entries = [m for m in manifest["images"] if m["n_kept"] > 0]
    entries.sort(key=lambda m: -m["n_kept"])
    entries = entries[:TOP_N]

    cases = []
    for entry in entries:
        name = entry["file_name"]
        cache_path = CACHE_DIR / f"{Path(name).stem}.npz"
        gray_path = IMAGE_DIR_GRAY / name
        enhanced_path = IMAGE_DIR_ENHANCED / name
        if not (cache_path.exists() and gray_path.exists() and enhanced_path.exists()):
            continue
        masks, confidences, agreement = load_cache(cache_path)
        if not masks:
            continue

        gray_b64, width = encode_thumb(gray_path, color=False)
        enhanced_b64, _ = encode_thumb(enhanced_path, color=True)

        scale = 100.0 / width
        instances = []
        for mask in masks:
            polygons = mask_to_polygons(mask, scale)
            if polygons:
                instances.append({"polygons": polygons})
        if not instances:
            continue

        stamp = name[:14]
        date_str = f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"

        cases.append({
            "file_name": name,
            "date": date_str,
            "thumb_gray_b64": gray_b64,
            "thumb_color_b64": enhanced_b64,
            "n_instances": len(instances),
            "instances": instances,
        })

    OUT_JSON.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    print(f"cases: {len(cases)}")
    print(f"saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
