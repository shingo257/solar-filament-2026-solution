"""技術報告書用のセグメンテーション例図を、既存showcaseアセット(自社モデルの予測ポリゴン、
GTではない)から生成する。ローカルのみで完結、Colab不要。
出典: deploy_site/assets/solar-filament/detections/{case01_color.jpg, manifest.json}
(build_showcase_data.pyが生成、YOLO+refinerパイプラインの予測結果)。
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "deploy_site" / "assets" / "solar-filament" / "detections"
OUT = ROOT / "report" / "figures" / "segmentation_example.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
case = next(c for c in manifest if c["stem"] == "case01")

image = np.array(Image.open(ASSETS / "case01_color.jpg").convert("RGB"))
height, width = image.shape[:2]

# build_showcase_data.pyはポリゴン座標を0-100(パーセント、SVG viewBox用)へ
# 正規化して保存している(scale = 100.0/thumb_width)。720pxサムネイルへ
# 重畳するには逆変換(x100分率 -> ピクセル)が必要。
def to_pixels(poly):
    pts = np.array(poly, dtype=np.float64)
    return pts * (width / 100.0)

# 全個体を包含する領域をpad付きで切り出す
all_pts = np.concatenate([to_pixels(poly) for inst in case["instances"] for poly in inst["polygons"]])
pad = 50
x0, y0 = max(0, int(all_pts[:, 0].min()) - pad), max(0, int(all_pts[:, 1].min()) - pad)
x1, y1 = min(width, int(all_pts[:, 0].max()) + pad), min(height, int(all_pts[:, 1].max()) + pad)

fig, ax = plt.subplots(figsize=(7, 7 * (y1 - y0) / (x1 - x0)), dpi=150)
ax.imshow(image[y0:y1, x0:x1])
colors = plt.cm.tab10(np.linspace(0, 1, len(case["instances"])))
for idx, inst in enumerate(case["instances"]):
    for poly in inst["polygons"]:
        pts = to_pixels(poly)
        ax.plot(pts[:, 0] - x0, pts[:, 1] - y0, color=colors[idx], linewidth=1.8)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout(pad=0.2)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT} ({case['file_name']}, {len(case['instances'])} instances)")
