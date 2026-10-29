"""Kaggleネイティブ・augmentation強化プローブ v2 のビルダー(fold0のみ)。

## 背景(V38からの引き継ぎ、改訂8)

V38(mosaic 0.20→0.75 + copy_paste 0.3、epoch 70据え置き)はfold0 OOF PQ 0.4290で
不合格だった(基準0.4393)。生YOLO(refinerなし)でもV30基準を下回っており、
検出器そのものが悪化していた。ただしV35の70epochスケジュールをそのまま
流用しており、augmentationを強めたのにepoch数を増やしていなかった。
YOLOの標準運用は`close_mosaic`で終盤にmosaicを切って収束させる設計であり、
augmentationが強いほど収束に必要なepoch数も増えるのが一般的。70epoch据え置きは
収束不足だった可能性が高い。

## v2での変更点

- `epochs`: 70 → **100**(augmentation強化分の収束時間を確保)
- `warmup_epochs`: 3 → **5**(強いaugmentationによる序盤の不安定化を緩和)
- mosaic/copy_paste(0.75/0.3)・その他ハイパーパラメータはV38と同一

## なぜfold0だけか

3-fold全部を70 epochで回すのは高い。「単一foldのbestを一般化しない」という
このプロジェクトの方針どおり、**採用判定はしない**。fold0 OOF PQがV35の
fold0実測 0.4393 を上回ったときだけ fold1/2 に進む(P2-2/P2-3と同じゲート)。

## Kaggleネイティブ(Drive不要)

前処理済み画像・fold分割・refiner checkpointは Kaggle Dataset
`shingo257/solar-filament-v35-artifacts` から読む。COCO annotationは
competition `filament-segmentation-2026` を直接アタッチする。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v40_kaggle_augmentation_v2.ipynb"


try:
    from .build_v40_kaggle_augmentation_v2_cells import cells
except ImportError:  # Direct `python scratch/build_...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_v40_kaggle_augmentation_v2_cells import cells

for index, cell in enumerate(cells):
    cell["id"] = f"v40p-{index:02d}"
    if cell["cell_type"] != "code":
        continue
    stripped = re.sub(r"^[%!].*$", "", cell["source"], flags=re.M)
    ast.parse(stripped)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
generated = json.loads(OUTPUT.read_text(encoding="utf-8"))
text = "\n".join(
    "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    for cell in generated["cells"]
)
assert "class MediSegUNet" in text
assert "class PanopticQuality" in text
assert "MOSAIC = 0.75" in text
assert "COPY_PASTE = 0.3" in text
assert "solar-filament-v35-artifacts" in text
assert "P100" in text
print(OUTPUT)
