import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_BUILDER = Path(__file__).with_name("build_v30_disjoint_masks.py")
OUTPUT = ROOT / "solar_filament_v30_prize_colab_instance_ensemble.ipynb"

runpy.run_path(str(PREVIOUS_BUILDER), run_name="__main__")
notebook = json.loads(OUTPUT.read_text(encoding="utf-8"))

notebook["cells"][-1]["source"] = '''# 既存V30 CSVの重複だけを修復する完全独立セル（ランタイム再接続後も実行可能）
%pip install -q pycocotools

import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm

drive.mount("/content/drive")

ARTIFACT_ROOT = Path("/content/drive/MyDrive/solar_filament_v30/artifacts")
SOURCE_CANDIDATES = [
    Path("/content/submission_v30_prize_instance_ensemble.csv"),
    ARTIFACT_ROOT / "submission_v30_prize_instance_ensemble.csv",
]
SOURCE_CSV = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_CSV is None:
    raise FileNotFoundError(f"元のV30 CSVが見つかりません: {SOURCE_CANDIDATES}")

FIXED_CSV = Path("/content/submission_v30b_disjoint_instance_ensemble.csv")
MIN_AREA = 28  # V30の検証データで選択された値
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

source = pd.read_csv(SOURCE_CSV, keep_default_na=False)
assert source.columns.tolist() == ["filament_id", "segmentation_rle"]
assert not source.isna().any().any()

grouped = defaultdict(list)
for row in source.itertuples(index=False):
    image_id, suffix = row.filament_id.rsplit("_", 1)
    grouped[image_id].append((int(suffix), row.segmentation_rle))

fixed_rows = []
removed_pixels = 0
dropped_masks = 0
for image_id, records in tqdm(grouped.items(), desc="個体maskを排他化"):
    occupied = np.zeros((2048, 2048), dtype=bool)
    counter = 1
    # 元CSVは信頼度順なので、番号が小さい個体を優先する。
    for _, counts in sorted(records):
        mask = mask_utils.decode({
            "size": [2048, 2048],
            "counts": counts.encode("utf-8"),
        }).astype(bool)
        removed_pixels += int(np.logical_and(mask, occupied).sum())
        mask &= ~occupied
        if mask.sum() < MIN_AREA:
            dropped_masks += 1
            continue
        occupied |= mask
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        fixed_rows.append({
            "filament_id": f"{image_id}_{counter}",
            "segmentation_rle": encoded["counts"].decode("utf-8"),
        })
        counter += 1

fixed = pd.DataFrame(fixed_rows, columns=["filament_id", "segmentation_rle"])
fixed.to_csv(FIXED_CSV, index=False, lineterminator="\\n")
DRIVE_COPY = ARTIFACT_ROOT / FIXED_CSV.name
shutil.copy2(FIXED_CSV, DRIVE_COPY)

# 同一画像内の重複が完全にゼロであることを再復号して確認する。
verified_masks = 0
image_keys = fixed["filament_id"].str.rsplit("_", n=1).str[0]
for image_id, records in fixed.groupby(image_keys, sort=False):
    occupied = np.zeros((2048, 2048), dtype=bool)
    for counts in records["segmentation_rle"]:
        mask = mask_utils.decode({
            "size": [2048, 2048],
            "counts": counts.encode("utf-8"),
        }).astype(bool)
        assert not np.logical_and(occupied, mask).any(), image_id
        occupied |= mask
        verified_masks += 1

assert verified_masks == len(fixed)
assert not fixed["filament_id"].duplicated().any()
assert not fixed["segmentation_rle"].str.strip().eq("").any()

print({
    "元ファイル": str(SOURCE_CSV),
    "修復前の行数": len(source),
    "修復後の行数": len(fixed),
    "除去した重複画素": removed_pixels,
    "空または極小になり除外したmask": dropped_masks,
    "個体間重複": 0,
    "提出ファイル": str(FIXED_CSV),
    "Driveコピー": str(DRIVE_COPY),
})
'''

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
