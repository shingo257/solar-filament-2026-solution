import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_BUILDER = Path(__file__).with_name("build_v30_japanese_ram_adaptive.py")
OUTPUT = ROOT / "solar_filament_v30_prize_colab_instance_ensemble.ipynb"

runpy.run_path(str(PREVIOUS_BUILDER), run_name="__main__")
notebook = json.loads(OUTPUT.read_text(encoding="utf-8"))


def replace(cell_index, old, new):
    source = notebook["cells"][cell_index]["source"]
    if old not in source:
        raise RuntimeError(f"セル{cell_index}: 置換対象が見つかりません")
    notebook["cells"][cell_index]["source"] = source.replace(old, new, 1)


replace(
    11,
    '''    final_masks.sort(key=lambda item: item[0], reverse=True)
    image_id = Path(file_name).stem
    for index, (_, mask) in enumerate(final_masks, start=1):
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rows.append({
            "filament_id": f"{image_id}_{index}",
            "segmentation_rle": encoded["counts"].decode("utf-8"),
        })
    audit.append({"file_name": file_name, "proposals": len(proposals), "submitted": len(final_masks)})''',
    '''    # コンペでは同一画像内の個体maskが1ピクセルでも重なると提出エラーになる。
    # 信頼度の高い個体から画素を確保し、後続個体から重複画素を除去する。
    final_masks.sort(key=lambda item: item[0], reverse=True)
    occupied = np.zeros_like(final_masks[0][1], dtype=bool) if final_masks else np.zeros(image.shape[:2], dtype=bool)
    disjoint_masks = []
    overlap_pixels_removed = 0
    for confidence, mask in final_masks:
        overlap_pixels_removed += int(np.logical_and(mask, occupied).sum())
        exclusive = np.logical_and(mask, ~occupied)
        if exclusive.sum() < int(best_params["min_area"]):
            continue
        disjoint_masks.append((confidence, exclusive))
        occupied |= exclusive

    image_id = Path(file_name).stem
    for index, (_, mask) in enumerate(disjoint_masks, start=1):
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rows.append({
            "filament_id": f"{image_id}_{index}",
            "segmentation_rle": encoded["counts"].decode("utf-8"),
        })
    audit.append({
        "file_name": file_name,
        "proposals": len(proposals),
        "before_exclusive": len(final_masks),
        "submitted": len(disjoint_masks),
        "overlap_pixels_removed": overlap_pixels_removed,
    })''',
)

replace(
    12,
    '''areas = []
for counts in tqdm(checked["segmentation_rle"], desc="decode official RLE"):
    decoded = mask_utils.decode({"size": [2048, 2048], "counts": counts.encode("utf-8")})
    assert decoded.shape == (2048, 2048)
    area = int(decoded.sum())
    assert area > 0
    areas.append(area)''',
    '''areas = []
occupied_by_image = {}
for row in tqdm(checked.itertuples(index=False), total=len(checked), desc="公式RLEと個体間重複を検証"):
    decoded = mask_utils.decode({
        "size": [2048, 2048],
        "counts": row.segmentation_rle.encode("utf-8"),
    }).astype(bool)
    assert decoded.shape == (2048, 2048)
    area = int(decoded.sum())
    assert area > 0
    image_id = row.filament_id.rsplit("_", 1)[0]
    occupied = occupied_by_image.setdefault(image_id, np.zeros((2048, 2048), dtype=bool))
    overlap = int(np.logical_and(occupied, decoded).sum())
    assert overlap == 0, f"同一画像内でmaskが重複しています: {image_id}, {overlap} pixels"
    occupied |= decoded
    areas.append(area)''',
)

# 既にV30 CSVを生成済みの場合に、推論をやり直さず修復できる独立セル。
repair_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": '''# 既存V30 CSVの個体mask重複だけを修復する（学習・推論の再実行は不要）
from collections import defaultdict
from pathlib import Path

SOURCE_CSV = Path("/content/submission_v30_prize_instance_ensemble.csv")
if not SOURCE_CSV.exists():
    SOURCE_CSV = CFG.ARTIFACT_ROOT / "submission_v30_prize_instance_ensemble.csv"
FIXED_CSV = Path("/content/submission_v30b_disjoint_instance_ensemble.csv")

source = pd.read_csv(SOURCE_CSV, keep_default_na=False)
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
    # 元CSVは信頼度順に番号付けされているため、小さい番号を優先する。
    for _, counts in sorted(records):
        mask = mask_utils.decode({
            "size": [2048, 2048], "counts": counts.encode("utf-8")
        }).astype(bool)
        removed_pixels += int(np.logical_and(mask, occupied).sum())
        mask &= ~occupied
        if mask.sum() < int(best_params["min_area"]):
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
shutil.copy2(FIXED_CSV, CFG.ARTIFACT_ROOT / FIXED_CSV.name)

# Kaggleへアップロードする前に、同一画像内の重複が完全にゼロであることを確認する。
for image_id, records in fixed.groupby(fixed["filament_id"].str.rsplit("_", n=1).str[0]):
    occupied = np.zeros((2048, 2048), dtype=bool)
    for counts in records["segmentation_rle"]:
        mask = mask_utils.decode({
            "size": [2048, 2048], "counts": counts.encode("utf-8")
        }).astype(bool)
        assert not np.logical_and(occupied, mask).any(), image_id
        occupied |= mask

print({
    "修復前の行数": len(source),
    "修復後の行数": len(fixed),
    "除去した重複画素": removed_pixels,
    "空または極小になり除外したmask": dropped_masks,
    "提出ファイル": str(FIXED_CSV),
    "Driveコピー": str(CFG.ARTIFACT_ROOT / FIXED_CSV.name),
    "個体間重複": 0,
})
''',
}
notebook["cells"].append(repair_cell)

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
