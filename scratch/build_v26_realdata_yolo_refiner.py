import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_RESPONSE = ROOT / "scratch" / "public_yolo_unet_baseline.zip"
OUTPUT = ROOT / "solar_kaggle_ensemble_mediseg_v26_realdata_yolo_refiner.ipynb"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


with API_RESPONSE.open("r", encoding="utf-8") as handle:
    api_payload = json.load(handle)

notebook = json.loads(api_payload["blob"]["source"])
cells = deepcopy(notebook["cells"])

for cell in cells:
    cell["metadata"] = {}
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

cells[0]["source"] = '%pip install -q "ultralytics>=8.3,<9" pycocotools'

for cell in cells:
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)

    if "class CFG:" in source:
        source = source.replace("    EPOCHS = 1 if DEBUG else 30", "    EPOCHS = 1 if DEBUG else 30")
        source = source.replace("    MODEL = \"yolo11s-seg.pt\"", "    MODEL = \"yolo11s-seg.pt\"")
        cell["source"] = source

    if 'submission.to_csv("submission.csv", index=False)' in source:
        source = source.replace(
            'submission.to_csv("submission.csv", index=False)',
            'OUTPUT_PATH = "/kaggle/working/submission_v26_realdata_yolo_refiner.csv"\n'
            'submission.to_csv(OUTPUT_PATH, index=False, lineterminator="\\n")',
        )
        source = source.replace(
            'print(f"Total predicted filaments: {len(submission)}")',
            'print(f"Total predicted filaments: {len(submission)}")\nprint("Saved:", OUTPUT_PATH)',
        )
        cell["source"] = source

title = markdown_cell(
    "# Solar Filament Segmentation - V26 real-data YOLO + U-Net refiner\n\n"
    "This version fixes the `0.00` submission at its source: the previous checkpoint was trained "
    "only on synthetic random images and produced one semantic mask per image. V26 trains on the "
    "competition's real COCO annotations, predicts individual filament instances, refines each "
    "instance boundary, and uses `pycocotools` for the official compressed RLE.\n\n"
    "**Kaggle settings:** GPU on, Internet on. The attached `best_solar_unet` dataset is not used.\n\n"
    "Implementation adapted from the public Apache-2.0 Kaggle notebook "
    "[YOLO + U-Net | Solar Filament Segmentation](https://www.kaggle.com/code/ektarr/yolo-u-net-solar-filament-segmentation), "
    "which reported a public score of 0.69."
)

path_check = code_cell(
    "# Fail early if the competition input layout is not attached correctly.\n"
    "required_paths = [CFG.TRAIN_DIR, CFG.TEST_DIR, CFG.TRAIN_JSON]\n"
    "missing_paths = [path for path in required_paths if not os.path.exists(path)]\n"
    "if missing_paths:\n"
    "    raise FileNotFoundError(f'Missing competition inputs: {missing_paths}')\n"
    "print({\n"
    "    'train_images': len(os.listdir(CFG.TRAIN_DIR)),\n"
    "    'test_images': len(os.listdir(CFG.TEST_DIR)),\n"
    "    'annotations': CFG.TRAIN_JSON,\n"
    "})"
)

validation = code_cell(
    "# Final structural validation with the official pycocotools decoder.\n"
    "from pathlib import Path\n"
    "checked = pd.read_csv(OUTPUT_PATH, keep_default_na=False)\n"
    "expected_columns = ['filament_id', 'segmentation_rle']\n"
    "assert checked.columns.tolist() == expected_columns\n"
    "assert len(checked) > 0\n"
    "assert not checked.isna().any().any()\n"
    "assert not checked['filament_id'].duplicated().any()\n"
    "assert not checked['segmentation_rle'].str.strip().eq('').any()\n"
    "test_ids = {Path(name).stem for name in test_files}\n"
    "submitted_ids = {value.rsplit('_', 1)[0] for value in checked['filament_id']}\n"
    "assert submitted_ids <= test_ids\n"
    "areas = []\n"
    "for counts in tqdm(checked['segmentation_rle'], desc='official RLE validation'):\n"
    "    decoded = mask_utils.decode({\n"
    "        'size': [2048, 2048],\n"
    "        'counts': counts.encode('utf-8'),\n"
    "    })\n"
    "    assert decoded.shape == (2048, 2048)\n"
    "    area = int(decoded.sum())\n"
    "    assert area > 0\n"
    "    areas.append(area)\n"
    "print({\n"
    "    'validated_csv': OUTPUT_PATH,\n"
    "    'rows': len(checked),\n"
    "    'images_with_predictions': len(submitted_ids),\n"
    "    'test_images': len(test_ids),\n"
    "    'area_min': min(areas),\n"
    "    'area_median': int(np.median(areas)),\n"
    "    'area_max': max(areas),\n"
    "})"
)

# Config is the third original cell (after pip and imports), so insert the path
# check immediately after it.
cells = [title, *cells[:3], path_check, *cells[3:], validation]

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}
notebook["nbformat"] = 4
notebook["nbformat_minor"] = 5

with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(notebook, handle, ensure_ascii=False, indent=1)
    handle.write("\n")

print(OUTPUT)
