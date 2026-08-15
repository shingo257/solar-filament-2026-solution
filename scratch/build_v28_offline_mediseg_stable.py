import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_kaggle_ensemble_mediseg_v27_offline_realdata.ipynb"
OUTPUT = ROOT / "solar_kaggle_ensemble_mediseg_v28_offline_realdata_stable.ipynb"

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

for cell in notebook["cells"]:
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    source = source.replace(
        "Solar Filament Segmentation - V27 offline real-data MediSeg",
        "Solar Filament Segmentation - V28 stable offline real-data MediSeg",
    )
    source = source.replace(
        "This is the MediSeg-based correction of V26.",
        "This is the stable MediSeg-based correction of V27.",
    )
    source = source.replace(
        "NUM_WORKERS = 2",
        "NUM_WORKERS = 0  # Required for stable Kaggle/Papermill finalization",
    )
    source = source.replace(
        "submission_v27_offline_realdata_mediseg.csv",
        "submission_v28_offline_realdata_mediseg.csv",
    )
    source = source.replace(
        "best_v27_mediseg_unet.pth",
        "best_v28_mediseg_unet.pth",
    )
    cell["source"] = source
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(notebook, handle, ensure_ascii=False, indent=1)
    handle.write("\n")

print(OUTPUT)
