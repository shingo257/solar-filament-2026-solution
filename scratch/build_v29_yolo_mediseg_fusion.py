import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_kaggle_ensemble_mediseg_v26_realdata_yolo_refiner.ipynb"
OUTPUT = ROOT / "solar_kaggle_ensemble_mediseg_v29_yolo_mediseg_fusion.ipynb"
MEDISEG = (ROOT / "src" / "mediseg_models.py").read_text(encoding="utf-8")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

notebook["cells"][0]["source"] = """# Solar Filament Segmentation - V29 YOLO + MediSeg fusion

V28 proved that the repository's MediSegUNet learns the real MAGFiLO masks, but
its semantic connected-component export scored only 0.19 because the competition
matches individual filament instances. V29 uses YOLO only to identify separate
filament candidates and applies the repository's 31M-parameter MediSegUNet to
refine each candidate boundary.

**Kaggle settings:** GPU on and Internet on. Internet is required to install
Ultralytics and download the YOLO11 pretrained weights. The synthetic
`best_solar_unet` dataset is not used.

The instance pipeline is adapted from the Apache-2.0 public notebook
[YOLO + U-Net | Solar Filament Segmentation](https://www.kaggle.com/code/ektarr/yolo-u-net-solar-filament-segmentation),
while the crop refiner and loss are replaced by this repository's MediSegUNet
and CombinedLoss.
"""

notebook["cells"][9]["source"] = (
    "# MediSeg-AI crop refiner (repository implementation)\n" + MEDISEG
)

for cell in notebook["cells"]:
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)

    source = source.replace(
        "submission_v26_realdata_yolo_refiner.csv",
        "submission_v29_yolo_mediseg_fusion.csv",
    )
    source = source.replace("best_refiner.pt", "best_mediseg_refiner.pt")
    source = source.replace(
        "CropUNet(pretrained=True)",
        "MediSegUNet(in_channels=3, out_channels=1, bilinear=False)",
    )
    source = source.replace(
        "CropUNet(pretrained=False)",
        "MediSegUNet(in_channels=3, out_channels=1, bilinear=False)",
    )
    source = source.replace(
        "batch_size=16, shuffle=True, num_workers=2, pin_memory=True",
        "batch_size=4, shuffle=True, num_workers=0, pin_memory=True",
    )
    source = source.replace(
        "batch_size=32, shuffle=False, num_workers=2, pin_memory=True",
        "batch_size=4, shuffle=False, num_workers=0, pin_memory=True",
    )
    source = source.replace(
        "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)",
        "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)\n"
        "    criterion = CombinedLoss(bce_weight=0.35, dice_weight=0.65)",
    )
    source = source.replace(
        "loss = 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, masks) + 0.5 * soft_dice_loss(logits, masks)",
        "loss = criterion(logits, masks)",
    )
    source = source.replace(
        "best_refiner_dice = train_refiner(refiner, train_crop_loader, valid_crop_loader, epochs=8, out_path=\"best_mediseg_refiner.pt\")",
        "best_refiner_dice = train_refiner(refiner, train_crop_loader, valid_crop_loader, epochs=12, out_path=\"best_mediseg_refiner.pt\")\n"
        "print({'best_mediseg_refiner_dice': best_refiner_dice})",
    )

    cell["source"] = source
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(notebook, handle, ensure_ascii=False, indent=1)
    handle.write("\n")

print(OUTPUT)
