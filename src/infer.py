import os
import torch
import pandas as pd
import numpy as np
from src.config import Config
from src.model import build_model
from src.utils import rle_encode

@torch.no_grad()
def run_inference():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = config.DEVICE

    print(f"=== Starting Solar Filament Inference on Device: {device} ===")

    model_path = os.path.join(config.OUTPUT_DIR, "best_solar_unet.pth")
    model = build_model(config)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from: {model_path}")
    else:
        print("Warning: Model checkpoint not found. Running inference with initial weights.")

    model.to(device)
    model.eval()

    # Generate Dummy RLE Submissions for Test Verification
    print("Generating sample submission file with RLE encoding...")
    test_ids = [f"solar_img_{i:04d}.png" for i in range(50)]
    rle_list = []

    for _ in test_ids:
        # Dummy prediction mask
        dummy_mask = np.zeros((512, 512), dtype=np.uint8)
        dummy_mask[150:250, 150:250] = 1
        rle = rle_encode(dummy_mask)
        rle_list.append(rle)

    sub_df = pd.DataFrame({
        "id": test_ids,
        "rle_mask": rle_list
    })

    sub_path = os.path.join(config.OUTPUT_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved successfully to: {sub_path}")
    print(sub_df.head())

if __name__ == "__main__":
    run_inference()
