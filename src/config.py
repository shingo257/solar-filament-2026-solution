import os
import torch

class Config:
    # Competition Info
    COMPETITION_NAME = "filament-segmentation-2026"
    SEED = 42

    # Paths (Auto adapt for Colab / Local)
    IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")
    
    if IS_COLAB:
        BASE_DIR = "/content/solar_project"
        DATA_DIR = "/content/drive/MyDrive/kaggle_solar_project/data"
        OUTPUT_DIR = "/content/drive/MyDrive/kaggle_solar_project/output"
    else:
        BASE_DIR = r"C:\work\code\ER-FlowScan\Solar_Filament_Segmentation"
        DATA_DIR = os.path.join(BASE_DIR, "data")
        OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    # Image & Model Parameters
    IMAGE_SIZE = (512, 512)
    IN_CHANNELS = 3
    NUM_CLASSES = 1  # Binary segmentation

    # Segmentation Architecture (segmentation_models_pytorch)
    # Options: 'Unet', 'UnetPlusPlus', 'DeepLabV3Plus', 'FPN'
    ARCHITECTURES = "UnetPlusPlus"
    BACKBONE = "efficientnet-b3"
    ENCODER_WEIGHTS = "imagenet"

    # Training Hyperparameters (Colab Pro+ A100 optimized)
    BATCH_SIZE = 16 if IS_COLAB else 4
    NUM_WORKERS = 4 if IS_COLAB else 0
    EPOCHS = 15
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_FOLDS = 5

    # Acceleration
    USE_AMP = torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
