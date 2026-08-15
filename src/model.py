import torch
import torch.nn as nn
import torch.nn.functional as F

class FallbackUNet2D(nn.Module):
    """Fallback PyTorch 2D UNet if segmentation_models_pytorch is not installed."""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final_conv(d1)


def build_model(config):
    """
    Build 2D Segmentation Model using smp or MediSeg-AI custom architectures.
    """
    arch_name = getattr(config, 'ARCHITECTURES', 'UnetPlusPlus')
    
    if arch_name == 'MediSegUNet':
        from .mediseg_models import MediSegUNet
        print("Built MediSeg Custom Model: MediSegUNet (31M params)")
        return MediSegUNet(in_channels=config.IN_CHANNELS, out_channels=config.NUM_CLASSES)
    
    elif arch_name == 'MediSegDeepLabV3Plus':
        from .mediseg_models import MediSegDeepLabV3Plus
        print("Built MediSeg Custom Model: MediSegDeepLabV3Plus (with ASPP)")
        return MediSegDeepLabV3Plus(in_channels=config.IN_CHANNELS, out_channels=config.NUM_CLASSES)

    try:
        import segmentation_models_pytorch as smp
        arch = getattr(smp, arch_name, smp.UnetPlusPlus)
        model = arch(
            encoder_name=config.BACKBONE,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=config.IN_CHANNELS,
            classes=config.NUM_CLASSES,
        )
        print(f"Built SMP Model: {arch_name} (Backbone: {config.BACKBONE})")
        return model
    except ImportError:
        print("segmentation_models_pytorch not installed. Using Fallback PyTorch UNet2D.")
        from .mediseg_models import MediSegUNet
        return MediSegUNet(in_channels=config.IN_CHANNELS, out_channels=config.NUM_CLASSES)

