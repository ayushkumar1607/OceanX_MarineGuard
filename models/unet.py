"""
MarineGuard — Lightweight U-Net for CPU training
Small enough to train in ~1-2 minutes on a normal laptop CPU.

Params: ~1.9M (vs 31M for the full version)

Input:  (B, 1, 128, 128) SAR grayscale
Output: (B, 1, 128, 128) sigmoid probability map
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple


class DoubleConv(nn.Module):
    """(Conv 3x3 → BN → ReLU) × 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for CPU-friendly oil slick segmentation.
    Feature channels reduced from (64,128,256,512) to (16,32,64,128).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: Tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # Encoder
        self.enc1 = DoubleConv(in_channels, features[0])
        self.enc2 = DoubleConv(features[0], features[1])
        self.enc3 = DoubleConv(features[1], features[2])
        self.enc4 = DoubleConv(features[2], features[3])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3] * 2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0])

        # Output head
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))

    def get_architecture_summary(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "model_name": "U-Net (Lightweight, CPU-friendly)",
            "input": f"({self.in_channels}, 128, 128) — Sentinel-1 SAR",
            "output": f"({self.out_channels}, 128, 128) — Slick probability map",
            "encoder_features": list(self.features),
            "total_params": f"{n_params / 1e6:.2f}M",
            "loss_function": "BCE + Dice Loss",
            "optimizer": "Adam (lr=1e-3)",
            "training_data": "Synthetic SAR oil spill dataset",
        }

    # Fallback used only if trained weights are unavailable
    def simulate_inference(self, sar_shape: Tuple[int, int],
                           slick_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        H, W = sar_shape
        prob_map = np.random.uniform(0.0, 0.05, (H, W)).astype(np.float32)

        if slick_params is None:
            return prob_map

        center_r = slick_params.get("center_row", H // 2)
        center_c = slick_params.get("center_col", W // 2)
        radius = slick_params.get("radius", 50)
        confidence = slick_params.get("confidence", 0.85)
        elongation = slick_params.get("elongation", 1.5)
        angle = slick_params.get("angle", 30)

        y, x = np.ogrid[:H, :W]
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        x_rot = (x - center_c) * cos_a + (y - center_r) * sin_a
        y_rot = -(x - center_c) * sin_a + (y - center_r) * cos_a
        dist = np.sqrt((x_rot / (radius * elongation)) ** 2 + (y_rot / radius) ** 2)

        slick_mask = np.exp(-2 * dist ** 2) * confidence
        noise = np.random.normal(0, 0.08, (H, W))
        slick_mask = np.clip(slick_mask + noise * (slick_mask > 0.1), 0, 1)
        return np.maximum(prob_map, slick_mask).astype(np.float32)