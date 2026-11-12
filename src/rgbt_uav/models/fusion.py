from __future__ import annotations

import torch
from torch import Tensor, nn


class ReliabilityGatedFusion(nn.Module):
    """Channel-wise two-modality gate with low-cost spatial refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        hidden = max(channels // 4, 8)
        self.channels = channels
        self.gate = nn.Sequential(
            nn.Linear(2 * channels + 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * channels),
        )
        self.mix = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        self.refine = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, rgb: Tensor, ir: Tensor, rgb_quality: Tensor) -> tuple[Tensor, Tensor]:
        if rgb.shape != ir.shape:
            raise ValueError("RGB and IR features must have the same shape")
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(
                f"expected BCHW features with {self.channels} channels, got {tuple(rgb.shape)}"
            )
        if rgb_quality.shape != (rgb.shape[0], 2):
            raise ValueError("rgb_quality must have shape [batch, 2] for mean and contrast")

        rgb_descriptor = rgb.mean(dim=(2, 3))
        ir_descriptor = ir.mean(dim=(2, 3))
        descriptor = torch.cat((rgb_descriptor, ir_descriptor, rgb_quality), dim=1)
        logits = self.gate(descriptor).reshape(rgb.shape[0], 2, self.channels, 1, 1)
        weights = logits.softmax(dim=1)
        weighted_rgb = weights[:, 0] * rgb
        weighted_ir = weights[:, 1] * ir
        mixed = self.mix(torch.cat((weighted_rgb, weighted_ir), dim=1))
        residual = self.refine(weighted_rgb + weighted_ir)
        fused = self.activation(self.norm(mixed + residual))
        return fused, weights

