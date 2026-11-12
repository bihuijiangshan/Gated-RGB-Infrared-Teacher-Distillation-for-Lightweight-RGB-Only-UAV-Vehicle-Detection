from __future__ import annotations

import torch
from torch import Tensor, nn

from rgbt_uav.models.fusion import ReliabilityGatedFusion


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class PyramidBranch(nn.Module):
    """Compact YOLO-style downsampling branch exposing P3/P4/P5."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels, stride=2)
        self.p2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.p3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.p4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        self.p5 = ConvBlock(base_channels * 8, base_channels * 16, stride=2)

    def forward(self, image: Tensor) -> list[Tensor]:
        value = self.stem(image)
        value = self.p2(value)
        p3 = self.p3(value)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]


class ShallowBranch(nn.Module):
    """Modality-specific stem that ends at stride-8 P3."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=2),
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.layers(image)


class DetectionHead(nn.Module):
    """Raw three-scale anchor-free prediction head for loss/NMS integration."""

    def __init__(self, channels: list[int], num_classes: int, reg_max: int) -> None:
        super().__init__()
        output_channels = num_classes + 4 * reg_max
        self.heads = nn.ModuleList(
            nn.Conv2d(channel, output_channels, kernel_size=1) for channel in channels
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != len(self.heads):
            raise ValueError("feature level count does not match detection heads")
        return [head(feature) for head, feature in zip(self.heads, features, strict=True)]


def _rgb_quality(rgb: Tensor) -> Tensor:
    mean = rgb.mean(dim=(1, 2, 3))
    contrast = rgb.std(dim=(1, 2, 3), unbiased=False)
    return torch.stack((mean, contrast), dim=1)


def _validate_modalities(rgb: Tensor, ir: Tensor) -> None:
    if rgb.ndim != 4 or ir.ndim != 4:
        raise ValueError("RGB and IR inputs must be BCHW tensors")
    if rgb.shape[0] != ir.shape[0] or rgb.shape[2:] != ir.shape[2:]:
        raise ValueError("RGB and IR inputs must have the same batch and spatial size")
    if rgb.shape[1] != 3 or ir.shape[1] != 1:
        raise ValueError("expected 3-channel RGB and 1-channel IR inputs")

class AddFusionRGBTDetector(nn.Module):
    """Simple dual-stream RGB-IR baseline with feature addition at P3-P5."""

    def __init__(
        self,
        *,
        num_classes: int,
        base_channels: int = 32,
        reg_max: int = 16,
    ) -> None:
        super().__init__()

        channels = [
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.rgb_backbone = PyramidBranch(3, base_channels)
        self.ir_backbone = PyramidBranch(1, base_channels)
        self.head = DetectionHead(channels, num_classes, reg_max)

    def forward(self, rgb: Tensor, ir: Tensor) -> dict[str, list[Tensor]]:
        _validate_modalities(rgb, ir)

        rgb_features = self.rgb_backbone(rgb)
        ir_features = self.ir_backbone(ir)

        fused_features = [
            rgb_feature + ir_feature
            for rgb_feature, ir_feature in zip(
                rgb_features,
                ir_features,
                strict=True,
            )
        ]

        return {
            "predictions": self.head(fused_features),
            "features": fused_features,
        }

class TeacherRGBTDetector(nn.Module):
    """Full dual-stream teacher with gated fusion at P3, P4, and P5."""

    def __init__(self, *, num_classes: int, base_channels: int = 32, reg_max: int = 16) -> None:
        super().__init__()
        channels = [base_channels * 4, base_channels * 8, base_channels * 16]
        self.rgb_backbone = PyramidBranch(3, base_channels)
        self.ir_backbone = PyramidBranch(1, base_channels)
        self.fusions = nn.ModuleList(ReliabilityGatedFusion(channel) for channel in channels)
        self.head = DetectionHead(channels, num_classes, reg_max)

    def forward(self, rgb: Tensor, ir: Tensor) -> dict[str, list[Tensor]]:
        _validate_modalities(rgb, ir)
        quality = _rgb_quality(rgb)
        rgb_features = self.rgb_backbone(rgb)
        ir_features = self.ir_backbone(ir)
        fused_features: list[Tensor] = []
        gates: list[Tensor] = []
        for module, rgb_feature, ir_feature in zip(
            self.fusions, rgb_features, ir_features, strict=True
        ):
            fused, weights = module(rgb_feature, ir_feature, quality)
            fused_features.append(fused)
            gates.append(weights)
        return {
            "predictions": self.head(fused_features),
            "features": fused_features,
            "gates": gates,
        }


class StudentRGBTDetector(nn.Module):
    """Shallow dual-stream student that becomes single-stream after P3 fusion."""

    def __init__(self, *, num_classes: int, base_channels: int = 16, reg_max: int = 16) -> None:
        super().__init__()
        channels = [base_channels * 4, base_channels * 8, base_channels * 16]
        self.rgb_shallow = ShallowBranch(3, base_channels)
        self.ir_shallow = ShallowBranch(1, base_channels)
        self.p3_fusion = ReliabilityGatedFusion(channels[0])
        self.shared_p4 = ConvBlock(channels[0], channels[1], stride=2)
        self.shared_p5 = ConvBlock(channels[1], channels[2], stride=2)
        self.head = DetectionHead(channels, num_classes, reg_max)

    def forward(self, rgb: Tensor, ir: Tensor) -> dict[str, list[Tensor]]:
        _validate_modalities(rgb, ir)
        quality = _rgb_quality(rgb)
        rgb_p3 = self.rgb_shallow(rgb)
        ir_p3 = self.ir_shallow(ir)
        p3, weights = self.p3_fusion(rgb_p3, ir_p3, quality)
        p4 = self.shared_p4(p3)
        p5 = self.shared_p5(p4)
        features = [p3, p4, p5]
        return {
            "predictions": self.head(features),
            "features": features,
            "gates": [weights],
        }

