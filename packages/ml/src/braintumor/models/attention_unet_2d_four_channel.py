"""Independent four-modality 2D Attention U-Net experiment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    return next(groups for groups in range(min(maximum, channels), 0, -1) if channels % groups == 0)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class AttentionGate2D(nn.Module):
    """Attention gate that scales encoder skip features using decoder context."""

    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        intermediate = max(1, skip_channels // 2)
        self.skip_projection = nn.Conv2d(
            skip_channels, intermediate, kernel_size=2, stride=2, bias=False
        )
        self.gate_projection = nn.Conv2d(
            gate_channels, intermediate, kernel_size=1, bias=True
        )
        self.activation = nn.ReLU(inplace=True)
        self.coefficient = nn.Sequential(
            nn.Conv2d(intermediate, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        projected_skip = self.skip_projection(skip)
        projected_gate = F.interpolate(
            self.gate_projection(gate),
            size=projected_skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        weights = self.coefficient(
            self.activation(projected_skip + projected_gate)
        )
        weights = F.interpolate(
            weights, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return skip * weights


class DecoderStage2D(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.attention = AttentionGate2D(skip_channels, in_channels)
        self.fusion = ConvBlock2D(in_channels + skip_channels, out_channels)

    def forward(
        self, inputs: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        inputs = F.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        attended_skip = self.attention(skip, inputs)
        return self.fusion(torch.cat((inputs, attended_skip), dim=1))


class AttentionUNet2DFourChannel(nn.Module):
    """2D U-Net with attention-gated skips and four native BraTS classes.

    Input channel order is FLAIR, T1, T1ce, T2. Output channels correspond to
    contiguous training classes 0, 1, 2, 3; class 3 maps back to native label 4.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    ) -> None:
        super().__init__()
        if in_channels != 4:
            raise ValueError("The four-channel experiment requires four inputs.")
        if out_channels != 4:
            raise ValueError("BraTS segmentation requires four output classes.")
        if len(channels) < 3 or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain at least three positive widths.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        previous = in_channels
        for width in channels[:-1]:
            self.encoders.append(ConvBlock2D(previous, width))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            previous = width

        self.bottleneck = ConvBlock2D(channels[-2], channels[-1])
        self.decoders = nn.ModuleList(
            DecoderStage2D(channels[index + 1], channels[index], channels[index])
            for index in range(len(channels) - 2, -1, -1)
        )
        self.classifier = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input (B, 4, H, W), got {tuple(inputs.shape)}"
            )
        minimum_size = 2 ** len(self.pools)
        if min(inputs.shape[-2:]) < minimum_size:
            raise ValueError(
                f"Input spatial dimensions must be at least {minimum_size}."
            )
        skips = []
        current = inputs
        for encoder, pool in zip(self.encoders, self.pools):
            current = encoder(current)
            skips.append(current)
            current = pool(current)
        current = self.bottleneck(current)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            current = decoder(current, skip)
        return self.classifier(current)
