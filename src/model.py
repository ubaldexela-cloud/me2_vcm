"""VCM-DS-Tiny: depthwise-separable CNN (locked 95,376 parameters)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, conv: nn.Module, channels: int):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class VCMDSTiny(nn.Module):
    """
    Exact architecture from VCM_BASELINE_ARCHITECTURE.csv / design report.

    Softmax is NOT applied in forward (training uses CrossEntropyLoss on logits).
    Expected trainable parameters: 95,376
    """

    def __init__(self, num_classes: int = 16, dropout: float = 0.20):
        super().__init__()
        # Stem: Conv2d 1->32, k3, s1, p1
        self.stem = ConvBNAct(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True),
            32,
        )
        # Block 1
        self.dw1 = ConvBNAct(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=True),
            32,
        )
        self.pw1 = ConvBNAct(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=True),
            64,
        )
        # Block 2
        self.dw2 = ConvBNAct(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=True),
            64,
        )
        self.pw2 = ConvBNAct(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True),
            128,
        )
        # Block 3
        self.dw3 = ConvBNAct(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=True),
            128,
        )
        self.pw3 = ConvBNAct(
            nn.Conv2d(128, 192, kernel_size=1, stride=1, padding=0, bias=True),
            192,
        )
        # Block 4
        self.dw4 = ConvBNAct(
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, groups=192, bias=True),
            192,
        )
        self.pw4 = ConvBNAct(
            nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0, bias=True),
            256,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 40, 397)
        x = self.stem(x)
        x = self.pw1(self.dw1(x))
        x = self.pw2(self.dw2(x))
        x = self.pw3(self.dw3(x))
        x = self.pw4(self.dw4(x))
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)  # logits (B, 16)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "fp32_bytes": trainable * 4,
        "fp16_bytes": trainable * 2,
        "int8_bytes": trainable * 1,
    }
