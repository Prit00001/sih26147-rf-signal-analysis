"""CNN architecture for modulation classification.

Train-time only (imports PyTorch -- part of the "train" optional dependency
group, NOT required at runtime; inference uses ONNX Runtime instead).

Covers: FR-06.
"""

from __future__ import annotations

import torch
from torch import nn


class ModulationCNN(nn.Module):
    """Small 1D CNN over [I, Q] channels -> per-class logits."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).squeeze(-1)
        return self.classifier(features)  # type: ignore[no-any-return]
