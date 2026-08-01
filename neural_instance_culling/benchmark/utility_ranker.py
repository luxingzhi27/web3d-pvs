"""Shared model definition for the independent GLB-utility ranking baseline."""
from __future__ import annotations

import torch
from torch import nn

from aabb_ray_feature_utils import FEATURE_DIM


class IndependentUtilityRankerMLP(nn.Module):
    """Small pairwise ranker that never receives the visibility probability."""

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
