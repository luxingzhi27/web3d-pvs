"""Offline point-cloud encoder for the fixed 96D instance geometry table."""
from __future__ import annotations

import torch
from torch import nn


def _mlp(dims: list[int], *, last_relu: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2 or last_relu:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class InstancePointNetPPGeoEncoder(nn.Module):
    """Encode one sampled instance point cloud into a fixed geometry vector."""

    def __init__(
        self,
        point_feature_dim: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 96,
        center_count: int = 16,
        neighbor_count: int = 8,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.center_count = int(center_count)
        self.neighbor_count = int(neighbor_count)
        self.local_mlp = _mlp(
            [point_feature_dim + 3, hidden_dim, hidden_dim], last_relu=True
        )
        self.local_pool = _mlp(
            [hidden_dim * 2, hidden_dim, hidden_dim], last_relu=True
        )
        self.global_mlp = _mlp(
            [hidden_dim * 2, hidden_dim, out_dim], last_relu=True
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        batch_size, point_count, feature_dim = point_features.shape
        center_count = min(max(1, self.center_count), point_count)
        neighbor_count = min(max(1, self.neighbor_count), point_count)
        center_indices = torch.linspace(
            0,
            point_count - 1,
            steps=center_count,
            device=point_features.device,
        ).round().long()
        xyz = point_features[..., :3]
        centers = xyz[:, center_indices, :]
        squared_distance = (
            centers[:, :, None, :] - xyz[:, None, :, :]
        ).square().sum(dim=-1)
        neighbor_indices = torch.topk(
            squared_distance, k=neighbor_count, dim=-1, largest=False
        ).indices
        expanded = point_features[:, None, :, :].expand(
            batch_size, center_count, point_count, feature_dim
        )
        grouped = torch.gather(
            expanded,
            2,
            neighbor_indices[..., None].expand(
                batch_size, center_count, neighbor_count, feature_dim
            ),
        )
        center_features = point_features[:, center_indices, :]
        local_input = torch.cat(
            [grouped, grouped[..., :3] - center_features[:, :, None, :3]], dim=-1
        )
        encoded = self.local_mlp(
            local_input.reshape(
                batch_size * center_count * neighbor_count, feature_dim + 3
            )
        ).reshape(batch_size, center_count, neighbor_count, -1)
        pooled = torch.cat(
            [encoded.mean(dim=2), encoded.max(dim=2).values], dim=-1
        )
        local = self.local_pool(
            pooled.reshape(batch_size * center_count, -1)
        ).reshape(batch_size, center_count, -1)
        global_pooled = torch.cat(
            [local.mean(dim=1), local.max(dim=1).values], dim=-1
        )
        return self.global_mlp(global_pooled)


__all__ = ["InstancePointNetPPGeoEncoder"]
