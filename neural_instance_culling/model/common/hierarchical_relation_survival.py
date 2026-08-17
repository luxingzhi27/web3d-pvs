"""Offline hierarchical encoder for train-only observed occlusion relations.

The encoder consumes fixed instance geometry and the validated CSR from
``train_observed_relation_csr``.  It is intentionally an offline module:
source IDs, direction/depth message passing, and hierarchy mappings must not
be exported to the browser.  The output is a per-instance ``[N, 4, 7]``
survival-field coefficient tensor.

Message semantics are directional.  Each stored edge is ``source -> target``
where source is the front instance and target is the deeper instance.  Edges
are first aggregated within ``target/direction/depth-shell`` segments, then
decoded through instance, local-group, and structural-group scales.  The
direction and depth dimensions remain explicit until the instance update, so
opposite directions and different depth ordering are not mixed by an
undirected global pool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .train_observed_relation_csr import (
    RELATION_FEATURE_DIM,
    ObservedRelationCSR,
)


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    layers: int = 2,
) -> nn.Sequential:
    if int(layers) < 1:
        raise ValueError("layers must be positive")
    modules: list[nn.Module] = []
    current = int(input_dim)
    for _ in range(int(layers) - 1):
        modules.extend((nn.Linear(current, int(hidden_dim)), nn.SiLU()))
        current = int(hidden_dim)
    modules.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*modules)


def _validate_segment_ids(
    segment_ids: torch.Tensor,
    *,
    expected_size: int,
    name: str,
) -> tuple[torch.Tensor, int]:
    values = torch.as_tensor(segment_ids, dtype=torch.long)
    if values.ndim != 1 or int(values.numel()) != int(expected_size):
        raise ValueError(f"{name} must have shape [{expected_size}]")
    if values.numel() == 0:
        return values, 0
    if int(values.min()) < 0:
        raise ValueError(f"{name} must be non-negative")
    count = int(values.max().item()) + 1
    expected = torch.arange(count, dtype=torch.long, device=values.device)
    present = torch.unique(values, sorted=True)
    if not torch.equal(present, expected):
        raise ValueError(f"{name} must contain contiguous IDs from zero")
    return values, count


def segment_mean(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean values and counts without requiring torch-scatter."""
    if values.ndim != 2 or segment_ids.ndim != 1 or values.shape[0] != segment_ids.numel():
        raise ValueError("segment_mean expects values [E, C] and segment_ids [E]")
    output = values.new_zeros((int(num_segments), values.shape[1]))
    counts = values.new_zeros((int(num_segments), 1))
    if values.numel():
        output.index_add_(0, segment_ids, values)
        counts.index_add_(
            0,
            segment_ids,
            torch.ones((segment_ids.numel(), 1), dtype=values.dtype, device=values.device),
        )
    return output / counts.clamp_min(1.0), counts


def segment_max(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Return segment maxima, using zero for empty segments."""
    if values.ndim != 2 or segment_ids.ndim != 1 or values.shape[0] != segment_ids.numel():
        raise ValueError("segment_max expects values [E, C] and segment_ids [E]")
    if not values.is_floating_point():
        raise ValueError("segment_max expects floating point values")
    output = torch.full(
        (int(num_segments), values.shape[1]),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    if values.numel():
        expanded = segment_ids[:, None].expand(-1, values.shape[1])
        output.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


@dataclass(frozen=True)
class HierarchySpec:
    """Contiguous instance-to-group mappings for the three encoder scales."""

    local_group_ids: torch.Tensor
    structural_group_ids: torch.Tensor
    local_group_count: int
    structural_group_count: int

    @classmethod
    def from_ids(
        cls,
        local_group_ids: torch.Tensor,
        structural_group_ids: torch.Tensor,
        *,
        num_instances: int,
    ) -> "HierarchySpec":
        local, local_count = _validate_segment_ids(
            local_group_ids, expected_size=int(num_instances), name="local_group_ids"
        )
        structural, structural_count = _validate_segment_ids(
            structural_group_ids,
            expected_size=local_count,
            name="structural_group_ids",
        )
        return cls(local, structural, local_count, structural_count)


def identity_hierarchy(
    num_instances: int,
    *,
    local_group_size: int = 1,
    structural_group_size: int | None = None,
    device: torch.device | None = None,
) -> HierarchySpec:
    """Create a deterministic hierarchy useful for controls and smoke tests."""
    if int(num_instances) <= 0 or int(local_group_size) <= 0:
        raise ValueError("num_instances and local_group_size must be positive")
    local = torch.arange(int(num_instances), device=device, dtype=torch.long) // int(local_group_size)
    local_count = int(local.max().item()) + 1
    size = int(structural_group_size or local_count)
    if size <= 0:
        raise ValueError("structural_group_size must be positive")
    structural = torch.arange(local_count, device=device, dtype=torch.long) // size
    return HierarchySpec.from_ids(local, structural, num_instances=int(num_instances))


def _relation_tensors(
    relation: ObservedRelationCSR | Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if isinstance(relation, ObservedRelationCSR):
        relation.validate()
        tensors = relation.to_torch(device=device)
    elif isinstance(relation, Mapping):
        required = (
            "source_ids",
            "target_ids",
            "direction_ids",
            "depth_shell_ids",
            "edge_features",
        )
        if any(key not in relation for key in required):
            raise ValueError(f"relation mapping needs {required}")
        tensors = {
            key: torch.as_tensor(value, device=device)
            for key, value in relation.items()
        }
    else:
        raise TypeError("relation must be ObservedRelationCSR or a tensor mapping")
    result = {
        key: torch.as_tensor(value, device=device)
        for key, value in tensors.items()
    }
    for key in ("source_ids", "target_ids", "direction_ids", "depth_shell_ids"):
        result[key] = result[key].long().reshape(-1)
    result["edge_features"] = result["edge_features"].float()
    edge_count = int(result["source_ids"].numel())
    if any(int(result[key].numel()) != edge_count for key in ("target_ids", "direction_ids", "depth_shell_ids")):
        raise ValueError("relation index arrays have different edge counts")
    if result["edge_features"].ndim != 2 or result["edge_features"].shape != (edge_count, RELATION_FEATURE_DIM):
        raise ValueError(f"relation edge_features must have shape [{edge_count}, {RELATION_FEATURE_DIM}]")
    return result


class HierarchicalOcclusionSurvivalEncoder(nn.Module):
    """Three-scale directed relation encoder producing survival coefficients.

    Inputs are ``geo_features [N, geo_dim]``, a validated observed relation
    CSR, and contiguous mappings ``instance -> local group`` and ``local group
    -> structural group``.  The output of :meth:`forward` is ``[N, 4, 7]``.
    """

    def __init__(
        self,
        *,
        geo_dim: int = 96,
        hidden_dim: int = 64,
        direction_bins: int = 12,
        depth_shells: int = 3,
        survival_rank: int = 4,
        survival_parameter_dim: int = 7,
    ) -> None:
        super().__init__()
        if min(int(geo_dim), int(hidden_dim), int(direction_bins), int(depth_shells)) <= 0:
            raise ValueError("encoder dimensions must be positive")
        if int(survival_rank) != 4 or int(survival_parameter_dim) != 7:
            raise ValueError("the registered first encoder uses a fixed [4, 7] survival field")
        self.geo_dim = int(geo_dim)
        self.hidden_dim = int(hidden_dim)
        self.direction_bins = int(direction_bins)
        self.depth_shells = int(depth_shells)
        self.survival_rank = int(survival_rank)
        self.survival_parameter_dim = int(survival_parameter_dim)

        self.node_projection = _mlp(self.geo_dim, self.hidden_dim, self.hidden_dim)
        self.direction_embedding = nn.Parameter(torch.zeros(self.direction_bins, self.hidden_dim))
        self.depth_embedding = nn.Parameter(torch.zeros(self.depth_shells, self.hidden_dim))
        nn.init.normal_(self.direction_embedding, std=0.04)
        nn.init.normal_(self.depth_embedding, std=0.04)
        self.edge_message = _mlp(
            2 * self.hidden_dim + RELATION_FEATURE_DIM + 2 * self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
        )
        # A per-shell score creates an ordered depth mixture.  It is applied
        # after segment aggregation, so a deeper shell cannot be confused with
        # a different source within the same shell.
        self.shell_gate = _mlp(2 * self.hidden_dim + 1, self.hidden_dim, 1)
        self.direction_update = _mlp(
            self.direction_bins * self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
        )
        self.instance_update = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.local_update = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.structural_update = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.local_decode = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.instance_decode = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.coefficient_head = nn.Linear(self.hidden_dim, self.survival_rank * self.survival_parameter_dim)
        nn.init.normal_(self.coefficient_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.coefficient_head.bias)

    def _directed_messages(
        self,
        node_state: torch.Tensor,
        relation: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_instances = int(node_state.shape[0])
        source = relation["source_ids"]
        target = relation["target_ids"]
        direction = relation["direction_ids"]
        shell = relation["depth_shell_ids"]
        features = relation["edge_features"]
        if source.numel():
            if int(source.min()) < 0 or int(source.max()) >= num_instances:
                raise ValueError("relation source ID is outside geometry table")
            if int(target.min()) < 0 or int(target.max()) >= num_instances:
                raise ValueError("relation target ID is outside geometry table")
            if int(direction.min()) < 0 or int(direction.max()) >= self.direction_bins:
                raise ValueError("relation direction ID exceeds encoder direction_bins")
            if int(shell.min()) < 0 or int(shell.max()) >= self.depth_shells:
                raise ValueError("relation depth shell exceeds encoder depth_shells")
            if torch.any(source == target):
                raise ValueError("directed relation source and target must differ")
        edge_input = torch.cat(
            [
                node_state[source],
                node_state[target],
                features,
                self.direction_embedding[direction],
                self.depth_embedding[shell],
            ],
            dim=-1,
        )
        edge_message = self.edge_message(edge_input)
        segment = target * (self.direction_bins * self.depth_shells) + direction * self.depth_shells + shell
        segment_count = num_instances * self.direction_bins * self.depth_shells
        shell_values, counts = segment_mean(edge_message, segment, segment_count)
        shell_values = shell_values.reshape(
            num_instances, self.direction_bins, self.depth_shells, self.hidden_dim
        )
        shell_counts = counts.reshape(num_instances, self.direction_bins, self.depth_shells, 1)
        shell_axis = torch.linspace(
            0.0, 1.0, self.depth_shells, device=node_state.device, dtype=node_state.dtype
        ).reshape(1, 1, self.depth_shells, 1).expand(
            num_instances, self.direction_bins, -1, -1
        )
        gate_input = torch.cat(
            [
                shell_values,
                self.depth_embedding.view(1, 1, self.depth_shells, self.hidden_dim).expand(
                    num_instances, self.direction_bins, -1, -1
                ),
                shell_axis,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.shell_gate(gate_input.reshape(-1, gate_input.shape[-1])))
        gate = gate.reshape(num_instances, self.direction_bins, self.depth_shells, 1)
        gate = gate * (shell_counts > 0).to(node_state.dtype)
        directional = (shell_values * gate).sum(dim=2)
        directional = directional / gate.sum(dim=2).clamp_min(1.0)
        direction_flat = directional.reshape(num_instances, self.direction_bins * self.hidden_dim)
        direction_state = self.direction_update(direction_flat)
        return directional, direction_state, shell_counts

    def forward(
        self,
        geo_features: torch.Tensor,
        relation: ObservedRelationCSR | Mapping[str, torch.Tensor],
        local_group_ids: torch.Tensor,
        structural_group_ids: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Encode geometry and observed relations into ``[N, 4, 7]``."""
        geo = torch.as_tensor(geo_features, dtype=torch.float32)
        if geo.ndim != 2 or geo.shape[1] != self.geo_dim:
            raise ValueError(f"geo_features must have shape [N, {self.geo_dim}]")
        device = geo.device
        tensors = _relation_tensors(relation, device=device)
        num_instances = int(geo.shape[0])
        if num_instances <= 0:
            raise ValueError("geo_features must contain at least one instance")
        if tensors["source_ids"].numel():
            if int(tensors["source_ids"].max()) >= num_instances or int(tensors["target_ids"].max()) >= num_instances:
                raise ValueError("relation IDs exceed geo_features")
        local, local_count = _validate_segment_ids(
            local_group_ids, expected_size=num_instances, name="local_group_ids"
        )
        structural, structural_count = _validate_segment_ids(
            structural_group_ids, expected_size=local_count, name="structural_group_ids"
        )
        hierarchy = HierarchySpec(local, structural, local_count, structural_count)
        node_state = self.node_projection(geo)
        directional, direction_state, shell_counts = self._directed_messages(node_state, tensors)
        instance_relation = self.instance_update(torch.cat([node_state, direction_state], dim=-1))

        local_mean, local_count_tensor = segment_mean(instance_relation, hierarchy.local_group_ids.to(device), local_count)
        local_max = segment_max(instance_relation, hierarchy.local_group_ids.to(device), local_count)
        local_state = self.local_update(torch.cat([local_mean, local_max], dim=-1))

        structural_mean, structural_count_tensor = segment_mean(
            local_state, hierarchy.structural_group_ids.to(device), structural_count
        )
        structural_max = segment_max(
            local_state, hierarchy.structural_group_ids.to(device), structural_count
        )
        structural_state = self.structural_update(torch.cat([structural_mean, structural_max], dim=-1))

        decoded_local = self.local_decode(
            torch.cat([local_state, structural_state[hierarchy.structural_group_ids.to(device)]], dim=-1)
        )
        decoded_instance = self.instance_decode(
            torch.cat([instance_relation, decoded_local[hierarchy.local_group_ids.to(device)]], dim=-1)
        )
        coefficients = self.coefficient_head(decoded_instance).reshape(
            num_instances, self.survival_rank, self.survival_parameter_dim
        )
        if not return_diagnostics:
            return coefficients
        return {
            "survival_coefficients": coefficients,
            "instance_features": instance_relation,
            "directional_messages": directional,
            "direction_state": direction_state,
            "local_features": decoded_local,
            "structural_features": structural_state,
            "shell_counts": shell_counts,
            "local_group_counts": local_count_tensor,
            "structural_group_counts": structural_count_tensor,
            "hierarchy": hierarchy,
        }


def query_monotone_survival(
    coefficients: torch.Tensor,
    direction_basis: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Query the same monotone two-component survival parameterization.

    This helper is intentionally small and training-side only.  It provides a
    numerical smoke target for the coefficient shape while the full model
    decides how its learned direction basis is parameterized.
    """
    coefficients = torch.as_tensor(coefficients, dtype=torch.float32)
    direction_basis = torch.as_tensor(direction_basis, dtype=coefficients.dtype, device=coefficients.device)
    rho = torch.as_tensor(rho, dtype=coefficients.dtype, device=coefficients.device).reshape(-1, 1)
    if coefficients.ndim != 3 or coefficients.shape[1:] != (4, 7):
        raise ValueError("coefficients must have shape [N, 4, 7]")
    if direction_basis.ndim != 2 or direction_basis.shape[1] != 4:
        raise ValueError("direction_basis must have shape [N, 4]")
    if direction_basis.shape[0] != coefficients.shape[0] or rho.shape[0] != coefficients.shape[0]:
        raise ValueError("survival query batch dimensions do not match")
    params = torch.einsum("bi,bik->bk", direction_basis, coefficients)
    no_block = torch.sigmoid(params[:, 0:1])
    weights = torch.softmax(params[:, 1:3], dim=-1)
    depths = 0.05 + 0.90 * torch.sigmoid(params[:, 3:5])
    scales = 0.03 + F.softplus(params[:, 5:7])
    depths, order = torch.sort(depths, dim=-1)
    weights = torch.gather(weights, -1, order)
    scales = torch.gather(scales, -1, order)
    cdf = (1.0 - no_block) * (
        weights * torch.sigmoid((rho - depths) / scales)
    ).sum(dim=-1, keepdim=True)
    return torch.clamp(1.0 - cdf, 1e-5, 1.0)


__all__ = [
    "HierarchySpec",
    "identity_hierarchy",
    "segment_mean",
    "segment_max",
    "HierarchicalOcclusionSurvivalEncoder",
    "query_monotone_survival",
]
