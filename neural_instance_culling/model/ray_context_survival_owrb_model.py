#!/usr/bin/env python3
"""Offline relation encoder and lightweight ray-query PVS model.

The model has two phases.  During offline preparation, a relation encoder
reads fixed geometry and the triangle-depth relation table and produces
direction tokens.  A small direction basis fits those tokens to 32 fixed
coefficients per instance.  Browser inference only evaluates the basis and a
semantic eight-value survival query; it never loads source IDs or performs
relation propagation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from current_pvs_utils import InstancePointNetPPGeoEncoder, fourier_features, mlp


def spherical_direction_anchors(device: torch.device | None = None) -> torch.Tensor:
    values: list[list[float]] = []
    for pitch in (0.0, math.pi / 4.0, -math.pi / 4.0):
        for yaw in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
            cp = math.cos(pitch)
            values.append([math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp])
    return torch.tensor(values, dtype=torch.float32, device=device)


class RayRelationContextEncoder(nn.Module):
    """Offline ordered attention over sources, depth shells and directions."""

    def __init__(self, geo_dim: int = 96, hidden_dim: int = 64, token_dim: int = 8):
        super().__init__()
        self.geo_dim = int(geo_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.edge = mlp([2 * self.geo_dim + 11, hidden_dim, hidden_dim], last_relu=True)
        self.source_score = nn.Linear(hidden_dim, 1)
        self.shell = mlp([hidden_dim + 8, hidden_dim, hidden_dim], last_relu=True)
        self.shell_score = nn.Linear(hidden_dim, 1)
        self.direction = mlp([hidden_dim + 4, hidden_dim, token_dim], last_relu=True)
        self.direction_query = nn.Linear(token_dim, token_dim, bias=False)
        self.direction_key = nn.Linear(token_dim, token_dim, bias=False)
        self.direction_value = nn.Linear(token_dim, token_dim, bias=False)
        self.direction_output = nn.Linear(token_dim, token_dim)
        self.register_buffer("anchors", spherical_direction_anchors())

    def forward(
        self,
        target_geo: torch.Tensor,
        source_geo: torch.Tensor,
        relative_features: torch.Tensor,
        relation_stats: torch.Tensor,
        evidence_strength: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # target [B,G], source [B,D,S,K,G], stats [B,D,S,K,6].
        batch, directions, shells, source_k, _ = source_geo.shape
        target_rep = target_geo[:, None, None, None, :].expand(batch, directions, shells, source_k, -1)
        edge_input = torch.cat([target_rep, source_geo, relative_features], dim=-1)
        edge = self.edge(edge_input.reshape(-1, edge_input.shape[-1]))
        edge = edge.reshape(batch, directions, shells, source_k, -1)
        source_score = torch.clamp(relation_stats[..., 0], min=0.0)
        source_logits = self.source_score(edge).squeeze(-1)
        source_logits = source_logits + torch.log(torch.clamp(source_score, min=1e-6))
        source_valid = source_score > 0.0
        source_logits = source_logits.masked_fill(~source_valid, -1e4)
        source_attention = torch.softmax(source_logits, dim=-1) * source_valid.float()
        source_pool = (edge * source_attention.unsqueeze(-1)).sum(dim=-2)
        source_mass = source_attention.sum(dim=-1, keepdim=True)
        source_pool = source_pool * torch.clamp(source_mass, max=1.0)

        shell_index = torch.linspace(-1.0, 1.0, shells, device=edge.device, dtype=edge.dtype)
        relation_mean = relation_stats.mean(dim=-2)
        shell_input = torch.cat([
            source_pool,
            relation_mean,
            shell_index.view(1, 1, shells, 1).expand(batch, directions, shells, 1),
            evidence_strength.unsqueeze(-1),
        ], dim=-1)
        shell_features = self.shell(shell_input.reshape(-1, shell_input.shape[-1]))
        shell_features = shell_features.reshape(batch, directions, shells, -1)
        shell_logits = self.shell_score(shell_features).squeeze(-1)
        shell_logits = shell_logits - 0.15 * torch.arange(
            shells, device=edge.device, dtype=edge.dtype
        ).view(1, 1, shells)
        shell_valid = evidence_strength > 0.0
        shell_logits = shell_logits.masked_fill(~shell_valid, -1e4)
        shell_attention = torch.softmax(shell_logits, dim=-1) * shell_valid.float()
        shell_pool = (shell_features * shell_attention.unsqueeze(-1)).sum(dim=-2)
        shell_mass = shell_attention.sum(dim=-1, keepdim=True)
        shell_pool = shell_pool * torch.clamp(shell_mass, max=1.0)

        anchors = self.anchors.to(device=edge.device, dtype=edge.dtype)
        direction_input = torch.cat([
            shell_pool,
            anchors.view(1, directions, 3).expand(batch, directions, 3),
            evidence_strength.sum(dim=-1, keepdim=True),
        ], dim=-1)
        tokens = self.direction(direction_input.reshape(-1, direction_input.shape[-1]))
        tokens = torch.tanh(tokens.reshape(batch, directions, self.token_dim))

        # Relative-angle attention lets nearby spherical anchors exchange
        # information while retaining the anchor identity and its ordering.
        query = self.direction_query(tokens)
        key = self.direction_key(tokens)
        value = self.direction_value(tokens)
        attention_logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.token_dim)
        attention_logits = attention_logits + 0.15 * torch.matmul(anchors, anchors.T).view(1, directions, directions)
        direction_valid = evidence_strength.sum(dim=-1) > 0.0
        attention_logits = attention_logits.masked_fill(~direction_valid[:, None, :], -1e4)
        attention = torch.softmax(attention_logits, dim=-1) * direction_valid[:, None, :].float()
        tokens = torch.tanh(tokens + self.direction_output(torch.matmul(attention, value)))
        return tokens, evidence_strength


class RayContextSurvivalOWRBModel(nn.Module):
    """Offline fixed-feature PVS model with explicit supplement switches.

    The default configuration is the frozen 156-dimensional table, nine
    direct ray values, and the monotone survival field.  The optional context
    width, Fourier ray input, and unconstrained 28-parameter survival field
    are isolated for registered supplement comparisons and are rejected by
    the browser exporter.  This keeps the deployable path unambiguous.
    """

    geo_dim = 96
    context_coeff_dim = 32
    survival_coeff_dim = 28
    context_query_dim = 8
    survival_query_dim = 8
    ray_dim = 9
    base_hidden_dim = 32

    def __init__(
        self,
        num_instances: int,
        num_glbs: int,
        point_hidden_dim: int = 160,
        pointnetpp_centers: int = 24,
        pointnetpp_neighbors: int = 12,
        relation_hidden_dim: int = 64,
        context_mode: str = "directional",
        survival_enabled: bool = True,
        survival_input_mode: str = "semantic",
        context_query_dim: int = 8,
        ray_mode: str = "direct9",
        survival_parameterization: str = "monotone",
        scene_size_m: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        if context_mode not in ("zero", "pooled", "directional"):
            raise ValueError("context_mode must be zero, pooled, or directional")
        if survival_input_mode not in ("semantic", "raw"):
            raise ValueError("survival_input_mode must be semantic or raw")
        if int(context_query_dim) not in (8, 16):
            raise ValueError("context_query_dim must be 8 (32-D) or 16 (64-D)")
        if ray_mode not in ("direct9", "fourier117"):
            raise ValueError("ray_mode must be direct9 or fourier117")
        if survival_parameterization not in ("monotone", "unconstrained28"):
            raise ValueError("survival_parameterization must be monotone or unconstrained28")
        self.num_instances = int(num_instances)
        self.num_glbs = int(num_glbs)
        self.point_hidden_dim = int(point_hidden_dim)
        self.pointnetpp_centers = int(pointnetpp_centers)
        self.pointnetpp_neighbors = int(pointnetpp_neighbors)
        self.relation_hidden_dim = int(relation_hidden_dim)
        self.context_mode = str(context_mode)
        self.survival_enabled = bool(survival_enabled)
        self.survival_input_mode = str(survival_input_mode)
        self.context_query_dim = int(context_query_dim)
        self.context_coeff_dim = 4 * self.context_query_dim
        self.ray_mode = str(ray_mode)
        self.ray_dim = 9 if self.ray_mode == "direct9" else 117
        self.survival_parameterization = str(survival_parameterization)
        self.point_feature_dim = 16

        self.geo_encoder = InstancePointNetPPGeoEncoder(
            point_feature_dim=self.point_feature_dim,
            hidden_dim=self.point_hidden_dim,
            out_dim=self.geo_dim,
            center_count=self.pointnetpp_centers,
            neighbor_count=self.pointnetpp_neighbors,
        )
        self.relation_encoder = RayRelationContextEncoder(
            geo_dim=self.geo_dim,
            hidden_dim=self.relation_hidden_dim,
            token_dim=self.context_query_dim,
        )
        self.context_direction_basis = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(inplace=True), nn.Linear(16, 4), nn.Tanh()
        )
        self.survival_direction_basis = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(inplace=True), nn.Linear(16, 4), nn.Tanh()
        )
        self.relation_aux_head = nn.Sequential(
            nn.Linear(self.context_query_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 3)
        )
        self.survival_coefficients = nn.Parameter(torch.zeros(self.num_instances, 4, 7))
        nn.init.normal_(self.survival_coefficients, mean=0.0, std=0.02)

        # The default formal query is the complete fixed table (96 geometry
        # + 32 context coefficients + 28 survival coefficients) plus 21 cheap
        # view-query values.  Supplement variants change only the registered
        # input representation; they never enter the browser export path.
        self.base_input_dim = self.runtime_feature_dim + self.ray_dim + self.context_query_dim + 4
        self.base_trunk = nn.Sequential(
            nn.Linear(self.base_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.base_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.base_visibility_head = nn.Linear(self.base_hidden_dim, 1)
        self.utility_head = nn.Sequential(nn.Linear(33, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.download_head = nn.Sequential(nn.Linear(33, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        with torch.no_grad():
            self.utility_head[-1].bias.fill_(-4.0)

        self.register_buffer("instance_world_aabbs", torch.zeros(self.num_instances, 6))
        self.register_buffer("instance_to_glb", torch.zeros(self.num_instances, dtype=torch.long))
        self.register_buffer("scene_min", torch.zeros(3))
        self.register_buffer(
            "scene_size_m", torch.tensor(scene_size_m or [1.0, 1.0, 1.0], dtype=torch.float32)
        )
        self.register_buffer(
            "relation_source_ids", torch.zeros(self.num_instances, 12, 3, 8, dtype=torch.long)
        )
        self.register_buffer(
            "relation_stats", torch.zeros(self.num_instances, 12, 3, 8, 6)
        )
        self.register_buffer("relation_strength", torch.zeros(self.num_instances, 12, 3))
        self.register_buffer("relation_count", torch.zeros(self.num_instances, 12, 3, dtype=torch.long))
        self.register_buffer(
            "context_coefficients", torch.zeros(self.num_instances, 4, self.context_query_dim)
        )

    @property
    def runtime_feature_dim(self) -> int:
        return self.geo_dim + self.context_coeff_dim + self.survival_coeff_dim

    @property
    def query_dim(self) -> int:
        return self.base_input_dim

    @property
    def config(self) -> dict[str, Any]:
        return {
            "runtimeSchema": "ray-context-survival-scheduler-v1",
            "numInstances": self.num_instances,
            "numGlbs": self.num_glbs,
            "pointHiddenDim": self.point_hidden_dim,
            "pointnetppCenters": self.pointnetpp_centers,
            "pointnetppNeighbors": self.pointnetpp_neighbors,
            "relationHiddenDim": self.relation_hidden_dim,
            "geoDim": self.geo_dim,
            "contextCoefficientShape": [4, self.context_query_dim],
            "contextCoefficientDim": self.context_coeff_dim,
            "survivalCoefficientShape": [4, 7],
            "survivalCoefficientDim": self.survival_coeff_dim,
            "runtimeFeatureDim": self.runtime_feature_dim,
            "contextQueryDim": self.context_query_dim,
            "survivalQueryDim": self.survival_query_dim,
            "survivalVisibilityInput": ["survival", "noBlock", "expectedDepth", "uncertainty"],
            "rayDim": self.ray_dim,
            "baseInputDim": self.base_input_dim,
            "baseTrunk": [self.base_input_dim, 64, 32],
            "visibilityHead": [self.base_input_dim, 64, 32, 1],
            "contextMode": self.context_mode,
            "survivalEnabled": self.survival_enabled,
            "survivalInputMode": self.survival_input_mode,
            "contextQueryDim": self.context_query_dim,
            "rayMode": self.ray_mode,
            "survivalParameterization": self.survival_parameterization,
            "relationDirectionBins": 12,
            "relationDepthShells": 3,
            "relationSourceK": 8,
            "frontendOnlineOperations": [
                "nine direct ray-space scalars" if self.ray_mode == "direct9" else "117-dimensional Fourier ray features",
                "shared 3-to-4 direction bases",
                "semantic survival query",
                f"{self.context_query_dim}-dimensional context query",
                "shared small MLP heads",
            ],
            "frontendOfflineOnlyOperations": [
                "PointNet++ geometry encoding",
                "source relation attention",
                "depth-shell attention",
                "direction attention",
            ],
        }

    def set_scene_bounds(self, scene_min: torch.Tensor, scene_size: torch.Tensor) -> None:
        self.scene_min.copy_(scene_min.detach().to(self.scene_min))
        self.scene_size_m.copy_(scene_size.detach().to(self.scene_size_m))

    def set_instance_world_aabbs(self, aabbs: torch.Tensor) -> None:
        if tuple(aabbs.shape) != (self.num_instances, 6):
            raise ValueError("instance AABBs must have shape [num_instances, 6]")
        self.instance_world_aabbs.copy_(aabbs.detach().to(self.instance_world_aabbs))

    def set_instance_to_glb(self, values: torch.Tensor) -> None:
        if int(values.numel()) != self.num_instances:
            raise ValueError("instance_to_glb length does not match num_instances")
        self.instance_to_glb.copy_(values.detach().long().to(self.instance_to_glb))

    def set_relation_evidence(
        self,
        source_ids: torch.Tensor,
        relation_stats: torch.Tensor,
        strength: torch.Tensor,
        count: torch.Tensor | None = None,
    ) -> None:
        expected_ids = (self.num_instances, 12, 3, 8)
        expected_stats = (self.num_instances, 12, 3, 8, 6)
        if tuple(source_ids.shape) != expected_ids or tuple(relation_stats.shape) != expected_stats:
            raise ValueError("relation evidence shapes do not match the v2 schema")
        self.relation_source_ids.copy_(source_ids.detach().long().to(self.relation_source_ids))
        self.relation_stats.copy_(relation_stats.detach().float().to(self.relation_stats))
        self.relation_strength.copy_(strength.detach().float().to(self.relation_strength))
        if count is not None:
            self.relation_count.copy_(count.detach().long().to(self.relation_count))

    def set_context_coefficients(self, coefficients: torch.Tensor) -> None:
        expected = (self.num_instances, 4, self.context_query_dim)
        if tuple(coefficients.shape) != expected:
            raise ValueError(f"context coefficients must have shape {expected}")
        self.context_coefficients.copy_(coefficients.detach().to(self.context_coefficients))

    def point_features(self, instance_ids: torch.Tensor, glb_points: torch.Tensor) -> torch.Tensor:
        ids = instance_ids.long()
        local = glb_points[self.instance_to_glb[ids]]
        bounds = self.instance_world_aabbs[ids].to(local.device)
        mins, maxs = bounds[:, :3], bounds[:, 3:]
        center = (mins + maxs) * 0.5
        size = torch.clamp(maxs - mins, min=1e-4)
        scene_min = self.scene_min.to(local.device)
        scene_size = torch.clamp(self.scene_size_m.to(local.device), min=1e-6)
        center_norm = torch.clamp((center - scene_min) / scene_size, 0.0, 1.0)
        size_norm = torch.clamp(size / scene_size, 0.0, 4.0)
        uniform_scale = size.max(dim=-1, keepdim=True).values
        world = center[:, None, :] + local * uniform_scale[:, None, :]
        world_norm = torch.clamp(
            (world - scene_min[None, None, :]) / scene_size[None, None, :], 0.0, 1.0
        )
        scale_ratio = size / torch.clamp(uniform_scale, min=1e-6)
        return torch.cat([
            local,
            world_norm,
            center_norm[:, None, :].expand_as(local),
            size_norm[:, None, :].expand_as(local),
            scale_ratio[:, None, :].expand_as(local),
            torch.linalg.norm(local, dim=-1, keepdim=True),
        ], dim=-1)

    def instance_geo_features(self, instance_ids: torch.Tensor, glb_points: torch.Tensor) -> torch.Tensor:
        return self.geo_encoder(self.point_features(instance_ids, glb_points))

    @staticmethod
    def _relative_relation_features(
        target_ids: torch.Tensor,
        source_ids: torch.Tensor,
        relation_stats: torch.Tensor,
        aabbs: torch.Tensor,
        scene_size: torch.Tensor,
    ) -> torch.Tensor:
        target_bounds = aabbs[target_ids.long()]
        source_bounds = aabbs[source_ids.long()]
        target_center = (target_bounds[..., :3] + target_bounds[..., 3:]) * 0.5
        source_center = (source_bounds[..., :3] + source_bounds[..., 3:]) * 0.5
        target_size = torch.clamp(target_bounds[..., 3:] - target_bounds[..., :3], min=1e-4)
        source_size = torch.clamp(source_bounds[..., 3:] - source_bounds[..., :3], min=1e-4)
        delta = (source_center - target_center) / torch.clamp(scene_size, min=1e-6)
        distance = torch.linalg.norm(source_center - target_center, dim=-1, keepdim=True)
        radius_ratio = torch.linalg.norm(source_size, dim=-1, keepdim=True) / torch.clamp(
            torch.linalg.norm(target_size, dim=-1, keepdim=True), min=1e-4
        )
        return torch.cat([
            torch.clamp(delta, -4.0, 4.0),
            torch.clamp(torch.log1p(distance / 100.0), 0.0, 4.0),
            torch.clamp(radius_ratio / 8.0, 0.0, 1.0),
            relation_stats,
        ], dim=-1)

    def _relation_tokens(self, target_ids: torch.Tensor, geo_table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = target_ids.long().reshape(-1)
        target_geo = geo_table[ids]
        source_ids = self.relation_source_ids[ids]
        source_geo = geo_table[source_ids.reshape(-1)].reshape(*source_ids.shape, self.geo_dim)
        target_rep = ids[:, None, None, None].expand_as(source_ids)
        relative = self._relative_relation_features(
            target_rep,
            source_ids,
            self.relation_stats[ids],
            self.instance_world_aabbs,
            self.scene_size_m.to(geo_table.device),
        )
        # The relation encoder receives relative geometry/evidence, while the
        # fixed table remains the only runtime representation.
        return self.relation_encoder(
            target_geo,
            source_geo,
            relative,
            self.relation_stats[ids],
            self.relation_strength[ids],
        )

    def context_coefficients_from_geo(
        self,
        instance_ids: torch.Tensor,
        geo_table: torch.Tensor,
        mode: str | None = None,
        query_batch_size: int = 128,
    ) -> torch.Tensor:
        mode = mode or self.context_mode
        ids = instance_ids.long().reshape(-1)
        if mode == "zero":
            return torch.zeros(
                (ids.numel(), 4, self.context_query_dim),
                device=geo_table.device,
                dtype=geo_table.dtype,
            )
        pieces: list[torch.Tensor] = []
        for start in range(0, ids.numel(), int(query_batch_size)):
            local_ids = ids[start : start + int(query_batch_size)]
            tokens, strength = self._relation_tokens(local_ids, geo_table)
            if mode == "pooled":
                weights = strength.sum(dim=-1, keepdim=True)
                pooled = (tokens * weights).sum(dim=1, keepdim=True) / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)
                tokens = pooled.expand(-1, tokens.shape[1], -1)
            elif mode != "directional":
                raise ValueError("context mode must be zero, pooled, or directional")
            pieces.append(self.fit_context_coefficients(tokens))
        return torch.cat(pieces, dim=0) if pieces else torch.zeros(
            (0, 4, self.context_query_dim), device=geo_table.device
        )

    def fit_context_coefficients(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (12, self.context_query_dim):
            raise ValueError(f"tokens must have shape [B, 12, {self.context_query_dim}]")
        anchors = self.relation_encoder.anchors.to(device=tokens.device, dtype=tokens.dtype)
        phi = self.context_direction_basis(anchors)
        gram = phi.T @ phi + 1e-3 * torch.eye(4, device=phi.device, dtype=phi.dtype)
        projection = torch.linalg.solve(gram, phi.T)
        return torch.einsum("am,bmk->bak", projection, tokens)

    @torch.no_grad()
    def compute_offline_geometry(
        self,
        glb_points: torch.Tensor,
        batch_size: int = 512,
    ) -> torch.Tensor:
        self.eval()
        device = self.instance_world_aabbs.device
        ids = torch.arange(self.num_instances, device=device, dtype=torch.long)
        pieces = []
        for start in range(0, self.num_instances, int(batch_size)):
            pieces.append(self.instance_geo_features(ids[start : start + int(batch_size)], glb_points).detach())
        return torch.cat(pieces, dim=0)

    @torch.no_grad()
    def compute_offline_context_coefficients(
        self,
        geo_table: torch.Tensor,
        mode: str | None = None,
        batch_size: int = 128,
    ) -> torch.Tensor:
        self.eval()
        ids = torch.arange(self.num_instances, device=geo_table.device, dtype=torch.long)
        return self.context_coefficients_from_geo(ids, geo_table, mode=mode, query_batch_size=batch_size)

    def runtime_features_from_offline(
        self,
        geo_table: torch.Tensor,
        context_coefficients: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tuple(geo_table.shape) != (self.num_instances, self.geo_dim):
            raise ValueError("geo table has the wrong shape")
        context = self.context_coefficients if context_coefficients is None else context_coefficients
        if tuple(context.shape) != (self.num_instances, 4, self.context_query_dim):
            raise ValueError("context coefficient table has the wrong shape")
        # Keep the fixed 156-D schema identical across the ablation matrix, but
        # make the disabled factor a true zero input.  Otherwise the random
        # 28-D survival parameters would still be visible to the main MLP even
        # when the semantic survival query was disabled.
        survival = self.survival_coefficients if self.survival_enabled else torch.zeros_like(self.survival_coefficients)
        return torch.cat([geo_table, context.reshape(self.num_instances, -1), survival.reshape(self.num_instances, -1)], dim=-1)

    def _split_runtime(self, runtime_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if runtime_features.ndim != 2 or runtime_features.shape[1] != self.runtime_feature_dim:
            raise ValueError(f"runtime_features must have shape [B, {self.runtime_feature_dim}]")
        geo = runtime_features[:, : self.geo_dim]
        context = runtime_features[:, self.geo_dim : self.geo_dim + self.context_coeff_dim].reshape(
            -1, 4, self.context_query_dim
        )
        survival = runtime_features[:, self.geo_dim + self.context_coeff_dim :].reshape(-1, 4, 7)
        return geo, context, survival

    def ray_features(
        self,
        camera_view: torch.Tensor,
        instance_ids: torch.Tensor,
        camera_pos_world: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bounds = self.instance_world_aabbs[instance_ids.long()].to(camera_pos_world.device)
        center = (bounds[:, :3] + bounds[:, 3:]) * 0.5
        size = torch.clamp(bounds[:, 3:] - bounds[:, :3], min=1e-4)
        delta = center - camera_pos_world
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        ray_dir = delta / torch.clamp(distance, min=1e-4)
        forward = F.normalize(camera_view[:, :3], dim=-1, eps=1e-6)
        up_seed = torch.zeros_like(forward)
        up_seed[:, 1] = 1.0
        alternative = torch.zeros_like(forward)
        alternative[:, 2] = 1.0
        up_seed = torch.where(torch.abs((forward * up_seed).sum(dim=-1, keepdim=True)) > 0.98, alternative, up_seed)
        right = F.normalize(torch.cross(forward, up_seed, dim=-1), dim=-1, eps=1e-6)
        up = F.normalize(torch.cross(right, forward, dim=-1), dim=-1, eps=1e-6)
        dot_forward = (ray_dir * forward).sum(dim=-1, keepdim=True)
        dot_right = (ray_dir * right).sum(dim=-1, keepdim=True)
        dot_up = (ray_dir * up).sum(dim=-1, keepdim=True)
        tan_x = torch.clamp(camera_view[:, 3:4], min=1e-4)
        tan_y = torch.clamp(camera_view[:, 4:5], min=1e-4)
        u = dot_right / torch.clamp(torch.abs(dot_forward) * tan_x, min=1e-4)
        v = dot_up / torch.clamp(torch.abs(dot_forward) * tan_y, min=1e-4)
        radius = torch.linalg.norm(size, dim=-1, keepdim=True) * 0.5
        angular = radius / torch.clamp(distance, min=1.0)
        scalar_values = torch.cat([
            torch.clamp(torch.log1p(distance / 100.0) / 4.0, 0.0, 2.0) - 1.0,
            torch.clamp(dot_forward, -1.0, 1.0),
            torch.clamp(u, -4.0, 4.0) / 4.0,
            torch.clamp(v, -4.0, 4.0) / 4.0,
            torch.clamp(angular / tan_x, 0.0, 4.0) / 2.0 - 1.0,
            torch.clamp(angular / tan_y, 0.0, 4.0) / 2.0 - 1.0,
        ], dim=-1)
        ray9 = torch.cat([
            ray_dir,
            scalar_values,
        ], dim=-1)
        if self.ray_mode == "direct9":
            return ray_dir, ray9
        # This is the historical Fourier input used by the old proxy model:
        # 3 direction values with ten bands (63 values) plus six scalar ray
        # values with four bands (54 values), for 117 values total.
        return ray_dir, torch.cat([
            fourier_features(ray_dir, 10),
            fourier_features(scalar_values, 4),
        ], dim=-1)

    def survival_query_from_direction(
        self,
        direction: torch.Tensor,
        rho: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi = self.survival_direction_basis(direction)
        params = torch.einsum("bi,bik->bk", phi, coefficients)
        if self.survival_parameterization == "unconstrained28":
            # Equal-capacity diagnostic control.  It retains the same 4x7
            # coefficient table and direction basis, but does not construct a
            # sorted positive mixture CDF.  The slope of the visibility-like
            # signal is unconstrained, so it can increase with distance.
            survival = torch.sigmoid(params[:, 0:1] + params[:, 1:2] * rho)
            no_block = torch.sigmoid(params[:, 2:3])
            expected_depth = torch.sigmoid(params[:, 3:4])
            uncertainty = torch.sigmoid(params[:, 4:5])
            weights = torch.softmax(params[:, 5:7], dim=-1)
            semantic = torch.cat([
                survival,
                1.0 - survival,
                no_block,
                expected_depth,
                uncertainty,
                weights,
                torch.abs(params[:, 1:2]),
            ], dim=-1)
            return semantic, phi
        no_block = torch.sigmoid(params[:, 0:1])
        weights = torch.softmax(params[:, 1:3], dim=-1)
        depths = 0.05 + 0.90 * torch.sigmoid(params[:, 3:5])
        scales = 0.03 + F.softplus(params[:, 5:7])
        depths, order = torch.sort(depths, dim=-1)
        weights = torch.gather(weights, dim=-1, index=order)
        scales = torch.gather(scales, dim=-1, index=order)
        cdf = (1.0 - no_block) * (weights * torch.sigmoid((rho - depths) / scales)).sum(dim=-1, keepdim=True)
        survival = torch.clamp(1.0 - cdf, 1e-5, 1.0)
        expected_depth = (weights * depths).sum(dim=-1, keepdim=True)
        variance = (weights * (depths - expected_depth).pow(2)).sum(dim=-1, keepdim=True)
        uncertainty = torch.sqrt(torch.clamp(variance, min=1e-8))
        local_slope = (1.0 - no_block) * (weights / scales * torch.sigmoid((rho - depths) / scales) * (1.0 - torch.sigmoid((rho - depths) / scales))).sum(dim=-1, keepdim=True)
        semantic = torch.cat([
            survival,
            1.0 - survival,
            no_block,
            expected_depth,
            uncertainty,
            weights,
            local_slope,
        ], dim=-1)
        return semantic, phi

    def context_query_from_direction(
        self,
        direction: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        phi = self.context_direction_basis(direction)
        return torch.einsum("bi,bik->bk", phi, coefficients)

    def query_runtime_features(
        self,
        camera_view: torch.Tensor,
        camera_pos_world: torch.Tensor,
        instance_ids: torch.Tensor,
        runtime_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        geo, context_coefficients, survival_coefficients = self._split_runtime(runtime_features)
        ray_dir, ray9 = self.ray_features(camera_view, instance_ids, camera_pos_world)
        context_query = self.context_query_from_direction(ray_dir, context_coefficients)
        bounds = self.instance_world_aabbs[instance_ids.long()].to(camera_pos_world.device)
        radius = torch.linalg.norm(torch.clamp(bounds[:, 3:] - bounds[:, :3], min=1e-4), dim=-1, keepdim=True) * 0.5
        distance = torch.linalg.norm((bounds[:, :3] + bounds[:, 3:]) * 0.5 - camera_pos_world, dim=-1, keepdim=True)
        rho = torch.clamp(distance / torch.clamp(distance + radius, min=1e-4), 0.0, 1.0)
        if self.survival_enabled:
            survival_semantic, survival_phi = self.survival_query_from_direction(ray_dir, rho, survival_coefficients)
        else:
            survival_semantic = torch.zeros((ray_dir.shape[0], self.survival_query_dim), device=ray_dir.device, dtype=ray_dir.dtype)
            survival_phi = torch.zeros((ray_dir.shape[0], 4), device=ray_dir.device, dtype=ray_dir.dtype)
            survival_coefficients = torch.zeros_like(survival_coefficients)
        return geo, context_query, survival_semantic, survival_phi, ray9, survival_coefficients

    def compute_logits_with_aux(
        self,
        camera_pos_norm: torch.Tensor,
        camera_view: torch.Tensor,
        camera_pos_world: torch.Tensor,
        instance_ids: torch.Tensor,
        runtime_features: torch.Tensor,
        **_unused: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del camera_pos_norm
        if runtime_features.shape[0] == self.num_instances:
            runtime_for_ids = runtime_features[instance_ids.long()]
        elif runtime_features.shape[0] == instance_ids.shape[0]:
            runtime_for_ids = runtime_features
        else:
            raise ValueError("runtime_features must be global [N,F] or aligned with instance_ids [B,F]")
        geo, context_query, survival_semantic, survival_phi, ray_query, survival_coefficients = self.query_runtime_features(
            camera_view, camera_pos_world, instance_ids, runtime_for_ids
        )
        # Reassemble the sanitized runtime table.  This also protects the
        # ablation when an evaluator supplies an externally exported table
        # containing non-zero survival coefficients.
        context_flat = runtime_for_ids[:, self.geo_dim : self.geo_dim + self.context_coeff_dim].reshape(-1, self.context_coeff_dim)
        # The base path is identical with and without the optional relation
        # residual.  The relation factor is therefore measured as an actual
        # additive contribution rather than as a hidden replacement of the
        # fixed context representation.
        context_query_for_base = context_query
        runtime_query = torch.cat([
            geo,
            context_flat,
            survival_coefficients.reshape(-1, self.survival_coeff_dim),
        ], dim=-1)
        # Keep the registered 177-D head while feeding it semantic survival
        # values rather than the direction basis that generated them.
        survival_visibility = torch.stack([
            survival_semantic[:, 0],
            survival_semantic[:, 2],
            survival_semantic[:, 3],
            survival_semantic[:, 4],
        ], dim=-1)
        base_input = torch.cat([
            runtime_query,
            ray_query,
            context_query_for_base,
            survival_visibility,
        ], dim=-1)
        base_hidden = self.base_trunk(base_input)
        base_logits = self.base_visibility_head(base_hidden)
        final_logits = base_logits
        final_probability = torch.sigmoid(final_logits)
        utility_input = torch.cat([base_hidden, final_probability], dim=-1)
        utility_logits = self.utility_head(utility_input)
        download_logits = self.download_head(utility_input)
        return final_logits, {
            "base_logits": base_logits,
            "final_logits": final_logits,
            "runtime_features_for_ids": runtime_query,
            "query_features_for_ids": base_hidden,
            "base_input_features": base_input,
            "context_query": context_query,
            "survival_basis": survival_phi,
            "survival_semantic": survival_semantic,
            "survival_visibility_features": survival_visibility,
            "survival_occlusion_probability": survival_semantic[:, 1:2],
            "ray_features": ray_query,
            "utility_logits": utility_logits,
            "utility_value": torch.sigmoid(utility_logits),
            "download_logits": download_logits,
            "evidence_target": torch.zeros_like(final_logits),
        }

    def compute_visibility_logits(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.compute_logits_with_aux(*args, **kwargs)[0]

    def regularization(self) -> torch.Tensor:
        terms = [
            parameter.pow(2).mean()
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.ndim > 1
        ]
        return torch.stack(terms).mean() if terms else torch.zeros((), device=self.scene_min.device)


def load_ray_context_survival_evidence(path: str | Path, num_instances: int) -> dict[str, Any]:
    root = Path(path)
    meta = json.loads((root / "evidence_meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != "ray-context-relation-evidence-v2":
        raise ValueError("evidence is not ray-context-relation-evidence-v2")
    if int(meta.get("directionBins")) != 12 or int(meta.get("depthShells")) != 3 or int(meta.get("sourceK")) != 8:
        raise ValueError("v2 evidence must use 12 directions, 3 shells, and 8 sources")
    n = int(num_instances)
    files = meta["files"]
    source_ids = np.fromfile(root / files["sourceIds"], dtype="<u4").reshape(n, 12, 3, 8).astype(np.int64)
    relation_stats = np.fromfile(root / files["relationStats"], dtype=np.float16).reshape(n, 12, 3, 8, 6).astype(np.float32)
    strength = np.fromfile(root / files["evidenceStrength"], dtype=np.float16).reshape(n, 12, 3).astype(np.float32)
    count = np.fromfile(root / files["evidenceCount"], dtype="<u4").reshape(n, 12, 3).astype(np.int64)
    observations = {
        "instance": np.fromfile(root / files["observationInstances"], dtype="<u4").astype(np.int64),
        "direction": np.fromfile(root / files["observationDirections"], dtype=np.uint8).astype(np.int64),
        "rho": np.fromfile(root / files["observationRho"], dtype=np.float16).astype(np.float32),
        "event": np.fromfile(root / files["observationEvent"], dtype=np.uint8).astype(np.float32),
        "weight": np.fromfile(root / files["observationWeight"], dtype=np.float16).astype(np.float32),
    }
    length = int(observations["instance"].size)
    if any(int(value.size) != length for value in observations.values()):
        raise ValueError("survival observation arrays have inconsistent lengths")
    checksums = meta.get("survivalObservationChecksums", {})
    if checksums and not bool(checksums.get("byteIdentical", False)):
        raise ValueError("v2 evidence does not prove byte-identical survival observations")
    return {
        "meta": meta,
        "source_ids": source_ids,
        "relation_stats": relation_stats,
        "strength": strength,
        "count": count,
        "observations": observations,
    }
