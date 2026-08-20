from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from current_pvs_utils import InstancePointNetPPGeoEncoder, VisibilityMLP, fourier_dim, fourier_features, mlp


class DirectionalOcclusionProxyEncoderPVSModel(nn.Module):
    """Offline-heavy instance point-cloud encoder with directional occlusion proxies.

    Runtime uses exported fixed features only. It does not read source occluder
    instances and does not run graph propagation in the browser.
    """

    point_feature_dim = 16

    def __init__(
        self,
        num_instances: int,
        num_glbs: int,
        geo_dim: int = 96,
        context_dim: int = 64,
        proxy_dim: int = 8,
        direction_bins: int = 8,
        depth_shells: int = 3,
        source_k: int = 8,
        point_hidden_dim: int = 160,
        pointnetpp_centers: int = 24,
        pointnetpp_neighbors: int = 12,
        graph_hidden_dim: int = 160,
        graph_message_dim: int = 96,
        ray_fourier_bands: int = 10,
        ray_scalar_fourier_bands: int = 4,
        camera_location_dim: int = 0,
        mlp_hidden: int = 128,
        interaction_dim: int = 64,
        scene_size_m: list[float] | tuple[float, ...] | None = None,
        runtime_feature_ablation: str = "none",
        use_explicit_inhibition: bool = True,
    ):
        super().__init__()
        valid_ablations = {
            "none",
            "proxy_zero",
            "context_proxy_zero",
            "geo_context_proxy_zero",
        }
        if runtime_feature_ablation not in valid_ablations:
            raise ValueError(
                f"runtime_feature_ablation must be one of {sorted(valid_ablations)}, "
                f"got {runtime_feature_ablation!r}"
            )
        self.num_instances = int(num_instances)
        self.num_glbs = int(num_glbs)
        self.geo_dim = int(geo_dim)
        self.context_dim = int(context_dim)
        self.proxy_dim = int(proxy_dim)
        self.direction_bins = int(direction_bins)
        self.depth_shells = int(depth_shells)
        self.source_k = int(source_k)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.graph_message_dim = int(graph_message_dim)
        self.ray_fourier_bands = int(ray_fourier_bands)
        self.ray_scalar_fourier_bands = int(ray_scalar_fourier_bands)
        self.camera_location_dim = int(max(0, camera_location_dim))
        self.runtime_feature_ablation = str(runtime_feature_ablation)
        self.use_explicit_inhibition = bool(use_explicit_inhibition)
        self.runtime_feature_dim = self.geo_dim + self.context_dim + self.direction_bins * self.depth_shells * self.proxy_dim
        self.query_feature_dim = self.geo_dim + self.context_dim + self.proxy_dim
        self.ray_direction_feature_dim = fourier_dim(3, self.ray_fourier_bands)
        self.ray_scalar_feature_dim = fourier_dim(6, self.ray_scalar_fourier_bands)
        self.camera_feature_dim = self.ray_direction_feature_dim + self.ray_scalar_feature_dim + self.camera_location_dim

        self.geo_encoder = InstancePointNetPPGeoEncoder(
            point_feature_dim=self.point_feature_dim,
            hidden_dim=point_hidden_dim,
            out_dim=geo_dim,
            center_count=pointnetpp_centers,
            neighbor_count=pointnetpp_neighbors,
        )
        node_dim = self.geo_dim + 6
        edge_dim = node_dim * 2 + 7
        self.edge_encoder = mlp([edge_dim, graph_hidden_dim, graph_message_dim], last_relu=True)
        self.context_head = mlp([node_dim + graph_message_dim * 2, graph_hidden_dim, context_dim], last_relu=True)
        self.proxy_head = mlp([node_dim + graph_message_dim + 2, graph_hidden_dim, proxy_dim], last_relu=True)
        self.ray_proxy_gate = mlp(
            [self.camera_feature_dim, max(64, graph_hidden_dim // 2), self.direction_bins * self.depth_shells],
            last_relu=False,
        )
        self.ray_runtime_gate = nn.Linear(self.ray_direction_feature_dim, self.query_feature_dim)
        self.visibility_mlp = VisibilityMLP(
            self.camera_feature_dim,
            0,
            self.query_feature_dim,
            mlp_hidden,
            interaction_dim,
        )
        self.inhibition_head = mlp([self.query_feature_dim + self.camera_feature_dim, mlp_hidden, 1])
        self.utility_head = mlp([self.query_feature_dim + 1, 96, 96, 1])
        self.download_head = mlp([self.query_feature_dim + 2, 96, 1])
        with torch.no_grad():
            last = self.utility_head[-1]
            if isinstance(last, nn.Linear):
                last.bias.fill_(-4.0)

        self.register_buffer("instance_world_aabbs", torch.zeros(self.num_instances, 6))
        self.register_buffer("instance_to_glb", torch.zeros(self.num_instances, dtype=torch.long))
        self.register_buffer("scene_min", torch.zeros(3))
        self.register_buffer("scene_size_m", torch.tensor(scene_size_m or [1.0, 1.0, 1.0], dtype=torch.float32))
        self.register_buffer("evidence_source_ids", torch.zeros(self.num_instances, self.direction_bins, self.depth_shells, self.source_k, dtype=torch.long))
        self.register_buffer("evidence_source_scores", torch.zeros(self.num_instances, self.direction_bins, self.depth_shells, self.source_k))
        self.register_buffer("evidence_strength", torch.zeros(self.num_instances, self.direction_bins, self.depth_shells))

    @property
    def config(self) -> dict[str, Any]:
        return {
            "runtimeSchema": "directional-occlusion-proxy-scheduler-v1",
            "numInstances": int(self.num_instances),
            "numGlbs": int(self.num_glbs),
            "geoDim": int(self.geo_dim),
            "contextDim": int(self.context_dim),
            "proxyDim": int(self.proxy_dim),
            "directionBins": int(self.direction_bins),
            "depthShells": int(self.depth_shells),
            "sourceK": int(self.source_k),
            "runtimeFeatureDim": int(self.runtime_feature_dim),
            "queryFeatureDim": int(self.query_feature_dim),
            "rayDirectionFeatureDim": int(self.ray_direction_feature_dim),
            "rayScalarFeatureDim": int(self.ray_scalar_feature_dim),
            "cameraFeatureDim": int(self.camera_feature_dim),
            "cameraLocationFeatureDim": int(self.camera_location_dim),
            "usesCameraLocationAux": bool(self.camera_location_dim > 0),
            "outputsVisualUtility": True,
            "outputsGlbPriority": True,
            "usesDynamicOcclusionPool": False,
            "usesDynamicTeacher": False,
            "runtimeUsesPointNet": False,
            "runtimeUsesGraphPropagation": False,
            "runtimeFeatureAblation": self.runtime_feature_ablation,
            "usesExplicitInhibition": bool(self.use_explicit_inhibition),
            "predictionSemantics": "fixed instance context and directional occlusion proxy features queried by current ray-space features",
        }

    def set_scene_bounds(self, scene_min: torch.Tensor, scene_size: torch.Tensor) -> None:
        self.scene_min.copy_(scene_min.float())
        self.scene_size_m.copy_(scene_size.float())

    def set_instance_world_aabbs(self, aabbs: torch.Tensor) -> None:
        if aabbs.shape != self.instance_world_aabbs.shape:
            raise ValueError(f"Expected instance_world_aabbs {tuple(self.instance_world_aabbs.shape)}, got {tuple(aabbs.shape)}")
        self.instance_world_aabbs.copy_(aabbs.float())

    def set_instance_to_glb(self, mapping: torch.Tensor) -> None:
        if mapping.shape != self.instance_to_glb.shape:
            raise ValueError(f"Expected instance_to_glb {tuple(self.instance_to_glb.shape)}, got {tuple(mapping.shape)}")
        self.instance_to_glb.copy_(mapping.long())

    def set_evidence(self, source_ids: torch.Tensor, source_scores: torch.Tensor, strength: torch.Tensor) -> None:
        if source_ids.shape != self.evidence_source_ids.shape:
            raise ValueError(f"Expected evidence source ids {tuple(self.evidence_source_ids.shape)}, got {tuple(source_ids.shape)}")
        if source_scores.shape != self.evidence_source_scores.shape:
            raise ValueError(f"Expected evidence source scores {tuple(self.evidence_source_scores.shape)}, got {tuple(source_scores.shape)}")
        if strength.shape != self.evidence_strength.shape:
            raise ValueError(f"Expected evidence strength {tuple(self.evidence_strength.shape)}, got {tuple(strength.shape)}")
        self.evidence_source_ids.copy_(source_ids.long())
        self.evidence_source_scores.copy_(source_scores.float())
        self.evidence_strength.copy_(strength.float())

    def _instance_node_attrs(self, instance_ids: torch.Tensor) -> torch.Tensor:
        bounds = self.instance_world_aabbs[instance_ids.long()]
        mins = bounds[:, :3]
        maxs = bounds[:, 3:]
        center = (mins + maxs) * 0.5
        size = torch.clamp(maxs - mins, min=1e-4)
        scene_size = torch.clamp(self.scene_size_m.to(center.device), min=1e-6)
        center_norm = torch.clamp((center - self.scene_min.to(center.device)) / scene_size, 0.0, 1.0)
        size_norm = torch.clamp(size / scene_size, 0.0, 4.0)
        return torch.cat([center_norm, size_norm], dim=-1)

    def point_features(self, instance_ids: torch.Tensor, glb_points: torch.Tensor) -> torch.Tensor:
        ids = instance_ids.long()
        # The offline point cache is instance-aligned. Several instances may share
        # one downloadable GLB after instancing, but they still have distinct
        # world-space orientation and AABB-conditioned geometry features.
        # The offline cache is indexed by unique GLB, while runtime metadata
        # indexes component instances. Resolve the shared geometry through the
        # explicit instance-to-GLB mapping before adding instance AABB context.
        local = glb_points[self.instance_to_glb[ids]]
        bounds = self.instance_world_aabbs[ids].to(local.device)
        mins = bounds[:, :3]
        maxs = bounds[:, 3:]
        center = (mins + maxs) * 0.5
        size = torch.clamp(maxs - mins, min=1e-4)
        scene_min = self.scene_min.to(local.device)
        scene_size = torch.clamp(self.scene_size_m.to(local.device), min=1e-6)
        center_norm = torch.clamp((center - scene_min) / scene_size, 0.0, 1.0)
        size_norm = torch.clamp(size / scene_size, 0.0, 4.0)
        uniform_scale = size.max(dim=-1, keepdim=True).values
        world = center[:, None, :] + local * uniform_scale[:, None, :]
        world_norm = torch.clamp((world - scene_min[None, None, :]) / scene_size[None, None, :], 0.0, 1.0)
        scale_ratio = size / torch.clamp(uniform_scale, min=1e-6)
        return torch.cat(
            [
                local,
                world_norm,
                center_norm[:, None, :].expand_as(local),
                size_norm[:, None, :].expand_as(local),
                scale_ratio[:, None, :].expand_as(local),
                torch.linalg.norm(local, dim=-1, keepdim=True),
            ],
            dim=-1,
        )

    def instance_geo_features(self, instance_ids: torch.Tensor, glb_points: torch.Tensor) -> torch.Tensor:
        return self.geo_encoder(self.point_features(instance_ids, glb_points))

    @staticmethod
    def _map_to_local(expanded_ids: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.searchsorted(expanded_ids, ids.reshape(-1)).reshape_as(ids)
        valid = pos < expanded_ids.numel()
        safe = torch.clamp(pos, 0, max(0, expanded_ids.numel() - 1))
        matched = expanded_ids[safe] == ids
        return torch.where(valid & matched, safe, torch.zeros_like(safe))

    def _edge_relative_features(self, target_ids: torch.Tensor, source_ids: torch.Tensor, source_scores: torch.Tensor) -> torch.Tensor:
        bt = self.instance_world_aabbs[target_ids.long()]
        bs = self.instance_world_aabbs[source_ids.long()]
        ct = (bt[:, :3] + bt[:, 3:]) * 0.5
        cs = (bs[:, :3] + bs[:, 3:]) * 0.5
        st = torch.clamp(bt[:, 3:] - bt[:, :3], min=1e-4)
        ss = torch.clamp(bs[:, 3:] - bs[:, :3], min=1e-4)
        scene_size = torch.clamp(self.scene_size_m.to(ct.device), min=1e-6)
        delta = (cs - ct) / scene_size
        dist = torch.linalg.norm(cs - ct, dim=-1, keepdim=True)
        rt = torch.linalg.norm(st, dim=-1, keepdim=True) * 0.5
        rs = torch.linalg.norm(ss, dim=-1, keepdim=True) * 0.5
        return torch.cat(
            [
                torch.clamp(delta, -4.0, 4.0),
                torch.log1p(dist / 100.0),
                torch.clamp(rs / torch.clamp(rt, min=1.0), 0.0, 8.0) / 8.0,
                torch.clamp(source_scores.reshape(-1, 1), 0.0, 1.0),
                (self.instance_to_glb[source_ids.long()] == self.instance_to_glb[target_ids.long()]).float().reshape(-1, 1),
            ],
            dim=-1,
        )

    def _encode_context_proxy_from_geo(
        self,
        target_ids: torch.Tensor,
        expanded_ids: torch.Tensor,
        expanded_geo: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_ids = target_ids.long()
        t_local = self._map_to_local(expanded_ids, target_ids)
        target_node = torch.cat([expanded_geo[t_local], self._instance_node_attrs(target_ids).to(expanded_geo.device)], dim=-1)
        sources = self.evidence_source_ids[target_ids].to(target_ids.device)
        scores = self.evidence_source_scores[target_ids].to(expanded_geo.device)
        s_local = self._map_to_local(expanded_ids, sources)
        source_geo = expanded_geo[s_local.reshape(-1)].reshape(*sources.shape, -1)
        source_attrs = self._instance_node_attrs(sources.reshape(-1)).to(expanded_geo.device).reshape(*sources.shape, -1)
        source_node = torch.cat([source_geo, source_attrs], dim=-1)
        b, bins, shells, k, node_dim = source_node.shape
        target_rep = target_node[:, None, None, None, :].expand(b, bins, shells, k, node_dim)
        rel = self._edge_relative_features(
            target_ids[:, None, None, None].expand_as(sources).reshape(-1),
            sources.reshape(-1),
            scores.reshape(-1),
        ).to(expanded_geo.device).reshape(b, bins, shells, k, -1)
        edge_in = torch.cat([target_rep, source_node, rel], dim=-1)
        msg = self.edge_encoder(edge_in.reshape(b * bins * shells * k, -1)).reshape(b, bins, shells, k, -1)
        weights = torch.clamp(scores, min=0.0).unsqueeze(-1)
        pooled = (msg * weights).sum(dim=3) / torch.clamp(weights.sum(dim=3), min=1e-4)
        proxy_parts = []
        strength = self.evidence_strength[target_ids].to(expanded_geo.device)
        for shell_id in range(shells):
            shell_msg = pooled[:, :, shell_id, :]
            shell_strength = strength[:, :, shell_id:shell_id + 1]
            bin_ids = torch.linspace(-1.0, 1.0, steps=bins, device=expanded_geo.device).view(1, bins, 1).expand(b, bins, 1)
            proxy_in = torch.cat([target_node[:, None, :].expand(b, bins, target_node.shape[-1]), shell_msg, shell_strength, bin_ids], dim=-1)
            proxy_parts.append(self.proxy_head(proxy_in.reshape(b * bins, -1)).reshape(b, bins, self.proxy_dim))
        proxy = torch.stack(proxy_parts, dim=2)
        mean_msg = pooled.mean(dim=(1, 2))
        max_msg = pooled.max(dim=2).values.max(dim=1).values
        context = self.context_head(torch.cat([target_node, mean_msg, max_msg], dim=-1))
        return context, proxy.reshape(b, bins * shells * self.proxy_dim)

    def features_for_ids(self, instance_ids: torch.Tensor, glb_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_ids, inverse = torch.unique(instance_ids.long(), sorted=True, return_inverse=True)
        source_ids = self.evidence_source_ids[unique_ids].reshape(-1).to(unique_ids.device)
        expanded_ids = torch.unique(torch.cat([unique_ids, source_ids.long()], dim=0), sorted=True)
        expanded_geo = self.instance_geo_features(expanded_ids, glb_points)
        context, proxy = self._encode_context_proxy_from_geo(unique_ids, expanded_ids, expanded_geo)
        unique_geo = expanded_geo[self._map_to_local(expanded_ids, unique_ids)]
        return unique_geo[inverse], context[inverse], proxy[inverse]

    @torch.no_grad()
    def compute_all_self_geo_features(self, glb_points: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        out = []
        device = self.instance_world_aabbs.device
        for start in range(0, self.num_instances, batch_size):
            end = min(self.num_instances, start + batch_size)
            ids = torch.arange(start, end, device=device)
            out.append(self.instance_geo_features(ids, glb_points).detach().cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def compute_all_context_proxy_from_self(self, self_geo: torch.Tensor, batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
        out_context = []
        out_proxy = []
        device = self.instance_world_aabbs.device
        all_ids = torch.arange(self.num_instances, device=device)
        self_geo = self_geo.to(device)
        for start in range(0, self.num_instances, batch_size):
            end = min(self.num_instances, start + batch_size)
            ids = all_ids[start:end]
            context, proxy = self._encode_context_proxy_from_geo(ids, all_ids, self_geo)
            out_context.append(context.detach().cpu())
            out_proxy.append(proxy.detach().cpu())
        return torch.cat(out_context, dim=0), torch.cat(out_proxy, dim=0)

    @torch.no_grad()
    def compute_all_runtime_features(self, glb_points: torch.Tensor, batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        geo = self.compute_all_self_geo_features(glb_points, batch_size=batch_size)
        context, proxy = self.compute_all_context_proxy_from_self(geo, batch_size=max(256, batch_size))
        runtime = torch.cat([geo, context, proxy], dim=-1)
        return geo, context, proxy, runtime

    def _split_runtime(self, runtime_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geo = runtime_features[:, : self.geo_dim]
        context = runtime_features[:, self.geo_dim : self.geo_dim + self.context_dim]
        proxy = runtime_features[:, self.geo_dim + self.context_dim :].reshape(-1, self.direction_bins, self.depth_shells, self.proxy_dim)
        return geo, context, proxy

    def ray_features(
        self,
        camera_view: torch.Tensor,
        instance_ids: torch.Tensor,
        camera_pos_world: torch.Tensor,
        camera_pos_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bounds = self.instance_world_aabbs[instance_ids.long()].to(camera_pos_world.device)
        mins = bounds[:, :3]
        maxs = bounds[:, 3:]
        center = (mins + maxs) * 0.5
        size = torch.clamp(maxs - mins, min=1e-4)
        delta = center - camera_pos_world
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        ray_dir = delta / torch.clamp(dist, min=1e-4)
        ray_dir_feat = fourier_features(ray_dir, self.ray_fourier_bands)
        forward = F.normalize(camera_view[:, :3], dim=-1, eps=1e-6)
        up_ref = torch.zeros_like(forward)
        up_ref[:, 1] = 1.0
        alt_up = torch.zeros_like(forward)
        alt_up[:, 2] = 1.0
        up_seed = torch.where(torch.abs((forward * up_ref).sum(dim=-1, keepdim=True)) > 0.98, alt_up, up_ref)
        right = F.normalize(torch.cross(forward, up_seed, dim=-1), dim=-1, eps=1e-6)
        up = F.normalize(torch.cross(right, forward, dim=-1), dim=-1, eps=1e-6)
        dot_forward = (ray_dir * forward).sum(dim=-1, keepdim=True)
        dot_right = (ray_dir * right).sum(dim=-1, keepdim=True)
        dot_up = (ray_dir * up).sum(dim=-1, keepdim=True)
        tan_x = torch.clamp(camera_view[:, 3:4], min=1e-4)
        tan_y = torch.clamp(camera_view[:, 4:5], min=1e-4)
        denom_x = torch.clamp(torch.abs(dot_forward) * tan_x, min=1e-4)
        denom_y = torch.clamp(torch.abs(dot_forward) * tan_y, min=1e-4)
        u = dot_right / denom_x
        v = dot_up / denom_y
        radius = torch.linalg.norm(size, dim=-1, keepdim=True) * 0.5
        angular = radius / torch.clamp(dist, min=1.0)
        scalar_values = torch.cat(
            [
                torch.clamp(torch.log1p(dist / 100.0) / 4.0, 0.0, 2.0) - 1.0,
                torch.clamp(dot_forward, -1.0, 1.0),
                torch.clamp(u, -4.0, 4.0) / 4.0,
                torch.clamp(v, -4.0, 4.0) / 4.0,
                torch.clamp(angular / tan_x, 0.0, 4.0) / 2.0 - 1.0,
                torch.clamp(angular / tan_y, 0.0, 4.0) / 2.0 - 1.0,
            ],
            dim=-1,
        )
        scalar_feat = fourier_features(scalar_values, self.ray_scalar_fourier_bands)
        camera_features = [ray_dir_feat, scalar_feat]
        if self.camera_location_dim > 0:
            if camera_pos_norm is None:
                location = torch.zeros(
                    (camera_view.shape[0], 3),
                    device=camera_view.device,
                    dtype=camera_view.dtype,
                )
            else:
                location = torch.clamp(camera_pos_norm[:, :3], 0.0, 1.0).to(camera_view.dtype)
            if self.camera_location_dim < 3:
                location = location[:, : self.camera_location_dim]
            elif self.camera_location_dim > 3:
                location = F.pad(location, (0, self.camera_location_dim - 3))
            camera_features.append(location)
        return ray_dir, ray_dir_feat, scalar_feat, torch.cat(camera_features, dim=-1)

    def query_runtime_features(
        self,
        camera_view: torch.Tensor,
        camera_pos_world: torch.Tensor,
        instance_ids: torch.Tensor,
        runtime_features: torch.Tensor,
        camera_pos_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geo, context, proxy = self._split_runtime(runtime_features)
        # M4 uses registered input ablations while keeping the architecture,
        # candidate set and threshold protocol identical.  The offline table
        # remains present for byte/schema accounting, but the selected branch
        # is prevented from reaching the runtime query and its task heads.
        if self.runtime_feature_ablation == "proxy_zero":
            proxy = torch.zeros_like(proxy)
        elif self.runtime_feature_ablation == "context_proxy_zero":
            context = torch.zeros_like(context)
            proxy = torch.zeros_like(proxy)
        elif self.runtime_feature_ablation == "geo_context_proxy_zero":
            geo = torch.zeros_like(geo)
            context = torch.zeros_like(context)
            proxy = torch.zeros_like(proxy)
        _ray_dir, ray_dir_feat, _scalar_feat, camera_feat = self.ray_features(
            camera_view,
            instance_ids,
            camera_pos_world,
            camera_pos_norm,
        )
        weights = torch.softmax(self.ray_proxy_gate(camera_feat), dim=1).reshape(-1, self.direction_bins, self.depth_shells)
        selected_proxy = (proxy * weights[..., None]).sum(dim=(1, 2))
        query = torch.cat([geo, context, selected_proxy], dim=-1)
        gate = torch.tanh(self.ray_runtime_gate(ray_dir_feat))
        query = query * (1.0 + gate)
        return camera_feat, query, selected_proxy

    def compute_logits_with_aux(
        self,
        camera_pos_norm: torch.Tensor,
        camera_view: torch.Tensor,
        camera_pos_world: torch.Tensor,
        instance_ids: torch.Tensor,
        glb_points: torch.Tensor | None = None,
        runtime_features: torch.Tensor | None = None,
        **_unused: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if runtime_features is None:
            if glb_points is None:
                raise ValueError("glb_points is required when runtime_features is not provided")
            geo, context, proxy = self.features_for_ids(instance_ids, glb_points)
            runtime_for_ids = torch.cat([geo, context, proxy], dim=-1)
        else:
            runtime_for_ids = runtime_features[instance_ids.long()]
        camera_feat, query, selected_proxy = self.query_runtime_features(
            camera_view,
            camera_pos_world,
            instance_ids,
            runtime_for_ids,
            camera_pos_norm=camera_pos_norm,
        )
        base_logits = self.visibility_mlp.logits(camera_feat, torch.empty((query.shape[0], 0), device=query.device, dtype=query.dtype), query)
        if self.use_explicit_inhibition:
            inhibition = torch.clamp(
                F.softplus(self.inhibition_head(torch.cat([query, camera_feat], dim=-1))), 0.0, 4.0
            )
        else:
            inhibition = torch.zeros_like(base_logits)
        final_logits = base_logits - inhibition
        utility_logits = self.compute_utility_logits(query, final_logits)
        download_logits = self.compute_download_logits(query, final_logits, utility_logits)
        evidence_target = self.query_evidence_strength(
            camera_view,
            camera_pos_world,
            instance_ids,
            camera_pos_norm=camera_pos_norm,
        )
        return final_logits, {
            "base_logits": base_logits,
            "prior_logits": base_logits,
            "final_logits": final_logits,
            "inhibition": inhibition,
            "runtime_features_for_ids": runtime_for_ids,
            "query_features_for_ids": query,
            "camera_features_for_ids": camera_feat,
            "selected_proxy": selected_proxy,
            "evidence_target": evidence_target,
            "utility_logits": utility_logits,
            "utility_value": torch.sigmoid(utility_logits),
            "download_logits": download_logits,
        }

    def compute_visibility_logits(self, *args, **kwargs) -> torch.Tensor:
        logits, _aux = self.compute_logits_with_aux(*args, **kwargs)
        return logits

    def query_evidence_strength(
        self,
        camera_view: torch.Tensor,
        camera_pos_world: torch.Tensor,
        instance_ids: torch.Tensor,
        camera_pos_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ray_dir, _ray_dir_feat, _scalar_feat, camera_feat = self.ray_features(
            camera_view,
            instance_ids,
            camera_pos_world,
            camera_pos_norm,
        )
        weights = torch.softmax(self.ray_proxy_gate(camera_feat), dim=1).reshape(-1, self.direction_bins, self.depth_shells)
        strength = self.evidence_strength[instance_ids.long()].to(camera_feat.device, camera_feat.dtype)
        return (strength * weights).sum(dim=(1, 2), keepdim=True)

    def compute_utility_logits(self, query_features: torch.Tensor, visibility_logits: torch.Tensor) -> torch.Tensor:
        return self.utility_head(torch.cat([query_features, torch.sigmoid(visibility_logits).view(-1, 1)], dim=-1))

    def compute_download_logits(self, query_features: torch.Tensor, visibility_logits: torch.Tensor, utility_logits: torch.Tensor | None = None) -> torch.Tensor:
        if utility_logits is None:
            utility_logits = self.compute_utility_logits(query_features, visibility_logits)
        return self.download_head(
            torch.cat(
                [query_features, torch.sigmoid(visibility_logits).view(-1, 1), torch.sigmoid(utility_logits).view(-1, 1)],
                dim=-1,
            )
        )

    def regularization(self) -> torch.Tensor:
        terms = [p.pow(2).mean() for p in self.parameters() if p.requires_grad and p.ndim > 1]
        return torch.stack(terms).mean() if terms else torch.zeros((), device=self.scene_min.device)


def load_directional_occlusion_evidence(path: str | Path, num_instances: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    evidence_dir = Path(path)
    meta = json.loads((evidence_dir / "evidence_meta.json").read_text(encoding="utf-8"))
    n = int(num_instances)
    bins = int(meta["directionBins"])
    shells = int(meta["depthShells"])
    k = int(meta["sourceK"])
    strength = np.fromfile(evidence_dir / meta["files"]["evidenceStrength"], dtype=np.float16).reshape(n, bins, shells).astype(np.float32)
    source_id_dtype_name = str(meta.get("sourceIdDtype", "uint16")).lower()
    source_id_dtype = {
        "uint16": np.uint16,
        "uint32": np.uint32,
    }.get(source_id_dtype_name)
    if source_id_dtype is None:
        raise ValueError(f"Unsupported directional evidence sourceIdDtype: {source_id_dtype_name}")
    source_ids = np.fromfile(
        evidence_dir / meta["files"]["sourceIds"],
        dtype=source_id_dtype,
    ).reshape(n, bins, shells, k).astype(np.int64)
    source_scores = np.fromfile(evidence_dir / meta["files"]["sourceScores"], dtype=np.float16).reshape(n, bins, shells, k).astype(np.float32)
    return source_ids, source_scores, strength, meta
