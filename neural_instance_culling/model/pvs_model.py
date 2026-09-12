#!/usr/bin/env python3
"""Current V4 PVS model and its registered paper controls.

The relation graph is train-only. Runtime receives one fixed geometry and
occlusion row per instance, then performs one horizontal view-cell query.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from common.relation_encoder import BoundedHierarchicalOcclusionSurvivalEncoder
from common.spectral_query import ViewCellMomentEnvelopeSpectralQuery
from common.viewcell_ray_space import (
    build_horizontal_disk_ray_query,
    export_ray_space_schema,
    normalized_relative_log_depth,
)


MODEL_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-envelope-v4"
GEO_DIM = 96
SURVIVAL_RANK = 4
SURVIVAL_PARAMETER_DIM = 7
SURVIVAL_DIM = SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM
SUPPORTED_SURVIVAL_RANKS = (2, 4, 8, 12)
RUNTIME_FEATURE_DIM = GEO_DIM + SURVIVAL_DIM
VIEW_DIM = 9
DISK_AXIS_DIM = 2
SPECTRAL_FREQUENCY_COUNT = 16
SPECTRAL_MOMENT_DIM = SPECTRAL_FREQUENCY_COUNT * 4
BOUNDARY_SUMMARY_DIM = 8
LOW_RANK_SUMMARY_DIM = 4
SURVIVAL_SEMANTIC_DIM = 8
RELATION_CONDITION_DIM = 8
INSTANCE_CALIBRATION_MODES = ("residual", "disabled")
OCCLUSION_REPRESENTATION_MODES = ("survival", "generic28", "none")
SPECTRAL_MODES = ("moment_envelope", "point")
RUNTIME_HEAD_INPUT_DIM = (
    GEO_DIM + SURVIVAL_RANK + SURVIVAL_SEMANTIC_DIM + BOUNDARY_SUMMARY_DIM
    + VIEW_DIM + LOW_RANK_SUMMARY_DIM + 1
)
RUNTIME_HEAD_INPUT_DIM_WITHOUT_OCCLUSION = (
    GEO_DIM + BOUNDARY_SUMMARY_DIM + VIEW_DIM + LOW_RANK_SUMMARY_DIM + 1
)


def _float_tensor(value: Any, name: str, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if device is not None:
        tensor = tensor.to(device=device)
    tensor = tensor.float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _rows(value: Any, width: int, name: str, *, device: torch.device) -> torch.Tensor:
    tensor = _float_tensor(value, name, device=device)
    if tensor.ndim == 1 and tensor.numel() == width:
        tensor = tensor.reshape(1, width)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape [{width}] or [B, {width}]")
    return tensor


def _broadcast_rows(
    value: Any, batch: int, width: int, name: str, *, device: torch.device
) -> torch.Tensor:
    tensor = _rows(value, width, name, device=device)
    if tensor.shape[0] == 1 and batch != 1:
        tensor = tensor.expand(batch, -1)
    if tensor.shape[0] != batch:
        raise ValueError(f"{name} must contain one row or {batch} rows")
    return tensor


def _broadcast_scalar(value: Any, batch: int, name: str, *, device: torch.device) -> torch.Tensor:
    tensor = _float_tensor(value, name, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    if tensor.ndim != 2 or tensor.shape[1] != 1 or tensor.shape[0] not in (1, batch):
        raise ValueError(f"{name} must be scalar, [B], or [B, 1]")
    if tensor.shape[0] == 1 and batch != 1:
        tensor = tensor.expand(batch, -1)
    return tensor


def _parameter_count(module: nn.Module) -> int:
    return sum(value.numel() for value in module.parameters() if value.requires_grad)


def _capacity_matched_widths(target_parameters: int, output_dim: int) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    for first in range(32, 513):
        constant = GEO_DIM * first + first + output_dim
        coefficient = first + output_dim + 1
        second = max(1, round((target_parameters - constant) / coefficient))
        count = constant + coefficient * second
        error = abs(count - target_parameters) / max(target_parameters, 1)
        candidate = (error + abs(first - second) * 1e-8, first, second)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    if best[0] > 0.01:
        raise RuntimeError("could not capacity-match the geometry-only encoder")
    return best[1], best[2]


def _query_survival_semantic(
    basis: torch.Tensor, normalized_depth: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    params = torch.einsum("br,brp->bp", basis, coefficients)
    no_block = torch.sigmoid(params[:, 0:1])
    weights = torch.softmax(params[:, 1:3], dim=-1)
    depths = 0.05 + 0.90 * torch.sigmoid(params[:, 3:5])
    scales = 0.03 + F.softplus(params[:, 5:7])
    depths, order = torch.sort(depths, dim=-1)
    weights = torch.gather(weights, -1, order)
    scales = torch.gather(scales, -1, order)
    cdf_terms = torch.sigmoid((normalized_depth - depths) / scales.clamp_min(1e-3))
    cdf = (1.0 - no_block) * (weights * cdf_terms).sum(-1, keepdim=True)
    survival = (1.0 - cdf).clamp(1e-5, 1.0)
    expected_depth = (weights * depths).sum(-1, keepdim=True)
    variance = (weights * (depths - expected_depth).square()).sum(-1, keepdim=True)
    slope = (1.0 - no_block) * (
        weights / scales.clamp_min(1e-3) * cdf_terms * (1.0 - cdf_terms)
    ).sum(-1, keepdim=True)
    result = torch.cat(
        [survival, 1.0 - survival, no_block, expected_depth,
         variance.clamp_min(1e-8).sqrt(), weights, slope], dim=-1
    )
    if result.shape[1] != SURVIVAL_SEMANTIC_DIM or not bool(torch.isfinite(result).all()):
        raise FloatingPointError("survival query produced invalid semantics")
    return result


def _query_generic_semantic(
    basis: torch.Tensor, normalized_depth: torch.Tensor, features: torch.Tensor
) -> torch.Tensor:
    projected = torch.einsum("br,brp->bp", basis, features)
    result = torch.tanh(
        torch.cat([projected, projected[:, :1] * (2.0 * normalized_depth - 1.0)], dim=-1)
    )
    if result.shape[1] != SURVIVAL_SEMANTIC_DIM or not bool(torch.isfinite(result).all()):
        raise FloatingPointError("generic occlusion query produced invalid semantics")
    return result


class BoundedRelationSurvivalMomentModel(nn.Module):
    """Offline relation PVS with one horizontal-disk runtime query."""

    def __init__(
        self,
        num_instances: int,
        num_glbs: int = 0,
        *,
        relation_hidden_dim: int = 64,
        hidden_dim: int = 64,
        survival_rank: int = SURVIVAL_RANK,
        relation_source: str = "bounded_hierarchical",
        occlusion_representation: str = "survival",
        spectral_mode: str = "moment_envelope",
        depth_q01: float = 0.0,
        depth_q99: float = 1.0,
        depth_epsilon: float = 1e-6,
        max_frequency_norm_cycles: float = 8.0,
        instance_calibration_mode: str = "residual",
        instance_calibration_max_abs: float = 4.0,
        sparse_instance_penalty: float = 3.0,
    ) -> None:
        super().__init__()
        if num_instances <= 0 or num_glbs < 0:
            raise ValueError("num_instances must be positive and num_glbs non-negative")
        if survival_rank not in SUPPORTED_SURVIVAL_RANKS:
            raise ValueError(f"survival_rank must be one of {SUPPORTED_SURVIVAL_RANKS}")
        if occlusion_representation not in OCCLUSION_REPRESENTATION_MODES:
            raise ValueError(f"unknown occlusion representation: {occlusion_representation}")
        if occlusion_representation == "generic28" and survival_rank != 4:
            raise ValueError("generic28 requires the four-by-seven layout")
        if occlusion_representation == "survival":
            if relation_source not in {"bounded_hierarchical", "geometry_only"}:
                raise ValueError("survival requires a registered relation source")
        elif relation_source != "none":
            raise ValueError("generic28 and none require relation_source='none'")
        if spectral_mode not in SPECTRAL_MODES:
            raise ValueError(f"spectral_mode must be one of {SPECTRAL_MODES}")
        if instance_calibration_mode not in INSTANCE_CALIBRATION_MODES:
            raise ValueError(f"unknown instance calibration: {instance_calibration_mode}")
        if occlusion_representation != "survival" and instance_calibration_mode != "disabled":
            raise ValueError("only survival supports instance calibration")
        if depth_q99 <= depth_q01:
            raise ValueError("depth_q99 must be greater than depth_q01")
        if depth_epsilon <= 0 or max_frequency_norm_cycles <= 0:
            raise ValueError("depth and frequency bounds must be positive")
        if instance_calibration_max_abs <= 0 or sparse_instance_penalty < 0:
            raise ValueError("instance calibration bounds are invalid")

        self.num_instances = int(num_instances)
        self.num_glbs = int(num_glbs)
        self.relation_hidden_dim = int(relation_hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.survival_rank = int(survival_rank)
        self.survival_parameter_dim = SURVIVAL_PARAMETER_DIM
        self.survival_dim = self.survival_rank * self.survival_parameter_dim
        self.relation_source = relation_source
        self.occlusion_representation = occlusion_representation
        self.spectral_mode = spectral_mode
        self.depth_q01 = float(depth_q01)
        self.depth_q99 = float(depth_q99)
        self.depth_epsilon = float(depth_epsilon)
        self.max_frequency_norm_cycles = float(max_frequency_norm_cycles)
        self.instance_calibration_mode = instance_calibration_mode
        self.instance_calibration_max_abs = float(instance_calibration_max_abs)
        self.sparse_instance_penalty = float(sparse_instance_penalty)
        self.runtime_feature_dim = GEO_DIM + (
            self.survival_dim if occlusion_representation != "none" else 0
        )
        self.runtime_head_input_dim = (
            GEO_DIM + self.survival_rank + SURVIVAL_SEMANTIC_DIM + BOUNDARY_SUMMARY_DIM
            + VIEW_DIM + LOW_RANK_SUMMARY_DIM + 1
            if occlusion_representation != "none"
            else RUNTIME_HEAD_INPUT_DIM_WITHOUT_OCCLUSION
        )

        if occlusion_representation == "survival":
            relation_encoder = BoundedHierarchicalOcclusionSurvivalEncoder(
                geo_dim=GEO_DIM,
                hidden_dim=self.relation_hidden_dim,
                survival_rank=self.survival_rank,
                survival_parameter_dim=self.survival_parameter_dim,
            )
            if relation_source == "bounded_hierarchical":
                self.offline_survival_encoder = relation_encoder
                self.geometry_only_survival_encoder = None
                self.geometry_only_widths = None
            else:
                widths = _capacity_matched_widths(
                    _parameter_count(relation_encoder), self.survival_dim
                )
                self.geometry_only_widths = widths
                self.geometry_only_survival_encoder = nn.Sequential(
                    nn.Linear(GEO_DIM, widths[0]), nn.SiLU(),
                    nn.Linear(widths[0], widths[1]), nn.SiLU(),
                    nn.Linear(widths[1], self.survival_dim),
                )
                self.offline_survival_encoder = None
        else:
            self.offline_survival_encoder = None
            self.geometry_only_survival_encoder = None
            self.geometry_only_widths = None

        if occlusion_representation == "generic28":
            self.generic_occlusion_features = nn.Parameter(
                torch.empty(self.num_instances, self.survival_rank, SURVIVAL_PARAMETER_DIM)
            )
            nn.init.normal_(self.generic_occlusion_features, mean=0.0, std=0.02)
        else:
            self.register_parameter("generic_occlusion_features", None)
        if occlusion_representation == "survival" and instance_calibration_mode == "residual":
            self.instance_calibration_residual_raw = nn.Parameter(
                torch.zeros(self.num_instances, self.survival_rank, SURVIVAL_PARAMETER_DIM)
            )
        else:
            self.register_parameter("instance_calibration_residual_raw", None)
        self.register_buffer(
            "instance_calibration_reliability", torch.zeros(self.num_instances), persistent=True
        )
        self.register_buffer("instance_calibration_blend", torch.zeros(()), persistent=True)

        self.moment_query = ViewCellMomentEnvelopeSpectralQuery(learnable_frequencies=True)
        if self.moment_query.frequency_count != SPECTRAL_FREQUENCY_COUNT:
            raise RuntimeError("V4 requires sixteen joint frequencies")
        if occlusion_representation != "none":
            self.relation_condition_head: nn.Module | None = nn.Sequential(
                nn.Linear(self.survival_dim, 16), nn.SiLU(), nn.Linear(16, RELATION_CONDITION_DIM)
            )
            self.direction_basis_head: nn.Module | None = nn.Sequential(
                nn.Linear(BOUNDARY_SUMMARY_DIM + RELATION_CONDITION_DIM + 3, 24),
                nn.SiLU(), nn.Linear(24, self.survival_rank), nn.Tanh(),
            )
        else:
            self.relation_condition_head = None
            self.direction_basis_head = None
        boundary_input_dim = SPECTRAL_MOMENT_DIM + VIEW_DIM + LOW_RANK_SUMMARY_DIM
        if occlusion_representation != "none":
            boundary_input_dim += RELATION_CONDITION_DIM
        self.boundary_summary_head = nn.Sequential(
            nn.Linear(boundary_input_dim, 48), nn.SiLU(),
            nn.Linear(48, BOUNDARY_SUMMARY_DIM), nn.Tanh(),
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(self.runtime_head_input_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
        )
        self.visibility_head = nn.Linear(self.hidden_dim, 1)
        self.utility_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )
        self.download_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )
        self.register_buffer(
            "instance_world_aabbs", torch.zeros(self.num_instances, 6), persistent=False
        )
        self.register_buffer(
            "instance_to_glb", torch.zeros(self.num_instances, dtype=torch.long), persistent=False
        )

    def reset_runtime_heads(self) -> None:
        """Reinitialize the lightweight runtime query trunk and task heads."""
        for root in (
            self.shared_trunk,
            self.visibility_head,
            self.utility_head,
            self.download_head,
        ):
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

    @property
    def config(self) -> dict[str, Any]:
        has_occlusion = self.occlusion_representation != "none"
        is_survival = self.occlusion_representation == "survival"
        return {
            "runtimeSchema": MODEL_SCHEMA,
            "numInstances": self.num_instances,
            "numGlbs": self.num_glbs,
            "relationSource": self.relation_source,
            "occlusionRepresentation": {
                "mode": self.occlusion_representation,
                "featureDim": self.survival_dim if has_occlusion else 0,
                "directionRank": self.survival_rank if has_occlusion else 0,
                "parameterDim": SURVIVAL_PARAMETER_DIM if has_occlusion else 0,
            },
            "spectralMode": self.spectral_mode,
            "hiddenDim": self.hidden_dim,
            "relationHiddenDim": self.relation_hidden_dim,
            "instanceCalibration": {
                "mode": self.instance_calibration_mode,
                "shape": [self.survival_rank, SURVIVAL_PARAMETER_DIM] if is_survival else [],
                "initialization": "zero",
                "maximumAbsoluteResidual": self.instance_calibration_max_abs,
                "sparseInstancePenalty": self.sparse_instance_penalty,
                "fusion": "survival_prior + blend * bounded_instance_residual" if is_survival else "not applicable",
                "runtimeExport": "fused coefficients only" if is_survival else "not applicable",
            },
            "geometryDim": GEO_DIM,
            "survivalCoefficientShape": [self.survival_rank, SURVIVAL_PARAMETER_DIM] if is_survival else [],
            "runtimeFeatureDim": self.runtime_feature_dim,
            "runtimeHeadInputDim": self.runtime_head_input_dim,
            "boundarySummaryDim": BOUNDARY_SUMMARY_DIM,
            "lowRankSummaryDim": LOW_RANK_SUMMARY_DIM,
            "depthNormalization": {
                "definition": "clip((log1p(distance/(radius+epsilon))-q01)/(q99-q01),0,1)",
                "q01": self.depth_q01,
                "q99": self.depth_q99,
                "epsilon": self.depth_epsilon,
                "sourceSplit": "train",
            },
            "frequency": {
                "count": SPECTRAL_FREQUENCY_COUNT,
                "units": "cycles",
                "maxNormCycles": self.max_frequency_norm_cycles,
            },
            "runtimeForbidden": [
                "relation CSR", "group IDs", "neighbor search",
                "subpose expansion", "online graph propagation",
            ],
        }

    def export_schema(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SCHEMA,
            "config": self.config,
            "raySpace": export_ray_space_schema(),
            "outputs": {
                "visibilityLogits": ["B", 1],
                "utilityLogits": ["B", 1],
                "downloadLogits": ["B", 1],
            },
        }

    def set_instance_world_aabbs(self, value: Any) -> None:
        tensor = _float_tensor(value, "instance_world_aabbs", device=self.instance_world_aabbs.device)
        if tensor.shape != (self.num_instances, 6):
            raise ValueError(f"instance_world_aabbs must have shape [{self.num_instances}, 6]")
        self.instance_world_aabbs = tensor

    def set_instance_to_glb(self, value: Any) -> None:
        tensor = torch.as_tensor(value, dtype=torch.long, device=self.instance_to_glb.device).reshape(-1)
        if tensor.numel() != self.num_instances or bool((tensor < 0).any()):
            raise ValueError("instance_to_glb must contain one non-negative ID per instance")
        self.instance_to_glb = tensor

    def set_instance_calibration_reliability(self, value: Any) -> None:
        tensor = _float_tensor(
            value, "instance_calibration_reliability",
            device=self.instance_calibration_reliability.device,
        ).reshape(-1)
        if tensor.numel() != self.num_instances or bool((tensor < 0).any()) or bool((tensor > 1).any()):
            raise ValueError("instance calibration reliability must contain values in [0, 1]")
        self.instance_calibration_reliability.copy_(tensor)

    def set_instance_calibration_blend(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("instance calibration blend must lie in [0, 1]")
        self.instance_calibration_blend.fill_(
            value if self.instance_calibration_mode == "residual" else 0.0
        )

    def _bounded_instance_calibration_residual(self) -> torch.Tensor:
        if self.instance_calibration_residual_raw is None:
            return self.instance_calibration_reliability.new_zeros(
                self.num_instances, self.survival_rank, SURVIVAL_PARAMETER_DIM
            )
        scale = self.instance_calibration_max_abs
        return scale * torch.tanh(self.instance_calibration_residual_raw / scale)

    def instance_calibration_regularization(self) -> torch.Tensor:
        residual = self._bounded_instance_calibration_residual()
        if self.instance_calibration_residual_raw is None:
            return residual.sum() * 0.0
        weights = 1.0 + self.sparse_instance_penalty * (1.0 - self.instance_calibration_reliability)
        return ((weights / weights.mean().clamp_min(1e-6)).view(-1, 1, 1) * residual.square()).mean()

    def instance_calibration_diagnostics(self) -> dict[str, torch.Tensor]:
        residual = self._bounded_instance_calibration_residual()
        applied = self.instance_calibration_blend * residual
        return {
            "instanceCalibrationBlend": self.instance_calibration_blend,
            "instanceCalibrationReliabilityMean": self.instance_calibration_reliability.mean(),
            "instanceCalibrationReliabilityZeroFraction": (self.instance_calibration_reliability <= 0).float().mean(),
            "instanceCalibrationResidualRms": residual.square().mean().sqrt(),
            "instanceCalibrationAppliedRms": applied.square().mean().sqrt(),
            "instanceCalibrationResidualMaxAbs": residual.abs().amax(),
            "instanceCalibrationActiveFraction": (residual.flatten(1).norm(dim=-1) > 1e-6).float().mean(),
        }

    def offline_encode_survival(
        self, geometry: Any, relation: Any, local_group_ids: Any,
        structural_group_ids: Any, *, return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        geo = _float_tensor(geometry, "geometry")
        if geo.shape != (self.num_instances, GEO_DIM):
            raise ValueError(f"geometry must have shape [{self.num_instances}, {GEO_DIM}]")
        if self.occlusion_representation == "generic28":
            assert self.generic_occlusion_features is not None
            features = self.generic_occlusion_features
            zeros = torch.zeros_like(features)
            diagnostics: dict[str, Any] = {
                "occlusion_features": features,
                "survival_coefficients": features,
                "survival_prior_coefficients": features,
                "instance_calibration_residual": zeros,
                "instance_calibration_applied_residual": zeros,
                "instance_calibration_blend": self.instance_calibration_blend,
                "instance_calibration_reliability": self.instance_calibration_reliability,
                "hierarchy": "visibility_trained_unstructured_generic28",
            }
            return diagnostics if return_diagnostics else features
        if self.occlusion_representation == "none":
            features = geo.new_empty((self.num_instances, 0))
            diagnostics = {
                "occlusion_features": features,
                "survival_coefficients": features,
                "survival_prior_coefficients": features,
                "instance_calibration_residual": features,
                "instance_calibration_applied_residual": features,
                "instance_calibration_blend": self.instance_calibration_blend,
                "instance_calibration_reliability": self.instance_calibration_reliability,
                "hierarchy": "disabled",
            }
            return diagnostics if return_diagnostics else features
        if self.relation_source == "geometry_only":
            assert self.geometry_only_survival_encoder is not None
            prior = self.geometry_only_survival_encoder(geo).reshape(
                self.num_instances, self.survival_rank, SURVIVAL_PARAMETER_DIM
            )
            diagnostics = {
                "survival_prior_coefficients": prior,
                "geometry_base_coefficients": prior,
                "relation_delta_coefficients": torch.zeros_like(prior),
                "relation_gate": torch.zeros(self.num_instances, 1, 1, device=geo.device),
                "hierarchy": "capacity_matched_geometry_only",
            }
        else:
            assert self.offline_survival_encoder is not None
            result = self.offline_survival_encoder(
                geo, relation,
                torch.as_tensor(local_group_ids, dtype=torch.long, device=geo.device),
                torch.as_tensor(structural_group_ids, dtype=torch.long, device=geo.device),
                return_diagnostics=True,
            )
            if not isinstance(result, Mapping):
                raise TypeError("offline relation encoder did not return diagnostics")
            diagnostics = dict(result)
            prior = diagnostics["survival_coefficients"]
            diagnostics["survival_prior_coefficients"] = prior
        residual = self._bounded_instance_calibration_residual()
        applied = self.instance_calibration_blend * residual
        coefficients = prior + applied
        if not bool(torch.isfinite(coefficients).all()):
            raise FloatingPointError("calibrated survival coefficients are non-finite")
        diagnostics.update({
            "occlusion_features": coefficients,
            "survival_coefficients": coefficients,
            "coefficients": coefficients,
            "instance_calibration_residual": residual,
            "instance_calibration_applied_residual": applied,
            "instance_calibration_blend": self.instance_calibration_blend,
            "instance_calibration_reliability": self.instance_calibration_reliability,
        })
        return diagnostics if return_diagnostics else coefficients

    def relation_consistency_loss(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.occlusion_representation != "survival" or self.offline_survival_encoder is None:
            zero = next(self.parameters()).sum() * 0.0
            return zero, {"lossRelationConsistency": zero, "relationPositiveCount": 0.0}
        if "instance_world_aabbs" in kwargs:
            raise ValueError("instance_world_aabbs is owned by the model")
        kwargs["instance_world_aabbs"] = self.instance_world_aabbs
        return self.offline_survival_encoder.relation_consistency_loss(*args, **kwargs)

    def _bounded_frequency_cycles(self) -> torch.Tensor:
        raw = self.moment_query.frequency_cycles.float()
        norm = torch.linalg.norm(raw, dim=-1, keepdim=True).clamp_min(1e-8)
        return raw * torch.clamp(self.max_frequency_norm_cycles / norm, max=1.0)

    @staticmethod
    def _low_rank_summary(disk_axes: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            torch.linalg.vector_norm(disk_axes, dim=1),
            torch.linalg.vector_norm(disk_axes, dim=(1, 2)).unsqueeze(1),
            disk_axes.abs().amax(dim=(1, 2)).unsqueeze(1),
        ], dim=-1)

    def _query_basis_and_semantic(
        self, center: torch.Tensor, axes: torch.Tensor, depth: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        effective_axes = axes if self.spectral_mode == "moment_envelope" else torch.zeros_like(axes)
        spectral = self.moment_query(
            center, effective_axes, frequency_cycles=self._bounded_frequency_cycles()
        )
        low_rank = self._low_rank_summary(effective_axes)
        if self.occlusion_representation == "none":
            condition = coefficients.new_empty(coefficients.shape[0], 0)
        else:
            assert self.relation_condition_head is not None
            condition = self.relation_condition_head(coefficients.flatten(1))
        boundary = self.boundary_summary_head(
            torch.cat([spectral.spectral_features, center, low_rank, condition], dim=-1)
        )
        if self.occlusion_representation == "none":
            basis = center.new_zeros(center.shape[0], self.survival_rank)
            semantic = center.new_zeros(center.shape[0], SURVIVAL_SEMANTIC_DIM)
        else:
            assert self.direction_basis_head is not None
            basis = self.direction_basis_head(torch.cat([boundary, condition, center[:, :3]], dim=-1))
            semantic = (
                _query_survival_semantic(basis, depth, coefficients)
                if self.occlusion_representation == "survival"
                else _query_generic_semantic(basis, depth, coefficients)
            )
        return basis, semantic, {
            "boundary_spectral_summary": boundary,
            "low_rank_summary": low_rank,
            "spectral_features": spectral.spectral_features,
            "spectral_radial_argument": spectral.s,
            "relation_condition": condition,
            "effective_disk_axes": effective_axes,
        }

    @staticmethod
    def _reject_graph_tables(tables: Mapping[str, Any]) -> None:
        forbidden = ("relation", "csr", "neighbor", "subpose", "group", "graph", "observation")
        invalid = [key for key in tables if any(token in str(key).lower() for token in forbidden)]
        if invalid:
            raise ValueError(f"runtime tables contain offline graph data: {invalid}")

    def _gather_runtime_tables(
        self, ids: torch.Tensor, tables: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "runtime_features" in tables:
            runtime = _float_tensor(tables["runtime_features"], "runtime_features")
            if runtime.ndim != 2 or runtime.shape[1] != self.runtime_feature_dim:
                raise ValueError(f"runtime_features must have shape [N, {self.runtime_feature_dim}]")
            rows = runtime[ids] if runtime.shape[0] == self.num_instances else runtime
            if rows.shape[0] != ids.numel():
                raise ValueError("runtime_features must be global or candidate-aligned")
            if self.occlusion_representation == "none":
                return rows[:, :GEO_DIM], rows.new_empty(rows.shape[0], 0)
            return rows[:, :GEO_DIM], rows[:, GEO_DIM:].reshape(
                -1, self.survival_rank, SURVIVAL_PARAMETER_DIM
            )
        if "geometry" not in tables:
            raise ValueError("runtime tables require geometry")
        geometry_table = _float_tensor(tables["geometry"], "geometry")
        local_ids = ids.to(geometry_table.device)
        geometry = geometry_table[local_ids] if geometry_table.shape[0] == self.num_instances else geometry_table
        if self.occlusion_representation == "none":
            if geometry.shape != (ids.numel(), GEO_DIM):
                raise ValueError("runtime geometry table has an invalid shape")
            return geometry, geometry.new_empty(ids.numel(), 0)
        key = "survival_coefficients" if self.occlusion_representation == "survival" else "occlusion_features"
        if key not in tables:
            raise ValueError(f"runtime tables require {key}")
        table = _float_tensor(tables[key], key, device=geometry_table.device)
        coefficients = table[local_ids] if table.shape[0] == self.num_instances else table
        if geometry.shape != (ids.numel(), GEO_DIM) or coefficients.shape != (
            ids.numel(), self.survival_rank, SURVIVAL_PARAMETER_DIM
        ):
            raise ValueError("runtime geometry or occlusion table has an invalid shape")
        return geometry, coefficients

    def forward_batch(
        self, instance_ids: Any, center_view: Any, disk_axes: Any,
        normalized_depth: Any, runtime_tables: Mapping[str, Any],
        *, pose_offsets: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del pose_offsets
        if not isinstance(runtime_tables, Mapping):
            raise TypeError("runtime_tables must be a mapping")
        self._reject_graph_tables(runtime_tables)
        ids = torch.as_tensor(instance_ids, dtype=torch.long).reshape(-1)
        if ids.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= self.num_instances:
            raise ValueError("instance_ids must be a non-empty in-range vector")
        anchor = runtime_tables.get("runtime_features", runtime_tables.get("geometry"))
        if anchor is None:
            raise ValueError("runtime tables have no fixed feature table")
        device = torch.as_tensor(anchor).device
        ids = ids.to(device)
        geometry, coefficients = self._gather_runtime_tables(ids, runtime_tables)
        batch = ids.numel()
        center = _broadcast_rows(center_view, batch, VIEW_DIM, "center_view", device=device)
        axes = _float_tensor(disk_axes, "disk_axes", device=device)
        if axes.ndim == 2 and axes.shape == (VIEW_DIM, DISK_AXIS_DIM):
            axes = axes.unsqueeze(0).expand(batch, -1, -1)
        if axes.shape != (batch, VIEW_DIM, DISK_AXIS_DIM):
            raise ValueError(f"disk_axes must have shape [{batch}, {VIEW_DIM}, {DISK_AXIS_DIM}]")
        depth = _broadcast_scalar(normalized_depth, batch, "normalized_depth", device=device).clamp(0, 1)
        basis, semantic, query = self._query_basis_and_semantic(center, axes, depth, coefficients)
        dynamic = torch.cat([
            *([basis, semantic] if self.occlusion_representation != "none" else []),
            query["boundary_spectral_summary"], center, query["low_rank_summary"], depth,
        ], dim=-1)
        trunk_input = torch.cat([geometry, dynamic], dim=-1)
        if trunk_input.shape[1] != self.runtime_head_input_dim:
            raise RuntimeError("runtime input layout drifted from its schema")
        hidden = self.shared_trunk(trunk_input)
        logits = self.visibility_head(hidden)
        task_input = torch.cat([hidden, torch.sigmoid(logits)], dim=-1)
        return logits, {
            "visibility_logits": logits,
            "base_visibility_logits": logits,
            "utility_logits": self.utility_head(task_input),
            "download_logits": self.download_head(task_input),
            "instance_ids": ids,
            "geometry": geometry,
            "survival_coefficients": coefficients,
            "survival_direction_basis": basis,
            "survival_semantic": semantic,
            "survival_occlusion_probability": semantic[:, 1:2],
            "normalized_depth": depth,
            "center_view": center,
            "disk_axes": axes,
            "query_features": hidden,
            **query,
        }

    def compute_logits_with_aux(
        self, camera_pos_norm: Any, camera_view: Any, candidate_camera_world: Any,
        instance_ids: Any, runtime_features: Any | None = None, *,
        query_center_world: Any | None = None, viewcell_radius_m: Any | None = None,
        pose_offsets: Any | None = None, **_unused: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del camera_pos_norm
        if runtime_features is None:
            raise ValueError(f"V4 requires the fixed {self.runtime_feature_dim}D runtime feature table")
        if query_center_world is None or viewcell_radius_m is None:
            raise ValueError("V4 requires query_center_world and viewcell_radius_m")
        runtime = _float_tensor(runtime_features, "runtime_features")
        ids = torch.as_tensor(instance_ids, dtype=torch.long, device=runtime.device).reshape(-1)
        view = _broadcast_rows(camera_view, ids.numel(), 5, "camera_view", device=runtime.device)
        center = _broadcast_rows(
            query_center_world, ids.numel(), 3, "query_center_world", device=runtime.device
        )
        radius = _broadcast_scalar(
            viewcell_radius_m, ids.numel(), "viewcell_radius_m", device=runtime.device
        )
        ray = build_horizontal_disk_ray_query(
            center, view, self.instance_world_aabbs[ids].to(runtime.device), radius
        )
        depth = normalized_relative_log_depth(
            ray.distance_m, ray.instance_radius_m, self.depth_q01, self.depth_q99,
            epsilon=self.depth_epsilon,
        )
        candidate_camera = _broadcast_rows(
            candidate_camera_world, ids.numel(), 3, "candidate_camera_world", device=runtime.device
        )
        logits, aux = self.forward_batch(
            ids, ray.center_view, ray.disk_axes, depth,
            {"runtime_features": runtime}, pose_offsets=pose_offsets,
        )
        aux.update({
            "candidate_camera_world": candidate_camera,
            "query_center_world": center,
            "viewcell_radius_m": radius,
            "evidence_target": torch.zeros_like(logits),
            "final_logits": logits,
            "base_logits": logits,
            "utility_value": torch.sigmoid(aux["utility_logits"]),
        })
        return logits, aux

    def compute_visibility_logits(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.compute_logits_with_aux(*args, **kwargs)[0]

    def query_survival_from_direction(
        self, direction: Any, normalized_depth: Any, coefficients: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.occlusion_representation != "survival":
            raise RuntimeError("directional survival query requires survival mode")
        coefficients = _float_tensor(coefficients, "coefficients")
        if coefficients.ndim != 3 or coefficients.shape[1:] != (
            self.survival_rank, SURVIVAL_PARAMETER_DIM
        ):
            raise ValueError("coefficients have the wrong rank or parameter dimension")
        batch = coefficients.shape[0]
        direction = F.normalize(
            _broadcast_rows(direction, batch, 3, "direction", device=coefficients.device),
            dim=-1, eps=1e-6,
        )
        center = torch.cat([direction, direction.new_zeros(batch, VIEW_DIM - 3)], dim=-1)
        axes = direction.new_zeros(batch, VIEW_DIM, DISK_AXIS_DIM)
        depth = _broadcast_scalar(
            normalized_depth, batch, "normalized_depth", device=coefficients.device
        ).clamp(0, 1)
        basis, semantic, _ = self._query_basis_and_semantic(center, axes, depth, coefficients)
        return semantic, basis

    def regularization(self) -> torch.Tensor:
        terms = [
            parameter.square().mean()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and parameter.ndim > 1
            and name not in {"instance_calibration_residual_raw", "generic_occlusion_features"}
        ]
        frequency_norm = torch.linalg.norm(self.moment_query.frequency_cycles.float(), dim=-1)
        overflow = F.relu(frequency_norm - self.max_frequency_norm_cycles).square().mean()
        return (torch.stack(terms).mean() if terms else overflow * 0.0) + overflow


__all__ = [
    "BOUNDARY_SUMMARY_DIM", "BoundedRelationSurvivalMomentModel", "GEO_DIM",
    "LOW_RANK_SUMMARY_DIM", "MODEL_SCHEMA", "OCCLUSION_REPRESENTATION_MODES",
    "RUNTIME_FEATURE_DIM", "RUNTIME_HEAD_INPUT_DIM_WITHOUT_OCCLUSION",
    "SUPPORTED_SURVIVAL_RANKS", "SURVIVAL_PARAMETER_DIM", "SURVIVAL_RANK",
]
