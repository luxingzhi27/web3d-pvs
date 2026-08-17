#!/usr/bin/env python3
"""PVS model with an offline relation prior and per-instance calibration.

The relation graph is consumed only by :meth:`offline_encode_survival`.  The
shared hierarchy first generates a scene-level relation prior, then a
zero-initialized train-only residual calibrates each instance independently.
Export fuses both terms into the same 28 survival coefficients used by the
runtime.  The browser still gathers one 96-value geometry row and those 28
coefficients; it never receives the residual table, relation graph, hierarchy,
or subpose observations.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from common.bounded_hierarchical_relation_survival import (
    BoundedHierarchicalOcclusionSurvivalEncoder,
)
from common.viewcell_moment_envelope_spectral_query import (
    ViewCellMomentEnvelopeSpectralQuery,
)
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
RUNTIME_HEAD_INPUT_DIM = (
    GEO_DIM
    + SURVIVAL_RANK
    + SURVIVAL_SEMANTIC_DIM
    + BOUNDARY_SUMMARY_DIM
    + VIEW_DIM
    + LOW_RANK_SUMMARY_DIM
    + 1
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
    value: Any,
    batch: int,
    width: int,
    name: str,
    *,
    device: torch.device,
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


def _capacity_matched_widths(target_parameters: int) -> tuple[int, int]:
    """Find a two-hidden-layer geometry MLP close to the relation encoder."""
    target = int(target_parameters)
    best: tuple[float, int, int] | None = None
    for first in range(32, 513):
        constant = GEO_DIM * first + first + SURVIVAL_DIM
        coefficient = first + SURVIVAL_DIM + 1
        second = max(1, round((target - constant) / coefficient))
        count = constant + coefficient * second
        relative = abs(count - target) / max(target, 1)
        candidate = (relative + abs(first - second) * 1e-8, first, second)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    if best[0] > 0.01:
        raise RuntimeError("could not capacity-match the geometry-only survival encoder")
    return best[1], best[2]


def _query_survival_semantic(
    basis: torch.Tensor,
    normalized_depth: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Read a monotone two-component survival distribution."""
    params = torch.einsum("br,brp->bp", basis, coefficients)
    no_block = torch.sigmoid(params[:, 0:1])
    weights = torch.softmax(params[:, 1:3], dim=-1)
    depths = 0.05 + 0.90 * torch.sigmoid(params[:, 3:5])
    scales = 0.03 + F.softplus(params[:, 5:7])
    depths, order = torch.sort(depths, dim=-1)
    weights = torch.gather(weights, dim=-1, index=order)
    scales = torch.gather(scales, dim=-1, index=order)
    cdf_terms = torch.sigmoid((normalized_depth - depths) / scales.clamp_min(1e-3))
    cdf = (1.0 - no_block) * (weights * cdf_terms).sum(dim=-1, keepdim=True)
    survival = (1.0 - cdf).clamp(1e-5, 1.0)
    expected_depth = (weights * depths).sum(dim=-1, keepdim=True)
    variance = (weights * (depths - expected_depth).square()).sum(dim=-1, keepdim=True)
    uncertainty = variance.clamp_min(1e-8).sqrt()
    local_slope = (1.0 - no_block) * (
        weights / scales.clamp_min(1e-3) * cdf_terms * (1.0 - cdf_terms)
    ).sum(dim=-1, keepdim=True)
    result = torch.cat(
        [survival, 1.0 - survival, no_block, expected_depth, uncertainty, weights, local_slope],
        dim=-1,
    )
    if result.shape[1] != SURVIVAL_SEMANTIC_DIM or not bool(torch.isfinite(result).all()):
        raise FloatingPointError("survival-field query produced invalid semantics")
    return result


class BoundedRelationSurvivalMomentModel(nn.Module):
    """Offline-relation PVS model with a one-query horizontal-disk runtime."""

    def __init__(
        self,
        num_instances: int,
        num_glbs: int = 0,
        *,
        relation_hidden_dim: int = 64,
        hidden_dim: int = 64,
        relation_source: str = "bounded_hierarchical",
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
        if int(num_instances) <= 0 or int(num_glbs) < 0:
            raise ValueError("num_instances must be positive and num_glbs non-negative")
        if relation_source not in {"bounded_hierarchical", "geometry_only"}:
            raise ValueError("relation_source must be bounded_hierarchical or geometry_only")
        if spectral_mode not in {"moment_envelope", "point"}:
            raise ValueError("spectral_mode must be moment_envelope or point")
        if instance_calibration_mode not in INSTANCE_CALIBRATION_MODES:
            raise ValueError(
                f"instance_calibration_mode must be one of {INSTANCE_CALIBRATION_MODES}"
            )
        if not float(depth_q99) > float(depth_q01):
            raise ValueError("depth_q99 must be greater than depth_q01")
        if (
            float(depth_epsilon) <= 0.0
            or float(max_frequency_norm_cycles) <= 0.0
            or float(instance_calibration_max_abs) <= 0.0
            or float(sparse_instance_penalty) < 0.0
        ):
            raise ValueError("depth epsilon and frequency norm bound must be positive")

        self.num_instances = int(num_instances)
        self.num_glbs = int(num_glbs)
        self.relation_source = str(relation_source)
        self.spectral_mode = str(spectral_mode)
        self.hidden_dim = int(hidden_dim)
        self.relation_hidden_dim = int(relation_hidden_dim)
        self.depth_q01 = float(depth_q01)
        self.depth_q99 = float(depth_q99)
        self.depth_epsilon = float(depth_epsilon)
        self.max_frequency_norm_cycles = float(max_frequency_norm_cycles)
        self.instance_calibration_mode = str(instance_calibration_mode)
        self.instance_calibration_max_abs = float(instance_calibration_max_abs)
        self.sparse_instance_penalty = float(sparse_instance_penalty)

        relation_encoder = BoundedHierarchicalOcclusionSurvivalEncoder(
            geo_dim=GEO_DIM,
            hidden_dim=self.relation_hidden_dim,
            survival_rank=SURVIVAL_RANK,
            survival_parameter_dim=SURVIVAL_PARAMETER_DIM,
        )
        if self.relation_source == "bounded_hierarchical":
            self.offline_survival_encoder = relation_encoder
            self.geometry_only_survival_encoder = None
            self.geometry_only_widths = None
        else:
            target = _parameter_count(relation_encoder)
            first, second = _capacity_matched_widths(target)
            self.geometry_only_widths = (first, second)
            self.geometry_only_survival_encoder = nn.Sequential(
                nn.Linear(GEO_DIM, first),
                nn.SiLU(),
                nn.Linear(first, second),
                nn.SiLU(),
                nn.Linear(second, SURVIVAL_DIM),
            )
            self.offline_survival_encoder = None

        if self.instance_calibration_mode == "residual":
            self.instance_calibration_residual_raw = nn.Parameter(
                torch.zeros(
                    (self.num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("instance_calibration_residual_raw", None)
        self.register_buffer(
            "instance_calibration_reliability",
            torch.zeros((self.num_instances,), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "instance_calibration_blend",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )

        self.moment_query = ViewCellMomentEnvelopeSpectralQuery(learnable_frequencies=True)
        if self.moment_query.frequency_count != SPECTRAL_FREQUENCY_COUNT:
            raise RuntimeError("the registered v4 model requires sixteen joint frequencies")
        self.relation_condition_head = nn.Sequential(
            nn.Linear(SURVIVAL_DIM, 16),
            nn.SiLU(),
            nn.Linear(16, RELATION_CONDITION_DIM),
        )
        spectral_fuse_dim = SPECTRAL_MOMENT_DIM + VIEW_DIM + LOW_RANK_SUMMARY_DIM + RELATION_CONDITION_DIM
        self.boundary_summary_head = nn.Sequential(
            nn.Linear(spectral_fuse_dim, 48),
            nn.SiLU(),
            nn.Linear(48, BOUNDARY_SUMMARY_DIM),
            nn.Tanh(),
        )
        self.direction_basis_head = nn.Sequential(
            nn.Linear(BOUNDARY_SUMMARY_DIM + RELATION_CONDITION_DIM + 3, 24),
            nn.SiLU(),
            nn.Linear(24, SURVIVAL_RANK),
            nn.Tanh(),
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(RUNTIME_HEAD_INPUT_DIM, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.visibility_head = nn.Linear(self.hidden_dim, 1)
        self.utility_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )
        self.download_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )
        self.register_buffer(
            "instance_world_aabbs",
            torch.zeros((self.num_instances, 6), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "instance_to_glb",
            torch.zeros((self.num_instances,), dtype=torch.long),
            persistent=False,
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "runtimeSchema": MODEL_SCHEMA,
            "numInstances": self.num_instances,
            "numGlbs": self.num_glbs,
            "relationSource": self.relation_source,
            "spectralMode": self.spectral_mode,
            "hiddenDim": self.hidden_dim,
            "relationHiddenDim": self.relation_hidden_dim,
            "instanceCalibration": {
                "mode": self.instance_calibration_mode,
                "shape": [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
                "initialization": "zero",
                "maximumAbsoluteResidual": self.instance_calibration_max_abs,
                "sparseInstancePenalty": self.sparse_instance_penalty,
                "fusion": "survival_prior + blend * bounded_instance_residual",
                "runtimeExport": "fused coefficients only",
            },
            "geometryDim": GEO_DIM,
            "survivalCoefficientShape": [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
            "runtimeFeatureDim": RUNTIME_FEATURE_DIM,
            "runtimeHeadInputDim": RUNTIME_HEAD_INPUT_DIM,
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
                "relation CSR",
                "group IDs",
                "neighbor search",
                "subpose expansion",
                "online graph propagation",
            ],
        }

    def export_schema(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SCHEMA,
            "modelConfig": self.config,
            "fixedTable": {
                "shape": ["N", RUNTIME_FEATURE_DIM],
                "dtype": "float16",
                "layout": [
                    {"name": "geometry", "offset": 0, "dim": GEO_DIM},
                    {"name": "survivalCoefficients", "offset": GEO_DIM, "dim": SURVIVAL_DIM},
                ],
            },
            "query": {
                "candidateCameraSemantics": "66-degree back-camera candidate identity only",
                "queryCenterSemantics": "center of the same-direction view-cell visibility union",
                "raySpace": export_ray_space_schema(),
                "spectral": self.moment_query.export_schema(),
                "spectralMode": self.spectral_mode,
            },
            "outputs": {
                "visibilityLogits": ["B", 1],
                "utilityLogits": ["B", 1],
                "downloadLogits": ["B", 1],
            },
            "offlineOnly": [
                "bounded relation CSR",
                "local and structural hierarchy",
                "event/censor observations",
                "relation consistency supervision",
                "per-instance calibration residual parameters",
                "per-instance calibration reliability",
            ],
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
            value,
            "instance_calibration_reliability",
            device=self.instance_calibration_reliability.device,
        ).reshape(-1)
        if tensor.numel() != self.num_instances or bool((tensor < 0.0).any()) or bool((tensor > 1.0).any()):
            raise ValueError(
                "instance_calibration_reliability must contain one value in [0, 1] per instance"
            )
        self.instance_calibration_reliability.copy_(tensor)

    def set_instance_calibration_blend(self, value: float) -> None:
        blend = float(value)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("instance calibration blend must lie in [0, 1]")
        self.instance_calibration_blend.fill_(
            blend if self.instance_calibration_mode == "residual" else 0.0
        )

    def _bounded_instance_calibration_residual(self) -> torch.Tensor:
        if self.instance_calibration_residual_raw is None:
            return self.instance_calibration_reliability.new_zeros(
                (self.num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM)
            )
        scale = self.instance_calibration_max_abs
        return scale * torch.tanh(self.instance_calibration_residual_raw / scale)

    def instance_calibration_regularization(self) -> torch.Tensor:
        """Keep weakly supervised instance corrections close to the shared prior."""
        residual = self._bounded_instance_calibration_residual()
        if self.instance_calibration_residual_raw is None:
            return residual.sum() * 0.0
        weights = 1.0 + self.sparse_instance_penalty * (
            1.0 - self.instance_calibration_reliability
        )
        weights = weights / weights.mean().clamp_min(1e-6)
        return (weights.view(-1, 1, 1) * residual.square()).mean()

    def instance_calibration_diagnostics(self) -> dict[str, torch.Tensor]:
        residual = self._bounded_instance_calibration_residual()
        applied = self.instance_calibration_blend * residual
        per_instance_norm = residual.flatten(1).norm(dim=-1)
        active = (per_instance_norm > 1e-6).float()
        return {
            "instanceCalibrationBlend": self.instance_calibration_blend,
            "instanceCalibrationReliabilityMean": self.instance_calibration_reliability.mean(),
            "instanceCalibrationReliabilityZeroFraction": (
                self.instance_calibration_reliability <= 0.0
            ).float().mean(),
            "instanceCalibrationResidualRms": residual.square().mean().sqrt(),
            "instanceCalibrationAppliedRms": applied.square().mean().sqrt(),
            "instanceCalibrationResidualMaxAbs": residual.abs().amax(),
            "instanceCalibrationActiveFraction": active.mean(),
        }

    def offline_encode_survival(
        self,
        geometry: Any,
        relation: Any,
        local_group_ids: Any,
        structural_group_ids: Any,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        geo = _float_tensor(geometry, "geometry")
        if geo.shape != (self.num_instances, GEO_DIM):
            raise ValueError(f"geometry must have shape [{self.num_instances}, {GEO_DIM}]")
        if self.relation_source == "geometry_only":
            assert self.geometry_only_survival_encoder is not None
            prior = self.geometry_only_survival_encoder(geo).reshape(
                self.num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM
            )
            diagnostics: dict[str, Any] = {
                "survival_prior_coefficients": prior,
                "geometry_base_coefficients": prior,
                "relation_delta_coefficients": torch.zeros_like(prior),
                "relation_gate": torch.zeros((self.num_instances, 1, 1), device=geo.device),
                "hierarchy": "capacity_matched_geometry_only",
            }
        else:
            assert self.offline_survival_encoder is not None
            relation_result = self.offline_survival_encoder(
                geo,
                relation,
                torch.as_tensor(local_group_ids, dtype=torch.long, device=geo.device),
                torch.as_tensor(structural_group_ids, dtype=torch.long, device=geo.device),
                return_diagnostics=True,
            )
            if not isinstance(relation_result, Mapping):
                raise TypeError("offline relation encoder did not return diagnostics")
            diagnostics = dict(relation_result)
            prior = diagnostics["survival_coefficients"]
            diagnostics["survival_prior_coefficients"] = prior

        residual = self._bounded_instance_calibration_residual()
        applied_residual = self.instance_calibration_blend * residual
        coefficients = prior + applied_residual
        if not bool(torch.isfinite(coefficients).all()):
            raise FloatingPointError("instance-calibrated survival coefficients are non-finite")
        diagnostics.update(
            {
                "survival_coefficients": coefficients,
                "coefficients": coefficients,
                "instance_calibration_residual": residual,
                "instance_calibration_applied_residual": applied_residual,
                "instance_calibration_blend": self.instance_calibration_blend,
                "instance_calibration_reliability": self.instance_calibration_reliability,
            }
        )
        return diagnostics if return_diagnostics else coefficients

    def relation_consistency_loss(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.offline_survival_encoder is None:
            zero = next(self.parameters()).sum() * 0.0
            return zero, {"lossRelationConsistency": zero, "relationPositiveCount": 0.0}
        if "instance_world_aabbs" in kwargs:
            raise ValueError("instance_world_aabbs is owned by the v4 model runtime metadata")
        kwargs["instance_world_aabbs"] = self.instance_world_aabbs
        return self.offline_survival_encoder.relation_consistency_loss(*args, **kwargs)

    def _bounded_frequency_cycles(self) -> torch.Tensor:
        raw = self.moment_query.frequency_cycles.float()
        norm = torch.linalg.norm(raw, dim=-1, keepdim=True).clamp_min(1e-8)
        return raw * torch.clamp(self.max_frequency_norm_cycles / norm, max=1.0)

    @staticmethod
    def _low_rank_summary(disk_axes: torch.Tensor) -> torch.Tensor:
        axis_norms = torch.linalg.vector_norm(disk_axes, dim=1)
        frobenius = torch.linalg.vector_norm(disk_axes, dim=(1, 2), keepdim=False).unsqueeze(1)
        maximum = disk_axes.abs().amax(dim=(1, 2), keepdim=False).unsqueeze(1)
        return torch.cat([axis_norms, frobenius, maximum], dim=-1)

    def _query_basis_and_semantic(
        self,
        center_view: torch.Tensor,
        disk_axes: torch.Tensor,
        normalized_depth: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        effective_axes = disk_axes if self.spectral_mode == "moment_envelope" else torch.zeros_like(disk_axes)
        spectral = self.moment_query(
            center_view,
            effective_axes,
            frequency_cycles=self._bounded_frequency_cycles(),
        )
        low_rank = self._low_rank_summary(effective_axes)
        relation_condition = self.relation_condition_head(coefficients.reshape(coefficients.shape[0], -1))
        boundary = self.boundary_summary_head(
            torch.cat([spectral.spectral_features, center_view, low_rank, relation_condition], dim=-1)
        )
        basis = self.direction_basis_head(
            torch.cat([boundary, relation_condition, center_view[:, :3]], dim=-1)
        )
        semantic = _query_survival_semantic(basis, normalized_depth, coefficients)
        return basis, semantic, {
            "boundary_spectral_summary": boundary,
            "low_rank_summary": low_rank,
            "spectral_features": spectral.spectral_features,
            "spectral_radial_argument": spectral.s,
            "relation_condition": relation_condition,
            "effective_disk_axes": effective_axes,
        }

    @staticmethod
    def _reject_graph_tables(runtime_tables: Mapping[str, Any]) -> None:
        forbidden = ("relation", "csr", "neighbor", "subpose", "group", "graph", "observation")
        invalid = [key for key in runtime_tables if any(token in str(key).lower() for token in forbidden)]
        if invalid:
            raise ValueError(f"runtime tables contain offline graph data: {invalid}")

    def _gather_runtime_tables(
        self,
        instance_ids: torch.Tensor,
        runtime_tables: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "runtime_features" in runtime_tables:
            runtime = _float_tensor(runtime_tables["runtime_features"], "runtime_features")
            if runtime.ndim != 2 or runtime.shape[1] != RUNTIME_FEATURE_DIM:
                raise ValueError(f"runtime_features must have shape [N, {RUNTIME_FEATURE_DIM}]")
            rows = runtime[instance_ids] if runtime.shape[0] == self.num_instances else runtime
            if rows.shape[0] != instance_ids.shape[0]:
                raise ValueError("runtime_features must be global or candidate-aligned")
            return rows[:, :GEO_DIM], rows[:, GEO_DIM:].reshape(-1, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM)
        if "geometry" not in runtime_tables or "survival_coefficients" not in runtime_tables:
            raise ValueError("runtime tables require geometry and survival_coefficients")
        geometry_table = _float_tensor(runtime_tables["geometry"], "geometry")
        coefficient_table = _float_tensor(
            runtime_tables["survival_coefficients"], "survival_coefficients", device=geometry_table.device
        )
        ids = instance_ids.to(geometry_table.device)
        geometry = geometry_table[ids] if geometry_table.shape[0] == self.num_instances else geometry_table
        coefficients = coefficient_table[ids] if coefficient_table.shape[0] == self.num_instances else coefficient_table
        if geometry.shape != (ids.numel(), GEO_DIM) or coefficients.shape != (
            ids.numel(), SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM
        ):
            raise ValueError("runtime geometry or coefficient table has an invalid shape")
        return geometry, coefficients

    def forward_batch(
        self,
        instance_ids: Any,
        center_view: Any,
        disk_axes: Any,
        normalized_depth: Any,
        runtime_tables: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(runtime_tables, Mapping):
            raise TypeError("runtime_tables must be a mapping")
        self._reject_graph_tables(runtime_tables)
        ids = torch.as_tensor(instance_ids, dtype=torch.long).reshape(-1)
        if ids.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= self.num_instances:
            raise ValueError("instance_ids must be a non-empty in-range vector")
        table_anchor = runtime_tables.get("runtime_features", runtime_tables.get("geometry"))
        if table_anchor is None:
            raise ValueError("runtime tables have no fixed feature table")
        device = torch.as_tensor(table_anchor).device
        ids = ids.to(device=device)
        geometry, coefficients = self._gather_runtime_tables(ids, runtime_tables)
        batch = int(ids.numel())
        center = _broadcast_rows(center_view, batch, VIEW_DIM, "center_view", device=device)
        axes = _float_tensor(disk_axes, "disk_axes", device=device)
        if axes.ndim == 2 and axes.shape == (VIEW_DIM, DISK_AXIS_DIM):
            axes = axes.unsqueeze(0).expand(batch, -1, -1)
        if axes.shape != (batch, VIEW_DIM, DISK_AXIS_DIM):
            raise ValueError(f"disk_axes must have shape [{batch}, {VIEW_DIM}, {DISK_AXIS_DIM}]")
        depth = _broadcast_scalar(normalized_depth, batch, "normalized_depth", device=device).clamp(0.0, 1.0)
        basis, semantic, query_aux = self._query_basis_and_semantic(center, axes, depth, coefficients)
        trunk_input = torch.cat(
            [
                geometry,
                basis,
                semantic,
                query_aux["boundary_spectral_summary"],
                center,
                query_aux["low_rank_summary"],
                depth,
            ],
            dim=-1,
        )
        if trunk_input.shape[1] != RUNTIME_HEAD_INPUT_DIM:
            raise RuntimeError("v4 runtime input layout drifted from its schema")
        hidden = self.shared_trunk(trunk_input)
        visibility_logits = self.visibility_head(hidden)
        task_input = torch.cat([hidden, torch.sigmoid(visibility_logits)], dim=-1)
        utility_logits = self.utility_head(task_input)
        download_logits = self.download_head(task_input)
        return visibility_logits, {
            "visibility_logits": visibility_logits,
            "utility_logits": utility_logits,
            "download_logits": download_logits,
            "instance_ids": ids,
            "geometry": geometry,
            "survival_coefficients": coefficients,
            "survival_direction_basis": basis,
            "survival_semantic": semantic,
            "survival_occlusion_probability": semantic[:, 1:2],
            "normalized_depth": depth,
            "query_features": hidden,
            **query_aux,
        }

    def compute_logits_with_aux(
        self,
        camera_pos_norm: Any,
        camera_view: Any,
        candidate_camera_world: Any,
        instance_ids: Any,
        runtime_features: Any | None = None,
        *,
        query_center_world: Any | None = None,
        viewcell_radius_m: Any | None = None,
        **_unused: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del camera_pos_norm
        if runtime_features is None:
            raise ValueError("v4 requires the fixed 124D runtime feature table")
        if query_center_world is None or viewcell_radius_m is None:
            raise ValueError("v4 requires query_center_world and viewcell_radius_m; candidate camera is not a query fallback")
        runtime = _float_tensor(runtime_features, "runtime_features")
        ids = torch.as_tensor(instance_ids, dtype=torch.long, device=runtime.device).reshape(-1)
        view = _broadcast_rows(camera_view, ids.numel(), 5, "camera_view", device=runtime.device)
        query_center = _broadcast_rows(
            query_center_world, ids.numel(), 3, "query_center_world", device=runtime.device
        )
        radius = _broadcast_scalar(
            viewcell_radius_m, ids.numel(), "viewcell_radius_m", device=runtime.device
        )
        aabbs = self.instance_world_aabbs[ids].to(device=runtime.device)
        ray_query = build_horizontal_disk_ray_query(query_center, view, aabbs, radius)
        depth = normalized_relative_log_depth(
            ray_query.distance_m,
            ray_query.instance_radius_m,
            self.depth_q01,
            self.depth_q99,
            epsilon=self.depth_epsilon,
        )
        logits, aux = self.forward_batch(
            ids,
            ray_query.center_view,
            ray_query.disk_axes,
            depth,
            {"runtime_features": runtime},
        )
        candidate_camera = _broadcast_rows(
            candidate_camera_world,
            ids.numel(),
            3,
            "candidate_camera_world",
            device=runtime.device,
        )
        aux.update(
            {
                "candidate_camera_world": candidate_camera,
                "query_center_world": query_center,
                "viewcell_radius_m": radius,
                "evidence_target": torch.zeros_like(logits),
                "final_logits": logits,
                "base_logits": logits,
                "utility_value": torch.sigmoid(aux["utility_logits"]),
            }
        )
        return logits, aux

    def compute_visibility_logits(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.compute_logits_with_aux(*args, **kwargs)[0]

    def query_survival_from_direction(
        self,
        direction: Any,
        normalized_depth: Any,
        coefficients: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficient_tensor = _float_tensor(coefficients, "coefficients")
        if coefficient_tensor.ndim != 3 or coefficient_tensor.shape[1:] != (
            SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM
        ):
            raise ValueError("coefficients must have shape [B, 4, 7]")
        batch = int(coefficient_tensor.shape[0])
        direction_tensor = _broadcast_rows(
            direction, batch, 3, "direction", device=coefficient_tensor.device
        )
        direction_tensor = F.normalize(direction_tensor, dim=-1, eps=1e-6)
        center = torch.cat(
            [direction_tensor, torch.zeros((batch, VIEW_DIM - 3), device=coefficient_tensor.device)],
            dim=-1,
        )
        axes = torch.zeros((batch, VIEW_DIM, DISK_AXIS_DIM), device=coefficient_tensor.device)
        depth = _broadcast_scalar(
            normalized_depth, batch, "normalized_depth", device=coefficient_tensor.device
        ).clamp(0.0, 1.0)
        basis, semantic, _ = self._query_basis_and_semantic(center, axes, depth, coefficient_tensor)
        return semantic, basis

    def regularization(self) -> torch.Tensor:
        terms = [
            parameter.square().mean()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and parameter.ndim > 1
            and name != "instance_calibration_residual_raw"
        ]
        raw_norm = torch.linalg.norm(self.moment_query.frequency_cycles.float(), dim=-1)
        frequency_overflow = F.relu(raw_norm - self.max_frequency_norm_cycles).square().mean()
        return (torch.stack(terms).mean() if terms else frequency_overflow * 0.0) + frequency_overflow


__all__ = [
    "BOUNDARY_SUMMARY_DIM",
    "BoundedRelationSurvivalMomentModel",
    "GEO_DIM",
    "LOW_RANK_SUMMARY_DIM",
    "MODEL_SCHEMA",
    "RUNTIME_FEATURE_DIM",
]
