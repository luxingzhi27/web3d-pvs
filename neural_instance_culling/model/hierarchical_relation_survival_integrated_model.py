#!/usr/bin/env python3
"""Unified offline-relation and runtime view-cell PVS model.

The model has a deliberately explicit two-phase contract:

* :meth:`offline_encode_survival` consumes fixed geometry and the train-only
  observed-relation CSR.  Hierarchical message passing is used here to write
  one ``[4, 7]`` survival field for every instance.
* :meth:`forward_batch` consumes only the exported geometry table, survival
  coefficient table, one nine-value view-cell center query, its diagonal
  variance, and ``rho``.  It performs one integrated spectral query and one
  monotone survival read per candidate.  No graph, neighbor lookup, or
  subpose expansion is possible on this path.

The three heads share the same queried representation.  ``visibility_logits``
is the instance render-filter score, ``utility_logits`` is an instance visual
utility score, and ``download_logits`` is an instance download-priority score
that can be aggregated by ``instance_to_glb`` by the scheduler.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from common.hierarchical_relation_survival import HierarchicalOcclusionSurvivalEncoder
from common.viewcell_integrated_spectral_query import (
    ViewCellIntegratedSpectralQuery,
    analytic_viewcell_diagonal_variance,
)


MODEL_SCHEMA = "pvs-hierarchical-relation-survival-integrated-v1"
GEO_DIM = 96
SURVIVAL_RANK = 4
SURVIVAL_PARAMETER_DIM = 7
VIEW_DIM = 9
SURVIVAL_SEMANTIC_DIM = 8
RELATION_CONDITION_DIM = 8


def _trainable_parameter_count(module: nn.Module) -> int:
    """Count trainable parameters in a module, including no dead padding."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _geometry_only_parameter_count(
    input_dim: int,
    first_hidden_dim: int,
    second_hidden_dim: int,
    output_dim: int,
) -> int:
    """Parameter count for the two-hidden-layer geometry-only MLP."""

    return (
        int(input_dim) * int(first_hidden_dim)
        + int(first_hidden_dim)
        + int(first_hidden_dim) * int(second_hidden_dim)
        + int(second_hidden_dim)
        + int(second_hidden_dim) * int(output_dim)
        + int(output_dim)
    )


def _capacity_matched_geometry_widths(target_parameter_count: int) -> tuple[int, int, int]:
    """Choose useful hidden widths within one percent of a target capacity.

    The search prefers similarly sized hidden layers.  A and B are actual
    trainable widths; no unused parameter tensor is added to make the count
    look equal.  The returned count is included to make the contract easy to
    audit in tests and checkpoint metadata.
    """

    target = int(target_parameter_count)
    if target <= 0:
        raise ValueError("target_parameter_count must be positive")
    # The target is normally the 64-wide relation encoder (158,429 params).
    # This bound also covers substantially larger registered hidden sizes.
    limit = max(64, int(target**0.5 * 4) + 8)
    candidates: list[tuple[float, int, int, int, int]] = []
    for first in range(1, limit + 1):
        # For fixed first width, the parameter count is affine in second.
        constant = GEO_DIM * first + first + SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM
        coefficient = first + SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM + 1
        estimate = round((target - constant) / coefficient)
        for second in range(max(1, estimate - 2), estimate + 3):
            count = _geometry_only_parameter_count(
                GEO_DIM,
                first,
                second,
                SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM,
            )
            relative_error = abs(count - target) / target
            if relative_error <= 0.01:
                candidates.append((abs(first - second), relative_error, first + second, first, second))
    if not candidates:
        raise RuntimeError(
            f"could not find a geometry-only capacity within 1% of {target} parameters"
        )
    _balance, _error, _width_sum, first, second = min(candidates)
    count = _geometry_only_parameter_count(
        GEO_DIM,
        first,
        second,
        SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM,
    )
    return first, second, count


class _GeometryOnlyAnchorOwner(nn.Module):
    """Non-trainable compatibility owner for the shared survival loss.

    The loss only uses ``direction_embedding.new_tensor(...)`` to place its
    fixed 12 direction anchors on the right device and dtype.  Geometry-only
    must not retain the full relation encoder merely to provide that helper.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("direction_embedding", torch.zeros(1), persistent=False)


def _as_float_tensor(value: Any, *, name: str, device: torch.device | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if device is not None:
        tensor = tensor.to(device=device)
    tensor = tensor.float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _broadcast_view(value: Any, batch: int, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = _as_float_tensor(value, name=name, device=device)
    if tensor.ndim == 1 and tuple(tensor.shape) == (VIEW_DIM,):
        tensor = tensor.unsqueeze(0).expand(batch, -1)
    if tensor.ndim != 2 or tuple(tensor.shape) != (batch, VIEW_DIM):
        raise ValueError(f"{name} must have shape [9] or [{batch}, 9]")
    return tensor


def _broadcast_scalar(value: Any, batch: int, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = _as_float_tensor(value, name=name, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1).expand(batch, 1)
    elif tensor.ndim == 1 and tensor.numel() in (1, batch):
        tensor = tensor.reshape(-1, 1)
        if tensor.shape[0] == 1:
            tensor = tensor.expand(batch, -1)
    elif tensor.ndim == 2 and tuple(tensor.shape) == (batch, 1):
        pass
    else:
        raise ValueError(f"{name} must be scalar, [{batch}], or [{batch}, 1]")
    return tensor


def _fourier_features(values: torch.Tensor, bands: int) -> torch.Tensor:
    parts = [values]
    for band in range(max(0, int(bands))):
        frequency = float(2 ** band) * torch.pi
        parts.extend([torch.sin(values * frequency), torch.cos(values * frequency)])
    return torch.cat(parts, dim=-1)


class HierarchicalRelationSurvivalIntegratedModel(nn.Module):
    """Fixed-geometry, offline-survival, integrated-spectral PVS model."""

    geo_dim = GEO_DIM
    survival_coefficient_shape = (SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM)
    view_dim = VIEW_DIM
    survival_semantic_dim = SURVIVAL_SEMANTIC_DIM

    def __init__(
        self,
        num_instances: int,
        num_glbs: int = 0,
        *,
        relation_hidden_dim: int = 64,
        query_basis_dim: int = 4,
        hidden_dim: int = 64,
        spectral_mode: str = "integrated",
        relation_source: str = "hierarchical",
    ) -> None:
        super().__init__()
        if int(num_instances) <= 0:
            raise ValueError("num_instances must be positive")
        if int(num_glbs) < 0:
            raise ValueError("num_glbs must be non-negative")
        if int(query_basis_dim) != SURVIVAL_RANK:
            raise ValueError("the registered model requires query_basis_dim=4")
        if int(hidden_dim) <= 0 or int(relation_hidden_dim) <= 0:
            raise ValueError("hidden dimensions must be positive")
        if spectral_mode not in ("integrated", "learned_point", "fourier117"):
            raise ValueError("spectral_mode must be integrated, learned_point, or fourier117")
        if relation_source not in ("hierarchical", "single_scale", "geometry_only", "free", "shuffled"):
            raise ValueError("relation_source must be hierarchical, single_scale, geometry_only, free, or shuffled")

        self.num_instances = int(num_instances)
        self.num_glbs = int(num_glbs)
        self.relation_hidden_dim = int(relation_hidden_dim)
        self.query_basis_dim = int(query_basis_dim)
        self.hidden_dim = int(hidden_dim)
        self.spectral_mode = str(spectral_mode)
        self.relation_source = str(relation_source)

        if self.relation_source == "geometry_only":
            # Capacity-matched control: remove the relation graph, retain the
            # same geometry input and [4, 7] output, and match the complete
            # relation encoder's trainable parameter count within one percent.
            relation_encoder = HierarchicalOcclusionSurvivalEncoder(
                geo_dim=GEO_DIM,
                hidden_dim=self.relation_hidden_dim,
                survival_rank=SURVIVAL_RANK,
                survival_parameter_dim=SURVIVAL_PARAMETER_DIM,
            )
            target_parameters = _trainable_parameter_count(relation_encoder)
            del relation_encoder
            first_width, second_width, matched_parameters = _capacity_matched_geometry_widths(
                target_parameters
            )
            self.geometry_only_widths = (first_width, second_width)
            self.geometry_only_target_parameters = target_parameters
            self.geometry_only_parameters = matched_parameters
            self.offline_survival_encoder = _GeometryOnlyAnchorOwner()
            self.geometry_only_survival_encoder = nn.Sequential(
                nn.Linear(GEO_DIM, first_width),
                nn.SiLU(),
                nn.Linear(first_width, second_width),
                nn.SiLU(),
                nn.Linear(second_width, SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM),
            )
        else:
            self.offline_survival_encoder = HierarchicalOcclusionSurvivalEncoder(
                geo_dim=GEO_DIM,
                hidden_dim=self.relation_hidden_dim,
                survival_rank=SURVIVAL_RANK,
                survival_parameter_dim=SURVIVAL_PARAMETER_DIM,
            )
        if self.relation_source == "free":
            # This control deliberately has no relation input.  It is an
            # independent per-instance field used to measure the contribution
            # of the observed relation graph, not a relation encoder with an
            # empty edge list hidden behind the same name.
            self.free_survival_coefficients = nn.Parameter(
                torch.randn(
                    self.num_instances,
                    SURVIVAL_RANK,
                    SURVIVAL_PARAMETER_DIM,
                    dtype=torch.float32,
                ) * 0.02
            )
        self.integrated_query = ViewCellIntegratedSpectralQuery(
            query_basis_dim=self.query_basis_dim,
            relation_condition_dim=RELATION_CONDITION_DIM,
        )
        self.fourier_query = nn.Sequential(
            nn.Linear(117 + RELATION_CONDITION_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.query_basis_dim),
        )
        self.relation_condition_head = nn.Sequential(
            nn.Linear(SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, RELATION_CONDITION_DIM),
        )

        # Geometry, integrated query basis, semantic survival values, center
        # view, variance summary, and the scalar radial query form the common
        # representation used by all three task heads.
        self.runtime_input_dim = (
            GEO_DIM
            + self.query_basis_dim
            + SURVIVAL_SEMANTIC_DIM
            + VIEW_DIM
            + 2
            + 1
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(self.runtime_input_dim, self.hidden_dim),
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
        # Runtime-only geometry needed to construct the candidate ray query.
        # The relation CSR and hierarchy are intentionally absent from these
        # buffers and are never consulted by the online path.
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
        """Return the checkpoint configuration without runtime objects."""

        return {
            "runtimeSchema": MODEL_SCHEMA,
            "numInstances": self.num_instances,
            "numGlbs": self.num_glbs,
            "geoDim": GEO_DIM,
            "survivalCoefficientShape": [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
            "survivalCoefficientDim": SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM,
            "runtimeFeatureDim": GEO_DIM + SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM,
            "viewCellCenterDim": VIEW_DIM,
            "viewCellVarianceDim": VIEW_DIM,
            "survivalSemanticDim": SURVIVAL_SEMANTIC_DIM,
            "relationConditionDim": RELATION_CONDITION_DIM,
            "queryBasisDim": self.query_basis_dim,
            "runtimeInputDim": self.runtime_input_dim,
            "hiddenDim": self.hidden_dim,
            "relationHiddenDim": self.relation_hidden_dim,
            "spectralMode": self.spectral_mode,
            "relationSource": self.relation_source,
            "relationEncoderParameters": (
                _trainable_parameter_count(self.offline_survival_encoder)
                if self.relation_source != "geometry_only"
                else None
            ),
            "geometryOnlyParameters": (
                self.geometry_only_parameters
                if self.relation_source == "geometry_only"
                else None
            ),
            "geometryOnlyTargetParameters": (
                self.geometry_only_target_parameters
                if self.relation_source == "geometry_only"
                else None
            ),
            "geometryOnlyWidths": (
                list(self.geometry_only_widths)
                if self.relation_source == "geometry_only"
                else None
            ),
            "heads": {
                "visibility": [self.hidden_dim, 1],
                "utility": [self.hidden_dim + 1, 32, 1],
                "download": [self.hidden_dim + 1, 32, 1],
            },
            "offlineOnly": [
                "geometry-to-survival hierarchical relation encoding",
                "observed relation CSR traversal",
                "local and structural hierarchy aggregation",
            ],
            "runtimeOnlineOperations": [
                "fixed geometry table gather",
                "fixed survival coefficient table gather",
                "one integrated spectral query per candidate batch",
                "one semantic survival field query per candidate",
                "shared lightweight trunk and three heads",
            ],
            "runtimeForbidden": [
                "relation CSR",
                "graph propagation",
                "neighbor search",
                "subpose expansion",
            ],
        }

    def export_schema(self) -> dict[str, Any]:
        """Return the JSON-compatible deployable table and query contract."""

        return {
            "schema": MODEL_SCHEMA,
            "modelConfig": self.config,
            "offline": {
                "inputs": {
                    "geometry": {"shape": ["N", GEO_DIM], "dtype": "float16/float32"},
                    "relationCsr": "train-only observed directed relation CSR",
                    "localGroupIds": {"shape": ["N"], "dtype": "int64"},
                    "structuralGroupIds": {
                        "shape": ["localGroupCount"],
                        "dtype": "int64",
                        "meaning": "local group to structural group mapping",
                    },
                },
                "output": {
                    "survivalCoefficients": {
                        "shape": ["N", SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
                        "dtype": "float16/float32",
                    }
                },
            },
            "runtime": {
                "tables": {
                    "geometry": {"shape": ["N", GEO_DIM], "required": True},
                    "survivalCoefficients": {
                        "shape": ["N", SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
                        "required": True,
                    },
                    "instanceToGlb": {"shape": ["N"], "required": False},
                },
                "query": {
                    "instanceIds": {"shape": ["B"], "dtype": "int64"},
                    "centerView": {"shape": [VIEW_DIM], "broadcastable": True},
                    "diagonalVariance": {"shape": [VIEW_DIM], "broadcastable": True},
                    "rho": {"shape": [1], "broadcastable": True, "range": [0.0, 1.0]},
                },
                "outputs": {
                    "visibilityLogits": {"shape": ["B", 1]},
                    "utilityLogits": {"shape": ["B", 1]},
                    "downloadLogits": {"shape": ["B", 1]},
                    "survivalSemantic": {"shape": ["B", SURVIVAL_SEMANTIC_DIM]},
                },
                "execution": {
                    "oneCenterQueryPerCandidateBatch": True,
                    "fixedTableGatherOnly": True,
                    "offlineRelationEncoding": True,
                },
            },
        }

    def offline_encode_survival(
        self,
        geo: torch.Tensor,
        relation: Any,
        local_ids: torch.Tensor,
        structural_ids: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Encode train-only relations into fixed per-instance survival fields.

        ``local_ids`` maps each instance to a contiguous local group.  The
        ``structural_ids`` tensor maps each local group to a contiguous
        structural group, matching the common encoder's hierarchy contract.
        This method is never called by :meth:`forward_batch`.
        """

        geometry = _as_float_tensor(geo, name="geo")
        if geometry.ndim != 2 or tuple(geometry.shape[1:]) != (GEO_DIM,):
            raise ValueError(f"geo must have shape [N, {GEO_DIM}]")
        if int(geometry.shape[0]) != self.num_instances:
            raise ValueError("geo row count does not match num_instances")
        if self.relation_source == "free":
            coefficients = self.free_survival_coefficients
            if return_diagnostics:
                return {
                    "survival_coefficients": coefficients,
                    "instance_features": geometry.new_zeros((self.num_instances, self.relation_hidden_dim)),
                    "hierarchy": "free_per_instance",
                }
            return coefficients
        if self.relation_source == "geometry_only":
            coefficients = self.geometry_only_survival_encoder(geometry).reshape(
                self.num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM
            )
            if return_diagnostics:
                return {
                    "survival_coefficients": coefficients,
                    "instance_features": geometry.new_zeros((self.num_instances, self.relation_hidden_dim)),
                    "hierarchy": "geometry_only",
                }
            return coefficients
        result = self.offline_survival_encoder(
            geometry,
            relation,
            torch.as_tensor(local_ids, dtype=torch.long, device=geometry.device),
            torch.as_tensor(structural_ids, dtype=torch.long, device=geometry.device),
            return_diagnostics=return_diagnostics,
        )
        return result

    def _query_basis_and_semantic(
        self,
        center: torch.Tensor,
        variance: torch.Tensor,
        rho: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        relation_condition = self.relation_condition_head(coefficients.reshape(coefficients.shape[0], -1))
        if self.spectral_mode in ("integrated", "learned_point"):
            # The learned-point control uses the same joint learnable
            # frequencies and relation-conditioned query network, but removes
            # the view-cell covariance.  This isolates learnable spectral
            # directions from the region-integration contribution.
            query_variance = variance if self.spectral_mode == "integrated" else torch.zeros_like(variance)
            query = self.integrated_query(
                center,
                query_variance,
                relation_condition=relation_condition,
                relation_survival_coefficients=coefficients,
                rho=rho,
            )
            assert query.query_basis is not None and query.relation_survival is not None
            return query.query_basis, query.relation_survival, {
                "integrated_query_basis": query.query_basis,
                "spectral_features": query.spectral_features,
                "high_frequency_gate": query.high_frequency_gate,
                "variance_summary": torch.cat([
                    query_variance.mean(dim=-1, keepdim=True),
                    query_variance.max(dim=-1, keepdim=True).values,
                ], dim=-1),
                "relation_condition": relation_condition,
            }
        spectral = torch.cat([
            _fourier_features(center[:, :3], 10),
            _fourier_features(center[:, 3:], 4),
        ], dim=-1)
        basis = torch.tanh(self.fourier_query(torch.cat([spectral, relation_condition], dim=-1)))
        semantic = self.integrated_query.query_relation_survival(basis, rho, coefficients)
        return basis, semantic, {
            "integrated_query_basis": basis,
            "spectral_features": spectral,
            "high_frequency_gate": torch.ones((center.shape[0], 1), device=center.device),
            "variance_summary": torch.cat([
                variance.mean(dim=-1, keepdim=True),
                variance.max(dim=-1, keepdim=True).values,
            ], dim=-1),
            "relation_condition": relation_condition,
        }

    def query_survival_from_direction(
        self,
        direction: torch.Tensor,
        rho: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the offline field for event/censor supervision."""
        direction = _as_float_tensor(direction, name="direction")
        if direction.ndim == 1:
            direction = direction.unsqueeze(0)
        if direction.ndim != 2 or direction.shape[1] != 3:
            raise ValueError("direction must have shape [B, 3]")
        center = torch.cat([direction, torch.zeros((direction.shape[0], 6), device=direction.device)], dim=-1)
        variance = torch.zeros_like(center)
        basis, semantic, _ = self._query_basis_and_semantic(center, variance, rho, coefficients)
        return semantic, basis

    def set_instance_world_aabbs(self, value: torch.Tensor) -> None:
        aabbs = torch.as_tensor(value, dtype=torch.float32, device=self.instance_world_aabbs.device)
        if tuple(aabbs.shape) != (self.num_instances, 6) or not bool(torch.isfinite(aabbs).all()):
            raise ValueError(f"instance_world_aabbs must have shape [{self.num_instances}, 6]")
        self.instance_world_aabbs = aabbs

    def set_instance_to_glb(self, value: torch.Tensor) -> None:
        mapping = torch.as_tensor(value, dtype=torch.long, device=self.instance_to_glb.device).reshape(-1)
        if mapping.numel() != self.num_instances or bool((mapping < 0).any()):
            raise ValueError("instance_to_glb must contain one non-negative ID per instance")
        self.instance_to_glb = mapping

    def _ray_space_query(
        self,
        camera_view: torch.Tensor,
        camera_world: torch.Tensor,
        instance_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct one nine-value ray-space query for a candidate batch."""
        bounds = self.instance_world_aabbs[instance_ids.long()].to(camera_world.device)
        center = (bounds[:, :3] + bounds[:, 3:]) * 0.5
        size = torch.clamp(bounds[:, 3:] - bounds[:, :3], min=1e-4)
        delta = center - camera_world
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-4)
        ray = delta / distance
        forward = F.normalize(camera_view[:, :3], dim=-1, eps=1e-6)
        up_seed = torch.zeros_like(forward)
        up_seed[:, 1] = 1.0
        alternative = torch.zeros_like(forward)
        alternative[:, 2] = 1.0
        up_seed = torch.where(
            torch.abs((forward * up_seed).sum(dim=-1, keepdim=True)) > 0.98,
            alternative,
            up_seed,
        )
        right = F.normalize(torch.cross(forward, up_seed, dim=-1), dim=-1, eps=1e-6)
        up = F.normalize(torch.cross(right, forward, dim=-1), dim=-1, eps=1e-6)
        dot_forward = (ray * forward).sum(dim=-1, keepdim=True)
        dot_right = (ray * right).sum(dim=-1, keepdim=True)
        dot_up = (ray * up).sum(dim=-1, keepdim=True)
        tan_x = camera_view[:, 3:4].clamp_min(1e-4)
        tan_y = camera_view[:, 4:5].clamp_min(1e-4)
        u = dot_right / (dot_forward.abs() * tan_x).clamp_min(1e-4)
        v = dot_up / (dot_forward.abs() * tan_y).clamp_min(1e-4)
        radius = torch.linalg.norm(size, dim=-1, keepdim=True) * 0.5
        angular = radius / distance.clamp_min(1.0)
        scalars = torch.cat([
            (torch.log1p(distance / 100.0) / 4.0).clamp(0.0, 2.0) - 1.0,
            dot_forward.clamp(-1.0, 1.0),
            u.clamp(-4.0, 4.0) / 4.0,
            v.clamp(-4.0, 4.0) / 4.0,
            (angular / tan_x).clamp(0.0, 4.0) / 2.0 - 1.0,
            (angular / tan_y).clamp(0.0, 4.0) / 2.0 - 1.0,
        ], dim=-1)
        center_view = torch.cat([ray, scalars], dim=-1)
        variance = analytic_viewcell_diagonal_variance(center_view, distance)
        rho = (distance / (distance + radius).clamp_min(1e-4)).clamp(0.0, 1.0)
        return center_view, variance, rho

    def compute_logits_with_aux(
        self,
        camera_pos_norm: torch.Tensor,
        camera_view: torch.Tensor,
        camera_pos_world: torch.Tensor,
        instance_ids: torch.Tensor,
        runtime_features: torch.Tensor | None = None,
        **_unused: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Adapter for the common pose-set evaluator and training loop."""
        del camera_pos_norm
        ids = torch.as_tensor(instance_ids, dtype=torch.long, device=camera_pos_world.device).reshape(-1)
        if runtime_features is None:
            raise ValueError("the integrated model requires a fixed runtime feature table")
        if runtime_features.ndim != 2 or runtime_features.shape[1] != GEO_DIM + SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM:
            raise ValueError("runtime_features must be [N, 124] geometry plus survival coefficients")
        center_view, variance, rho = self._ray_space_query(camera_view, camera_pos_world, ids)
        logits, aux = self.forward_batch(
            ids,
            center_view,
            variance,
            rho,
            {
                "geometry": runtime_features[:, :GEO_DIM],
                "survival_coefficients": runtime_features[:, GEO_DIM:].reshape(
                    runtime_features.shape[0], SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM
                ),
            },
        )
        aux["evidence_target"] = torch.zeros_like(logits)
        aux["final_logits"] = logits
        aux["base_logits"] = logits
        aux["utility_value"] = torch.sigmoid(aux["utility_logits"])
        return logits, aux

    def compute_visibility_logits(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.compute_logits_with_aux(*args, **kwargs)[0]

    @staticmethod
    def _table_value(runtime_tables: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in runtime_tables:
                return runtime_tables[name]
        joined = ", ".join(names)
        raise ValueError(f"runtime_tables is missing required table: {joined}")

    def _gather_runtime_table(
        self,
        table: Any,
        instance_ids: torch.Tensor,
        *,
        expected_tail: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        tensor = _as_float_tensor(table, name=name, device=instance_ids.device)
        if tuple(tensor.shape[1:]) != expected_tail:
            raise ValueError(f"{name} must have shape [N, {', '.join(map(str, expected_tail))}]")
        if tensor.shape[0] == self.num_instances:
            return tensor[instance_ids]
        if tensor.shape[0] == instance_ids.shape[0]:
            return tensor
        raise ValueError(f"{name} must be global [N, ...] or aligned [B, ...]")

    def _reject_runtime_graph_inputs(self, runtime_tables: Mapping[str, Any]) -> None:
        forbidden_tokens = ("relation", "csr", "neighbor", "subpose", "graph")
        present = [
            str(key)
            for key in runtime_tables
            if any(token in str(key).lower() for token in forbidden_tokens)
        ]
        if present:
            raise ValueError(
                "runtime_tables must contain exported fixed tables only; "
                f"graph-like fields are forbidden: {present}"
            )

    def forward_batch(
        self,
        instance_ids: torch.Tensor,
        center_view: torch.Tensor,
        variance: torch.Tensor,
        rho: torch.Tensor | float,
        runtime_tables: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one view-cell batch using fixed tables and no online graph data."""

        if not isinstance(runtime_tables, Mapping):
            raise TypeError("runtime_tables must be a mapping of exported fixed tables")
        self._reject_runtime_graph_inputs(runtime_tables)
        ids = torch.as_tensor(instance_ids, dtype=torch.long)
        if ids.ndim == 0:
            ids = ids.reshape(1)
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("instance_ids must have shape [B] with B > 0")
        if int(ids.min()) < 0 or int(ids.max()) >= self.num_instances:
            raise ValueError("instance_ids contains an ID outside the model table")

        geometry_table = self._table_value(runtime_tables, "geometry", "geo")
        coefficient_table = self._table_value(
            runtime_tables, "survival_coefficients", "survivalCoefficients"
        )
        # Use the geometry table as the device anchor so CPU tables and GPU
        # queries are normalized before any indexed gather occurs.
        geometry_anchor = _as_float_tensor(geometry_table, name="geometry")
        ids = ids.to(device=geometry_anchor.device)
        geometry = self._gather_runtime_table(
            geometry_table, ids, expected_tail=(GEO_DIM,), name="geometry"
        )
        coefficients = _as_float_tensor(
            coefficient_table, name="survival_coefficients", device=ids.device
        )
        if coefficients.ndim != 3 or tuple(coefficients.shape[1:]) != self.survival_coefficient_shape:
            raise ValueError("survival_coefficients must have shape [N, 4, 7]")
        if coefficients.shape[0] == self.num_instances:
            coefficients = coefficients[ids]
        elif coefficients.shape[0] != ids.shape[0]:
            raise ValueError("survival_coefficients must be global [N, 4, 7] or aligned [B, 4, 7]")

        batch = int(ids.shape[0])
        center = _broadcast_view(center_view, batch, name="center_view", device=ids.device)
        diagonal_variance = _broadcast_view(
            variance, batch, name="variance", device=ids.device
        )
        if bool((diagonal_variance < 0.0).any()):
            raise ValueError("variance must be non-negative")
        rho_tensor = torch.clamp(
            _broadcast_scalar(rho, batch, name="rho", device=ids.device), 0.0, 1.0
        )

        basis, semantic, query_aux = self._query_basis_and_semantic(
            center, diagonal_variance, rho_tensor, coefficients
        )
        # The query branch owns the effective region uncertainty. In the
        # learned-point control it is intentionally zero, so this comparison
        # cannot leak the original view-cell variance through the shared trunk.
        variance_summary = query_aux["variance_summary"]
        trunk_input = torch.cat(
            [geometry, basis, semantic, center, variance_summary, rho_tensor],
            dim=-1,
        )
        hidden = self.shared_trunk(trunk_input)
        visibility_logits = self.visibility_head(hidden)
        visibility_probability = torch.sigmoid(visibility_logits)
        task_input = torch.cat([hidden, visibility_probability], dim=-1)
        utility_logits = self.utility_head(task_input)
        download_logits = self.download_head(task_input)
        aux = {
            "logits": visibility_logits,
            "visibility_logits": visibility_logits,
            "utility_logits": utility_logits,
            "download_logits": download_logits,
            "instance_ids": ids,
            "geometry": geometry,
            "survival_coefficients": coefficients,
            "integrated_query_basis": basis,
            "survival_semantic": semantic,
            "survival_occlusion_probability": semantic[:, 1:2],
            "spectral_features": query_aux["spectral_features"],
            "high_frequency_gate": query_aux["high_frequency_gate"],
            "variance_summary": variance_summary,
            "query_features": hidden,
            "relation_condition": query_aux["relation_condition"],
        }
        return visibility_logits, aux

    def regularization(self) -> torch.Tensor:
        terms = [
            parameter.square().mean()
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.ndim > 1
        ]
        return torch.stack(terms).mean() if terms else torch.zeros(())


# A concise alias is useful to checkpoint loaders without changing the formal
# model name or introducing a second runtime path.
PvsHierarchicalRelationSurvivalIntegratedModel = HierarchicalRelationSurvivalIntegratedModel


__all__ = [
    "GEO_DIM",
    "MODEL_SCHEMA",
    "PvsHierarchicalRelationSurvivalIntegratedModel",
    "HierarchicalRelationSurvivalIntegratedModel",
]
