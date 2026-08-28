#!/usr/bin/env python3
"""PVS model with interchangeable compact occlusion representations.

The relation graph is consumed only by :meth:`offline_encode_survival`.  The
shared hierarchy first generates a scene-level relation prior, then a
zero-initialized train-only residual calibrates each instance independently.
Export fuses both terms into the same 28 survival coefficients used by the
runtime.  Formal controls can replace them with an unstructured trainable 28D
table or remove the table entirely.  The browser never receives the relation
graph, hierarchy, subpose observations, or separate calibration residuals.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from common.relation_encoder import (
    BoundedHierarchicalOcclusionSurvivalEncoder,
)
from common.spectral_query import (
    ViewCellMomentEnvelopeSpectralQuery,
)
from common.viewcell_extreme_envelope import (
    VIEWCELL_EXTREME_ENVELOPE_DIM,
    VIEWCELL_SUPPORT_ENVELOPE_DIM,
    export_viewcell_extreme_envelope_schema,
    export_viewcell_support_envelope_schema,
    viewcell_extreme_envelope,
    viewcell_support_envelope,
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
CULL_CERTIFICATE_INPUT_MODES = (
    "hidden",
    "evidence",
    "hidden_evidence",
    "runtime",
    "hidden_runtime",
)
RUNTIME_HEAD_INPUT_DIM = (
    GEO_DIM
    + SURVIVAL_RANK
    + SURVIVAL_SEMANTIC_DIM
    + BOUNDARY_SUMMARY_DIM
    + VIEW_DIM
    + LOW_RANK_SUMMARY_DIM
    + 1
)
RUNTIME_HEAD_INPUT_DIM_WITHOUT_OCCLUSION = (
    GEO_DIM
    + BOUNDARY_SUMMARY_DIM
    + VIEW_DIM
    + LOW_RANK_SUMMARY_DIM
    + 1
)
VIEW_RESIDUAL_DYNAMIC_INPUT_DIM = (
    SURVIVAL_RANK
    + SURVIVAL_SEMANTIC_DIM
    + BOUNDARY_SUMMARY_DIM
    + VIEW_DIM
    + LOW_RANK_SUMMARY_DIM
    + 1
    + 1
)
BOUNDARY_OPPORTUNITY_REGION_DIM = VIEWCELL_EXTREME_ENVELOPE_DIM + VIEW_DIM + 1
BOUNDARY_TAIL_RESIDUAL_REGION_DIM = (
    VIEWCELL_EXTREME_ENVELOPE_DIM + VIEW_DIM + 1
)
VIEWCELL_EXTREME_VISIBILITY_INPUT_DIM = BOUNDARY_TAIL_RESIDUAL_REGION_DIM
VIEWCELL_EXTREME_VISIBILITY_PROJECTION_DIM = 16
VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM = (
    VIEWCELL_EXTREME_ENVELOPE_DIM + VIEW_DIM + 1
)
VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM = 16
VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM = 48
VIEWCELL_REGION_CONDITIONED_VISIBILITY_HEAD_DIM = 16
DUAL_PROBE_CENTER_DIM = VIEW_DIM
DUAL_PROBE_SPECTRAL_DIM = SPECTRAL_MOMENT_DIM
DUAL_PROBE_BOUNDARY_DIM = BOUNDARY_SUMMARY_DIM
DUAL_PROBE_REGION_EXTREMA_DIM = VIEW_DIM * 3
DUAL_PROBE_RAW_QUERY_DIM = (
    DUAL_PROBE_CENTER_DIM
    + DUAL_PROBE_SPECTRAL_DIM
    + DUAL_PROBE_BOUNDARY_DIM
    + DUAL_PROBE_REGION_EXTREMA_DIM
)
DUAL_PROBE_COEFFICIENT_DIM = DUAL_PROBE_RAW_QUERY_DIM + 1
DUAL_PROBE_ATTENUATION_HIDDEN_DIM = 16
QUERY_TAIL_SEPARATOR_FAMILIES = ("disabled", "linear", "hinge", "mlp")
QUERY_TAIL_SEPARATOR_CENTERING_MODES = ("none", "pose_mean")
QUERY_TAIL_SEPARATOR_HINGE_KNOTS = (-1.0, 0.0, 1.0)


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


def _probe_vector(value: Any, width: int, name: str) -> torch.Tensor:
    tensor = _float_tensor(value, name).reshape(-1)
    if tensor.numel() != int(width):
        raise ValueError(f"{name} must contain exactly {width} values")
    return tensor.detach().clone()


def _probe_scalar(value: Any, name: str, *, minimum: float | None = None) -> float:
    tensor = _float_tensor(value, name).reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    result = float(tensor.item())
    if minimum is not None and result < float(minimum):
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _probe_risk_metadata(record: Any, role: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"dual probe {role} specification must be an object")
    metadata = record.get("riskMetadata", record.get("riskCertificate", record))
    if not isinstance(metadata, Mapping):
        raise ValueError(f"dual probe {role} risk metadata must be an object")
    source_split = metadata.get("sourceSplit", metadata.get("fitSplit"))
    fit_rows_only = metadata.get("fitRowsOnly")
    if source_split != "train" or fit_rows_only is not True:
        raise ValueError(
            f"dual probe {role} risk metadata must be train-owned with fitRowsOnly=true"
        )
    raw_risk = metadata.get("risk", record.get("risk"))
    if raw_risk is None:
        raw_risk = metadata.get("registeredRiskUpperBound")
    risk = _probe_scalar(raw_risk, f"dual probe {role} risk", minimum=0.0)
    observed_raw = metadata.get("observedRisk", risk)
    registered_raw = metadata.get("registeredRiskUpperBound", risk)
    observed = _probe_scalar(
        observed_raw, f"dual probe {role} observed risk", minimum=0.0
    )
    registered = _probe_scalar(
        registered_raw,
        f"dual probe {role} registered risk upper bound",
        minimum=0.0,
    )
    if observed > registered:
        raise ValueError(f"dual probe {role} observed risk exceeds its registered bound")
    if "riskWithinRegisteredBound" in metadata and metadata["riskWithinRegisteredBound"] is not True:
        raise ValueError(f"dual probe {role} risk certificate is outside its registered bound")
    return {
        "sourceSplit": "train",
        "fitRowsOnly": True,
        "risk": risk,
        "observedRisk": observed,
        "registeredRiskUpperBound": registered,
        "riskWithinRegisteredBound": True,
    }


def _normalize_dual_probe_rescue_spec(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("dual_probe_rescue must be a mapping or None")
    if value.get("enabled", True) is False:
        return None
    primary = value.get("primary")
    coverage = value.get("coverage")
    if not isinstance(primary, Mapping) or not isinstance(coverage, Mapping):
        raise ValueError("dual_probe_rescue requires primary and coverage probe specifications")
    normalized: dict[str, Any] = {
        "enabled": True,
        "primary": {
            "mean": _probe_vector(
                primary.get("mean"), DUAL_PROBE_RAW_QUERY_DIM, "primary probe mean"
            ),
            "scale": _probe_vector(
                primary.get("scale"), DUAL_PROBE_RAW_QUERY_DIM, "primary probe scale"
            ),
            "coefficients": _probe_vector(
                primary.get("coefficients"), DUAL_PROBE_COEFFICIENT_DIM, "primary probe coefficients"
            ),
            "threshold": _probe_scalar(
                primary.get("threshold"), "primary probe threshold"
            ),
            "riskMetadata": _probe_risk_metadata(primary, "primary"),
        },
        "coverage": {
            "mean": _probe_vector(
                coverage.get("mean"), DUAL_PROBE_RAW_QUERY_DIM, "coverage probe mean"
            ),
            "scale": _probe_vector(
                coverage.get("scale"), DUAL_PROBE_RAW_QUERY_DIM, "coverage probe scale"
            ),
            "coefficients": _probe_vector(
                coverage.get("coefficients"), DUAL_PROBE_COEFFICIENT_DIM, "coverage probe coefficients"
            ),
            "threshold": _probe_scalar(
                coverage.get("threshold"), "coverage probe threshold"
            ),
            "riskMetadata": _probe_risk_metadata(coverage, "coverage"),
        },
        "alpha": _probe_scalar(value.get("alpha", 0.5), "dual probe alpha", minimum=0.0),
        "temperature": _probe_scalar(
            value.get("temperature", 0.05), "dual probe temperature", minimum=1e-12
        ),
        "coverageWeight": _probe_scalar(
            value.get("coverageWeight", 2.0),
            "dual probe coverageWeight",
            minimum=0.0,
        ),
    }
    for role in ("primary", "coverage"):
        if bool((normalized[role]["scale"] <= 0.0).any()):
            raise ValueError(f"{role} probe scale must be strictly positive")
    return normalized


def _parameter_count(module: nn.Module) -> int:
    return sum(value.numel() for value in module.parameters() if value.requires_grad)


def _pose_centered_residual(
    residual: torch.Tensor,
    pose_offsets: Any | None,
) -> torch.Tensor:
    """Remove the per-pose common shift from a candidate-aligned residual."""
    flat = residual.reshape(-1, 1)
    count = int(flat.shape[0])
    if count == 0:
        return flat
    def center_and_bound(values: torch.Tensor) -> torch.Tensor:
        return values - values.mean(dim=0, keepdim=True)

    if pose_offsets is None:
        return center_and_bound(flat)
    offsets = torch.as_tensor(pose_offsets, device=flat.device)
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("pose_offsets must be a one-dimensional boundary vector")
    if torch.is_floating_point(offsets):
        if not bool(torch.isfinite(offsets).all()) or not bool(
            torch.equal(offsets, offsets.round())
        ):
            raise ValueError("pose_offsets must contain finite integers")
        offsets = offsets.round()
    offsets = offsets.to(dtype=torch.long)
    values = [int(value) for value in offsets.detach().cpu().tolist()]
    if values[0] != 0 or values[-1] != count or any(
        end <= start for start, end in zip(values, values[1:])
    ):
        raise ValueError(
            "pose_offsets must start at zero, end at the candidate count, "
            "and contain non-empty poses"
        )
    return torch.cat(
        [
            center_and_bound(flat[start:end])
            for start, end in zip(values, values[1:])
        ],
        dim=0,
    )


BOUNDARY_TAIL_RESIDUAL_CENTERING_MODES = ("pose_mean", "none")
VIEWCELL_REGION_CONDITIONED_VISIBILITY_CENTERING_MODES = ("pose_mean", "none")
BOUNDARY_TAIL_RESIDUAL_SHORTCUT_MODES = ("none", "region_linear")
BOUNDARY_TAIL_RESIDUAL_FUSION_MODES = ("product", "affine_region")


def _capacity_matched_widths(
    target_parameters: int,
    survival_dim: int = SURVIVAL_DIM,
) -> tuple[int, int]:
    """Find a two-hidden-layer geometry MLP close to the relation encoder."""
    target = int(target_parameters)
    best: tuple[float, int, int] | None = None
    for first in range(32, 513):
        constant = GEO_DIM * first + first + int(survival_dim)
        coefficient = first + int(survival_dim) + 1
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


def _query_generic_occlusion_semantic(
    basis: torch.Tensor,
    normalized_depth: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Query an unstructured 28D directional latent without survival semantics."""
    projected = torch.einsum("br,brp->bp", basis, features)
    depth_interaction = projected[:, :1] * (2.0 * normalized_depth - 1.0)
    result = torch.tanh(torch.cat([projected, depth_interaction], dim=-1))
    if result.shape[1] != SURVIVAL_SEMANTIC_DIM or not bool(
        torch.isfinite(result).all()
    ):
        raise FloatingPointError("generic occlusion query produced invalid semantics")
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
        cull_certificate_max_suppression: float = 0.0,
        cull_certificate_initial_suppression: float = 0.05,
        cull_certificate_hidden_dim: int = 0,
        cull_certificate_input_mode: str = "hidden",
        view_residual_max_abs: float = 0.0,
        view_residual_hidden_dim: int = 16,
        boundary_opportunity_hidden_dim: int = 0,
        boundary_opportunity_projection_dim: int = 24,
        boundary_opportunity_initial_logit: float = -6.0,
        boundary_opportunity_max_logit_uplift: float = 6.0,
        boundary_tail_residual_hidden_dim: int = 0,
        boundary_tail_residual_projection_dim: int = 24,
        boundary_tail_residual_max_abs: float = 1.0,
        boundary_tail_residual_centering: str = "pose_mean",
        boundary_tail_residual_shortcut: str = "none",
        boundary_tail_residual_fusion: str = "product",
        boundary_tail_residual_output_init_std: float = 0.0,
        viewcell_extreme_visibility_enabled: bool = False,
        viewcell_region_conditioned_visibility_enabled: bool = False,
        viewcell_region_conditioned_visibility_centering: str = "none",
        query_tail_separator_family: str = "disabled",
        query_tail_separator_hidden_dim: int = 8,
        query_tail_separator_max_abs: float = 0.5,
        query_tail_separator_centering: str = "pose_mean",
        dual_probe_rescue: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        normalized_dual_probe_rescue = _normalize_dual_probe_rescue_spec(
            dual_probe_rescue
        )
        if int(num_instances) <= 0 or int(num_glbs) < 0:
            raise ValueError("num_instances must be positive and num_glbs non-negative")
        if int(survival_rank) not in SUPPORTED_SURVIVAL_RANKS:
            raise ValueError(
                f"survival_rank must be one of {SUPPORTED_SURVIVAL_RANKS}"
            )
        if occlusion_representation not in OCCLUSION_REPRESENTATION_MODES:
            raise ValueError(
                f"occlusion_representation must be one of {OCCLUSION_REPRESENTATION_MODES}"
            )
        if occlusion_representation == "generic28" and int(survival_rank) != 4:
            raise ValueError("generic28 requires the registered four-by-seven layout")
        if occlusion_representation == "survival":
            if relation_source not in {"bounded_hierarchical", "geometry_only"}:
                raise ValueError(
                    "survival representation requires bounded_hierarchical or geometry_only relation_source"
                )
        elif relation_source != "none":
            raise ValueError(
                "generic28 and none representations require relation_source='none'"
            )
        if spectral_mode not in {
            "moment_envelope",
            "moment_extrema",
            "moment_support",
            "moment_extrema_support",
            "point",
        }:
            raise ValueError(
                "spectral_mode must be moment_envelope, moment_extrema, "
                "moment_support, moment_extrema_support, or point"
            )
        if instance_calibration_mode not in INSTANCE_CALIBRATION_MODES:
            raise ValueError(
                f"instance_calibration_mode must be one of {INSTANCE_CALIBRATION_MODES}"
            )
        if (
            occlusion_representation != "survival"
            and instance_calibration_mode != "disabled"
        ):
            raise ValueError(
                "generic28 and none representations require disabled instance calibration"
            )
        if not float(depth_q99) > float(depth_q01):
            raise ValueError("depth_q99 must be greater than depth_q01")
        if (
            float(depth_epsilon) <= 0.0
            or float(max_frequency_norm_cycles) <= 0.0
            or float(instance_calibration_max_abs) <= 0.0
            or float(sparse_instance_penalty) < 0.0
            or float(cull_certificate_max_suppression) < 0.0
            or int(cull_certificate_hidden_dim) < 0
            or str(cull_certificate_input_mode) not in CULL_CERTIFICATE_INPUT_MODES
            or not math.isfinite(float(view_residual_max_abs))
            or float(view_residual_max_abs) < 0.0
            or int(view_residual_hidden_dim) < 0
            or int(boundary_opportunity_hidden_dim) < 0
            or int(boundary_opportunity_projection_dim) <= 0
            or not math.isfinite(float(boundary_opportunity_initial_logit))
            or not math.isfinite(float(boundary_opportunity_max_logit_uplift))
            or float(boundary_opportunity_max_logit_uplift) <= 0.0
            or int(boundary_tail_residual_hidden_dim) < 0
            or int(boundary_tail_residual_projection_dim) <= 0
            or not math.isfinite(float(boundary_tail_residual_max_abs))
            or float(boundary_tail_residual_max_abs) <= 0.0
            or str(boundary_tail_residual_centering)
            not in BOUNDARY_TAIL_RESIDUAL_CENTERING_MODES
            or str(boundary_tail_residual_shortcut)
            not in BOUNDARY_TAIL_RESIDUAL_SHORTCUT_MODES
            or str(boundary_tail_residual_fusion)
            not in BOUNDARY_TAIL_RESIDUAL_FUSION_MODES
            or str(viewcell_region_conditioned_visibility_centering)
            not in VIEWCELL_REGION_CONDITIONED_VISIBILITY_CENTERING_MODES
            or str(query_tail_separator_family)
            not in QUERY_TAIL_SEPARATOR_FAMILIES
            or int(query_tail_separator_hidden_dim) <= 0
            or not math.isfinite(float(query_tail_separator_max_abs))
            or float(query_tail_separator_max_abs) <= 0.0
            or str(query_tail_separator_centering)
            not in QUERY_TAIL_SEPARATOR_CENTERING_MODES
            or not math.isfinite(float(boundary_tail_residual_output_init_std))
            or float(boundary_tail_residual_output_init_std) < 0.0
        ):
            raise ValueError("depth epsilon and frequency norm bound must be positive")
        if float(cull_certificate_max_suppression) > 0.0 and not (
            0.0
            < float(cull_certificate_initial_suppression)
            < float(cull_certificate_max_suppression)
        ):
            raise ValueError(
                "enabled cull certificate requires 0 < initial suppression < maximum suppression"
            )
        if float(view_residual_max_abs) > 0.0 and int(view_residual_hidden_dim) <= 0:
            raise ValueError("enabled view residual requires a positive hidden dimension")
        if int(boundary_opportunity_hidden_dim) > 0 and spectral_mode not in {
            "moment_extrema",
            "moment_extrema_support",
        }:
            raise ValueError(
                "boundary opportunity requires an analytic moment-extrema query"
            )
        if int(boundary_tail_residual_hidden_dim) > 0 and spectral_mode not in {
            "moment_extrema",
            "moment_extrema_support",
        }:
            raise ValueError(
                "boundary tail residual requires an analytic moment-extrema query"
            )
        if bool(viewcell_extreme_visibility_enabled) and spectral_mode not in {
            "moment_extrema",
            "moment_extrema_support",
        }:
            raise ValueError(
                "viewcell extreme visibility requires an analytic moment-extrema query"
            )
        if bool(viewcell_region_conditioned_visibility_enabled) and spectral_mode not in {
            "moment_extrema",
            "moment_extrema_support",
        }:
            raise ValueError(
                "viewcell region-conditioned visibility requires an analytic moment-extrema query"
            )

        self.num_instances = int(num_instances)
        self.num_glbs = int(num_glbs)
        self.relation_source = str(relation_source)
        self.occlusion_representation = str(occlusion_representation)
        self.spectral_mode = str(spectral_mode)
        self.hidden_dim = int(hidden_dim)
        self.relation_hidden_dim = int(relation_hidden_dim)
        self.survival_rank = int(survival_rank)
        self.survival_parameter_dim = SURVIVAL_PARAMETER_DIM
        self.survival_dim = self.survival_rank * self.survival_parameter_dim
        self.depth_q01 = float(depth_q01)
        self.depth_q99 = float(depth_q99)
        self.depth_epsilon = float(depth_epsilon)
        self.max_frequency_norm_cycles = float(max_frequency_norm_cycles)
        self.instance_calibration_mode = str(instance_calibration_mode)
        self.instance_calibration_max_abs = float(instance_calibration_max_abs)
        self.sparse_instance_penalty = float(sparse_instance_penalty)
        self.cull_certificate_max_suppression = float(
            cull_certificate_max_suppression
        )
        self.cull_certificate_initial_suppression = float(
            cull_certificate_initial_suppression
        )
        self.cull_certificate_hidden_dim = int(cull_certificate_hidden_dim)
        self.cull_certificate_input_mode = str(cull_certificate_input_mode)
        self.view_residual_max_abs = float(view_residual_max_abs)
        self.view_residual_hidden_dim = int(view_residual_hidden_dim)
        self.view_residual_projection_dim = self.hidden_dim
        self.boundary_opportunity_hidden_dim = int(
            boundary_opportunity_hidden_dim
        )
        self.boundary_opportunity_projection_dim = int(
            boundary_opportunity_projection_dim
        )
        self.boundary_opportunity_initial_logit = float(
            boundary_opportunity_initial_logit
        )
        self.boundary_opportunity_max_logit_uplift = float(
            boundary_opportunity_max_logit_uplift
        )
        self.boundary_tail_residual_hidden_dim = int(
            boundary_tail_residual_hidden_dim
        )
        self.boundary_tail_residual_projection_dim = int(
            boundary_tail_residual_projection_dim
        )
        self.boundary_tail_residual_max_abs = float(
            boundary_tail_residual_max_abs
        )
        self.boundary_tail_residual_centering = str(
            boundary_tail_residual_centering
        )
        self.boundary_tail_residual_shortcut = str(
            boundary_tail_residual_shortcut
        )
        self.boundary_tail_residual_fusion = str(
            boundary_tail_residual_fusion
        )
        self.boundary_tail_residual_output_init_std = float(
            boundary_tail_residual_output_init_std
        )
        self.viewcell_extreme_visibility_enabled = bool(
            viewcell_extreme_visibility_enabled
        )
        self.viewcell_region_conditioned_visibility_enabled = bool(
            viewcell_region_conditioned_visibility_enabled
        )
        self.viewcell_region_conditioned_visibility_centering = str(
            viewcell_region_conditioned_visibility_centering
        )
        self.query_tail_separator_family = str(query_tail_separator_family)
        self.query_tail_separator_hidden_dim = int(query_tail_separator_hidden_dim)
        self.query_tail_separator_max_abs = float(query_tail_separator_max_abs)
        self.query_tail_separator_centering = str(query_tail_separator_centering)
        self.runtime_feature_dim = GEO_DIM + (
            self.survival_dim if self.occlusion_representation != "none" else 0
        )
        self.runtime_head_input_dim = (
            GEO_DIM
            + self.survival_rank
            + SURVIVAL_SEMANTIC_DIM
            + BOUNDARY_SUMMARY_DIM
            + VIEW_DIM
            + LOW_RANK_SUMMARY_DIM
            + 1
            if self.occlusion_representation != "none"
            else RUNTIME_HEAD_INPUT_DIM_WITHOUT_OCCLUSION
        )
        self.view_residual_dynamic_input_dim = (
            self.survival_rank
            + SURVIVAL_SEMANTIC_DIM
            + BOUNDARY_SUMMARY_DIM
            + VIEW_DIM
            + LOW_RANK_SUMMARY_DIM
            + 1
            + 1
        )
        self.dual_probe_rescue_enabled = normalized_dual_probe_rescue is not None
        self._dual_probe_rescue_spec = normalized_dual_probe_rescue
        evidence_dim = 1 + SURVIVAL_SEMANTIC_DIM + 1
        self.cull_certificate_input_dim = {
            "hidden": self.hidden_dim,
            "evidence": evidence_dim,
            "hidden_evidence": self.hidden_dim + evidence_dim,
            "runtime": self.runtime_head_input_dim,
            "hidden_runtime": self.hidden_dim + self.runtime_head_input_dim,
        }[self.cull_certificate_input_mode]

        if self.occlusion_representation == "survival":
            relation_encoder = BoundedHierarchicalOcclusionSurvivalEncoder(
                geo_dim=GEO_DIM,
                hidden_dim=self.relation_hidden_dim,
                survival_rank=self.survival_rank,
                survival_parameter_dim=self.survival_parameter_dim,
            )
            if self.relation_source == "bounded_hierarchical":
                self.offline_survival_encoder = relation_encoder
                self.geometry_only_survival_encoder = None
                self.geometry_only_widths = None
            else:
                target = _parameter_count(relation_encoder)
                first, second = _capacity_matched_widths(target, self.survival_dim)
                self.geometry_only_widths = (first, second)
                self.geometry_only_survival_encoder = nn.Sequential(
                    nn.Linear(GEO_DIM, first),
                    nn.SiLU(),
                    nn.Linear(first, second),
                    nn.SiLU(),
                    nn.Linear(second, self.survival_dim),
                )
                self.offline_survival_encoder = None
        else:
            self.offline_survival_encoder = None
            self.geometry_only_survival_encoder = None
            self.geometry_only_widths = None

        if self.occlusion_representation == "generic28":
            self.generic_occlusion_features = nn.Parameter(
                torch.empty(
                    (
                        self.num_instances,
                        self.survival_rank,
                        self.survival_parameter_dim,
                    ),
                    dtype=torch.float32,
                )
            )
            nn.init.normal_(self.generic_occlusion_features, mean=0.0, std=0.02)
        else:
            self.register_parameter("generic_occlusion_features", None)

        if (
            self.occlusion_representation == "survival"
            and self.instance_calibration_mode == "residual"
        ):
            self.instance_calibration_residual_raw = nn.Parameter(
                torch.zeros(
                    (
                        self.num_instances,
                        self.survival_rank,
                        self.survival_parameter_dim,
                    ),
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
        self.register_buffer(
            "boundary_opportunity_scale",
            torch.zeros((), dtype=torch.float32),
            persistent=self.boundary_opportunity_hidden_dim > 0,
        )

        self.moment_query = ViewCellMomentEnvelopeSpectralQuery(learnable_frequencies=True)
        if self.moment_query.frequency_count != SPECTRAL_FREQUENCY_COUNT:
            raise RuntimeError("the registered v4 model requires sixteen joint frequencies")
        if self.occlusion_representation != "none":
            self.relation_condition_head: nn.Module | None = nn.Sequential(
                nn.Linear(self.survival_dim, 16),
                nn.SiLU(),
                nn.Linear(16, RELATION_CONDITION_DIM),
            )
        else:
            self.relation_condition_head = None
        spectral_fuse_dim = (
            SPECTRAL_MOMENT_DIM
            + VIEW_DIM
            + LOW_RANK_SUMMARY_DIM
            + (
                RELATION_CONDITION_DIM
                if self.occlusion_representation != "none"
                else 0
            )
            + (
                VIEWCELL_EXTREME_ENVELOPE_DIM
                if self.spectral_mode in {"moment_extrema", "moment_extrema_support"}
                else 0
            )
            + (
                VIEWCELL_SUPPORT_ENVELOPE_DIM
                if self.spectral_mode in {"moment_support", "moment_extrema_support"}
                else 0
            )
        )
        self.boundary_summary_head = nn.Sequential(
            nn.Linear(spectral_fuse_dim, 48),
            nn.SiLU(),
            nn.Linear(48, BOUNDARY_SUMMARY_DIM),
            nn.Tanh(),
        )
        if self.occlusion_representation != "none":
            self.direction_basis_head: nn.Module | None = nn.Sequential(
                nn.Linear(BOUNDARY_SUMMARY_DIM + RELATION_CONDITION_DIM + 3, 24),
                nn.SiLU(),
                nn.Linear(24, self.survival_rank),
                nn.Tanh(),
            )
        else:
            self.direction_basis_head = None
        self.shared_trunk = nn.Sequential(
            nn.Linear(self.runtime_head_input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.visibility_head = nn.Linear(self.hidden_dim, 1)
        if self.view_residual_max_abs > 0.0:
            self.view_residual_geometry_projection: nn.Module | None = nn.Linear(
                GEO_DIM, self.view_residual_projection_dim
            )
            self.view_residual_dynamic_projection: nn.Module | None = nn.Linear(
                self.view_residual_dynamic_input_dim,
                self.view_residual_projection_dim,
            )
            view_residual_output = nn.Linear(self.view_residual_hidden_dim, 1)
            nn.init.zeros_(view_residual_output.weight)
            nn.init.zeros_(view_residual_output.bias)
            self.view_residual_head: nn.Module | None = nn.Sequential(
                nn.Linear(
                    self.view_residual_projection_dim * 2,
                    self.view_residual_hidden_dim,
                ),
                nn.SiLU(),
                view_residual_output,
            )
        else:
            self.view_residual_geometry_projection = None
            self.view_residual_dynamic_projection = None
            self.view_residual_head = None
        if self.cull_certificate_max_suppression > 0.0:
            initial_ratio = (
                self.cull_certificate_initial_suppression
                / self.cull_certificate_max_suppression
            )
            initial_bias = math.log(initial_ratio / (1.0 - initial_ratio))
            if self.cull_certificate_hidden_dim > 0:
                certificate_output = nn.Linear(
                    self.cull_certificate_hidden_dim, 1
                )
                nn.init.zeros_(certificate_output.weight)
                nn.init.constant_(certificate_output.bias, initial_bias)
                self.cull_certificate_head: nn.Module | None = nn.Sequential(
                    nn.Linear(
                        self.cull_certificate_input_dim,
                        self.cull_certificate_hidden_dim,
                    ),
                    nn.SiLU(),
                    certificate_output,
                )
            else:
                certificate_output = nn.Linear(self.cull_certificate_input_dim, 1)
                nn.init.zeros_(certificate_output.weight)
                nn.init.constant_(certificate_output.bias, initial_bias)
                self.cull_certificate_head = certificate_output
        else:
            self.cull_certificate_head = None
        if self.boundary_opportunity_hidden_dim > 0:
            projection_dim = self.boundary_opportunity_projection_dim
            self.boundary_opportunity_geometry_projection: nn.Module | None = nn.Sequential(
                nn.Linear(GEO_DIM, projection_dim),
                nn.SiLU(),
            )
            self.boundary_opportunity_region_projection: nn.Module | None = nn.Sequential(
                nn.Linear(BOUNDARY_OPPORTUNITY_REGION_DIM, projection_dim),
                nn.SiLU(),
            )
            opportunity_output = nn.Linear(
                self.boundary_opportunity_hidden_dim, 1
            )
            nn.init.zeros_(opportunity_output.weight)
            nn.init.constant_(
                opportunity_output.bias,
                self.boundary_opportunity_initial_logit,
            )
            self.boundary_opportunity_head: nn.Module | None = nn.Sequential(
                nn.Linear(
                    2 * projection_dim + SURVIVAL_SEMANTIC_DIM + 1,
                    self.boundary_opportunity_hidden_dim,
                ),
                nn.SiLU(),
                opportunity_output,
            )
        else:
            self.boundary_opportunity_geometry_projection = None
            self.boundary_opportunity_region_projection = None
            self.boundary_opportunity_head = None
        if self.boundary_tail_residual_hidden_dim > 0:
            projection_dim = self.boundary_tail_residual_projection_dim
            self.boundary_tail_residual_geometry_projection: nn.Module | None = (
                nn.Sequential(nn.Linear(GEO_DIM, projection_dim), nn.SiLU())
            )
            self.boundary_tail_residual_region_projection: nn.Module | None = (
                nn.Sequential(
                    nn.Linear(BOUNDARY_TAIL_RESIDUAL_REGION_DIM, projection_dim),
                    nn.SiLU(),
                )
            )
            tail_output = nn.Linear(self.boundary_tail_residual_hidden_dim, 1)
            if self.boundary_tail_residual_output_init_std > 0.0:
                nn.init.normal_(
                    tail_output.weight,
                    mean=0.0,
                    std=self.boundary_tail_residual_output_init_std,
                )
            else:
                nn.init.zeros_(tail_output.weight)
            nn.init.zeros_(tail_output.bias)
            tail_input_dim = (
                2 * projection_dim
                + BOUNDARY_TAIL_RESIDUAL_REGION_DIM
                + SURVIVAL_SEMANTIC_DIM
                + 1
                if self.boundary_tail_residual_fusion == "affine_region"
                else 2 * projection_dim + SURVIVAL_SEMANTIC_DIM + 1
            )
            self.boundary_tail_residual_head: nn.Module | None = nn.Sequential(
                nn.Linear(
                    tail_input_dim,
                    self.boundary_tail_residual_hidden_dim,
                ),
                nn.SiLU(),
                tail_output,
            )
            if self.boundary_tail_residual_shortcut == "region_linear":
                self.boundary_tail_residual_region_shortcut: nn.Module | None = (
                    nn.Linear(BOUNDARY_TAIL_RESIDUAL_REGION_DIM, 1)
                )
                nn.init.zeros_(self.boundary_tail_residual_region_shortcut.weight)
                nn.init.zeros_(self.boundary_tail_residual_region_shortcut.bias)
            else:
                self.boundary_tail_residual_region_shortcut = None
        else:
            self.boundary_tail_residual_geometry_projection = None
            self.boundary_tail_residual_region_projection = None
            self.boundary_tail_residual_head = None
            self.boundary_tail_residual_region_shortcut = None
        if self.viewcell_extreme_visibility_enabled:
            self.viewcell_extreme_visibility_projection: nn.Module | None = (
                nn.Sequential(
                    nn.Linear(
                        VIEWCELL_EXTREME_VISIBILITY_INPUT_DIM,
                        VIEWCELL_EXTREME_VISIBILITY_PROJECTION_DIM,
                    ),
                    nn.SiLU(),
                )
            )
            # Zero the feature projection so a checkpoint without this branch
            # has exactly the old visibility score after non-strict loading.
            nn.init.zeros_(self.viewcell_extreme_visibility_projection[0].weight)
            nn.init.zeros_(self.viewcell_extreme_visibility_projection[0].bias)
            self.viewcell_extreme_visibility_head: nn.Module | None = nn.Linear(
                VIEWCELL_EXTREME_VISIBILITY_PROJECTION_DIM, 1
            )
            nn.init.zeros_(self.viewcell_extreme_visibility_head.bias)
        else:
            self.viewcell_extreme_visibility_projection = None
            self.viewcell_extreme_visibility_head = None
        if self.viewcell_region_conditioned_visibility_enabled:
            self.viewcell_region_conditioned_visibility_region_projection: (
                nn.Module | None
            ) = nn.Sequential(
                nn.Linear(
                    VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM,
                    VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM,
                ),
                nn.SiLU(),
            )
            self.viewcell_region_conditioned_visibility_hidden_projection: (
                nn.Module | None
            ) = nn.Sequential(
                nn.Linear(
                    self.hidden_dim,
                    VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM,
                ),
                nn.SiLU(),
            )
            self.viewcell_region_conditioned_visibility_head: nn.Module | None = (
                nn.Sequential(
                    nn.Linear(
                        VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM,
                        VIEWCELL_REGION_CONDITIONED_VISIBILITY_HEAD_DIM,
                    ),
                    nn.SiLU(),
                    nn.Linear(VIEWCELL_REGION_CONDITIONED_VISIBILITY_HEAD_DIM, 1),
                )
            )
            # The branch is initially an exact no-op, while the two input
            # projections retain normal initialization for later training.
            nn.init.zeros_(self.viewcell_region_conditioned_visibility_head[-1].weight)
            nn.init.zeros_(self.viewcell_region_conditioned_visibility_head[-1].bias)
        else:
            self.viewcell_region_conditioned_visibility_region_projection = None
            self.viewcell_region_conditioned_visibility_hidden_projection = None
            self.viewcell_region_conditioned_visibility_head = None
        # The ablated branch must not shift the random initialization of the
        # shared utility/download heads or the later training RNG stream.
        query_tail_rng_state = torch.random.get_rng_state()
        try:
            if self.query_tail_separator_family == "linear":
                self.query_tail_separator: nn.Module | None = nn.Linear(
                    DUAL_PROBE_RAW_QUERY_DIM, 1
                )
            elif self.query_tail_separator_family == "hinge":
                self.query_tail_separator = nn.Linear(
                    DUAL_PROBE_RAW_QUERY_DIM
                    * (1 + len(QUERY_TAIL_SEPARATOR_HINGE_KNOTS)),
                    1,
                )
            elif self.query_tail_separator_family == "mlp":
                self.query_tail_separator = nn.Sequential(
                    nn.Linear(
                        DUAL_PROBE_RAW_QUERY_DIM,
                        self.query_tail_separator_hidden_dim,
                    ),
                    nn.SiLU(),
                    nn.Linear(self.query_tail_separator_hidden_dim, 1),
                )
            else:
                self.query_tail_separator = None
            if self.query_tail_separator is not None:
                output = (
                    self.query_tail_separator[-1]
                    if isinstance(self.query_tail_separator, nn.Sequential)
                    else self.query_tail_separator
                )
                assert isinstance(output, nn.Linear)
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
        finally:
            torch.random.set_rng_state(query_tail_rng_state)
        if self.dual_probe_rescue_enabled:
            assert normalized_dual_probe_rescue is not None
            primary = normalized_dual_probe_rescue["primary"]
            coverage = normalized_dual_probe_rescue["coverage"]
            for name, value in (
                ("dual_probe_primary_mean", primary["mean"]),
                ("dual_probe_primary_scale", primary["scale"]),
                ("dual_probe_primary_coefficients", primary["coefficients"]),
                ("dual_probe_primary_threshold", primary["threshold"]),
                ("dual_probe_coverage_mean", coverage["mean"]),
                ("dual_probe_coverage_scale", coverage["scale"]),
                ("dual_probe_coverage_coefficients", coverage["coefficients"]),
                ("dual_probe_coverage_threshold", coverage["threshold"]),
                ("dual_probe_alpha", normalized_dual_probe_rescue["alpha"]),
                ("dual_probe_temperature", normalized_dual_probe_rescue["temperature"]),
                ("dual_probe_coverage_weight", normalized_dual_probe_rescue["coverageWeight"]),
                (
                    "dual_probe_primary_risk",
                    primary["riskMetadata"]["risk"],
                ),
                (
                    "dual_probe_primary_observed_risk",
                    primary["riskMetadata"]["observedRisk"],
                ),
                (
                    "dual_probe_primary_registered_risk_upper_bound",
                    primary["riskMetadata"]["registeredRiskUpperBound"],
                ),
                (
                    "dual_probe_coverage_risk",
                    coverage["riskMetadata"]["risk"],
                ),
                (
                    "dual_probe_coverage_observed_risk",
                    coverage["riskMetadata"]["observedRisk"],
                ),
                (
                    "dual_probe_coverage_registered_risk_upper_bound",
                    coverage["riskMetadata"]["registeredRiskUpperBound"],
                ),
            ):
                self.register_buffer(
                    name,
                    torch.as_tensor(value, dtype=torch.float32).detach().clone(),
                    persistent=True,
                )
            self.register_buffer(
                "dual_probe_primary_fit_rows_only",
                torch.ones((), dtype=torch.bool),
                persistent=True,
            )
            self.register_buffer(
                "dual_probe_coverage_fit_rows_only",
                torch.ones((), dtype=torch.bool),
                persistent=True,
            )
            self.dual_probe_primary_source_split = "train"
            self.dual_probe_coverage_source_split = "train"
            self.dual_probe_rescue_attenuation_head: nn.Module | None = nn.Sequential(
                nn.Linear(DUAL_PROBE_COEFFICIENT_DIM, DUAL_PROBE_ATTENUATION_HIDDEN_DIM),
                nn.SiLU(),
                nn.Linear(DUAL_PROBE_ATTENUATION_HIDDEN_DIM, 2),
            )
            nn.init.zeros_(self.dual_probe_rescue_attenuation_head[-1].weight)
            nn.init.zeros_(self.dual_probe_rescue_attenuation_head[-1].bias)
        else:
            self.dual_probe_rescue_attenuation_head = None
            self.dual_probe_primary_source_split = None
            self.dual_probe_coverage_source_split = None
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
        result = {
            "runtimeSchema": MODEL_SCHEMA,
            "numInstances": self.num_instances,
            "numGlbs": self.num_glbs,
            "relationSource": self.relation_source,
            "occlusionRepresentation": {
                "mode": self.occlusion_representation,
                "featureDim": (
                    self.survival_dim
                    if self.occlusion_representation != "none"
                    else 0
                ),
                "directionRank": (
                    self.survival_rank
                    if self.occlusion_representation != "none"
                    else 0
                ),
                "parameterDim": (
                    self.survival_parameter_dim
                    if self.occlusion_representation != "none"
                    else 0
                ),
                "querySemantics": (
                    "monotone_logistic_survival"
                    if self.occlusion_representation == "survival"
                    else "unstructured_direction_conditioned_latent"
                    if self.occlusion_representation == "generic28"
                    else "disabled"
                ),
                "offlineSupervision": (
                    "train_only_relation_and_depth_censoring"
                    if self.occlusion_representation == "survival"
                    else "visibility_labels_only"
                    if self.occlusion_representation == "generic28"
                    else "none"
                ),
            },
            "spectralMode": self.spectral_mode,
            "hiddenDim": self.hidden_dim,
            "relationHiddenDim": self.relation_hidden_dim,
            "instanceCalibration": {
                "mode": self.instance_calibration_mode,
                "shape": (
                    [self.survival_rank, self.survival_parameter_dim]
                    if self.occlusion_representation == "survival"
                    else []
                ),
                "initialization": "zero",
                "maximumAbsoluteResidual": self.instance_calibration_max_abs,
                "sparseInstancePenalty": self.sparse_instance_penalty,
                "fusion": (
                    "survival_prior + blend * bounded_instance_residual"
                    if self.occlusion_representation == "survival"
                    else "not applicable"
                ),
                "runtimeExport": (
                    "fused coefficients only"
                    if self.occlusion_representation == "survival"
                    else "not applicable"
                ),
            },
            "geometryDim": GEO_DIM,
            "survivalCoefficientShape": (
                [self.survival_rank, self.survival_parameter_dim]
                if self.occlusion_representation == "survival"
                else []
            ),
            "runtimeFeatureDim": self.runtime_feature_dim,
            "runtimeHeadInputDim": self.runtime_head_input_dim,
            "queryTailSeparator": {
                "enabled": self.query_tail_separator is not None,
                "family": self.query_tail_separator_family,
                "inputDim": DUAL_PROBE_RAW_QUERY_DIM,
                "hiddenDim": (
                    self.query_tail_separator_hidden_dim
                    if self.query_tail_separator_family == "mlp"
                    else 0
                ),
                "maximumAbsoluteResidual": self.query_tail_separator_max_abs,
                "centering": self.query_tail_separator_centering,
                "hingeKnots": list(QUERY_TAIL_SEPARATOR_HINGE_KNOTS),
                "outputInitialization": "zero",
                "training": "joint from-scratch visibility-tail separation",
                "fusion": "visibility logit plus bounded signed residual before utility and download heads",
                "runtimeAssets": "no additional per-instance asset",
            },
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
        if self.spectral_mode in {"moment_extrema", "moment_extrema_support"}:
            result["viewcellExtremeEnvelope"] = (
                export_viewcell_extreme_envelope_schema()
            )
        if self.spectral_mode in {"moment_support", "moment_extrema_support"}:
            result["viewcellSupportEnvelope"] = (
                export_viewcell_support_envelope_schema()
            )
        if self.cull_certificate_head is not None:
            result["cullCertificate"] = {
                "enabled": True,
                "input": (
                    "detached shared hidden feature"
                    if self.cull_certificate_input_mode == "hidden"
                    else "detached base logit, survival semantic, and normalized depth"
                    if self.cull_certificate_input_mode == "evidence"
                    else "detached runtime head input"
                    if self.cull_certificate_input_mode == "runtime"
                    else "detached shared hidden feature plus runtime head input"
                    if self.cull_certificate_input_mode == "hidden_runtime"
                    else "detached shared hidden feature plus base logit, survival semantic, and normalized depth"
                ),
                "inputMode": self.cull_certificate_input_mode,
                "inputDim": self.cull_certificate_input_dim,
                "hiddenDim": self.cull_certificate_hidden_dim,
                "architecture": (
                    [
                        self.cull_certificate_input_dim,
                        self.cull_certificate_hidden_dim,
                        1,
                    ]
                    if self.cull_certificate_hidden_dim > 0
                    else [self.cull_certificate_input_dim, 1]
                ),
                "maximumSuppressionLogit": self.cull_certificate_max_suppression,
                "initialSuppressionLogit": self.cull_certificate_initial_suppression,
                "fusion": "final visibility logit = base logit - non-negative suppression",
            }
        if self.view_residual_head is not None:
            result["viewResidual"] = {
                "enabled": True,
                "maximumAbsoluteResidual": self.view_residual_max_abs,
                "hiddenDim": self.view_residual_hidden_dim,
                "projectionDim": self.view_residual_projection_dim,
                "dynamicInputDim": self.view_residual_dynamic_input_dim,
                "runtimeInputs": [
                    "geometry",
                    "survival_direction_basis",
                    "survival_semantic",
                    "boundary_spectral_summary",
                    "center_view",
                    "low_rank_summary",
                    "normalized_depth",
                    "base_visibility_logit",
                ],
                "headInputs": [
                    "geometry_projection * dynamic_projection",
                    "dynamic_projection",
                ],
                "usesInstanceIds": False,
                "lastLayerInitialization": "zero",
                "gate": "none; train-only exact-tail selection and residual anchoring prevent global drift",
                "fusion": "base visibility logit - certificate suppression + bounded signed residual",
                "runtimeAssets": "existing geometry row and query inputs; no additional per-instance asset",
            }
        if self.boundary_opportunity_head is not None:
            result["boundaryOpportunity"] = {
                "enabled": True,
                "hiddenDim": self.boundary_opportunity_hidden_dim,
                "projectionDim": self.boundary_opportunity_projection_dim,
                "initialLogit": self.boundary_opportunity_initial_logit,
                "maximumLogitUplift": self.boundary_opportunity_max_logit_uplift,
                "regionInputDim": BOUNDARY_OPPORTUNITY_REGION_DIM,
                "regionInputs": [
                    "analytic view-cell extrema",
                    "center ray-space query",
                    "normalized depth",
                ],
                "instanceInputs": ["existing 96D geometry row"],
                "fusion": "final visibility logit = base logit + bounded non-negative uplift",
                "direction": "visibility logit can only increase",
                "initialEquivalence": "scale zero gives exact base logits",
                "training": "frozen-base weighted positive-tail rescue against the same-pose negative quantile boundary, with all-negative uplift preservation",
                "runtimeAssets": "no additional per-instance asset",
            }
        if self.boundary_tail_residual_head is not None:
            result["boundaryTailResidual"] = {
                "enabled": True,
                "hiddenDim": self.boundary_tail_residual_hidden_dim,
                "projectionDim": self.boundary_tail_residual_projection_dim,
                "maximumAbsoluteResidual": self.boundary_tail_residual_max_abs,
                "centering": self.boundary_tail_residual_centering,
                "shortcut": self.boundary_tail_residual_shortcut,
                "fusionMode": self.boundary_tail_residual_fusion,
                "outputInitializationStd": self.boundary_tail_residual_output_init_std,
                "regionInputDim": BOUNDARY_TAIL_RESIDUAL_REGION_DIM,
                "regionInputs": [
                    "analytic view-cell extrema",
                    "center ray-space query",
                    "normalized depth",
                ],
                "instanceInputs": ["existing 96D geometry row"],
                "fusion": (
                    "final visibility logit = base logit + bounded signed residual"
                ),
                "poseCentering": (
                    "subtract candidate-set mean residual for each pose"
                    if self.boundary_tail_residual_centering == "pose_mean"
                    else "disabled"
                ),
                "training": "frozen-base weighted positive-tail local partial-AUC against same-pose hard negatives",
                "runtimeAssets": "no additional per-instance asset",
                "shortcutSemantics": (
                    "zero-initialized direct linear path from the 27D analytic region input"
                    if self.boundary_tail_residual_shortcut == "region_linear"
                    else "disabled"
                ),
                "fusionInputs": (
                    "projected geometry plus/minus projected region, raw 27D region, survival semantic, detached base logit"
                    if self.boundary_tail_residual_fusion == "affine_region"
                    else "projected geometry times projected region, projected region, survival semantic, detached base logit"
                ),
                "runtimeReduction": (
                    "one scalar mean over the current candidate set"
                    if self.boundary_tail_residual_centering == "pose_mean"
                    else "none"
                ),
            }
        if self.query_tail_separator is not None:
            result["queryTailSeparator"]["architecture"] = (
                [DUAL_PROBE_RAW_QUERY_DIM, 1]
                if self.query_tail_separator_family == "linear"
                else [
                    DUAL_PROBE_RAW_QUERY_DIM
                    * (1 + len(QUERY_TAIL_SEPARATOR_HINGE_KNOTS)),
                    1,
                ]
                if self.query_tail_separator_family == "hinge"
                else [
                    DUAL_PROBE_RAW_QUERY_DIM,
                    self.query_tail_separator_hidden_dim,
                    1,
                ]
            )
        if self.viewcell_extreme_visibility_enabled:
            result["viewcellExtremeVisibility"] = {
                "enabled": True,
                "inputDim": VIEWCELL_EXTREME_VISIBILITY_INPUT_DIM,
                "input": (
                    "raw analytic view-cell extrema: 17D envelope, "
                    "9D center ray-space query, and 1D normalized depth"
                ),
                "queryAuxKey": "viewcell_extreme_features",
                "queryAuxFeatureDim": VIEWCELL_EXTREME_ENVELOPE_DIM,
                "projectionDim": VIEWCELL_EXTREME_VISIBILITY_PROJECTION_DIM,
                "activation": "SiLU",
                "fusion": "base visibility logit + direct view-cell extreme logit",
                "initialization": (
                    "zero projection and zero output bias; output weights remain trainable"
                ),
                "poseReduction": "none",
                "boundedCorrection": False,
                "runtimeAssets": "no additional per-instance asset",
            }
        if self.viewcell_region_conditioned_visibility_enabled:
            result["viewcellRegionConditionedVisibility"] = {
                "enabled": True,
                "regionInputDim": VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM,
                "regionInput": (
                    "query_aux['viewcell_extreme_features'] (17D), "
                    "center ray-space query (9D), and normalized depth (1D)"
                ),
                "queryAuxKey": "viewcell_extreme_features",
                "queryAuxFeatureDim": VIEWCELL_EXTREME_ENVELOPE_DIM,
                "hiddenInputDim": self.hidden_dim,
                "projectionDim": VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM,
                "fusionDim": VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM,
                "headHiddenDim": VIEWCELL_REGION_CONDITIONED_VISIBILITY_HEAD_DIM,
                "activation": "SiLU",
                "fusion": (
                    "concat(region_projection, hidden_projection, "
                    "region_projection * hidden_projection)"
                ),
                "output": "unbounded additive main visibility logit",
                "outputInitialization": "zero weight and bias",
                "centering": self.viewcell_region_conditioned_visibility_centering,
                "poseReduction": (
                    "candidate_mean_per_pose"
                    if self.viewcell_region_conditioned_visibility_centering
                    == "pose_mean"
                    else "none"
                ),
                "runtimeReduction": (
                    "candidate_mean_per_pose"
                    if self.viewcell_region_conditioned_visibility_centering
                    == "pose_mean"
                    else "none"
                ),
                "boundedCorrection": False,
                "runtimeAssets": "no additional per-instance asset",
            }
        if self.dual_probe_rescue_enabled:
            result["dualProbeRescue"] = {
                "enabled": True,
                "rawQueryDim": DUAL_PROBE_RAW_QUERY_DIM,
                "rawQueryLayout": [
                    {"name": "center_view", "offset": 0, "dim": DUAL_PROBE_CENTER_DIM},
                    {
                        "name": "spectral_features",
                        "offset": DUAL_PROBE_CENTER_DIM,
                        "dim": DUAL_PROBE_SPECTRAL_DIM,
                    },
                    {
                        "name": "boundary_spectral_summary",
                        "offset": DUAL_PROBE_CENTER_DIM + DUAL_PROBE_SPECTRAL_DIM,
                        "dim": DUAL_PROBE_BOUNDARY_DIM,
                    },
                    {
                        "name": "region_extrema",
                        "offset": DUAL_PROBE_CENTER_DIM
                        + DUAL_PROBE_SPECTRAL_DIM
                        + DUAL_PROBE_BOUNDARY_DIM,
                        "dim": DUAL_PROBE_REGION_EXTREMA_DIM,
                        "definition": "concat(center_view - extent, center_view + extent, 2 * extent), extent=row_norm(disk_axes)",
                    },
                ],
                "probeCoefficientDim": DUAL_PROBE_COEFFICIENT_DIM,
                "primary": {
                    "fitSplit": "train",
                    "mean": self.dual_probe_primary_mean.detach().cpu().tolist(),
                    "scale": self.dual_probe_primary_scale.detach().cpu().tolist(),
                    "coefficients": self.dual_probe_primary_coefficients.detach().cpu().tolist(),
                    "threshold": float(self.dual_probe_primary_threshold),
                    "riskMetadata": {
                        "sourceSplit": self.dual_probe_primary_source_split,
                        "fitRowsOnly": True,
                        "risk": float(self.dual_probe_primary_risk),
                        "observedRisk": float(self.dual_probe_primary_observed_risk),
                        "registeredRiskUpperBound": float(
                            self.dual_probe_primary_registered_risk_upper_bound
                        ),
                        "riskWithinRegisteredBound": True,
                    },
                    "role": "high_visual_weight_low_score_positive_rescue",
                    "frozenBuffers": [
                        "dual_probe_primary_mean",
                        "dual_probe_primary_scale",
                        "dual_probe_primary_coefficients",
                        "dual_probe_primary_threshold",
                    ],
                },
                "coverage": {
                    "fitSplit": "train",
                    "mean": self.dual_probe_coverage_mean.detach().cpu().tolist(),
                    "scale": self.dual_probe_coverage_scale.detach().cpu().tolist(),
                    "coefficients": self.dual_probe_coverage_coefficients.detach().cpu().tolist(),
                    "threshold": float(self.dual_probe_coverage_threshold),
                    "riskMetadata": {
                        "sourceSplit": self.dual_probe_coverage_source_split,
                        "fitRowsOnly": True,
                        "risk": float(self.dual_probe_coverage_risk),
                        "observedRisk": float(self.dual_probe_coverage_observed_risk),
                        "registeredRiskUpperBound": float(
                            self.dual_probe_coverage_registered_risk_upper_bound
                        ),
                        "riskWithinRegisteredBound": True,
                    },
                    "role": "ordinary_low_score_positive_rescue",
                    "frozenBuffers": [
                        "dual_probe_coverage_mean",
                        "dual_probe_coverage_scale",
                        "dual_probe_coverage_coefficients",
                        "dual_probe_coverage_threshold",
                    ],
                },
                "alpha": float(self.dual_probe_alpha),
                "temperature": float(self.dual_probe_temperature),
                "coverageWeight": float(self.dual_probe_coverage_weight),
                "gate": "max(0, tanh((threshold - standardized_ridge_score) / temperature))",
                "fixedResidual": "alpha * max(primary_gate, coverageWeight * coverage_gate)",
                "attenuation": "clamp(1 - (softplus(d) - softplus(0)), 0, 1)",
                "residual": "alpha * max(primary_attenuation * primary_gate, coverageWeight * coverage_attenuation * coverage_gate)",
                "fusion": "opportunity visibility logit + non-negative attenuated rescue residual",
                "poseCentering": "disabled",
                "attenuationHead": {
                    "inputDim": DUAL_PROBE_COEFFICIENT_DIM,
                    "hiddenDim": DUAL_PROBE_ATTENUATION_HIDDEN_DIM,
                    "outputDim": 2,
                    "inputs": ["detached 108D raw query", "detached base visibility logit"],
                    "lastLayerInitialization": "zero; attenuation initializes exactly to one",
                },
                "runtimeAssets": "no additional per-instance asset; frozen probe buffers are shared",
                "noPerInstanceAssets": True,
            }
        return result

    def export_schema(self) -> dict[str, Any]:
        fixed_layout: list[dict[str, Any]] = [
            {"name": "geometry", "offset": 0, "dim": GEO_DIM}
        ]
        if self.occlusion_representation != "none":
            fixed_layout.append(
                {
                    "name": (
                        "survivalCoefficients"
                        if self.occlusion_representation == "survival"
                        else "genericOcclusionFeatures"
                    ),
                    "offset": GEO_DIM,
                    "dim": self.survival_dim,
                }
            )
        result = {
            "schema": MODEL_SCHEMA,
            "modelConfig": self.config,
            "fixedTable": {
                "shape": ["N", self.runtime_feature_dim],
                "dtype": "float16",
                "layout": fixed_layout,
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
        if self.view_residual_head is not None:
            result["viewResidual"] = {
                "enabled": True,
                "maximumAbsoluteResidual": self.view_residual_max_abs,
                "hiddenDim": self.view_residual_hidden_dim,
                "projectionDim": self.view_residual_projection_dim,
                "dynamicInputDim": self.view_residual_dynamic_input_dim,
                "runtimeInputs": [
                    "geometry",
                    "survival_direction_basis",
                    "survival_semantic",
                    "boundary_spectral_summary",
                    "center_view",
                    "low_rank_summary",
                    "normalized_depth",
                    "base_visibility_logit",
                ],
                "headInputs": [
                    "geometry_projection * dynamic_projection",
                    "dynamic_projection",
                ],
                "usesInstanceIds": False,
                "lastLayerInitialization": "zero",
                "gate": "none",
                "noAdditionalPerInstanceAssets": True,
            }
            result["outputs"]["viewConditionedResidual"] = ["B", 1]
        if self.boundary_opportunity_head is not None:
            result["boundaryOpportunity"] = self.config["boundaryOpportunity"]
            result["outputs"]["boundaryOpportunityLogits"] = ["B", 1]
            result["outputs"]["boundaryOpportunityProbability"] = ["B", 1]
        if self.boundary_tail_residual_head is not None:
            result["boundaryTailResidual"] = self.config["boundaryTailResidual"]
            result["outputs"]["boundaryTailResidualRaw"] = ["B", 1]
            result["outputs"]["boundaryTailResidualCentered"] = ["B", 1]
        if self.query_tail_separator is not None:
            result["queryTailSeparator"] = self.config["queryTailSeparator"]
            result["outputs"]["queryTailSeparatorRawFeatures"] = [
                "B",
                DUAL_PROBE_RAW_QUERY_DIM,
            ]
            result["outputs"]["queryTailSeparatorResidualRaw"] = ["B", 1]
            result["outputs"]["queryTailSeparatorResidual"] = ["B", 1]
        if self.viewcell_extreme_visibility_enabled:
            result["viewcellExtremeVisibility"] = self.config[
                "viewcellExtremeVisibility"
            ]
            result["outputs"]["viewcellExtremeVisibilityLogit"] = ["B", 1]
        if self.viewcell_region_conditioned_visibility_enabled:
            result["viewcellRegionConditionedVisibility"] = self.config[
                "viewcellRegionConditionedVisibility"
            ]
            result["outputs"]["viewcellRegionConditionedVisibilityLogit"] = [
                "B",
                1,
            ]
            result["outputs"]["viewcellRegionConditionedVisibilityRawLogit"] = [
                "B",
                1,
            ]
        if self.dual_probe_rescue_enabled:
            result["dualProbeRescue"] = self.config["dualProbeRescue"]
            result["outputs"].update(
                {
                    "dualProbeRawFeatures": ["B", DUAL_PROBE_RAW_QUERY_DIM],
                    "dualProbePrimaryScore": ["B", 1],
                    "dualProbeCoverageScore": ["B", 1],
                    "dualProbePrimaryGate": ["B", 1],
                    "dualProbeCoverageGate": ["B", 1],
                    "dualProbePrimaryAttenuation": ["B", 1],
                    "dualProbeCoverageAttenuation": ["B", 1],
                    "dualProbeFixedResidual": ["B", 1],
                    "dualProbeResidual": ["B", 1],
                }
            )
        return result

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

    def set_boundary_opportunity_scale(self, value: float) -> None:
        scale = float(value)
        if not 0.0 <= scale <= 1.0:
            raise ValueError("boundary opportunity scale must lie in [0, 1]")
        self.boundary_opportunity_scale.fill_(
            scale if self.boundary_opportunity_head is not None else 0.0
        )

    def _bounded_instance_calibration_residual(self) -> torch.Tensor:
        if self.instance_calibration_residual_raw is None:
            return self.instance_calibration_reliability.new_zeros(
                (
                    self.num_instances,
                    self.survival_rank,
                    self.survival_parameter_dim,
                )
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
                self.num_instances,
                self.survival_rank,
                self.survival_parameter_dim,
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
                "occlusion_features": coefficients,
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
        if (
            self.occlusion_representation != "survival"
            or self.offline_survival_encoder is None
        ):
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

    @staticmethod
    def _dual_probe_raw_query(
        center_view: torch.Tensor,
        spectral_features: torch.Tensor,
        boundary_spectral_summary: torch.Tensor,
        disk_axes: torch.Tensor,
    ) -> torch.Tensor:
        """Build the fixed 108D probe layout without any learned projection."""
        if center_view.ndim != 2 or center_view.shape[1] != DUAL_PROBE_CENTER_DIM:
            raise RuntimeError("dual probe center_view layout must be [B, 9]")
        if spectral_features.ndim != 2 or spectral_features.shape[1] != DUAL_PROBE_SPECTRAL_DIM:
            raise RuntimeError("dual probe spectral feature layout must be [B, 64]")
        if boundary_spectral_summary.ndim != 2 or boundary_spectral_summary.shape[1] != DUAL_PROBE_BOUNDARY_DIM:
            raise RuntimeError("dual probe boundary summary layout must be [B, 8]")
        if disk_axes.ndim != 3 or disk_axes.shape[1:] != (VIEW_DIM, DISK_AXIS_DIM):
            raise RuntimeError("dual probe disk_axes layout must be [B, 9, 2]")
        batch = center_view.shape[0]
        if not (
            spectral_features.shape[0]
            == boundary_spectral_summary.shape[0]
            == disk_axes.shape[0]
            == batch
        ):
            raise RuntimeError("dual probe raw query rows are not aligned")
        extent = torch.linalg.vector_norm(disk_axes, dim=-1)
        region_extrema = torch.cat(
            [center_view - extent, center_view + extent, 2.0 * extent], dim=-1
        )
        result = torch.cat(
            [center_view, spectral_features, boundary_spectral_summary, region_extrema],
            dim=-1,
        )
        if result.shape != (batch, DUAL_PROBE_RAW_QUERY_DIM):
            raise RuntimeError("dual probe raw query layout drifted from 108D contract")
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("dual probe raw query is non-finite")
        return result

    def _query_tail_separator_residual(
        self,
        raw_query: torch.Tensor,
        pose_offsets: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the shared 108D query to a bounded, optionally pose-centered residual."""
        if self.query_tail_separator is None:
            zero = raw_query.new_zeros((raw_query.shape[0], 1))
            return zero, zero
        if raw_query.ndim != 2 or raw_query.shape[1] != DUAL_PROBE_RAW_QUERY_DIM:
            raise RuntimeError("query-tail separator input must follow the 108D layout")
        if self.query_tail_separator_family == "hinge":
            separator_input = torch.cat(
                [raw_query]
                + [
                    F.relu(raw_query - float(knot))
                    for knot in QUERY_TAIL_SEPARATOR_HINGE_KNOTS
                ],
                dim=-1,
            )
        else:
            separator_input = raw_query
        raw = self.query_tail_separator(separator_input)
        if raw.shape != (raw_query.shape[0], 1):
            raise RuntimeError("query-tail separator output must have shape [B, 1]")
        if self.query_tail_separator_centering == "pose_mean":
            precenter_bound = 0.5 * self.query_tail_separator_max_abs
            bounded = precenter_bound * torch.tanh(raw / precenter_bound)
            applied = _pose_centered_residual(bounded, pose_offsets)
        else:
            bound = self.query_tail_separator_max_abs
            bounded = bound * torch.tanh(raw / bound)
            applied = bounded
        if not bool(torch.isfinite(raw).all() and torch.isfinite(applied).all()):
            raise FloatingPointError("query-tail separator produced non-finite values")
        if bool((applied.abs() > self.query_tail_separator_max_abs + 1e-6).any()):
            raise FloatingPointError("query-tail separator exceeded its residual bound")
        return raw, applied

    def _query_basis_and_semantic(
        self,
        center_view: torch.Tensor,
        disk_axes: torch.Tensor,
        normalized_depth: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        effective_axes = (
            disk_axes
            if self.spectral_mode
            in {
                "moment_envelope",
                "moment_extrema",
                "moment_support",
                "moment_extrema_support",
            }
            else torch.zeros_like(disk_axes)
        )
        spectral = self.moment_query(
            center_view,
            effective_axes,
            frequency_cycles=self._bounded_frequency_cycles(),
        )
        low_rank = self._low_rank_summary(effective_axes)
        if self.occlusion_representation == "none":
            relation_condition = coefficients.new_empty((coefficients.shape[0], 0))
        else:
            assert self.relation_condition_head is not None
            relation_condition = self.relation_condition_head(
                coefficients.reshape(coefficients.shape[0], -1)
            )
        boundary_inputs = [
            spectral.spectral_features,
            center_view,
            low_rank,
            relation_condition,
        ]
        extreme = None
        support = None
        if self.spectral_mode in {"moment_extrema", "moment_extrema_support"}:
            extreme = viewcell_extreme_envelope(center_view, effective_axes)
            boundary_inputs.append(extreme.features)
        if self.spectral_mode in {"moment_support", "moment_extrema_support"}:
            support = viewcell_support_envelope(
                center_view,
                effective_axes,
                spectral.frequencies_cycles,
            )
            boundary_inputs.append(support.features)
        boundary = self.boundary_summary_head(torch.cat(boundary_inputs, dim=-1))
        if self.occlusion_representation == "none":
            basis = center_view.new_zeros(
                (center_view.shape[0], self.survival_rank)
            )
            semantic = center_view.new_zeros(
                (center_view.shape[0], SURVIVAL_SEMANTIC_DIM)
            )
        else:
            assert self.direction_basis_head is not None
            basis = self.direction_basis_head(
                torch.cat(
                    [boundary, relation_condition, center_view[:, :3]], dim=-1
                )
            )
            semantic = (
                _query_survival_semantic(basis, normalized_depth, coefficients)
                if self.occlusion_representation == "survival"
                else _query_generic_occlusion_semantic(
                    basis, normalized_depth, coefficients
                )
            )
        auxiliary = {
            "boundary_spectral_summary": boundary,
            "low_rank_summary": low_rank,
            "spectral_features": spectral.spectral_features,
            "spectral_radial_argument": spectral.s,
            "relation_condition": relation_condition,
            "effective_disk_axes": effective_axes,
        }
        if extreme is not None:
            auxiliary.update(
                {
                    "viewcell_extreme_features": extreme.features,
                    "viewcell_extreme_lower": extreme.lower,
                    "viewcell_extreme_upper": extreme.upper,
                }
            )
        if support is not None:
            auxiliary.update(
                {
                    "viewcell_support_features": support.features,
                    "viewcell_support_lower": support.lower,
                    "viewcell_support_upper": support.upper,
                }
            )
        return basis, semantic, auxiliary

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
            if runtime.ndim != 2 or runtime.shape[1] != self.runtime_feature_dim:
                raise ValueError(
                    f"runtime_features must have shape [N, {self.runtime_feature_dim}]"
                )
            rows = runtime[instance_ids] if runtime.shape[0] == self.num_instances else runtime
            if rows.shape[0] != instance_ids.shape[0]:
                raise ValueError("runtime_features must be global or candidate-aligned")
            if self.occlusion_representation == "none":
                return rows[:, :GEO_DIM], rows.new_empty((rows.shape[0], 0))
            return rows[:, :GEO_DIM], rows[:, GEO_DIM:].reshape(
                -1,
                self.survival_rank,
                self.survival_parameter_dim,
            )
        if "geometry" not in runtime_tables:
            raise ValueError("runtime tables require geometry")
        geometry_table = _float_tensor(runtime_tables["geometry"], "geometry")
        if self.occlusion_representation == "none":
            ids = instance_ids.to(geometry_table.device)
            geometry = (
                geometry_table[ids]
                if geometry_table.shape[0] == self.num_instances
                else geometry_table
            )
            if geometry.shape != (ids.numel(), GEO_DIM):
                raise ValueError("runtime geometry table has an invalid shape")
            return geometry, geometry.new_empty((ids.numel(), 0))
        feature_key = (
            "survival_coefficients"
            if self.occlusion_representation == "survival"
            else "occlusion_features"
        )
        if feature_key not in runtime_tables:
            raise ValueError(f"runtime tables require {feature_key}")
        coefficient_table = _float_tensor(
            runtime_tables[feature_key], feature_key, device=geometry_table.device
        )
        ids = instance_ids.to(geometry_table.device)
        geometry = geometry_table[ids] if geometry_table.shape[0] == self.num_instances else geometry_table
        coefficients = coefficient_table[ids] if coefficient_table.shape[0] == self.num_instances else coefficient_table
        if geometry.shape != (ids.numel(), GEO_DIM) or coefficients.shape != (
            ids.numel(),
            self.survival_rank,
            self.survival_parameter_dim,
        ):
            raise ValueError("runtime geometry or coefficient table has an invalid shape")
        return geometry, coefficients

    @staticmethod
    def _viewcell_extreme_visibility_region_input(
        query_aux: Mapping[str, torch.Tensor],
        center: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Build the registered 27D raw region input for the main logit branch."""
        extreme = query_aux.get("viewcell_extreme_features")
        if extreme is None:
            raise RuntimeError(
                "viewcell extreme visibility requires query_aux['viewcell_extreme_features']"
            )
        if extreme.ndim != 2 or extreme.shape != (
            center.shape[0],
            VIEWCELL_EXTREME_ENVELOPE_DIM,
        ):
            raise RuntimeError(
                "query_aux['viewcell_extreme_features'] must have shape "
                f"[{center.shape[0]}, {VIEWCELL_EXTREME_ENVELOPE_DIM}]"
            )
        if center.ndim != 2 or center.shape[1] != VIEW_DIM:
            raise RuntimeError(
                f"view-cell extreme visibility center input must have shape [B, {VIEW_DIM}]"
            )
        if depth.ndim != 2 or depth.shape != (center.shape[0], 1):
            raise RuntimeError(
                "view-cell extreme visibility depth input must have shape [B, 1]"
            )
        region_input = torch.cat([extreme, center, depth], dim=-1)
        if region_input.shape != (
            center.shape[0],
            VIEWCELL_EXTREME_VISIBILITY_INPUT_DIM,
        ):
            raise RuntimeError(
                "view-cell extreme visibility input layout drifted from its 27D schema"
            )
        if not bool(torch.isfinite(region_input).all()):
            raise FloatingPointError(
                "view-cell extreme visibility input is non-finite"
            )
        return region_input

    def _viewcell_extreme_visibility_logit(
        self,
        query_aux: Mapping[str, torch.Tensor],
        center: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.viewcell_extreme_visibility_projection is None:
            zero = depth.new_zeros((depth.shape[0], 1))
            return zero, depth.new_empty((depth.shape[0], 0))
        assert self.viewcell_extreme_visibility_head is not None
        region_input = self._viewcell_extreme_visibility_region_input(
            query_aux, center, depth
        )
        projected = self.viewcell_extreme_visibility_projection(region_input)
        if projected.shape != (
            region_input.shape[0],
            VIEWCELL_EXTREME_VISIBILITY_PROJECTION_DIM,
        ):
            raise RuntimeError(
                "view-cell extreme visibility projection output layout drifted"
            )
        branch_logit = self.viewcell_extreme_visibility_head(projected)
        if branch_logit.shape != (region_input.shape[0], 1):
            raise RuntimeError(
                "view-cell extreme visibility logit must have shape [B, 1]"
            )
        if not bool(torch.isfinite(branch_logit).all()):
            raise FloatingPointError(
                "view-cell extreme visibility logit is non-finite"
            )
        return branch_logit, region_input

    def _viewcell_region_conditioned_visibility_logit(
        self,
        hidden: torch.Tensor,
        query_aux: Mapping[str, torch.Tensor],
        center: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.viewcell_region_conditioned_visibility_region_projection is None:
            zero = depth.new_zeros((depth.shape[0], 1))
            return zero, {}
        assert self.viewcell_region_conditioned_visibility_hidden_projection is not None
        assert self.viewcell_region_conditioned_visibility_head is not None
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_dim:
            raise RuntimeError(
                "viewcell region-conditioned visibility hidden input must have "
                f"shape [B, {self.hidden_dim}]"
            )
        region_input = self._viewcell_extreme_visibility_region_input(
            query_aux, center, depth
        )
        if region_input.shape[1] != VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM:
            raise RuntimeError(
                "viewcell region-conditioned visibility region input must have "
                f"width {VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM}"
            )
        region_projection = self.viewcell_region_conditioned_visibility_region_projection(
            region_input
        )
        hidden_projection = self.viewcell_region_conditioned_visibility_hidden_projection(
            hidden
        )
        expected_projection_shape = (
            hidden.shape[0],
            VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM,
        )
        if region_projection.shape != expected_projection_shape:
            raise RuntimeError(
                "viewcell region-conditioned visibility region projection layout drifted"
            )
        if hidden_projection.shape != expected_projection_shape:
            raise RuntimeError(
                "viewcell region-conditioned visibility hidden projection layout drifted"
            )
        interaction = region_projection * hidden_projection
        fusion = torch.cat(
            [region_projection, hidden_projection, interaction], dim=-1
        )
        if fusion.shape != (
            hidden.shape[0],
            VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM,
        ):
            raise RuntimeError(
                "viewcell region-conditioned visibility fusion layout drifted"
            )
        branch_logit = self.viewcell_region_conditioned_visibility_head(fusion)
        if branch_logit.shape != (hidden.shape[0], 1):
            raise RuntimeError(
                "viewcell region-conditioned visibility logit must have shape [B, 1]"
            )
        if not bool(torch.isfinite(branch_logit).all()):
            raise FloatingPointError(
                "viewcell region-conditioned visibility logit is non-finite"
            )
        return branch_logit, {
            "viewcell_region_conditioned_visibility_input": region_input,
            "viewcell_region_conditioned_visibility_region_projection": (
                region_projection
            ),
            "viewcell_region_conditioned_visibility_hidden_projection": (
                hidden_projection
            ),
            "viewcell_region_conditioned_visibility_interaction": interaction,
            "viewcell_region_conditioned_visibility_fusion": fusion,
        }

    def _view_conditioned_residual(
        self,
        geometry: torch.Tensor,
        basis: torch.Tensor,
        semantic: torch.Tensor,
        query_aux: Mapping[str, torch.Tensor],
        center: torch.Tensor,
        depth: torch.Tensor,
        base_visibility_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.view_residual_head is None:
            return torch.zeros_like(depth)
        assert self.view_residual_geometry_projection is not None
        assert self.view_residual_dynamic_projection is not None
        dynamic_input = torch.cat(
            [
                basis,
                semantic,
                query_aux["boundary_spectral_summary"],
                center,
                query_aux["low_rank_summary"],
                depth,
                base_visibility_logits.detach(),
            ],
            dim=-1,
        )
        if dynamic_input.shape[1] != self.view_residual_dynamic_input_dim:
            raise RuntimeError("view residual dynamic input layout drifted from its schema")
        geometry_projection = self.view_residual_geometry_projection(geometry)
        dynamic_projection = self.view_residual_dynamic_projection(dynamic_input)
        interaction = geometry_projection * dynamic_projection
        residual_raw = self.view_residual_head(
            torch.cat([interaction, dynamic_projection], dim=-1)
        )
        residual = self.view_residual_max_abs * torch.tanh(residual_raw)
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError("view-conditioned residual is non-finite")
        return residual

    def _boundary_visibility_opportunity(
        self,
        geometry: torch.Tensor,
        semantic: torch.Tensor,
        query_aux: Mapping[str, torch.Tensor],
        center: torch.Tensor,
        depth: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a one-way, bounded logit uplift for view-cell tail rescue."""
        if self.boundary_opportunity_head is None:
            zero = torch.zeros_like(base_logits)
            return base_logits, zero, zero, zero
        assert self.boundary_opportunity_geometry_projection is not None
        assert self.boundary_opportunity_region_projection is not None
        if "viewcell_extreme_features" not in query_aux:
            raise RuntimeError(
                "boundary opportunity requires analytic view-cell extrema"
            )
        region_input = torch.cat(
            [query_aux["viewcell_extreme_features"], center, depth], dim=-1
        )
        if region_input.shape[1] != BOUNDARY_OPPORTUNITY_REGION_DIM:
            raise RuntimeError("boundary opportunity region layout drifted")
        geometry_projection = self.boundary_opportunity_geometry_projection(
            geometry
        )
        region_projection = self.boundary_opportunity_region_projection(
            region_input
        )
        opportunity_input = torch.cat(
            [
                geometry_projection * region_projection,
                region_projection,
                semantic,
                base_logits.detach(),
            ],
            dim=-1,
        )
        opportunity_logits = self.boundary_opportunity_head(opportunity_input)
        raw_logit_uplift = (
            self.boundary_opportunity_max_logit_uplift
            * torch.sigmoid(opportunity_logits)
        )
        applied_logit_uplift = self.boundary_opportunity_scale * raw_logit_uplift
        if float(self.boundary_opportunity_scale.detach()) == 0.0:
            return base_logits, opportunity_logits, raw_logit_uplift, applied_logit_uplift
        final_logits = base_logits + applied_logit_uplift
        if not bool(torch.isfinite(final_logits).all()):
            raise FloatingPointError("boundary opportunity fusion is non-finite")
        return final_logits, opportunity_logits, raw_logit_uplift, applied_logit_uplift

    def _dual_probe_rescue(
        self,
        raw_query: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply the frozen two-probe posterior rescue with learned attenuation."""
        if self.dual_probe_rescue_attenuation_head is None:
            return base_logits, {}
        if raw_query.shape != (base_logits.shape[0], DUAL_PROBE_RAW_QUERY_DIM):
            raise RuntimeError("dual probe raw query is not candidate-aligned")
        if base_logits.ndim != 2 or base_logits.shape[1] != 1:
            raise RuntimeError("dual probe base logits must have shape [B, 1]")
        detached_query = raw_query.detach()

        def score(
            mean: torch.Tensor,
            scale: torch.Tensor,
            coefficients: torch.Tensor,
        ) -> torch.Tensor:
            standardized = (detached_query - mean) / scale
            return coefficients[0] + standardized @ coefficients[1:]

        primary_score = score(
            self.dual_probe_primary_mean,
            self.dual_probe_primary_scale,
            self.dual_probe_primary_coefficients,
        )
        coverage_score = score(
            self.dual_probe_coverage_mean,
            self.dual_probe_coverage_scale,
            self.dual_probe_coverage_coefficients,
        )
        primary_gate = torch.clamp(
            torch.tanh(
                (self.dual_probe_primary_threshold - primary_score)
                / self.dual_probe_temperature
            ),
            min=0.0,
        )
        coverage_gate = torch.clamp(
            torch.tanh(
                (self.dual_probe_coverage_threshold - coverage_score)
                / self.dual_probe_temperature
            ),
            min=0.0,
        )
        attenuation_input = torch.cat([detached_query, base_logits.detach()], dim=-1)
        attenuation_logits = self.dual_probe_rescue_attenuation_head(attenuation_input)
        attenuation = torch.clamp(
            1.0
            - (
                F.softplus(attenuation_logits)
                - math.log(2.0)
            ),
            min=0.0,
            max=1.0,
        )
        primary_attenuation = attenuation[:, 0:1]
        coverage_attenuation = attenuation[:, 1:2]
        primary_gate = primary_gate.unsqueeze(-1)
        coverage_gate = coverage_gate.unsqueeze(-1)
        fixed_residual = self.dual_probe_alpha * torch.maximum(
            primary_gate,
            self.dual_probe_coverage_weight * coverage_gate,
        )
        residual = self.dual_probe_alpha * torch.maximum(
            primary_attenuation * primary_gate,
            self.dual_probe_coverage_weight
            * coverage_attenuation
            * coverage_gate,
        )
        if not bool(torch.isfinite(residual).all() and torch.isfinite(fixed_residual).all()):
            raise FloatingPointError("dual probe rescue residual is non-finite")
        if bool((residual < -1e-7).any()):
            raise FloatingPointError("dual probe rescue residual became negative")
        if bool((residual > fixed_residual + 1e-6).any()):
            raise FloatingPointError("dual probe rescue residual exceeded fixed posterior residual")
        final_logits = base_logits + residual
        if not bool(torch.isfinite(final_logits).all()):
            raise FloatingPointError("dual probe rescue fusion is non-finite")
        return final_logits, {
            "dual_probe_raw_features": raw_query,
            "dual_probe_primary_score": primary_score.unsqueeze(-1),
            "dual_probe_coverage_score": coverage_score.unsqueeze(-1),
            "dual_probe_primary_gate": primary_gate,
            "dual_probe_coverage_gate": coverage_gate,
            "dual_probe_attenuation_logits": attenuation_logits,
            "dual_probe_primary_attenuation": primary_attenuation,
            "dual_probe_coverage_attenuation": coverage_attenuation,
            "dual_probe_fixed_residual": fixed_residual,
            "dual_probe_residual": residual,
            "dual_probe_rescue_base_logit": base_logits.detach(),
        }

    def _boundary_tail_residual(
        self,
        geometry: torch.Tensor,
        semantic: torch.Tensor,
        query_aux: Mapping[str, torch.Tensor],
        center: torch.Tensor,
        depth: torch.Tensor,
        base_logits: torch.Tensor,
        pose_offsets: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the configured bounded signed correction to frozen base logits."""
        if self.boundary_tail_residual_head is None:
            zero = torch.zeros_like(base_logits)
            return base_logits, zero, zero
        assert self.boundary_tail_residual_geometry_projection is not None
        assert self.boundary_tail_residual_region_projection is not None
        if "viewcell_extreme_features" not in query_aux:
            raise RuntimeError(
                "boundary tail residual requires analytic view-cell extrema"
            )
        region_input = torch.cat(
            [query_aux["viewcell_extreme_features"], center, depth], dim=-1
        )
        if region_input.shape[1] != BOUNDARY_TAIL_RESIDUAL_REGION_DIM:
            raise RuntimeError("boundary tail residual region layout drifted")
        geometry_projection = self.boundary_tail_residual_geometry_projection(
            geometry
        )
        region_projection = self.boundary_tail_residual_region_projection(region_input)
        if self.boundary_tail_residual_fusion == "affine_region":
            head_input = torch.cat(
                [
                    geometry_projection + region_projection,
                    geometry_projection - region_projection,
                    region_input,
                    semantic,
                    base_logits.detach(),
                ],
                dim=-1,
            )
        else:
            head_input = torch.cat(
                [
                    geometry_projection * region_projection,
                    region_projection,
                    semantic,
                    base_logits.detach(),
                ],
                dim=-1,
            )
        raw = self.boundary_tail_residual_head(head_input)
        if self.boundary_tail_residual_region_shortcut is not None:
            raw = raw + self.boundary_tail_residual_region_shortcut(region_input)
        if self.boundary_tail_residual_centering == "pose_mean":
            # Centering can double the absolute range when one candidate is at
            # one bound and the pose mean is at the opposite bound.
            precenter_bound = 0.5 * self.boundary_tail_residual_max_abs
            bounded = precenter_bound * torch.tanh(raw / precenter_bound)
            applied = _pose_centered_residual(bounded, pose_offsets)
        else:
            bound = self.boundary_tail_residual_max_abs
            bounded = bound * torch.tanh(raw / bound)
            applied = bounded
        final = base_logits + applied
        if not bool(torch.isfinite(final).all()):
            raise FloatingPointError("boundary tail residual fusion is non-finite")
        return final, bounded, applied

    def forward_batch(
        self,
        instance_ids: Any,
        center_view: Any,
        disk_axes: Any,
        normalized_depth: Any,
        runtime_tables: Mapping[str, Any],
        *,
        pose_offsets: Any | None = None,
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
                *(
                    [basis, semantic]
                    if self.occlusion_representation != "none"
                    else []
                ),
                query_aux["boundary_spectral_summary"],
                center,
                query_aux["low_rank_summary"],
                depth,
            ],
            dim=-1,
        )
        if trunk_input.shape[1] != self.runtime_head_input_dim:
            raise RuntimeError("v4 runtime input layout drifted from its schema")
        hidden = self.shared_trunk(trunk_input)
        base_visibility_logits = self.visibility_head(hidden)
        viewcell_extreme_visibility_logit: torch.Tensor | None = None
        viewcell_extreme_visibility_input: torch.Tensor | None = None
        viewcell_region_conditioned_visibility_raw_logit: torch.Tensor | None = None
        viewcell_region_conditioned_visibility_logit: torch.Tensor | None = None
        viewcell_region_conditioned_visibility_aux: dict[str, torch.Tensor] = {}
        if self.viewcell_extreme_visibility_enabled:
            (
                viewcell_extreme_visibility_logit,
                viewcell_extreme_visibility_input,
            ) = self._viewcell_extreme_visibility_logit(
                query_aux,
                center,
                depth,
            )
            base_visibility_logits = (
                base_visibility_logits + viewcell_extreme_visibility_logit
            )
        if self.viewcell_region_conditioned_visibility_enabled:
            (
                viewcell_region_conditioned_visibility_raw_logit,
                viewcell_region_conditioned_visibility_aux,
            ) = self._viewcell_region_conditioned_visibility_logit(
                hidden,
                query_aux,
                center,
                depth,
            )
            if self.viewcell_region_conditioned_visibility_centering == "pose_mean":
                viewcell_region_conditioned_visibility_logit = (
                    _pose_centered_residual(
                        viewcell_region_conditioned_visibility_raw_logit,
                        pose_offsets,
                    )
                )
            else:
                viewcell_region_conditioned_visibility_logit = (
                    viewcell_region_conditioned_visibility_raw_logit
                )
            base_visibility_logits = (
                base_visibility_logits + viewcell_region_conditioned_visibility_logit
            )
        if self.cull_certificate_head is None:
            certificate_raw = torch.zeros_like(base_visibility_logits)
            certificate_suppression = torch.zeros_like(base_visibility_logits)
        else:
            certificate_evidence = torch.cat(
                [base_visibility_logits, semantic, depth], dim=-1
            ).detach()
            certificate_input = {
                "hidden": hidden.detach(),
                "evidence": certificate_evidence,
                "hidden_evidence": torch.cat(
                    [hidden.detach(), certificate_evidence], dim=-1
                ),
                "runtime": trunk_input.detach(),
                "hidden_runtime": torch.cat(
                    [hidden.detach(), trunk_input.detach()], dim=-1
                ),
            }[self.cull_certificate_input_mode]
            certificate_raw = self.cull_certificate_head(certificate_input)
            certificate_suppression = self.cull_certificate_max_suppression * torch.sigmoid(
                certificate_raw
            )
        view_conditioned_residual = self._view_conditioned_residual(
            geometry,
            basis,
            semantic,
            query_aux,
            center,
            depth,
            base_visibility_logits,
        )
        pre_tail_residual_visibility_logits = (
            base_visibility_logits
            - certificate_suppression
            + view_conditioned_residual
        )
        (
            visibility_logits,
            boundary_tail_residual_raw,
            boundary_tail_residual_centered,
        ) = self._boundary_tail_residual(
            geometry,
            semantic,
            query_aux,
            center,
            depth,
            pre_tail_residual_visibility_logits,
            pose_offsets,
        )
        # Tail suppression is allowed to move scores in either direction.  Apply
        # the one-way rescue last so a frozen suppression head cannot cancel an
        # uplift assigned to a difficult positive.
        pre_opportunity_visibility_logits = visibility_logits
        (
            visibility_logits,
            boundary_opportunity_logits,
            boundary_opportunity_raw_logit_uplift,
            boundary_opportunity_logit_uplift,
        ) = self._boundary_visibility_opportunity(
            geometry,
            semantic,
            query_aux,
            center,
            depth,
            pre_opportunity_visibility_logits,
        )
        pre_query_tail_separator_visibility_logits = visibility_logits
        query_tail_separator_raw_features = visibility_logits.new_empty(
            (visibility_logits.shape[0], 0)
        )
        query_tail_separator_residual_raw = torch.zeros_like(visibility_logits)
        query_tail_separator_residual = torch.zeros_like(visibility_logits)
        if self.query_tail_separator is not None:
            query_tail_separator_raw_features = self._dual_probe_raw_query(
                center,
                query_aux["spectral_features"],
                query_aux["boundary_spectral_summary"],
                axes,
            )
            (
                query_tail_separator_residual_raw,
                query_tail_separator_residual,
            ) = self._query_tail_separator_residual(
                query_tail_separator_raw_features,
                pose_offsets,
            )
            visibility_logits = (
                pre_query_tail_separator_visibility_logits
                + query_tail_separator_residual
            )
        dual_probe_rescue_aux: dict[str, torch.Tensor] = {}
        if self.dual_probe_rescue_enabled:
            dual_probe_raw_features = self._dual_probe_raw_query(
                center,
                query_aux["spectral_features"],
                query_aux["boundary_spectral_summary"],
                axes,
            )
            visibility_logits, dual_probe_rescue_aux = self._dual_probe_rescue(
                dual_probe_raw_features,
                visibility_logits,
            )
        task_input = torch.cat([hidden, torch.sigmoid(visibility_logits)], dim=-1)
        utility_logits = self.utility_head(task_input)
        download_logits = self.download_head(task_input)
        auxiliary = {
            "visibility_logits": visibility_logits,
            "base_visibility_logits": base_visibility_logits,
            "cull_certificate_raw": certificate_raw,
            "cull_certificate_suppression": certificate_suppression,
            "view_conditioned_residual": view_conditioned_residual,
            "pre_opportunity_visibility_logits": pre_opportunity_visibility_logits,
            "boundary_opportunity_logits": boundary_opportunity_logits,
            "boundary_opportunity_raw_logit_uplift": boundary_opportunity_raw_logit_uplift,
            "boundary_opportunity_logit_uplift": boundary_opportunity_logit_uplift,
            "pre_query_tail_separator_visibility_logits": (
                pre_query_tail_separator_visibility_logits
            ),
            "query_tail_separator_raw_features": (
                query_tail_separator_raw_features
            ),
            "query_tail_separator_residual_raw": (
                query_tail_separator_residual_raw
            ),
            "query_tail_separator_residual": query_tail_separator_residual,
            "pre_tail_residual_visibility_logits": (
                pre_tail_residual_visibility_logits
            ),
            "boundary_tail_residual_raw": boundary_tail_residual_raw,
            "boundary_tail_residual_centered": boundary_tail_residual_centered,
            "utility_logits": utility_logits,
            "download_logits": download_logits,
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
            **query_aux,
            **dual_probe_rescue_aux,
        }
        if self.viewcell_extreme_visibility_enabled:
            assert viewcell_extreme_visibility_logit is not None
            assert viewcell_extreme_visibility_input is not None
            auxiliary.update(
                {
                    "viewcell_extreme_visibility_logit": (
                        viewcell_extreme_visibility_logit
                    ),
                    "viewcell_extreme_visibility_input": (
                        viewcell_extreme_visibility_input
                    ),
                }
            )
        if self.viewcell_region_conditioned_visibility_enabled:
            assert viewcell_region_conditioned_visibility_raw_logit is not None
            assert viewcell_region_conditioned_visibility_logit is not None
            auxiliary.update(
                {
                    "viewcell_region_conditioned_visibility_raw_logit": (
                        viewcell_region_conditioned_visibility_raw_logit
                    ),
                    "viewcell_region_conditioned_visibility_logit": (
                        viewcell_region_conditioned_visibility_logit
                    ),
                    **viewcell_region_conditioned_visibility_aux,
                }
            )
        return visibility_logits, auxiliary

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
        pose_offsets: Any | None = None,
        **_unused: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del camera_pos_norm
        if runtime_features is None:
            raise ValueError(
                f"v4 requires the fixed {self.runtime_feature_dim}D runtime feature table"
            )
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
        candidate_camera = _broadcast_rows(
            candidate_camera_world,
            ids.numel(),
            3,
            "candidate_camera_world",
            device=runtime.device,
        )
        logits, aux = self.forward_batch(
            ids,
            ray_query.center_view,
            ray_query.disk_axes,
            depth,
            {"runtime_features": runtime},
            pose_offsets=pose_offsets,
        )
        aux.update(
            {
                "candidate_camera_world": candidate_camera,
                "query_center_world": query_center,
                "viewcell_radius_m": radius,
                "evidence_target": torch.zeros_like(logits),
                "final_logits": logits,
                "base_logits": aux["base_visibility_logits"],
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
        if self.occlusion_representation != "survival":
            raise RuntimeError(
                "query_survival_from_direction is only valid for survival representation"
            )
        coefficient_tensor = _float_tensor(coefficients, "coefficients")
        if coefficient_tensor.ndim != 3 or coefficient_tensor.shape[1:] != (
            self.survival_rank,
            self.survival_parameter_dim,
        ):
            raise ValueError(
                "coefficients must have shape "
                f"[B, {self.survival_rank}, {self.survival_parameter_dim}]"
            )
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
            and name
            not in {"instance_calibration_residual_raw", "generic_occlusion_features"}
        ]
        raw_norm = torch.linalg.norm(self.moment_query.frequency_cycles.float(), dim=-1)
        frequency_overflow = F.relu(raw_norm - self.max_frequency_norm_cycles).square().mean()
        return (torch.stack(terms).mean() if terms else frequency_overflow * 0.0) + frequency_overflow


__all__ = [
    "BOUNDARY_SUMMARY_DIM",
    "BoundedRelationSurvivalMomentModel",
    "DUAL_PROBE_ATTENUATION_HIDDEN_DIM",
    "DUAL_PROBE_RAW_QUERY_DIM",
    "DUAL_PROBE_REGION_EXTREMA_DIM",
    "GEO_DIM",
    "LOW_RANK_SUMMARY_DIM",
    "MODEL_SCHEMA",
    "OCCLUSION_REPRESENTATION_MODES",
    "RUNTIME_FEATURE_DIM",
    "RUNTIME_HEAD_INPUT_DIM_WITHOUT_OCCLUSION",
    "SUPPORTED_SURVIVAL_RANKS",
    "SURVIVAL_PARAMETER_DIM",
    "SURVIVAL_RANK",
    "VIEWCELL_EXTREME_VISIBILITY_INPUT_DIM",
    "VIEWCELL_EXTREME_VISIBILITY_PROJECTION_DIM",
    "VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM",
    "VIEWCELL_REGION_CONDITIONED_VISIBILITY_HEAD_DIM",
    "VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM",
    "VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM",
    "VIEW_RESIDUAL_DYNAMIC_INPUT_DIM",
]
