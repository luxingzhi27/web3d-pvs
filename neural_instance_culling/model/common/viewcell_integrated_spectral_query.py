"""Compact analytic spectral queries for one view-cell candidate.

The runtime contract is deliberately smaller than the historical Fourier117
path.  A caller supplies one nine-value ray-space query at the view-cell
center and a diagonal variance for that query.  The module analytically
integrates sixteen shared, learnable joint frequencies over the Gaussian
approximation of the cell.  It never materializes or loops over subposes.

The first nine output values are the center query.  Each frequency contributes
integrated sine, integrated cosine, and unresolved phase energy, so the full
spectral feature has ``9 + 16 * 3 = 57`` values.  A small query head keeps the
low-frequency body and the high-frequency boundary branch separate; the latter
is multiplied by a continuous gate rather than selected with a runtime branch.
The optional relation condition is intended for an offline-generated survival
field summary.  Relation coefficients can also be queried once with the
resulting basis, which keeps the browser path at one batch query per candidate
set.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


VIEWCELL_INTEGRATED_SPECTRAL_SCHEMA = "view-cell-integrated-spectral-query-v1"
VIEW_DIM = 9
FREQUENCY_COUNT = 16
INTEGRATED_COMPONENTS_PER_FREQUENCY = 3
SPECTRAL_FEATURE_DIM = VIEW_DIM + FREQUENCY_COUNT * INTEGRATED_COMPONENTS_PER_FREQUENCY
LOW_FREQUENCY_COUNT = 8
MIDDLE_FREQUENCY_COUNT = 4
HIGH_FREQUENCY_COUNT = 4
DEFAULT_QUERY_BASIS_DIM = 4
MAX_QUERY_BASIS_DIM = 6
DEFAULT_RELATION_CONDITION_DIM = 8
DEFAULT_VIEWCELL_RADIUS_M = 2.0
DEFAULT_MODEL_FOV_Y_DEG = 66.0
DEFAULT_ASPECT_RATIO = 16.0 / 9.0


def _as_batched_float(value: torch.Tensor | Any, dim: int, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 1 and tensor.shape[0] == dim:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] != dim:
        raise ValueError(f"{name} must have shape [B, {dim}] or [{dim}]")
    if not tensor.is_floating_point():
        tensor = tensor.float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor.float()


def _broadcast_batch_scalar(
    value: torch.Tensor | float,
    batch: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1).expand(batch, 1)
    elif tensor.ndim == 1:
        if tensor.numel() == 1:
            tensor = tensor.reshape(1, 1).expand(batch, 1)
        elif tensor.numel() == batch:
            tensor = tensor.reshape(batch, 1)
        else:
            raise ValueError(f"{name} must be scalar, [B], or [B, 1]")
    elif tensor.ndim == 2 and tuple(tensor.shape) == (batch, 1):
        pass
    else:
        raise ValueError(f"{name} must be scalar, [B], or [B, 1]")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def analytic_viewcell_diagonal_variance(
    center_view: torch.Tensor,
    instance_distance_m: torch.Tensor | float,
    viewcell_radius_m: float = DEFAULT_VIEWCELL_RADIUS_M,
    tan_x: torch.Tensor | float | None = None,
    tan_y: torch.Tensor | float | None = None,
    rotation_angular_std_radians: float = 0.0,
) -> torch.Tensor:
    """Approximate the query covariance of the registered view-cell contract.

    ``center_view`` uses the existing nine-value ray-space layout:
    three unit ray components followed by normalized log distance, forward
    cosine, normalized screen ``u``/``v``, and normalized horizontal/vertical
    angular size.  The returned tensor is a diagonal *variance*, not a
    standard deviation.  Position uncertainty is modeled by the cell radius;
    optional rotation uncertainty is an explicit caller-owned extension and
    defaults to zero for the current fixed-heading HKUST contract.

    This is a conservative first-order estimate.  It uses only arithmetic,
    reciprocal, and square operations; it does not project AABB corners or
    enumerate the legal subposes.
    """

    center = _as_batched_float(center_view, VIEW_DIM, "center_view")
    if not math.isfinite(float(viewcell_radius_m)) or float(viewcell_radius_m) < 0.0:
        raise ValueError("viewcell_radius_m must be finite and non-negative")
    if not math.isfinite(float(rotation_angular_std_radians)) or float(rotation_angular_std_radians) < 0.0:
        raise ValueError("rotation_angular_std_radians must be finite and non-negative")

    batch = int(center.shape[0])
    device = center.device
    distance = _broadcast_batch_scalar(instance_distance_m, batch, "instance_distance_m", device)
    distance = torch.clamp(distance, min=1e-3, max=1e6)

    default_tan_y = math.tan(math.radians(DEFAULT_MODEL_FOV_Y_DEG) * 0.5)
    default_tan_x = default_tan_y * DEFAULT_ASPECT_RATIO
    tan_x_tensor = _broadcast_batch_scalar(
        default_tan_x if tan_x is None else tan_x, batch, "tan_x", device
    )
    tan_y_tensor = _broadcast_batch_scalar(
        default_tan_y if tan_y is None else tan_y, batch, "tan_y", device
    )
    tan_x_tensor = torch.clamp(tan_x_tensor, min=1e-4, max=1e4)
    tan_y_tensor = torch.clamp(tan_y_tensor, min=1e-4, max=1e4)

    ray = F.normalize(center[:, :3], dim=-1, eps=1e-6)
    angular_std = torch.clamp(
        float(viewcell_radius_m) / distance + float(rotation_angular_std_radians),
        min=0.0,
        max=math.pi * 0.5,
    )
    angular_variance = angular_std.square()
    variance = torch.zeros_like(center)

    # A tangent-plane perturbation changes each direction component according
    # to its available orthogonal component.  This preserves zero uncertainty
    # exactly and avoids inventing an angular range for a fixed heading.
    variance[:, :3] = angular_variance * torch.clamp(1.0 - ray.square(), min=0.0, max=1.0)

    forward_cosine = torch.clamp(center[:, 4:5], min=-1.0, max=1.0)
    variance[:, 4:5] = angular_variance
    screen_denominator_x = torch.clamp(torch.abs(forward_cosine) * tan_x_tensor, min=1e-3)
    screen_denominator_y = torch.clamp(torch.abs(forward_cosine) * tan_y_tensor, min=1e-3)
    variance[:, 5:6] = (angular_std / (screen_denominator_x * 4.0)).square()
    variance[:, 6:7] = (angular_std / (screen_denominator_y * 4.0)).square()

    # The existing normalized log-distance feature has derivative
    # 1 / (4 * (100 + distance)).  A radius-sized radial displacement is used
    # as a conservative local span.
    log_distance_derivative = 1.0 / (4.0 * (100.0 + distance))
    variance[:, 3:4] = (log_distance_derivative * float(viewcell_radius_m)).square()

    radial_fraction = float(viewcell_radius_m) / distance
    horizontal_angular = torch.clamp(center[:, 7:8] + 1.0, min=0.0)
    vertical_angular = torch.clamp(center[:, 8:9] + 1.0, min=0.0)
    variance[:, 7:8] = (0.5 * horizontal_angular * radial_fraction).square()
    variance[:, 8:9] = (0.5 * vertical_angular * radial_fraction).square()
    return torch.clamp(variance, min=0.0, max=1e4)


def build_viewcell_query(
    center_view: torch.Tensor,
    instance_distance_m: torch.Tensor | float,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the single center query and its analytic diagonal variance."""

    center = _as_batched_float(center_view, VIEW_DIM, "center_view")
    return center, analytic_viewcell_diagonal_variance(center, instance_distance_m, **kwargs)


# Readable alias for callers that prefer the term used in experiment reports.
analytic_viewcell_uncertainty = analytic_viewcell_diagonal_variance


@dataclass(frozen=True)
class IntegratedSpectralQueryResult:
    """Named outputs of one batched view-cell query."""

    center_view: torch.Tensor
    diagonal_variance: torch.Tensor
    frequencies: torch.Tensor
    phase: torch.Tensor
    attenuation: torch.Tensor
    integrated_sin_cos: torch.Tensor
    unresolved_energy: torch.Tensor
    per_frequency: torch.Tensor
    low_frequency: torch.Tensor
    middle_frequency: torch.Tensor
    high_frequency: torch.Tensor
    high_frequency_energy: torch.Tensor
    high_frequency_gate: torch.Tensor
    spectral_features: torch.Tensor
    query_basis: torch.Tensor | None = None
    relation_condition: torch.Tensor | None = None
    relation_survival: torch.Tensor | None = None

    def as_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            "centerView": self.center_view,
            "diagonalVariance": self.diagonal_variance,
            "frequencies": self.frequencies,
            "phase": self.phase,
            "attenuation": self.attenuation,
            "integratedSinCos": self.integrated_sin_cos,
            "unresolvedEnergy": self.unresolved_energy,
            "perFrequency": self.per_frequency,
            "lowFrequency": self.low_frequency,
            "middleFrequency": self.middle_frequency,
            "highFrequency": self.high_frequency,
            "highFrequencyEnergy": self.high_frequency_energy,
            "highFrequencyGate": self.high_frequency_gate,
            "spectralFeatures": self.spectral_features,
            "queryBasis": self.query_basis,
            "relationCondition": self.relation_condition,
            "relationSurvival": self.relation_survival,
        }


def _initial_frequency_vectors() -> torch.Tensor:
    """Create deterministic, non-collapsed low/middle/high joint frequencies."""

    index = torch.arange(1, FREQUENCY_COUNT + 1, dtype=torch.float32).unsqueeze(1)
    dimension = torch.arange(1, VIEW_DIM + 1, dtype=torch.float32).unsqueeze(0)
    directions = torch.sin(index * dimension * 0.71) + 0.5 * torch.cos(index * dimension * 0.37)
    directions = F.normalize(directions, dim=-1)
    target_norms = torch.cat([
        torch.logspace(math.log10(0.25), math.log10(1.8), LOW_FREQUENCY_COUNT),
        torch.logspace(math.log10(2.2), math.log10(3.5), MIDDLE_FREQUENCY_COUNT),
        torch.logspace(math.log10(4.2), math.log10(6.8), HIGH_FREQUENCY_COUNT),
    ])
    return directions * target_norms.unsqueeze(1)


class ViewCellIntegratedSpectralQuery(nn.Module):
    """One-query-per-candidate integrated spectral view-cell encoder.

    ``center_view`` and ``diagonal_variance`` are both ``[B, 9]``.  The
    optional ``relation_condition`` is a fixed-size summary of an offline
    relation survival field, normally the eight semantic values emitted by
    the current monotone survival query.  It conditions the continuous
    high-frequency gate and the compact query basis; no relation graph or
    subpose data is needed at runtime.
    """

    def __init__(
        self,
        query_basis_dim: int = DEFAULT_QUERY_BASIS_DIM,
        relation_condition_dim: int = DEFAULT_RELATION_CONDITION_DIM,
        max_frequency_norm: float = 8.0,
    ) -> None:
        super().__init__()
        if int(query_basis_dim) < 1 or int(query_basis_dim) > MAX_QUERY_BASIS_DIM:
            raise ValueError(f"query_basis_dim must be in [1, {MAX_QUERY_BASIS_DIM}]")
        if int(relation_condition_dim) < 1:
            raise ValueError("relation_condition_dim must be positive")
        if not math.isfinite(float(max_frequency_norm)) or float(max_frequency_norm) <= 0.0:
            raise ValueError("max_frequency_norm must be finite and positive")

        self.query_basis_dim = int(query_basis_dim)
        self.relation_condition_dim = int(relation_condition_dim)
        self.max_frequency_norm = float(max_frequency_norm)
        self.frequency_vectors = nn.Parameter(_initial_frequency_vectors())
        self.register_buffer(
            "frequency_group_ids",
            torch.tensor(
                [0] * LOW_FREQUENCY_COUNT
                + [1] * MIDDLE_FREQUENCY_COUNT
                + [2] * HIGH_FREQUENCY_COUNT,
                dtype=torch.int64,
            ),
        )

        low_input_dim = VIEW_DIM + (
            LOW_FREQUENCY_COUNT + MIDDLE_FREQUENCY_COUNT
        ) * INTEGRATED_COMPONENTS_PER_FREQUENCY
        high_input_dim = HIGH_FREQUENCY_COUNT * INTEGRATED_COMPONENTS_PER_FREQUENCY + 2
        self.low_frequency_body = nn.Sequential(
            nn.Linear(low_input_dim, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 16),
            nn.ReLU(inplace=True),
        )
        self.high_frequency_body = nn.Sequential(
            nn.Linear(high_input_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 12),
            nn.ReLU(inplace=True),
        )
        self.relation_body = nn.Sequential(
            nn.Linear(self.relation_condition_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
        )
        self.high_frequency_gate = nn.Sequential(
            nn.Linear(self.relation_condition_dim + 4, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 1),
        )
        self.query_fuse = nn.Linear(16 + 12 + 8, self.query_basis_dim)

    @property
    def spectral_feature_dim(self) -> int:
        return SPECTRAL_FEATURE_DIM

    @property
    def frequency_count(self) -> int:
        return FREQUENCY_COUNT

    @property
    def input_dim(self) -> int:
        return VIEW_DIM

    def bounded_frequency_vectors(self) -> torch.Tensor:
        """Return learnable frequencies with a differentiable norm ceiling."""

        raw = self.frequency_vectors.float()
        norms = torch.linalg.norm(raw, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(self.max_frequency_norm / norms, max=1.0)
        return raw * scale

    def frequency_norms(self) -> torch.Tensor:
        return torch.linalg.norm(self.bounded_frequency_vectors(), dim=-1)

    def frequency_regularization(self) -> torch.Tensor:
        """Penalize norm overflow without forcing low and high groups together."""

        norms = torch.linalg.norm(self.frequency_vectors.float(), dim=-1)
        overflow = F.relu(norms - self.max_frequency_norm).square().mean()
        return overflow

    def export_schema(self) -> dict[str, Any]:
        """Return the JSON-compatible runtime layout for an exporter."""

        return {
            "schema": VIEWCELL_INTEGRATED_SPECTRAL_SCHEMA,
            "input": {
                "centerView": {
                    "offset": 0,
                    "dim": VIEW_DIM,
                    "components": [
                        "rayDirectionX",
                        "rayDirectionY",
                        "rayDirectionZ",
                        "normalizedLogDistance",
                        "dotForward",
                        "screenU",
                        "screenV",
                        "horizontalAngularSize",
                        "verticalAngularSize",
                    ],
                },
                "uncertainty": {
                    "type": "diagonalVariance",
                    "dim": VIEW_DIM,
                    "nonNegative": True,
                    "analyticSource": "registered-viewcell-radius-and-fixed-heading-contract",
                },
            },
            "frequency": {
                "count": FREQUENCY_COUNT,
                "jointInputDim": VIEW_DIM,
                "learnable": True,
                "phase": "2*pi*dot(jointFrequency, centerView)",
                "groups": {
                    "low": {"offset": 0, "count": LOW_FREQUENCY_COUNT},
                    "middle": {"offset": LOW_FREQUENCY_COUNT, "count": MIDDLE_FREQUENCY_COUNT},
                    "high": {
                        "offset": LOW_FREQUENCY_COUNT + MIDDLE_FREQUENCY_COUNT,
                        "count": HIGH_FREQUENCY_COUNT,
                    },
                },
                "maxNorm": self.max_frequency_norm,
            },
            "gaussianIntegration": {
                "componentsPerFrequency": INTEGRATED_COMPONENTS_PER_FREQUENCY,
                "layout": ["attenuatedSin", "attenuatedCos", "unresolvedPhaseEnergy"],
                "attenuation": "exp(-0.5 * frequency^T * diagonalVariance * frequency)",
                "unresolvedEnergy": "sqrt(max(0, 1 - attenuation^2))",
                "spectralFeatureDim": SPECTRAL_FEATURE_DIM,
                "spectralFeatureOrder": "centerView_then_frequency_major_triplets",
            },
            "query": {
                "lowFrequencyBody": "centerView_plus_low_frequency_triplets",
                "highFrequencyBody": "high_frequency_triplets_plus_mean_max_unresolved_energy",
                "highFrequencyGate": "continuous_sigmoid_conditioned_on_relation_and_uncertainty",
                "queryBasisDim": self.query_basis_dim,
                "maxQueryBasisDim": MAX_QUERY_BASIS_DIM,
                "relationConditionDim": self.relation_condition_dim,
            },
            "relationSurvivalField": {
                "supported": True,
                "coefficientShape": [self.query_basis_dim, 7],
                "rhoDim": 1,
                "semanticDim": 8,
                "parameterization": "monotone_two_logistic_survival",
            },
            "runtime": {
                "oneCenterQueryPerCandidateBatch": True,
                "subposeExpansion": False,
                "onlineRelationPropagation": False,
                "onlineNeighborSearch": False,
            },
        }

    def _prepare_inputs(
        self,
        center_view: torch.Tensor,
        diagonal_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = _as_batched_float(center_view, VIEW_DIM, "center_view")
        variance = _as_batched_float(diagonal_variance, VIEW_DIM, "diagonal_variance")
        if center.shape[0] != variance.shape[0]:
            raise ValueError("center_view and diagonal_variance must have the same batch size")
        if bool((variance < 0.0).any()):
            raise ValueError("diagonal_variance must be non-negative")
        return torch.clamp(center, min=-1e4, max=1e4), torch.clamp(variance, min=0.0, max=1e4)

    def _prepare_relation_condition(
        self,
        relation_condition: torch.Tensor | None,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if relation_condition is None:
            return torch.zeros(
                (batch, self.relation_condition_dim), device=device, dtype=torch.float32
            )
        condition = _as_batched_float(
            relation_condition, self.relation_condition_dim, "relation_condition"
        ).to(device=device)
        if condition.shape[0] != batch:
            raise ValueError("relation_condition and center_view must have the same batch size")
        return torch.clamp(condition, min=-1e4, max=1e4)

    def encode(
        self,
        center_view: torch.Tensor,
        diagonal_variance: torch.Tensor,
    ) -> IntegratedSpectralQueryResult:
        """Compute the analytic 57-value center-plus-integrated spectrum."""

        center, variance = self._prepare_inputs(center_view, diagonal_variance)
        frequencies = self.bounded_frequency_vectors().to(device=center.device, dtype=torch.float32)
        phase = 2.0 * math.pi * (center @ frequencies.T)
        phase = torch.clamp(phase, min=-1e6, max=1e6)
        quadratic = torch.einsum("kd,bd,kd->bk", frequencies, variance, frequencies)
        attenuation = torch.exp(torch.clamp(-0.5 * quadratic, min=-80.0, max=0.0))
        integrated_sin = attenuation * torch.sin(phase)
        integrated_cos = attenuation * torch.cos(phase)
        # ``sqrt(1 - attenuation**2)`` is exactly zero for a point query,
        # where its raw derivative is singular.  Subtracting the same small
        # floor after the square root preserves the exact zero limit while
        # giving learnable frequency vectors a finite gradient at zero
        # view-cell variance.
        unresolved_floor = 1e-4
        unresolved = (
            torch.sqrt(torch.clamp(1.0 - attenuation.square(), min=0.0, max=1.0) + unresolved_floor**2)
            - unresolved_floor
        )
        unresolved = torch.clamp(unresolved, min=0.0, max=1.0)
        integrated_sin_cos = torch.stack([integrated_sin, integrated_cos], dim=-1)
        per_frequency = torch.cat([integrated_sin_cos, unresolved.unsqueeze(-1)], dim=-1)
        low = per_frequency[:, :LOW_FREQUENCY_COUNT].reshape(center.shape[0], -1)
        middle_start = LOW_FREQUENCY_COUNT
        middle_end = middle_start + MIDDLE_FREQUENCY_COUNT
        middle = per_frequency[:, middle_start:middle_end].reshape(center.shape[0], -1)
        high = per_frequency[:, middle_end:].reshape(center.shape[0], -1)
        high_energy = unresolved[:, middle_end:]
        high_energy_summary = torch.cat([
            high_energy.mean(dim=-1, keepdim=True),
            high_energy.max(dim=-1, keepdim=True).values,
        ], dim=-1)
        spectral_features = torch.cat([center, per_frequency.reshape(center.shape[0], -1)], dim=-1)
        return IntegratedSpectralQueryResult(
            center_view=center,
            diagonal_variance=variance,
            frequencies=frequencies,
            phase=phase,
            attenuation=attenuation,
            integrated_sin_cos=integrated_sin_cos,
            unresolved_energy=unresolved,
            per_frequency=per_frequency,
            low_frequency=low,
            middle_frequency=middle,
            high_frequency=high,
            high_frequency_energy=high_energy_summary,
            high_frequency_gate=torch.ones((center.shape[0], 1), device=center.device),
            spectral_features=spectral_features,
        )

    def center_query(self, center_view: torch.Tensor) -> IntegratedSpectralQueryResult:
        """Evaluate the exact center limit, with no region uncertainty."""

        center = _as_batched_float(center_view, VIEW_DIM, "center_view")
        return self.forward(center, torch.zeros_like(center))

    def forward(
        self,
        center_view: torch.Tensor,
        diagonal_variance: torch.Tensor,
        relation_condition: torch.Tensor | None = None,
        relation_survival_coefficients: torch.Tensor | None = None,
        rho: torch.Tensor | None = None,
    ) -> IntegratedSpectralQueryResult:
        encoded = self.encode(center_view, diagonal_variance)
        relation = self._prepare_relation_condition(
            relation_condition,
            encoded.center_view.shape[0],
            encoded.center_view.device,
        )
        variance_summary = torch.cat([
            encoded.diagonal_variance.mean(dim=-1, keepdim=True),
            encoded.diagonal_variance.max(dim=-1, keepdim=True).values,
        ], dim=-1)
        gate_input = torch.cat([relation, variance_summary, encoded.high_frequency_energy], dim=-1)
        gate = torch.sigmoid(self.high_frequency_gate(gate_input))
        low_hidden = self.low_frequency_body(
            torch.cat([encoded.center_view, encoded.low_frequency, encoded.middle_frequency], dim=-1)
        )
        high_hidden = self.high_frequency_body(
            torch.cat([encoded.high_frequency, encoded.high_frequency_energy], dim=-1)
        )
        relation_hidden = self.relation_body(relation)
        query_basis = torch.tanh(self.query_fuse(
            torch.cat([low_hidden, gate * high_hidden, relation_hidden], dim=-1)
        ))
        relation_survival = None
        if relation_survival_coefficients is not None:
            if rho is None:
                raise ValueError("rho is required with relation_survival_coefficients")
            relation_survival = self.query_relation_survival(
                query_basis, rho, relation_survival_coefficients
            )
        elif rho is not None:
            raise ValueError("relation_survival_coefficients is required with rho")
        return IntegratedSpectralQueryResult(
            center_view=encoded.center_view,
            diagonal_variance=encoded.diagonal_variance,
            frequencies=encoded.frequencies,
            phase=encoded.phase,
            attenuation=encoded.attenuation,
            integrated_sin_cos=encoded.integrated_sin_cos,
            unresolved_energy=encoded.unresolved_energy,
            per_frequency=encoded.per_frequency,
            low_frequency=encoded.low_frequency,
            middle_frequency=encoded.middle_frequency,
            high_frequency=encoded.high_frequency,
            high_frequency_energy=encoded.high_frequency_energy,
            high_frequency_gate=gate,
            spectral_features=encoded.spectral_features,
            query_basis=query_basis,
            relation_condition=relation,
            relation_survival=relation_survival,
        )

    def query_relation_survival(
        self,
        query_basis: torch.Tensor,
        rho: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Read one monotone relation survival field from the query basis.

        ``coefficients`` are the offline relation encoder's ``[B, R, 7]``
        values.  This method evaluates one ``rho`` per candidate and returns
        the same eight semantic values used by the existing visibility head.
        It is a conditional field read, not graph propagation.
        """

        basis = _as_batched_float(query_basis, self.query_basis_dim, "query_basis")
        coefficient_tensor = coefficients if isinstance(coefficients, torch.Tensor) else torch.as_tensor(coefficients)
        if coefficient_tensor.ndim == 2 and tuple(coefficient_tensor.shape) == (self.query_basis_dim, 7):
            coefficient_tensor = coefficient_tensor.unsqueeze(0).expand(basis.shape[0], -1, -1)
        if coefficient_tensor.ndim != 3 or tuple(coefficient_tensor.shape[1:]) != (self.query_basis_dim, 7):
            raise ValueError(
                f"coefficients must have shape [B, {self.query_basis_dim}, 7] or "
                f"[{self.query_basis_dim}, 7]"
            )
        if coefficient_tensor.shape[0] != basis.shape[0]:
            raise ValueError("coefficients and query_basis must have the same batch size")
        coefficient_tensor = coefficient_tensor.to(device=basis.device, dtype=torch.float32)
        if not bool(torch.isfinite(coefficient_tensor).all()):
            raise ValueError("coefficients contains non-finite values")
        coefficient_tensor = torch.clamp(coefficient_tensor, min=-1e4, max=1e4)
        rho_tensor = _broadcast_batch_scalar(rho, basis.shape[0], "rho", basis.device)
        rho_tensor = torch.clamp(rho_tensor, min=0.0, max=1.0)

        params = torch.einsum("br,brp->bp", basis, coefficient_tensor)
        no_block = torch.sigmoid(params[:, 0:1])
        weights = torch.softmax(params[:, 1:3], dim=-1)
        depths = 0.05 + 0.90 * torch.sigmoid(params[:, 3:5])
        scales = 0.03 + F.softplus(params[:, 5:7])
        depths, order = torch.sort(depths, dim=-1)
        weights = torch.gather(weights, dim=-1, index=order)
        scales = torch.gather(scales, dim=-1, index=order)
        cdf_terms = torch.sigmoid((rho_tensor - depths) / torch.clamp(scales, min=1e-3))
        cdf = (1.0 - no_block) * (weights * cdf_terms).sum(dim=-1, keepdim=True)
        survival = torch.clamp(1.0 - cdf, min=1e-5, max=1.0)
        expected_depth = (weights * depths).sum(dim=-1, keepdim=True)
        variance = (weights * (depths - expected_depth).square()).sum(dim=-1, keepdim=True)
        uncertainty = torch.sqrt(torch.clamp(variance, min=1e-8))
        local_slope = (1.0 - no_block) * (
            weights
            / torch.clamp(scales, min=1e-3)
            * cdf_terms
            * (1.0 - cdf_terms)
        ).sum(dim=-1, keepdim=True)
        semantic = torch.cat([
            survival,
            1.0 - survival,
            no_block,
            expected_depth,
            uncertainty,
            weights,
            local_slope,
        ], dim=-1)
        if not bool(torch.isfinite(semantic).all()):
            raise FloatingPointError("relation survival query produced non-finite values")
        return semantic


# Concise aliases for importers that use the shorter query name.
IntegratedSpectralQuery = ViewCellIntegratedSpectralQuery


__all__ = [
    "DEFAULT_ASPECT_RATIO",
    "DEFAULT_MODEL_FOV_Y_DEG",
    "DEFAULT_QUERY_BASIS_DIM",
    "DEFAULT_RELATION_CONDITION_DIM",
    "DEFAULT_VIEWCELL_RADIUS_M",
    "FREQUENCY_COUNT",
    "HIGH_FREQUENCY_COUNT",
    "INTEGRATED_COMPONENTS_PER_FREQUENCY",
    "IntegratedSpectralQuery",
    "IntegratedSpectralQueryResult",
    "LOW_FREQUENCY_COUNT",
    "MAX_QUERY_BASIS_DIM",
    "MIDDLE_FREQUENCY_COUNT",
    "SPECTRAL_FEATURE_DIM",
    "VIEWCELL_INTEGRATED_SPECTRAL_SCHEMA",
    "VIEW_DIM",
    "ViewCellIntegratedSpectralQuery",
    "analytic_viewcell_diagonal_variance",
    "analytic_viewcell_uncertainty",
    "build_viewcell_query",
]
