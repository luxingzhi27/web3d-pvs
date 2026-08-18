"""Pose-local two-sided Huber deficits for a frozen visibility boundary.

The frozen logits define which positives and negatives are difficult for each
pose.  The trainable ``applied_residual`` is then asked to move the selected
positive and negative logits to opposite sides of a detached, smoothed
boundary.  Selection and the boundary are deliberately detached so this
objective cannot change its mining policy through the residual itself.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _flat_float(value: Any, name: str, device: torch.device) -> torch.Tensor:
    """Convert an aligned vector or singleton-column vector to float32."""
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc
    if tensor.ndim == 0 or tensor.ndim > 2 or (
        tensor.ndim == 2 and 1 not in tensor.shape
    ):
        raise ValueError(f"{name} must be a vector or singleton-column vector")
    if torch.is_complex(tensor):
        raise ValueError(f"{name} must be real-valued")
    return tensor.to(device=device, dtype=torch.float32).reshape(-1)


def _pose_offsets(value: Any, item_count: int) -> list[int]:
    """Validate and materialize contiguous pose boundaries on the host."""
    try:
        offsets = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("pose_offsets must be tensor-like") from exc
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("pose_offsets must be a one-dimensional boundary vector")
    if torch.is_complex(offsets) or offsets.dtype == torch.bool:
        raise ValueError("pose_offsets must contain integer offsets")
    if not bool(torch.isfinite(offsets).all()):
        raise ValueError("pose_offsets must be finite")
    if torch.is_floating_point(offsets):
        rounded = offsets.round()
        if not bool(torch.equal(offsets, rounded)):
            raise ValueError("pose_offsets must contain integer offsets")
        offsets = rounded

    values = [int(item) for item in offsets.detach().cpu().tolist()]
    if values[0] != 0 or values[-1] != int(item_count):
        raise ValueError("pose_offsets must start at zero and end at the candidate count")
    if any(
        start < 0 or end < start or end > int(item_count)
        for start, end in zip(values, values[1:])
    ):
        raise ValueError("pose_offsets must be non-decreasing within the candidate range")
    return values


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    try:
        exact = float(value) == float(result)
    except (TypeError, ValueError, OverflowError):
        exact = False
    if result < 0 or not exact:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = _non_negative_integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _weighted_lower_tail(
    frozen_base_logits: torch.Tensor,
    positive_indices: torch.Tensor,
    visible_weights: torch.Tensor,
    *,
    mass_fraction: float,
    count_cap: int,
    min_count: int,
) -> torch.Tensor:
    """Select the lowest frozen-base positives up to a weighted mass budget."""
    if positive_indices.numel() == 0:
        return positive_indices

    order = torch.argsort(frozen_base_logits[positive_indices].detach(), stable=True)
    ordered_indices = positive_indices[order]
    ordered_weights = visible_weights[ordered_indices].detach().to(torch.float64)
    ordered_weights = ordered_weights.clamp_min(0.0)
    if not bool(ordered_weights.sum() > 0.0):
        ordered_weights = torch.ones_like(ordered_weights)

    if mass_fraction >= 1.0:
        mass_count = int(ordered_indices.numel())
    else:
        cumulative = torch.cumsum(ordered_weights, dim=0)
        target_mass = cumulative[-1] * float(mass_fraction)
        mass_count = int(
            torch.searchsorted(cumulative, target_mass, right=False).item()
        ) + 1
        mass_count = max(1, min(mass_count, int(ordered_indices.numel())))

    selected_count = min(
        int(ordered_indices.numel()),
        int(count_cap),
        max(int(min_count), int(mass_count)),
    )
    return ordered_indices[:selected_count]


def _negative_upper_tail(
    frozen_base_logits: torch.Tensor,
    negative_indices: torch.Tensor,
    *,
    fraction: float,
    count_cap: int,
    min_count: int,
) -> torch.Tensor:
    """Select the highest frozen-base negatives under count constraints."""
    if negative_indices.numel() == 0:
        return negative_indices
    count = min(
        int(negative_indices.numel()),
        int(count_cap),
        max(int(min_count), int(math.ceil(fraction * int(negative_indices.numel())))),
    )
    order = torch.argsort(
        frozen_base_logits[negative_indices].detach(), descending=True, stable=True
    )
    return negative_indices[order[:count]]


def _weighted_softmin(
    values: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute a finite normalized weighted soft minimum."""
    minimum = values.detach().min()
    scaled = ((values.detach() - minimum) / float(temperature)).clamp(0.0, 80.0)
    log_weights = weights.detach().clamp_min(0.0).log()
    log_average = torch.logsumexp(-scaled + log_weights, dim=0) - torch.log(
        weights.detach().sum().clamp_min(1e-12)
    )
    return minimum - float(temperature) * log_average


def _softmax_mean(values: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute a finite uniform soft maximum in logit units."""
    maximum = values.detach().max()
    scaled = ((values.detach() - maximum) / float(temperature)).clamp(-80.0, 0.0)
    log_average = torch.logsumexp(scaled, dim=0) - math.log(int(values.numel()))
    return maximum + float(temperature) * log_average


def _mean_or_zero(values: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else zero


def viewcell_tail_boundary_deficit_loss(
    applied_residual: torch.Tensor,
    frozen_base_logits: torch.Tensor,
    target: torch.Tensor,
    visible_hit_rates: torch.Tensor,
    visible_weights: torch.Tensor,
    pose_offsets: torch.Tensor,
    *,
    positive_mass_fraction: float = 0.005,
    positive_count_cap: int = 64,
    min_positive_count: int = 1,
    negative_fraction: float = 0.04,
    negative_count_cap: int = 256,
    min_negative_count: int = 1,
    margin: float = 0.2,
    temperature: float = 0.25,
    huber_beta: float = 0.1,
    gap_weight: float = 1.0,
    residual_l2_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return a pose-balanced, two-sided Huber boundary-deficit loss.

    For every pose containing both classes, positives are selected from the
    lower frozen-base tail until ``positive_mass_fraction`` of their visible
    weight is covered, and negatives are selected from the upper frozen-base
    tail.  The selected positive importance is
    ``log1p(visible_weight) * (2 - visible_hit_rate)``.  A detached boundary
    is the midpoint of the weighted positive soft minimum and uniform
    negative soft maximum.  Final positive logits are pushed above
    ``boundary + margin / 2`` and final negative logits below
    ``boundary - margin / 2``.

    Positive and negative deficits are averaged separately within each pose
    before the two sides are combined, so a large negative class cannot
    drown out the positive safety term.  Poses without both classes do not
    contribute, including to residual regularization; in that case the
    returned zero remains connected to ``applied_residual`` for backward.
    """
    if isinstance(applied_residual, torch.Tensor):
        residual = _flat_float(
            applied_residual, "applied_residual", applied_residual.device
        )
    else:
        residual = _flat_float(
            applied_residual, "applied_residual", torch.device("cpu")
        )
    if residual.numel() == 0:
        raise ValueError("loss requires at least one candidate")
    device = residual.device

    base = _flat_float(frozen_base_logits, "frozen_base_logits", device).detach()
    labels = _flat_float(target, "target", device)
    hit_rates = _flat_float(visible_hit_rates, "visible_hit_rates", device)
    weights = _flat_float(visible_weights, "visible_weights", device)
    count = int(residual.numel())
    tensors = {
        "applied_residual": residual,
        "frozen_base_logits": base,
        "target": labels,
        "visible_hit_rates": hit_rates,
        "visible_weights": weights,
    }
    for name, tensor in tensors.items():
        if tensor.numel() != count:
            raise ValueError(f"{name} must align with applied_residual")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("target must contain only binary 0/1 values")
    if bool((hit_rates < 0.0).any() or (hit_rates > 1.0).any()):
        raise ValueError("visible_hit_rates must lie in [0, 1]")
    if bool((weights < 0.0).any()):
        raise ValueError("visible_weights must be non-negative")

    positive_fraction = _finite_scalar(positive_mass_fraction, "positive_mass_fraction")
    negative_fraction_value = _finite_scalar(negative_fraction, "negative_fraction")
    if not 0.0 < positive_fraction <= 1.0:
        raise ValueError("positive_mass_fraction must lie in (0, 1]")
    if not 0.0 < negative_fraction_value <= 1.0:
        raise ValueError("negative_fraction must lie in (0, 1]")

    positive_cap = _positive_integer(positive_count_cap, "positive_count_cap")
    negative_cap = _positive_integer(negative_count_cap, "negative_count_cap")
    minimum_positive = _non_negative_integer(min_positive_count, "min_positive_count")
    minimum_negative = _non_negative_integer(min_negative_count, "min_negative_count")
    if minimum_positive > positive_cap:
        raise ValueError("min_positive_count cannot exceed positive_count_cap")
    if minimum_negative > negative_cap:
        raise ValueError("min_negative_count cannot exceed negative_count_cap")

    margin_value = _non_negative_scalar(margin, "margin")
    temperature_value = _finite_scalar(temperature, "temperature")
    if temperature_value <= 0.0:
        raise ValueError("temperature must be positive")
    huber_beta_value = _finite_scalar(huber_beta, "huber_beta")
    if huber_beta_value <= 0.0:
        raise ValueError("huber_beta must be positive")
    gap_weight_value = _non_negative_scalar(gap_weight, "gap_weight")
    residual_l2_weight_value = _non_negative_scalar(
        residual_l2_weight, "residual_l2_weight"
    )
    offsets = _pose_offsets(pose_offsets, count)

    final_logits = base + residual
    if not bool(torch.isfinite(final_logits).all()):
        raise ValueError("frozen_base_logits plus applied_residual must be finite")

    zero = residual.sum() * 0.0
    positive_losses: list[torch.Tensor] = []
    negative_losses: list[torch.Tensor] = []
    gap_losses: list[torch.Tensor] = []
    boundary_values: list[torch.Tensor] = []
    positive_softmin_values: list[torch.Tensor] = []
    negative_softmax_values: list[torch.Tensor] = []
    positive_final_means: list[torch.Tensor] = []
    negative_final_means: list[torch.Tensor] = []
    mean_gaps: list[torch.Tensor] = []
    positive_importance_means: list[torch.Tensor] = []
    positive_deficit_means: list[torch.Tensor] = []
    negative_deficit_means: list[torch.Tensor] = []
    selected_positive_count = 0
    selected_negative_count = 0

    for start, end in zip(offsets, offsets[1:]):
        if end <= start:
            continue
        local_target = labels[start:end]
        local_base = base[start:end]
        local_final = final_logits[start:end]
        local_weights = weights[start:end]
        local_hit_rates = hit_rates[start:end]

        positive_indices = torch.nonzero(
            local_target == 1.0, as_tuple=False
        ).reshape(-1)
        negative_indices = torch.nonzero(
            local_target == 0.0, as_tuple=False
        ).reshape(-1)
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        selected_positive = _weighted_lower_tail(
            local_base,
            positive_indices,
            local_weights,
            mass_fraction=positive_fraction,
            count_cap=positive_cap,
            min_count=minimum_positive,
        )
        selected_negative = _negative_upper_tail(
            local_base,
            negative_indices,
            fraction=negative_fraction_value,
            count_cap=negative_cap,
            min_count=minimum_negative,
        )
        if selected_positive.numel() == 0 or selected_negative.numel() == 0:
            continue

        positive_importance = torch.log1p(
            local_weights[selected_positive].detach().clamp_min(0.0)
        ) * (2.0 - local_hit_rates[selected_positive].detach())
        positive_importance = positive_importance.clamp_min(0.0)
        if not bool(positive_importance.sum() > 0.0):
            positive_importance = torch.ones_like(positive_importance)
        positive_importance_sum = positive_importance.sum().clamp_min(1e-12)

        positive_softmin = _weighted_softmin(
            local_base[selected_positive],
            positive_importance,
            temperature_value,
        ).detach()
        negative_softmax = _softmax_mean(
            local_base[selected_negative], temperature_value
        ).detach()
        boundary = (positive_softmin + negative_softmax) * 0.5

        positive_final = local_final[selected_positive]
        negative_final = local_final[selected_negative]
        positive_deficit = torch.relu(
            boundary + 0.5 * margin_value - positive_final
        ).clamp_min(0.0)
        negative_deficit = torch.relu(
            negative_final - (boundary - 0.5 * margin_value)
        ).clamp_min(0.0)
        positive_huber = F.smooth_l1_loss(
            positive_deficit,
            torch.zeros_like(positive_deficit),
            beta=huber_beta_value,
            reduction="none",
        )
        negative_huber = F.smooth_l1_loss(
            negative_deficit,
            torch.zeros_like(negative_deficit),
            beta=huber_beta_value,
            reduction="none",
        )
        positive_loss = (
            positive_huber * positive_importance
        ).sum() / positive_importance_sum
        negative_loss = negative_huber.mean()

        positive_final_mean = (
            positive_final * positive_importance
        ).sum() / positive_importance_sum
        negative_final_mean = negative_final.mean()
        gap_argument = (
            negative_final_mean - positive_final_mean + margin_value
        ) / temperature_value
        gap_loss = temperature_value * F.softplus(gap_argument)

        positive_losses.append(positive_loss)
        negative_losses.append(negative_loss)
        gap_losses.append(gap_loss)
        boundary_values.append(boundary)
        positive_softmin_values.append(positive_softmin)
        negative_softmax_values.append(negative_softmax)
        positive_final_means.append(positive_final_mean.detach())
        negative_final_means.append(negative_final_mean.detach())
        mean_gaps.append((positive_final_mean - negative_final_mean).detach())
        positive_importance_means.append(positive_importance.detach().mean())
        positive_deficit_means.append(positive_deficit.detach().mean())
        negative_deficit_means.append(negative_deficit.detach().mean())
        selected_positive_count += int(selected_positive.numel())
        selected_negative_count += int(selected_negative.numel())

    if positive_losses:
        loss_positive_deficit = torch.stack(positive_losses).mean()
        loss_negative_deficit = torch.stack(negative_losses).mean()
        loss_tail_boundary_deficit = 0.5 * (
            loss_positive_deficit + loss_negative_deficit
        )
        loss_gap = torch.stack(gap_losses).mean()
        loss_residual_l2 = residual.square().mean()
        total = (
            loss_tail_boundary_deficit
            + gap_weight_value * loss_gap
            + residual_l2_weight_value * loss_residual_l2
        )
    else:
        loss_positive_deficit = zero
        loss_negative_deficit = zero
        loss_tail_boundary_deficit = zero
        loss_gap = zero
        loss_residual_l2 = zero
        total = zero

    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("view-cell tail boundary deficit loss is non-finite")

    stats: dict[str, torch.Tensor | float] = {
        "lossPositiveDeficit": loss_positive_deficit,
        "lossNegativeDeficit": loss_negative_deficit,
        "lossTailBoundaryDeficit": loss_tail_boundary_deficit,
        "lossBoundaryDeficit": loss_tail_boundary_deficit,
        "lossGap": loss_gap,
        "lossResidualL2": loss_residual_l2,
        "lossViewcellTailBoundaryDeficit": total,
        "tailBoundaryLogit": _mean_or_zero(boundary_values, zero.detach()),
        "boundaryLogit": _mean_or_zero(boundary_values, zero.detach()),
        "positiveSoftminLogit": _mean_or_zero(
            positive_softmin_values, zero.detach()
        ),
        "negativeSoftmaxLogit": _mean_or_zero(
            negative_softmax_values, zero.detach()
        ),
        "positiveFinalMeanLogit": _mean_or_zero(
            positive_final_means, zero.detach()
        ),
        "negativeFinalMeanLogit": _mean_or_zero(
            negative_final_means, zero.detach()
        ),
        "meanFinalGap": _mean_or_zero(mean_gaps, zero.detach()),
        "positiveImportanceMean": _mean_or_zero(
            positive_importance_means, zero.detach()
        ),
        "positiveDeficitMean": _mean_or_zero(
            positive_deficit_means, zero.detach()
        ),
        "negativeDeficitMean": _mean_or_zero(
            negative_deficit_means, zero.detach()
        ),
        "selectedPositiveCount": float(selected_positive_count),
        "selectedNegativeCount": float(selected_negative_count),
        "poseCount": float(len(positive_losses)),
        "contributingPoseCount": float(len(positive_losses)),
    }
    return total, stats


__all__ = ["viewcell_tail_boundary_deficit_loss"]
