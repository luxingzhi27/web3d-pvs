"""Tail-rescue objective for conservative view-cell visibility prediction.

The frozen base model already ranks ordinary candidates well.  Its failure is
localized at the safe operating boundary: a small weighted tail of visible
instances lies below the high-score tail of invisible candidates.  This loss
therefore does not train a second visibility probability.  It trains a
non-negative, bounded logit uplift and asks only the frozen-base positive tail
to clear the same-pose negative quantile boundary.  Every negative candidate
is simultaneously charged for receiving uplift, with an extra charge on the
hard negative tail.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _flat_float(value: Any, name: str, device: torch.device) -> torch.Tensor:
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc
    if torch.is_complex(tensor):
        raise ValueError(f"{name} must be real-valued")
    return tensor.to(device=device, dtype=torch.float32).reshape(-1)


def _pose_offsets(value: Any, item_count: int) -> list[int]:
    try:
        offsets = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("pose_offsets must be tensor-like") from exc
    if offsets.ndim != 1 or offsets.numel() == 0:
        raise ValueError("pose_offsets must be a non-empty one-dimensional tensor")
    if torch.is_complex(offsets) or offsets.dtype == torch.bool:
        raise ValueError("pose_offsets must contain integer offsets")
    if torch.is_floating_point(offsets):
        if not bool(torch.isfinite(offsets).all()):
            raise ValueError("pose_offsets must be finite")
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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or float(value) != float(result):
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_importance_weights(
    weights: torch.Tensor,
    hit_rates: torch.Tensor,
    rare_threshold: float,
) -> torch.Tensor:
    """Keep weighted-recall semantics while emphasizing boundary-only hits."""
    transformed = torch.log1p(weights.detach().clamp_min(0.0))
    mean = transformed.mean() if transformed.numel() else transformed.new_zeros(())
    if not bool(mean > 0.0):
        transformed = torch.ones_like(transformed)
    else:
        transformed = 1.0 + transformed / mean.clamp_min(1e-12)
    boundary_bonus = 1.0 + (
        (hit_rates.detach() > 0.0)
        & (hit_rates.detach() <= float(rare_threshold))
    ).to(dtype=transformed.dtype)
    return transformed * boundary_bonus


def _weighted_low_tail(
    logits: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    mass_fraction: float,
    count_cap_fraction: float,
    min_count: int,
) -> torch.Tensor:
    """Select the frozen-score lower tail until its positive weight budget is met."""
    order = torch.argsort(logits[indices].detach(), descending=False)
    ordered_indices = indices[order]
    ordered_weights = weights[ordered_indices].detach().clamp_min(0.0)
    if not bool(ordered_weights.sum() > 0.0):
        ordered_weights = torch.ones_like(ordered_weights)
    cumulative_before = torch.cumsum(ordered_weights, dim=0) - ordered_weights
    mass_limit = float(mass_fraction) * ordered_weights.sum()
    mass_count = int((cumulative_before < mass_limit).sum().item())
    count_cap = max(
        int(min_count),
        int(math.ceil(float(count_cap_fraction) * int(ordered_indices.numel()))),
    )
    selected_count = min(
        int(ordered_indices.numel()),
        int(count_cap),
        max(int(min_count), int(mass_count)),
    )
    return ordered_indices[:selected_count]


def viewcell_boundary_opportunity_loss(
    raw_logit_uplift: torch.Tensor,
    frozen_base_logits: torch.Tensor,
    target: torch.Tensor,
    visible_hit_rates: torch.Tensor,
    visible_weights: torch.Tensor,
    pose_offsets: torch.Tensor,
    *,
    rare_threshold: float = 0.05,
    positive_tail_mass_fraction: float = 0.02,
    positive_tail_count_cap_fraction: float = 0.05,
    negative_top_fraction: float = 0.01,
    min_positive_count: int = 2,
    min_negative_count: int = 32,
    positive_margin_weight: float = 1.0,
    all_negative_uplift_weight: float = 1.0,
    hard_negative_uplift_weight: float = 1.0,
    margin: float = 0.25,
    temperature: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Separate the frozen-base positive lower tail from the negative upper tail.

    Hard-positive and hard-negative membership is selected only from detached
    frozen scores.  The positive loss uses the lower boundary of the selected
    negative upper tail, which approximates the same-pose q(1-tail_fraction)
    operating boundary.  This avoids the impossible objective of ranking a
    rescued positive above every extreme negative.
    """
    if isinstance(raw_logit_uplift, torch.Tensor):
        if torch.is_complex(raw_logit_uplift):
            raise ValueError("raw_logit_uplift must be real-valued")
        flat_uplift = raw_logit_uplift.to(dtype=torch.float32).reshape(-1)
    else:
        flat_uplift = _flat_float(
            raw_logit_uplift, "raw_logit_uplift", torch.device("cpu")
        )
    device = flat_uplift.device
    flat_base = _flat_float(frozen_base_logits, "frozen_base_logits", device).detach()
    flat_target = _flat_float(target, "target", device)
    flat_hit_rates = _flat_float(visible_hit_rates, "visible_hit_rates", device)
    flat_weights = _flat_float(visible_weights, "visible_weights", device)

    count = flat_uplift.numel()
    for name, tensor in {
        "raw_logit_uplift": flat_uplift,
        "frozen_base_logits": flat_base,
        "target": flat_target,
        "visible_hit_rates": flat_hit_rates,
        "visible_weights": flat_weights,
    }.items():
        if tensor.numel() != count:
            raise ValueError(f"{name} must align with raw_logit_uplift")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")
    if bool((flat_uplift < 0.0).any()):
        raise ValueError("raw_logit_uplift must be non-negative")
    if not bool(((flat_target == 0.0) | (flat_target == 1.0)).all()):
        raise ValueError("target must contain only binary 0/1 values")
    if bool((flat_hit_rates < 0.0).any() or (flat_hit_rates > 1.0).any()):
        raise ValueError("visible_hit_rates must lie in [0, 1]")
    if bool((flat_weights < 0.0).any()):
        raise ValueError("visible_weights must be non-negative")

    rare_limit = _finite_scalar(rare_threshold, "rare_threshold")
    positive_fraction = _finite_scalar(
        positive_tail_mass_fraction, "positive_tail_mass_fraction"
    )
    positive_count_cap = _finite_scalar(
        positive_tail_count_cap_fraction, "positive_tail_count_cap_fraction"
    )
    negative_fraction = _finite_scalar(
        negative_top_fraction, "negative_top_fraction"
    )
    if not 0.0 < rare_limit <= 1.0:
        raise ValueError("rare_threshold must lie in (0, 1]")
    if not 0.0 < positive_fraction <= 1.0:
        raise ValueError("positive_tail_mass_fraction must lie in (0, 1]")
    if not 0.0 < positive_count_cap <= 1.0:
        raise ValueError("positive_tail_count_cap_fraction must lie in (0, 1]")
    if not 0.0 < negative_fraction <= 1.0:
        raise ValueError("negative_top_fraction must lie in (0, 1]")
    minimum_positive = _positive_integer(min_positive_count, "min_positive_count")
    minimum_negative = _positive_integer(min_negative_count, "min_negative_count")
    positive_weight = _non_negative(positive_margin_weight, "positive_margin_weight")
    all_negative_weight = _non_negative(
        all_negative_uplift_weight, "all_negative_uplift_weight"
    )
    hard_negative_weight = _non_negative(
        hard_negative_uplift_weight, "hard_negative_uplift_weight"
    )
    margin_value = _non_negative(margin, "margin")
    temperature_value = _finite_scalar(temperature, "temperature")
    if temperature_value <= 0.0:
        raise ValueError("temperature must be positive")
    offsets = _pose_offsets(pose_offsets, count)

    zero = flat_uplift.sum() * 0.0
    positive_terms: list[torch.Tensor] = []
    all_negative_terms: list[torch.Tensor] = []
    hard_negative_terms: list[torch.Tensor] = []
    gap_values: list[torch.Tensor] = []
    boundary_values: list[torch.Tensor] = []
    positive_uplift_values: list[torch.Tensor] = []
    negative_uplift_values: list[torch.Tensor] = []
    selected_positive_count = 0
    selected_rare_positive_count = 0
    selected_negative_count = 0
    contributing_pose_count = 0

    for start, end in zip(offsets, offsets[1:]):
        local_target = flat_target[start:end]
        local_base = flat_base[start:end]
        local_uplift = flat_uplift[start:end]
        local_weights = flat_weights[start:end]
        local_hit_rates = flat_hit_rates[start:end]
        positive_indices = torch.nonzero(local_target == 1.0, as_tuple=False).flatten()
        negative_indices = torch.nonzero(local_target == 0.0, as_tuple=False).flatten()
        if negative_indices.numel() > 0:
            all_negative_terms.append(local_uplift[negative_indices].square().mean())
            negative_uplift_values.append(local_uplift[negative_indices].detach().mean())
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        hard_positive_indices = _weighted_low_tail(
            local_base,
            local_weights,
            positive_indices,
            mass_fraction=positive_fraction,
            count_cap_fraction=positive_count_cap,
            min_count=minimum_positive,
        )
        negative_count = int(negative_indices.numel())
        selected_count = min(
            negative_count,
            max(minimum_negative, int(math.ceil(negative_fraction * negative_count))),
        )
        negative_order = torch.argsort(
            local_base[negative_indices].detach(), descending=True
        )[:selected_count]
        hard_negative_indices = negative_indices[negative_order]
        negative_boundary = local_base[hard_negative_indices].detach().min()
        rescued_positive_logits = (
            local_base[hard_positive_indices] + local_uplift[hard_positive_indices]
        )
        positive_importance = _positive_importance_weights(
            local_weights[hard_positive_indices],
            local_hit_rates[hard_positive_indices],
            rare_limit,
        )
        margin_argument = (
            negative_boundary
            + margin_value
            - rescued_positive_logits
        ) / temperature_value
        positive_raw = temperature_value * F.softplus(
            margin_argument.clamp(-80.0, 80.0)
        )
        positive_terms.append(
            (positive_raw * positive_importance).sum()
            / positive_importance.sum().clamp_min(1e-12)
        )
        hard_negative_terms.append(
            local_uplift[hard_negative_indices].square().mean()
        )

        selected_positive_count += int(hard_positive_indices.numel())
        selected_rare_positive_count += int(
            (
                (local_hit_rates[hard_positive_indices] > 0.0)
                & (local_hit_rates[hard_positive_indices] <= rare_limit)
            ).sum().item()
        )
        selected_negative_count += int(hard_negative_indices.numel())
        contributing_pose_count += 1
        boundary_values.append(negative_boundary)
        gap_values.append(
            (rescued_positive_logits.detach().min() - negative_boundary).reshape(())
        )
        positive_uplift_values.append(
            local_uplift[hard_positive_indices].detach().mean()
        )

    loss_positive_margin = torch.stack(positive_terms).mean() if positive_terms else zero
    loss_all_negative_uplift = (
        torch.stack(all_negative_terms).mean() if all_negative_terms else zero
    )
    loss_hard_negative_uplift = (
        torch.stack(hard_negative_terms).mean() if hard_negative_terms else zero
    )
    total = (
        positive_weight * loss_positive_margin
        + all_negative_weight * loss_all_negative_uplift
        + hard_negative_weight * loss_hard_negative_uplift
    )
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("view-cell tail-rescue loss is non-finite")

    def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero.detach()

    stats: dict[str, torch.Tensor | float] = {
        "lossOpportunityPositiveTailMargin": loss_positive_margin,
        "lossOpportunityAllNegativeUplift": loss_all_negative_uplift,
        "lossOpportunityHardNegativeUplift": loss_hard_negative_uplift,
        "lossViewcellBoundaryOpportunity": total,
        "opportunityTailGap": mean_or_zero(gap_values),
        "opportunityNegativeBoundaryLogit": mean_or_zero(boundary_values),
        "opportunityPositiveUpliftMean": mean_or_zero(positive_uplift_values),
        "opportunityNegativeUpliftMean": mean_or_zero(negative_uplift_values),
        "tailPositiveCount": float(selected_positive_count),
        "tailRarePositiveCount": float(selected_rare_positive_count),
        "hardNegativeCount": float(selected_negative_count),
        "tailPoseCount": float(contributing_pose_count),
    }
    return total, stats


__all__ = ["viewcell_boundary_opportunity_loss"]
