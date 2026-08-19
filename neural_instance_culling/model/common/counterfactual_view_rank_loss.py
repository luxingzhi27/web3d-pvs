"""Visual-mass weighted counterfactual ranking across views of one instance."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _segment_soft_extreme(
    values: torch.Tensor,
    groups: torch.Tensor,
    group_count: int,
    *,
    largest: bool,
    temperature: float,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable weighted log-mean-exp extrema and group row counts."""
    if values.numel() == 0:
        return (
            values.new_zeros((group_count,)),
            torch.zeros((group_count,), dtype=torch.long, device=values.device),
        )
    sign = 1.0 if largest else -1.0
    scaled = sign * values / float(temperature)
    maximum = values.new_full((group_count,), float("-inf"))
    maximum.scatter_reduce_(
        0, groups, scaled, reduce="amax", include_self=True
    )
    row_weights = (
        torch.ones_like(values)
        if weights is None
        else weights.to(device=values.device, dtype=values.dtype).clamp_min(1e-12)
    )
    stable = row_weights * torch.exp(scaled - maximum[groups])
    numerator = values.new_zeros((group_count,))
    denominator = values.new_zeros((group_count,))
    numerator.scatter_add_(0, groups, stable)
    denominator.scatter_add_(0, groups, row_weights)
    counts = torch.zeros((group_count,), dtype=torch.long, device=values.device)
    counts.scatter_add_(0, groups, torch.ones_like(groups, dtype=torch.long))
    valid = counts > 0
    log_mean_exp = values.new_zeros((group_count,))
    log_mean_exp[valid] = maximum[valid] + torch.log(
        numerator[valid] / denominator[valid].clamp_min(1e-12)
    )
    return sign * float(temperature) * log_mean_exp, counts


def counterfactual_view_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    instance_ids: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    recurrence_priority: torch.Tensor | None = None,
    margin: float = 0.50,
    temperature: float = 0.25,
    positive_weight_power: float = 0.50,
    positive_importance_mix: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
    """Separate opposite labels for the same instance without static shortcuts.

    A visual-mass weighted soft minimum summarizes positive views, while a
    soft maximum summarizes negative views.  A fixed per-instance score bias
    appears in both anchors and therefore cancels from their gap.
    """
    numeric = (
        float(margin),
        float(temperature),
        float(positive_weight_power),
        float(positive_importance_mix),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("counterfactual rank parameters must be finite")
    if (
        float(margin) < 0.0
        or float(temperature) <= 0.0
        or not 0.0 < float(positive_weight_power) <= 1.0
        or not 0.0 <= float(positive_importance_mix) <= 1.0
    ):
        raise ValueError("counterfactual rank parameters are invalid")

    source_logits = torch.as_tensor(logits).float().reshape(-1)
    labels = torch.as_tensor(target, device=source_logits.device).float().reshape(-1)
    ids = torch.as_tensor(instance_ids, device=source_logits.device).long().reshape(-1)
    weights = torch.as_tensor(
        positive_weights, device=source_logits.device
    ).float().reshape(-1)
    if source_logits.numel() == 0 or not (
        source_logits.numel() == labels.numel() == ids.numel() == weights.numel()
    ):
        raise ValueError("counterfactual rank inputs must be non-empty and aligned")
    if bool((ids < 0).any()):
        raise ValueError("counterfactual rank instance IDs must be non-negative")
    if not bool(
        torch.isfinite(source_logits).all()
        and torch.isfinite(labels).all()
        and torch.isfinite(weights).all()
    ) or bool((weights < 0.0).any()):
        raise ValueError("counterfactual rank inputs must be finite and non-negative")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("counterfactual rank targets must be binary")

    priority = None
    if recurrence_priority is not None:
        priority = torch.as_tensor(
            recurrence_priority, device=source_logits.device
        ).float().reshape(-1)
        if priority.numel() != source_logits.numel():
            raise ValueError("counterfactual recurrence priority must align with logits")
        if not bool(torch.isfinite(priority).all()) or bool((priority < 0.0).any()):
            raise ValueError("counterfactual recurrence priority must be non-negative")
        active = priority > 0.0
        if not bool(active.any()):
            zero = source_logits.sum() * 0.0
            return zero, zero, {
                "lossCounterfactualViewPositive": zero,
                "lossCounterfactualViewNegative": zero,
                "counterfactualViewGap": zero.detach(),
                "counterfactualViewWorstGap": zero.detach(),
                "counterfactualViewViolationFraction": zero.detach(),
                "counterfactualViewPairCount": 0.0,
                "counterfactualViewPositiveRows": 0.0,
                "counterfactualViewNegativeRows": 0.0,
            }
        source_logits = source_logits[active]
        labels = labels[active]
        ids = ids[active]
        weights = weights[active]
        priority = priority[active]

    _, inverse = torch.unique(ids, sorted=False, return_inverse=True)
    group_count = int(inverse.max()) + 1
    positive = labels > 0.5
    negative = ~positive
    positive_mass = weights[positive].clamp_min(1e-12).pow(
        float(positive_weight_power)
    )
    positive_anchor, positive_count = _segment_soft_extreme(
        source_logits[positive],
        inverse[positive],
        group_count,
        largest=False,
        temperature=float(temperature),
        weights=positive_mass,
    )
    negative_anchor, negative_count = _segment_soft_extreme(
        source_logits[negative],
        inverse[negative],
        group_count,
        largest=True,
        temperature=float(temperature),
    )
    paired = (positive_count > 0) & (negative_count > 0)
    zero = source_logits.sum() * 0.0
    if not bool(paired.any()):
        return zero, zero, {
            "lossCounterfactualViewPositive": zero,
            "lossCounterfactualViewNegative": zero,
            "counterfactualViewGap": zero.detach(),
            "counterfactualViewWorstGap": zero.detach(),
            "counterfactualViewViolationFraction": zero.detach(),
            "counterfactualViewPairCount": 0.0,
            "counterfactualViewPositiveRows": float(positive.sum()),
            "counterfactualViewNegativeRows": float(negative.sum()),
        }

    group_importance = source_logits.new_zeros((group_count,))
    group_importance.scatter_reduce_(
        0,
        inverse[positive],
        torch.log1p(weights[positive]),
        reduce="amax",
        include_self=True,
    )
    group_importance = group_importance[paired]
    if bool((group_importance > 0.0).any()):
        group_importance = group_importance / group_importance.mean().clamp_min(1e-6)
    else:
        group_importance = torch.ones_like(group_importance)
    group_weights = (
        (1.0 - float(positive_importance_mix))
        + float(positive_importance_mix) * group_importance
    )
    if priority is not None:
        group_priority = source_logits.new_zeros((group_count,))
        group_priority.scatter_reduce_(
            0, inverse, priority, reduce="amax", include_self=True
        )
        paired_priority = group_priority[paired]
        group_weights = group_weights * (
            paired_priority / paired_priority.mean().clamp_min(1e-6)
        )
    group_weights = group_weights / group_weights.sum().clamp_min(1e-6)

    positive_tail = positive_anchor[paired]
    negative_tail = negative_anchor[paired]
    positive_violation = (
        negative_tail.detach() - positive_tail + float(margin)
    ) / float(temperature)
    negative_violation = (
        negative_tail - positive_tail.detach() + float(margin)
    ) / float(temperature)
    positive_loss = (
        group_weights * F.softplus(positive_violation) * float(temperature)
    ).sum()
    negative_loss = (
        group_weights * F.softplus(negative_violation) * float(temperature)
    ).sum()
    gap = positive_tail.detach() - negative_tail.detach()
    diagnostics: dict[str, Any] = {
        "lossCounterfactualViewPositive": positive_loss,
        "lossCounterfactualViewNegative": negative_loss,
        "counterfactualViewGap": (group_weights * gap).sum(),
        "counterfactualViewWorstGap": gap.min(),
        "counterfactualViewViolationFraction": (gap < float(margin)).float().mean(),
        "counterfactualViewPairCount": float(paired.sum()),
        "counterfactualViewPositiveRows": float(positive.sum()),
        "counterfactualViewNegativeRows": float(negative.sum()),
    }
    if not bool(
        torch.isfinite(
            torch.stack(
                [
                    positive_loss,
                    negative_loss,
                    diagnostics["counterfactualViewGap"],
                    diagnostics["counterfactualViewWorstGap"],
                ]
            )
        ).all()
    ):
        raise FloatingPointError("counterfactual view ranking became non-finite")
    return positive_loss, negative_loss, diagnostics


__all__ = ["counterfactual_view_rank_loss"]
