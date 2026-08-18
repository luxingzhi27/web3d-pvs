"""Training-only safety-reserve loss for low-threshold PVS operation.

The safety objective keeps the registered RVL control and adds a positive
boundary-tail CVaR. Efficiency is optimized only after a detached, warmup-aware
safety-reserve gate opens. No calibration, validation, or test threshold is an
input to this module: operating thresholds are derived solely from the train
seed and optimizer step.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from common.safety_constraint_utility_loss import rvl_strong_v2_visibility_loss


OPERATING_THRESHOLD_RANGE = (1e-4, 0.2)
OPERATING_THRESHOLD_ANCHORS = (0.001, 0.005, 0.02, 0.10)

GradientList = Sequence[torch.Tensor | None]


def _pose_slices(offsets: torch.Tensor, item_count: int):
    # Pose boundaries are metadata, not differentiable inputs.  Keeping one
    # detached CPU copy avoids a scalar device synchronization for every pose
    # when this train-only loss is evaluated on CUDA.
    values = torch.as_tensor(offsets, dtype=torch.long).detach().cpu().reshape(-1)
    if values.numel() < 2:
        raise ValueError("pose_offsets must contain at least a start and end")
    if int(values[0]) != 0 or int(values[-1]) != int(item_count):
        raise ValueError("pose_offsets must start at zero and cover every item")
    if bool((values[1:] < values[:-1]).any()):
        raise ValueError("pose_offsets must be non-decreasing")
    for pose_index in range(values.numel() - 1):
        start = int(values[pose_index])
        end = int(values[pose_index + 1])
        if end > start:
            yield pose_index, start, end


def _hash_uniform(train_seed: int, global_step: int, sample_index: int) -> float:
    payload = f"pvs-v3:{int(train_seed)}:{int(global_step)}:{int(sample_index)}".encode("ascii")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little")
    return (float(integer) + 0.5) / float(1 << 64)


def sample_train_operating_thresholds(
    train_seed: int,
    global_step: int,
    *,
    sample_count: int = 2,
    minimum: float = OPERATING_THRESHOLD_RANGE[0],
    maximum: float = OPERATING_THRESHOLD_RANGE[1],
    anchors: Sequence[float] = OPERATING_THRESHOLD_ANCHORS,
    anchor_period: int = 4,
    split: str = "train",
) -> tuple[float, ...]:
    """Return deterministic train-only log-uniform thresholds for one step.

    Two random thresholds are used by default. Every ``anchor_period`` steps,
    one fixed anchor is appended; successive anchor steps cycle through the
    registered anchor set. The function does not touch global Python or Torch
    RNG state.
    """
    if str(split) != "train":
        raise ValueError("operating thresholds are train-only")
    if int(train_seed) < 0 or int(global_step) < 0:
        raise ValueError("train_seed and global_step must be non-negative")
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 < float(minimum) < float(maximum) < 1.0:
        raise ValueError("threshold range must satisfy 0 < minimum < maximum < 1")
    if int(anchor_period) < 0:
        raise ValueError("anchor_period must be non-negative")
    anchor_values = tuple(float(value) for value in anchors)
    if any(not float(minimum) <= value <= float(maximum) for value in anchor_values):
        raise ValueError("every anchor must lie inside the operating threshold range")

    log_minimum = math.log(float(minimum))
    log_span = math.log(float(maximum)) - log_minimum
    result = [
        math.exp(log_minimum + _hash_uniform(train_seed, global_step, index) * log_span)
        for index in range(int(sample_count))
    ]
    if anchor_values and int(anchor_period) > 0 and int(global_step) % int(anchor_period) == 0:
        anchor_index = (int(global_step) // int(anchor_period)) % len(anchor_values)
        result.append(anchor_values[anchor_index])
    return tuple(result)


def bounded_topk_smooth_max(
    logits: torch.Tensor,
    *,
    top_k: int = 8,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Return a bounded smooth maximum over at most the highest eight logits."""
    values = logits.float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("bounded smooth-max requires at least one logit")
    if int(top_k) <= 0 or float(temperature) <= 0.0:
        raise ValueError("top_k and temperature must be positive")
    selected = torch.topk(values, k=min(int(top_k), values.numel()), sorted=False).values
    weights = torch.softmax(selected / float(temperature), dim=0)
    return (weights * selected).sum()


def _group_topk_smooth_max(
    logits: torch.Tensor,
    inverse_groups: torch.Tensor,
    group_count: int,
    *,
    top_k: int = 8,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Compute bounded smooth-max for every group without host-side loops."""
    values = logits.float().reshape(-1)
    groups = torch.as_tensor(
        inverse_groups, device=values.device, dtype=torch.long
    ).reshape(-1)
    if values.numel() == 0 or values.numel() != groups.numel():
        raise ValueError("group smooth-max inputs must be non-empty and aligned")
    if int(group_count) <= 0 or int(top_k) <= 0 or float(temperature) <= 0.0:
        raise ValueError("group_count, top_k, and temperature must be positive")

    # Sort logits first, then stably sort by group.  This leaves each group in
    # descending logit order, so a rank mask selects exactly its top-k values.
    value_order = torch.argsort(values, descending=True, stable=True)
    order = value_order[torch.argsort(groups[value_order], stable=True)]
    ordered_values = values[order]
    ordered_groups = groups[order]
    starts_mask = torch.ones(ordered_values.numel(), device=values.device, dtype=torch.bool)
    if ordered_values.numel() > 1:
        starts_mask[1:] = ordered_groups[1:] != ordered_groups[:-1]
    starts = torch.nonzero(starts_mask, as_tuple=False).reshape(-1)
    ends = torch.cat(
        [starts[1:], starts.new_tensor([ordered_values.numel()])]
    )
    ranks = torch.arange(ordered_values.numel(), device=values.device) - torch.repeat_interleave(
        starts, ends - starts
    )
    selected = ranks < int(top_k)
    selected_values = ordered_values[selected]
    selected_groups = ordered_groups[selected]
    group_max = torch.full(
        (int(group_count),), float("-inf"), device=values.device, dtype=values.dtype
    )
    group_max.scatter_reduce_(
        0, selected_groups, selected_values, reduce="amax", include_self=True
    )
    exp_values = torch.exp(
        (selected_values - group_max[selected_groups]) / float(temperature)
    )
    denominator = torch.zeros_like(group_max)
    denominator.scatter_add_(0, selected_groups, exp_values)
    numerator = torch.zeros_like(group_max)
    numerator.scatter_add_(0, selected_groups, exp_values * selected_values)
    return numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny)


def soft_request_probability(
    group_logit: torch.Tensor,
    threshold: float,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Map one bounded GLB logit to a smooth request probability."""
    tau = float(threshold)
    if not 0.0 < tau < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    value = group_logit.float()
    boundary = value.new_tensor(math.log(tau / (1.0 - tau)))
    return torch.sigmoid((value - boundary) / float(temperature))


def _normalize_pose_visible_weights(
    visible_weights: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    *,
    kappa: float = 99.0,
) -> torch.Tensor:
    weights = visible_weights.float().reshape(-1).clamp_min(0.0)
    labels = target.float().reshape(-1)
    if weights.numel() != labels.numel():
        raise ValueError("visible_weights and target must align")
    if float(kappa) <= 0.0:
        raise ValueError("weight normalization kappa must be positive")
    normalized = torch.zeros_like(weights)
    denominator = weights.new_tensor(math.log1p(float(kappa)))
    for _pose, start, end in _pose_slices(pose_offsets, weights.numel()):
        positive = labels[start:end] > 0.5
        if not bool(positive.any()):
            continue
        local = weights[start:end]
        maximum = local[positive].max().clamp_min(1e-6)
        relative = (local / maximum).clamp(0.0, 1.0)
        normalized[start:end] = torch.log1p(float(kappa) * relative) / denominator
        normalized[start:end] *= positive.to(normalized.dtype)
    return normalized


def _upper_tail_cvar(values: torch.Tensor, fraction: float) -> tuple[torch.Tensor, int]:
    flat = values.float().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("CVaR requires at least one value")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("CVaR fraction must lie in (0, 1]")
    count = min(flat.numel(), max(1, int(math.ceil(float(fraction) * flat.numel()))))
    return torch.topk(flat, k=count, sorted=False).values.mean(), count


def _linear_schedule_scale(
    global_step: int,
    total_optimizer_steps: int,
    *,
    initial_scale: float,
    start_fraction: float,
    ramp_fraction: float,
) -> float:
    """Ramp a train-only regularizer from ``initial_scale`` to one."""
    if int(global_step) < 0 or int(total_optimizer_steps) <= 0:
        raise ValueError("optimizer step values must be valid")
    if not 0.0 <= float(initial_scale) <= 1.0:
        raise ValueError("initial schedule scale must lie in [0, 1]")
    if not 0.0 <= float(start_fraction) <= 1.0 or not 0.0 <= float(ramp_fraction) <= 1.0:
        raise ValueError("schedule fractions must lie in [0, 1]")
    progress = float(global_step) / float(max(1, total_optimizer_steps - 1))
    if progress <= float(start_fraction):
        return float(initial_scale)
    if float(ramp_fraction) == 0.0:
        return 1.0
    ratio = min(1.0, (progress - float(start_fraction)) / float(ramp_fraction))
    return float(initial_scale) + ratio * (1.0 - float(initial_scale))


def _weighted_lower_tail(
    values: torch.Tensor,
    weights: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select an exact weighted lower tail with detached membership."""
    indices, masses = _weighted_lower_tail_indices(values, weights, fraction)
    return values.float().reshape(-1)[indices], masses


def _weighted_lower_tail_indices(
    selection_values: torch.Tensor,
    weights: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached lower-tail membership and its normalized mass."""
    flat_values = selection_values.float().reshape(-1)
    flat_weights = weights.float().reshape(-1).clamp_min(0.0)
    if flat_values.numel() == 0 or flat_values.numel() != flat_weights.numel():
        raise ValueError("weighted tail values and weights must be non-empty and aligned")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("weighted tail fraction must lie in (0, 1]")
    if not bool((flat_weights > 0.0).any()):
        flat_weights = torch.ones_like(flat_weights)
    order = torch.argsort(flat_values.detach())
    ordered_values = flat_values[order]
    ordered_weights = flat_weights[order]
    target_mass = ordered_weights.sum() * float(fraction)
    cumulative_before = torch.cumsum(ordered_weights, dim=0) - ordered_weights
    selected_mass = torch.minimum(
        ordered_weights,
        torch.clamp(target_mass - cumulative_before, min=0.0),
    )
    selected = selected_mass > 0.0
    masses = selected_mass[selected]
    return order[selected], masses / masses.sum().clamp_min(1e-12)


def same_instance_cross_view_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    instance_ids: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    recurrence_priority: torch.Tensor | None = None,
    margin: float = 0.50,
    temperature: float = 0.25,
    positive_importance_mix: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
    """Separate opposite labels for the same instance across sampled poses.

    This comparison cannot be satisfied by a fixed per-instance visibility
    prior.  It forces the runtime view query to explain why one instance is
    visible in one view-cell and invisible in another.  Positive and negative
    gradients are returned separately for the trainer's safety/efficiency
    gradient routing.
    """
    if float(margin) < 0.0 or float(temperature) <= 0.0:
        raise ValueError("cross-view rank margin and temperature are invalid")
    if not 0.0 <= float(positive_importance_mix) <= 1.0:
        raise ValueError("cross-view positive importance mix must lie in [0, 1]")
    source_logits = logits.float().reshape(-1)
    flat_logits = source_logits
    labels = target.float().reshape(-1)
    ids = instance_ids.long().reshape(-1)
    weights = positive_weights.float().reshape(-1).clamp_min(0.0)
    if (
        flat_logits.numel() == 0
        or labels.numel() != flat_logits.numel()
        or ids.numel() != flat_logits.numel()
        or weights.numel() != flat_logits.numel()
    ):
        raise ValueError("cross-view rank inputs must be non-empty and aligned")
    if bool((ids < 0).any()):
        raise ValueError("cross-view rank instance IDs must be non-negative")
    if not bool(torch.isfinite(flat_logits).all() and torch.isfinite(weights).all()):
        raise ValueError("cross-view rank inputs must be finite")
    priority = None
    if recurrence_priority is not None:
        priority = recurrence_priority.float().reshape(-1)
        if priority.numel() != flat_logits.numel():
            raise ValueError("cross-view recurrence priority must align with logits")
        if not bool(torch.isfinite(priority).all()) or bool((priority < 0.0).any()):
            raise ValueError("cross-view recurrence priority must be finite and non-negative")
        active = priority > 0.0
        if not bool(active.any()):
            zero = source_logits.sum() * 0.0
            return zero, zero, {
                "lossSameInstanceCrossViewPositive": zero,
                "lossSameInstanceCrossViewNegative": zero,
                "sameInstanceCrossViewGap": zero.detach(),
                "sameInstanceCrossViewWorstGap": zero.detach(),
                "sameInstanceCrossViewViolationFraction": zero.detach(),
                "sameInstanceCrossViewPairCount": 0.0,
                "sameInstanceCrossViewPriorPairCount": 0.0,
                "sameInstanceCrossViewPriorityMean": 0.0,
            }
        flat_logits = flat_logits[active]
        labels = labels[active]
        ids = ids[active]
        weights = weights[active]
        priority = priority[active]

    unique_ids, inverse = torch.unique(ids, sorted=False, return_inverse=True)
    group_count = int(unique_ids.numel())
    positive = labels > 0.5
    negative = ~positive
    positive_min = torch.full(
        (group_count,), float("inf"), device=flat_logits.device, dtype=flat_logits.dtype
    )
    negative_max = torch.full(
        (group_count,), float("-inf"), device=flat_logits.device, dtype=flat_logits.dtype
    )
    positive_min.scatter_reduce_(
        0,
        inverse,
        torch.where(positive, flat_logits, torch.full_like(flat_logits, float("inf"))),
        reduce="amin",
        include_self=True,
    )
    negative_max.scatter_reduce_(
        0,
        inverse,
        torch.where(negative, flat_logits, torch.full_like(flat_logits, float("-inf"))),
        reduce="amax",
        include_self=True,
    )
    paired = torch.isfinite(positive_min) & torch.isfinite(negative_max)
    group_priority = None
    if priority is not None:
        group_priority = torch.zeros(
            (group_count,), device=flat_logits.device, dtype=flat_logits.dtype
        )
        group_priority.scatter_reduce_(
            0,
            inverse,
            priority,
            reduce="amax",
            include_self=True,
        )
        paired = paired & (group_priority > 0.0)
    zero = source_logits.sum() * 0.0
    if not bool(paired.any()):
        return zero, zero, {
            "lossSameInstanceCrossViewPositive": zero,
            "lossSameInstanceCrossViewNegative": zero,
            "sameInstanceCrossViewGap": zero.detach(),
            "sameInstanceCrossViewWorstGap": zero.detach(),
            "sameInstanceCrossViewViolationFraction": zero.detach(),
            "sameInstanceCrossViewPairCount": 0.0,
            "sameInstanceCrossViewPriorPairCount": 0.0,
            "sameInstanceCrossViewPriorityMean": 0.0,
        }

    positive_importance = torch.log1p(weights)
    group_importance = torch.zeros(
        (group_count,), device=flat_logits.device, dtype=flat_logits.dtype
    )
    group_importance.scatter_reduce_(
        0,
        inverse,
        torch.where(positive, positive_importance, torch.zeros_like(positive_importance)),
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
    if group_priority is not None:
        paired_priority = group_priority[paired]
        group_weights = group_weights * (
            paired_priority / paired_priority.mean().clamp_min(1e-6)
        )
    group_weights = group_weights / group_weights.sum().clamp_min(1e-6)

    positive_tail = positive_min[paired]
    negative_tail = negative_max[paired]
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
    return positive_loss, negative_loss, {
        "lossSameInstanceCrossViewPositive": positive_loss,
        "lossSameInstanceCrossViewNegative": negative_loss,
        "sameInstanceCrossViewGap": (group_weights * gap).sum(),
        "sameInstanceCrossViewWorstGap": gap.min(),
        "sameInstanceCrossViewViolationFraction": (gap < float(margin)).float().mean(),
        "sameInstanceCrossViewPairCount": float(paired.sum()),
        "sameInstanceCrossViewPriorPairCount": float(paired.sum()),
        "sameInstanceCrossViewPriorityMean": (
            group_priority[paired].mean().detach()
            if group_priority is not None
            else flat_logits.new_ones(())
        ),
    }


def instance_exposure_balanced_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    instance_ids: torch.Tensor,
    positive_observation_weights: torch.Tensor,
    negative_observation_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return class-separated BCE with train-only per-instance exposure weights."""
    flat_logits = logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    ids = instance_ids.long().reshape(-1)
    positive_table = positive_observation_weights.float().reshape(-1)
    negative_table = negative_observation_weights.float().reshape(-1)
    if flat_logits.numel() == 0 or labels.numel() != flat_logits.numel() or ids.numel() != flat_logits.numel():
        raise ValueError("instance exposure BCE inputs must be non-empty and aligned")
    if positive_table.numel() != negative_table.numel() or positive_table.numel() == 0:
        raise ValueError("instance exposure weight tables must be non-empty and aligned")
    if bool((ids < 0).any()) or int(ids.max()) >= positive_table.numel():
        raise ValueError("instance exposure BCE instance IDs are out of range")
    if not bool(
        torch.isfinite(flat_logits).all()
        and torch.isfinite(positive_table).all()
        and torch.isfinite(negative_table).all()
    ):
        raise ValueError("instance exposure BCE inputs must be finite")
    if bool((positive_table < 0.0).any() or (negative_table < 0.0).any()):
        raise ValueError("instance exposure weights must be non-negative")

    positive = labels > 0.5
    negative = ~positive
    zero = flat_logits.sum() * 0.0
    if bool(positive.any()):
        local_weight = positive_table[ids[positive]]
        positive_loss = (
            F.softplus(-flat_logits[positive]) * local_weight
        ).sum() / local_weight.sum().clamp_min(1e-6)
    else:
        positive_loss = zero
    if bool(negative.any()):
        local_weight = negative_table[ids[negative]]
        negative_loss = (
            F.softplus(flat_logits[negative]) * local_weight
        ).sum() / local_weight.sum().clamp_min(1e-6)
    else:
        negative_loss = zero
    return positive_loss, negative_loss, {
        "lossInstanceExposurePositive": positive_loss,
        "lossInstanceExposureNegative": negative_loss,
        "instanceExposurePositiveCount": float(positive.sum()),
        "instanceExposureNegativeCount": float(negative.sum()),
    }


def view_residual_tail_regularizers(
    final_logits: torch.Tensor,
    base_logits: torch.Tensor,
    residual: torch.Tensor,
    target: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    positive_importance_mix: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
    """Keep a tail-ranking residual from becoming a global score shift.

    The pairwise tail objective is translation invariant.  These companion
    terms preserve the initialized model's positive scores and keep the new
    residual small and centered without introducing a probability anchor.
    """
    final = final_logits.float().reshape(-1)
    base = base_logits.float().reshape(-1)
    correction = residual.float().reshape(-1)
    labels = target.float().reshape(-1)
    importance = positive_weights.float().reshape(-1).clamp_min(0.0)
    if final.numel() == 0 or not (
        final.numel()
        == base.numel()
        == correction.numel()
        == labels.numel()
        == importance.numel()
    ):
        raise ValueError("view residual regularizer inputs must be non-empty and aligned")
    if not bool(
        torch.isfinite(final).all()
        and torch.isfinite(base).all()
        and torch.isfinite(correction).all()
        and torch.isfinite(importance).all()
    ):
        raise ValueError("view residual regularizer inputs must be finite")
    if not 0.0 <= float(positive_importance_mix) <= 1.0:
        raise ValueError("view residual positive importance mix must lie in [0, 1]")

    positive = labels > 0.5
    zero = final.sum() * 0.0
    if bool(positive.any()):
        local_importance = importance[positive]
        uniform_mass = torch.full_like(
            local_importance,
            1.0 / float(local_importance.numel()),
        )
        if bool((local_importance > 0.0).any()):
            importance_mass = (
                local_importance / local_importance.sum().clamp_min(1e-12)
            )
        else:
            importance_mass = uniform_mass
        positive_mass = (
            (1.0 - float(positive_importance_mix)) * uniform_mass
            + float(positive_importance_mix) * importance_mass
        )
        positive_drop = torch.relu(
            base[positive].detach() - final[positive]
        )
        positive_guard = (positive_mass * positive_drop.square()).sum()
    else:
        positive_drop = final.new_zeros((0,))
        positive_guard = zero
    residual_anchor = correction.square().mean()
    residual_center = correction.mean().square()
    return positive_guard, residual_anchor, residual_center, {
        "lossViewResidualPositiveGuard": positive_guard,
        "lossViewResidualAnchor": residual_anchor,
        "lossViewResidualCenter": residual_center,
        "viewResidualPositiveDropMean": (
            positive_drop.mean().detach()
            if positive_drop.numel()
            else zero.detach()
        ),
        "viewResidualPositiveDropFraction": (
            (positive_drop.detach() > 0.0).float().mean()
            if positive_drop.numel()
            else zero.detach()
        ),
        "viewResidualPositiveImportanceMix": float(positive_importance_mix),
    }


def extreme_tail_separation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    positive_tail_fraction: float = 0.01,
    negative_tail_fraction: float = 0.01,
    margin: float = 0.30,
    temperature: float = 0.25,
    pose_cvar_fraction: float = 0.50,
    pose_cvar_weight: float = 0.25,
    positive_importance_mix: float = 0.80,
    positive_gradient_scale: float = 1.0,
    selection_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Separate the safety-critical positive and negative score tails.

    Tail membership is selected from detached logits, while every selected
    logit receives a finite pairwise softplus gradient.  Differences are
    divided by a detached robust logit range, so the objective cannot be
    satisfied merely by changing the global score scale.
    """
    if not 0.0 < float(positive_tail_fraction) <= 1.0:
        raise ValueError("positive tail fraction must lie in (0, 1]")
    if not 0.0 < float(negative_tail_fraction) <= 1.0:
        raise ValueError("negative tail fraction must lie in (0, 1]")
    if float(margin) < 0.0 or float(temperature) <= 0.0:
        raise ValueError("tail margin must be non-negative and temperature positive")
    if not 0.0 < float(pose_cvar_fraction) <= 1.0 or float(pose_cvar_weight) < 0.0:
        raise ValueError("pose tail CVaR arguments are invalid")
    if not 0.0 <= float(positive_importance_mix) <= 1.0:
        raise ValueError("positive importance mix must lie in [0, 1]")
    if not 0.0 <= float(positive_gradient_scale) <= 1.0:
        raise ValueError("positive tail gradient scale must lie in [0, 1]")

    flat_logits = logits.float().reshape(-1)
    flat_selection_logits = (
        flat_logits
        if selection_logits is None
        else torch.as_tensor(
            selection_logits,
            device=flat_logits.device,
            dtype=flat_logits.dtype,
        ).reshape(-1)
    )
    labels = target.float().reshape(-1)
    weights = positive_weights.float().reshape(-1).clamp_min(0.0)
    if (
        flat_logits.numel() == 0
        or labels.numel() != flat_logits.numel()
        or weights.numel() != flat_logits.numel()
        or flat_selection_logits.numel() != flat_logits.numel()
    ):
        raise ValueError("tail separation inputs must be non-empty and aligned")
    if (
        not bool(torch.isfinite(flat_logits).all())
        or not bool(torch.isfinite(flat_selection_logits).all())
        or not bool(torch.isfinite(weights).all())
    ):
        raise ValueError("tail separation logits and weights must be finite")

    pose_losses: list[torch.Tensor] = []
    pose_gaps: list[torch.Tensor] = []
    violation_fractions: list[torch.Tensor] = []
    positive_count = 0
    negative_count = 0
    for _pose, start, end in _pose_slices(pose_offsets, flat_logits.numel()):
        local_logits = flat_logits[start:end]
        local_selection_logits = flat_selection_logits[start:end].detach()
        local_labels = labels[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        if not bool(positive.any()) or not bool(negative.any()):
            continue
        local_positive_weights = weights[start:end][positive]
        uniform_mass = torch.full_like(
            local_positive_weights,
            1.0 / float(local_positive_weights.numel()),
        )
        if bool((local_positive_weights > 0.0).any()):
            importance_mass = local_positive_weights / local_positive_weights.sum().clamp_min(1e-12)
        else:
            importance_mass = uniform_mass
        selection_mass = (
            (1.0 - float(positive_importance_mix)) * uniform_mass
            + float(positive_importance_mix) * importance_mass
        )
        positive_indices, positive_masses = _weighted_lower_tail_indices(
            local_selection_logits[positive],
            selection_mass,
            positive_tail_fraction,
        )
        positive_values = local_logits[positive][positive_indices]
        negative_indices = torch.topk(
            local_selection_logits[negative],
            k=min(
                local_selection_logits[negative].numel(),
                max(1, int(math.ceil(float(negative_tail_fraction) * int(negative.sum())))),
            ),
            sorted=False,
        ).indices
        negative_values = local_logits[negative][negative_indices]
        q10 = torch.quantile(local_logits.detach(), 0.10)
        q90 = torch.quantile(local_logits.detach(), 0.90)
        robust_scale = (q90 - q10).clamp_min(1.0)
        positive_loss_values = (
            positive_values.detach()
            + float(positive_gradient_scale)
            * (positive_values - positive_values.detach())
        )
        normalized_violation = (
            negative_values[:, None]
            - positive_loss_values[None, :]
        ) / robust_scale + float(margin)
        pair_losses = (
            F.softplus(normalized_violation / float(temperature))
            * float(temperature)
        )
        pose_loss = (pair_losses * positive_masses[None, :]).sum(dim=1).mean()
        positive_tail_mean = (positive_values * positive_masses).sum()
        negative_tail_mean = negative_values.mean()
        pose_losses.append(pose_loss)
        pose_gaps.append((positive_tail_mean - negative_tail_mean) / robust_scale)
        violation_fractions.append((normalized_violation.detach() > 0.0).float().mean())
        positive_count += int(positive_values.numel())
        negative_count += int(negative_values.numel())

    zero = flat_logits.sum() * 0.0
    if not pose_losses:
        return zero, {
            "lossExtremeTailSeparation": zero,
            "extremeTailGap": zero.detach(),
            "extremeTailViolationFraction": zero.detach(),
            "extremeTailPoseCount": 0.0,
            "extremeTailPositiveCount": 0.0,
            "extremeTailNegativeCount": 0.0,
        }
    stacked_losses = torch.stack(pose_losses)
    pose_cvar, _count = _upper_tail_cvar(stacked_losses, pose_cvar_fraction)
    loss = stacked_losses.mean() + float(pose_cvar_weight) * pose_cvar
    return loss, {
        "lossExtremeTailSeparation": loss,
        "lossExtremeTailSeparationPoseMean": stacked_losses.mean(),
        "lossExtremeTailSeparationPoseCvar": pose_cvar,
        "extremeTailGap": torch.stack(pose_gaps).mean().detach(),
        "extremeTailWorstGap": torch.stack(pose_gaps).min().detach(),
        "extremeTailViolationFraction": torch.stack(violation_fractions).mean().detach(),
        "extremeTailPoseCount": float(len(pose_losses)),
        "extremeTailPositiveCount": float(positive_count),
        "extremeTailNegativeCount": float(negative_count),
    }


def weighted_positive_tail_compactness_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    tail_mass_fraction: float = 0.01,
    reference_quantile: float = 0.05,
    allowed_relative_gap: float = 0.20,
    temperature: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Raise only the weighted positive low tail toward a detached reference.

    Both the tail and reference are defined within the current train batch, so
    the objective is invariant to a global logit shift and has no fixed 0.5
    probability anchor.  Negatives never enter this safety objective.
    """
    if not 0.0 < float(tail_mass_fraction) < float(reference_quantile) < 1.0:
        raise ValueError(
            "positive tail mass must be below a reference quantile in (0, 1)"
        )
    if float(allowed_relative_gap) < 0.0 or float(temperature) <= 0.0:
        raise ValueError("positive tail gap must be non-negative and temperature positive")
    flat_logits = logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    weights = positive_weights.float().reshape(-1).clamp_min(0.0)
    if (
        flat_logits.numel() == 0
        or labels.numel() != flat_logits.numel()
        or weights.numel() != flat_logits.numel()
    ):
        raise ValueError("positive tail compactness inputs must be non-empty and aligned")
    if not bool(torch.isfinite(flat_logits).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("positive tail compactness inputs must be finite")
    positive = labels > 0.5
    zero = flat_logits.sum() * 0.0
    if not bool(positive.any()):
        return zero, {
            "lossPositiveTailCompactness": zero,
            "positiveTailReferenceLogit": zero.detach(),
            "positiveTailMeanLogit": zero.detach(),
            "positiveTailReferenceGap": zero.detach(),
            "positiveTailRobustLogitScale": zero.detach(),
            "positiveTailViolationFraction": zero.detach(),
            "positiveTailSelectedCount": 0.0,
        }
    positive_logits = flat_logits[positive]
    positive_mass = weights[positive]
    if not bool((positive_mass > 0.0).any()):
        positive_mass = torch.ones_like(positive_mass)
    tail_values, tail_masses = _weighted_lower_tail(
        positive_logits,
        positive_mass,
        tail_mass_fraction,
    )
    reference = _weighted_lower_quantile(
        positive_logits,
        positive_mass,
        reference_quantile,
    ).detach()
    robust_scale = (
        torch.quantile(flat_logits.detach(), 0.90)
        - torch.quantile(flat_logits.detach(), 0.10)
    ).clamp_min(1.0)
    normalized_violation = (
        reference
        - tail_values
        - float(allowed_relative_gap) * robust_scale
    ) / (float(temperature) * robust_scale)
    active = normalized_violation.detach() > 0.0
    penalties = (
        F.softplus(normalized_violation)
        * float(temperature)
        * robust_scale
        * active.to(dtype=tail_values.dtype)
    )
    loss = (penalties * tail_masses).sum()
    tail_mean = (tail_values * tail_masses).sum()
    return loss, {
        "lossPositiveTailCompactness": loss,
        "positiveTailReferenceLogit": reference,
        "positiveTailMeanLogit": tail_mean.detach(),
        "positiveTailReferenceGap": (reference - tail_mean).detach(),
        "positiveTailRobustLogitScale": robust_scale,
        "positiveTailViolationFraction": (
            active.float().mean()
        ),
        "positiveTailSelectedCount": float(tail_values.numel()),
    }


def safety_boundary_negative_excess_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    scope: str = "batch",
    positive_mass_fraction: float = 0.01,
    negative_tail_fraction: float = 1.0,
    margin: float = 0.0,
    temperature: float = 0.20,
    pose_cvar_fraction: float = 0.50,
    pose_cvar_weight: float = 0.50,
    reference_boundary_logit: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Penalize negatives that exceed a detached weighted-recall boundary.

    The boundary is the weighted positive quantile corresponding to the
    allowed visual-utility miss mass.  Only negative logits receive gradients.
    ``batch`` scope uses the same raw visual-weight mass definition as
    aggregate weighted recall, but remains a minibatch surrogate. ``pose``
    scope is retained as a diagnostic for pose-balanced pressure.
    """
    if str(scope) not in {"batch", "pose"}:
        raise ValueError("safety boundary scope must be batch or pose")
    if not 0.0 < float(positive_mass_fraction) < 1.0:
        raise ValueError("positive mass fraction must lie in (0, 1)")
    if not 0.0 < float(negative_tail_fraction) <= 1.0:
        raise ValueError("negative tail fraction must lie in (0, 1]")
    if float(margin) < 0.0 or float(temperature) <= 0.0:
        raise ValueError("safety boundary margin must be non-negative and temperature positive")
    if not 0.0 < float(pose_cvar_fraction) <= 1.0 or float(pose_cvar_weight) < 0.0:
        raise ValueError("safety boundary pose CVaR arguments are invalid")

    flat_logits = logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    weights = positive_weights.float().reshape(-1).clamp_min(0.0)
    if flat_logits.numel() == 0 or labels.numel() != flat_logits.numel() or weights.numel() != flat_logits.numel():
        raise ValueError("safety boundary inputs must be non-empty and aligned")
    if not bool(torch.isfinite(flat_logits).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("safety boundary logits and weights must be finite")

    slices = list(_pose_slices(pose_offsets, flat_logits.numel()))
    if reference_boundary_logit is not None and str(scope) != "batch":
        raise ValueError("a reference safety boundary is only valid for batch scope")
    reference_boundary = None
    if reference_boundary_logit is not None:
        reference_boundary = torch.as_tensor(
            reference_boundary_logit,
            device=flat_logits.device,
            dtype=flat_logits.dtype,
        ).reshape(())
        if not bool(torch.isfinite(reference_boundary)):
            raise ValueError("reference safety boundary must be finite")
        reference_boundary = reference_boundary.detach()

    groups = [(0, 0, flat_logits.numel())] if str(scope) == "batch" else slices
    group_losses: list[torch.Tensor] = []
    batch_boundaries: list[torch.Tensor] = []
    boundaries: list[torch.Tensor] = []
    violation_fractions: list[torch.Tensor] = []
    selected_negative_count = 0
    for _group, start, end in groups:
        local_logits = flat_logits[start:end]
        local_labels = labels[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        if not bool(positive.any()) or not bool(negative.any()):
            continue
        local_weights = weights[start:end][positive]
        if not bool((local_weights > 0.0).any()):
            local_weights = torch.ones_like(local_weights)
        batch_boundary = _weighted_lower_quantile(
            local_logits[positive],
            local_weights,
            positive_mass_fraction,
        ).detach()
        boundary = (
            torch.minimum(batch_boundary, reference_boundary)
            if reference_boundary is not None
            else batch_boundary
        )
        negative_logits = local_logits[negative]
        excess = (
            negative_logits - boundary + float(margin)
        ) / float(temperature)
        penalties = F.softplus(excess) * float(temperature)
        count = min(
            penalties.numel(),
            max(1, int(math.ceil(float(negative_tail_fraction) * penalties.numel()))),
        )
        selected_penalties = torch.topk(penalties, k=count, sorted=False).values
        group_losses.append(selected_penalties.mean())
        batch_boundaries.append(batch_boundary)
        boundaries.append(boundary)
        violation_fractions.append((excess.detach() > 0.0).float().mean())
        selected_negative_count += int(count)

    zero = flat_logits.sum() * 0.0
    if not group_losses:
        return zero, {
            "lossSafetyBoundaryNegativeExcess": zero,
            "lossSafetyBoundaryNegativeExcessMean": zero,
            "lossSafetyBoundaryNegativeExcessPoseCvar": zero,
            "safetyBoundaryBatchLogit": zero.detach(),
            "safetyBoundaryReferenceLogit": zero.detach(),
            "safetyBoundaryLogit": zero.detach(),
            "safetyBoundaryViolationFraction": zero.detach(),
            "safetyBoundaryGroupCount": 0.0,
            "safetyBoundarySelectedNegativeCount": 0.0,
        }
    stacked = torch.stack(group_losses)
    if str(scope) == "pose":
        pose_cvar, _count = _upper_tail_cvar(stacked, pose_cvar_fraction)
        loss = stacked.mean() + float(pose_cvar_weight) * pose_cvar
    else:
        pose_cvar = zero
        loss = stacked.mean()
    return loss, {
        "lossSafetyBoundaryNegativeExcess": loss,
        "lossSafetyBoundaryNegativeExcessMean": stacked.mean(),
        "lossSafetyBoundaryNegativeExcessPoseCvar": pose_cvar,
        "safetyBoundaryBatchLogit": torch.stack(batch_boundaries).mean().detach(),
        "safetyBoundaryReferenceLogit": (
            reference_boundary
            if reference_boundary is not None
            else torch.stack(batch_boundaries).mean().detach()
        ),
        "safetyBoundaryLogit": torch.stack(boundaries).mean().detach(),
        "safetyBoundaryViolationFraction": torch.stack(violation_fractions).mean().detach(),
        "safetyBoundaryGroupCount": float(len(group_losses)),
        "safetyBoundarySelectedNegativeCount": float(selected_negative_count),
    }


def candidate_boundary_hard_negative_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    positive_weights: torch.Tensor,
    boundary_proximity: torch.Tensor,
    *,
    positive_mass_fraction: float = 0.01,
    negative_tail_fraction: float = 0.01,
    selection_bias: float = 1.0,
    proximity_gain: float = 1.0,
    margin: float = 0.0,
    temperature: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Suppress high-score negatives concentrated near candidate-frustum edges.

    The positive weighted boundary is detached and only negative logits receive
    gradients.  Boundary proximity is used for detached hard-example selection
    and for a bounded sample weight.  This keeps the branch in the efficiency
    gradient group and avoids turning legitimate edge-visible positives into
    negatives.
    """
    if not 0.0 < float(positive_mass_fraction) < 1.0:
        raise ValueError("positive mass fraction must lie in (0, 1)")
    if not 0.0 < float(negative_tail_fraction) <= 1.0:
        raise ValueError("negative tail fraction must lie in (0, 1]")
    if (
        float(selection_bias) < 0.0
        or float(proximity_gain) < 0.0
        or float(margin) < 0.0
        or float(temperature) <= 0.0
    ):
        raise ValueError("candidate boundary hard-negative arguments are invalid")

    flat_logits = logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    weights = positive_weights.float().reshape(-1).clamp_min(0.0)
    proximity = boundary_proximity.float().reshape(-1)
    if flat_logits.numel() == 0 or not (
        flat_logits.numel() == labels.numel() == weights.numel() == proximity.numel()
    ):
        raise ValueError("candidate boundary inputs must be non-empty and aligned")
    if not bool(
        torch.isfinite(flat_logits).all()
        and torch.isfinite(weights).all()
        and torch.isfinite(proximity).all()
    ):
        raise ValueError("candidate boundary inputs must be finite")
    if bool(((proximity < 0.0) | (proximity > 1.0)).any()):
        raise ValueError("candidate boundary proximity must lie in [0, 1]")

    pose_losses: list[torch.Tensor] = []
    selected_proximity: list[torch.Tensor] = []
    selected_count = 0
    zero = flat_logits.sum() * 0.0
    for _pose, start, end in _pose_slices(pose_offsets, flat_logits.numel()):
        local_logits = flat_logits[start:end]
        local_labels = labels[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        if not bool(positive.any()) or not bool(negative.any()):
            continue
        local_weights = weights[start:end][positive]
        if not bool((local_weights > 0.0).any()):
            local_weights = torch.ones_like(local_weights)
        safety_boundary = _weighted_lower_quantile(
            local_logits[positive],
            local_weights,
            positive_mass_fraction,
        ).detach()
        negative_logits = local_logits[negative]
        negative_proximity = proximity[start:end][negative].detach()
        count = min(
            negative_logits.numel(),
            max(
                1,
                int(
                    math.ceil(
                        float(negative_tail_fraction) * negative_logits.numel()
                    )
                ),
            ),
        )
        selection_key = (
            negative_logits.detach()
            + float(selection_bias) * negative_proximity
        )
        selected = torch.topk(selection_key, k=count, sorted=False).indices
        selected_logits = negative_logits[selected]
        selected_boundary_proximity = negative_proximity[selected]
        sample_weight = 1.0 + float(proximity_gain) * selected_boundary_proximity
        penalty = (
            F.softplus(
                (
                    selected_logits
                    - safety_boundary
                    + float(margin)
                )
                / float(temperature)
            )
            * float(temperature)
        )
        pose_losses.append(
            (sample_weight * penalty).sum() / sample_weight.sum().clamp_min(1e-12)
        )
        selected_proximity.append(selected_boundary_proximity.mean())
        selected_count += int(count)

    if not pose_losses:
        return zero, {
            "lossCandidateBoundaryHardNegative": zero,
            "candidateBoundarySelectedProximity": zero.detach(),
            "candidateBoundarySelectedNegativeCount": 0.0,
            "candidateBoundaryPoseCount": 0.0,
        }
    loss = torch.stack(pose_losses).mean()
    return loss, {
        "lossCandidateBoundaryHardNegative": loss,
        "candidateBoundarySelectedProximity": torch.stack(
            selected_proximity
        ).mean(),
        "candidateBoundarySelectedNegativeCount": float(selected_count),
        "candidateBoundaryPoseCount": float(len(pose_losses)),
    }


def cull_certificate_tail_classification_loss(
    certificate_raw: torch.Tensor,
    base_logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    *,
    visible_hit_rates: torch.Tensor | None = None,
    negative_tail_fraction: float = 0.02,
    positive_uniform_mix: float = 0.20,
    rare_positive_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
    """Train a detached-feature certificate on positives and hard negatives.

    The highest-scoring train negatives in each pose are the only samples that
    may request suppression.  Every train positive guards against suppression,
    with its raw visual-utility weight normalized across the batch.  Base
    logits are used only for detached membership selection and receive no
    gradient from either certificate term.
    """
    if not 0.0 < float(negative_tail_fraction) <= 1.0:
        raise ValueError("certificate negative tail fraction must lie in (0, 1]")
    if not 0.0 <= float(positive_uniform_mix) <= 1.0:
        raise ValueError("certificate positive uniform mix must lie in [0, 1]")
    if float(rare_positive_weight) < 0.0:
        raise ValueError("certificate rare positive weight must be non-negative")
    raw = certificate_raw.float().reshape(-1)
    base = base_logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    weights = visible_weights.float().reshape(-1).clamp_min(0.0)
    if raw.numel() == 0 or not (
        raw.numel() == base.numel() == labels.numel() == weights.numel()
    ):
        raise ValueError("certificate inputs must be non-empty and aligned")
    if not (
        bool(torch.isfinite(raw).all())
        and bool(torch.isfinite(base).all())
        and bool(torch.isfinite(weights).all())
    ):
        raise ValueError("certificate inputs must be finite")
    hit_rates = (
        torch.ones_like(raw)
        if visible_hit_rates is None
        else torch.as_tensor(
            visible_hit_rates,
            device=raw.device,
            dtype=raw.dtype,
        ).reshape(-1)
    )
    if hit_rates.numel() != raw.numel() or not bool(torch.isfinite(hit_rates).all()):
        raise ValueError("certificate visible hit rates must be finite and aligned")
    hit_rates = hit_rates.clamp(0.0, 1.0)

    positive = labels > 0.5
    zero = raw.sum() * 0.0
    if bool(positive.any()):
        positive_weights = weights[positive]
        positive_weights = positive_weights * (
            1.0
            + float(rare_positive_weight)
            * (1.0 - torch.sqrt(hit_rates[positive]))
        )
        if not bool((positive_weights > 0.0).any()):
            positive_weights = torch.ones_like(positive_weights)
        weighted_mass = positive_weights / positive_weights.sum().clamp_min(1e-12)
        uniform_mass = torch.full_like(weighted_mass, 1.0 / weighted_mass.numel())
        positive_mass = (
            float(positive_uniform_mix) * uniform_mass
            + (1.0 - float(positive_uniform_mix)) * weighted_mass
        )
        positive_guard = (
            positive_mass * F.softplus(raw[positive])
        ).sum()
    else:
        positive_guard = zero

    hard_negative_indices: list[torch.Tensor] = []
    for _pose, start, end in _pose_slices(pose_offsets, raw.numel()):
        local_negative = torch.nonzero(
            labels[start:end] <= 0.5,
            as_tuple=False,
        ).reshape(-1)
        if local_negative.numel() == 0:
            continue
        count = min(
            local_negative.numel(),
            max(
                1,
                int(
                    math.ceil(
                        float(negative_tail_fraction)
                        * local_negative.numel()
                    )
                ),
            ),
        )
        selected = torch.topk(
            base.detach()[start:end][local_negative],
            k=count,
            sorted=False,
        ).indices
        hard_negative_indices.append(start + local_negative[selected])
    if hard_negative_indices:
        hard_indices = torch.cat(hard_negative_indices)
        negative_certificate = F.softplus(-raw[hard_indices]).mean()
        selected_raw_mean = raw[hard_indices].mean().detach()
        selected_count = float(hard_indices.numel())
    else:
        negative_certificate = zero
        selected_raw_mean = zero.detach()
        selected_count = 0.0

    parts: dict[str, torch.Tensor | float] = {
        "lossCullCertificatePositiveGuard": positive_guard,
        "lossCullCertificateHardNegative": negative_certificate,
        "cullCertificatePositiveRawMean": (
            raw[positive].mean().detach() if bool(positive.any()) else zero.detach()
        ),
        "cullCertificateHardNegativeRawMean": selected_raw_mean,
        "cullCertificatePositiveCount": float(positive.sum()),
        "cullCertificateHardNegativeCount": selected_count,
        "cullCertificatePositiveUniformMix": float(positive_uniform_mix),
        "cullCertificateRarePositiveWeight": float(rare_positive_weight),
    }
    return positive_guard, negative_certificate, parts


def cull_certificate_pose_tail_pair_loss(
    final_logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    *,
    positive_tail_fraction: float = 0.01,
    negative_tail_fraction: float = 0.01,
    margin: float = 0.25,
    temperature: float = 0.25,
    pose_cvar_fraction: float = 0.50,
    pose_cvar_weight: float = 0.25,
    positive_uniform_mix: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
    """Separate the final-score tails without coupling their gradients.

    Every pose contributes equally.  The positive branch can only reduce
    suppression on the weighted low-tail positives, while the negative branch
    can only increase suppression on the highest-scoring negatives.  Keeping
    the two branches separate lets the caller place them in the safety and
    efficiency gradient groups respectively.
    """
    if not 0.0 < float(positive_tail_fraction) <= 1.0:
        raise ValueError("certificate positive tail fraction must lie in (0, 1]")
    if not 0.0 < float(negative_tail_fraction) <= 1.0:
        raise ValueError("certificate negative tail fraction must lie in (0, 1]")
    if float(margin) < 0.0 or float(temperature) <= 0.0:
        raise ValueError("certificate pair margin and temperature are invalid")
    if not 0.0 < float(pose_cvar_fraction) <= 1.0 or float(pose_cvar_weight) < 0.0:
        raise ValueError("certificate pose CVaR arguments are invalid")
    if not 0.0 <= float(positive_uniform_mix) <= 1.0:
        raise ValueError("certificate positive uniform mix must lie in [0, 1]")

    logits = final_logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    weights = visible_weights.float().reshape(-1).clamp_min(0.0)
    if logits.numel() == 0 or not (
        logits.numel() == labels.numel() == weights.numel()
    ):
        raise ValueError("certificate pair inputs must be non-empty and aligned")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("certificate pair inputs must be finite")

    positive_pose_losses: list[torch.Tensor] = []
    negative_pose_losses: list[torch.Tensor] = []
    pose_gaps: list[torch.Tensor] = []
    violation_fractions: list[torch.Tensor] = []
    selected_positive_count = 0
    selected_negative_count = 0
    for _pose, start, end in _pose_slices(pose_offsets, logits.numel()):
        local_logits = logits[start:end]
        local_labels = labels[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        if not bool(positive.any()) or not bool(negative.any()):
            continue

        local_positive_weights = weights[start:end][positive]
        uniform_mass = torch.full_like(
            local_positive_weights,
            1.0 / float(local_positive_weights.numel()),
        )
        if bool((local_positive_weights > 0.0).any()):
            importance_mass = (
                local_positive_weights
                / local_positive_weights.sum().clamp_min(1e-12)
            )
        else:
            importance_mass = uniform_mass
        selection_mass = (
            float(positive_uniform_mix) * uniform_mass
            + (1.0 - float(positive_uniform_mix)) * importance_mass
        )
        positive_values, positive_masses = _weighted_lower_tail(
            local_logits[positive],
            selection_mass,
            positive_tail_fraction,
        )
        negative_values = torch.topk(
            local_logits[negative],
            k=min(
                int(negative.sum()),
                max(
                    1,
                    int(
                        math.ceil(
                            float(negative_tail_fraction) * int(negative.sum())
                        )
                    ),
                ),
            ),
            sorted=False,
        ).values

        positive_violation = (
            negative_values.detach()[:, None]
            - positive_values[None, :]
            + float(margin)
        ) / float(temperature)
        negative_violation = (
            negative_values[:, None]
            - positive_values.detach()[None, :]
            + float(margin)
        ) / float(temperature)
        positive_pairs = F.softplus(positive_violation) * float(temperature)
        negative_pairs = F.softplus(negative_violation) * float(temperature)
        positive_pose_losses.append(
            (positive_pairs * positive_masses[None, :]).sum(dim=1).mean()
        )
        negative_pose_losses.append(
            (negative_pairs * positive_masses[None, :]).sum(dim=1).mean()
        )
        positive_mean = (positive_values * positive_masses).sum()
        pose_gaps.append((positive_mean - negative_values.mean()).detach())
        violation_fractions.append((positive_violation.detach() > 0.0).float().mean())
        selected_positive_count += int(positive_values.numel())
        selected_negative_count += int(negative_values.numel())

    zero = logits.sum() * 0.0
    if not positive_pose_losses:
        return zero, zero, {
            "lossCullCertificatePairPositive": zero,
            "lossCullCertificatePairNegative": zero,
            "cullCertificatePairGap": zero.detach(),
            "cullCertificatePairWorstGap": zero.detach(),
            "cullCertificatePairViolationFraction": zero.detach(),
            "cullCertificatePairPoseCount": 0.0,
            "cullCertificatePairPositiveCount": 0.0,
            "cullCertificatePairNegativeCount": 0.0,
        }
    positive_stacked = torch.stack(positive_pose_losses)
    negative_stacked = torch.stack(negative_pose_losses)
    positive_cvar, _ = _upper_tail_cvar(positive_stacked, pose_cvar_fraction)
    negative_cvar, _ = _upper_tail_cvar(negative_stacked, pose_cvar_fraction)
    positive_loss = (
        positive_stacked.mean() + float(pose_cvar_weight) * positive_cvar
    )
    negative_loss = (
        negative_stacked.mean() + float(pose_cvar_weight) * negative_cvar
    )
    return positive_loss, negative_loss, {
        "lossCullCertificatePairPositive": positive_loss,
        "lossCullCertificatePairNegative": negative_loss,
        "lossCullCertificatePairPositivePoseMean": positive_stacked.mean(),
        "lossCullCertificatePairNegativePoseMean": negative_stacked.mean(),
        "lossCullCertificatePairPositivePoseCvar": positive_cvar,
        "lossCullCertificatePairNegativePoseCvar": negative_cvar,
        "cullCertificatePairGap": torch.stack(pose_gaps).mean(),
        "cullCertificatePairWorstGap": torch.stack(pose_gaps).min(),
        "cullCertificatePairViolationFraction": torch.stack(
            violation_fractions
        ).mean(),
        "cullCertificatePairPoseCount": float(len(positive_pose_losses)),
        "cullCertificatePairPositiveCount": float(selected_positive_count),
        "cullCertificatePairNegativeCount": float(selected_negative_count),
    }


def _weighted_lower_quantile(
    values: torch.Tensor,
    weights: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    flat_values = values.float().reshape(-1)
    flat_weights = weights.float().reshape(-1).clamp_min(0.0)
    if flat_values.numel() == 0 or flat_values.numel() != flat_weights.numel():
        raise ValueError("weighted quantile values and weights must be non-empty and aligned")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    positive_weight = flat_weights > 0.0
    if bool(positive_weight.any()):
        flat_values = flat_values[positive_weight]
        flat_weights = flat_weights[positive_weight]
    else:
        flat_weights = torch.ones_like(flat_values)
    order = torch.argsort(flat_values.detach())
    cumulative = torch.cumsum(flat_weights[order], dim=0)
    target = cumulative[-1] * float(quantile)
    index = torch.searchsorted(cumulative, target, right=False).clamp_max(cumulative.numel() - 1)
    return flat_values[order[index]]


def weighted_positive_safety_boundary_logit(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    positive_mass_fraction: float = 0.01,
) -> torch.Tensor:
    """Return a detached train-batch boundary for weighted positive recall."""
    flat_logits = logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    weights = positive_weights.float().reshape(-1).clamp_min(0.0)
    if not 0.0 < float(positive_mass_fraction) < 1.0:
        raise ValueError("positive mass fraction must lie in (0, 1)")
    if (
        flat_logits.numel() == 0
        or labels.numel() != flat_logits.numel()
        or weights.numel() != flat_logits.numel()
    ):
        raise ValueError("positive safety boundary inputs must be non-empty and aligned")
    if not bool(torch.isfinite(flat_logits).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("positive safety boundary inputs must be finite")
    positive = labels > 0.5
    if not bool(positive.any()):
        raise ValueError("positive safety boundary requires at least one positive")
    local_weights = weights[positive]
    if not bool((local_weights > 0.0).any()):
        local_weights = torch.ones_like(local_weights)
    return _weighted_lower_quantile(
        flat_logits[positive],
        local_weights,
        positive_mass_fraction,
    ).detach()


def safety_reserve_gate(
    positive_logits: torch.Tensor,
    positive_weights: torch.Tensor,
    threshold: float,
    *,
    global_step: int,
    total_optimizer_steps: int,
    warmup_fraction: float = 0.10,
    reserve_quantile: float = 0.01,
    reserve_margin: float = 0.0,
    gate_temperature: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a detached reserve gate and weighted low-tail logit margin."""
    values = positive_logits.float().reshape(-1)
    weights = positive_weights.float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("safety reserve requires at least one positive logit")
    if values.numel() != weights.numel():
        raise ValueError("positive logits and weights must align")
    if int(global_step) < 0 or int(total_optimizer_steps) <= 0:
        raise ValueError("optimizer step values must be valid")
    if not 0.0 <= float(warmup_fraction) <= 1.0:
        raise ValueError("warmup_fraction must lie in [0, 1]")
    if float(gate_temperature) <= 0.0:
        raise ValueError("gate_temperature must be positive")
    tau = float(threshold)
    if not 0.0 < tau < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")

    lower = _weighted_lower_quantile(values, weights, reserve_quantile)
    boundary = lower.new_tensor(math.log(tau / (1.0 - tau)))
    margin = (lower - boundary).detach()
    warmup_steps = int(math.ceil(float(warmup_fraction) * int(total_optimizer_steps)))
    if int(global_step) < warmup_steps:
        gate = margin.new_zeros(())
    else:
        gate = torch.sigmoid((margin - float(reserve_margin)) / float(gate_temperature)).detach()
    return gate, margin


def safety_reserve_operating_utility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    visible_hit_rates: torch.Tensor | None,
    instance_ids: torch.Tensor | None,
    instance_to_glb: torch.Tensor,
    glb_bytes: torch.Tensor,
    *,
    train_seed: int,
    global_step: int,
    total_optimizer_steps: int,
    split: str = "train",
    efficiency_logits: torch.Tensor | None = None,
    boundary_tail_weight: float = 0.30,
    boundary_tail_fraction: float = 0.01,
    rare_positive_weight: float = 1.0,
    negative_band_weight: float = 0.03,
    negative_tail_fraction: float = 0.02,
    glb_resource_weight: float = 0.05,
    threshold_temperature: float = 0.10,
    negative_band_shape: str = "sigmoid",
    group_temperature: float = 0.10,
    request_temperature: float = 0.10,
    reserve_quantile: float = 0.01,
    reserve_margin: float = 0.0,
    gate_temperature: float = 0.25,
    warmup_fraction: float = 0.10,
    threshold_sample_count: int = 2,
    threshold_anchor_period: int = 4,
    rvl_bce_positive_weight: float = 14.0,
    rvl_tversky_fn_weight: float = 7.0,
    rvl_count_weight: float = 0.10,
    rvl_fp_normalization: str = "positive",
    rvl_rank_weight: float = 0.45,
    rvl_rank_negative_top_k: int = 256,
    positive_tail_compactness_weight: float = 0.0,
    positive_tail_mass_fraction: float = 0.01,
    positive_tail_reference_quantile: float = 0.05,
    positive_tail_allowed_relative_gap: float = 0.20,
    positive_tail_temperature: float = 0.25,
    positive_tail_ramp_fraction: float = 0.0,
    tail_separation_weight: float = 0.0,
    tail_selection_logits: torch.Tensor | None = None,
    tail_positive_fraction: float = 0.01,
    tail_negative_fraction: float = 0.01,
    tail_margin: float = 0.30,
    tail_temperature: float = 0.25,
    tail_pose_cvar_fraction: float = 0.50,
    tail_pose_cvar_weight: float = 0.25,
    tail_positive_importance_mix: float = 0.80,
    tail_positive_gradient_scale: float = 1.0,
    coverage_tail_separation_weight: float = 0.0,
    coverage_tail_positive_fraction: float = 0.05,
    tail_objective_group: str = "safety",
    tail_ramp_fraction: float = 0.05,
    safety_boundary_excess_weight: float = 0.0,
    safety_boundary_scope: str = "batch",
    safety_boundary_positive_mass_fraction: float = 0.01,
    safety_boundary_negative_tail_fraction: float = 1.0,
    safety_boundary_margin: float = 0.0,
    safety_boundary_temperature: float = 0.20,
    safety_boundary_pose_cvar_fraction: float = 0.50,
    safety_boundary_pose_cvar_weight: float = 0.50,
    safety_boundary_ramp_fraction: float = 0.05,
    safety_boundary_reference_logit: torch.Tensor | float | None = None,
    rvl_budget_initial_scale: float = 1.0,
    rvl_budget_start_fraction: float = 0.0,
    rvl_budget_ramp_fraction: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Compute RVL safety, boundary protection, and reserve-gated efficiency."""
    if str(split) != "train":
        raise ValueError("safety-reserve operating utility loss is train-only")
    for name, value in (
        ("boundary_tail_weight", boundary_tail_weight),
        ("rare_positive_weight", rare_positive_weight),
        ("negative_band_weight", negative_band_weight),
        ("glb_resource_weight", glb_resource_weight),
        ("positive_tail_compactness_weight", positive_tail_compactness_weight),
        ("tail_separation_weight", tail_separation_weight),
        ("coverage_tail_separation_weight", coverage_tail_separation_weight),
        ("safety_boundary_excess_weight", safety_boundary_excess_weight),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if float(threshold_temperature) <= 0.0:
        raise ValueError("threshold_temperature must be positive")
    if str(negative_band_shape) not in {"sigmoid", "softplus"}:
        raise ValueError("negative_band_shape must be sigmoid or softplus")
    if str(tail_objective_group) not in {"safety", "efficiency"}:
        raise ValueError("tail objective group must be safety or efficiency")
    if not 0.0 < float(coverage_tail_positive_fraction) <= 1.0:
        raise ValueError("coverage tail positive fraction must lie in (0, 1]")
    if str(safety_boundary_scope) not in {"batch", "pose"}:
        raise ValueError("safety boundary scope must be batch or pose")

    flat_logits = logits.float().reshape(-1)
    flat_efficiency_logits = (
        flat_logits
        if efficiency_logits is None
        else torch.as_tensor(
            efficiency_logits,
            device=flat_logits.device,
            dtype=flat_logits.dtype,
        ).reshape(-1)
    )
    labels = target.float().reshape(-1)
    flat_tail_selection_logits = (
        None
        if tail_selection_logits is None
        else torch.as_tensor(
            tail_selection_logits,
            device=flat_logits.device,
            dtype=flat_logits.dtype,
        ).reshape(-1)
    )
    raw_weights = visible_weights.float().reshape(-1)
    if flat_logits.numel() == 0:
        raise ValueError("loss requires at least one candidate")
    if labels.numel() != flat_logits.numel() or raw_weights.numel() != flat_logits.numel():
        raise ValueError("logits, target, and visible_weights must align")
    if flat_efficiency_logits.numel() != flat_logits.numel():
        raise ValueError("efficiency_logits must align with logits")
    if (
        flat_tail_selection_logits is not None
        and flat_tail_selection_logits.numel() != flat_logits.numel()
    ):
        raise ValueError("tail_selection_logits must align with logits")
    if (
        not bool(torch.isfinite(flat_logits).all())
        or not bool(torch.isfinite(flat_efficiency_logits).all())
        or (
            flat_tail_selection_logits is not None
            and not bool(torch.isfinite(flat_tail_selection_logits).all())
        )
        or not bool(torch.isfinite(raw_weights).all())
    ):
        raise ValueError("loss inputs must be finite")

    hit_rates = (
        torch.ones_like(flat_logits)
        if visible_hit_rates is None
        else torch.as_tensor(visible_hit_rates, device=flat_logits.device).float().reshape(-1)
    )
    if hit_rates.numel() != flat_logits.numel() or not bool(torch.isfinite(hit_rates).all()):
        raise ValueError("visible_hit_rates must be finite and align with logits")
    hit_rates = hit_rates.clamp(0.0, 1.0)
    ids = (
        torch.arange(flat_logits.numel(), device=flat_logits.device, dtype=torch.long)
        if instance_ids is None
        else torch.as_tensor(instance_ids, device=flat_logits.device, dtype=torch.long).reshape(-1)
    )
    if ids.numel() != flat_logits.numel():
        raise ValueError("instance_ids must align with logits")
    mapping = torch.as_tensor(instance_to_glb, device=flat_logits.device, dtype=torch.long).reshape(-1)
    if mapping.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= mapping.numel():
        raise ValueError("instance_ids contains an ID outside instance_to_glb")
    candidate_glbs = mapping[ids]
    costs = torch.as_tensor(glb_bytes, device=flat_logits.device).float().reshape(-1)
    if candidate_glbs.numel() and (int(candidate_glbs.min()) < 0 or int(candidate_glbs.max()) >= costs.numel()):
        raise ValueError("instance_to_glb points outside glb_bytes")
    if not bool(torch.isfinite(costs).all()) or bool((costs < 0.0).any()):
        raise ValueError("glb_bytes must be finite and non-negative")

    # All downstream pose loops only need integer boundaries.  Reuse one CPU
    # metadata copy, including for the shared RVL control loss.
    pose_offsets_cpu = torch.as_tensor(pose_offsets, dtype=torch.long).detach().cpu().reshape(-1)
    normalized_weights = _normalize_pose_visible_weights(raw_weights, labels, pose_offsets_cpu)
    rvl_budget_scale = _linear_schedule_scale(
        global_step,
        total_optimizer_steps,
        initial_scale=rvl_budget_initial_scale,
        start_fraction=rvl_budget_start_fraction,
        ramp_fraction=rvl_budget_ramp_fraction,
    )
    rvl_loss, rvl_parts = rvl_strong_v2_visibility_loss(
        flat_logits,
        labels,
        pose_offsets_cpu,
        normalized_weights,
        torch.zeros_like(flat_logits),
        bce_positive_weight=rvl_bce_positive_weight,
        tversky_fn_weight=rvl_tversky_fn_weight,
        count_weight=rvl_count_weight,
        fp_normalization=rvl_fp_normalization,
        rank_weight=rvl_rank_weight,
        rank_negative_top_k=rvl_rank_negative_top_k,
        budget_scale=rvl_budget_scale,
    )
    zero = flat_logits.sum() * 0.0
    if float(positive_tail_compactness_weight) > 0.0:
        positive_tail_loss, positive_tail_parts = weighted_positive_tail_compactness_loss(
            flat_logits,
            labels,
            raw_weights,
            tail_mass_fraction=positive_tail_mass_fraction,
            reference_quantile=positive_tail_reference_quantile,
            allowed_relative_gap=positive_tail_allowed_relative_gap,
            temperature=positive_tail_temperature,
        )
        positive_tail_ramp_scale = _linear_schedule_scale(
            global_step,
            total_optimizer_steps,
            initial_scale=0.0,
            start_fraction=0.0,
            ramp_fraction=positive_tail_ramp_fraction,
        )
    else:
        positive_tail_loss = zero
        positive_tail_ramp_scale = 0.0
        positive_tail_parts = {
            "lossPositiveTailCompactness": zero,
            "positiveTailReferenceLogit": zero.detach(),
            "positiveTailMeanLogit": zero.detach(),
            "positiveTailReferenceGap": zero.detach(),
            "positiveTailRobustLogitScale": zero.detach(),
            "positiveTailViolationFraction": zero.detach(),
            "positiveTailSelectedCount": 0.0,
        }
    positive_tail_scaled = (
        float(positive_tail_compactness_weight)
        * float(positive_tail_ramp_scale)
        * positive_tail_loss
    )
    if float(tail_separation_weight) > 0.0:
        tail_loss, tail_parts = extreme_tail_separation_loss(
            flat_logits,
            labels,
            pose_offsets_cpu,
            normalized_weights,
            positive_tail_fraction=tail_positive_fraction,
            negative_tail_fraction=tail_negative_fraction,
            margin=tail_margin,
            temperature=tail_temperature,
            pose_cvar_fraction=tail_pose_cvar_fraction,
            pose_cvar_weight=tail_pose_cvar_weight,
            positive_importance_mix=tail_positive_importance_mix,
            positive_gradient_scale=tail_positive_gradient_scale,
            selection_logits=flat_tail_selection_logits,
        )
    else:
        tail_loss = zero
        tail_parts = {
            "lossExtremeTailSeparation": zero,
            "lossExtremeTailSeparationPoseMean": zero,
            "lossExtremeTailSeparationPoseCvar": zero,
            "extremeTailGap": zero.detach(),
            "extremeTailWorstGap": zero.detach(),
            "extremeTailViolationFraction": zero.detach(),
            "extremeTailPoseCount": 0.0,
            "extremeTailPositiveCount": 0.0,
            "extremeTailNegativeCount": 0.0,
        }
    tail_ramp_scale = (
        _linear_schedule_scale(
            global_step,
            total_optimizer_steps,
            initial_scale=0.0,
            start_fraction=0.0,
            ramp_fraction=tail_ramp_fraction,
        )
        if (
            float(tail_separation_weight) > 0.0
            or float(coverage_tail_separation_weight) > 0.0
        )
        else 0.0
    )
    tail_scaled = float(tail_separation_weight) * float(tail_ramp_scale) * tail_loss
    if float(coverage_tail_separation_weight) > 0.0:
        coverage_tail_loss, coverage_tail_parts = extreme_tail_separation_loss(
            flat_logits,
            labels,
            pose_offsets_cpu,
            normalized_weights,
            positive_tail_fraction=coverage_tail_positive_fraction,
            negative_tail_fraction=tail_negative_fraction,
            margin=tail_margin,
            temperature=tail_temperature,
            pose_cvar_fraction=tail_pose_cvar_fraction,
            pose_cvar_weight=tail_pose_cvar_weight,
            positive_importance_mix=0.0,
            positive_gradient_scale=tail_positive_gradient_scale,
        )
    else:
        coverage_tail_loss = zero
        coverage_tail_parts = {
            "lossExtremeTailSeparation": zero,
            "lossExtremeTailSeparationPoseMean": zero,
            "lossExtremeTailSeparationPoseCvar": zero,
            "extremeTailGap": zero.detach(),
            "extremeTailWorstGap": zero.detach(),
            "extremeTailViolationFraction": zero.detach(),
            "extremeTailPoseCount": 0.0,
            "extremeTailPositiveCount": 0.0,
            "extremeTailNegativeCount": 0.0,
        }
    coverage_tail_scaled = (
        float(coverage_tail_separation_weight)
        * float(tail_ramp_scale)
        * coverage_tail_loss
    )
    combined_tail_scaled = tail_scaled + coverage_tail_scaled
    if float(safety_boundary_excess_weight) > 0.0:
        safety_boundary_loss, safety_boundary_parts = safety_boundary_negative_excess_loss(
            flat_efficiency_logits,
            labels,
            pose_offsets_cpu,
            raw_weights,
            scope=safety_boundary_scope,
            positive_mass_fraction=safety_boundary_positive_mass_fraction,
            negative_tail_fraction=safety_boundary_negative_tail_fraction,
            margin=safety_boundary_margin,
            temperature=safety_boundary_temperature,
            pose_cvar_fraction=safety_boundary_pose_cvar_fraction,
            pose_cvar_weight=safety_boundary_pose_cvar_weight,
            reference_boundary_logit=safety_boundary_reference_logit,
        )
        safety_boundary_ramp_scale = _linear_schedule_scale(
            global_step,
            total_optimizer_steps,
            initial_scale=0.0,
            start_fraction=0.0,
            ramp_fraction=safety_boundary_ramp_fraction,
        )
    else:
        safety_boundary_loss = zero
        safety_boundary_ramp_scale = 0.0
        safety_boundary_parts = {
            "lossSafetyBoundaryNegativeExcess": zero,
            "lossSafetyBoundaryNegativeExcessMean": zero,
            "lossSafetyBoundaryNegativeExcessPoseCvar": zero,
            "safetyBoundaryBatchLogit": zero.detach(),
            "safetyBoundaryReferenceLogit": zero.detach(),
            "safetyBoundaryLogit": zero.detach(),
            "safetyBoundaryViolationFraction": zero.detach(),
            "safetyBoundaryGroupCount": 0.0,
            "safetyBoundarySelectedNegativeCount": 0.0,
        }
    safety_boundary_scaled = (
        float(safety_boundary_excess_weight)
        * float(safety_boundary_ramp_scale)
        * safety_boundary_loss
    )
    thresholds = sample_train_operating_thresholds(
        train_seed,
        global_step,
        sample_count=threshold_sample_count,
        anchor_period=threshold_anchor_period,
        split=split,
    )
    scores = torch.sigmoid(flat_logits)
    boundary_terms: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []
    resource_terms: list[torch.Tensor] = []
    ungated_terms: list[torch.Tensor] = []
    gated_terms: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    boundary_count = 0
    negative_count = 0
    negative_saturation_terms: list[torch.Tensor] = []
    no_demand_glb_count = flat_logits.new_zeros(())

    for _pose, start, end in _pose_slices(pose_offsets_cpu, flat_logits.numel()):
        local_labels = labels[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        local_weights = normalized_weights[start:end]
        if bool(positive.any()):
            risk = (
                local_weights[positive]
                * (1.0 - scores[start:end][positive])
                * (1.0 + float(rare_positive_weight) * (1.0 - torch.sqrt(hit_rates[start:end][positive])))
            )
            boundary, count = _upper_tail_cvar(risk, boundary_tail_fraction)
            boundary_terms.append(boundary)
            boundary_count += count

        local_glbs = candidate_glbs[start:end]
        unique_glbs, inverse_glbs = torch.unique(local_glbs, sorted=True, return_inverse=True)
        positive_count = torch.zeros(
            (unique_glbs.numel(),), device=flat_logits.device, dtype=torch.long
        )
        positive_count.scatter_add_(0, inverse_glbs, positive.to(dtype=torch.long))
        no_demand_mask = positive_count == 0
        no_demand_glb_count = no_demand_glb_count + no_demand_mask.sum().to(flat_logits.dtype)
        stacked_group_logits = _group_topk_smooth_max(
            flat_efficiency_logits[start:end],
            inverse_glbs,
            unique_glbs.numel(),
            top_k=8,
            temperature=group_temperature,
        )
        stacked_costs = (
            torch.log1p(costs[unique_glbs])
            if unique_glbs.numel()
            else flat_logits.new_zeros((0,))
        )

        for threshold in thresholds:
            if bool(negative.any()):
                boundary_logit = flat_logits.new_tensor(math.log(threshold / (1.0 - threshold)))
                normalized_margin = (
                    flat_efficiency_logits[start:end][negative] - boundary_logit
                ) / float(threshold_temperature)
                if negative_band_shape == "sigmoid":
                    soft_keep = torch.sigmoid(normalized_margin)
                    negative_saturation_terms.append((soft_keep > 0.999).float().mean())
                else:
                    soft_keep = (
                        F.softplus(normalized_margin) * float(threshold_temperature)
                    )
                    negative_saturation_terms.append(torch.zeros((), device=soft_keep.device))
                negative_loss, count = _upper_tail_cvar(soft_keep, negative_tail_fraction)
                negative_count += count
            else:
                negative_loss = zero
            negative_terms.append(negative_loss)

            selected_costs = stacked_costs[no_demand_mask]
            if selected_costs.numel():
                requests = soft_request_probability(
                    stacked_group_logits, threshold, temperature=request_temperature
                )
                resource_loss = (
                    requests[no_demand_mask] * selected_costs
                ).sum() / selected_costs.sum().clamp_min(1e-12)
            else:
                resource_loss = zero
            resource_terms.append(resource_loss)

            if bool(positive.any()):
                gate, margin = safety_reserve_gate(
                    flat_logits[start:end][positive],
                    local_weights[positive],
                    threshold,
                    global_step=global_step,
                    total_optimizer_steps=total_optimizer_steps,
                    warmup_fraction=warmup_fraction,
                    reserve_quantile=reserve_quantile,
                    reserve_margin=reserve_margin,
                    gate_temperature=gate_temperature,
                )
            else:
                gate, margin = zero.detach(), zero.detach()
            ungated = float(negative_band_weight) * negative_loss + float(glb_resource_weight) * resource_loss
            ungated_terms.append(ungated)
            gated_terms.append(gate * ungated)
            gates.append(gate)
            margins.append(margin)

    boundary_loss = torch.stack(boundary_terms).mean() if boundary_terms else zero
    negative_loss = torch.stack(negative_terms).mean() if negative_terms else zero
    resource_loss = torch.stack(resource_terms).mean() if resource_terms else zero
    ungated_efficiency = torch.stack(ungated_terms).mean() if ungated_terms else zero
    operating_efficiency_loss = torch.stack(gated_terms).mean() if gated_terms else zero
    tail_safety = (
        combined_tail_scaled if str(tail_objective_group) == "safety" else zero
    )
    tail_efficiency = (
        combined_tail_scaled if str(tail_objective_group) == "efficiency" else zero
    )
    efficiency_loss = operating_efficiency_loss + tail_efficiency + safety_boundary_scaled
    safety_loss = (
        rvl_loss
        + float(boundary_tail_weight) * boundary_loss
        + positive_tail_scaled
        + tail_safety
    )
    total = safety_loss + efficiency_loss
    gate_mean = torch.stack(gates).mean() if gates else zero.detach()
    margin_mean = torch.stack(margins).mean() if margins else zero.detach()
    negative_saturation = (
        torch.stack(negative_saturation_terms).mean()
        if negative_saturation_terms
        else zero.detach()
    )
    parts: dict[str, torch.Tensor | float] = {
        **{f"rvl_{key}": value for key, value in rvl_parts.items()},
        "lossSafetyRvl": rvl_loss,
        "lossBoundaryTail": boundary_loss,
        "lossBoundaryTailScaled": float(boundary_tail_weight) * boundary_loss,
        **positive_tail_parts,
        "lossPositiveTailCompactnessScaled": positive_tail_scaled,
        "positiveTailCompactnessRampScale": float(positive_tail_ramp_scale),
        **tail_parts,
        "lossImportanceTailSeparationScaled": tail_scaled,
        **{
            f"coverageTail{key[0].upper()}{key[1:]}": value
            for key, value in coverage_tail_parts.items()
        },
        "lossUniformCoverageTailSeparationScaled": coverage_tail_scaled,
        "lossExtremeTailSeparationScaled": combined_tail_scaled,
        "lossExtremeTailSeparationSafetyScaled": tail_safety,
        "lossExtremeTailSeparationEfficiencyScaled": tail_efficiency,
        "extremeTailRampScale": float(tail_ramp_scale),
        **safety_boundary_parts,
        "lossSafetyBoundaryNegativeExcessScaled": safety_boundary_scaled,
        "safetyBoundaryRampScale": float(safety_boundary_ramp_scale),
        "rvlBudgetScheduleScale": float(rvl_budget_scale),
        "lossSafety": safety_loss,
        "lossNegativeBand": negative_loss,
        "lossNoDemandGlbResource": resource_loss,
        "lossEfficiencyUngated": ungated_efficiency,
        "lossOperatingEfficiency": operating_efficiency_loss,
        "lossEfficiency": efficiency_loss,
        "lossSafetyReserveOperatingUtility": total,
        "safetyReserveGate": gate_mean,
        "safetyReserveMargin": margin_mean,
        "operatingThresholds": flat_logits.new_tensor(thresholds).detach(),
        "boundaryTailPositiveCount": float(boundary_count),
        "thresholdNegativeTailCount": float(negative_count),
        "negativeBandSaturationFraction": negative_saturation,
        "noDemandGlbCount": no_demand_glb_count,
        "normalizedVisibleWeightMean": normalized_weights.mean(),
        "normalizedVisibleWeightMax": normalized_weights.max(),
    }
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("safety-reserve operating utility loss is non-finite")
    for name, value in parts.items():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite loss part: {name}")
    return total, parts


def _gradient_device(groups: Sequence[GradientList]) -> torch.device:
    devices = {value.device for group in groups for value in group if value is not None}
    if len(devices) > 1:
        raise ValueError("all gradient tensors must be on the same device")
    return next(iter(devices), torch.device("cpu"))


def _gradient_norm_sq(gradients: GradientList, device: torch.device) -> torch.Tensor:
    result = torch.zeros((), device=device, dtype=torch.float64)
    for value in gradients:
        if value is not None:
            result = result + value.double().square().sum()
    return result


def _gradient_dot(
    left: GradientList,
    right: GradientList,
    device: torch.device,
) -> torch.Tensor:
    result = torch.zeros((), device=device, dtype=torch.float64)
    for left_value, right_value in zip(left, right, strict=True):
        if left_value is not None and right_value is not None:
            result = result + (left_value.double() * right_value.double()).sum()
    return result


def _project_one_gradient_group(
    safety_gradients: GradientList,
    auxiliary_gradients: GradientList,
    *,
    norm_cap_ratio: float,
    device: torch.device,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    if float(norm_cap_ratio) < 0.0:
        raise ValueError("gradient norm cap ratios must be non-negative")
    dot = _gradient_dot(safety_gradients, auxiliary_gradients, device)
    safety_norm_sq = _gradient_norm_sq(safety_gradients, device)
    auxiliary_norm_sq = _gradient_norm_sq(auxiliary_gradients, device)
    apply_projection = bool(dot.detach() < 0.0 and safety_norm_sq.detach() > 0.0)
    coefficient = dot / safety_norm_sq.clamp_min(1e-12) if apply_projection else dot.new_zeros(())
    projected: list[torch.Tensor | None] = []
    for safety, auxiliary in zip(safety_gradients, auxiliary_gradients, strict=True):
        if auxiliary is None:
            projected.append(None)
        elif apply_projection and safety is not None:
            projected.append(auxiliary - coefficient.to(auxiliary.dtype) * safety)
        else:
            projected.append(auxiliary)
    for _attempt in range(3):
        dot_after_projection = _gradient_dot(safety_gradients, projected, device)
        if bool(dot_after_projection.detach() >= 0.0 or safety_norm_sq.detach() <= 0.0):
            break
        # FP32 gradient tensors can leave a negative residual after the
        # analytically exact projection.  Correct toward a small positive
        # relative margin, then verify the represented tensors again.
        projected_norm_sq_before_correction = _gradient_norm_sq(projected, device)
        margin = (
            torch.sqrt(safety_norm_sq)
            * torch.sqrt(projected_norm_sq_before_correction)
            * 1e-5
        )
        correction = (dot_after_projection - margin) / safety_norm_sq.clamp_min(1e-24)
        projected = [
            value
            if value is None or safety is None
            else value - correction.to(value.dtype) * safety
            for safety, value in zip(safety_gradients, projected, strict=True)
        ]
    projected_norm_sq = _gradient_norm_sq(projected, device)
    safety_norm = torch.sqrt(safety_norm_sq)
    projected_norm = torch.sqrt(projected_norm_sq)
    cap = float(norm_cap_ratio) * safety_norm
    scale = torch.minimum(
        projected_norm.new_ones(()), cap / projected_norm.clamp_min(1e-12)
    )
    projected = [None if value is None else value * scale.to(value.dtype) for value in projected]
    dot_after = _gradient_dot(safety_gradients, projected, device)
    fallback_zeroed = False
    if bool(dot_after.detach() < 0.0):
        # The projection is computed in float64 but the optimizer receives
        # float32 tensors.  With large, nearly cancelling gradients, casting
        # the correction back to float32 can leave a small negative residual.
        # Dropping this auxiliary group for the current step is the only
        # representation-safe fallback: it gives an exactly zero safety dot
        # product instead of silently applying a safety-conflicting update.
        projected = [None if value is None else torch.zeros_like(value) for value in projected]
        dot_after = _gradient_dot(safety_gradients, projected, device)
        fallback_zeroed = True
    if bool(dot_after.detach() < 0.0):
        raise FloatingPointError(
            f"auxiliary gradient remains safety-conflicting after zero fallback: {float(dot_after.detach().cpu())}"
        )
    return projected, {
        "dotBeforeProjection": float(dot.detach().cpu()),
        "dotAfterProjection": float(dot_after.detach().cpu()),
        "normBeforeProjection": float(torch.sqrt(auxiliary_norm_sq).detach().cpu()),
        "normAfterProjection": float(torch.sqrt(projected_norm_sq).detach().cpu()),
        "normAfterCap": float((projected_norm * scale).detach().cpu()),
        "projectionApplied": float(apply_projection),
        "projectionFallbackZeroed": float(fallback_zeroed),
        "capScale": float(scale.detach().cpu()),
        "normCapRatio": float(norm_cap_ratio),
    }


def project_operating_utility_gradient_groups(
    safety_gradients: GradientList,
    relation_gradients: GradientList,
    schedule_gradients: GradientList,
    efficiency_gradients: GradientList,
    *,
    relation_norm_cap: float = 1.0,
    schedule_norm_cap: float = 1.0,
    efficiency_norm_cap: float = 0.25,
) -> tuple[dict[str, list[torch.Tensor | None]], dict[str, float]]:
    """Project relation, scheduling, and efficiency gradients against safety.

    Each auxiliary group is protected independently. Consequently every
    projected group, and their sum, has a non-negative dot product with the
    safety gradient. Norm caps are expressed as fractions of the safety norm.
    """
    groups = (safety_gradients, relation_gradients, schedule_gradients, efficiency_gradients)
    lengths = {len(group) for group in groups}
    if len(lengths) != 1:
        raise ValueError("all gradient groups must have equal length")
    for name, gradients in zip(
        ("safety", "relation", "schedule", "efficiency"),
        groups,
        strict=True,
    ):
        invalid = [
            index
            for index, value in enumerate(gradients)
            if value is not None and not bool(torch.isfinite(value).all())
        ]
        if invalid:
            raise FloatingPointError(
                f"{name} gradient group contains non-finite tensors at parameter indices {invalid[:16]}"
            )
    device = _gradient_device(groups)
    safety_norm = float(torch.sqrt(_gradient_norm_sq(safety_gradients, device)).detach().cpu())
    projected: dict[str, list[torch.Tensor | None]] = {"safety": list(safety_gradients)}
    statistics: dict[str, float] = {"safetyGradientNorm": safety_norm}
    for name, gradients, cap in (
        ("relation", relation_gradients, relation_norm_cap),
        ("schedule", schedule_gradients, schedule_norm_cap),
        ("efficiency", efficiency_gradients, efficiency_norm_cap),
    ):
        values, group_stats = _project_one_gradient_group(
            safety_gradients, gradients, norm_cap_ratio=cap, device=device
        )
        projected[name] = values
        statistics.update({f"{name}Gradient{key[0].upper()}{key[1:]}": value for key, value in group_stats.items()})
    if not all(math.isfinite(value) for value in statistics.values()):
        raise FloatingPointError("gradient projection statistics are non-finite")
    return projected, statistics


__all__ = [
    "OPERATING_THRESHOLD_ANCHORS",
    "OPERATING_THRESHOLD_RANGE",
    "bounded_topk_smooth_max",
    "candidate_boundary_hard_negative_loss",
    "extreme_tail_separation_loss",
    "instance_exposure_balanced_bce_loss",
    "project_operating_utility_gradient_groups",
    "same_instance_cross_view_rank_loss",
    "safety_boundary_negative_excess_loss",
    "safety_reserve_gate",
    "safety_reserve_operating_utility_loss",
    "sample_train_operating_thresholds",
    "soft_request_probability",
    "view_residual_tail_regularizers",
]
