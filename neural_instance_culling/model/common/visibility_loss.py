"""Pose-balanced visibility learning with an RVL safety guard and tail margin.

The objective deliberately does not add the complete historical RVL loss to
the pose-balanced frontier loss.  Both contain classification and ranking
terms, and stacking them repeatedly pushes every positive score upward.  This
module assigns one responsibility to each term instead:

* pose-balanced BCE learns the ordinary visible/invisible decision;
* a one-sided, visible-weighted recall guard supplies the RVL safety signal;
* one shared hard-tail selection separates difficult positive and negative
  logits without adding a second classifier or runtime asset.
"""
from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn.functional as F


def _pose_slices(
    pose_offsets: torch.Tensor,
    item_count: int,
) -> Iterator[tuple[int, int, int]]:
    offsets = torch.as_tensor(pose_offsets, dtype=torch.long).detach().cpu().reshape(-1)
    if offsets.numel() < 2:
        raise ValueError("pose_offsets must contain at least two entries")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(item_count):
        raise ValueError("pose_offsets must start at zero and cover every item")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("pose_offsets must be non-decreasing")
    for pose_index in range(offsets.numel() - 1):
        start = int(offsets[pose_index])
        end = int(offsets[pose_index + 1])
        if end > start:
            yield pose_index, start, end


def _flat_inputs(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weights: torch.Tensor,
    pose_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    weights = positive_weights.float().reshape(-1).clamp_min(0.0)
    if values.numel() == 0 or not (
        values.numel() == labels.numel() == weights.numel()
    ):
        raise ValueError("logits, labels, and positive weights must align")
    if not bool(
        torch.isfinite(values).all()
        and torch.isfinite(labels).all()
        and torch.isfinite(weights).all()
    ):
        raise ValueError("visibility loss inputs must be finite")
    tuple(_pose_slices(pose_offsets, values.numel()))
    return values, labels, weights


def _positive_mass(weights: torch.Tensor) -> torch.Tensor:
    local = weights.float().reshape(-1).clamp_min(0.0)
    if local.numel() == 0:
        return local
    if not bool((local > 0.0).any()):
        return torch.ones_like(local) / float(local.numel())
    return local / local.sum().clamp_min(1e-12)


def _compressed_positive_weights(
    weights: torch.Tensor,
    *,
    floor: float,
    power: float,
) -> torch.Tensor:
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError("positive importance floor must lie in [0, 1]")
    if not 0.0 < float(power) <= 1.0:
        raise ValueError("positive importance power must lie in (0, 1]")
    local = weights.float().reshape(-1).clamp_min(0.0)
    if local.numel() == 0:
        return local
    maximum = local.max()
    relative = (
        local / maximum.clamp_min(1e-12)
        if bool(maximum > 0.0)
        else torch.ones_like(local)
    )
    return float(floor) + (1.0 - float(floor)) * relative.pow(float(power))


def pose_balanced_binary_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    positive_importance_floor: float = 0.5,
    positive_importance_power: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Average a class-balanced BCE inside each pose, then across poses."""
    raw_weights = positive_weights.float().reshape(-1)
    if not bool(torch.isfinite(raw_weights).all()) or bool((raw_weights < 0.0).any()):
        raise ValueError("positive weights must be finite and non-negative")
    values, labels, weights = _flat_inputs(
        logits, target, positive_weights, pose_offsets
    )
    losses: list[torch.Tensor] = []
    positive_terms: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []
    positive_count = 0
    negative_count = 0
    mixed_pose_count = 0
    for _pose, start, end in _pose_slices(pose_offsets, values.numel()):
        local_values = values[start:end]
        local_labels = labels[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        terms: list[torch.Tensor] = []
        if bool(positive.any()):
            importance = _compressed_positive_weights(
                weights[start:end][positive],
                floor=positive_importance_floor,
                power=positive_importance_power,
            )
            mass = importance / importance.sum().clamp_min(1e-12)
            positive_loss = (
                mass
                * F.binary_cross_entropy_with_logits(
                    local_values[positive],
                    torch.ones_like(local_values[positive]),
                    reduction="none",
                )
            ).sum()
            positive_terms.append(positive_loss)
            terms.append(positive_loss)
            positive_count += int(positive.sum())
        if bool(negative.any()):
            negative_loss = F.binary_cross_entropy_with_logits(
                local_values[negative],
                torch.zeros_like(local_values[negative]),
            )
            negative_terms.append(negative_loss)
            terms.append(negative_loss)
            negative_count += int(negative.sum())
        if len(terms) == 2:
            mixed_pose_count += 1
        if terms:
            losses.append(torch.stack(terms).mean())
    zero = values.sum() * 0.0
    total = torch.stack(losses).mean() if losses else zero
    positive_mean = torch.stack(positive_terms).mean() if positive_terms else zero
    negative_mean = torch.stack(negative_terms).mean() if negative_terms else zero
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("pose-balanced BCE is non-finite")
    return total, {
        "lossPoseBalancedBce": total,
        "lossPoseBalancedPositiveBce": positive_mean,
        "lossPoseBalancedNegativeBce": negative_mean,
        "poseBalancedPoseCount": float(len(losses)),
        "poseBalancedMixedPoseCount": float(mixed_pose_count),
        "poseBalancedPositiveCount": float(positive_count),
        "poseBalancedNegativeCount": float(negative_count),
    }


def _smooth_excess(value: torch.Tensor, boundary: float, temperature: float) -> torch.Tensor:
    return float(temperature) * F.softplus(
        (value - float(boundary)) / float(temperature)
    )


def pose_weighted_recall_guard_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    target_recall: float = 0.99,
    temperature: float = 0.05,
    pose_cvar_fraction: float = 0.25,
    pose_cvar_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Apply RVL only as a one-sided soft weighted-recall constraint.

    The aggregate term follows the formal weighted-recall definition.  A
    smaller pose-CVaR term prevents all of the reserve from being supplied by
    a few easy view cells, but ordinary pose recall is not treated as the
    formal safety gate.
    """
    if not 0.0 < float(target_recall) < 1.0:
        raise ValueError("RVL recall target must lie in (0, 1)")
    if float(temperature) <= 0.0:
        raise ValueError("RVL recall temperature must be positive")
    if not 0.0 < float(pose_cvar_fraction) <= 1.0:
        raise ValueError("RVL pose CVaR fraction must lie in (0, 1]")
    if float(pose_cvar_weight) < 0.0:
        raise ValueError("RVL pose CVaR weight must be non-negative")
    values, labels, weights = _flat_inputs(
        logits, target, positive_weights, pose_offsets
    )
    scores = torch.sigmoid(values)
    positive = labels > 0.5
    zero = values.sum() * 0.0
    if not bool(positive.any()):
        return zero, {
            "lossRvlWeightedRecallGuard": zero,
            "lossRvlAggregateRecallGuard": zero,
            "lossRvlPoseCvarRecallGuard": zero,
            "softAggregateWeightedRecall": values.new_ones(()),
            "softWorstPoseWeightedRecall": values.new_ones(()),
            "rvlRecallGuardPoseCount": 0.0,
        }

    aggregate_mass = _positive_mass(weights[positive])
    aggregate_recall = (aggregate_mass * scores[positive]).sum()
    allowed_miss = 1.0 - float(target_recall)
    aggregate_guard = _smooth_excess(
        1.0 - aggregate_recall,
        allowed_miss,
        temperature,
    )

    pose_misses: list[torch.Tensor] = []
    for _pose, start, end in _pose_slices(pose_offsets, values.numel()):
        local_positive = labels[start:end] > 0.5
        if not bool(local_positive.any()):
            continue
        mass = _positive_mass(weights[start:end][local_positive])
        local_recall = (mass * scores[start:end][local_positive]).sum()
        pose_misses.append(1.0 - local_recall)
    if pose_misses:
        ordered = torch.stack(pose_misses)
        tail_count = max(1, int(math.ceil(float(pose_cvar_fraction) * ordered.numel())))
        pose_cvar_miss = torch.topk(ordered, k=tail_count, largest=True).values.mean()
        pose_cvar_guard = _smooth_excess(
            pose_cvar_miss,
            allowed_miss,
            temperature,
        )
        worst_pose_recall = 1.0 - ordered.max()
    else:
        pose_cvar_guard = zero
        worst_pose_recall = values.new_ones(())
    total = aggregate_guard + float(pose_cvar_weight) * pose_cvar_guard
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("RVL weighted-recall guard is non-finite")
    return total, {
        "lossRvlWeightedRecallGuard": total,
        "lossRvlAggregateRecallGuard": aggregate_guard,
        "lossRvlPoseCvarRecallGuard": pose_cvar_guard,
        "softAggregateWeightedRecall": aggregate_recall.detach(),
        "softWorstPoseWeightedRecall": worst_pose_recall.detach(),
        "rvlRecallGuardPoseCount": float(len(pose_misses)),
    }


def _weighted_low_tail_indices(
    logits: torch.Tensor,
    weights: torch.Tensor,
    *,
    mass_fraction: float,
    count_cap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mass = weights.float().reshape(-1).clamp_min(0.0)
    if not bool((mass > 0.0).any()):
        mass = torch.ones_like(mass)
    order = torch.argsort(logits.detach(), descending=False, stable=True)
    ordered_mass = mass[order]
    target_mass = ordered_mass.sum() * float(mass_fraction)
    cumulative_before = torch.cumsum(ordered_mass, dim=0) - ordered_mass
    selected_mass = torch.minimum(
        ordered_mass,
        torch.clamp(target_mass - cumulative_before, min=0.0),
    )
    selected = torch.nonzero(selected_mass > 0.0, as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        selected = torch.zeros((1,), dtype=torch.long, device=logits.device)
        selected_mass = selected_mass.clone()
        selected_mass[0] = ordered_mass[0]
    selected = selected[: int(count_cap)]
    indices = order[selected]
    normalized_mass = selected_mass[selected]
    return indices, normalized_mass / normalized_mass.sum().clamp_min(1e-12)


def shared_tail_logit_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    positive_mass_fraction: float = 0.005,
    positive_count_cap: int = 64,
    negative_top_fraction: float = 0.01,
    negative_count_cap: int = 256,
    margin: float = 0.5,
    logit_temperature: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Separate difficult weighted positives from high-scoring negatives."""
    if not 0.0 < float(positive_mass_fraction) <= 1.0:
        raise ValueError("tail positive mass fraction must lie in (0, 1]")
    if int(positive_count_cap) <= 0:
        raise ValueError("tail positive count cap must be positive")
    if not 0.0 < float(negative_top_fraction) <= 1.0:
        raise ValueError("tail negative fraction must lie in (0, 1]")
    if int(negative_count_cap) <= 0:
        raise ValueError("tail negative count cap must be positive")
    if float(margin) < 0.0 or float(logit_temperature) <= 0.0:
        raise ValueError("tail margins and temperatures are invalid")
    values, labels, weights = _flat_inputs(
        logits, target, positive_weights, pose_offsets
    )

    logit_terms: list[torch.Tensor] = []
    gap_terms: list[torch.Tensor] = []
    selected_positive_count = 0
    selected_negative_count = 0
    active_pair_count = 0
    for _pose, start, end in _pose_slices(pose_offsets, values.numel()):
        local_labels = labels[start:end]
        positive_mask = local_labels > 0.5
        negative_mask = ~positive_mask
        if int(positive_mask.sum()) < 2 or not bool(negative_mask.any()):
            continue
        positive_values = values[start:end][positive_mask]
        positive_weights_local = weights[start:end][positive_mask]
        tail_positive, tail_mass = _weighted_low_tail_indices(
            positive_values,
            positive_weights_local,
            mass_fraction=positive_mass_fraction,
            count_cap=positive_count_cap,
        )
        negative_values = values[start:end][negative_mask]
        negative_count = min(
            int(negative_count_cap),
            max(1, int(math.ceil(float(negative_top_fraction) * negative_values.numel()))),
        )
        tail_negative = torch.topk(
            negative_values.detach(),
            k=negative_count,
            largest=True,
            sorted=False,
        ).indices
        positive_tail_values = positive_values[tail_positive]
        negative_tail_values = negative_values[tail_negative]
        violation = (
            negative_tail_values[:, None].detach()
            - positive_tail_values[None, :].detach()
            + float(margin)
        ) > 0.0
        if not bool(violation.any()):
            continue

        pair_scaled = (
            negative_tail_values[:, None]
            - positive_tail_values[None, :]
            + float(margin)
        ) / float(logit_temperature)
        pair_weight = violation.to(pair_scaled.dtype) * tail_mass[None, :]
        logit_terms.append(
            (F.softplus(pair_scaled) * pair_weight).sum()
            / pair_weight.sum().clamp_min(1e-12)
        )
        pair_gap = positive_tail_values[None, :] - negative_tail_values[:, None]
        gap_terms.append(
            (pair_gap.detach() * pair_weight).sum()
            / pair_weight.sum().clamp_min(1e-12)
        )

        selected_positive_count += int(tail_positive.numel())
        selected_negative_count += int(tail_negative.numel())
        active_pair_count += int(violation.sum())

    zero = values.sum() * 0.0
    total = torch.stack(logit_terms).mean() if logit_terms else zero
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("shared tail margin loss is non-finite")
    mean_gap = torch.stack(gap_terms).mean() if gap_terms else zero.detach()
    return total, {
        "lossSharedTailMargin": total,
        "sharedTailPoseCount": float(len(logit_terms)),
        "sharedTailPositiveCount": float(selected_positive_count),
        "sharedTailNegativeCount": float(selected_negative_count),
        "sharedTailActivePairCount": float(active_pair_count),
        "sharedTailMeanActiveGap": mean_gap,
    }


def pose_balanced_rvl_contrastive_visibility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    recall_guard_weight: float = 0.30,
    recall_target: float = 0.99,
    recall_temperature: float = 0.05,
    recall_pose_cvar_fraction: float = 0.25,
    recall_pose_cvar_weight: float = 0.25,
    separation_weight: float = 0.20,
    separation_scale: float = 1.0,
    positive_mass_fraction: float = 0.005,
    positive_count_cap: int = 64,
    negative_top_fraction: float = 0.01,
    negative_count_cap: int = 256,
    margin: float = 0.5,
    logit_temperature: float = 0.25,
    positive_importance_floor: float = 0.5,
    positive_importance_power: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return the single registered visibility objective for the V4 mainline."""
    if min(float(recall_guard_weight), float(separation_weight), float(separation_scale)) < 0.0:
        raise ValueError("integrated visibility loss weights must be non-negative")
    balanced, balanced_parts = pose_balanced_binary_cross_entropy(
        logits,
        target,
        pose_offsets,
        positive_weights,
        positive_importance_floor=positive_importance_floor,
        positive_importance_power=positive_importance_power,
    )
    recall_guard, recall_parts = pose_weighted_recall_guard_loss(
        logits,
        target,
        pose_offsets,
        positive_weights,
        target_recall=recall_target,
        temperature=recall_temperature,
        pose_cvar_fraction=recall_pose_cvar_fraction,
        pose_cvar_weight=recall_pose_cvar_weight,
    )
    separation, separation_parts = shared_tail_logit_margin_loss(
        logits,
        target,
        pose_offsets,
        positive_weights,
        positive_mass_fraction=positive_mass_fraction,
        positive_count_cap=positive_count_cap,
        negative_top_fraction=negative_top_fraction,
        negative_count_cap=negative_count_cap,
        margin=margin,
        logit_temperature=logit_temperature,
    )
    weighted_recall_guard = float(recall_guard_weight) * recall_guard
    weighted_separation = (
        float(separation_weight) * float(separation_scale) * separation
    )
    total = balanced + weighted_recall_guard + weighted_separation
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("integrated V4 visibility loss is non-finite")
    return total, {
        "lossVisibility": total,
        "lossVisibilityPoseBalanced": balanced,
        "lossVisibilityRvlRecallGuardWeighted": weighted_recall_guard,
        "lossVisibilitySharedTailWeighted": weighted_separation,
        "rvlRecallGuardWeight": float(recall_guard_weight),
        "sharedTailSeparationWeight": float(separation_weight),
        "sharedTailSeparationScale": float(separation_scale),
        **balanced_parts,
        **recall_parts,
        **separation_parts,
    }


__all__ = [
    "pose_balanced_binary_cross_entropy",
    "pose_balanced_rvl_contrastive_visibility_loss",
    "pose_weighted_recall_guard_loss",
    "shared_tail_logit_margin_loss",
]
