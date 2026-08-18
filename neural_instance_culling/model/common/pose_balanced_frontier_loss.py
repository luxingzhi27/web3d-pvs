"""Pose-balanced visibility classification with a dynamic safety frontier."""
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


def _validate_inputs(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    positive_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_logits = logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    weights = positive_weights.float().reshape(-1)
    if flat_logits.numel() == 0 or not (
        flat_logits.numel() == labels.numel() == weights.numel()
    ):
        raise ValueError("visibility logits, labels, and positive weights must align")
    if not bool(
        torch.isfinite(flat_logits).all()
        and torch.isfinite(labels).all()
        and torch.isfinite(weights).all()
    ):
        raise ValueError("visibility loss inputs must be finite")
    if bool((weights < 0.0).any()):
        raise ValueError("positive weights must be non-negative")
    # Validate offsets before any loss is accumulated.
    tuple(_pose_slices(pose_offsets, flat_logits.numel()))
    return flat_logits, labels, weights.clamp_min(0.0)


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
    """Average equally normalized positive/negative BCE within each pose."""
    flat_logits, labels, weights = _validate_inputs(
        logits, target, pose_offsets, positive_weights
    )
    pose_losses: list[torch.Tensor] = []
    positive_losses: list[torch.Tensor] = []
    negative_losses: list[torch.Tensor] = []
    positive_count = 0
    negative_count = 0
    mixed_pose_count = 0

    for _pose, start, end in _pose_slices(pose_offsets, flat_logits.numel()):
        local_logits = flat_logits[start:end]
        local_labels = labels[start:end]
        local_weights = weights[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        class_losses: list[torch.Tensor] = []
        if bool(positive.any()):
            compressed = _compressed_positive_weights(
                local_weights[positive],
                floor=positive_importance_floor,
                power=positive_importance_power,
            )
            mass = compressed / compressed.sum().clamp_min(1e-12)
            loss = (
                mass
                * F.binary_cross_entropy_with_logits(
                    local_logits[positive],
                    torch.ones_like(local_logits[positive]),
                    reduction="none",
                )
            ).sum()
            positive_losses.append(loss)
            class_losses.append(loss)
            positive_count += int(positive.sum())
        if bool(negative.any()):
            loss = F.binary_cross_entropy_with_logits(
                local_logits[negative],
                torch.zeros_like(local_logits[negative]),
            )
            negative_losses.append(loss)
            class_losses.append(loss)
            negative_count += int(negative.sum())
        if len(class_losses) == 2:
            mixed_pose_count += 1
        if class_losses:
            pose_losses.append(torch.stack(class_losses).mean())

    zero = flat_logits.sum() * 0.0
    loss = torch.stack(pose_losses).mean() if pose_losses else zero
    positive_mean = torch.stack(positive_losses).mean() if positive_losses else zero
    negative_mean = torch.stack(negative_losses).mean() if negative_losses else zero
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("pose-balanced BCE is non-finite")
    return loss, {
        "lossPoseBalancedBce": loss,
        "lossPoseBalancedPositiveBce": positive_mean,
        "lossPoseBalancedNegativeBce": negative_mean,
        "poseBalancedPoseCount": float(len(pose_losses)),
        "poseBalancedMixedPoseCount": float(mixed_pose_count),
        "poseBalancedPositiveCount": float(positive_count),
        "poseBalancedNegativeCount": float(negative_count),
    }


def _weighted_low_tail(
    selection_logits: torch.Tensor,
    weights: torch.Tensor,
    *,
    mass_fraction: float,
    count_cap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < float(mass_fraction) <= 1.0:
        raise ValueError("frontier positive mass fraction must lie in (0, 1]")
    if int(count_cap) <= 0:
        raise ValueError("frontier positive count cap must be positive")
    values = selection_logits.float().reshape(-1)
    mass_source = weights.float().reshape(-1).clamp_min(0.0)
    if values.numel() == 0 or values.numel() != mass_source.numel():
        raise ValueError("frontier positive values and weights must align")
    if not bool((mass_source > 0.0).any()):
        mass_source = torch.ones_like(mass_source)
    order = torch.argsort(values.detach(), descending=False, stable=True)
    ordered_mass = mass_source[order]
    target_mass = ordered_mass.sum() * float(mass_fraction)
    cumulative_before = torch.cumsum(ordered_mass, dim=0) - ordered_mass
    selected_mass = torch.minimum(
        ordered_mass,
        torch.clamp(target_mass - cumulative_before, min=0.0),
    )
    selected = torch.nonzero(selected_mass > 0.0, as_tuple=False).reshape(-1)
    selected = selected[: int(count_cap)]
    indices = order[selected]
    masses = selected_mass[selected]
    return indices, masses / masses.sum().clamp_min(1e-12)


def dynamic_weighted_safety_frontier_loss(
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
    temperature: float = 0.25,
    positive_importance_floor: float = 0.5,
    positive_importance_power: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Separate low-score important positives from high-score negatives per pose."""
    if not 0.0 < float(negative_top_fraction) <= 1.0:
        raise ValueError("frontier negative fraction must lie in (0, 1]")
    if int(negative_count_cap) <= 0:
        raise ValueError("frontier negative count cap must be positive")
    if float(margin) < 0.0 or float(temperature) <= 0.0:
        raise ValueError("frontier margin and temperature are invalid")
    flat_logits, labels, weights = _validate_inputs(
        logits, target, pose_offsets, positive_weights
    )
    pose_losses: list[torch.Tensor] = []
    pose_gaps: list[torch.Tensor] = []
    violation_fractions: list[torch.Tensor] = []
    positive_count = 0
    negative_count = 0
    pair_count = 0

    for _pose, start, end in _pose_slices(pose_offsets, flat_logits.numel()):
        local_logits = flat_logits[start:end]
        local_labels = labels[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        if not bool(positive.any()) or not bool(negative.any()):
            continue

        positive_logits = local_logits[positive]
        positive_source_weights = weights[start:end][positive]
        selected_positive, selected_mass = _weighted_low_tail(
            positive_logits.detach(),
            positive_source_weights,
            mass_fraction=positive_mass_fraction,
            count_cap=positive_count_cap,
        )
        negative_logits = local_logits[negative]
        selected_negative_count = min(
            int(negative_count_cap),
            max(1, int(math.ceil(float(negative_top_fraction) * negative_logits.numel()))),
        )
        selected_negative = torch.topk(
            negative_logits.detach(),
            k=selected_negative_count,
            largest=True,
            sorted=False,
        ).indices
        positive_values = positive_logits[selected_positive]
        negative_values = negative_logits[selected_negative]
        if positive_values.numel() == 0 or negative_values.numel() == 0:
            continue

        importance = _compressed_positive_weights(
            positive_source_weights[selected_positive],
            floor=positive_importance_floor,
            power=positive_importance_power,
        )
        positive_pair_mass = selected_mass * importance
        positive_pair_mass = positive_pair_mass / positive_pair_mass.sum().clamp_min(1e-12)
        pair_violation = (
            negative_values[:, None] - positive_values[None, :] + float(margin)
        ) / float(temperature)
        pair_loss = F.softplus(pair_violation)
        pose_loss = (pair_loss * positive_pair_mass[None, :]).sum(dim=1).mean()
        pair_gap = positive_values[None, :].detach() - negative_values[:, None].detach()
        weighted_gap = (pair_gap * positive_pair_mass[None, :]).sum(dim=1)
        pose_losses.append(pose_loss)
        pose_gaps.append(weighted_gap.mean())
        violation_fractions.append((pair_gap < float(margin)).float().mean())
        positive_count += int(positive_values.numel())
        negative_count += int(negative_values.numel())
        pair_count += int(pair_gap.numel())

    zero = flat_logits.sum() * 0.0
    loss = torch.stack(pose_losses).mean() if pose_losses else zero
    mean_gap = torch.stack(pose_gaps).mean() if pose_gaps else zero.detach()
    worst_gap = torch.stack(pose_gaps).min() if pose_gaps else zero.detach()
    violation = (
        torch.stack(violation_fractions).mean()
        if violation_fractions
        else zero.detach()
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("dynamic weighted safety-frontier loss is non-finite")
    return loss, {
        "lossDynamicSafetyFrontier": loss,
        "dynamicSafetyFrontierPoseCount": float(len(pose_losses)),
        "dynamicSafetyFrontierPositiveCount": float(positive_count),
        "dynamicSafetyFrontierNegativeCount": float(negative_count),
        "dynamicSafetyFrontierPairCount": float(pair_count),
        "dynamicSafetyFrontierMeanGap": mean_gap,
        "dynamicSafetyFrontierWorstPoseMeanGap": worst_gap,
        "dynamicSafetyFrontierViolationFraction": violation,
    }


def pose_balanced_frontier_visibility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    frontier_weight: float = 0.2,
    positive_mass_fraction: float = 0.005,
    positive_count_cap: int = 64,
    negative_top_fraction: float = 0.01,
    negative_count_cap: int = 256,
    margin: float = 0.5,
    temperature: float = 0.25,
    positive_importance_floor: float = 0.5,
    positive_importance_power: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return the complete two-term visibility objective and diagnostics."""
    if float(frontier_weight) < 0.0:
        raise ValueError("frontier loss weight must be non-negative")
    balanced, balanced_parts = pose_balanced_binary_cross_entropy(
        logits,
        target,
        pose_offsets,
        positive_weights,
        positive_importance_floor=positive_importance_floor,
        positive_importance_power=positive_importance_power,
    )
    frontier, frontier_parts = dynamic_weighted_safety_frontier_loss(
        logits,
        target,
        pose_offsets,
        positive_weights,
        positive_mass_fraction=positive_mass_fraction,
        positive_count_cap=positive_count_cap,
        negative_top_fraction=negative_top_fraction,
        negative_count_cap=negative_count_cap,
        margin=margin,
        temperature=temperature,
        positive_importance_floor=positive_importance_floor,
        positive_importance_power=positive_importance_power,
    )
    total = balanced + float(frontier_weight) * frontier
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("pose-balanced frontier visibility loss is non-finite")
    return total, {
        "lossVisibility": total,
        "lossVisibilityPoseBalanced": balanced,
        "lossVisibilityFrontierWeighted": float(frontier_weight) * frontier,
        "frontierLossWeight": float(frontier_weight),
        **balanced_parts,
        **frontier_parts,
    }


__all__ = [
    "dynamic_weighted_safety_frontier_loss",
    "pose_balanced_binary_cross_entropy",
    "pose_balanced_frontier_visibility_loss",
]
