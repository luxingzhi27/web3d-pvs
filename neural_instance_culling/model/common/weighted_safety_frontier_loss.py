"""Pairwise loss for a train-frozen weighted safety frontier.

The frontier is mined before refinement and is passed in as instance IDs per
pose.  This module never mines members from the current logits: it only uses
``torch.isin`` to locate the frozen IDs in the current batch.  The objective
therefore trains the positive-minus-negative logit gap without introducing an
absolute-logit anchor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


FrontierByPose = Mapping[int, tuple[torch.Tensor, torch.Tensor]]


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc
    return tensor


def _flat_real(value: Any, name: str, device: torch.device) -> torch.Tensor:
    tensor = _as_tensor(value, name)
    if tensor.ndim == 0 or tensor.ndim > 2 or (
        tensor.ndim == 2 and 1 not in tensor.shape
    ):
        raise ValueError(f"{name} must be a vector or singleton-column vector")
    if torch.is_complex(tensor) or tensor.dtype == torch.bool:
        raise ValueError(f"{name} must be a real-valued vector")
    result = tensor.to(device=device, dtype=torch.float32).reshape(-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _flat_integer(value: Any, name: str, device: torch.device) -> torch.Tensor:
    tensor = _as_tensor(value, name)
    if tensor.ndim == 0 or tensor.ndim > 2 or (
        tensor.ndim == 2 and 1 not in tensor.shape
    ):
        raise ValueError(f"{name} must be a vector or singleton-column vector")
    if torch.is_complex(tensor) or tensor.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer values")
    if torch.is_floating_point(tensor):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")
        rounded = tensor.round()
        if not bool(torch.equal(tensor, rounded)):
            raise ValueError(f"{name} must contain integer values")
        tensor = rounded
    return tensor.to(device=device, dtype=torch.long).reshape(-1)


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_pose_offsets(value: Any, item_count: int, device: torch.device) -> list[int]:
    offsets = _flat_integer(value, "pose_offsets", device)
    if offsets.numel() < 2:
        raise ValueError("pose_offsets must contain a start and an end")
    values = [int(item) for item in offsets.detach().cpu().tolist()]
    if values[0] != 0 or values[-1] != int(item_count):
        raise ValueError("pose_offsets must start at zero and end at the candidate count")
    if any(
        start < 0 or end < start or end > int(item_count)
        for start, end in zip(values, values[1:])
    ):
        raise ValueError("pose_offsets must be non-decreasing within the candidate range")
    return values


def _frontier_key(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("frontier pose keys must be integers")
    try:
        result = int(value)
        exact = float(value) == float(result)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("frontier pose keys must be integers") from exc
    if not exact:
        raise ValueError("frontier pose keys must be integers")
    return result


def _prepare_frontier(
    frontier_by_pose: Any,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if not isinstance(frontier_by_pose, Mapping):
        raise ValueError("frontier_by_pose must be a mapping")

    prepared: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for raw_pose, members in frontier_by_pose.items():
        pose = _frontier_key(raw_pose)
        if not isinstance(members, tuple) or len(members) != 2:
            raise ValueError(
                "frontier_by_pose values must be (positive_ids, negative_ids) tuples"
            )
        positive_ids = _flat_integer(members[0], "frontier positive IDs", device).detach()
        negative_ids = _flat_integer(members[1], "frontier negative IDs", device).detach()
        if bool(torch.isin(positive_ids, negative_ids).any()):
            raise ValueError("frontier positive and negative IDs must be disjoint")
        prepared[pose] = (positive_ids, negative_ids)
    return prepared


def _normalized_positive_importance(
    positive_weights: torch.Tensor,
    power: float,
) -> torch.Tensor:
    importance = positive_weights.pow(float(power))
    if not bool(torch.isfinite(importance).all()):
        raise ValueError("positive importance weights must be finite")
    if not bool((importance > 0.0).any()):
        importance = torch.ones_like(importance)
    return importance / importance.sum().clamp_min(torch.finfo(importance.dtype).tiny)


def _upper_cvar(values: torch.Tensor, fraction: float) -> torch.Tensor:
    count = min(
        int(values.numel()),
        max(1, int(math.ceil(float(fraction) * int(values.numel())))),
    )
    # The selected tail is a risk-weighting decision, not a trainable path.
    selected = torch.topk(values.detach(), k=count, sorted=False).indices
    return values[selected].mean()


def _empty_result(zero: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    detached_zero = zero.detach()
    return zero, {
        "lossWeightedSafetyFrontier": zero,
        "lossPoseMean": zero,
        "lossPoseCvar": zero,
        "lossBatchGlobalPair": zero,
        "poseCount": 0.0,
        "positiveCount": 0.0,
        "negativeCount": 0.0,
        "pairCount": 0.0,
        "meanGap": detached_zero,
        "worstGap": detached_zero,
        "globalPairCount": 0.0,
    }


def weighted_safety_frontier_pair_loss(
    logits: torch.Tensor,
    instance_ids: torch.Tensor,
    visible_weights: torch.Tensor,
    pose_indices: torch.Tensor,
    pose_offsets: torch.Tensor,
    frontier_by_pose: FrontierByPose,
    *,
    margin: float = 0.20,
    temperature: float = 0.25,
    positive_importance_power: float = 1.0,
    pose_cvar_fraction: float = 0.25,
    pose_cvar_weight: float = 0.50,
    batch_global_pair_weight: float = 0.0,
    global_pair_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return the fixed-frontier pair loss and detached frontier diagnostics.

    ``frontier_by_pose[pose]`` contains the positive and negative instance IDs
    frozen from a train-only checkpoint.  The IDs are matched to each pose's
    candidate slice with ``torch.isin``; current logits never determine
    membership.  For a pose with both classes, each pair contributes

    ``temperature * softplus((z_negative - z_positive + margin) / temperature)``.

    Positive terms are normalized by ``visible_weights **
    positive_importance_power`` inside each pose.  The total is the mean of
    pose losses plus an upper-tail pose CVaR term and an optional pair loss
    over all fixed members in the batch.  The global term normalizes positive
    importance over the whole batch.  If all selected positive weights are
    zero, uniform positive weights are used so the frontier remains trainable.

    The returned ``meanGap`` is the mean pose gap (weighted positive logit
    minus mean negative logit), and ``worstGap`` is the smallest such gap.
    Empty or non-contributing frontiers return a differentiable zero.
    """
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a torch.Tensor")
    device = logits.device
    flat_logits = _flat_real(logits, "logits", device)

    flat_instance_ids = _flat_integer(instance_ids, "instance_ids", device)
    flat_weights = _flat_real(visible_weights, "visible_weights", device).detach()
    if flat_instance_ids.numel() != flat_logits.numel():
        raise ValueError("instance_ids must align with logits")
    if flat_weights.numel() != flat_logits.numel():
        raise ValueError("visible_weights must align with logits")
    if bool((flat_weights < 0.0).any()):
        raise ValueError("visible_weights must be non-negative")

    flat_pose_indices = _flat_integer(pose_indices, "pose_indices", device)
    offsets = _validate_pose_offsets(pose_offsets, flat_logits.numel(), device)
    if flat_pose_indices.numel() != len(offsets) - 1:
        raise ValueError("pose_indices must contain one pose ID per pose offset interval")

    margin_value = _finite_scalar(margin, "margin")
    temperature_value = _finite_scalar(temperature, "temperature")
    importance_power = _finite_scalar(
        positive_importance_power, "positive_importance_power"
    )
    cvar_fraction = _finite_scalar(pose_cvar_fraction, "pose_cvar_fraction")
    cvar_weight = _finite_scalar(pose_cvar_weight, "pose_cvar_weight")
    if global_pair_weight is not None:
        alias_weight = _finite_scalar(global_pair_weight, "global_pair_weight")
        if batch_global_pair_weight != 0.0 and float(batch_global_pair_weight) != alias_weight:
            raise ValueError(
                "batch_global_pair_weight and global_pair_weight disagree"
            )
        batch_global_pair_weight = alias_weight
    global_weight = _finite_scalar(
        batch_global_pair_weight, "batch_global_pair_weight"
    )
    if margin_value < 0.0:
        raise ValueError("margin must be non-negative")
    if temperature_value <= 0.0:
        raise ValueError("temperature must be positive")
    if importance_power <= 0.0:
        raise ValueError("positive_importance_power must be positive")
    if not 0.0 < cvar_fraction <= 1.0:
        raise ValueError("pose_cvar_fraction must lie in (0, 1]")
    if cvar_weight < 0.0:
        raise ValueError("pose_cvar_weight must be non-negative")
    if global_weight < 0.0:
        raise ValueError("batch_global_pair_weight must be non-negative")

    frontier = _prepare_frontier(frontier_by_pose, device)
    zero = flat_logits.sum() * 0.0
    if not frontier or flat_logits.numel() == 0:
        return _empty_result(zero)

    pose_losses: list[torch.Tensor] = []
    pose_gaps: list[torch.Tensor] = []
    positive_indices: list[torch.Tensor] = []
    negative_indices: list[torch.Tensor] = []
    positive_count = 0
    negative_count = 0
    pair_count = 0

    pose_ids = [int(value) for value in flat_pose_indices.detach().cpu().tolist()]
    for pose_slot, (start, end) in enumerate(zip(offsets, offsets[1:])):
        members = frontier.get(pose_ids[pose_slot])
        if members is None or end <= start:
            continue
        positive_ids, negative_ids = members
        local_ids = flat_instance_ids[start:end]
        # The masks depend only on detached IDs and the fixed frontier.
        positive_mask = torch.isin(local_ids, positive_ids.detach())
        negative_mask = torch.isin(local_ids, negative_ids.detach())
        if bool((positive_mask & negative_mask).any()):
            raise ValueError("a batch instance ID is both a fixed positive and negative")
        if not bool(positive_mask.any()) or not bool(negative_mask.any()):
            continue

        local_positive = torch.nonzero(positive_mask, as_tuple=False).reshape(-1) + start
        local_negative = torch.nonzero(negative_mask, as_tuple=False).reshape(-1) + start
        positive_values = flat_logits[local_positive]
        negative_values = flat_logits[local_negative]
        positive_importance = _normalized_positive_importance(
            flat_weights[local_positive], importance_power
        )
        violation = (
            negative_values[:, None]
            - positive_values[None, :]
            + margin_value
        ) / temperature_value
        if not bool(torch.isfinite(violation).all()):
            raise ValueError("frontier pair gaps must be finite")
        pair_values = temperature_value * F.softplus(violation)
        pose_loss = (
            pair_values * positive_importance[None, :]
        ).sum() / float(local_negative.numel())
        pose_losses.append(pose_loss)
        pose_gaps.append(
            (
                (positive_values * positive_importance).sum()
                - negative_values.mean()
            ).detach()
        )
        positive_indices.append(local_positive)
        negative_indices.append(local_negative)
        positive_count += int(local_positive.numel())
        negative_count += int(local_negative.numel())
        pair_count += int(local_positive.numel() * local_negative.numel())

    if not pose_losses:
        return _empty_result(zero)

    pose_values = torch.stack(pose_losses)
    loss_pose_mean = pose_values.mean()
    loss_pose_cvar = _upper_cvar(pose_values, cvar_fraction)

    if global_weight > 0.0:
        global_positive = torch.cat(positive_indices)
        global_negative = torch.cat(negative_indices)
        global_positive_importance = _normalized_positive_importance(
            flat_weights[global_positive], importance_power
        )
        global_violation = (
            flat_logits[global_negative][:, None]
            - flat_logits[global_positive][None, :]
            + margin_value
        ) / temperature_value
        if not bool(torch.isfinite(global_violation).all()):
            raise ValueError("global frontier pair gaps must be finite")
        global_pair_values = temperature_value * F.softplus(global_violation)
        loss_global_pair = (
            global_pair_values * global_positive_importance[None, :]
        ).sum() / float(global_negative.numel())
        global_pair_count = float(
            global_positive.numel() * global_negative.numel()
        )
    else:
        loss_global_pair = zero
        global_pair_count = 0.0

    total = (
        loss_pose_mean
        + cvar_weight * loss_pose_cvar
        + global_weight * loss_global_pair
    )
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("weighted safety frontier loss is non-finite")

    gaps = torch.stack(pose_gaps)
    diagnostics: dict[str, torch.Tensor | float] = {
        "lossWeightedSafetyFrontier": total,
        "lossPoseMean": loss_pose_mean,
        "lossPoseCvar": loss_pose_cvar,
        "lossBatchGlobalPair": loss_global_pair,
        "poseCount": float(len(pose_losses)),
        "positiveCount": float(positive_count),
        "negativeCount": float(negative_count),
        "pairCount": float(pair_count),
        "meanGap": gaps.mean(),
        "worstGap": gaps.min(),
        "globalPairCount": global_pair_count,
        "poseCvarFraction": float(cvar_fraction),
        "poseCvarWeight": float(cvar_weight),
        "batchGlobalPairWeight": float(global_weight),
        "positiveImportancePower": float(importance_power),
    }
    return total, diagnostics


weighted_safety_frontier_loss = weighted_safety_frontier_pair_loss


__all__ = [
    "FrontierByPose",
    "weighted_safety_frontier_loss",
    "weighted_safety_frontier_pair_loss",
]
