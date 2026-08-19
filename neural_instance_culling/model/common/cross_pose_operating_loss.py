"""Cross-pose visibility loss for a recall-constrained operating boundary."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import torch
import torch.nn.functional as F


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _pose_slices(
    pose_offsets: torch.Tensor,
    item_count: int,
) -> Iterator[tuple[int, int]]:
    offsets = torch.as_tensor(pose_offsets, dtype=torch.long).detach().cpu().reshape(-1)
    if offsets.numel() < 2:
        raise ValueError("pose_offsets must contain at least two entries")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(item_count):
        raise ValueError("pose_offsets must start at zero and cover every item")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("pose_offsets must be non-decreasing")
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        if int(end) > int(start):
            yield int(start), int(end)


def _validated_inputs(
    logits: torch.Tensor,
    target: torch.Tensor,
    visible_weights: torch.Tensor,
    pose_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.as_tensor(logits).float().reshape(-1)
    labels = torch.as_tensor(target, device=scores.device).float().reshape(-1)
    weights = torch.as_tensor(
        visible_weights, device=scores.device
    ).float().reshape(-1)
    if scores.numel() == 0 or not (
        scores.numel() == labels.numel() == weights.numel()
    ):
        raise ValueError("logits, target, and visible_weights must align and be non-empty")
    if not bool(
        torch.isfinite(scores).all()
        and torch.isfinite(labels).all()
        and torch.isfinite(weights).all()
    ):
        raise ValueError("cross-pose operating inputs must be finite")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("target must contain only binary values")
    if bool((weights < 0.0).any()):
        raise ValueError("visible_weights must be non-negative")
    tuple(_pose_slices(pose_offsets, scores.numel()))
    return scores, labels.detach(), weights.detach()


def _positive_importance(
    weights: torch.Tensor,
    *,
    floor: float,
    power: float,
) -> torch.Tensor:
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError("positive importance floor must lie in [0, 1]")
    if not 0.0 < float(power) <= 1.0:
        raise ValueError("positive importance power must lie in (0, 1]")
    maximum = weights.max() if weights.numel() else weights.new_zeros(())
    relative = (
        weights / maximum.clamp_min(1e-12)
        if bool(maximum > 0.0)
        else torch.ones_like(weights)
    )
    return float(floor) + (1.0 - float(floor)) * relative.pow(float(power))


def pose_asymmetric_binary_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    *,
    positive_class_fraction: float = 0.25,
    positive_importance_floor: float = 0.5,
    positive_importance_power: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Normalize classes per pose while assigning a configurable positive share."""
    scores, labels, weights = _validated_inputs(
        logits, target, visible_weights, pose_offsets
    )
    positive_share = _finite_scalar(
        positive_class_fraction, "positive_class_fraction"
    )
    if not 0.0 < positive_share < 1.0:
        raise ValueError("positive_class_fraction must lie in (0, 1)")

    pose_losses: list[torch.Tensor] = []
    positive_losses: list[torch.Tensor] = []
    negative_losses: list[torch.Tensor] = []
    positive_count = 0
    negative_count = 0
    for start, end in _pose_slices(pose_offsets, scores.numel()):
        local_scores = scores[start:end]
        local_labels = labels[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        if not bool(positive.any()) or not bool(negative.any()):
            continue
        importance = _positive_importance(
            weights[start:end][positive],
            floor=positive_importance_floor,
            power=positive_importance_power,
        )
        importance = importance / importance.sum().clamp_min(1e-12)
        positive_loss = (
            importance
            * F.binary_cross_entropy_with_logits(
                local_scores[positive],
                torch.ones_like(local_scores[positive]),
                reduction="none",
            )
        ).sum()
        negative_loss = F.binary_cross_entropy_with_logits(
            local_scores[negative],
            torch.zeros_like(local_scores[negative]),
        )
        pose_losses.append(
            positive_share * positive_loss + (1.0 - positive_share) * negative_loss
        )
        positive_losses.append(positive_loss)
        negative_losses.append(negative_loss)
        positive_count += int(positive.sum())
        negative_count += int(negative.sum())

    if not pose_losses:
        raise ValueError("each cross-pose batch must contain a mixed-label pose")
    loss = torch.stack(pose_losses).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("pose-asymmetric BCE became non-finite")
    return loss, {
        "lossPoseAsymmetricBce": loss,
        "lossPoseAsymmetricPositiveBce": torch.stack(positive_losses).mean(),
        "lossPoseAsymmetricNegativeBce": torch.stack(negative_losses).mean(),
        "poseAsymmetricPoseCount": float(len(pose_losses)),
        "poseAsymmetricPositiveCount": float(positive_count),
        "poseAsymmetricNegativeCount": float(negative_count),
        "poseAsymmetricPositiveClassFraction": positive_share,
    }


@dataclass
class CrossPoseOperatingDualState:
    """Autograd-independent multiplier for the weighted-recall constraint."""

    multiplier: float = 1.0
    learning_rate: float = 0.05
    maximum: float = 20.0

    def __post_init__(self) -> None:
        self.multiplier = _finite_scalar(self.multiplier, "multiplier")
        self.learning_rate = _finite_scalar(self.learning_rate, "learning_rate")
        self.maximum = _finite_scalar(self.maximum, "maximum")
        if self.multiplier < 0.0 or self.learning_rate < 0.0 or self.maximum < 0.0:
            raise ValueError("dual state values must be non-negative")
        if self.multiplier > self.maximum:
            raise ValueError("dual multiplier must not exceed maximum")

    def update(self, raw_violation: torch.Tensor | float) -> None:
        violation = float(torch.as_tensor(raw_violation).detach().cpu())
        if not math.isfinite(violation):
            raise ValueError("raw_violation must be finite")
        updated = self.multiplier + self.learning_rate * violation
        if not math.isfinite(updated):
            raise FloatingPointError("dual multiplier update became non-finite")
        self.multiplier = min(self.maximum, max(0.0, updated))

    def as_dict(self) -> dict[str, float]:
        return {
            "multiplier": float(self.multiplier),
            "learningRate": float(self.learning_rate),
            "maximum": float(self.maximum),
        }


def cross_pose_recall_constrained_operating_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    boundary_logit: torch.Tensor,
    *,
    positive_class_fraction: float = 0.25,
    positive_importance_floor: float = 0.5,
    positive_importance_power: float = 0.5,
    operating_weight: float = 1.0,
    weighted_recall_target: float = 0.995,
    temperature: float = 0.25,
    hard_negative_fraction: float = 0.01,
    hard_negative_count_cap: int = 512,
    hard_negative_mix: float = 0.5,
    augmented_penalty: float = 10.0,
    dual_multiplier: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Optimize false positives under one batch-global weighted-recall boundary.

    Unlike an unconstrained Lagrangian, the primal safety penalty uses only the
    positive part of the violation.  Exceeding the recall target therefore
    cannot keep rewarding higher positive scores after the constraint is met.
    """
    scores, labels, weights = _validated_inputs(
        logits, target, visible_weights, pose_offsets
    )
    boundary = torch.as_tensor(
        boundary_logit, device=scores.device, dtype=torch.float32
    ).reshape(-1)
    if boundary.numel() != 1 or not bool(torch.isfinite(boundary).all()):
        raise ValueError("boundary_logit must be one finite scalar")
    boundary = boundary.reshape(())

    operating_scale = _finite_scalar(operating_weight, "operating_weight")
    recall_target = _finite_scalar(weighted_recall_target, "weighted_recall_target")
    temperature_value = _finite_scalar(temperature, "temperature")
    negative_fraction = _finite_scalar(
        hard_negative_fraction, "hard_negative_fraction"
    )
    tail_mix = _finite_scalar(hard_negative_mix, "hard_negative_mix")
    penalty = _finite_scalar(augmented_penalty, "augmented_penalty")
    dual = _finite_scalar(dual_multiplier, "dual_multiplier")
    if operating_scale < 0.0 or penalty < 0.0 or dual < 0.0:
        raise ValueError("operating weights and penalties must be non-negative")
    if not 0.0 < recall_target < 1.0:
        raise ValueError("weighted_recall_target must lie in (0, 1)")
    if temperature_value <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 < negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must lie in (0, 1]")
    if int(hard_negative_count_cap) <= 0:
        raise ValueError("hard_negative_count_cap must be positive")
    if not 0.0 <= tail_mix <= 1.0:
        raise ValueError("hard_negative_mix must lie in [0, 1]")

    base_loss, base_parts = pose_asymmetric_binary_cross_entropy(
        scores,
        labels,
        pose_offsets,
        weights,
        positive_class_fraction=positive_class_fraction,
        positive_importance_floor=positive_importance_floor,
        positive_importance_power=positive_importance_power,
    )
    positive = labels > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("cross-pose operating loss requires both classes")
    positive_mass = weights[positive].sum()
    if not bool(positive_mass > 0.0):
        raise ValueError("positive visible_weights must have positive mass")

    soft_keep = torch.sigmoid((scores - boundary) / temperature_value)
    soft_weighted_recall = (
        soft_keep[positive] * weights[positive]
    ).sum() / positive_mass
    aggregate_soft_fpr = soft_keep[negative].mean()
    pose_soft_fprs = []
    for start, end in _pose_slices(pose_offsets, scores.numel()):
        local_negative = negative[start:end]
        if bool(local_negative.any()):
            pose_soft_fprs.append(soft_keep[start:end][local_negative].mean())
    if not pose_soft_fprs:
        raise ValueError("cross-pose operating loss requires a negative candidate")
    pose_macro_soft_fpr = torch.stack(pose_soft_fprs).mean()
    negative_count = int(negative.sum())
    hard_count = min(
        int(hard_negative_count_cap),
        negative_count,
        max(1, int(math.ceil(negative_fraction * negative_count))),
    )
    hard_negative_keep = torch.topk(
        soft_keep[negative], hard_count, largest=True, sorted=False
    ).values
    hard_negative_fpr = hard_negative_keep.mean()
    false_positive_objective = (
        (1.0 - tail_mix) * pose_macro_soft_fpr + tail_mix * hard_negative_fpr
    )

    raw_violation = recall_target - soft_weighted_recall
    active_violation = F.relu(raw_violation)
    safety_penalty = dual * active_violation + 0.5 * penalty * active_violation.square()
    operating_objective = false_positive_objective + safety_penalty
    loss = base_loss + operating_scale * operating_objective
    values = (
        loss,
        soft_weighted_recall,
        pose_macro_soft_fpr,
        aggregate_soft_fpr,
        hard_negative_fpr,
        raw_violation,
        active_violation,
        safety_penalty,
    )
    if not bool(torch.isfinite(torch.stack([value.reshape(()) for value in values])).all()):
        raise FloatingPointError("cross-pose operating loss became non-finite")

    return loss, {
        **base_parts,
        "lossCrossPoseOperating": loss,
        "lossCrossPoseOperatingWeighted": operating_scale * operating_objective,
        "crossPoseSoftWeightedRecall": soft_weighted_recall,
        "crossPosePoseMacroSoftFpr": pose_macro_soft_fpr,
        "crossPoseAggregateSoftFpr": aggregate_soft_fpr,
        "crossPoseHardNegativeFpr": hard_negative_fpr,
        "crossPoseFalsePositiveObjective": false_positive_objective,
        "crossPoseWeightedRecallRawViolation": raw_violation,
        "crossPoseWeightedRecallActiveViolation": active_violation,
        "crossPoseSafetyPenalty": safety_penalty,
        "crossPoseBoundaryLogit": boundary.detach(),
        "crossPoseBoundaryProbability": torch.sigmoid(boundary.detach()),
        "crossPoseDualMultiplier": dual,
        "crossPoseHardNegativeCount": float(hard_count),
        "crossPosePositiveWeightMass": positive_mass,
    }


__all__ = [
    "CrossPoseOperatingDualState",
    "cross_pose_recall_constrained_operating_loss",
    "pose_asymmetric_binary_cross_entropy",
]
