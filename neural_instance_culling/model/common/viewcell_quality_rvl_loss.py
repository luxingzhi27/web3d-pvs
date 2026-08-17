"""Threshold-aligned extension of the registered RVL visibility loss.

RVL remains the safety objective.  The v2 additions put coverage weights into
a pose-local, resolution-independent range and evaluate the resource penalty
around the low probability thresholds used by calibration instead of around a
fixed probability of 0.5.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from common.safety_constraint_utility_loss import rvl_strong_v2_visibility_loss


def _pose_slices(offsets: torch.Tensor):
    for pose in range(max(0, int(offsets.numel()) - 1)):
        start, end = int(offsets[pose]), int(offsets[pose + 1])
        if end > start:
            yield pose, start, end


def noisy_or(probabilities: torch.Tensor, inverse: torch.Tensor, group_count: int) -> torch.Tensor:
    """Compute one soft request probability per GLB group."""
    result = probabilities.new_ones((int(group_count),))
    if probabilities.numel() == 0:
        return 1.0 - result
    result.scatter_reduce_(0, inverse.long(), 1.0 - probabilities, reduce="prod", include_self=True)
    return 1.0 - result


def normalize_pose_visible_weights(
    visible_weights: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    *,
    kappa: float = 99.0,
) -> torch.Tensor:
    """Normalize Color-ID coverage weights independently inside each pose.

    The raw values are useful for reporting but are not comparable across
    poses. The largest positive value supplies the pose scale, and the
    logarithm limits gradient dominance by one large surface. Negative
    candidates receive zero weight.
    """
    if float(kappa) <= 0.0:
        raise ValueError("kappa must be positive")
    weights = visible_weights.float().reshape(-1).clamp_min(0.0)
    labels = target.float().reshape(-1)
    if weights.numel() != labels.numel():
        raise ValueError("visible_weights and target must have equal length")
    normalized = torch.zeros_like(weights)
    denominator = torch.log1p(
        torch.as_tensor(float(kappa), device=weights.device, dtype=weights.dtype)
    )
    for _pose, start, end in _pose_slices(pose_offsets):
        positive = labels[start:end] > 0.5
        if not bool(positive.any()):
            continue
        maximum = weights[start:end][positive].max().clamp_min(1e-6)
        relative = (weights[start:end] / maximum).clamp(0.0, 1.0)
        normalized[start:end] = torch.log1p(float(kappa) * relative) / denominator
        normalized[start:end] = normalized[start:end] * positive.to(normalized.dtype)
    return normalized


def soft_keep_at_threshold(
    logits: torch.Tensor,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    """Return a smooth keep probability centered at a probability threshold."""
    tau = float(threshold)
    if not 0.0 < tau < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    logit_threshold = torch.log(
        torch.as_tensor(tau / (1.0 - tau), device=logits.device, dtype=logits.dtype)
    )
    return torch.sigmoid((logits.float() - logit_threshold) / float(temperature))


def quality_tail_rvl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    visible_hit_rates: torch.Tensor | None,
    instance_ids: torch.Tensor | None,
    instance_to_glb: torch.Tensor,
    glb_cost_norm: torch.Tensor,
    *,
    quality_weight: float = 0.5,
    resource_weight: float = 0.05,
    quality_temperature: float = 0.10,
    tail_fraction: float = 0.01,
    rare_positive_weight: float = 1.0,
    request_thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20),
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return RVL plus view-cell tail and low-threshold GLB penalties.

    ``visible_weights`` is normalized only inside this function. Formal
    reports continue to use the original coverage values. The caller can
    project the resource gradient against the safety gradient.
    """
    if not request_thresholds:
        raise ValueError("request_thresholds must not be empty")
    flat_logits = logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    raw_weights = visible_weights.float().reshape(-1).clamp_min(0.0)
    if flat_logits.numel() != labels.numel() or labels.numel() != raw_weights.numel():
        raise ValueError("logits, target, and visible_weights must align")
    ids = (
        torch.arange(flat_logits.numel(), device=flat_logits.device, dtype=torch.long)
        if instance_ids is None
        else torch.as_tensor(instance_ids, dtype=torch.long, device=flat_logits.device).reshape(-1)
    )
    if ids.numel() != flat_logits.numel():
        raise ValueError("instance_ids must align with flattened logits")
    mapping = torch.as_tensor(instance_to_glb, dtype=torch.long, device=flat_logits.device).reshape(-1)
    if mapping.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= mapping.numel():
        raise ValueError("instance_ids contains an ID outside instance_to_glb")
    costs_all = torch.as_tensor(glb_cost_norm, dtype=torch.float32, device=flat_logits.device).reshape(-1)
    normalized_weights = normalize_pose_visible_weights(raw_weights, labels, pose_offsets)
    safety_loss, safety_parts = rvl_strong_v2_visibility_loss(
        logits, target, pose_offsets, normalized_weights, torch.zeros_like(logits)
    )
    hit = (
        torch.ones_like(raw_weights)
        if visible_hit_rates is None
        else visible_hit_rates.float().reshape(-1).clamp(0.0, 1.0)
    )
    scores = torch.sigmoid(flat_logits)
    quality_terms: list[torch.Tensor] = []
    quality_recall: list[torch.Tensor] = []
    resource_terms: list[torch.Tensor] = []
    rare_count = 0
    resource_groups = 0

    for _pose, start, end in _pose_slices(pose_offsets):
        positive = labels[start:end] > 0.5
        q = normalized_weights[start:end]
        if bool(positive.any()):
            q_pos = q[positive].clamp_min(1e-6)
            pos_scores = scores[start:end][positive]
            mass = q_pos.sum().clamp_min(1e-6)
            recall = (q_pos * pos_scores).sum() / mass
            quality_recall.append(recall)
            tail_mass = float(max(0.0, min(1.0, tail_fraction))) * float(mass.detach())
            order = torch.argsort(pos_scores.detach())
            cumulative = torch.cumsum(q_pos[order], dim=0)
            tail_count = int(
                torch.searchsorted(
                    cumulative, torch.tensor(tail_mass, device=cumulative.device)
                ).item()
            ) + 1
            tail_count = min(tail_count, pos_scores.numel())
            tail_indices = order[:tail_count]
            rare_count += tail_count
            tail_logits = flat_logits[start:end][positive][tail_indices]
            tail_weights = q_pos[tail_indices]
            rare_factor = 1.0 + float(rare_positive_weight) * (1.0 - torch.sqrt(hit[start:end][positive][tail_indices]))
            quality_terms.append(
                F.relu(1.0 - recall - 0.005).square()
                + (
                    F.softplus(0.5 - tail_logits) * tail_weights * rare_factor
                ).sum() / tail_weights.sum().clamp_min(1e-6)
            )

        # The mapping is indexed by global instance ID, not batch position.
        glb = mapping[ids[start:end]].clamp_min(0)
        if glb.numel() and int(glb.max()) >= costs_all.numel():
            raise ValueError("instance_to_glb points outside glb_cost_norm")
        unique, inverse = torch.unique(glb, sorted=True, return_inverse=True)
        required = torch.zeros(unique.numel(), device=flat_logits.device, dtype=flat_logits.dtype)
        if bool(positive.any()):
            required = required.scatter_reduce(
                0,
                inverse[positive],
                torch.ones_like(inverse[positive], dtype=required.dtype),
                reduce="amax",
                include_self=True,
            )
        unnecessary = required < 0.5
        if bool(unnecessary.any()):
            costs = costs_all[unique].float().clamp_min(0.05)
            threshold_terms: list[torch.Tensor] = []
            for threshold in request_thresholds:
                request = noisy_or(
                    soft_keep_at_threshold(flat_logits[start:end], threshold, quality_temperature),
                    inverse,
                    unique.numel(),
                )
                threshold_terms.append(
                    (request[unnecessary] * costs[unnecessary]).sum()
                    / unnecessary.to(request.dtype).sum().clamp_min(1.0)
                )
            resource_terms.append(torch.stack(threshold_terms).mean())
            resource_groups += int(unnecessary.sum())

    zero = flat_logits.sum() * 0.0
    quality_loss = torch.stack(quality_terms).mean() if quality_terms else zero
    resource_loss = torch.stack(resource_terms).mean() if resource_terms else zero
    total = safety_loss + float(quality_weight) * quality_loss + float(resource_weight) * resource_loss
    parts = {
        **{f"rvl_{key}": value for key, value in safety_parts.items()},
        "lossSafetyRvl": safety_loss,
        "lossQualityTail": quality_loss,
        "lossResourceRepulsion": resource_loss,
        "qualityRecall": torch.stack(quality_recall).mean() if quality_recall else zero,
        "qualityTailPositiveCount": float(rare_count),
        "resourceUnnecessaryGlbCount": float(resource_groups),
        "normalizedVisibleWeightMean": normalized_weights.mean(),
        "normalizedVisibleWeightMax": normalized_weights.max() if normalized_weights.numel() else zero,
        "lossQualityResourceRvl": total,
    }
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("quality/resource RVL loss is non-finite")
    return total, parts

