from __future__ import annotations

"""Losses for the direction-depth survival experiment.

The visibility part is evaluated per pose, while the resource term estimates
the GLB bytes that would be requested by soft predictions.  Dual variables are
kept outside autograd and updated once per training step/epoch.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from common.survival_loss import survival_censoring_loss


@dataclass
class SafetyDualState:
    pose: float = 0.0
    weighted: float = 0.0
    cvar: float = 0.0
    learning_rate: float = 0.05
    maximum: float = 20.0

    def update(self, metrics: dict[str, torch.Tensor | float]) -> None:
        values = {
            "pose": float(metrics["pose_miss"].detach().cpu() if isinstance(metrics["pose_miss"], torch.Tensor) else metrics["pose_miss"]) - 0.05,
            "weighted": float(metrics["weighted_miss"].detach().cpu() if isinstance(metrics["weighted_miss"], torch.Tensor) else metrics["weighted_miss"]) - 0.01,
            "cvar": float(metrics["weighted_miss_cvar"].detach().cpu() if isinstance(metrics["weighted_miss_cvar"], torch.Tensor) else metrics["weighted_miss_cvar"]) - 0.03,
        }
        self.pose = min(self.maximum, max(0.0, self.pose + self.learning_rate * values["pose"]))
        self.weighted = min(self.maximum, max(0.0, self.weighted + self.learning_rate * values["weighted"]))
        self.cvar = min(self.maximum, max(0.0, self.cvar + self.learning_rate * values["cvar"]))

    def as_dict(self) -> dict[str, float]:
        return {"pose": self.pose, "weighted": self.weighted, "cvar": self.cvar, "learningRate": self.learning_rate, "maximum": self.maximum}


def _mean_or_zero(values: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack(values).mean() if values else torch.zeros((), device=device, dtype=dtype)


def _soft_group_union(
    probabilities: torch.Tensor,
    inverse_group: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    """Combine instance probabilities into one probability per GLB.

    A GLB is requested when at least one of its instances is requested.  The
    differentiable noisy-OR below counts a shared GLB once, while preserving
    gradients for every instance that contributes to the group.
    """
    if probabilities.ndim != 1 or inverse_group.ndim != 1:
        raise ValueError("probabilities and inverse_group must be one-dimensional")
    if probabilities.numel() != inverse_group.numel():
        raise ValueError("probabilities and inverse_group must have equal length")
    if int(group_count) <= 0:
        return probabilities.new_zeros((0,))
    clipped = torch.clamp(probabilities.float(), 0.0, 1.0 - 1e-6)
    log_not = torch.log1p(-clipped)
    grouped_log_not = torch.zeros((int(group_count),), device=probabilities.device, dtype=log_not.dtype)
    grouped_log_not.scatter_add_(0, inverse_group.long(), log_not)
    return 1.0 - torch.exp(grouped_log_not)


def safety_constraint_utility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    instance_ids: torch.Tensor,
    instance_to_glb: torch.Tensor,
    glb_cost_norm: torch.Tensor,
    gamma: float,
    dual_state: SafetyDualState,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Compute the registered safety-constrained visibility/resource loss."""
    scores = torch.sigmoid(logits.float().view(-1))
    y = target.float().view(-1)
    weights = torch.clamp(visible_weights.float().view(-1), min=0.0)
    ids = instance_ids.long().view(-1)
    mapping = instance_to_glb.long().view(-1)
    if ids.numel() and int(ids.max()) >= mapping.numel():
        raise ValueError("instance_ids contains an ID outside instance_to_glb")
    glb_ids = mapping[torch.clamp(ids, 0, max(0, mapping.numel() - 1))]
    bce_terms: list[torch.Tensor] = []
    soft_fp_terms: list[torch.Tensor] = []
    excess_bytes_terms: list[torch.Tensor] = []
    pose_miss_terms: list[torch.Tensor] = []
    weighted_miss_terms: list[torch.Tensor] = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        local_p, local_y = scores[start:end], y[start:end]
        local_w = weights[start:end]
        pos = local_y > 0.5
        neg = ~pos
        if pos.any():
            bce_terms.append(F.softplus(-logits.float().view(-1)[start:end][pos]).mean())
        if neg.any():
            bce_terms.append(F.softplus(logits.float().view(-1)[start:end][neg]).mean())
            soft_fp_terms.append((local_p[neg]).sum() / max(1.0, float(neg.sum().item())))
        gt_count = torch.clamp(local_y.sum(), min=1.0)
        gt_weight = torch.clamp((local_y * local_w).sum(), min=1e-6)
        pose_miss_terms.append(1.0 - (local_p * local_y).sum() / gt_count)
        weighted_miss_terms.append(1.0 - (local_p * local_y * local_w).sum() / gt_weight)
        # Download cost is GLB-level, while visibility labels are instance-
        # level.  Aggregate instances sharing one GLB before applying bytes;
        # otherwise a GLB with many instances is charged repeatedly.
        local_glb = glb_ids[start:end]
        unique_glb, inverse_glb = torch.unique(local_glb, sorted=True, return_inverse=True)
        group_probability = _soft_group_union(local_p, inverse_glb, int(unique_glb.numel()))
        group_required = torch.zeros_like(group_probability)
        group_required = group_required.scatter_reduce(
            0, inverse_glb, local_y, reduce="amax", include_self=True
        )
        group_cost = torch.clamp(glb_cost_norm[unique_glb], min=0.05)
        predicted_bytes = (group_probability * group_cost).sum()
        required_bytes = (group_required * group_cost).sum()
        excess_bytes_terms.append(F.relu(predicted_bytes - required_bytes) / torch.clamp(required_bytes, min=1.0))
    balanced_bce = _mean_or_zero(bce_terms, logits.device, logits.dtype)
    soft_fp = _mean_or_zero(soft_fp_terms, logits.device, logits.dtype)
    excess_bytes = _mean_or_zero(excess_bytes_terms, logits.device, logits.dtype)
    pose_miss = _mean_or_zero(pose_miss_terms, logits.device, logits.dtype)
    weighted_miss = _mean_or_zero(weighted_miss_terms, logits.device, logits.dtype)
    if weighted_miss_terms:
        ordered = torch.stack(weighted_miss_terms)
        tail_count = max(1, int(torch.ceil(torch.tensor(0.05 * ordered.numel())).item()))
        weighted_miss_cvar = torch.topk(ordered, k=min(tail_count, ordered.numel())).values.mean()
    else:
        weighted_miss_cvar = torch.zeros((), device=logits.device, dtype=logits.dtype)
    gamma_value = float(gamma)
    base = 0.1 * balanced_bce + (1.0 - gamma_value) * soft_fp + gamma_value * excess_bytes
    constraint = (
        float(dual_state.pose) * (pose_miss - 0.05)
        + float(dual_state.weighted) * (weighted_miss - 0.01)
        + float(dual_state.cvar) * (weighted_miss_cvar - 0.03)
    )
    loss = base + constraint
    metrics: dict[str, torch.Tensor | float] = {
        "lossBalancedBce": balanced_bce,
        "lossSoftFp": soft_fp,
        "lossExcessGlbBytes": excess_bytes,
        "lossSafetyConstraint": constraint,
        "lossSafetyUtility": loss,
        "pose_miss": pose_miss,
        "weighted_miss": weighted_miss,
        "weighted_miss_cvar": weighted_miss_cvar,
    }
    return loss, metrics


def rvl_strong_v2_visibility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    evidence: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compact registered control matching the current rvl_strong_v2 intent."""
    scores = torch.sigmoid(logits.float().view(-1))
    y = target.float().view(-1)
    weights = torch.clamp(visible_weights.float().view(-1), min=0.0)
    bce = F.binary_cross_entropy_with_logits(logits.float().view(-1), y, pos_weight=torch.tensor(14.0, device=logits.device))
    tversky: list[torch.Tensor] = []
    count: list[torch.Tensor] = []
    rank: list[torch.Tensor] = []
    fn: list[torch.Tensor] = []
    fp: list[torch.Tensor] = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        p, local_y = scores[start:end], y[start:end]
        soft_tp = (p * local_y).sum()
        soft_fp = (p * (1.0 - local_y)).sum()
        soft_fn = ((1.0 - p) * local_y).sum()
        tversky.append(1.0 - (soft_tp + 1e-6) / (soft_tp + soft_fp + 7.0 * soft_fn + 1e-6))
        count.append(F.smooth_l1_loss(p.sum() / torch.clamp(local_y.sum(), min=1.0), torch.ones((), device=p.device)))
        pos = logits.view(-1)[start:end][local_y > 0.5]
        neg = logits.view(-1)[start:end][local_y <= 0.5]
        if pos.numel() and neg.numel():
            rank.append(F.softplus(torch.topk(neg, k=min(256, neg.numel())).values[:, None] - pos[None, :] + 0.3).mean())
        evidence_local = torch.clamp(evidence.float().view(-1)[start:end], 0.0, 1.0)
        fn.append(((1.0 - p) * local_y * (1.0 + weights[start:end])).sum() / torch.clamp(local_y.sum(), min=1.0))
        fp.append((p * (1.0 - local_y) * (1.0 + evidence_local)).sum() / torch.clamp(local_y.sum(), min=1.0))
    parts = {
        "lossBce": bce,
        "lossTversky": _mean_or_zero(tversky, logits.device, logits.dtype),
        "lossCount": _mean_or_zero(count, logits.device, logits.dtype),
        "lossRank": _mean_or_zero(rank, logits.device, logits.dtype),
        "lossRvlFn": _mean_or_zero(fn, logits.device, logits.dtype),
        "lossRvlFp": _mean_or_zero(fp, logits.device, logits.dtype),
    }
    loss = 0.28 * parts["lossBce"] + 1.35 * parts["lossTversky"] + 0.10 * parts["lossCount"] + 0.45 * parts["lossRank"] + 0.12 * (0.25 * parts["lossRvlFn"] + parts["lossRvlFp"])
    parts["lossRvlStrongV2"] = loss
    return loss, parts


def neutral_rvl_control_evidence(logits: torch.Tensor) -> torch.Tensor:
    """Return the fixed evidence tensor used by the new RVL control.

    The ray-context experiment must not let its newly generated triangle-depth
    relation table change the RVL control loss. Historical callers may still
    pass their own evidence tensor to ``rvl_strong_v2_visibility_loss``; the
    new formal trainer uses this explicit neutral tensor instead.
    """
    return torch.zeros_like(logits)
