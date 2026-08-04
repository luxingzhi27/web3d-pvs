from __future__ import annotations

import torch
import torch.nn.functional as F


def pose_set_visibility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    false_negative_weight: float = 4.0,
    bce_weight: float = 0.35,
    tversky_weight: float = 1.0,
    count_weight: float = 0.25,
    rank_weight: float = 0.2,
    tversky_alpha: float = 1.5,
    tversky_beta: float = 2.0,
    rank_margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pose-level set loss used by both implicit and fixed-geo experiments.

    BCE keeps per-instance classification stable; Tversky directly penalizes pose-set
    FP/FN imbalance; Count discourages predicting far too many/few visible instances;
    Rank pushes visible logits above hard negatives inside the same pose.
    """
    target = target.float().view(-1, 1)
    logits = logits.view(-1, 1)
    pos_weight = torch.full((), float(false_negative_weight), device=logits.device, dtype=logits.dtype)
    loss_bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    tversky_terms = []
    count_terms = []
    rank_terms = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        p = prob[start:end]
        y = target[start:end]
        soft_tp = (p * y).sum()
        soft_fp = (p * (1.0 - y)).sum()
        soft_fn = ((1.0 - p) * y).sum()
        tversky_terms.append(1.0 - (soft_tp + 1e-6) / (soft_tp + tversky_alpha * soft_fp + tversky_beta * soft_fn + 1e-6))
        gt_count = torch.clamp(y.sum(), min=1.0)
        count_terms.append(F.smooth_l1_loss(p.sum() / gt_count, torch.ones((), device=logits.device, dtype=logits.dtype)))
        pos_logits = logits[start:end][y.view(-1) > 0.5].view(-1)
        neg_logits = logits[start:end][y.view(-1) <= 0.5].view(-1)
        if pos_logits.numel() > 0 and neg_logits.numel() > 0:
            if pos_logits.numel() > 64:
                pos_logits = pos_logits[torch.linspace(0, pos_logits.numel() - 1, 64, device=logits.device).long()]
            neg_top = torch.topk(neg_logits, k=min(256, neg_logits.numel())).values
            rank_terms.append(F.softplus(neg_top[:, None] - pos_logits[None, :] + rank_margin).mean())
    loss_tversky = torch.stack(tversky_terms).mean() if tversky_terms else torch.zeros((), device=logits.device)
    loss_count = torch.stack(count_terms).mean() if count_terms else torch.zeros((), device=logits.device)
    loss_rank = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=logits.device)
    loss = bce_weight * loss_bce + tversky_weight * loss_tversky + count_weight * loss_count + rank_weight * loss_rank
    return loss, {
        "lossBce": float(loss_bce.detach().cpu()),
        "lossTversky": float(loss_tversky.detach().cpu()),
        "lossCount": float(loss_count.detach().cpu()),
        "lossRank": float(loss_rank.detach().cpu()),
    }


def pose_set_visibility_loss_with_importance(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor | None = None,
    false_negative_weight: float = 4.0,
    bce_weight: float = 0.35,
    tversky_weight: float = 1.0,
    count_weight: float = 0.25,
    rank_weight: float = 0.2,
    tversky_alpha: float = 1.5,
    tversky_beta: float = 2.0,
    rank_margin: float = 0.25,
    importance_weight_scale: float = 1.0,
    importance_log_clip: float = 1024.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pose-level set loss with rvcServer component_weights as weak importance.

    `visible_weights` are not pixel counts. They only increase the weight of positive
    BCE/ranking examples so high-importance visible components are less likely to be
    missed. Tversky/count terms remain set-based and unweighted.
    """
    target = target.float().view(-1, 1)
    logits = logits.view(-1, 1)
    if visible_weights is None:
        importance = torch.zeros_like(target)
    else:
        raw = torch.clamp(visible_weights.float().view(-1, 1), min=0.0)
        denom = torch.log1p(torch.full((), float(max(1.0, importance_log_clip)), device=raw.device, dtype=raw.dtype))
        importance = torch.clamp(torch.log1p(raw) / torch.clamp(denom, min=1e-6), 0.0, 1.0) * target

    pos_weight = torch.full((), float(false_negative_weight), device=logits.device, dtype=logits.dtype)
    bce_raw = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    bce_sample_weight = 1.0 + float(importance_weight_scale) * importance
    loss_bce = (bce_raw * bce_sample_weight).sum() / torch.clamp(bce_sample_weight.sum(), min=1.0)

    prob = torch.sigmoid(logits)
    tversky_terms = []
    count_terms = []
    rank_terms = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        p = prob[start:end]
        y = target[start:end]
        soft_tp = (p * y).sum()
        soft_fp = (p * (1.0 - y)).sum()
        soft_fn = ((1.0 - p) * y).sum()
        tversky_terms.append(1.0 - (soft_tp + 1e-6) / (soft_tp + tversky_alpha * soft_fp + tversky_beta * soft_fn + 1e-6))
        gt_count = torch.clamp(y.sum(), min=1.0)
        count_terms.append(F.smooth_l1_loss(p.sum() / gt_count, torch.ones((), device=logits.device, dtype=logits.dtype)))
        pos_mask = y.view(-1) > 0.5
        neg_mask = ~pos_mask
        pos_logits = logits[start:end][pos_mask].view(-1)
        neg_logits = logits[start:end][neg_mask].view(-1)
        if pos_logits.numel() > 0 and neg_logits.numel() > 0:
            pos_importance = importance[start:end][pos_mask].view(-1)
            if pos_logits.numel() > 64:
                # Pick the most important positives first, but keep deterministic order.
                top = torch.topk(pos_importance, k=64).indices
                pos_logits = pos_logits[top]
                pos_importance = pos_importance[top]
            neg_top = torch.topk(neg_logits, k=min(256, neg_logits.numel())).values
            rank_raw = F.softplus(neg_top[:, None] - pos_logits[None, :] + rank_margin)
            rank_pos_weight = 1.0 + float(importance_weight_scale) * pos_importance[None, :]
            rank_terms.append((rank_raw * rank_pos_weight).sum() / torch.clamp(rank_pos_weight.sum() * rank_raw.shape[0], min=1.0))
    loss_tversky = torch.stack(tversky_terms).mean() if tversky_terms else torch.zeros((), device=logits.device)
    loss_count = torch.stack(count_terms).mean() if count_terms else torch.zeros((), device=logits.device)
    loss_rank = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=logits.device)
    loss = bce_weight * loss_bce + tversky_weight * loss_tversky + count_weight * loss_count + rank_weight * loss_rank
    return loss, {
        "lossBce": float(loss_bce.detach().cpu()),
        "lossTversky": float(loss_tversky.detach().cpu()),
        "lossCount": float(loss_count.detach().cpu()),
        "lossRank": float(loss_rank.detach().cpu()),
        "importanceMean": float(importance.detach().mean().cpu()) if importance.numel() else 0.0,
    }


def pose_set_visibility_loss_with_calibration(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor | None = None,
    bce_weight: float = 0.8,
    tversky_weight: float = 0.8,
    count_weight: float = 0.08,
    rank_weight: float = 0.45,
    positive_margin_weight: float = 0.55,
    hard_negative_margin_weight: float = 0.20,
    pos_class_weight: float = 2.0,
    neg_class_weight: float = 1.0,
    tversky_alpha: float = 1.0,
    tversky_beta: float = 6.0,
    rank_margin: float = 0.5,
    positive_logit_margin: float = 1.0,
    hard_negative_logit_margin: float = -1.0,
    hard_negative_top_k: int = 512,
    importance_weight_scale: float = 1.0,
    importance_log_clip: float = 1024.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pose-balanced visibility loss with explicit score calibration.

    The older set loss can keep high recall by lowering the runtime threshold because
    positives are not forced into a high-logit region. This variant treats each pose
    as an imbalanced binary classification problem: positives and hard negatives are
    normalized separately, then positive and negative logit margins make the score
    scale useful for thresholding.
    """
    target = target.float().view(-1, 1)
    logits = logits.view(-1, 1)
    if visible_weights is None:
        importance = torch.zeros_like(target)
    else:
        raw = torch.clamp(visible_weights.float().view(-1, 1), min=0.0)
        denom = torch.log1p(torch.full((), float(max(1.0, importance_log_clip)), device=raw.device, dtype=raw.dtype))
        importance = torch.clamp(torch.log1p(raw) / torch.clamp(denom, min=1e-6), 0.0, 1.0) * target

    prob = torch.sigmoid(logits)
    bce_terms = []
    pos_margin_terms = []
    neg_margin_terms = []
    tversky_terms = []
    count_terms = []
    rank_terms = []
    pos_logit_means = []
    neg_logit_means = []
    hard_neg_logit_means = []
    pos_prob_means = []
    neg_prob_means = []

    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        local_logits = logits[start:end].view(-1)
        local_prob = prob[start:end].view(-1)
        y = target[start:end].view(-1)
        local_importance = importance[start:end].view(-1)
        pos_mask = y > 0.5
        neg_mask = ~pos_mask
        if not pos_mask.any():
            continue

        pos_logits = local_logits[pos_mask]
        pos_prob = local_prob[pos_mask]
        pos_importance = local_importance[pos_mask]
        pos_weight = 1.0 + float(importance_weight_scale) * pos_importance
        pos_bce = F.softplus(-pos_logits)
        pos_loss = (pos_bce * pos_weight).sum() / torch.clamp(pos_weight.sum(), min=1.0)
        pos_margin_raw = F.softplus(float(positive_logit_margin) - pos_logits)
        pos_margin_terms.append((pos_margin_raw * pos_weight).sum() / torch.clamp(pos_weight.sum(), min=1.0))
        pos_logit_means.append(pos_logits.mean())
        pos_prob_means.append(pos_prob.mean())

        if neg_mask.any():
            neg_logits = local_logits[neg_mask]
            neg_prob = local_prob[neg_mask]
            top_k = min(max(1, int(hard_negative_top_k)), neg_logits.numel())
            hard_neg_logits = torch.topk(neg_logits, k=top_k).values
            neg_loss = F.softplus(hard_neg_logits).mean()
            neg_margin_terms.append(F.softplus(hard_neg_logits - float(hard_negative_logit_margin)).mean())
            neg_logit_means.append(neg_logits.mean())
            neg_prob_means.append(neg_prob.mean())
            hard_neg_logit_means.append(hard_neg_logits.mean())
            bce_terms.append(float(pos_class_weight) * pos_loss + float(neg_class_weight) * neg_loss)

            pos_rank_logits = pos_logits
            pos_rank_weight = pos_weight
            if pos_rank_logits.numel() > 64:
                top = torch.topk(pos_rank_weight, k=64).indices
                pos_rank_logits = pos_rank_logits[top]
                pos_rank_weight = pos_rank_weight[top]
            rank_raw = F.softplus(hard_neg_logits[:, None] - pos_rank_logits[None, :] + float(rank_margin))
            rank_weight_local = pos_rank_weight[None, :]
            rank_terms.append((rank_raw * rank_weight_local).sum() / torch.clamp(rank_weight_local.sum() * rank_raw.shape[0], min=1.0))
        else:
            bce_terms.append(float(pos_class_weight) * pos_loss)

        p = local_prob.view(-1, 1)
        yy = y.view(-1, 1)
        soft_tp = (p * yy).sum()
        soft_fp = (p * (1.0 - yy)).sum()
        soft_fn = ((1.0 - p) * yy).sum()
        tversky_terms.append(1.0 - (soft_tp + 1e-6) / (soft_tp + float(tversky_alpha) * soft_fp + float(tversky_beta) * soft_fn + 1e-6))
        gt_count = torch.clamp(yy.sum(), min=1.0)
        count_terms.append(F.smooth_l1_loss(p.sum() / gt_count, torch.ones((), device=logits.device, dtype=logits.dtype)))

    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    loss_bce = torch.stack(bce_terms).mean() if bce_terms else zero
    loss_pos_margin = torch.stack(pos_margin_terms).mean() if pos_margin_terms else zero
    loss_neg_margin = torch.stack(neg_margin_terms).mean() if neg_margin_terms else zero
    loss_tversky = torch.stack(tversky_terms).mean() if tversky_terms else zero
    loss_count = torch.stack(count_terms).mean() if count_terms else zero
    loss_rank = torch.stack(rank_terms).mean() if rank_terms else zero
    loss = (
        float(bce_weight) * loss_bce
        + float(tversky_weight) * loss_tversky
        + float(count_weight) * loss_count
        + float(rank_weight) * loss_rank
        + float(positive_margin_weight) * loss_pos_margin
        + float(hard_negative_margin_weight) * loss_neg_margin
    )

    def _mean_or_zero(values: list[torch.Tensor]) -> float:
        if not values:
            return 0.0
        return float(torch.stack([v.detach().float() for v in values]).mean().cpu())

    return loss, {
        "lossBalancedBce": float(loss_bce.detach().cpu()),
        "lossTversky": float(loss_tversky.detach().cpu()),
        "lossCount": float(loss_count.detach().cpu()),
        "lossRank": float(loss_rank.detach().cpu()),
        "lossPositiveMargin": float(loss_pos_margin.detach().cpu()),
        "lossHardNegativeMargin": float(loss_neg_margin.detach().cpu()),
        "importanceMean": float(importance.detach().mean().cpu()) if importance.numel() else 0.0,
        "posLogitMean": _mean_or_zero(pos_logit_means),
        "negLogitMean": _mean_or_zero(neg_logit_means),
        "hardNegLogitMean": _mean_or_zero(hard_neg_logit_means),
        "posProbMean": _mean_or_zero(pos_prob_means),
        "negProbMean": _mean_or_zero(neg_prob_means),
    }


def pose_visual_safety_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor | None,
    weight_power: float = 1.0,
    tail_k: int = 8,
    tail_margin: float = 1.0,
    tail_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiate the visual cost of missing visible instances.

    ``visible_weights`` is a screen-coverage proxy for Color-ID datasets and a
    weak importance value for legacy rvcServer datasets. Per-pose
    normalization removes dependence on the absolute unit while preserving the
    relative cost of missing high-contribution positives. The optional top-k
    margin protects the largest positives from being hidden by an average.
    """
    scores = logits.float().view(-1)
    y = target.float().view(-1)
    raw_weights = (
        torch.zeros_like(scores)
        if visible_weights is None
        else torch.clamp(visible_weights.float().view(-1), min=0.0)
    )
    mass_terms: list[torch.Tensor] = []
    tail_terms: list[torch.Tensor] = []
    coverage_terms: list[torch.Tensor] = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        pos_mask = y[start:end] > 0.5
        if not pos_mask.any():
            continue
        pos_scores = scores[start:end][pos_mask]
        pos_prob = torch.sigmoid(pos_scores)
        pos_raw_weights = raw_weights[start:end][pos_mask]
        # Legacy sources may not provide a meaningful positive weight. Fall
        # back to an ordinary positive miss loss instead of zero gradient.
        if float(pos_raw_weights.sum().detach().cpu()) <= 1e-8:
            importance = torch.ones_like(pos_raw_weights)
        else:
            importance = torch.pow(pos_raw_weights, max(0.0, float(weight_power)))
        total_importance = torch.clamp(importance.sum(), min=1e-8)
        mass_terms.append(((1.0 - pos_prob) * importance).sum() / total_importance)
        coverage_terms.append(pos_raw_weights.mean())

        k = min(max(0, int(tail_k)), pos_scores.numel())
        if k > 0 and float(tail_weight) > 0.0:
            top_indices = torch.topk(pos_raw_weights, k=k, largest=True, sorted=False).indices
            top_scores = pos_scores[top_indices]
            tail_terms.append(F.softplus(float(tail_margin) - top_scores).mean())

    zero = torch.zeros((), device=scores.device, dtype=scores.dtype)
    mass = torch.stack(mass_terms).mean() if mass_terms else zero
    tail = torch.stack(tail_terms).mean() if tail_terms else zero
    loss = mass + float(tail_weight) * tail
    coverage = torch.stack(coverage_terms).mean() if coverage_terms else zero
    return loss, {
        "lossVisualSafety": float(loss.detach().cpu()),
        "lossVisualSafetyMass": float(mass.detach().cpu()),
        "lossVisualSafetyTail": float(tail.detach().cpu()),
        "visualSafetyPositiveWeightMean": float(coverage.detach().cpu()),
    }


def pose_subpose_robust_safety_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor | None,
    visible_hit_rates: torch.Tensor | None,
    rare_weight: float = 1.0,
    frequency_power: float = 0.5,
    tail_k: int = 8,
    tail_margin: float = 1.5,
    tail_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Protect view-cell positives that are easy to miss at dense subposes.

    A view-cell target is a union over same-direction subposes. ``visible_weights``
    stores the maximum visual contribution and ``visible_hit_rates`` stores the
    fraction of subposes in which the instance appeared. The risk weight keeps
    high-contribution positives important and adds a bounded bonus to rare
    positives, which targets worst-case subpose coverage without changing the
    candidate set or adding any negative examples.
    """
    scores = logits.float().view(-1)
    labels = target.float().view(-1)
    raw_weights = (
        torch.zeros_like(scores)
        if visible_weights is None
        else torch.clamp(visible_weights.float().view(-1), min=0.0)
    )
    hit_rates = (
        torch.zeros_like(scores)
        if visible_hit_rates is None
        else torch.clamp(visible_hit_rates.float().view(-1), min=0.0, max=1.0)
    )
    mass_terms: list[torch.Tensor] = []
    tail_terms: list[torch.Tensor] = []
    risk_means: list[torch.Tensor] = []
    rare_means: list[torch.Tensor] = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        pos_mask = labels[start:end] > 0.5
        if not pos_mask.any():
            continue
        pos_scores = scores[start:end][pos_mask]
        pos_weights = raw_weights[start:end][pos_mask]
        pos_rates = hit_rates[start:end][pos_mask]
        if float(pos_weights.sum().detach().cpu()) <= 1e-8:
            visual_mass = torch.ones_like(pos_weights)
        else:
            visual_mass = torch.log1p(pos_weights)
        frequency = torch.pow(
            torch.clamp(pos_rates, min=0.0, max=1.0),
            max(0.0, float(frequency_power)),
        )
        risk = visual_mass * (1.0 + float(rare_weight) * (1.0 - frequency))
        risk = torch.clamp(risk, min=1e-6)
        mass_terms.append(
            ((1.0 - torch.sigmoid(pos_scores)) * risk).sum()
            / torch.clamp(risk.sum(), min=1e-6)
        )
        risk_means.append(risk.mean())
        rare_means.append((1.0 - frequency).mean())

        k = min(max(0, int(tail_k)), pos_scores.numel())
        if k > 0 and float(tail_weight) > 0.0:
            top_indices = torch.topk(risk, k=k, largest=True, sorted=False).indices
            tail_terms.append(F.softplus(float(tail_margin) - pos_scores[top_indices]).mean())

    zero = torch.zeros((), device=scores.device, dtype=scores.dtype)
    mass = torch.stack(mass_terms).mean() if mass_terms else zero
    tail = torch.stack(tail_terms).mean() if tail_terms else zero
    risk_mean = torch.stack(risk_means).mean() if risk_means else zero
    rare_mean = torch.stack(rare_means).mean() if rare_means else zero
    loss = mass + float(tail_weight) * tail
    return loss, {
        "lossSubposeRobust": float(loss.detach().cpu()),
        "lossSubposeRobustMass": float(mass.detach().cpu()),
        "lossSubposeRobustTail": float(tail.detach().cpu()),
        "subposeRobustRiskMean": float(risk_mean.detach().cpu()),
        "subposeRobustRareMean": float(rare_mean.detach().cpu()),
    }
