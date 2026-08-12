"""Occlusion-aware weighted-recall boundary loss (OWRB).

OWRB is an independent loss for this project.  It is intentionally not a
renamed RVL implementation: it first estimates a per-pose score boundary that
retains a registered fraction of visible weight, then suppresses only the
hardest negatives above that boundary and protects high-utility positives.
The survival-field probability is detached before it changes negative
weights, so the visibility head cannot exploit a joint gradient shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from common.survival_loss import survival_censoring_loss


@dataclass
class OWRBConfig:
    target_visible_mass: float = 0.995
    top_k_hard_negatives: int = 128
    boundary_margin: float = 0.20
    positive_margin: float = 0.05
    lambda_balanced_bce: float = 0.50
    lambda_hard_negative: float = 1.00
    lambda_positive_protection: float = 0.75
    lambda_boundary_alignment: float = 0.10
    survival_negative_beta: float = 0.50
    boundary_ema_momentum: float = 0.95


@dataclass
class OWRBState:
    """Non-gradient state shared by batches within one training run."""

    boundary_ema: float = 0.0
    initialized: bool = False
    boundary_history: list[float] = field(default_factory=list)

    def update(self, boundaries: list[torch.Tensor | float]) -> None:
        finite = []
        for value in boundaries:
            numeric = float(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            if numeric == numeric and abs(numeric) < float("inf"):
                finite.append(numeric)
        if not finite:
            return
        current = sum(finite) / len(finite)
        # The momentum is supplied by the loss config at the call site and is
        # recorded in the returned state update; this default is only for
        # direct use in tests.
        momentum = 0.95
        if not self.initialized:
            self.boundary_ema = current
            self.initialized = True
        else:
            self.boundary_ema = momentum * self.boundary_ema + (1.0 - momentum) * current
        self.boundary_history.append(float(current))
        if len(self.boundary_history) > 256:
            del self.boundary_history[:-256]

    def update_with_momentum(self, boundaries: list[torch.Tensor | float], momentum: float) -> None:
        finite = []
        for value in boundaries:
            numeric = float(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            if numeric == numeric and abs(numeric) < float("inf"):
                finite.append(numeric)
        if not finite:
            return
        current = sum(finite) / len(finite)
        if not self.initialized:
            self.boundary_ema = current
            self.initialized = True
        else:
            self.boundary_ema = float(momentum) * self.boundary_ema + (1.0 - float(momentum)) * current
        self.boundary_history.append(float(current))
        if len(self.boundary_history) > 256:
            del self.boundary_history[:-256]

    def as_dict(self) -> dict[str, Any]:
        return {
            "boundaryEma": float(self.boundary_ema),
            "initialized": bool(self.initialized),
            "recentBoundaryMean": float(sum(self.boundary_history) / len(self.boundary_history)) if self.boundary_history else 0.0,
            "historyCount": len(self.boundary_history),
        }


def weighted_recall_boundary(
    positive_scores: torch.Tensor,
    positive_weights: torch.Tensor,
    target_visible_mass: float,
) -> torch.Tensor:
    """Return the score threshold retaining ``target_visible_mass`` weight."""
    scores = positive_scores.reshape(-1).float()
    weights = torch.clamp(positive_weights.reshape(-1).float(), min=1e-6)
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    cumulative = torch.cumsum(weights[order], dim=0)
    target = min(max(float(target_visible_mass), 0.0), 1.0) * weights.sum()
    index = torch.searchsorted(cumulative, target, right=False).clamp(max=sorted_scores.numel() - 1)
    return sorted_scores[index]


def _pose_ranges(pose_offsets: torch.Tensor):
    for pose_id in range(max(0, int(pose_offsets.numel()) - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end > start:
            yield pose_id, start, end


def owrb_visibility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    survival_occlusion_probability: torch.Tensor | None = None,
    config: OWRBConfig | None = None,
    state: OWRBState | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Compute OWRB over a batch of variable-length candidate poses."""
    cfg = config or OWRBConfig()
    state = state or OWRBState()
    flat_logits = logits.float().reshape(-1)
    scores = torch.sigmoid(flat_logits)
    labels = target.float().reshape(-1)
    weights = torch.clamp(visible_weights.float().reshape(-1), min=0.0)
    if survival_occlusion_probability is None:
        occlusion = torch.zeros_like(scores)
    else:
        occlusion = torch.clamp(survival_occlusion_probability.float().reshape(-1), 0.0, 1.0)
    balanced_terms: list[torch.Tensor] = []
    hard_negative_terms: list[torch.Tensor] = []
    positive_protection_terms: list[torch.Tensor] = []
    alignment_terms: list[torch.Tensor] = []
    boundaries: list[torch.Tensor] = []
    pose_recall_surrogates: list[torch.Tensor] = []
    hard_negative_count = 0
    protected_positive_count = 0

    for _pose_id, start, end in _pose_ranges(pose_offsets):
        local_logits = flat_logits[start:end]
        local_scores = scores[start:end]
        local_labels = labels[start:end] > 0.5
        local_weights = weights[start:end]
        local_occlusion = occlusion[start:end].detach()
        positives = local_labels
        negatives = ~local_labels
        if positives.any():
            boundary = weighted_recall_boundary(
                local_scores[positives], local_weights[positives], cfg.target_visible_mass
            )
            boundaries.append(boundary.detach())
            positive_weight = torch.clamp(local_weights[positives], min=1e-6)
            positive_weight = positive_weight / torch.clamp(positive_weight.mean(), min=1e-6)
            positive_bce = F.binary_cross_entropy_with_logits(
                local_logits[positives], torch.ones_like(local_logits[positives]), reduction="none"
            )
            negative_bce = torch.zeros((), device=flat_logits.device, dtype=flat_logits.dtype)
            if negatives.any():
                negative_bce = F.binary_cross_entropy_with_logits(
                    local_logits[negatives], torch.zeros_like(local_logits[negatives]), reduction="mean"
                )
            balanced_terms.append((positive_bce * positive_weight).mean() + negative_bce)
            protected = F.softplus(float(cfg.positive_margin) + boundary - local_scores[positives])
            positive_protection_terms.append((protected * positive_weight).mean())
            protected_positive_count += int(positives.sum().item())
            pose_recall_surrogates.append(
                (local_scores[positives] * local_weights[positives]).sum()
                / torch.clamp(local_weights[positives].sum(), min=1e-6)
            )
        elif negatives.any():
            balanced_terms.append(
                F.binary_cross_entropy_with_logits(
                    local_logits[negatives], torch.zeros_like(local_logits[negatives]), reduction="mean"
                )
            )

        if positives.any() and negatives.any():
            negative_scores = local_scores[negatives]
            negative_occlusion = local_occlusion[negatives]
            k = min(int(cfg.top_k_hard_negatives), int(negative_scores.numel()))
            hard_indices = torch.topk(negative_scores, k=k, largest=True).indices
            selected_scores = negative_scores[hard_indices]
            selected_occlusion = negative_occlusion[hard_indices]
            negative_weight = 1.0 + float(cfg.survival_negative_beta) * selected_occlusion
            target_score = boundary.detach() - float(cfg.boundary_margin)
            hard_negative_terms.append(
                (F.softplus(selected_scores - target_score) * negative_weight).mean()
            )
            hard_negative_count += int(k)

    if not balanced_terms:
        zero = flat_logits.sum() * 0.0
        return zero, {"lossOWRB": zero, "owrbBoundary": 0.0}
    balanced = torch.stack(balanced_terms).mean()
    hard_negative = torch.stack(hard_negative_terms).mean() if hard_negative_terms else balanced * 0.0
    protection = torch.stack(positive_protection_terms).mean() if positive_protection_terms else balanced * 0.0
    if state.initialized and boundaries:
        ema = flat_logits.new_tensor(float(state.boundary_ema))
        alignment = torch.stack([(boundary - ema).abs() for boundary in boundaries]).mean()
    else:
        alignment = balanced * 0.0
    loss = (
        float(cfg.lambda_balanced_bce) * balanced
        + float(cfg.lambda_hard_negative) * hard_negative
        + float(cfg.lambda_positive_protection) * protection
        + float(cfg.lambda_boundary_alignment) * alignment
    )
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("OWRB produced a non-finite loss")
    if state is not None:
        state.update_with_momentum(boundaries, cfg.boundary_ema_momentum)
    parts: dict[str, torch.Tensor | float] = {
        "lossOWRB": loss,
        "lossOWRBBalancedBce": balanced,
        "lossOWRBHardNegative": hard_negative,
        "lossOWRBPositiveProtection": protection,
        "lossOWRBBoundaryAlignment": alignment,
        "owrbBoundary": float(sum(float(value.detach().cpu()) for value in boundaries) / len(boundaries)) if boundaries else 0.0,
        "owrbBoundaryEma": float(state.boundary_ema),
        "owrbHardNegativeCount": float(hard_negative_count),
        "owrbProtectedPositiveCount": float(protected_positive_count),
        "owrbSoftWeightedRecall": torch.stack(pose_recall_surrogates).mean() if pose_recall_surrogates else balanced * 0.0,
    }
    return loss, parts
