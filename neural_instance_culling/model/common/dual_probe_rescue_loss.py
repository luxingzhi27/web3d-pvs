"""Loss for training only the attenuation of a frozen dual-probe rescue.

The two ridge probes and their gates are fixed posterior quantities.  This loss
therefore treats the base logit, gates, and visual weights as detached labels:
low-score positives are encouraged to keep their rescue, while negatives are
encouraged to attenuate it.  Positive primary-tail terms use square-root visual
importance; coverage-tail terms are class-balanced and unweighted.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _flat_real(value: Any, name: str, device: torch.device) -> torch.Tensor:
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc
    if torch.is_complex(tensor) or tensor.dtype == torch.bool:
        raise ValueError(f"{name} must be a real-valued vector")
    if tensor.ndim == 0 or tensor.ndim > 2 or (
        tensor.ndim == 2 and 1 not in tensor.shape
    ):
        raise ValueError(f"{name} must be a vector or singleton-column vector")
    result = tensor.to(device=device, dtype=torch.float32).reshape(-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _flat_mask(value: Any, name: str, count: int, device: torch.device) -> torch.Tensor:
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc
    if tensor.ndim == 0 or tensor.ndim > 2 or (
        tensor.ndim == 2 and 1 not in tensor.shape
    ):
        raise ValueError(f"{name} must be a vector or singleton-column vector")
    if tensor.dtype == torch.bool:
        result = tensor.to(device=device).reshape(-1)
    else:
        result = tensor.to(device=device, dtype=torch.float32).reshape(-1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} must contain only finite values")
        if not bool(((result == 0.0) | (result == 1.0)).all()):
            raise ValueError(f"{name} must contain binary 0/1 values")
        result = result > 0.5
    if result.numel() != int(count):
        raise ValueError(f"{name} must align with the attenuation batch")
    return result


def _finite_non_negative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    selected = values[mask]
    if selected.numel() == 0:
        return zero, 0.0
    if weights is None:
        return selected.mean(), float(selected.numel())
    selected_weights = weights[mask]
    mass = selected_weights.sum()
    if not bool(mass > 0.0):
        return selected.mean(), float(selected.numel())
    return (selected * selected_weights).sum() / mass, float(selected.numel())


def dual_probe_rescue_attenuation_loss(
    attenuation: torch.Tensor,
    base_logit: torch.Tensor,
    primary_gate: torch.Tensor,
    coverage_gate: torch.Tensor,
    target: torch.Tensor,
    visible_weights: torch.Tensor,
    *,
    primary_low_score_positive: Any | None = None,
    coverage_low_score_positive: Any | None = None,
    high_score_negative: Any | None = None,
    fixed_residual: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    coverage_weight: float = 2.0,
    primary_keep_weight: float = 1.0,
    coverage_keep_weight: float = 1.0,
    negative_decay_weight: float = 1.0,
    high_negative_decay_weight: float = 1.0,
    visible_weight_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return a class-normalized loss for the trainable attenuation head.

    ``attenuation`` has shape ``[B, 2]``.  Its first column protects the
    primary high-visual-weight positive tail and its second column protects
    the ordinary coverage tail.  All masks and posterior quantities are
    detached before membership or weighting decisions.  If masks are omitted,
    positive members with a strictly positive corresponding frozen gate are
    used; high-score negatives default to the upper half of detached base
    logits among negatives.
    """
    if not isinstance(attenuation, torch.Tensor):
        raise ValueError("attenuation must be a torch.Tensor")
    if attenuation.ndim != 2 or attenuation.shape[1] != 2:
        raise ValueError("attenuation must have shape [B, 2]")
    if not bool(torch.isfinite(attenuation).all()):
        raise ValueError("attenuation must be finite")
    if bool((attenuation < 0.0).any() or (attenuation > 1.0).any()):
        raise ValueError("attenuation must lie in [0, 1]")
    device = attenuation.device
    count = int(attenuation.shape[0])
    if count <= 0:
        raise ValueError("dual probe rescue loss requires at least one candidate")

    base = _flat_real(base_logit, "base_logit", device).detach()
    primary = _flat_real(primary_gate, "primary_gate", device).detach()
    coverage = _flat_real(coverage_gate, "coverage_gate", device).detach()
    labels = _flat_real(target, "target", device)
    weights = _flat_real(visible_weights, "visible_weights", device).detach()
    for name, value in {
        "base_logit": base,
        "primary_gate": primary,
        "coverage_gate": coverage,
        "target": labels,
        "visible_weights": weights,
    }.items():
        if value.numel() != count:
            raise ValueError(f"{name} must align with attenuation")
    if bool((primary < 0.0).any() or (coverage < 0.0).any()):
        raise ValueError("frozen probe gates must be non-negative")
    if bool((primary > 1.0 + 1e-6).any() or (coverage > 1.0 + 1e-6).any()):
        raise ValueError("frozen probe gates must lie in [0, 1]")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("target must contain binary 0/1 values")
    if bool((weights < 0.0).any()):
        raise ValueError("visible_weights must be non-negative")
    coverage_weight_value = _finite_non_negative(coverage_weight, "coverage_weight")
    primary_keep = _finite_non_negative(primary_keep_weight, "primary_keep_weight")
    coverage_keep = _finite_non_negative(coverage_keep_weight, "coverage_keep_weight")
    negative_decay = _finite_non_negative(negative_decay_weight, "negative_decay_weight")
    hard_negative_decay = _finite_non_negative(
        high_negative_decay_weight, "high_negative_decay_weight"
    )
    epsilon = _finite_non_negative(visible_weight_epsilon, "visible_weight_epsilon")
    if epsilon <= 0.0:
        raise ValueError("visible_weight_epsilon must be positive")

    positives = labels > 0.5
    negatives = ~positives
    if primary_low_score_positive is None:
        primary_tail = positives & (primary > 0.0)
    else:
        primary_tail = _flat_mask(
            primary_low_score_positive,
            "primary_low_score_positive",
            count,
            device,
        )
        if bool((primary_tail & ~positives).any()):
            raise ValueError("primary low-score mask contains negative candidates")
    if coverage_low_score_positive is None:
        coverage_tail = positives & (coverage > 0.0)
    else:
        coverage_tail = _flat_mask(
            coverage_low_score_positive,
            "coverage_low_score_positive",
            count,
            device,
        )
        if bool((coverage_tail & ~positives).any()):
            raise ValueError("coverage low-score mask contains negative candidates")
    if high_score_negative is None:
        if bool(negatives.any()):
            median = base[negatives].median()
            hard_negative = negatives & (base >= median)
        else:
            hard_negative = torch.zeros_like(negatives)
    else:
        hard_negative = _flat_mask(high_score_negative, "high_score_negative", count, device)
        if bool((hard_negative & ~negatives).any()):
            raise ValueError("high-score negative mask contains positive candidates")

    if fixed_residual is not None:
        fixed = _flat_real(fixed_residual, "fixed_residual", device).detach()
        if fixed.numel() != count:
            raise ValueError("fixed_residual must align with attenuation")
        if bool((fixed < 0.0).any()):
            raise ValueError("fixed_residual must be non-negative")
    else:
        fixed = None
    if residual is not None:
        applied = _flat_real(residual, "residual", device).detach()
        if applied.numel() != count:
            raise ValueError("residual must align with attenuation")
        if bool((applied < 0.0).any()):
            raise ValueError("residual must be non-negative")
        if fixed is not None and bool((applied > fixed + 1e-6).any()):
            raise ValueError("residual must not exceed fixed_residual")
    else:
        applied = None

    zero = attenuation.sum() * 0.0
    primary_keep_loss, primary_count = _masked_mean(
        1.0 - attenuation[:, 0],
        primary_tail,
        zero,
        torch.sqrt(weights.clamp_min(epsilon)),
    )
    coverage_keep_loss, coverage_count = _masked_mean(
        1.0 - attenuation[:, 1],
        coverage_tail,
        zero,
    )
    negative_values = attenuation[negatives].square().mean(dim=-1)
    negative_decay_loss, negative_count = _masked_mean(
        negative_values,
        torch.ones(negative_values.shape, dtype=torch.bool, device=device),
        zero,
    )
    # The hard-negative term is separately normalized, so a large negative
    # candidate pool cannot drown out the positive-tail preservation terms.
    hard_values = attenuation[hard_negative].square().mean(dim=-1)
    hard_decay_loss, hard_negative_count = _masked_mean(
        hard_values,
        torch.ones(hard_values.shape, dtype=torch.bool, device=device),
        zero,
    )
    loss = (
        primary_keep * primary_keep_loss
        + coverage_keep * coverage_keep_loss
        + negative_decay * negative_decay_loss
        + hard_negative_decay * hard_decay_loss
    )
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("dual probe rescue attenuation loss is non-finite")
    diagnostics: dict[str, torch.Tensor | float] = {
        "lossDualProbeRescueAttenuation": loss,
        "lossPrimaryKeep": primary_keep_loss,
        "lossCoverageKeep": coverage_keep_loss,
        "lossNegativeDecay": negative_decay_loss,
        "lossHighNegativeDecay": hard_decay_loss,
        "primaryLowScorePositiveCount": primary_count,
        "coverageLowScorePositiveCount": coverage_count,
        "negativeCount": negative_count,
        "highScoreNegativeCount": hard_negative_count,
        "primaryVisibleWeightSqrtMass": float(
            torch.sqrt(weights.clamp_min(epsilon))[primary_tail].sum().detach()
        ),
        "coverageWeighting": 1.0,
        "coverageWeight": coverage_weight_value,
        "baseLogitDetached": True,
        "frozenGatesDetached": True,
        "classNormalized": True,
    }
    if fixed is not None:
        diagnostics["fixedResidualMean"] = fixed.mean()
    if applied is not None:
        diagnostics["residualMean"] = applied.mean()
        diagnostics["residualWithinFixedPosterior"] = True
    return loss, diagnostics


def dual_probe_rescue_loss(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Short alias for :func:`dual_probe_rescue_attenuation_loss`."""
    return dual_probe_rescue_attenuation_loss(*args, **kwargs)


__all__ = ["dual_probe_rescue_attenuation_loss", "dual_probe_rescue_loss"]
