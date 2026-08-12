"""Training-only projection for objectives that must not oppose safety.

This module is deliberately independent of any particular visibility loss.
The caller supplies a safety gradient and an auxiliary efficiency/resource
gradient. Conflicting auxiliary components are removed, then their norm is
capped relative to the safety gradient.
"""
from __future__ import annotations

import torch


def project_auxiliary_gradients(
    safety_gradients: list[torch.Tensor | None],
    auxiliary_gradients: list[torch.Tensor | None],
    gradient_cap: float = 0.25,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Project and cap an auxiliary gradient against a safety gradient."""
    if len(safety_gradients) != len(auxiliary_gradients):
        raise ValueError("safety and auxiliary gradient lists must have equal length")
    device = next(
        (
            value.device
            for value in safety_gradients + auxiliary_gradients
            if value is not None
        ),
        torch.device("cpu"),
    )
    dot = torch.zeros((), device=device)
    safety_norm_sq = torch.zeros((), device=device)
    auxiliary_norm_sq = torch.zeros((), device=device)
    for safety, auxiliary in zip(safety_gradients, auxiliary_gradients):
        if safety is None or auxiliary is None:
            continue
        safety_value = safety.float()
        auxiliary_value = auxiliary.float()
        dot = dot + (safety_value * auxiliary_value).sum()
        safety_norm_sq = safety_norm_sq + safety_value.square().sum()
        auxiliary_norm_sq = auxiliary_norm_sq + auxiliary_value.square().sum()

    projection_applied = bool(dot.detach().item() < 0.0)
    coefficient = (
        dot / torch.clamp(safety_norm_sq, min=1e-12)
        if projection_applied
        else torch.zeros_like(dot)
    )
    projected: list[torch.Tensor | None] = []
    for safety, auxiliary in zip(safety_gradients, auxiliary_gradients):
        if auxiliary is None:
            projected.append(None)
            continue
        value = auxiliary
        if projection_applied and safety is not None:
            value = auxiliary - coefficient.to(auxiliary.dtype) * safety
        projected.append(value)

    projected_norm_sq = torch.zeros((), device=device)
    for value in projected:
        if value is not None:
            projected_norm_sq = projected_norm_sq + value.float().square().sum()
    safety_norm = torch.sqrt(torch.clamp(safety_norm_sq, min=1e-12))
    projected_norm = torch.sqrt(torch.clamp(projected_norm_sq, min=1e-12))
    cap_value = float(max(0.0, gradient_cap)) * safety_norm
    scale = torch.minimum(
        torch.ones_like(projected_norm),
        cap_value / torch.clamp(projected_norm, min=1e-12),
    )
    projected = [
        None if value is None else value * scale.to(value.dtype)
        for value in projected
    ]
    return projected, {
        "gradientDotBeforeProjection": float(dot.detach().cpu()),
        "gradientSafetyNorm": float(safety_norm.detach().cpu()),
        "gradientAuxiliaryNormBeforeCap": float(auxiliary_norm_sq.sqrt().detach().cpu()),
        "gradientAuxiliaryNormAfterCap": float((projected_norm * scale).detach().cpu()),
        "gradientProjectionApplied": float(projection_applied),
        "gradientAuxiliaryScale": float(scale.detach().cpu()),
    }
