"""Train-only view-cell exposure supervision for shared query features."""
from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn.functional as F


def _pose_slices(pose_offsets: torch.Tensor, count: int) -> Iterator[tuple[int, int]]:
    offsets = torch.as_tensor(pose_offsets, dtype=torch.long).detach().cpu().reshape(-1)
    if offsets.numel() < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != int(count):
        raise ValueError("pose_offsets must start at zero and cover every exposure row")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("pose_offsets must be non-decreasing")
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        if int(end) > int(start):
            yield int(start), int(end)


def viewcell_exposure_supervision_loss(
    exposure_logits: torch.Tensor,
    union_target: torch.Tensor,
    visible_hit_rates: torch.Tensor,
    pose_offsets: torch.Tensor,
    *,
    positive_class_fraction: float = 0.25,
    boundary_hit_rate: float = 0.20,
    stable_hit_rate: float = 0.80,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Fit continuous subpose hit rates with per-pose class normalization."""
    logits = torch.as_tensor(exposure_logits).float().reshape(-1)
    labels = torch.as_tensor(union_target, device=logits.device).float().reshape(-1)
    hit_rates = torch.as_tensor(
        visible_hit_rates, device=logits.device
    ).float().reshape(-1)
    if logits.numel() == 0 or not (
        logits.numel() == labels.numel() == hit_rates.numel()
    ):
        raise ValueError("exposure logits, union targets, and hit rates must align")
    if not bool(
        torch.isfinite(logits).all()
        and torch.isfinite(labels).all()
        and torch.isfinite(hit_rates).all()
    ):
        raise ValueError("exposure supervision inputs must be finite")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("union_target must contain only binary values")
    if bool((hit_rates < 0.0).any() or (hit_rates > 1.0).any()):
        raise ValueError("visible_hit_rates must lie in [0, 1]")
    if bool((hit_rates[labels < 0.5] != 0.0).any()):
        raise ValueError("negative candidates must have zero subpose hit rate")
    if bool((hit_rates[labels > 0.5] <= 0.0).any()):
        raise ValueError("visible union members must have a positive subpose hit rate")
    positive_share = float(positive_class_fraction)
    if not math.isfinite(positive_share) or not 0.0 < positive_share < 1.0:
        raise ValueError("positive_class_fraction must lie in (0, 1)")
    if not 0.0 < float(boundary_hit_rate) < float(stable_hit_rate) <= 1.0:
        raise ValueError("exposure diagnostic hit-rate boundaries are invalid")
    tuple(_pose_slices(pose_offsets, logits.numel()))

    row_loss = F.binary_cross_entropy_with_logits(
        logits, hit_rates, reduction="none"
    )
    pose_losses: list[torch.Tensor] = []
    for start, end in _pose_slices(pose_offsets, logits.numel()):
        local_labels = labels[start:end] > 0.5
        if not bool(local_labels.any()) or not bool((~local_labels).any()):
            continue
        pose_losses.append(
            positive_share * row_loss[start:end][local_labels].mean()
            + (1.0 - positive_share) * row_loss[start:end][~local_labels].mean()
        )
    if not pose_losses:
        raise ValueError("exposure supervision requires a mixed-label pose")
    loss = torch.stack(pose_losses).mean()
    probability = torch.sigmoid(logits)
    absolute_error = (probability - hit_rates).abs()
    positive = labels > 0.5
    boundary = positive & (hit_rates <= float(boundary_hit_rate))
    stable = positive & (hit_rates >= float(stable_hit_rate))
    invisible = ~positive

    def group_mean(mask: torch.Tensor) -> torch.Tensor:
        return (
            absolute_error[mask].mean()
            if bool(mask.any())
            else absolute_error.sum() * 0.0
        )

    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("view-cell exposure loss became non-finite")
    return loss, {
        "lossViewcellExposureSupervision": loss,
        "viewcellExposureMae": absolute_error.mean(),
        "viewcellExposureBoundaryMae": group_mean(boundary),
        "viewcellExposureStableMae": group_mean(stable),
        "viewcellExposureInvisibleMae": group_mean(invisible),
        "viewcellExposureBoundaryCount": float(boundary.sum()),
        "viewcellExposureStableCount": float(stable.sum()),
        "viewcellExposureInvisibleCount": float(invisible.sum()),
        "viewcellExposurePositiveClassFraction": positive_share,
    }


__all__ = ["viewcell_exposure_supervision_loss"]
