"""Pose-local partial-AUC loss for a visibility score boundary.

The residual is the only trainable quantity in this objective. Positive and
negative membership can follow either the frozen base score or a detached
current score, while the pair loss always uses
``frozen_base_logits + signed_centered_residual``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _flat_float(value: Any, name: str, device: torch.device) -> torch.Tensor:
    """Convert an aligned vector or singleton-column vector to float32."""
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc
    if tensor.ndim == 0 or tensor.ndim > 2 or (
        tensor.ndim == 2 and 1 not in tensor.shape
    ):
        raise ValueError(f"{name} must be a vector or singleton-column vector")
    if torch.is_complex(tensor):
        raise ValueError(f"{name} must be real-valued")
    return tensor.to(device=device, dtype=torch.float32).reshape(-1)


def _pose_offsets(value: Any, item_count: int) -> list[int]:
    """Validate and materialize contiguous pose boundaries on the host."""
    try:
        offsets = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("pose_offsets must be tensor-like") from exc
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("pose_offsets must be a one-dimensional boundary vector")
    if torch.is_complex(offsets) or offsets.dtype == torch.bool:
        raise ValueError("pose_offsets must contain integer offsets")
    if not bool(torch.isfinite(offsets).all()):
        raise ValueError("pose_offsets must be finite")
    if torch.is_floating_point(offsets):
        rounded = offsets.round()
        if not bool(torch.equal(offsets, rounded)):
            raise ValueError("pose_offsets must contain integer offsets")
        offsets = rounded

    values = [int(item) for item in offsets.detach().cpu().tolist()]
    if values[0] != 0 or values[-1] != int(item_count):
        raise ValueError("pose_offsets must start at zero and end at the candidate count")
    if any(
        start < 0 or end < start or end > int(item_count)
        for start, end in zip(values, values[1:])
    ):
        raise ValueError("pose_offsets must be non-decreasing within the candidate range")
    return values


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    try:
        exact = float(value) == float(result)
    except (TypeError, ValueError, OverflowError):
        exact = False
    if result < 0 or not exact:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = _non_negative_integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _weighted_lower_tail(
    base_logits: torch.Tensor,
    positive_indices: torch.Tensor,
    visible_weights: torch.Tensor,
    *,
    mass_fraction: float,
    count_cap: int,
    min_count: int,
) -> torch.Tensor:
    """Select the lowest frozen-base positives up to a weighted mass budget."""
    if positive_indices.numel() == 0:
        return positive_indices

    order = torch.argsort(base_logits[positive_indices].detach(), stable=True)
    ordered_indices = positive_indices[order]
    # Use float64 for cumulative mass so finite large input weights cannot
    # overflow while the membership decision is being made.
    ordered_weights = visible_weights[ordered_indices].detach().to(torch.float64)
    ordered_weights = ordered_weights.clamp_min(0.0)
    if not bool(ordered_weights.sum() > 0.0):
        ordered_weights = torch.ones_like(ordered_weights)

    if float(mass_fraction) >= 1.0:
        mass_count = int(ordered_indices.numel())
    else:
        cumulative = torch.cumsum(ordered_weights, dim=0)
        target_mass = cumulative[-1] * float(mass_fraction)
        mass_count = int(torch.searchsorted(cumulative, target_mass, right=False).item()) + 1
        mass_count = max(1, min(mass_count, int(ordered_indices.numel())))

    selected_count = min(
        int(ordered_indices.numel()),
        int(count_cap),
        max(int(min_count), int(mass_count)),
    )
    return ordered_indices[:selected_count]


def _upper_cvar(values: torch.Tensor, fraction: float) -> torch.Tensor:
    count = min(
        int(values.numel()),
        max(1, int(math.ceil(float(fraction) * int(values.numel())))),
    )
    return torch.topk(values, k=count, sorted=False).values.mean()


def _mean_or_zero(values: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else zero.detach()


def _positive_pair_importance(
    visible_weights: torch.Tensor,
    *,
    transform: str,
    power: float,
) -> torch.Tensor:
    """Map visual utility to positive-pair weights without changing membership."""
    non_negative = visible_weights.detach().clamp_min(0.0)
    if transform == "log1p":
        return torch.log1p(non_negative)
    if transform == "power":
        return non_negative.pow(power)
    raise ValueError("positive_importance_transform must be log1p or power")


def viewcell_tail_partial_auc_loss(
    signed_centered_residual: torch.Tensor,
    frozen_base_logits: torch.Tensor,
    target: torch.Tensor,
    visible_hit_rates: torch.Tensor,
    visible_weights: torch.Tensor,
    pose_offsets: torch.Tensor,
    *,
    tail_selection_logits: torch.Tensor | None = None,
    positive_tail_mass_fraction: float = 0.005,
    positive_tail_count_cap: int = 64,
    min_positive_count: int = 1,
    negative_top_fraction: float = 0.04,
    min_negative_count: int = 1,
    max_negative_count: int = 256,
    margin: float = 0.2,
    temperature: float = 0.25,
    pose_cvar_fraction: float = 0.25,
    pose_cvar_weight: float = 0.5,
    negative_positive_residual_weight: float = 0.5,
    selected_positive_negative_residual_weight: float = 0.5,
    all_positive_negative_residual_weight: float = 0.0,
    tail_classification_weight: float = 0.0,
    tail_classification_margin: float = 0.2,
    residual_l2_weight: float = 0.02,
    positive_importance_transform: str = "log1p",
    positive_importance_power: float = 1.0,
    cross_pose_pair_weight: float = 0.0,
    global_tail_pair_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return a pose-balanced, safety-boundary partial-AUC residual loss.

    Tail membership is detached from either the frozen base score or an
    explicitly supplied current score.  For each pose with both
    classes, all selected positive-tail by negative-tail pairs receive
    ``temperature * softplus((negative - positive + margin) / temperature)``.
    Positive pair weights use either ``log1p(visible_weight)`` or
    ``visible_weight ** power`` times a hit-rate factor in ``[1, 2]``.  The
    power form can align the pair gradient with the linear visual utility used
    by weighted recall.  The residual penalties are one-sided: positive
    residual on negatives and negative residual on selected positives are
    discouraged.

    The count cap is an absolute cap.  Positive and negative minimum counts are
    allowed to exceed the available class count, in which case all available
    members are selected; an impossible minimum/cap combination is rejected.
    """
    if isinstance(signed_centered_residual, torch.Tensor):
        residual = _flat_float(
            signed_centered_residual,
            "signed_centered_residual",
            signed_centered_residual.device,
        )
    else:
        residual = _flat_float(
            signed_centered_residual,
            "signed_centered_residual",
            torch.device("cpu"),
        )
    if residual.numel() == 0:
        raise ValueError("loss requires at least one candidate")
    device = residual.device

    base = _flat_float(frozen_base_logits, "frozen_base_logits", device).detach()
    selection = (
        base
        if tail_selection_logits is None
        else _flat_float(
            tail_selection_logits,
            "tail_selection_logits",
            device,
        ).detach()
    )
    labels = _flat_float(target, "target", device)
    hit_rates = _flat_float(visible_hit_rates, "visible_hit_rates", device)
    weights = _flat_float(visible_weights, "visible_weights", device)
    count = int(residual.numel())
    tensors = {
        "signed_centered_residual": residual,
        "frozen_base_logits": base,
        "tail_selection_logits": selection,
        "target": labels,
        "visible_hit_rates": hit_rates,
        "visible_weights": weights,
    }
    for name, tensor in tensors.items():
        if tensor.numel() != count:
            raise ValueError(f"{name} must align with signed_centered_residual")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("target must contain only binary 0/1 values")
    if bool((hit_rates < 0.0).any() or (hit_rates > 1.0).any()):
        raise ValueError("visible_hit_rates must lie in [0, 1]")
    if bool((weights < 0.0).any()):
        raise ValueError("visible_weights must be non-negative")

    positive_fraction = _finite_scalar(
        positive_tail_mass_fraction, "positive_tail_mass_fraction"
    )
    negative_fraction = _finite_scalar(negative_top_fraction, "negative_top_fraction")
    cvar_fraction = _finite_scalar(pose_cvar_fraction, "pose_cvar_fraction")
    if not 0.0 < positive_fraction <= 1.0:
        raise ValueError("positive_tail_mass_fraction must lie in (0, 1]")
    if not 0.0 < negative_fraction <= 1.0:
        raise ValueError("negative_top_fraction must lie in (0, 1]")
    if not 0.0 < cvar_fraction <= 1.0:
        raise ValueError("pose_cvar_fraction must lie in (0, 1]")

    positive_cap = _positive_integer(
        positive_tail_count_cap, "positive_tail_count_cap"
    )
    minimum_positive = _non_negative_integer(min_positive_count, "min_positive_count")
    minimum_negative = _non_negative_integer(min_negative_count, "min_negative_count")
    maximum_negative = _positive_integer(max_negative_count, "max_negative_count")
    if minimum_positive > positive_cap:
        raise ValueError("min_positive_count cannot exceed positive_tail_count_cap")
    if minimum_negative > maximum_negative:
        raise ValueError("min_negative_count cannot exceed max_negative_count")

    margin_value = _non_negative_scalar(margin, "margin")
    temperature_value = _finite_scalar(temperature, "temperature")
    if temperature_value <= 0.0:
        raise ValueError("temperature must be positive")
    importance_transform = str(positive_importance_transform)
    importance_power = _finite_scalar(
        positive_importance_power,
        "positive_importance_power",
    )
    if importance_transform not in {"log1p", "power"}:
        raise ValueError(
            "positive_importance_transform must be log1p or power"
        )
    if importance_power <= 0.0:
        raise ValueError("positive_importance_power must be positive")
    cvar_weight = _non_negative_scalar(pose_cvar_weight, "pose_cvar_weight")
    cross_pose_weight = _non_negative_scalar(
        cross_pose_pair_weight,
        "cross_pose_pair_weight",
    )
    global_tail_weight = _non_negative_scalar(
        global_tail_pair_weight,
        "global_tail_pair_weight",
    )
    negative_positive_weight = _non_negative_scalar(
        negative_positive_residual_weight,
        "negative_positive_residual_weight",
    )
    selected_positive_negative_weight = _non_negative_scalar(
        selected_positive_negative_residual_weight,
        "selected_positive_negative_residual_weight",
    )
    all_positive_negative_weight = _non_negative_scalar(
        all_positive_negative_residual_weight,
        "all_positive_negative_residual_weight",
    )
    classification_weight = _non_negative_scalar(
        tail_classification_weight,
        "tail_classification_weight",
    )
    classification_margin = _non_negative_scalar(
        tail_classification_margin,
        "tail_classification_margin",
    )
    residual_l2_weight_value = _non_negative_scalar(residual_l2_weight, "residual_l2_weight")
    offsets = _pose_offsets(pose_offsets, count)

    final_logits = base + residual
    if not bool(torch.isfinite(final_logits).all()):
        raise ValueError("frozen_base_logits plus residual must be finite")

    zero = residual.sum() * 0.0
    pose_pair_losses: list[torch.Tensor] = []
    tail_gaps: list[torch.Tensor] = []
    violation_fractions: list[torch.Tensor] = []
    negative_residual_terms: list[torch.Tensor] = []
    selected_positive_residual_terms: list[torch.Tensor] = []
    all_positive_residual_terms: list[torch.Tensor] = []
    tail_classification_terms: list[torch.Tensor] = []
    positive_classification_terms: list[torch.Tensor] = []
    negative_classification_terms: list[torch.Tensor] = []
    positive_correction_satisfied: list[torch.Tensor] = []
    negative_correction_satisfied: list[torch.Tensor] = []
    pose_residual_abs: list[torch.Tensor] = []
    selected_positive_indices: list[torch.Tensor] = []
    selected_negative_indices: list[torch.Tensor] = []
    selected_pair_positive_indices: list[torch.Tensor] = []
    selected_pair_negative_indices: list[torch.Tensor] = []
    selected_pair_positive_pose_ids: list[torch.Tensor] = []
    selected_pair_negative_pose_ids: list[torch.Tensor] = []
    selected_positive_weight_mass_fractions: list[torch.Tensor] = []

    for pose, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if end <= start:
            continue
        local_target = labels[start:end]
        local_base = base[start:end]
        local_selection = selection[start:end]
        local_residual = residual[start:end]
        local_weights = weights[start:end]
        local_hit_rates = hit_rates[start:end]
        pose_residual_abs.append(local_residual.abs().mean())

        positive_indices = torch.nonzero(local_target == 1.0, as_tuple=False).reshape(-1)
        negative_indices = torch.nonzero(local_target == 0.0, as_tuple=False).reshape(-1)
        if positive_indices.numel():
            all_positive_residual_terms.append(
                F.relu(-local_residual[positive_indices]).square().mean()
            )

        selected_positive = _weighted_lower_tail(
            local_selection,
            positive_indices,
            local_weights,
            mass_fraction=positive_fraction,
            count_cap=positive_cap,
            min_count=minimum_positive,
        )
        if selected_positive.numel():
            selected_positive_indices.append(start + selected_positive)
            selected_positive_residual_terms.append(
                F.relu(-local_residual[selected_positive]).square().mean()
            )
            positive_weight_mass = local_weights[positive_indices].sum()
            selected_positive_weight_mass_fractions.append(
                local_weights[selected_positive].sum()
                / positive_weight_mass.clamp_min(1e-12)
            )

        if negative_indices.numel():
            negative_count = min(
                int(negative_indices.numel()),
                maximum_negative,
                max(
                    minimum_negative,
                    int(math.ceil(negative_fraction * int(negative_indices.numel()))),
                ),
            )
            negative_order = torch.argsort(
                local_selection[negative_indices], descending=True, stable=True
            )
            selected_negative = negative_indices[negative_order[:negative_count]]
            selected_negative_indices.append(start + selected_negative)
            negative_residual_terms.append(
                F.relu(local_residual[negative_indices]).square().mean()
            )
        else:
            selected_negative = negative_indices

        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        selected_pair_positive_indices.append(start + selected_positive)
        selected_pair_negative_indices.append(start + selected_negative)
        selected_pair_positive_pose_ids.append(
            torch.full_like(selected_positive, int(pose))
        )
        selected_pair_negative_pose_ids.append(
            torch.full_like(selected_negative, int(pose))
        )
        positive_final = final_logits[start:end][selected_positive]
        negative_final = final_logits[start:end][selected_negative]

        positive_importance = _positive_pair_importance(
            local_weights[selected_positive],
            transform=importance_transform,
            power=importance_power,
        )
        if not bool((positive_importance > 0.0).any()):
            positive_importance = torch.ones_like(positive_importance)
        # A zero-hit positive receives at most twice the pair weight of a
        # hit-rate-one positive, while remaining continuous in hit rate.
        positive_importance = positive_importance * (
            2.0 - local_hit_rates[selected_positive].detach()
        )
        positive_importance = positive_importance.clamp_min(0.0)
        if not bool(positive_importance.sum() > 0.0):
            positive_importance = torch.ones_like(positive_importance)

        positive_correction_values = temperature_value * F.softplus(
            (
                classification_margin
                - local_residual[selected_positive]
            )
            / temperature_value
        )
        positive_correction_loss = (
            positive_correction_values * positive_importance
        ).sum() / positive_importance.sum().clamp_min(1e-12)
        negative_correction_loss = (
            temperature_value
            * F.softplus(
                (
                    classification_margin
                    + local_residual[selected_negative]
                )
                / temperature_value
            )
        ).mean()
        positive_classification_terms.append(positive_correction_loss)
        negative_classification_terms.append(negative_correction_loss)
        tail_classification_terms.append(
            0.5 * (positive_correction_loss + negative_correction_loss)
        )
        positive_correction_satisfied.append(
            (
                (local_residual[selected_positive].detach() >= classification_margin).to(
                    dtype=residual.dtype
                )
                * positive_importance
            ).sum()
            / positive_importance.sum().clamp_min(1e-12)
        )
        negative_correction_satisfied.append(
            (
                local_residual[selected_negative].detach()
                <= -classification_margin
            ).to(dtype=residual.dtype).mean()
        )

        violation = (
            negative_final[:, None]
            - positive_final[None, :]
            + margin_value
        ) / temperature_value
        pair_values = temperature_value * F.softplus(violation)
        normalizer = positive_importance.sum() * float(negative_final.numel())
        pose_pair_losses.append(
            (pair_values * positive_importance[None, :]).sum() / normalizer
        )
        positive_mean = (
            positive_final * positive_importance
        ).sum() / positive_importance.sum().clamp_min(1e-12)
        tail_gaps.append((positive_mean - negative_final.mean()).detach())
        violation_fractions.append(
            (
                (violation.detach() > 0.0).to(dtype=pair_values.dtype)
                * positive_importance[None, :]
            ).sum()
            / normalizer
        )

    loss_pose_mean = torch.stack(pose_pair_losses).mean() if pose_pair_losses else zero
    loss_pose_cvar = (
        _upper_cvar(torch.stack(pose_pair_losses), cvar_fraction)
        if pose_pair_losses
        else zero
    )
    loss_negative_positive_residual = (
        torch.stack(negative_residual_terms).mean() if negative_residual_terms else zero
    )
    loss_selected_positive_negative_residual = (
        torch.stack(selected_positive_residual_terms).mean()
        if selected_positive_residual_terms
        else zero
    )
    loss_all_positive_negative_residual = (
        torch.stack(all_positive_residual_terms).mean()
        if all_positive_residual_terms
        else zero
    )
    loss_tail_classification = (
        torch.stack(tail_classification_terms).mean()
        if tail_classification_terms
        else zero
    )
    if selected_pair_positive_indices and selected_pair_negative_indices:
        cross_positive_indices = torch.cat(selected_pair_positive_indices)
        cross_negative_indices = torch.cat(selected_pair_negative_indices)
        cross_positive_pose_ids = torch.cat(selected_pair_positive_pose_ids)
        cross_negative_pose_ids = torch.cat(selected_pair_negative_pose_ids)
        cross_positive_importance = _positive_pair_importance(
            weights[cross_positive_indices],
            transform=importance_transform,
            power=importance_power,
        ) * (2.0 - hit_rates[cross_positive_indices].detach())
        cross_positive_importance = cross_positive_importance.clamp_min(0.0)
        if not bool(cross_positive_importance.sum() > 0.0):
            cross_positive_importance = torch.ones_like(cross_positive_importance)
        cross_pose_mask = (
            cross_negative_pose_ids[:, None]
            != cross_positive_pose_ids[None, :]
        )
        cross_pose_normalizer = (
            cross_pose_mask.to(dtype=residual.dtype)
            * cross_positive_importance[None, :]
        ).sum()
        if bool(cross_pose_normalizer > 0.0):
            cross_pose_violation = (
                final_logits[cross_negative_indices, None]
                - final_logits[cross_positive_indices][None, :]
                + margin_value
            ) / temperature_value
            cross_pose_pair_values = temperature_value * F.softplus(
                cross_pose_violation
            )
            cross_pose_weights = (
                cross_pose_mask.to(dtype=residual.dtype)
                * cross_positive_importance[None, :]
            )
            loss_cross_pose_pair = (
                cross_pose_pair_values * cross_pose_weights
            ).sum() / cross_pose_normalizer
            cross_pose_violation_fraction = (
                (cross_pose_violation.detach() > 0.0).to(dtype=residual.dtype)
                * cross_pose_weights
            ).sum() / cross_pose_normalizer
        else:
            loss_cross_pose_pair = zero
            cross_pose_violation_fraction = zero.detach()
    else:
        loss_cross_pose_pair = zero
        cross_pose_violation_fraction = zero.detach()

    global_positive_indices = torch.nonzero(
        labels == 1.0,
        as_tuple=False,
    ).reshape(-1)
    global_negative_indices = torch.nonzero(
        labels == 0.0,
        as_tuple=False,
    ).reshape(-1)
    global_selected_positive = _weighted_lower_tail(
        selection,
        global_positive_indices,
        weights,
        mass_fraction=positive_fraction,
        count_cap=positive_cap,
        min_count=minimum_positive,
    )
    if global_negative_indices.numel():
        global_negative_count = min(
            int(global_negative_indices.numel()),
            maximum_negative,
            max(
                minimum_negative,
                int(
                    math.ceil(
                        negative_fraction * int(global_negative_indices.numel())
                    )
                ),
            ),
        )
        global_negative_order = torch.argsort(
            selection[global_negative_indices],
            descending=True,
            stable=True,
        )
        global_selected_negative = global_negative_indices[
            global_negative_order[:global_negative_count]
        ]
    else:
        global_selected_negative = global_negative_indices
    if global_selected_positive.numel() and global_selected_negative.numel():
        global_positive_importance = _positive_pair_importance(
            weights[global_selected_positive],
            transform=importance_transform,
            power=importance_power,
        ) * (2.0 - hit_rates[global_selected_positive].detach())
        global_positive_importance = global_positive_importance.clamp_min(0.0)
        if not bool(global_positive_importance.sum() > 0.0):
            global_positive_importance = torch.ones_like(
                global_positive_importance
            )
        global_violation = (
            final_logits[global_selected_negative][:, None]
            - final_logits[global_selected_positive][None, :]
            + margin_value
        ) / temperature_value
        global_pair_values = temperature_value * F.softplus(global_violation)
        global_normalizer = (
            global_positive_importance.sum()
            * float(global_selected_negative.numel())
        )
        loss_global_tail_pair = (
            global_pair_values * global_positive_importance[None, :]
        ).sum() / global_normalizer.clamp_min(1e-12)
        global_tail_violation_fraction = (
            (global_violation.detach() > 0.0).to(dtype=residual.dtype)
            * global_positive_importance[None, :]
        ).sum() / global_normalizer.clamp_min(1e-12)
    else:
        loss_global_tail_pair = zero
        global_tail_violation_fraction = zero.detach()
    loss_residual_l2 = residual.square().mean()
    total = (
        loss_pose_mean
        + cvar_weight * loss_pose_cvar
        + cross_pose_weight * loss_cross_pose_pair
        + global_tail_weight * loss_global_tail_pair
        + negative_positive_weight * loss_negative_positive_residual
        + selected_positive_negative_weight * loss_selected_positive_negative_residual
        + all_positive_negative_weight * loss_all_positive_negative_residual
        + classification_weight * loss_tail_classification
        + residual_l2_weight_value * loss_residual_l2
    )
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("view-cell tail partial-AUC loss is non-finite")

    positive_mask = labels == 1.0
    negative_mask = labels == 0.0
    all_positive_mean = (
        residual[positive_mask].mean().detach() if bool(positive_mask.any()) else zero.detach()
    )
    all_negative_mean = (
        residual[negative_mask].mean().detach() if bool(negative_mask.any()) else zero.detach()
    )
    selected_positive = (
        torch.cat(selected_positive_indices)
        if selected_positive_indices
        else torch.empty(0, dtype=torch.long, device=device)
    )
    selected_negative = (
        torch.cat(selected_negative_indices)
        if selected_negative_indices
        else torch.empty(0, dtype=torch.long, device=device)
    )
    pair_positive = (
        torch.cat(selected_pair_positive_indices)
        if selected_pair_positive_indices
        else torch.empty(0, dtype=torch.long, device=device)
    )
    pair_negative = (
        torch.cat(selected_pair_negative_indices)
        if selected_pair_negative_indices
        else torch.empty(0, dtype=torch.long, device=device)
    )

    selected_positive_mean = (
        residual[selected_positive].mean().detach()
        if selected_positive.numel()
        else zero.detach()
    )
    selected_negative_mean = (
        residual[selected_negative].mean().detach()
        if selected_negative.numel()
        else zero.detach()
    )
    stats: dict[str, torch.Tensor | float] = {
        "lossViewcellTailPartialAuc": total,
        "lossTailPartialAuc": total,
        "lossPoseMean": loss_pose_mean,
        "lossPoseCvar": loss_pose_cvar,
        "lossCrossPosePair": loss_cross_pose_pair,
        "lossCrossPosePairWeighted": cross_pose_weight * loss_cross_pose_pair,
        "lossGlobalTailPair": loss_global_tail_pair,
        "lossGlobalTailPairWeighted": global_tail_weight * loss_global_tail_pair,
        "lossNegativePositiveResidual": loss_negative_positive_residual,
        "lossSelectedPositiveNegativeResidual": loss_selected_positive_negative_residual,
        "lossAllPositiveNegativeResidual": loss_all_positive_negative_residual,
        "lossTailClassification": loss_tail_classification,
        "lossTailClassificationPositive": _mean_or_zero(
            positive_classification_terms, zero
        ),
        "lossTailClassificationNegative": _mean_or_zero(
            negative_classification_terms, zero
        ),
        "lossResidualL2": loss_residual_l2,
        "lossPoseCvarWeighted": cvar_weight * loss_pose_cvar,
        "lossNegativePositiveResidualWeighted": (
            negative_positive_weight * loss_negative_positive_residual
        ),
        "lossSelectedPositiveNegativeResidualWeighted": (
            selected_positive_negative_weight * loss_selected_positive_negative_residual
        ),
        "lossAllPositiveNegativeResidualWeighted": (
            all_positive_negative_weight * loss_all_positive_negative_residual
        ),
        "lossTailClassificationWeighted": (
            classification_weight * loss_tail_classification
        ),
        "lossResidualL2Weighted": residual_l2_weight_value * loss_residual_l2,
        "tailGap": _mean_or_zero(tail_gaps, zero),
        "tailViolationFraction": _mean_or_zero(violation_fractions, zero),
        "crossPoseViolationFraction": cross_pose_violation_fraction,
        "crossPosePairWeight": float(cross_pose_weight),
        "globalTailViolationFraction": global_tail_violation_fraction,
        "globalTailPairWeight": float(global_tail_weight),
        "globalTailPositiveCount": float(global_selected_positive.numel()),
        "globalTailNegativeCount": float(global_selected_negative.numel()),
        "poseCount": float(len(pose_pair_losses)),
        "nonEmptyPoseCount": float(len(pose_residual_abs)),
        "positiveCount": float(selected_positive.numel()),
        "negativeCount": float(selected_negative.numel()),
        "selectedPositiveCount": float(selected_positive.numel()),
        "selectedNegativeCount": float(selected_negative.numel()),
        "selectedPositiveWeightMassFraction": _mean_or_zero(
            selected_positive_weight_mass_fractions,
            zero,
        ),
        "positiveImportancePower": float(importance_power),
        "positiveCandidateCount": float(positive_mask.sum()),
        "negativeCandidateCount": float(negative_mask.sum()),
        "positiveResidualMean": all_positive_mean,
        "negativeResidualMean": all_negative_mean,
        "selectedPositiveResidualMean": selected_positive_mean,
        "selectedNegativeResidualMean": selected_negative_mean,
        "residualMean": residual.mean().detach(),
        "residualMeanAbs": residual.abs().mean().detach(),
        "poseResidualMeanAbs": _mean_or_zero(pose_residual_abs, zero),
        "pairPositiveResidualMean": (
            residual[pair_positive].mean().detach()
            if pair_positive.numel()
            else zero.detach()
        ),
        "pairNegativeResidualMean": (
            residual[pair_negative].mean().detach()
            if pair_negative.numel()
            else zero.detach()
        ),
        "tailPositiveCorrectionSatisfiedFraction": _mean_or_zero(
            positive_correction_satisfied, zero
        ),
        "tailNegativeCorrectionSatisfiedFraction": _mean_or_zero(
            negative_correction_satisfied, zero
        ),
    }
    return total, stats


__all__ = ["viewcell_tail_partial_auc_loss"]
