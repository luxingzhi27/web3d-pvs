"""Shared event/right-censoring loss for the offline survival field."""
from __future__ import annotations

from typing import Any

import torch


def survival_censoring_loss(
    model: Any,
    observations: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Optimize the event/right-censored likelihood of the monotone field.

    ``event=1`` means an occluding surface was observed before the target and
    contributes ``1-survival``.  ``event=0`` is right-censored and contributes
    ``survival``.  Both current model families expose the same anchor buffer
    and survival-query contract.
    """
    if not observations or observations["instance"].numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"lossSurvivalCensor": zero, "survivalEventRate": 0.0}

    ids = observations["instance"].long()
    direction_ids = observations["direction"].long()
    rho = observations["rho"].float().view(-1, 1)
    event = observations["event"].float().view(-1, 1)
    weight = torch.clamp(observations["weight"].float().view(-1, 1), min=0.1)
    anchor_owner = getattr(model, "relation_encoder", None)
    if anchor_owner is None:
        anchor_owner = getattr(model, "context_encoder", None)
    if anchor_owner is None or not hasattr(anchor_owner, "anchors"):
        raise AttributeError("survival model must expose relation_encoder.anchors or context_encoder.anchors")
    anchors = anchor_owner.anchors.to(device=rho.device, dtype=rho.dtype)
    if direction_ids.numel() and int(direction_ids.max()) >= int(anchors.shape[0]):
        raise ValueError("survival observation direction exceeds the model anchor count")
    direction = anchors[direction_ids]
    coefficients = model.survival_coefficients[ids]
    semantic, _phi = model.survival_query_from_direction(direction, rho, coefficients)
    survival = torch.clamp(semantic[:, :1], 1e-5, 1.0 - 1e-5)
    nll = -(event * torch.log(1.0 - survival) + (1.0 - event) * torch.log(survival))
    loss = (nll * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("survival censoring loss is non-finite")
    return loss, {"lossSurvivalCensor": loss, "survivalEventRate": event.mean()}


def relation_survival_censoring_loss(
    model: Any,
    observations: dict[str, torch.Tensor],
    geo_table: torch.Tensor,
    query_batch_size: int = 128,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Censoring loss for the offline relation-conditioned survival field.

    Unlike the historical parameter-table field, coefficients are generated
    from the frozen geometry table and the trainable offline relation encoder.
    The relation encoder is therefore optimized by the true event/censoring
    labels and by the final visibility objective.
    """
    if not observations or observations["instance"].numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"lossSurvivalCensor": zero, "survivalEventRate": 0.0}
    if not bool(getattr(model, "relation_field_enabled", False)):
        raise ValueError("relation_survival_censoring_loss requires relation_field_enabled")
    ids = observations["instance"].long()
    direction_ids = observations["direction"].long()
    rho = observations["rho"].float().view(-1, 1)
    event = observations["event"].float().view(-1, 1)
    weight = torch.clamp(observations["weight"].float().view(-1, 1), min=0.1)
    anchor_owner = getattr(model, "context_encoder", None)
    if anchor_owner is None or not hasattr(anchor_owner, "anchors"):
        raise AttributeError("relation model must expose context_encoder.anchors")
    anchors = anchor_owner.anchors.to(device=rho.device, dtype=rho.dtype)
    if direction_ids.numel() and int(direction_ids.max()) >= int(anchors.shape[0]):
        raise ValueError("survival observation direction exceeds evidence anchor count")
    direction = anchors[direction_ids]
    coefficients = model.relation_coefficients_from_geo(
        ids,
        geo_table,
        query_batch_size=query_batch_size,
    )
    semantic, _phi = model.survival_query_from_direction(direction, rho, coefficients)
    survival = torch.clamp(semantic[:, :1], 1e-5, 1.0 - 1e-5)
    nll = -(event * torch.log(1.0 - survival) + (1.0 - event) * torch.log(survival))
    loss = (nll * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("relation survival censoring loss is non-finite")
    return loss, {"lossSurvivalCensor": loss, "survivalEventRate": event.mean()}
