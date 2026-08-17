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


def integrated_relation_survival_censoring_loss(
    model: Any,
    coefficients: torch.Tensor,
    observations: dict[str, torch.Tensor],
    *,
    max_observations: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Train the new offline relation encoder with event/right-censor labels.

    The coefficient table is generated offline from the directed relation CSR.
    This loss never reads validation/test visibility labels.  A layer event is
    a failure of survival before ``rho``; a first-layer observation is a
    right-censored sample.  The model owns the direction-conditioned query so
    the same monotone field is used by both offline supervision and runtime.
    """
    if not observations or observations["instance"].numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"lossSurvivalCensor": zero, "survivalEventRate": 0.0}
    ids = observations["instance"].long().reshape(-1)
    direction_ids = observations["direction"].long().reshape(-1)
    rho = observations["rho"].float().reshape(-1, 1)
    event = observations["event"].float().reshape(-1, 1)
    weight = observations["weight"].float().reshape(-1, 1).clamp_min(0.1)
    if max_observations is not None and ids.numel() > int(max_observations):
        # Deterministic stride sampling keeps the loss bounded without using a
        # random validation/test signal or changing the observation semantics.
        keep = torch.linspace(
            0, ids.numel() - 1, steps=int(max_observations), device=ids.device
        ).long()
        ids, direction_ids, rho, event, weight = (
            value[keep] for value in (ids, direction_ids, rho, event, weight)
        )
    direction_anchors = model.offline_survival_encoder.direction_embedding.new_tensor(
        [
            [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0],
            [0.0, 0.70710677, 0.70710677], [0.70710677, 0.70710677, 0.0],
            [0.0, 0.70710677, -0.70710677], [-0.70710677, 0.70710677, 0.0],
            [0.0, -0.70710677, 0.70710677], [0.70710677, -0.70710677, 0.0],
            [0.0, -0.70710677, -0.70710677], [-0.70710677, -0.70710677, 0.0],
        ]
    )
    if direction_ids.numel() and int(direction_ids.max()) >= direction_anchors.shape[0]:
        raise ValueError("survival observation direction exceeds the registered 12 anchors")
    direction = direction_anchors[direction_ids]
    semantic, _basis = model.query_survival_from_direction(
        direction,
        rho,
        coefficients[ids],
    )
    survival = semantic[:, :1].clamp(1e-5, 1.0 - 1e-5)
    nll = -(event * torch.log1p(-survival) + (1.0 - event) * torch.log(survival))
    loss = (nll * weight).sum() / weight.sum().clamp_min(1.0)
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("integrated relation survival censoring loss is non-finite")
    return loss, {
        "lossSurvivalCensor": loss,
        "survivalEventRate": event.mean(),
        "survivalObservationCount": float(ids.numel()),
    }


def stratified_survival_censoring_loss(
    model: Any,
    coefficients: torch.Tensor,
    observations: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """V3 event/censor likelihood with probability-corrected train weights.

    The caller must provide observations selected by the train-only stratified
    sampler.  ``normalized_depth`` is the frozen radius-relative log-depth;
    the historical ``rho=d/(d+r)`` coordinate is deliberately not accepted.
    """
    required = ("instance", "direction", "normalized_depth", "event", "weight")
    if any(key not in observations for key in required):
        raise ValueError(f"v3 survival observations require {required}")
    ids = observations["instance"].long().reshape(-1)
    if ids.numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {
            "lossSurvivalCensor": zero,
            "survivalEventRate": 0.0,
            "survivalObservationCount": 0.0,
        }
    direction_ids = observations["direction"].long().reshape(-1)
    depth = observations["normalized_depth"].float().reshape(-1, 1)
    event = observations["event"].float().reshape(-1, 1)
    weight = observations["weight"].float().reshape(-1, 1)
    if any(value.numel() != ids.numel() for value in (direction_ids, depth, event, weight)):
        raise ValueError("v3 survival observation arrays must align")
    if bool((ids < 0).any()) or int(ids.max()) >= int(coefficients.shape[0]):
        raise ValueError("v3 survival observation instance is outside the coefficient table")
    if bool((direction_ids < 0).any()) or bool((direction_ids >= 12).any()):
        raise ValueError("v3 survival direction must be in [0, 11]")
    if bool((depth < 0.0).any()) or bool((depth > 1.0).any()):
        raise ValueError("v3 normalized survival depth must be in [0, 1]")
    if bool(((event != 0.0) & (event != 1.0)).any()):
        raise ValueError("v3 survival event must be zero or one")
    if bool((weight < 0.0).any()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("v3 survival weights must be finite and non-negative")
    anchors = depth.new_tensor(
        [
            [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0],
            [0.0, 0.70710677, 0.70710677], [0.70710677, 0.70710677, 0.0],
            [0.0, 0.70710677, -0.70710677], [-0.70710677, 0.70710677, 0.0],
            [0.0, -0.70710677, 0.70710677], [0.70710677, -0.70710677, 0.0],
            [0.0, -0.70710677, -0.70710677], [-0.70710677, -0.70710677, 0.0],
        ]
    )
    semantic, _basis = model.query_survival_from_direction(
        anchors[direction_ids], depth, coefficients[ids]
    )
    survival = semantic[:, :1].clamp(1e-5, 1.0 - 1e-5)
    nll = -(event * torch.log1p(-survival) + (1.0 - event) * torch.log(survival))
    denominator = weight.sum().clamp_min(1e-8)
    loss = (nll * weight).sum() / denominator
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("v3 stratified survival loss is non-finite")
    return loss, {
        "lossSurvivalCensor": loss,
        "survivalEventRate": (event * weight).sum() / denominator,
        "survivalObservationCount": float(ids.numel()),
        "survivalEffectiveWeight": denominator,
    }
