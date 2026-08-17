"""Training-only safety-reserve loss for low-threshold PVS operation.

The safety objective keeps the registered RVL control and adds a positive
boundary-tail CVaR. Efficiency is optimized only after a detached, warmup-aware
safety-reserve gate opens. No calibration, validation, or test threshold is an
input to this module: operating thresholds are derived solely from the train
seed and optimizer step.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from common.safety_constraint_utility_loss import rvl_strong_v2_visibility_loss


OPERATING_THRESHOLD_RANGE = (1e-4, 0.2)
OPERATING_THRESHOLD_ANCHORS = (0.001, 0.005, 0.02, 0.10)

GradientList = Sequence[torch.Tensor | None]


def _pose_slices(offsets: torch.Tensor, item_count: int):
    # Pose boundaries are metadata, not differentiable inputs.  Keeping one
    # detached CPU copy avoids a scalar device synchronization for every pose
    # when this train-only loss is evaluated on CUDA.
    values = torch.as_tensor(offsets, dtype=torch.long).detach().cpu().reshape(-1)
    if values.numel() < 2:
        raise ValueError("pose_offsets must contain at least a start and end")
    if int(values[0]) != 0 or int(values[-1]) != int(item_count):
        raise ValueError("pose_offsets must start at zero and cover every item")
    if bool((values[1:] < values[:-1]).any()):
        raise ValueError("pose_offsets must be non-decreasing")
    for pose_index in range(values.numel() - 1):
        start = int(values[pose_index])
        end = int(values[pose_index + 1])
        if end > start:
            yield pose_index, start, end


def _hash_uniform(train_seed: int, global_step: int, sample_index: int) -> float:
    payload = f"pvs-v3:{int(train_seed)}:{int(global_step)}:{int(sample_index)}".encode("ascii")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little")
    return (float(integer) + 0.5) / float(1 << 64)


def sample_train_operating_thresholds(
    train_seed: int,
    global_step: int,
    *,
    sample_count: int = 2,
    minimum: float = OPERATING_THRESHOLD_RANGE[0],
    maximum: float = OPERATING_THRESHOLD_RANGE[1],
    anchors: Sequence[float] = OPERATING_THRESHOLD_ANCHORS,
    anchor_period: int = 4,
    split: str = "train",
) -> tuple[float, ...]:
    """Return deterministic train-only log-uniform thresholds for one step.

    Two random thresholds are used by default. Every ``anchor_period`` steps,
    one fixed anchor is appended; successive anchor steps cycle through the
    registered anchor set. The function does not touch global Python or Torch
    RNG state.
    """
    if str(split) != "train":
        raise ValueError("operating thresholds are train-only")
    if int(train_seed) < 0 or int(global_step) < 0:
        raise ValueError("train_seed and global_step must be non-negative")
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 < float(minimum) < float(maximum) < 1.0:
        raise ValueError("threshold range must satisfy 0 < minimum < maximum < 1")
    if int(anchor_period) < 0:
        raise ValueError("anchor_period must be non-negative")
    anchor_values = tuple(float(value) for value in anchors)
    if any(not float(minimum) <= value <= float(maximum) for value in anchor_values):
        raise ValueError("every anchor must lie inside the operating threshold range")

    log_minimum = math.log(float(minimum))
    log_span = math.log(float(maximum)) - log_minimum
    result = [
        math.exp(log_minimum + _hash_uniform(train_seed, global_step, index) * log_span)
        for index in range(int(sample_count))
    ]
    if anchor_values and int(anchor_period) > 0 and int(global_step) % int(anchor_period) == 0:
        anchor_index = (int(global_step) // int(anchor_period)) % len(anchor_values)
        result.append(anchor_values[anchor_index])
    return tuple(result)


def bounded_topk_smooth_max(
    logits: torch.Tensor,
    *,
    top_k: int = 8,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Return a bounded smooth maximum over at most the highest eight logits."""
    values = logits.float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("bounded smooth-max requires at least one logit")
    if int(top_k) <= 0 or float(temperature) <= 0.0:
        raise ValueError("top_k and temperature must be positive")
    selected = torch.topk(values, k=min(int(top_k), values.numel()), sorted=False).values
    weights = torch.softmax(selected / float(temperature), dim=0)
    return (weights * selected).sum()


def _group_topk_smooth_max(
    logits: torch.Tensor,
    inverse_groups: torch.Tensor,
    group_count: int,
    *,
    top_k: int = 8,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Compute bounded smooth-max for every group without host-side loops."""
    values = logits.float().reshape(-1)
    groups = torch.as_tensor(
        inverse_groups, device=values.device, dtype=torch.long
    ).reshape(-1)
    if values.numel() == 0 or values.numel() != groups.numel():
        raise ValueError("group smooth-max inputs must be non-empty and aligned")
    if int(group_count) <= 0 or int(top_k) <= 0 or float(temperature) <= 0.0:
        raise ValueError("group_count, top_k, and temperature must be positive")

    # Sort logits first, then stably sort by group.  This leaves each group in
    # descending logit order, so a rank mask selects exactly its top-k values.
    value_order = torch.argsort(values, descending=True, stable=True)
    order = value_order[torch.argsort(groups[value_order], stable=True)]
    ordered_values = values[order]
    ordered_groups = groups[order]
    starts_mask = torch.ones(ordered_values.numel(), device=values.device, dtype=torch.bool)
    if ordered_values.numel() > 1:
        starts_mask[1:] = ordered_groups[1:] != ordered_groups[:-1]
    starts = torch.nonzero(starts_mask, as_tuple=False).reshape(-1)
    ends = torch.cat(
        [starts[1:], starts.new_tensor([ordered_values.numel()])]
    )
    ranks = torch.arange(ordered_values.numel(), device=values.device) - torch.repeat_interleave(
        starts, ends - starts
    )
    selected = ranks < int(top_k)
    selected_values = ordered_values[selected]
    selected_groups = ordered_groups[selected]
    group_max = torch.full(
        (int(group_count),), float("-inf"), device=values.device, dtype=values.dtype
    )
    group_max.scatter_reduce_(
        0, selected_groups, selected_values, reduce="amax", include_self=True
    )
    exp_values = torch.exp(
        (selected_values - group_max[selected_groups]) / float(temperature)
    )
    denominator = torch.zeros_like(group_max)
    denominator.scatter_add_(0, selected_groups, exp_values)
    numerator = torch.zeros_like(group_max)
    numerator.scatter_add_(0, selected_groups, exp_values * selected_values)
    return numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny)


def soft_request_probability(
    group_logit: torch.Tensor,
    threshold: float,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Map one bounded GLB logit to a smooth request probability."""
    tau = float(threshold)
    if not 0.0 < tau < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    value = group_logit.float()
    boundary = value.new_tensor(math.log(tau / (1.0 - tau)))
    return torch.sigmoid((value - boundary) / float(temperature))


def _normalize_pose_visible_weights(
    visible_weights: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    *,
    kappa: float = 99.0,
) -> torch.Tensor:
    weights = visible_weights.float().reshape(-1).clamp_min(0.0)
    labels = target.float().reshape(-1)
    if weights.numel() != labels.numel():
        raise ValueError("visible_weights and target must align")
    if float(kappa) <= 0.0:
        raise ValueError("weight normalization kappa must be positive")
    normalized = torch.zeros_like(weights)
    denominator = weights.new_tensor(math.log1p(float(kappa)))
    for _pose, start, end in _pose_slices(pose_offsets, weights.numel()):
        positive = labels[start:end] > 0.5
        if not bool(positive.any()):
            continue
        local = weights[start:end]
        maximum = local[positive].max().clamp_min(1e-6)
        relative = (local / maximum).clamp(0.0, 1.0)
        normalized[start:end] = torch.log1p(float(kappa) * relative) / denominator
        normalized[start:end] *= positive.to(normalized.dtype)
    return normalized


def _upper_tail_cvar(values: torch.Tensor, fraction: float) -> tuple[torch.Tensor, int]:
    flat = values.float().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("CVaR requires at least one value")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("CVaR fraction must lie in (0, 1]")
    count = min(flat.numel(), max(1, int(math.ceil(float(fraction) * flat.numel()))))
    return torch.topk(flat, k=count, sorted=False).values.mean(), count


def _weighted_lower_quantile(
    values: torch.Tensor,
    weights: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    flat_values = values.float().reshape(-1)
    flat_weights = weights.float().reshape(-1).clamp_min(0.0)
    if flat_values.numel() == 0 or flat_values.numel() != flat_weights.numel():
        raise ValueError("weighted quantile values and weights must be non-empty and aligned")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    positive_weight = flat_weights > 0.0
    if bool(positive_weight.any()):
        flat_values = flat_values[positive_weight]
        flat_weights = flat_weights[positive_weight]
    else:
        flat_weights = torch.ones_like(flat_values)
    order = torch.argsort(flat_values.detach())
    cumulative = torch.cumsum(flat_weights[order], dim=0)
    target = cumulative[-1] * float(quantile)
    index = torch.searchsorted(cumulative, target, right=False).clamp_max(cumulative.numel() - 1)
    return flat_values[order[index]]


def safety_reserve_gate(
    positive_logits: torch.Tensor,
    positive_weights: torch.Tensor,
    threshold: float,
    *,
    global_step: int,
    total_optimizer_steps: int,
    warmup_fraction: float = 0.10,
    reserve_quantile: float = 0.01,
    reserve_margin: float = 0.0,
    gate_temperature: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a detached reserve gate and weighted low-tail logit margin."""
    values = positive_logits.float().reshape(-1)
    weights = positive_weights.float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("safety reserve requires at least one positive logit")
    if values.numel() != weights.numel():
        raise ValueError("positive logits and weights must align")
    if int(global_step) < 0 or int(total_optimizer_steps) <= 0:
        raise ValueError("optimizer step values must be valid")
    if not 0.0 <= float(warmup_fraction) <= 1.0:
        raise ValueError("warmup_fraction must lie in [0, 1]")
    if float(gate_temperature) <= 0.0:
        raise ValueError("gate_temperature must be positive")
    tau = float(threshold)
    if not 0.0 < tau < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")

    lower = _weighted_lower_quantile(values, weights, reserve_quantile)
    boundary = lower.new_tensor(math.log(tau / (1.0 - tau)))
    margin = (lower - boundary).detach()
    warmup_steps = int(math.ceil(float(warmup_fraction) * int(total_optimizer_steps)))
    if int(global_step) < warmup_steps:
        gate = margin.new_zeros(())
    else:
        gate = torch.sigmoid((margin - float(reserve_margin)) / float(gate_temperature)).detach()
    return gate, margin


def safety_reserve_operating_utility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    visible_hit_rates: torch.Tensor | None,
    instance_ids: torch.Tensor | None,
    instance_to_glb: torch.Tensor,
    glb_bytes: torch.Tensor,
    *,
    train_seed: int,
    global_step: int,
    total_optimizer_steps: int,
    split: str = "train",
    boundary_tail_weight: float = 0.30,
    boundary_tail_fraction: float = 0.01,
    rare_positive_weight: float = 1.0,
    negative_band_weight: float = 0.03,
    negative_tail_fraction: float = 0.02,
    glb_resource_weight: float = 0.05,
    threshold_temperature: float = 0.10,
    negative_band_shape: str = "sigmoid",
    group_temperature: float = 0.10,
    request_temperature: float = 0.10,
    reserve_quantile: float = 0.01,
    reserve_margin: float = 0.0,
    gate_temperature: float = 0.25,
    warmup_fraction: float = 0.10,
    threshold_sample_count: int = 2,
    threshold_anchor_period: int = 4,
    rvl_bce_positive_weight: float = 14.0,
    rvl_tversky_fn_weight: float = 7.0,
    rvl_count_weight: float = 0.10,
    rvl_fp_normalization: str = "positive",
    rvl_rank_weight: float = 0.45,
    rvl_rank_negative_top_k: int = 256,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Compute RVL safety, boundary protection, and reserve-gated efficiency."""
    if str(split) != "train":
        raise ValueError("safety-reserve operating utility loss is train-only")
    for name, value in (
        ("boundary_tail_weight", boundary_tail_weight),
        ("rare_positive_weight", rare_positive_weight),
        ("negative_band_weight", negative_band_weight),
        ("glb_resource_weight", glb_resource_weight),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if float(threshold_temperature) <= 0.0:
        raise ValueError("threshold_temperature must be positive")
    if str(negative_band_shape) not in {"sigmoid", "softplus"}:
        raise ValueError("negative_band_shape must be sigmoid or softplus")

    flat_logits = logits.float().reshape(-1)
    labels = target.float().reshape(-1)
    raw_weights = visible_weights.float().reshape(-1)
    if flat_logits.numel() == 0:
        raise ValueError("loss requires at least one candidate")
    if labels.numel() != flat_logits.numel() or raw_weights.numel() != flat_logits.numel():
        raise ValueError("logits, target, and visible_weights must align")
    if not bool(torch.isfinite(flat_logits).all()) or not bool(torch.isfinite(raw_weights).all()):
        raise ValueError("loss inputs must be finite")

    hit_rates = (
        torch.ones_like(flat_logits)
        if visible_hit_rates is None
        else torch.as_tensor(visible_hit_rates, device=flat_logits.device).float().reshape(-1)
    )
    if hit_rates.numel() != flat_logits.numel() or not bool(torch.isfinite(hit_rates).all()):
        raise ValueError("visible_hit_rates must be finite and align with logits")
    hit_rates = hit_rates.clamp(0.0, 1.0)
    ids = (
        torch.arange(flat_logits.numel(), device=flat_logits.device, dtype=torch.long)
        if instance_ids is None
        else torch.as_tensor(instance_ids, device=flat_logits.device, dtype=torch.long).reshape(-1)
    )
    if ids.numel() != flat_logits.numel():
        raise ValueError("instance_ids must align with logits")
    mapping = torch.as_tensor(instance_to_glb, device=flat_logits.device, dtype=torch.long).reshape(-1)
    if mapping.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= mapping.numel():
        raise ValueError("instance_ids contains an ID outside instance_to_glb")
    candidate_glbs = mapping[ids]
    costs = torch.as_tensor(glb_bytes, device=flat_logits.device).float().reshape(-1)
    if candidate_glbs.numel() and (int(candidate_glbs.min()) < 0 or int(candidate_glbs.max()) >= costs.numel()):
        raise ValueError("instance_to_glb points outside glb_bytes")
    if not bool(torch.isfinite(costs).all()) or bool((costs < 0.0).any()):
        raise ValueError("glb_bytes must be finite and non-negative")

    # All downstream pose loops only need integer boundaries.  Reuse one CPU
    # metadata copy, including for the shared RVL control loss.
    pose_offsets_cpu = torch.as_tensor(pose_offsets, dtype=torch.long).detach().cpu().reshape(-1)
    normalized_weights = _normalize_pose_visible_weights(raw_weights, labels, pose_offsets_cpu)
    rvl_loss, rvl_parts = rvl_strong_v2_visibility_loss(
        flat_logits,
        labels,
        pose_offsets_cpu,
        normalized_weights,
        torch.zeros_like(flat_logits),
        bce_positive_weight=rvl_bce_positive_weight,
        tversky_fn_weight=rvl_tversky_fn_weight,
        count_weight=rvl_count_weight,
        fp_normalization=rvl_fp_normalization,
        rank_weight=rvl_rank_weight,
        rank_negative_top_k=rvl_rank_negative_top_k,
    )
    thresholds = sample_train_operating_thresholds(
        train_seed,
        global_step,
        sample_count=threshold_sample_count,
        anchor_period=threshold_anchor_period,
        split=split,
    )
    scores = torch.sigmoid(flat_logits)
    zero = flat_logits.sum() * 0.0
    boundary_terms: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []
    resource_terms: list[torch.Tensor] = []
    ungated_terms: list[torch.Tensor] = []
    gated_terms: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    boundary_count = 0
    negative_count = 0
    negative_saturation_terms: list[torch.Tensor] = []
    no_demand_glb_count = flat_logits.new_zeros(())

    for _pose, start, end in _pose_slices(pose_offsets_cpu, flat_logits.numel()):
        local_labels = labels[start:end]
        positive = local_labels > 0.5
        negative = ~positive
        local_weights = normalized_weights[start:end]
        if bool(positive.any()):
            risk = (
                local_weights[positive]
                * (1.0 - scores[start:end][positive])
                * (1.0 + float(rare_positive_weight) * (1.0 - torch.sqrt(hit_rates[start:end][positive])))
            )
            boundary, count = _upper_tail_cvar(risk, boundary_tail_fraction)
            boundary_terms.append(boundary)
            boundary_count += count

        local_glbs = candidate_glbs[start:end]
        unique_glbs, inverse_glbs = torch.unique(local_glbs, sorted=True, return_inverse=True)
        positive_count = torch.zeros(
            (unique_glbs.numel(),), device=flat_logits.device, dtype=torch.long
        )
        positive_count.scatter_add_(0, inverse_glbs, positive.to(dtype=torch.long))
        no_demand_mask = positive_count == 0
        no_demand_glb_count = no_demand_glb_count + no_demand_mask.sum().to(flat_logits.dtype)
        stacked_group_logits = _group_topk_smooth_max(
            flat_logits[start:end],
            inverse_glbs,
            unique_glbs.numel(),
            top_k=8,
            temperature=group_temperature,
        )
        stacked_costs = (
            torch.log1p(costs[unique_glbs])
            if unique_glbs.numel()
            else flat_logits.new_zeros((0,))
        )

        for threshold in thresholds:
            if bool(negative.any()):
                boundary_logit = flat_logits.new_tensor(math.log(threshold / (1.0 - threshold)))
                normalized_margin = (
                    flat_logits[start:end][negative] - boundary_logit
                ) / float(threshold_temperature)
                if negative_band_shape == "sigmoid":
                    soft_keep = torch.sigmoid(normalized_margin)
                    negative_saturation_terms.append((soft_keep > 0.999).float().mean())
                else:
                    soft_keep = (
                        F.softplus(normalized_margin) * float(threshold_temperature)
                    )
                    negative_saturation_terms.append(torch.zeros((), device=soft_keep.device))
                negative_loss, count = _upper_tail_cvar(soft_keep, negative_tail_fraction)
                negative_count += count
            else:
                negative_loss = zero
            negative_terms.append(negative_loss)

            selected_costs = stacked_costs[no_demand_mask]
            if selected_costs.numel():
                requests = soft_request_probability(
                    stacked_group_logits, threshold, temperature=request_temperature
                )
                resource_loss = (
                    requests[no_demand_mask] * selected_costs
                ).sum() / selected_costs.sum().clamp_min(1e-12)
            else:
                resource_loss = zero
            resource_terms.append(resource_loss)

            if bool(positive.any()):
                gate, margin = safety_reserve_gate(
                    flat_logits[start:end][positive],
                    local_weights[positive],
                    threshold,
                    global_step=global_step,
                    total_optimizer_steps=total_optimizer_steps,
                    warmup_fraction=warmup_fraction,
                    reserve_quantile=reserve_quantile,
                    reserve_margin=reserve_margin,
                    gate_temperature=gate_temperature,
                )
            else:
                gate, margin = zero.detach(), zero.detach()
            ungated = float(negative_band_weight) * negative_loss + float(glb_resource_weight) * resource_loss
            ungated_terms.append(ungated)
            gated_terms.append(gate * ungated)
            gates.append(gate)
            margins.append(margin)

    boundary_loss = torch.stack(boundary_terms).mean() if boundary_terms else zero
    negative_loss = torch.stack(negative_terms).mean() if negative_terms else zero
    resource_loss = torch.stack(resource_terms).mean() if resource_terms else zero
    ungated_efficiency = torch.stack(ungated_terms).mean() if ungated_terms else zero
    efficiency_loss = torch.stack(gated_terms).mean() if gated_terms else zero
    safety_loss = rvl_loss + float(boundary_tail_weight) * boundary_loss
    total = safety_loss + efficiency_loss
    gate_mean = torch.stack(gates).mean() if gates else zero.detach()
    margin_mean = torch.stack(margins).mean() if margins else zero.detach()
    negative_saturation = (
        torch.stack(negative_saturation_terms).mean()
        if negative_saturation_terms
        else zero.detach()
    )
    parts: dict[str, torch.Tensor | float] = {
        **{f"rvl_{key}": value for key, value in rvl_parts.items()},
        "lossSafetyRvl": rvl_loss,
        "lossBoundaryTail": boundary_loss,
        "lossBoundaryTailScaled": float(boundary_tail_weight) * boundary_loss,
        "lossSafety": safety_loss,
        "lossNegativeBand": negative_loss,
        "lossNoDemandGlbResource": resource_loss,
        "lossEfficiencyUngated": ungated_efficiency,
        "lossEfficiency": efficiency_loss,
        "lossSafetyReserveOperatingUtility": total,
        "safetyReserveGate": gate_mean,
        "safetyReserveMargin": margin_mean,
        "operatingThresholds": flat_logits.new_tensor(thresholds).detach(),
        "boundaryTailPositiveCount": float(boundary_count),
        "thresholdNegativeTailCount": float(negative_count),
        "negativeBandSaturationFraction": negative_saturation,
        "noDemandGlbCount": no_demand_glb_count,
        "normalizedVisibleWeightMean": normalized_weights.mean(),
        "normalizedVisibleWeightMax": normalized_weights.max(),
    }
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("safety-reserve operating utility loss is non-finite")
    for name, value in parts.items():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite loss part: {name}")
    return total, parts


def _gradient_device(groups: Sequence[GradientList]) -> torch.device:
    devices = {value.device for group in groups for value in group if value is not None}
    if len(devices) > 1:
        raise ValueError("all gradient tensors must be on the same device")
    return next(iter(devices), torch.device("cpu"))


def _gradient_norm_sq(gradients: GradientList, device: torch.device) -> torch.Tensor:
    result = torch.zeros((), device=device, dtype=torch.float64)
    for value in gradients:
        if value is not None:
            result = result + value.double().square().sum()
    return result


def _gradient_dot(
    left: GradientList,
    right: GradientList,
    device: torch.device,
) -> torch.Tensor:
    result = torch.zeros((), device=device, dtype=torch.float64)
    for left_value, right_value in zip(left, right, strict=True):
        if left_value is not None and right_value is not None:
            result = result + (left_value.double() * right_value.double()).sum()
    return result


def _project_one_gradient_group(
    safety_gradients: GradientList,
    auxiliary_gradients: GradientList,
    *,
    norm_cap_ratio: float,
    device: torch.device,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    if float(norm_cap_ratio) < 0.0:
        raise ValueError("gradient norm cap ratios must be non-negative")
    dot = _gradient_dot(safety_gradients, auxiliary_gradients, device)
    safety_norm_sq = _gradient_norm_sq(safety_gradients, device)
    auxiliary_norm_sq = _gradient_norm_sq(auxiliary_gradients, device)
    apply_projection = bool(dot.detach() < 0.0 and safety_norm_sq.detach() > 0.0)
    coefficient = dot / safety_norm_sq.clamp_min(1e-12) if apply_projection else dot.new_zeros(())
    projected: list[torch.Tensor | None] = []
    for safety, auxiliary in zip(safety_gradients, auxiliary_gradients, strict=True):
        if auxiliary is None:
            projected.append(None)
        elif apply_projection and safety is not None:
            projected.append(auxiliary - coefficient.to(auxiliary.dtype) * safety)
        else:
            projected.append(auxiliary)
    for _attempt in range(3):
        dot_after_projection = _gradient_dot(safety_gradients, projected, device)
        if bool(dot_after_projection.detach() >= 0.0 or safety_norm_sq.detach() <= 0.0):
            break
        # FP32 gradient tensors can leave a negative residual after the
        # analytically exact projection.  Correct toward a small positive
        # relative margin, then verify the represented tensors again.
        projected_norm_sq_before_correction = _gradient_norm_sq(projected, device)
        margin = (
            torch.sqrt(safety_norm_sq)
            * torch.sqrt(projected_norm_sq_before_correction)
            * 1e-5
        )
        correction = (dot_after_projection - margin) / safety_norm_sq.clamp_min(1e-24)
        projected = [
            value
            if value is None or safety is None
            else value - correction.to(value.dtype) * safety
            for safety, value in zip(safety_gradients, projected, strict=True)
        ]
    projected_norm_sq = _gradient_norm_sq(projected, device)
    safety_norm = torch.sqrt(safety_norm_sq)
    projected_norm = torch.sqrt(projected_norm_sq)
    cap = float(norm_cap_ratio) * safety_norm
    scale = torch.minimum(
        projected_norm.new_ones(()), cap / projected_norm.clamp_min(1e-12)
    )
    projected = [None if value is None else value * scale.to(value.dtype) for value in projected]
    dot_after = _gradient_dot(safety_gradients, projected, device)
    fallback_zeroed = False
    if bool(dot_after.detach() < 0.0):
        # The projection is computed in float64 but the optimizer receives
        # float32 tensors.  With large, nearly cancelling gradients, casting
        # the correction back to float32 can leave a small negative residual.
        # Dropping this auxiliary group for the current step is the only
        # representation-safe fallback: it gives an exactly zero safety dot
        # product instead of silently applying a safety-conflicting update.
        projected = [None if value is None else torch.zeros_like(value) for value in projected]
        dot_after = _gradient_dot(safety_gradients, projected, device)
        fallback_zeroed = True
    if bool(dot_after.detach() < 0.0):
        raise FloatingPointError(
            f"auxiliary gradient remains safety-conflicting after zero fallback: {float(dot_after.detach().cpu())}"
        )
    return projected, {
        "dotBeforeProjection": float(dot.detach().cpu()),
        "dotAfterProjection": float(dot_after.detach().cpu()),
        "normBeforeProjection": float(torch.sqrt(auxiliary_norm_sq).detach().cpu()),
        "normAfterProjection": float(torch.sqrt(projected_norm_sq).detach().cpu()),
        "normAfterCap": float((projected_norm * scale).detach().cpu()),
        "projectionApplied": float(apply_projection),
        "projectionFallbackZeroed": float(fallback_zeroed),
        "capScale": float(scale.detach().cpu()),
        "normCapRatio": float(norm_cap_ratio),
    }


def project_operating_utility_gradient_groups(
    safety_gradients: GradientList,
    relation_gradients: GradientList,
    schedule_gradients: GradientList,
    efficiency_gradients: GradientList,
    *,
    relation_norm_cap: float = 1.0,
    schedule_norm_cap: float = 1.0,
    efficiency_norm_cap: float = 0.25,
) -> tuple[dict[str, list[torch.Tensor | None]], dict[str, float]]:
    """Project relation, scheduling, and efficiency gradients against safety.

    Each auxiliary group is protected independently. Consequently every
    projected group, and their sum, has a non-negative dot product with the
    safety gradient. Norm caps are expressed as fractions of the safety norm.
    """
    groups = (safety_gradients, relation_gradients, schedule_gradients, efficiency_gradients)
    lengths = {len(group) for group in groups}
    if len(lengths) != 1:
        raise ValueError("all gradient groups must have equal length")
    for name, gradients in zip(
        ("safety", "relation", "schedule", "efficiency"),
        groups,
        strict=True,
    ):
        invalid = [
            index
            for index, value in enumerate(gradients)
            if value is not None and not bool(torch.isfinite(value).all())
        ]
        if invalid:
            raise FloatingPointError(
                f"{name} gradient group contains non-finite tensors at parameter indices {invalid[:16]}"
            )
    device = _gradient_device(groups)
    safety_norm = float(torch.sqrt(_gradient_norm_sq(safety_gradients, device)).detach().cpu())
    projected: dict[str, list[torch.Tensor | None]] = {"safety": list(safety_gradients)}
    statistics: dict[str, float] = {"safetyGradientNorm": safety_norm}
    for name, gradients, cap in (
        ("relation", relation_gradients, relation_norm_cap),
        ("schedule", schedule_gradients, schedule_norm_cap),
        ("efficiency", efficiency_gradients, efficiency_norm_cap),
    ):
        values, group_stats = _project_one_gradient_group(
            safety_gradients, gradients, norm_cap_ratio=cap, device=device
        )
        projected[name] = values
        statistics.update({f"{name}Gradient{key[0].upper()}{key[1:]}": value for key, value in group_stats.items()})
    if not all(math.isfinite(value) for value in statistics.values()):
        raise FloatingPointError("gradient projection statistics are non-finite")
    return projected, statistics


__all__ = [
    "OPERATING_THRESHOLD_ANCHORS",
    "OPERATING_THRESHOLD_RANGE",
    "bounded_topk_smooth_max",
    "project_operating_utility_gradient_groups",
    "safety_reserve_gate",
    "safety_reserve_operating_utility_loss",
    "sample_train_operating_thresholds",
    "soft_request_probability",
]
