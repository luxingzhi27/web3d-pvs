"""GCOF-PVS V5 visibility loss, field loss, and scene dual state.

The training loop processes one source scene per optimizer step.  The public
interfaces in this module therefore keep one scene's global ``G_s`` and
``W_s`` explicit instead of silently averaging minibatch denominators.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

try:  # Support the path-based imports used by the repository tests.
    from .survival import directional_parameters
except ImportError:  # pragma: no cover
    from survival import directional_parameters  # type: ignore[no-redef]


COUNT_BUDGET = 0.02
VISUAL_BUDGET = 0.01
FIELD_NLL_WEIGHT = 0.25
DISTANCE_GRID_COUNT = 13
EXTERNAL_HIT_SCHEMA = "parallel_external_hit_current_status-v1"
DUAL_STATE_SCHEMA = "gcof-pvs-v5-dual-state-v1"
_LN2 = math.log(2.0)


def _float_tensor(
    value: Any,
    name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device) if device is not None else value
    else:
        result = torch.as_tensor(value, device=device)
    if not result.is_floating_point():
        result = result.float()
    if dtype is not None:
        result = result.to(dtype=dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    return value.sum() * 0.0


def _scalar_total(value: Any, name: str, *, positive: bool) -> float:
    tensor = _float_tensor(value, name, dtype=torch.float64).reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    result = float(tensor.item())
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class ConstrainedPoseRisks:
    """One scene's unbiased estimates of the three Full-task terms."""

    extra: torch.Tensor
    miss_count: torch.Tensor
    miss_visual: torch.Tensor

    @property
    def J_extra(self) -> torch.Tensor:
        return self.extra

    @property
    def R_count(self) -> torch.Tensor:
        return self.miss_count

    @property
    def R_visual(self) -> torch.Tensor:
        return self.miss_visual

    def __getitem__(self, key: str) -> torch.Tensor:
        values = {
            "extra": self.extra,
            "J_extra": self.extra,
            "miss_count": self.miss_count,
            "R_count": self.miss_count,
            "miss_visual": self.miss_visual,
            "R_visual": self.miss_visual,
        }
        if key not in values:
            raise KeyError(key)
        return values[key]


def _candidate_scale(pose_scale: Any, logits: torch.Tensor) -> torch.Tensor:
    scale = _float_tensor(
        pose_scale,
        "pose_scale",
        device=logits.device,
        dtype=logits.dtype,
    )
    if scale.ndim == 0:
        scale = scale.expand_as(logits)
    elif scale.shape == logits.shape:
        pass
    elif logits.ndim == 2 and scale.numel() == logits.shape[0]:
        scale = scale.reshape(logits.shape[0], 1).expand_as(logits)
    elif scale.numel() == logits.numel():
        scale = scale.reshape_as(logits)
    else:
        raise ValueError("pose_scale must be scalar, pose-aligned, or candidate-aligned")
    if bool((scale < 0.0).any()):
        raise ValueError("pose_scale must be finite and non-negative")
    return scale.reshape(-1)


def constrained_pose_risks(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    pose_scale: torch.Tensor | float,
    G_s: torch.Tensor | float,
    W_s: torch.Tensor | float,
) -> ConstrainedPoseRisks:
    """Compute unbiased scene-global ``J_extra``, ``R_count``, ``R_visual``.

    ``pose_scale`` is the inverse inclusion probability of a sampled pose,
    already broadcast conceptually to every candidate from that pose.  It may
    be scalar, candidate-aligned, or one value per row when ``logits`` is
    shaped ``[num_poses, candidates_per_pose]``.  ``G_s`` and ``W_s`` are the
    complete source-scene GT occurrence and visible-weight totals.
    """
    values = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if not values.is_floating_point():
        values = values.float()
    labels = _float_tensor(targets, "targets", device=values.device, dtype=values.dtype)
    visible = _float_tensor(weights, "weights", device=values.device, dtype=values.dtype)
    if labels.numel() != values.numel() or visible.numel() != values.numel():
        raise ValueError("logits, targets, and weights must have equal item counts")
    labels = labels.reshape_as(values)
    visible = visible.reshape_as(values)
    if bool(((labels != 0.0) & (labels != 1.0)).any()):
        raise ValueError("targets must contain only zero and one")
    if bool((visible < 0.0).any()):
        raise ValueError("weights must be finite and non-negative")
    occurrence_total = _scalar_total(G_s, "G_s", positive=True)
    visible_total = _scalar_total(W_s, "W_s", positive=False)
    scale = _candidate_scale(pose_scale, values)
    flat_logits = values.reshape(-1)
    flat_labels = labels.reshape(-1)
    flat_visible = visible.reshape(-1)
    flat_scale = scale.reshape(-1)
    h_keep = F.softplus(flat_logits) / _LN2
    h_miss = F.softplus(-flat_logits) / _LN2
    negative = flat_labels == 0.0
    positive = flat_labels == 1.0
    extra = (h_keep[negative] * flat_scale[negative]).sum() / occurrence_total
    miss_count = (h_miss[positive] * flat_scale[positive]).sum() / occurrence_total
    if visible_total > 0.0:
        miss_visual = (
            h_miss[positive]
            * flat_visible[positive]
            * flat_scale[positive]
        ).sum() / visible_total
    else:
        miss_visual = _zero_like(values)
    risks = ConstrainedPoseRisks(extra, miss_count, miss_visual)
    if not bool(torch.isfinite(torch.stack([extra, miss_count, miss_visual])).all()):
        raise FloatingPointError("constrained V5 risks are non-finite")
    return risks


class SceneDualState:
    """Two projected non-negative multipliers for every source scene."""

    def __init__(
        self,
        scene_ids: Sequence[Hashable],
        dual_lr: float,
        count_budget: float = COUNT_BUDGET,
        visual_budget: float = VISUAL_BUDGET,
        device: torch.device | str | None = None,
    ) -> None:
        if not math.isfinite(float(dual_lr)) or float(dual_lr) <= 0.0:
            raise ValueError("dual_lr must be finite and positive")
        if not math.isfinite(float(count_budget)) or float(count_budget) < 0.0:
            raise ValueError("count_budget must be finite and non-negative")
        if not math.isfinite(float(visual_budget)) or float(visual_budget) < 0.0:
            raise ValueError("visual_budget must be finite and non-negative")
        self.dual_lr = float(dual_lr)
        self.count_budget = float(count_budget)
        self.visual_budget = float(visual_budget)
        self.device = torch.device("cpu" if device is None else device)
        self.scene_ids = tuple(scene_ids)
        if len(set(self.scene_ids)) != len(self.scene_ids):
            raise ValueError("scene_ids must be unique")
        self._scene_index = {scene_id: index for index, scene_id in enumerate(self.scene_ids)}
        self.lambda_count = torch.zeros(
            len(self.scene_ids), dtype=torch.float64, device=self.device
        )
        self.lambda_visual = torch.zeros_like(self.lambda_count)

    def _index(self, scene_id: Hashable) -> int:
        try:
            return self._scene_index[scene_id]
        except KeyError as exc:
            raise KeyError(f"unknown V5 source scene: {scene_id!r}") from exc

    @staticmethod
    def _risk(risks: Any, *names: str) -> torch.Tensor:
        for name in names:
            if hasattr(risks, name):
                value = getattr(risks, name)
                return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if isinstance(risks, Mapping) and name in risks:
                value = risks[name]
                return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        raise ValueError(f"risks must provide one of {names}")

    def loss(
        self,
        scene_id: Hashable,
        risks: ConstrainedPoseRisks,
        field_nll: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return the exact Full loss for one scene and one optimizer step."""
        index = self._index(scene_id)
        extra = self._risk(risks, "extra", "J_extra")
        miss_count = self._risk(risks, "miss_count", "R_count")
        miss_visual = self._risk(risks, "miss_visual", "R_visual")
        if not isinstance(extra, torch.Tensor):
            extra = torch.as_tensor(extra, device=self.device)
        field = (
            field_nll.to(device=extra.device, dtype=extra.dtype)
            if isinstance(field_nll, torch.Tensor)
            else extra.new_tensor(float(field_nll))
        )
        if field.numel() == 0 or not bool(torch.isfinite(field).all()):
            raise ValueError("field_nll must be finite and non-empty")
        count_multiplier = self.lambda_count[index].detach().to(extra)
        visual_multiplier = self.lambda_visual[index].detach().to(extra)
        total = (
            extra
            + count_multiplier * miss_count
            + visual_multiplier * miss_visual
            + FIELD_NLL_WEIGHT * field.mean()
        )
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError("V5 Full loss is non-finite")
        return total

    def update(
        self,
        scene_id: Hashable,
        risks: ConstrainedPoseRisks,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply projected ascent using detached ``R_count``/``R_visual``."""
        index = self._index(scene_id)
        miss_count = self._risk(risks, "miss_count", "R_count")
        miss_visual = self._risk(risks, "miss_visual", "R_visual")
        count_value = float(miss_count.detach().item())
        visual_value = float(miss_visual.detach().item())
        if not math.isfinite(count_value) or not math.isfinite(visual_value):
            raise ValueError("dual risks must be finite")
        if count_value < 0.0 or visual_value < 0.0:
            raise ValueError("dual risks must be non-negative")
        with torch.no_grad():
            self.lambda_count[index].copy_(torch.clamp(
                self.lambda_count[index]
                + self.dual_lr * (count_value - self.count_budget),
                min=0.0,
            ))
            self.lambda_visual[index].copy_(torch.clamp(
                self.lambda_visual[index]
                + self.dual_lr * (visual_value - self.visual_budget),
                min=0.0,
            ))
        return self.lambda_count[index], self.lambda_visual[index]

    def state_dict(self) -> dict[str, Any]:
        """Return checkpointable state without model gradients or device ties."""
        return {
            "schema": DUAL_STATE_SCHEMA,
            "version": 1,
            "scene_ids": list(self.scene_ids),
            "lambda_count": self.lambda_count.detach().cpu().clone(),
            "lambda_visual": self.lambda_visual.detach().cpu().clone(),
            "dual_lr": self.dual_lr,
            "count_budget": self.count_budget,
            "visual_budget": self.visual_budget,
        }

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != DUAL_STATE_SCHEMA:
            raise ValueError("unsupported V5 dual checkpoint schema")
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported V5 dual checkpoint version")
        scene_ids = tuple(state.get("scene_ids", ()))
        if strict and scene_ids != self.scene_ids:
            raise ValueError("dual checkpoint scenes do not match current scene_ids")
        count = torch.as_tensor(state.get("lambda_count"), dtype=torch.float64).reshape(-1)
        visual = torch.as_tensor(state.get("lambda_visual"), dtype=torch.float64).reshape(-1)
        if count.numel() != len(scene_ids) or visual.numel() != len(scene_ids):
            raise ValueError("dual checkpoint multipliers do not match scene_ids")
        if not bool(torch.isfinite(count).all() and torch.isfinite(visual).all()):
            raise ValueError("dual checkpoint multipliers are non-finite")
        if bool((count < 0.0).any()) or bool((visual < 0.0).any()):
            raise ValueError("dual checkpoint multipliers must be non-negative")
        if float(state.get("count_budget", self.count_budget)) != self.count_budget:
            raise ValueError("dual checkpoint count budget disagrees")
        if float(state.get("visual_budget", self.visual_budget)) != self.visual_budget:
            raise ValueError("dual checkpoint visual budget disagrees")
        dual_lr = float(state.get("dual_lr", self.dual_lr))
        if not math.isfinite(dual_lr) or dual_lr <= 0.0:
            raise ValueError("dual checkpoint dual_lr is invalid")
        if not strict:
            self.scene_ids = scene_ids
            self._scene_index = {scene_id: index for index, scene_id in enumerate(scene_ids)}
            self.lambda_count = torch.zeros(len(scene_ids), dtype=torch.float64, device=self.device)
            self.lambda_visual = torch.zeros_like(self.lambda_count)
        if len(scene_ids) != len(self.scene_ids):
            raise ValueError("dual checkpoint scene count disagrees")
        self.dual_lr = dual_lr
        with torch.no_grad():
            self.lambda_count.copy_(count.to(self.device))
            self.lambda_visual.copy_(visual.to(self.device))


def _pose_ranges(
    pose_rows: Any,
    item_count: int,
    num_poses: int,
) -> list[tuple[int, int]]:
    if isinstance(pose_rows, torch.Tensor):
        rows = pose_rows.detach().cpu()
    else:
        rows = torch.as_tensor(pose_rows)
    if rows.ndim == 1:
        values = rows.to(dtype=torch.long).reshape(-1)
        if values.numel() == num_poses + 1 and int(values[0]) == 0 and int(values[-1]) == item_count:
            if bool((values[1:] < values[:-1]).any()):
                raise ValueError("pose offset rows must be non-decreasing")
            return [(int(values[i]), int(values[i + 1])) for i in range(num_poses)]
        if values.numel() != item_count:
            raise ValueError("pose_rows must be pose IDs, offsets, or [num_poses,2] ranges")
        if bool((values < 0).any()) or bool((values >= num_poses).any()):
            raise ValueError("pose IDs are outside [0, num_poses)")
        return [
            (
                int(torch.nonzero(values == pose, as_tuple=False).min().item())
                if bool((values == pose).any())
                else 0,
                int(torch.nonzero(values == pose, as_tuple=False).max().item()) + 1
                if bool((values == pose).any())
                else 0,
            )
            for pose in range(num_poses)
        ]
    if rows.ndim == 2 and tuple(rows.shape) == (num_poses, 2):
        ranges = rows.to(dtype=torch.long).tolist()
        result = [(int(start), int(end)) for start, end in ranges]
        if any(start < 0 or end < start or end > item_count for start, end in result):
            raise ValueError("pose ranges are outside the flattened batch")
        return result
    raise ValueError("pose_rows must be pose IDs, offsets, or [num_poses,2] ranges")


def pose_balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pose_rows: Any,
    num_poses: int,
) -> torch.Tensor:
    """Average positive/negative-balanced BCE per pose.

    ``pose_rows`` may be one pose ID per flattened candidate, ``num_poses+1``
    offsets, or ``[num_poses, 2]`` half-open ranges.  A pose with no positive
    or no negative candidates uses the class that is present; empty poses are
    skipped.
    """
    if int(num_poses) <= 0:
        raise ValueError("num_poses must be positive")
    values = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if not values.is_floating_point():
        values = values.float()
    flat_logits = values.reshape(-1)
    labels = _float_tensor(targets, "targets", device=values.device, dtype=values.dtype).reshape(-1)
    if labels.numel() != flat_logits.numel():
        raise ValueError("logits and targets must align")
    if bool(((labels != 0.0) & (labels != 1.0)).any()):
        raise ValueError("targets must contain only zero and one")
    pose_losses: list[torch.Tensor] = []
    for start, end in _pose_ranges(pose_rows, flat_logits.numel(), int(num_poses)):
        if start == end:
            continue
        local_logits = flat_logits[start:end]
        local_labels = labels[start:end]
        positive = local_labels == 1.0
        negative = ~positive
        class_losses: list[torch.Tensor] = []
        if bool(positive.any()):
            class_losses.append(F.softplus(-local_logits[positive]).mean())
        if bool(negative.any()):
            class_losses.append(F.softplus(local_logits[negative]).mean())
        pose_losses.append(torch.stack(class_losses).mean())
    total = torch.stack(pose_losses).mean() if pose_losses else _zero_like(values)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("pose-balanced BCE is non-finite")
    return total


def pbce_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pose_rows: Any,
    num_poses: int,
    field_nll: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """PBCE ablation objective with the unchanged ``0.25 L_field`` term."""
    balanced = pose_balanced_bce(logits, targets, pose_rows, num_poses)
    field = (
        field_nll.to(device=balanced.device, dtype=balanced.dtype)
        if isinstance(field_nll, torch.Tensor)
        else balanced.new_tensor(float(field_nll))
    )
    total = balanced + FIELD_NLL_WEIGHT * field.mean()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("PBCE objective is non-finite")
    return total


def _prepare_field_queries(
    coefficients: torch.Tensor,
    directions: torch.Tensor,
    distances: torch.Tensor,
    radii: torch.Tensor,
    events: torch.Tensor | None,
    *,
    require_grid: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    coeff = _float_tensor(coefficients, "coefficients")
    if coeff.ndim == 2:
        if tuple(coeff.shape) != (4, 7):
            raise ValueError("coefficients must have shape [B,4,7] or [4,7]")
        coeff = coeff.unsqueeze(0)
    if coeff.ndim != 3 or tuple(coeff.shape[1:]) != (4, 7):
        raise ValueError("coefficients must have shape [B,4,7] or [4,7]")
    batch = coeff.shape[0]
    device = coeff.device
    work_dtype = torch.float64
    raw_distance = _float_tensor(distances, "distances", device=device, dtype=work_dtype)
    if raw_distance.ndim == 0:
        distance = raw_distance.reshape(1, 1)
    elif raw_distance.ndim == 1:
        if raw_distance.numel() == batch and raw_distance.numel() != DISTANCE_GRID_COUNT:
            distance = raw_distance.reshape(batch, 1)
        else:
            distance = raw_distance.reshape(1, -1)
    elif raw_distance.ndim == 2:
        distance = raw_distance
    else:
        raise ValueError("distances must have shape [B,Q], [B], or [Q]")
    if distance.shape[0] not in (1, batch):
        raise ValueError("distance batch does not match coefficients")
    if distance.shape[0] == 1 and batch != 1:
        distance = distance.expand(batch, -1)
    query_count = distance.shape[1]
    if require_grid and query_count != DISTANCE_GRID_COUNT:
        raise ValueError("current-status external-hit loss requires exactly 13 distance grid points")
    if bool((distance < 0.0).any()):
        raise ValueError("distances must be non-negative")

    raw_radius = _float_tensor(radii, "radii", device=device, dtype=work_dtype)
    if raw_radius.ndim == 0:
        radius = raw_radius.reshape(1, 1)
    elif raw_radius.ndim == 1:
        if raw_radius.numel() == batch and raw_radius.numel() != query_count:
            radius = raw_radius.reshape(batch, 1)
        elif raw_radius.numel() == query_count:
            radius = raw_radius.reshape(1, query_count)
        else:
            raise ValueError("radii must be scalar, [B], or [B,Q]")
    elif raw_radius.ndim == 2:
        radius = raw_radius
    else:
        raise ValueError("radii must be scalar, [B], or [B,Q]")
    if radius.shape[0] not in (1, batch) or radius.shape[1] not in (1, query_count):
        raise ValueError("radii do not match field query shape")
    radius = radius.expand(batch, query_count)
    if bool((radius <= 0.0).any()):
        raise ValueError("radii must be positive")

    raw_direction = _float_tensor(directions, "directions", device=device, dtype=work_dtype)
    if raw_direction.ndim == 1:
        if raw_direction.numel() != 3:
            raise ValueError("directions must have final dimension 3")
        direction = raw_direction.reshape(1, 1, 3)
    elif raw_direction.ndim == 2:
        if raw_direction.shape[1] != 3:
            raise ValueError("directions must have final dimension 3")
        if raw_direction.shape[0] == batch and raw_direction.shape[0] != query_count:
            direction = raw_direction[:, None, :]
        elif raw_direction.shape[0] == query_count:
            direction = raw_direction[None, :, :]
        elif raw_direction.shape[0] == 1:
            direction = raw_direction[None, :, :]
        else:
            raise ValueError("directions do not match field query shape")
    elif raw_direction.ndim == 3 and raw_direction.shape[2] == 3:
        direction = raw_direction
    else:
        raise ValueError("directions must have shape [3], [B,3], or [B,Q,3]")
    if direction.shape[0] not in (1, batch) or direction.shape[1] not in (1, query_count):
        raise ValueError("directions do not match field query shape")
    direction = direction.expand(batch, query_count, 3)
    if not bool(torch.isfinite(direction).all()):
        raise ValueError("directions contain non-finite values")

    coeff = coeff.to(dtype=work_dtype)
    coeff = coeff.expand(batch, -1, -1)
    projected = directional_parameters(coeff, direction)
    if projected.shape != (batch, query_count, 7):
        raise RuntimeError("directional field projection has an unexpected shape")

    event_values: torch.Tensor | None = None
    if events is not None:
        raw_event = _float_tensor(events, "events", device=device, dtype=work_dtype)
        if raw_event.ndim == 0:
            event_values = raw_event.reshape(1, 1).expand(batch, query_count)
        elif raw_event.ndim == 1:
            if raw_event.numel() == query_count:
                event_values = raw_event.reshape(1, query_count).expand(batch, query_count)
            elif raw_event.numel() == batch and query_count == 1:
                event_values = raw_event.reshape(batch, 1)
            else:
                raise ValueError("events do not match the distance grid")
        elif raw_event.ndim == 2 and tuple(raw_event.shape) in {
            (batch, query_count),
            (1, query_count),
            (batch, 1),
        }:
            event_values = raw_event.expand(batch, query_count)
        else:
            raise ValueError("events must align with distances")
        if bool(((event_values != 0.0) & (event_values != 1.0)).any()):
            raise ValueError("events must contain only zero and one")
    return projected, distance.expand(batch, query_count), radius, direction, event_values


def _analytic_log_terms(projected: torch.Tensor, distance: torch.Tensor, radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = projected.to(dtype=torch.float64)
    log_no_hit = -F.softplus(-raw[..., 0])
    log_hit_mass = -F.softplus(raw[..., 0])
    log_mixture = F.log_softmax(raw[..., 1:3], dim=-1)
    location = F.softplus(raw[..., 3:5])
    scale = 0.02 + F.softplus(raw[..., 5:7])
    t = torch.log1p(distance / radius)
    standardized = (t.unsqueeze(-1) - location) / scale
    log_component_survival = F.softplus(-location / scale) - F.softplus(standardized)
    log_survival = torch.logaddexp(
        log_no_hit,
        log_hit_mass
        + torch.logsumexp(log_mixture + log_component_survival, dim=-1),
    )
    log_component_density_t = (
        log_component_survival
        + F.logsigmoid(standardized)
        - torch.log(scale)
    )
    log_density_distance = (
        log_hit_mass
        + torch.logsumexp(log_mixture + log_component_density_t, dim=-1)
        - torch.log(radius + distance)
    )
    log_survival = torch.where(
        distance == 0.0,
        torch.zeros_like(log_survival),
        torch.minimum(log_survival, torch.zeros_like(log_survival)),
    )
    if not bool(torch.isfinite(log_survival).all() and torch.isfinite(log_density_distance).all()):
        raise FloatingPointError("analytic V5 field terms are non-finite")
    return log_survival, log_density_distance


def external_hit_nll(
    coefficients: torch.Tensor,
    directions: torch.Tensor,
    distances: torch.Tensor,
    radii: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Current-status Bernoulli NLL for observations sampled from the 13-point grid.

    For every sampled external-hit distance ``d``, the label is
    ``e = 1[h <= d]``.  Thus the likelihood is
    ``-e*log(1-S(d)) - (1-e)*log(S(d))``.  ``distances`` and ``events`` are
    A batch may contain all 13 distances for each ray or independently sampled
    ray-distance pairs; the dataset sampler controls that choice.
    """
    projected, distance, radius, _direction, event = _prepare_field_queries(
        coefficients,
        directions,
        distances,
        radii,
        events,
        require_grid=False,
    )
    assert event is not None
    log_survival, _log_density = _analytic_log_terms(projected, distance, radius)
    # log(1-exp(log_survival)), evaluated without subtractive cancellation.
    log_survival = torch.minimum(log_survival, torch.zeros_like(log_survival))
    log_one_minus = torch.where(
        log_survival < math.log(0.5),
        torch.log1p(-torch.exp(log_survival)),
        torch.log(torch.clamp(-torch.expm1(log_survival), min=torch.finfo(log_survival.dtype).tiny)),
    )
    log_one_minus = torch.clamp(log_one_minus, min=math.log(torch.finfo(log_survival.dtype).tiny))
    nll = -(event * log_one_minus + (1.0 - event) * log_survival)
    result = nll.mean()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("current-status external-hit NLL is non-finite")
    return result


def external_hit_event_density(
    coefficients: torch.Tensor,
    directions: torch.Tensor,
    distances: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Diagnostic event density ``-dS/dd`` for the analytic field."""
    projected, distance, radius, _direction, _event = _prepare_field_queries(
        coefficients,
        directions,
        distances,
        radii,
        None,
        require_grid=False,
    )
    return _analytic_log_terms(projected, distance, radius)[1].exp()


def external_hit_censor_survival(
    coefficients: torch.Tensor,
    directions: torch.Tensor,
    distances: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Diagnostic right-censor survival ``S(d)`` for the analytic field."""
    projected, distance, radius, _direction, _event = _prepare_field_queries(
        coefficients,
        directions,
        distances,
        radii,
        None,
        require_grid=False,
    )
    return _analytic_log_terms(projected, distance, radius)[0].exp()


def external_hit_event_density_nll(
    coefficients: torch.Tensor,
    directions: torch.Tensor,
    distances: torch.Tensor,
    radii: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Diagnostic density/censor likelihood; not the V5 main field loss."""
    projected, distance, radius, _direction, event = _prepare_field_queries(
        coefficients,
        directions,
        distances,
        radii,
        events,
        require_grid=False,
    )
    assert event is not None
    log_survival, log_density = _analytic_log_terms(projected, distance, radius)
    result = -(event * log_density + (1.0 - event) * log_survival).mean()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("diagnostic external-hit density NLL is non-finite")
    return result


__all__ = [
    "COUNT_BUDGET",
    "DISTANCE_GRID_COUNT",
    "DUAL_STATE_SCHEMA",
    "EXTERNAL_HIT_SCHEMA",
    "FIELD_NLL_WEIGHT",
    "VISUAL_BUDGET",
    "ConstrainedPoseRisks",
    "SceneDualState",
    "constrained_pose_risks",
    "external_hit_censor_survival",
    "external_hit_event_density",
    "external_hit_event_density_nll",
    "external_hit_nll",
    "pbce_objective",
    "pose_balanced_bce",
]
