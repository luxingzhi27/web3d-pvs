"""Analytic monotone survival fields and the fixed nine-point region query.

The field coefficients are produced by :mod:`core`.  This module contains no
learned table and no scene-specific state.  A coefficient row has seven
values: one no-hit mass, two mixture logits, two positive locations, and two
positive scales after the parameter transform below.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


FIELD_RANK = 4
SURVIVAL_PARAMETER_DIM = 7
SUPPORT_POINT_COUNT = 9


def _as_field(coefficients: torch.Tensor) -> torch.Tensor:
    field = torch.as_tensor(coefficients).float()
    if field.ndim == 2:
        field = field.unsqueeze(0)
    if field.ndim != 3 or tuple(field.shape[1:]) != (FIELD_RANK, SURVIVAL_PARAMETER_DIM):
        raise ValueError(
            f"coefficients must have shape [B, {FIELD_RANK}, {SURVIVAL_PARAMETER_DIM}]"
        )
    if not bool(torch.isfinite(field).all()):
        raise ValueError("coefficients contain non-finite values")
    return field


def normalize_query_directions(directions: torch.Tensor) -> torch.Tensor:
    """Normalize directions, using +Z for a zero-length point query."""
    values = torch.as_tensor(directions)
    if not values.is_floating_point():
        values = values.float()
    if values.shape[-1] != 3:
        raise ValueError("directions must have a final dimension of 3")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("directions contain non-finite values")
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    fallback = values.new_tensor([0.0, 0.0, 1.0])
    normalized = values / norm.clamp_min(torch.finfo(values.dtype).eps)
    return torch.where(norm > torch.finfo(values.dtype).eps, normalized, fallback)


def first_order_direction_basis(directions: torch.Tensor) -> torch.Tensor:
    """Evaluate the fixed unnormalized order-one basis ``[1, dx, dy, dz]``."""
    normalized = normalize_query_directions(directions)
    return torch.cat([torch.ones_like(normalized[..., :1]), normalized], dim=-1)


def _query_layout(
    field: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fields and directions in a common ``[B, Q, ...]`` layout."""
    coefficients = _as_field(field)
    values = torch.as_tensor(directions, device=coefficients.device).float()
    if values.ndim == 1:
        if values.numel() != 3:
            raise ValueError("directions must have shape [3], [B, 3], or [B, Q, 3]")
        values = values.reshape(1, 1, 3)
    elif values.ndim == 2:
        if values.shape[1] != 3:
            raise ValueError("directions must have a final dimension of 3")
        if values.shape[0] == coefficients.shape[0]:
            values = values.unsqueeze(1)
        else:
            values = values.unsqueeze(0)
    elif values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("directions must have shape [3], [B, 3], or [B, Q, 3]")

    if values.shape[0] == 1 and coefficients.shape[0] != 1:
        values = values.expand(coefficients.shape[0], -1, -1)
    if values.shape[0] != coefficients.shape[0]:
        raise ValueError("field and directions have incompatible batch sizes")
    return coefficients, normalize_query_directions(values)


def directional_parameters(
    coefficients: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Project ``[B, 4, 7]`` fields onto one or more query directions."""
    field, normalized = _query_layout(coefficients, directions)
    basis = torch.cat([torch.ones_like(normalized[..., :1]), normalized], dim=-1)
    return torch.einsum("bqr,brp->bqp", basis, field)


def _query_distances(
    distance: torch.Tensor,
    radius: torch.Tensor,
    batch: int,
    query_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.as_tensor(distance).float()
    r = torch.as_tensor(radius, device=d.device).float()
    if d.ndim == 0:
        d = d.reshape(1, 1)
    elif d.ndim == 1:
        if d.shape[0] == batch:
            d = d.reshape(batch, 1)
        else:
            d = d.reshape(1, -1)
    elif d.ndim != 2:
        raise ValueError("distance must have shape scalar, [B], [Q], or [B, Q]")
    if d.shape[0] == 1 and batch != 1:
        d = d.expand(batch, -1)
    if d.shape != (batch, query_count):
        raise ValueError("distance does not match the field/direction query layout")

    if r.ndim == 0:
        r = r.reshape(1, 1)
    elif r.ndim == 1:
        r = r.reshape(-1, 1)
    elif r.ndim == 2 and r.shape[1] == 1:
        pass
    else:
        raise ValueError("radius must be scalar, [B], or [B, 1]")
    if r.shape[0] == 1 and batch != 1:
        r = r.expand(batch, -1)
    if r.shape != (batch, 1) or bool((r <= 0).any()):
        raise ValueError("radius must be positive and match the field batch")
    if bool((d < 0).any()) or not bool(torch.isfinite(d).all()):
        raise ValueError("distance must be finite and non-negative")
    if not bool(torch.isfinite(r).all()):
        raise ValueError("radius contains non-finite values")
    return d, r


def double_truncated_logistic_survival(
    coefficients: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    radius: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate the two-component, zero-truncated logistic survival field.

    For ``t = log1p(distance / radius)`` the component survival is

    ``exp(softplus(-mu / scale) - softplus((t - mu) / scale))``.

    Each component is normalized to one at ``t=0``.  The no-hit mass and the
    positive mixture therefore make the complete field exactly one at zero,
    while every component is non-increasing for increasing distance.
    """
    field, normalized = _query_layout(coefficients, directions)
    batch, query_count = normalized.shape[:2]
    distances, radii = _query_distances(distance, radius, batch, query_count)
    basis = torch.cat([torch.ones_like(normalized[..., :1]), normalized], dim=-1)
    raw = torch.einsum("bqr,brp->bqp", basis, field)

    no_hit_mass = torch.sigmoid(raw[..., 0])
    mixture = F.softmax(raw[..., 1:3], dim=-1)
    location = F.softplus(raw[..., 3:5])
    scale = 0.02 + F.softplus(raw[..., 5:7])
    normalized_distance = torch.log1p(distances / radii)
    component = torch.exp(
        F.softplus(-location / scale)
        - F.softplus((normalized_distance.unsqueeze(-1) - location) / scale)
    )
    survival = no_hit_mass + (1.0 - no_hit_mass) * (
        mixture * component
    ).sum(dim=-1)
    # Keep the contractual boundary exact even under reduced precision.
    survival = torch.where(
        distances == 0,
        torch.ones_like(survival),
        survival,
    )
    if not bool(torch.isfinite(survival).all()):
        raise FloatingPointError("survival field produced non-finite values")
    return survival


class DoubleTruncatedLogisticSurvival(nn.Module):
    """Stateless module wrapper for the shared analytic survival transform."""

    def forward(
        self,
        coefficients: torch.Tensor,
        directions: torch.Tensor,
        distance: torch.Tensor,
        radius: torch.Tensor | float,
    ) -> torch.Tensor:
        return double_truncated_logistic_survival(
            coefficients, directions, distance, radius
        )


def _unit_vectors(values: torch.Tensor, name: str) -> torch.Tensor:
    result = normalize_query_directions(values)
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape [B, 3]")
    return result


def build_disk_support_points(
    center: torch.Tensor,
    axis_u: torch.Tensor,
    axis_v: torch.Tensor,
    radius: torch.Tensor | float,
) -> torch.Tensor:
    """Return a center plus eight equal-angle circumference points."""
    c = torch.as_tensor(center).float()
    if c.ndim == 1:
        c = c.unsqueeze(0)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("center must have shape [B, 3]")
    u = _unit_vectors(torch.as_tensor(axis_u, device=c.device).float(), "axis_u")
    v = _unit_vectors(torch.as_tensor(axis_v, device=c.device).float(), "axis_v")
    if u.shape != c.shape or v.shape != c.shape:
        raise ValueError("disk axes must match center")
    r = torch.as_tensor(radius, device=c.device).float()
    if r.ndim == 0:
        r = r.expand(c.shape[0])
    if r.ndim == 2 and r.shape[1] == 1:
        r = r[:, 0]
    if r.shape != (c.shape[0],) or bool((r < 0).any()):
        raise ValueError("radius must be non-negative and match center batch")
    angles = torch.arange(8, device=c.device, dtype=c.dtype) * (2.0 * math.pi / 8.0)
    circle = (
        torch.cos(angles).reshape(1, 8, 1) * u[:, None, :]
        + torch.sin(angles).reshape(1, 8, 1) * v[:, None, :]
    ) * r[:, None, None]
    return torch.cat([c[:, None, :], c[:, None, :] + circle], dim=1)


def build_oriented_box_support_points(
    center: torch.Tensor,
    half_axes: torch.Tensor,
    half_axis_v: torch.Tensor | None = None,
    half_axis_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a center plus eight corners from three directed half-axes.

    The canonical form is ``half_axes=[B, 3, 3]`` where the three rows are
    the actual directed half-axis vectors from the box center.  For callers
    that already keep three separate vectors, ``half_axis_v`` and
    ``half_axis_w`` may be supplied instead.  The eight sign combinations are
    all emitted once; no two-dimensional rectangle is substituted for the
    oriented box.
    """
    c = torch.as_tensor(center).float()
    if c.ndim == 1:
        c = c.unsqueeze(0)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("center must have shape [B, 3]")
    first = torch.as_tensor(half_axes, device=c.device).float()
    if half_axis_v is None and half_axis_w is None:
        if first.ndim == 2 and first.shape == (3, 3):
            first = first.unsqueeze(0)
        if first.ndim != 3 or first.shape[1:] != (3, 3):
            raise ValueError("half_axes must have shape [B, 3, 3]")
        axes = first
    elif half_axis_v is not None and half_axis_w is not None:
        second = torch.as_tensor(half_axis_v, device=c.device).float()
        third = torch.as_tensor(half_axis_w, device=c.device).float()
        if first.ndim == 1:
            first = first.unsqueeze(0)
        if second.ndim == 1:
            second = second.unsqueeze(0)
        if third.ndim == 1:
            third = third.unsqueeze(0)
        if first.shape != c.shape or second.shape != c.shape or third.shape != c.shape:
            raise ValueError("three directed half-axes must each have shape [B, 3]")
        axes = torch.stack([first, second, third], dim=1)
    else:
        raise ValueError("provide either half_axes=[B, 3, 3] or all three half-axis vectors")
    if axes.shape[0] == 1 and c.shape[0] != 1:
        axes = axes.expand(c.shape[0], -1, -1)
    if axes.shape[0] != c.shape[0] or not bool(torch.isfinite(axes).all()):
        raise ValueError("half_axes must be finite and match center batch")
    if bool((torch.linalg.vector_norm(axes, dim=-1) <= 0).any()):
        raise ValueError("all directed half-axes must be non-zero")
    determinant = torch.linalg.det(axes)
    if bool((determinant.abs() <= 1.0e-7).any()):
        raise ValueError("three directed half-axes must span a non-degenerate box")
    signs = c.new_tensor(
        [
            [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0], [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0], [1.0, 1.0, 1.0],
        ]
    )
    corners = torch.einsum("cs,bsv->bcv", signs, axes)
    return torch.cat([c[:, None, :], c[:, None, :] + corners], dim=1)


build_box_support_points = build_oriented_box_support_points


def region_survival_statistics(
    coefficients: torch.Tensor,
    target_center: torch.Tensor,
    support_points: torch.Tensor,
    target_radius: torch.Tensor | float,
) -> torch.Tensor:
    """Query the fixed nine supports and return center/max/mean/min survival."""
    field = _as_field(coefficients)
    center = torch.as_tensor(target_center, device=field.device).float()
    points = torch.as_tensor(support_points, device=field.device).float()
    if center.ndim == 1:
        center = center.unsqueeze(0)
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if center.shape != (field.shape[0], 3):
        raise ValueError("target_center must have shape [B, 3]")
    if points.shape != (field.shape[0], SUPPORT_POINT_COUNT, 3):
        raise ValueError(
            f"support_points must have shape [B, {SUPPORT_POINT_COUNT}, 3]"
        )
    if not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(points).all()):
        raise ValueError("region query geometry contains non-finite values")
    relative = points - center[:, None, :]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    survival = double_truncated_logistic_survival(
        field,
        relative,
        distance,
        target_radius,
    )
    stats = torch.stack(
        [survival[:, 0], survival.amax(dim=1), survival.mean(dim=1), survival.amin(dim=1)],
        dim=-1,
    )
    if not bool(torch.isfinite(stats).all()):
        raise FloatingPointError("region survival statistics are non-finite")
    return stats


query_region_statistics = region_survival_statistics
survival_from_field = double_truncated_logistic_survival


__all__ = [
    "DoubleTruncatedLogisticSurvival",
    "FIELD_RANK",
    "SUPPORT_POINT_COUNT",
    "SURVIVAL_PARAMETER_DIM",
    "build_disk_support_points",
    "build_box_support_points",
    "build_oriented_box_support_points",
    "directional_parameters",
    "double_truncated_logistic_survival",
    "first_order_direction_basis",
    "normalize_query_directions",
    "query_region_statistics",
    "region_survival_statistics",
    "survival_from_field",
]
