"""Ray-space query and horizontal view-cell disk linearization.

The candidate camera and the query center have different meanings.  Candidate
identity is produced once from the backed-up camera.  The functions in this
module only describe the registered view-cell center and its horizontal disk;
they never regenerate or alter a candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch.nn import functional as F


VIEW_DIM = 9
VIEWCELL_RAY_SPACE_SCHEMA = "viewcell-ray-space-horizontal-disk-v3"
FEATURE_DOMAIN_ABS_MAX = 1.0


def _float_tensor(value: Any, name: str, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if device is not None:
        tensor = tensor.to(device=device)
    tensor = tensor.float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _rows(value: Any, width: int, name: str, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = _float_tensor(value, name, device=device)
    if tensor.ndim == 1 and tensor.shape[0] == width:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape [{width}] or [B, {width}]")
    return tensor


def _broadcast_rows(value: Any, batch: int, width: int, name: str, device: torch.device) -> torch.Tensor:
    tensor = _rows(value, width, name, device=device)
    if tensor.shape[0] == 1 and batch != 1:
        tensor = tensor.expand(batch, -1)
    if tensor.shape[0] != batch:
        raise ValueError(f"{name} batch size does not match")
    return tensor


def _broadcast_scalar(value: Any, batch: int, name: str, device: torch.device) -> torch.Tensor:
    tensor = _float_tensor(value, name, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1).expand(batch, 1)
    elif tensor.ndim == 1 and tensor.numel() in (1, batch):
        tensor = tensor.reshape(-1, 1)
        if tensor.shape[0] == 1:
            tensor = tensor.expand(batch, -1)
    elif tensor.ndim == 2 and tuple(tensor.shape) == (batch, 1):
        pass
    else:
        raise ValueError(f"{name} must be scalar, [B], or [B, 1]")
    return tensor


def _camera_frame(camera_view: torch.Tensor) -> tuple[torch.Tensor, ...]:
    forward = F.normalize(camera_view[:, :3], dim=-1, eps=1e-6)
    up_seed = torch.zeros_like(forward)
    up_seed[:, 1] = 1.0
    alternative = torch.zeros_like(forward)
    alternative[:, 2] = 1.0
    up_seed = torch.where(
        torch.abs((forward * up_seed).sum(dim=-1, keepdim=True)) > 0.98,
        alternative,
        up_seed,
    )
    right = F.normalize(torch.cross(forward, up_seed, dim=-1), dim=-1, eps=1e-6)
    up = F.normalize(torch.cross(right, forward, dim=-1), dim=-1, eps=1e-6)
    tan_x = camera_view[:, 3:4].clamp_min(1e-4)
    tan_y = camera_view[:, 4:5].clamp_min(1e-4)
    return forward, right, up, tan_x, tan_y


@dataclass(frozen=True)
class ViewCellRaySpaceQuery:
    center_view: torch.Tensor
    disk_axes: torch.Tensor
    distance_m: torch.Tensor
    instance_radius_m: torch.Tensor


def _features_and_differentials(
    query_center_world: torch.Tensor,
    camera_view: torch.Tensor,
    instance_aabbs: torch.Tensor,
    derivative_axis: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    centers = (instance_aabbs[:, :3] + instance_aabbs[:, 3:]) * 0.5
    sizes = torch.clamp(instance_aabbs[:, 3:] - instance_aabbs[:, :3], min=1e-4)
    delta = centers - query_center_world
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-4)
    ray = delta / distance
    radius = torch.linalg.norm(sizes, dim=-1, keepdim=True) * 0.5
    forward, right, up, tan_x, tan_y = _camera_frame(camera_view)

    dot_forward = (ray * forward).sum(dim=-1, keepdim=True)
    dot_right = (ray * right).sum(dim=-1, keepdim=True)
    dot_up = (ray * up).sum(dim=-1, keepdim=True)
    denominator_x_raw = dot_forward.abs() * tan_x
    denominator_y_raw = dot_forward.abs() * tan_y
    denominator_x = denominator_x_raw.clamp_min(1e-4)
    denominator_y = denominator_y_raw.clamp_min(1e-4)
    screen_u_raw = dot_right / denominator_x
    screen_v_raw = dot_up / denominator_y
    screen_u = screen_u_raw.clamp(-4.0, 4.0) / 4.0
    screen_v = screen_v_raw.clamp(-4.0, 4.0) / 4.0
    safe_distance = distance.clamp_min(1.0)
    angular = radius / safe_distance
    horizontal_raw = angular / tan_x
    vertical_raw = angular / tan_y
    horizontal = horizontal_raw.clamp(0.0, 4.0) / 2.0 - 1.0
    vertical = vertical_raw.clamp(0.0, 4.0) / 2.0 - 1.0
    log_distance = (torch.log1p(distance / 100.0) / 4.0).clamp(0.0, 2.0) - 1.0
    center_view = torch.cat(
        [
            ray,
            log_distance,
            dot_forward.clamp(-1.0, 1.0),
            screen_u,
            screen_v,
            horizontal,
            vertical,
        ],
        dim=-1,
    )
    if derivative_axis is None:
        return center_view, None, distance, radius

    # The derivative is with respect to moving the query camera along one
    # world-space axis.  delta = center - camera, so both signs matter.
    axis = derivative_axis
    ray_axis = (ray * axis).sum(dim=-1, keepdim=True)
    d_distance = -ray_axis
    d_ray = -(axis - ray * ray_axis) / distance
    d_log = d_distance / (4.0 * (100.0 + distance))
    d_forward = (d_ray * forward).sum(dim=-1, keepdim=True)
    d_right = (d_ray * right).sum(dim=-1, keepdim=True)
    d_up = (d_ray * up).sum(dim=-1, keepdim=True)

    sign_forward = torch.sign(dot_forward)
    d_denominator_x = sign_forward * d_forward * tan_x
    d_denominator_y = sign_forward * d_forward * tan_y
    d_denominator_x = torch.where(denominator_x_raw > 1e-4, d_denominator_x, torch.zeros_like(d_denominator_x))
    d_denominator_y = torch.where(denominator_y_raw > 1e-4, d_denominator_y, torch.zeros_like(d_denominator_y))
    d_u_raw = (d_right * denominator_x - dot_right * d_denominator_x) / denominator_x.square()
    d_v_raw = (d_up * denominator_y - dot_up * d_denominator_y) / denominator_y.square()
    d_u = torch.where(screen_u_raw.abs() < 4.0, d_u_raw / 4.0, torch.zeros_like(d_u_raw))
    d_v = torch.where(screen_v_raw.abs() < 4.0, d_v_raw / 4.0, torch.zeros_like(d_v_raw))

    d_angular = torch.where(
        distance > 1.0,
        -radius * d_distance / distance.square(),
        torch.zeros_like(distance),
    )
    d_horizontal_raw = d_angular / tan_x
    d_vertical_raw = d_angular / tan_y
    d_horizontal = torch.where(
        (horizontal_raw > 0.0) & (horizontal_raw < 4.0),
        d_horizontal_raw / 2.0,
        torch.zeros_like(d_horizontal_raw),
    )
    d_vertical = torch.where(
        (vertical_raw > 0.0) & (vertical_raw < 4.0),
        d_vertical_raw / 2.0,
        torch.zeros_like(d_vertical_raw),
    )
    d_feature = torch.cat(
        [d_ray, d_log, d_forward, d_u, d_v, d_horizontal, d_vertical], dim=-1
    )
    return center_view, d_feature, distance, radius


def build_horizontal_disk_ray_query(
    query_center_world: Any,
    camera_view: Any,
    instance_aabbs: Any,
    viewcell_radius_m: Any,
) -> ViewCellRaySpaceQuery:
    """Return one center query and two low-rank disk axes per instance."""

    centers = _rows(query_center_world, 3, "query_center_world")
    batch = centers.shape[0]
    view = _broadcast_rows(camera_view, batch, 5, "camera_view", centers.device)
    aabbs = _broadcast_rows(instance_aabbs, batch, 6, "instance_aabbs", centers.device)
    radius = _broadcast_scalar(viewcell_radius_m, batch, "viewcell_radius_m", centers.device)
    if bool((radius < 0.0).any()):
        raise ValueError("viewcell_radius_m must be non-negative")

    axis_x = torch.zeros((batch, 3), dtype=torch.float32, device=centers.device)
    axis_z = torch.zeros_like(axis_x)
    axis_x[:, 0] = 1.0
    axis_z[:, 2] = 1.0
    center_view, dx, distance, instance_radius = _features_and_differentials(
        centers, view, aabbs, axis_x
    )
    _, dz, _, _ = _features_and_differentials(centers, view, aabbs, axis_z)
    assert dx is not None and dz is not None
    raw_disk_axes = torch.stack([dx * radius, dz * radius], dim=-1)
    # Every center feature is registered in [-1, 1].  A first-order disk can
    # otherwise explode near a projection singularity even though the actual
    # clipped feature remains bounded.  Project each two-axis row into the
    # largest symmetric disk that stays inside the registered feature domain.
    row_budget = (FEATURE_DOMAIN_ABS_MAX - center_view.abs()).clamp_min(0.0)
    row_norm = torch.linalg.vector_norm(raw_disk_axes, dim=-1)
    row_scale = torch.where(
        row_norm > row_budget,
        row_budget / row_norm.clamp_min(1e-12),
        torch.ones_like(row_norm),
    )
    disk_axes = raw_disk_axes * row_scale.unsqueeze(-1)
    if tuple(center_view.shape) != (batch, VIEW_DIM) or tuple(disk_axes.shape) != (batch, VIEW_DIM, 2):
        raise RuntimeError("internal view-cell ray-space shape error")
    if not bool(torch.isfinite(center_view).all() and torch.isfinite(disk_axes).all()):
        raise FloatingPointError("view-cell ray-space query is non-finite")
    return ViewCellRaySpaceQuery(center_view, disk_axes, distance, instance_radius)


def normalized_relative_log_depth(
    distance_m: Any,
    instance_radius_m: Any,
    q01: float,
    q99: float,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Map radius-relative log distance to a train-frozen monotone coordinate."""

    distance = _float_tensor(distance_m, "distance_m")
    radius = _float_tensor(instance_radius_m, "instance_radius_m", device=distance.device)
    distance, radius = torch.broadcast_tensors(distance, radius)
    if bool((distance < 0.0).any() or (radius < 0.0).any()):
        raise ValueError("distance and radius must be non-negative")
    if not math.isfinite(float(q01)) or not math.isfinite(float(q99)) or float(q99) <= float(q01):
        raise ValueError("q01/q99 must be finite and strictly increasing")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    raw = torch.log1p(distance / (radius + float(epsilon)))
    return ((raw - float(q01)) / (float(q99) - float(q01))).clamp(0.0, 1.0)


def export_ray_space_schema() -> dict[str, Any]:
    return {
        "schema": VIEWCELL_RAY_SPACE_SCHEMA,
        "centerDim": VIEW_DIM,
        "viewcellShape": "horizontal_disk",
        "diskAxisShape": [VIEW_DIM, 2],
        "diskAxisBound": "per-feature row norm <= 1 - abs(center feature)",
        "featureDomain": [-FEATURE_DOMAIN_ABS_MAX, FEATURE_DOMAIN_ABS_MAX],
        "candidateCameraRole": "candidate identity only",
        "queryCenterRole": "center of same-direction view-cell visibility union",
        "depth": "train-frozen radius-relative log1p q01/q99 normalization with exported epsilon",
        "onlineSubposeExpansion": False,
    }
