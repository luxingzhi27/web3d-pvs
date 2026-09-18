"""Fixed nine-point support manifests for disk and oriented-box regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .schemas import (
    REGION_SUPPORT_SCHEMA,
    SUPPORT_POINT_COUNT,
    SchemaError,
    validate_region_manifest,
)


_DISK_ANGLES = np.arange(8, dtype=np.float64) * (np.pi / 4.0)
_BOX_SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RegionSupport:
    region_type: str
    center: np.ndarray
    support_points: np.ndarray
    geometry: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        manifest = {
            "schema": REGION_SUPPORT_SCHEMA,
            "version": 1,
            "regionType": self.region_type,
            "supportCount": SUPPORT_POINT_COUNT,
            "center": self.center.astype(np.float64).tolist(),
            "supportPoints": self.support_points.astype(np.float64).tolist(),
            "geometry": self.geometry,
        }
        validate_region_manifest(manifest)
        return manifest


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape [3]")
    return result


def _validate_disk_basis(right: np.ndarray, forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right_norm = np.linalg.norm(right)
    forward_norm = np.linalg.norm(forward)
    if right_norm <= 1e-12 or forward_norm <= 1e-12:
        raise ValueError("disk basis vectors must be non-zero")
    right = right / right_norm
    forward = forward / forward_norm
    if abs(float(right @ forward)) > 1e-6:
        raise ValueError("disk right and forward vectors must be orthogonal")
    return right, forward


def build_disk_support_points(
    center: Any,
    right: Any,
    forward: Any,
    radius: float,
) -> np.ndarray:
    center_value = _vector(center, "disk center")
    right_value, forward_value = _validate_disk_basis(
        _vector(right, "disk right"), _vector(forward, "disk forward")
    )
    if not isinstance(radius, (int, float, np.integer, np.floating)) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("disk radius must be a positive finite number")
    circle = (
        np.cos(_DISK_ANGLES)[:, None] * right_value[None, :]
        + np.sin(_DISK_ANGLES)[:, None] * forward_value[None, :]
    )
    return np.concatenate(
        [center_value[None, :], center_value[None, :] + float(radius) * circle], axis=0
    ).astype(np.float32)


def build_oriented_box_support_points(center: Any, half_axes: Any) -> np.ndarray:
    center_value = _vector(center, "oriented-box center")
    axes = np.asarray(half_axes, dtype=np.float64)
    if axes.shape != (3, 3) or not np.isfinite(axes).all():
        raise ValueError("oriented-box half_axes must have shape [3,3]")
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("oriented-box half_axes must be non-zero")
    gram = axes @ axes.T
    if not np.allclose(gram, np.diag(np.diag(gram)), atol=1e-6):
        raise ValueError("oriented-box half_axes must be mutually orthogonal")
    corners = center_value[None, :] + _BOX_SIGNS @ axes
    return np.concatenate([center_value[None, :], corners], axis=0).astype(np.float32)


def build_support_points(
    center: Any,
    right: Any,
    forward: Any,
    radius: float,
    *,
    count: int = SUPPORT_POINT_COUNT,
) -> np.ndarray:
    """Compatibility-shaped disk constructor with a fixed count of nine."""

    if count != SUPPORT_POINT_COUNT:
        raise ValueError("V5 support point count is fixed at 9")
    return build_disk_support_points(center, right, forward, radius)


def build_region_support_manifest(
    *,
    region_type: str,
    center: Any,
    radius: float | None = None,
    right: Any | None = None,
    forward: Any | None = None,
    half_axes: Any | None = None,
) -> dict[str, Any]:
    center_value = _vector(center, "region center")
    if region_type == "disk":
        if radius is None or right is None or forward is None or half_axes is not None:
            raise ValueError("disk manifest requires radius, right, and forward only")
        right_value, forward_value = _validate_disk_basis(
            _vector(right, "disk right"), _vector(forward, "disk forward")
        )
        support_points = build_disk_support_points(center_value, right_value, forward_value, radius)
        geometry = {
            "radius": float(radius),
            "right": right_value.tolist(),
            "forward": forward_value.tolist(),
        }
    elif region_type == "oriented_box":
        if half_axes is None or radius is not None or right is not None or forward is not None:
            raise ValueError("oriented_box manifest requires half_axes only")
        axes = np.asarray(half_axes, dtype=np.float64)
        support_points = build_oriented_box_support_points(center_value, axes)
        geometry = {"halfAxes": axes.tolist()}
    else:
        raise ValueError("region_type must be disk or oriented_box")
    return RegionSupport(region_type, center_value, support_points, geometry).to_manifest()


def build_region_support_points(manifest: Mapping[str, Any]) -> np.ndarray:
    """Rebuild and validate the nine points recorded in a region manifest."""

    value = validate_region_manifest(manifest)
    center = np.asarray(value["center"], dtype=np.float64)
    geometry = value["geometry"]
    if value["regionType"] == "disk":
        points = build_disk_support_points(
            center,
            geometry["right"],
            geometry["forward"],
            geometry["radius"],
        )
    else:
        points = build_oriented_box_support_points(center, geometry["halfAxes"])
    stored = np.asarray(value["supportPoints"], dtype=np.float32)
    if not np.allclose(points, stored, rtol=0.0, atol=1e-6):
        raise SchemaError("region manifest supportPoints do not match its geometry")
    return points


def query_region_stats(survivals: Any) -> np.ndarray:
    """Compress nine field queries as center, max, mean, and min."""

    values = np.asarray(survivals, dtype=np.float32)
    if values.shape[-1] != SUPPORT_POINT_COUNT:
        raise ValueError("survivals must have a final dimension of nine")
    if not np.isfinite(values).all():
        raise ValueError("survivals contain non-finite values")
    center = values[..., 0]
    return np.stack(
        [center, values.max(axis=-1), values.mean(axis=-1), values.min(axis=-1)], axis=-1
    )


def validate_region_support_manifest(manifest: Mapping[str, Any]) -> None:
    build_region_support_points(manifest)
