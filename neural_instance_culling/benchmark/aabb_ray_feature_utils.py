"""Shared cold-start features for the learned AABB-plus-ray baseline.

The feature layout is deliberately limited to data available before target GLB
geometry arrives: an instance AABB, the current camera ray/FOV, and the
instance's conservative AABB projection in the supplied MVP.  The training
script and benchmark runner import this module so their schemas cannot drift.
"""
from __future__ import annotations

import numpy as np

from pose_csr_dataset import _project_aabb_features_numpy


FEATURE_DIM = 18


def _normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return array / norm


def _camera_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = _normalize(forward, np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    up_reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(f, up_reference))) > 0.98:
        up_reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = _normalize(np.cross(f, up_reference), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    up = _normalize(np.cross(right, f), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    return f, right, up


def build_aabb_ray_features(
    world_aabbs: np.ndarray,
    instance_ids: np.ndarray,
    camera_world: np.ndarray,
    camera_view: np.ndarray,
    mvp: np.ndarray,
    scene_diagonal: float,
) -> np.ndarray:
    """Build one deterministic feature row for each supplied candidate.

    Layout: camera-space AABB center (3), camera-space half extent (3), ray
    direction (3), FOV tangents (2), clipped AABB rectangle (4), projected
    area (1), normalized projected depth (1), and projection-valid flag (1).
    """
    ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= int(world_aabbs.shape[0])):
        raise IndexError("AABB-ray feature ids are outside the runtime instance range")
    view = np.asarray(camera_view, dtype=np.float32).reshape(-1)
    if view.size < 5:
        raise ValueError("AABB-ray features require camera_view=[forward_x, forward_y, forward_z, tan_x, tan_y]")
    camera = np.asarray(camera_world, dtype=np.float32).reshape(3)
    diagonal = float(scene_diagonal)
    if not np.isfinite(diagonal) or diagonal <= 1e-6:
        raise ValueError("scene_diagonal must be finite and positive")
    if ids.size == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    forward, right, up = _camera_basis(view[:3])
    tan_x = max(float(view[3]), 1e-4)
    tan_y = max(float(view[4]), 1e-4)
    aabbs = np.asarray(world_aabbs, dtype=np.float32)[ids]
    mins = aabbs[:, :3]
    maxs = aabbs[:, 3:]
    centers = (mins + maxs) * 0.5
    extents = np.maximum((maxs - mins) * 0.5, 0.0)
    relative = centers - camera[None, :]
    center_camera = np.stack(
        [relative @ right, relative @ up, relative @ forward], axis=1
    ) / diagonal
    extent_camera = np.stack(
        [extents @ np.abs(right), extents @ np.abs(up), extents @ np.abs(forward)], axis=1
    ) / diagonal
    _rect, area, depth, valid = _project_aabb_features_numpy(aabbs, np.asarray(mvp, dtype=np.float32))
    depth_normalized = np.clip(depth / diagonal, 0.0, 32.0)
    rows = np.concatenate(
        [
            center_camera,
            extent_camera,
            np.repeat(forward[None, :], ids.size, axis=0),
            np.full((ids.size, 2), [tan_x, tan_y], dtype=np.float32),
            _rect,
            np.asarray(area, dtype=np.float32)[:, None],
            depth_normalized[:, None],
            np.asarray(valid, dtype=np.float32)[:, None],
        ],
        axis=1,
    )
    if rows.shape != (ids.size, FEATURE_DIM) or not np.isfinite(rows).all():
        raise ValueError("AABB-ray feature construction produced an invalid tensor")
    return rows.astype(np.float32, copy=False)
