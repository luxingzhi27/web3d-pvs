"""Actual surface-point fallback for uncovered triangle-layer targets.

This module never treats AABB overlap as an occlusion label.  It reconstructs
sampled surface points from the offline GLB point cache, projects them with the
recorded camera, and compares their geometric depth with the first peeled
triangle layer at the same pixel.  The fallback is intentionally bounded and
reports its cap in metadata because it is an offline completion source, not a
replacement for triangle depth peeling.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 1e-8 else fallback.astype(np.float32)


def camera_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = _normalize(forward, np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    up_reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(f, up_reference))) > 0.98:
        up_reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = _normalize(np.cross(f, up_reference), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    up = _normalize(np.cross(right, f), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    return f, right, up


def reconstruct_instance_points(
    normalized_glb_points: np.ndarray,
    instance_aabb: np.ndarray,
) -> np.ndarray:
    """Map normalized GLB surface points into an audited instance AABB frame."""
    points = np.asarray(normalized_glb_points, dtype=np.float32).reshape(-1, 3)
    aabb = np.asarray(instance_aabb, dtype=np.float32).reshape(6)
    center = (aabb[:3] + aabb[3:]) * 0.5
    size = np.maximum(aabb[3:] - aabb[:3], 1e-6)
    # generate_glb_points_v3 normalizes by the largest source dimension.
    return center[None, :] + points * float(np.max(size))


def project_surface_points(
    points_world: np.ndarray,
    camera_world: np.ndarray,
    camera_forward: np.ndarray,
    tan_x: float,
    tan_y: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    origin = np.asarray(camera_world, dtype=np.float32).reshape(3)
    forward, right, up = camera_basis(camera_forward)
    delta = points - origin[None, :]
    depth = delta @ forward
    front = depth > 0.05
    safe_depth = np.maximum(depth, 0.05)
    x = (delta @ right) / np.maximum(safe_depth * float(tan_x), 1e-6)
    y = (delta @ up) / np.maximum(safe_depth * float(tan_y), 1e-6)
    valid = front & (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
    px = np.floor((x * 0.5 + 0.5) * int(width)).astype(np.int64)
    py = np.floor((0.5 - y * 0.5) * int(height)).astype(np.int64)
    px = np.clip(px, 0, int(width) - 1)
    py = np.clip(py, 0, int(height) - 1)
    pixel = py * int(width) + px
    metric_range = np.linalg.norm(delta, axis=1)
    return pixel, metric_range.astype(np.float32), valid


def select_fallback_targets(
    candidate_ids: np.ndarray,
    observed_ids: np.ndarray,
    world_aabbs: np.ndarray,
    normalized_glb_points: np.ndarray,
    instance_to_glb: np.ndarray,
    camera_world: np.ndarray,
    camera_forward: np.ndarray,
    tan_x: float,
    tan_y: float,
    width: int,
    height: int,
    *,
    coarse_points: int = 16,
    max_targets: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose uncovered targets by actual projected surface footprint."""
    candidates = np.unique(np.asarray(candidate_ids, dtype=np.int64))
    observed = set(int(value) for value in np.asarray(observed_ids, dtype=np.int64).tolist())
    missing = np.asarray([value for value in candidates.tolist() if int(value) not in observed], dtype=np.int64)
    if missing.size == 0:
        return missing, {"candidateCount": int(candidates.size), "uncoveredTargetCount": 0, "selectedTargetCount": 0}
    scores = np.zeros((missing.size,), dtype=np.float64)
    chunk = max(1, min(512, missing.size))
    for start in range(0, missing.size, chunk):
        ids = missing[start : start + chunk]
        sample_count = min(int(coarse_points), normalized_glb_points.shape[1])
        points = normalized_glb_points[instance_to_glb[ids], :sample_count]
        centers = (world_aabbs[ids, :3] + world_aabbs[ids, 3:]) * 0.5
        sizes = np.maximum(world_aabbs[ids, 3:] - world_aabbs[ids, :3], 1e-6)
        points_world = centers[:, None, :] + points * np.max(sizes, axis=1)[:, None, None]
        flat = points_world.reshape(-1, 3)
        _pixel, depth, valid = project_surface_points(
            flat, camera_world, camera_forward, tan_x, tan_y, width, height
        )
        valid = valid.reshape(ids.size, sample_count)
        depth = depth.reshape(ids.size, sample_count)
        scores[start : start + ids.size] = valid.sum(axis=1) + np.where(valid, 1.0 / np.maximum(depth, 1e-3), 0.0).sum(axis=1) * 1e-3
    positive = np.flatnonzero(scores > 0.0)
    if positive.size > int(max_targets):
        order = positive[np.argsort(scores[positive])[::-1][: int(max_targets)]]
    else:
        order = positive[np.argsort(scores[positive])[::-1]]
    selected = missing[order]
    return selected.astype(np.int64, copy=False), {
        "candidateCount": int(candidates.size),
        "uncoveredTargetCount": int(missing.size),
        "projectedTargetCount": int(positive.size),
        "selectedTargetCount": int(selected.size),
        "maxTargets": int(max_targets),
        "coarsePointsPerTarget": int(coarse_points),
    }


def collect_front_surface_relations(
    target_id: int,
    source_ids: np.ndarray,
    normalized_points: np.ndarray,
    instance_aabb: np.ndarray,
    camera_world: np.ndarray,
    camera_forward: np.ndarray,
    tan_x: float,
    tan_y: float,
    first_layer_ids: np.ndarray,
    first_layer_depths: np.ndarray,
    width: int,
    height: int,
    *,
    min_depth_gap: float = 1e-4,
) -> list[tuple[int, int, float, float]]:
    """Return (front, target, metric-range-gap, sample-count) relations."""
    points = reconstruct_instance_points(normalized_points, instance_aabb)
    pixel, depth, valid = project_surface_points(
        points, camera_world, camera_forward, tan_x, tan_y, width, height
    )
    if not bool(valid.any()):
        return []
    ids = np.asarray(first_layer_ids, dtype=np.uint32).reshape(-1)
    peeled_depth = np.asarray(first_layer_depths, dtype=np.float32).reshape(-1)
    if ids.size != int(width) * int(height) or peeled_depth.size != ids.size:
        raise ValueError("first depth layer shape does not match fallback projection dimensions")
    valid_indices = np.flatnonzero(valid)
    front = ids[pixel[valid_indices]].astype(np.int64, copy=False)
    gap = depth[valid_indices] - peeled_depth[pixel[valid_indices]]
    usable = (front != 0xFFFFFFFF) & (front != int(target_id)) & np.isfinite(gap) & (gap > float(min_depth_gap))
    if not bool(usable.any()):
        return []
    front_values = front[usable]
    gap_values = gap[usable]
    unique, counts = np.unique(front_values, return_counts=True)
    result: list[tuple[int, int, float, float]] = []
    for source, count in zip(unique.tolist(), counts.tolist(), strict=False):
        selected_gap = gap_values[front_values == source]
        result.append((int(source), int(target_id), float(np.median(selected_gap)), float(count)))
    return result
