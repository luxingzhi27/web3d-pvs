"""Shared cache format and conservative queries for the triangle HZB baseline.

The cache is intentionally separate from the AABB depth proxy.  Its level-zero
depth comes from a browser rasterization of the scene triangles; higher levels
are built by min-pooling that level-zero buffer.  This module only handles the
binary cache and candidate queries, so it can be tested without Chrome.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


CACHE_SCHEMA = "neuralstreamweb3d-triangle-hzb-cache-v1"
DEPTH_ENCODING = "linear_view_depth_normalized_by_camera_far"


def _normalize_direction(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return value / norm


def camera_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = _normalize_direction(forward)
    up_reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(f, up_reference))) > 0.98:
        up_reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(f, up_reference)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(right, f)
    up /= max(float(np.linalg.norm(up)), 1e-8)
    return f, right.astype(np.float32), up.astype(np.float32)


def build_hzb_levels(depth: np.ndarray, width: int, height: int) -> list[np.ndarray]:
    """Build min-depth mip levels from a bottom-left-oriented depth image."""
    current = np.asarray(depth, dtype=np.float32).reshape(int(height), int(width))
    if not np.all(np.isfinite(current)):
        raise ValueError("triangle HZB level zero contains non-finite depth")
    current = np.clip(current, 0.0, 1.0).copy()
    levels = [current]
    current_height, current_width = current.shape
    while current_width > 1 or current_height > 1:
        next_width = max(1, (current_width + 1) // 2)
        next_height = max(1, (current_height + 1) // 2)
        pooled = np.ones((next_height, next_width), dtype=np.float32)
        for y in range(next_height):
            y0 = 2 * y
            y1 = min(y0 + 2, current_height)
            for x in range(next_width):
                x0 = 2 * x
                x1 = min(x0 + 2, current_width)
                pooled[y, x] = float(np.min(current[y0:y1, x0:x1]))
        levels.append(pooled)
        current = pooled
        current_height, current_width = current.shape
    return levels


def flatten_hzb_levels(levels: list[np.ndarray]) -> tuple[np.ndarray, list[dict[str, int]]]:
    chunks: list[np.ndarray] = []
    descriptors: list[dict[str, int]] = []
    offset = 0
    for level, values in enumerate(levels):
        array = np.ascontiguousarray(values, dtype=np.float32)
        height, width = array.shape
        chunks.append(array.reshape(-1))
        descriptors.append({"level": level, "width": int(width), "height": int(height), "offset": offset, "count": int(array.size)})
        offset += int(array.size)
    return np.concatenate(chunks).astype("<f4", copy=False), descriptors


def unflatten_hzb_levels(values: np.ndarray, descriptors: list[dict[str, Any]]) -> list[np.ndarray]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    result: list[np.ndarray] = []
    for descriptor in descriptors:
        offset = int(descriptor["offset"])
        count = int(descriptor["count"])
        width = int(descriptor["width"])
        height = int(descriptor["height"])
        if offset < 0 or count < 0 or offset + count > array.size or count != width * height:
            raise ValueError("invalid triangle HZB level descriptor")
        result.append(array[offset:offset + count].reshape(height, width))
    return result


def project_aabb_to_camera(
    aabbs: np.ndarray,
    camera_world: np.ndarray,
    camera_forward: np.ndarray,
    tan_x: float,
    tan_y: float,
    camera_far: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project AABBs using the same perspective query used by the browser.

    Returns NDC rectangles, nearest normalized linear depth, validity, and
    positive view-space depth of the farthest corner.  AABB projection is only
    the query primitive; the occluder depth in the cache is triangle-derived.
    """
    boxes = np.asarray(aabbs, dtype=np.float32).reshape(-1, 6)
    count = boxes.shape[0]
    if count == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    camera = np.asarray(camera_world, dtype=np.float32).reshape(3)
    forward, right, up = camera_basis(camera_forward)
    mins = boxes[:, :3]
    maxs = boxes[:, 3:]
    corners = np.stack(
        [
            np.stack([mins[:, 0], mins[:, 1], mins[:, 2]], axis=-1),
            np.stack([mins[:, 0], mins[:, 1], maxs[:, 2]], axis=-1),
            np.stack([mins[:, 0], maxs[:, 1], mins[:, 2]], axis=-1),
            np.stack([mins[:, 0], maxs[:, 1], maxs[:, 2]], axis=-1),
            np.stack([maxs[:, 0], mins[:, 1], mins[:, 2]], axis=-1),
            np.stack([maxs[:, 0], mins[:, 1], maxs[:, 2]], axis=-1),
            np.stack([maxs[:, 0], maxs[:, 1], mins[:, 2]], axis=-1),
            np.stack([maxs[:, 0], maxs[:, 1], maxs[:, 2]], axis=-1),
        ],
        axis=1,
    )
    relative = corners - camera[None, None, :]
    view_x = relative @ right
    view_y = relative @ up
    view_z = relative @ forward
    front = view_z > 0.0
    safe_z = np.maximum(view_z, 1e-5)
    ndc_x = view_x / (safe_z * max(float(tan_x), 1e-5))
    ndc_y = view_y / (safe_z * max(float(tan_y), 1e-5))
    valid = np.any(front, axis=1)
    ndc_x = np.clip(ndc_x, -8.0, 8.0)
    ndc_y = np.clip(ndc_y, -8.0, 8.0)
    rect = np.stack(
        [
            np.clip(np.min(np.where(front, ndc_x, 8.0), axis=1), -1.0, 1.0),
            np.clip(np.min(np.where(front, ndc_y, 8.0), axis=1), -1.0, 1.0),
            np.clip(np.max(np.where(front, ndc_x, -8.0), axis=1), -1.0, 1.0),
            np.clip(np.max(np.where(front, ndc_y, -8.0), axis=1), -1.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    nearest = np.min(np.where(front, view_z, np.inf), axis=1)
    farthest = np.max(np.where(front, view_z, 0.0), axis=1)
    nearest_normalized = np.clip(nearest / max(float(camera_far), 1e-5), 0.0, 1.0).astype(np.float32, copy=False)
    valid &= np.isfinite(nearest)
    return rect, nearest_normalized, farthest.astype(np.float32, copy=False), valid


def query_hzb_levels(
    levels: list[np.ndarray],
    rect: np.ndarray,
    candidate_depth: float,
    depth_bias: float,
) -> float:
    """Return conservative visible coverage in [0, 1] for one AABB.

    The selected mip covers the projected rectangle with at most a 2x2 query
    footprint.  A candidate is considered potentially visible when its nearest
    AABB depth is no farther than the minimum triangle depth in that footprint.
    The returned fraction is useful for threshold calibration but remains a
    geometric HZB approximation, not an exact triangle visibility mask.
    """
    if not levels:
        return 0.0
    width = int(levels[0].shape[1])
    height = int(levels[0].shape[0])
    x0, y0, x1, y1 = [float(value) for value in np.asarray(rect).reshape(4)]
    if x1 <= x0 or y1 <= y0 or not np.isfinite(candidate_depth):
        return 0.0
    pixel_width = max(1.0, (x1 - x0) * 0.5 * width)
    pixel_height = max(1.0, (y1 - y0) * 0.5 * height)
    mip = int(np.clip(np.floor(np.log2(max(pixel_width, pixel_height))), 0, len(levels) - 1))
    level = levels[mip]
    level_height, level_width = level.shape
    ix0 = int(np.clip(np.floor((x0 + 1.0) * 0.5 * level_width), 0, level_width - 1))
    iy0 = int(np.clip(np.floor((y0 + 1.0) * 0.5 * level_height), 0, level_height - 1))
    ix1 = int(np.clip(np.ceil((x1 + 1.0) * 0.5 * level_width) - 1.0, 0, level_width - 1))
    iy1 = int(np.clip(np.ceil((y1 + 1.0) * 0.5 * level_height) - 1.0, 0, level_height - 1))
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    samples = level[iy0:iy1 + 1, ix0:ix1 + 1]
    if samples.size == 0:
        return 0.0
    return float(np.mean(float(candidate_depth) <= samples + float(depth_bias)))


def load_triangle_hzb_cache(cache_path: str | Path) -> tuple[dict[str, Any], np.memmap]:
    """Load and validate a packed triangle HZB cache."""
    path = Path(cache_path)
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"triangle HZB metadata is missing: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"unsupported triangle HZB cache schema: {metadata.get('schema')!r}")
    if metadata.get("depthEncoding") != DEPTH_ENCODING:
        raise ValueError("triangle HZB cache uses an unsupported depth encoding")
    expected_bytes = int(metadata.get("valueCount", -1)) * 4
    if expected_bytes < 0 or path.stat().st_size != expected_bytes:
        raise ValueError(f"triangle HZB binary size mismatch for {path}")
    values = np.memmap(path, dtype="<f4", mode="r")
    return metadata, values
