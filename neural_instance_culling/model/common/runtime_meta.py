from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def scene_min_max(scene_bounds: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scene min/max/size from either `{min,max}` or `{center,size}` bounds."""
    if "min" in scene_bounds and "max" in scene_bounds:
        mn = np.asarray(scene_bounds["min"], dtype=np.float32)
        mx = np.asarray(scene_bounds["max"], dtype=np.float32)
        return mn, mx, mx - mn
    center = np.asarray(scene_bounds["center"], dtype=np.float32)
    size = np.asarray(scene_bounds["size"], dtype=np.float32)
    return center - size * 0.5, center + size * 0.5, size


def bounds_min_max(bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return min/max corners for one component AABB."""
    if "min" in bounds and "max" in bounds:
        return np.asarray(bounds["min"], dtype=np.float32), np.asarray(bounds["max"], dtype=np.float32)
    center = np.asarray(bounds["center"], dtype=np.float32)
    size = np.asarray(bounds["size"], dtype=np.float32)
    return center - size * 0.5, center + size * 0.5


def load_runtime_meta(runtime_meta: str | Path, num_instances: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load `runtimeVisibilityMeta.json` into dense instance AABB and GLB mapping arrays."""
    path = Path(runtime_meta)
    meta = json.loads(path.read_text(encoding="utf-8"))
    records = meta["componentRecords"]
    inferred = max(int(record["componentGlobalId"]) for record in records) + 1
    count = int(num_instances or inferred)
    world_aabbs = np.zeros((count, 6), dtype=np.float32)
    instance_to_glb = np.zeros((count,), dtype=np.int64)
    for record in records:
        idx = int(record["componentGlobalId"])
        if idx < 0 or idx >= count:
            continue
        mn, mx = bounds_min_max(record["bounds"])
        world_aabbs[idx, :3] = mn
        world_aabbs[idx, 3:] = mx
        instance_to_glb[idx] = int(record["globalGlbId"])
    return world_aabbs, instance_to_glb, meta

