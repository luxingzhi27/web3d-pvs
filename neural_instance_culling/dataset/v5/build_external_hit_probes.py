"""Columnar external-hit probe contract and current-status sampling."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .permissions import FoldAccessPolicy, authorize_asset_manifest_read
from .directions import icosahedron12_directions
from .schemas import EXTERNAL_HIT_PROBE_SCHEMA, EXTERNAL_HIT_RAY_SCHEMA, validate_probe_manifest


SURFACE_STARTS_PER_UNIT = 16
DIRECTIONS_PER_UNIT = 36
RAYS_PER_UNIT = SURFACE_STARTS_PER_UNIT * DIRECTIONS_PER_UNIT
PROBE_OBSERVATIONS_PER_UNIT = 16
PROBE_DISTANCE_RATIOS = np.asarray(
    [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
    dtype=np.float32,
)


def fibonacci_sphere_directions(count: int = 24) -> np.ndarray:
    if count != 24:
        raise ValueError("V5 Fibonacci direction count is fixed at 24")
    index = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    angle = math.pi * (3.0 - math.sqrt(5.0)) * index
    return np.column_stack([radius * np.cos(angle), y, radius * np.sin(angle)]).astype(np.float32)


def fixed_probe_directions() -> np.ndarray:
    return np.concatenate([icosahedron12_directions(), fibonacci_sphere_directions()], axis=0)


@dataclass(frozen=True)
class ExternalHitProbeTable:
    manifest: Mapping[str, Any]
    unit_ids: np.memmap
    directions: np.memmap
    hit_distances: np.memmap
    max_distances: np.memmap
    start_ids: np.memmap
    direction_ids: np.memmap

    @property
    def row_count(self) -> int:
        return int(self.manifest["rowCount"])


def make_probe_manifest(scene_id: str, num_units: int) -> dict[str, Any]:
    """Build the sole V5 columnar manifest spelling used by synthetic data."""
    row_count = int(num_units) * RAYS_PER_UNIT
    manifest = {
        "schema": EXTERNAL_HIT_PROBE_SCHEMA,
        "version": 2,
        "assetKind": "external_hit_probe",
        "sceneId": str(scene_id),
        "split": "train",
        "sourceRole": "source_train",
        "containsVisibilityLabels": False,
        "permissions": {
            "assetKind": "external_hit_probe",
            "access": "source_train_only",
            "heldOutReadable": False,
            "visibilityLabelsIncluded": False,
        },
        "raySchema": EXTERNAL_HIT_RAY_SCHEMA,
        "surfaceStartsPerUnit": SURFACE_STARTS_PER_UNIT,
        "directionsPerUnit": DIRECTIONS_PER_UNIT,
        "directionSet": "icosahedron12_plus_fibonacci24",
        "anchorDirections": icosahedron12_directions(dtype=np.float64).tolist(),
        "distanceRatios": PROBE_DISTANCE_RATIOS.astype(float).tolist(),
        "maxTraceDistanceRatio": 1024.0,
        "recordFields": [
            "sceneId", "unitId", "probeId", "startPointWorld", "directionWorld",
            "hitTargetCenterDepth", "maxTargetCenterDepth", "event",
        ],
        "recordsFile": "external_hit_probes.columnar",
        "storage": "columnar_memmap_little_endian_v1",
        "rowCount": row_count,
        "rowLayout": "[unit][directionId][startId]",
        "eventEncoding": "finite_hit_distance_is_event",
        "hitDistanceOrigin": "target_center_directional_projection",
        "rayOriginOffset": "direction_world_times_1e-5_times_unit_radius",
        "unitRadius": "component_aabb_half_diagonal",
        "unitOrder": "unitIds_column_order_matches_source_unit_order",
        "numUnits": int(num_units),
        "sceneNumUnits": int(num_units),
        "unitIds": list(range(int(num_units))),
        "columnDtypes": {
            "unitIds": "uint32", "directions": "float32[3]", "hitDistances": "float32",
            "maxDistances": "float32", "startIds": "uint8", "directionIds": "uint8",
        },
        "columnShapes": {
            "unitIds": [row_count], "directions": [row_count, 3], "hitDistances": [row_count],
            "maxDistances": [row_count], "startIds": [row_count], "directionIds": [row_count],
        },
        "byteOrder": "little-endian",
        "files": {
            "unitIds": "probe_unit_ids_uint32.bin",
            "directions": "probe_directions_fp32.bin",
            "hitDistances": "probe_hit_distances_fp32.bin",
            "maxDistances": "probe_max_distances_fp32.bin",
            "startIds": "probe_start_ids_uint8.bin",
            "directionIds": "probe_direction_ids_uint8.bin",
        },
        "materialRule": "formal_color_id_double_side_with_source_alpha_test_provenance",
        "acceleration": "three-mesh-bvh_exact_world_triangles",
    }
    return validate_probe_manifest(manifest)


def load_probe_table(
    output_dir: str | Path,
    *,
    policy: FoldAccessPolicy | Mapping[str, Any],
) -> ExternalHitProbeTable:
    root = Path(output_dir)
    manifest = validate_probe_manifest(
        json.loads((root / "external_hit_probe_manifest.json").read_text(encoding="utf-8"))
    )
    authorize_asset_manifest_read(policy, manifest)
    rows = int(manifest["rowCount"])
    files = manifest["files"]
    return ExternalHitProbeTable(
        manifest=manifest,
        unit_ids=np.memmap(root / files["unitIds"], mode="r", dtype="<u4", shape=(rows,)),
        directions=np.memmap(root / files["directions"], mode="r", dtype="<f4", shape=(rows, 3)),
        hit_distances=np.memmap(root / files["hitDistances"], mode="r", dtype="<f4", shape=(rows,)),
        max_distances=np.memmap(root / files["maxDistances"], mode="r", dtype="<f4", shape=(rows,)),
        start_ids=np.memmap(root / files["startIds"], mode="r", dtype="u1", shape=(rows,)),
        direction_ids=np.memmap(root / files["directionIds"], mode="r", dtype="u1", shape=(rows,)),
    )


def sample_current_status_observations(
    table: ExternalHitProbeTable,
    rng: np.random.Generator,
    count: int = 8192,
) -> dict[str, np.ndarray]:
    """Uniformly sample ray and distance-grid pairs without event balancing."""
    if count <= 0:
        raise ValueError("probe observation count must be positive")
    rows = rng.integers(0, table.row_count, size=count, dtype=np.int64)
    ratio_ids = rng.integers(0, PROBE_DISTANCE_RATIOS.size, size=count, dtype=np.int64)
    max_distance = np.asarray(table.max_distances[rows], dtype=np.float32)
    radius = max_distance / np.float32(1024.0)
    distance = radius * PROBE_DISTANCE_RATIOS[ratio_ids]
    hit = np.asarray(table.hit_distances[rows], dtype=np.float32)
    events = np.isfinite(hit) & (hit <= distance)
    return {
        "unit_ids": np.asarray(table.unit_ids[rows], dtype=np.int64),
        "directions": np.asarray(table.directions[rows], dtype=np.float32),
        "distances": distance.astype(np.float32, copy=False),
        "events": events.astype(np.float32),
    }


def sample_grouped_current_status_observations(
    table: ExternalHitProbeTable,
    rng: np.random.Generator,
    *,
    unit_count: int,
    observations_per_unit: int = PROBE_OBSERVATIONS_PER_UNIT,
) -> dict[str, np.ndarray]:
    """Sample an equal number of unbiased ray-distance pairs per sampled unit.

    Every probe unit owns the same 576-ray block, so uniform unit sampling
    followed by uniform within-block sampling has the same observation
    marginal as uniform full-table row sampling.  Grouping observations keeps
    the field batch at its formal size while bounding the number of geometry
    rows that need to be encoded by one optimizer step.
    """

    if unit_count <= 0 or observations_per_unit <= 0:
        raise ValueError("probe unit and per-unit observation counts must be positive")
    if table.row_count % RAYS_PER_UNIT:
        raise ValueError("probe table rows do not form complete per-unit ray blocks")
    probe_unit_count = table.row_count // RAYS_PER_UNIT
    if probe_unit_count <= 0:
        raise ValueError("probe table contains no units")
    block_ids = rng.choice(
        probe_unit_count,
        size=unit_count,
        replace=probe_unit_count < unit_count,
    ).astype(np.int64, copy=False)
    offsets = rng.integers(
        0,
        RAYS_PER_UNIT,
        size=(unit_count, observations_per_unit),
        dtype=np.int64,
    )
    rows = (block_ids[:, None] * RAYS_PER_UNIT + offsets).reshape(-1)
    ratio_ids = rng.integers(
        0,
        PROBE_DISTANCE_RATIOS.size,
        size=rows.size,
        dtype=np.int64,
    )
    max_distance = np.asarray(table.max_distances[rows], dtype=np.float32)
    radius = max_distance / np.float32(1024.0)
    distance = radius * PROBE_DISTANCE_RATIOS[ratio_ids]
    hit = np.asarray(table.hit_distances[rows], dtype=np.float32)
    events = np.isfinite(hit) & (hit <= distance)
    return {
        "unit_ids": np.asarray(table.unit_ids[rows], dtype=np.int64),
        "directions": np.asarray(table.directions[rows], dtype=np.float32),
        "distances": distance.astype(np.float32, copy=False),
        "events": events.astype(np.float32),
    }


__all__ = [
    "DIRECTIONS_PER_UNIT",
    "ExternalHitProbeTable",
    "PROBE_DISTANCE_RATIOS",
    "PROBE_OBSERVATIONS_PER_UNIT",
    "RAYS_PER_UNIT",
    "SURFACE_STARTS_PER_UNIT",
    "fibonacci_sphere_directions",
    "fixed_probe_directions",
    "load_probe_table",
    "make_probe_manifest",
    "sample_current_status_observations",
    "sample_grouped_current_status_observations",
]
