"""Generate deterministic procedural V5 scenes and training assets.

The generator deliberately keeps the four concerns separate:

* primitive meshes are the renderable units and are also used by the exact
  analytic ray caster;
* local surface samples and proxy relations are geometry-only assets;
* external-hit probes store one nearest hit per unit/start/direction row;
* Pose CSR visibility labels are produced by a small Color-ID style renderer.

The command is intended for reproducible offline generation.  It does not
modify the catalog or any of the existing V5 preprocessing modules.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Keep the documented ``python path/to/script.py`` entry point usable while
# retaining normal package imports for ``python -m`` and unit tests.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neural_instance_culling.dataset.v5.build_local_surface_points import (
    SurfaceUnitInput,
    build_local_surface_asset,
    write_local_surface_asset,
)
from neural_instance_culling.dataset.v5.proxy_relation_graph import (
    build_proxy_relation_graph,
    icosahedron12_directions,
    write_proxy_relation_graph,
)
from neural_instance_culling.dataset.v5.synthetic_scene_manifest import (
    generate_synthetic_scene_manifest,
)
from neural_instance_culling.dataset.v5.schemas import (
    EXTERNAL_HIT_PROBE_SCHEMA,
    EXTERNAL_HIT_RAY_SCHEMA,
    validate_relation_manifest,
    validate_probe_manifest,
)
from neural_instance_culling.model.pose_csr_dataset import (
    DIRECTIONAL_POSE_DTYPE,
    frustum_candidate_ids_for_pose,
)


EXPERIMENT_NAME = "pvs_v5_synthetic_procedural_primitives_v1"
DEFAULT_BASE_SEED = 20260918
DEFAULT_OUTPUT_ROOT = Path(
    "neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/synthetic"
)
DEFAULT_VIEWS_PER_SCENE = 64
DEFAULT_RENDER_WIDTH = 160
DEFAULT_RENDER_HEIGHT = 90
SUBPOSES_PER_VIEWCELL = 4
MODEL_FOV_Y_DEG = 66.0
RENDER_FOV_Y_DEG = 60.0
DEFAULT_NEAR = 0.05
DEFAULT_FAR = 1_000_000.0
SYNTHETIC_CAMERA_ORBIT_FRACTION = 0.20
SURFACE_STARTS_PER_UNIT = 16
PROBE_DIRECTIONS_PER_UNIT = 36
PROBE_RECORD_SCHEMA = EXTERNAL_HIT_PROBE_SCHEMA
PROBE_DISTANCE_RATIOS = [
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    512.0,
    1024.0,
]
DATASET_SCHEMA = "gcof-pvs-v5-synthetic-dataset-v1"
SCENE_OUTPUT_SCHEMA = "gcof-pvs-v5-synthetic-scene-output-v1"
COMPILED_SCHEMA = "gcof-pvs-v5-compiled-scene-v1"

_PRIMITIVE_TYPES = ("box", "cylinder", "beam", "panel")
_FAMILY_CATEGORY_IDS = {
    "rooms_doorways_long_corridors": 0,
    "multi_floor_campus_courtyards_colonnades": 1,
    "city_street_canyons_dense_mixed_height": 2,
    "industrial_pipes_equipment_beams_platforms": 3,
    "repeated_and_unique_cluttered_units": 4,
}
_SPLIT_IDS = {"train": 0, "validation": 1, "calibration": 2, "test": 3}
_POSE_SPLIT_ORDER = ("train", "calibration", "validation", "test")


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _normalize(value: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(array))
    if length <= 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return array / length


def _rotation_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return a row-vector-compatible local-to-world rotation."""

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _box_mesh(dimensions: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(dimensions, dtype=np.float64) * 0.5
    vertices = np.asarray(
        [
            (-half[0], -half[1], -half[2]),
            (half[0], -half[1], -half[2]),
            (half[0], half[1], -half[2]),
            (-half[0], half[1], -half[2]),
            (-half[0], -half[1], half[2]),
            (half[0], -half[1], half[2]),
            (half[0], half[1], half[2]),
            (-half[0], half[1], half[2]),
        ],
        dtype=np.float64,
    )
    triangles = np.asarray(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (0, 4, 5),
            (0, 5, 1),
            (3, 2, 6),
            (3, 6, 7),
            (0, 3, 7),
            (0, 7, 4),
            (1, 5, 6),
            (1, 6, 2),
        ],
        dtype=np.int64,
    )
    return vertices, triangles


def _cylinder_mesh(radius: float, height: float, segments: int = 16) -> tuple[np.ndarray, np.ndarray]:
    angles = np.arange(segments, dtype=np.float64) * (2.0 * math.pi / segments)
    ring = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius])
    vertices = np.vstack(
        [
            np.column_stack([ring[:, 0], np.full(segments, -height * 0.5), ring[:, 1]]),
            np.column_stack([ring[:, 0], np.full(segments, height * 0.5), ring[:, 1]]),
            np.asarray([[0.0, -height * 0.5, 0.0], [0.0, height * 0.5, 0.0]]),
        ]
    )
    bottom_center = 2 * segments
    top_center = bottom_center + 1
    triangles: list[tuple[int, int, int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.extend(
            [
                (index, nxt, segments + nxt),
                (index, segments + nxt, segments + index),
                (bottom_center, nxt, index),
                (top_center, segments + index, segments + nxt),
            ]
        )
    return vertices.astype(np.float64), np.asarray(triangles, dtype=np.int64)


@dataclass(frozen=True)
class PrimitiveUnit:
    """One actual renderable unit with analytic and triangle geometry."""

    unit_id: int
    primitive_type: str
    center: np.ndarray
    rotation: np.ndarray
    dimensions: np.ndarray
    local_vertices: np.ndarray
    local_triangles: np.ndarray
    resource_id: int
    world_aabb: np.ndarray

    def surface_input(self) -> SurfaceUnitInput:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.center
        return SurfaceUnitInput(
            self.unit_id,
            self.local_vertices,
            self.local_triangles,
            transform,
        )

    def mapping(self) -> dict[str, Any]:
        return {
            "unitId": self.unit_id,
            "primitiveType": self.primitive_type,
            "center": self.center.astype(float).tolist(),
            "rotation": self.rotation.astype(float).tolist(),
            "dimensions": self.dimensions.astype(float).tolist(),
            "resourceId": self.resource_id,
            "bounds": {
                "min": self.world_aabb[0].astype(float).tolist(),
                "max": self.world_aabb[1].astype(float).tolist(),
            },
        }


@dataclass
class PrimitiveScene:
    scene_id: str
    units: tuple[PrimitiveUnit, ...]
    bounds: np.ndarray
    _grid: Any | None = None

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def unit_ids(self) -> np.ndarray:
        return np.asarray([unit.unit_id for unit in self.units], dtype=np.int64)

    @property
    def primitive_types(self) -> np.ndarray:
        return np.asarray([unit.primitive_type for unit in self.units], dtype="U8")

    @property
    def centers(self) -> np.ndarray:
        return np.asarray([unit.center for unit in self.units], dtype=np.float64)

    @property
    def rotations(self) -> np.ndarray:
        return np.asarray([unit.rotation for unit in self.units], dtype=np.float64)

    @property
    def dimensions(self) -> np.ndarray:
        return np.asarray([unit.dimensions for unit in self.units], dtype=np.float64)

    @property
    def aabbs(self) -> np.ndarray:
        return np.asarray([unit.world_aabb for unit in self.units], dtype=np.float64)

    def surface_inputs(self) -> list[SurfaceUnitInput]:
        return [unit.surface_input() for unit in self.units]


@dataclass(frozen=True)
class _UniformGrid:
    low: np.ndarray
    cell_size: float
    dimensions: np.ndarray
    members: np.ndarray


def _build_uniform_grid(scene: PrimitiveScene) -> _UniformGrid:
    """Build a conservative AABB cell index for ray broad-phase queries."""

    extent = np.maximum(scene.bounds[1] - scene.bounds[0], 1.0e-6)
    target_axis = max(8, min(32, int(math.ceil(scene.unit_count ** (1.0 / 3.0) * 2.0))))
    cell_size = float(np.max(extent) / target_axis)
    dimensions = np.maximum(1, np.ceil(extent / cell_size).astype(np.int64))
    cell_count = int(np.prod(dimensions))
    cells: list[list[int]] = [[] for _ in range(cell_count)]

    def linear(index: np.ndarray) -> int:
        return int(index[0] + dimensions[0] * (index[1] + dimensions[1] * index[2]))

    for unit_index, aabb in enumerate(scene.aabbs):
        minimum = np.floor((aabb[0] - scene.bounds[0]) / cell_size).astype(np.int64)
        maximum = np.floor((aabb[1] - scene.bounds[0]) / cell_size).astype(np.int64)
        minimum = np.clip(minimum, 0, dimensions - 1)
        maximum = np.clip(maximum, 0, dimensions - 1)
        for z in range(int(minimum[2]), int(maximum[2]) + 1):
            for y in range(int(minimum[1]), int(maximum[1]) + 1):
                for x in range(int(minimum[0]), int(maximum[0]) + 1):
                    cells[linear(np.asarray([x, y, z]))].append(unit_index)
    max_members = max((len(cell) for cell in cells), default=1)
    members = np.full((cell_count, max(1, max_members)), -1, dtype=np.int64)
    for cell_index, cell in enumerate(cells):
        if cell:
            members[cell_index, : len(cell)] = np.asarray(cell, dtype=np.int64)
    return _UniformGrid(scene.bounds[0].astype(np.float64), cell_size, dimensions, members)


def _scene_grid(scene: PrimitiveScene) -> _UniformGrid:
    if scene._grid is None:
        scene._grid = _build_uniform_grid(scene)
    return scene._grid


def _grid_candidate_matrix(
    scene: PrimitiveScene,
    origins: np.ndarray,
    directions: np.ndarray,
    limits: np.ndarray | None,
) -> np.ndarray:
    """Return conservative per-ray unit-index rows from a 3-D DDA."""

    grid = _scene_grid(scene)
    low = grid.low
    high = low + grid.cell_size * grid.dimensions.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (low[None, :] - origins) / directions
        t1 = (high[None, :] - origins) / directions
    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)
    parallel = np.abs(directions) <= 1.0e-12
    parallel_outside = parallel & ((origins < low[None, :] - 1.0e-9) | (origins > high[None, :] + 1.0e-9))
    entry = np.max(np.where(parallel, -np.inf, t_min), axis=1)
    exit_time = np.min(np.where(parallel, np.inf, t_max), axis=1)
    valid = (~parallel_outside.any(axis=1)) & (exit_time >= np.maximum(entry, 0.0))
    entry = np.maximum(entry, 0.0)
    if limits is not None:
        exit_time = np.minimum(exit_time, limits)
        valid &= exit_time >= entry

    max_steps = int(grid.dimensions.sum()) + 3
    max_members = grid.members.shape[1]
    candidates = np.full((origins.shape[0], max_steps * max_members), -1, dtype=np.int64)
    if not bool(valid.any()):
        return candidates
    # Move a tiny amount inside the first cell so a ray entering exactly on a
    # cell boundary uses the cell it actually travels through.
    initial_t = entry + max(1.0e-9, grid.cell_size * 1.0e-9)
    points = origins + directions * initial_t[:, None]
    cells = np.floor((points - low[None, :]) / grid.cell_size).astype(np.int64)
    cells = np.clip(cells, 0, grid.dimensions - 1)
    steps = np.sign(directions).astype(np.int64)
    abs_direction = np.abs(directions)
    delta_t = np.divide(
        grid.cell_size,
        abs_direction,
        out=np.full_like(abs_direction, np.inf),
        where=abs_direction > 1.0e-12,
    )
    boundary_index = np.where(steps > 0, cells + 1, cells)
    boundaries = low[None, :] + boundary_index * grid.cell_size
    next_t = np.divide(
        boundaries - origins,
        directions,
        out=np.full_like(directions, np.inf),
        where=abs_direction > 1.0e-12,
    )
    active = valid.copy()
    for step_index in range(max_steps):
        linear_ids = cells[:, 0] + grid.dimensions[0] * (cells[:, 1] + grid.dimensions[1] * cells[:, 2])
        rows = grid.members[np.clip(linear_ids, 0, grid.members.shape[0] - 1)]
        start = step_index * max_members
        candidates[active, start : start + max_members] = rows[active]
        axis = np.argmin(next_t, axis=1)
        next_boundary = np.min(next_t, axis=1)
        advance = active & (next_boundary <= exit_time + 1.0e-9)
        if not bool(advance.any()):
            break
        rows_index = np.flatnonzero(advance)
        axes = axis[rows_index]
        cells[rows_index, axes] += steps[rows_index, axes]
        next_t[rows_index, axes] += delta_t[rows_index, axes]
        in_bounds = np.all((cells[rows_index] >= 0) & (cells[rows_index] < grid.dimensions[None, :]), axis=1)
        active[:] = False
        active[rows_index[in_bounds]] = True
    return candidates


def _primitive_mesh(primitive_type: str, dimensions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if primitive_type == "cylinder":
        return _cylinder_mesh(float(dimensions[0]) * 0.5, float(dimensions[1]))
    return _box_mesh(dimensions)


def _fibonacci_probe_directions(count: int = 24) -> np.ndarray:
    if count != 24:
        raise ValueError("V5 Fibonacci probe direction count is fixed at 24")
    index = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = math.pi * (3.0 - math.sqrt(5.0)) * index
    return np.column_stack([radius * np.cos(theta), y, radius * np.sin(theta)])


def fixed_probe_directions() -> np.ndarray:
    values = np.concatenate([icosahedron12_directions().astype(np.float64), _fibonacci_probe_directions()], axis=0)
    if values.shape != (PROBE_DIRECTIONS_PER_UNIT, 3):
        raise RuntimeError("fixed V5 probe direction table has the wrong shape")
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values.astype(np.float32)


def _family_position(
    family_index: int,
    unit_index: int,
    unit_count: int,
    *,
    rng: np.random.Generator,
    channel_width: float,
    layers: int,
    scale: float,
) -> tuple[np.ndarray, float, float, float]:
    """Place units on deterministic structural scaffolds, not random AABBs."""

    index = int(unit_index)
    jitter = rng.uniform(-0.18, 0.18, size=3) * scale
    if family_index == 0:
        span = max(4, int(math.ceil(math.sqrt(unit_count / max(1, layers)))))
        segment = index // 4
        lane = index % 4
        x = (segment % span - (span - 1) * 0.5) * channel_width
        z = (segment // span - max(1, layers - 1) * 0.5) * channel_width * 1.4
        y = (lane - 1.5) * max(1.4, channel_width * 0.22)
        return np.asarray([x, y, z]) + jitter, float(index % 3) * 0.2, 0.0, 0.0
    if family_index == 1:
        side = max(4, int(math.ceil(math.sqrt(unit_count / max(1, layers)))))
        gx = index % side
        gz = (index // side) % side
        level = (index // (side * side)) % max(1, layers)
        spacing = max(3.0, channel_width * 0.8)
        position = np.asarray(
            [(gx - (side - 1) * 0.5) * spacing, level * max(3.0, scale * 2.8), (gz - (side - 1) * 0.5) * spacing]
        )
        return position + jitter, float((gx + gz) % 4) * 0.15, 0.0, 0.0
    if family_index == 2:
        rows = max(2, int(math.ceil(math.sqrt(unit_count / 2.0))))
        row = index // 2
        side = -1.0 if index % 2 == 0 else 1.0
        z = (row % rows - (rows - 1) * 0.5) * max(4.0, channel_width)
        y = ((row // rows) % max(1, layers)) * max(5.0, scale * 3.5)
        x = side * max(8.0, channel_width * 1.8)
        return np.asarray([x, y, z]) + jitter, side * 0.08, 0.0, 0.0
    if family_index == 3:
        grid = max(3, int(math.ceil(unit_count ** (1.0 / 3.0))))
        gx = index % grid
        gz = (index // grid) % grid
        level = (index // (grid * grid)) % max(1, layers)
        position = np.asarray(
            [
                (gx - (grid - 1) * 0.5) * max(2.0, channel_width * 0.5),
                level * max(2.5, scale * 1.5),
                (gz - (grid - 1) * 0.5) * max(2.0, channel_width * 0.5),
            ]
        )
        return position + jitter, float(index % 8) * math.pi / 8.0, 0.0, 0.0
    # The clutter family uses a 3-D lattice with a deterministic low-discrepancy
    # offset.  Repeated lattice locations receive repeated primitive shapes,
    # while the jitter and type cycle retain non-repeated units.
    grid = max(4, int(math.ceil(unit_count ** (1.0 / 3.0))))
    gx = index % grid
    gz = (index // grid) % grid
    level = (index // (grid * grid)) % max(1, layers)
    position = np.asarray(
        [
            (gx - (grid - 1) * 0.5) * max(2.0, scale * 1.8),
            level * max(2.0, scale * 1.4),
            (gz - (grid - 1) * 0.5) * max(2.0, scale * 1.8),
        ]
    )
    return position + jitter, float((index * 13) % 16) * math.pi / 16.0, 0.0, 0.0


def build_primitive_scene(scene: Mapping[str, Any], *, allow_small: bool = False) -> PrimitiveScene:
    """Materialize all renderable units for one catalog entry."""

    required = {"sceneId", "generatorSeed", "unitCount", "geometryRecipe", "randomization"}
    missing = sorted(required - set(scene))
    if missing:
        raise ValueError(f"scene is missing geometry fields: {', '.join(missing)}")
    count = int(scene["unitCount"])
    if (not allow_small and not 256 <= count <= 4096) or count <= 0:
        raise ValueError("synthetic scenes must contain 256..4096 units")
    recipe = scene["geometryRecipe"]
    randomization = scene["randomization"]
    family_index = int(recipe["familyIndex"])
    if family_index not in range(5):
        raise ValueError("geometryRecipe.familyIndex must be in [0,4]")
    seed = int(scene["generatorSeed"])
    rng = np.random.default_rng(seed)
    scale_range = randomization.get("scaleRange", [0.8, 2.0])
    base_scale = float(rng.uniform(float(scale_range[0]), float(scale_range[1])))
    channel_width = float(randomization.get("channelWidth", 6.0))
    layers = int(randomization.get("layers", 3))
    size_distribution = float(randomization.get("unitSizeDistribution", 0.5))
    repeat_rate = float(randomization.get("repeatRate", 0.5))
    type_patterns = (
        ("panel", "box", "beam", "box", "cylinder"),
        ("cylinder", "beam", "box", "panel", "box"),
        ("box", "panel", "box", "beam", "cylinder"),
        ("cylinder", "beam", "panel", "box", "cylinder"),
        ("box", "cylinder", "beam", "panel", "box", "cylinder"),
    )
    units: list[PrimitiveUnit] = []
    for unit_id in range(count):
        if unit_id < 4:
            primitive_type = _PRIMITIVE_TYPES[unit_id]
        else:
            primitive_type = type_patterns[family_index][unit_id % len(type_patterns[family_index])]
        center, yaw, pitch, roll = _family_position(
            family_index,
            unit_id,
            count,
            rng=rng,
            channel_width=channel_width,
            layers=layers,
            scale=base_scale,
        )
        size_factor = float(rng.uniform(0.65, 1.45)) ** (0.5 + size_distribution)
        if primitive_type == "cylinder":
            radius = max(0.12, base_scale * 0.22 * size_factor)
            height = max(0.35, base_scale * rng.uniform(1.2, 3.8) * size_factor)
            dimensions = np.asarray([radius * 2.0, height, radius * 2.0])
        elif primitive_type == "beam":
            dimensions = np.asarray(
                [
                    max(0.5, base_scale * rng.uniform(2.0, 7.0) * size_factor),
                    max(0.22, base_scale * rng.uniform(0.18, 0.55)),
                    max(0.22, base_scale * rng.uniform(0.18, 0.55)),
                ]
            )
        elif primitive_type == "panel":
            dimensions = np.asarray(
                [
                    max(0.8, base_scale * rng.uniform(1.5, 5.5) * size_factor),
                    max(0.8, base_scale * rng.uniform(1.2, 4.0) * size_factor),
                    max(0.08, base_scale * rng.uniform(0.08, 0.24)),
                ]
            )
        else:
            dimensions = np.asarray(
                [
                    max(0.5, base_scale * rng.uniform(0.55, 2.4) * size_factor),
                    max(0.5, base_scale * rng.uniform(0.45, 2.4) * size_factor),
                    max(0.5, base_scale * rng.uniform(0.55, 2.4) * size_factor),
                ]
            )
        # Structural units sit on or around their scaffold rather than sharing
        # a fabricated AABB.  Only a small deterministic vertical adjustment is
        # applied; the mesh remains the source of the final bounds.
        if family_index in (0, 1, 3, 4):
            center[1] += float(dimensions[1]) * 0.5
        rotation = _rotation_from_euler(yaw, pitch, roll)
        local_vertices, local_triangles = _primitive_mesh(primitive_type, dimensions)
        world_vertices = local_vertices @ rotation.T + center[None, :]
        world_aabb = np.stack([world_vertices.min(axis=0), world_vertices.max(axis=0)], axis=0)
        # Repetition is represented by a shared resource ID, while every unit
        # retains its own geometry and component ID.
        if repeat_rate > 0.0 and unit_id >= 4:
            repeat_group = max(1, int(round(1.0 + repeat_rate * 7.0)))
            resource_id = int(unit_id % repeat_group)
        else:
            resource_id = unit_id
        units.append(
            PrimitiveUnit(
                unit_id,
                primitive_type,
                center.astype(np.float64),
                rotation.astype(np.float64),
                dimensions.astype(np.float64),
                local_vertices,
                local_triangles,
                resource_id,
                world_aabb.astype(np.float64),
            )
        )
    bounds = np.stack(
        [np.asarray([unit.world_aabb[0] for unit in units]).min(axis=0), np.asarray([unit.world_aabb[1] for unit in units]).max(axis=0)],
        axis=0,
    )
    return PrimitiveScene(str(scene["sceneId"]), tuple(units), bounds.astype(np.float64))


def _intersect_box_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    rotations: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    relative = origins[:, None, :] - centers[None, :, :]
    local_origins = np.einsum("bmi,mij->bmj", relative, rotations)
    local_directions = np.einsum("bi,mij->bmj", directions, rotations)
    half = dimensions[None, :, :] * 0.5
    lower = -half
    upper = half
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lower - local_origins) / local_directions
        t1 = (upper - local_origins) / local_directions
    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)
    parallel = np.abs(local_directions) <= 1.0e-12
    parallel_outside = parallel & ((local_origins < lower - 1.0e-10) | (local_origins > upper + 1.0e-10))
    near = np.max(np.where(parallel, -np.inf, t_min), axis=-1)
    far = np.min(np.where(parallel, np.inf, t_max), axis=-1)
    valid = (~parallel_outside.any(axis=-1)) & (far >= np.maximum(near, 0.0)) & (far > 1.0e-8)
    result = np.where(near > 1.0e-8, near, far)
    return np.where(valid, result, np.inf)


def _intersect_cylinder_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    rotations: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    relative = origins[:, None, :] - centers[None, :, :]
    local_origins = np.einsum("bmi,mij->bmj", relative, rotations)
    local_directions = np.einsum("bi,mij->bmj", directions, rotations)
    radius = dimensions[None, :, 0] * 0.5
    half_height = dimensions[None, :, 1] * 0.5
    ox, oy, oz = local_origins[..., 0], local_origins[..., 1], local_origins[..., 2]
    dx, dy, dz = local_directions[..., 0], local_directions[..., 1], local_directions[..., 2]
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid_quadratic = (a > 1.0e-14) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (-b - root) / (2.0 * a)
        t1 = (-b + root) / (2.0 * a)
    side0 = valid_quadratic & (t0 > 1.0e-8) & (np.abs(oy + t0 * dy) <= half_height + 1.0e-9)
    side1 = valid_quadratic & (t1 > 1.0e-8) & (np.abs(oy + t1 * dy) <= half_height + 1.0e-9)
    side_t = np.where(side0, t0, np.inf)
    side_t = np.minimum(side_t, np.where(side1, t1, np.inf))

    parallel_y = np.abs(dy) <= 1.0e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        bottom_t = (-half_height - oy) / dy
        top_t = (half_height - oy) / dy
    bottom_valid = (~parallel_y) & (bottom_t > 1.0e-8) & (
        (ox + bottom_t * dx) ** 2 + (oz + bottom_t * dz) ** 2 <= radius * radius + 1.0e-9
    )
    top_valid = (~parallel_y) & (top_t > 1.0e-8) & (
        (ox + top_t * dx) ** 2 + (oz + top_t * dz) ** 2 <= radius * radius + 1.0e-9
    )
    cap_t = np.minimum(np.where(bottom_valid, bottom_t, np.inf), np.where(top_valid, top_t, np.inf))
    return np.minimum(side_t, cap_t)


def _intersect_box_torch(
    origins: Any,
    directions: Any,
    centers: Any,
    rotations: Any,
    dimensions: Any,
) -> Any:
    import torch

    relative = origins[:, None, :] - centers[None, :, :]
    local_origins = torch.einsum("bmi,mij->bmj", relative, rotations)
    local_directions = torch.einsum("bi,mij->bmj", directions, rotations)
    half = dimensions[None, :, :] * 0.5
    lower = -half
    upper = half
    t0 = torch.where(torch.abs(local_directions) > 1.0e-12, (lower - local_origins) / local_directions, torch.full_like(local_directions, -torch.inf))
    t1 = torch.where(torch.abs(local_directions) > 1.0e-12, (upper - local_origins) / local_directions, torch.full_like(local_directions, torch.inf))
    t_min = torch.minimum(t0, t1)
    t_max = torch.maximum(t0, t1)
    parallel = torch.abs(local_directions) <= 1.0e-12
    parallel_outside = parallel & ((local_origins < lower - 1.0e-10) | (local_origins > upper + 1.0e-10))
    near = torch.where(parallel, torch.full_like(t_min, -torch.inf), t_min).amax(dim=-1)
    far = torch.where(parallel, torch.full_like(t_max, torch.inf), t_max).amin(dim=-1)
    valid = (~parallel_outside.any(dim=-1)) & (far >= torch.maximum(near, torch.zeros_like(near))) & (far > 1.0e-8)
    result = torch.where(near > 1.0e-8, near, far)
    return torch.where(valid, result, torch.full_like(result, torch.inf))


def _intersect_cylinder_torch(
    origins: Any,
    directions: Any,
    centers: Any,
    rotations: Any,
    dimensions: Any,
) -> Any:
    import torch

    relative = origins[:, None, :] - centers[None, :, :]
    local_origins = torch.einsum("bmi,mij->bmj", relative, rotations)
    local_directions = torch.einsum("bi,mij->bmj", directions, rotations)
    radius = dimensions[None, :, 0] * 0.5
    half_height = dimensions[None, :, 1] * 0.5
    ox, oy, oz = local_origins[..., 0], local_origins[..., 1], local_origins[..., 2]
    dx, dy, dz = local_directions[..., 0], local_directions[..., 1], local_directions[..., 2]
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid_quadratic = (a > 1.0e-14) & (discriminant >= 0.0)
    root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    t0 = torch.where(valid_quadratic, (-b - root) / (2.0 * a), torch.full_like(a, torch.inf))
    t1 = torch.where(valid_quadratic, (-b + root) / (2.0 * a), torch.full_like(a, torch.inf))
    side0 = valid_quadratic & (t0 > 1.0e-8) & (torch.abs(oy + t0 * dy) <= half_height + 1.0e-9)
    side1 = valid_quadratic & (t1 > 1.0e-8) & (torch.abs(oy + t1 * dy) <= half_height + 1.0e-9)
    side_t = torch.minimum(torch.where(side0, t0, torch.full_like(t0, torch.inf)), torch.where(side1, t1, torch.full_like(t1, torch.inf)))
    parallel_y = torch.abs(dy) <= 1.0e-14
    bottom_t = torch.where(parallel_y, torch.full_like(dy, torch.inf), (-half_height - oy) / dy)
    top_t = torch.where(parallel_y, torch.full_like(dy, torch.inf), (half_height - oy) / dy)
    bottom_valid = (~parallel_y) & (bottom_t > 1.0e-8) & ((ox + bottom_t * dx) ** 2 + (oz + bottom_t * dz) ** 2 <= radius * radius + 1.0e-9)
    top_valid = (~parallel_y) & (top_t > 1.0e-8) & ((ox + top_t * dx) ** 2 + (oz + top_t * dz) ** 2 <= radius * radius + 1.0e-9)
    cap_t = torch.minimum(torch.where(bottom_valid, bottom_t, torch.full_like(bottom_t, torch.inf)), torch.where(top_valid, top_t, torch.full_like(top_t, torch.inf)))
    return torch.minimum(side_t, cap_t)


def _intersect_box_candidates_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    candidate_indices: np.ndarray,
    scene: PrimitiveScene,
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.maximum(candidate_indices, 0)
    centers = scene.centers[safe]
    rotations = scene.rotations[safe]
    dimensions = scene.dimensions[safe]
    relative = origins[:, None, :] - centers
    local_origins = np.einsum("bki,bkij->bkj", relative, rotations)
    local_directions = np.einsum("bi,bkij->bkj", directions, rotations)
    half = dimensions * 0.5
    lower = -half
    upper = half
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lower - local_origins) / local_directions
        t1 = (upper - local_origins) / local_directions
    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)
    parallel = np.abs(local_directions) <= 1.0e-12
    parallel_outside = parallel & ((local_origins < lower - 1.0e-10) | (local_origins > upper + 1.0e-10))
    near = np.max(np.where(parallel, -np.inf, t_min), axis=-1)
    far = np.min(np.where(parallel, np.inf, t_max), axis=-1)
    valid = (~parallel_outside.any(axis=-1)) & (far >= np.maximum(near, 0.0)) & (far > 1.0e-8)
    result = np.where(valid, np.where(near > 1.0e-8, near, far), np.inf)
    result[candidate_indices < 0] = np.inf
    return result, scene.unit_ids[safe]


def _intersect_cylinder_candidates_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    candidate_indices: np.ndarray,
    scene: PrimitiveScene,
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.maximum(candidate_indices, 0)
    centers = scene.centers[safe]
    rotations = scene.rotations[safe]
    dimensions = scene.dimensions[safe]
    relative = origins[:, None, :] - centers
    local_origins = np.einsum("bki,bkij->bkj", relative, rotations)
    local_directions = np.einsum("bi,bkij->bkj", directions, rotations)
    radius = dimensions[..., 0] * 0.5
    half_height = dimensions[..., 1] * 0.5
    ox, oy, oz = local_origins[..., 0], local_origins[..., 1], local_origins[..., 2]
    dx, dy, dz = local_directions[..., 0], local_directions[..., 1], local_directions[..., 2]
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid_quadratic = (a > 1.0e-14) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (-b - root) / (2.0 * a)
        t1 = (-b + root) / (2.0 * a)
    side0 = valid_quadratic & (t0 > 1.0e-8) & (np.abs(oy + t0 * dy) <= half_height + 1.0e-9)
    side1 = valid_quadratic & (t1 > 1.0e-8) & (np.abs(oy + t1 * dy) <= half_height + 1.0e-9)
    side_t = np.minimum(np.where(side0, t0, np.inf), np.where(side1, t1, np.inf))
    parallel_y = np.abs(dy) <= 1.0e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        bottom_t = (-half_height - oy) / dy
        top_t = (half_height - oy) / dy
    bottom_valid = (~parallel_y) & (bottom_t > 1.0e-8) & ((ox + bottom_t * dx) ** 2 + (oz + bottom_t * dz) ** 2 <= radius * radius + 1.0e-9)
    top_valid = (~parallel_y) & (top_t > 1.0e-8) & ((ox + top_t * dx) ** 2 + (oz + top_t * dz) ** 2 <= radius * radius + 1.0e-9)
    cap_t = np.minimum(np.where(bottom_valid, bottom_t, np.inf), np.where(top_valid, top_t, np.inf))
    result = np.minimum(side_t, cap_t)
    result[candidate_indices < 0] = np.inf
    return result, scene.unit_ids[safe]


def _nearest_candidate_numpy(
    scene: PrimitiveScene,
    origins: np.ndarray,
    directions: np.ndarray,
    candidate_indices: np.ndarray,
    limit: np.ndarray,
    ignored: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    primitive_types = scene.primitive_types[candidate_indices.clip(min=0)]
    distances = np.full(candidate_indices.shape, np.inf, dtype=np.float64)
    candidate_ids = scene.unit_ids[np.maximum(candidate_indices, 0)]
    box_mask = (primitive_types != "cylinder") & (candidate_indices >= 0)
    cylinder_mask = (primitive_types == "cylinder") & (candidate_indices >= 0)
    if bool(box_mask.any()):
        box_candidates = np.where(box_mask, candidate_indices, -1)
        box_distances, _ = _intersect_box_candidates_numpy(origins, directions, box_candidates, scene)
        distances = np.where(box_mask, box_distances, distances)
    if bool(cylinder_mask.any()):
        cylinder_candidates = np.where(cylinder_mask, candidate_indices, -1)
        cylinder_distances, _ = _intersect_cylinder_candidates_numpy(origins, directions, cylinder_candidates, scene)
        distances = np.where(cylinder_mask, cylinder_distances, distances)
    if ignored is not None:
        distances = np.where(candidate_ids == ignored[:, None], np.inf, distances)
    distances = np.where(distances <= limit[:, None], distances, np.inf)
    finite = np.isfinite(distances)
    best_t = np.min(distances, axis=1)
    with np.errstate(invalid="ignore"):
        tie = finite & (np.abs(distances - best_t[:, None]) <= 1.0e-10)
    best_ids = np.min(np.where(tie, candidate_ids, np.iinfo(np.int64).max), axis=1)
    best_ids[~np.isfinite(best_t)] = -1
    return best_ids, best_t


def _nearest_candidate_torch(
    origins: Any,
    directions: Any,
    candidate_indices: np.ndarray,
    scene: PrimitiveScene,
    limit: Any,
    ignored: Any | None,
    scene_centers: Any,
    scene_rotations: Any,
    scene_dimensions: Any,
) -> tuple[Any, Any]:
    import torch

    safe_np = np.maximum(candidate_indices, 0)
    candidate_ids = torch.as_tensor(scene.unit_ids[safe_np], dtype=torch.int64, device=origins.device)
    cylinder_mask_np = (scene.primitive_types[safe_np] == "cylinder") & (candidate_indices >= 0)
    box_mask_np = (~cylinder_mask_np) & (candidate_indices >= 0)
    candidate_index_tensor = torch.as_tensor(candidate_indices, dtype=torch.int64, device=origins.device)
    distances = torch.full(candidate_index_tensor.shape, torch.inf, dtype=origins.dtype, device=origins.device)
    if bool(box_mask_np.any()):
        box_candidates = torch.where(
            torch.as_tensor(box_mask_np, dtype=torch.bool, device=origins.device),
            candidate_index_tensor,
            torch.full_like(candidate_index_tensor, -1),
        )
        box_distances = _intersect_box_candidates_torch(
            origins, directions, box_candidates, scene_centers, scene_rotations, scene_dimensions
        )
        distances = torch.where(torch.as_tensor(box_mask_np, dtype=torch.bool, device=origins.device), box_distances, distances)
    if bool(cylinder_mask_np.any()):
        cylinder_candidates = torch.where(
            torch.as_tensor(cylinder_mask_np, dtype=torch.bool, device=origins.device),
            candidate_index_tensor,
            torch.full_like(candidate_index_tensor, -1),
        )
        cylinder_distances = _intersect_cylinder_candidates_torch(
            origins, directions, cylinder_candidates, scene_centers, scene_rotations, scene_dimensions
        )
        distances = torch.where(torch.as_tensor(cylinder_mask_np, dtype=torch.bool, device=origins.device), cylinder_distances, distances)
    if ignored is not None:
        distances = torch.where(candidate_ids == ignored[:, None], torch.inf, distances)
    distances = torch.where(distances <= limit[:, None], distances, torch.inf)
    finite = torch.isfinite(distances)
    best_t = distances.amin(dim=1)
    ties = finite & (torch.abs(distances - best_t[:, None]) <= 1.0e-10)
    max_id = torch.full_like(candidate_ids, torch.iinfo(torch.int64).max)
    best_ids = torch.where(ties, candidate_ids, max_id).amin(dim=1)
    best_ids = torch.where(torch.isfinite(best_t), best_ids, torch.full_like(best_ids, -1))
    return best_ids, best_t


def _intersect_box_candidates_torch(
    origins: Any,
    directions: Any,
    candidate_indices: Any,
    scene_centers: Any,
    scene_rotations: Any,
    scene_dimensions: Any,
) -> Any:
    import torch

    safe = torch.clamp_min(candidate_indices, 0)
    centers = scene_centers[safe]
    rotations = scene_rotations[safe]
    dimensions = scene_dimensions[safe]
    relative = origins[:, None, :] - centers
    local_origins = torch.einsum("bki,bkij->bkj", relative, rotations)
    local_directions = torch.einsum("bi,bkij->bkj", directions, rotations)
    half = dimensions * 0.5
    lower, upper = -half, half
    parallel = torch.abs(local_directions) <= 1.0e-12
    t0 = torch.where(parallel, torch.full_like(local_directions, -torch.inf), (lower - local_origins) / local_directions)
    t1 = torch.where(parallel, torch.full_like(local_directions, torch.inf), (upper - local_origins) / local_directions)
    t_min, t_max = torch.minimum(t0, t1), torch.maximum(t0, t1)
    parallel_outside = parallel & ((local_origins < lower - 1.0e-10) | (local_origins > upper + 1.0e-10))
    near = torch.where(parallel, torch.full_like(t_min, -torch.inf), t_min).amax(dim=-1)
    far = torch.where(parallel, torch.full_like(t_max, torch.inf), t_max).amin(dim=-1)
    valid = (~parallel_outside.any(dim=-1)) & (far >= torch.maximum(near, torch.zeros_like(near))) & (far > 1.0e-8)
    result = torch.where(valid, torch.where(near > 1.0e-8, near, far), torch.full_like(far, torch.inf))
    return torch.where(candidate_indices >= 0, result, torch.full_like(result, torch.inf))


def _intersect_cylinder_candidates_torch(
    origins: Any,
    directions: Any,
    candidate_indices: Any,
    scene_centers: Any,
    scene_rotations: Any,
    scene_dimensions: Any,
) -> Any:
    import torch

    safe = torch.clamp_min(candidate_indices, 0)
    centers = scene_centers[safe]
    rotations = scene_rotations[safe]
    dimensions = scene_dimensions[safe]
    relative = origins[:, None, :] - centers
    local_origins = torch.einsum("bki,bkij->bkj", relative, rotations)
    local_directions = torch.einsum("bi,bkij->bkj", directions, rotations)
    radius = dimensions[..., 0] * 0.5
    half_height = dimensions[..., 1] * 0.5
    ox, oy, oz = local_origins[..., 0], local_origins[..., 1], local_origins[..., 2]
    dx, dy, dz = local_directions[..., 0], local_directions[..., 1], local_directions[..., 2]
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid_quadratic = (a > 1.0e-14) & (discriminant >= 0.0)
    root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    t0 = torch.where(valid_quadratic, (-b - root) / (2.0 * a), torch.full_like(a, torch.inf))
    t1 = torch.where(valid_quadratic, (-b + root) / (2.0 * a), torch.full_like(a, torch.inf))
    side0 = valid_quadratic & (t0 > 1.0e-8) & (torch.abs(oy + t0 * dy) <= half_height + 1.0e-9)
    side1 = valid_quadratic & (t1 > 1.0e-8) & (torch.abs(oy + t1 * dy) <= half_height + 1.0e-9)
    side_t = torch.minimum(torch.where(side0, t0, torch.full_like(t0, torch.inf)), torch.where(side1, t1, torch.full_like(t1, torch.inf)))
    parallel_y = torch.abs(dy) <= 1.0e-14
    bottom_t = torch.where(parallel_y, torch.full_like(dy, torch.inf), (-half_height - oy) / dy)
    top_t = torch.where(parallel_y, torch.full_like(dy, torch.inf), (half_height - oy) / dy)
    bottom_valid = (~parallel_y) & (bottom_t > 1.0e-8) & ((ox + bottom_t * dx) ** 2 + (oz + bottom_t * dz) ** 2 <= radius * radius + 1.0e-9)
    top_valid = (~parallel_y) & (top_t > 1.0e-8) & ((ox + top_t * dx) ** 2 + (oz + top_t * dz) ** 2 <= radius * radius + 1.0e-9)
    cap_t = torch.minimum(torch.where(bottom_valid, bottom_t, torch.full_like(bottom_t, torch.inf)), torch.where(top_valid, top_t, torch.full_like(top_t, torch.inf)))
    result = torch.minimum(side_t, cap_t)
    return torch.where(candidate_indices >= 0, result, torch.full_like(result, torch.inf))


def _update_nearest(
    best_t: np.ndarray,
    best_ids: np.ndarray,
    distances: np.ndarray,
    unit_ids: np.ndarray,
    max_distance: np.ndarray,
    ignore_ids: np.ndarray | None,
) -> None:
    if ignore_ids is not None:
        distances = np.where(unit_ids[None, :] == ignore_ids[:, None], np.inf, distances)
    distances = np.where(distances <= max_distance[:, None], distances, np.inf)
    candidate = unit_ids[None, :]
    finite = np.isfinite(distances)
    with np.errstate(invalid="ignore"):
        better = finite & (
            (distances < best_t[:, None] - 1.0e-10)
            | ((np.abs(distances - best_t[:, None]) <= 1.0e-10) & (candidate < best_ids[:, None]))
        )
    selected_distance = np.min(np.where(better, distances, np.inf), axis=1)
    selected_index = np.argmin(np.where(better, distances, np.inf), axis=1)
    has_update = np.isfinite(selected_distance)
    rows = np.flatnonzero(has_update)
    if rows.size:
        best_t[rows] = selected_distance[rows]
        best_ids[rows] = unit_ids[selected_index[rows]]


def _raycast_cpu(
    scene: PrimitiveScene,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    max_distance: np.ndarray | None,
    ignore_unit_ids: np.ndarray | None,
    candidate_matrix: np.ndarray | None = None,
    use_grid: bool = False,
    ray_chunk: int = 4096,
    unit_chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    count = origins.shape[0]
    output_ids = np.full(count, -1, dtype=np.int64)
    output_distances = np.full(count, np.inf, dtype=np.float64)
    type_values = scene.primitive_types
    for ray_start in range(0, count, ray_chunk):
        ray_end = min(count, ray_start + ray_chunk)
        local_origins = origins[ray_start:ray_end]
        local_directions = directions[ray_start:ray_end]
        best_t = np.full(ray_end - ray_start, np.inf, dtype=np.float64)
        best_ids = np.full(ray_end - ray_start, np.iinfo(np.int64).max, dtype=np.int64)
        limit = np.full(ray_end - ray_start, np.inf, dtype=np.float64) if max_distance is None else max_distance[ray_start:ray_end]
        ignored = None if ignore_unit_ids is None else ignore_unit_ids[ray_start:ray_end]
        if candidate_matrix is not None or use_grid:
            local_candidates = (
                candidate_matrix[ray_start:ray_end]
                if candidate_matrix is not None
                else _grid_candidate_matrix(scene, local_origins, local_directions, limit)
            )
            selected_ids, selected_distances = _nearest_candidate_numpy(
                scene,
                local_origins,
                local_directions,
                local_candidates,
                limit,
                ignored,
            )
            valid = selected_ids >= 0
            output_ids[ray_start:ray_end][valid] = selected_ids[valid]
            output_distances[ray_start:ray_end] = selected_distances
            continue
        for primitive_type in _PRIMITIVE_TYPES:
            indices = np.flatnonzero(type_values == primitive_type)
            for unit_start in range(0, indices.size, unit_chunk):
                selected = indices[unit_start : unit_start + unit_chunk]
                centers = scene.centers[selected]
                rotations = scene.rotations[selected]
                dimensions = scene.dimensions[selected]
                if primitive_type == "cylinder":
                    distances = _intersect_cylinder_numpy(local_origins, local_directions, centers, rotations, dimensions)
                else:
                    distances = _intersect_box_numpy(local_origins, local_directions, centers, rotations, dimensions)
                _update_nearest(best_t, best_ids, distances, scene.unit_ids[selected], limit, ignored)
        valid = np.isfinite(best_t)
        output_ids[ray_start:ray_end][valid] = best_ids[valid]
        output_distances[ray_start:ray_end] = best_t
    return output_ids, output_distances


def _raycast_cuda(
    scene: PrimitiveScene,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    max_distance: np.ndarray | None,
    ignore_unit_ids: np.ndarray | None,
    candidate_matrix: np.ndarray | None = None,
    use_grid: bool = False,
    ray_chunk: int = 4096,
    unit_chunk: int = 512,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    count = origins.shape[0]
    output_ids = np.full(count, -1, dtype=np.int64)
    output_distances = np.full(count, np.inf, dtype=np.float64)
    type_values = scene.primitive_types
    scene_centers = torch.as_tensor(scene.centers, dtype=torch.float64, device=target)
    scene_rotations = torch.as_tensor(scene.rotations, dtype=torch.float64, device=target)
    scene_dimensions = torch.as_tensor(scene.dimensions, dtype=torch.float64, device=target)
    for ray_start in range(0, count, ray_chunk):
        ray_end = min(count, ray_start + ray_chunk)
        local_origins = torch.as_tensor(origins[ray_start:ray_end], dtype=torch.float64, device=target)
        local_directions = torch.as_tensor(directions[ray_start:ray_end], dtype=torch.float64, device=target)
        best_t = torch.full((ray_end - ray_start,), torch.inf, dtype=torch.float64, device=target)
        best_ids = torch.full((ray_end - ray_start,), np.iinfo(np.int64).max, dtype=torch.int64, device=target)
        limit = torch.full((ray_end - ray_start,), torch.inf, dtype=torch.float64, device=target) if max_distance is None else torch.as_tensor(max_distance[ray_start:ray_end], dtype=torch.float64, device=target)
        ignored = None if ignore_unit_ids is None else torch.as_tensor(ignore_unit_ids[ray_start:ray_end], dtype=torch.int64, device=target)
        if candidate_matrix is not None or use_grid:
            local_candidates = (
                candidate_matrix[ray_start:ray_end]
                if candidate_matrix is not None
                else _grid_candidate_matrix(
                    scene,
                    origins[ray_start:ray_end],
                    directions[ray_start:ray_end],
                    None if max_distance is None else max_distance[ray_start:ray_end],
                )
            )
            selected_ids, selected_distances = _nearest_candidate_torch(
                local_origins,
                local_directions,
                local_candidates,
                scene,
                limit,
                ignored,
                scene_centers,
                scene_rotations,
                scene_dimensions,
            )
            ids_cpu = selected_ids.detach().cpu().numpy()
            distances_cpu = selected_distances.detach().cpu().numpy()
            valid = ids_cpu >= 0
            output_ids[ray_start:ray_end][valid] = ids_cpu[valid]
            output_distances[ray_start:ray_end] = distances_cpu
            continue
        for primitive_type in _PRIMITIVE_TYPES:
            indices_np = np.flatnonzero(type_values == primitive_type)
            for unit_start in range(0, indices_np.size, unit_chunk):
                selected_np = indices_np[unit_start : unit_start + unit_chunk]
                selected = torch.as_tensor(selected_np, dtype=torch.int64, device=target)
                selected_ids = torch.as_tensor(scene.unit_ids[selected_np], dtype=torch.int64, device=target)
                if primitive_type == "cylinder":
                    distances = _intersect_cylinder_torch(
                        local_origins,
                        local_directions,
                        scene_centers[selected],
                        scene_rotations[selected],
                        scene_dimensions[selected],
                    )
                else:
                    distances = _intersect_box_torch(
                        local_origins,
                        local_directions,
                        scene_centers[selected],
                        scene_rotations[selected],
                        scene_dimensions[selected],
                    )
                selected_ids = selected_ids[None, :]
                if ignored is not None:
                    distances = torch.where(selected_ids == ignored[:, None], torch.inf, distances)
                distances = torch.where(distances <= limit[:, None], distances, torch.inf)
                better = (distances < best_t[:, None] - 1.0e-10) | ((torch.abs(distances - best_t[:, None]) <= 1.0e-10) & (selected_ids < best_ids[:, None]))
                candidate_t = torch.where(better, distances, torch.inf)
                selected_distance, selected_slot = candidate_t.min(dim=1)
                update = torch.isfinite(selected_distance)
                best_t = torch.where(update, selected_distance, best_t)
                ids_for_update = selected[selected_slot]
                best_ids = torch.where(update, ids_for_update, best_ids)
        ids_cpu = best_ids.detach().cpu().numpy()
        distances_cpu = best_t.detach().cpu().numpy()
        valid = np.isfinite(distances_cpu)
        output_ids[ray_start:ray_end][valid] = ids_cpu[valid]
        output_distances[ray_start:ray_end] = distances_cpu
    return output_ids, output_distances


def raycast_primitive_hits(
    scene: PrimitiveScene,
    origins: Any,
    directions: Any,
    *,
    device: str = "cpu",
    max_distance: Any | None = None,
    ignore_unit_ids: Any | None = None,
    ray_chunk: int = 4096,
    unit_chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest analytic primitive IDs and distances for a ray batch.

    ``-1``/``inf`` denotes no hit.  ``ignore_unit_ids`` is per-ray and is used
    by external probes to skip the complete source primitive.  CPU and CUDA
    use the same box/cylinder intersection equations and deterministic ID tie
    break.
    """

    ray_origins = np.asarray(origins, dtype=np.float64)
    ray_directions = np.asarray(directions, dtype=np.float64)
    if ray_origins.ndim != 2 or ray_origins.shape[1] != 3 or ray_directions.shape != ray_origins.shape:
        raise ValueError("origins and directions must both have shape [R,3]")
    if ray_origins.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    lengths = np.linalg.norm(ray_directions, axis=1)
    if np.any(~np.isfinite(ray_origins)) or np.any(~np.isfinite(ray_directions)) or np.any(lengths <= 1.0e-12):
        raise ValueError("ray origins/directions must be finite and directions non-zero")
    ray_directions = ray_directions / lengths[:, None]
    limits = None if max_distance is None else np.asarray(max_distance, dtype=np.float64).reshape(-1)
    if limits is not None and (limits.shape != (ray_origins.shape[0],) or np.any(~np.isfinite(limits)) or np.any(limits <= 0.0)):
        raise ValueError("max_distance must be positive finite values with shape [R]")
    ignored = None if ignore_unit_ids is None else np.asarray(ignore_unit_ids, dtype=np.int64).reshape(-1)
    if ignored is not None and ignored.shape != (ray_origins.shape[0],):
        raise ValueError("ignore_unit_ids must have shape [R]")
    resolved = str(device).lower()
    if resolved == "auto":
        try:
            import torch

            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved = "cpu"
    if resolved == "cuda":
        return _raycast_cuda(
            scene,
            ray_origins,
            ray_directions,
            max_distance=limits,
            ignore_unit_ids=ignored,
            candidate_matrix=None,
            use_grid=scene.unit_count > 64,
            ray_chunk=int(ray_chunk),
            unit_chunk=int(unit_chunk),
            device=resolved,
        )
    if resolved != "cpu":
        raise ValueError("device must be cpu, cuda, or auto")
    return _raycast_cpu(
        scene,
        ray_origins,
        ray_directions,
        max_distance=limits,
        ignore_unit_ids=ignored,
        candidate_matrix=None,
        use_grid=scene.unit_count > 64,
        ray_chunk=int(ray_chunk),
        unit_chunk=int(unit_chunk),
    )


def raycast_primitives(
    scene: PrimitiveScene,
    origins: Any,
    directions: Any,
    *,
    device: str = "cpu",
    max_distance: Any | None = None,
    ignore_unit_ids: Any | None = None,
) -> np.ndarray:
    """Convenience wrapper returning only the nearest unit IDs."""

    return raycast_primitive_hits(
        scene,
        origins,
        directions,
        device=device,
        max_distance=max_distance,
        ignore_unit_ids=ignore_unit_ids,
    )[0]


def _camera_basis(forward: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = _normalize(np.asarray(forward, dtype=np.float64), [0.0, 0.0, -1.0])
    up_seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(f @ up_seed)) > 0.98:
        up_seed = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = _normalize(np.cross(f, up_seed), [1.0, 0.0, 0.0])
    up = _normalize(np.cross(right, f), [0.0, 1.0, 0.0])
    return f, right, up


def _camera_rays(
    origin: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_y_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    f, right, up = _camera_basis(forward)
    aspect = float(width) / float(height)
    tan_y = math.tan(math.radians(float(fov_y_deg)) * 0.5)
    tan_x = tan_y * aspect
    x = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0) * tan_x
    y = (1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0) * tan_y
    xx, yy = np.meshgrid(x, y, indexing="xy")
    directions = f[None, None, :] + xx[..., None] * right[None, None, :] + yy[..., None] * up[None, None, :]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(np.asarray(origin, dtype=np.float64), directions.shape).copy()
    return origins.reshape(-1, 3), directions.reshape(-1, 3)


@dataclass(frozen=True)
class ViewCell:
    viewcell_id: int
    center: np.ndarray
    forward: np.ndarray
    region_type: str
    radius: float
    half_axes: np.ndarray
    right: np.ndarray
    up: np.ndarray
    subpose_positions: np.ndarray
    subpose_forwards: np.ndarray
    yaw_deg: float
    pitch_deg: float
    back_offset: float

    def region_manifest(self) -> dict[str, Any]:
        if self.region_type == "disk":
            support = [self.center]
            angles = np.arange(8, dtype=np.float64) * math.pi / 4.0
            horizontal_forward = _normalize(self.half_axes[2], [0.0, 0.0, -1.0])
            support.extend(self.center + self.radius * (np.cos(angle) * self.right + np.sin(angle) * horizontal_forward) for angle in angles)
            return {
                "regionType": "disk",
                "center": self.center.tolist(),
                "radius": self.radius,
                "right": self.right.tolist(),
                "forward": horizontal_forward.tolist(),
                "supportPoints": np.asarray(support).tolist(),
            }
        axes = self.half_axes
        signs = np.asarray([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        return {
            "regionType": "oriented_box",
            "center": self.center.tolist(),
            "halfAxes": axes.tolist(),
            "supportPoints": np.vstack([self.center, self.center[None, :] + signs @ axes]).tolist(),
        }


def build_viewcells(
    scene: PrimitiveScene,
    scene_entry: Mapping[str, Any],
    *,
    views_per_scene: int,
) -> list[ViewCell]:
    if views_per_scene <= 0:
        raise ValueError("views_per_scene must be positive")
    seed = int(scene_entry["generatorSeed"]) + 0x51ED2705
    rng = np.random.default_rng(seed)
    scene_center = (scene.bounds[0] + scene.bounds[1]) * 0.5
    scene_extent = scene.bounds[1] - scene.bounds[0]
    horizontal_extent = max(float(scene_extent[0]), float(scene_extent[2]), 1.0)
    # The old external orbit put the complete synthetic scene inside one 66
    # degree frustum. V5 samples view cells in the navigable scene volume so
    # candidate generation exercises the intended geometry-only culling path.
    orbit_radius = max(horizontal_extent * SYNTHETIC_CAMERA_ORBIT_FRACTION, 8.0)
    viewcells: list[ViewCell] = []
    for viewcell_id in range(views_per_scene):
        azimuth = 2.0 * math.pi * viewcell_id / views_per_scene + float(rng.uniform(-0.04, 0.04))
        elevation = float(rng.uniform(-0.10, 0.24))
        horizontal = np.asarray([math.cos(azimuth), 0.0, math.sin(azimuth)])
        position = scene_center + horizontal * (orbit_radius * float(rng.uniform(1.0, 1.25)))
        position[1] = scene_center[1] + scene_extent[1] * (0.25 + 0.5 * float(rng.random()))
        forward = _normalize(scene_center + np.asarray([0.0, scene_extent[1] * 0.18, 0.0]) - position, [0.0, 0.0, -1.0])
        f, right, up = _camera_basis(forward)
        radius = max(0.45, min(3.0, horizontal_extent * float(rng.uniform(0.018, 0.032))))
        region_type = "disk" if viewcell_id % 2 == 0 else "oriented_box"
        if region_type == "disk":
            horizontal_forward = _normalize(np.cross(right, np.asarray([0.0, 1.0, 0.0])), [0.0, 0.0, -1.0])
            # Keep the disk in the world-horizontal plane.
            half_axes = np.stack([right * radius, np.zeros(3), horizontal_forward * radius], axis=0)
            offsets = np.asarray(
                [
                    [0.0, 0.0],
                    [0.55 * radius, 0.0],
                    [-0.55 * radius, 0.0],
                    [0.0, 0.55 * radius],
                ]
            )
            subpose_positions = position[None, :] + offsets[:, 0, None] * right[None, :] + offsets[:, 1, None] * horizontal_forward[None, :]
        else:
            vertical_radius = max(0.35, min(1.5, scene_extent[1] * 0.02))
            half_axes = np.stack([right * radius, f * radius, up * vertical_radius], axis=0)
            offsets = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.5 * radius, 0.35 * radius, 0.35 * vertical_radius],
                    [-0.5 * radius, -0.35 * radius, -0.35 * vertical_radius],
                    [0.25 * radius, -0.4 * radius, 0.4 * vertical_radius],
                ]
            )
            subpose_positions = position[None, :] + offsets[:, 0, None] * right[None, :] + offsets[:, 1, None] * f[None, :] + offsets[:, 2, None] * up[None, :]
        viewcells.append(
            ViewCell(
                viewcell_id,
                position.astype(np.float64),
                f.astype(np.float64),
                region_type,
                radius,
                half_axes.astype(np.float64),
                right.astype(np.float64),
                up.astype(np.float64),
                subpose_positions.astype(np.float64),
                np.broadcast_to(f[None, :], (SUBPOSES_PER_VIEWCELL, 3)).copy().astype(np.float64),
                math.degrees(azimuth),
                math.degrees(elevation),
                max(2.0 * radius, orbit_radius * 0.18),
            )
        )
    return viewcells


def _candidate_ids_for_viewcell(
    scene: PrimitiveScene,
    viewcell: ViewCell,
    *,
    width: int,
    height: int,
    near: float = DEFAULT_NEAR,
) -> np.ndarray:
    tan_y = math.tan(math.radians(MODEL_FOV_Y_DEG) * 0.5)
    tan_x = tan_y * float(width) / float(height)
    aabbs = scene.aabbs.reshape(scene.unit_count, 6)
    selected: list[np.ndarray] = []
    # A subpose stores the query position. Candidate generation uses the
    # corresponding backed candidate camera, matching the runtime formula
    # candidate_camera = query_center - forward * back_offset.
    candidate_origins = (
        viewcell.subpose_positions
        - viewcell.subpose_forwards * float(viewcell.back_offset)
    )
    for origin, forward in zip(candidate_origins, viewcell.subpose_forwards):
        selected.append(
            frustum_candidate_ids_for_pose(origin, forward, tan_x, tan_y, aabbs, near=near).astype(np.uint32)
        )
    if not selected:
        return np.zeros((0,), dtype=np.uint32)
    return np.unique(np.concatenate(selected)).astype(np.uint32)


def render_viewcell_color_id(
    scene: PrimitiveScene,
    viewcell: ViewCell,
    *,
    render_width: int,
    render_height: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render all subposes and aggregate IDs, max pixel-ppm, and hit counts."""

    max_weights = np.zeros(scene.unit_count, dtype=np.float32)
    hit_counts = np.zeros(scene.unit_count, dtype=np.uint16)
    for origin, forward in zip(viewcell.subpose_positions, viewcell.subpose_forwards):
        origins, directions = _camera_rays(origin, forward, render_width, render_height, RENDER_FOV_Y_DEG)
        ids = raycast_primitives(scene, origins, directions, device=device)
        valid = ids >= 0
        if not bool(valid.any()):
            continue
        counts = np.bincount(ids[valid], minlength=scene.unit_count).astype(np.float64)
        weights = (counts / float(render_width * render_height) * 1_000_000.0).astype(np.float32)
        present = counts > 0.0
        hit_counts[present] = np.minimum(hit_counts[present].astype(np.uint32) + 1, np.iinfo(np.uint16).max).astype(np.uint16)
        max_weights = np.maximum(max_weights, weights)
    visible = np.flatnonzero(max_weights > 0.0).astype(np.uint32)
    return visible, max_weights[visible].astype(np.float32), hit_counts[visible]


def _conservative_mvp(camera_world: np.ndarray, forward: np.ndarray, tan_x: float, tan_y: float) -> np.ndarray:
    f, right, up = _camera_basis(forward)
    camera = np.asarray(camera_world, dtype=np.float64)
    result = np.zeros((16,), dtype=np.float32)
    result[0], result[4], result[8], result[12] = right[0] / tan_x, right[1] / tan_x, right[2] / tan_x, -float(right @ camera) / tan_x
    result[1], result[5], result[9], result[13] = up[0] / tan_y, up[1] / tan_y, up[2] / tan_y, -float(up @ camera) / tan_y
    result[2], result[6], result[10], result[14] = f[0], f[1], f[2], -float(f @ camera)
    result[3], result[7], result[11], result[15] = f[0], f[1], f[2], -float(f @ camera)
    return result


def _camera_bounds(scene: PrimitiveScene, viewcells: Sequence[ViewCell]) -> dict[str, list[float]]:
    cameras = np.asarray([cell.center - cell.forward * cell.back_offset for cell in viewcells], dtype=np.float64)
    low = np.minimum(scene.bounds[0], cameras.min(axis=0))
    high = np.maximum(scene.bounds[1], cameras.max(axis=0))
    padding = np.maximum((high - low) * 0.05, 1.0)
    low -= padding
    high += padding
    return {
        "min": low.tolist(),
        "max": high.tolist(),
        "center": ((low + high) * 0.5).tolist(),
        "size": (high - low).tolist(),
    }


def _normalize_camera(camera: np.ndarray, bounds: Mapping[str, Any]) -> np.ndarray:
    low = np.asarray(bounds["min"], dtype=np.float64)
    high = np.asarray(bounds["max"], dtype=np.float64)
    return ((camera - low) / np.maximum(high - low, 1.0e-12) * 2.0 - 1.0).astype(np.float32)


@dataclass(frozen=True)
class ColumnarProbeAsset:
    unit_ids: np.ndarray
    directions: np.ndarray
    hit_distances: np.ndarray
    max_distances: np.ndarray
    start_ids: np.ndarray
    direction_ids: np.ndarray
    manifest: dict[str, Any]


def _probe_manifest(scene_id: str, split: str, num_units: int, files: Mapping[str, str]) -> dict[str, Any]:
    records = int(num_units) * SURFACE_STARTS_PER_UNIT * PROBE_DIRECTIONS_PER_UNIT
    return {
        "schema": PROBE_RECORD_SCHEMA,
        "version": 2,
        "assetKind": "external_hit_probe",
        "sceneId": str(scene_id),
        "split": str(split),
        "sourceRole": "source_train",
        "permissions": {
            "assetKind": "external_hit_probe",
            "access": "source_train_only",
            "heldOutReadable": False,
            "visibilityLabelsIncluded": False,
        },
        "containsVisibilityLabels": False,
        "raySchema": EXTERNAL_HIT_RAY_SCHEMA,
        "surfaceStartsPerUnit": SURFACE_STARTS_PER_UNIT,
        "directionsPerUnit": PROBE_DIRECTIONS_PER_UNIT,
        "recordsFile": "external_hit_probes.columnar",
        "storage": "columnar_memmap_little_endian_v1",
        "rowCount": records,
        "rowLayout": "[unit][directionId][startId]",
        "eventEncoding": "finite_hit_distance_is_event",
        "hitDistanceOrigin": "target_center_directional_projection",
        "rayOriginOffset": "direction_world_times_1e-5_times_unit_radius",
        "unitRadius": "component_aabb_half_diagonal",
        "unitOrder": "unitIds_column_order_matches_source_unit_order",
        "directionSet": "icosahedron12_plus_fibonacci24",
        "anchorDirections": icosahedron12_directions(dtype=np.float64).tolist(),
        "distanceRatios": list(PROBE_DISTANCE_RATIOS),
        "maxTraceDistanceRatio": 1024.0,
        "recordFields": ["sceneId", "unitId", "probeId", "startPointWorld", "directionWorld", "hitTargetCenterDepth", "maxTargetCenterDepth", "event"],
        "numUnits": int(num_units),
        "sceneNumUnits": int(num_units),
        "unitIds": list(range(int(num_units))),
        "columnDtypes": {
            "unitIds": "uint32",
            "directions": "float32[3]",
            "hitDistances": "float32",
            "maxDistances": "float32",
            "startIds": "uint8",
            "directionIds": "uint8",
        },
        "columnShapes": {
            "unitIds": [records],
            "directions": [records, 3],
            "hitDistances": [records],
            "maxDistances": [records],
            "startIds": [records],
            "directionIds": [records],
        },
        "byteOrder": "little-endian",
        "files": dict(files),
        "materialRule": "formal_color_id_double_side_with_source_alpha_test_provenance",
        "acceleration": "vectorized_exact_analytic_box_cylinder_intersection",
    }


def build_external_hit_probe_asset(
    scene: PrimitiveScene,
    surface_asset: Any,
    *,
    scene_split: str,
    device: str,
) -> ColumnarProbeAsset:
    """Build R=N*16*36 probes with one nearest external hit per row."""

    if scene_split != "train":
        raise ValueError("external-hit probes are source-train-only assets")
    directions_table = fixed_probe_directions().astype(np.float64)
    starts = np.asarray(surface_asset.points[:, :SURFACE_STARTS_PER_UNIT, :3], dtype=np.float64)
    centers = (scene.aabbs[:, 0, :] + scene.aabbs[:, 1, :]) * 0.5
    radii = np.linalg.norm(scene.aabbs[:, 1, :] - scene.aabbs[:, 0, :], axis=1) * 0.5
    world_starts = starts * radii[:, None, None] + centers[:, None, :]
    unit_ids = np.repeat(np.arange(scene.unit_count, dtype=np.uint32), SURFACE_STARTS_PER_UNIT * PROBE_DIRECTIONS_PER_UNIT)
    direction_ids = np.tile(np.repeat(np.arange(PROBE_DIRECTIONS_PER_UNIT, dtype=np.uint8), SURFACE_STARTS_PER_UNIT), scene.unit_count)
    start_ids = np.tile(np.arange(SURFACE_STARTS_PER_UNIT, dtype=np.uint8), scene.unit_count * PROBE_DIRECTIONS_PER_UNIT)
    probe_directions = directions_table[direction_ids]
    probe_starts = world_starts[unit_ids.astype(np.int64), start_ids.astype(np.int64)]
    epsilon = np.maximum(radii[unit_ids.astype(np.int64)] * 1.0e-5, 1.0e-7)
    origins = probe_starts + epsilon[:, None] * probe_directions
    max_distances = radii[unit_ids.astype(np.int64)] * 1024.0
    origin_depths = np.einsum(
        "bi,bi->b",
        origins - centers[unit_ids.astype(np.int64)],
        probe_directions,
    )
    trace_limits = max_distances - origin_depths
    if np.any(trace_limits <= 0.0):
        raise ValueError("synthetic surface origin exceeds target-centered probe range")
    hit_ids, hit_from_offset = raycast_primitive_hits(
        scene,
        origins,
        probe_directions,
        device=device,
        max_distance=trace_limits,
        ignore_unit_ids=unit_ids.astype(np.int64),
        ray_chunk=2048,
        unit_chunk=256,
    )
    del hit_ids
    hit_target_depths = np.maximum(0.0, origin_depths + hit_from_offset)
    finite = np.isfinite(hit_from_offset) & (hit_target_depths <= max_distances + 1.0e-6)
    hit_distances = np.where(finite, hit_target_depths, np.nan).astype(np.float32)
    files = {
        "unitIds": "probe_unit_ids_uint32.bin",
        "directions": "probe_directions_fp32.bin",
        "hitDistances": "probe_hit_distances_fp32.bin",
        "maxDistances": "probe_max_distances_fp32.bin",
        "startIds": "probe_start_ids_uint8.bin",
        "directionIds": "probe_direction_ids_uint8.bin",
    }
    manifest = _probe_manifest(scene.scene_id, scene_split, scene.unit_count, files)
    return ColumnarProbeAsset(
        unit_ids,
        probe_directions.astype(np.float32),
        hit_distances,
        max_distances.astype(np.float32),
        start_ids,
        direction_ids,
        manifest,
    )


def validate_columnar_probe_asset(asset: ColumnarProbeAsset) -> None:
    manifest = asset.manifest
    validate_probe_manifest(manifest)
    expected = int(manifest["numUnits"]) * SURFACE_STARTS_PER_UNIT * PROBE_DIRECTIONS_PER_UNIT
    arrays = (asset.unit_ids, asset.directions, asset.hit_distances, asset.max_distances, asset.start_ids, asset.direction_ids)
    if asset.manifest["rowCount"] != expected or any(array.shape[0] != expected for array in arrays):
        raise ValueError("columnar probe record count must equal N*16*36")
    if asset.directions.shape != (expected, 3):
        raise ValueError("probe directions must have shape [R,3]")
    if not np.isfinite(asset.directions).all() or not np.isfinite(asset.max_distances).all():
        raise ValueError("probe direction/max columns contain non-finite values")
    if np.any(~np.isfinite(asset.hit_distances) & ~np.isnan(asset.hit_distances)):
        raise ValueError("probe hit distances contain invalid non-finite values")
    if not np.allclose(np.linalg.norm(asset.directions, axis=1), 1.0, atol=1.0e-5):
        raise ValueError("probe directions must be unit vectors")
    if np.any(asset.max_distances <= 0.0) or np.any(np.isfinite(asset.hit_distances) & (asset.hit_distances < 0.0)) or np.any(np.isfinite(asset.hit_distances) & (asset.hit_distances > asset.max_distances)):
        raise ValueError("probe hit/max distance columns are invalid")


def write_columnar_probe_asset(asset: ColumnarProbeAsset, output_dir: str | Path) -> Path:
    validate_columnar_probe_asset(asset)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.asarray(asset.unit_ids, dtype="<u4").tofile(output / asset.manifest["files"]["unitIds"])
    np.asarray(asset.directions, dtype="<f4").tofile(output / asset.manifest["files"]["directions"])
    np.asarray(asset.hit_distances, dtype="<f4").tofile(output / asset.manifest["files"]["hitDistances"])
    np.asarray(asset.max_distances, dtype="<f4").tofile(output / asset.manifest["files"]["maxDistances"])
    np.asarray(asset.start_ids, dtype="u1").tofile(output / asset.manifest["files"]["startIds"])
    np.asarray(asset.direction_ids, dtype="u1").tofile(output / asset.manifest["files"]["directionIds"])
    # Keep the generated asset on the same canonical manifest name consumed
    # by the protected V5 loader and by registered real-scene probes.
    _json_dump(output / "external_hit_probe_manifest.json", asset.manifest)
    return output / "external_hit_probe_manifest.json"


def expand_probe_distance_grid(
    asset: ColumnarProbeAsset,
    unit_radii: Any,
) -> dict[str, np.ndarray]:
    """Expand one nearest hit into the frozen 13 distance-grid observations."""

    validate_columnar_probe_asset(asset)
    radii = np.asarray(unit_radii, dtype=np.float64).reshape(-1)
    if radii.size != int(asset.manifest["numUnits"]):
        raise ValueError("unit_radii does not match probe manifest")
    distances = radii[asset.unit_ids.astype(np.int64), None] * np.asarray(PROBE_DISTANCE_RATIOS, dtype=np.float64)[None, :]
    hit = asset.hit_distances[:, None]
    events = np.isfinite(hit) & (hit <= distances)
    return {
        "unitIds": np.broadcast_to(asset.unit_ids[:, None], events.shape).copy(),
        "directionIds": np.broadcast_to(asset.direction_ids[:, None], events.shape).copy(),
        "distances": distances.astype(np.float32),
        "events": events.astype(np.uint8),
        "hitDistances": np.broadcast_to(asset.hit_distances[:, None], events.shape).copy(),
    }


def _write_runtime_assets(scene: PrimitiveScene, scene_root: Path, streaming_granularity: int) -> None:
    resources: dict[int, list[int]] = {}
    for unit in scene.units:
        resources.setdefault(unit.resource_id, []).append(unit.unit_id)
    resource_ids = sorted(resources)
    component_records = []
    for unit in scene.units:
        component_records.append(
            {
                "componentGlobalId": unit.unit_id,
                "globalGlbId": unit.resource_id,
                "primitiveType": unit.primitive_type,
                "bounds": {"min": unit.world_aabb[0].tolist(), "max": unit.world_aabb[1].tolist()},
            }
        )
    resource_records = [
        {
            "globalGlbId": resource_id,
            "path": f"synthetic/{scene.scene_id}/resource_{resource_id:05d}.glb",
            "byteLength": int(streaming_granularity) * 1024,
            "componentGlobalIds": resources[resource_id],
        }
        for resource_id in resource_ids
    ]
    runtime = {
        "schema": "synthetic-runtime-visibility-meta-v1",
        "sceneId": scene.scene_id,
        "instanceCount": scene.unit_count,
        "componentCount": scene.unit_count,
        "globalGlbCount": len(resource_records),
        "sceneBounds": {"min": scene.bounds[0].tolist(), "max": scene.bounds[1].tolist()},
        "componentRecords": component_records,
        "globalGlbRecords": resource_records,
        "instanceToGlobalGlb": [unit.resource_id for unit in scene.units],
        "geometrySource": "deterministic_closed_box_cylinder_beam_panel_meshes",
        "streamingGranularityKiB": int(streaming_granularity),
    }
    glb_index = {
        "schema": "synthetic-glb-index-v1",
        "sceneId": scene.scene_id,
        "total": len(resource_records),
        "entries": resource_records,
    }
    _json_dump(scene_root / "runtimeVisibilityMeta.json", runtime)
    _json_dump(scene_root / "glbIndex.json", glb_index)


def _pose_split(index: int, count: int) -> str:
    if count <= 1:
        return "train"
    train_end = max(1, int(math.ceil(count * 0.75)))
    calibration_end = max(train_end, int(math.ceil(count * 0.875)))
    validation_end = max(calibration_end, int(math.ceil(count * 0.9375)))
    if index < train_end:
        return "train"
    if index < calibration_end:
        return "calibration"
    if index < validation_end:
        return "validation"
    return "test"


def _write_pose_csr(
    scene: PrimitiveScene,
    scene_entry: Mapping[str, Any],
    viewcells: Sequence[ViewCell],
    rendered: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    pose_dir: Path,
    render_width: int,
    render_height: int,
) -> dict[str, Any]:
    pose_dir.mkdir(parents=True, exist_ok=True)
    view_count = len(viewcells)
    aspect = float(render_width) / float(render_height)
    model_tan_y = math.tan(math.radians(MODEL_FOV_Y_DEG) * 0.5)
    model_tan_x = model_tan_y * aspect
    render_tan_y = math.tan(math.radians(RENDER_FOV_Y_DEG) * 0.5)
    render_tan_x = render_tan_y * aspect
    camera_bounds = _camera_bounds(scene, viewcells)
    poses = np.zeros((view_count,), dtype=DIRECTIONAL_POSE_DTYPE)
    visible_offsets = np.zeros((view_count + 1,), dtype=np.uint64)
    candidate_offsets = np.zeros((view_count + 1,), dtype=np.uint64)
    subpose_offsets = np.zeros((view_count + 1,), dtype=np.uint64)
    visible_ids: list[int] = []
    visible_weights: list[float] = []
    visible_hits: list[int] = []
    candidate_ids: list[int] = []
    subpose_positions: list[list[float]] = []
    subpose_forwards: list[list[float]] = []
    subpose_pose_indices: list[int] = []
    subpose_params: list[list[float]] = []
    mvps = np.zeros((view_count, 16), dtype=np.float32)
    viewcell_split_ids = np.zeros((view_count,), dtype=np.uint8)
    category_id = _FAMILY_CATEGORY_IDS.get(str(scene_entry.get("structureFamily")), 255)
    visible_subset_failures = 0
    per_view_stats: list[dict[str, Any]] = []
    for index, (viewcell, labels) in enumerate(zip(viewcells, rendered)):
        visible, weights, hits = labels
        candidates = _candidate_ids_for_viewcell(scene, viewcell, width=render_width, height=render_height)
        missing = np.setdiff1d(visible, candidates, assume_unique=True)
        if missing.size:
            visible_subset_failures += 1
            raise ValueError(
                f"{scene.scene_id} view-cell {index} has {missing.size} visible units "
                "outside the four-subpose 66-degree AABB candidate union"
            )
        pose_split = _pose_split(index, view_count)
        poses["camera_world"][index] = viewcell.center - viewcell.forward * viewcell.back_offset
        poses["camera_forward"][index] = viewcell.forward.astype(np.float32)
        poses["camera_view"][index] = np.asarray([model_tan_x, model_tan_y], dtype=np.float32)
        poses["split"][index] = _SPLIT_IDS[pose_split]
        poses["category"][index] = category_id
        visible_ids.extend(int(value) for value in visible.tolist())
        visible_weights.extend(float(value) for value in weights.tolist())
        visible_hits.extend(int(value) for value in hits.tolist())
        candidate_ids.extend(int(value) for value in candidates.tolist())
        visible_offsets[index + 1] = len(visible_ids)
        candidate_offsets[index + 1] = len(candidate_ids)
        subpose_positions.extend(viewcell.subpose_positions.astype(np.float32).tolist())
        subpose_forwards.extend(viewcell.subpose_forwards.astype(np.float32).tolist())
        subpose_pose_indices.extend(range(index * SUBPOSES_PER_VIEWCELL, (index + 1) * SUBPOSES_PER_VIEWCELL))
        subpose_params.extend([[RENDER_FOV_Y_DEG, aspect, float(render_width), float(render_height)]] * SUBPOSES_PER_VIEWCELL)
        subpose_offsets[index + 1] = len(subpose_positions)
        mvps[index] = _conservative_mvp(poses["camera_world"][index], viewcell.forward, model_tan_x, model_tan_y)
        viewcell_split_ids[index] = _SPLIT_IDS[pose_split]
        per_view_stats.append({"viewcellId": index, "poseSplit": pose_split, "candidateCount": int(candidates.size), "visibleCount": int(visible.size), "missingCandidateCount": int(missing.size)})
    poses["camera_norm"] = np.asarray([_normalize_camera(value, camera_bounds) for value in poses["camera_world"]], dtype=np.float32)
    poses.tofile(pose_dir / "poses.bin")
    mvps.tofile(pose_dir / "mvp.bin")
    visible_offsets.tofile(pose_dir / "visible_offsets.bin")
    np.asarray(visible_ids, dtype="<u4").tofile(pose_dir / "visible_ids.bin")
    np.asarray(visible_weights, dtype="<f4").tofile(pose_dir / "visible_weights.bin")
    np.asarray(visible_hits, dtype="<u2").tofile(pose_dir / "visible_hit_counts.bin")
    candidate_offsets.tofile(pose_dir / "candidate_offsets.bin")
    np.asarray(candidate_ids, dtype="<u4").tofile(pose_dir / "candidate_ids.bin")
    # Preserve the exact geometry-only candidate table under the audit names.
    np.asarray(candidate_ids, dtype="<u4").tofile(pose_dir / "raw_candidate_ids.bin")
    candidate_offsets.tofile(pose_dir / "raw_candidate_offsets.bin")
    np.asarray([cell.viewcell_id for cell in viewcells], dtype="<u4").tofile(pose_dir / "viewcell_ids.bin")
    np.asarray([cell.center for cell in viewcells], dtype="<f4").tofile(pose_dir / "viewcell_centers.bin")
    np.asarray([cell.center for cell in viewcells], dtype="<f4").tofile(pose_dir / "query_center_world.bin")
    np.asarray(poses["camera_world"], dtype="<f4").tofile(pose_dir / "candidate_camera_world.bin")
    np.asarray([cell.radius for cell in viewcells], dtype="<f4").tofile(pose_dir / "viewcell_radius_m.bin")
    np.asarray([cell.forward for cell in viewcells], dtype="<f4").tofile(pose_dir / "viewcell_forwards.bin")
    np.asarray(
        [[cell.radius, MODEL_FOV_Y_DEG, RENDER_FOV_Y_DEG, math.degrees(2.0 * math.atan(model_tan_x)), cell.back_offset, cell.yaw_deg, cell.pitch_deg, SUBPOSES_PER_VIEWCELL] for cell in viewcells],
        dtype="<f4",
    ).tofile(pose_dir / "viewcell_params.bin")
    np.full((view_count,), category_id, dtype="u1").tofile(pose_dir / "viewcell_category_ids.bin")
    viewcell_split_ids.tofile(pose_dir / "viewcell_split_ids.bin")
    subpose_offsets.tofile(pose_dir / "subpose_offsets.bin")
    np.asarray(subpose_positions, dtype="<f4").tofile(pose_dir / "subpose_camera_pos.bin")
    np.asarray(subpose_forwards, dtype="<f4").tofile(pose_dir / "subpose_camera_forward.bin")
    np.asarray(subpose_pose_indices, dtype="<u4").tofile(pose_dir / "subpose_pose_indices.bin")
    np.asarray(subpose_params, dtype="<f4").tofile(pose_dir / "subpose_params.bin")
    split_counts = {name: int(np.count_nonzero(poses["split"] == split_id)) for name, split_id in _SPLIT_IDS.items()}
    meta = {
        "schema": "pose-csr-explicit-four-way-split-v1",
        "sourceAggregationSchema": "viewcell-csr-color-id-approximate-primitive-v1",
        "experiment": EXPERIMENT_NAME,
        "sourceSampler": "synthetic_vectorized_analytic_color_id",
        "runtimeMeta": "../runtimeVisibilityMeta.json",
        "sceneId": scene.scene_id,
        "sceneSeedSplit": str(scene_entry["split"]),
        "viewcellCount": view_count,
        "poseCount": view_count,
        "numInstances": scene.unit_count,
        "poseStrideBytes": int(DIRECTIONAL_POSE_DTYPE.itemsize),
        "mvpStrideBytes": 64,
        "visibleCount": len(visible_ids),
        "candidateCount": len(candidate_ids),
        "candidateSemantics": "union of geometry-only 66-degree AABB frusta at the four backed subpose candidate cameras",
        "candidateVisibleUnionAllowed": False,
        "rawCandidateFile": "raw_candidate_ids.bin",
        "rawCandidateOffsets": "raw_candidate_offsets.bin",
        "rawCandidateSemantics": "same geometry-only candidate table before label aggregation",
        "cameraSemantics": "poses.camera_world is the backed 66-degree candidate camera for the view-cell center",
        "subposeCandidateCameraSemantics": "subpose_camera_pos - subpose_camera_forward * back_offset",
        "queryCenterSemantics": "view-cell center used by the nine-point region contract",
        "viewcellGeometrySemantics": "disk and camera-oriented box regions share one forward direction and four offline subposes",
        "sourceViewcellCount": view_count,
        "sourceSubposeCount": view_count * SUBPOSES_PER_VIEWCELL,
        "gtSemantics": "union of exact analytic primitive Color-ID hits over four same-direction subposes",
        "visibleWeightSemantics": "max pooled Color-ID pixel count in parts per million over subposes",
        "visibleWeightDtype": "float32",
        "modelInputFovYDeg": MODEL_FOV_Y_DEG,
        "frontendRenderFovYDeg": RENDER_FOV_Y_DEG,
        "renderWidth": render_width,
        "renderHeight": render_height,
        "cameraBounds": camera_bounds,
        "splitIds": _SPLIT_IDS,
        "categoryIds": _FAMILY_CATEGORY_IDS,
        "categoryCounts": {str(scene_entry.get("structureFamily")): view_count},
        "splitCounts": split_counts,
        "stats": {
            "avgCandidate": float(len(candidate_ids) / max(1, view_count)),
            "avgVisible": float(len(visible_ids) / max(1, view_count)),
            "avgSubposesPerViewcell": float(SUBPOSES_PER_VIEWCELL),
            "visibleSubsetFailures": visible_subset_failures,
            "viewcellStats": per_view_stats,
        },
        "files": {
            "poses": "poses.bin",
            "mvp": "mvp.bin",
            "viewcellIds": "viewcell_ids.bin",
            "viewcellCenters": "viewcell_centers.bin",
            "queryCenterWorld": "query_center_world.bin",
            "candidateCameraWorld": "candidate_camera_world.bin",
            "viewcellRadiusM": "viewcell_radius_m.bin",
            "viewcellForwards": "viewcell_forwards.bin",
            "viewcellParams": "viewcell_params.bin",
            "viewcellCategoryIds": "viewcell_category_ids.bin",
            "viewcellSplitIds": "viewcell_split_ids.bin",
            "subposeOffsets": "subpose_offsets.bin",
            "subposeCameraPos": "subpose_camera_pos.bin",
            "subposeCameraForward": "subpose_camera_forward.bin",
            "subposePoseIndices": "subpose_pose_indices.bin",
            "subposeParams": "subpose_params.bin",
            "visibleOffsets": "visible_offsets.bin",
            "visibleIds": "visible_ids.bin",
            "visibleWeights": "visible_weights.bin",
            "visibleHitCounts": "visible_hit_counts.bin",
            "candidateOffsets": "candidate_offsets.bin",
            "candidateIds": "candidate_ids.bin",
            "rawCandidateOffsets": "raw_candidate_offsets.bin",
            "rawCandidateIds": "raw_candidate_ids.bin",
        },
    }
    _json_dump(pose_dir / "dataset_meta.json", meta)
    return meta


def generate_scene_dataset(
    scene_entry: Mapping[str, Any],
    output_root: str | Path,
    *,
    device: str = "cpu",
    views_per_scene: int = DEFAULT_VIEWS_PER_SCENE,
    render_width: int = DEFAULT_RENDER_WIDTH,
    render_height: int = DEFAULT_RENDER_HEIGHT,
    allow_small_scene: bool = False,
) -> dict[str, Any]:
    """Generate one scene and return its output manifest."""

    if render_width <= 0 or render_height <= 0:
        raise ValueError("render dimensions must be positive")
    scene = build_primitive_scene(scene_entry, allow_small=allow_small_scene)
    root = Path(output_root)
    scene_root = root / scene.scene_id
    compiled_root = scene_root / "compiled"
    surface_root = compiled_root / "surface"
    relation_root = compiled_root / "relation"
    probes_root = compiled_root / "probes"
    pose_root = scene_root / "pose_csr"
    surface_asset = build_local_surface_asset(scene.surface_inputs(), sampling_seed=int(scene_entry["generatorSeed"]) + 17)
    write_local_surface_asset(surface_asset, surface_root)
    relation = build_proxy_relation_graph(scene.aabbs, unit_ids=scene.unit_ids.astype(np.uint64))
    write_proxy_relation_graph(relation, relation_root)
    probe_asset: ColumnarProbeAsset | None = None
    if str(scene_entry["split"]) == "train":
        probe_asset = build_external_hit_probe_asset(
            scene,
            surface_asset,
            scene_split="train",
            device=device,
        )
        write_columnar_probe_asset(probe_asset, probes_root)
    else:
        probes_root.mkdir(parents=True, exist_ok=True)
    viewcells = build_viewcells(scene, scene_entry, views_per_scene=views_per_scene)
    rendered = []
    for viewcell in viewcells:
        rendered.append(
            render_viewcell_color_id(
                scene,
                viewcell,
                render_width=render_width,
                render_height=render_height,
                device=device,
            )
        )
    pose_meta = _write_pose_csr(
        scene,
        scene_entry,
        viewcells,
        rendered,
        pose_dir=pose_root,
        render_width=render_width,
        render_height=render_height,
    )
    _write_runtime_assets(scene, scene_root, int(scene_entry["streamingGranularityKiB"]))
    _json_dump(
        compiled_root / "primitive_units.json",
        {
            "schema": "gcof-pvs-v5-procedural-primitive-units-v1",
            "sceneId": scene.scene_id,
            "unitCount": scene.unit_count,
            "primitiveTypes": list(_PRIMITIVE_TYPES),
            "units": [unit.mapping() for unit in scene.units],
        },
    )
    _json_dump(
        compiled_root / "compiled_manifest.json",
        {
            "schema": COMPILED_SCHEMA,
            "version": 1,
            "sceneId": scene.scene_id,
            "numUnits": scene.unit_count,
            "surface": "surface",
            "relation": "relation",
            "probes": "probes",
            "probeTrainingAllowed": probe_asset is not None,
            "geometrySource": "procedural_closed_primitive_meshes",
            "primitiveTypes": list(_PRIMITIVE_TYPES),
        },
    )
    output_manifest = {
        "schema": SCENE_OUTPUT_SCHEMA,
        "version": 1,
        "experiment": EXPERIMENT_NAME,
        "scene": dict(scene_entry),
        "sceneId": scene.scene_id,
        "seedSplit": str(scene_entry["split"]),
        "unitCount": scene.unit_count,
        "streamingGranularityKiB": int(scene_entry["streamingGranularityKiB"]),
        "geometry": {
            "primitiveTypes": list(_PRIMITIVE_TYPES),
            "asset": "compiled/primitive_units.json",
            "bounds": {"min": scene.bounds[0].tolist(), "max": scene.bounds[1].tolist()},
        },
        "viewCells": {
            "count": len(viewcells),
            "subposesPerViewcell": SUBPOSES_PER_VIEWCELL,
            "shapes": sorted({cell.region_type for cell in viewcells}),
            "renderWidth": render_width,
            "renderHeight": render_height,
            "modelFovYDeg": MODEL_FOV_Y_DEG,
            "renderFovYDeg": RENDER_FOV_Y_DEG,
            "regions": [cell.region_manifest() for cell in viewcells],
        },
        "assets": {
            "runtimeMeta": "runtimeVisibilityMeta.json",
            "glbIndex": "glbIndex.json",
            "poseCsr": "pose_csr",
            "compiled": "compiled",
            "surface": "compiled/surface",
            "relation": "compiled/relation",
            "probes": "compiled/probes",
        },
        "poseCsr": {
            "schema": pose_meta["schema"],
            "poseCount": pose_meta["poseCount"],
            "splitCounts": pose_meta["splitCounts"],
            "candidateCount": pose_meta["candidateCount"],
            "visibleCount": pose_meta["visibleCount"],
        },
        "probe": {
            "manifest": "compiled/probes/external_hit_probe_manifest.json" if probe_asset is not None else None,
            "rowCount": None if probe_asset is None else probe_asset.manifest["rowCount"],
            "distanceRatios": list(PROBE_DISTANCE_RATIOS),
            "trainingAllowed": probe_asset is not None,
            "generated": probe_asset is not None,
        },
        "determinism": {
            "generatorSeed": int(scene_entry["generatorSeed"]),
            "device": str(device),
            "fixedPrimitiveMeshBank": True,
        },
    }
    _json_dump(scene_root / "scene_manifest.json", output_manifest)
    return output_manifest


def rebuild_compiled_scene_assets(
    scene_entry: Mapping[str, Any],
    output_root: str | Path,
    *,
    device: str = "cpu",
    allow_small_scene: bool = False,
) -> dict[str, Any]:
    """Rebuild deterministic geometry assets without touching frozen pose GT."""

    scene = build_primitive_scene(scene_entry, allow_small=allow_small_scene)
    scene_root = Path(output_root) / scene.scene_id
    manifest_path = scene_root / "scene_manifest.json"
    required_frozen = (
        scene_root / "runtimeVisibilityMeta.json",
        scene_root / "glbIndex.json",
        scene_root / "pose_csr/dataset_meta.json",
        manifest_path,
    )
    missing = [str(path) for path in required_frozen if not path.is_file()]
    if missing:
        raise ValueError(
            f"cannot rebuild compiled assets for {scene.scene_id}; missing frozen inputs: "
            + ", ".join(missing)
        )
    output_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        output_manifest.get("sceneId") != scene.scene_id
        or int(output_manifest.get("unitCount", -1)) != scene.unit_count
        or output_manifest.get("seedSplit") != str(scene_entry["split"])
    ):
        raise ValueError(f"existing synthetic scene identity disagrees for {scene.scene_id}")

    compiled_root = scene_root / "compiled"
    surface_root = compiled_root / "surface"
    relation_root = compiled_root / "relation"
    probes_root = compiled_root / "probes"
    surface_asset = build_local_surface_asset(
        scene.surface_inputs(), sampling_seed=int(scene_entry["generatorSeed"]) + 17
    )
    write_local_surface_asset(surface_asset, surface_root)
    relation = build_proxy_relation_graph(scene.aabbs, unit_ids=scene.unit_ids.astype(np.uint64))
    write_proxy_relation_graph(relation, relation_root)
    probe_asset: ColumnarProbeAsset | None = None
    if str(scene_entry["split"]) == "train":
        probe_asset = build_external_hit_probe_asset(
            scene, surface_asset, scene_split="train", device=device
        )
        write_columnar_probe_asset(probe_asset, probes_root)
    else:
        probes_root.mkdir(parents=True, exist_ok=True)

    _json_dump(
        compiled_root / "primitive_units.json",
        {
            "schema": "gcof-pvs-v5-procedural-primitive-units-v1",
            "sceneId": scene.scene_id,
            "unitCount": scene.unit_count,
            "primitiveTypes": list(_PRIMITIVE_TYPES),
            "units": [unit.mapping() for unit in scene.units],
        },
    )
    _json_dump(
        compiled_root / "compiled_manifest.json",
        {
            "schema": COMPILED_SCHEMA,
            "version": 1,
            "sceneId": scene.scene_id,
            "numUnits": scene.unit_count,
            "surface": "surface",
            "relation": "relation",
            "probes": "probes",
            "probeTrainingAllowed": probe_asset is not None,
            "geometrySource": "procedural_closed_primitive_meshes",
            "primitiveTypes": list(_PRIMITIVE_TYPES),
        },
    )
    output_manifest["probe"] = {
        "manifest": "compiled/probes/external_hit_probe_manifest.json" if probe_asset is not None else None,
        "rowCount": None if probe_asset is None else probe_asset.manifest["rowCount"],
        "distanceRatios": list(PROBE_DISTANCE_RATIOS),
        "trainingAllowed": probe_asset is not None,
        "generated": probe_asset is not None,
    }
    output_manifest.setdefault("determinism", {})["compiledAssetDevice"] = str(device)
    _json_dump(manifest_path, output_manifest)
    return dict(output_manifest)


def load_complete_scene_output(
    scene_entry: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any] | None:
    """Return a complete matching scene output, or ``None`` when absent."""

    scene_id = str(scene_entry["sceneId"])
    scene_root = Path(output_root) / scene_id
    manifest_path = scene_root / "scene_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing synthetic scene manifest is invalid: {manifest_path}") from exc
    expected = {
        "schema": SCENE_OUTPUT_SCHEMA,
        "version": 1,
        "sceneId": scene_id,
        "seedSplit": str(scene_entry["split"]),
        "unitCount": int(scene_entry["unitCount"]),
        "streamingGranularityKiB": int(scene_entry["streamingGranularityKiB"]),
    }
    if not isinstance(manifest, Mapping) or any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"existing synthetic scene identity disagrees with catalog: {manifest_path}")
    required = [
        scene_root / "runtimeVisibilityMeta.json",
        scene_root / "glbIndex.json",
        scene_root / "pose_csr/dataset_meta.json",
        scene_root / "compiled/surface/surface_manifest.json",
        scene_root / "compiled/relation/relation_manifest.json",
        scene_root / "compiled/compiled_manifest.json",
    ]
    if str(scene_entry["split"]) == "train":
        required.append(scene_root / "compiled/probes/external_hit_probe_manifest.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"existing synthetic scene {scene_id} is incomplete despite its final manifest: "
            + ", ".join(missing)
        )
    relation_manifest = json.loads(required[4].read_text(encoding="utf-8"))
    validate_relation_manifest(relation_manifest)
    if str(scene_entry["split"]) == "train":
        probe_manifest = json.loads(required[-1].read_text(encoding="utf-8"))
        validate_probe_manifest(probe_manifest)
    return dict(manifest)


def build_dataset_manifest(
    catalog: Mapping[str, Any],
    *,
    generated_scene_ids: Sequence[str],
    device: str,
    views_per_scene: int,
    render_width: int,
    render_height: int,
    output_root: Path,
) -> dict[str, Any]:
    """Create the root manifest with all formal defaults explicitly recorded."""

    return {
        "schema": DATASET_SCHEMA,
        "version": 1,
        "experiment": EXPERIMENT_NAME,
        "catalogSchema": catalog["schema"],
        "method": "GCOF-PVS-V5",
        "baseSeed": int(catalog["baseSeed"]),
        "sceneCount": int(catalog["sceneCount"]),
        "splitCounts": dict(catalog["splitCounts"]),
        "structureFamilies": list(catalog["structureFamilies"]),
        "formalDefaults": {
            "viewsPerScene": int(views_per_scene),
            "subposesPerViewcell": SUBPOSES_PER_VIEWCELL,
            "renderWidth": int(render_width),
            "renderHeight": int(render_height),
            "modelFovYDeg": MODEL_FOV_Y_DEG,
            "renderFovYDeg": RENDER_FOV_Y_DEG,
            "near": DEFAULT_NEAR,
            "far": DEFAULT_FAR,
            "device": str(device),
            "probeSurfaceStartsPerUnit": SURFACE_STARTS_PER_UNIT,
            "probeDirectionsPerUnit": PROBE_DIRECTIONS_PER_UNIT,
            "probeDistanceRatios": list(PROBE_DISTANCE_RATIOS),
        },
        "outputRoot": output_root.as_posix(),
        "sceneOutputs": {scene_id: f"{scene_id}/scene_manifest.json" for scene_id in sorted(generated_scene_ids)},
        "splitProtocol": catalog["splitProtocol"],
        "assets": {
            "poseCsr": "<scene>/pose_csr",
            "compiled": "<scene>/compiled/{surface,relation,probes}",
            "runtimeMeta": "<scene>/runtimeVisibilityMeta.json",
        },
    }


def _resolve_device(value: str) -> str:
    resolved = str(value).lower()
    if resolved == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if resolved == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        except ImportError as exc:
            raise RuntimeError("CUDA generation requires PyTorch") from exc
    if resolved not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    return resolved


def select_catalog_scene_ids(
    entries: Mapping[str, Any],
    *,
    all_scenes: bool,
    requested_scene_ids: Sequence[str] | None,
    shard_index: int | None,
    shard_count: int | None,
) -> list[str]:
    """Select one deterministic, disjoint scene shard from the frozen catalog."""

    if (shard_index is None) != (shard_count is None):
        raise ValueError("scene-shard-index and scene-shard-count must be provided together")
    if shard_index is not None:
        if not all_scenes:
            raise ValueError("scene sharding is supported only with --all")
        if shard_count is None or shard_count <= 1 or not 0 <= shard_index < shard_count:
            raise ValueError("scene shard requires count > 1 and 0 <= index < count")
    selected = (
        sorted(str(scene_id) for scene_id in entries)
        if all_scenes
        else sorted(set(str(value) for value in (requested_scene_ids or ())))
    )
    unknown = sorted(set(selected) - set(entries))
    if unknown:
        raise ValueError(f"unknown synthetic scene IDs: {', '.join(unknown)}")
    if shard_index is not None and shard_count is not None:
        selected = selected[shard_index::shard_count]
    return selected


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic procedural synthetic V5 datasets.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene-id", action="append", help="Generate one catalog scene; repeat for several scenes.")
    selection.add_argument("--all", action="store_true", help="Generate all 120 catalog scenes.")
    selection.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Validate all 120 existing scene outputs and write the canonical root manifest.",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--views-per-scene", type=int, default=DEFAULT_VIEWS_PER_SCENE)
    parser.add_argument("--render-width", type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--render-height", type=int, default=DEFAULT_RENDER_HEIGHT)
    parser.add_argument("--scene-shard-index", type=int)
    parser.add_argument("--scene-shard-count", type=int)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="validate and skip scenes that already have a complete matching final manifest",
    )
    parser.add_argument(
        "--rebuild-compiled",
        action="store_true",
        help="rebuild deterministic surface/relation/probe assets while preserving frozen PoseCSR",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.views_per_scene <= 0 or args.render_width <= 0 or args.render_height <= 0:
        raise SystemExit("views-per-scene and render dimensions must be positive")
    device = _resolve_device(args.device)
    catalog = generate_synthetic_scene_manifest(int(args.base_seed))
    entries = {str(entry["sceneId"]): entry for entry in catalog["scenes"]}
    if args.finalize_existing:
        selected = sorted(entries)
        missing = [scene_id for scene_id in selected if load_complete_scene_output(entries[scene_id], args.output_root) is None]
        if missing:
            raise SystemExit(
                f"cannot finalize synthetic dataset; missing {len(missing)} scene manifests"
            )
        root_manifest = build_dataset_manifest(
            catalog,
            generated_scene_ids=selected,
            device=device,
            views_per_scene=int(args.views_per_scene),
            render_width=int(args.render_width),
            render_height=int(args.render_height),
            output_root=args.output_root,
        )
        _json_dump(args.output_root / "synthetic_dataset_manifest.json", root_manifest)
        print(json.dumps({
            "schema": DATASET_SCHEMA,
            "generatedSceneCount": len(selected),
            "finalized": True,
        }), flush=True)
        return
    try:
        selected = select_catalog_scene_ids(
            entries,
            all_scenes=bool(args.all),
            requested_scene_ids=args.scene_id,
            shard_index=args.scene_shard_index,
            shard_count=args.scene_shard_count,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output_root.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for scene_id in selected:
        if args.rebuild_compiled:
            print(f"rebuild compiled {scene_id} device={device}", flush=True)
            rebuild_compiled_scene_assets(entries[scene_id], args.output_root, device=device)
            generated.append(scene_id)
            continue
        if args.skip_existing and load_complete_scene_output(entries[scene_id], args.output_root) is not None:
            print(f"skip complete {scene_id}", flush=True)
            generated.append(scene_id)
            continue
        print(f"generate {scene_id} device={device}", flush=True)
        generate_scene_dataset(
            entries[scene_id],
            args.output_root,
            device=device,
            views_per_scene=int(args.views_per_scene),
            render_width=int(args.render_width),
            render_height=int(args.render_height),
        )
        generated.append(scene_id)
    root_manifest = build_dataset_manifest(
        catalog,
        generated_scene_ids=generated,
        device=device,
        views_per_scene=int(args.views_per_scene),
        render_width=int(args.render_width),
        render_height=int(args.render_height),
        output_root=args.output_root,
    )
    manifest_name = (
        "synthetic_dataset_manifest.json"
        if args.scene_shard_index is None
        else f"synthetic_dataset_manifest_shard_{args.scene_shard_index:02d}_of_{args.scene_shard_count:02d}.json"
    )
    _json_dump(args.output_root / manifest_name, root_manifest)
    print(json.dumps({"schema": DATASET_SCHEMA, "generatedSceneCount": len(generated), "scenes": generated}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
