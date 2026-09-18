"""Build deterministic, locally normalized surface-point assets for V5.

Formal dense assets use a 16-byte little-endian header:
``magic[4], version[2], feature_dim[2], num_units[4], items_per_unit[4]``.
The point payload is FP32 ``[num_units,256,6]`` and the size-ratio payload is
a separate FP32 ``[num_units,3]`` file with ``items_per_unit=1``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .schemas import (
    POINTS_PER_UNIT,
    LOCAL_SURFACE_SCHEMA,
    SchemaError,
    validate_local_surface_manifest,
    write_json_manifest,
    read_json_manifest,
)


SURFACE_BINARY_MAGIC = b"GPV5"
SURFACE_BINARY_VERSION = 1
SURFACE_BINARY_HEADER = struct.Struct("<4sHHII")


class DegenerateSurfaceError(ValueError):
    """Raised when a unit has no usable world-space triangle area."""


@dataclass(frozen=True)
class SurfaceUnitInput:
    unit_id: int
    vertices: np.ndarray
    triangles: np.ndarray
    transform: np.ndarray


@dataclass(frozen=True)
class LocalSurfaceSample:
    unit_id: int
    points: np.ndarray
    size_ratios: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    radius: float
    world_area: float
    degenerate: bool


@dataclass(frozen=True)
class LocalSurfaceAsset:
    unit_ids: np.ndarray
    points: np.ndarray
    size_ratios: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    degenerate_unit_ids: tuple[int, ...]
    manifest: dict[str, Any]


def _array(value: Any, shape: tuple[int, ...], name: str, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _coerce_unit(value: SurfaceUnitInput | Mapping[str, Any]) -> SurfaceUnitInput:
    if isinstance(value, SurfaceUnitInput):
        unit = value
    elif isinstance(value, Mapping):
        try:
            unit = SurfaceUnitInput(
                unit_id=int(value["unit_id"]),
                vertices=np.asarray(value["vertices"], dtype=np.float64),
                triangles=np.asarray(value["triangles"], dtype=np.int64),
                transform=np.asarray(value.get("transform", np.eye(4)), dtype=np.float64),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("surface unit mapping requires unit_id, vertices, and triangles") from exc
    else:
        raise TypeError("surface units must be SurfaceUnitInput or a mapping")
    if isinstance(unit.unit_id, bool) or unit.unit_id < 0:
        raise ValueError("unit_id must be a non-negative integer")
    vertices = np.asarray(unit.vertices, dtype=np.float64)
    triangles = np.asarray(unit.triangles, dtype=np.int64)
    transform = np.asarray(unit.transform, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError("vertices must have shape [V,3] and be non-empty")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape [T,3]")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= vertices.shape[0]):
        raise ValueError("triangle index is outside the vertex array")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError("transform must be an affine 4x4 matrix")
    if abs(float(np.linalg.det(transform[:3, :3]))) <= 1e-12:
        raise ValueError("transform linear part must be invertible for normal transformation")
    return SurfaceUnitInput(int(unit.unit_id), vertices, triangles, transform)


def _world_vertices(unit: SurfaceUnitInput) -> np.ndarray:
    homogeneous = np.concatenate(
        [unit.vertices, np.ones((unit.vertices.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (homogeneous @ unit.transform.T)[:, :3]


def sample_unit_surface(
    unit: SurfaceUnitInput | Mapping[str, Any],
    *,
    seed: int,
    points_per_unit: int = POINTS_PER_UNIT,
) -> LocalSurfaceSample:
    """Sample actual transformed triangle area and return six-dimensional points.

    Coordinates are normalized by the world AABB center and half diagonal.
    Normals use the inverse-transpose of the actual instance transform.  A
    degenerate unit returns zero samples instead of fabricated zero points.
    """

    if points_per_unit != POINTS_PER_UNIT:
        raise ValueError("V5 points_per_unit is fixed at 256")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("surface sampling seed must be a non-negative integer")
    value = _coerce_unit(unit)
    world = _world_vertices(value)
    aabb_min = world.min(axis=0)
    aabb_max = world.max(axis=0)
    extents = aabb_max - aabb_min
    center = (aabb_min + aabb_max) * 0.5
    radius = float(np.linalg.norm(extents) * 0.5)

    if value.triangles.shape[0] == 0 or radius <= 1e-12:
        return LocalSurfaceSample(
            value.unit_id,
            np.empty((0, 6), dtype=np.float32),
            _size_ratios(extents),
            aabb_min.astype(np.float32),
            aabb_max.astype(np.float32),
            radius,
            0.0,
            True,
        )

    triangles = world[value.triangles]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    triangle_areas = 0.5 * np.linalg.norm(cross, axis=1)
    usable = np.isfinite(triangle_areas) & (triangle_areas > 1e-14)
    if not bool(usable.any()):
        return LocalSurfaceSample(
            value.unit_id,
            np.empty((0, 6), dtype=np.float32),
            _size_ratios(extents),
            aabb_min.astype(np.float32),
            aabb_max.astype(np.float32),
            radius,
            0.0,
            True,
        )

    usable_indices = np.flatnonzero(usable)
    usable_areas = triangle_areas[usable]
    world_area = float(usable_areas.sum())
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(usable_indices, size=points_per_unit, replace=True, p=usable_areas / world_area)
    barycentric_u = np.sqrt(rng.random(points_per_unit))
    barycentric_v = rng.random(points_per_unit)
    weights = np.column_stack(
        [1.0 - barycentric_u, barycentric_u * (1.0 - barycentric_v), barycentric_u * barycentric_v]
    )
    sampled_triangles = triangles[selected]
    sampled_points = np.einsum("ni,nij->nj", weights, sampled_triangles)

    linear = value.transform[:3, :3]
    inverse_transpose = np.linalg.inv(linear).T
    local_normals = np.cross(
        value.vertices[value.triangles[:, 1]] - value.vertices[value.triangles[:, 0]],
        value.vertices[value.triangles[:, 2]] - value.vertices[value.triangles[:, 0]],
    )
    sampled_normals = local_normals[selected] @ inverse_transpose.T
    normal_lengths = np.linalg.norm(sampled_normals, axis=1, keepdims=True)
    sampled_normals /= np.maximum(normal_lengths, 1e-12)
    if not np.isfinite(sampled_normals).all():
        raise ValueError("inverse-transpose normal transformation produced non-finite values")

    normalized_xyz = (sampled_points - center[None, :]) / radius
    points = np.concatenate([normalized_xyz, sampled_normals], axis=1).astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("normalized surface points contain non-finite values")
    return LocalSurfaceSample(
        value.unit_id,
        points,
        _size_ratios(extents),
        aabb_min.astype(np.float32),
        aabb_max.astype(np.float32),
        radius,
        world_area,
        False,
    )


def _size_ratios(extents: np.ndarray) -> np.ndarray:
    maximum = float(np.max(extents)) if extents.size else 0.0
    if maximum <= 1e-12:
        return np.zeros((3,), dtype=np.float32)
    return (extents / maximum).astype(np.float32)


def _surface_manifest(
    units: list[SurfaceUnitInput],
    *,
    sampling_seed: int,
    degenerate_unit_ids: list[int],
    files: Mapping[str, str | None],
) -> dict[str, Any]:
    unit_ids = np.asarray([unit.unit_id for unit in units], dtype=np.uint64)
    manifest = {
        "schema": LOCAL_SURFACE_SCHEMA,
        "version": 1,
        "pointsPerUnit": POINTS_PER_UNIT,
        "pointFeatureDim": 6,
        "pointFeatureOrder": "normalized_xyz_unit_normal",
        "sizeRatioDim": 3,
        "storage": "dense_fp32_header16",
        "numUnits": len(units),
        "unitIds": [int(value) for value in unit_ids],
        "samplingSeed": int(sampling_seed),
        "transformPolicy": "world_triangle_area_inverse_transpose_normal",
        "degenerateUnitIds": [int(value) for value in degenerate_unit_ids],
        "files": dict(files),
    }
    validate_local_surface_manifest(manifest)
    return manifest


def build_local_surface_asset(
    units: Iterable[SurfaceUnitInput | Mapping[str, Any]],
    *,
    sampling_seed: int,
) -> LocalSurfaceAsset:
    """Build one fixed-order local-surface asset from transformed unit meshes."""

    coerced = sorted((_coerce_unit(unit) for unit in units), key=lambda item: item.unit_id)
    if not coerced:
        raise ValueError("at least one surface unit is required")
    unit_ids = [unit.unit_id for unit in coerced]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("surface unit IDs must be unique")
    samples = [
        sample_unit_surface(
            unit,
            seed=int(np.random.SeedSequence([int(sampling_seed), unit.unit_id]).generate_state(1)[0]),
        )
        for unit in coerced
    ]
    degenerate = [sample.unit_id for sample in samples if sample.degenerate]
    # Keep the formal dense shape even when a unit has no usable area.  Its
    # row is an explicit zero placeholder and the manifest records the unit so
    # training/export code can exclude or handle it deliberately.
    points = np.zeros((len(samples), POINTS_PER_UNIT, 6), dtype=np.float32)
    for index, sample in enumerate(samples):
        if not sample.degenerate:
            points[index] = sample.points
    size_ratios = np.stack([sample.size_ratios for sample in samples], axis=0).astype(np.float32, copy=False)
    aabb_min = np.stack([sample.aabb_min for sample in samples], axis=0).astype(np.float32, copy=False)
    aabb_max = np.stack([sample.aabb_max for sample in samples], axis=0).astype(np.float32, copy=False)
    files = {"points": "surface_points_fp32.bin",
             "sizeRatios": "size_ratios_fp32.bin",
             "aabbMin": "aabb_min_fp32.npy",
             "aabbMax": "aabb_max_fp32.npy",
             "unitIds": "unit_ids_uint64.npy"}
    manifest = _surface_manifest(
        coerced,
        sampling_seed=int(sampling_seed),
        degenerate_unit_ids=degenerate,
        files=files,
    )
    return LocalSurfaceAsset(
        np.asarray(unit_ids, dtype=np.uint64),
        points,
        size_ratios,
        aabb_min,
        aabb_max,
        tuple(degenerate),
        manifest,
    )


def _write_dense_fp32(path: Path, values: np.ndarray, *, feature_dim: int, items_per_unit: int) -> None:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != feature_dim or array.shape[1] != items_per_unit:
        raise ValueError(
            f"dense FP32 asset shape must be [N,{items_per_unit},{feature_dim}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("dense FP32 asset contains non-finite values")
    header = SURFACE_BINARY_HEADER.pack(
        SURFACE_BINARY_MAGIC,
        SURFACE_BINARY_VERSION,
        int(feature_dim),
        int(array.shape[0]),
        int(items_per_unit),
    )
    path.write_bytes(header + np.ascontiguousarray(array).tobytes(order="C"))


def _read_dense_fp32(
    path: Path, *, expected_units: int, feature_dim: int, items_per_unit: int
) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < SURFACE_BINARY_HEADER.size:
        raise SchemaError(f"dense FP32 asset is missing its 16-byte header: {path}")
    magic, version, stored_dim, stored_units, stored_items = SURFACE_BINARY_HEADER.unpack_from(payload)
    if magic != SURFACE_BINARY_MAGIC or version != SURFACE_BINARY_VERSION:
        raise SchemaError(f"unsupported dense FP32 asset header: {path}")
    if (stored_dim, stored_units, stored_items) != (feature_dim, expected_units, items_per_unit):
        raise SchemaError(
            f"dense FP32 header mismatch: got {(stored_dim, stored_units, stored_items)}, "
            f"expected {(feature_dim, expected_units, items_per_unit)}"
        )
    expected_bytes = SURFACE_BINARY_HEADER.size + expected_units * items_per_unit * feature_dim * 4
    if len(payload) != expected_bytes:
        raise SchemaError(f"dense FP32 payload length mismatch: {path}")
    values = np.frombuffer(payload, dtype="<f4", offset=SURFACE_BINARY_HEADER.size)
    return values.reshape(expected_units, items_per_unit, feature_dim).astype(np.float32, copy=True)


def write_local_surface_asset(asset: LocalSurfaceAsset, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = validate_local_surface_manifest(asset.manifest)
    expected_shape = (manifest["numUnits"], POINTS_PER_UNIT, 6)
    if asset.points is None or asset.points.shape != expected_shape:
        raise ValueError(f"points must have shape {expected_shape}")
    _write_dense_fp32(
        output / manifest["files"]["points"],
        np.asarray(asset.points, dtype=np.float32),
        feature_dim=6,
        items_per_unit=POINTS_PER_UNIT,
    )
    size_ratios = np.asarray(asset.size_ratios, dtype=np.float32)
    if size_ratios.shape != (manifest["numUnits"], 3):
        raise ValueError("size ratios must have shape [numUnits,3]")
    _write_dense_fp32(
        output / manifest["files"]["sizeRatios"],
        size_ratios[:, None, :],
        feature_dim=3,
        items_per_unit=1,
    )
    np.save(output / manifest["files"]["aabbMin"], np.asarray(asset.aabb_min, dtype=np.float32))
    np.save(output / manifest["files"]["aabbMax"], np.asarray(asset.aabb_max, dtype=np.float32))
    np.save(output / manifest["files"]["unitIds"], np.asarray(asset.unit_ids, dtype=np.uint64))
    write_json_manifest(output / "surface_manifest.json", manifest)
    return output / "surface_manifest.json"


def load_local_surface_asset(output_dir: str | Path) -> LocalSurfaceAsset:
    output = Path(output_dir)
    manifest = validate_local_surface_manifest(read_json_manifest(output / "surface_manifest.json"))
    unit_ids = np.load(output / manifest["files"]["unitIds"], allow_pickle=False)
    size_ratio_payload = _read_dense_fp32(
        output / manifest["files"]["sizeRatios"],
        expected_units=manifest["numUnits"],
        feature_dim=3,
        items_per_unit=1,
    )
    size_ratios = size_ratio_payload[:, 0, :]
    aabb_min = np.load(output / manifest["files"]["aabbMin"], allow_pickle=False)
    aabb_max = np.load(output / manifest["files"]["aabbMax"], allow_pickle=False)
    if unit_ids.shape != (manifest["numUnits"],):
        raise SchemaError("stored local-surface unit IDs have the wrong shape")
    if size_ratios.shape != (manifest["numUnits"], 3):
        raise SchemaError("stored local-surface size ratios have the wrong shape")
    if aabb_min.shape != (manifest["numUnits"], 3) or aabb_max.shape != (manifest["numUnits"], 3):
        raise SchemaError("stored local-surface AABB arrays have the wrong shape")
    points = _read_dense_fp32(
        output / manifest["files"]["points"],
        expected_units=manifest["numUnits"],
        feature_dim=6,
        items_per_unit=POINTS_PER_UNIT,
    )
    return LocalSurfaceAsset(
        unit_ids.astype(np.uint64, copy=False),
        points.astype(np.float32, copy=False),
        size_ratios.astype(np.float32, copy=False),
        aabb_min.astype(np.float32, copy=False),
        aabb_max.astype(np.float32, copy=False),
        tuple(int(item) for item in manifest["degenerateUnitIds"]),
        manifest,
    )
