"""Scene-balanced pose batches and compiled V5 geometry assets."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np
import torch

from neural_instance_culling.dataset.v5.build_external_hit_probes import (
    ExternalHitProbeTable,
    PROBE_OBSERVATIONS_PER_UNIT,
    load_probe_table,
    sample_grouped_current_status_observations,
)
from neural_instance_culling.dataset.v5.permissions import FoldAccessPolicy
from neural_instance_culling.dataset.v5.schemas import validate_relation_manifest
from neural_instance_culling.model.common.runtime_meta import load_runtime_meta
from neural_instance_culling.model.pose_csr_dataset import PoseCSRDataset


SURFACE_HEADER = struct.Struct("<4sHHII")
SURFACE_MAGIC = b"GPV5"


@dataclass(frozen=True)
class SceneRiskDenominators:
    pose_count: int
    eligible_pose_count: int
    candidate_occurrences: int
    visible_occurrences: int
    visible_weight_sum: float


@dataclass(frozen=True)
class PoseBatch:
    pose_ids: np.ndarray
    candidate_ids: np.ndarray
    pose_rows: np.ndarray
    targets: np.ndarray
    visible_weights: np.ndarray
    query_geometry: np.ndarray
    support_points: np.ndarray
    target_centers: np.ndarray
    target_radii: np.ndarray
    denominators: SceneRiskDenominators


@dataclass(frozen=True)
class ProbeBatch:
    unit_ids: np.ndarray
    directions: np.ndarray
    distances: np.ndarray
    events: np.ndarray


def _dense_fp32_memmap(
    path: Path,
    *,
    feature_dim: int,
    items_per_unit: int,
) -> np.memmap:
    with path.open("rb") as stream:
        header = stream.read(SURFACE_HEADER.size)
    if len(header) != SURFACE_HEADER.size:
        raise ValueError(f"missing V5 dense header: {path}")
    magic, version, stored_dim, units, stored_items = SURFACE_HEADER.unpack(header)
    if magic != SURFACE_MAGIC or version != 1:
        raise ValueError(f"unsupported V5 dense asset: {path}")
    if stored_dim != feature_dim or stored_items != items_per_unit:
        raise ValueError(f"V5 dense asset shape mismatch: {path}")
    expected = SURFACE_HEADER.size + int(units) * items_per_unit * feature_dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"V5 dense asset byte length mismatch: {path}")
    return np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=SURFACE_HEADER.size,
        shape=(int(units), items_per_unit, feature_dim),
    )


def _camera_basis(forward: np.ndarray) -> np.ndarray:
    f = np.asarray(forward, dtype=np.float64)
    f /= max(float(np.linalg.norm(f)), 1e-12)
    up_seed = np.asarray([0.0, 0.0, 1.0] if abs(float(f[1])) > 0.98 else [0.0, 1.0, 0.0])
    right = np.cross(f, up_seed)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, f)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    return np.stack([right, up, f], axis=0).astype(np.float32)


def _support_points(
    center: np.ndarray,
    basis: np.ndarray,
    shape: str,
    half_extent: np.ndarray,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half_extent, dtype=np.float64)
    if shape == "horizontal_disk":
        angles = np.arange(8, dtype=np.float64) * (2.0 * math.pi / 8.0)
        circle = np.column_stack(
            [np.cos(angles) * half[0], np.zeros((8,)), np.sin(angles) * half[0]]
        )
        return np.concatenate([center[None, :], center[None, :] + circle], axis=0).astype(np.float32)
    if shape != "camera_aligned_box":
        raise ValueError(f"unsupported V5 view-cell shape: {shape}")
    # Registry order is half-right, half-forward, half-up.
    axes = np.stack(
        [basis[0] * half[0], basis[2] * half[1], basis[1] * half[2]], axis=0
    )
    signs = np.asarray(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    corners = center[None, :] + signs @ axes
    return np.concatenate([center[None, :], corners], axis=0).astype(np.float32)


def _query_geometry_numpy(
    target_centers: np.ndarray,
    target_radii: np.ndarray,
    region_center: np.ndarray,
    basis: np.ndarray,
    half_axes: np.ndarray,
    tan_fov: np.ndarray,
    region_type: float,
    near: float,
    far: float,
) -> np.ndarray:
    offset = np.asarray(region_center, dtype=np.float64)[None, :] - target_centers
    distance = np.linalg.norm(offset, axis=1)
    world = np.divide(
        offset,
        distance[:, None],
        out=np.zeros_like(offset),
        where=distance[:, None] > 1e-12,
    )
    reverse_camera = (-world) @ basis.T
    radius = target_radii.astype(np.float64)
    denominator = distance + radius
    axis_denominator = denominator + float(np.max(half_axes))
    return np.column_stack(
        [
            world,
            reverse_camera,
            np.log1p(distance / radius),
            radius / denominator,
            np.broadcast_to(half_axes[None, :], (target_centers.shape[0], 3))
            / axis_denominator[:, None],
            np.broadcast_to(tan_fov[None, :], (target_centers.shape[0], 2)),
            np.full((target_centers.shape[0],), region_type),
            np.full((target_centers.shape[0],), near) / denominator,
            np.log1p(float(far) / denominator),
        ]
    ).astype(np.float32)


class V5SceneTrainingData:
    """Read-only V5 assets for one real or synthetic scene."""

    def __init__(
        self,
        *,
        scene_id: str,
        pose_dataset: Path,
        runtime_meta: Path,
        compiled_dir: Path,
        viewcell_shape: str,
        viewcell_half_extent_m: tuple[float, float, float],
        camera_clip_m: tuple[float, float],
        probe_policy: FoldAccessPolicy | Mapping[str, Any] | None = None,
        region_manifest: Path | None = None,
    ) -> None:
        self.scene_id = str(scene_id)
        self.pose_dataset_path = Path(pose_dataset)
        self.compiled_dir = Path(compiled_dir)
        world_aabbs, _instance_to_glb, _meta = load_runtime_meta(runtime_meta)
        self.world_aabbs = world_aabbs
        self.num_units = int(world_aabbs.shape[0])
        self.dataset = PoseCSRDataset(self.pose_dataset_path, self.num_units)
        self.train_split = self.dataset.split("train")
        self.viewcell_shape = str(viewcell_shape)
        self.viewcell_half_extent_m = np.asarray(viewcell_half_extent_m, dtype=np.float32)
        self.camera_clip_m = tuple(float(value) for value in camera_clip_m)
        self.region_specs: tuple[dict[str, Any], ...] | None = None
        if region_manifest is not None:
            manifest = json.loads(Path(region_manifest).read_text(encoding="utf-8"))
            regions = manifest.get("viewCells", {}).get("regions")
            if not isinstance(regions, list) or len(regions) != self.dataset.poses.size:
                raise ValueError(f"{self.scene_id} region manifest does not match pose count")
            parsed: list[dict[str, Any]] = []
            for pose_id, region in enumerate(regions):
                if not isinstance(region, Mapping):
                    raise ValueError(f"{self.scene_id} region {pose_id} is not an object")
                region_type = str(region.get("regionType", ""))
                support = np.asarray(region.get("supportPoints"), dtype=np.float32)
                if support.shape != (9, 3) or not bool(np.isfinite(support).all()):
                    raise ValueError(f"{self.scene_id} region {pose_id} has invalid support points")
                if region_type == "disk":
                    radius = float(region.get("radius", 0.0))
                    if not math.isfinite(radius) or radius <= 0.0:
                        raise ValueError(f"{self.scene_id} region {pose_id} has invalid disk radius")
                    half_axes = np.asarray([radius, 0.0, radius], dtype=np.float32)
                    encoded_type = 0.0
                elif region_type == "oriented_box":
                    axes = np.asarray(region.get("halfAxes"), dtype=np.float32)
                    if axes.shape != (3, 3) or not bool(np.isfinite(axes).all()):
                        raise ValueError(f"{self.scene_id} region {pose_id} has invalid box axes")
                    half_axes = np.linalg.norm(axes, axis=1).astype(np.float32)
                    if bool((half_axes <= 0.0).any()):
                        raise ValueError(f"{self.scene_id} region {pose_id} has degenerate box axes")
                    encoded_type = 1.0
                else:
                    raise ValueError(f"{self.scene_id} region {pose_id} has unknown type {region_type!r}")
                parsed.append({"support": support, "half_axes": half_axes, "region_type": encoded_type})
            self.region_specs = tuple(parsed)

        surface_dir = self.compiled_dir / "surface"
        surface_manifest = json.loads((surface_dir / "surface_manifest.json").read_text(encoding="utf-8"))
        if int(surface_manifest["numUnits"]) != self.num_units:
            raise ValueError(f"{self.scene_id} surface/runtime unit count mismatch")
        files = surface_manifest["files"]
        self.surface_points = _dense_fp32_memmap(
            surface_dir / files["points"], feature_dim=6, items_per_unit=256
        )
        self.size_ratios = _dense_fp32_memmap(
            surface_dir / files["sizeRatios"], feature_dim=3, items_per_unit=1
        )[:, 0, :]
        self.degenerate_unit_ids = np.asarray(
            surface_manifest.get("degenerateUnitIds", []), dtype=np.int64
        )
        if self.degenerate_unit_ids.size and (
            bool((self.degenerate_unit_ids < 0).any())
            or bool((self.degenerate_unit_ids >= self.num_units).any())
        ):
            raise ValueError(f"{self.scene_id} degenerate unit IDs are outside the scene")
        self.valid_unit_mask = np.ones((self.num_units,), dtype=bool)
        self.valid_unit_mask[self.degenerate_unit_ids] = False

        relation_dir = self.compiled_dir / "relation"
        relation_manifest = validate_relation_manifest(
            json.loads((relation_dir / "relation_manifest.json").read_text(encoding="utf-8"))
        )
        if int(relation_manifest["numUnits"]) != self.num_units:
            raise ValueError(f"{self.scene_id} relation/runtime unit count mismatch")
        self.relation_source_ids = np.load(
            relation_dir / "source_ids_int64.npy", mmap_mode="r", allow_pickle=False
        )
        self.relation_valid = np.load(
            relation_dir / "valid_mask_bool.npy", mmap_mode="r", allow_pickle=False
        )
        self.relation_features = np.load(
            relation_dir / "edge_features_fp32.npy", mmap_mode="r", allow_pickle=False
        )
        self.relation_metadata = relation_manifest
        self.probe_table: ExternalHitProbeTable | None = None
        probe_dir = self.compiled_dir / "probes"
        if probe_policy is not None and (probe_dir / "external_hit_probe_manifest.json").is_file():
            self.probe_table = load_probe_table(probe_dir, policy=probe_policy)
        self.denominators = self._compute_train_denominators()

    def _compute_train_denominators(self) -> SceneRiskDenominators:
        occurrences = 0
        candidate_occurrences = 0
        weight_sum = 0.0
        indices = np.asarray(self.train_split.pose_indices, dtype=np.int64)
        eligible = indices[self.dataset.candidate_counts[indices] > 0]
        for pose_id in indices.tolist():
            candidate_ids = np.asarray(self.dataset.candidate_slice(int(pose_id)), dtype=np.int64)
            candidate_occurrences += int(np.count_nonzero(self.valid_unit_mask[candidate_ids]))
            visible_ids, weights = self.dataset.visible_slice(int(pose_id))
            visible_ids = np.asarray(visible_ids, dtype=np.int64)
            if self.degenerate_unit_ids.size and visible_ids.size and bool(
                np.isin(visible_ids, self.degenerate_unit_ids).any()
            ):
                raise ValueError(
                    f"{self.scene_id} train pose {pose_id} contains visible degenerate units"
                )
            occurrences += int(weights.size)
            weight_sum += float(np.asarray(weights, dtype=np.float64).sum())
        if occurrences <= 0 or weight_sum <= 0:
            raise ValueError(f"{self.scene_id} train split has no positive supervision")
        if candidate_occurrences < occurrences:
            raise ValueError(f"{self.scene_id} train candidates contain fewer rows than visible GT")
        if eligible.size <= 0:
            raise ValueError(f"{self.scene_id} train split has no candidate-nonempty poses")
        return SceneRiskDenominators(
            int(indices.size), int(eligible.size), candidate_occurrences, occurrences, weight_sum
        )

    def sample_pose_batch(self, rng: np.random.Generator, pose_count: int = 4) -> PoseBatch:
        eligible = np.asarray(self.train_split.pose_indices, dtype=np.int64)
        eligible = eligible[self.dataset.candidate_counts[eligible] > 0]
        if eligible.size == 0:
            raise ValueError(f"{self.scene_id} has no candidate-nonempty train poses")
        replace = eligible.size < pose_count
        selected = rng.choice(eligible, size=pose_count, replace=replace).astype(np.int64)
        all_candidates: list[np.ndarray] = []
        all_pose_rows: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_weights: list[np.ndarray] = []
        all_query: list[np.ndarray] = []
        all_supports: list[np.ndarray] = []
        all_centers: list[np.ndarray] = []
        all_radii: list[np.ndarray] = []
        for pose_row, pose_id in enumerate(selected.tolist()):
            candidates = np.asarray(self.dataset.candidate_slice(pose_id), dtype=np.int64)
            candidates = candidates[self.valid_unit_mask[candidates]]
            visible_ids, visible_weights = self.dataset.visible_slice(pose_id)
            visible_ids = np.asarray(visible_ids, dtype=np.int64)
            visible_weights = np.asarray(visible_weights, dtype=np.float32)
            locations = np.searchsorted(visible_ids, candidates)
            matched = (locations < visible_ids.size)
            matched[matched] &= visible_ids[locations[matched]] == candidates[matched]
            targets = matched.astype(np.float32)
            weights = np.zeros(candidates.size, dtype=np.float32)
            weights[matched] = visible_weights[locations[matched]]

            bounds = self.world_aabbs[candidates]
            centers = (bounds[:, :3] + bounds[:, 3:]) * 0.5
            radii = np.linalg.norm(bounds[:, 3:] - bounds[:, :3], axis=1) * 0.5
            if np.any(radii <= 1e-8):
                raise ValueError(f"{self.scene_id} pose {pose_id} contains a degenerate candidate AABB")
            region_center = self.dataset.query_center_world(pose_id, required=True)
            view = self.dataset.camera_view(pose_id)
            basis = _camera_basis(view[:3])
            if self.region_specs is None:
                half_axes = self.viewcell_half_extent_m.copy()
                if self.viewcell_shape == "horizontal_disk":
                    half_axes = np.asarray([half_axes[0], 0.0, half_axes[1]], dtype=np.float32)
                region_type = 0.0 if self.viewcell_shape == "horizontal_disk" else 1.0
                support = _support_points(
                    region_center,
                    basis,
                    self.viewcell_shape,
                    self.viewcell_half_extent_m,
                )
            else:
                spec = self.region_specs[int(pose_id)]
                half_axes = np.asarray(spec["half_axes"], dtype=np.float32)
                region_type = float(spec["region_type"])
                support = np.asarray(spec["support"], dtype=np.float32)
            query = _query_geometry_numpy(
                centers,
                radii,
                region_center,
                basis,
                half_axes,
                view[3:5],
                region_type,
                self.camera_clip_m[0],
                self.camera_clip_m[1],
            )
            all_candidates.append(candidates)
            all_pose_rows.append(np.full(candidates.size, pose_row, dtype=np.int64))
            all_targets.append(targets)
            all_weights.append(weights)
            all_query.append(query)
            all_supports.append(np.broadcast_to(support[None, :, :], (candidates.size, 9, 3)).copy())
            all_centers.append(centers.astype(np.float32))
            all_radii.append(radii.astype(np.float32))
        return PoseBatch(
            pose_ids=selected,
            candidate_ids=np.concatenate(all_candidates),
            pose_rows=np.concatenate(all_pose_rows),
            targets=np.concatenate(all_targets),
            visible_weights=np.concatenate(all_weights),
            query_geometry=np.concatenate(all_query),
            support_points=np.concatenate(all_supports),
            target_centers=np.concatenate(all_centers),
            target_radii=np.concatenate(all_radii),
            denominators=self.denominators,
        )

    def relation_slice(self, target_ids: np.ndarray) -> dict[str, Any]:
        ids = np.asarray(target_ids, dtype=np.int64)
        return {
            "source_ids": np.asarray(self.relation_source_ids[ids]),
            "valid_mask": np.asarray(self.relation_valid[ids]),
            "edge_features": np.asarray(self.relation_features[ids]),
            "metadata": self.relation_metadata,
        }

    def geometry_dependency_ids(self, target_ids: np.ndarray) -> np.ndarray:
        ids = np.unique(np.asarray(target_ids, dtype=np.int64))
        sources = np.asarray(self.relation_source_ids[ids])
        valid = np.asarray(self.relation_valid[ids])
        return np.union1d(ids, sources[valid].astype(np.int64, copy=False))

    def sample_probe_batch(
        self,
        rng: np.random.Generator,
        observation_count: int = 8192,
    ) -> ProbeBatch:
        if self.probe_table is None:
            raise ValueError(f"{self.scene_id} has no authorized external-hit probe table")
        if observation_count <= 0 or observation_count % PROBE_OBSERVATIONS_PER_UNIT:
            raise ValueError(
                f"probe observation count must be a positive multiple of "
                f"{PROBE_OBSERVATIONS_PER_UNIT}"
            )
        values = sample_grouped_current_status_observations(
            self.probe_table,
            rng,
            unit_count=observation_count // PROBE_OBSERVATIONS_PER_UNIT,
            observations_per_unit=PROBE_OBSERVATIONS_PER_UNIT,
        )
        if not bool(self.valid_unit_mask[values["unit_ids"]].all()):
            raise ValueError(
                f"{self.scene_id} probe table contains a degenerate or excluded unit"
            )
        return ProbeBatch(
            unit_ids=values["unit_ids"],
            directions=values["directions"],
            distances=values["distances"],
            events=values["events"],
        )

    def geometry_tensors(self, unit_ids: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ids = np.asarray(unit_ids, dtype=np.int64)
        points = np.asarray(self.surface_points[ids], dtype=np.float32).copy()
        ratios = np.asarray(self.size_ratios[ids], dtype=np.float32).copy()
        return torch.from_numpy(points).to(device), torch.from_numpy(ratios).to(device)


__all__ = [
    "PoseBatch",
    "ProbeBatch",
    "SceneRiskDenominators",
    "V5SceneTrainingData",
]
