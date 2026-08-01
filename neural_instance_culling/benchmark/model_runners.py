from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from common.threshold_selection import (  # noqa: E402
    select_weighted_precision_workpoint,
    target_weighted_recall_from_payload,
)
from directional_occlusion_proxy_encoder_model import DirectionalOcclusionProxyEncoderPVSModel  # noqa: E402
from pose_csr_dataset import PoseCSRDataset, _project_aabb_features_numpy  # noqa: E402
from aabb_ray_feature_utils import FEATURE_DIM as AABB_RAY_FEATURE_DIM, build_aabb_ray_features  # noqa: E402
from train_aabb_ray_baseline import AabbRayMLP  # noqa: E402


DEFAULT_MODEL_SPECS: dict[str, dict[str, str]] = {
    "baseline_keep_all": {
        "kind": "keep_all",
    },
    "baseline_static_frequency_train": {
        "kind": "static_frequency_train",
    },
    "baseline_camera_distance": {
        "kind": "camera_distance",
    },
    "baseline_projected_aabb_area": {
        "kind": "projected_aabb_area",
    },
    "baseline_aabb_ray": {
        "kind": "aabb_ray",
    },
    "baseline_viewcell_bitset_train": {
        "kind": "viewcell_bitset_train",
    },
    "baseline_aabb_hzb": {
        "kind": "aabb_hzb",
        "display_name": "baseline_aabb_depth_proxy",
        "legacy_name": "baseline_aabb_hzb",
    },
    "pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best": {
        "kind": "directional_occlusion_proxy_encoder",
        "checkpoint": str(ROOT / "model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/best.pt"),
        "runtime_features": str(ROOT / "model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/instance_runtime_features_fp16.bin"),
        "eval_summary": str(ROOT / "model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/eval_summary.json"),
    },
    "pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best": {
        "kind": "directional_occlusion_proxy_encoder",
        "checkpoint": str(ROOT / "model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66/best.pt"),
        "runtime_features": str(ROOT / "model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66/instance_runtime_features_fp16.bin"),
        "eval_summary": str(ROOT / "model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66/eval_summary.json"),
    },
    "pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best": {
        "kind": "directional_occlusion_proxy_encoder",
        "checkpoint": str(ROOT / "model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40/best.pt"),
        "runtime_features": str(ROOT / "model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40/instance_runtime_features_fp16.bin"),
        "eval_summary": str(ROOT / "model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40/eval_summary.json"),
    },
}

MODEL_ALIASES: dict[str, str] = {
    "pvs_directional_occlusion_proxy_encoder_full40_best": "pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best",
}


@dataclass
class PredictionResult:
    scores: np.ndarray
    forward_ms: float
    total_ms: float
    diagnostics: dict | None = None
    utility_scores: np.ndarray | None = None
    download_scores: np.ndarray | None = None


class BaseModelRunner:
    """Shared benchmark interface for PVS methods and visibility baselines.

    A runner receives the candidate rows already assembled by the CSR loader.
    It must return one score per input row, in exactly the same order.  Runner
    implementations must not add GT positives or rebuild the candidate set.
    """

    def __init__(
        self,
        name: str,
        kind: str,
        model,
        world_aabbs: np.ndarray,
        instance_to_glb: np.ndarray,
        runtime_meta: dict,
        checkpoint: dict,
        threshold: float,
        device: torch.device,
    ):
        self.name = name
        self.kind = kind
        self.display_name = str(
            (checkpoint or {}).get("displayName")
            or (checkpoint or {}).get("display_name")
            or ("baseline_aabb_depth_proxy" if kind == "aabb_hzb" else name)
        )
        self.model = model
        self.world_aabbs = world_aabbs
        self.instance_to_glb = instance_to_glb
        self.runtime_meta = runtime_meta
        self.checkpoint = checkpoint
        self.threshold = float(threshold)
        self.device = device
        self.camera_bounds = checkpoint.get("cameraBounds") or runtime_meta.get("cameraBounds") or runtime_meta["sceneBounds"]
        self.requires_mvp = False
        self.information_level = "L0_metadata_cold_start"
        self.decision_mode = "threshold"

    def score_arrays(
        self,
        camera_norm: np.ndarray,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        instance_ids: np.ndarray,
        mvp: np.ndarray | None = None,
    ) -> PredictionResult:
        raise NotImplementedError

    def score_batch(self, batch: dict[str, np.ndarray]) -> PredictionResult:
        return self.score_arrays(batch["camera"], batch["camera_world"], batch["camera_view"], batch["instance"], batch.get("mvp"))

    def predict_ids(
        self,
        camera_norm: np.ndarray,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        candidate_ids: np.ndarray,
        mvp: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, PredictionResult]:
        count = int(candidate_ids.size)
        camera = np.repeat(camera_norm[None, :], count, axis=0).astype(np.float32, copy=False)
        world = np.repeat(camera_world[None, :], count, axis=0).astype(np.float32, copy=False)
        view = np.repeat(camera_view[None, :], count, axis=0).astype(np.float32, copy=False)
        mvp_rows = None
        if mvp is not None:
            mvp_rows = np.repeat(np.asarray(mvp, dtype=np.float32)[None, :], count, axis=0)
        result = self.score_arrays(camera, world, view, candidate_ids.astype(np.int64, copy=False), mvp_rows)
        pred = candidate_ids[result.scores >= float(self.threshold if threshold is None else threshold)]
        return pred.astype(np.uint32, copy=False), result


class StaticRuleRunner(BaseModelRunner):
    """Base class for deterministic candidate-preserving visibility rules."""

    def _validate_instance_ids(self, instance_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.world_aabbs.shape[0]):
            raise IndexError(
                f"{self.name} received instance ids outside [0, {self.world_aabbs.shape[0]})."
            )
        return ids

    def _score_rows(
        self,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        instance_ids: np.ndarray,
        mvp: np.ndarray | None,
    ) -> np.ndarray:
        raise NotImplementedError

    def score_arrays(
        self,
        camera_norm: np.ndarray,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        instance_ids: np.ndarray,
        mvp: np.ndarray | None = None,
    ) -> PredictionResult:
        del camera_norm
        t0 = time.perf_counter()
        ids = self._validate_instance_ids(instance_ids)
        scores = np.asarray(
            self._score_rows(
                np.asarray(camera_world, dtype=np.float32),
                np.asarray(camera_view, dtype=np.float32),
                ids,
                None if mvp is None else np.asarray(mvp, dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        if scores.shape != ids.shape:
            raise ValueError(f"{self.name} returned {scores.size} scores for {ids.size} candidates.")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{self.name} returned non-finite visibility scores.")
        t1 = time.perf_counter()
        return PredictionResult(scores=scores, forward_ms=(t1 - t0) * 1000.0, total_ms=(t1 - t0) * 1000.0)


def _normalize_directions(values: np.ndarray) -> np.ndarray:
    directions = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normalized = directions / np.maximum(norms, 1e-8)
    fallback = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return np.where(norms > 1e-8, normalized, fallback[None, :]).astype(np.float32, copy=False)


class KeepAllRunner(StaticRuleRunner):
    """Retain every supplied back-camera candidate.

    This is a fixed safety baseline.  It does not use GT, a threshold learned
    from evaluation data, or target geometry beyond the candidate list.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_mode = "fixed_keep_all"

    def _score_rows(self, camera_world, camera_view, instance_ids, mvp):
        del camera_world, camera_view, mvp
        return np.ones((instance_ids.size,), dtype=np.float32)


class StaticFrequencyRunner(StaticRuleRunner):
    """Use a per-instance visibility frequency computed from train CSR only."""

    def __init__(self, *args, train_visibility_frequency: np.ndarray, train_pose_count: int, **kwargs):
        super().__init__(*args, **kwargs)
        frequency = np.asarray(train_visibility_frequency, dtype=np.float32).reshape(-1)
        if frequency.size != self.world_aabbs.shape[0]:
            raise ValueError(
                f"Train frequency has {frequency.size} rows, expected {self.world_aabbs.shape[0]}."
            )
        if not np.all(np.isfinite(frequency)) or np.any(frequency < 0.0) or np.any(frequency > 1.0):
            raise ValueError("Train visibility frequency must be finite and lie in [0, 1].")
        self.train_visibility_frequency = frequency
        self.train_pose_count = int(train_pose_count)

    def _score_rows(self, camera_world, camera_view, instance_ids, mvp):
        del camera_world, camera_view, mvp
        return self.train_visibility_frequency[instance_ids]


class CameraDistanceRunner(StaticRuleRunner):
    """Rank candidates by inverse distance from the camera to their AABB."""

    def _score_rows(self, camera_world, camera_view, instance_ids, mvp):
        del camera_view, mvp
        aabbs = self.world_aabbs[instance_ids]
        camera = np.asarray(camera_world, dtype=np.float32).reshape(-1, 3)
        mins = aabbs[:, :3]
        maxs = aabbs[:, 3:]
        delta = np.maximum(np.maximum(mins - camera, camera - maxs), 0.0)
        distance = np.linalg.norm(delta, axis=1)
        return (1.0 / (1.0 + distance)).astype(np.float32, copy=False)


class ProjectedAabbAreaRunner(StaticRuleRunner):
    """Rank candidates by their projected AABB area in the supplied MVP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_mvp = True

    def _scores_for_pose(self, instance_ids: np.ndarray, mvp: np.ndarray) -> np.ndarray:
        rect, area, _depth, valid = _project_aabb_features_numpy(self.world_aabbs[instance_ids], mvp)
        del rect
        scores = np.where(valid, np.clip(area, 0.0, 1.0), 0.0)
        return scores.astype(np.float32, copy=False)

    def _score_rows(self, camera_world, camera_view, instance_ids, mvp):
        del camera_world, camera_view
        if mvp is None:
            raise RuntimeError("projected_aabb_area runner requires mvp rows in the CSR dataset.")
        mvp_rows = np.asarray(mvp, dtype=np.float32)
        if mvp_rows.ndim == 1:
            return self._scores_for_pose(instance_ids, mvp_rows)
        if mvp_rows.shape[0] != instance_ids.size or mvp_rows.shape[1] != 16:
            raise ValueError("projected_aabb_area mvp rows must have shape [candidate_count, 16].")
        # score_arrays is normally called for one pose.  Supporting distinct
        # MVP rows here keeps the runner's row-alignment contract explicit.
        return np.asarray(
            [self._scores_for_pose(np.asarray([instance_id], dtype=np.int64), row)[0] for instance_id, row in zip(instance_ids, mvp_rows)],
            dtype=np.float32,
        )

    def score_batch(self, batch: dict[str, np.ndarray]) -> PredictionResult:
        if "mvp" not in batch:
            raise RuntimeError("projected_aabb_area runner requires mvp.bin in the CSR dataset.")
        t0 = time.perf_counter()
        ids_all = batch["instance"].astype(np.int64, copy=False)
        offsets = batch["pose_offsets"]
        scores = np.zeros((ids_all.size,), dtype=np.float32)
        for pose_id in range(offsets.size - 1):
            start = int(offsets[pose_id])
            end = int(offsets[pose_id + 1])
            if end <= start:
                continue
            scores[start:end] = self._scores_for_pose(ids_all[start:end], batch["mvp"][start])
        t1 = time.perf_counter()
        return PredictionResult(scores=scores, forward_ms=(t1 - t0) * 1000.0, total_ms=(t1 - t0) * 1000.0)


class AabbRayRunner(StaticRuleRunner):
    """Deterministic AABB-plus-ray visibility ranking baseline.

    This is deliberately a metadata-only baseline.  For each candidate it uses
    the current camera position and forward ray, the per-pose FOV tangents, the
    instance AABB, and the supplied MVP.  The score is the product of:

    * a smooth conservative angular proximity of the camera ray to the AABB;
    * the square root of the AABB's clipped MVP screen area; and
    * a scene-diagonal-normalized near-range factor.

    It does not inspect GLB files or triangles, build a depth buffer, use GT,
    or alter the supplied candidate rows.  It is therefore an L0 geometric
    baseline, not an HZB or a learned AABB+ray model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_mvp = True
        self.information_level = "L0_metadata_cold_start"
        self.resource_assumption = "instance_aabb_camera_ray_mvp_no_glb_geometry"
        self.score_formula = "ray_angular_proximity * sqrt(mvp_aabb_area) * near_range_factor"
        _scene_min, _scene_max, scene_size = scene_min_max(self.runtime_meta["sceneBounds"])
        self.scene_diagonal = float(np.linalg.norm(scene_size))
        if not np.isfinite(self.scene_diagonal) or self.scene_diagonal <= 1e-6:
            raise ValueError("AABB-ray runner requires a finite, non-zero scene diagonal.")
        self.near_plane = 0.05

    @staticmethod
    def _camera_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = _normalize_directions(np.asarray(forward, dtype=np.float32).reshape(1, 3))[0]
        up_reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(f, up_reference))) > 0.98:
            up_reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        right = _normalize_directions(np.cross(f, up_reference).reshape(1, 3))[0]
        up = _normalize_directions(np.cross(right, f).reshape(1, 3))[0]
        return f, right, up

    def _scores_for_pose(
        self,
        instance_ids: np.ndarray,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        mvp: np.ndarray,
    ) -> np.ndarray:
        ids = self._validate_instance_ids(instance_ids)
        if ids.size == 0:
            return np.zeros((0,), dtype=np.float32)
        view = np.asarray(camera_view, dtype=np.float32).reshape(-1)
        if view.size < 5:
            raise ValueError("aabb_ray runner requires camera_view=[forward_x, forward_y, forward_z, tan_x, tan_y].")
        camera = np.asarray(camera_world, dtype=np.float32).reshape(-1)
        if camera.size != 3:
            raise ValueError("aabb_ray runner requires a 3D camera_world position.")
        mvp_values = np.asarray(mvp, dtype=np.float32).reshape(-1)
        if mvp_values.size != 16:
            raise ValueError("aabb_ray runner requires one 4x4 MVP row with 16 values.")

        forward, right, up = self._camera_basis(view[:3])
        tan_x = max(float(view[3]), 1e-4)
        tan_y = max(float(view[4]), 1e-4)
        aabbs = self.world_aabbs[ids]
        mins = aabbs[:, :3]
        maxs = aabbs[:, 3:]
        centers = (mins + maxs) * 0.5
        extents = np.maximum((maxs - mins) * 0.5, 0.0)
        relative = centers - camera[None, :]
        center_x = relative @ right
        center_y = relative @ up
        center_z = relative @ forward
        radius_x = extents @ np.abs(right)
        radius_y = extents @ np.abs(up)
        radius_z = extents @ np.abs(forward)

        # The residual is zero whenever the central camera ray falls inside
        # the conservative angular rectangle of the AABB.  Outside it, the
        # exponential tail gives a deterministic soft ranking instead of a
        # brittle binary ray/box test.
        z_far = center_z + radius_z
        angular_denominator_x = np.maximum(z_far, self.near_plane) * tan_x + radius_x
        angular_denominator_y = np.maximum(z_far, self.near_plane) * tan_y + radius_y
        residual_x = np.maximum(np.abs(center_x) - radius_x, 0.0) / np.maximum(angular_denominator_x, 1e-6)
        residual_y = np.maximum(np.abs(center_y) - radius_y, 0.0) / np.maximum(angular_denominator_y, 1e-6)
        ray_score = np.exp(-0.5 * (residual_x * residual_x + residual_y * residual_y))
        ray_score = np.where(z_far >= self.near_plane, ray_score, 0.0)

        _rect, area, _depth, valid = _project_aabb_features_numpy(aabbs, mvp_values)
        footprint_score = np.sqrt(np.clip(area, 0.0, 1.0))
        near_depth = np.maximum(center_z - radius_z, 0.0)
        near_range_score = 1.0 / (1.0 + near_depth / self.scene_diagonal)
        scores = ray_score * footprint_score * near_range_score
        scores = np.where(valid & (z_far >= self.near_plane), scores, 0.0)
        return np.asarray(scores, dtype=np.float32)

    def _score_rows(self, camera_world, camera_view, instance_ids, mvp):
        if mvp is None:
            raise RuntimeError("aabb_ray runner requires mvp rows in the CSR dataset or live raw input.")
        ids = self._validate_instance_ids(instance_ids)
        worlds = np.asarray(camera_world, dtype=np.float32).reshape(-1, 3)
        views = np.asarray(camera_view, dtype=np.float32).reshape(-1, 5)
        mvp_rows = np.asarray(mvp, dtype=np.float32)
        if worlds.shape[0] != ids.size or views.shape[0] != ids.size:
            raise ValueError("aabb_ray runner received misaligned camera and candidate rows.")
        if mvp_rows.ndim == 1:
            return self._scores_for_pose(ids, worlds[0], views[0], mvp_rows)
        if mvp_rows.ndim != 2 or mvp_rows.shape[1] != 16:
            raise ValueError("aabb_ray mvp rows must have shape [candidate_count, 16].")
        if mvp_rows.shape[0] == 1:
            return self._scores_for_pose(ids, worlds[0], views[0], mvp_rows[0])
        if mvp_rows.shape[0] != ids.size:
            raise ValueError("aabb_ray mvp rows must match the candidate row count.")
        return np.asarray(
            [
                self._scores_for_pose(
                    np.asarray([instance_id], dtype=np.int64),
                    world,
                    view,
                    mvp_row,
                )[0]
                for instance_id, world, view, mvp_row in zip(ids, worlds, views, mvp_rows)
            ],
            dtype=np.float32,
        )

    def score_arrays(
        self,
        camera_norm: np.ndarray,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        instance_ids: np.ndarray,
        mvp: np.ndarray | None = None,
    ) -> PredictionResult:
        del camera_norm
        t0 = time.perf_counter()
        ids = self._validate_instance_ids(instance_ids)
        scores = self._score_rows(camera_world, camera_view, ids, mvp)
        if scores.shape != ids.shape or not np.all(np.isfinite(scores)):
            raise ValueError("aabb_ray runner returned invalid scores.")
        t1 = time.perf_counter()
        return PredictionResult(scores=scores, forward_ms=(t1 - t0) * 1000.0, total_ms=(t1 - t0) * 1000.0)

    def score_batch(self, batch: dict[str, np.ndarray]) -> PredictionResult:
        if "mvp" not in batch:
            raise RuntimeError("aabb_ray runner requires mvp.bin in the CSR dataset.")
        t0 = time.perf_counter()
        ids_all = batch["instance"].astype(np.int64, copy=False)
        offsets = np.asarray(batch["pose_offsets"], dtype=np.int64)
        scores = np.zeros((ids_all.size,), dtype=np.float32)
        for pose_id in range(offsets.size - 1):
            start = int(offsets[pose_id])
            end = int(offsets[pose_id + 1])
            if end <= start:
                continue
            scores[start:end] = self._scores_for_pose(
                ids_all[start:end],
                batch["camera_world"][start],
                batch["camera_view"][start],
                batch["mvp"][start],
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("aabb_ray runner returned non-finite visibility scores.")
        t1 = time.perf_counter()
        return PredictionResult(scores=scores, forward_ms=(t1 - t0) * 1000.0, total_ms=(t1 - t0) * 1000.0)


class TrainViewcellBitsetRunner(StaticRuleRunner):
    """Query a train-only view-cell visibility bitset.

    The query pose is mapped to the nearest train pose using camera position
    and forward direction.  The returned membership is intersected with the
    supplied candidate rows implicitly by scoring only those rows.
    """

    def __init__(
        self,
        *args,
        train_pose_world: np.ndarray,
        train_pose_forward: np.ndarray,
        train_bitsets: np.ndarray,
        direction_penalty_m2: float = 25.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.information_level = "L1_train_viewcell_bitset"
        self.decision_mode = "fixed_train_bitset"
        self.train_pose_world = np.asarray(train_pose_world, dtype=np.float32).reshape(-1, 3)
        self.train_pose_forward = _normalize_directions(train_pose_forward)
        self.train_bitsets = np.asarray(train_bitsets, dtype=np.uint8)
        if self.train_pose_world.shape[0] == 0:
            raise ValueError("viewcell bitset runner requires at least one train pose.")
        expected_width = (self.world_aabbs.shape[0] + 7) // 8
        if self.train_pose_forward.shape != self.train_pose_world.shape:
            raise ValueError("Train pose position and direction tables must have matching shapes.")
        if self.train_bitsets.shape != (self.train_pose_world.shape[0], expected_width):
            raise ValueError(
                f"Train bitsets have shape {self.train_bitsets.shape}, expected "
                f"({self.train_pose_world.shape[0]}, {expected_width})."
            )
        self.direction_penalty_m2 = float(direction_penalty_m2)
        if not np.isfinite(self.direction_penalty_m2) or self.direction_penalty_m2 < 0.0:
            raise ValueError("direction_penalty_m2 must be finite and non-negative.")
        self.train_pose_count = int(self.train_pose_world.shape[0])
        self.bitset_bytes = int(self.train_bitsets.nbytes)

    def _nearest_train_pose(self, camera_world: np.ndarray, camera_forward: np.ndarray) -> int:
        position_delta = self.train_pose_world - np.asarray(camera_world, dtype=np.float32).reshape(1, 3)
        position_distance_sq = np.sum(position_delta * position_delta, axis=1)
        forward = _normalize_directions(np.asarray(camera_forward, dtype=np.float32).reshape(1, 3))[0]
        direction_mismatch = 1.0 - np.clip(self.train_pose_forward @ forward, -1.0, 1.0)
        metric = position_distance_sq + self.direction_penalty_m2 * direction_mismatch
        return int(np.argmin(metric))

    def _scores_for_pose(
        self,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        instance_ids: np.ndarray,
    ) -> np.ndarray:
        camera_view = np.asarray(camera_view, dtype=np.float32).reshape(-1)
        if camera_view.size < 3:
            raise ValueError("viewcell bitset runner requires a 3D camera forward direction.")
        pose_id = self._nearest_train_pose(camera_world, camera_view[:3])
        ids = self._validate_instance_ids(instance_ids)
        byte_ids = ids // 8
        bit_ids = ids & 7
        return ((self.train_bitsets[pose_id, byte_ids] >> bit_ids) & 1).astype(np.float32, copy=False)

    def _score_rows(self, camera_world, camera_view, instance_ids, mvp):
        del mvp
        camera_world = np.asarray(camera_world, dtype=np.float32).reshape(-1, 3)
        camera_view = np.asarray(camera_view, dtype=np.float32).reshape(-1, 5)
        if camera_world.shape[0] != instance_ids.size or camera_view.shape[0] != instance_ids.size:
            raise ValueError("viewcell bitset runner received misaligned camera and candidate rows.")
        if instance_ids.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if np.all(camera_world == camera_world[0]) and np.all(camera_view == camera_view[0]):
            return self._scores_for_pose(camera_world[0], camera_view[0], instance_ids)
        return np.asarray(
            [
                self._scores_for_pose(world, view, np.asarray([instance_id], dtype=np.int64))[0]
                for world, view, instance_id in zip(camera_world, camera_view, instance_ids)
            ],
            dtype=np.float32,
        )

    def score_batch(self, batch: dict[str, np.ndarray]) -> PredictionResult:
        t0 = time.perf_counter()
        ids_all = batch["instance"].astype(np.int64, copy=False)
        offsets = batch["pose_offsets"]
        scores = np.zeros((ids_all.size,), dtype=np.float32)
        for pose_id in range(offsets.size - 1):
            start = int(offsets[pose_id])
            end = int(offsets[pose_id + 1])
            if end <= start:
                continue
            scores[start:end] = self._scores_for_pose(
                batch["camera_world"][start],
                batch["camera_view"][start],
                ids_all[start:end],
            )
        t1 = time.perf_counter()
        return PredictionResult(scores=scores, forward_ms=(t1 - t0) * 1000.0, total_ms=(t1 - t0) * 1000.0)


class DirectionalOcclusionProxyEncoderRunner(BaseModelRunner):
    """Current deployed model runner.

    Runtime reads fixed instance features and queries them with ray-space camera
    features. It does not run PointNet++ or graph propagation during benchmark
    or in the frontend path.
    """

    def __init__(self, *args, runtime_features: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime_features = runtime_features

    @torch.no_grad()
    def score_arrays(
        self,
        camera_norm: np.ndarray,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        instance_ids: np.ndarray,
        mvp: np.ndarray | None = None,
    ) -> PredictionResult:
        count = int(instance_ids.size)
        return self.score_batch({
            "camera": camera_norm.astype(np.float32, copy=False),
            "camera_world": camera_world.astype(np.float32, copy=False),
            "camera_view": camera_view.astype(np.float32, copy=False),
            "instance": instance_ids.astype(np.int64, copy=False),
            "pose_offsets": np.asarray([0, count], dtype=np.int64),
        })

    @torch.no_grad()
    def score_batch(self, batch: dict[str, np.ndarray]) -> PredictionResult:
        t0 = time.perf_counter()
        ids = torch.from_numpy(batch["instance"].astype(np.int64, copy=False)).to(self.device)
        camera = torch.from_numpy(batch["camera"].astype(np.float32, copy=False)).to(self.device)
        world = torch.from_numpy(batch["camera_world"].astype(np.float32, copy=False)).to(self.device)
        view = torch.from_numpy(batch["camera_view"].astype(np.float32, copy=False)).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits, aux = self.model.compute_logits_with_aux(camera, view, world, ids, runtime_features=self.runtime_features)
        scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        utility_scores = None
        download_scores = None
        diagnostics = {}
        if "utility_logits" in aux:
            utility_scores = torch.sigmoid(aux["utility_logits"]).detach().cpu().numpy().reshape(-1)
            diagnostics["avgUtilityScore"] = float(np.mean(utility_scores)) if utility_scores.size else 0.0
        if "download_logits" in aux:
            download_scores = aux["download_logits"].detach().cpu().numpy().reshape(-1)
            diagnostics["avgDownloadScore"] = float(np.mean(download_scores)) if download_scores.size else 0.0
        if "inhibition" in aux:
            diagnostics["avgProxyInhibition"] = float(aux["inhibition"].detach().mean().cpu())
        if "selected_proxy" in aux:
            diagnostics["avgSelectedProxyAbs"] = float(aux["selected_proxy"].detach().abs().mean().cpu())
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        return PredictionResult(
            scores=scores,
            forward_ms=(t2 - t1) * 1000.0,
            total_ms=(t2 - t0) * 1000.0,
            diagnostics=diagnostics,
            utility_scores=utility_scores,
            download_scores=download_scores,
        )


class AabbHzbRunner(BaseModelRunner):
    """AABB depth-proxy baseline kept under a legacy compatibility key.

    It rasterizes projected AABB rectangles into CPU proxy depth grids.  It does
    not rasterize triangles and therefore is not a standard geometric HZB.
    """

    def __init__(self, *args, levels: tuple[tuple[int, int], ...] = ((64, 36), (128, 72)), depth_margin: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = tuple((int(w), int(h)) for w, h in levels)
        self.depth_margin = float(depth_margin)
        self.requires_mvp = True

    def _scores_for_pose(self, instance_ids: np.ndarray, mvp: np.ndarray) -> np.ndarray:
        if instance_ids.size == 0:
            return np.zeros((0,), dtype=np.float32)
        rect, area, depth, valid = _project_aabb_features_numpy(self.world_aabbs[instance_ids.astype(np.int64, copy=False)], mvp)
        log_depth = np.log1p(np.maximum(depth, 1e-3) / 100.0).astype(np.float32, copy=False)
        scores = np.zeros((instance_ids.size,), dtype=np.float32)
        order = np.argsort(log_depth)
        for width, height in self.levels:
            depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
            level_scores = np.zeros((instance_ids.size,), dtype=np.float32)
            for idx in order.tolist():
                if not bool(valid[idx]) or area[idx] <= 0.0:
                    continue
                x0 = int(np.floor(np.clip((rect[idx, 0] + 1.0) * 0.5 * width, 0, width - 1)))
                y0 = int(np.floor(np.clip((rect[idx, 1] + 1.0) * 0.5 * height, 0, height - 1)))
                x1 = int(np.floor(np.clip((rect[idx, 2] + 1.0) * 0.5 * width, 0, width - 1)))
                y1 = int(np.floor(np.clip((rect[idx, 3] + 1.0) * 0.5 * height, 0, height - 1)))
                if x1 < x0 or y1 < y0:
                    continue
                tile = depth_buffer[y0:y1 + 1, x0:x1 + 1]
                visible_mask = log_depth[idx] <= (tile + self.depth_margin)
                level_scores[idx] = float(np.mean(visible_mask)) if visible_mask.size else 0.0
                np.minimum(tile, log_depth[idx], out=tile)
            scores = np.maximum(scores, level_scores)
        return scores

    def score_arrays(
        self,
        camera_norm: np.ndarray,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        instance_ids: np.ndarray,
        mvp: np.ndarray | None = None,
    ) -> PredictionResult:
        if mvp is None:
            raise RuntimeError("AABB depth-proxy runner requires mvp rows in dataset or live raw.")
        t0 = time.perf_counter()
        first_mvp = np.asarray(mvp[0] if np.asarray(mvp).ndim == 2 else mvp, dtype=np.float32)
        scores = self._scores_for_pose(instance_ids.astype(np.uint32, copy=False), first_mvp)
        t1 = time.perf_counter()
        return PredictionResult(scores=scores, forward_ms=(t1 - t0) * 1000.0, total_ms=(t1 - t0) * 1000.0)


    def score_batch(self, batch: dict[str, np.ndarray]) -> PredictionResult:
        if "mvp" not in batch:
            raise RuntimeError("AABB depth-proxy runner requires mvp.bin in CSR dataset.")
        t0 = time.perf_counter()
        scores = np.zeros((batch["instance"].shape[0],), dtype=np.float32)
        offsets = batch["pose_offsets"]
        for pose_id in range(offsets.size - 1):
            start = int(offsets[pose_id])
            end = int(offsets[pose_id + 1])
            if end <= start:
                continue
            scores[start:end] = self._scores_for_pose(
                batch["instance"][start:end].astype(np.uint32, copy=False),
                batch["mvp"][start],
            )
        t1 = time.perf_counter()
        return PredictionResult(scores=scores, forward_ms=(t1 - t0) * 1000.0, total_ms=(t1 - t0) * 1000.0)


class LearnedAabbRayRunner(BaseModelRunner):
    """Small learned L0 baseline using only AABB/ray/MVP features."""

    def __init__(self, *args, learned_model: torch.nn.Module, scene_diagonal: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.learned_model = learned_model
        self.scene_diagonal = float(scene_diagonal)
        self.requires_mvp = True
        self.information_level = "L0_metadata_cold_start_learned"
        self.resource_assumption = "instance_aabb_camera_ray_mvp_no_glb_geometry"
        self.feature_dim = AABB_RAY_FEATURE_DIM

    @torch.no_grad()
    def _score_pose(
        self,
        instance_ids: np.ndarray,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        mvp: np.ndarray,
    ) -> np.ndarray:
        features = build_aabb_ray_features(
            self.world_aabbs,
            instance_ids,
            camera_world,
            camera_view,
            mvp,
            self.scene_diagonal,
        )
        tensor = torch.from_numpy(features).to(self.device)
        return torch.sigmoid(self.learned_model(tensor)).detach().cpu().numpy().astype(np.float32, copy=False)

    def score_arrays(
        self,
        camera_norm: np.ndarray,
        camera_world: np.ndarray,
        camera_view: np.ndarray,
        instance_ids: np.ndarray,
        mvp: np.ndarray | None = None,
    ) -> PredictionResult:
        del camera_norm
        if mvp is None:
            raise RuntimeError("learned AABB-ray runner requires mvp rows in the CSR dataset or live raw input")
        t0 = time.perf_counter()
        mvp_row = np.asarray(mvp[0] if np.asarray(mvp).ndim == 2 else mvp, dtype=np.float32)
        scores = self._score_pose(instance_ids, np.asarray(camera_world)[0], np.asarray(camera_view)[0], mvp_row)
        t1 = time.perf_counter()
        return PredictionResult(scores=scores, forward_ms=(t1 - t0) * 1000.0, total_ms=(t1 - t0) * 1000.0)

    def score_batch(self, batch: dict[str, np.ndarray]) -> PredictionResult:
        if "mvp" not in batch:
            raise RuntimeError("learned AABB-ray runner requires mvp.bin in the CSR dataset")
        t0 = time.perf_counter()
        ids_all = np.asarray(batch["instance"], dtype=np.int64).reshape(-1)
        offsets = np.asarray(batch["pose_offsets"], dtype=np.int64)
        scores = np.zeros((ids_all.size,), dtype=np.float32)
        for pose_id in range(offsets.size - 1):
            start = int(offsets[pose_id])
            end = int(offsets[pose_id + 1])
            if end <= start:
                continue
            scores[start:end] = self._score_pose(
                ids_all[start:end],
                np.asarray(batch["camera_world"][start], dtype=np.float32),
                np.asarray(batch["camera_view"][start], dtype=np.float32),
                np.asarray(batch["mvp"][start], dtype=np.float32),
            )
        t1 = time.perf_counter()
        return PredictionResult(scores=scores, forward_ms=(t1 - t0) * 1000.0, total_ms=(t1 - t0) * 1000.0)

def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def load_threshold(eval_summary: str | Path | None, fallback: float) -> float:
    if not eval_summary:
        return float(fallback)
    path = Path(eval_summary)
    if not path.exists():
        return float(fallback)
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = str(data.get("protocol", ""))
    if protocol in {"calibration_ready_pre_test", "frozen_calibration_one_shot_test"}:
        expected_test_count = 0 if protocol == "calibration_ready_pre_test" else 1
        if int(data.get("testEvaluationCount", -1)) != expected_test_count:
            requirement = "zero test evaluations" if expected_test_count == 0 else "exactly one test evaluation"
            raise RuntimeError(
                f"{path} has protocol={protocol!r} but does not record {requirement}."
            )
        frozen = data.get("frozenThreshold")
        if frozen is None:
            raise RuntimeError(f"{path} is missing frozenThreshold; refusing to use a fallback threshold.")
        threshold = float(frozen)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError(f"{path} contains an invalid frozenThreshold: {frozen!r}")
        return threshold
    rows = data.get("thresholdRows") or []
    selected = select_weighted_precision_workpoint(
        rows,
        target_weighted_recall_from_payload(data),
    )
    if selected is not None:
        return float(selected.get("threshold", fallback))
    if rows:
        raise RuntimeError(
            f"{path} has no threshold satisfying the strict weighted-recall rule; "
            "refusing to load an unsafe benchmark threshold."
        )
    return float(fallback)


def load_aabb_hzb_runner(name: str, runtime_meta_path: str | Path, fallback_threshold: float, device: torch.device) -> AabbHzbRunner:
    world_aabbs, instance_to_glb, runtime = load_runtime_meta(runtime_meta_path)
    return AabbHzbRunner(
        name=name,
        kind="aabb_hzb",
        model=None,
        world_aabbs=world_aabbs,
        instance_to_glb=instance_to_glb,
        runtime_meta=runtime,
        checkpoint={"config": {"numInstances": int(world_aabbs.shape[0])}},
        threshold=fallback_threshold,
        device=device,
    )


def load_learned_aabb_ray_runner(
    name: str,
    checkpoint_path: str | Path,
    runtime_meta_path: str | Path,
    eval_summary: str | Path | None,
    fallback_threshold: float,
    device: torch.device,
) -> LearnedAabbRayRunner:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config") or {}
    if checkpoint.get("schema") != "neuralstreamweb3d-learned-aabb-ray-v1":
        raise ValueError(f"{checkpoint_path} is not a learned AABB-ray v1 checkpoint")
    feature_dim = int(config.get("featureDim", -1))
    if feature_dim != AABB_RAY_FEATURE_DIM:
        raise ValueError(f"{checkpoint_path} featureDim={feature_dim}, expected {AABB_RAY_FEATURE_DIM}")
    world_aabbs, instance_to_glb, runtime = load_runtime_meta(runtime_meta_path, int(config["numInstances"]))
    model = AabbRayMLP(feature_dim=feature_dim).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    scene_diagonal = float(config.get("sceneDiagonal", 0.0))
    if not np.isfinite(scene_diagonal) or scene_diagonal <= 1e-6:
        _scene_min, _scene_max, scene_size = scene_min_max(runtime["sceneBounds"])
        scene_diagonal = float(np.linalg.norm(scene_size))
    return LearnedAabbRayRunner(
        name=name,
        kind="learned_aabb_ray",
        model=model,
        learned_model=model,
        world_aabbs=world_aabbs,
        instance_to_glb=instance_to_glb,
        runtime_meta=runtime,
        checkpoint=checkpoint,
        threshold=load_threshold(eval_summary, fallback_threshold),
        device=device,
        scene_diagonal=scene_diagonal,
    )


def compute_train_only_visibility_frequency(
    dataset_dir: str | Path,
    num_instances: int,
) -> tuple[np.ndarray, int]:
    """Compute candidate-conditioned visibility frequency from the train split.

    The numerator counts train poses where an instance is GT-visible.  The
    denominator counts train poses where the same instance appears in the
    stored candidate set.  No validation/test rows are read and no visible ID
    is added to a candidate set.  A missing train positive is a data-semantic
    error rather than something this baseline repairs.
    """

    dataset = PoseCSRDataset(dataset_dir, num_instances=int(num_instances))
    train = dataset.split("train")
    visible_counts = np.zeros((int(num_instances),), dtype=np.int64)
    candidate_counts = np.zeros((int(num_instances),), dtype=np.int64)
    for pose_index in train.pose_indices.tolist():
        candidate_ids = np.unique(dataset.frustum_slice(int(pose_index)).astype(np.int64, copy=False))
        visible_ids = np.unique(dataset.visible_slice(int(pose_index))[0].astype(np.int64, copy=False))
        if candidate_ids.size and (int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= int(num_instances)):
            raise ValueError(f"Train candidate ids are outside the configured instance range at pose {pose_index}.")
        if visible_ids.size and (int(visible_ids.min()) < 0 or int(visible_ids.max()) >= int(num_instances)):
            raise ValueError(f"Train visible ids are outside the configured instance range at pose {pose_index}.")
        missing = np.setdiff1d(visible_ids, candidate_ids, assume_unique=True)
        if missing.size:
            raise ValueError(
                f"Train pose {pose_index} has {missing.size} visible ids outside its stored candidate set; "
                "static frequency refuses to repair candidate semantics."
            )
        np.add.at(candidate_counts, candidate_ids, 1)
        np.add.at(visible_counts, visible_ids, 1)
    frequency = np.divide(
        visible_counts.astype(np.float32),
        np.maximum(candidate_counts, 1),
        dtype=np.float32,
    )
    return frequency, int(train.pose_indices.size)


def build_train_viewcell_bitsets(
    dataset_dir: str | Path,
    num_instances: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build packed visibility bitsets from train view-cell rows only."""

    dataset = PoseCSRDataset(dataset_dir, num_instances=int(num_instances))
    train = dataset.split("train")
    train_indices = train.pose_indices.astype(np.int64, copy=False)
    if train_indices.size == 0:
        raise ValueError("viewcell bitset runner requires at least one train pose.")
    train_world = np.asarray(dataset.poses["camera_world"][train_indices], dtype=np.float32)
    if "camera_forward" in dataset.poses.dtype.names:
        train_forward = np.asarray(dataset.poses["camera_forward"][train_indices], dtype=np.float32)
    else:
        train_forward = np.tile(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), (train_indices.size, 1))

    byte_width = (int(num_instances) + 7) // 8
    bitsets = np.zeros((train_indices.size, byte_width), dtype=np.uint8)
    for row, pose_index in enumerate(train_indices.tolist()):
        candidate_ids = np.unique(dataset.frustum_slice(int(pose_index)).astype(np.int64, copy=False))
        visible_ids = np.unique(dataset.visible_slice(int(pose_index))[0].astype(np.int64, copy=False))
        if candidate_ids.size and (int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= int(num_instances)):
            raise ValueError(f"Train candidate ids are outside the configured instance range at pose {pose_index}.")
        if visible_ids.size and (int(visible_ids.min()) < 0 or int(visible_ids.max()) >= int(num_instances)):
            raise ValueError(f"Train visible ids are outside the configured instance range at pose {pose_index}.")
        missing = np.setdiff1d(visible_ids, candidate_ids, assume_unique=True)
        if missing.size:
            raise ValueError(
                f"Train pose {pose_index} has {missing.size} visible ids outside its stored candidate set; "
                "viewcell bitset refuses to repair candidate semantics."
            )
        if visible_ids.size:
            byte_ids = visible_ids // 8
            bit_values = np.left_shift(np.uint8(1), (visible_ids & 7).astype(np.uint8))
            np.bitwise_or.at(bitsets[row], byte_ids, bit_values)
    return train_world, _normalize_directions(train_forward), bitsets


def load_static_rule_runner(
    name: str,
    kind: str,
    runtime_meta_path: str | Path,
    fallback_threshold: float,
    device: torch.device,
    dataset_dir: str | Path | None = None,
) -> StaticRuleRunner:
    """Construct one of the deterministic, candidate-preserving L0 rules."""

    world_aabbs, instance_to_glb, runtime = load_runtime_meta(runtime_meta_path)
    common = {
        "name": name,
        "kind": kind,
        "model": None,
        "world_aabbs": world_aabbs,
        "instance_to_glb": instance_to_glb,
        "runtime_meta": runtime,
        "checkpoint": {"config": {"numInstances": int(world_aabbs.shape[0])}},
        "threshold": fallback_threshold,
        "device": device,
    }
    if kind == "keep_all":
        return KeepAllRunner(**common)
    if kind == "static_frequency_train":
        if dataset_dir is None:
            raise ValueError("static_frequency_train runner requires the CSR dataset directory.")
        frequency, train_pose_count = compute_train_only_visibility_frequency(
            dataset_dir,
            int(world_aabbs.shape[0]),
        )
        return StaticFrequencyRunner(
            **common,
            train_visibility_frequency=frequency,
            train_pose_count=train_pose_count,
        )
    if kind == "camera_distance":
        return CameraDistanceRunner(**common)
    if kind == "projected_aabb_area":
        return ProjectedAabbAreaRunner(**common)
    if kind == "aabb_ray":
        return AabbRayRunner(**common)
    raise ValueError(f"Unsupported static rule kind: {kind}")


def load_train_viewcell_bitset_runner(
    name: str,
    runtime_meta_path: str | Path,
    fallback_threshold: float,
    device: torch.device,
    dataset_dir: str | Path | None,
) -> TrainViewcellBitsetRunner:
    if dataset_dir is None:
        raise ValueError("viewcell_bitset_train runner requires the CSR dataset directory.")
    world_aabbs, instance_to_glb, runtime = load_runtime_meta(runtime_meta_path)
    train_world, train_forward, bitsets = build_train_viewcell_bitsets(
        dataset_dir,
        int(world_aabbs.shape[0]),
    )
    return TrainViewcellBitsetRunner(
        name=name,
        kind="viewcell_bitset_train",
        model=None,
        world_aabbs=world_aabbs,
        instance_to_glb=instance_to_glb,
        runtime_meta=runtime,
        checkpoint={"config": {"numInstances": int(world_aabbs.shape[0])}},
        threshold=fallback_threshold,
        device=device,
        train_pose_world=train_world,
        train_pose_forward=train_forward,
        train_bitsets=bitsets,
    )


def load_directional_occlusion_proxy_encoder_runner(
    name: str,
    checkpoint_path: str | Path,
    runtime_meta_path: str | Path,
    runtime_features_path: str | Path,
    eval_summary: str | Path | None,
    fallback_threshold: float,
    device: torch.device,
) -> DirectionalOcclusionProxyEncoderRunner:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    world_aabbs, instance_to_glb, runtime = load_runtime_meta(runtime_meta_path, config["numInstances"])
    scene_min, _scene_max, scene_size = scene_min_max(runtime["sceneBounds"])
    data = np.fromfile(runtime_features_path, dtype=np.float16)
    expected = int(config["numInstances"]) * int(config["runtimeFeatureDim"])
    if data.size != expected:
        raise ValueError(f"{runtime_features_path} has {data.size} fp16 values, expected {expected}")
    runtime_np = data.reshape(config["numInstances"], config["runtimeFeatureDim"]).astype(np.float32)
    train_args = checkpoint.get("args", {})
    model = DirectionalOcclusionProxyEncoderPVSModel(
        num_instances=config["numInstances"],
        num_glbs=config["numGlbs"],
        geo_dim=config["geoDim"],
        context_dim=config["contextDim"],
        proxy_dim=config["proxyDim"],
        direction_bins=config["directionBins"],
        depth_shells=config["depthShells"],
        source_k=config.get("sourceK", 8),
        point_hidden_dim=train_args.get("point_hidden_dim", 160),
        pointnetpp_centers=train_args.get("pointnetpp_centers", 24),
        pointnetpp_neighbors=train_args.get("pointnetpp_neighbors", 12),
        graph_hidden_dim=train_args.get("graph_hidden_dim", 160),
        graph_message_dim=train_args.get("graph_message_dim", 96),
        ray_fourier_bands=train_args.get("ray_fourier_bands", 10),
        ray_scalar_fourier_bands=train_args.get("ray_scalar_fourier_bands", 4),
        camera_location_dim=config.get("cameraLocationFeatureDim", train_args.get("camera_location_dim", 0)),
        mlp_hidden=train_args.get("mlp_hidden", 128),
        interaction_dim=train_args.get("interaction_dim", 64),
        scene_size_m=config.get("sceneSizeM", scene_size.tolist()),
        runtime_feature_ablation=str(config.get("runtimeFeatureAblation", "none")),
    ).to(device)
    model.set_scene_bounds(torch.from_numpy(scene_min).to(device), torch.from_numpy(scene_size).to(device))
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return DirectionalOcclusionProxyEncoderRunner(
        name=name,
        kind="directional_occlusion_proxy_encoder",
        model=model,
        world_aabbs=world_aabbs,
        instance_to_glb=instance_to_glb,
        runtime_meta=runtime,
        checkpoint=checkpoint,
        threshold=load_threshold(eval_summary, fallback_threshold),
        device=device,
        runtime_features=torch.from_numpy(runtime_np).to(device),
    )


def load_runner(
    name: str,
    spec: dict[str, str],
    runtime_meta: str | Path,
    device: torch.device,
    fallback_threshold: float = 0.5,
    dataset_dir: str | Path | None = None,
) -> BaseModelRunner:
    kind = spec["kind"]
    if kind == "viewcell_bitset_train":
        return load_train_viewcell_bitset_runner(name, runtime_meta, fallback_threshold, device, dataset_dir)
    if kind in {"keep_all", "static_frequency_train", "camera_distance", "projected_aabb_area", "aabb_ray"}:
        return load_static_rule_runner(name, kind, runtime_meta, fallback_threshold, device, dataset_dir)
    if kind == "learned_aabb_ray":
        return load_learned_aabb_ray_runner(
            name,
            spec["checkpoint"],
            runtime_meta,
            spec.get("eval_summary"),
            fallback_threshold,
            device,
        )
    if kind == "aabb_hzb":
        return load_aabb_hzb_runner(name, runtime_meta, fallback_threshold, device)
    if kind == "directional_occlusion_proxy_encoder":
        return load_directional_occlusion_proxy_encoder_runner(
            name,
            spec["checkpoint"],
            runtime_meta,
            spec["runtime_features"],
            spec.get("eval_summary"),
            fallback_threshold,
            device,
        )
    raise ValueError(f"Unsupported retained model kind: {kind}")


def selected_default_specs(names: str) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for raw in names.split(","):
        name = raw.strip()
        if not name:
            continue
        canonical_name = MODEL_ALIASES.get(name, name)
        if canonical_name not in DEFAULT_MODEL_SPECS:
            raise KeyError(f"Unknown model '{name}'. Available: {', '.join(DEFAULT_MODEL_SPECS)}")
        spec = DEFAULT_MODEL_SPECS[canonical_name]
        selected[canonical_name] = spec
    return selected
