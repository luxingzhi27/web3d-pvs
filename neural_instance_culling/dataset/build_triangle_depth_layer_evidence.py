#!/usr/bin/env python3
"""Build fixed instance relations from a browser triangle depth-peeling cache.

The browser writes a stack of component-ID and linear-depth images for each
stored pose.  This module turns adjacent different IDs in those stacks into
two independent offline products:

* a compact spherical/depth ordered source table used by the context encoder;
* right-censored/occluded observations used by the monotone survival field.

The stored candidate CSR is authoritative.  This script never unions visible
IDs into candidates and never uses AABB overlap as an evidence source.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from common.candidate_identity import (  # noqa: E402
    audit_native_aabb_candidates,
    candidate_digest_for_pose_sequence,
    pose_sequence_for_splits,
)
from common.glb_points import load_glb_points  # noqa: E402
from common.train_observed_relation_csr import (  # noqa: E402
    normalize_radius_relative_log_depth,
    train_depth_quantiles,
    validate_survival_observations_v3,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from surface_point_occlusion_fallback import (  # noqa: E402
    collect_front_surface_relations,
    select_fallback_targets,
)


LEGACY_SCHEMA = "triangle-depth-layer-evidence-v1"
SCHEMA = "triangle-depth-layer-evidence-v3"
SCHEMA_V3 = SCHEMA
CACHE_SCHEMA_V1 = "triangle-depth-layer-cache-v1"
CACHE_SCHEMA_V2 = "triangle-depth-layer-cache-v2"
CACHE_SCHEMA = "triangle-depth-layer-cache-v3"
SUPPORTED_CACHE_SCHEMAS = {CACHE_SCHEMA_V1, CACHE_SCHEMA_V2, CACHE_SCHEMA}
LINEAR_DEPTH_ENCODING = "camera_forward_axial_depth_divided_by_camera_far"
GPU_EVIDENCE_SCHEMA = "triangle-depth-layer-gpu-evidence-v1"
BACKGROUND_ID = np.uint32(0xFFFFFFFF)

# Fallback relations are retained per rendered subpose so that the later
# context-table builder can merge them with depth-peeling relations without
# inventing pose support from an already aggregated top-k table.
SURFACE_FALLBACK_RELATION_SCHEMA = "triangle-surface-fallback-relation-v2"
SURFACE_FALLBACK_RELATION_DTYPE = np.dtype([
    ("renderPoseId", "<u4"),
    ("source", "<u4"),
    ("target", "<u4"),
    ("pixelCount", "<u4"),
    ("gapMean", "<f4"),
    ("gapStd", "<f4"),
])


def select_topk_evidence(entries: list[dict[str, Any]], k: int) -> tuple[list[dict[str, Any]], float, float, bool]:
    """Select true evidence mass, preserving deterministic source ranking.

    Each entry must contain ``source`` and ``pixelMass``.  The returned
    retained quality is retained mass divided by all observed mass; it is not
    inferred from candidate or GT membership.
    """
    if int(k) <= 0:
        raise ValueError("evidence top-k must be positive")
    normalized = [dict(entry) for entry in entries if float(entry.get("pixelMass", 0.0)) > 0.0]
    normalized.sort(key=lambda item: (-float(item["pixelMass"]), int(item.get("source", 0))))
    total = float(sum(float(item["pixelMass"]) for item in normalized))
    kept = normalized[: int(k)]
    retained = float(sum(float(item["pixelMass"]) for item in kept))
    return kept, total, retained / total if total > 0.0 else 1.0, len(normalized) > int(k)


def build_survival_evidence_records(
    ids: np.ndarray,
    depths: np.ndarray,
    candidate_set: set[int],
    directions: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    camera_world: np.ndarray,
    min_depth_gap: float,
    *,
    subpose_id: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Create conserved event/censor evidence for one rendered subpose.

    A visible target receives a right-censor record and a later different
    front layer independently receives an event record.  Thus both records
    can exist for the same target and subpose.  Weights are normalized over
    the raw retained evidence mass for this subpose and therefore sum to one.
    """
    ids = np.asarray(ids, dtype=np.uint32)
    depths = np.asarray(depths, dtype=np.float32)
    visible: dict[int, int] = {}
    visible_depth: dict[int, float] = {}
    first_ids = ids[0].reshape(-1)
    first_depth = depths[0].reshape(-1)
    first_valid = (first_ids != BACKGROUND_ID) & np.isfinite(first_depth)
    if bool(first_valid.any()):
        valid_ids = first_ids[first_valid]
        valid_depth = first_depth[first_valid]
        order = np.lexsort((valid_depth, valid_ids))
        sorted_ids = valid_ids[order]
        sorted_depth = valid_depth[order]
        starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
        ends = np.r_[starts[1:], sorted_ids.size]
        values = sorted_ids[starts]
        counts = ends - starts
        low = starts + (counts - 1) // 2
        high = starts + counts // 2
        medians = (sorted_depth[low] + sorted_depth[high]) * 0.5
        for value, count, median in zip(
            values.tolist(), counts.tolist(), medians.tolist(), strict=True
        ):
            if int(value) in candidate_set:
                visible[int(value)] = int(count)
                visible_depth[int(value)] = float(median)
    events: dict[int, int] = defaultdict(int)
    event_depth_sum: dict[int, float] = defaultdict(float)
    for layer in range(max(0, int(ids.shape[0]) - 1)):
        front = ids[layer].reshape(-1)
        back = ids[layer + 1].reshape(-1)
        gap = depths[layer + 1].reshape(-1) - depths[layer].reshape(-1)
        mask = (front != BACKGROUND_ID) & (back != BACKGROUND_ID) & (front != back) & np.isfinite(gap) & (gap > float(min_depth_gap))
        if not bool(mask.any()):
            continue
        selected_back = back[mask]
        selected_depth = depths[layer + 1].reshape(-1)[mask]
        back_values, inverse, back_counts = np.unique(
            selected_back, return_inverse=True, return_counts=True
        )
        depth_sums = np.bincount(
            inverse,
            weights=selected_depth.astype(np.float64, copy=False),
            minlength=back_values.size,
        )
        for value, count, depth_sum in zip(
            back_values.tolist(), back_counts.tolist(), depth_sums.tolist(), strict=True
        ):
            if int(value) in candidate_set:
                events[int(value)] += int(count)
                event_depth_sum[int(value)] += float(depth_sum)
    raw_total = float(sum(visible.values()) + sum(events.values()))
    if raw_total <= 0.0:
        return [], {"rawEvidenceMass": 0.0, "weightSum": 0.0}
    records: list[dict[str, Any]] = []
    def append(instance: int, raw_pixels: int, event: int, evidence_level: int, depth_value: float) -> None:
        records.append({
            "subpose": int(subpose_id),
            "instance": int(instance),
            "direction": int(directions[instance]),
            # Input depths are metric camera-ray ranges reconstructed from the
            # raster cache.  Dividing by the target radius gives the same
            # physical coordinate used by the runtime center-distance query.
            "depth": float(max(depth_value, 0.0) / max(float(radii[instance]), 1e-6)),
            "event": int(event),
            "rawPixelCount": int(raw_pixels),
            "weight": float(raw_pixels / raw_total),
            "confidence": float(np.clip(raw_pixels / raw_total, 0.0, 1.0)),
            "evidenceLevel": int(evidence_level),
        })
    for instance, count in visible.items():
        append(instance, count, 0, 0, visible_depth[instance])
    for instance, count in events.items():
        append(instance, count, 1, 1, event_depth_sum[instance] / max(float(count), 1.0))
    return records, {"rawEvidenceMass": raw_total, "weightSum": float(sum(item["weight"] for item in records))}


def spherical_direction_anchors() -> np.ndarray:
    """Return twelve fixed anchors: four equatorial and two slanted rings."""
    values: list[list[float]] = []
    for pitch in (0.0, math.pi / 4.0, -math.pi / 4.0):
        for yaw in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
            cp = math.cos(pitch)
            values.append([math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp])
    return np.asarray(values, dtype=np.float32)


def spherical_direction_bins(camera_world: np.ndarray, centers: np.ndarray) -> np.ndarray:
    ray = np.asarray(centers, dtype=np.float32) - np.asarray(camera_world, dtype=np.float32)[None, :]
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-6)
    anchors = spherical_direction_anchors()
    return np.argmax(ray @ anchors.T, axis=1).astype(np.int64, copy=False)


def _load_formal_gpu_evidence(root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Require browser and host evidence before using a formal cache."""
    roots: list[Path] = []
    source_shards = meta.get("sourceShards") or []
    if source_shards:
        roots.extend(Path(value) for value in source_shards)
    else:
        roots.append(root)
    evidence_paths: list[Path] = []
    for candidate_root in roots:
        evidence_paths.extend(sorted(candidate_root.glob("*.gpu_evidence.json")))
    if not evidence_paths:
        raise ValueError(f"triangle depth cache has no GPU evidence JSON: {root}")

    summaries: list[dict[str, Any]] = []
    for evidence_path in evidence_paths:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        gate = evidence.get("gpuGate") or {}
        before = evidence.get("hostGpuBefore") or {}
        during = evidence.get("hostGpuDuring") or {}
        nvidia_before = before.get("nvidiaSmi") or {}
        pmon_before = before.get("nvidiaSmiPmon") or {}
        nvidia_during = during.get("nvidiaSmi") or {}
        pmon_during = during.get("nvidiaSmiPmon") or {}
        formal = bool(
            evidence.get("schema") == GPU_EVIDENCE_SCHEMA
            and evidence.get("formalReady") is True
            and gate.get("hardware") is True
            and nvidia_before.get("available") is True
            and pmon_before.get("available") is True
            and nvidia_during.get("available") is True
            and pmon_during.get("available") is True
        )
        if not formal:
            raise ValueError(
                f"{evidence_path}: formal GPU evidence requires hardware WebGL, "
                "formalReady=true, nvidia-smi and nvidia-smi pmon before/during"
            )
        summaries.append({
            "path": str(evidence_path.resolve()),
            "schema": evidence.get("schema"),
            "formalReady": True,
            "gpuGate": gate,
            "hostGpuEvidence": {
                "beforeNvidiaSmi": True,
                "beforeNvidiaSmiPmon": True,
                "duringNvidiaSmi": True,
                "duringNvidiaSmiPmon": True,
            },
        })
    return {
        "schema": "triangle-depth-layer-gpu-evidence-summary-v1",
        "formalReady": True,
        "sourceCount": len(summaries),
        "sources": summaries,
    }


def load_layer_cache(path: str | Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(path)
    meta = json.loads((root / "layer_cache_meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") not in SUPPORTED_CACHE_SCHEMAS:
        raise ValueError(f"Expected one of {sorted(SUPPORTED_CACHE_SCHEMAS)}, got {meta.get('schema')!r}")
    meta["gpuEvidenceSummary"] = _load_formal_gpu_evidence(root, meta)
    pose_count = int(meta["poseCount"])
    max_layers = int(meta["maxLayers"])
    width = int(meta["width"])
    height = int(meta["height"])
    render_pose_ids = np.fromfile(root / meta["files"]["poseIndices"], dtype="<u4")
    if render_pose_ids.size != pose_count:
        raise ValueError("layer cache render pose ID count does not match metadata")
    if np.unique(render_pose_ids).size != render_pose_ids.size:
        raise ValueError("layer cache contains duplicate render pose IDs")
    if meta.get("schema") in (CACHE_SCHEMA_V2, CACHE_SCHEMA):
        files = meta.get("files", {})
        if not files.get("sourcePoseIndices") or not files.get("renderPoseIds"):
            raise ValueError("v2 layer cache must contain explicit sourcePoseIndices and renderPoseIds")
        explicit_render_ids = np.fromfile(root / files["renderPoseIds"], dtype="<u4")
        source_pose_indices = np.fromfile(root / files["sourcePoseIndices"], dtype="<u4")
        if explicit_render_ids.size != pose_count or not np.array_equal(explicit_render_ids, render_pose_ids):
            raise ValueError("layer cache renderPoseIds do not match poseIndices")
        if source_pose_indices.size != pose_count:
            raise ValueError("layer cache source pose index count does not match metadata")
    else:
        source_pose_indices = render_pose_ids.copy()
    manifest_path = meta.get("sourceManifest")
    if manifest_path:
        manifest = Path(str(manifest_path))
        if manifest.is_file():
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_camera_far = float(manifest_payload.get("cameraFar", float("nan")))
            if np.isfinite(manifest_camera_far) and manifest_camera_far > 0.0:
                stored_camera_far = float(meta.get("cameraFarMeters", manifest_camera_far))
                if not np.isclose(stored_camera_far, manifest_camera_far, rtol=1e-6, atol=1e-6):
                    raise ValueError("layer cache cameraFarMeters disagrees with its source manifest")
                meta["cameraFarMeters"] = manifest_camera_far
            records = {}
            for ordinal, record in enumerate(manifest_payload.get("poses", [])):
                render_id = int(record.get("renderPoseId", record.get("poseIndex", ordinal)))
                records[render_id] = record
            if len(records) != pose_count:
                raise ValueError(
                    f"layer cache manifest has {len(records)} render records, expected {pose_count}"
                )
            meta["renderPoseRecords"] = records
    if meta.get("schema") == CACHE_SCHEMA:
        if meta.get("linearDepthEncoding") != LINEAR_DEPTH_ENCODING:
            raise ValueError("v3 layer cache has an unknown linear-depth encoding")
    else:
        # V1/V2 stored the same normalized axial value but did not spell the
        # coordinate out.  Decoding it here repairs the consumer contract; no
        # relation or survival builder is allowed to consume the raw value.
        meta["linearDepthEncoding"] = LINEAR_DEPTH_ENCODING
    camera_far = float(meta.get("cameraFarMeters", float("nan")))
    if not np.isfinite(camera_far) or camera_far <= 0.0:
        raise ValueError("layer cache needs a finite positive cameraFarMeters for metric decoding")
    shape = (pose_count, max_layers, height, width)
    ids = np.memmap(root / meta["files"]["instanceIds"], dtype="<u4", mode="r", shape=shape)
    depths = np.memmap(root / meta["files"]["linearDepth"], dtype="<f4", mode="r", shape=shape)
    if ids.shape != depths.shape:
        raise ValueError("layer ID and depth cache shapes differ")
    return meta, render_pose_ids.astype(np.int64, copy=False), source_pose_indices.astype(np.int64, copy=False), ids, depths


def metric_ray_depths_for_cache_row(
    cache_meta: dict[str, Any],
    encoded_depths: np.ndarray,
    camera_view: np.ndarray,
) -> np.ndarray:
    """Decode normalized axial raster depth into metric camera-ray range.

    The peel comparison is performed in normalized camera-forward depth.  A
    survival observation, however, is queried at runtime with the Euclidean
    camera-to-instance range.  For a perspective pixel the conversion is
    ``z * sqrt(1 + (u*tan_x)^2 + (v*tan_y)^2)``.
    """
    if cache_meta.get("linearDepthEncoding") != LINEAR_DEPTH_ENCODING:
        raise ValueError("unsupported triangle cache linear-depth encoding")
    depth = np.asarray(encoded_depths, dtype=np.float32)
    if depth.ndim != 3:
        raise ValueError("encoded depth row must have shape [layers, height, width]")
    finite = np.isfinite(depth)
    if np.any(finite & ((depth < 0.0) | (depth > 1.0))):
        raise ValueError("normalized axial depth must be in [0, 1]")
    view = np.asarray(camera_view, dtype=np.float32).reshape(5)
    tan_x = float(view[3])
    tan_y = float(view[4])
    camera_far = float(cache_meta.get("cameraFarMeters", float("nan")))
    if not np.isfinite(camera_far) or camera_far <= 0.0 or tan_x <= 0.0 or tan_y <= 0.0:
        raise ValueError("metric depth decoding needs positive cameraFar/tan_x/tan_y")
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    u = ((np.arange(width, dtype=np.float32) + 0.5) / float(width)) * 2.0 - 1.0
    v = ((np.arange(height, dtype=np.float32) + 0.5) / float(height)) * 2.0 - 1.0
    ray_scale = np.sqrt(
        1.0
        + np.square(u[None, :] * np.float32(tan_x))
        + np.square(v[:, None] * np.float32(tan_y))
    ).astype(np.float32, copy=False)
    return depth * np.float32(camera_far) * ray_scale[None, :, :]


def _camera_for_cache_row(
    cache_meta: dict[str, Any],
    render_pose_id: int,
    dataset: PoseCSRDataset,
    source_pose_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = (cache_meta.get("renderPoseRecords") or {}).get(int(render_pose_id))
    if isinstance(record, dict) and record.get("cameraWorld") and record.get("cameraForward"):
        camera_world = np.asarray(record["cameraWorld"], dtype=np.float32).reshape(3)
        camera_forward = np.asarray(record["cameraForward"], dtype=np.float32).reshape(3)
        camera_view = np.asarray(record.get("cameraView") or dataset.camera_view(int(source_pose_index)), dtype=np.float32).reshape(5)
        return camera_world, camera_forward, camera_view
    pose = dataset.poses[int(source_pose_index)]
    return (
        np.asarray(pose["camera_world"], dtype=np.float32),
        np.asarray(pose["camera_forward"], dtype=np.float32),
        np.asarray(dataset.camera_view(int(source_pose_index)), dtype=np.float32),
    )


def _topk_insert(
    scores: np.ndarray,
    ids: np.ndarray,
    gaps: np.ndarray,
    target: int,
    direction: int,
    shell: int,
    source: int,
    score: float,
    gap: float,
    source_k: int,
) -> None:
    if score <= 0.0 or source == target:
        return
    row = scores[target, direction, shell]
    position = int(np.argmin(row))
    if score <= float(row[position]):
        return
    row[position] = float(score)
    ids[target, direction, shell, position] = int(source)
    gaps[target, direction, shell, position] = float(gap)


def _shell_id(relative_gap: float, shells: int) -> int:
    if shells <= 1:
        return 0
    if shells == 2:
        return int(relative_gap >= 0.2)
    return int(np.clip(np.digitize(relative_gap, np.asarray([0.10, 0.35], dtype=np.float32)), 0, shells - 1))


def _read_candidate_pose(dataset: PoseCSRDataset, pose_index: int) -> np.ndarray:
    return np.asarray(dataset.frustum_slice(int(pose_index)), dtype=np.uint32)


def _adjacent_occlusion_pixel_counts(
    layer_ids: np.ndarray,
    candidate_set: set[int],
) -> dict[int, int]:
    """Count only back instances preceded by a different front instance.

    Repeated surfaces belonging to the same component are not an occlusion
    event.  Keeping this distinction is important for the right-censored
    survival targets: an event must have a different front component.
    """
    values: dict[int, int] = {}
    ids = np.asarray(layer_ids, dtype=np.uint32)
    if ids.ndim < 3 or ids.shape[0] < 2:
        return values
    for layer in range(ids.shape[0] - 1):
        front = ids[layer].reshape(-1)
        back = ids[layer + 1].reshape(-1)
        mask = (
            (front != BACKGROUND_ID)
            & (back != BACKGROUND_ID)
            & (front != back)
        )
        if not bool(mask.any()):
            continue
        back_values, counts = np.unique(back[mask], return_counts=True)
        for value, count in zip(back_values.tolist(), counts.tolist(), strict=False):
            instance = int(value)
            if instance in candidate_set:
                values[instance] = values.get(instance, 0) + int(count)
    return values


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    cache_dir = Path(args.layer_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(Path(args.runtime_meta))
    num_instances = int(world_aabbs.shape[0])
    _scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    centers = ((world_aabbs[:, :3] + world_aabbs[:, 3:]) * 0.5).astype(np.float32, copy=False)
    radii = np.linalg.norm(np.maximum(world_aabbs[:, 3:] - world_aabbs[:, :3], 1e-4), axis=1) * 0.5
    dataset = PoseCSRDataset(dataset_dir, num_instances=num_instances)
    cache_meta, cache_render_pose_ids, cache_source_pose_indices, cache_ids, cache_depths = load_layer_cache(cache_dir)
    if int(cache_meta.get("modelInputFovYDeg", 66)) != 66:
        raise ValueError("triangle evidence cache must use the 66 degree model-input FOV")
    if int(cache_meta.get("maxLayers", 0)) < 2:
        raise ValueError("triangle evidence requires at least two depth layers")
    if cache_meta.get("schema") in (CACHE_SCHEMA_V2, CACHE_SCHEMA):
        first_layer_reference = cache_meta.get("firstLayerReference") or {}
        if int(first_layer_reference.get("checks", 0)) != int(cache_render_pose_ids.size) or int(first_layer_reference.get("mismatches", 0)) != 0:
            raise ValueError("v2 triangle cache lacks a matching unpeeled Color-ID reference for every pose")
    cache_rows = list(range(int(cache_render_pose_ids.size)))

    requested_split_names = [name.strip() for name in str(args.splits).split(",") if name.strip()]
    if requested_split_names != ["train"]:
        raise ValueError("v3 triangle evidence survival supervision is train-only")
    missing_splits = [name for name in requested_split_names if name not in dataset.split_ids]
    if missing_splits:
        raise ValueError(f"dataset does not define requested splits: {missing_splits}")
    split_values = {int(dataset.split_ids[name]) for name in requested_split_names}
    selected_entries = [
        (row, int(cache_render_pose_ids[row]), int(cache_source_pose_indices[row]))
        for row in cache_rows
        if 0 <= int(cache_source_pose_indices[row]) < dataset.poses.shape[0]
        and int(dataset.poses[int(cache_source_pose_indices[row])]["split"]) in split_values
    ]
    if args.max_poses > 0:
        selected_entries = selected_entries[: int(args.max_poses)]
    if not selected_entries:
        raise ValueError("triangle cache has no poses in the requested split")

    surface_points = None
    surface_fallback_config = {
        "enabled": bool(getattr(args, "surface_point_fallback", False)),
        "status": "disabled",
    }
    if surface_fallback_config["enabled"]:
        points_path = getattr(args, "glb_points", None)
        if not points_path:
            raise ValueError("surface point fallback requires --glb-points")
        surface_points, point_meta = load_glb_points(points_path)
        if surface_points.ndim != 3 or surface_points.shape[0] <= int(instance_to_glb.max()):
            raise ValueError("GLB point cache does not cover all runtime global GLB IDs")
        surface_fallback_config.update({
            "status": "enabled",
            "glbPoints": str(Path(points_path).resolve()),
            "glbPointCount": int(surface_points.shape[1]),
            "glbCount": int(surface_points.shape[0]),
            "pointCacheMeta": point_meta,
            "maxTargetsPerPose": int(getattr(args, "surface_max_targets", 256)),
            "pointsPerTarget": int(getattr(args, "surface_points_per_target", 64)),
        })

    direction_bins = 12
    depth_shells = 3
    source_k = int(args.source_k)
    source_scores = np.zeros((num_instances, direction_bins, depth_shells, source_k), dtype=np.float32)
    source_ids = np.zeros((num_instances, direction_bins, depth_shells, source_k), dtype=np.uint32)
    source_gaps = np.zeros_like(source_scores)
    source_confidences = np.zeros_like(source_scores)
    source_evidence_levels = np.zeros((num_instances, direction_bins, depth_shells, source_k), dtype=np.uint8)
    source_layer_counts = np.zeros_like(source_evidence_levels)
    evidence_strength = np.zeros((num_instances, direction_bins, depth_shells), dtype=np.float32)
    evidence_count = np.zeros_like(evidence_strength, dtype=np.uint32)
    total_evidence_mass = np.zeros_like(evidence_strength)
    retained_evidence_mass = np.zeros_like(evidence_strength)
    retained_quality = np.ones_like(evidence_strength)
    source_counts = np.zeros_like(evidence_count)
    topk_truncated = np.zeros_like(evidence_count, dtype=np.uint8)
    observation_instances: list[int] = []
    observation_directions: list[int] = []
    observation_depth: list[float] = []
    observation_event: list[int] = []
    observation_weight: list[float] = []
    observation_subpose: list[int] = []
    observation_raw_pixels: list[int] = []
    observation_confidence: list[float] = []
    observation_evidence_level: list[int] = []
    first_layer_failures: list[dict[str, Any]] = []
    total_layer_pairs = 0
    total_pixels = 0
    fallback_relation_count = 0
    fallback_target_count = 0
    fallback_pose_count = 0
    fallback_relation_rows: list[tuple[int, int, int, int, float, float]] = []
    depth_monotonicity_violations = 0
    topk_entries: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)

    started_at = time.perf_counter()
    progress_interval = max(1, min(500, len(selected_entries) // 20 or 1))
    for entry_index, (row, render_pose_id, pose_index) in enumerate(selected_entries, start=1):
        ids = np.asarray(cache_ids[row], dtype=np.uint32)
        encoded_depths = np.asarray(cache_depths[row], dtype=np.float32)
        candidate = _read_candidate_pose(dataset, pose_index)
        candidate_set = set(int(value) for value in candidate.tolist())
        first_unique = np.unique(ids[0][ids[0] != BACKGROUND_ID])
        outside_first = [int(value) for value in first_unique.tolist() if int(value) not in candidate_set]
        if outside_first:
            first_layer_failures.append({"poseIndex": int(pose_index), "renderPoseId": int(render_pose_id), "outsideCandidate": outside_first[:16]})
            if not args.allow_outside_candidate:
                raise ValueError(
                    f"first depth layer contains {len(outside_first)} IDs outside stored candidates at pose {pose_index}"
                )
        camera_world, camera_forward, camera_view = _camera_for_cache_row(
            cache_meta, render_pose_id, dataset, pose_index
        )
        directions = spherical_direction_bins(camera_world, centers)
        finite_depth = np.isfinite(encoded_depths)
        if np.any(finite_depth & ((encoded_depths < 0.0) | (encoded_depths > 1.0))):
            raise ValueError(f"depth cache contains values outside [0,1] at pose {pose_index}")
        adjacent_valid = (
            (ids[:-1] != BACKGROUND_ID)
            & (ids[1:] != BACKGROUND_ID)
            & finite_depth[:-1]
            & finite_depth[1:]
        )
        non_increasing = adjacent_valid & (encoded_depths[1:] <= encoded_depths[:-1])
        violation_count = int(non_increasing.sum())
        depth_monotonicity_violations += violation_count
        if violation_count:
            raise ValueError(
                f"depth layers are not strictly increasing at pose {pose_index}: "
                f"{violation_count} valid pixel transitions"
            )
        depths = metric_ray_depths_for_cache_row(cache_meta, encoded_depths, camera_view)

        # A target may legitimately contribute both records in this subpose:
        # layer zero is right-censored, while a deeper different layer is an
        # independent occlusion event.  The helper also enforces weight
        # conservation over the retained evidence mass.
        pose_records, pose_mass = build_survival_evidence_records(
            ids, depths, candidate_set, directions, centers, radii,
            camera_world, float(args.min_depth_gap), subpose_id=int(render_pose_id),
        )
        for record in pose_records:
            observation_instances.append(int(record["instance"]))
            observation_directions.append(int(record["direction"]))
            observation_depth.append(float(record["depth"]))
            observation_event.append(int(record["event"]))
            observation_weight.append(float(record["weight"]))
            observation_subpose.append(int(record["subpose"]))
            observation_raw_pixels.append(int(record["rawPixelCount"]))
            observation_confidence.append(float(record["confidence"]))
            observation_evidence_level.append(int(record["evidenceLevel"]))

        # Adjacent different layers are the formal directed evidence.  The
        # operation is vectorised over pixels and aggregates repeated pairs.
        for layer in range(ids.shape[0] - 1):
            front = ids[layer].reshape(-1)
            back = ids[layer + 1].reshape(-1)
            gap = depths[layer + 1].reshape(-1) - depths[layer].reshape(-1)
            mask = (front != int(BACKGROUND_ID)) & (back != int(BACKGROUND_ID)) & (front != back) & np.isfinite(gap) & (gap > float(args.min_depth_gap))
            if not bool(mask.any()):
                continue
            front = front[mask].astype(np.int64, copy=False)
            back = back[mask].astype(np.int64, copy=False)
            gap = gap[mask].astype(np.float32, copy=False)
            valid = np.asarray([int(a) in candidate_set and int(b) in candidate_set for a, b in zip(front, back, strict=False)])
            if not args.allow_outside_candidate:
                front, back, gap = front[valid], back[valid], gap[valid]
            if front.size == 0:
                continue
            pairs = np.stack([front, back], axis=1)
            unique_pairs, inverse, pair_counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
            total_layer_pairs += int(unique_pairs.shape[0])
            total_pixels += int(front.size)
            for pair_index, (source, target) in enumerate(unique_pairs.tolist()):
                pair_gap = float(np.median(gap[inverse == pair_index]))
                source = int(source)
                target = int(target)
                target_depth = max(float(np.median(depths[layer + 1][ids[layer + 1] == target])), 1e-4)
                relative_gap = max(0.0, pair_gap) / max(target_depth, 1e-4)
                direction = int(directions[target])
                shell = _shell_id(relative_gap, depth_shells)
                score = float(pair_counts[pair_index]) / max(1.0, float(ids.shape[-1] * ids.shape[-2]))
                score *= float(np.clip(pair_gap / 0.05, 0.0, 1.0))
                _topk_insert(source_scores, source_ids, source_gaps, target, direction, shell, source, score, relative_gap, source_k)
                topk_entries[(target, direction, shell)].append({
                    "source": source,
                    "pixelMass": float(pair_counts[pair_index]),
                    "depthGap": relative_gap,
                    "confidence": float(np.clip(pair_gap / 0.05, 0.0, 1.0)),
                    "evidenceLevel": int(layer + 1),
                })
                total_evidence_mass[target, direction, shell] += float(pair_counts[pair_index])
                evidence_strength[target, direction, shell] += score
                evidence_count[target, direction, shell] += 1

        if surface_points is not None:
            observed_all = np.unique(ids[ids != BACKGROUND_ID])
            fallback_targets, fallback_stats = select_fallback_targets(
                candidate,
                observed_all,
                world_aabbs,
                surface_points,
                instance_to_glb,
                camera_world,
                camera_forward,
                float(camera_view[3]),
                float(camera_view[4]),
                int(cache_meta["width"]),
                int(cache_meta["height"]),
                coarse_points=min(16, int(surface_points.shape[1])),
                max_targets=int(getattr(args, "surface_max_targets", 256)),
            )
            if fallback_targets.size:
                fallback_pose_count += 1
                fallback_target_count += int(fallback_targets.size)
            for target in fallback_targets.tolist():
                glb_id = int(instance_to_glb[int(target)])
                relations = collect_front_surface_relations(
                    int(target),
                    observed_all,
                    surface_points[glb_id, : int(getattr(args, "surface_points_per_target", 64))],
                    world_aabbs[int(target)],
                    camera_world,
                    camera_forward,
                    float(camera_view[3]),
                    float(camera_view[4]),
                    ids[0],
                    depths[0],
                    int(cache_meta["width"]),
                    int(cache_meta["height"]),
                    min_depth_gap=float(args.min_depth_gap),
                )
                relation_count_for_target = 0.0
                for source, target_id, gap, count in relations:
                    if source not in candidate_set or target_id not in candidate_set:
                        continue
                    direction = int(directions[target_id])
                    distance = float(np.linalg.norm(centers[target_id] - camera_world))
                    target_depth = max(distance, 1e-4)
                    relative_gap = max(0.0, float(gap)) / target_depth
                    shell = _shell_id(relative_gap, depth_shells)
                    score = float(count) / max(1.0, float(getattr(args, "surface_points_per_target", 64)))
                    score *= float(np.clip(float(gap) / 0.05, 0.0, 1.0))
                    _topk_insert(source_scores, source_ids, source_gaps, target_id, direction, shell, source, score, relative_gap, source_k)
                    topk_entries[(target_id, direction, shell)].append({
                        "source": source,
                        "pixelMass": float(count),
                        "depthGap": relative_gap,
                        "confidence": float(np.clip(float(gap) / 0.05, 0.0, 1.0)),
                        "evidenceLevel": 1,
                    })
                    total_evidence_mass[target_id, direction, shell] += float(count)
                    evidence_strength[target_id, direction, shell] += score
                    evidence_count[target_id, direction, shell] += 1
                    relation_count_for_target += float(count)
                    fallback_relation_count += 1
                    fallback_relation_rows.append((
                        int(render_pose_id),
                        int(source),
                        int(target_id),
                        int(count),
                        float(gap),
                        0.0,
                    ))
                if relation_count_for_target > 0.0:
                    distance = float(np.linalg.norm(centers[int(target)] - camera_world))
                    observation_instances.append(int(target))
                    observation_directions.append(int(directions[int(target)]))
                    observation_depth.append(float(max(distance, 0.0) / max(float(radii[int(target)]), 1e-6)))
                    observation_event.append(1)
                    observation_weight.append(float(max(relation_count_for_target, 1.0)))
                    observation_subpose.append(int(render_pose_id))
                    observation_raw_pixels.append(int(max(relation_count_for_target, 1.0)))
                    observation_confidence.append(float(np.clip(relation_count_for_target / max(float(cache_meta["width"] * cache_meta["height"]), 1.0), 0.0, 1.0)))
                    observation_evidence_level.append(1)

        if entry_index == 1 or entry_index % progress_interval == 0 or entry_index == len(selected_entries):
            elapsed = max(1e-6, time.perf_counter() - started_at)
            rate = entry_index / elapsed
            remaining = (len(selected_entries) - entry_index) / max(rate, 1e-6)
            print(json.dumps({
                "status": "triangle_evidence_progress",
                "processed": entry_index,
                "total": len(selected_entries),
                "ratePosesPerSecond": rate,
                "etaSeconds": remaining,
                "fallbackTargets": fallback_target_count,
                "fallbackRelations": fallback_relation_count,
            }, ensure_ascii=False), flush=True)

    # Collapse duplicate source observations before selecting top-k.  Ranking
    # is by real retained pixel mass, with quality retained beside the table.
    for (target, direction, shell), entries in topk_entries.items():
        by_source: dict[int, dict[str, Any]] = {}
        for entry in entries:
            source = int(entry["source"])
            current = by_source.setdefault(source, {
                "source": source, "pixelMass": 0.0, "depthGap": 0.0,
                "confidence": 0.0, "evidenceLevel": 0, "_weightedGap": 0.0,
            })
            mass = float(entry["pixelMass"])
            current["pixelMass"] += mass
            current["_weightedGap"] += mass * float(entry["depthGap"])
            current["confidence"] = max(float(current["confidence"]), float(entry.get("confidence", 0.0)))
            current["evidenceLevel"] = max(int(current["evidenceLevel"]), int(entry.get("evidenceLevel", 0)))
        compact = list(by_source.values())
        for entry in compact:
            entry["depthGap"] = float(entry["_weightedGap"] / max(float(entry["pixelMass"]), 1e-6))
        kept, total_mass, quality, truncated = select_topk_evidence(compact, source_k)
        total_evidence_mass[target, direction, shell] = float(total_mass)
        retained_evidence_mass[target, direction, shell] = float(sum(float(item["pixelMass"]) for item in kept))
        retained_quality[target, direction, shell] = float(quality)
        topk_truncated[target, direction, shell] = np.uint8(truncated)
        source_counts[target, direction, shell] = np.uint32(len(compact))
        source_ids[target, direction, shell, :] = 0
        source_scores[target, direction, shell, :] = 0.0
        source_gaps[target, direction, shell, :] = 0.0
        source_confidences[target, direction, shell, :] = 0.0
        source_evidence_levels[target, direction, shell, :] = 0
        source_layer_counts[target, direction, shell, :] = 0
        for position, entry in enumerate(kept):
            source_ids[target, direction, shell, position] = np.uint32(entry["source"])
            source_scores[target, direction, shell, position] = np.float32(entry["pixelMass"] / max(total_mass, 1.0))
            source_gaps[target, direction, shell, position] = np.float32(entry["depthGap"])
            source_confidences[target, direction, shell, position] = np.float32(entry["confidence"])
            source_evidence_levels[target, direction, shell, position] = np.uint8(entry["evidenceLevel"])
            source_layer_counts[target, direction, shell, position] = np.uint8(entry["evidenceLevel"])

    # Surface fallback rows are additional evidence for the same render
    # subpose. Normalize only after all sources have been appended.
    if observation_subpose:
        subposes = np.asarray(observation_subpose, dtype=np.int64)
        raw = np.asarray(observation_raw_pixels, dtype=np.float64)
        weights = np.zeros_like(raw)
        for subpose in np.unique(subposes):
            mask = subposes == subpose
            weights[mask] = raw[mask] / max(float(raw[mask].sum()), 1.0)
        observation_weight = weights.tolist()
    strength = evidence_strength / np.maximum(evidence_count, 1.0)
    files = {
        "sourceIds": "source_ids_uint32.bin",
        "sourceScores": "source_scores_fp16.bin",
        "sourceDepthGaps": "source_depth_gaps_fp16.bin",
        "sourceConfidences": "source_confidences_fp16.bin",
        "sourceEvidenceLevels": "source_evidence_levels_uint8.bin",
        "sourceLayerCounts": "source_layer_counts_uint8.bin",
        "evidenceStrength": "evidence_strength_fp16.bin",
        "evidenceCount": "evidence_count_uint32.bin",
        "totalEvidenceMass": "total_evidence_mass_fp32.bin",
        "retainedEvidenceMass": "retained_evidence_mass_fp32.bin",
        "retainedQuality": "retained_quality_fp16.bin",
        "sourceCounts": "source_counts_uint32.bin",
        "topkTruncated": "topk_truncated_uint8.bin",
        "observationInstances": "survival_observation_instances_uint32.bin",
        "observationDirections": "survival_observation_directions_uint8.bin",
        "observationDepth": "survival_observation_depth_fp32.bin",
        "observationNormalizedDepth": "survival_observation_normalized_depth_fp16.bin",
        "observationEvent": "survival_observation_event_uint8.bin",
        "observationWeight": "survival_observation_weight_fp16.bin",
        "observationSubpose": "survival_observation_subpose_uint32.bin",
        "observationRawPixels": "survival_observation_raw_pixels_uint32.bin",
        "observationConfidence": "survival_observation_confidence_fp16.bin",
        "observationEvidenceLevel": "survival_observation_evidence_level_uint8.bin",
        "surfaceFallbackRelations": "surface_fallback_relations_v1.bin",
    }
    source_ids.tofile(output_dir / files["sourceIds"])
    source_scores.astype(np.float16).tofile(output_dir / files["sourceScores"])
    source_gaps.astype(np.float16).tofile(output_dir / files["sourceDepthGaps"])
    source_confidences.astype(np.float16).tofile(output_dir / files["sourceConfidences"])
    source_evidence_levels.tofile(output_dir / files["sourceEvidenceLevels"])
    source_layer_counts.tofile(output_dir / files["sourceLayerCounts"])
    strength.astype(np.float16).tofile(output_dir / files["evidenceStrength"])
    evidence_count.tofile(output_dir / files["evidenceCount"])
    total_evidence_mass.astype(np.float32).tofile(output_dir / files["totalEvidenceMass"])
    retained_evidence_mass.astype(np.float32).tofile(output_dir / files["retainedEvidenceMass"])
    retained_quality.astype(np.float16).tofile(output_dir / files["retainedQuality"])
    source_counts.tofile(output_dir / files["sourceCounts"])
    topk_truncated.tofile(output_dir / files["topkTruncated"])
    np.asarray(observation_instances, dtype="<u4").tofile(output_dir / files["observationInstances"])
    np.asarray(observation_directions, dtype=np.uint8).tofile(output_dir / files["observationDirections"])
    np.asarray(observation_depth, dtype=np.float32).tofile(output_dir / files["observationDepth"])
    raw_depth_values = np.asarray(observation_depth, dtype=np.float64)
    depth_stats = train_depth_quantiles(np.log1p(raw_depth_values)) if observation_depth else train_depth_quantiles(np.asarray([0.0], dtype=np.float64))
    normalize_radius_relative_log_depth(
        np.log1p(raw_depth_values), depth_stats
    ).astype(np.float16).tofile(output_dir / files["observationNormalizedDepth"])
    validate_survival_observations_v3(
        {
            "instance": np.asarray(observation_instances, dtype="<u4"),
            "direction": np.asarray(observation_directions, dtype=np.uint8),
            "normalizedDepth": np.asarray(normalize_radius_relative_log_depth(np.log1p(raw_depth_values), depth_stats), dtype=np.float16),
            "event": np.asarray(observation_event, dtype=np.uint8),
            "weight": np.asarray(observation_weight, dtype=np.float16),
            "subpose": np.asarray(observation_subpose, dtype="<u4"),
            "rawPixelCount": np.asarray(observation_raw_pixels, dtype="<u4"),
        },
        num_instances=num_instances,
        direction_bins=direction_bins,
    )
    np.asarray(observation_event, dtype=np.uint8).tofile(output_dir / files["observationEvent"])
    np.asarray(observation_weight, dtype=np.float16).tofile(output_dir / files["observationWeight"])
    np.asarray(observation_subpose, dtype="<u4").tofile(output_dir / files["observationSubpose"])
    np.asarray(observation_raw_pixels, dtype="<u4").tofile(output_dir / files["observationRawPixels"])
    np.asarray(observation_confidence, dtype=np.float16).tofile(output_dir / files["observationConfidence"])
    np.asarray(observation_evidence_level, dtype=np.uint8).tofile(output_dir / files["observationEvidenceLevel"])
    fallback_array = np.asarray(fallback_relation_rows, dtype=SURFACE_FALLBACK_RELATION_DTYPE)
    fallback_array.tofile(output_dir / files["surfaceFallbackRelations"])

    pose_indices = np.asarray([entry[2] for entry in selected_entries], dtype="<i8")
    canonical_pose_indices = pose_sequence_for_splits(dataset, requested_split_names)
    candidate_audit = audit_native_aabb_candidates(
        dataset,
        world_aabbs,
        canonical_pose_indices,
    )
    canonical_digest = candidate_digest_for_pose_sequence(dataset, canonical_pose_indices)
    render_digest = candidate_digest_for_pose_sequence(dataset, pose_indices)
    cache_identity = cache_meta.get("candidateIdentity") or {}
    if cache_identity.get("renderCandidateDigest") and cache_identity["renderCandidateDigest"] != render_digest:
        raise ValueError("triangle depth cache render candidate digest does not match its source manifest")
    meta: dict[str, Any] = {
        "schema": SCHEMA,
        "legacySchema": LEGACY_SCHEMA,
        "cacheSchema": cache_meta["schema"],
        "gpuEvidenceSummary": cache_meta.get("gpuEvidenceSummary"),
        "datasetDir": dataset_dir.as_posix(),
        "runtimeMeta": Path(args.runtime_meta).as_posix(),
        "numInstances": num_instances,
        "directionBins": direction_bins,
        "depthShells": depth_shells,
        "sourceK": source_k,
        "sourceIdDtype": "uint32",
        "splits": requested_split_names,
        "modelInputFovYDeg": 66.0,
        "formalCandidatePolicy": "stored back-camera candidate CSR; no GT-visible union and no frontend whitelist",
        "candidateAudit": candidate_audit,
        "evidenceSemantics": "adjacent different component IDs from triangle depth peeling plus optional actual GLB surface-point depth comparison for uncovered targets; AABB overlap is not used",
        "survivalSemantics": "layer-zero observations are right-censored and deeper-layer observations are independent occlusion events; both may coexist for one instance/subpose",
        "pixelWeightPolicy": "per_render_subpose_sum_to_one",
        "depthDefinition": "metric per-pixel camera-ray range divided by target radius; normalizedDepth stores clipped train-only log1p quantiles",
        "sourceDepthEncoding": cache_meta.get("linearDepthEncoding"),
        "cameraFarMeters": float(cache_meta["cameraFarMeters"]),
        "depthNormalization": depth_stats,
        "candidateHashScope": "canonical stored candidate CSR rows in the requested split order",
        "candidateDigest": canonical_digest,
        "candidateDigestScope": "canonical PoseCSR split rows; repeated representative render rows are not included",
        "renderCandidateDigest": render_digest,
        "renderCandidateDigestScope": "stored candidate CSR rows repeated in cache renderPose order",
        "candidateIdentity": {
            "canonicalCandidateDigest": canonical_digest,
            "renderCandidateDigest": render_digest,
            "canonicalPoseCount": int(canonical_pose_indices.size),
            "renderPoseCount": int(pose_indices.size),
        },
        "firstLayerValidation": {
            "status": "passed" if not first_layer_failures else "outside_candidate_detected",
            "failureCount": len(first_layer_failures),
            "failures": first_layer_failures[:32],
        },
        "stats": {
            "poseCount": int(len(selected_entries)),
            "renderPoseCount": int(len(selected_entries)),
            "sourcePoseCount": int(np.unique(pose_indices).size),
            "representativeSubposeCache": bool(cache_meta.get("schema") in (CACHE_SCHEMA_V2, CACHE_SCHEMA)),
            "layerPairCount": int(total_layer_pairs),
            "layerPairPixelCount": int(total_pixels),
            "nonzeroEvidenceCells": int((evidence_count > 0).sum()),
            "survivalObservationCount": len(observation_instances),
            "survivalEventCount": int(np.sum(np.asarray(observation_event, dtype=np.uint8))),
            "surfaceFallbackRelationCount": int(fallback_relation_count),
            "surfaceFallbackTargetCount": int(fallback_target_count),
            "surfaceFallbackPoseCount": int(fallback_pose_count),
            "depthMonotonicityViolations": int(depth_monotonicity_violations),
            "survivalWeightSum": float(np.sum(np.asarray(observation_weight, dtype=np.float64))),
            "survivalSubposeCount": int(np.unique(np.asarray(observation_subpose, dtype=np.int64)).size) if observation_subpose else 0,
            "survivalRawEvidenceMass": float(np.sum(np.asarray(observation_raw_pixels, dtype=np.float64))),
        },
        "evidenceTopK": {
            "k": int(source_k),
            "ranking": "descending aggregated raw pixel mass per target/direction/depth shell",
            "qualityDefinition": "retainedEvidenceMass / totalEvidenceMass",
            "files": {
                "totalEvidenceMass": files["totalEvidenceMass"],
                "retainedEvidenceMass": files["retainedEvidenceMass"],
                "retainedQuality": files["retainedQuality"],
                "sourceCounts": files["sourceCounts"],
                "truncated": files["topkTruncated"],
            },
        },
        "survivalObservations": {
            "schema": "triangle-depth-layer-survival-observation-v3",
            "depthField": "normalizedDepth",
            "rawDepthField": "depth",
            "normalization": depth_stats,
            "eventAndCensorMayCoexist": True,
        },
        "surfacePointFallback": surface_fallback_config,
        "surfaceFallbackRelationSchema": SURFACE_FALLBACK_RELATION_SCHEMA,
        "files": files,
        "cachePoseIdSemantics": cache_meta.get("poseIdSemantics", "poseIndices stores canonical PoseCSR poseIndex"),
    }
    (output_dir / "evidence_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_triangle_depth_evidence(path: str | Path, num_instances: int) -> dict[str, Any]:
    root = Path(path)
    meta = json.loads((root / "evidence_meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}; historical {LEGACY_SCHEMA} requires an explicit legacy loader")
    n = int(num_instances)
    bins = int(meta["directionBins"])
    shells = int(meta["depthShells"])
    k = int(meta["sourceK"])
    files = meta["files"]
    source_ids = np.fromfile(root / files["sourceIds"], dtype="<u4").reshape(n, bins, shells, k).astype(np.int64)
    source_scores = np.fromfile(root / files["sourceScores"], dtype=np.float16).reshape(n, bins, shells, k).astype(np.float32)
    source_gaps = np.fromfile(root / files["sourceDepthGaps"], dtype=np.float16).reshape(n, bins, shells, k).astype(np.float32)
    source_confidences = np.fromfile(root / files["sourceConfidences"], dtype=np.float16).reshape(n, bins, shells, k).astype(np.float32)
    source_evidence_levels = np.fromfile(root / files["sourceEvidenceLevels"], dtype=np.uint8).reshape(n, bins, shells, k)
    source_layer_counts = np.fromfile(root / files["sourceLayerCounts"], dtype=np.uint8).reshape(n, bins, shells, k)
    strength = np.fromfile(root / files["evidenceStrength"], dtype=np.float16).reshape(n, bins, shells).astype(np.float32)
    count = np.fromfile(root / files["evidenceCount"], dtype="<u4").reshape(n, bins, shells)
    total_mass = np.fromfile(root / files["totalEvidenceMass"], dtype="<f4").reshape(n, bins, shells)
    retained_mass = np.fromfile(root / files["retainedEvidenceMass"], dtype="<f4").reshape(n, bins, shells)
    quality = np.fromfile(root / files["retainedQuality"], dtype=np.float16).reshape(n, bins, shells).astype(np.float32)
    source_count = np.fromfile(root / files["sourceCounts"], dtype="<u4").reshape(n, bins, shells)
    truncated = np.fromfile(root / files["topkTruncated"], dtype=np.uint8).reshape(n, bins, shells)
    obs_count = int(meta["stats"]["survivalObservationCount"])
    observations = {
        "instance": np.fromfile(root / files["observationInstances"], dtype="<u4").astype(np.int64),
        "direction": np.fromfile(root / files["observationDirections"], dtype=np.uint8).astype(np.int64),
        "depth": np.fromfile(root / files["observationDepth"], dtype="<f4").astype(np.float32),
        "normalizedDepth": np.fromfile(root / files["observationNormalizedDepth"], dtype=np.float16).astype(np.float32),
        "event": np.fromfile(root / files["observationEvent"], dtype=np.uint8).astype(np.float32),
        "weight": np.fromfile(root / files["observationWeight"], dtype=np.float16).astype(np.float32),
        "subpose": np.fromfile(root / files["observationSubpose"], dtype="<u4").astype(np.int64),
        "rawPixelCount": np.fromfile(root / files["observationRawPixels"], dtype="<u4").astype(np.float32),
        "confidence": np.fromfile(root / files["observationConfidence"], dtype=np.float16).astype(np.float32),
        "evidenceLevel": np.fromfile(root / files["observationEvidenceLevel"], dtype=np.uint8),
    }
    if any(value.size != obs_count for value in observations.values()):
        raise ValueError("survival observation arrays have inconsistent lengths")
    return {
        "meta": meta,
        "source_ids": source_ids,
        "source_scores": source_scores,
        "source_depth_gaps": source_gaps,
        "source_confidences": source_confidences,
        "source_evidence_levels": source_evidence_levels,
        "source_layer_counts": source_layer_counts,
        "strength": strength,
        "count": count,
        "total_evidence_mass": total_mass,
        "retained_evidence_mass": retained_mass,
        "retained_quality": quality,
        "source_count": source_count,
        "topk_truncated": truncated,
        "observations": observations,
    }


def load_surface_fallback_relations(path: str | Path) -> np.ndarray:
    """Load per-render fallback relations, if the evidence has them."""
    root = Path(path)
    meta = json.loads((root / "evidence_meta.json").read_text(encoding="utf-8"))
    if meta.get("surfaceFallbackRelationSchema") != SURFACE_FALLBACK_RELATION_SCHEMA:
        return np.zeros((0,), dtype=SURFACE_FALLBACK_RELATION_DTYPE)
    filename = (meta.get("files") or {}).get("surfaceFallbackRelations")
    if not filename:
        return np.zeros((0,), dtype=SURFACE_FALLBACK_RELATION_DTYPE)
    relation_path = root / str(filename)
    if not relation_path.is_file():
        raise FileNotFoundError(f"missing surface fallback relation array: {relation_path}")
    raw = relation_path.read_bytes()
    item_size = SURFACE_FALLBACK_RELATION_DTYPE.itemsize
    if len(raw) % item_size:
        raise ValueError("surface fallback relation array has a partial record")
    return np.fromfile(relation_path, dtype=SURFACE_FALLBACK_RELATION_DTYPE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--layer-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="train")
    parser.add_argument("--source-k", type=int, default=8)
    parser.add_argument("--min-depth-gap", type=float, default=1e-4)
    parser.add_argument("--max-poses", type=int, default=0)
    parser.add_argument("--allow-outside-candidate", action="store_true")
    parser.add_argument("--surface-point-fallback", action="store_true")
    parser.add_argument("--glb-points", default=None)
    parser.add_argument("--surface-points-per-target", type=int, default=64)
    parser.add_argument("--surface-max-targets", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(build_evidence(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
