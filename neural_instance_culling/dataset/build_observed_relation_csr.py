#!/usr/bin/env python3
"""Build the train-only observed occlusion relation CSR for the new model.

The source is the registered triangle depth-layer cache.  Every relation is
kept within the cache's observed coverage; this command never adds a visible
instance to the native candidate set and never uses AABB overlap as an
occlusion label.  The resulting graph is an offline training artifact.  The
browser receives only the coefficients produced by the offline relation
encoder.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATASET_DIR = ROOT / "dataset"
for value in (ROOT, MODEL_DIR, DATASET_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_triangle_depth_layer_evidence import (  # noqa: E402
    BACKGROUND_ID,
    CACHE_SCHEMA,
    CACHE_SCHEMA_V2,
    _camera_for_cache_row,
    build_survival_evidence_records,
    load_layer_cache,
    load_surface_fallback_relations,
    metric_ray_depths_for_cache_row,
    spherical_direction_bins,
)
from build_ray_context_relation_evidence import (  # noqa: E402
    _reduce_pose_events,
    _surface_fallback_pose_rows,
)
from common.candidate_identity import (  # noqa: E402
    audit_native_aabb_candidates,
    pose_sequence_for_splits,
)
from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from common.train_observed_relation_csr import (  # noqa: E402
    SOURCE_TYPES,
    ObservedRelationCSR,
    PIXEL_SUPPORT_FEATURE,
    CONFIDENCE_FEATURE,
    SCHEMA_V3,
    bounded_hierarchy_ids,
    make_relation_metadata_v3,
    normalize_radius_relative_log_depth,
    radius_relative_log_depth,
    summarize_bounded_hierarchy,
    summarize_candidate_csr,
    train_depth_quantiles,
    validate_survival_observations_v3,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


SCHEMA = SCHEMA_V3
DIRECTION_BINS = 12
DEPTH_SHELLS = 3
LOCAL_DIAMETER_QUANTILE = 0.75
STRUCTURAL_DIAMETER_QUANTILE = 0.95


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_summary(dataset: PoseCSRDataset, poses: np.ndarray, num_instances: int):
    poses = np.asarray(poses, dtype="<i8").reshape(-1)
    rows = [np.asarray(dataset.frustum_slice(int(p)), dtype="<u4") for p in poses.tolist()]
    offsets = np.zeros((poses.size + 1,), dtype="<i8")
    if rows:
        offsets[1:] = np.cumsum([row.size for row in rows], dtype=np.int64)
        ids = np.concatenate(rows).astype("<u4", copy=False)
    else:
        ids = np.zeros((0,), dtype="<u4")
    return summarize_candidate_csr(ids, poses, offsets, num_instances)


def _input_hashes(
    dataset_dir: Path,
    runtime_meta_path: Path,
    cache_dir: Path,
    cache_meta: dict[str, Any],
    fallback_dir: Path | None,
) -> dict[str, str]:
    """Hash the exact provenance envelopes used by the builder."""
    paths: dict[str, Path] = {
        "datasetMeta": dataset_dir / "dataset_meta.json",
        "runtimeMeta": runtime_meta_path,
        "layerCacheMeta": cache_dir / "layer_cache_meta.json",
    }
    for key, name in (cache_meta.get("files") or {}).items():
        candidate = cache_dir / str(name)
        if candidate.is_file():
            paths[f"layerCache:{key}"] = candidate
    if fallback_dir is not None:
        evidence_meta = fallback_dir / "evidence_meta.json"
        if evidence_meta.is_file():
            paths["surfaceFallbackMeta"] = evidence_meta
        fallback_files = {}
        if evidence_meta.is_file():
            fallback_files = json.loads(evidence_meta.read_text(encoding="utf-8")).get("files") or {}
        for key, name in fallback_files.items():
            candidate = fallback_dir / str(name)
            if candidate.is_file():
                paths[f"surfaceFallback:{key}"] = candidate
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"relation provenance file(s) are missing: {missing}")
    return {key: _sha256(path) for key, path in paths.items()}


def _update_moment(
    accumulator: dict[str, Any],
    *,
    pose_id: int,
    pixels: float,
    gap: float,
    relative_gap: float,
    relative_center: np.ndarray,
    relative_log_scale: float,
    relative_log_depth: float,
    source_type: int,
) -> None:
    weight = max(float(pixels), 1.0)
    accumulator["poses"].add(int(pose_id))
    accumulator["pixels"] += float(max(pixels, 0.0))
    accumulator["weight"] += weight
    accumulator["gap_sum"] += weight * float(gap)
    accumulator["gap_sq_sum"] += weight * float(gap) ** 2
    accumulator["relative_gap_sum"] += weight * float(relative_gap)
    accumulator["relative_gap_sq_sum"] += weight * float(relative_gap) ** 2
    accumulator["center_sum"] += weight * np.asarray(relative_center, dtype=np.float64)
    accumulator["scale_sum"] += weight * float(relative_log_scale)
    accumulator["depth_values"].append(float(relative_log_depth))
    if accumulator["source_type"] < 0:
        accumulator["source_type"] = int(source_type)
    elif accumulator["source_type"] != int(source_type):
        accumulator["source_type"] = SOURCE_TYPES["merged"]


def _accumulator() -> dict[str, Any]:
    return {
        "poses": set(),
        "pixels": 0.0,
        "weight": 0.0,
        "gap_sum": 0.0,
        "gap_sq_sum": 0.0,
        "relative_gap_sum": 0.0,
        "relative_gap_sq_sum": 0.0,
        "center_sum": np.zeros((3,), dtype=np.float64),
        "scale_sum": 0.0,
        "depth_values": [],
        "source_type": -1,
    }


def truncate_relation_topk(
    target_ids: Any,
    direction_ids: Any,
    shell_ids: Any,
    source_ids: Any,
    edge_features: Any,
    *,
    k: int,
    score_indices: tuple[int, int] = (PIXEL_SUPPORT_FEATURE, CONFIDENCE_FEATURE),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the formal per-target/direction/shell evidence top-k.

    The score multiplies cell-normalized confidence, logarithmic pixel
    support, and logarithmic pose support. Ties are broken by source ID.
    """
    if int(k) <= 0:
        raise ValueError("relation top-k must be positive")
    target = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    direction = np.asarray(direction_ids, dtype=np.int64).reshape(-1)
    shell = np.asarray(shell_ids, dtype=np.int64).reshape(-1)
    source = np.asarray(source_ids, dtype=np.uint32).reshape(-1)
    features = np.asarray(edge_features, dtype=np.float32)
    if features.ndim != 2 or not (target.size == direction.size == shell.size == source.size == features.shape[0]):
        raise ValueError("relation top-k arrays have incompatible shapes")
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(target.tolist(), direction.tolist(), shell.tolist(), strict=True)):
        groups[key].append(index)
    keep: list[int] = []
    quality: dict[str, dict[str, float | int | bool]] = {}
    for key in sorted(groups):
        indices = groups[key]
        local = features[indices]
        pixel_component = np.log1p(np.maximum(local[:, int(score_indices[0])], 0.0))
        confidence_component = np.maximum(local[:, int(score_indices[1])], 0.0)
        pose_component = np.log1p(1.0 + np.maximum(local[:, 0], 0.0))

        def normalized(values: np.ndarray) -> np.ndarray:
            maximum = float(np.max(values)) if values.size else 0.0
            return values / maximum if maximum > 0.0 else np.zeros_like(values)

        local_score = (
            normalized(pixel_component)
            * normalized(confidence_component)
            * normalized(pose_component)
        )
        score_by_index = {
            original: float(local_score[position])
            for position, original in enumerate(indices)
        }
        ordered = sorted(
            indices, key=lambda i: (-score_by_index[i], int(source[i]))
        )
        total = float(np.sum(local_score, dtype=np.float64))
        chosen = ordered[: int(k)]
        retained = float(sum(score_by_index[index] for index in chosen))
        retained_quality = float(np.clip(retained / total, 0.0, 1.0)) if total > 0.0 else 1.0
        cell = ":".join(str(int(value)) for value in key)
        quality[cell] = {
            "totalScore": total,
            "retainedScore": retained,
            "retainedQuality": retained_quality,
            "totalCount": len(indices),
            "retainedCount": len(chosen),
            "truncated": len(indices) > int(k),
        }
        keep.extend(chosen)
    keep_array = np.asarray(sorted(keep, key=lambda i: (target[i], direction[i], shell[i], source[i])), dtype=np.int64)
    out_features = features[keep_array].copy()
    for row, original in enumerate(keep_array.tolist()):
        key = ":".join(str(int(value)) for value in (target[original], direction[original], shell[original]))
        out_features[row, 13] = np.float32(quality[key]["retainedQuality"])
        out_features[row, 16] = np.float32(bool(quality[key]["truncated"]))
        out_features[row, 19] = np.float32(quality[key]["totalCount"])
    return (
        target[keep_array], direction[keep_array], shell[keep_array],
        source[keep_array], out_features, {
            "k": int(k),
            "score": "cell_normalized(evidence_confidence) * cell_normalized(log1p(pixel_support)) * cell_normalized(log1p(1 + pose_support_count))",
            "cells": quality,
        }
    )


def derive_hierarchy_diameter_limits(
    centers: Any,
    target_ids: Any,
    source_ids: Any,
    *,
    local_override: float = 0.0,
    structural_override: float = 0.0,
) -> tuple[float, float, dict[str, Any]]:
    """Freeze finite hierarchy scales from train-only observed edges."""
    points = np.asarray(centers, dtype=np.float64)
    target = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    source = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or target.size == 0 or target.size != source.size:
        raise ValueError("hierarchy diameter derivation requires observed edges and instance centers")
    edge_distance = np.linalg.norm(points[source] - points[target], axis=1)
    edge_distance = edge_distance[np.isfinite(edge_distance) & (edge_distance > 0.0)]
    if edge_distance.size == 0:
        raise ValueError("train-only observed edges have no finite positive center distances")
    derived_local = float(np.quantile(edge_distance, LOCAL_DIAMETER_QUANTILE))
    derived_structural = float(np.quantile(edge_distance, STRUCTURAL_DIAMETER_QUANTILE))
    local = float(local_override) if float(local_override) > 0.0 else derived_local
    structural = float(structural_override) if float(structural_override) > 0.0 else derived_structural
    local = max(local, 1e-4)
    structural = max(structural, local)
    if not np.isfinite(local) or not np.isfinite(structural):
        raise ValueError("hierarchy diameter limits must be finite")
    return local, structural, {
        "source": "train-only retained observed relation edge center distances",
        "metric": "world-space instance-center Euclidean distance",
        "edgeCount": int(edge_distance.size),
        "localQuantile": LOCAL_DIAMETER_QUANTILE,
        "structuralQuantile": STRUCTURAL_DIAMETER_QUANTILE,
        "derivedLocalDiameter": derived_local,
        "derivedStructuralDiameter": derived_structural,
        "localOverride": float(local_override) if float(local_override) > 0.0 else None,
        "structuralOverride": float(structural_override) if float(structural_override) > 0.0 else None,
    }


def summarize_topk_diagnostics_for_metadata(
    diagnostics: dict[str, Any],
    *,
    worst_cell_limit: int = 64,
) -> dict[str, Any]:
    """Keep formal top-k diagnostics bounded while preserving worst cases."""
    cells = diagnostics.get("cells") or {}
    ordered = sorted(
        cells.items(),
        key=lambda item: (float(item[1]["retainedQuality"]), item[0]),
    )
    qualities = np.asarray(
        [float(value["retainedQuality"]) for _, value in ordered],
        dtype=np.float64,
    )
    truncated_count = sum(bool(value["truncated"]) for _, value in ordered)
    return {
        "k": int(diagnostics["k"]),
        "score": str(diagnostics["score"]),
        "cellCount": len(ordered),
        "truncatedCellCount": int(truncated_count),
        "retainedQualityQuantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "q01", "q05", "q50", "q95", "q99", "max"),
                np.quantile(qualities, (0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0))
                if qualities.size
                else np.ones((7,), dtype=np.float64),
                strict=True,
            )
        },
        "worstCells": {
            key: value for key, value in ordered[: max(0, int(worst_cell_limit))]
        },
    }


def _connected_components(
    num_nodes: int,
    target_ids: np.ndarray,
    source_ids: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> np.ndarray:
    parent = np.arange(int(num_nodes), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for target, source, score in zip(target_ids.tolist(), source_ids.tolist(), confidence.tolist(), strict=True):
        if float(score) < float(threshold):
            continue
        left, right = find(int(target)), find(int(source))
        if left != right:
            parent[right] = left
    roots = np.asarray([find(index) for index in range(int(num_nodes))], dtype=np.int64)
    _, inverse = np.unique(roots, return_inverse=True)
    return inverse.astype(np.uint32, copy=False)


def _hierarchy_ids(
    num_instances: int,
    target_ids: np.ndarray,
    source_ids: np.ndarray,
    confidence: np.ndarray,
    local_threshold: float,
    structure_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create instance->local and local->structure contiguous mappings."""
    local = _connected_components(
        num_instances, target_ids, source_ids, confidence, local_threshold
    )
    local_count = int(local.max()) + 1 if local.size else 0
    if local_count == 0:
        return local, np.zeros((0,), dtype=np.uint32)
    local_edges_target = local[target_ids.astype(np.int64, copy=False)]
    local_edges_source = local[source_ids.astype(np.int64, copy=False)]
    keep = local_edges_target != local_edges_source
    structural = _connected_components(
        local_count,
        local_edges_target[keep],
        local_edges_source[keep],
        confidence[keep],
        structure_threshold,
    )
    return local, structural


def _observations_for_pose(
    ids: np.ndarray,
    depths: np.ndarray,
    candidate_set: set[int],
    directions: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    camera_world: np.ndarray,
    min_depth_gap: float,
    *,
    pose_id: int,
) -> list[dict[str, Any]]:
    """Return independent conserved censor/event records for one subpose."""
    records, _mass = build_survival_evidence_records(
        ids, depths, candidate_set, directions, centers, radii, camera_world,
        min_depth_gap, subpose_id=pose_id,
    )
    return records


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    runtime_meta_path = Path(args.runtime_meta)
    cache_dir = Path(args.layer_cache_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [value.strip() for value in str(args.splits).split(",") if value.strip()]
    if requested != ["train"]:
        raise ValueError("the hierarchical relation CSR is train-only; use --splits train")

    world_aabbs, _instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    num_instances = int(world_aabbs.shape[0])
    centers = ((world_aabbs[:, :3] + world_aabbs[:, 3:]) * 0.5).astype(np.float32)
    extents = np.maximum(world_aabbs[:, 3:] - world_aabbs[:, :3], 1e-4)
    radii = np.linalg.norm(extents, axis=1) * 0.5
    scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    scene_size = np.maximum(np.asarray(scene_size, dtype=np.float32), 1e-4)
    dataset = PoseCSRDataset(dataset_dir, num_instances=num_instances)
    canonical_poses = pose_sequence_for_splits(dataset, requested)

    cache_meta, render_ids, source_pose_indices, cache_ids, cache_depths = load_layer_cache(cache_dir)
    if cache_meta.get("schema") not in (CACHE_SCHEMA, CACHE_SCHEMA_V2):
        raise ValueError(f"unsupported depth cache schema: {cache_meta.get('schema')!r}")
    if not np.isclose(float(cache_meta.get("modelInputFovYDeg", 0.0)), 66.0):
        raise ValueError("observed relation CSR requires the registered model-input FOV of 66 degrees")
    if int(cache_meta.get("maxLayers", 0)) < 2:
        raise ValueError("observed relation CSR requires at least two depth layers")
    if not (cache_meta.get("gpuEvidenceSummary") or {}).get("formalReady", False):
        raise ValueError("depth cache has no formal hardware-GPU evidence")

    registered_selected = [
        (row, int(render_ids[row]), int(source_pose_indices[row]))
        for row in range(int(render_ids.size))
        if int(dataset.poses[int(source_pose_indices[row])]["split"]) == int(dataset.split_ids["train"])
    ]
    selected = registered_selected
    if args.max_poses > 0:
        selected = selected[: int(args.max_poses)]
    if not selected:
        raise ValueError("depth cache has no train poses")
    render_poses = np.asarray([entry[2] for entry in selected], dtype="<i8")
    canonical_summary = _candidate_summary(dataset, canonical_poses, num_instances)
    render_summary = _candidate_summary(dataset, render_poses, num_instances)
    # The depth cache is a registered rendering artifact.  Its candidate
    # identity is part of the provenance contract, so recomputing a summary
    # is not enough: the builder must reject a cache whose render rows no
    # longer match the native candidate rows used to produce it.
    cache_identity = cache_meta.get("candidateIdentity") or {}
    expected_canonical_digest = str(cache_identity.get("canonicalCandidateDigest", ""))
    expected_render_digest = str(cache_identity.get("renderCandidateDigest", ""))
    expected_canonical_count = int(cache_identity.get("canonicalPoseCount", -1))
    expected_render_count = int(cache_identity.get("renderPoseCount", -1))
    if not expected_canonical_digest or not expected_render_digest:
        raise ValueError("depth cache is missing its canonical/render candidate identity")
    # A split-only PoseCSR view may move poses between train, calibration, and
    # validation without changing any candidate row.  Validate the immutable
    # cache against the exact canonical/render pose coverage stored in that
    # cache, then filter its rows by the current train labels above.  The new
    # relation metadata still records the complete current train candidate
    # identity through ``canonical_summary``.
    cache_canonical_poses = np.unique(source_pose_indices.astype("<i8", copy=False))
    cache_canonical_summary = _candidate_summary(
        dataset, cache_canonical_poses, num_instances
    )
    cache_render_summary = _candidate_summary(
        dataset, source_pose_indices.astype("<i8", copy=False), num_instances
    )
    if cache_canonical_summary.digest != expected_canonical_digest:
        raise ValueError(
            "depth cache canonical candidate digest does not match the registered dataset: "
            f"{expected_canonical_digest!r} != {cache_canonical_summary.digest!r}"
        )
    if cache_render_summary.digest != expected_render_digest:
        raise ValueError(
            "depth cache render candidate digest does not match the selected render pose order: "
            f"{expected_render_digest!r} != {cache_render_summary.digest!r}"
        )
    if expected_canonical_count != int(cache_canonical_poses.size):
        raise ValueError(
            "depth cache canonical pose count does not match its stored coverage: "
            f"{expected_canonical_count} != {cache_canonical_poses.size}"
        )
    if expected_render_count != int(source_pose_indices.size):
        raise ValueError(
            "depth cache render pose count does not match its stored rows: "
            f"{expected_render_count} != {source_pose_indices.size}"
        )
    candidate_audit = audit_native_aabb_candidates(dataset, world_aabbs, canonical_poses)

    fallback_dir = Path(args.surface_fallback_dir) if args.surface_fallback_dir else None
    fallback = load_surface_fallback_relations(fallback_dir) if fallback_dir else np.zeros((0,), dtype=np.dtype([]))
    fallback_by_pose: dict[int, np.ndarray] = {}
    if fallback.size:
        for pose in np.unique(fallback["renderPoseId"]).tolist():
            fallback_by_pose[int(pose)] = fallback[fallback["renderPoseId"] == pose]

    accumulators: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    observations = {key: [] for key in ("instance", "direction", "depth", "event", "weight", "subpose", "rawPixelCount", "confidence", "evidenceLevel")}
    started = time.perf_counter()
    width = int(cache_meta["width"])
    height = int(cache_meta["height"])
    for ordinal, (cache_row, render_pose_id, pose_index) in enumerate(selected, start=1):
        ids = np.asarray(cache_ids[cache_row], dtype=np.uint32)
        encoded_depths = np.asarray(cache_depths[cache_row], dtype=np.float32)
        candidate = np.asarray(dataset.frustum_slice(pose_index), dtype=np.uint32)
        candidate_set = set(int(value) for value in candidate.tolist())
        first_unique = np.unique(ids[0][ids[0] != BACKGROUND_ID])
        outside = [int(value) for value in first_unique.tolist() if int(value) not in candidate_set]
        if outside:
            raise ValueError(f"first depth layer contains IDs outside candidates at pose {pose_index}: {outside[:8]}")
        camera_world, _camera_forward, camera_view = _camera_for_cache_row(
            cache_meta, render_pose_id, dataset, pose_index
        )
        depths = metric_ray_depths_for_cache_row(cache_meta, encoded_depths, camera_view)
        directions = spherical_direction_bins(camera_world, centers)
        pose_records = _observations_for_pose(
            ids, depths, candidate_set, directions, centers, radii, camera_world,
            float(args.min_depth_gap), pose_id=render_pose_id,
        )
        for record in pose_records:
            for key in observations:
                observations[key].append(record[key])

        rows_with_type: list[tuple[np.ndarray, int]] = [
            (_reduce_pose_events(ids, depths, np.isin(np.arange(num_instances), candidate), float(args.min_depth_gap)), SOURCE_TYPES["depth_peeling"])
        ]
        if render_pose_id in fallback_by_pose:
            extra, _ = _surface_fallback_pose_rows(
                fallback_by_pose[render_pose_id],
                np.isin(np.arange(num_instances), candidate),
                centers,
                camera_world,
                float(args.min_depth_gap),
            )
            if extra.size:
                rows_with_type.append((extra, SOURCE_TYPES["surface_fallback"]))
                for row in extra:
                    target = int(row[0])
                    observations["instance"].append(target)
                    observations["direction"].append(int(directions[target]))
                    target_distance = float(np.linalg.norm(centers[target] - camera_world))
                    observations["depth"].append(float(max(target_distance, 0.0) / max(float(radii[target]), 1e-6)))
                    observations["event"].append(1)
                    observations["weight"].append(float(max(float(row[3]), 1.0)))
                    observations["subpose"].append(int(render_pose_id))
                    observations["rawPixelCount"].append(float(max(float(row[3]), 1.0)))
                    observations["confidence"].append(float(np.clip(float(row[3]) / max(float(width * height), 1.0), 0.0, 1.0)))
                    observations["evidenceLevel"].append(1)

        for pose_rows, source_type in rows_with_type:
            if pose_rows.size == 0:
                continue
            for target_value, source_value, shell_value, pixels, gap, gap_std in pose_rows.tolist():
                target = int(target_value)
                source = int(source_value)
                if target not in candidate_set or source not in candidate_set or target == source:
                    raise ValueError("relation row escaped the stored native candidate set")
                target_distance = max(float(np.linalg.norm(centers[target] - camera_world)), 1e-4)
                relative_gap = float(gap) / target_distance
                relative_log_depth = float(radius_relative_log_depth(
                    np.asarray([target_distance]), np.asarray([radii[target]])
                )[0])
                source_radius = max(float(radii[source]), 1e-5)
                target_radius = max(float(radii[target]), 1e-5)
                relative_center = (centers[source] - centers[target]) / scene_size
                relative_scale = float(np.log(source_radius / target_radius))
                key = (target, int(directions[target]), int(shell_value), source)
                state = accumulators.setdefault(key, _accumulator())
                _update_moment(
                    state,
                    pose_id=render_pose_id,
                    pixels=float(pixels),
                    gap=float(gap),
                    relative_gap=relative_gap,
                    relative_center=relative_center,
                    relative_log_scale=relative_scale,
                    relative_log_depth=relative_log_depth,
                    source_type=source_type,
                )

        if ordinal == 1 or ordinal == len(selected) or ordinal % max(1, len(selected) // 20) == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            rate = ordinal / elapsed
            print(json.dumps({
                "status": "observed_relation_progress",
                "processed": ordinal,
                "total": len(selected),
                "relationEdges": len(accumulators),
                "ratePosesPerSecond": rate,
                "etaSeconds": (len(selected) - ordinal) / max(rate, 1e-6),
            }, ensure_ascii=False), flush=True)

    if observations["subpose"]:
        subpose_values = np.asarray(observations["subpose"], dtype=np.int64)
        raw_values = np.asarray(observations["rawPixelCount"], dtype=np.float64)
        weights = np.zeros_like(raw_values)
        for subpose in np.unique(subpose_values):
            mask = subpose_values == subpose
            weights[mask] = raw_values[mask] / max(float(raw_values[mask].sum()), 1.0)
        observations["weight"] = weights.tolist()
    if not accumulators:
        raise ValueError("no observed relation rows were generated")
    keys = sorted(accumulators)
    target_ids = np.asarray([key[0] for key in keys], dtype=np.int64)
    direction_ids = np.asarray([key[1] for key in keys], dtype=np.int64)
    shell_ids = np.asarray([key[2] for key in keys], dtype=np.int64)
    source_ids = np.asarray([key[3] for key in keys], dtype=np.uint32)
    total_pose_count = max(len(selected), 1)
    all_depth_values = np.asarray(
        [value for state in accumulators.values() for value in state["depth_values"]],
        dtype=np.float64,
    )
    depth_stats = train_depth_quantiles(all_depth_values)
    edge_features = np.zeros((len(keys), 20), dtype=np.float32)
    source_types = np.zeros((len(keys),), dtype=np.uint8)
    group_totals: dict[tuple[int, int, int], float] = defaultdict(float)
    for key, state in accumulators.items():
        group_totals[key[:3]] += float(state["pixels"])
    for index, key in enumerate(keys):
        state = accumulators[key]
        weight = max(float(state["weight"]), 1e-6)
        pose_count = len(state["poses"])
        gap_mean = state["gap_sum"] / weight
        gap_variance = max(state["gap_sq_sum"] / weight - gap_mean * gap_mean, 0.0)
        rel_mean = state["relative_gap_sum"] / weight
        rel_variance = max(state["relative_gap_sq_sum"] / weight - rel_mean * rel_mean, 0.0)
        pixel_fraction = float(state["pixels"]) / max(group_totals[key[:3]], 1.0)
        pose_rate = pose_count / total_pose_count
        depth_values = np.asarray(state["depth_values"], dtype=np.float64)
        normalized_depth = normalize_radius_relative_log_depth(
            depth_values, depth_stats
        )
        edge_features[index] = np.asarray([
            float(pose_count),
            np.clip(pose_rate, 0.0, 1.0),
            float(state["pixels"]),
            np.clip(pixel_fraction, 0.0, 1.0),
            max(gap_mean, float(args.min_depth_gap)),
            np.sqrt(gap_variance),
            max(rel_mean, float(args.min_depth_gap) / max(float(np.linalg.norm(scene_size)), 1.0)),
            np.sqrt(rel_variance),
            *np.clip(state["center_sum"] / weight, -8.0, 8.0),
            np.clip(state["scale_sum"] / weight, -8.0, 8.0),
            np.clip(np.sqrt(max(pose_rate * pixel_fraction, 0.0)), 0.0, 1.0),
            np.clip(pixel_fraction, 0.0, 1.0),
            1.0,
            float(max(1, len(state["poses"]))),
            0.0,
            float(np.mean(normalized_depth)),
            float(np.std(normalized_depth)),
            1.0,
        ], dtype=np.float32)
        source_types[index] = np.uint8(state["source_type"])

    (
        target_ids, direction_ids, shell_ids, source_ids, edge_features,
        topk_diagnostics,
    ) = truncate_relation_topk(
        target_ids, direction_ids, shell_ids, source_ids, edge_features,
        k=int(args.source_k),
    )
    # Re-index source types with the same stable retained-row order.
    retained_type = []
    for target, direction, shell, source in zip(
        target_ids.tolist(), direction_ids.tolist(), shell_ids.tolist(), source_ids.tolist(), strict=True
    ):
        state = accumulators[(int(target), int(direction), int(shell), int(source))]
        retained_type.append(int(state["source_type"]))
    source_types = np.asarray(retained_type, dtype=np.uint8)
    retained_qualities = [float(value["retainedQuality"]) for value in topk_diagnostics["cells"].values()]
    min_retained_quality = min(retained_qualities) if retained_qualities else 1.0
    compact_topk_diagnostics = summarize_topk_diagnostics_for_metadata(topk_diagnostics)
    if min_retained_quality < float(args.min_retained_quality):
        raise ValueError(
            f"relation top-k retained quality {min_retained_quality:.6f} is below "
            f"the registered floor {float(args.min_retained_quality):.6f}"
        )

    input_sha256 = _input_hashes(dataset_dir, runtime_meta_path, cache_dir, cache_meta, fallback_dir)
    local_diameter, structural_diameter, diameter_derivation = derive_hierarchy_diameter_limits(
        centers,
        target_ids,
        source_ids,
        local_override=float(args.local_group_diameter),
        structural_override=float(args.structural_group_diameter),
    )
    local_groups, structural_groups = bounded_hierarchy_ids(
        num_instances,
        target_ids,
        source_ids,
        edge_features[:, 12],
        centers,
        local_max_size=32,
        local_diameter=local_diameter,
        structural_max_size=64,
        structural_diameter=structural_diameter,
        max_structural_fraction=0.10,
        local_threshold=float(args.local_group_score),
        structural_threshold=float(args.structure_group_score),
    )
    hierarchy_stats = summarize_bounded_hierarchy(local_groups, structural_groups, centers)
    hierarchy_metadata = {
        "construction": "confidence-ordered bounded relation packing; no KNN and no unbounded connected components",
        "diameterMetric": "world-space instance-center AABB diagonal upper bound",
        "diameterDerivation": diameter_derivation,
        "localMaxSize": 32,
        "localDiameter": local_diameter,
        "structuralMaxLocalGroups": 64,
        "structuralDiameter": structural_diameter,
        "maxStructuralFraction": 0.10,
        "localThreshold": float(args.local_group_score),
        "structureThreshold": float(args.structure_group_score),
        **hierarchy_stats,
        "files": {
            "localGroupIds": "instance_local_group_ids_uint32.bin",
            "structuralGroupIds": "local_structural_group_ids_uint32.bin",
        },
    }
    metadata = make_relation_metadata_v3(
        num_instances=num_instances,
        direction_bins=DIRECTION_BINS,
        depth_shells=DEPTH_SHELLS,
        canonical_candidate_summary=canonical_summary,
        render_candidate_summary=render_summary,
        input_sha256=input_sha256,
        cache_schema=str(cache_meta["schema"]),
        max_layers=int(cache_meta["maxLayers"]),
        model_input_fov_y_deg=66.0,
        depth_normalization=depth_stats,
        hierarchy=hierarchy_metadata,
        evidence_topk={
            "k": int(args.source_k),
            "ranking": "cell-normalized confidence * log pixel support * log pose support",
            "retainedQuality": "retainedEvidenceMass / totalEvidenceMass",
            "score": topk_diagnostics["score"],
            "cellCount": len(topk_diagnostics["cells"]),
            "minRetainedQuality": float(min_retained_quality),
            "qualityFloor": float(args.min_retained_quality),
            "diagnostics": compact_topk_diagnostics,
        },
        surface_fallback={
            "included": bool(fallback_dir and fallback.size),
            "relationCount": int(fallback.size),
            "directory": str(fallback_dir.resolve()) if fallback_dir else None,
        },
    )
    metadata["candidateAudit"] = candidate_audit
    selected_unique_pose_count = int(np.unique(render_poses).size)
    metadata["cacheCoverage"] = {
        "cacheCanonicalPoseCount": int(cache_canonical_poses.size),
        "currentTrainPoseCount": int(canonical_poses.size),
        "selectedTrainRenderRowCount": int(render_poses.size),
        "selectedTrainUniquePoseCount": selected_unique_pose_count,
        "uncoveredCurrentTrainPoseCount": int(
            canonical_poses.size - selected_unique_pose_count
        ),
        "policy": (
            "filter the registered hardware depth cache to current train labels; "
            "never synthesize relations for uncovered train poses"
        ),
    }
    metadata["scene"] = {
        "sceneMin": scene_min.astype(float).tolist(),
        "sceneSize": scene_size.astype(float).tolist(),
        "directionAnchorCount": DIRECTION_BINS,
    }
    metadata["sourceTypeIds"] = {key: int(value) for key, value in SOURCE_TYPES.items()}

    relation = ObservedRelationCSR.from_rows(
        num_instances=num_instances,
        direction_bins=DIRECTION_BINS,
        depth_shells=DEPTH_SHELLS,
        target_ids=target_ids,
        direction_ids=direction_ids,
        depth_shell_ids=shell_ids,
        source_ids=source_ids,
        edge_features=edge_features,
        source_types=source_types,
        metadata=metadata,
    )
    depth_observations = np.asarray(observations["depth"], dtype=np.float64)
    if depth_observations.size == 0:
        depth_observations = np.asarray(all_depth_values, dtype=np.float64)
    normalized_observations = normalize_radius_relative_log_depth(np.log1p(depth_observations), depth_stats)
    observation_arrays = {
        key: np.asarray(values, dtype=dtype)
        for key, dtype, values in (
            ("instance", "<u4", observations["instance"]),
            ("direction", "<u1", observations["direction"]),
            ("normalizedDepth", "<f2", normalized_observations),
            ("rawDepth", "<f4", depth_observations),
            ("event", "<u1", observations["event"]),
            ("weight", "<f2", observations["weight"]),
            ("subpose", "<u4", observations["subpose"]),
            ("rawPixelCount", "<f4", observations["rawPixelCount"]),
            ("confidence", "<f2", observations["confidence"]),
            ("evidenceLevel", "<u1", observations["evidenceLevel"]),
        )
    }
    validate_survival_observations_v3(
        observation_arrays,
        num_instances=num_instances,
        direction_bins=DIRECTION_BINS,
    )
    metadata["hierarchy"] = hierarchy_metadata
    metadata["survivalObservations"] = {
        "schema": "pvs-viewcell-train-observed-survival-censoring-v3",
        "depthField": "normalizedDepth",
        "rawDepthDefinition": "metric camera-ray range / max(target radius, epsilon)",
        "normalization": metadata["depthNormalization"],
        "count": int(observation_arrays["instance"].size),
        "eventCount": int(np.sum(observation_arrays["event"] == 1)),
        "files": {
            "instance": "survival_observation_instances_uint32.bin",
            "direction": "survival_observation_directions_uint8.bin",
            "normalizedDepth": "survival_observation_normalized_depth_fp16.bin",
            "event": "survival_observation_event_uint8.bin",
            "weight": "survival_observation_weight_fp16.bin",
            "subpose": "survival_observation_subpose_uint32.bin",
            "rawPixelCount": "survival_observation_raw_pixels_fp32.bin",
            "rawDepth": "survival_observation_raw_depth_fp32.bin",
            "confidence": "survival_observation_confidence_fp16.bin",
            "evidenceLevel": "survival_observation_evidence_level_uint8.bin",
        },
    }
    metadata["stats"] = {
        "edgeCount": int(relation.edge_count),
        "rowCount": int(relation.row_count),
        "sourceTruncation": True,
        "sourceTopK": int(args.source_k),
        "topKCellCount": len(topk_diagnostics["cells"]),
        "topKMinRetainedQuality": float(min_retained_quality),
        "renderPoseCount": int(len(selected)),
        "survivalObservationCount": int(observation_arrays["instance"].size),
        "survivalEventCount": int(np.sum(observation_arrays["event"] == 1)),
    }
    relation.metadata = metadata
    relation.save(output_dir)
    local_groups.tofile(output_dir / metadata["hierarchy"]["files"]["localGroupIds"])
    structural_groups.tofile(output_dir / metadata["hierarchy"]["files"]["structuralGroupIds"])
    for key, values in observation_arrays.items():
        values.tofile(output_dir / metadata["survivalObservations"]["files"][key])
    # Rewrite metadata after the optional hierarchy and censoring fields have
    # been attached; the strict loader ignores unknown fields but preserves
    # them for the independent training entry point.
    relation.metadata = metadata
    relation.save(output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--layer-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--surface-fallback-dir", default="")
    parser.add_argument("--splits", default="train")
    parser.add_argument("--min-depth-gap", type=float, default=1e-4)
    parser.add_argument("--source-k", type=int, default=8)
    parser.add_argument("--min-retained-quality", type=float, default=0.0)
    parser.add_argument("--local-group-score", type=float, default=0.08)
    parser.add_argument("--structure-group-score", type=float, default=0.01)
    parser.add_argument(
        "--local-group-diameter",
        type=float,
        default=0.0,
        help="finite world-space limit; zero derives train-edge q75",
    )
    parser.add_argument(
        "--structural-group-diameter",
        type=float,
        default=0.0,
        help="finite world-space limit; zero derives train-edge q95",
    )
    parser.add_argument("--max-poses", type=int, default=0)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
