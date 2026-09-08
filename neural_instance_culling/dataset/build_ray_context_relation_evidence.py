#!/usr/bin/env python3
"""Build unique, pose-conditioned occlusion relations from depth-layer caches.

The cache is produced by the formal hardware-GPU triangle depth-peeling path.
This builder does not render, does not add visible IDs to the candidate CSR,
and copies the registered survival observations byte-for-byte.  Relation
aggregation follows four explicit stages:

1. reduce adjacent depth-layer events inside one pose;
2. assign each target to its three nearest spherical directions for that pose;
3. remove duplicate ``(target, direction, shell, pose, source)`` events;
4. aggregate unique sources and compute pixel fraction and pose support rate.

The result is an offline-only relation table.  The browser never receives
source IDs or this table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

DATASET_DIR = Path(__file__).resolve().parent
MODEL_DIR = DATASET_DIR.parent / "model"
for _path in (DATASET_DIR, MODEL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_triangle_depth_layer_evidence import (
    BACKGROUND_ID,
    CACHE_SCHEMA,
    CACHE_SCHEMA_V2,
    _camera_for_cache_row,
    load_layer_cache,
    load_surface_fallback_relations,
    metric_ray_depths_for_cache_row,
    spherical_direction_anchors,
)
from common.candidate_identity import (  # noqa: E402
    audit_native_aabb_candidates,
    candidate_digest_for_pose_sequence,
    pose_sequence_for_splits,
)
from common.runtime_meta import load_runtime_meta, scene_min_max
from pose_csr_dataset import PoseCSRDataset


SCHEMA = "ray-context-relation-evidence-v2"
SOURCE_K = 8
DIRECTION_BINS = 12
DEPTH_SHELLS = 3
RELATION_STAT_DIM = 6


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _soft_direction_neighbors(
    directions: np.ndarray,
    anchors: np.ndarray,
    top_k: int,
    concentration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest spherical anchors and normalized soft weights."""
    directions = np.asarray(directions, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape [N, 3]")
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("anchors must have shape [D, 3]")
    if not 1 <= int(top_k) <= anchors.shape[0]:
        raise ValueError("top_k must be within the number of anchors")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-6)
    scores = directions @ anchors.T
    order = np.argsort(-scores, axis=1, kind="stable")[:, : int(top_k)]
    selected = np.take_along_axis(scores, order, axis=1)
    selected = np.exp(float(concentration) * (selected - selected.max(axis=1, keepdims=True)))
    selected /= np.maximum(selected.sum(axis=1, keepdims=True), 1e-8)
    return order.astype(np.int16, copy=False), selected.astype(np.float32, copy=False)


def _shell_id(relative_gap: np.ndarray) -> np.ndarray:
    return np.clip(
        np.digitize(
            np.asarray(relative_gap, dtype=np.float32),
            np.asarray([0.10, 0.35], dtype=np.float32),
        ),
        0,
        DEPTH_SHELLS - 1,
    ).astype(np.int8, copy=False)


def _surface_fallback_pose_rows(
    relation_rows: np.ndarray,
    candidate_mask: np.ndarray,
    centers: np.ndarray,
    camera_world: np.ndarray,
    min_depth_gap: float,
) -> tuple[np.ndarray, int]:
    """Convert actual surface fallback records into relation-table rows.

    The returned rows use the same six columns as ``_reduce_pose_events``.
    Keeping this conversion per rendered pose preserves the view-cell
    subpose support denominator in the later aggregation step.
    """
    if relation_rows.size == 0:
        return np.zeros((0, 6), dtype=np.float32), 0
    num_instances = int(candidate_mask.size)
    source = relation_rows["source"].astype(np.int64, copy=False)
    target = relation_rows["target"].astype(np.int64, copy=False)
    pixels = relation_rows["pixelCount"].astype(np.float32, copy=False)
    gap = relation_rows["gapMean"].astype(np.float32, copy=False)
    source_in_range = (source >= 0) & (source < num_instances)
    target_in_range = (target >= 0) & (target < num_instances)
    source_candidate = np.zeros(source.shape, dtype=bool)
    target_candidate = np.zeros(target.shape, dtype=bool)
    if bool(source_in_range.any()):
        source_candidate[source_in_range] = candidate_mask[source[source_in_range]]
    if bool(target_in_range.any()):
        target_candidate[target_in_range] = candidate_mask[target[target_in_range]]
    valid = (
        source_in_range
        & target_in_range
        & (source != target)
        & source_candidate
        & target_candidate
        & np.isfinite(pixels)
        & (pixels > 0.0)
        & np.isfinite(gap)
        & (gap > float(min_depth_gap))
    )
    if not bool(valid.all()):
        invalid_count = int((~valid).sum())
        raise ValueError(
            f"surface fallback has {invalid_count} invalid/out-of-candidate relations"
        )
    source = source[valid]
    target = target[valid]
    pixels = pixels[valid]
    gap = gap[valid]
    # Surface fallback and triangle peeling both use metric camera-ray range.
    target_distance = np.linalg.norm(centers[target] - camera_world[None, :], axis=1)
    relative_gap = gap / np.maximum(target_distance, 1e-4)
    shell = _shell_id(relative_gap)
    rows = np.column_stack([
        target,
        source,
        shell.astype(np.float32, copy=False),
        pixels,
        gap,
        relation_rows["gapStd"].astype(np.float32, copy=False)[valid],
    ]).astype(np.float32, copy=False)
    return rows, int(np.unique(target).size)


def _weighted_gap_statistics(
    pixels: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> tuple[float, float]:
    """Combine per-event depth-gap moments using event pixel counts."""
    weights = np.maximum(np.asarray(pixels, dtype=np.float64), 1.0)
    mean_values = np.asarray(means, dtype=np.float64)
    std_values = np.asarray(standard_deviations, dtype=np.float64)
    total = max(float(weights.sum()), 1e-12)
    mean = float(np.sum(weights * mean_values) / total)
    variance = float(
        np.sum(weights * (np.square(std_values) + np.square(mean_values - mean))) / total
    )
    return mean, float(math.sqrt(max(0.0, variance)))


def _reduce_pose_events(
    ids: np.ndarray,
    depths: np.ndarray,
    candidate_mask: np.ndarray,
    min_depth_gap: float,
) -> np.ndarray:
    """Return [target, source, shell, pixels, gap_mean, gap_std] rows.

    A target is the deeper ID and a source is the adjacent front ID.  The
    target direction is deliberately assigned by the caller because it must
    use that target's ray in the current pose.
    """
    ids = np.asarray(ids, dtype=np.uint32)
    depths = np.asarray(depths, dtype=np.float32)
    if ids.ndim != 3 or depths.shape != ids.shape:
        raise ValueError("ids and depths must have shape [layers, height, width]")
    rows: list[np.ndarray] = []
    num_instances = int(candidate_mask.size)
    for layer in range(int(ids.shape[0]) - 1):
        front_all = ids[layer].reshape(-1)
        back_all = ids[layer + 1].reshape(-1)
        gap_all = depths[layer + 1].reshape(-1) - depths[layer].reshape(-1)
        valid = (
            (front_all != BACKGROUND_ID)
            & (back_all != BACKGROUND_ID)
            & (front_all != back_all)
            & (front_all < num_instances)
            & (back_all < num_instances)
            & np.isfinite(gap_all)
            & (gap_all > float(min_depth_gap))
        )
        if not bool(valid.any()):
            continue
        front = front_all[valid].astype(np.int64, copy=False)
        back = back_all[valid].astype(np.int64, copy=False)
        gap = gap_all[valid].astype(np.float32, copy=False)
        keep = candidate_mask[front] & candidate_mask[back]
        if not bool(keep.any()):
            continue
        front, back, gap = front[keep], back[keep], gap[keep]
        pairs = np.stack([front, back], axis=1)
        unique_pairs, inverse, counts = np.unique(
            pairs, axis=0, return_inverse=True, return_counts=True
        )
        gap_sum = np.bincount(
            inverse,
            weights=gap.astype(np.float64, copy=False),
            minlength=unique_pairs.shape[0],
        )
        gap_square_sum = np.bincount(
            inverse,
            weights=np.square(gap.astype(np.float64, copy=False)),
            minlength=unique_pairs.shape[0],
        )
        gap_mean64 = gap_sum / np.maximum(counts.astype(np.float64), 1.0)
        gap_variance = np.maximum(
            gap_square_sum / np.maximum(counts.astype(np.float64), 1.0)
            - np.square(gap_mean64),
            0.0,
        )
        gap_mean = gap_mean64.astype(np.float32)
        gap_std = np.sqrt(gap_variance).astype(np.float32)

        # Estimate target depth from the complete deeper layer.  This gives a
        # stable relative gap for ordered depth shells without AABB overlap.
        finite = (back_all != BACKGROUND_ID) & (back_all < num_instances) & np.isfinite(
            depths[layer + 1].reshape(-1)
        )
        target_values, target_inverse = np.unique(back_all[finite], return_inverse=True)
        target_depth = np.bincount(
            target_inverse,
            weights=depths[layer + 1].reshape(-1)[finite].astype(np.float64, copy=False),
            minlength=target_values.size,
        ).astype(np.float32)
        target_count = np.bincount(target_inverse, minlength=target_values.size).astype(np.float32)
        target_depth /= np.maximum(target_count, 1.0)
        target_depth_map = np.zeros((num_instances,), dtype=np.float32)
        target_depth_map[target_values.astype(np.int64, copy=False)] = target_depth

        source = unique_pairs[:, 0].astype(np.int32, copy=False)
        target = unique_pairs[:, 1].astype(np.int32, copy=False)
        relative_gap = gap_mean / np.maximum(target_depth_map[target], 1e-4)
        shell = _shell_id(relative_gap)
        rows.append(
            np.column_stack(
                [target, source, shell.astype(np.int32), counts.astype(np.float32), gap_mean, gap_std]
            ).astype(np.float32, copy=False)
        )
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def _deduplicate_pose_rows(expanded_rows: np.ndarray) -> np.ndarray:
    """Deduplicate [target, source, direction, shell, pixels, mean, std, pose]."""
    if expanded_rows.size == 0:
        return np.zeros((0, 8), dtype=np.float32)
    key = expanded_rows[:, [0, 2, 3, 7, 1]].astype(np.int64, copy=False)
    order = np.lexsort((key[:, 4], key[:, 3], key[:, 2], key[:, 1], key[:, 0]))
    values = expanded_rows[order]
    keys = key[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1]
    ends = np.r_[starts[1:], keys.shape[0]]
    reduced: list[list[float]] = []
    for start, end in zip(starts.tolist(), ends.tolist(), strict=False):
        group = values[start:end]
        mean, std = _weighted_gap_statistics(group[:, 4], group[:, 5], group[:, 6])
        reduced.append([
            float(group[0, 0]),
            float(group[0, 1]),
            float(group[0, 2]),
            float(group[0, 3]),
            float(group[:, 4].sum()),
            mean,
            std,
            float(group[0, 7]),
        ])
    return np.asarray(reduced, dtype=np.float32)


def _aggregate_rows(
    expanded_rows: np.ndarray,
    num_instances: int,
    source_k: int,
    pixel_denominator: int,
    pose_rows_are_unique: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Aggregate unique sources with pose support, then select top-k."""
    if expanded_rows.size == 0:
        raise ValueError("depth cache produced no candidate occlusion relations")
    if pose_rows_are_unique:
        if expanded_rows.ndim != 2 or expanded_rows.shape[1] != 8:
            raise ValueError("pre-deduplicated relation rows must have shape [N, 8]")
        pose_rows = np.asarray(expanded_rows, dtype=np.float32)
    else:
        pose_rows = _deduplicate_pose_rows(expanded_rows)

    # The cell denominator is the number of poses in which any relation was
    # observed for this target/direction/shell.  It must not be inferred from
    # the maximum support of one source, otherwise a source that appears in a
    # single pose can receive an artificial support rate of one.
    cell_pose_keys = pose_rows[:, [0, 2, 3, 7]].astype(np.int64, copy=False)
    unique_cell_pose = np.unique(cell_pose_keys, axis=0)
    unique_cell_keys, cell_pose_counts = np.unique(
        unique_cell_pose[:, :3], axis=0, return_counts=True
    )
    cell_pose_count_map = {
        tuple(int(value) for value in key): int(count)
        for key, count in zip(unique_cell_keys.tolist(), cell_pose_counts.tolist(), strict=False)
    }

    # Aggregate one source across all poses in a target/direction/shell cell.
    source_key = pose_rows[:, [0, 2, 3, 1]].astype(np.int64, copy=False)
    order = np.lexsort((source_key[:, 3], source_key[:, 2], source_key[:, 1], source_key[:, 0]))
    values = pose_rows[order]
    keys = source_key[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1]
    ends = np.r_[starts[1:], keys.shape[0]]
    source_rows: list[list[float]] = []
    for start, end in zip(starts.tolist(), ends.tolist(), strict=False):
        group = values[start:end]
        mean, std = _weighted_gap_statistics(group[:, 4], group[:, 5], group[:, 6])
        source_rows.append([
            float(group[0, 0]),
            float(group[0, 1]),
            float(group[0, 2]),
            float(group[0, 3]),
            float(group[:, 4].sum()),
            float(np.unique(group[:, 7].astype(np.int64)).size),
            mean,
            std,
        ])
    source_values = np.asarray(source_rows, dtype=np.float32)

    source_ids = np.zeros((num_instances, DIRECTION_BINS, DEPTH_SHELLS, source_k), dtype=np.uint32)
    relation_stats = np.zeros(
        (num_instances, DIRECTION_BINS, DEPTH_SHELLS, source_k, RELATION_STAT_DIM),
        dtype=np.float32,
    )
    strength = np.zeros((num_instances, DIRECTION_BINS, DEPTH_SHELLS), dtype=np.float32)
    evidence_count = np.zeros((num_instances, DIRECTION_BINS, DEPTH_SHELLS), dtype=np.uint32)
    cell_key = source_values[:, [0, 2, 3]].astype(np.int64, copy=False)
    cell_order = np.lexsort((cell_key[:, 2], cell_key[:, 1], cell_key[:, 0]))
    cell_values = source_values[cell_order]
    cell_keys = cell_key[cell_order]
    cell_starts = np.r_[0, np.flatnonzero(np.any(np.diff(cell_keys, axis=0) != 0, axis=1)) + 1]
    cell_ends = np.r_[cell_starts[1:], cell_keys.shape[0]]
    dropped_sources = 0
    nonzero_cells = 0
    for start, end in zip(cell_starts.tolist(), cell_ends.tolist(), strict=False):
        group = cell_values[start:end]
        target, direction, shell = (int(group[0, 0]), int(group[0, 2]), int(group[0, 3]))
        cell_pixels = max(float(group[:, 4].sum()), 1e-6)
        cell_pose_count = max(
            float(cell_pose_count_map.get((target, direction, shell), 0)), 1.0
        )
        pixel_fraction = np.clip(group[:, 4] / cell_pixels, 0.0, 1.0)
        support_rate = np.clip(group[:, 5] / cell_pose_count, 0.0, 1.0)
        scores = np.sqrt(pixel_fraction * support_rate)
        tie = group[:, 1].astype(np.int64, copy=False)
        ranking = np.lexsort((tie, -scores))[: int(source_k)]
        nonzero_cells += 1
        strength[target, direction, shell] = float(np.clip(scores[ranking].sum(), 0.0, 1.0))
        evidence_count[target, direction, shell] = int(group.shape[0])
        dropped_sources += max(0, int(group.shape[0]) - int(source_k))
        for slot, index in enumerate(ranking.tolist()):
            row = group[int(index)]
            source = int(row[1])
            if source == target:
                raise RuntimeError("a target instance appeared as its own occlusion source")
            source_ids[target, direction, shell, slot] = np.uint32(source)
            relation_stats[target, direction, shell, slot] = np.asarray([
                float(scores[index]),
                float(pixel_fraction[index]),
                float(support_rate[index]),
                float(np.clip(row[6], 0.0, 1.0)),
                float(np.clip(row[7], 0.0, 1.0)),
                float(np.log1p(row[4]) / np.log1p(max(1, pixel_denominator))),
            ], dtype=np.float32)

    return source_ids, relation_stats, strength, evidence_count, {
        "nonzeroCells": int(nonzero_cells),
        "droppedUniqueSourcesBeyondTopK": int(dropped_sources),
        "deduplicatedPoseSourceRows": int(pose_rows.shape[0]),
        "uniqueTargetSourceCells": int(source_values.shape[0]),
    }


def _copy_survival_observations(
    survival_dir: Path,
    output_dir: Path,
    selected_pose_count: int,
    expected_candidate_digest: str,
    expected_cache_schema: str,
    expected_splits: list[str],
    expected_render_candidate_digest: str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    source_meta_path = survival_dir / "evidence_meta.json"
    if not source_meta_path.is_file():
        raise FileNotFoundError(f"missing registered survival evidence metadata: {source_meta_path}")
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    if source_meta.get("schema") != "triangle-depth-layer-evidence-v1":
        raise ValueError("survival evidence source must be triangle-depth-layer-evidence-v1")
    if source_meta.get("cacheSchema") != expected_cache_schema:
        raise ValueError(
            "registered survival evidence uses a different depth-cache schema: "
            f"{source_meta.get('cacheSchema')!r} != {expected_cache_schema!r}"
        )
    if float(source_meta.get("modelInputFovYDeg", 0.0)) != 66.0:
        raise ValueError("registered survival evidence must use the 66 degree model-input FOV")
    if sorted(str(value) for value in source_meta.get("splits", [])) != sorted(expected_splits):
        raise ValueError(
            "registered survival evidence uses different dataset splits: "
            f"{source_meta.get('splits')!r} != {expected_splits!r}"
        )
    source_digest = str(source_meta.get("candidateDigest", ""))
    if source_digest != str(expected_candidate_digest):
        raise ValueError(
            "registered survival evidence candidate digest does not match the current cache/dataset: "
            f"{source_digest!r} != {expected_candidate_digest!r}"
        )
    source_render_digest = str(source_meta.get("renderCandidateDigest", ""))
    if expected_render_candidate_digest and source_render_digest and source_render_digest != str(expected_render_candidate_digest):
        raise ValueError(
            "registered survival evidence render candidate digest does not match the current depth cache: "
            f"{source_render_digest!r} != {expected_render_candidate_digest!r}"
        )
    source_checksums: dict[str, str] = {}
    copied_checksums: dict[str, str] = {}
    for key in (
        "observationInstances",
        "observationDirections",
        "observationRho",
        "observationEvent",
        "observationWeight",
    ):
        filename = str(source_meta["files"][key])
        source = survival_dir / filename
        destination = output_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing survival observation array: {source}")
        shutil.copy2(source, destination)
        source_checksums[key] = _sha256(source)
        copied_checksums[key] = _sha256(destination)
        if source_checksums[key] != copied_checksums[key]:
            raise IOError(f"survival observation copy changed bytes for {key}")
    if int(source_meta.get("stats", {}).get("poseCount", selected_pose_count)) < selected_pose_count:
        raise ValueError("registered survival evidence has fewer poses than the requested relation evidence")
    return source_checksums, copied_checksums


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    runtime_meta_path = Path(args.runtime_meta)
    cache_dir = Path(args.layer_cache_dir)
    survival_dir = Path(args.survival_evidence_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    world_aabbs, _instance_to_glb, _runtime_meta = load_runtime_meta(runtime_meta_path)
    num_instances = int(world_aabbs.shape[0])
    centers = ((world_aabbs[:, :3] + world_aabbs[:, 3:]) * 0.5).astype(np.float32, copy=False)
    _scene_min, _scene_max, scene_size = scene_min_max(_runtime_meta["sceneBounds"])
    dataset = PoseCSRDataset(dataset_dir, num_instances=num_instances)
    cache_meta, cache_render_pose_ids, cache_source_pose_indices, cache_ids, cache_depths = load_layer_cache(cache_dir)
    if cache_meta.get("schema") not in (CACHE_SCHEMA, CACHE_SCHEMA_V2) or float(cache_meta.get("modelInputFovYDeg", 0)) != 66.0:
        raise ValueError("formal relation evidence requires triangle-depth cache v1/v2 at FOV 66")
    gpu_evidence_summary = cache_meta.get("gpuEvidenceSummary") or {}
    if gpu_evidence_summary.get("formalReady") is not True:
        raise ValueError("triangle-depth cache lacks the registered hardware GPU evidence")
    if int(cache_meta.get("maxLayers", 0)) < 2:
        raise ValueError("relation evidence requires at least two depth layers")
    fallback_relations = load_surface_fallback_relations(survival_dir)
    require_fallback = bool(getattr(args, "require_surface_fallback", False))
    if require_fallback and fallback_relations.size == 0:
        raise ValueError(
            "formal relation evidence requires per-render surface-point fallback relations; "
            f"none were found in {survival_dir}"
        )
    cache_render_id_set = set(int(value) for value in cache_render_pose_ids.tolist())
    fallback_render_ids = np.unique(fallback_relations["renderPoseId"]) if fallback_relations.size else np.zeros((0,), dtype=np.uint32)
    unknown_fallback_render_ids = [
        int(value) for value in fallback_render_ids.tolist() if int(value) not in cache_render_id_set
    ]
    if unknown_fallback_render_ids:
        raise ValueError(
            "surface fallback relations reference render poses absent from the depth cache: "
            f"{unknown_fallback_render_ids[:8]}"
        )
    fallback_by_render_pose: dict[int, np.ndarray] = {}
    if fallback_relations.size:
        for render_pose_id in fallback_render_ids.tolist():
            pose_id = int(render_pose_id)
            fallback_by_render_pose[pose_id] = fallback_relations[
                fallback_relations["renderPoseId"] == pose_id
            ]
    cache_rows = list(range(int(cache_render_pose_ids.size)))

    requested_splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    missing_splits = [name for name in requested_splits if name not in dataset.split_ids]
    if missing_splits:
        raise ValueError(f"dataset does not define requested splits: {missing_splits}")
    split_values = {int(dataset.split_ids[name]) for name in requested_splits}
    selected_entries = [
        (row, int(cache_render_pose_ids[row]), int(cache_source_pose_indices[row]))
        for row in cache_rows
        if 0 <= int(cache_source_pose_indices[row]) < dataset.poses.shape[0]
        and int(dataset.poses[int(cache_source_pose_indices[row])]["split"]) in split_values
    ]
    if args.max_poses > 0:
        selected_entries = selected_entries[: int(args.max_poses)]
    if not selected_entries:
        raise ValueError("cache contains no requested split poses")

    anchors = spherical_direction_anchors()
    all_expanded_rows: list[np.ndarray] = []
    total_pair_rows = 0
    total_pair_pixels = 0
    fallback_relation_count = 0
    fallback_target_count = 0
    fallback_pose_count = 0
    first_layer_failures: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    progress_interval = max(1, min(500, len(selected_entries) // 20 or 1))
    for entry_index, (cache_row, render_pose_id, pose_index) in enumerate(selected_entries, start=1):
        ids = np.asarray(cache_ids[cache_row], dtype=np.uint32)
        encoded_depths = np.asarray(cache_depths[cache_row], dtype=np.float32)
        candidate = np.asarray(dataset.candidate_slice(int(pose_index)), dtype=np.uint32)
        candidate_mask = np.zeros((num_instances,), dtype=bool)
        if candidate.size and int(candidate.max()) >= num_instances:
            raise ValueError(f"candidate ID exceeds runtime instance count at pose {pose_index}")
        candidate_mask[candidate.astype(np.int64, copy=False)] = True
        first_ids = np.unique(ids[0][ids[0] != BACKGROUND_ID])
        invalid_first = first_ids[first_ids >= num_instances]
        if invalid_first.size:
            raise ValueError(
                f"first depth layer contains IDs outside runtime instance count at pose {pose_index}: "
                f"{invalid_first[:8].astype(int).tolist()}"
            )
        outside_first = first_ids[~candidate_mask[first_ids.astype(np.int64, copy=False)]]
        if outside_first.size:
            first_layer_failures.append({
                "poseIndex": int(pose_index),
                "renderPoseId": int(render_pose_id),
                "outsideCandidate": outside_first[:16].astype(int).tolist(),
                "outsideCount": int(outside_first.size),
            })
            raise ValueError(
                f"first depth layer contains {outside_first.size} IDs outside stored candidates at pose {pose_index}"
            )

        camera_world, _camera_forward, camera_view = _camera_for_cache_row(
            cache_meta, render_pose_id, dataset, pose_index
        )
        depths = metric_ray_depths_for_cache_row(cache_meta, encoded_depths, camera_view)
        pose_rows = _reduce_pose_events(ids, depths, candidate_mask, float(args.min_depth_gap))
        pose_fallback = fallback_by_render_pose.get(int(render_pose_id))
        if pose_fallback is not None and pose_fallback.size:
            fallback_pose_rows, fallback_pose_target_count = _surface_fallback_pose_rows(
                pose_fallback,
                candidate_mask,
                centers,
                camera_world,
                float(args.min_depth_gap),
            )
            if pose_rows.size:
                pose_rows = np.concatenate([pose_rows, fallback_pose_rows], axis=0)
            else:
                pose_rows = fallback_pose_rows
            fallback_relation_count += int(fallback_pose_rows.shape[0])
            fallback_target_count += int(fallback_pose_target_count)
            fallback_pose_count += 1
        if pose_rows.size == 0:
            continue
        target_ids, inverse = np.unique(pose_rows[:, 0].astype(np.int64, copy=False), return_inverse=True)
        target_rays = centers[target_ids] - camera_world[None, :]
        soft_ids, soft_weights = _soft_direction_neighbors(
            target_rays,
            anchors,
            top_k=int(args.soft_direction_k),
            concentration=float(args.direction_concentration),
        )
        k = int(args.soft_direction_k)
        expanded = np.empty((pose_rows.shape[0] * k, 8), dtype=np.float32)
        expanded[:, 0] = np.repeat(pose_rows[:, 0], k)
        expanded[:, 1] = np.repeat(pose_rows[:, 1], k)
        expanded[:, 2] = soft_ids[inverse].reshape(-1)
        expanded[:, 3] = np.repeat(pose_rows[:, 2], k)
        expanded[:, 4] = np.repeat(pose_rows[:, 3], k) * soft_weights[inverse].reshape(-1)
        expanded[:, 5] = np.repeat(pose_rows[:, 4], k)
        expanded[:, 6] = np.repeat(pose_rows[:, 5], k)
        # Deduplication and pose-support statistics operate on the actual
        # render, not the canonical source row.  Five representatives of one
        # viewcell are therefore five independent observations.
        expanded[:, 7] = float(render_pose_id)
        all_expanded_rows.append(expanded)
        total_pair_rows += int(pose_rows.shape[0])
        total_pair_pixels += int(np.rint(pose_rows[:, 3].sum()))

        if entry_index == 1 or entry_index % progress_interval == 0 or entry_index == len(selected_entries):
            elapsed = max(1e-6, time.perf_counter() - started_at)
            rate = entry_index / elapsed
            remaining = (len(selected_entries) - entry_index) / max(rate, 1e-6)
            print(json.dumps({
                "status": "relation_evidence_progress",
                "processed": entry_index,
                "total": len(selected_entries),
                "ratePosesPerSecond": rate,
                "etaSeconds": remaining,
                "reducedRelationRows": total_pair_rows,
            }, ensure_ascii=False), flush=True)

    if not all_expanded_rows:
        raise ValueError("no relation rows were generated")
    expanded_rows = np.concatenate(all_expanded_rows, axis=0)
    source_ids, relation_stats, strength, evidence_count, aggregation_stats = _aggregate_rows(
        expanded_rows,
        num_instances=num_instances,
        source_k=int(args.source_k),
        pixel_denominator=int(cache_meta["width"]) * int(cache_meta["height"]),
    )
    valid_sources = relation_stats[..., 0] > 0.0
    duplicate_slots = 0
    for target in range(num_instances):
        for direction in range(DIRECTION_BINS):
            for shell in range(DEPTH_SHELLS):
                values = source_ids[target, direction, shell][valid_sources[target, direction, shell]]
                if values.size != np.unique(values).size:
                    duplicate_slots += int(values.size - np.unique(values).size)
    if duplicate_slots:
        raise RuntimeError(f"v2 relation table contains {duplicate_slots} duplicate source IDs")

    source_pose_sequence = np.asarray([entry[2] for entry in selected_entries], dtype="<i8")
    canonical_pose_sequence = pose_sequence_for_splits(dataset, requested_splits)
    candidate_audit = audit_native_aabb_candidates(
        dataset,
        world_aabbs,
        canonical_pose_sequence,
    )
    digest = candidate_digest_for_pose_sequence(dataset, canonical_pose_sequence)
    render_digest = candidate_digest_for_pose_sequence(dataset, source_pose_sequence)
    cache_identity = cache_meta.get("candidateIdentity") or {}
    if cache_identity.get("renderCandidateDigest") and cache_identity["renderCandidateDigest"] != render_digest:
        raise ValueError("relation cache render candidate digest does not match its native cache identity")

    source_checksums, copied_checksums = _copy_survival_observations(
        survival_dir,
        output_dir,
        selected_pose_count=int(len(selected_entries)),
        expected_candidate_digest=digest,
        expected_render_candidate_digest=render_digest,
        expected_cache_schema=str(cache_meta["schema"]),
        expected_splits=requested_splits,
    )

    relation_files = {
        "sourceIds": "relation_source_ids_uint32.bin",
        "relationStats": "relation_stats_fp16.bin",
        "evidenceStrength": "relation_evidence_strength_fp16.bin",
        "evidenceCount": "relation_evidence_count_uint32.bin",
    }
    source_ids.tofile(output_dir / relation_files["sourceIds"])
    relation_stats.astype(np.float16).tofile(output_dir / relation_files["relationStats"])
    strength.astype(np.float16).tofile(output_dir / relation_files["evidenceStrength"])
    evidence_count.astype(np.uint32).tofile(output_dir / relation_files["evidenceCount"])

    meta: dict[str, Any] = {
        "schema": SCHEMA,
        "cacheSchema": cache_meta["schema"],
        "datasetDir": dataset_dir.as_posix(),
        "runtimeMeta": runtime_meta_path.as_posix(),
        "survivalEvidenceSource": survival_dir.as_posix(),
        "numInstances": num_instances,
        "directionBins": DIRECTION_BINS,
        "depthShells": DEPTH_SHELLS,
        "sourceK": int(args.source_k),
        "relationStatDim": RELATION_STAT_DIM,
        "relationStatSemantics": [
            "unique_relation_score",
            "conditional_pixel_fraction",
            "pose_support_rate",
            "weighted_mean_relative_depth_gap",
            "weighted_depth_gap_std",
            "normalized_log_pixel_count",
        ],
        "softDirectionK": int(args.soft_direction_k),
        "directionConcentration": float(args.direction_concentration),
        "modelInputFovYDeg": 66.0,
        "splits": requested_splits,
        "formalCandidatePolicy": "stored back-camera candidate CSR; no GT-visible union and no frontend whitelist",
        "relationBuildVariant": "surface_fallback_merged_v1" if fallback_relations.size else "depth_peeling_only_v2",
        "surfaceFallbackMerged": bool(fallback_relations.size),
        "surfaceFallbackEvidenceSource": survival_dir.as_posix() if fallback_relations.size else None,
        "evidenceSemantics": (
            "unique adjacent different component IDs from hardware-GPU triangle depth peeling "
            "plus per-render actual GLB surface-point fallback relations; AABB overlap is not used"
            if fallback_relations.size
            else "unique adjacent different component IDs from hardware-GPU triangle depth peeling; AABB overlap is not used"
        ),
        "survivalSemantics": "copied byte-for-byte from registered v1 evidence; layer-zero observations are right-censored",
        "candidateDigest": digest,
        "candidateDigestScope": "canonical PoseCSR split rows; representative render rows are not included",
        "renderCandidateDigest": render_digest,
        "renderCandidateDigestScope": "stored candidate CSR rows repeated in relation-cache renderPose order",
        "candidateIdentity": {
            "canonicalCandidateDigest": digest,
            "renderCandidateDigest": render_digest,
            "canonicalPoseCount": int(canonical_pose_sequence.size),
            "renderPoseCount": int(source_pose_sequence.size),
        },
        "gpuEvidence": gpu_evidence_summary,
        "candidateAudit": candidate_audit,
        "firstLayerValidation": {
            "status": "passed",
            "failureCount": len(first_layer_failures),
            "failures": first_layer_failures,
        },
        "survivalObservationChecksums": {
            "source": source_checksums,
            "copied": copied_checksums,
            "byteIdentical": source_checksums == copied_checksums,
        },
        "stats": {
            "poseCount": int(len(selected_entries)),
            "renderPoseCount": int(len(selected_entries)),
            "sourcePoseCount": int(np.unique(source_pose_sequence).size),
            "representativeSubposeCache": bool(cache_meta.get("schema") == CACHE_SCHEMA_V2),
            "perPoseReducedRelationRows": int(total_pair_rows),
            "perPoseRelationPixelCount": int(total_pair_pixels),
            "surfaceFallbackRelationCount": int(fallback_relation_count),
            "surfaceFallbackTargetCount": int(fallback_target_count),
            "surfaceFallbackPoseCount": int(fallback_pose_count),
            "nonzeroRelationCells": int(np.count_nonzero(valid_sources)),
            "uniqueSourceSlots": int(np.count_nonzero(valid_sources)),
            "duplicateSourceSlots": int(duplicate_slots),
            **aggregation_stats,
        },
        "files": {**relation_files, **{
            "observationInstances": "survival_observation_instances_uint32.bin",
            "observationDirections": "survival_observation_directions_uint8.bin",
            "observationRho": "survival_observation_rho_fp16.bin",
            "observationEvent": "survival_observation_event_uint8.bin",
            "observationWeight": "survival_observation_weight_fp16.bin",
        }},
    }
    (output_dir / "evidence_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--layer-cache-dir", required=True)
    parser.add_argument("--survival-evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="train")
    parser.add_argument("--source-k", type=int, default=SOURCE_K)
    parser.add_argument("--soft-direction-k", type=int, default=3)
    parser.add_argument("--direction-concentration", type=float, default=8.0)
    parser.add_argument("--min-depth-gap", type=float, default=1e-4)
    parser.add_argument("--max-poses", type=int, default=0)
    parser.add_argument(
        "--require-surface-fallback",
        action="store_true",
        help="require and merge per-render surface-point fallback relations from the triangle evidence directory",
    )
    args = parser.parse_args()
    if not 1 <= args.soft_direction_k <= DIRECTION_BINS:
        parser.error("--soft-direction-k must be between 1 and 12")
    if args.source_k <= 0:
        parser.error("--source-k must be positive")
    if args.max_poses < 0:
        parser.error("--max-poses must be non-negative")
    print(json.dumps(build_evidence(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
