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

from compact_triangle_depth_relation_shard import (  # noqa: E402
    RELATION_MOMENT_DTYPE,
    RELATION_ROW_DTYPE,
    SURVIVAL_OBSERVATION_DTYPE,
    SCHEMA as SPARSE_SHARD_SCHEMA,
    load_sparse_shard,
)
from common.candidate_identity import (  # noqa: E402
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
    rows = [np.asarray(dataset.candidate_slice(int(p)), dtype="<u4") for p in poses.tolist()]
    offsets = np.zeros((poses.size + 1,), dtype="<i8")
    if rows:
        offsets[1:] = np.cumsum([row.size for row in rows], dtype=np.int64)
        ids = np.concatenate(rows).astype("<u4", copy=False)
    else:
        ids = np.zeros((0,), dtype="<u4")
    return summarize_candidate_csr(ids, poses, offsets, num_instances)


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


def _segment_starts(*columns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return starts, ends and row-to-segment IDs for sorted key columns."""
    if not columns:
        raise ValueError("at least one segment key is required")
    size = int(np.asarray(columns[0]).size)
    if any(int(np.asarray(value).size) != size for value in columns):
        raise ValueError("segment key columns have different lengths")
    if size == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, empty
    change = np.ones((size,), dtype=np.bool_)
    change[1:] = False
    for value in columns:
        array = np.asarray(value).reshape(-1)
        change[1:] |= array[1:] != array[:-1]
    starts = np.flatnonzero(change).astype(np.int64, copy=False)
    ends = np.r_[starts[1:], size].astype(np.int64, copy=False)
    segment_ids = np.cumsum(change, dtype=np.int64) - 1
    return starts, ends, segment_ids


def _vectorized_topk(
    target: np.ndarray,
    direction: np.ndarray,
    shell: np.ndarray,
    source: np.ndarray,
    features: np.ndarray,
    *,
    k: int,
    worst_cell_limit: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Select formal source top-k without constructing a Python dict per cell."""
    if int(k) <= 0:
        raise ValueError("relation top-k must be positive")
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    direction = np.asarray(direction, dtype=np.int64).reshape(-1)
    shell = np.asarray(shell, dtype=np.int64).reshape(-1)
    source = np.asarray(source, dtype=np.uint32).reshape(-1)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not (
        target.size == direction.size == shell.size == source.size == features.shape[0]
    ):
        raise ValueError("relation top-k arrays have incompatible shapes")

    pixel_component = np.log1p(np.maximum(features[:, PIXEL_SUPPORT_FEATURE], 0.0))
    confidence_component = np.maximum(features[:, CONFIDENCE_FEATURE], 0.0)
    pose_component = np.log1p(1.0 + np.maximum(features[:, 0], 0.0))
    starts, ends, cell_ids = _segment_starts(target, direction, shell)
    counts = ends - starts

    def normalized(values: np.ndarray) -> np.ndarray:
        maxima = np.maximum.reduceat(values, starts)
        denominator = np.maximum(maxima[cell_ids], 1e-30)
        result = values / denominator
        result[maxima[cell_ids] <= 0.0] = 0.0
        return result

    score = normalized(pixel_component) * normalized(confidence_component) * normalized(pose_component)
    ranking = np.lexsort(
        (
            source.astype(np.int64, copy=False),
            -score,
            shell,
            direction,
            target,
        )
    )
    rt, rd, rs = target[ranking], direction[ranking], shell[ranking]
    rank_starts, rank_ends, rank_cell_ids = _segment_starts(rt, rd, rs)
    rank_counts = rank_ends - rank_starts
    within_rank = np.arange(ranking.size, dtype=np.int64) - rank_starts[rank_cell_ids]
    selected_in_ranking = within_rank < int(k)
    total_score = np.add.reduceat(score[ranking].astype(np.float64), rank_starts)
    retained_score = np.add.reduceat(
        np.where(selected_in_ranking, score[ranking], 0.0).astype(np.float64),
        rank_starts,
    )
    quality = np.divide(
        retained_score,
        total_score,
        out=np.ones_like(retained_score),
        where=total_score > 0.0,
    )
    selected = ranking[selected_in_ranking]
    selected_cell = rank_cell_ids[selected_in_ranking]
    out_features = features[selected].copy()
    out_features[:, 13] = quality[selected_cell].astype(np.float32, copy=False)
    out_features[:, 16] = (rank_counts[selected_cell] > int(k)).astype(np.float32)
    out_features[:, 19] = rank_counts[selected_cell].astype(np.float32, copy=False)

    canonical = np.lexsort(
        (
            source[selected].astype(np.int64, copy=False),
            shell[selected],
            direction[selected],
            target[selected],
        )
    )
    selected = selected[canonical]
    out_features = out_features[canonical]

    quality_order = np.lexsort((np.arange(quality.size, dtype=np.int64), quality))
    worst: dict[str, Any] = {}
    for cell in quality_order[: max(0, int(worst_cell_limit))].tolist():
        row = int(rank_starts[cell])
        key = f"{int(rt[row])}:{int(rd[row])}:{int(rs[row])}"
        worst[key] = {
            "totalScore": float(total_score[cell]),
            "retainedScore": float(retained_score[cell]),
            "retainedQuality": float(quality[cell]),
            "totalCount": int(rank_counts[cell]),
            "retainedCount": int(min(int(k), int(rank_counts[cell]))),
            "truncated": bool(rank_counts[cell] > int(k)),
        }
    quantiles = np.quantile(
        quality if quality.size else np.ones((1,), dtype=np.float64),
        (0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0),
    )
    diagnostics = {
        "k": int(k),
        "score": "cell_normalized(evidence_confidence) * cell_normalized(log1p(pixel_support)) * cell_normalized(log1p(1 + pose_support_count))",
        "cellCount": int(quality.size),
        "truncatedCellCount": int(np.count_nonzero(rank_counts > int(k))),
        "retainedQualityQuantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "q01", "q05", "q50", "q95", "q99", "max"),
                quantiles.tolist(),
                strict=True,
            )
        },
        "worstCells": worst,
    }
    return (
        target[selected],
        direction[selected],
        shell[selected],
        source[selected],
        out_features,
        diagnostics,
    )


def _aggregate_sparse_relation_rows(
    rows: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    scene_size: np.ndarray,
    *,
    total_pose_count: int,
    source_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Aggregate sparse per-pose rows into retained relation features."""
    rows = np.asarray(rows, dtype=RELATION_ROW_DTYPE)
    if rows.size == 0:
        raise ValueError("sparse shards contain no relation rows")
    order = np.lexsort(
        (
            rows["renderPoseId"].astype(np.int64, copy=False),
            rows["source"].astype(np.int64, copy=False),
            rows["shell"].astype(np.int64, copy=False),
            rows["direction"].astype(np.int64, copy=False),
            rows["target"].astype(np.int64, copy=False),
        )
    )
    values = rows[order]
    target_row = values["target"].astype(np.int64, copy=False)
    direction_row = values["direction"].astype(np.int64, copy=False)
    shell_row = values["shell"].astype(np.int64, copy=False)
    source_row = values["source"].astype(np.int64, copy=False)
    starts, ends, group_ids = _segment_starts(
        target_row, direction_row, shell_row, source_row
    )
    group_count = ends - starts
    weight = np.maximum(values["pixelCount"].astype(np.float64), 1.0)
    pixel_sum = np.add.reduceat(values["pixelCount"].astype(np.float64), starts)
    weight_sum = np.add.reduceat(weight, starts)

    def weighted_moment(mean_name: str, std_name: str) -> tuple[np.ndarray, np.ndarray]:
        mean = values[mean_name].astype(np.float64)
        std = values[std_name].astype(np.float64)
        combined_mean = np.add.reduceat(weight * mean, starts) / np.maximum(weight_sum, 1e-12)
        second = np.add.reduceat(weight * (np.square(std) + np.square(mean)), starts)
        variance = np.maximum(second / np.maximum(weight_sum, 1e-12) - np.square(combined_mean), 0.0)
        return combined_mean, np.sqrt(variance)

    gap_mean, gap_std = weighted_moment("gapMean", "gapStd")
    relative_mean, relative_std = weighted_moment("relativeGapMean", "relativeGapStd")
    new_pose = np.ones((values.size,), dtype=np.bool_)
    new_pose[1:] = (
        (group_ids[1:] != group_ids[:-1])
        | (values["renderPoseId"][1:] != values["renderPoseId"][:-1])
    )
    pose_count = np.add.reduceat(new_pose.astype(np.int64), starts)

    depth_stats = train_depth_quantiles(values["relativeLogDepth"].astype(np.float64))
    normalized_depth = normalize_radius_relative_log_depth(
        values["relativeLogDepth"].astype(np.float64), depth_stats
    ).astype(np.float64)
    depth_mean = np.add.reduceat(normalized_depth, starts) / np.maximum(group_count, 1)
    depth_second = np.add.reduceat(np.square(normalized_depth), starts) / np.maximum(group_count, 1)
    depth_std = np.sqrt(np.maximum(depth_second - np.square(depth_mean), 0.0))

    target = target_row[starts]
    direction = direction_row[starts]
    shell = shell_row[starts]
    source = source_row[starts]
    source_min = np.minimum.reduceat(values["sourceType"], starts)
    source_max = np.maximum.reduceat(values["sourceType"], starts)
    source_types = np.where(
        source_min == source_max,
        source_min,
        np.uint8(SOURCE_TYPES["merged"]),
    ).astype(np.uint8, copy=False)

    cell_starts, cell_ends, cell_ids = _segment_starts(target, direction, shell)
    cell_pixels = np.add.reduceat(pixel_sum, cell_starts)
    pixel_fraction = pixel_sum / np.maximum(cell_pixels[cell_ids], 1.0)
    pose_rate = pose_count.astype(np.float64) / max(int(total_pose_count), 1)
    relative_center = (centers[source] - centers[target]) / scene_size[None, :]
    relative_scale = np.log(
        np.maximum(radii[source], 1e-5) / np.maximum(radii[target], 1e-5)
    )
    edge_features = np.zeros((target.size, 20), dtype=np.float32)
    edge_features[:, 0] = pose_count
    edge_features[:, 1] = np.clip(pose_rate, 0.0, 1.0)
    edge_features[:, 2] = pixel_sum
    edge_features[:, 3] = np.clip(pixel_fraction, 0.0, 1.0)
    edge_features[:, 4] = np.maximum(gap_mean, 1e-4)
    edge_features[:, 5] = gap_std
    edge_features[:, 6] = np.maximum(
        relative_mean, 1e-4 / max(float(np.linalg.norm(scene_size)), 1.0)
    )
    edge_features[:, 7] = relative_std
    edge_features[:, 8:11] = np.clip(relative_center, -8.0, 8.0)
    edge_features[:, 11] = np.clip(relative_scale, -8.0, 8.0)
    edge_features[:, 12] = np.clip(
        np.sqrt(np.maximum(pose_rate * pixel_fraction, 0.0)), 0.0, 1.0
    )
    edge_features[:, 13] = np.clip(pixel_fraction, 0.0, 1.0)
    edge_features[:, 14] = 1.0
    edge_features[:, 15] = np.maximum(1, pose_count)
    edge_features[:, 16] = 0.0
    edge_features[:, 17] = depth_mean
    edge_features[:, 18] = depth_std
    edge_features[:, 19] = 1.0

    (
        target,
        direction,
        shell,
        source,
        edge_features,
        topk_diagnostics,
    ) = _vectorized_topk(
        target,
        direction,
        shell,
        source.astype(np.uint32, copy=False),
        edge_features,
        k=int(source_k),
    )
    retained_key = np.rec.fromarrays(
        [target, direction, shell, source], names="target,direction,shell,source"
    )
    original_key = np.rec.fromarrays(
        [target_row[starts], direction_row[starts], shell_row[starts], source_row[starts]],
        names="target,direction,shell,source",
    )
    retained_indices = np.searchsorted(original_key, retained_key)
    retained_types = source_types[retained_indices]
    aggregate_stats = {
        "rawRelationRowCount": int(rows.size),
        "aggregatedEdgeCount": int(starts.size),
        "retainedEdgeCount": int(target.size),
        "cellCountBeforeTopK": int(cell_starts.size),
    }
    return (
        target,
        direction,
        shell,
        source.astype(np.uint32, copy=False),
        edge_features,
        retained_types,
        topk_diagnostics,
        {"depthNormalization": depth_stats, **aggregate_stats},
    )


def _merge_sparse_relation_moments(
    moment_chunks: list[np.ndarray],
    depth_offset_chunks: list[np.ndarray],
    depth_value_chunks: list[np.ndarray],
    centers: np.ndarray,
    radii: np.ndarray,
    scene_size: np.ndarray,
    *,
    total_pose_count: int,
    source_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Merge exact per-shard moments after freezing train-only depth quantiles."""
    if not moment_chunks or not (
        len(moment_chunks) == len(depth_offset_chunks) == len(depth_value_chunks)
    ):
        raise ValueError("sparse relation moment shards are incomplete")
    all_depth_values = np.concatenate(depth_value_chunks).astype(np.float64, copy=False)
    depth_stats = train_depth_quantiles(all_depth_values)
    depth_count_chunks: list[np.ndarray] = []
    depth_sum_chunks: list[np.ndarray] = []
    depth_square_sum_chunks: list[np.ndarray] = []
    for moments, offsets, raw_depth in zip(
        moment_chunks, depth_offset_chunks, depth_value_chunks, strict=True
    ):
        offsets = np.asarray(offsets, dtype=np.int64)
        if offsets.size != moments.size + 1 or int(offsets[-1]) != raw_depth.size:
            raise ValueError("sparse relation moment/depth segmentation mismatch")
        counts = np.diff(offsets).astype(np.int64, copy=False)
        normalized = normalize_radius_relative_log_depth(raw_depth, depth_stats).astype(
            np.float64
        )
        starts = offsets[:-1]
        depth_count_chunks.append(counts)
        depth_sum_chunks.append(np.add.reduceat(normalized, starts))
        depth_square_sum_chunks.append(np.add.reduceat(np.square(normalized), starts))

    moments = np.concatenate(moment_chunks).astype(RELATION_MOMENT_DTYPE, copy=False)
    depth_count_input = np.concatenate(depth_count_chunks)
    depth_sum_input = np.concatenate(depth_sum_chunks)
    depth_square_sum_input = np.concatenate(depth_square_sum_chunks)
    order = np.lexsort(
        (
            moments["source"].astype(np.int64, copy=False),
            moments["shell"].astype(np.int64, copy=False),
            moments["direction"].astype(np.int64, copy=False),
            moments["target"].astype(np.int64, copy=False),
        )
    )
    values = moments[order]
    depth_count_input = depth_count_input[order]
    depth_sum_input = depth_sum_input[order]
    depth_square_sum_input = depth_square_sum_input[order]
    target_row = values["target"].astype(np.int64, copy=False)
    direction_row = values["direction"].astype(np.int64, copy=False)
    shell_row = values["shell"].astype(np.int64, copy=False)
    source_row = values["source"].astype(np.int64, copy=False)
    starts, _ends, _group_ids = _segment_starts(
        target_row, direction_row, shell_row, source_row
    )

    pixel_sum = np.add.reduceat(values["pixelSum"], starts)
    weight_sum = np.add.reduceat(values["weightSum"], starts)
    pose_count = np.add.reduceat(
        values["poseSupportCount"].astype(np.int64), starts
    )

    def merged_moment(sum_name: str, second_name: str) -> tuple[np.ndarray, np.ndarray]:
        total = np.add.reduceat(values[sum_name], starts)
        second = np.add.reduceat(values[second_name], starts)
        mean = total / np.maximum(weight_sum, 1e-12)
        variance = np.maximum(second / np.maximum(weight_sum, 1e-12) - np.square(mean), 0.0)
        return mean, np.sqrt(variance)

    gap_mean, gap_std = merged_moment("gapWeightedSum", "gapWeightedSecond")
    relative_mean, relative_std = merged_moment(
        "relativeGapWeightedSum", "relativeGapWeightedSecond"
    )
    depth_count = np.add.reduceat(depth_count_input, starts)
    depth_sum = np.add.reduceat(depth_sum_input, starts)
    depth_square_sum = np.add.reduceat(depth_square_sum_input, starts)
    depth_mean = depth_sum / np.maximum(depth_count, 1)
    depth_std = np.sqrt(
        np.maximum(
            depth_square_sum / np.maximum(depth_count, 1) - np.square(depth_mean),
            0.0,
        )
    )

    target = target_row[starts]
    direction = direction_row[starts]
    shell = shell_row[starts]
    source = source_row[starts]
    source_min = np.minimum.reduceat(values["sourceType"], starts)
    source_max = np.maximum.reduceat(values["sourceType"], starts)
    source_types = np.where(
        source_min == source_max,
        source_min,
        np.uint8(SOURCE_TYPES["merged"]),
    ).astype(np.uint8, copy=False)
    cell_starts, _cell_ends, cell_ids = _segment_starts(target, direction, shell)
    cell_pixels = np.add.reduceat(pixel_sum, cell_starts)
    pixel_fraction = pixel_sum / np.maximum(cell_pixels[cell_ids], 1.0)
    pose_rate = pose_count.astype(np.float64) / max(int(total_pose_count), 1)
    relative_center = (centers[source] - centers[target]) / scene_size[None, :]
    relative_scale = np.log(
        np.maximum(radii[source], 1e-5) / np.maximum(radii[target], 1e-5)
    )
    edge_features = np.zeros((target.size, 20), dtype=np.float32)
    edge_features[:, 0] = pose_count
    edge_features[:, 1] = np.clip(pose_rate, 0.0, 1.0)
    edge_features[:, 2] = pixel_sum
    edge_features[:, 3] = np.clip(pixel_fraction, 0.0, 1.0)
    edge_features[:, 4] = np.maximum(gap_mean, 1e-4)
    edge_features[:, 5] = gap_std
    edge_features[:, 6] = np.maximum(
        relative_mean, 1e-4 / max(float(np.linalg.norm(scene_size)), 1.0)
    )
    edge_features[:, 7] = relative_std
    edge_features[:, 8:11] = np.clip(relative_center, -8.0, 8.0)
    edge_features[:, 11] = np.clip(relative_scale, -8.0, 8.0)
    edge_features[:, 12] = np.clip(
        np.sqrt(np.maximum(pose_rate * pixel_fraction, 0.0)), 0.0, 1.0
    )
    edge_features[:, 13] = np.clip(pixel_fraction, 0.0, 1.0)
    edge_features[:, 14] = 1.0
    edge_features[:, 15] = np.maximum(1, pose_count)
    edge_features[:, 16] = 0.0
    edge_features[:, 17] = depth_mean
    edge_features[:, 18] = depth_std
    edge_features[:, 19] = 1.0

    original_target, original_direction = target, direction
    original_shell, original_source = shell, source
    (
        target,
        direction,
        shell,
        source,
        edge_features,
        topk_diagnostics,
    ) = _vectorized_topk(
        target,
        direction,
        shell,
        source.astype(np.uint32, copy=False),
        edge_features,
        k=int(source_k),
    )
    retained_key = np.rec.fromarrays(
        [target, direction, shell, source], names="target,direction,shell,source"
    )
    original_key = np.rec.fromarrays(
        [original_target, original_direction, original_shell, original_source],
        names="target,direction,shell,source",
    )
    retained_indices = np.searchsorted(original_key, retained_key)
    retained_types = source_types[retained_indices]
    stats = {
        "rawRelationRowCount": int(all_depth_values.size),
        "shardMomentCount": int(moments.size),
        "aggregatedEdgeCount": int(starts.size),
        "retainedEdgeCount": int(target.size),
        "cellCountBeforeTopK": int(cell_starts.size),
    }
    return (
        target,
        direction,
        shell,
        source.astype(np.uint32, copy=False),
        edge_features,
        retained_types,
        topk_diagnostics,
        {"depthNormalization": depth_stats, **stats},
    )


def _discover_sparse_shards(root: Path) -> list[Path]:
    candidates = sorted(root.glob("shard_*/sparse"))
    if not candidates and (root / "sparse_relation_meta.json").is_file():
        candidates = [root]
    if not candidates:
        candidates = sorted(
            path.parent for path in root.glob("**/sparse_relation_meta.json")
        )
    if not candidates:
        raise FileNotFoundError(f"no sparse relation shards under {root}")
    return candidates


def _load_sparse_shards(
    root: Path,
    dataset: PoseCSRDataset,
    *,
    num_instances: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    list[Path],
]:
    moment_chunks: list[np.ndarray] = []
    depth_offset_chunks: list[np.ndarray] = []
    depth_value_chunks: list[np.ndarray] = []
    observation_chunks: list[np.ndarray] = []
    render_chunks: list[np.ndarray] = []
    source_pose_chunks: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    paths = _discover_sparse_shards(root)
    canonical_identity: str | None = None
    for shard_ordinal, path in enumerate(paths, start=1):
        (
            meta,
            moments,
            depth_offsets,
            depth_values,
            observations,
            render_ids,
            source_poses,
        ) = load_sparse_shard(path)
        if int(meta.get("numInstances", -1)) != int(num_instances):
            raise ValueError(f"sparse shard instance count mismatch: {path}")
        if float(meta.get("modelInputFovYDeg", 0.0)) != 66.0:
            raise ValueError(f"sparse shard model FOV is not 66 degrees: {path}")
        if meta.get("trainOnly") is not True or meta.get("splitNames") != ["train"]:
            raise ValueError(f"sparse shard is not train-only: {path}")
        if np.any(dataset.poses[source_poses.astype(np.int64)]["split"] != int(dataset.split_ids["train"])):
            raise ValueError(f"sparse shard references a non-train pose: {path}")
        identity = meta.get("candidateIdentity") or {}
        canonical = str(identity.get("canonicalCandidateDigest", ""))
        if not canonical:
            raise ValueError(f"sparse shard lacks canonical candidate identity: {path}")
        if canonical_identity is None:
            canonical_identity = canonical
        elif canonical != canonical_identity:
            raise ValueError("sparse shards disagree on canonical candidate identity")
        expected_render = str(identity.get("renderCandidateDigest", ""))
        actual_render = _candidate_summary(
            dataset, source_poses.astype(np.int64), num_instances
        ).digest
        if expected_render != actual_render:
            raise ValueError(f"sparse shard render candidate identity mismatch: {path}")
        moment_chunks.append(moments)
        depth_offset_chunks.append(depth_offsets)
        depth_value_chunks.append(depth_values)
        observation_chunks.append(observations)
        render_chunks.append(render_ids)
        source_pose_chunks.append(source_poses)
        metadata_rows.append(meta)
        if shard_ordinal == 1 or shard_ordinal == len(paths) or shard_ordinal % 4 == 0:
            print(
                json.dumps(
                    {
                        "status": "sparse_relation_load_progress",
                        "processedShards": shard_ordinal,
                        "totalShards": len(paths),
                        "relationMoments": int(
                            sum(value.size for value in moment_chunks)
                        ),
                        "survivalObservations": int(
                            sum(value.size for value in observation_chunks)
                        ),
                    }
                ),
                flush=True,
            )

    render_ids = np.concatenate(render_chunks).astype("<u4", copy=False)
    source_poses = np.concatenate(source_pose_chunks).astype("<u4", copy=False)
    if np.unique(render_ids).size != render_ids.size:
        raise ValueError("sparse shards contain duplicate render pose IDs")
    order = np.argsort(render_ids, kind="stable")
    if not np.array_equal(render_ids[order], np.arange(render_ids.size, dtype=np.uint32)):
        raise ValueError("formal sparse render pose IDs must cover one contiguous sequence")
    return (
        moment_chunks,
        depth_offset_chunks,
        depth_value_chunks,
        np.concatenate(observation_chunks),
        render_ids[order],
        source_poses[order],
        metadata_rows,
        paths,
    )


def _sparse_input_hashes(
    dataset_dir: Path,
    runtime_meta_path: Path,
    sparse_paths: list[Path],
) -> dict[str, str]:
    # Only small schema envelopes are hashed.  The 200+ GiB transient pixel
    # payload is deliberately absent from the sparse protocol.
    paths = {
        "datasetMeta": dataset_dir / "dataset_meta.json",
        "runtimeMeta": runtime_meta_path,
    }
    for index, root in enumerate(sparse_paths):
        paths[f"sparseShardMeta:{index}"] = root / "sparse_relation_meta.json"
    return {key: _sha256(path) for key, path in paths.items()}


def _collapse_hierarchy_edges(
    target: np.ndarray,
    source: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the strongest edge per unordered pair for union-find packing."""
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    source = np.asarray(source, dtype=np.int64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    low = np.minimum(target, source)
    high = np.maximum(target, source)
    order = np.lexsort((-confidence, high, low))
    low_ordered, high_ordered = low[order], high[order]
    first = np.ones((order.size,), dtype=np.bool_)
    first[1:] = (low_ordered[1:] != low_ordered[:-1]) | (high_ordered[1:] != high_ordered[:-1])
    selected = order[first]
    return target[selected], source[selected], confidence[selected]


def build_sparse(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()

    def progress(stage: str, **values: Any) -> None:
        print(
            json.dumps(
                {
                    "status": "sparse_relation_build_progress",
                    "stage": stage,
                    "elapsedSeconds": float(time.perf_counter() - started),
                    **values,
                }
            ),
            flush=True,
        )

    dataset_dir = Path(args.dataset_dir)
    runtime_meta_path = Path(args.runtime_meta)
    sparse_root = Path(args.sparse_cache_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.surface_fallback_dir:
        raise ValueError("the sparse formal path does not accept a separate surface fallback")
    requested = [value.strip() for value in str(args.splits).split(",") if value.strip()]
    if requested != ["train"]:
        raise ValueError("the hierarchical sparse relation CSR is train-only")

    world_aabbs, _instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    num_instances = int(world_aabbs.shape[0])
    centers = ((world_aabbs[:, :3] + world_aabbs[:, 3:]) * 0.5).astype(np.float32)
    extents = np.maximum(world_aabbs[:, 3:] - world_aabbs[:, :3], 1e-4)
    radii = (np.linalg.norm(extents, axis=1) * 0.5).astype(np.float32)
    scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    scene_size = np.maximum(np.asarray(scene_size, dtype=np.float32), 1e-4)
    dataset = PoseCSRDataset(dataset_dir, num_instances=num_instances)
    canonical_poses = pose_sequence_for_splits(dataset, requested)
    progress("preflight_complete", trainPoseCount=int(canonical_poses.size))

    (
        relation_moment_chunks,
        relation_depth_offset_chunks,
        relation_depth_value_chunks,
        observations,
        render_ids,
        source_poses,
        shard_metadata,
        sparse_paths,
    ) = _load_sparse_shards(sparse_root, dataset, num_instances=num_instances)
    progress(
        "sparse_shards_loaded",
        shardCount=len(sparse_paths),
        renderPoseCount=int(render_ids.size),
        survivalObservationCount=int(observations.size),
    )
    if int(args.max_poses) > 0:
        raise ValueError("formal sparse moment shards do not support partial-pose truncation")

    canonical_summary = _candidate_summary(dataset, canonical_poses, num_instances)
    render_summary = _candidate_summary(dataset, source_poses.astype(np.int64), num_instances)
    expected_canonical = str(
        (shard_metadata[0].get("candidateIdentity") or {}).get(
            "canonicalCandidateDigest", ""
        )
    )
    if canonical_summary.digest != expected_canonical:
        raise ValueError("sparse relation canonical candidate identity changed")
    candidate_audit = dict(shard_metadata[0].get("candidateAudit") or {})
    if candidate_audit.get("status") != "passed" or candidate_audit.get("gtUnionUsed") is not False:
        raise ValueError("sparse relation requires the registered native-candidate audit")

    (
        target_ids,
        direction_ids,
        shell_ids,
        source_ids,
        edge_features,
        source_types,
        topk_diagnostics,
        aggregate_stats,
    ) = _merge_sparse_relation_moments(
        relation_moment_chunks,
        relation_depth_offset_chunks,
        relation_depth_value_chunks,
        centers,
        radii,
        scene_size,
        total_pose_count=int(render_ids.size),
        source_k=int(args.source_k),
    )
    progress(
        "relation_moments_merged",
        retainedEdgeCount=int(target_ids.size),
        sourceK=int(args.source_k),
    )
    depth_stats = aggregate_stats.pop("depthNormalization")
    min_retained_quality = float(
        topk_diagnostics["retainedQualityQuantiles"]["min"]
    )
    if min_retained_quality < float(args.min_retained_quality):
        raise ValueError(
            f"relation top-k retained quality {min_retained_quality:.6f} is below "
            f"the registered floor {float(args.min_retained_quality):.6f}"
        )

    local_diameter, structural_diameter, diameter_derivation = derive_hierarchy_diameter_limits(
        centers,
        target_ids,
        source_ids,
        local_override=float(args.local_group_diameter),
        structural_override=float(args.structural_group_diameter),
    )
    hierarchy_target, hierarchy_source, hierarchy_confidence = _collapse_hierarchy_edges(
        target_ids, source_ids, edge_features[:, 12]
    )
    local_groups, structural_groups = bounded_hierarchy_ids(
        num_instances,
        hierarchy_target,
        hierarchy_source,
        hierarchy_confidence,
        centers,
        local_max_size=32,
        local_diameter=local_diameter,
        structural_max_size=64,
        structural_diameter=structural_diameter,
        max_structural_fraction=0.10,
        local_threshold=float(args.local_group_score),
        structural_threshold=float(args.structure_group_score),
    )
    hierarchy_stats = summarize_bounded_hierarchy(
        local_groups, structural_groups, centers
    )
    progress(
        "hierarchy_built",
        localGroupCount=int(hierarchy_stats["localGroupCount"]),
        structuralGroupCount=int(hierarchy_stats["structuralGroupCount"]),
    )
    hierarchy_metadata = {
        "construction": "confidence-ordered bounded relation packing from unique instance pairs",
        "diameterMetric": "world-space instance-center AABB diagonal upper bound",
        "diameterDerivation": diameter_derivation,
        "localMaxSize": 32,
        "localDiameter": local_diameter,
        "structuralMaxLocalGroups": 64,
        "structuralDiameter": structural_diameter,
        "maxStructuralFraction": 0.10,
        "localThreshold": float(args.local_group_score),
        "structureThreshold": float(args.structure_group_score),
        "uniqueHierarchyEdgeCount": int(hierarchy_target.size),
        **hierarchy_stats,
        "files": {
            "localGroupIds": "instance_local_group_ids_uint32.bin",
            "structuralGroupIds": "local_structural_group_ids_uint32.bin",
        },
    }
    max_layers = {int(meta["maxLayers"]) for meta in shard_metadata}
    if len(max_layers) != 1:
        raise ValueError("sparse shards disagree on depth layer count")
    input_sha256 = _sparse_input_hashes(
        dataset_dir, runtime_meta_path, sparse_paths
    )
    metadata = make_relation_metadata_v3(
        num_instances=num_instances,
        direction_bins=DIRECTION_BINS,
        depth_shells=DEPTH_SHELLS,
        canonical_candidate_summary=canonical_summary,
        render_candidate_summary=render_summary,
        input_sha256=input_sha256,
        cache_schema=SPARSE_SHARD_SCHEMA,
        max_layers=max_layers.pop(),
        model_input_fov_y_deg=66.0,
        depth_normalization=depth_stats,
        hierarchy=hierarchy_metadata,
        evidence_topk={
            "k": int(args.source_k),
            "ranking": "cell-normalized confidence * log pixel support * log pose support",
            "retainedQuality": "retainedEvidenceMass / totalEvidenceMass",
            "score": topk_diagnostics["score"],
            "cellCount": int(topk_diagnostics["cellCount"]),
            "minRetainedQuality": min_retained_quality,
            "qualityFloor": float(args.min_retained_quality),
            "diagnostics": topk_diagnostics,
        },
        surface_fallback={"included": False, "relationCount": 0, "directory": None},
    )
    metadata["candidateAudit"] = candidate_audit
    metadata["cacheCoverage"] = {
        "sparseShardCount": len(sparse_paths),
        "currentTrainPoseCount": int(canonical_poses.size),
        "selectedTrainRenderRowCount": int(render_ids.size),
        "selectedTrainUniquePoseCount": int(np.unique(source_poses).size),
        "uncoveredCurrentTrainPoseCount": int(
            canonical_poses.size - np.unique(source_poses).size
        ),
        "policy": "merge registered train-only sparse render shards; never synthesize relations",
    }
    metadata["scene"] = {
        "sceneMin": scene_min.astype(float).tolist(),
        "sceneSize": scene_size.astype(float).tolist(),
        "directionAnchorCount": DIRECTION_BINS,
    }
    metadata["sourceTypeIds"] = {key: int(value) for key, value in SOURCE_TYPES.items()}
    metadata["sparsePreprocessing"] = {
        "schema": SPARSE_SHARD_SCHEMA,
        "shardCount": len(sparse_paths),
        "denseLayersRequiredByRelationBuild": False,
        **aggregate_stats,
    }

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

    raw_depth = observations["rawDepth"].astype(np.float64)
    normalized_depth = normalize_radius_relative_log_depth(
        np.log1p(raw_depth), depth_stats
    )
    observation_arrays = {
        "instance": observations["instance"].astype("<u4", copy=False),
        "direction": observations["direction"].astype("<u1", copy=False),
        "normalizedDepth": normalized_depth.astype("<f2"),
        "rawDepth": observations["rawDepth"].astype("<f4", copy=False),
        "event": observations["event"].astype("<u1", copy=False),
        "weight": observations["weight"].astype("<f2"),
        "subpose": observations["renderPoseId"].astype("<u4", copy=False),
        "rawPixelCount": observations["rawPixelCount"].astype("<f4"),
        "confidence": observations["confidence"].astype("<f2"),
        "evidenceLevel": observations["evidenceLevel"].astype("<u1", copy=False),
    }
    observation_validation = validate_survival_observations_v3(
        observation_arrays,
        num_instances=num_instances,
        direction_bins=DIRECTION_BINS,
    )
    metadata["survivalObservations"] = {
        "schema": "pvs-viewcell-train-observed-survival-censoring-v3",
        "depthField": "normalizedDepth",
        "rawDepthDefinition": "metric camera-ray range / max(target radius, epsilon)",
        "normalization": metadata["depthNormalization"],
        "count": int(observations.size),
        "eventCount": int(np.count_nonzero(observations["event"] == 1)),
        "validation": observation_validation,
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
        "topKCellCount": int(topk_diagnostics["cellCount"]),
        "topKMinRetainedQuality": min_retained_quality,
        "renderPoseCount": int(render_ids.size),
        "survivalObservationCount": int(observations.size),
        "survivalEventCount": int(np.count_nonzero(observations["event"] == 1)),
    }
    relation.metadata = metadata
    relation.save(output_dir)
    local_groups.tofile(output_dir / hierarchy_metadata["files"]["localGroupIds"])
    structural_groups.tofile(
        output_dir / hierarchy_metadata["files"]["structuralGroupIds"]
    )
    for key, values in observation_arrays.items():
        values.tofile(output_dir / metadata["survivalObservations"]["files"][key])
    relation.metadata = metadata
    relation.save(output_dir)
    progress(
        "complete",
        edgeCount=int(relation.edge_count),
        survivalObservationCount=int(observations.size),
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--sparse-cache-root", required=True)
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
    print(json.dumps(build_sparse(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
