#!/usr/bin/env python3
"""Build the formal degree-preserving shuffled v3 relation control.

The control keeps the native train-only candidate identity, edge evidence
values, source/target degrees, direction/depth strata, survival observations,
and hierarchy budgets. It swaps edge targets only within the same
direction/depth/distance stratum, recomputes source-to-target geometry, and
rebuilds the bounded hierarchy. No validation or test data is opened.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
for value in (ROOT, MODEL_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from common.train_observed_relation_csr import (  # noqa: E402
    ObservedRelationCSR,
    bounded_hierarchy_ids,
    degree_preserving_directed_edge_swap,
    summarize_bounded_hierarchy,
)


CONTROL_SCHEMA = "pvs-degree-preserving-shuffled-relation-control-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _distance_buckets(values: np.ndarray, bucket_count: int) -> tuple[np.ndarray, list[float]]:
    distances = np.asarray(values, dtype=np.float64).reshape(-1)
    if distances.size == 0 or np.any(~np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("edge distances must be a non-empty finite non-negative vector")
    if int(bucket_count) <= 0:
        raise ValueError("distance bucket count must be positive")
    boundaries = np.unique(
        np.quantile(distances, np.linspace(0.0, 1.0, int(bucket_count) + 1)[1:-1])
    )
    return np.searchsorted(boundaries, distances, side="right").astype(np.int64), boundaries.tolist()


def _degree_digest(
    source: np.ndarray,
    target: np.ndarray,
    direction: np.ndarray,
    shell: np.ndarray,
    distance_bucket: np.ndarray,
) -> str:
    rows = np.column_stack((source, target, direction, shell, distance_bucket)).astype("<i8", copy=False)
    return hashlib.sha256(rows.tobytes(order="C")).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    relation_dir = Path(args.relation_dir).resolve()
    runtime_meta_path = Path(args.runtime_meta).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite shuffled relation output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    relation = ObservedRelationCSR.load(relation_dir)
    if relation.metadata.get("splitNames") != ["train"] or relation.metadata.get("trainOnly") is not True:
        raise ValueError("shuffled control requires the train-only v3 relation artifact")
    target, direction, shell = relation.row_indices()
    source = relation.source_ids.astype(np.int64, copy=False)
    world_aabbs, _instance_to_glb, runtime_meta = load_runtime_meta(
        runtime_meta_path, relation.num_instances
    )
    centers = ((world_aabbs[:, :3] + world_aabbs[:, 3:]) * 0.5).astype(np.float64)
    extents = np.maximum(world_aabbs[:, 3:] - world_aabbs[:, :3], 1e-6)
    radii = np.linalg.norm(extents, axis=1) * 0.5
    _scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    scene_size = np.maximum(np.asarray(scene_size, dtype=np.float64), 1e-6)
    distances = np.linalg.norm(centers[source] - centers[target], axis=1)
    distance_bucket, distance_boundaries = _distance_buckets(distances, int(args.distance_buckets))

    shuffled_source, shuffled_target = degree_preserving_directed_edge_swap(
        source,
        target,
        direction,
        shell,
        distance_bucket_ids=distance_bucket,
        seed=int(args.seed),
        swap_fraction=float(args.swap_fraction),
    )
    shuffled_source_i64 = shuffled_source.astype(np.int64, copy=False)
    shuffled_target_i64 = shuffled_target.astype(np.int64, copy=False)
    changed = shuffled_target_i64 != target
    if not bool(changed.any()):
        raise ValueError("degree-preserving shuffle changed no relation edge")

    before_source_degree = np.bincount(source, minlength=relation.num_instances)
    after_source_degree = np.bincount(shuffled_source_i64, minlength=relation.num_instances)
    before_target_degree = np.bincount(target, minlength=relation.num_instances)
    after_target_degree = np.bincount(shuffled_target_i64, minlength=relation.num_instances)
    if not np.array_equal(before_source_degree, after_source_degree):
        raise RuntimeError("shuffled control changed source degree")
    if not np.array_equal(before_target_degree, after_target_degree):
        raise RuntimeError("shuffled control changed target degree")

    features = relation.edge_features.copy()
    features[:, 8:11] = np.clip(
        (centers[shuffled_source_i64] - centers[shuffled_target_i64]) / scene_size,
        -8.0,
        8.0,
    ).astype(np.float32)
    features[:, 11] = np.clip(
        np.log(
            np.maximum(radii[shuffled_source_i64], 1e-6)
            / np.maximum(radii[shuffled_target_i64], 1e-6)
        ),
        -8.0,
        8.0,
    ).astype(np.float32)

    hierarchy = relation.metadata["hierarchy"]
    local, structural = bounded_hierarchy_ids(
        relation.num_instances,
        shuffled_target_i64,
        shuffled_source_i64,
        features[:, 12],
        centers,
        local_max_size=int(hierarchy["localMaxSize"]),
        local_diameter=float(hierarchy["localDiameter"]),
        structural_max_size=int(hierarchy["structuralMaxLocalGroups"]),
        structural_diameter=float(hierarchy["structuralDiameter"]),
        max_structural_fraction=float(hierarchy["maxStructuralFraction"]),
        local_threshold=float(hierarchy["localThreshold"]),
        structural_threshold=float(hierarchy["structureThreshold"]),
    )
    hierarchy_stats = summarize_bounded_hierarchy(local, structural, centers)

    metadata = copy.deepcopy(relation.metadata)
    input_hashes = dict(metadata["inputSha256"])
    for key, relative in metadata["files"].items():
        input_hashes[f"nativeRelation:{key}"] = _sha256(relation_dir / relative)
    input_hashes["nativeRelation:metadata"] = _sha256(relation_dir / "relation_csr_meta.json")
    metadata["inputSha256"] = input_hashes
    metadata["control"] = {
        "schema": CONTROL_SCHEMA,
        "seed": int(args.seed),
        "swapFractionRequested": float(args.swap_fraction),
        "changedEdgeCount": int(changed.sum()),
        "changedEdgeFraction": float(changed.mean()),
        "strata": ["direction", "depth_shell", "train_edge_distance_quantile"],
        "distanceBucketCountRequested": int(args.distance_buckets),
        "distanceBucketCountActual": int(distance_bucket.max()) + 1,
        "distanceBucketBoundariesMeters": distance_boundaries,
        "sourceAndTargetDegreesPreserved": True,
        "relativeCenterAndScaleRecomputed": True,
        "boundedHierarchyRebuilt": True,
        "nativeRelationDirectory": str(relation_dir),
        "nativeEdgeOrderDigest": _degree_digest(source, target, direction, shell, distance_bucket),
        "shuffledEdgeOrderDigest": _degree_digest(
            shuffled_source_i64, shuffled_target_i64, direction, shell, distance_bucket
        ),
    }
    shuffled_hierarchy = {
        **hierarchy,
        **hierarchy_stats,
        "construction": "bounded hierarchy rebuilt from degree-preserving shuffled train relation",
    }
    shuffled_hierarchy.pop("structureGroupCount", None)
    metadata["hierarchy"] = shuffled_hierarchy
    metadata["stats"] = {
        **dict(metadata.get("stats") or {}),
        "edgeCount": int(relation.edge_count),
        "shuffleChangedEdgeCount": int(changed.sum()),
        "shuffleChangedEdgeFraction": float(changed.mean()),
    }

    shuffled_relation = ObservedRelationCSR.from_rows(
        num_instances=relation.num_instances,
        direction_bins=relation.direction_bins,
        depth_shells=relation.depth_shells,
        target_ids=shuffled_target_i64,
        direction_ids=direction,
        depth_shell_ids=shell,
        source_ids=shuffled_source,
        edge_features=features,
        source_types=relation.source_types,
        metadata=metadata,
    )
    shuffled_relation.save(output_dir)
    local.astype("<u4", copy=False).tofile(
        output_dir / metadata["hierarchy"]["files"]["localGroupIds"]
    )
    structural.astype("<u4", copy=False).tofile(
        output_dir / metadata["hierarchy"]["files"]["structuralGroupIds"]
    )
    observation_files = (metadata.get("survivalObservations") or {}).get("files") or {}
    for relative in observation_files.values():
        source_path = relation_dir / str(relative)
        if not source_path.is_file():
            raise FileNotFoundError(f"native relation observation is missing: {source_path}")
        shutil.copy2(source_path, output_dir / str(relative))
    (output_dir / "relation_csr_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ObservedRelationCSR.load(output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--distance-buckets", type=int, default=8)
    parser.add_argument("--swap-fraction", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
