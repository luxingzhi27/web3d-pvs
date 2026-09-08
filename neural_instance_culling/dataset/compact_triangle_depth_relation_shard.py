#!/usr/bin/env python3
"""Convert one dense triangle-depth shard into compact train-only evidence.

The dense ID/depth layers are a transient renderer product.  This module
extracts the only information consumed by the V4 relation and survival
training paths, then writes fixed-width sparse arrays that can be merged
without revisiting pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATASET_DIR = ROOT / "dataset"
for value in (ROOT, MODEL_DIR, DATASET_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_ray_context_relation_evidence import _reduce_pose_events  # noqa: E402
from build_triangle_depth_layer_evidence import (  # noqa: E402
    BACKGROUND_ID,
    _camera_for_cache_row,
    build_survival_evidence_records,
    load_layer_cache,
    metric_ray_depths_for_cache_row,
    spherical_direction_bins,
)
from common.runtime_meta import load_runtime_meta  # noqa: E402
from common.train_observed_relation_csr import (  # noqa: E402
    SOURCE_TYPES,
    radius_relative_log_depth,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


SCHEMA = "triangle-depth-relation-sparse-shard-v1"
DIRECTION_BINS = 12
DEPTH_SHELLS = 3

# One row represents one adjacent-layer relation observation.  Rows are not
# globally aggregated here: renderPoseId remains explicit so final pose
# support is computed exactly during the vectorized merge.
RELATION_ROW_DTYPE = np.dtype(
    [
        ("target", "<u4"),
        ("source", "<u4"),
        ("direction", "u1"),
        ("shell", "u1"),
        ("sourceType", "u1"),
        ("reserved", "u1"),
        ("renderPoseId", "<u4"),
        ("pixelCount", "<f4"),
        ("gapMean", "<f4"),
        ("gapStd", "<f4"),
        ("relativeGapMean", "<f4"),
        ("relativeGapStd", "<f4"),
        ("relativeLogDepth", "<f4"),
    ],
    align=False,
)

# Per-shard sufficient statistics for every target/direction/shell/source key.
# Depth samples stay in a separate segmented vector because their global
# q01/q99 clipping is only known after every shard has completed.
RELATION_MOMENT_DTYPE = np.dtype(
    [
        ("target", "<u4"),
        ("source", "<u4"),
        ("direction", "u1"),
        ("shell", "u1"),
        ("sourceType", "u1"),
        ("reserved", "u1"),
        ("poseSupportCount", "<u4"),
        ("pixelSum", "<f8"),
        ("weightSum", "<f8"),
        ("gapWeightedSum", "<f8"),
        ("gapWeightedSecond", "<f8"),
        ("relativeGapWeightedSum", "<f8"),
        ("relativeGapWeightedSecond", "<f8"),
    ],
    align=False,
)

SURVIVAL_OBSERVATION_DTYPE = np.dtype(
    [
        ("instance", "<u4"),
        ("direction", "u1"),
        ("event", "u1"),
        ("evidenceLevel", "u1"),
        ("reserved", "u1"),
        ("renderPoseId", "<u4"),
        ("rawDepth", "<f4"),
        ("rawPixelCount", "<u4"),
        ("weight", "<f4"),
        ("confidence", "<f4"),
    ],
    align=False,
)


def _dtype_description(dtype: np.dtype) -> list[list[str]]:
    return [[str(name), str(field[0].str)] for name, field in dtype.fields.items()]


def _structured_rows(records: list[dict[str, Any]]) -> np.ndarray:
    rows = np.zeros((len(records),), dtype=SURVIVAL_OBSERVATION_DTYPE)
    for index, record in enumerate(records):
        rows[index]["instance"] = int(record["instance"])
        rows[index]["direction"] = int(record["direction"])
        rows[index]["event"] = int(record["event"])
        rows[index]["evidenceLevel"] = int(record["evidenceLevel"])
        rows[index]["renderPoseId"] = int(record["subpose"])
        rows[index]["rawDepth"] = float(record["depth"])
        rows[index]["rawPixelCount"] = int(record["rawPixelCount"])
        rows[index]["weight"] = float(record["weight"])
        rows[index]["confidence"] = float(record["confidence"])
    return rows


def aggregate_relation_moments(
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce repeated pose rows while retaining exact global-depth samples."""
    rows = np.asarray(rows, dtype=RELATION_ROW_DTYPE)
    if rows.size == 0:
        return (
            np.zeros((0,), dtype=RELATION_MOMENT_DTYPE),
            np.zeros((1,), dtype="<u8"),
            np.zeros((0,), dtype="<f4"),
        )
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
    key_change = np.ones((values.size,), dtype=np.bool_)
    key_change[1:] = (
        (values["target"][1:] != values["target"][:-1])
        | (values["direction"][1:] != values["direction"][:-1])
        | (values["shell"][1:] != values["shell"][:-1])
        | (values["source"][1:] != values["source"][:-1])
    )
    starts = np.flatnonzero(key_change).astype(np.int64, copy=False)
    ends = np.r_[starts[1:], values.size].astype(np.int64, copy=False)
    group_ids = np.cumsum(key_change, dtype=np.int64) - 1
    new_pose = np.ones((values.size,), dtype=np.bool_)
    new_pose[1:] = key_change[1:] | (
        values["renderPoseId"][1:] != values["renderPoseId"][:-1]
    )
    weight = np.maximum(values["pixelCount"].astype(np.float64), 1.0)
    moments = np.zeros((starts.size,), dtype=RELATION_MOMENT_DTYPE)
    moments["target"] = values["target"][starts]
    moments["source"] = values["source"][starts]
    moments["direction"] = values["direction"][starts]
    moments["shell"] = values["shell"][starts]
    source_min = np.minimum.reduceat(values["sourceType"], starts)
    source_max = np.maximum.reduceat(values["sourceType"], starts)
    moments["sourceType"] = np.where(
        source_min == source_max,
        source_min,
        np.uint8(SOURCE_TYPES["merged"]),
    )
    moments["poseSupportCount"] = np.add.reduceat(
        new_pose.astype(np.uint32), starts
    )
    moments["pixelSum"] = np.add.reduceat(
        values["pixelCount"].astype(np.float64), starts
    )
    moments["weightSum"] = np.add.reduceat(weight, starts)
    gap_mean = values["gapMean"].astype(np.float64)
    gap_std = values["gapStd"].astype(np.float64)
    relative_mean = values["relativeGapMean"].astype(np.float64)
    relative_std = values["relativeGapStd"].astype(np.float64)
    moments["gapWeightedSum"] = np.add.reduceat(weight * gap_mean, starts)
    moments["gapWeightedSecond"] = np.add.reduceat(
        weight * (np.square(gap_std) + np.square(gap_mean)), starts
    )
    moments["relativeGapWeightedSum"] = np.add.reduceat(
        weight * relative_mean, starts
    )
    moments["relativeGapWeightedSecond"] = np.add.reduceat(
        weight * (np.square(relative_std) + np.square(relative_mean)), starts
    )
    depth_offsets = np.r_[starts, values.size].astype("<u8", copy=False)
    depth_values = values["relativeLogDepth"].astype("<f4", copy=True)
    if not np.array_equal(group_ids, np.repeat(np.arange(starts.size), ends - starts)):
        raise ValueError("relation moment depth segmentation is inconsistent")
    return moments, depth_offsets, depth_values


def extract_sparse_pose(
    ids: np.ndarray,
    metric_depths: np.ndarray,
    candidate_ids: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    camera_world: np.ndarray,
    *,
    render_pose_id: int,
    min_depth_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract relation rows and conserved survival observations for one pose."""
    ids = np.asarray(ids, dtype=np.uint32)
    depths = np.asarray(metric_depths, dtype=np.float32)
    candidates = np.asarray(candidate_ids, dtype=np.uint32).reshape(-1)
    centers = np.asarray(centers, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    camera = np.asarray(camera_world, dtype=np.float32).reshape(3)
    if ids.ndim != 3 or depths.shape != ids.shape:
        raise ValueError("pose ID/depth layers must have shape [layers, height, width]")
    if centers.ndim != 2 or centers.shape[1] != 3 or radii.size != centers.shape[0]:
        raise ValueError("instance centers/radii have incompatible shapes")
    if candidates.size and int(candidates.max()) >= centers.shape[0]:
        raise ValueError("candidate ID is outside the runtime instance table")

    candidate_mask = np.zeros((centers.shape[0],), dtype=np.bool_)
    candidate_mask[candidates.astype(np.int64, copy=False)] = True
    first = np.unique(ids[0][ids[0] != BACKGROUND_ID])
    if first.size and not bool(np.all(candidate_mask[first.astype(np.int64, copy=False)])):
        outside = first[~candidate_mask[first.astype(np.int64, copy=False)]]
        raise ValueError(
            f"first depth layer contains IDs outside native candidates: {outside[:8].tolist()}"
        )

    directions = np.zeros((centers.shape[0],), dtype=np.int64)
    if candidates.size:
        candidate_index = candidates.astype(np.int64, copy=False)
        directions[candidate_index] = spherical_direction_bins(
            camera, centers[candidate_index]
        )
    reduced = _reduce_pose_events(ids, depths, candidate_mask, float(min_depth_gap))
    relation = np.zeros((reduced.shape[0],), dtype=RELATION_ROW_DTYPE)
    if reduced.size:
        target = reduced[:, 0].astype(np.int64, copy=False)
        source = reduced[:, 1].astype(np.int64, copy=False)
        target_distance = np.maximum(
            np.linalg.norm(centers[target] - camera[None, :], axis=1), 1e-4
        )
        relation["target"] = target.astype(np.uint32, copy=False)
        relation["source"] = source.astype(np.uint32, copy=False)
        relation["direction"] = directions[target].astype(np.uint8, copy=False)
        relation["shell"] = reduced[:, 2].astype(np.uint8, copy=False)
        relation["sourceType"] = np.uint8(SOURCE_TYPES["depth_peeling"])
        relation["renderPoseId"] = np.uint32(render_pose_id)
        relation["pixelCount"] = reduced[:, 3]
        relation["gapMean"] = reduced[:, 4]
        relation["gapStd"] = reduced[:, 5]
        relation["relativeGapMean"] = reduced[:, 4] / target_distance
        relation["relativeGapStd"] = reduced[:, 5] / target_distance
        relation["relativeLogDepth"] = radius_relative_log_depth(
            target_distance, radii[target]
        ).astype(np.float32, copy=False)

    observations, mass = build_survival_evidence_records(
        ids,
        depths,
        set(int(value) for value in candidates.tolist()),
        directions,
        centers,
        radii,
        camera,
        float(min_depth_gap),
        subpose_id=int(render_pose_id),
    )
    observation_rows = _structured_rows(observations)
    if observation_rows.size and not np.isclose(
        float(observation_rows["weight"].sum(dtype=np.float64)), 1.0, rtol=1e-5, atol=1e-5
    ):
        raise ValueError(
            f"survival weights do not sum to one for render pose {render_pose_id}: "
            f"{mass.get('weightSum')}"
        )
    return relation, observation_rows


def _write_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    np.ascontiguousarray(values).tofile(temporary)
    temporary.replace(path)


def _write_compressed_array(path: Path, values: np.ndarray) -> tuple[int, int]:
    """Write one fixed-width array with fast, deterministic zstd framing."""
    executable = shutil.which("zstd")
    if not executable:
        raise FileNotFoundError("formal sparse caches require the zstd executable")
    raw_path = path.with_suffix("") if path.suffix == ".zst" else path
    _write_array(raw_path, values)
    raw_bytes = raw_path.stat().st_size
    result = subprocess.run(
        [executable, "-q", "-T1", "-3", "-f", "--rm", str(raw_path), "-o", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not path.is_file():
        raise RuntimeError(f"zstd compression failed for {raw_path}: {result.stderr}")
    return int(raw_bytes), int(path.stat().st_size)


def _read_array(path: Path, dtype: np.dtype) -> np.ndarray:
    if path.suffix != ".zst":
        return np.fromfile(path, dtype=dtype)
    executable = shutil.which("zstd")
    if not executable:
        raise FileNotFoundError("reading sparse caches requires the zstd executable")
    with tempfile.NamedTemporaryFile(prefix="pvs_sparse_", suffix=".bin") as temporary:
        result = subprocess.run(
            [executable, "-q", "-d", "-f", str(path), "-o", temporary.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"zstd decompression failed for {path}: {result.stderr}")
        return np.fromfile(temporary.name, dtype=dtype)


def compact_shard(
    cache_dir: Path,
    output_dir: Path,
    *,
    min_depth_gap: float = 1e-4,
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty sparse shard: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_meta, render_ids, source_pose_indices, cache_ids, cache_depths = load_layer_cache(
        cache_dir
    )
    manifest_path = Path(str(cache_meta.get("sourceManifest", "")))
    if not manifest_path.is_file():
        raise FileNotFoundError("dense shard metadata must reference its render manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_dir = Path(str(manifest.get("datasetDir", "")))
    runtime_meta_path = Path(str(manifest.get("runtimeMeta", "")))
    if not dataset_dir.is_dir() or not runtime_meta_path.is_file():
        raise FileNotFoundError("render manifest datasetDir/runtimeMeta is unavailable")

    world_aabbs, _mapping, runtime_meta = load_runtime_meta(runtime_meta_path)
    num_instances = int(world_aabbs.shape[0])
    centers = ((world_aabbs[:, :3] + world_aabbs[:, 3:]) * 0.5).astype(np.float32)
    radii = (
        np.linalg.norm(np.maximum(world_aabbs[:, 3:] - world_aabbs[:, :3], 1e-4), axis=1)
        * 0.5
    ).astype(np.float32)
    dataset = PoseCSRDataset(dataset_dir, num_instances=num_instances)
    train_id = int(dataset.split_ids["train"])
    if render_ids.size != source_pose_indices.size:
        raise ValueError("render/source pose identity arrays differ in length")
    if np.unique(render_ids).size != render_ids.size:
        raise ValueError("render pose IDs must be unique within a sparse shard")

    relation_chunks: list[np.ndarray] = []
    observation_chunks: list[np.ndarray] = []
    started = time.perf_counter()
    for row in range(int(render_ids.size)):
        source_pose = int(source_pose_indices[row])
        if int(dataset.poses[source_pose]["split"]) != train_id:
            raise ValueError("sparse relation compaction may only consume train poses")
        encoded_depth = np.asarray(cache_depths[row], dtype=np.float32)
        camera_world, _forward, camera_view = _camera_for_cache_row(
            cache_meta, int(render_ids[row]), dataset, source_pose
        )
        metric_depth = metric_ray_depths_for_cache_row(
            cache_meta, encoded_depth, camera_view
        )
        relation, observations = extract_sparse_pose(
            np.asarray(cache_ids[row], dtype=np.uint32),
            metric_depth,
            np.asarray(dataset.frustum_slice(source_pose), dtype=np.uint32),
            centers,
            radii,
            camera_world,
            render_pose_id=int(render_ids[row]),
            min_depth_gap=float(min_depth_gap),
        )
        if relation.size:
            relation_chunks.append(relation)
        if observations.size:
            observation_chunks.append(observations)
        ordinal = row + 1
        if ordinal == 1 or ordinal == render_ids.size or ordinal % max(1, render_ids.size // 10) == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "status": "sparse_triangle_compaction_progress",
                        "processed": ordinal,
                        "total": int(render_ids.size),
                        "posesPerSecond": ordinal / elapsed,
                        "relationRows": int(sum(value.size for value in relation_chunks)),
                        "survivalRows": int(sum(value.size for value in observation_chunks)),
                    }
                ),
                flush=True,
            )

    relations = (
        np.concatenate(relation_chunks)
        if relation_chunks
        else np.zeros((0,), dtype=RELATION_ROW_DTYPE)
    )
    observations = (
        np.concatenate(observation_chunks)
        if observation_chunks
        else np.zeros((0,), dtype=SURVIVAL_OBSERVATION_DTYPE)
    )
    if relations.size == 0:
        raise ValueError("dense shard produced no sparse relation rows")
    for name in (
        "pixelCount",
        "gapMean",
        "gapStd",
        "relativeGapMean",
        "relativeGapStd",
        "relativeLogDepth",
    ):
        if not bool(np.isfinite(relations[name]).all()):
            raise ValueError(f"non-finite sparse relation field: {name}")
    for name in ("rawDepth", "weight", "confidence"):
        if not bool(np.isfinite(observations[name]).all()):
            raise ValueError(f"non-finite sparse survival field: {name}")

    relation_moments, relation_depth_offsets, relation_depth_values = (
        aggregate_relation_moments(relations)
    )
    raw_relation_row_count = int(relations.size)
    del relations

    files = {
        "relationMoments": "relation_moments.bin.zst",
        "relationDepthOffsets": "relation_depth_offsets_uint64.bin.zst",
        "relationDepthValues": "relation_depth_values_fp32.bin.zst",
        "survivalObservations": "survival_observations.bin.zst",
        "renderPoseIds": "render_pose_ids_uint32.bin",
        "sourcePoseIndices": "source_pose_indices_uint32.bin",
    }
    moment_raw_bytes, moment_compressed_bytes = _write_compressed_array(
        output_dir / files["relationMoments"], relation_moments
    )
    depth_offset_raw_bytes, depth_offset_compressed_bytes = _write_compressed_array(
        output_dir / files["relationDepthOffsets"], relation_depth_offsets
    )
    depth_value_raw_bytes, depth_value_compressed_bytes = _write_compressed_array(
        output_dir / files["relationDepthValues"], relation_depth_values
    )
    observation_raw_bytes, observation_compressed_bytes = _write_compressed_array(
        output_dir / files["survivalObservations"], observations
    )
    _write_array(output_dir / files["renderPoseIds"], render_ids.astype("<u4", copy=False))
    _write_array(
        output_dir / files["sourcePoseIndices"],
        source_pose_indices.astype("<u4", copy=False),
    )

    first_layer = cache_meta.get("firstLayerReference") or {}
    formal_ready = bool(
        int(first_layer.get("checks", -1)) == int(render_ids.size)
        and int(first_layer.get("mismatches", -1)) == 0
        and (cache_meta.get("gpuEvidenceSummary") or {}).get("formalReady") is True
    )
    if not formal_ready:
        raise ValueError("sparse shard cannot inherit an incomplete formal GPU/first-layer gate")
    metadata = {
        "schema": SCHEMA,
        "formalReady": True,
        "trainOnly": True,
        "splitNames": ["train"],
        "modelInputFovYDeg": float(cache_meta["modelInputFovYDeg"]),
        "width": int(cache_meta["width"]),
        "height": int(cache_meta["height"]),
        "maxLayers": int(cache_meta["maxLayers"]),
        "cameraFarMeters": float(cache_meta["cameraFarMeters"]),
        "linearDepthEncoding": str(cache_meta["linearDepthEncoding"]),
        "numInstances": num_instances,
        "poseCount": int(render_ids.size),
        "relationRowCount": raw_relation_row_count,
        "relationMomentCount": int(relation_moments.size),
        "relationDepthValueCount": int(relation_depth_values.size),
        "survivalObservationCount": int(observations.size),
        "survivalEventCount": int(np.count_nonzero(observations["event"] == 1)),
        "directionBins": DIRECTION_BINS,
        "depthShells": DEPTH_SHELLS,
        "minDepthGap": float(min_depth_gap),
        "relationRowDtype": _dtype_description(RELATION_ROW_DTYPE),
        "relationMomentDtype": _dtype_description(RELATION_MOMENT_DTYPE),
        "survivalObservationDtype": _dtype_description(SURVIVAL_OBSERVATION_DTYPE),
        "files": files,
        "compression": {
            "codec": "zstd",
            "level": 3,
            "relationMomentRawBytes": moment_raw_bytes,
            "relationMomentCompressedBytes": moment_compressed_bytes,
            "relationDepthOffsetRawBytes": depth_offset_raw_bytes,
            "relationDepthOffsetCompressedBytes": depth_offset_compressed_bytes,
            "relationDepthValueRawBytes": depth_value_raw_bytes,
            "relationDepthValueCompressedBytes": depth_value_compressed_bytes,
            "survivalRawBytes": observation_raw_bytes,
            "survivalCompressedBytes": observation_compressed_bytes,
        },
        "sourceDenseCache": str(cache_dir),
        "sourceManifest": str(manifest_path.resolve()),
        "datasetDir": str(dataset_dir.resolve()),
        "runtimeMeta": str(runtime_meta_path.resolve()),
        "candidateIdentity": cache_meta.get("candidateIdentity") or {},
        "candidateAudit": manifest.get("candidateAudit") or {},
        "firstLayerReference": first_layer,
        "gpuEvidenceSummary": cache_meta.get("gpuEvidenceSummary"),
        "candidateSemantics": "native stored candidate CSR only; no GT union",
        "elapsedSeconds": float(time.perf_counter() - started),
    }
    meta_path = output_dir / "sparse_relation_meta.json"
    temporary_meta = meta_path.with_name(f".{meta_path.name}.partial")
    temporary_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_meta.replace(meta_path)
    return metadata


def load_sparse_shard(
    path: str | Path,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    root = Path(path)
    meta_path = root / "sparse_relation_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA or metadata.get("formalReady") is not True:
        raise ValueError(f"invalid sparse relation shard: {root}")
    files = metadata.get("files") or {}
    moments = _read_array(root / files["relationMoments"], RELATION_MOMENT_DTYPE)
    depth_offsets = _read_array(root / files["relationDepthOffsets"], np.dtype("<u8"))
    depth_values = _read_array(root / files["relationDepthValues"], np.dtype("<f4"))
    observations = _read_array(
        root / files["survivalObservations"], SURVIVAL_OBSERVATION_DTYPE
    )
    render_ids = np.fromfile(root / files["renderPoseIds"], dtype="<u4")
    source_ids = np.fromfile(root / files["sourcePoseIndices"], dtype="<u4")
    expected = {
        "relationMomentCount": moments.size,
        "relationDepthValueCount": depth_values.size,
        "survivalObservationCount": observations.size,
        "poseCount": render_ids.size,
    }
    for key, actual in expected.items():
        if int(metadata.get(key, -1)) != int(actual):
            raise ValueError(f"sparse shard {key} mismatch: {metadata.get(key)} != {actual}")
    if (
        depth_offsets.size != moments.size + 1
        or int(depth_offsets[0]) != 0
        or int(depth_offsets[-1]) != depth_values.size
        or np.any(np.diff(depth_offsets) <= 0)
    ):
        raise ValueError("sparse shard relation depth segmentation is invalid")
    if source_ids.size != render_ids.size or np.unique(render_ids).size != render_ids.size:
        raise ValueError("sparse shard pose identity is incomplete or duplicated")
    return (
        metadata,
        moments,
        depth_offsets,
        depth_values,
        observations,
        render_ids,
        source_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-depth-gap", type=float, default=1e-4)
    args = parser.parse_args()
    print(
        json.dumps(
            compact_shard(
                args.cache_dir,
                args.output_dir,
                min_depth_gap=float(args.min_depth_gap),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
