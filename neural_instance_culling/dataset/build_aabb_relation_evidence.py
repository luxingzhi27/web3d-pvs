#!/usr/bin/env python3
"""Build the registered AABB-projection relation supplement.

This is an offline evidence ablation only.  It uses the unchanged native
back-camera candidate CSR and the GT visible IDs already stored for each
training pose.  A candidate-negative target receives evidence from a
candidate-visible source when their projected AABBs overlap and the source
is closer to the camera.  The resulting table has the same 12 spherical
directions, 3 depth shells, 8 source slots, and survival-observation files as
the triangle-depth relation table, so the model comparison changes only the
relation evidence source.

The AABB evidence is deliberately not used as the formal occlusion
supervision in the paper.  It exists to quantify the gap between coarse
projection overlap and triangle-depth evidence without changing candidates,
GT, split rows, or the survival loss observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys
from typing import Any

import numpy as np

DATASET_DIR = Path(__file__).resolve().parent
MODEL_DIR = DATASET_DIR.parent / "model"
for _path in (DATASET_DIR, MODEL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_ray_context_relation_evidence import (  # noqa: E402
    DIRECTION_BINS,
    DEPTH_SHELLS,
    RELATION_STAT_DIM,
    SOURCE_K,
    _aggregate_rows,
    _shell_id,
    _soft_direction_neighbors,
)
from common.candidate_identity import (  # noqa: E402
    audit_native_aabb_candidates,
    candidate_digest_for_pose_sequence,
    pose_sequence_for_splits,
)
from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from pose_csr_dataset import PoseCSRDataset, _project_aabb_features_numpy  # noqa: E402


SCHEMA = "ray-context-relation-evidence-v2"
SURVIVAL_SCHEMA = "triangle-depth-layer-evidence-v1"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _rect_overlap_ratio(target_rect: np.ndarray, source_rect: np.ndarray) -> np.ndarray:
    x0 = np.maximum(target_rect[:, None, 0], source_rect[None, :, 0])
    y0 = np.maximum(target_rect[:, None, 1], source_rect[None, :, 1])
    x1 = np.minimum(target_rect[:, None, 2], source_rect[None, :, 2])
    y1 = np.minimum(target_rect[:, None, 3], source_rect[None, :, 3])
    intersection = np.maximum(x1 - x0, 0.0) * np.maximum(y1 - y0, 0.0)
    target_area = np.maximum(target_rect[:, 2] - target_rect[:, 0], 0.0) * np.maximum(
        target_rect[:, 3] - target_rect[:, 1], 0.0
    )
    return intersection / np.maximum(target_area[:, None], 1e-8)


def _aabb_pose_rows(
    dataset: PoseCSRDataset,
    pose_index: int,
    world_aabbs: np.ndarray,
    centers: np.ndarray,
    num_instances: int,
    min_overlap: float,
    min_evidence: float,
    min_depth_gap: float,
    min_source_area: float,
    min_target_area: float,
    target_chunk_size: int,
    max_targets: int,
    max_sources: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return per-pose rows [target, source, shell, pixels, gap, std]."""
    visible_slice = dataset.visible_slice(int(pose_index))
    visible_values = visible_slice[0] if isinstance(visible_slice, tuple) else visible_slice
    visible = np.unique(np.asarray(visible_values, dtype=np.int64))
    candidate = np.asarray(dataset.frustum_slice(int(pose_index)), dtype=np.int64)
    if candidate.size and int(candidate.max()) >= num_instances:
        raise ValueError(f"candidate ID exceeds runtime instance count at pose {pose_index}")
    candidate = np.unique(candidate)
    if visible.size and not np.all(np.isin(visible, candidate)):
        missing = np.setdiff1d(visible, candidate, assume_unique=False)
        raise ValueError(
            f"stored candidates omit {missing.size} visible IDs at pose {pose_index}; "
            "AABB supplement cannot repair the candidate set"
        )
    targets = np.setdiff1d(candidate, visible, assume_unique=True)
    if targets.size == 0 or visible.size == 0:
        return np.zeros((0, 6), dtype=np.float32), {
            "candidateCount": int(candidate.size),
            "targetCount": int(targets.size),
            "sourceCount": int(visible.size),
            "positivePairs": 0,
        }
    mvp = dataset.mvp_slice(int(pose_index))
    target_rect_all, target_area_all, target_depth_all, target_valid_all = _project_aabb_features_numpy(
        world_aabbs[targets], mvp
    )
    target_keep = target_valid_all & (target_area_all > float(min_target_area))
    targets = targets[target_keep]
    target_rect_all = target_rect_all[target_keep]
    target_area_all = target_area_all[target_keep]
    target_depth_all = target_depth_all[target_keep]
    if max_targets > 0 and targets.size > max_targets:
        target_score = np.log1p(np.clip(target_area_all, 0.0, 4.0) * 2048.0)
        keep = np.argpartition(target_score, -int(max_targets))[-int(max_targets):]
        targets = targets[keep]
        target_rect_all = target_rect_all[keep]
        target_area_all = target_area_all[keep]
        target_depth_all = target_depth_all[keep]
    source_rect, source_area, source_depth, source_valid = _project_aabb_features_numpy(
        world_aabbs[visible], mvp
    )
    source_keep = source_valid & (source_area > float(min_source_area))
    source_ids = visible[source_keep]
    source_rect = source_rect[source_keep]
    source_area = source_area[source_keep]
    source_depth = source_depth[source_keep]
    if max_sources > 0 and source_ids.size > max_sources:
        source_score = np.log1p(np.clip(source_area, 0.0, 4.0) * 2048.0)
        keep = np.argpartition(source_score, -int(max_sources))[-int(max_sources):]
        source_ids = source_ids[keep]
        source_rect = source_rect[keep]
        source_area = source_area[keep]
        source_depth = source_depth[keep]
    if targets.size == 0 or source_ids.size == 0:
        return np.zeros((0, 6), dtype=np.float32), {
            "candidateCount": int(candidate.size),
            "targetCount": int(targets.size),
            "sourceCount": int(source_ids.size),
            "positivePairs": 0,
        }

    camera_world = np.asarray(dataset.poses["camera_world"][int(pose_index)], dtype=np.float32)
    anchors = _spherical_anchors()
    rows: list[np.ndarray] = []
    positive_pairs = 0
    for start in range(0, targets.size, int(target_chunk_size)):
        end = min(targets.size, start + int(target_chunk_size))
        local_ids = targets[start:end]
        target_rect = target_rect_all[start:end]
        target_area = target_area_all[start:end]
        target_depth = target_depth_all[start:end]
        overlap = _rect_overlap_ratio(target_rect, source_rect)
        gap = target_depth[:, None] - source_depth[None, :]
        front = gap > float(min_depth_gap)
        overlap_keep = overlap > float(min_overlap)
        depth_confidence = np.clip(
            gap / np.maximum(target_depth[:, None] * 0.25, 25.0), 0.0, 1.0
        )
        source_area_score = np.log1p(np.clip(source_area, 0.0, 4.0) * 2048.0) / np.log1p(4.0 * 2048.0)
        evidence = overlap * depth_confidence * (
            0.35 + 0.65 * source_area_score[None, :]
        )
        evidence = np.where(front & overlap_keep, evidence, 0.0).astype(np.float32, copy=False)
        target_local, source_local = np.nonzero(evidence > float(min_evidence))
        if target_local.size == 0:
            continue
        positive_pairs += int(target_local.size)
        target_global = local_ids[target_local].astype(np.int64, copy=False)
        rays = centers[target_global] - camera_world[None, :]
        direction_ids, direction_weights = _soft_direction_neighbors(
            rays, anchors, top_k=3, concentration=8.0
        )
        for neighbor in range(direction_ids.shape[1]):
            weight = direction_weights[:, neighbor]
            pair_gap = gap[target_local, source_local].astype(np.float32, copy=False)
            relative_gap = pair_gap / np.maximum(target_depth[target_local], 1e-4)
            shell = _shell_id(relative_gap)
            pixel_count = np.maximum(
                target_area[target_local] * float(width * height) * evidence[target_local, source_local] * weight,
                1.0,
            )
            rows.append(np.column_stack([
                target_global,
                source_ids[source_local],
                direction_ids[:, neighbor],
                shell,
                pixel_count,
                relative_gap,
            ]).astype(np.float32, copy=False))
    if not rows:
        result = np.zeros((0, 6), dtype=np.float32)
    else:
        result = np.concatenate(rows, axis=0)
        # The final table has eight source slots.  Keep the same bounded
        # capacity per pose/cell before the cross-pose support aggregation so
        # coarse AABB overlap cannot create an unbounded intermediate table.
        keys = result[:, [0, 2, 3]].astype(np.int64, copy=False)
        order = np.lexsort((
            result[:, 1].astype(np.int64, copy=False),
            -result[:, 4].astype(np.float64, copy=False),
            keys[:, 2], keys[:, 1], keys[:, 0],
        ))
        sorted_keys = keys[order]
        starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)) + 1]
        ends = np.r_[starts[1:], sorted_keys.shape[0]]
        ranks = np.arange(sorted_keys.shape[0], dtype=np.int64) - np.repeat(starts, ends - starts)
        result = result[order[ranks < SOURCE_K]]
    return result, {
        "candidateCount": int(candidate.size),
        "targetCount": int(targets.size),
        "sourceCount": int(source_ids.size),
        "positivePairs": int(positive_pairs),
    }


def _spherical_anchors() -> np.ndarray:
    values: list[list[float]] = []
    for pitch in (0.0, np.pi / 4.0, -np.pi / 4.0):
        for yaw in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0):
            cp = np.cos(pitch)
            values.append([np.sin(yaw) * cp, np.sin(pitch), np.cos(yaw) * cp])
    return np.asarray(values, dtype=np.float32)


def _copy_survival_observations(
    survival_dir: Path,
    output_dir: Path,
    expected_digest: str,
    expected_splits: list[str],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    source_meta_path = survival_dir / "evidence_meta.json"
    if not source_meta_path.is_file():
        raise FileNotFoundError(f"missing triangle survival evidence metadata: {source_meta_path}")
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    if source_meta.get("schema") != SURVIVAL_SCHEMA:
        raise ValueError("AABB supplement requires the triangle evidence survival source")
    if str(source_meta.get("candidateDigest", "")) != str(expected_digest):
        raise ValueError("triangle survival observations use a different candidate digest")
    source_splits = sorted(str(value) for value in source_meta.get("splits", []))
    if source_splits != sorted(expected_splits):
        raise ValueError(f"triangle survival observations use splits {source_splits}, expected {expected_splits}")
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
    return source_meta, source_checksums, copied_checksums


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    runtime_meta_path = Path(args.runtime_meta)
    survival_dir = Path(args.survival_evidence_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    world_aabbs, _instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    num_instances = int(world_aabbs.shape[0])
    centers = ((world_aabbs[:, :3] + world_aabbs[:, 3:]) * 0.5).astype(np.float32, copy=False)
    _scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    del scene_size
    dataset = PoseCSRDataset(dataset_dir, num_instances=num_instances)
    if dataset.mvp is None:
        raise RuntimeError("AABB relation evidence requires mvp.bin in the pose CSR dataset")
    requested_splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    missing = [name for name in requested_splits if name not in dataset.split_ids]
    if missing:
        raise ValueError(f"dataset does not define requested splits: {missing}")
    pose_sequence = pose_sequence_for_splits(dataset, requested_splits)
    if args.max_poses > 0:
        pose_sequence = pose_sequence[: int(args.max_poses)]
    if pose_sequence.size == 0:
        raise ValueError("requested splits contain no poses")
    digest = candidate_digest_for_pose_sequence(dataset, pose_sequence)
    candidate_audit = audit_native_aabb_candidates(dataset, world_aabbs, pose_sequence)
    all_rows: list[np.ndarray] = []
    stats = {
        "poseCount": int(pose_sequence.size),
        "positivePairs": 0,
        "candidateRefs": 0,
        "targetRefs": 0,
        "sourceRefs": 0,
        "posesWithRelations": 0,
    }
    for pose_number, pose_index in enumerate(pose_sequence.tolist(), start=1):
        rows, pose_stats = _aabb_pose_rows(
            dataset,
            int(pose_index),
            world_aabbs,
            centers,
            num_instances,
            min_overlap=float(args.min_overlap),
            min_evidence=float(args.min_evidence),
            min_depth_gap=float(args.min_depth_gap),
            min_source_area=float(args.min_source_area),
            min_target_area=float(args.min_target_area),
            target_chunk_size=int(args.target_chunk_size),
            max_targets=int(args.max_targets_per_pose),
            max_sources=int(args.max_sources_per_pose),
            width=int(args.width),
            height=int(args.height),
        )
        stats["positivePairs"] += int(pose_stats["positivePairs"])
        stats["candidateRefs"] += int(pose_stats["candidateCount"])
        stats["targetRefs"] += int(pose_stats["targetCount"])
        stats["sourceRefs"] += int(pose_stats["sourceCount"])
        if rows.size:
            stats["posesWithRelations"] += 1
            # Add a pose identity column expected by the shared aggregation
            # helper.  The helper consumes [target, source, direction, shell,
            # pixels, gap, std, pose].
            pose_column = np.full((rows.shape[0], 1), float(pose_index), dtype=np.float32)
            all_rows.append(np.concatenate([
                rows[:, :4],
                rows[:, 4:5],
                rows[:, 5:6],
                np.zeros((rows.shape[0], 1), dtype=np.float32),
                pose_column,
            ], axis=1))
        if pose_number == 1 or pose_number % max(1, pose_sequence.size // 20) == 0 or pose_number == pose_sequence.size:
            print(json.dumps({
                "status": "aabb_relation_progress",
                "processed": pose_number,
                "total": int(pose_sequence.size),
                "positivePairs": int(stats["positivePairs"]),
            }, ensure_ascii=False), flush=True)
    if not all_rows:
        raise ValueError("AABB projection produced no relation rows")
    expanded = np.concatenate(all_rows, axis=0)
    source_ids, relation_stats, strength, evidence_count, aggregation_stats = _aggregate_rows(
        expanded,
        num_instances=num_instances,
        source_k=SOURCE_K,
        pixel_denominator=int(args.width) * int(args.height),
        pose_rows_are_unique=True,
    )
    valid_sources = relation_stats[..., 0] > 0.0
    source_meta, source_checksums, copied_checksums = _copy_survival_observations(
        survival_dir, output_dir, digest, requested_splits
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
        "datasetDir": dataset_dir.as_posix(),
        "runtimeMeta": runtime_meta_path.as_posix(),
        "survivalEvidenceSource": survival_dir.as_posix(),
        "numInstances": num_instances,
        "directionBins": DIRECTION_BINS,
        "depthShells": DEPTH_SHELLS,
        "sourceK": SOURCE_K,
        "relationStatDim": RELATION_STAT_DIM,
        "relationStatSemantics": [
            "unique_relation_score",
            "conditional_pixel_fraction",
            "pose_support_rate",
            "weighted_mean_relative_depth_gap",
            "weighted_depth_gap_std",
            "normalized_log_pixel_count",
        ],
        "modelInputFovYDeg": 66.0,
        "splits": requested_splits,
        "formalCandidatePolicy": "stored back-camera candidate CSR; no GT-visible union and no frontend whitelist",
        "relationBuildVariant": "aabb_projection_supplement_v1",
        "evidenceSemantics": "projected AABB overlap plus positive AABB depth ordering; coarse supplement only",
        "survivalSemantics": "copied byte-for-byte from the registered triangle evidence; relation source is changed only for this supplement",
        "candidateDigest": digest,
        "candidateDigestScope": "canonical PoseCSR split rows",
        "renderCandidateDigest": digest,
        "renderCandidateDigestScope": "canonical rows used as the AABB relation pose sequence",
        "candidateIdentity": {
            "canonicalCandidateDigest": digest,
            "renderCandidateDigest": digest,
            "canonicalPoseCount": int(pose_sequence.size),
            "renderPoseCount": int(pose_sequence.size),
        },
        "candidateAudit": candidate_audit,
        "relationEvidenceSource": {
            "projection": "AABB eight-corner MVP projection from stored mvp.bin",
            "resolution": [int(args.width), int(args.height)],
            "thresholds": {
                "minOverlap": float(args.min_overlap),
                "minEvidence": float(args.min_evidence),
                "minDepthGap": float(args.min_depth_gap),
                "minSourceArea": float(args.min_source_area),
                "minTargetArea": float(args.min_target_area),
            },
            "triangleDepthUsedForRelations": False,
            "gtVisibleSourceOnly": True,
            "perPoseSourceTopK": SOURCE_K,
        },
        "survivalObservationChecksums": {
            "source": source_checksums,
            "copied": copied_checksums,
            "byteIdentical": source_checksums == copied_checksums,
            "sourceMetaSchema": source_meta.get("schema"),
        },
        "stats": {
            **stats,
            "nonzeroRelationCells": int(np.count_nonzero(valid_sources)),
            **aggregation_stats,
        },
        "files": {
            **relation_files,
            "observationInstances": "survival_observation_instances_uint32.bin",
            "observationDirections": "survival_observation_directions_uint8.bin",
            "observationRho": "survival_observation_rho_fp16.bin",
            "observationEvent": "survival_observation_event_uint8.bin",
            "observationWeight": "survival_observation_weight_fp16.bin",
        },
    }
    (output_dir / "evidence_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--survival-evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="train")
    parser.add_argument("--target-chunk-size", type=int, default=512)
    parser.add_argument("--max-poses", type=int, default=0)
    parser.add_argument("--max-targets-per-pose", type=int, default=4096)
    parser.add_argument("--max-sources-per-pose", type=int, default=512)
    parser.add_argument("--min-overlap", type=float, default=0.02)
    parser.add_argument("--min-evidence", type=float, default=0.01)
    parser.add_argument("--min-depth-gap", type=float, default=1.0)
    parser.add_argument("--min-source-area", type=float, default=1e-6)
    parser.add_argument("--min-target-area", type=float, default=1e-7)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()
    if args.target_chunk_size <= 0 or args.max_poses < 0:
        parser.error("target chunk size must be positive and max poses non-negative")
    print(json.dumps(build_evidence(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
