#!/usr/bin/env python3
"""Build data-derived directional occlusion evidence for proxy-encoder training.

The evidence is computed from one pose at a time:
- targets are candidate instances that are not visible in the GT set;
- sources are GT-visible instances in the same pose;
- a source contributes if it is in front of the target and their screen AABB
  projections overlap under the stored MVP.

This is not a teacher model. It is weak supervision derived from the sampled
visible set and view geometry.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.runtime_meta import load_runtime_meta  # noqa: E402
from pose_csr_dataset import PoseCSRDataset, _project_aabb_features_numpy  # noqa: E402


def _rect_overlap_ratio(target_rect: np.ndarray, source_rect: np.ndarray) -> np.ndarray:
    x0 = np.maximum(target_rect[:, None, 0], source_rect[None, :, 0])
    y0 = np.maximum(target_rect[:, None, 1], source_rect[None, :, 1])
    x1 = np.minimum(target_rect[:, None, 2], source_rect[None, :, 2])
    y1 = np.minimum(target_rect[:, None, 3], source_rect[None, :, 3])
    inter = np.maximum(x1 - x0, 0.0) * np.maximum(y1 - y0, 0.0)
    target_area = np.maximum(target_rect[:, 2] - target_rect[:, 0], 0.0) * np.maximum(target_rect[:, 3] - target_rect[:, 1], 0.0)
    return inter / np.maximum(target_area[:, None], 1e-8)


def _ray_direction_bins(camera_world: np.ndarray, target_centers: np.ndarray, bins: int) -> np.ndarray:
    ray = target_centers - camera_world[None, :]
    norm = np.linalg.norm(ray, axis=1, keepdims=True)
    ray = ray / np.maximum(norm, 1e-6)
    angle = np.arctan2(ray[:, 0], ray[:, 2])
    angle = np.mod(angle + 2.0 * np.pi, 2.0 * np.pi)
    return np.clip(np.floor(angle / (2.0 * np.pi) * float(bins)).astype(np.int64), 0, bins - 1)


def _shell_ids(gap: np.ndarray, target_depth: np.ndarray, shells: int) -> np.ndarray:
    # gap is target_depth - source_depth. Positive means source is in front.
    rel = gap / np.maximum(target_depth, 1.0)
    if shells <= 1:
        return np.zeros_like(rel, dtype=np.int64)
    if shells == 2:
        return np.where(rel < 0.20, 0, 1).astype(np.int64)
    # near / middle / far in front of the target.
    return np.clip(np.digitize(rel, np.asarray([0.10, 0.35], dtype=np.float32)), 0, shells - 1).astype(np.int64)


def _update_topk(
    top_scores: np.ndarray,
    top_ids: np.ndarray,
    target_ids: np.ndarray,
    bin_ids: np.ndarray,
    shell_ids: np.ndarray,
    source_ids: np.ndarray,
    scores: np.ndarray,
) -> None:
    for local_t, bin_id, shell_id, source_id, score in zip(target_ids, bin_ids, shell_ids, source_ids, scores, strict=False):
        if score <= 0.0:
            continue
        row = top_scores[int(local_t), int(bin_id), int(shell_id)]
        min_pos = int(np.argmin(row))
        if float(score) <= float(row[min_pos]):
            continue
        top_ids[int(local_t), int(bin_id), int(shell_id), min_pos] = int(source_id)
        row[min_pos] = float(score)


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    runtime_meta_path = Path(args.runtime_meta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    world_aabbs, _instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    centers = ((world_aabbs[:, :3] + world_aabbs[:, 3:]) * 0.5).astype(np.float32, copy=False)
    sizes = np.maximum(world_aabbs[:, 3:] - world_aabbs[:, :3], 1e-4).astype(np.float32, copy=False)
    volumes = np.maximum(sizes[:, 0] * sizes[:, 1] * sizes[:, 2], 1e-4).astype(np.float32, copy=False)
    volume_norm = np.log1p(volumes) / max(1e-6, float(np.log1p(volumes.max())))
    dataset = PoseCSRDataset(dataset_dir, num_instances=world_aabbs.shape[0])
    if dataset.mvp is None:
        raise RuntimeError("Directional occlusion evidence requires mvp.bin in the pose CSR dataset.")

    direction_bins = int(args.direction_bins)
    depth_shells = int(args.depth_shells)
    source_k = int(args.source_k)
    n = int(world_aabbs.shape[0])
    strength_sum = np.zeros((n, direction_bins, depth_shells), dtype=np.float32)
    weight_sum = np.zeros_like(strength_sum)
    count = np.zeros((n, direction_bins, depth_shells), dtype=np.uint32)
    top_scores = np.zeros((n, direction_bins, depth_shells, source_k), dtype=np.float32)
    # Keep the compact uint16 format for smaller scenes, but use uint32 when a
    # scene exceeds that range so proxy supervision never wraps an ID.
    source_id_dtype = np.uint32 if n > np.iinfo(np.uint16).max else np.uint16
    source_id_dtype_name = "uint32" if source_id_dtype == np.uint32 else "uint16"
    top_ids = np.zeros((n, direction_bins, depth_shells, source_k), dtype=source_id_dtype)

    split_ids = dataset.split_ids
    selected_splits = {split_ids[name] for name in args.splits.split(",") if name in split_ids}
    pose_indices = np.arange(dataset.poses.shape[0], dtype=np.int64)
    if selected_splits:
        pose_indices = pose_indices[np.isin(dataset.poses["split"], list(selected_splits))]
    if args.max_poses > 0:
        pose_indices = pose_indices[: int(args.max_poses)]

    total_pairs = 0
    positive_pairs = 0
    skipped_no_visible = 0
    skipped_no_targets = 0
    for pose_index in tqdm(pose_indices.tolist(), desc="directional occlusion evidence"):
        visible_ids, _visible_weights = dataset.visible_slice(int(pose_index))
        if visible_ids.size == 0:
            skipped_no_visible += 1
            continue
        candidate_ids = dataset.frustum_slice(int(pose_index)).astype(np.uint32, copy=False)
        visible_unique = np.unique(visible_ids.astype(np.uint32, copy=False))
        missing_visible = np.setdiff1d(visible_unique, candidate_ids, assume_unique=False)
        if missing_visible.size and not args.allow_candidate_visible_union:
            raise ValueError(
                "Formal evidence candidate semantics failed at pose "
                f"{pose_index}: {missing_visible.size} visible instances are absent from the stored candidate set."
            )
        if args.allow_candidate_visible_union:
            candidate_ids = np.union1d(candidate_ids, visible_unique)
        target_ids = np.setdiff1d(candidate_ids, visible_unique, assume_unique=False)
        if target_ids.size == 0:
            skipped_no_targets += 1
            continue
        if args.max_targets_per_pose > 0 and target_ids.size > args.max_targets_per_pose:
            # Prefer larger projected targets because they have clearer overlap evidence.
            rect_t_all, area_t_all, _depth_t_all, valid_t_all = _project_aabb_features_numpy(world_aabbs[target_ids], dataset.mvp_slice(int(pose_index)))
            score = area_t_all + valid_t_all.astype(np.float32) * 1e-3
            keep = np.argpartition(score, -int(args.max_targets_per_pose))[-int(args.max_targets_per_pose):]
            target_ids = target_ids[keep]
        source_ids = visible_ids.astype(np.uint32, copy=False)

        mvp = dataset.mvp_slice(int(pose_index))
        source_rect, source_area, source_depth, source_valid = _project_aabb_features_numpy(world_aabbs[source_ids], mvp)
        source_keep = source_valid & (source_area > float(args.min_source_area))
        if not bool(source_keep.any()):
            continue
        source_ids = source_ids[source_keep]
        source_rect = source_rect[source_keep]
        source_area = source_area[source_keep]
        source_depth = source_depth[source_keep]
        source_size_weight = volume_norm[source_ids.astype(np.int64)]
        if args.max_sources_per_pose > 0 and source_ids.size > args.max_sources_per_pose:
            source_score = np.log1p(source_area * 2048.0) + source_size_weight
            keep = np.argpartition(source_score, -int(args.max_sources_per_pose))[-int(args.max_sources_per_pose):]
            source_ids = source_ids[keep]
            source_rect = source_rect[keep]
            source_area = source_area[keep]
            source_depth = source_depth[keep]
            source_size_weight = source_size_weight[keep]

        for start in range(0, target_ids.size, int(args.target_chunk_size)):
            end = min(target_ids.size, start + int(args.target_chunk_size))
            local_targets = target_ids[start:end]
            target_rect, target_area, target_depth, target_valid = _project_aabb_features_numpy(world_aabbs[local_targets], mvp)
            valid_targets = target_valid & (target_area > float(args.min_target_area))
            if not bool(valid_targets.any()):
                continue
            local_targets = local_targets[valid_targets]
            target_rect = target_rect[valid_targets]
            target_depth = target_depth[valid_targets]
            overlap = _rect_overlap_ratio(target_rect, source_rect)
            gap = target_depth[:, None] - source_depth[None, :]
            front = gap > float(args.min_depth_gap)
            total_pairs += int(overlap.size)
            if not bool((front & (overlap > float(args.min_overlap))).any()):
                continue
            depth_conf = np.clip(gap / np.maximum(target_depth[:, None] * 0.25, 25.0), 0.0, 1.0)
            source_area_score = np.log1p(np.clip(source_area, 0.0, 4.0) * 2048.0) / math.log1p(4.0 * 2048.0)
            evidence = overlap * depth_conf * (0.35 + 0.45 * source_area_score[None, :] + 0.20 * source_size_weight[None, :])
            evidence = np.where(front & (overlap > float(args.min_overlap)), evidence, 0.0).astype(np.float32, copy=False)
            ti, sj = np.nonzero(evidence > float(args.min_evidence))
            if ti.size == 0:
                continue
            positive_pairs += int(ti.size)
            scores = evidence[ti, sj]
            global_targets = local_targets[ti].astype(np.int64, copy=False)
            bins = _ray_direction_bins(np.asarray(dataset.poses["camera_world"][pose_index], dtype=np.float32), centers[global_targets], direction_bins)
            shells = _shell_ids(gap[ti, sj], target_depth[ti], depth_shells)
            np.add.at(strength_sum, (global_targets, bins, shells), scores)
            np.add.at(weight_sum, (global_targets, bins, shells), np.ones_like(scores, dtype=np.float32))
            np.add.at(count, (global_targets, bins, shells), np.ones_like(scores, dtype=np.uint32))
            _update_topk(top_scores, top_ids, global_targets, bins, shells, source_ids[sj], scores)

    mean_strength = strength_sum / np.maximum(weight_sum, 1.0)
    strength_path = output_dir / "evidence_strength_fp16.bin"
    count_path = output_dir / "evidence_count_uint32.bin"
    source_ids_path = output_dir / f"evidence_source_ids_{source_id_dtype_name}.bin"
    source_scores_path = output_dir / "evidence_source_scores_fp16.bin"
    mean_strength.astype(np.float16).tofile(strength_path)
    count.tofile(count_path)
    top_ids.astype(source_id_dtype, copy=False).tofile(source_ids_path)
    top_scores.astype(np.float16).tofile(source_scores_path)
    meta: dict[str, Any] = {
        "schema": "directional-occlusion-evidence-v1",
        "runtimeMeta": runtime_meta_path.as_posix(),
        "datasetDir": dataset_dir.as_posix(),
        "numInstances": n,
        "directionBins": direction_bins,
        "depthShells": depth_shells,
        "sourceK": source_k,
        "sourceIdDtype": source_id_dtype_name,
        "splits": args.splits,
        "semantics": (
            "Evidence is derived from GT-visible sources and GT-invisible candidate targets "
            "under the same pose projection; no dynamic-pool teacher is used. "
            f"Candidate-visible union allowed: {bool(args.allow_candidate_visible_union)}."
        ),
        "thresholds": {
            "minOverlap": float(args.min_overlap),
            "minEvidence": float(args.min_evidence),
            "minDepthGap": float(args.min_depth_gap),
            "minSourceArea": float(args.min_source_area),
            "minTargetArea": float(args.min_target_area),
        },
        "stats": {
            "poseCount": int(pose_indices.size),
            "positivePairs": int(positive_pairs),
            "totalPairsScanned": int(total_pairs),
            "nonzeroEvidenceCells": int((count > 0).sum()),
            "skippedNoVisible": int(skipped_no_visible),
            "skippedNoTargets": int(skipped_no_targets),
        },
        "files": {
            "evidenceStrength": strength_path.name,
            "evidenceCount": count_path.name,
            "sourceIds": source_ids_path.name,
            "sourceScores": source_scores_path.name,
        },
    }
    (output_dir / "evidence_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build directional occlusion evidence from pose CSR visibility data.")
    parser.add_argument("--dataset-dir", default="neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66")
    parser.add_argument("--runtime-meta", default="hkust-v3/assets/runtimeVisibilityMeta.json")
    parser.add_argument("--output-dir", default="neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_fov66")
    parser.add_argument("--splits", default="train")
    parser.add_argument("--direction-bins", type=int, default=8)
    parser.add_argument("--depth-shells", type=int, default=3)
    parser.add_argument("--source-k", type=int, default=8)
    parser.add_argument("--target-chunk-size", type=int, default=512)
    parser.add_argument("--max-poses", type=int, default=0)
    parser.add_argument("--max-targets-per-pose", type=int, default=4096)
    parser.add_argument("--max-sources-per-pose", type=int, default=512)
    parser.add_argument("--min-overlap", type=float, default=0.02)
    parser.add_argument("--min-evidence", type=float, default=0.01)
    parser.add_argument("--min-depth-gap", type=float, default=1.0)
    parser.add_argument("--min-source-area", type=float, default=1e-6)
    parser.add_argument("--min-target-area", type=float, default=1e-7)
    parser.add_argument(
        "--allow-candidate-visible-union",
        action="store_true",
        help="Exploratory-only legacy repair. Formal evidence generation must use stored candidates unchanged.",
    )
    args = parser.parse_args()
    meta = build_evidence(args)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
