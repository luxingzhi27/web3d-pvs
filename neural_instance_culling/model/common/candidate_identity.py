"""Canonical identity helpers for stored frustum candidate CSR data.

The formal experiments use two row sequences: the canonical PoseCSR split,
where every source pose appears once, and a render sequence, where a source
pose may have several representative subposes.  They contain the same stored
candidate IDs but have different row order and offsets, so both identities are
recorded explicitly.
"""
from __future__ import annotations

from typing import Any, Iterable
import hashlib

import numpy as np


def candidate_digest(
    candidate_ids: np.ndarray,
    pose_indices: np.ndarray,
    offsets: np.ndarray,
) -> str:
    """Hash pose order, CSR offsets and candidate IDs without normalization."""
    poses = np.asarray(pose_indices, dtype="<i8").reshape(-1)
    csr_offsets = np.asarray(offsets, dtype="<i8").reshape(-1)
    ids = np.asarray(candidate_ids, dtype="<u4").reshape(-1)
    if csr_offsets.size != poses.size + 1:
        raise ValueError("candidate offsets must have one more entry than pose indices")
    if int(csr_offsets[0]) != 0 or np.any(np.diff(csr_offsets) < 0):
        raise ValueError("candidate offsets must be non-decreasing and start at zero")
    if int(csr_offsets[-1]) != ids.size:
        raise ValueError("candidate offsets do not cover the candidate ID array")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(poses).tobytes())
    digest.update(np.ascontiguousarray(csr_offsets).tobytes())
    digest.update(np.ascontiguousarray(ids).tobytes())
    return digest.hexdigest()


def candidate_digest_for_pose_sequence(dataset: object, pose_indices: Iterable[int]) -> str:
    """Hash the native stored candidate CSR in the supplied row order."""
    poses = np.asarray(list(pose_indices), dtype=np.int64).reshape(-1)
    parts = [
        np.asarray(dataset.frustum_slice(int(pose)), dtype=np.uint32).reshape(-1)
        for pose in poses.tolist()
    ]
    offsets = np.zeros((poses.size + 1,), dtype="<i8")
    if parts:
        offsets[1:] = np.cumsum([part.size for part in parts], dtype=np.int64)
        flat = np.concatenate(parts).astype("<u4", copy=False)
    else:
        flat = np.zeros((0,), dtype="<u4")
    return candidate_digest(flat, poses, offsets)


def pose_sequence_for_splits(dataset: object, split_names: Iterable[str]) -> np.ndarray:
    """Return split rows in explicit protocol order, without deduplication."""
    parts = [np.asarray(dataset.split(str(name)).pose_indices, dtype=np.int64) for name in split_names]
    return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.int64)


def audit_native_aabb_candidates(
    dataset: Any,
    world_aabbs: np.ndarray,
    pose_indices: Iterable[int],
    near: float = 0.05,
) -> dict[str, Any]:
    """Recompute and verify the stored native AABB candidate CSR.

    Some historical dataset metadata describes a candidate file as a union
    with dense-subpose positives even when the stored ``frustum_ids`` are the
    original AABB result.  Formal experiments must establish the candidate
    semantics from the pose and AABB data, not from that description.  This
    audit therefore recomputes the exact ordered result for every requested
    pose and refuses any mismatch.  It does not read visible IDs and cannot
    add a GT instance to the candidate set.
    """
    from pose_csr_dataset import frustum_candidate_ids_for_pose

    aabbs = np.asarray(world_aabbs, dtype=np.float32)
    if aabbs.ndim != 2 or aabbs.shape[1] != 6:
        raise ValueError("world_aabbs must have shape [num_instances, 6]")
    poses = np.asarray(list(pose_indices), dtype=np.int64).reshape(-1)
    if poses.size and (int(poses.min()) < 0 or int(poses.max()) >= len(dataset.poses)):
        raise ValueError("candidate audit pose index is outside the dataset")
    mismatch_rows: list[dict[str, int]] = []
    stored_refs = 0
    recomputed_refs = 0
    for pose_index in poses.tolist():
        record = dataset.poses[int(pose_index)]
        expected = frustum_candidate_ids_for_pose(
            np.asarray(record["camera_world"], dtype=np.float32),
            np.asarray(record["camera_forward"], dtype=np.float32),
            float(record["camera_view"][0]),
            float(record["camera_view"][1]),
            aabbs,
            near=float(near),
        )
        stored = np.asarray(dataset.frustum_slice(int(pose_index)), dtype=np.uint32)
        stored_refs += int(stored.size)
        recomputed_refs += int(expected.size)
        if not np.array_equal(stored, expected):
            missing = np.setdiff1d(expected, stored, assume_unique=False)
            extra = np.setdiff1d(stored, expected, assume_unique=False)
            mismatch_rows.append({
                "poseIndex": int(pose_index),
                "storedCount": int(stored.size),
                "recomputedCount": int(expected.size),
                "missingCount": int(missing.size),
                "extraCount": int(extra.size),
            })
            if len(mismatch_rows) >= 16:
                break
    if mismatch_rows:
        raise ValueError(
            "formal candidate audit failed: stored frustum_ids do not equal "
            f"native AABB candidates at {len(mismatch_rows)} pose(s); "
            f"first={mismatch_rows[0]}"
        )
    return {
        "status": "passed",
        "source": "recomputed_native_back_camera_aabb",
        "poseCount": int(poses.size),
        "storedCandidateRefs": int(stored_refs),
        "recomputedCandidateRefs": int(recomputed_refs),
        "near": float(near),
        "gtUnionUsed": False,
        "metadataCandidateSemantics": str(getattr(dataset, "meta", {}).get("candidateSemantics", "")),
    }
