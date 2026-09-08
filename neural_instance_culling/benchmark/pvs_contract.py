"""Structured contracts for the v4 validation replay.

The v4 pipeline deliberately validates semantic fields directly.  It does
not calculate or compare file fingerprints.  This module is shared by the
runner and the summarizer so replay reuse and formal aggregation use the same
pose, candidate, GT, and count checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


COUNT_FIELDS = ("tp", "fp", "fn", "tn")
WEIGHT_FIELDS = ("weightedTp", "weightedGt")


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def pose_indices_for_split(dataset: Any, split_name: str) -> np.ndarray:
    if split_name not in dataset.split_ids:
        raise ValueError(f"dataset has no explicit split {split_name!r}")
    return np.asarray(dataset.split(split_name).pose_indices, dtype="<i8")


def _as_ids(value: Any, *, label: str, allow_none: bool = False) -> np.ndarray:
    if value is None and allow_none:
        return np.zeros((0,), dtype=np.int64)
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list; old non-semantic replay cannot be reused")
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size and (int(values.min()) < 0):
        raise ValueError(f"{label} contains a negative instance ID")
    if values.size != np.unique(values).size:
        raise ValueError(f"{label} contains duplicate instance IDs")
    return values


def validate_native_candidate_contract(
    dataset: Any,
    pose_indices: Iterable[int],
    *,
    num_instances: int | None = None,
) -> dict[str, Any]:
    """Validate native candidates and GT inclusion for an explicit pose list."""
    poses = np.asarray(list(pose_indices), dtype="<i8").reshape(-1)
    candidate_refs = 0
    visible_refs = 0
    candidate_sizes: list[int] = []
    visible_sizes: list[int] = []
    for pose in poses.tolist():
        candidates = np.asarray(dataset.candidate_slice(int(pose)), dtype=np.int64).reshape(-1)
        visible = np.unique(np.asarray(dataset.visible_slice(int(pose))[0], dtype=np.int64).reshape(-1))
        if candidates.size != np.unique(candidates).size:
            raise ValueError(f"candidate IDs contain duplicates at pose {pose}")
        if num_instances is not None and candidates.size and int(candidates.max()) >= int(num_instances):
            raise ValueError(f"candidate ID is outside num_instances at pose {pose}")
        if num_instances is not None and visible.size and int(visible.max()) >= int(num_instances):
            raise ValueError(f"GT ID is outside num_instances at pose {pose}")
        missing = np.setdiff1d(visible, candidates, assume_unique=False)
        if missing.size:
            raise ValueError(
                f"GT visible IDs are absent from native candidates at pose {pose}: {missing[:8].tolist()}"
            )
        candidate_refs += int(candidates.size)
        visible_refs += int(visible.size)
        candidate_sizes.append(int(candidates.size))
        visible_sizes.append(int(visible.size))
    return {
        "poseCount": int(poses.size),
        "poseIndices": [int(value) for value in poses.tolist()],
        "candidateReferenceCount": int(candidate_refs),
        "visibleReferenceCount": int(visible_refs),
        "candidateCountMin": int(min(candidate_sizes)) if candidate_sizes else 0,
        "candidateCountMax": int(max(candidate_sizes)) if candidate_sizes else 0,
        "candidateCountMean": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
        "visibleCountMin": int(min(visible_sizes)) if visible_sizes else 0,
        "visibleCountMax": int(max(visible_sizes)) if visible_sizes else 0,
        "visibleCountMean": float(np.mean(visible_sizes)) if visible_sizes else 0.0,
        "candidateSemantics": "stored native back-camera candidates; GT union disabled",
        "testRead": False,
    }


def _compare_count(actual: Mapping[str, Any], expected: Mapping[str, float], label: str) -> None:
    for field in COUNT_FIELDS:
        value = _finite(actual.get(field), f"{label}.metrics.{field}")
        if not np.isclose(value, float(expected[field]), rtol=0.0, atol=1e-5):
            raise ValueError(
                f"{label}.metrics.{field} disagrees with instance IDs: {value} != {expected[field]}"
            )


def validate_replay_payload(
    payload: Mapping[str, Any],
    dataset: Any,
    expected_pose_indices: Iterable[int],
    *,
    expected_variant: str | None = None,
    expected_seed: int | None = None,
    expected_checkpoint: str | Path | None = None,
    expected_threshold: float | None = None,
    num_instances: int | None = None,
) -> dict[str, Any]:
    """Validate one replay using explicit candidate/prediction ID arrays."""
    poses = [int(value) for value in expected_pose_indices]
    if payload.get("split") != "validation":
        raise ValueError("replay is not validation-only")
    if payload.get("testRead") is not False:
        raise ValueError("replay claims that test was read")
    if expected_variant is not None and str(payload.get("variant", "")) != str(expected_variant):
        raise ValueError("replay variant disagrees with its member")
    if expected_seed is not None and int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError("replay seed disagrees with its member")
    if expected_checkpoint is not None:
        actual_checkpoint = Path(str(payload.get("checkpoint", ""))).resolve()
        if actual_checkpoint != Path(expected_checkpoint).resolve():
            raise ValueError("replay checkpoint disagrees with its member")
    if expected_threshold is not None:
        threshold = _finite(payload.get("threshold"), "replay.threshold")
        if not np.isclose(threshold, float(expected_threshold), rtol=0.0, atol=1e-7):
            raise ValueError("replay threshold disagrees with its checkpoint")
    if [int(value) for value in payload.get("poseIndices", [])] != poses:
        raise ValueError("replay pose order disagrees with the registered validation split")
    rows = payload.get("perPose")
    if not isinstance(rows, list) or len(rows) != len(poses):
        raise ValueError("replay perPose rows do not cover the validation split")

    for row, pose in zip(rows, poses, strict=True):
        if not isinstance(row, Mapping) or int(row.get("poseIndex", -1)) != pose:
            raise ValueError(f"replay perPose order disagrees at pose {pose}")
        candidates = _as_ids(row.get("candidateIds"), label=f"pose {pose} candidateIds")
        predicted = _as_ids(row.get("predictedIds"), label=f"pose {pose} predictedIds")
        expected_candidates = np.asarray(dataset.candidate_slice(pose), dtype=np.int64).reshape(-1)
        if not np.array_equal(candidates, expected_candidates):
            raise ValueError(f"replay candidate IDs disagree with the native candidate set at pose {pose}")
        if num_instances is not None and predicted.size and int(predicted.max()) >= int(num_instances):
            raise ValueError(f"replay predicted ID is outside num_instances at pose {pose}")
        if not np.all(np.isin(predicted, candidates, assume_unique=False)):
            raise ValueError(f"replay predicted IDs are not a subset of candidates at pose {pose}")
        visible = np.unique(np.asarray(dataset.visible_slice(pose)[0], dtype=np.int64).reshape(-1))
        if not np.all(np.isin(visible, candidates, assume_unique=False)):
            raise ValueError(f"GT visible IDs are not a subset of candidates at pose {pose}")
        candidate_mask = np.isin(candidates, visible, assume_unique=False)
        predicted_mask = np.isin(candidates, predicted, assume_unique=False)
        expected_counts = {
            "tp": float(np.logical_and(candidate_mask, predicted_mask).sum()),
            "fp": float(np.logical_and(~candidate_mask, predicted_mask).sum()),
            "fn": float(np.logical_and(candidate_mask, ~predicted_mask).sum()),
            "tn": float(np.logical_and(~candidate_mask, ~predicted_mask).sum()),
        }
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"replay pose {pose} has no metrics")
        _compare_count(metrics, expected_counts, f"pose {pose}")
        if int(round(_finite(metrics.get("candidateCount"), f"pose {pose}.candidateCount"))) != int(candidates.size):
            raise ValueError(f"replay candidate count disagrees at pose {pose}")
        if int(round(_finite(metrics.get("gtCount"), f"pose {pose}.gtCount"))) != int(visible.size):
            raise ValueError(f"replay GT count disagrees at pose {pose}")
        if int(round(_finite(metrics.get("predCount"), f"pose {pose}.predCount"))) != int(predicted.size):
            raise ValueError(f"replay prediction count disagrees at pose {pose}")
        visible_weights = np.asarray(dataset.visible_slice(pose)[1], dtype=np.float64).reshape(-1)
        weighted_gt = float(visible_weights.sum())
        weighted_tp = float(visible_weights[np.isin(visible, predicted, assume_unique=False)].sum())
        if not np.isclose(_finite(metrics.get("weightedGt"), f"pose {pose}.weightedGt"), weighted_gt, rtol=0.0, atol=1e-3):
            raise ValueError(f"replay weighted GT disagrees at pose {pose}")
        if not np.isclose(_finite(metrics.get("weightedTp"), f"pose {pose}.weightedTp"), weighted_tp, rtol=0.0, atol=1e-3):
            raise ValueError(f"replay weighted TP disagrees at pose {pose}")
        for name, value in metrics.items():
            _finite(value, f"pose {pose}.metrics.{name}")
    return {
        "poseCount": len(poses),
        "poseIndices": poses,
        "candidateReferenceCount": int(sum(len(row["candidateIds"]) for row in rows)),
        "predictionReferenceCount": int(sum(len(row["predictedIds"]) for row in rows)),
        "testRead": False,
    }


__all__ = [
    "COUNT_FIELDS",
    "WEIGHT_FIELDS",
    "pose_indices_for_split",
    "validate_native_candidate_contract",
    "validate_replay_payload",
]
