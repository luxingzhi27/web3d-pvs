"""Shared, test-free exact threshold calibration primitives.

Calibration is evaluated on the actual float32 scores produced for the
calibration split.  The bootstrap implementation deliberately generates pose
indices in bounded chunks, so its memory use does not depend on the number of
replicates times the number of score change-points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EXACT_THRESHOLD_SOURCE = "calibration_actual_float32_score_change_points"


@dataclass(frozen=True)
class FixedPoseBootstrap:
    """Deterministic pose bootstrap plan evaluated without a large index matrix."""

    valid_pose_indices: np.ndarray
    replicates: int
    seed: int
    confidence: float = 0.95
    chunk_size: int = 256
    indices: np.ndarray | None = None

    @classmethod
    def from_gt_mass(
        cls,
        gt_mass: np.ndarray,
        *,
        replicates: int,
        seed: int,
        confidence: float = 0.95,
        chunk_size: int = 256,
    ) -> "FixedPoseBootstrap":
        masses = np.asarray(gt_mass, dtype=np.float64).reshape(-1)
        if not bool(np.isfinite(masses).all()) or bool(np.any(masses < 0.0)):
            raise ValueError("pose ground-truth mass must be finite and non-negative")
        if int(replicates) <= 0:
            raise ValueError("bootstrap replicate count must be positive")
        if not 0.0 < float(confidence) < 1.0:
            raise ValueError("bootstrap confidence must lie strictly between 0 and 1")
        if int(chunk_size) <= 0:
            raise ValueError("bootstrap chunk size must be positive")
        valid = np.flatnonzero(masses > 1e-12).astype(np.int64, copy=False)
        if valid.size == 0:
            raise ValueError("calibration data contains no positive visible weight mass")
        return cls(
            valid_pose_indices=valid.copy(),
            replicates=int(replicates),
            seed=int(seed),
            confidence=float(confidence),
            chunk_size=int(chunk_size),
        )

    def lower_confidence_bound(
        self,
        weighted_tp: np.ndarray,
        gt_mass: np.ndarray,
    ) -> float:
        """Return the one-sided percentile bound using this fixed pose plan."""
        tp = np.asarray(weighted_tp, dtype=np.float64).reshape(-1)
        gt = np.asarray(gt_mass, dtype=np.float64).reshape(-1)
        if tp.size != gt.size or tp.size == 0:
            raise ValueError("weighted TP and GT mass must be non-empty and pose-aligned")
        if not bool(np.isfinite(tp).all()) or not bool(np.isfinite(gt).all()):
            raise FloatingPointError("bootstrap inputs contain non-finite values")
        if bool(np.any(tp < 0.0)) or bool(np.any(gt < 0.0)):
            raise ValueError("bootstrap inputs must be non-negative")
        valid = self.valid_pose_indices
        if valid.size == 0 or int(valid.max()) >= tp.size:
            raise ValueError("fixed bootstrap poses do not match the calibration data")

        tp_valid = tp[valid]
        gt_valid = gt[valid]
        values = np.empty((self.replicates,), dtype=np.float64)
        pose_count = int(valid.size)
        if self.indices is not None:
            fixed_indices = np.asarray(self.indices)
            if fixed_indices.shape != (self.replicates, pose_count):
                raise ValueError("fixed bootstrap index shape disagrees with the pose plan")
            if fixed_indices.size and (
                int(fixed_indices.min()) < 0 or int(fixed_indices.max()) >= pose_count
            ):
                raise ValueError("fixed bootstrap indices are outside the valid pose range")
            index_source = fixed_indices
        else:
            index_source = None
            rng = np.random.default_rng(self.seed)
        for start in range(0, self.replicates, self.chunk_size):
            end = min(self.replicates, start + self.chunk_size)
            indices = (
                index_source[start:end]
                if index_source is not None
                else rng.integers(
                    0,
                    pose_count,
                    size=(end - start, pose_count),
                    dtype=np.int32,
                )
            )
            numerator = tp_valid[indices].sum(axis=1)
            denominator = gt_valid[indices].sum(axis=1)
            values[start:end] = np.divide(
                numerator,
                denominator,
                out=np.ones_like(numerator),
                where=denominator > 1e-12,
            )
        return float(np.quantile(values, 1.0 - self.confidence))

    def metadata(self) -> dict[str, Any]:
        return {
            "unit": "pose",
            "replicates": int(self.replicates),
            "seed": int(self.seed),
            "confidence": float(self.confidence),
            "validPoseCount": int(self.valid_pose_indices.size),
            "chunkSize": int(self.chunk_size),
            "indexStorage": (
                "provided_read_only_indices"
                if self.indices is not None
                else "deterministic_chunked_generation"
            ),
        }


def float32_score_change_points(scores: np.ndarray) -> np.ndarray:
    """Return all unique finite score values in the model's float32 domain."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not bool(np.isfinite(values).all()):
        raise FloatingPointError("calibration scores contain non-finite values")
    return np.unique(values)


def _pose_rows_from_offsets(pose_offsets: np.ndarray, score_count: int) -> np.ndarray:
    offsets = np.asarray(pose_offsets, dtype=np.int64).reshape(-1)
    if offsets.size == 0 or int(offsets[0]) != 0:
        raise ValueError("pose offsets must start at zero")
    if bool(np.any(np.diff(offsets) < 0)) or int(offsets[-1]) != int(score_count):
        raise ValueError("pose offsets do not cover the score stream")
    return np.repeat(
        np.arange(offsets.size - 1, dtype=np.int32),
        np.diff(offsets),
    )


def select_highest_safe_score_change_point(
    scores: np.ndarray,
    labels: np.ndarray,
    visible_weights: np.ndarray,
    pose_offsets: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    target_weighted_recall: float = 0.99,
    bootstrap: FixedPoseBootstrap | None = None,
) -> dict[str, Any]:
    """Select the highest actual float32 score point passing both safety gates.

    Safety is monotonic in the threshold.  The search therefore evaluates the
    complete float32 score-point domain with a binary search.  A negative-only
    point can be the formal boundary of an interval with unchanged weighted
    recall, so it must remain part of the searchable domain rather than merely
    being reported as a diagnostic.
    """
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    targets = np.asarray(labels, dtype=np.float32).reshape(-1)
    weights = np.asarray(visible_weights, dtype=np.float64).reshape(-1)
    if not (values.size == targets.size == weights.size):
        raise ValueError("scores, labels, and visible weights must be candidate-aligned")
    if bool(np.any((targets < 0.0) | (targets > 1.0))):
        raise ValueError("labels must lie in [0, 1]")
    if bool(np.any(weights < 0.0)) or not bool(np.isfinite(weights).all()):
        raise ValueError("visible weights must be finite and non-negative")
    row_ids = _pose_rows_from_offsets(pose_offsets, values.size)
    pose_count = int(np.asarray(pose_offsets).size - 1)
    positive = targets > 0.5
    gt_mass = np.bincount(
        row_ids,
        weights=np.where(positive, weights, 0.0),
        minlength=pose_count,
    ).astype(np.float64, copy=False)
    if bootstrap is None:
        bootstrap = FixedPoseBootstrap.from_gt_mass(
            gt_mass,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    elif (
        bootstrap.replicates != int(bootstrap_replicates)
        or bootstrap.seed != int(bootstrap_seed)
        or not np.array_equal(
            bootstrap.valid_pose_indices,
            np.flatnonzero(gt_mass > 1e-12).astype(np.int64, copy=False),
        )
    ):
        raise ValueError("provided fixed bootstrap does not match calibration poses")
    score_points = float32_score_change_points(values)
    positive_scores = values[positive]
    positive_pose_rows = row_ids[positive]
    positive_weights = weights[positive]
    positive_points = float32_score_change_points(positive_scores)

    def safety(threshold: np.float32) -> tuple[float, float, bool]:
        selected = positive_scores >= np.float32(threshold)
        weighted_tp = np.bincount(
            positive_pose_rows[selected],
            weights=positive_weights[selected],
            minlength=pose_count,
        ).astype(np.float64, copy=False)
        aggregate = float(weighted_tp.sum() / max(1e-12, float(gt_mass.sum())))
        lower = bootstrap.lower_confidence_bound(weighted_tp, gt_mass)
        return aggregate, lower, bool(
            aggregate > float(target_weighted_recall)
            and lower > float(target_weighted_recall)
        )

    best = -1
    low = 0
    high = int(score_points.size) - 1
    while low <= high:
        middle = (low + high) // 2
        _aggregate, _lower, safe = safety(np.float32(score_points[middle]))
        if safe:
            best = middle
            low = middle + 1
        else:
            high = middle - 1

    if best >= 0:
        threshold = np.float32(score_points[best])
        status = "safe"
        selected_aggregate, selected_lower, selected_safe = safety(threshold)
        selected_index = best
    elif score_points.size:
        threshold = np.float32(score_points[0])
        status = "no_qualified_safety_workpoint"
        selected_aggregate, selected_lower, selected_safe = safety(threshold)
        selected_index = 0
    else:
        threshold = np.float32(0.0)
        status = "no_qualified_safety_workpoint"
        selected_aggregate = 1.0
        selected_lower = 1.0
        selected_safe = False
        selected_index = None

    next_higher = None
    next_higher_safety: dict[str, Any] | None = None
    if selected_index is not None and selected_index + 1 < score_points.size:
        next_higher = np.float32(score_points[selected_index + 1])
        aggregate, lower, safe = safety(next_higher)
        next_higher_safety = {
            "aggregateWeightedRecall": float(aggregate),
            "aggregateWeightedRecallLowerConfidenceBound": float(lower),
            "safe": bool(safe),
        }
    return {
        "status": status,
        "threshold": float(threshold),
        "thresholdDtype": "float32",
        "thresholdIsScoreChangePoint": bool(score_points.size and np.any(score_points == threshold)),
        "selectedCandidateIndex": selected_index,
        "candidateCount": int(score_points.size),
        "allScoreChangePointCount": int(score_points.size),
        "positiveScoreChangePointCount": int(positive_points.size),
        "minimumScore": float(score_points[0]) if score_points.size else None,
        "maximumScore": float(score_points[-1]) if score_points.size else None,
        "aggregateWeightedRecall": float(selected_aggregate),
        "aggregateWeightedRecallLowerConfidenceBound": float(selected_lower),
        "safe": bool(selected_safe),
        "nextHigherThreshold": None if next_higher is None else float(next_higher),
        "nextHigherSafety": next_higher_safety,
        "targetWeightedRecall": float(target_weighted_recall),
        "searchDomain": "all_candidate_float32_score_change_points",
        "searchOptimization": "binary_search_over_all_points; safety_from_positive_weight_mass",
        "bootstrap": bootstrap.metadata(),
        "rule": (
            "highest float32 score change-point with aggregateWeightedRecall > "
            f"{float(target_weighted_recall):.2f} and fixed-pose bootstrap LCB > "
            f"{float(target_weighted_recall):.2f}"
        ),
    }


__all__ = [
    "EXACT_THRESHOLD_SOURCE",
    "FixedPoseBootstrap",
    "float32_score_change_points",
    "select_highest_safe_score_change_point",
]
