"""Calibration over re-openable columnar score streams."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import TARGET_WEIGHTED_RECALL, _weighted_lcb, evaluate_scene
from .score_bundle import PoseScores


@dataclass(frozen=True)
class CalibrationSelection:
    """A threshold frozen from calibration only."""

    protocol: str
    threshold: float
    status: str
    target_weighted_recall: float
    mean_weighted_recall: float
    mean_weighted_recall_lcb: float | None
    mean_target_met: bool
    confidence_target_met: bool
    selection_split: str
    test_read: bool
    scene_metrics: Mapping[str, Mapping[str, Any]]
    threshold_rows: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if self.selection_split != "calibration":
            raise ValueError("V5 calibration selections must come from calibration")
        if self.test_read:
            raise ValueError("V5 calibration selection cannot be test-tainted")
        if self.status not in {"strict_lcb_target", "mean_target", "diagnostic"}:
            raise ValueError(f"unknown V5 calibration status: {self.status!r}")
        if not np.isfinite(float(self.threshold)):
            raise ValueError("calibration threshold must be finite")

    @property
    def qualification(self) -> str:
        return self.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "threshold": float(self.threshold),
            "status": self.status,
            "qualification": self.qualification,
            "target_weighted_recall": float(self.target_weighted_recall),
            "mean_weighted_recall": float(self.mean_weighted_recall),
            "mean_weighted_recall_lcb": None if self.mean_weighted_recall_lcb is None else float(self.mean_weighted_recall_lcb),
            "mean_target_met": bool(self.mean_target_met),
            "confidence_target_met": bool(self.confidence_target_met),
            "selection_split": "calibration",
            "test_read": False,
            "scene_metrics": {str(scene): dict(metrics) for scene, metrics in self.scene_metrics.items()},
            "threshold_rows": [dict(row) for row in self.threshold_rows],
        }


@dataclass(frozen=True)
class _SceneRecallIndex:
    """Positive-only exact recall index.

    Weighted recall changes only when a positive score crosses the threshold;
    retaining positive scores avoids concatenating the much larger negative
    candidate population during calibration.
    """

    sorted_scores: tuple[np.ndarray, ...]
    prefix_weights: tuple[np.ndarray, ...]
    weighted_gt: np.ndarray

    @classmethod
    def from_records(cls, records: Iterable[PoseScores], *, scene: str) -> "_SceneRecallIndex":
        scores: list[np.ndarray] = []
        prefixes: list[np.ndarray] = []
        totals: list[float] = []
        seen = 0
        for record in records:
            if not isinstance(record, PoseScores):
                raise TypeError("V5 calibration requires PoseScores from a columnar sidecar")
            if record.scene != scene:
                raise ValueError(f"calibration record scene disagrees with {scene!r}")
            if record.split != "calibration":
                raise ValueError("V5 threshold selection accepts calibration rows only")
            positive = record.targets.astype(bool, copy=False)
            pose_scores = np.asarray(record.scores[positive], dtype=np.float64).copy()
            pose_weights = np.asarray(record.visible_weights[positive], dtype=np.float64).copy()
            if pose_scores.size:
                order = np.argsort(pose_scores, kind="stable")
                pose_scores = pose_scores[order]
                pose_weights = pose_weights[order]
                prefix = np.concatenate([
                    np.zeros((1,), dtype=np.float64),
                    np.cumsum(pose_weights, dtype=np.float64),
                ])
            else:
                prefix = np.zeros((1,), dtype=np.float64)
            scores.append(pose_scores)
            prefixes.append(prefix)
            totals.append(float(pose_weights.sum()))
            seen += 1
        if seen == 0:
            raise ValueError(f"calibration split is empty for scene {scene!r}")
        return cls(tuple(scores), tuple(prefixes), np.asarray(totals, dtype=np.float64))

    def thresholds(self) -> np.ndarray:
        values = [values for values in self.sorted_scores if values.size]
        if not values:
            raise ValueError("calibration contains no positive scores")
        result = np.unique(np.concatenate(values))
        if not bool(np.isfinite(result).all()):
            raise ValueError("calibration thresholds contain non-finite scores")
        return result

    def weighted_tp(self, threshold: float) -> np.ndarray:
        result = np.empty_like(self.weighted_gt)
        for index, (scores, prefix, total) in enumerate(zip(self.sorted_scores, self.prefix_weights, self.weighted_gt, strict=True)):
            missed = int(np.searchsorted(scores, threshold, side="left"))
            result[index] = float(total) - float(prefix[missed])
        return result

    def weighted_recall(self, threshold: float) -> float:
        denominator = float(self.weighted_gt.sum())
        if denominator <= 1e-12:
            return 1.0
        return float(self.weighted_tp(threshold).sum() / denominator)

    def lower_confidence_bound(self, threshold: float, *, bootstrap_replicates: int, bootstrap_seed: int) -> float:
        value = _weighted_lcb(
            self.weighted_tp(threshold),
            self.weighted_gt,
            bootstrap_replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed),
        )
        return 1.0 if value is None else float(value)


def _highest_monotone_threshold(thresholds: np.ndarray, predicate: Any) -> float | None:
    lower = 0
    upper = int(thresholds.size) - 1
    best = -1
    while lower <= upper:
        middle = (lower + upper) // 2
        if bool(predicate(float(thresholds[middle]))):
            best = middle
            lower = middle + 1
        else:
            upper = middle - 1
    return None if best < 0 else float(thresholds[best])


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in metrics.items() if key != "per_pose"}


def _select_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    target_weighted_recall: float,
) -> CalibrationSelection:
    if not rows:
        raise ValueError("calibration produced no threshold rows")
    target = float(target_weighted_recall)
    if not 0.0 < target < 1.0:
        raise ValueError("target weighted recall must lie strictly between zero and one")

    def strict(row: Mapping[str, Any]) -> bool:
        return all(
            float(metrics["weighted_recall"]) > target
            and metrics.get("weighted_recall_lcb") is not None
            and float(metrics["weighted_recall_lcb"]) > target
            for metrics in row["scene_metrics"].values()
        )

    def mean_target(row: Mapping[str, Any]) -> bool:
        return all(float(metrics["weighted_recall"]) > target for metrics in row["scene_metrics"].values())

    strict_rows = [row for row in rows if strict(row)]
    mean_rows = [row for row in rows if mean_target(row)]
    if strict_rows:
        selected = max(strict_rows, key=lambda row: float(row["threshold"]))
        status = "strict_lcb_target"
    elif mean_rows:
        selected = max(mean_rows, key=lambda row: float(row["threshold"]))
        status = "mean_target"
    else:
        selected = max(
            rows,
            key=lambda row: (
                float(row["mean_weighted_recall"]),
                -np.inf if row["mean_weighted_recall_lcb"] is None else float(row["mean_weighted_recall_lcb"]),
                float(row["threshold"]),
            ),
        )
        status = "diagnostic"
    mean_lcb = selected["mean_weighted_recall_lcb"]
    return CalibrationSelection(
        protocol=str(protocol),
        threshold=float(selected["threshold"]),
        status=status,
        target_weighted_recall=target,
        mean_weighted_recall=float(selected["mean_weighted_recall"]),
        mean_weighted_recall_lcb=None if mean_lcb is None else float(mean_lcb),
        mean_target_met=status in {"strict_lcb_target", "mean_target"},
        confidence_target_met=status == "strict_lcb_target",
        selection_split="calibration",
        test_read=False,
        scene_metrics={str(scene): dict(metrics) for scene, metrics in selected["scene_metrics"].items()},
        threshold_rows=tuple(dict(row) for row in rows),
    )


def select_calibration_workpoint(
    threshold_rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    target_weighted_recall: float = TARGET_WEIGHTED_RECALL,
) -> CalibrationSelection:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(threshold_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"calibration threshold row {index} is not an object")
        if str(row.get("selection_split", "calibration")) != "calibration":
            raise ValueError("V5 calibration rows must declare selection_split=calibration")
        if row.get("test_read") is True:
            raise ValueError("test-tainted V5 calibration row")
        scene_metrics = row.get("scene_metrics")
        if not isinstance(scene_metrics, Mapping) or not scene_metrics:
            raise ValueError("V5 calibration row has no scene_metrics")
        means = [float(metrics["weighted_recall"]) for metrics in scene_metrics.values()]
        lcb_values = [metrics.get("weighted_recall_lcb") for metrics in scene_metrics.values()]
        finite_lcb = [float(value) for value in lcb_values if value is not None]
        normalized.append({
            "threshold": float(row["threshold"]),
            "scene_metrics": {str(scene): dict(metrics) for scene, metrics in scene_metrics.items()},
            "mean_weighted_recall": float(np.mean(means)),
            "mean_weighted_recall_lcb": float(np.mean(finite_lcb)) if finite_lcb else None,
        })
    return _select_from_rows(normalized, protocol=protocol, target_weighted_recall=target_weighted_recall)


def _calibrate(
    scene_records: Mapping[str, Iterable[PoseScores]],
    *,
    protocol: str,
    target_weighted_recall: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> CalibrationSelection:
    if not scene_records:
        raise ValueError("V5 calibration requires at least one scene")
    records = {str(scene): value for scene, value in scene_records.items()}
    indices = {
        scene: _SceneRecallIndex.from_records(values, scene=scene)
        for scene, values in records.items()
    }
    scenes = tuple(sorted(indices))
    threshold_values = [index.thresholds() for index in indices.values()]
    thresholds = np.unique(np.concatenate(threshold_values))
    target = float(target_weighted_recall)
    scene_seeds = {scene: int(bootstrap_seed) + scene_index * 100_003 for scene_index, scene in enumerate(scenes)}
    mean_threshold = _highest_monotone_threshold(
        thresholds,
        lambda value: all(indices[scene].weighted_recall(value) > target for scene in scenes),
    )
    strict_threshold = _highest_monotone_threshold(
        thresholds,
        lambda value: all(
            indices[scene].weighted_recall(value) > target
            and indices[scene].lower_confidence_bound(
                value,
                bootstrap_replicates=int(bootstrap_replicates),
                bootstrap_seed=scene_seeds[scene],
            ) > target
            for scene in scenes
        ),
    )
    if strict_threshold is not None:
        selected_threshold, status = strict_threshold, "strict_lcb_target"
    elif mean_threshold is not None:
        selected_threshold, status = mean_threshold, "mean_target"
    else:
        selected_threshold, status = float(thresholds[0]), "diagnostic"
    scene_metrics = {
        scene: _compact_metrics(
            evaluate_scene(
                records[scene],
                selected_threshold,
                bootstrap_replicates=int(bootstrap_replicates),
                bootstrap_seed=scene_seeds[scene],
            )
        )
        for scene in scenes
    }
    recalls = [float(metrics["weighted_recall"]) for metrics in scene_metrics.values()]
    lcbs = [float(metrics["weighted_recall_lcb"]) for metrics in scene_metrics.values() if metrics.get("weighted_recall_lcb") is not None]
    final_row = {
        "threshold": selected_threshold,
        "selection_split": "calibration",
        "test_read": False,
        "scene_metrics": scene_metrics,
        "mean_weighted_recall": float(np.mean(recalls)),
        "mean_weighted_recall_lcb": float(np.mean(lcbs)) if lcbs else None,
    }
    return CalibrationSelection(
        protocol=str(protocol),
        threshold=selected_threshold,
        status=status,
        target_weighted_recall=target,
        mean_weighted_recall=float(np.mean(recalls)),
        mean_weighted_recall_lcb=float(np.mean(lcbs)) if lcbs else None,
        mean_target_met=status in {"strict_lcb_target", "mean_target"},
        confidence_target_met=status == "strict_lcb_target",
        selection_split="calibration",
        test_read=False,
        scene_metrics=scene_metrics,
        threshold_rows=(final_row,),
    )


def calibrate_source_global(
    source_calibration: Mapping[str, Iterable[PoseScores]],
    *,
    target_weighted_recall: float = TARGET_WEIGHTED_RECALL,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> CalibrationSelection:
    return _calibrate(
        source_calibration,
        protocol="source_global",
        target_weighted_recall=target_weighted_recall,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def calibrate_target(
    target_calibration: Iterable[PoseScores],
    *,
    scene: str = "target",
    target_weighted_recall: float = TARGET_WEIGHTED_RECALL,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> CalibrationSelection:
    return _calibrate(
        {str(scene): target_calibration},
        protocol="target_calibrated",
        target_weighted_recall=target_weighted_recall,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


__all__ = [
    "CalibrationSelection",
    "calibrate_source_global",
    "calibrate_target",
    "select_calibration_workpoint",
]
