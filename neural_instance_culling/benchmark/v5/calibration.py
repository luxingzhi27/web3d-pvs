"""Calibration protocols for shared and five-fold LOSO V5 evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import PoseRecord
from .metrics import TARGET_WEIGHTED_RECALL, _weighted_lcb, evaluate_scene


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
            "mean_weighted_recall_lcb": (
                None if self.mean_weighted_recall_lcb is None else float(self.mean_weighted_recall_lcb)
            ),
            "mean_target_met": bool(self.mean_target_met),
            "confidence_target_met": bool(self.confidence_target_met),
            "selection_split": self.selection_split,
            "test_read": False,
            "scene_metrics": {
                str(scene): dict(metrics) for scene, metrics in self.scene_metrics.items()
            },
            "threshold_rows": [dict(row) for row in self.threshold_rows],
        }


def _materialize_calibration(
    records: Iterable[PoseRecord],
    *,
    scene: str,
) -> tuple[PoseRecord, ...]:
    materialized = tuple(records)
    if not materialized:
        raise ValueError(f"calibration split is empty for scene {scene!r}")
    if any(record.split != "calibration" for record in materialized):
        raise ValueError("V5 threshold selection accepts calibration rows only")
    if any(record.scene not in (None, scene) for record in materialized):
        raise ValueError(f"calibration row scene disagrees with {scene!r}")
    return materialized


def _candidate_thresholds(scene_records: Mapping[str, Sequence[PoseRecord]]) -> np.ndarray:
    values: list[np.ndarray] = []
    for scene, records in scene_records.items():
        if not records:
            raise ValueError(f"cannot calibrate an empty scene: {scene!r}")
        for record in records:
            if record.split == "test":
                raise ValueError("test rows are forbidden during V5 threshold selection")
            if record.scores.size:
                values.append(np.asarray(record.scores, dtype=np.float64))
    if not values:
        raise ValueError("calibration contains no candidate scores")
    thresholds = np.unique(np.concatenate(values))
    if not bool(np.isfinite(thresholds).all()):
        raise ValueError("calibration thresholds contain non-finite scores")
    return thresholds


@dataclass(frozen=True)
class _SceneRecallIndex:
    """Positive-only exact change-point index for one scene.

    Weighted recall and its pose bootstrap depend only on positive scores.  A
    threshold lookup therefore never rescans the much larger negative set.
    """

    sorted_scores: tuple[np.ndarray, ...]
    prefix_weights: tuple[np.ndarray, ...]
    weighted_gt: np.ndarray

    @classmethod
    def from_records(cls, records: Sequence[PoseRecord]) -> "_SceneRecallIndex":
        scores: list[np.ndarray] = []
        prefixes: list[np.ndarray] = []
        totals: list[float] = []
        for record in records:
            positive = record.targets.astype(bool, copy=False)
            pose_scores = np.asarray(record.scores[positive], dtype=np.float64)
            pose_weights = np.asarray(record.visible_weights[positive], dtype=np.float64)
            if pose_scores.size:
                order = np.argsort(pose_scores, kind="stable")
                pose_scores = pose_scores[order]
                pose_weights = pose_weights[order]
                prefix = np.concatenate(
                    [np.zeros((1,), dtype=np.float64), np.cumsum(pose_weights, dtype=np.float64)]
                )
            else:
                prefix = np.zeros((1,), dtype=np.float64)
            scores.append(pose_scores)
            prefixes.append(prefix)
            totals.append(float(pose_weights.sum()))
        return cls(tuple(scores), tuple(prefixes), np.asarray(totals, dtype=np.float64))

    def weighted_tp(self, threshold: float) -> np.ndarray:
        result = np.empty_like(self.weighted_gt)
        for pose_index, (scores, prefix, total) in enumerate(
            zip(self.sorted_scores, self.prefix_weights, self.weighted_gt, strict=True)
        ):
            missed = int(np.searchsorted(scores, threshold, side="left"))
            result[pose_index] = float(total) - float(prefix[missed])
        return result

    def weighted_recall(self, threshold: float) -> float:
        denominator = float(self.weighted_gt.sum())
        if denominator <= 1e-12:
            return 1.0
        return float(self.weighted_tp(threshold).sum() / denominator)

    def lower_confidence_bound(
        self,
        threshold: float,
        *,
        bootstrap_replicates: int,
        bootstrap_seed: int,
    ) -> float:
        value = _weighted_lcb(
            self.weighted_tp(threshold),
            self.weighted_gt,
            bootstrap_replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed),
        )
        return 1.0 if value is None else float(value)


def _highest_monotone_threshold(
    thresholds: np.ndarray,
    predicate: Any,
) -> float | None:
    """Return the highest ascending change point satisfying a monotone predicate."""

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
    return {
        str(key): value
        for key, value in metrics.items()
        if key != "per_pose"
    }


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
        scene_rows = row["scene_metrics"]
        return all(
            float(metrics["weighted_recall"]) > target
            and metrics.get("weighted_recall_lcb") is not None
            and float(metrics["weighted_recall_lcb"]) > target
            for metrics in scene_rows.values()
        )

    def mean_target(row: Mapping[str, Any]) -> bool:
        # source_global is a common threshold.  Requiring every source's
        # point estimate prevents a large source from hiding an unsafe source;
        # the reported mean remains scene-equal for comparison.
        return all(
            float(metrics["weighted_recall"]) > target
            for metrics in row["scene_metrics"].values()
        )

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
        mean_target_met=bool(status in {"strict_lcb_target", "mean_target"}),
        confidence_target_met=bool(status == "strict_lcb_target"),
        selection_split="calibration",
        test_read=False,
        scene_metrics={
            str(scene): dict(metrics) for scene, metrics in selected["scene_metrics"].items()
        },
        threshold_rows=tuple(dict(row) for row in rows),
    )


def select_calibration_workpoint(
    threshold_rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    target_weighted_recall: float = TARGET_WEIGHTED_RECALL,
) -> CalibrationSelection:
    """Select from already computed calibration rows.

    This is intentionally a V5-specific selector: rows must contain
    ``scene_metrics`` and cannot be built from a test split.
    """
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
        normalized.append(
            {
                "threshold": float(row["threshold"]),
                "scene_metrics": {str(scene): dict(metrics) for scene, metrics in scene_metrics.items()},
                "mean_weighted_recall": float(np.mean(means)),
                "mean_weighted_recall_lcb": float(np.mean(finite_lcb)) if finite_lcb else None,
            }
        )
    return _select_from_rows(
        normalized,
        protocol=protocol,
        target_weighted_recall=target_weighted_recall,
    )


def _calibrate(
    scene_records: Mapping[str, Iterable[PoseRecord]],
    *,
    protocol: str,
    target_weighted_recall: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> CalibrationSelection:
    if not scene_records:
        raise ValueError("V5 calibration requires at least one scene")
    normalized = {
        str(scene): _materialize_calibration(records, scene=str(scene))
        for scene, records in scene_records.items()
    }
    scenes = tuple(sorted(normalized))
    thresholds = _candidate_thresholds(normalized)
    indices = {scene: _SceneRecallIndex.from_records(normalized[scene]) for scene in scenes}
    target = float(target_weighted_recall)
    scene_seeds = {
        scene: int(bootstrap_seed) + scene_index * 100_003
        for scene_index, scene in enumerate(scenes)
    }

    mean_threshold = _highest_monotone_threshold(
        thresholds,
        lambda threshold: all(
            indices[scene].weighted_recall(threshold) > target for scene in scenes
        ),
    )
    strict_threshold = _highest_monotone_threshold(
        thresholds,
        lambda threshold: all(
            indices[scene].weighted_recall(threshold) > target
            and indices[scene].lower_confidence_bound(
                threshold,
                bootstrap_replicates=int(bootstrap_replicates),
                bootstrap_seed=scene_seeds[scene],
            ) > target
            for scene in scenes
        ),
    )
    if strict_threshold is not None:
        selected_threshold = strict_threshold
        status = "strict_lcb_target"
    elif mean_threshold is not None:
        selected_threshold = mean_threshold
        status = "mean_target"
    else:
        selected_threshold = float(thresholds[0])
        status = "diagnostic"

    scene_metrics: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        scene_metrics[scene] = _compact_metrics(
            evaluate_scene(
                normalized[scene],
                selected_threshold,
                bootstrap_replicates=int(bootstrap_replicates),
                bootstrap_seed=scene_seeds[scene],
            )
        )
    recalls = [float(metrics["weighted_recall"]) for metrics in scene_metrics.values()]
    lcb_values = [
        float(metrics["weighted_recall_lcb"])
        for metrics in scene_metrics.values()
        if metrics.get("weighted_recall_lcb") is not None
    ]
    mean_recall = float(np.mean(recalls))
    mean_lcb = float(np.mean(lcb_values)) if lcb_values else None
    final_row = {
        "threshold": selected_threshold,
        "selection_split": "calibration",
        "test_read": False,
        "scene_metrics": scene_metrics,
        "mean_weighted_recall": mean_recall,
        "mean_weighted_recall_lcb": mean_lcb,
    }
    return CalibrationSelection(
        protocol=str(protocol),
        threshold=selected_threshold,
        status=status,
        target_weighted_recall=target,
        mean_weighted_recall=mean_recall,
        mean_weighted_recall_lcb=mean_lcb,
        mean_target_met=status in {"strict_lcb_target", "mean_target"},
        confidence_target_met=status == "strict_lcb_target",
        selection_split="calibration",
        test_read=False,
        scene_metrics=scene_metrics,
        threshold_rows=(final_row,),
    )


def calibrate_source_global(
    source_calibration: Mapping[str, Iterable[PoseRecord]],
    *,
    target_weighted_recall: float = TARGET_WEIGHTED_RECALL,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> CalibrationSelection:
    """Freeze one common threshold using only the four source calibrations."""
    return _calibrate(
        source_calibration,
        protocol="source_global",
        target_weighted_recall=target_weighted_recall,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def calibrate_target(
    target_calibration: Iterable[PoseRecord],
    *,
    scene: str = "target",
    target_weighted_recall: float = TARGET_WEIGHTED_RECALL,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> CalibrationSelection:
    """Freeze one scalar threshold from the held-out scene calibration only."""
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
