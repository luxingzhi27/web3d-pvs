"""Shared threshold and checkpoint workpoint selection rules."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


DEFAULT_TARGET_WEIGHTED_RECALL = 0.99


def target_weighted_recall_from_payload(
    payload: Mapping[str, Any] | None,
    default: float = DEFAULT_TARGET_WEIGHTED_RECALL,
) -> float:
    """Read the recorded weighted-recall target from an eval payload.

    Older summaries keep CLI arguments under ``args`` while benchmark
    summaries keep them under ``meta``.  Centralizing this lookup prevents a
    reader from silently applying a different safety target.
    """
    if not isinstance(payload, Mapping):
        return float(default)
    candidates: list[Any] = [payload.get("targetWeightedRecall"), payload.get("target_weighted_recall")]
    for container_name in ("args", "meta", "config"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            candidates.extend(
                [container.get("targetWeightedRecall"), container.get("target_weighted_recall")]
            )
    for value in candidates:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= numeric <= 1.0:
            return numeric
    return float(default)


def weighted_recall_safe_rows(
    rows: list[dict[str, Any]],
    target_weighted_recall: float = DEFAULT_TARGET_WEIGHTED_RECALL,
    minimum_point_estimate: float | None = None,
    minimum_lower_confidence_bound: float | None = None,
    minimum_pose_recall: float | None = None,
) -> list[dict[str, Any]]:
    """Return rows that strictly satisfy the weighted-recall safety target."""
    target = float(target_weighted_recall)
    return [
        row
        for row in rows
        if float(row.get("pose_weighted_recall", -1.0)) > target
        and (
            minimum_point_estimate is None
            or float(row.get("pose_weighted_recall", -1.0)) >= float(minimum_point_estimate)
        )
        and (
            minimum_lower_confidence_bound is None
            or float(row.get("weighted_recall_lower_confidence_bound", -1.0))
            > float(minimum_lower_confidence_bound)
        )
        and (
            minimum_pose_recall is None
            or float(row.get("pose_recall", -1.0)) >= float(minimum_pose_recall)
        )
    ]


def select_weighted_precision_workpoint(
    rows: list[dict[str, Any]],
    target_weighted_recall: float = DEFAULT_TARGET_WEIGHTED_RECALL,
    minimum_point_estimate: float | None = None,
    minimum_lower_confidence_bound: float | None = None,
    minimum_pose_recall: float | None = None,
) -> dict[str, Any] | None:
    """Select the highest pose-precision row after the strict safety filter."""
    safe = weighted_recall_safe_rows(
        rows,
        target_weighted_recall,
        minimum_point_estimate=minimum_point_estimate,
        minimum_lower_confidence_bound=minimum_lower_confidence_bound,
        minimum_pose_recall=minimum_pose_recall,
    )
    if not safe:
        return None
    return max(
        safe,
        key=lambda row: (
            float(row.get("pose_precision", 0.0)),
            float(row.get("pose_f1", 0.0)),
            float(row.get("pose_weighted_recall", 0.0)),
            -float(row.get("avg_pred_count", 0.0)),
        ),
    )


def weighted_precision_selection_key(
    row: dict[str, Any],
    target_weighted_recall: float = DEFAULT_TARGET_WEIGHTED_RECALL,
) -> tuple[float, float, float, float] | None:
    """Return the checkpoint comparison key, or None when the row is unsafe."""
    if float(row.get("pose_weighted_recall", -1.0)) <= float(target_weighted_recall):
        return None
    return (
        float(row.get("pose_precision", 0.0)),
        float(row.get("pose_f1", 0.0)),
        float(row.get("pose_weighted_recall", 0.0)),
        -float(row.get("avg_pred_count", 0.0)),
    )


def weighted_precision_selection_rule(
    target_weighted_recall: float = DEFAULT_TARGET_WEIGHTED_RECALL,
    minimum_point_estimate: float | None = None,
    minimum_lower_confidence_bound: float | None = None,
    minimum_pose_recall: float | None = None,
) -> str:
    safety = f"pose_weighted_recall > {float(target_weighted_recall):.3f}"
    if minimum_point_estimate is not None:
        safety += f" and point estimate >= {float(minimum_point_estimate):.4f}"
    if minimum_lower_confidence_bound is not None:
        safety += f" and one-sided lower bound > {float(minimum_lower_confidence_bound):.3f}"
    if minimum_pose_recall is not None:
        safety += f" and pose_recall >= {float(minimum_pose_recall):.3f}"
    return (
        safety + "; "
        "among safe rows maximize pose_precision, then pose_f1, weighted_recall, and minimize avg_pred_count"
    )


def attach_viewcell_weighted_recall_bounds(
    rows: list[dict[str, Any]],
    bootstrap_replicates: int,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Attach a one-sided view-cell bootstrap lower bound to calibration rows.

    The evaluator stores per-pose weighted recall only for the calibration
    pass.  Keeping the bootstrap here makes the safety rule shared by training,
    standalone calibration and future benchmark entry points.
    """
    count = int(bootstrap_replicates)
    if count <= 0:
        return rows
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    rng = np.random.default_rng(int(seed))
    for row in rows:
        values = np.asarray(row.pop("_pose_weighted_recall_values", []), dtype=np.float64)
        if values.size == 0:
            row["weighted_recall_lower_confidence_bound"] = 0.0
            row["weighted_recall_bootstrap_replicates"] = count
            continue
        indices = rng.integers(0, values.size, size=(count, values.size), endpoint=False)
        means = values[indices].mean(axis=1)
        row["weighted_recall_lower_confidence_bound"] = float(np.quantile(means, 1.0 - float(confidence)))
        row["weighted_recall_bootstrap_replicates"] = count
        row["weighted_recall_bootstrap_confidence"] = float(confidence)
    return rows
