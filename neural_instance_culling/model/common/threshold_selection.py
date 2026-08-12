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


def select_weighted_cull_workpoint(
    rows: list[dict[str, Any]],
    target_weighted_recall: float = DEFAULT_TARGET_WEIGHTED_RECALL,
    minimum_lower_confidence_bound: float | None = None,
    minimum_threshold: float | None = None,
    maximum_threshold: float | None = None,
) -> dict[str, Any] | None:
    """Select the most useful-culling row after the weighted safety gate.

    This is the selection rule for the direction-depth survival experiment.
    Weighted recall and its one-sided lower bound define screen safety; useful
    cull is optimized only after that filter.  Ordinary pose recall is not a
    gate because it weights every instance equally and can reject a model that
    preserves the visible mass that determines the rendered image.
    """
    safe = weighted_recall_safe_rows(
        rows,
        target_weighted_recall,
        minimum_lower_confidence_bound=minimum_lower_confidence_bound,
    )
    if minimum_threshold is not None:
        safe = [row for row in safe if float(row.get("threshold", -np.inf)) >= float(minimum_threshold)]
    if maximum_threshold is not None:
        safe = [row for row in safe if float(row.get("threshold", np.inf)) <= float(maximum_threshold)]
    if not safe:
        return None
    return max(
        safe,
        key=lambda row: (
            float(row.get("pose_useful_cull", 0.0)),
            float(row.get("pose_balanced_accuracy", 0.0)),
            float(row.get("pose_precision", 0.0)),
            -float(row.get("avg_pred_count", 0.0)),
        ),
    )


def _threshold_row(rows: list[dict[str, Any]], threshold: float, atol: float = 1e-6) -> dict[str, Any] | None:
    """Return the row registered for one scalar threshold, if present."""
    for row in rows:
        try:
            if abs(float(row.get("threshold", -np.inf)) - float(threshold)) <= float(atol):
                return row
        except (TypeError, ValueError):
            continue
    return None


def select_healthy_weighted_cull_workpoint(
    rows: list[dict[str, Any]],
    target_weighted_recall: float = DEFAULT_TARGET_WEIGHTED_RECALL,
    minimum_lower_confidence_bound: float | None = None,
    minimum_threshold: float = 0.40,
    maximum_threshold: float = 0.60,
    anchor_threshold: float = 0.50,
) -> dict[str, Any] | None:
    """Select a safe workpoint only after the score distribution passes health gates.

    A low threshold can satisfy weighted recall simply by predicting almost all
    candidates.  For the new relation-survival line, the natural probability
    boundary at 0.5 must itself satisfy the weighted-recall safety rule.  A
    second safe row must exist in the registered middle interval, and the
    positive/negative score quantile gap must be available and positive.  This
    function deliberately remains separate from ``select_weighted_cull_workpoint``
    so historical baselines keep their original calibration semantics.
    """
    health = summarize_threshold_health(
        rows,
        target_weighted_recall=target_weighted_recall,
        minimum_lower_confidence_bound=minimum_lower_confidence_bound,
        healthy_minimum=minimum_threshold,
        healthy_maximum=maximum_threshold,
        anchor_threshold=anchor_threshold,
    )
    if not bool(health.get("hasHealthySafeWorkpoint")):
        return None
    return select_weighted_cull_workpoint(
        rows,
        target_weighted_recall=target_weighted_recall,
        minimum_lower_confidence_bound=minimum_lower_confidence_bound,
        minimum_threshold=minimum_threshold,
        maximum_threshold=maximum_threshold,
    )


def summarize_threshold_health(
    rows: list[dict[str, Any]],
    target_weighted_recall: float = DEFAULT_TARGET_WEIGHTED_RECALL,
    minimum_lower_confidence_bound: float | None = None,
    healthy_minimum: float = 0.40,
    healthy_maximum: float = 0.60,
    anchor_threshold: float = 0.50,
) -> dict[str, Any]:
    """Describe whether safety is achieved near the fixed 0.5 score anchor.

    ``safe`` only means that some threshold can preserve the visible mass.  A
    ``healthy`` workpoint additionally requires the fixed probability boundary
    itself to be safe, a safe threshold in the registered middle interval, and
    score-distribution evidence.  This prevents a low-threshold recall rescue
    from being reported as a well-separated classifier.
    """
    safe = weighted_recall_safe_rows(
        rows,
        target_weighted_recall=target_weighted_recall,
        minimum_lower_confidence_bound=minimum_lower_confidence_bound,
    )
    safe_thresholds = sorted(float(row["threshold"]) for row in safe if "threshold" in row)
    healthy = [
        row for row in safe
        if float(healthy_minimum) <= float(row.get("threshold", -np.inf)) <= float(healthy_maximum)
    ]
    healthy_thresholds = sorted(float(row["threshold"]) for row in healthy if "threshold" in row)
    distribution = next(
        (
            row.get("scoreDistribution")
            for row in rows
            if isinstance(row.get("scoreDistribution"), dict)
        ),
        None,
    )
    score_gap = None
    if isinstance(distribution, dict) and distribution.get("positiveNegativeGapQ05Q95") is not None:
        score_gap = float(distribution["positiveNegativeGapQ05Q95"])
    positive_score_gap = score_gap is not None and score_gap > 0.0
    anchor_row = _threshold_row(rows, anchor_threshold)
    anchor_safe = bool(
        anchor_row is not None
        and weighted_recall_safe_rows(
            [anchor_row],
            target_weighted_recall=target_weighted_recall,
            minimum_lower_confidence_bound=minimum_lower_confidence_bound,
        )
    )
    healthy_with_gap = bool(healthy_thresholds) and anchor_safe and positive_score_gap
    anchor_metrics = {
        "threshold": float(anchor_threshold),
        "weightedRecall": None if anchor_row is None else anchor_row.get("pose_weighted_recall"),
        "weightedRecallLowerConfidenceBound": (
            None if anchor_row is None else anchor_row.get("weighted_recall_lower_confidence_bound")
        ),
        "poseRecall": None if anchor_row is None else anchor_row.get("pose_recall"),
        "posePrecision": None if anchor_row is None else anchor_row.get("pose_precision"),
        "poseUsefulCull": None if anchor_row is None else anchor_row.get("pose_useful_cull"),
        "avgPredCount": None if anchor_row is None else anchor_row.get("avg_pred_count"),
    }
    if healthy_with_gap:
        interpretation = "healthy_anchor_and_middle_safe_workpoint"
    elif healthy_thresholds and not anchor_safe:
        interpretation = "middle_safe_but_anchor_unsafe"
    elif healthy_thresholds and not positive_score_gap:
        interpretation = "middle_safe_but_score_distribution_unhealthy"
    elif safe_thresholds and not healthy_thresholds:
        interpretation = "safe_but_score_distribution_unhealthy"
    elif safe_thresholds and distribution is None:
        interpretation = "safe_but_score_distribution_missing"
    else:
        interpretation = "no_safe_workpoint"
    return {
        "safeThresholdCount": len(safe_thresholds),
        "safeThresholdMin": safe_thresholds[0] if safe_thresholds else None,
        "safeThresholdMax": safe_thresholds[-1] if safe_thresholds else None,
        "safeThresholdWidth": (safe_thresholds[-1] - safe_thresholds[0]) if len(safe_thresholds) >= 2 else 0.0,
        "healthyThresholdInterval": [float(healthy_minimum), float(healthy_maximum)],
        "healthySafeThresholdCount": len(healthy_thresholds),
        "healthySafeThresholdMin": healthy_thresholds[0] if healthy_thresholds else None,
        "healthySafeThresholdMax": healthy_thresholds[-1] if healthy_thresholds else None,
        "scoreGapQ05Q95": score_gap,
        "scoreDistributionAvailable": distribution is not None,
        "positiveScoreGap": positive_score_gap,
        "anchorThreshold": float(anchor_threshold),
        "anchorSafe": anchor_safe,
        "anchor": anchor_metrics,
        "hasHealthySafeWorkpoint": healthy_with_gap,
        "safeOnlyBelowHealthyInterval": bool(safe_thresholds) and not bool(healthy_thresholds) and safe_thresholds[-1] < float(healthy_minimum),
        "middleSafeButAnchorUnsafe": bool(healthy_thresholds) and not anchor_safe,
        "safeButOverlappingScoreDistribution": bool(healthy_thresholds) and not positive_score_gap,
        "interpretation": interpretation,
    }


def healthy_weighted_cull_selection_rule(
    target_weighted_recall: float = DEFAULT_TARGET_WEIGHTED_RECALL,
    minimum_lower_confidence_bound: float | None = None,
    minimum_threshold: float = 0.40,
    maximum_threshold: float = 0.60,
    anchor_threshold: float = 0.50,
) -> str:
    """Describe the stricter workpoint rule for new model candidates."""
    safety = weighted_cull_selection_rule(target_weighted_recall, minimum_lower_confidence_bound)
    return (
        f"{safety}; fixed anchor threshold {float(anchor_threshold):.2f} must also be safe; "
        f"select only within [{float(minimum_threshold):.2f},{float(maximum_threshold):.2f}] "
        "with positive weighted-positive-q05 minus negative-q95 score gap"
    )


def weighted_cull_selection_rule(
    target_weighted_recall: float = DEFAULT_TARGET_WEIGHTED_RECALL,
    minimum_lower_confidence_bound: float | None = None,
) -> str:
    """Describe the weighted-first culling workpoint rule for metadata."""
    rule = f"pose_weighted_recall > {float(target_weighted_recall):.3f}"
    if minimum_lower_confidence_bound is not None:
        rule += f" and one-sided lower bound > {float(minimum_lower_confidence_bound):.3f}"
    return (
        rule
        + "; among safe rows maximize pose_useful_cull, then pose_balanced_accuracy, "
        "pose_precision, and minimize avg_pred_count; pose_recall is diagnostic only"
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
