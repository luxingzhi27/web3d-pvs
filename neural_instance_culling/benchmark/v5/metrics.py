"""Pose-set metrics used by the GCOF-PVS V5 evaluation layer."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

from neural_instance_culling.benchmark.score_sidecar import average_precision
from neural_instance_culling.model.common.culling_metrics import (
    candidate_normalized_occlusion_recall,
)

# The existing threshold-metric module is a model-side script and imports its
# sibling ``common`` package as a top-level module.  Register that model root
# once so V5 can reuse its bootstrap implementation without changing V4 code.
_MODEL_ROOT = Path(__file__).resolve().parents[2] / "model"
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))
from neural_instance_culling.model.pvs_threshold_metrics import (
    weighted_recall_lower_confidence_bound,
)

from .contracts import PoseRecord


TARGET_WEIGHTED_RECALL = 0.99
METRIC_FIELDS = (
    "weighted_recall",
    "weighted_recall_lcb",
    "ordinary_recall",
    "fn_over_gt",
    "bad_cull",
    "cnor",
    "useful_cull",
    "fp_over_gt",
    "pred_over_gt",
    "pose_pr_auc",
    "pose_prevalence",
    "pose_ap_lift",
)


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def _materialize_records(records: Iterable[PoseRecord]) -> tuple[PoseRecord, ...]:
    materialized = tuple(records)
    if not materialized:
        raise ValueError("V5 evaluation requires at least one pose")
    split = materialized[0].split
    scenes = {record.scene for record in materialized if record.scene is not None}
    if any(record.split != split for record in materialized):
        raise ValueError("V5 evaluation records must belong to one split")
    if len(scenes) > 1:
        raise ValueError("V5 scene evaluation records must belong to one scene")
    return materialized


def _weighted_lcb(
    weighted_tp: np.ndarray,
    weighted_gt: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> float | None:
    if int(bootstrap_replicates) <= 0:
        return None
    valid = np.isfinite(weighted_tp) & np.isfinite(weighted_gt) & (weighted_gt > 1e-12)
    if not bool(np.any(valid)):
        return 1.0
    return float(
        weighted_recall_lower_confidence_bound(
            weighted_tp[valid],
            weighted_gt[valid],
            replicates=int(bootstrap_replicates),
            seed=int(seed),
        )
    )


def _pose_row(record: PoseRecord, threshold: float) -> dict[str, Any]:
    predicted = record.scores >= float(threshold)
    truth = record.targets.astype(bool, copy=False)
    positive = truth
    negative = ~truth
    candidate_count = int(truth.size)
    gt_count = int(positive.sum())
    pred_count = int(predicted.sum())
    tp = int(np.logical_and(predicted, positive).sum())
    fp = int(np.logical_and(predicted, negative).sum())
    fn = int(np.logical_and(~predicted, positive).sum())
    tn = int(np.logical_and(~predicted, negative).sum())
    weighted_tp = float(record.visible_weights[np.logical_and(predicted, positive)].sum())
    weighted_gt = float(record.visible_weights[positive].sum())
    precision = _safe_ratio(tp, tp + fp, 1.0)
    recall = _safe_ratio(tp, gt_count, 1.0)
    specificity = _safe_ratio(tn, tn + fp, 1.0)
    accuracy = _safe_ratio(tp + tn, candidate_count, 1.0)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn)
    jaccard = _safe_ratio(tp, tp + fp + fn, 1.0)
    pose_ap = average_precision(
        record.scores.astype(np.float32, copy=False),
        record.targets.astype(np.uint8, copy=False),
    ) if gt_count > 0 else None
    return {
        "pose_id": record.pose_id,
        "candidate_count": candidate_count,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "weighted_tp": weighted_tp,
        "weighted_gt": weighted_gt,
        "ordinary_recall": recall,
        "weighted_recall": _safe_ratio(weighted_tp, weighted_gt, 1.0),
        "precision": precision,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "jaccard": jaccard,
        "useful_cull": _safe_ratio(tn, candidate_count),
        "bad_cull": _safe_ratio(fn, candidate_count),
        "fp_over_gt": _safe_ratio(fp, gt_count),
        "fn_over_gt": _safe_ratio(fn, gt_count),
        "pred_over_gt": _safe_ratio(pred_count, gt_count),
        "prevalence": _safe_ratio(gt_count, candidate_count),
        "average_precision": pose_ap,
    }


def evaluate_scene(
    records: Iterable[PoseRecord],
    threshold: float,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate one scene at one frozen threshold.

    Aggregate safety and culling metrics are computed from summed confusion
    counts.  CNOR is delegated to the repository's shared implementation and
    therefore keeps its per-pose candidate-normalized denominator.  AP and
    prevalence are pose-macro quantities over poses with at least one GT
    positive; aggregate AP/prevalence are included separately for diagnostics.
    """
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("V5 threshold must be finite")
    materialized = _materialize_records(records)
    rows = [_pose_row(record, threshold_value) for record in materialized]

    tp = float(sum(row["tp"] for row in rows))
    fp = float(sum(row["fp"] for row in rows))
    fn = float(sum(row["fn"] for row in rows))
    tn = float(sum(row["tn"] for row in rows))
    weighted_tp = float(sum(row["weighted_tp"] for row in rows))
    weighted_gt = float(sum(row["weighted_gt"] for row in rows))
    candidate_count = float(sum(row["candidate_count"] for row in rows))
    gt_count = float(sum(row["gt_count"] for row in rows))
    pred_count = float(sum(row["pred_count"] for row in rows))
    per_pose_candidate = np.asarray([row["candidate_count"] for row in rows], dtype=np.float64)
    per_pose_tn = np.asarray([row["tn"] for row in rows], dtype=np.float64)
    per_pose_fp = np.asarray([row["fp"] for row in rows], dtype=np.float64)
    positive_rows = [row for row in rows if row["gt_count"] > 0]
    pose_ap_values = np.asarray(
        [row["average_precision"] for row in positive_rows], dtype=np.float64
    )
    pose_prevalence_values = np.asarray(
        [row["prevalence"] for row in positive_rows], dtype=np.float64
    )
    all_scores = np.concatenate([record.scores for record in materialized])
    all_targets = np.concatenate([record.targets for record in materialized])
    aggregate_ap = average_precision(
        all_scores.astype(np.float32, copy=False), all_targets.astype(np.uint8, copy=False)
    ) if gt_count > 0 else None
    aggregate_prevalence = _safe_ratio(gt_count, candidate_count)
    pose_pr_auc = float(np.mean(pose_ap_values)) if pose_ap_values.size else None
    pose_prevalence = float(np.mean(pose_prevalence_values)) if pose_prevalence_values.size else None
    aggregate_lift = _safe_ratio(aggregate_ap, aggregate_prevalence) if aggregate_ap is not None else None
    pose_lift = _safe_ratio(pose_pr_auc, pose_prevalence) if pose_pr_auc is not None else None
    ordinary_recall = _safe_ratio(tp, gt_count, 1.0)
    weighted_recall = _safe_ratio(weighted_tp, weighted_gt, 1.0)
    precision = _safe_ratio(tp, tp + fp, 1.0)
    specificity = _safe_ratio(tn, tn + fp, 1.0)
    lcb = _weighted_lcb(
        np.asarray([row["weighted_tp"] for row in rows], dtype=np.float64),
        np.asarray([row["weighted_gt"] for row in rows], dtype=np.float64),
        bootstrap_replicates=int(bootstrap_replicates),
        seed=int(bootstrap_seed),
    )
    scene_values = {record.scene for record in materialized if record.scene is not None}
    scene = next(iter(scene_values)) if scene_values else None
    return {
        "scene": scene,
        "split": materialized[0].split,
        "threshold": threshold_value,
        "pose_count": len(rows),
        "positive_pose_count": len(positive_rows),
        "empty_gt_pose_count": len(rows) - len(positive_rows),
        "candidate_count": int(candidate_count),
        "gt_count": int(gt_count),
        "pred_count": int(pred_count),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "weighted_tp": weighted_tp,
        "weighted_gt": weighted_gt,
        "weighted_recall": weighted_recall,
        "weighted_recall_lcb": lcb,
        "ordinary_recall": ordinary_recall,
        "pose_recall": float(np.mean([row["ordinary_recall"] for row in rows])),
        "fn_over_gt": _safe_ratio(fn, gt_count),
        "bad_cull": _safe_ratio(fn, candidate_count),
        "cnor": candidate_normalized_occlusion_recall(
            per_pose_tn,
            per_pose_fp,
            per_pose_candidate,
        ),
        "useful_cull": _safe_ratio(tn, candidate_count),
        "fp_over_gt": _safe_ratio(fp, gt_count),
        "pred_over_gt": _safe_ratio(pred_count, gt_count),
        "precision": precision,
        "specificity": specificity,
        "accuracy": _safe_ratio(tp + tn, candidate_count, 1.0),
        "balanced_accuracy": 0.5 * (ordinary_recall + specificity),
        "f1": _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "jaccard": _safe_ratio(tp, tp + fp + fn, 1.0),
        "avg_candidate_count": _safe_ratio(candidate_count, len(rows)),
        "avg_gt_count": _safe_ratio(gt_count, len(rows)),
        "avg_pred_count": _safe_ratio(pred_count, len(rows)),
        "pose_pr_auc": pose_pr_auc,
        "pose_prevalence": pose_prevalence,
        "pose_ap_lift": pose_lift,
        "aggregate_pr_auc": aggregate_ap,
        "aggregate_prevalence": aggregate_prevalence,
        "aggregate_ap_lift": aggregate_lift,
        "per_pose": rows,
        "test_read": False,
    }


def metric_projection(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fixed result-table metric subset without changing values."""
    return {field: metrics.get(field) for field in METRIC_FIELDS}


__all__ = [
    "METRIC_FIELDS",
    "TARGET_WEIGHTED_RECALL",
    "evaluate_scene",
    "metric_projection",
]
