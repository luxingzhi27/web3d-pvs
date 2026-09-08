"""Threshold sweeps and score diagnostics for the current PVS model."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from common.threshold_selection import (
    select_weighted_precision_workpoint,
    target_weighted_recall_from_payload,
)


def threshold_grid() -> np.ndarray:
    # Spatially held-out positives can occupy the low-score tail even when
    # calibration still has a safe workpoint.  The grid must cover that tail
    # without turning threshold selection into test-time tuning.
    # A coarse geometric grid can hide real improvements when the safe
    # boundary lies around 1e-4--1e-3: adjacent values in the former grid were
    # more than twofold apart.  Model scores are computed once, so a denser
    # calibration-only grid adds negligible inference cost and avoids freezing
    # an unnecessarily conservative threshold.
    low = np.geomspace(1e-8, 1e-2, 97, dtype=np.float32)
    low = np.concatenate([np.asarray([0.0], dtype=np.float32), low])
    mid = np.asarray([0.0125, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2], dtype=np.float32)
    high = np.linspace(0.22, 0.9, 35, dtype=np.float32)
    return np.unique(np.concatenate([low, mid, high]))


def weighted_recall_lower_confidence_bound(
    weighted_tp: np.ndarray,
    weighted_gt: np.ndarray,
    replicates: int = 10000,
    seed: int = 0,
) -> float:
    """Return the percentile lower bound used by calibration safety gates.

    The resampling unit is a pose.  ``weighted_tp`` and ``weighted_gt`` are
    therefore per-pose weighted hit and GT mass, not already aggregated
    ratios.  A zero replicate count is reserved for lightweight diagnostics;
    formal calibration always passes at least 10,000.
    """
    tp = np.asarray(weighted_tp, dtype=np.float64).reshape(-1)
    gt = np.asarray(weighted_gt, dtype=np.float64).reshape(-1)
    if tp.size == 0 or tp.size != gt.size:
        return 0.0
    # Empty-candidate poses have no visible weight and must not affect the
    # aggregate bootstrap denominator.  Keeping them in the raw pose stream
    # is useful for pose-count diagnostics, but resampling them as successful
    # recall observations would make the lower bound depend on empty poses.
    valid = np.isfinite(tp) & np.isfinite(gt) & (gt > 1e-12)
    tp = tp[valid]
    gt = gt[valid]
    if tp.size == 0:
        return 1.0
    if int(replicates) <= 0:
        denominator = float(gt.sum())
        return 1.0 if denominator <= 1e-12 else float(tp.sum() / denominator)
    rng = np.random.default_rng(int(seed))
    # Keep memory bounded for larger calibration sets while retaining exactly
    # the requested number of bootstrap draws.
    values = np.empty((int(replicates),), dtype=np.float64)
    chunk = max(1, min(int(replicates), 1_000_000 // max(1, tp.size)))
    for start in range(0, int(replicates), chunk):
        end = min(int(replicates), start + chunk)
        indices = rng.integers(0, tp.size, size=(end - start, tp.size), endpoint=False)
        numerator = tp[indices].sum(axis=1)
        denominator = gt[indices].sum(axis=1)
        values[start:end] = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=denominator > 1e-12,
        )
    return float(np.quantile(values, 0.05))


def pose_macro_weighted_recall_lower_confidence_bound(
    pose_weighted_recall: np.ndarray,
    replicates: int = 10000,
    seed: int = 0,
) -> float:
    """Return the one-sided bootstrap bound for the pose-macro diagnostic.

    This is intentionally separate from ``weighted_recall_lower_confidence_bound``:
    the latter resamples weighted TP/GT sufficient statistics and therefore
    estimates the aggregate safety quantity.  A macro bound resamples already
    computed per-pose ratios and is diagnostic only.
    """
    values = np.asarray(pose_weighted_recall, dtype=np.float64).reshape(-1)
    if values.size == 0 or not bool(np.isfinite(values).all()):
        return 0.0
    if int(replicates) <= 0:
        return float(np.mean(values))
    rng = np.random.default_rng(int(seed))
    result = np.empty((int(replicates),), dtype=np.float64)
    chunk = max(1, min(int(replicates), 1_000_000 // max(1, values.size)))
    for start in range(0, int(replicates), chunk):
        end = min(int(replicates), start + chunk)
        indices = rng.integers(0, values.size, size=(end - start, values.size), endpoint=False)
        result[start:end] = values[indices].mean(axis=1)
    return float(np.quantile(result, 0.05))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return a finite weighted quantile for score-distribution diagnostics."""
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value_array.size == 0 or value_array.size != weight_array.size:
        return float("nan")
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    value_array = value_array[valid]
    weight_array = weight_array[valid]
    if value_array.size == 0:
        return float("nan")
    order = np.argsort(value_array, kind="mergesort")
    ordered_values = value_array[order]
    ordered_weights = weight_array[order]
    cumulative = np.cumsum(ordered_weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def score_distribution_summary(
    scores: np.ndarray,
    target: np.ndarray,
    visible_weights: np.ndarray,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Summarize score separation without selecting a threshold.

    Positive scores use their visible-weight mass; negative scores use unit
    candidate mass.  This keeps the summary aligned with the safety metric
    while still exposing whether the model only becomes safe after moving the
    threshold into the numerical tail.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    visible = np.asarray(visible_weights, dtype=np.float64).reshape(-1)
    if not (values.size == labels.size == visible.size):
        raise ValueError("score, target, and visible weight arrays must have equal length")
    if values.size == 0:
        return {
            "scoreCount": 0,
            "positiveCount": 0,
            "negativeCount": 0,
            "positiveWeightedQ005": None,
            "positiveWeightedQ01": None,
            "positiveWeightedQ05": None,
            "negativeQ95": None,
            "negativeQ99": None,
            "negativeQ995": None,
            "positiveNegativeGapQ01Q99": None,
            "positiveNegativeGapQ005Q995": None,
            "positiveNegativeGapQ05Q95": None,
            "rocAuc": None,
            "averagePrecision": None,
            "weightedRocAuc": None,
            "brierScore": None,
            "expectedCalibrationError": None,
        }
    if not np.isfinite(values).all() or not np.isfinite(labels).all() or not np.isfinite(visible).all():
        raise FloatingPointError("score distribution contains non-finite values")
    positive = labels > 0.5
    negative = ~positive
    positive_weights = np.maximum(visible, 1e-6)
    positive_q005 = _weighted_quantile(values[positive], positive_weights[positive], 0.005)
    positive_q01 = _weighted_quantile(values[positive], positive_weights[positive], 0.01)
    positive_q05 = _weighted_quantile(values[positive], positive_weights[positive], 0.05)
    negative_q95 = _weighted_quantile(values[negative], np.ones(np.count_nonzero(negative)), 0.95)
    negative_q99 = _weighted_quantile(values[negative], np.ones(np.count_nonzero(negative)), 0.99)
    negative_q995 = _weighted_quantile(values[negative], np.ones(np.count_nonzero(negative)), 0.995)
    gap = positive_q05 - negative_q95 if np.isfinite(positive_q05) and np.isfinite(negative_q95) else float("nan")
    gap_q01_q99 = positive_q01 - negative_q99 if np.isfinite(positive_q01) and np.isfinite(negative_q99) else float("nan")
    gap_q005_q995 = positive_q005 - negative_q995 if np.isfinite(positive_q005) and np.isfinite(negative_q995) else float("nan")

    unique_scores, inverse = np.unique(values, return_inverse=True)
    positive_counts = np.bincount(inverse, weights=positive.astype(np.float64), minlength=unique_scores.size)
    positive_mass = np.bincount(
        inverse,
        weights=np.where(positive, positive_weights, 0.0),
        minlength=unique_scores.size,
    )
    negative_counts = np.bincount(inverse, weights=negative.astype(np.float64), minlength=unique_scores.size)
    total_positive = float(positive_counts.sum())
    total_positive_mass = float(positive_mass.sum())
    total_negative = float(negative_counts.sum())
    negatives_before = np.cumsum(negative_counts) - negative_counts
    roc_auc = (
        float(np.sum(positive_counts * (negatives_before + 0.5 * negative_counts)))
        / (total_positive * total_negative)
        if total_positive > 0.0 and total_negative > 0.0
        else float("nan")
    )
    weighted_roc_auc = (
        float(np.sum(positive_mass * (negatives_before + 0.5 * negative_counts)))
        / (total_positive_mass * total_negative)
        if total_positive_mass > 0.0 and total_negative > 0.0
        else float("nan")
    )
    descending_positive = positive_counts[::-1]
    descending_negative = negative_counts[::-1]
    cumulative_positive = np.cumsum(descending_positive)
    cumulative_negative = np.cumsum(descending_negative)
    average_precision = (
        float(
            np.sum(
                (descending_positive / total_positive)
                * cumulative_positive
                / np.maximum(cumulative_positive + cumulative_negative, 1e-12)
            )
        )
        if total_positive > 0.0
        else float("nan")
    )

    evaluation_weights = np.where(positive, positive_weights, 1.0)
    brier = float(
        np.sum(evaluation_weights * np.square(values - labels))
        / max(1e-12, float(np.sum(evaluation_weights)))
    )
    bin_count = max(1, int(calibration_bins))
    bin_index = np.minimum((np.clip(values, 0.0, 1.0) * bin_count).astype(np.int64), bin_count - 1)
    ece = 0.0
    for current_bin in range(bin_count):
        selected = bin_index == current_bin
        if not np.any(selected):
            continue
        bin_weights = evaluation_weights[selected]
        total = float(np.sum(bin_weights))
        ece += (total / max(1e-12, float(np.sum(evaluation_weights)))) * abs(
            float(np.sum(bin_weights * values[selected]) / max(1e-12, total))
            - float(np.sum(bin_weights * labels[selected]) / max(1e-12, total))
        )
    return {
        "scoreCount": int(values.size),
        "positiveCount": int(np.count_nonzero(positive)),
        "negativeCount": int(np.count_nonzero(negative)),
        "positiveWeightedQ005": None if not np.isfinite(positive_q005) else float(positive_q005),
        "positiveWeightedQ01": None if not np.isfinite(positive_q01) else float(positive_q01),
        "positiveWeightedQ05": None if not np.isfinite(positive_q05) else float(positive_q05),
        "negativeQ95": None if not np.isfinite(negative_q95) else float(negative_q95),
        "negativeQ99": None if not np.isfinite(negative_q99) else float(negative_q99),
        "negativeQ995": None if not np.isfinite(negative_q995) else float(negative_q995),
        "positiveNegativeGapQ01Q99": None if not np.isfinite(gap_q01_q99) else float(gap_q01_q99),
        "positiveNegativeGapQ005Q995": None if not np.isfinite(gap_q005_q995) else float(gap_q005_q995),
        "positiveNegativeGapQ05Q95": None if not np.isfinite(gap) else float(gap),
        "rocAuc": None if not np.isfinite(roc_auc) else float(roc_auc),
        "averagePrecision": None if not np.isfinite(average_precision) else float(average_precision),
        "weightedRocAuc": None if not np.isfinite(weighted_roc_auc) else float(weighted_roc_auc),
        "positiveMean": float(np.mean(values[positive])) if np.any(positive) else None,
        "negativeMean": float(np.mean(values[negative])) if np.any(negative) else None,
        "brierScore": brier,
        "expectedCalibrationError": float(ece),
    }


def evaluate_thresholds(
    model,
    split,
    runtime_features: torch.Tensor,
    world_aabbs: np.ndarray,
    device: torch.device,
    poses_per_batch: int,
    max_steps: int | None,
    max_candidates_per_pose: int,
    seed: int,
    thresholds: np.ndarray,
    collect_pose_stats: bool = False,
    allow_candidate_visible_union: bool = False,
    bootstrap_replicates: int = 0,
    collect_score_stats: bool = False,
    collect_per_pose: bool = False,
    collect_raw_scores: bool = False,
    instance_to_glb: np.ndarray | None = None,
    glb_bytes: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if collect_raw_scores and not collect_per_pose:
        raise ValueError("raw score capture requires per-pose collection")
    model.eval()
    rng = np.random.default_rng(seed)
    th = thresholds.astype(np.float32)
    n_th = int(th.size)
    pose_recalls = np.zeros((n_th,), dtype=np.float64)
    pose_precisions = np.zeros((n_th,), dtype=np.float64)
    pose_f1s = np.zeros((n_th,), dtype=np.float64)
    pose_jaccards = np.zeros((n_th,), dtype=np.float64)
    pose_weighted_recalls = np.zeros((n_th,), dtype=np.float64)
    pose_specificities = np.zeros((n_th,), dtype=np.float64)
    pose_accuracies = np.zeros((n_th,), dtype=np.float64)
    pose_balanced_accuracies = np.zeros((n_th,), dtype=np.float64)
    pose_useful_culls = np.zeros((n_th,), dtype=np.float64)
    pose_bad_culls = np.zeros((n_th,), dtype=np.float64)
    pred_counts = np.zeros((n_th,), dtype=np.float64)
    predicted_glb_counts = np.zeros((n_th,), dtype=np.float64)
    predicted_glb_bytes = np.zeros((n_th,), dtype=np.float64)
    candidate_glb_count_sum = 0.0
    candidate_glb_bytes_sum = 0.0
    gt_count_sum = 0.0
    candidate_count_sum = 0.0
    agg_tp = np.zeros((n_th,), dtype=np.float64)
    agg_fp = np.zeros((n_th,), dtype=np.float64)
    agg_fn = np.zeros((n_th,), dtype=np.float64)
    agg_tn = np.zeros((n_th,), dtype=np.float64)
    agg_weighted_tp = np.zeros((n_th,), dtype=np.float64)
    agg_weighted_gt = 0.0
    pose_weighted_recall_values: list[list[float]] = [[] for _ in range(n_th)]
    pose_weighted_tp_values: list[list[float]] = [[] for _ in range(n_th)]
    pose_weighted_gt_values: list[list[float]] = [[] for _ in range(n_th)]
    score_values: list[np.ndarray] = []
    score_targets: list[np.ndarray] = []
    score_weights: list[np.ndarray] = []
    per_pose_values: list[list[dict[str, Any]]] = [[] for _ in range(n_th)]
    if (instance_to_glb is None) != (glb_bytes is None):
        raise ValueError("instance_to_glb and glb_bytes must be supplied together")
    resource_mapping = None
    resource_bytes = None
    if instance_to_glb is not None and glb_bytes is not None:
        resource_mapping = np.asarray(instance_to_glb, dtype=np.int64).reshape(-1)
        resource_bytes = np.asarray(glb_bytes, dtype=np.float64).reshape(-1)
        if resource_mapping.size != int(world_aabbs.shape[0]):
            raise ValueError("instance_to_glb does not match the runtime instance count")
        if resource_mapping.size and (
            int(resource_mapping.min()) < 0 or int(resource_mapping.max()) >= resource_bytes.size
        ):
            raise ValueError("instance_to_glb contains an id outside glb_bytes")
        if not bool(np.isfinite(resource_bytes).all()) or bool(np.any(resource_bytes < 0.0)):
            raise ValueError("glb_bytes must contain finite non-negative costs")
    pose_count = 0
    for pose_indices in split.pose_set_batches(poses_per_batch, rng, max_steps, include_empty=True):
        batch = split.build_pose_set_batch(
            pose_indices,
            world_aabbs,
            rng,
            max_candidates_per_pose=max_candidates_per_pose,
            allow_candidate_visible_union=allow_candidate_visible_union,
            include_empty=True,
        )
        batch_offsets = np.asarray(batch["pose_offsets"], dtype=np.int64)
        batch_visible_counts = np.asarray(batch.get("visible_counts", []), dtype=np.int64)
        empty_rows = np.flatnonzero(np.diff(batch_offsets) == 0)
        for row_index in empty_rows.tolist():
            if row_index < batch_visible_counts.size and int(batch_visible_counts[row_index]) != 0:
                raise ValueError(
                    f"Stored candidate semantics are invalid at empty-candidate pose row {row_index}: "
                    "the pose has visible GT instances but no candidate references."
                )
            pose_precisions += 1.0
            pose_recalls += 1.0
            pose_f1s += 1.0
            pose_jaccards += 1.0
            pose_weighted_recalls += 1.0
            pose_specificities += 1.0
            pose_accuracies += 1.0
            pose_balanced_accuracies += 1.0
            pose_useful_culls += 1.0
            if collect_pose_stats:
                for threshold_index in range(n_th):
                    pose_weighted_recall_values[threshold_index].append(1.0)
                    pose_weighted_tp_values[threshold_index].append(0.0)
                    pose_weighted_gt_values[threshold_index].append(0.0)
            if collect_per_pose:
                empty_metrics = {
                    "precision": 1.0, "recall": 1.0, "weightedRecall": 1.0,
                    "f1": 1.0, "jaccard": 1.0, "accuracy": 1.0,
                    "balancedAccuracy": 1.0, "specificity": 1.0,
                    "usefulCull": 1.0, "badCull": 0.0,
                    "avgPredCount": 0.0, "avgCandidateCount": 0.0,
                    "avgGtCount": 0.0, "predictedGlbCount": 0.0,
                    "candidateGlbCount": 0.0, "candidateGlbBytes": 0.0,
                    "predictedGlbBytes": 0.0, "glbByteReduction": 0.0,
                    "missPixelRate": 0.0, "wrongIdPixelRate": 0.0,
                    "extraPixelRate": 0.0,
                }
                for threshold_index in range(n_th):
                    per_pose_values[threshold_index].append({
                        "poseIndex": int(pose_indices[row_index]),
                        "candidateIds": [], "predictedIds": [],
                        "candidateScores": [] if collect_raw_scores else None,
                        "targets": [] if collect_raw_scores else None,
                        "visibleWeights": [] if collect_raw_scores else None,
                        "metrics": dict(empty_metrics),
                    })
            pose_count += 1
        if batch["instance"].size == 0:
            continue
        camera = torch.from_numpy(batch["camera"]).to(device)
        camera_world = torch.from_numpy(batch["camera_world"]).to(device)
        view = torch.from_numpy(batch["camera_view"]).to(device)
        ids = torch.from_numpy(batch["instance"]).to(device)
        query_center_world = torch.from_numpy(batch["query_center_world"]).to(device)
        viewcell_radius_m = torch.from_numpy(batch["viewcell_radius_m"]).to(device)
        logits = model.compute_visibility_logits(
            camera,
            view,
            camera_world,
            ids,
            runtime_features=runtime_features,
            query_center_world=query_center_world,
            viewcell_radius_m=viewcell_radius_m,
            pose_offsets=batch["pose_offsets"],
        )
        scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        target = batch["target"].astype(bool, copy=False)
        weights = batch.get("visible_weights", np.zeros_like(batch["target"], dtype=np.float32)).astype(np.float64, copy=False)
        if collect_score_stats:
            score_values.append(scores.astype(np.float64, copy=True))
            score_targets.append(batch["target"].astype(np.float64, copy=True))
            score_weights.append(weights.astype(np.float64, copy=True))
        offsets = batch["pose_offsets"]
        for i in range(offsets.size - 1):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            if end <= start:
                continue
            pred = scores[start:end, None] >= th[None, :]
            yy = target[start:end, None]
            ww = weights[start:end, None]
            local_tp = np.logical_and(pred, yy).sum(axis=0).astype(np.float64)
            local_fp = np.logical_and(pred, ~yy).sum(axis=0).astype(np.float64)
            local_fn = np.logical_and(~pred, yy).sum(axis=0).astype(np.float64)
            local_tn = np.logical_and(~pred, ~yy).sum(axis=0).astype(np.float64)
            precision = np.divide(
                local_tp,
                local_tp + local_fp,
                out=np.ones_like(local_tp),
                where=(local_tp + local_fp) > 0.0,
            )
            recall = np.divide(
                local_tp,
                local_tp + local_fn,
                out=np.ones_like(local_tp),
                where=(local_tp + local_fn) > 0.0,
            )
            specificity = np.divide(
                local_tn,
                local_tn + local_fp,
                out=np.ones_like(local_tn),
                where=(local_tn + local_fp) > 0.0,
            )
            accuracy = (local_tp + local_tn) / max(1.0, float(end - start))
            f1 = 2.0 * precision * recall / np.maximum(1e-8, precision + recall)
            pose_precisions += precision
            pose_recalls += recall
            pose_f1s += f1
            pose_jaccards += np.divide(
                local_tp,
                local_tp + local_fp + local_fn,
                out=np.ones_like(local_tp),
                where=(local_tp + local_fp + local_fn) > 0.0,
            )
            weighted_tp = (np.logical_and(pred, yy) * ww).sum(axis=0)
            # The GT weight mass does not depend on the threshold.  Expand
            # the scalar to the threshold axis so pose-level bootstrap rows
            # have the same shape for both one and many thresholds.
            weighted_gt_scalar = float((yy * ww).sum())
            weighted_gt = np.full((n_th,), weighted_gt_scalar, dtype=np.float64)
            weighted_recall = np.divide(
                weighted_tp,
                weighted_gt,
                out=np.ones_like(weighted_tp),
                where=weighted_gt > 0.0,
            )
            pose_weighted_recalls += weighted_recall
            agg_weighted_tp += weighted_tp
            agg_weighted_gt += weighted_gt_scalar
            pose_specificities += specificity
            pose_accuracies += accuracy
            pose_balanced_accuracies += 0.5 * (recall + specificity)
            pose_useful_culls += local_tn / max(1.0, float(end - start))
            pose_bad_culls += local_fn / max(1.0, float(end - start))
            local_predicted_glb_counts = np.zeros((n_th,), dtype=np.float64)
            local_predicted_glb_bytes = np.zeros((n_th,), dtype=np.float64)
            local_candidate_glb_count = 0.0
            local_candidate_glb_bytes = 0.0
            candidate_ids = np.asarray(batch["instance"][start:end], dtype=np.uint32)
            if resource_mapping is not None and resource_bytes is not None:
                candidate_glbs = np.unique(resource_mapping[candidate_ids.astype(np.int64, copy=False)])
                local_candidate_glb_count = float(candidate_glbs.size)
                local_candidate_glb_bytes = (
                    float(resource_bytes[candidate_glbs].sum()) if candidate_glbs.size else 0.0
                )
                candidate_glb_count_sum += local_candidate_glb_count
                candidate_glb_bytes_sum += local_candidate_glb_bytes
                instance_glbs = resource_mapping[candidate_ids.astype(np.int64, copy=False)]
                for threshold_index in range(n_th):
                    selected_glbs = np.unique(instance_glbs[pred[:, threshold_index]])
                    local_predicted_glb_counts[threshold_index] = float(selected_glbs.size)
                    local_predicted_glb_bytes[threshold_index] = (
                        float(resource_bytes[selected_glbs].sum()) if selected_glbs.size else 0.0
                    )
                predicted_glb_counts += local_predicted_glb_counts
                predicted_glb_bytes += local_predicted_glb_bytes
            if collect_pose_stats:
                for threshold_index in range(n_th):
                    pose_weighted_recall_values[threshold_index].append(float(weighted_recall[threshold_index]))
                    pose_weighted_tp_values[threshold_index].append(float(weighted_tp[threshold_index]))
                    pose_weighted_gt_values[threshold_index].append(float(weighted_gt[threshold_index]))
            if collect_per_pose:
                for threshold_index in range(n_th):
                    local_pred_ids = candidate_ids[pred[:, threshold_index]].astype(np.uint32, copy=False)
                    per_pose_values[threshold_index].append({
                        "poseIndex": int(pose_indices[i]),
                        "candidateIds": candidate_ids.tolist(),
                        "predictedIds": local_pred_ids.tolist(),
                        "candidateScores": (
                            scores[start:end].astype(float).tolist()
                            if collect_raw_scores
                            else None
                        ),
                        "targets": (
                            target[start:end].astype(np.uint8).tolist()
                            if collect_raw_scores
                            else None
                        ),
                        "visibleWeights": (
                            weights[start:end].astype(float).tolist()
                            if collect_raw_scores
                            else None
                        ),
                        "metrics": {
                            "precision": float(precision[threshold_index]),
                            "recall": float(recall[threshold_index]),
                            "weightedRecall": float(weighted_recall[threshold_index]),
                            "f1": float(f1[threshold_index]),
                            "jaccard": float(local_tp[threshold_index] / max(1.0, local_tp[threshold_index] + local_fp[threshold_index] + local_fn[threshold_index])),
                            "accuracy": float(accuracy[threshold_index]),
                            "balancedAccuracy": float(0.5 * (recall[threshold_index] + specificity[threshold_index])),
                            "specificity": float(specificity[threshold_index]),
                            "usefulCull": float(local_tn[threshold_index] / max(1.0, end - start)),
                            "badCull": float(local_fn[threshold_index] / max(1.0, end - start)),
                            "avgPredCount": float(pred[:, threshold_index].sum()),
                            "avgCandidateCount": float(end - start),
                            "avgGtCount": float(yy.sum()),
                            "candidateGlbCount": local_candidate_glb_count,
                            "candidateGlbBytes": local_candidate_glb_bytes,
                            "predictedGlbCount": float(local_predicted_glb_counts[threshold_index]),
                            "predictedGlbBytes": float(local_predicted_glb_bytes[threshold_index]),
                            "glbByteReduction": float(
                                1.0 - local_predicted_glb_bytes[threshold_index] / local_candidate_glb_bytes
                                if local_candidate_glb_bytes > 0.0
                                else 0.0
                            ),
                            "missPixelRate": 0.0,
                            "wrongIdPixelRate": 0.0,
                            "extraPixelRate": 0.0,
                        },
                    })
            pred_counts += pred.sum(axis=0).astype(np.float64)
            gt_count_sum += float(yy.sum())
            candidate_count_sum += float(end - start)
            agg_tp += local_tp
            agg_fp += local_fp
            agg_fn += local_fn
            agg_tn += local_tn
            pose_count += 1
    rows = []
    distribution = None
    if collect_score_stats:
        distribution = score_distribution_summary(
            np.concatenate(score_values) if score_values else np.zeros((0,), dtype=np.float64),
            np.concatenate(score_targets) if score_targets else np.zeros((0,), dtype=np.float64),
            np.concatenate(score_weights) if score_weights else np.zeros((0,), dtype=np.float64),
        )
    for i, value in enumerate(th):
        agg_precision = agg_tp[i] / max(1.0, agg_tp[i] + agg_fp[i])
        agg_recall = agg_tp[i] / max(1.0, agg_tp[i] + agg_fn[i])
        agg_specificity = agg_tn[i] / max(1.0, agg_tn[i] + agg_fp[i])
        avg_candidate = float(candidate_count_sum / max(1, pose_count))
        avg_pred = float(pred_counts[i] / max(1, pose_count))
        row = {
            "threshold": float(value),
            "pose_precision": float(pose_precisions[i] / max(1, pose_count)),
            "pose_recall": float(pose_recalls[i] / max(1, pose_count)),
            "pose_f1": float(pose_f1s[i] / max(1, pose_count)),
            "pose_jaccard": float(pose_jaccards[i] / max(1, pose_count)),
            "pose_weighted_recall": float(pose_weighted_recalls[i] / max(1, pose_count)),
            "pose_specificity": float(pose_specificities[i] / max(1, pose_count)),
            "pose_accuracy": float(pose_accuracies[i] / max(1, pose_count)),
            "pose_balanced_accuracy": float(pose_balanced_accuracies[i] / max(1, pose_count)),
            "pose_useful_cull": float(pose_useful_culls[i] / max(1, pose_count)),
            "pose_bad_cull": float(pose_bad_culls[i] / max(1, pose_count)),
            "agg_precision": float(agg_precision),
            "agg_recall": float(agg_recall),
            "agg_weighted_recall": float(agg_weighted_tp[i] / max(1e-12, agg_weighted_gt)),
            "agg_specificity": float(agg_specificity),
            "agg_accuracy": float((agg_tp[i] + agg_tn[i]) / max(1.0, agg_tp[i] + agg_fp[i] + agg_fn[i] + agg_tn[i])),
            "agg_balanced_accuracy": float(0.5 * (agg_recall + agg_specificity)),
            "agg_useful_cull": float(agg_tn[i] / max(1.0, agg_tp[i] + agg_fp[i] + agg_fn[i] + agg_tn[i])),
            "agg_bad_cull": float(agg_fn[i] / max(1.0, agg_tp[i] + agg_fp[i] + agg_fn[i] + agg_tn[i])),
            "agg_f1": float(2.0 * agg_precision * agg_recall / max(1e-8, agg_precision + agg_recall)),
            "avg_pred_count": avg_pred,
            "avg_gt_count": float(gt_count_sum / max(1, pose_count)),
            "avg_candidate_count": avg_candidate,
            "avg_pred_glb_count": float(predicted_glb_counts[i] / max(1, pose_count)),
            "avg_pred_glb_bytes": float(predicted_glb_bytes[i] / max(1, pose_count)),
            "avg_candidate_glb_count": float(candidate_glb_count_sum / max(1, pose_count)),
            "avg_candidate_glb_bytes": float(candidate_glb_bytes_sum / max(1, pose_count)),
            "candidate_reduction_ratio": float(1.0 - avg_pred / max(1.0, avg_candidate)),
            "tp": int(agg_tp[i]),
            "fp": int(agg_fp[i]),
            "fn": int(agg_fn[i]),
            "tn": int(agg_tn[i]),
            "eval_pose_count": int(pose_count),
        }
        aggregate_weighted_recall = float(
            agg_weighted_tp[i] / agg_weighted_gt if agg_weighted_gt > 1e-12 else 1.0
        )
        pose_macro_weighted_recall = float(pose_weighted_recalls[i] / max(1, pose_count))
        aggregate_lcb = None
        pose_macro_lcb = None
        if collect_pose_stats and int(bootstrap_replicates) > 0:
            aggregate_lcb = weighted_recall_lower_confidence_bound(
                np.asarray(pose_weighted_tp_values[i]),
                np.asarray(pose_weighted_gt_values[i]),
                replicates=bootstrap_replicates,
                seed=int(seed) + i,
            )
            pose_macro_lcb = pose_macro_weighted_recall_lower_confidence_bound(
                np.asarray(pose_weighted_recall_values[i]),
                replicates=bootstrap_replicates,
                seed=int(seed) + 100000 + i,
            )
        if collect_pose_stats:
            row["_pose_weighted_recall_values"] = pose_weighted_recall_values[i]
            # A zero-replicate evaluation has only a point estimate.  Do not
            # serialize that estimate under a confidence-bound field.
            row["weighted_recall_lower_confidence_bound"] = (
                None
                if int(bootstrap_replicates) <= 0
                else weighted_recall_lower_confidence_bound(
                    np.asarray(pose_weighted_tp_values[i]),
                    np.asarray(pose_weighted_gt_values[i]),
                    replicates=bootstrap_replicates,
                    seed=int(seed) + i,
                )
            )
        row["aggregateWeightedRecall"] = aggregate_weighted_recall
        row["aggregate_weighted_recall"] = aggregate_weighted_recall
        row["poseMacroWeightedRecall"] = pose_macro_weighted_recall
        row["aggregateWeightedRecallLowerConfidenceBound"] = aggregate_lcb
        row["poseMacroWeightedRecallLowerConfidenceBound"] = pose_macro_lcb
        row["weightedRecallBootstrapReplicates"] = int(max(0, bootstrap_replicates))
        if distribution is not None:
            row["scoreDistribution"] = distribution
        if collect_per_pose:
            row["_per_pose"] = per_pose_values[i]
        rows.append(row)
    return rows
