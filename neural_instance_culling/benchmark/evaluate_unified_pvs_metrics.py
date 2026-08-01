#!/usr/bin/env python3
"""Unified PVS evaluation under visual-safety constraints.

The main ranking is not raw candidate reduction. Raw reduction mixes useful
culls (true negatives) with harmful culls (false negatives). This script reports
both, and only ranks culling efficiency at workpoints that satisfy recall and
weak-utility safety constraints.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_runners import load_runner, load_runtime_meta, selected_default_specs, select_device  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from common.threshold_selection import (
    select_weighted_precision_workpoint,
    weighted_precision_selection_rule,
    weighted_recall_safe_rows,
)  # noqa: E402


def threshold_grid() -> np.ndarray:
    low = np.asarray([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2], dtype=np.float32)
    high = np.linspace(0.22, 0.9, 35, dtype=np.float32)
    return np.unique(np.concatenate([low, high]))


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def load_frozen_thresholds(path: str | Path, model_names: list[str]) -> dict[str, float]:
    """Load one pre-registered threshold per model for a frozen evaluation.

    The file may either be a direct ``{model: threshold}`` mapping or contain
    that mapping under ``thresholds``.  A missing model is an error: silently
    falling back to a default would turn a one-shot test into an unregistered
    threshold choice.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("thresholds", payload) if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        raise ValueError(f"Frozen threshold file {path} must contain a model-to-threshold mapping.")
    result: dict[str, float] = {}
    for name in model_names:
        if name not in values:
            raise ValueError(f"Frozen threshold file {path} has no threshold for model {name!r}.")
        threshold = float(values[name])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Frozen threshold for {name!r} must be in [0, 1], got {threshold}.")
        result[name] = threshold
    return result


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b > 0 else float(default)


def load_glb_byte_costs(
    glb_index: str | Path | None,
    glb_root: str | Path | None,
    num_glbs: int,
    instance_to_glb: np.ndarray,
    allow_missing: bool = False,
) -> np.ndarray:
    costs = np.zeros((num_glbs,), dtype=np.float64)
    if glb_index and Path(glb_index).exists():
        path = Path(glb_index)
        root = Path(glb_root or path.parent)
        obj = json.loads(path.read_text(encoding="utf-8"))
        for entry in obj.get("entries", []):
            gid = int(entry.get("globalId", -1))
            if gid < 0 or gid >= num_glbs:
                continue
            glb_path = root / str(entry.get("path", ""))
            if glb_path.exists() and glb_path.stat().st_size > 0:
                costs[gid] = float(glb_path.stat().st_size)
    used_glbs = np.unique(np.asarray(instance_to_glb, dtype=np.int64))
    missing_used = used_glbs[(used_glbs < 0) | (used_glbs >= num_glbs) | (costs[np.clip(used_glbs, 0, max(0, num_glbs - 1))] <= 0)]
    if missing_used.size and not allow_missing:
        raise FileNotFoundError(
            "Missing byte cost for runtime GLB ids "
            f"{missing_used[:20].tolist()}" + (" ..." if missing_used.size > 20 else "") + ". "
            "Formal evaluation refuses median-size imputation; use --allow-missing-glb-cost only for exploratory runs."
        )
    if not np.any(costs > 0):
        if not allow_missing:
            raise FileNotFoundError("No usable GLB byte costs were found for formal evaluation.")
        counts = np.bincount(np.clip(instance_to_glb.astype(np.int64), 0, num_glbs - 1), minlength=num_glbs).astype(np.float64)
        costs = np.maximum(counts, 1.0)
    else:
        fallback = float(np.median(costs[costs > 0]))
        if allow_missing:
            costs[costs <= 0] = max(1.0, fallback)
    return costs


def glb_budget_rows(
    glbs: np.ndarray,
    rank_scores: np.ndarray,
    utility: np.ndarray,
    glb_byte_cost: np.ndarray,
    budgets: tuple[int, ...],
    target_utility_recall: float,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    unique_glbs, inverse = np.unique(glbs, return_inverse=True)
    if unique_glbs.size == 0:
        empty = {
            b: {
                "utilityRecall": 1.0,
                "requiredRecall": 1.0,
                "precision": 1.0,
                "selectedGlbCount": 0.0,
                "selectedBytes": 0.0,
                "byteReductionVsCandidate": 1.0,
            }
            for b in budgets
        }
        return empty, {
            "selectedGlbCount": 0.0,
            "selectedBytes": 0.0,
            "byteReductionVsCandidate": 1.0,
            "utilityRecall": 1.0,
            "candidateGlbCount": 0.0,
            "candidateBytes": 0.0,
        }

    glb_scores = np.full((unique_glbs.size,), -1e9, dtype=np.float64)
    np.maximum.at(glb_scores, inverse, rank_scores.astype(np.float64, copy=False))
    glb_utility = np.zeros((unique_glbs.size,), dtype=np.float64)
    np.add.at(glb_utility, inverse, utility.astype(np.float64, copy=False))
    required = glb_utility > 0.0
    costs = glb_byte_cost[unique_glbs]
    # Stable, pre-registered tie break: descending score, then ascending GLB id.
    order = np.lexsort((unique_glbs.astype(np.int64, copy=False), -glb_scores))
    total_utility = max(1e-9, float(glb_utility.sum()))
    required_count = max(1, int(required.sum()))
    candidate_bytes = max(1.0, float(costs.sum()))

    rows: dict[int, dict[str, float]] = {}
    for budget in budgets:
        top = order[: min(int(budget), order.size)]
        selected_utility = float(glb_utility[top].sum())
        selected_required = int(required[top].sum())
        selected_bytes = float(costs[top].sum())
        rows[int(budget)] = {
            "utilityRecall": selected_utility / total_utility,
            "requiredRecall": selected_required / required_count,
            "precision": selected_required / max(1, int(top.size)),
            "selectedGlbCount": float(top.size),
            "selectedBytes": selected_bytes,
            "byteReductionVsCandidate": 1.0 - selected_bytes / candidate_bytes,
        }

    cumulative_utility = np.cumsum(glb_utility[order])
    hit = np.flatnonzero(cumulative_utility >= float(target_utility_recall) * total_utility)
    prefix_count = int(hit[0] + 1) if hit.size else int(order.size)
    prefix = order[:prefix_count]
    selected_bytes = float(costs[prefix].sum()) if prefix.size else 0.0
    prefix_row = {
        "selectedGlbCount": float(prefix_count),
        "selectedBytes": selected_bytes,
        "byteReductionVsCandidate": 1.0 - selected_bytes / candidate_bytes,
        "utilityRecall": float(cumulative_utility[prefix_count - 1] / total_utility) if prefix_count > 0 else 1.0,
        "candidateGlbCount": float(unique_glbs.size),
        "candidateBytes": candidate_bytes,
    }
    return rows, prefix_row


def select_workpoints(
    rows: list[dict[str, Any]],
    target_recall: float,
    target_weighted_recall: float,
    target_utility_recall: float,
) -> dict[str, Any]:
    if not rows:
        return {}
    weighted_safe = weighted_recall_safe_rows(rows, target_weighted_recall)
    utility_safe = [
        r
        for r in weighted_safe
        if r["pose_visual_utility_recall"] >= target_utility_recall
    ]
    any_high_recall = [r for r in rows if r["pose_recall"] >= target_recall]
    return {
        "primaryWeightedPrecision": select_weighted_precision_workpoint(rows, target_weighted_recall),
        "primarySetSafeUsefulCull": max(weighted_safe, key=lambda r: (r["pose_useful_cull_candidate_ratio"], -r["avg_pred_count"])) if weighted_safe else None,
        "primaryUtilitySafeUsefulCull": max(utility_safe, key=lambda r: (r["pose_useful_cull_candidate_ratio"], -r["avg_pred_glb_bytes"])) if utility_safe else None,
        "bestPrecisionAtRecall": max(any_high_recall, key=lambda r: r["pose_precision"]) if any_high_recall else None,
        "bestBalancedAccuracy": max(rows, key=lambda r: r["pose_balanced_accuracy"]),
        "bestAccuracy": max(rows, key=lambda r: r["pose_accuracy"]),
        "bestF1": max(rows, key=lambda r: r["pose_f1"]),
        "bestSafetyAdjustedCullScore": max(weighted_safe, key=lambda r: r["safety_adjusted_cull_score"]) if weighted_safe else max(rows, key=lambda r: r["safety_adjusted_cull_score"]),
    }


def evaluate_runner(
    runner,
    split,
    thresholds: np.ndarray,
    budgets: tuple[int, ...],
    glb_byte_cost: np.ndarray,
    poses_per_batch: int,
    max_eval_poses: int,
    max_candidates_per_pose: int,
    seed: int,
    target_recall: float,
    target_weighted_recall: float,
    target_utility_recall: float,
    sample_with_replacement: bool,
    allow_candidate_visible_union: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if sample_with_replacement:
        steps = max(1, int(np.ceil(max(1, max_eval_poses) / max(1, poses_per_batch))))
        total_steps = steps
    else:
        steps = None
        total_steps = int(np.ceil(max(1, split.pose_indices.size) / max(1, poses_per_batch)))

    n_th = int(thresholds.size)
    agg_tp = np.zeros((n_th,), dtype=np.float64)
    agg_fp = np.zeros((n_th,), dtype=np.float64)
    agg_fn = np.zeros((n_th,), dtype=np.float64)
    agg_tn = np.zeros((n_th,), dtype=np.float64)
    pose_precision = np.zeros((n_th,), dtype=np.float64)
    pose_recall = np.zeros((n_th,), dtype=np.float64)
    pose_specificity = np.zeros((n_th,), dtype=np.float64)
    pose_accuracy = np.zeros((n_th,), dtype=np.float64)
    pose_balanced_accuracy = np.zeros((n_th,), dtype=np.float64)
    pose_f1 = np.zeros((n_th,), dtype=np.float64)
    pose_jaccard = np.zeros((n_th,), dtype=np.float64)
    pose_weighted_recall = np.zeros((n_th,), dtype=np.float64)
    pose_utility_recall = np.zeros((n_th,), dtype=np.float64)
    pose_pred_count = np.zeros((n_th,), dtype=np.float64)
    pose_pred_glb_count = np.zeros((n_th,), dtype=np.float64)
    pose_pred_bytes = np.zeros((n_th,), dtype=np.float64)
    pose_useful_cull = np.zeros((n_th,), dtype=np.float64)
    pose_bad_cull = np.zeros((n_th,), dtype=np.float64)
    pose_negative_fpr = np.zeros((n_th,), dtype=np.float64)
    pose_candidate_byte_reduction = np.zeros((n_th,), dtype=np.float64)

    budget_acc = {
        b: {"utilityRecall": 0.0, "requiredRecall": 0.0, "precision": 0.0, "selectedGlbCount": 0.0, "selectedBytes": 0.0, "byteReductionVsCandidate": 0.0}
        for b in budgets
    }
    prefix_acc = {"selectedGlbCount": 0.0, "selectedBytes": 0.0, "byteReductionVsCandidate": 0.0, "utilityRecall": 0.0, "candidateGlbCount": 0.0, "candidateBytes": 0.0}
    gt_count_sum = 0.0
    candidate_count_sum = 0.0
    candidate_glb_count_sum = 0.0
    candidate_bytes_sum = 0.0
    total_utility_sum = 0.0
    pose_count = 0
    empty_gt_pose_count = 0
    empty_candidate_pose_count = 0
    forward_ms: list[float] = []
    total_ms: list[float] = []
    diagnostics: dict[str, list[float]] = {}

    for pose_indices in tqdm(
        split.pose_set_batches(poses_per_batch, rng, steps, include_empty=True),
        total=total_steps,
        desc=runner.name,
        ascii=True,
    ):
        batch = split.build_pose_set_batch(
            pose_indices,
            runner.world_aabbs,
            rng,
            max_candidates_per_pose=max_candidates_per_pose,
            allow_candidate_visible_union=allow_candidate_visible_union,
            include_empty=True,
        )

        # Empty candidate rows are valid only for empty-GT poses. Count them
        # explicitly instead of dropping them from a nominally complete split.
        batch_offsets = np.asarray(batch["pose_offsets"], dtype=np.int64)
        batch_candidate_counts = np.asarray(batch.get("candidate_counts", []), dtype=np.int64)
        batch_visible_counts = np.asarray(batch.get("visible_counts", []), dtype=np.int64)
        empty_rows = np.flatnonzero(np.diff(batch_offsets) == 0)
        for row_index in empty_rows.tolist():
            if row_index < batch_visible_counts.size and int(batch_visible_counts[row_index]) != 0:
                raise ValueError(
                    f"Stored candidate semantics are invalid at empty-candidate pose row {row_index}: "
                    "the pose has visible GT instances but no candidate references."
                )
            empty_gt_pose_count += 1
            empty_candidate_pose_count += 1
            pose_accuracy += 1.0
            pose_balanced_accuracy += 1.0
            pose_precision += 1.0
            pose_recall += 1.0
            pose_specificity += 1.0
            pose_f1 += 1.0
            pose_jaccard += 1.0
            pose_weighted_recall += 1.0
            pose_utility_recall += 1.0
            pose_count += 1
        if batch["instance"].size == 0:
            continue
        result = runner.score_batch(batch)
        forward_ms.append(float(result.forward_ms))
        total_ms.append(float(result.total_ms))
        if result.diagnostics:
            for key, value in result.diagnostics.items():
                try:
                    diagnostics.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    pass

        rank_scores_all = result.download_scores if getattr(result, "download_scores", None) is not None else result.scores
        ids_all = batch["instance"].astype(np.int64, copy=False)
        target_all = batch["target"].astype(bool, copy=False)
        weights_all = batch.get("visible_weights", np.zeros_like(batch["target"], dtype=np.float32)).astype(np.float64, copy=False)
        utility_all = np.where(target_all, np.log1p(np.maximum(weights_all, 0.0)), 0.0)
        offsets = batch["pose_offsets"]
        glb_all = np.clip(runner.instance_to_glb[ids_all], 0, glb_byte_cost.size - 1)

        for pose_i in range(offsets.size - 1):
            start = int(offsets[pose_i])
            end = int(offsets[pose_i + 1])
            if end <= start:
                continue
            scores = result.scores[start:end].astype(np.float64, copy=False)
            rank_scores = rank_scores_all[start:end].astype(np.float64, copy=False)
            truth = target_all[start:end]
            weights = weights_all[start:end]
            utility = utility_all[start:end]
            glbs = glb_all[start:end]
            candidate_count = float(end - start)
            gt_count = float(truth.sum())
            neg_count = max(0.0, candidate_count - gt_count)
            total_utility = max(1e-9, float(utility.sum()))
            candidate_glbs = np.unique(glbs)
            candidate_bytes = max(1.0, float(glb_byte_cost[candidate_glbs].sum())) if candidate_glbs.size else 1.0

            pred = scores[:, None] >= thresholds[None, :]
            truth_col = truth[:, None]
            local_tp = np.logical_and(pred, truth_col).sum(axis=0).astype(np.float64)
            local_fp = np.logical_and(pred, ~truth_col).sum(axis=0).astype(np.float64)
            local_fn = np.logical_and(~pred, truth_col).sum(axis=0).astype(np.float64)
            local_tn = np.logical_and(~pred, ~truth_col).sum(axis=0).astype(np.float64)
            pred_counts = pred.sum(axis=0).astype(np.float64)

            weighted_tp = (np.logical_and(pred, truth_col) * weights[:, None]).sum(axis=0)
            weighted_gt = max(1.0, float((truth * weights).sum()))
            hit_utility = (np.logical_and(pred, truth_col) * utility[:, None]).sum(axis=0)

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
            accuracy = (local_tp + local_tn) / np.maximum(1.0, candidate_count)
            balanced_accuracy = 0.5 * (recall + specificity)
            f1 = 2.0 * precision * recall / np.maximum(1e-8, precision + recall)
            jaccard = np.divide(
                local_tp,
                local_tp + local_fp + local_fn,
                out=np.ones_like(local_tp),
                where=(local_tp + local_fp + local_fn) > 0.0,
            )
            useful_cull = local_tn / np.maximum(1.0, candidate_count)
            bad_cull = local_fn / np.maximum(1.0, candidate_count)
            negative_fpr = local_fp / np.maximum(1.0, neg_count)
            weighted_recall = np.divide(
                weighted_tp,
                weighted_gt,
                out=np.ones_like(weighted_tp),
                where=weighted_gt > 0.0,
            )
            util_recall = hit_utility / total_utility if total_utility > 0.0 else np.ones_like(hit_utility)

            for th_i in range(n_th):
                pred_mask = pred[:, th_i]
                pred_glbs = np.unique(glbs[pred_mask])
                pred_bytes = float(glb_byte_cost[pred_glbs].sum()) if pred_glbs.size else 0.0
                pose_pred_glb_count[th_i] += float(pred_glbs.size)
                pose_pred_bytes[th_i] += pred_bytes
                pose_candidate_byte_reduction[th_i] += 1.0 - pred_bytes / candidate_bytes

            agg_tp += local_tp
            agg_fp += local_fp
            agg_fn += local_fn
            agg_tn += local_tn
            pose_precision += precision
            pose_recall += recall
            pose_specificity += specificity
            pose_accuracy += accuracy
            pose_balanced_accuracy += balanced_accuracy
            pose_f1 += f1
            pose_jaccard += jaccard
            pose_weighted_recall += weighted_recall
            pose_utility_recall += util_recall
            pose_pred_count += pred_counts
            pose_useful_cull += useful_cull
            pose_bad_cull += bad_cull
            pose_negative_fpr += negative_fpr
            gt_count_sum += gt_count
            candidate_count_sum += candidate_count
            candidate_glb_count_sum += float(candidate_glbs.size)
            candidate_bytes_sum += candidate_bytes
            total_utility_sum += float(utility.sum())

            budget_rows, prefix_row = glb_budget_rows(glbs, rank_scores, utility, glb_byte_cost, budgets, target_utility_recall)
            for b, values in budget_rows.items():
                for key, value in values.items():
                    budget_acc[b][key] += float(value)
            for key, value in prefix_row.items():
                prefix_acc[key] += float(value)
            pose_count += 1

    rows: list[dict[str, Any]] = []
    avg_gt = gt_count_sum / max(1, pose_count)
    avg_candidate = candidate_count_sum / max(1, pose_count)
    avg_candidate_glb = candidate_glb_count_sum / max(1, pose_count)
    avg_candidate_bytes = candidate_bytes_sum / max(1, pose_count)
    for i, threshold in enumerate(thresholds):
        agg_precision = safe_div(agg_tp[i], agg_tp[i] + agg_fp[i])
        agg_recall = safe_div(agg_tp[i], agg_tp[i] + agg_fn[i])
        agg_specificity = safe_div(agg_tn[i], agg_tn[i] + agg_fp[i], default=1.0)
        agg_accuracy = safe_div(agg_tp[i] + agg_tn[i], agg_tp[i] + agg_fp[i] + agg_fn[i] + agg_tn[i])
        avg_pred = pose_pred_count[i] / max(1, pose_count)
        avg_fp = agg_fp[i] / max(1, pose_count)
        avg_fn = agg_fn[i] / max(1, pose_count)
        avg_tn = agg_tn[i] / max(1, pose_count)
        raw_candidate_reduction = 1.0 - avg_pred / max(1.0, avg_candidate)
        useful_cull_ratio = pose_useful_cull[i] / max(1, pose_count)
        bad_cull_ratio = pose_bad_cull[i] / max(1, pose_count)
        pose_rec = pose_recall[i] / max(1, pose_count)
        pose_weighted = pose_weighted_recall[i] / max(1, pose_count)
        safety_multiplier = min(1.0, pose_rec / max(1e-8, target_recall)) * min(1.0, pose_weighted / max(1e-8, target_weighted_recall))
        rows.append(
            {
                "threshold": float(threshold),
                "pose_accuracy": float(pose_accuracy[i] / max(1, pose_count)),
                "pose_balanced_accuracy": float(pose_balanced_accuracy[i] / max(1, pose_count)),
                "pose_precision": float(pose_precision[i] / max(1, pose_count)),
                "pose_recall": float(pose_rec),
                "pose_specificity": float(pose_specificity[i] / max(1, pose_count)),
                "pose_f1": float(pose_f1[i] / max(1, pose_count)),
                "pose_jaccard": float(pose_jaccard[i] / max(1, pose_count)),
                "pose_weighted_recall": float(pose_weighted),
                "pose_visual_utility_recall": float(pose_utility_recall[i] / max(1, pose_count)),
                "pose_miss_visual_utility_rate": float(1.0 - pose_utility_recall[i] / max(1, pose_count)),
                "pose_negative_fpr": float(pose_negative_fpr[i] / max(1, pose_count)),
                "pose_raw_candidate_reduction_ratio": float(raw_candidate_reduction),
                "pose_useful_cull_candidate_ratio": float(useful_cull_ratio),
                "pose_bad_cull_candidate_ratio": float(bad_cull_ratio),
                "safety_adjusted_cull_score": float(useful_cull_ratio * safety_multiplier),
                "agg_accuracy": float(agg_accuracy),
                "agg_balanced_accuracy": float(0.5 * (agg_recall + agg_specificity)),
                "agg_precision": float(agg_precision),
                "agg_recall": float(agg_recall),
                "agg_specificity": float(agg_specificity),
                "agg_f1": float(2.0 * agg_precision * agg_recall / max(1e-8, agg_precision + agg_recall)),
                "avg_candidate_count": float(avg_candidate),
                "avg_gt_count": float(avg_gt),
                "avg_pred_count": float(avg_pred),
                "avg_fp_count": float(avg_fp),
                "avg_fn_count": float(avg_fn),
                "avg_tn_count": float(avg_tn),
                "overfetch_factor": float(avg_pred / max(1.0, avg_gt)),
                "extra_pred_per_gt": float(avg_fp / max(1.0, avg_gt)),
                "avg_candidate_glb_count": float(avg_candidate_glb),
                "avg_candidate_glb_bytes": float(avg_candidate_bytes),
                "avg_pred_glb_count": float(pose_pred_glb_count[i] / max(1, pose_count)),
                "avg_pred_glb_bytes": float(pose_pred_bytes[i] / max(1, pose_count)),
                "pose_candidate_byte_reduction_ratio": float(pose_candidate_byte_reduction[i] / max(1, pose_count)),
                "tp": int(agg_tp[i]),
                "fp": int(agg_fp[i]),
                "fn": int(agg_fn[i]),
                "tn": int(agg_tn[i]),
                "eval_pose_count": int(pose_count),
            }
        )

    workpoints = select_workpoints(rows, target_recall, target_weighted_recall, target_utility_recall)
    budget_summary = {
        str(b): {key: float(value / max(1, pose_count)) for key, value in values.items()}
        for b, values in budget_acc.items()
    }
    prefix_summary = {key: float(value / max(1, pose_count)) for key, value in prefix_acc.items()}
    runner_info: dict[str, Any] = {
        "informationLevel": str(getattr(runner, "information_level", "unspecified")),
        "decisionMode": str(getattr(runner, "decision_mode", "threshold")),
        "displayName": str(getattr(runner, "display_name", runner.name)),
    }
    if hasattr(runner, "requires_mvp"):
        runner_info["requiresMvp"] = bool(getattr(runner, "requires_mvp"))
    for attribute, output_key in (
        ("resource_assumption", "resourceAssumption"),
        ("score_formula", "scoreFormula"),
    ):
        if hasattr(runner, attribute):
            runner_info[output_key] = str(getattr(runner, attribute))
    for attribute in ("train_pose_count", "bitset_bytes"):
        if hasattr(runner, attribute):
            runner_info[attribute] = int(getattr(runner, attribute))
    if hasattr(runner, "direction_penalty_m2"):
        runner_info["directionPenaltyM2"] = float(getattr(runner, "direction_penalty_m2"))
    return {
        "name": runner.name,
        "kind": runner.kind,
        "runnerInfo": runner_info,
        "best": workpoints.get("primaryWeightedPrecision"),
        "diagnosticBestSafetyAdjustedCull": workpoints.get("bestSafetyAdjustedCullScore"),
        "workpoints": workpoints,
        "thresholdRows": rows,
        "budgetedGlbUtility": budget_summary,
        "bytePrefixAtUtilityTarget": prefix_summary,
        "totalWeakUtility": float(total_utility_sum),
        "avgForwardMsPerBatch": float(np.mean(forward_ms)) if forward_ms else 0.0,
        "avgTotalMsPerBatch": float(np.mean(total_ms)) if total_ms else 0.0,
        "emptyGtPoseCount": int(empty_gt_pose_count),
        "emptyCandidatePoseCount": int(empty_candidate_pose_count),
        "runnerDiagnostics": {key: float(np.mean(values)) for key, values in diagnostics.items() if values},
    }


def fmt(row: dict[str, Any] | None, key: str, digits: int = 3, default: str = "-") -> str:
    if not row or row.get(key) is None:
        return default
    value = float(row[key])
    return f"{value:.{digits}f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    lines = [
        "# Unified PVS Metrics",
        "",
        f"- Created: `{meta['created']}`",
        f"- Dataset: `{meta['datasetDir']}`",
        f"- Split: `{meta['split']}`",
        f"- Eval mode: `{meta['evalMode']}`",
        f"- Threshold mode: `{meta['thresholdMode']}`",
        f"- Candidate semantics: `{meta['candidateSemanticsMode']}`",
        f"- Evaluated poses: `{meta['evaluatedPoses']}`",
        f"- Max candidates per pose: `{meta['maxCandidatesPerPose']}`",
        f"- Primary threshold rule: `weighted recall > {meta['targetWeightedRecall']}`; choose highest pose precision",
        f"- Pose recall target is reported as a diagnostic: `pose recall >= {meta['targetRecall']}`",
        f"- Utility target: `weak utility recall >= {meta['targetUtilityRecall']}`",
        "",
        "说明：`raw reduction` 会把正确剔除的不可见实例和错误剔除的可见实例混在一起；主表使用 `useful cull = TN / candidate`，并报告 `bad cull = FN / candidate`。逐实例 accuracy 作为辅助准确性指标，不能单独代表剔除质量。",
        "",
        "## Weighted-Safe Primary Workpoint",
        "",
        "| Model | Threshold | Acc | Bal Acc | Recall | Weighted Recall | Utility Recall | Useful Cull | Bad Cull | Raw Reduction | Avg Candidate | Avg GT | Avg Pred | Overfetch | Byte Reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in payload["summaries"]:
        row = (summary.get("workpoints") or {}).get("primaryWeightedPrecision")
        lines.append(
            f"| {summary['name']} | {fmt(row, 'threshold')} | {fmt(row, 'pose_accuracy')} | {fmt(row, 'pose_balanced_accuracy')} | "
            f"{fmt(row, 'pose_recall')} | {fmt(row, 'pose_weighted_recall')} | {fmt(row, 'pose_visual_utility_recall')} | "
            f"{fmt(row, 'pose_useful_cull_candidate_ratio')} | {fmt(row, 'pose_bad_cull_candidate_ratio', 5)} | "
            f"{fmt(row, 'pose_raw_candidate_reduction_ratio')} | {fmt(row, 'avg_candidate_count', 2)} | "
            f"{fmt(row, 'avg_gt_count', 2)} | {fmt(row, 'avg_pred_count', 2)} | {fmt(row, 'overfetch_factor', 2)} | "
            f"{fmt(row, 'pose_candidate_byte_reduction_ratio')} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostic Workpoints",
            "",
            "| Model | Workpoint | Threshold | Acc | Bal Acc | Precision | Recall | Specificity | Useful Cull | Bad Cull | Avg Pred | Avg FP | Avg FN |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in payload["summaries"]:
        for key, label in [
            ("primaryWeightedPrecision", "weighted_safe_precision"),
            ("primarySetSafeUsefulCull", "set_safe_useful_cull"),
            ("primaryUtilitySafeUsefulCull", "utility_safe_useful_cull"),
            ("bestPrecisionAtRecall", "best_precision_at_recall"),
            ("bestBalancedAccuracy", "best_balanced_accuracy"),
            ("bestAccuracy", "best_accuracy"),
            ("bestF1", "best_f1"),
        ]:
            row = (summary.get("workpoints") or {}).get(key)
            if not row:
                continue
            lines.append(
                f"| {summary['name']} | {label} | {fmt(row, 'threshold')} | {fmt(row, 'pose_accuracy')} | "
                f"{fmt(row, 'pose_balanced_accuracy')} | {fmt(row, 'pose_precision')} | {fmt(row, 'pose_recall')} | "
                f"{fmt(row, 'pose_specificity')} | {fmt(row, 'pose_useful_cull_candidate_ratio')} | "
                f"{fmt(row, 'pose_bad_cull_candidate_ratio', 5)} | {fmt(row, 'avg_pred_count', 2)} | "
                f"{fmt(row, 'avg_fp_count', 2)} | {fmt(row, 'avg_fn_count', 2)} |"
            )

    lines.extend(
        [
            "",
            "## Budgeted GLB Utility",
            "",
            "| Model | Budget | Utility Recall | Required Recall | Precision | Avg Selected GLB | Avg Selected Bytes | Byte Reduction vs Candidate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in payload["summaries"]:
        for budget, row in (summary.get("budgetedGlbUtility") or {}).items():
            lines.append(
                f"| {summary['name']} | {budget} | {row['utilityRecall']:.3f} | {row['requiredRecall']:.3f} | "
                f"{row['precision']:.3f} | {row['selectedGlbCount']:.2f} | {row['selectedBytes']:.0f} | "
                f"{row['byteReductionVsCandidate']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Byte Prefix At Utility Target",
            "",
            "| Model | Target Utility Recall | Avg GLB Count | Avg Bytes | Byte Reduction vs Candidate | Achieved Utility Recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in payload["summaries"]:
        row = summary.get("bytePrefixAtUtilityTarget") or {}
        lines.append(
            f"| {summary['name']} | {meta['targetUtilityRecall']:.3f} | {row.get('selectedGlbCount', 0.0):.2f} | "
            f"{row.get('selectedBytes', 0.0):.0f} | {row.get('byteReductionVsCandidate', 0.0):.3f} | "
            f"{row.get('utilityRecall', 0.0):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PVS accuracy, visual safety, useful culling, and GLB utility on a CSR split.")
    parser.add_argument("--models", default="baseline_aabb_hzb,baseline_aabb_ray,pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66"))
    parser.add_argument("--runtime-meta", default="hkust-v3/assets/runtimeVisibilityMeta.json")
    parser.add_argument("--glb-index", default="hkust-v3/assets/glbIndex.json")
    parser.add_argument("--glb-root", default="hkust-v3/assets")
    parser.add_argument(
        "--triangle-hzb-cache",
        default="",
        help="Explicit .bin cache for baseline_triangle_hzb; required when that model is selected.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark/out/unified_pvs_metrics"))
    parser.add_argument("--split", choices=["train", "val", "validation", "calibration", "test"], default="test")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--target-weighted-recall", type=float, default=0.99)
    parser.add_argument("--target-utility-recall", type=float, default=0.98)
    parser.add_argument("--budgets", default="50,100,200,384")
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--max-candidates-per-pose", type=int, default=0)
    parser.add_argument("--max-eval-poses", type=int, default=0)
    parser.add_argument("--sample-with-replacement", action="store_true")
    parser.add_argument(
        "--frozen-threshold-file",
        default="",
        help="JSON file containing one pre-registered threshold per model; required for formal test evaluation.",
    )
    parser.add_argument(
        "--exploratory-test-threshold-scan",
        action="store_true",
        help="Explicitly allow the historical test threshold scan. Its output is exploratory and must not be used as a formal test result.",
    )
    parser.add_argument(
        "--allow-candidate-visible-union",
        action="store_true",
        help="Exploratory-only legacy candidate repair. Formal evaluation must use the stored raw candidate set unchanged.",
    )
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--allow-missing-glb-cost",
        action="store_true",
        help="Exploratory-only median/count imputation for missing GLB files; formal evaluation must leave this disabled.",
    )
    args = parser.parse_args()
    if args.sample_with_replacement and int(args.max_eval_poses) <= 0:
        parser.error("--sample-with-replacement requires --max-eval-poses > 0")
    if args.split == "test" and not args.exploratory_test_threshold_scan:
        if not args.frozen_threshold_file:
            parser.error(
                "Formal test evaluation requires --frozen-threshold-file. "
                "Use --exploratory-test-threshold-scan only for explicitly labelled exploratory results."
            )
        if args.sample_with_replacement or int(args.max_eval_poses) > 0:
            parser.error("Formal test evaluation must traverse all unique test poses without replacement.")
    if args.allow_candidate_visible_union and args.split == "test" and not args.exploratory_test_threshold_scan:
        parser.error("--allow-candidate-visible-union cannot be used in a formal test evaluation.")

    device = select_device(args.device)
    specs = selected_default_specs(args.models)
    if "baseline_triangle_hzb" in specs:
        if not args.triangle_hzb_cache:
            parser.error("--triangle-hzb-cache is required when --models includes baseline_triangle_hzb")
        specs["baseline_triangle_hzb"] = {
            **specs["baseline_triangle_hzb"],
            "cache": str(Path(args.triangle_hzb_cache).resolve()),
        }
    elif args.triangle_hzb_cache:
        parser.error("--triangle-hzb-cache is only valid with --models baseline_triangle_hzb")
    first = next(iter(specs.values()))
    if first.get("checkpoint"):
        checkpoint = torch.load(first["checkpoint"], map_location="cpu")
        num_instances = int(checkpoint["config"]["numInstances"])
    else:
        world_aabbs, _instance_to_glb, _runtime = load_runtime_meta(args.runtime_meta)
        num_instances = int(world_aabbs.shape[0])
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=num_instances)
    requested_split = args.split
    if requested_split == "val" and "validation" in dataset.split_ids:
        requested_split = "validation"
    split = dataset.split(requested_split)
    budgets = parse_int_list(args.budgets)
    model_names = list(specs.keys())
    frozen_thresholds = (
        load_frozen_thresholds(args.frozen_threshold_file, model_names)
        if args.frozen_threshold_file
        else {}
    )
    scan_thresholds = threshold_grid() if not frozen_thresholds else None

    summaries = []
    for name, spec in specs.items():
        runner = load_runner(name, spec, args.runtime_meta, device, dataset_dir=args.dataset_dir)
        num_glbs = int(np.max(runner.instance_to_glb)) + 1
        glb_byte_cost = load_glb_byte_costs(
            args.glb_index,
            args.glb_root,
            num_glbs,
            runner.instance_to_glb,
            allow_missing=args.allow_missing_glb_cost,
        )
        thresholds = (
            np.asarray([frozen_thresholds[name]], dtype=np.float32)
            if name in frozen_thresholds
            else np.asarray([float(runner.threshold)], dtype=np.float32)
            if str(getattr(runner, "decision_mode", "threshold")).startswith("fixed_")
            else scan_thresholds
        )
        summaries.append(
            evaluate_runner(
                runner,
                split,
                thresholds,
                budgets,
                glb_byte_cost,
                poses_per_batch=args.poses_per_batch,
                max_eval_poses=args.max_eval_poses,
                max_candidates_per_pose=args.max_candidates_per_pose,
                seed=args.seed,
                target_recall=args.target_recall,
                target_weighted_recall=args.target_weighted_recall,
                target_utility_recall=args.target_utility_recall,
                sample_with_replacement=args.sample_with_replacement,
                allow_candidate_visible_union=args.allow_candidate_visible_union,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_summary = summaries[0] if summaries else {}
    first_row = first_summary.get("best") or ((first_summary.get("thresholdRows") or [None])[0] or {})
    evaluated = int(first_row.get("eval_pose_count", 0))
    payload = {
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "datasetDir": args.dataset_dir,
            "runtimeMeta": args.runtime_meta,
            "split": requested_split,
            "requestedSplit": args.split,
            "evalMode": "sampled_with_replacement" if args.sample_with_replacement else "all_unique_visible_poses",
            "thresholdMode": "exploratory_test_scan" if args.exploratory_test_threshold_scan else ("frozen_one_shot" if frozen_thresholds else "split_threshold_scan"),
            "frozenThresholdFile": args.frozen_threshold_file or None,
            "testEvaluationCount": 1 if args.split == "test" and not args.exploratory_test_threshold_scan else None,
            "candidateSemanticsMode": "exploratory_visible_union" if args.allow_candidate_visible_union else "stored_candidate_set_strict",
            "glbByteCostMode": "imputed_exploratory" if args.allow_missing_glb_cost else "strict_filesystem_bytes",
            "evaluatedPoses": evaluated,
            "maxCandidatesPerPose": int(args.max_candidates_per_pose),
            "device": str(device),
            "targetRecall": float(args.target_recall),
            "targetWeightedRecall": float(args.target_weighted_recall),
            "thresholdSelectionRule": weighted_precision_selection_rule(args.target_weighted_recall),
            "targetUtilityRecall": float(args.target_utility_recall),
            "budgets": list(budgets),
            "utilityTeacher": "weak_log1p_visible_weights_not_pixel_coverage",
            "primaryMetric": "maximize pose precision only where weighted recall is strictly above the safety target",
        },
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_dir / "summary.md", payload)
    print(json.dumps({"outputDir": str(output_dir), "evaluatedPoses": evaluated, "models": [s["name"] for s in summaries]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
