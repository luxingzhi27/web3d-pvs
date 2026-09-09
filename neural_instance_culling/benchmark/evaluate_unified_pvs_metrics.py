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
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_runners import DEFAULT_MODEL_SPECS, load_runner, load_runtime_meta, selected_default_specs, select_device  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from common.threshold_selection import (
    aggregate_weighted_recall_safe_rows,
    weighted_precision_selection_rule,
)  # noqa: E402

from pvs_threshold_metrics import weighted_recall_lower_confidence_bound  # noqa: E402

try:
    from .score_sidecar import ScoreSidecarWriter, average_precision  # noqa: E402
except ImportError:  # Direct script execution.
    from score_sidecar import ScoreSidecarWriter, average_precision  # noqa: E402


DEFAULT_DATA_ROOT = Path("/mnt/sda/rhyang/slm")
DEFAULT_MODEL_NAMES = frozenset(DEFAULT_MODEL_SPECS)


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
    if isinstance(payload, dict) and payload.get("testRead") is True:
        raise ValueError("frozen threshold registration must be created before test is read")
    result: dict[str, float] = {}
    for name in model_names:
        if name not in values:
            raise ValueError(f"Frozen threshold file {path} has no threshold for model {name!r}.")
        registered = values[name]
        if isinstance(registered, dict):
            if registered.get("selectionSplit") not in (None, "calibration"):
                raise ValueError(f"Frozen threshold for {name!r} was not selected on calibration.")
            threshold = float(registered.get("threshold", np.nan))
        else:
            threshold = float(registered)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Frozen threshold for {name!r} must be in [0, 1], got {threshold}.")
        result[name] = threshold
    return result


_FORMAL_TEST_FIXED_KINDS = frozenset(
    {
        "keep_all",
        "static_frequency_train",
        "camera_distance",
        "projected_aabb_area",
        "aabb_ray",
        "viewcell_bitset_train",
    }
)
_FORMAL_TEST_LEARNED_KINDS = frozenset(
    {"learned_aabb_ray", "bounded_relation_survival_moment_v4"}
)


def validate_formal_test_specs(
    specs: Mapping[str, Mapping[str, str]],
    frozen_threshold_file: str | Path,
    frozen_thresholds: Mapping[str, float],
) -> None:
    """Require each learned test runner to name its frozen calibration inputs."""
    payload = json.loads(Path(frozen_threshold_file).read_text(encoding="utf-8"))
    values = payload.get("thresholds", payload) if isinstance(payload, Mapping) else None
    if not isinstance(values, Mapping):
        raise ValueError("formal test threshold registration must contain thresholds")
    for name, spec in specs.items():
        kind = str(spec.get("kind", ""))
        registered = values.get(name)
        if registered is None:
            raise ValueError(f"formal test threshold registration is missing model {name!r}")
        if kind in _FORMAL_TEST_FIXED_KINDS:
            if isinstance(registered, Mapping) and registered.get("testRead") is not False:
                raise ValueError(f"formal test model {name!r} threshold registration is test-tainted")
            continue
        if kind not in _FORMAL_TEST_LEARNED_KINDS:
            raise ValueError(f"formal test runner kind is not registered: {kind!r}")
        checkpoint_value = spec.get("checkpoint")
        calibration_value = spec.get("calibration") or spec.get("eval_summary")
        if not checkpoint_value or not calibration_value:
            raise ValueError(
                f"formal test model {name!r} requires explicit checkpoint and calibration"
            )
        checkpoint_path = Path(checkpoint_value).resolve()
        calibration_path = Path(calibration_value).resolve()
        if not checkpoint_path.is_file() or not calibration_path.is_file():
            raise FileNotFoundError(
                f"formal test model {name!r} checkpoint/calibration is missing"
            )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not isinstance(calibration, Mapping) or calibration.get("testRead") is not False:
            raise ValueError(f"formal test model {name!r} calibration is not test-free")
        test_evaluation_count = calibration.get("testEvaluationCount", 0)
        if test_evaluation_count is None:
            test_evaluation_count = 0
        if int(test_evaluation_count) != 0:
            raise ValueError(f"formal test model {name!r} calibration already read test")
        if calibration.get("status") != "safe":
            raise ValueError(f"formal test model {name!r} has no safe calibration workpoint")
        best = calibration.get("bestSafe")
        selection = best.get("selection", best) if isinstance(best, Mapping) else None
        if not isinstance(selection, Mapping):
            raise ValueError(f"formal test model {name!r} has no frozen calibration threshold")
        selected_threshold = float(selection.get("threshold", np.nan))
        if not np.isfinite(selected_threshold) or not 0.0 <= selected_threshold <= 1.0:
            raise ValueError(f"formal test model {name!r} calibration threshold is invalid")
        if not isinstance(registered, Mapping):
            raise ValueError(
                f"formal test model {name!r} must register threshold provenance from calibration"
            )
        if registered.get("selectionSplit") != "calibration":
            raise ValueError(f"formal test model {name!r} threshold was not selected on calibration")
        if registered.get("testRead") is not False:
            raise ValueError(f"formal test model {name!r} threshold registration is test-tainted")
        registered_threshold = float(registered.get("threshold", np.nan))
        if not np.isclose(registered_threshold, selected_threshold, rtol=0.0, atol=1e-7):
            raise ValueError(f"formal test model {name!r} threshold disagrees with calibration")
        declared_checkpoint = registered.get("checkpoint")
        declared_calibration = registered.get("calibration")
        if declared_checkpoint is not None and Path(str(declared_checkpoint)).resolve() != checkpoint_path:
            raise ValueError(f"formal test model {name!r} checkpoint provenance disagrees")
        if declared_calibration is not None and Path(str(declared_calibration)).resolve() != calibration_path:
            raise ValueError(f"formal test model {name!r} calibration provenance disagrees")
        if name not in frozen_thresholds or not np.isclose(
            float(frozen_thresholds[name]), selected_threshold, rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"formal test model {name!r} has no matching frozen threshold")


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b > 0 else float(default)


def aggregate_weighted_recall_lcb(
    weighted_tp: Sequence[float],
    weighted_gt: Sequence[float],
    replicates: int,
    seed: int,
) -> float | None:
    """Return the pose-bootstrap lower bound without using test for selection."""
    if int(replicates) <= 0:
        return None
    return float(
        weighted_recall_lower_confidence_bound(
            np.asarray(weighted_tp, dtype=np.float64),
            np.asarray(weighted_gt, dtype=np.float64),
            replicates=int(replicates),
            seed=int(seed),
        )
    )


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
    weighted_safe = aggregate_weighted_recall_safe_rows(
        rows,
        target_weighted_recall,
        minimum_lower_confidence_bound=target_weighted_recall,
    )
    utility_safe = [
        r
        for r in weighted_safe
        if r["pose_visual_utility_recall"] >= target_utility_recall
    ]
    any_high_recall = [r for r in rows if r["pose_recall"] >= target_recall]
    return {
        "primaryWeightedPrecision": max(
            weighted_safe,
            key=lambda row: (
                float(row.get("pose_precision", 0.0)),
                float(row.get("pose_f1", 0.0)),
                float(row.get("aggregateWeightedRecall", 0.0)),
                -float(row.get("avg_pred_count", 0.0)),
            ),
        ) if weighted_safe else None,
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
    bootstrap_replicates: int = 0,
    sidecar_writer: ScoreSidecarWriter | None = None,
    collect_score_stats: bool = False,
) -> dict[str, Any]:
    if sidecar_writer is not None and sample_with_replacement:
        raise ValueError("score sidecar requires one ordered pass without replacement")
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
    aggregate_weighted_tp = np.zeros((n_th,), dtype=np.float64)
    aggregate_weighted_gt = 0.0
    pose_weighted_tp_values: list[list[float]] = [[] for _ in range(n_th)]
    pose_weighted_gt_values: list[list[float]] = [[] for _ in range(n_th)]
    pose_ap_values: list[float] = []
    pose_positive_rates: list[float] = []
    score_values: list[np.ndarray] = []
    score_targets: list[np.ndarray] = []

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
    zero_gt_pose_count = 0
    forward_ms: list[float] = []
    total_ms: list[float] = []
    diagnostics: dict[str, list[float]] = {}

    if sidecar_writer is not None and not sample_with_replacement:
        ordered_batches = (
            split.pose_indices[start : min(split.pose_indices.size, start + max(1, poses_per_batch))]
            for start in range(0, split.pose_indices.size, max(1, poses_per_batch))
        )
    else:
        ordered_batches = split.pose_set_batches(poses_per_batch, rng, steps, include_empty=True)
    for pose_indices in tqdm(
        ordered_batches,
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

        sidecar_batch_records: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
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
            zero_gt_pose_count += 1
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
            if sidecar_writer is not None:
                sidecar_batch_records[int(pose_indices[row_index])] = (
                    np.zeros((0,), dtype=np.uint32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.uint8),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.uint32),
                )
            for threshold_index in range(n_th):
                pose_weighted_tp_values[threshold_index].append(0.0)
                pose_weighted_gt_values[threshold_index].append(0.0)
        if batch["instance"].size == 0:
            if sidecar_writer is not None:
                for pose_index in pose_indices.tolist():
                    sidecar_writer.append_pose(int(pose_index), *sidecar_batch_records[int(pose_index)])
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
            weighted_gt_scalar = float((truth * weights).sum())
            weighted_gt = np.full((n_th,), weighted_gt_scalar, dtype=np.float64)
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

            positive_count = int(truth.sum())
            if positive_count > 0:
                pose_ap = average_precision(scores, truth.astype(np.uint8, copy=False))
                if pose_ap is None:
                    raise ValueError("a pose with positive labels produced no AP")
                pose_ap_values.append(float(pose_ap))
                pose_positive_rates.append(float(positive_count / max(1, truth.size)))
            else:
                zero_gt_pose_count += 1
            if collect_score_stats:
                score_values.append(scores.astype(np.float32, copy=True))
                score_targets.append(truth.astype(np.uint8, copy=True))
            for threshold_index in range(n_th):
                pose_weighted_tp_values[threshold_index].append(float(weighted_tp[threshold_index]))
                pose_weighted_gt_values[threshold_index].append(weighted_gt_scalar)

            for th_i in range(n_th):
                pred_mask = pred[:, th_i]
                pred_glbs = np.unique(glbs[pred_mask])
                pred_bytes = float(glb_byte_cost[pred_glbs].sum()) if pred_glbs.size else 0.0
                pose_pred_glb_count[th_i] += float(pred_glbs.size)
                pose_pred_bytes[th_i] += pred_bytes
                pose_candidate_byte_reduction[th_i] += 1.0 - pred_bytes / candidate_bytes

            if sidecar_writer is not None:
                sidecar_batch_records[int(pose_indices[pose_i])] = (
                    ids_all[start:end].astype(np.uint32, copy=False),
                    scores.astype(np.float32, copy=False),
                    truth.astype(np.uint8, copy=False),
                    weights.astype(np.float32, copy=False),
                    ids_all[start:end][pred[:, 0]].astype(np.uint32, copy=False)
                    if n_th == 1 else np.zeros((0,), dtype=np.uint32),
                )

            agg_tp += local_tp
            agg_fp += local_fp
            agg_fn += local_fn
            agg_tn += local_tn
            aggregate_weighted_tp += weighted_tp
            aggregate_weighted_gt += weighted_gt_scalar
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
        if sidecar_writer is not None:
            for pose_index in pose_indices.tolist():
                sidecar_writer.append_pose(int(pose_index), *sidecar_batch_records[int(pose_index)])

    rows: list[dict[str, Any]] = []
    avg_gt = gt_count_sum / max(1, pose_count)
    avg_candidate = candidate_count_sum / max(1, pose_count)
    avg_candidate_glb = candidate_glb_count_sum / max(1, pose_count)
    avg_candidate_bytes = candidate_bytes_sum / max(1, pose_count)
    aggregate_ap = None
    aggregate_positive_rate = 0.0
    if collect_score_stats:
        all_scores = np.concatenate(score_values) if score_values else np.zeros((0,), dtype=np.float32)
        all_targets = np.concatenate(score_targets) if score_targets else np.zeros((0,), dtype=np.uint8)
        aggregate_ap = average_precision(all_scores, all_targets)
        aggregate_positive_rate = float(all_targets.mean()) if all_targets.size else 0.0
    pose_ap = float(np.mean(pose_ap_values)) if pose_ap_values else None
    pose_positive_rate = float(np.mean(pose_positive_rates)) if pose_positive_rates else None
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
        aggregate_weighted = (
            float(aggregate_weighted_tp[i] / aggregate_weighted_gt)
            if aggregate_weighted_gt > 1e-12 else 1.0
        )
        aggregate_lcb = aggregate_weighted_recall_lcb(
            pose_weighted_tp_values[i],
            pose_weighted_gt_values[i],
            bootstrap_replicates,
            seed + i,
        )
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
                "aggregate_weighted_recall": aggregate_weighted,
                "aggregateWeightedRecall": aggregate_weighted,
                "aggregate_weighted_recall_lower_confidence_bound": aggregate_lcb,
                "aggregateWeightedRecallLowerConfidenceBound": aggregate_lcb,
                "poseMacroAveragePrecision": pose_ap,
                "aggregateAveragePrecision": aggregate_ap,
                "poseMacroPositiveRate": pose_positive_rate,
                "aggregatePositiveRate": aggregate_positive_rate,
                "poseMacroApLift": (
                    None if pose_ap is None or pose_positive_rate is None or pose_positive_rate <= 0.0
                    else float(pose_ap / pose_positive_rate)
                ),
                "aggregateApLift": (
                    None if aggregate_ap is None or aggregate_positive_rate <= 0.0
                    else float(aggregate_ap / aggregate_positive_rate)
                ),
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
        "aggregateWeightedRecall": (
            float(aggregate_weighted_tp[0] / aggregate_weighted_gt)
            if n_th == 1 and aggregate_weighted_gt > 1e-12 else None
        ),
        "aggregateWeightedRecallLowerConfidenceBound": (
            aggregate_weighted_recall_lcb(
                pose_weighted_tp_values[0], pose_weighted_gt_values[0], bootstrap_replicates, seed
            ) if n_th == 1 else None
        ),
        "zeroGtPoseCount": int(zero_gt_pose_count),
        "testRead": False,
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
        f"- Primary threshold rule: `aggregate weighted recall > {meta['targetWeightedRecall']}` and its LCB > target; choose highest pose precision",
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
    parser.add_argument("--models", default="baseline_keep_all,baseline_aabb_ray")
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATA_ROOT / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1"),
    )
    parser.add_argument("--runtime-meta", default=str(DEFAULT_DATA_ROOT / "hkust-v3/assets/runtimeVisibilityMeta.json"))
    parser.add_argument("--glb-index", default=str(DEFAULT_DATA_ROOT / "hkust-v3/assets/glbIndex.json"))
    parser.add_argument("--glb-root", default=str(DEFAULT_DATA_ROOT / "hkust-v3/assets"))
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
        help="Retained as a rejected historical option; test-time threshold scanning is disabled.",
    )
    parser.add_argument(
        "--allow-candidate-visible-union",
        action="store_true",
        help="Exploratory-only legacy candidate repair. Formal evaluation must use the stored raw candidate set unchanged.",
    )
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--sidecar-dir", type=Path, default=None)
    parser.add_argument("--model-spec-file", type=Path, default=None)
    parser.add_argument("--scene-name", default="")
    parser.add_argument("--collect-score-stats", action="store_true")
    parser.add_argument(
        "--allow-missing-glb-cost",
        action="store_true",
        help="Exploratory-only median/count imputation for missing GLB files; formal evaluation must leave this disabled.",
    )
    args = parser.parse_args()
    if args.sample_with_replacement and int(args.max_eval_poses) <= 0:
        parser.error("--sample-with-replacement requires --max-eval-poses > 0")
    if args.split == "test":
        if args.exploratory_test_threshold_scan:
            parser.error("test-time threshold scanning is disabled; use an explicit frozen threshold file")
        if not args.frozen_threshold_file:
            parser.error(
                "Formal test evaluation requires --frozen-threshold-file."
            )
        if args.sample_with_replacement or int(args.max_eval_poses) > 0:
            parser.error("Formal test evaluation must traverse all unique test poses without replacement.")
        if int(args.max_candidates_per_pose) > 0:
            parser.error("Formal test evaluation must use the stored full candidate set.")
    if args.allow_candidate_visible_union and args.split == "test":
        parser.error("--allow-candidate-visible-union cannot be used in a formal test evaluation.")

    device = select_device(args.device)
    spec_file_values: Mapping[str, Any] = {}
    if args.model_spec_file is not None:
        spec_payload = json.loads(args.model_spec_file.read_text(encoding="utf-8"))
        values = spec_payload.get("models", spec_payload) if isinstance(spec_payload, dict) else None
        if not isinstance(values, Mapping):
            parser.error("--model-spec-file must contain a model-to-spec mapping")
        spec_file_values = values
    requested_models = [name.strip() for name in str(args.models).split(",") if name.strip()]
    defaults = selected_default_specs(",".join(name for name in requested_models if name in DEFAULT_MODEL_NAMES))
    specs: dict[str, dict[str, str]] = {}
    for name in requested_models:
        if name in defaults:
            specs[name] = dict(defaults[name])
        elif name in spec_file_values and isinstance(spec_file_values[name], Mapping):
            specs[name] = {str(key): str(value) for key, value in spec_file_values[name].items()}
        else:
            raise KeyError(f"Unknown model '{name}'. Supply it in --model-spec-file or use a registered default.")
        if name in spec_file_values and isinstance(spec_file_values[name], Mapping):
            specs[name].update({str(key): str(value) for key, value in spec_file_values[name].items()})
    first = next(iter(specs.values()))
    if first.get("checkpoint"):
        checkpoint = torch.load(first["checkpoint"], map_location="cpu")
        checkpoint_config = checkpoint.get("config") or checkpoint.get("modelConfig")
        if not isinstance(checkpoint_config, Mapping):
            parser.error("checkpoint spec has no config or modelConfig")
        num_instances = int(checkpoint_config["numInstances"])
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
    if args.split == "test":
        validate_formal_test_specs(
            specs, args.frozen_threshold_file, frozen_thresholds
        )
    scan_thresholds = threshold_grid() if not frozen_thresholds else None

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
        sidecar_writer = None
        if args.split == "test" or args.sidecar_dir is not None:
            sidecar_root = (
                Path(args.sidecar_dir).resolve() / name
                if args.sidecar_dir is not None
                else output_dir / "score_sidecars" / name
            )
            sidecar_writer = ScoreSidecarWriter(
                sidecar_root,
                split=requested_split,
                threshold=float(thresholds[0]) if thresholds.size == 1 else None,
                checkpoint=spec.get("checkpoint"),
                calibration=spec.get("calibration") or spec.get("eval_summary"),
            )
        summary = evaluate_runner(
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
            bootstrap_replicates=(
                int(args.bootstrap_replicates)
                if int(args.bootstrap_replicates) > 0
                else 10000 if args.split == "test" else 0
            ),
            sidecar_writer=sidecar_writer,
            collect_score_stats=args.split == "test" or args.collect_score_stats,
        )
        if sidecar_writer is not None:
            summary["scoreSidecar"] = str(sidecar_writer.close())
        summary["testRead"] = args.split == "test"
        summaries.append(summary)

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
            "thresholdMode": "frozen_one_shot" if frozen_thresholds else "split_threshold_scan",
            "frozenThresholdFile": args.frozen_threshold_file or None,
            "testEvaluationCount": 1 if args.split == "test" else 0,
            "testRead": args.split == "test",
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
        "testRead": args.split == "test",
        "scene": str(args.scene_name) or None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "summary.md", payload)
    print(json.dumps({"outputDir": str(output_dir), "evaluatedPoses": evaluated, "models": [s["name"] for s in summaries]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
