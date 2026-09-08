#!/usr/bin/env python3
"""Replay one calibrated relation-prior checkpoint on train/calibration/validation.

This entry point owns the v4 replay contract.  It reads the checkpoint's
calibration-frozen threshold, queries the fixed geometry and survival table,
and writes per-pose instance/resource metrics.  Train replay is diagnostic-only:
it consumes the checkpoint-owned threshold and cannot replace it.  This entry
point never reads the test split and does not import a legacy evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.runtime_meta import load_runtime_meta  # noqa: E402
from common.threshold_selection import (  # noqa: E402
    aggregate_weighted_cull_selection_rule,
    select_aggregate_weighted_cull_workpoint,
)
from pvs_model import (  # noqa: E402
    BoundedRelationSurvivalMomentModel,
    GEO_DIM,
    MODEL_SCHEMA,
    SURVIVAL_PARAMETER_DIM,
    SURVIVAL_RANK,
)
from pvs_threshold_metrics import evaluate_thresholds, threshold_grid  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


CHECKPOINT_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4"
REPLAY_SPLITS = ("train", "calibration", "validation")
TRAINING_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4"
CALIBRATION_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4"
EVALUATION_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-v4-evaluation-v1"
REGISTERED_MAX_NORM_CYCLES = 8.0
# Keep the schema names discoverable under the v4-specific vocabulary used by
# the runner manifest and checkpoint validator.
V4_CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
V4_TRAINING_SCHEMA = TRAINING_SCHEMA
V4_CALIBRATION_SCHEMA = CALIBRATION_SCHEMA
V4_EVALUATION_SCHEMA = EVALUATION_SCHEMA
DIAGNOSTIC_RECALIBRATION_SCHEMA = (
    "pvs-bounded-relation-prior-instance-calibrated-moment-v4-diagnostic-recalibration-v1"
)
CORE_METRICS = (
    "precision",
    "recall",
    "weightedRecall",
    "f1",
    "jaccard",
    "accuracy",
    "balancedAccuracy",
    "specificity",
    "usefulCull",
    "badCull",
    "avgPredCount",
    "avgCandidateCount",
    "avgGtCount",
    "predictedGlbCount",
    "predictedGlbBytes",
    "glbByteReduction",
    "requiredGlbCount",
    "requiredGlbBytes",
    "predOverCandidate",
    "predOverGt",
    "downloadUtilityRecall",
    "glbBytesAtAchievedVisualUtility",
)


def _max_frequency_norm_cycles_from_config(config: Mapping[str, Any]) -> float:
    frequency = config.get("frequency")
    if not isinstance(frequency, Mapping) or "maxNormCycles" not in frequency:
        raise ValueError("v4 checkpoint frequency.maxNormCycles is required")
    try:
        value = float(frequency["maxNormCycles"])
    except (TypeError, ValueError) as exc:
        raise ValueError("v4 checkpoint frequency.maxNormCycles is invalid") from exc
    if not np.isfinite(value) or value <= 0.0 or value > REGISTERED_MAX_NORM_CYCLES:
        raise ValueError(
            "v4 checkpoint frequency.maxNormCycles must be finite, positive, "
            f"and <= {REGISTERED_MAX_NORM_CYCLES:g}"
        )
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe_metadata(value: Any) -> Any:
    """Convert open numeric bounds to JSON null without hiding metric errors."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _load_geometry(path: Path, num_instances: int) -> np.ndarray:
    expected = int(num_instances) * GEO_DIM
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed geometry table: {path}")
    values = np.fromfile(path, dtype=np.float16)
    if values.size != expected or not bool(np.isfinite(values).all()):
        raise ValueError(f"invalid geometry table {path}: {values.size} values, expected {expected}")
    return values.reshape(int(num_instances), GEO_DIM).astype(np.float32)


def _load_glb_bytes(index_path: Path, root: Path, num_glbs: int) -> np.ndarray:
    if not index_path.is_file():
        raise FileNotFoundError(f"missing GLB index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    values = np.zeros((int(num_glbs),), dtype=np.float64)
    for entry in payload.get("entries", []):
        gid = int(entry.get("globalId", -1))
        if 0 <= gid < values.size:
            path = root / str(entry.get("path", ""))
            if path.is_file():
                values[gid] = float(path.stat().st_size)
    if values.size and np.any(values <= 0.0):
        missing = np.flatnonzero(values <= 0.0)[:8].tolist()
        raise FileNotFoundError(f"GLB byte index has missing files, first IDs: {missing}")
    return values


def _strip_fingerprint_fields(value: Any) -> Any:
    """Keep readable provenance while omitting legacy hash-like fields."""

    def _is_legacy_fingerprint_key(key: Any) -> bool:
        normalized = str(key).lower()
        return (
            normalized in {"sha", "sha1", "sha256", "digest", "hash", "checksum"}
            or normalized.endswith("sha1")
            or normalized.endswith("sha256")
            or normalized.endswith("digest")
            or normalized.endswith("hash")
            or normalized.endswith("checksum")
        )

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _is_legacy_fingerprint_key(key):
                continue
            result[str(key)] = _strip_fingerprint_fields(item)
        return result
    if isinstance(value, list):
        return [_strip_fingerprint_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_fingerprint_fields(item) for item in value]
    return value


def _frozen_threshold(
    checkpoint: Mapping[str, Any],
    calibration_summary: Mapping[str, Any],
    *,
    allow_unsafe: bool,
) -> tuple[float, dict[str, Any]]:
    """Read the checkpoint-owned calibration workpoint without reselection."""
    if calibration_summary.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("v4 calibration summary schema is invalid")
    if calibration_summary.get("testRead") is not False:
        raise ValueError("v4 calibration summary is not test-free")
    status = str(calibration_summary.get("status", ""))
    if status == "safe":
        best = calibration_summary.get("bestSafe")
        safe = True
    elif status == "no_qualified_safety_workpoint" and allow_unsafe:
        best = calibration_summary.get("bestDiagnostic")
        safe = False
    else:
        raise ValueError("v4 checkpoint has no permitted calibration-frozen workpoint")
    if not isinstance(best, Mapping):
        raise ValueError("v4 calibration summary has no frozen workpoint")
    selection = best.get("selection", best)
    if not isinstance(selection, Mapping):
        raise ValueError("v4 calibration workpoint has no selection")
    try:
        threshold = float(selection["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v4 calibration workpoint has no threshold") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("v4 calibration threshold is outside [0, 1]")
    if safe and (
        float(selection.get("aggregateWeightedRecall", -1.0)) <= 0.99
        or float(selection.get("aggregateWeightedRecallLowerConfidenceBound", -1.0)) <= 0.99
    ):
        raise ValueError("v4 safe calibration workpoint fails the weighted-recall safety gate")
    calibration = calibration_summary.get("calibration")
    rows = calibration.get("thresholdRows") if isinstance(calibration, Mapping) else None
    if not isinstance(rows, list) or not any(
        isinstance(row, Mapping)
        and np.isfinite(float(row.get("threshold", np.nan)))
        and abs(float(row["threshold"]) - threshold) <= 1e-7
        for row in rows
    ):
        raise ValueError("v4 frozen threshold is absent from calibration threshold rows")
    checkpoint_best = checkpoint.get("best")
    if not isinstance(checkpoint_best, Mapping):
        raise ValueError("v4 checkpoint has no frozen best workpoint")
    checkpoint_selection = checkpoint_best.get("selection", checkpoint_best)
    if not isinstance(checkpoint_selection, Mapping) or not np.isclose(
        float(checkpoint_selection.get("threshold", np.nan)), threshold, rtol=0.0, atol=1e-7
    ):
        raise ValueError("v4 checkpoint threshold disagrees with calibration summary")
    return threshold, {
        "protocol": "checkpoint_own_calibration_only",
        "selectionSplit": "calibration",
        "selectedThreshold": threshold,
        "selectedFromTest": False,
        "testEvaluationCount": 0,
        "safeWorkpoint": safe,
        "selection": dict(selection),
    }


def _v4_frozen_threshold(
    checkpoint: Mapping[str, Any],
    calibration_summary: Mapping[str, Any],
    *,
    allow_unsafe: bool,
) -> tuple[float, dict[str, Any]]:
    """Named v4 entry for callers that record the checkpoint contract."""
    return _frozen_threshold(checkpoint, calibration_summary, allow_unsafe=allow_unsafe)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def _resource_metrics(
    candidate_ids: np.ndarray,
    predicted_ids: np.ndarray,
    visible_ids: np.ndarray,
    visible_weights: np.ndarray,
    instance_to_glb: np.ndarray,
    glb_bytes: np.ndarray,
) -> dict[str, float]:
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    predicted = np.asarray(predicted_ids, dtype=np.int64)
    visible = np.asarray(visible_ids, dtype=np.int64)
    weights = np.asarray(visible_weights, dtype=np.float64)
    candidate_glbs = np.unique(instance_to_glb[candidates]) if candidates.size else np.zeros(0, dtype=np.int64)
    predicted_glbs = np.unique(instance_to_glb[predicted]) if predicted.size else np.zeros(0, dtype=np.int64)
    visible_glbs = np.unique(instance_to_glb[visible]) if visible.size else np.zeros(0, dtype=np.int64)
    candidate_bytes = float(glb_bytes[candidate_glbs].sum()) if candidate_glbs.size else 0.0
    predicted_bytes = float(glb_bytes[predicted_glbs].sum()) if predicted_glbs.size else 0.0
    visible_instance_glbs = instance_to_glb[visible] if visible.size else np.zeros(0, dtype=np.int64)
    found = np.isin(visible_instance_glbs, predicted_glbs, assume_unique=False)
    utility = float(weights[found].sum()) if weights.size else 0.0
    total_utility = float(weights.sum())
    utility_by_glb: dict[int, float] = {}
    for instance, weight in zip(visible.tolist(), weights.tolist(), strict=True):
        gid = int(instance_to_glb[instance])
        utility_by_glb[gid] = utility_by_glb.get(gid, 0.0) + float(weight)
    equivalent_bytes = 0.0
    accumulated = 0.0
    for gid in sorted(
        utility_by_glb,
        key=lambda value: utility_by_glb[value] / max(1.0, float(glb_bytes[value])),
        reverse=True,
    ):
        if utility <= 0.0:
            break
        equivalent_bytes += float(glb_bytes[gid])
        accumulated += utility_by_glb[gid]
        if accumulated >= utility:
            break
    return {
        "predictedGlbCount": float(predicted_glbs.size),
        "candidateGlbCount": float(candidate_glbs.size),
        "candidateGlbBytes": candidate_bytes,
        "requiredGlbCount": float(visible_glbs.size),
        "requiredGlbBytes": float(glb_bytes[visible_glbs].sum()) if visible_glbs.size else 0.0,
        "predictedGlbBytes": predicted_bytes,
        "glbCountReduction": 1.0 - _safe_div(float(predicted_glbs.size), float(candidate_glbs.size)),
        "glbByteReduction": 1.0 - _safe_div(predicted_bytes, candidate_bytes),
        "downloadUtilityRecall": _safe_div(utility, total_utility, 1.0),
        "glbBytesAtAchievedVisualUtility": equivalent_bytes,
    }


def _camel_metrics(raw: Mapping[str, Any], resources: Mapping[str, float]) -> dict[str, float]:
    mapping = {
        "pose_precision": "precision",
        "pose_recall": "recall",
        "pose_weighted_recall": "weightedRecall",
        "pose_f1": "f1",
        "pose_jaccard": "jaccard",
        "pose_accuracy": "accuracy",
        "pose_balanced_accuracy": "balancedAccuracy",
        "pose_specificity": "specificity",
        "pose_useful_cull": "usefulCull",
        "pose_bad_cull": "badCull",
        "avg_pred_count": "avgPredCount",
        "avg_candidate_count": "avgCandidateCount",
        "avg_gt_count": "avgGtCount",
        "pred_over_candidate": "predOverCandidate",
        "pred_over_gt": "predOverGt",
    }
    result = {target: float(raw.get(source, raw.get(target, 0.0))) for source, target in mapping.items()}
    result["poseMacroWeightedRecall"] = result["weightedRecall"]
    result.update({key: float(value) for key, value in resources.items()})
    return result


def _weighted_lcb_from_rows(rows: list[Mapping[str, Any]], replicates: int, seed: int) -> float:
    if not rows:
        return 0.0
    weighted_tp = np.asarray([float(row["metrics"].get("weightedTp", 0.0)) for row in rows], dtype=np.float64)
    weighted_gt = np.asarray([float(row["metrics"].get("weightedGt", 0.0)) for row in rows], dtype=np.float64)
    valid = np.isfinite(weighted_tp) & np.isfinite(weighted_gt) & (weighted_gt > 1e-12)
    weighted_tp = weighted_tp[valid]
    weighted_gt = weighted_gt[valid]
    if weighted_tp.size == 0:
        return 1.0
    if int(replicates) <= 0:
        return 0.0
    rng = np.random.default_rng(int(seed))
    batch = max(1, min(int(replicates), 2_000_000 // max(1, weighted_tp.size)))
    values = np.empty((int(replicates),), dtype=np.float64)
    for start in range(0, int(replicates), batch):
        end = min(int(replicates), start + batch)
        indices = rng.integers(0, weighted_tp.size, size=(end - start, weighted_tp.size), endpoint=False)
        numerator = weighted_tp[indices].sum(axis=1)
        denominator = weighted_gt[indices].sum(axis=1)
        values[start:end] = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 1e-12)
    return float(np.quantile(values, 0.05))


def _pose_macro_lcb_from_rows(rows: list[Mapping[str, Any]], replicates: int, seed: int) -> float:
    values = np.asarray(
        [float(row["metrics"].get("poseMacroWeightedRecall", row["metrics"].get("weightedRecall", 0.0))) for row in rows],
        dtype=np.float64,
    )
    if values.size == 0:
        return 0.0
    if int(replicates) <= 0:
        return float(values.mean())
    rng = np.random.default_rng(int(seed))
    batch = max(1, min(int(replicates), 2_000_000 // max(1, values.size)))
    samples = np.empty((int(replicates),), dtype=np.float64)
    for start in range(0, int(replicates), batch):
        end = min(int(replicates), start + batch)
        indices = rng.integers(0, values.size, size=(end - start, values.size), endpoint=False)
        samples[start:end] = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.05))


def _summarize_pose_rows(rows: list[dict[str, Any]], lcb_replicates: int = 0, seed: int = 0) -> dict[str, Any]:
    if not rows:
        raise ValueError("v4 evaluator produced no pose rows")
    macro = {
        metric: float(np.mean([float(row["metrics"].get(metric, 0.0)) for row in rows]))
        for metric in CORE_METRICS
    }
    macro["poseMacroWeightedRecall"] = float(
        np.mean([float(row["metrics"].get("poseMacroWeightedRecall", row["metrics"].get("weightedRecall", 0.0))) for row in rows])
    )
    totals = {key: float(sum(float(row["metrics"].get(key, 0.0)) for row in rows)) for key in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    candidate = tp + fp + fn + tn
    gt = tp + fn
    pred = tp + fp
    recall = _safe_div(tp, gt, 1.0)
    precision = _safe_div(tp, tp + fp, 1.0)
    specificity = _safe_div(tn, tn + fp, 1.0)
    weighted_tp = float(sum(float(row["metrics"].get("weightedTp", 0.0)) for row in rows))
    weighted_gt = float(sum(float(row["metrics"].get("weightedGt", 0.0)) for row in rows))
    aggregate = {
        "precision": precision,
        "recall": recall,
        "weightedRecall": _safe_div(weighted_tp, weighted_gt, 1.0),
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "jaccard": _safe_div(tp, tp + fp + fn),
        "accuracy": _safe_div(tp + tn, candidate, 1.0),
        "balancedAccuracy": 0.5 * (recall + specificity),
        "specificity": specificity,
        "usefulCull": _safe_div(tn, candidate),
        "badCull": _safe_div(fn, candidate),
        "avgPredCount": _safe_div(pred, len(rows)),
        "avgCandidateCount": _safe_div(candidate, len(rows)),
        "avgGtCount": _safe_div(gt, len(rows)),
        "predOverCandidate": _safe_div(pred, candidate),
        "predOverGt": _safe_div(pred, gt),
        "tp": tp / len(rows),
        "fp": fp / len(rows),
        "fn": fn / len(rows),
        "tn": tn / len(rows),
        "aggregateWeightedRecall": _safe_div(weighted_tp, weighted_gt, 1.0),
        "poseMacroWeightedRecall": macro["poseMacroWeightedRecall"],
        "weightedTp": weighted_tp / len(rows),
        "weightedGt": weighted_gt / len(rows),
    }
    for key in (
        "candidateGlbCount", "predictedGlbCount", "candidateGlbBytes", "predictedGlbBytes",
        "glbCountReduction", "glbByteReduction", "downloadUtilityRecall",
        "glbBytesAtAchievedVisualUtility", "requiredGlbCount", "requiredGlbBytes",
    ):
        aggregate[key] = float(np.mean([float(row["metrics"].get(key, 0.0)) for row in rows]))
    if int(lcb_replicates) > 0:
        aggregate_lcb = _weighted_lcb_from_rows(rows, int(lcb_replicates), int(seed))
        macro_lcb = _pose_macro_lcb_from_rows(rows, int(lcb_replicates), int(seed) + 100000)
    else:
        aggregate_lcb = None
        macro_lcb = None
    aggregate["aggregateWeightedRecallLowerConfidenceBound"] = aggregate_lcb
    aggregate["poseMacroWeightedRecallLowerConfidenceBound"] = macro_lcb
    aggregate["weightedRecallLowerConfidenceBound"] = aggregate_lcb
    macro["aggregateWeightedRecallLowerConfidenceBound"] = aggregate_lcb
    macro["poseMacroWeightedRecallLowerConfidenceBound"] = macro_lcb
    return {
        "poseMacro": macro,
        "aggregate": aggregate,
        "counts": totals,
        "poseCount": len(rows),
        "weightedRecallLowerConfidenceBound": aggregate_lcb,
        "testRead": False,
    }


def _diagnostic_workpoint(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("aggregateWeightedRecallLowerConfidenceBound") or -1.0),
            float(row.get("aggregateWeightedRecall") or -1.0),
            float(row.get("agg_balanced_accuracy") or 0.0),
            float(row.get("agg_useful_cull") or 0.0),
            float(row.get("agg_precision") or 0.0),
            -float(row.get("avg_pred_count") or 0.0),
        ),
    )


def _diagnostic_recalibration(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    model: BoundedRelationSurvivalMomentModel,
    dataset: PoseCSRDataset,
    runtime_features: torch.Tensor,
    world_aabbs: np.ndarray,
    instance_to_glb: np.ndarray,
    glb_bytes: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Rescan complete calibration without modifying checkpoint-owned assets."""
    replicates = int(getattr(args, "recalibration_bootstrap_replicates", 2000))
    if replicates <= 0:
        raise ValueError("diagnostic recalibration bootstrap count must be positive")
    calibration_seed = int(args.seed)
    validation_seed_arg = getattr(args, "recalibration_validation_seed", None)
    validation_seed = (
        calibration_seed + 1
        if validation_seed_arg is None
        else int(validation_seed_arg)
    )
    calibration_split = dataset.split("calibration")
    validation_split = dataset.split("validation")
    calibration_rows = evaluate_thresholds(
        model,
        calibration_split,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=max(1, int(args.poses_per_batch)),
        max_steps=None,
        max_candidates_per_pose=0,
        seed=calibration_seed,
        thresholds=threshold_grid(),
        collect_pose_stats=True,
        allow_candidate_visible_union=False,
        bootstrap_replicates=replicates,
        collect_score_stats=True,
        collect_per_pose=False,
        instance_to_glb=instance_to_glb,
        glb_bytes=glb_bytes,
    )
    selected = select_aggregate_weighted_cull_workpoint(
        calibration_rows,
        target_weighted_recall=0.99,
        minimum_lower_confidence_bound=0.99,
    )
    diagnostic = _diagnostic_workpoint(calibration_rows)
    chosen = selected if selected is not None else diagnostic
    validation = None
    if chosen is not None:
        validation_rows = evaluate_thresholds(
            model,
            validation_split,
            runtime_features,
            world_aabbs,
            device,
            poses_per_batch=max(1, int(args.poses_per_batch)),
            max_steps=None,
            max_candidates_per_pose=0,
            seed=validation_seed,
            thresholds=np.asarray([float(chosen["threshold"])], dtype=np.float32),
            collect_pose_stats=True,
            allow_candidate_visible_union=False,
            bootstrap_replicates=replicates,
            collect_score_stats=True,
            collect_per_pose=False,
            instance_to_glb=instance_to_glb,
            glb_bytes=glb_bytes,
        )
        validation = validation_rows[0] if validation_rows else None
    protocol = checkpoint.get("protocol")
    return {
        "schema": DIAGNOSTIC_RECALIBRATION_SCHEMA,
        "status": "safe" if selected is not None else "no_qualified_safety_workpoint",
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", 0)),
        "seed": int(protocol.get("seed", args.seed)) if isinstance(protocol, Mapping) else int(args.seed),
        "selectionRule": aggregate_weighted_cull_selection_rule(0.99, 0.99),
        "bootstrapReplicates": replicates,
        "bootstrapSeeds": {
            "calibration": calibration_seed,
            "validation": validation_seed,
        },
        "calibrationPoseCount": int(calibration_split.pose_indices.size),
        "validationPoseCount": int(validation_split.pose_indices.size),
        "selectedSafe": _json_safe_metadata(selected),
        "diagnostic": _json_safe_metadata(diagnostic),
        "validationAtSelectedThreshold": _json_safe_metadata(validation),
        "calibrationThresholdRows": _json_safe_metadata(calibration_rows),
        "thresholdSource": "complete-calibration-split-diagnostic-rescan",
        "testRead": False,
    }


@torch.no_grad()
def _evaluate_checkpoint(args: argparse.Namespace, checkpoint: Mapping[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("v4 evaluator received a non-v4 checkpoint")
    if checkpoint.get("runtimeSchema") != MODEL_SCHEMA or checkpoint.get("testRead") is not False:
        raise ValueError("v4 checkpoint schema or test provenance is invalid")
    protocol = checkpoint.get("protocol")
    config = checkpoint.get("modelConfig")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != TRAINING_SCHEMA or protocol.get("testRead") is not False:
        raise ValueError("v4 checkpoint training protocol is invalid")
    if not isinstance(config, Mapping):
        raise ValueError("v4 checkpoint modelConfig is missing")
    runtime_meta_path = Path(args.runtime_meta).resolve()
    world_aabbs, instance_to_glb, _ = load_runtime_meta(runtime_meta_path)
    num_instances = int(world_aabbs.shape[0])
    representation_config = config.get("occlusionRepresentation")
    representation_mode = (
        str(representation_config.get("mode"))
        if isinstance(representation_config, Mapping)
        else str(checkpoint.get("occlusionRepresentation", "survival"))
    )
    survival_shape = config.get("survivalCoefficientShape")
    if representation_mode == "survival":
        if not (
            isinstance(survival_shape, list)
            and len(survival_shape) == 2
            and int(survival_shape[1]) == SURVIVAL_PARAMETER_DIM
        ):
            raise ValueError("v4 checkpoint has an invalid survival coefficient shape")
        survival_rank = int(survival_shape[0])
    elif representation_mode == "generic28":
        survival_rank = int(
            representation_config.get("directionRank", SURVIVAL_RANK)
            if isinstance(representation_config, Mapping)
            else SURVIVAL_RANK
        )
    else:
        survival_rank = SURVIVAL_RANK
    expected_runtime_dim = (
        GEO_DIM
        if representation_mode == "none"
        else GEO_DIM + survival_rank * SURVIVAL_PARAMETER_DIM
    )
    if int(config.get("numInstances", -1)) != num_instances or int(
        config.get("runtimeFeatureDim", -1)
    ) != expected_runtime_dim:
        raise ValueError("v4 checkpoint modelConfig does not match runtime metadata")
    depth = config.get("depthNormalization")
    if not isinstance(depth, Mapping):
        raise ValueError("v4 checkpoint depth normalization is missing")
    max_frequency_norm_cycles = _max_frequency_norm_cycles_from_config(config)
    instance_calibration = config.get("instanceCalibration")
    if not isinstance(instance_calibration, Mapping):
        raise ValueError("checkpoint instance calibration config is missing")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = BoundedRelationSurvivalMomentModel(
        num_instances=num_instances,
        num_glbs=int(config.get("numGlbs", int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0)),
        relation_hidden_dim=int(config.get("relationHiddenDim", 64)),
        hidden_dim=int(config.get("hiddenDim", 64)),
        survival_rank=survival_rank,
        relation_source=str(config.get("relationSource")),
        occlusion_representation=representation_mode,
        spectral_mode=str(config.get("spectralMode")),
        depth_q01=float(depth["q01"]),
        depth_q99=float(depth["q99"]),
        depth_epsilon=float(depth["epsilon"]),
        max_frequency_norm_cycles=max_frequency_norm_cycles,
        instance_calibration_mode=str(instance_calibration.get("mode")),
        instance_calibration_max_abs=float(
            instance_calibration.get("maximumAbsoluteResidual")
        ),
        sparse_instance_penalty=float(
            instance_calibration.get("sparseInstancePenalty")
        ),
    ).to(device)
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    state = checkpoint.get("modelState")
    if not isinstance(state, Mapping):
        raise ValueError("v4 checkpoint modelState is missing")
    model.load_state_dict(state, strict=True)

    geometry_path = Path(args.initial_geo_features).resolve()
    geometry = _load_geometry(geometry_path, num_instances)
    geometry_meta = checkpoint.get("geometry")
    if (
        not isinstance(geometry_meta, Mapping)
        or list(geometry_meta.get("shape", [])) != [num_instances, GEO_DIM]
        or geometry_meta.get("dtype") != "float16"
    ):
        raise ValueError("v4 geometry table shape or dtype disagrees with checkpoint provenance")
    geometry_tensor = torch.from_numpy(geometry).to(device)
    if representation_mode == "survival":
        features = torch.as_tensor(
            checkpoint.get("instanceSurvivalCoefficients"),
            dtype=torch.float32,
            device=device,
        )
        if tuple(features.shape) != (
            num_instances,
            survival_rank,
            SURVIVAL_PARAMETER_DIM,
        ) or not bool(
            torch.isfinite(features).all()
        ):
            raise ValueError(
                "v4 checkpoint instanceSurvivalCoefficients has an invalid shape"
            )
        prior_coefficients = torch.as_tensor(
            checkpoint.get("instanceSurvivalPriorCoefficients"),
            dtype=torch.float32,
            device=device,
        )
        calibration_residual = torch.as_tensor(
            checkpoint.get("instanceSurvivalCalibrationResidual"),
            dtype=torch.float32,
            device=device,
        )
        if (
            prior_coefficients.shape != features.shape
            or calibration_residual.shape != features.shape
            or not bool(torch.isfinite(prior_coefficients).all())
            or not bool(torch.isfinite(calibration_residual).all())
            or not torch.allclose(
                features,
                prior_coefficients + calibration_residual,
                rtol=5e-3,
                atol=1e-2,
            )
        ):
            raise ValueError(
                "checkpoint fused coefficients disagree with prior plus residual"
            )
        runtime_features = torch.cat(
            [geometry_tensor, features.reshape(num_instances, -1)], dim=-1
        )
    elif representation_mode == "generic28":
        features = torch.as_tensor(
            checkpoint.get("instanceOcclusionFeatures"),
            dtype=torch.float32,
            device=device,
        )
        if tuple(features.shape) != (
            num_instances,
            survival_rank,
            SURVIVAL_PARAMETER_DIM,
        ) or not bool(
            torch.isfinite(features).all()
        ):
            raise ValueError(
                "generic28 checkpoint instanceOcclusionFeatures has an invalid shape"
            )
        runtime_features = torch.cat(
            [geometry_tensor, features.reshape(num_instances, -1)], dim=-1
        )
    elif representation_mode == "none":
        runtime_features = geometry_tensor
    else:
        raise ValueError(f"unsupported occlusion representation: {representation_mode}")
    dataset = PoseCSRDataset(Path(args.dataset_dir).resolve(), num_instances=num_instances)
    num_glbs = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
    glb_bytes = _load_glb_bytes(
        Path(args.glb_index).resolve(), Path(args.glb_root).resolve(), num_glbs
    )
    if bool(getattr(args, "diagnostic_recalibrate", False)):
        return _diagnostic_recalibration(
            args,
            checkpoint,
            checkpoint_path,
            model,
            dataset,
            runtime_features,
            world_aabbs,
            instance_to_glb,
            glb_bytes,
            device,
        )
    split_name = str(args.split)
    if split_name.lower() == "test":
        raise ValueError("test is not allowed in the v4 evaluator")
    if split_name not in dataset.split_ids:
        raise ValueError(f"dataset has no explicit split {split_name!r}")
    split = dataset.split(split_name)
    if int(split.pose_indices.size) <= 0:
        raise ValueError("v4 replay split is empty")

    calibration_path = Path(args.calibration).resolve() if args.calibration else checkpoint_path.parent / "calibration_ready_summary.json"
    expected_calibration = (checkpoint_path.parent / "calibration_ready_summary.json").resolve()
    if calibration_path != expected_calibration:
        raise ValueError("v4 replay must use this checkpoint's calibration_ready_summary.json")
    calibration_summary = json.loads(calibration_path.read_text(encoding="utf-8"))
    threshold, threshold_source = _frozen_threshold(
        checkpoint,
        calibration_summary,
        allow_unsafe=bool(args.allow_unsafe_diagnostic),
    )
    model_meta_path = Path(args.model_meta).resolve() if args.model_meta else checkpoint_path.parent / "model_meta.json"
    if not model_meta_path.is_file():
        raise FileNotFoundError(f"v4 checkpoint has no model_meta.json: {model_meta_path}")
    model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    if model_meta.get("schema") != MODEL_SCHEMA or model_meta.get("testRead") is not False:
        raise ValueError("v4 model_meta schema or test provenance is invalid")
    if model_meta.get("modelConfig") != dict(config) or model_meta.get("protocol") != dict(protocol):
        raise ValueError("v4 model_meta does not match checkpoint")

    relation = checkpoint.get("relation")
    if not isinstance(relation, Mapping):
        raise ValueError("v4 checkpoint relation provenance is missing")
    if representation_mode == "survival" and args.relation_dir is not None:
        relation_dir = Path(args.relation_dir).resolve()
        relation_meta_path = relation_dir / "relation_csr_meta.json"
        if not relation_meta_path.is_file():
            raise FileNotFoundError(f"missing v4 relation metadata: {relation_meta_path}")
        relation_meta = json.loads(relation_meta_path.read_text(encoding="utf-8"))
        if relation_meta.get("schema") != relation.get("schema"):
            raise ValueError("v4 replay relation schema disagrees with checkpoint")
        if int(relation_meta.get("numInstances", -1)) != num_instances:
            raise ValueError("v4 replay relation instance count disagrees with checkpoint")

    started = time.perf_counter()
    rows = evaluate_thresholds(
        model,
        split,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=max(1, int(args.poses_per_batch)),
        max_steps=None,
        max_candidates_per_pose=0,
        seed=int(args.seed),
        thresholds=np.asarray([threshold], dtype=np.float32),
        collect_pose_stats=True,
        allow_candidate_visible_union=False,
        bootstrap_replicates=0,
        collect_score_stats=True,
        collect_per_pose=True,
        collect_raw_scores=bool(getattr(args, "persist_scores", False)),
    )
    if len(rows) != 1 or not isinstance(rows[0].get("_per_pose"), list):
        raise ValueError("v4 evaluator did not produce per-pose rows")
    raw_rows = sorted(rows[0]["_per_pose"], key=lambda row: int(row["poseIndex"]))
    pose_indices = [int(value) for value in split.pose_indices.tolist()]
    if [int(row["poseIndex"]) for row in raw_rows] != pose_indices:
        raise ValueError("v4 evaluation pose order does not match the split")
    per_pose: list[dict[str, Any]] = []
    for raw in raw_rows:
        pose = int(raw["poseIndex"])
        candidate_ids = np.asarray(raw.get("candidateIds", []), dtype=np.uint32)
        predicted_ids = np.asarray(raw.get("predictedIds", []), dtype=np.uint32)
        visible_ids, visible_weights = dataset.visible_slice(pose)
        resources = _resource_metrics(candidate_ids, predicted_ids, visible_ids, visible_weights, instance_to_glb, glb_bytes)
        metrics = _camel_metrics(raw.get("metrics", {}), resources)
        target_mask = np.isin(candidate_ids, visible_ids.astype(np.uint32), assume_unique=False)
        predicted_mask = np.isin(candidate_ids, predicted_ids, assume_unique=False)
        metrics.update({
            "weightedTp": float(visible_weights[np.isin(visible_ids, predicted_ids, assume_unique=False)].sum()),
            "weightedGt": float(visible_weights.sum()),
            "candidateCount": float(candidate_ids.size),
            "gtCount": float(visible_ids.size),
            "predCount": float(predicted_ids.size),
            "tp": float(np.logical_and(predicted_mask, target_mask).sum()),
            "fp": float(np.logical_and(predicted_mask, ~target_mask).sum()),
            "fn": float(np.logical_and(~predicted_mask, target_mask).sum()),
            "tn": float(np.logical_and(~predicted_mask, ~target_mask).sum()),
        })
        per_pose.append({
            "poseIndex": pose,
            "candidateCount": int(candidate_ids.size),
            "candidateIds": candidate_ids.astype(int).tolist() if args.persist_ids else None,
            "predictedIds": predicted_ids.astype(int).tolist() if args.persist_ids else None,
            "candidateScores": (
                raw.get("candidateScores")
                if bool(getattr(args, "persist_scores", False))
                else None
            ),
            "targets": (
                raw.get("targets")
                if bool(getattr(args, "persist_scores", False))
                else None
            ),
            "visibleWeights": (
                raw.get("visibleWeights")
                if bool(getattr(args, "persist_scores", False))
                else None
            ),
            "metrics": metrics,
        })
    pose_array = np.asarray(split.pose_indices, dtype="<i8")
    summary = _summarize_pose_rows(
        per_pose,
        lcb_replicates=(
            10000 if split_name in {"calibration", "validation"} else 0
        ),
        seed=int(args.seed),
    )
    return {
        "schema": EVALUATION_SCHEMA,
        "version": 1,
        "split": split_name,
        "testRead": False,
        "variant": str(protocol.get("variant", "")),
        "variantSpec": {
            "relationSource": config.get("relationSource"),
            "spectralMode": config.get("spectralMode"),
            "lossVariant": str(protocol.get("lossVariant", "")),
        },
        "relationVariant": _strip_fingerprint_fields(relation),
        "seed": int(protocol.get("seed")),
        "checkpointSeed": int(protocol.get("seed")),
        "checkpoint": str(checkpoint_path),
        "modelMeta": str(model_meta_path),
        "calibrationSummary": str(calibration_path),
        "epoch": int(checkpoint.get("epoch", 0)),
        "poseIndices": pose_indices,
        "poseCount": len(pose_indices),
        "threshold": threshold,
        "thresholdSource": threshold_source,
        "metrics": summary["aggregate"],
        "aggregateWeightedRecall": summary["aggregate"]["weightedRecall"],
        "poseMacroWeightedRecall": summary["poseMacro"]["poseMacroWeightedRecall"],
        "aggregateWeightedRecallLowerConfidenceBound": summary["aggregate"].get("aggregateWeightedRecallLowerConfidenceBound"),
        "poseMacroWeightedRecallLowerConfidenceBound": summary["poseMacro"].get("poseMacroWeightedRecallLowerConfidenceBound"),
        "poseMacro": summary["poseMacro"],
        "aggregate": summary["aggregate"],
        "scoreDistribution": rows[0].get("scoreDistribution"),
        "perPose": per_pose,
        "runtime": {
            "device": str(device),
            "elapsedSeconds": float(time.perf_counter() - started),
            "runtimeFeatureBytes": int(runtime_features.numel() * 2),
            "runtimeFeatureDim": int(runtime_features.shape[1]),
            "inferenceInputDim": int(config.get("runtimeHeadInputDim", -1)),
            "testRead": False,
        },
        "imageMetrics": {
            "status": "not_available",
            "reason": "Color-ID image evaluation is a separate formal stage",
            "testRead": False,
        },
        "unavailableMetrics": {
            "missPixelRate": "not_available: Color-ID image evaluation not attached",
            "wrongIdPixelRate": "not_available: Color-ID image evaluation not attached",
            "extraPixelRate": "not_available: Color-ID image evaluation not attached",
            "browserWebGpuLatency": "not_available: hardware browser replay not attached",
        },
        # Older checkpoints encode unbounded sampler bucket edges as +/-inf.
        # Keep the open-bound semantics as JSON null; metrics remain strictly
        # finite and are not silently sanitized.
        "protocol": _strip_fingerprint_fields(_json_safe_metadata(protocol)),
        "inputProvenance": {
            "datasetDir": str(Path(args.dataset_dir).resolve()),
            "runtimeMeta": str(runtime_meta_path),
            "geometry": {
                "path": str(geometry_path),
                "shape": [num_instances, GEO_DIM],
                "dtype": "float16",
            },
            "glbIndex": str(Path(args.glb_index).resolve()),
            "candidateSemantics": "stored native back-camera candidates; GT union disabled",
            "testRead": False,
        },
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    return _evaluate_checkpoint(args, _load_checkpoint(checkpoint_path), checkpoint_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--initial-geo-features", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=REPLAY_SPLITS, default="validation")
    parser.add_argument("--poses-per-batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-unsafe-diagnostic", action="store_true")
    parser.add_argument("--persist-ids", action="store_true")
    parser.add_argument("--persist-scores", action="store_true")
    parser.add_argument("--relation-dir", type=Path, default=None)
    parser.add_argument("--model-meta", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--diagnostic-recalibrate", action="store_true")
    parser.add_argument("--recalibration-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--recalibration-validation-seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = evaluate_checkpoint(args)
    _write_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "poseCount": payload.get("poseCount"),
                "calibrationPoseCount": payload.get("calibrationPoseCount"),
                "validationPoseCount": payload.get("validationPoseCount"),
                "testRead": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
