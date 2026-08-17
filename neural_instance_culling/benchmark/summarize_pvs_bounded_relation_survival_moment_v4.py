#!/usr/bin/env python3
"""Summarize the isolated v4 validation replay matrix.

The summarizer accepts only the v4 matrix manifest and v4 evaluation schema.
It computes pose-macro and pooled metrics, then performs the registered
paired seed/pose bootstrap.  It does not discover or reinterpret a legacy
matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from v4_semantic_contract import (  # noqa: E402
    validate_native_candidate_contract,
    validate_replay_payload,
)


SUMMARY_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-v4-validation-summary-v1"
MATRIX_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-v4-formal-validation-manifest-v1"
EVALUATION_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-v4-evaluation-v1"
V4_FORMAL_VARIANTS = (
    "full",
    "without_bounded_relation",
    "without_viewcell_moment_envelope",
    "without_safety_reserve_utility",
    "without_instance_calibration_residual",
)
V4_SEEDS = (20260801, 20260802, 20260803)
METRIC_NAMES = (
    "precision", "recall", "weightedRecall", "f1", "jaccard", "accuracy",
    "balancedAccuracy", "specificity", "usefulCull", "badCull", "avgPredCount",
    "avgCandidateCount", "avgGtCount", "predOverCandidate", "predOverGt",
    "candidateGlbCount", "predictedGlbCount", "predictedGlbBytes", "glbCountReduction",
    "glbByteReduction", "downloadUtilityRecall", "glbBytesAtAchievedVisualUtility",
    "candidateGlbBytes", "requiredGlbCount", "requiredGlbBytes",
)
COUNT_METRICS = ("tp", "fp", "fn", "tn")
BOOTSTRAP_CLUSTER_UNIT = "outer seed cluster, inner validation pose resampling"
IMAGE_SUMMARY_SCHEMA = "viewcell-image-per-benchmark-v2"
IMAGE_RENDER_SCHEMA = "local-true-component-id-browser-summary-v3"
IMAGE_RATE_FIELDS = (
    "missPixelRate",
    "wrongInstancePixelRate",
    "extraPixelRateOverImage",
)
IMAGE_REPORT_METRICS = (
    "meanMissPixelRate",
    "p95MissPixelRate",
    "meanWrongInstancePixelRate",
    "p95WrongInstancePixelRate",
    "meanExtraPixelRateOverImage",
    "p95ExtraPixelRateOverImage",
)
V4_SUMMARY_SCHEMA = SUMMARY_SCHEMA


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite metric: {value!r}")
    return result


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def _metric_from_counts(counts: Mapping[str, float], metric: str) -> float:
    tp, fp, fn, tn = (float(counts[name]) for name in COUNT_METRICS)
    candidate = tp + fp + fn + tn
    gt = tp + fn
    pred = tp + fp
    recall = _safe_div(tp, gt, 1.0)
    precision = _safe_div(tp, pred, 1.0)
    specificity = _safe_div(tn, tn + fp, 1.0)
    values = {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "jaccard": _safe_div(tp, tp + fp + fn),
        "accuracy": _safe_div(tp + tn, candidate, 1.0),
        "balancedAccuracy": 0.5 * (recall + specificity),
        "specificity": specificity,
        "usefulCull": _safe_div(tn, candidate),
        "badCull": _safe_div(fn, candidate),
        "avgPredCount": pred,
        "avgCandidateCount": candidate,
        "avgGtCount": gt,
        "predOverCandidate": _safe_div(pred, candidate),
        "predOverGt": _safe_div(pred, gt),
    }
    return float(values[metric])


def _weighted_lcb_from_rows(rows: list[Mapping[str, Any]], replicates: int, seed: int) -> float:
    if not rows:
        return 0.0
    weighted_tp = np.asarray([_finite(row["metrics"].get("weightedTp", 0.0)) for row in rows], dtype=np.float64)
    weighted_gt = np.asarray([_finite(row["metrics"].get("weightedGt", 0.0)) for row in rows], dtype=np.float64)
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


def _summarize_rows(
    rows: list[Mapping[str, Any]],
    lcb_replicates: int = 0,
    lcb_seed: int = 0,
    metric_names: Iterable[str] = METRIC_NAMES,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty v4 validation pose list")
    metric_names = tuple(metric_names)
    for row in rows:
        if not isinstance(row.get("metrics"), Mapping):
            raise ValueError("v4 validation pose row has no metrics")
    pose_macro = {
        metric: float(np.mean([_finite(row["metrics"].get(metric, 0.0)) for row in rows]))
        for metric in metric_names
    }
    totals = {name: float(sum(_finite(row["metrics"].get(name, 0.0)) for row in rows)) for name in COUNT_METRICS}
    aggregate = {
        metric: _metric_from_counts(totals, metric)
        for metric in (
            "precision", "recall", "f1", "jaccard", "accuracy", "balancedAccuracy",
            "specificity", "usefulCull", "badCull", "predOverCandidate", "predOverGt",
        )
    }
    count = float(len(rows))
    aggregate["avgPredCount"] = _safe_div(totals["tp"] + totals["fp"], count)
    aggregate["avgCandidateCount"] = _safe_div(
        sum(_finite(row["metrics"].get("candidateCount", row["metrics"].get("avgCandidateCount", 0.0))) for row in rows),
        count,
    )
    aggregate["avgGtCount"] = _safe_div(totals["tp"] + totals["fn"], count)
    weighted_tp = float(sum(_finite(row["metrics"].get("weightedTp", 0.0)) for row in rows))
    weighted_gt = float(sum(_finite(row["metrics"].get("weightedGt", 0.0)) for row in rows))
    aggregate["weightedRecall"] = _safe_div(weighted_tp, weighted_gt, 1.0)
    for metric in (
        "candidateGlbCount", "predictedGlbCount", "predictedGlbBytes", "glbCountReduction",
        "glbByteReduction", "downloadUtilityRecall", "glbBytesAtAchievedVisualUtility",
        "candidateGlbBytes", "requiredGlbCount", "requiredGlbBytes",
    ):
        aggregate[metric] = float(np.mean([_finite(row["metrics"].get(metric, 0.0)) for row in rows]))
    aggregate.update({name: value / count for name, value in totals.items()})
    aggregate["weightedTp"] = weighted_tp / count
    aggregate["weightedGt"] = weighted_gt / count
    aggregate["aggregateWeightedRecall"] = aggregate["weightedRecall"]
    aggregate["poseMacroWeightedRecall"] = float(
        np.mean([_finite(row["metrics"].get("poseMacroWeightedRecall", row["metrics"].get("weightedRecall", 0.0))) for row in rows])
    )
    pose_macro["poseMacroWeightedRecall"] = aggregate["poseMacroWeightedRecall"]
    if int(lcb_replicates) > 0:
        aggregate_lcb = _weighted_lcb_from_rows(rows, int(lcb_replicates), int(lcb_seed))
        pose_macro_lcb = _pose_macro_lcb_from_rows(rows, int(lcb_replicates), int(lcb_seed) + 100000)
    else:
        aggregate_lcb = None
        pose_macro_lcb = None
    aggregate["weightedRecallLowerConfidenceBound"] = aggregate_lcb
    aggregate["aggregateWeightedRecallLowerConfidenceBound"] = aggregate_lcb
    aggregate["poseMacroWeightedRecallLowerConfidenceBound"] = pose_macro_lcb
    pose_macro["weightedRecallLowerConfidenceBound"] = pose_macro_lcb
    pose_macro["aggregateWeightedRecallLowerConfidenceBound"] = aggregate_lcb
    pose_macro["poseMacroWeightedRecallLowerConfidenceBound"] = pose_macro_lcb
    safety = {
        "weightedRecallFloor": 0.99,
        "weightedRecallLowerConfidenceBoundFloor": 0.99,
        "weightedRecallSatisfied": bool(aggregate["weightedRecall"] > 0.99),
        "weightedRecallLowerConfidenceBoundSatisfied": bool(aggregate_lcb is not None and aggregate_lcb > 0.99),
        "gateScope": "aggregate",
    }
    return {
        "poseCount": len(rows),
        "poseMacro": pose_macro,
        "aggregate": aggregate,
        "counts": totals,
        "safety": safety,
    }


def _validate_threshold_provenance(payload: Mapping[str, Any], path: Path) -> None:
    source = payload.get("thresholdSource")
    if not isinstance(source, Mapping):
        raise ValueError(f"v4 validation replay has no thresholdSource: {path}")
    if source.get("protocol") != "checkpoint_own_calibration_only":
        raise ValueError(f"v4 validation replay threshold was not read from checkpoint calibration: {path}")
    if source.get("selectionSplit") != "calibration" or source.get("selectedFromTest") is not False:
        raise ValueError(f"v4 validation replay threshold provenance is invalid: {path}")
    if int(source.get("testEvaluationCount", -1)) != 0:
        raise ValueError(f"v4 validation replay threshold has test provenance: {path}")
    if "safeWorkpoint" not in source:
        raise ValueError(f"v4 validation replay threshold has no safety classification: {path}")
    try:
        threshold = float(payload["threshold"])
        selected = float(source["selectedThreshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"v4 validation replay threshold is incomplete: {path}") from exc
    if not np.isclose(threshold, selected, rtol=0.0, atol=1e-7):
        raise ValueError(f"v4 validation replay threshold disagrees with its source: {path}")


def _resolve_member_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_members(root: Path, manifest: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    if manifest.get("schema") != MATRIX_SCHEMA or manifest.get("testRead") is not False:
        raise ValueError("v4 matrix manifest is missing or not test-free")
    if manifest.get("mode") != "formal80_validation_replay" or int(manifest.get("epochs", -1)) != 80:
        raise ValueError("v4 matrix manifest is not formal80 validation replay")
    variants = list((manifest.get("variants") or {}).keys())
    if tuple(variants) != V4_FORMAL_VARIANTS:
        raise ValueError(f"v4 matrix variants are incomplete: {variants}")
    seeds = tuple(int(value) for value in manifest.get("seeds", []))
    if seeds != V4_SEEDS:
        raise ValueError("v4 matrix does not contain the registered three seeds")
    split_counts = manifest.get("splitPoseCounts") or {}
    expected_pose_count = int(split_counts.get("validation", 0))
    if expected_pose_count <= 0:
        raise ValueError("v4 matrix has no validation pose count")
    expected_poses_raw = manifest.get("validationPoseIndices")
    if not isinstance(expected_poses_raw, list) or len(expected_poses_raw) != expected_pose_count:
        raise ValueError("v4 matrix has no complete validation pose sequence")
    expected_poses = [int(value) for value in expected_poses_raw]
    provenance = manifest.get("inputProvenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("v4 matrix has no structured input provenance")
    dataset_info = provenance.get("dataset")
    runtime_info = provenance.get("runtimeMeta")
    if not isinstance(dataset_info, Mapping) or not str(dataset_info.get("path", "")):
        raise ValueError("v4 matrix has no dataset path for semantic replay validation")
    if not isinstance(runtime_info, Mapping) or int(runtime_info.get("numInstances", 0)) <= 0:
        raise ValueError("v4 matrix has no runtime instance count")
    dataset_path = _resolve_member_path(root, dataset_info["path"])
    dataset = PoseCSRDataset(dataset_path, num_instances=int(runtime_info["numInstances"]))
    if "validation" not in dataset.split_ids:
        raise ValueError("v4 matrix dataset has no validation split")
    actual_poses = [int(value) for value in dataset.split("validation").pose_indices.tolist()]
    if actual_poses != expected_poses:
        raise ValueError("v4 matrix validation pose order disagrees with the dataset")
    native_summary = validate_native_candidate_contract(
        dataset,
        expected_poses,
        num_instances=int(runtime_info["numInstances"]),
    )
    registered_semantics = (manifest.get("splitSemantics") or {}).get("validation")
    if isinstance(registered_semantics, Mapping):
        for field in ("poseCount", "candidateReferenceCount", "visibleReferenceCount"):
            if int(registered_semantics.get(field, -1)) != int(native_summary[field]):
                raise ValueError(f"v4 matrix validation {field} disagrees with the dataset")
    outputs = manifest.get("replayOutputs")
    member_meta = manifest.get("members")
    expected_member_count = len(V4_FORMAL_VARIANTS) * len(V4_SEEDS)
    if (
        not isinstance(outputs, list)
        or not isinstance(member_meta, list)
        or len(outputs) != expected_member_count
        or len(member_meta) != expected_member_count
    ):
        raise ValueError(
            f"v4 matrix must contain exactly {expected_member_count} replay members"
        )
    expected_poses = manifest.get("validationPoseIndices")
    if expected_poses is not None:
        expected_poses = [int(value) for value in expected_poses]
        if len(expected_poses) != expected_pose_count:
            raise ValueError("v4 matrix validationPoseIndices count disagrees with splitPoseCounts")
    members: dict[tuple[str, int], dict[str, Any]] = {}
    for meta, raw_path in zip(member_meta, outputs, strict=True):
        if not isinstance(meta, Mapping):
            raise ValueError("v4 matrix member metadata is not an object")
        variant = str(meta.get("variant", ""))
        seed = int(meta.get("seed", -1))
        key = (variant, seed)
        if key in members:
            raise ValueError(f"duplicate v4 matrix member {variant}/seed{seed}")
        path = _resolve_member_path(root, raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing v4 validation replay: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != EVALUATION_SCHEMA:
            raise ValueError(f"unexpected v4 evaluation schema: {path}")
        if payload.get("split") != "validation" or payload.get("testRead") is not False:
            raise ValueError(f"v4 replay is not validation-only: {path}")
        if str(payload.get("variant")) != variant or int(payload.get("seed", -1)) != seed:
            raise ValueError(f"v4 replay member identity mismatch: {path}")
        _validate_threshold_provenance(payload, path)
        pose_indices = [int(value) for value in payload.get("poseIndices", [])]
        checkpoint_info = meta.get("checkpointInfo")
        if not isinstance(checkpoint_info, Mapping):
            raise ValueError(f"v4 matrix member has no checkpointInfo: {path}")
        expected_checkpoint = Path(str(checkpoint_info.get("checkpoint", ""))).resolve()
        if not expected_checkpoint.is_file():
            raise FileNotFoundError(f"v4 replay checkpoint is missing: {expected_checkpoint}")
        calibration = expected_checkpoint.parent / "calibration_ready_summary.json"
        if not calibration.is_file():
            raise FileNotFoundError(f"v4 replay calibration summary is missing: {calibration}")
        expected_threshold = _finite(checkpoint_info.get("threshold"))
        contract = validate_replay_payload(
            payload,
            dataset,
            expected_poses,
            expected_variant=variant,
            expected_seed=seed,
            expected_checkpoint=expected_checkpoint,
            expected_threshold=expected_threshold,
            num_instances=int(runtime_info["numInstances"]),
        )
        if pose_indices != expected_poses or len(pose_indices) != expected_pose_count:
            raise ValueError(f"v4 replay pose order/count disagrees with matrix manifest: {path}")
        rows = payload.get("perPose")
        if not isinstance(rows, list) or len(rows) != expected_pose_count:
            raise ValueError(f"v4 replay has incomplete perPose rows: {path}")
        row_indices: list[int] = []
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("metrics"), Mapping):
                raise ValueError(f"v4 replay pose row is invalid: {path}")
            pose = int(row.get("poseIndex", -1))
            row_indices.append(pose)
            for metric in METRIC_NAMES:
                if metric not in row["metrics"]:
                    raise ValueError(f"v4 replay pose row is missing metric {metric}: {path}")
                _finite(row["metrics"][metric])
            for metric in COUNT_METRICS + ("weightedTp", "weightedGt"):
                _finite(row["metrics"].get(metric, 0.0))
        if row_indices != pose_indices:
            raise ValueError(f"v4 replay perPose order disagrees with poseIndices: {path}")
        members[key] = {
            "payload": payload,
            "rows": rows,
            "path": str(path),
            "poseIndices": pose_indices,
            "candidateReferenceCount": int(contract["candidateReferenceCount"]),
            "predictionReferenceCount": int(contract["predictionReferenceCount"]),
            "visibleReferenceCount": int(native_summary["visibleReferenceCount"]),
        }
    expected_keys = {(variant, seed) for variant in V4_FORMAL_VARIANTS for seed in V4_SEEDS}
    if set(members) != expected_keys:
        raise ValueError("v4 matrix member coverage is incomplete")
    reference_key = (V4_FORMAL_VARIANTS[0], V4_SEEDS[0])
    reference = members[reference_key]
    for key, member in members.items():
        if member["poseIndices"] != reference["poseIndices"]:
            raise ValueError(f"v4 matrix pose order is not paired for {key}")
        if member["candidateReferenceCount"] != reference["candidateReferenceCount"]:
            raise ValueError(f"v4 matrix candidate reference count is not paired for {key}")
    return members


def _member_arrays(member: Mapping[str, Any], metric_names: Iterable[str] = METRIC_NAMES) -> dict[str, np.ndarray]:
    names = (*COUNT_METRICS, "weightedTp", "weightedGt", *tuple(metric_names))
    return {
        name: np.asarray([float(row["metrics"].get(name, 0.0)) for row in member["rows"]], dtype=np.float64)
        for name in names
    }


def _gather_pose_field(
    arrays_by_seed: list[dict[str, np.ndarray]],
    field: str,
    selected_seed: np.ndarray,
    pose_indices: np.ndarray,
) -> np.ndarray:
    stacked = np.stack([arrays[field] for arrays in arrays_by_seed], axis=0)
    selected = stacked[selected_seed]
    return np.take_along_axis(selected, pose_indices, axis=2)


def _scope_bootstrap_values(
    arrays_by_seed: list[dict[str, np.ndarray]],
    scope: str,
    selected_seed: np.ndarray,
    pose_indices: np.ndarray,
    metric_names: Iterable[str] = METRIC_NAMES,
) -> dict[str, np.ndarray]:
    metric_names = tuple(metric_names)
    gathered = {
        field: _gather_pose_field(arrays_by_seed, field, selected_seed, pose_indices)
        for field in (*COUNT_METRICS, "weightedTp", "weightedGt")
    }
    if scope == "poseMacro":
        return {
            metric: _gather_pose_field(arrays_by_seed, metric, selected_seed, pose_indices).mean(axis=2)
            for metric in metric_names
        }
    totals = {name: values.sum(axis=2) for name, values in gathered.items()}
    tp, fp, fn, tn = (totals[name] for name in COUNT_METRICS)
    candidate = tp + fp + fn + tn
    gt = tp + fn
    pred = tp + fp
    recall = tp / np.maximum(1.0, gt)
    precision = tp / np.maximum(1.0, pred)
    specificity = tn / np.maximum(1.0, tn + fp)
    result: dict[str, np.ndarray] = {
        "precision": precision,
        "recall": recall,
        "weightedRecall": totals["weightedTp"] / np.maximum(1e-12, totals["weightedGt"]),
        "f1": 2.0 * precision * recall / np.maximum(1e-8, precision + recall),
        "jaccard": tp / np.maximum(1.0, tp + fp + fn),
        "accuracy": (tp + tn) / np.maximum(1.0, candidate),
        "balancedAccuracy": 0.5 * (recall + specificity),
        "specificity": specificity,
        "usefulCull": tn / np.maximum(1.0, candidate),
        "badCull": fn / np.maximum(1.0, candidate),
        "avgPredCount": pred / max(1, pose_indices.shape[2]),
        "avgCandidateCount": candidate / max(1, pose_indices.shape[2]),
        "avgGtCount": gt / max(1, pose_indices.shape[2]),
        "predOverCandidate": pred / np.maximum(1.0, candidate),
        "predOverGt": pred / np.maximum(1.0, gt),
    }
    resource_metrics = (
        "candidateGlbCount", "predictedGlbCount", "predictedGlbBytes", "glbCountReduction",
        "glbByteReduction", "downloadUtilityRecall", "glbBytesAtAchievedVisualUtility",
        "candidateGlbBytes", "requiredGlbCount", "requiredGlbBytes",
    )
    for metric in resource_metrics:
        if metric in metric_names:
            result[metric] = _gather_pose_field(arrays_by_seed, metric, selected_seed, pose_indices).mean(axis=2)
    return result


def _paired_bootstrap(
    members: Mapping[tuple[str, int], Mapping[str, Any]],
    left: str,
    right: str,
    seeds: list[int],
    metrics: Iterable[str],
    replicates: int,
    seed: int,
    metric_names: Iterable[str] = METRIC_NAMES,
) -> dict[str, Any]:
    if int(replicates) < 10000:
        raise ValueError("v4 formal paired bootstrap requires at least 10000 replicates")
    if tuple(int(value) for value in seeds) != V4_SEEDS:
        raise ValueError("v4 formal paired bootstrap requires the registered three seeds")
    if left == right:
        raise ValueError("paired bootstrap requires two distinct members")
    metric_names = tuple(metrics)
    rng = np.random.default_rng(int(seed))
    pose_count = len(members[(left, seeds[0])]["rows"])
    reference_poses = members[(left, seeds[0])].get("poseIndices")
    reference_candidate_count = members[(left, seeds[0])].get("candidateReferenceCount")
    for variant in (left, right):
        for current_seed in seeds:
            member = members.get((variant, current_seed))
            if member is None:
                raise ValueError(f"paired bootstrap input is missing {variant}/seed{current_seed}")
            if len(member["rows"]) != pose_count:
                raise ValueError("paired bootstrap members have different pose counts")
            if (
                member.get("poseIndices") != reference_poses
                or member.get("candidateReferenceCount") != reference_candidate_count
            ):
                raise ValueError("paired bootstrap members do not share pose/candidate structure")
    left_arrays = [_member_arrays(members[(left, current_seed)], metric_names) for current_seed in seeds]
    right_arrays = [_member_arrays(members[(right, current_seed)], metric_names) for current_seed in seeds]
    boot = {f"poseMacro.{metric}": np.empty((replicates,), dtype=np.float64) for metric in metric_names}
    boot.update({f"aggregate.{metric}": np.empty((replicates,), dtype=np.float64) for metric in metric_names})
    observed: dict[str, float] = {}
    for scope in ("poseMacro", "aggregate"):
        for metric in metric_names:
            right_value = [
                _summarize_rows(members[(right, current_seed)]["rows"], metric_names=metric_names)[scope][metric]
                for current_seed in seeds
            ]
            left_value = [
                _summarize_rows(members[(left, current_seed)]["rows"], metric_names=metric_names)[scope][metric]
                for current_seed in seeds
            ]
            observed[f"{scope}.{metric}"] = float(np.mean(np.asarray(right_value) - np.asarray(left_value)))
    batch = 128
    for start in range(0, int(replicates), batch):
        end = min(int(replicates), start + batch)
        count = end - start
        selected_seed = rng.integers(0, len(seeds), size=(count, len(seeds)), endpoint=False)
        pose_indices = rng.integers(0, pose_count, size=(count, len(seeds), pose_count), endpoint=False)
        for scope in ("poseMacro", "aggregate"):
            left_values = _scope_bootstrap_values(left_arrays, scope, selected_seed, pose_indices, metric_names)
            right_values = _scope_bootstrap_values(right_arrays, scope, selected_seed, pose_indices, metric_names)
            for metric in metric_names:
                boot[f"{scope}.{metric}"][start:end] = (right_values[metric] - left_values[metric]).mean(axis=1)
    effects: dict[str, Any] = {}
    for key, values in boot.items():
        interval = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        effects[key] = {
            "meanDelta": float(observed[key]),
            "ci95": interval,
            "direction": "positive" if observed[key] > 0.0 else "negative" if observed[key] < 0.0 else "zero",
            "crossesZero": bool(interval[0] <= 0.0 <= interval[1]),
            "bootstrapReplicates": int(replicates),
            "clusterUnit": BOOTSTRAP_CLUSTER_UNIT,
            "paired": True,
        }
    return effects


def _comparison_pairs(variants: list[str]) -> list[tuple[str, str, str]]:
    if "full" not in variants:
        raise ValueError("v4 comparison matrix must contain full")
    return [
        (f"full_minus_{reference}", reference, "full")
        for reference in variants
        if reference != "full"
    ]


def _image_member_name(variant: str, seed: int) -> str:
    return f"{variant}_seed{int(seed)}"


def _image_metric_summary(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("formal image member has no sample rows")
    rates = {
        field: np.asarray([_finite(row["imageMetrics"][field]) for row in rows], dtype=np.float64)
        for field in IMAGE_RATE_FIELDS
    }
    totals = {
        field: float(sum(_finite(row["imageMetrics"].get(field, 0.0)) for row in rows))
        for field in (
            "totalPixels", "validReferencePixels", "errorPixels", "missPixels",
            "wrongInstancePixels", "extraPixels",
        )
    }
    return {
        "PER": _safe_div(totals["errorPixels"], totals["validReferencePixels"]),
        "missPixelRate": _safe_div(totals["missPixels"], totals["validReferencePixels"]),
        "wrongInstancePixelRate": _safe_div(
            totals["wrongInstancePixels"], totals["validReferencePixels"]
        ),
        "extraPixelRateOverImage": _safe_div(totals["extraPixels"], totals["totalPixels"]),
        "meanMissPixelRate": float(np.mean(rates["missPixelRate"])),
        "p95MissPixelRate": float(np.quantile(rates["missPixelRate"], 0.95)),
        "meanWrongInstancePixelRate": float(np.mean(rates["wrongInstancePixelRate"])),
        "p95WrongInstancePixelRate": float(np.quantile(rates["wrongInstancePixelRate"], 0.95)),
        "meanExtraPixelRateOverImage": float(np.mean(rates["extraPixelRateOverImage"])),
        "p95ExtraPixelRateOverImage": float(np.quantile(rates["extraPixelRateOverImage"], 0.95)),
        **totals,
        "evaluatedSubposeCount": float(len(rows)),
    }


def _image_pose_arrays(
    rows: list[Mapping[str, Any]],
    pose_indices: list[int],
) -> dict[str, np.ndarray]:
    grouped: dict[int, list[Mapping[str, Any]]] = {int(pose): [] for pose in pose_indices}
    for row in rows:
        pose = int(row.get("viewcellRow", -1))
        if pose not in grouped:
            raise ValueError(f"formal image sample references a non-validation view-cell: {pose}")
        grouped[pose].append(row)
    if any(not grouped[pose] for pose in pose_indices):
        missing = [pose for pose in pose_indices if not grouped[pose]][:8]
        raise ValueError(f"formal image samples do not cover every validation view-cell: {missing}")
    width = max(len(grouped[pose]) for pose in pose_indices)
    arrays = {
        field: np.full((len(pose_indices), width), np.nan, dtype=np.float64)
        for field in IMAGE_RATE_FIELDS
    }
    for pose_row, pose in enumerate(pose_indices):
        samples = sorted(grouped[pose], key=lambda row: str(row.get("sampleId", "")))
        for field in IMAGE_RATE_FIELDS:
            values = [_finite(sample["imageMetrics"][field]) for sample in samples]
            arrays[field][pose_row, : len(values)] = values
    return arrays


def _load_image_members(
    image_root: Path,
    members: Mapping[tuple[str, int], Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"formal image root is missing: {image_root}")
    expected_pose_indices = [int(value) for value in manifest.get("validationPoseIndices", [])]
    if not expected_pose_indices:
        raise ValueError("matrix manifest has no validation pose sequence for image pairing")
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    reference_sample_keys: list[tuple[str, int, int]] | None = None
    for variant in V4_FORMAL_VARIANTS:
        for seed in V4_SEEDS:
            key = (variant, seed)
            name = _image_member_name(variant, seed)
            output = image_root / name
            summary_path = output / "summary.json"
            render_path = output / "true_glb_render" / "render_summary.json"
            sample_path = output / "true_glb_render" / "sample_image_metrics.json"
            for path in (summary_path, render_path, sample_path):
                if not path.is_file():
                    raise FileNotFoundError(f"formal image artifact is missing: {path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            render = json.loads(render_path.read_text(encoding="utf-8"))
            rows = json.loads(sample_path.read_text(encoding="utf-8"))
            if summary.get("schema") != IMAGE_SUMMARY_SCHEMA or summary.get("modelName") != name:
                raise ValueError(f"formal image member identity/schema is invalid: {summary_path}")
            if summary.get("split") != "validation" or summary.get("formalImageEvaluationReady") is not True:
                raise ValueError(f"formal image member is not complete validation output: {summary_path}")
            if render.get("schema") != IMAGE_RENDER_SCHEMA or render.get("formalImageEvaluationReady") is not True:
                raise ValueError(f"formal image renderer output is not complete: {render_path}")
            gpu_gate = render.get("gpuGate") or ((summary.get("renderer") or {}).get("gpuGate"))
            if not isinstance(gpu_gate, Mapping) or gpu_gate.get("hardware") is not True:
                raise ValueError(f"formal image member failed the hardware GPU gate: {render_path}")
            member = members[key]
            payload = member["payload"]
            threshold = _finite(summary.get("threshold"))
            if not np.isclose(threshold, _finite(payload.get("threshold")), rtol=0.0, atol=1e-7):
                raise ValueError(f"formal image threshold disagrees with validation replay: {summary_path}")
            model_spec = summary.get("modelSpec")
            runtime_bundle = next(
                (
                    item.get("runtimeBundle")
                    for item in manifest.get("members", [])
                    if isinstance(item, Mapping)
                    and str(item.get("variant")) == variant
                    and int(item.get("seed", -1)) == seed
                ),
                None,
            )
            if not isinstance(model_spec, Mapping) or not isinstance(runtime_bundle, Mapping):
                raise ValueError(f"formal image member has no model/bundle provenance: {summary_path}")
            if Path(str(model_spec.get("checkpoint", ""))).resolve() != Path(str(payload["checkpoint"])).resolve():
                raise ValueError(f"formal image checkpoint disagrees with validation replay: {summary_path}")
            if Path(str(model_spec.get("runtime_features", ""))).resolve() != Path(
                str(runtime_bundle.get("runtimeFeatures", ""))
            ).resolve():
                raise ValueError(f"formal image runtime table disagrees with checkpoint bundle: {summary_path}")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"formal image sample file is empty or invalid: {sample_path}")
            sample_keys = [
                (str(row.get("sampleId", "")), int(row.get("viewcellRow", -1)), int(row.get("poseIndex", -1)))
                for row in rows
            ]
            if len(set(sample_keys)) != len(sample_keys) or any(not value[0] for value in sample_keys):
                raise ValueError(f"formal image sample identity is invalid: {sample_path}")
            if reference_sample_keys is None:
                reference_sample_keys = sample_keys
            elif sample_keys != reference_sample_keys:
                raise ValueError("formal image members do not share the same ordered view-cell/subpose samples")
            for row in rows:
                metrics = row.get("imageMetrics")
                if not isinstance(metrics, Mapping):
                    raise ValueError(f"formal image sample has no metrics: {sample_path}")
                for field in IMAGE_RATE_FIELDS:
                    value = _finite(metrics.get(field))
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(f"formal image rate is outside [0,1]: {sample_path}/{field}")
            computed = _image_metric_summary(rows)
            recorded = render.get("imageMetrics") or {}
            for metric in IMAGE_REPORT_METRICS:
                if not np.isclose(computed[metric], _finite(recorded.get(metric)), rtol=0.0, atol=1e-10):
                    raise ValueError(f"formal image summary disagrees with sample metrics: {render_path}/{metric}")
            arrays = _image_pose_arrays(rows, expected_pose_indices)
            loaded[key] = {
                "summary": computed,
                "poseArrays": arrays,
                "poseIndices": expected_pose_indices,
                "sampleKeys": sample_keys,
                "sampleCount": len(rows),
                "gpuGate": dict(gpu_gate),
                "summaryPath": str(summary_path.resolve()),
                "renderSummaryPath": str(render_path.resolve()),
                "sampleMetricsPath": str(sample_path.resolve()),
                "runtimeFeatures": runtime_bundle.get("runtimeFeatures"),
                "testRead": False,
            }
    return loaded


def _paired_image_bootstrap(
    image_members: Mapping[tuple[str, int], Mapping[str, Any]],
    left: str,
    right: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if int(replicates) < 10000:
        raise ValueError("formal image bootstrap requires at least 10000 replicates")
    pose_count = len(image_members[(left, V4_SEEDS[0])]["poseIndices"])
    rng = np.random.default_rng(int(seed))
    effects: dict[str, np.ndarray] = {
        metric: np.empty((int(replicates),), dtype=np.float64)
        for metric in IMAGE_REPORT_METRICS
    }
    observed: dict[str, float] = {}
    field_to_names = {
        "missPixelRate": ("meanMissPixelRate", "p95MissPixelRate"),
        "wrongInstancePixelRate": (
            "meanWrongInstancePixelRate", "p95WrongInstancePixelRate",
        ),
        "extraPixelRateOverImage": (
            "meanExtraPixelRateOverImage", "p95ExtraPixelRateOverImage",
        ),
    }
    for metric in IMAGE_REPORT_METRICS:
        observed[metric] = float(np.mean([
            image_members[(right, current_seed)]["summary"][metric]
            - image_members[(left, current_seed)]["summary"][metric]
            for current_seed in V4_SEEDS
        ]))
    left_arrays = {
        field: np.stack(
            [image_members[(left, current_seed)]["poseArrays"][field] for current_seed in V4_SEEDS],
            axis=0,
        )
        for field in IMAGE_RATE_FIELDS
    }
    right_arrays = {
        field: np.stack(
            [image_members[(right, current_seed)]["poseArrays"][field] for current_seed in V4_SEEDS],
            axis=0,
        )
        for field in IMAGE_RATE_FIELDS
    }
    batch_size = 32
    for start in range(0, int(replicates), batch_size):
        end = min(int(replicates), start + batch_size)
        count = end - start
        selected_seed = rng.integers(0, len(V4_SEEDS), size=(count, len(V4_SEEDS)), endpoint=False)
        selected_pose = rng.integers(
            0,
            pose_count,
            size=(count, len(V4_SEEDS), pose_count),
            endpoint=False,
        )
        for field, (mean_name, p95_name) in field_to_names.items():
            side_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for side, arrays in (("left", left_arrays[field]), ("right", right_arrays[field])):
                selected = arrays[selected_seed]
                gathered = np.take_along_axis(selected, selected_pose[..., None], axis=2)
                flattened = gathered.reshape(count, len(V4_SEEDS), -1)
                mean_values = np.nanmean(flattened, axis=2).mean(axis=1)
                p95_values = np.nanquantile(flattened, 0.95, axis=2).mean(axis=1)
                side_values[side] = (mean_values, p95_values)
            effects[mean_name][start:end] = side_values["right"][0] - side_values["left"][0]
            effects[p95_name][start:end] = side_values["right"][1] - side_values["left"][1]
    result: dict[str, Any] = {}
    for metric, values in effects.items():
        interval = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        result[metric] = {
            "meanDelta": observed[metric],
            "ci95": interval,
            "direction": "positive" if observed[metric] > 0.0 else "negative" if observed[metric] < 0.0 else "zero",
            "crossesZero": bool(interval[0] <= 0.0 <= interval[1]),
            "desiredDirection": "decrease",
            "bootstrapReplicates": int(replicates),
            "clusterUnit": BOOTSTRAP_CLUSTER_UNIT,
            "paired": True,
            "statistic": "mean across resampled seed clusters; mean or p95 across resampled subposes",
        }
    return result


def _resource_number(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in row and row[name] is not None:
            return _finite(row[name])
    raise ValueError(f"resource threshold row is missing one of {names}")


def _load_resource_evaluation(
    resource_root: Path,
    members: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Load the frozen-threshold GLB resource evaluation into the v4 summary.

    The resource evaluator reports one threshold row per model plus richer
    budget workpoints.  This loader binds that row to the checkpoint-owned
    validation replay threshold and rejects partial or non-validation output.
    """

    summary_path = resource_root.resolve() / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"formal resource summary is missing: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"formal resource summary is not an object: {summary_path}")
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError(f"formal resource summary has no meta: {summary_path}")
    if meta.get("requestedSplit") != "validation" or meta.get("split") != "validation":
        raise ValueError(f"formal resource summary is not validation-only: {summary_path}")
    if meta.get("testEvaluationCount") not in (None, 0):
        raise ValueError(f"formal resource summary contains test evaluation provenance: {summary_path}")
    raw_summaries = payload.get("summaries")
    if not isinstance(raw_summaries, list):
        raise ValueError(f"formal resource summary has no summaries list: {summary_path}")
    by_name: dict[str, Mapping[str, Any]] = {}
    for item in raw_summaries:
        if not isinstance(item, Mapping) or not str(item.get("name", "")):
            raise ValueError(f"formal resource summary contains an invalid model entry: {summary_path}")
        name = str(item["name"])
        if name in by_name:
            raise ValueError(f"formal resource summary contains duplicate model {name}: {summary_path}")
        by_name[name] = item
    expected_names = {
        f"{variant}_seed{int(seed)}"
        for variant in V4_FORMAL_VARIANTS
        for seed in V4_SEEDS
    }
    if set(by_name) != expected_names:
        missing = sorted(expected_names - set(by_name))
        extra = sorted(set(by_name) - expected_names)
        raise ValueError(f"formal resource member coverage is incomplete; missing={missing}, extra={extra}")

    normalized: dict[str, dict[str, Any]] = {}
    for (variant, seed), member in members.items():
        name = _image_member_name(variant, seed)
        resource = by_name[name]
        threshold = _finite(member["payload"].get("threshold"))
        rows = resource.get("thresholdRows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"formal resource member has no threshold rows: {name}")
        matching = [
            row for row in rows
            if isinstance(row, Mapping)
            and np.isclose(_finite(row.get("threshold")), threshold, rtol=0.0, atol=1e-7)
        ]
        if len(matching) != 1:
            raise ValueError(f"formal resource threshold does not match checkpoint threshold: {name}")
        row = matching[0]
        candidate_glb_count = _resource_number(row, "avg_candidate_glb_count", "avgCandidateGlbCount")
        predicted_glb_count = _resource_number(row, "avg_pred_glb_count", "avgPredGlbCount")
        candidate_glb_bytes = _resource_number(row, "avg_candidate_glb_bytes", "avgCandidateGlbBytes")
        predicted_glb_bytes = _resource_number(row, "avg_pred_glb_bytes", "avgPredGlbBytes")
        normalized[name] = {
            "status": "available",
            "modelName": name,
            "requestedSplit": meta.get("requestedSplit"),
            "split": meta.get("split"),
            "evaluatedPoses": int(meta.get("evaluatedPoses", row.get("eval_pose_count", 0))),
            "threshold": threshold,
            "thresholdRow": dict(row),
            "averageCandidateGlbCount": candidate_glb_count,
            "averagePredictedGlbCount": predicted_glb_count,
            "averageCandidateGlbBytes": candidate_glb_bytes,
            "averagePredictedGlbBytes": predicted_glb_bytes,
            "glbCountReduction": 1.0 - _safe_div(predicted_glb_count, candidate_glb_count),
            "glbByteReduction": 1.0 - _safe_div(predicted_glb_bytes, candidate_glb_bytes),
            "workpoints": resource.get("workpoints", {}),
            "scoreModes": resource.get("scoreModes", {}),
            "testRead": False,
        }
    return {
        "status": "available",
        "schema": "pvs-bounded-relation-prior-instance-calibrated-v4-resource-evaluation-v1",
        "root": str(resource_root.resolve()),
        "summaryPath": str(summary_path.resolve()),
        "requestedSplit": meta.get("requestedSplit"),
        "split": meta.get("split"),
        "evaluatedPoses": int(meta.get("evaluatedPoses", 0)),
        "memberCount": len(normalized),
        "members": normalized,
        "meta": dict(meta),
        "testRead": False,
    }


def _build_summary(
    root: Path,
    replicates: int,
    lcb_replicates: int,
    image_root: Path | None = None,
    resource_root: Path | None = None,
) -> dict[str, Any]:
    manifest_path = root / "matrix_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"v4 matrix manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = _load_members(root, manifest)
    image_members = _load_image_members(image_root.resolve(), members, manifest) if image_root else None
    resource_evaluation = (
        _load_resource_evaluation(resource_root.resolve(), members)
        if resource_root is not None
        else {"status": "not_available", "testRead": False}
    )
    resource_members = resource_evaluation.get("members", {})
    if int(replicates) < 10000:
        raise ValueError("v4 formal summary requires at least 10000 paired bootstrap replicates")
    reference = members[(V4_FORMAL_VARIANTS[0], V4_SEEDS[0])]
    comparisons: dict[str, Any] = {}
    for index, (name, left, right) in enumerate(_comparison_pairs(list(V4_FORMAL_VARIANTS))):
        comparisons[name] = {
            "left": left,
            "right": right,
            "contrast": "full_minus_ablation" if name.startswith("full_minus_") else "right_minus_left",
            "metrics": _paired_bootstrap(
                members, left, right, list(V4_SEEDS), METRIC_NAMES, int(replicates), 20260814 + index, METRIC_NAMES
            ),
        }
        if image_members is not None and name.startswith("full_minus_"):
            comparisons[name]["metrics"].update({
                f"image.{metric}": effect
                for metric, effect in _paired_image_bootstrap(
                    image_members,
                    left,
                    right,
                    int(replicates),
                    20261814 + index,
                ).items()
            })
    member_summary: dict[str, Any] = {}
    calibration_safe_members: list[str] = []
    validation_safe_members: list[str] = []
    diagnostic_members: list[str] = []
    for variant in V4_FORMAL_VARIANTS:
        for seed in V4_SEEDS:
            member = members[(variant, seed)]
            member_metrics = _summarize_rows(
                member["rows"],
                lcb_replicates=int(lcb_replicates),
                lcb_seed=seed + 7013,
                metric_names=METRIC_NAMES,
            )
            key = f"{variant}:{seed}"
            payload = member["payload"]
            source = payload.get("thresholdSource") or {}
            calibration_safe = bool(source.get("safeWorkpoint", False))
            validation_safe = bool(
                member_metrics["safety"]["weightedRecallSatisfied"]
                and member_metrics["safety"]["weightedRecallLowerConfidenceBoundSatisfied"]
            )
            (calibration_safe_members if calibration_safe else diagnostic_members).append(key)
            if calibration_safe and validation_safe:
                validation_safe_members.append(key)
            member_summary[key] = {
                "variant": variant,
                "seed": seed,
                "path": member["path"],
                "checkpoint": payload.get("checkpoint"),
                "threshold": payload.get("threshold"),
                "thresholdSource": source,
                "safeWorkpoint": calibration_safe,
                "calibrationSafeWorkpoint": calibration_safe,
                "validationSafetyPassed": validation_safe,
                "memberClass": (
                    "calibration_and_validation_safe"
                    if calibration_safe and validation_safe
                    else "calibration_safe_validation_unsafe"
                    if calibration_safe
                    else "diagnostic_non_safe"
                ),
                "poseCount": member_metrics["poseCount"],
                "poseMacro": member_metrics["poseMacro"],
                "aggregate": member_metrics["aggregate"],
                "safety": member_metrics["safety"],
                "runtime": payload.get("runtime", {}),
                "imageMetrics": (
                    {
                        "status": "available",
                        **image_members[(variant, seed)]["summary"],
                        "gpuGate": image_members[(variant, seed)]["gpuGate"],
                        "sampleCount": image_members[(variant, seed)]["sampleCount"],
                        "source": {
                            name: value
                            for name, value in image_members[(variant, seed)].items()
                            if name.endswith("Path") or name in {"runtimeFeatures"}
                        },
                        "testRead": False,
                    }
                    if image_members is not None
                    else payload.get("imageMetrics", {})
                ),
                "resourceMetrics": resource_members.get(
                    _image_member_name(variant, seed),
                    {"status": "not_available", "testRead": False},
                ),
                "candidateReferenceCount": member.get("candidateReferenceCount"),
                "predictionReferenceCount": member.get("predictionReferenceCount"),
                "visibleReferenceCount": member.get("visibleReferenceCount"),
            }
    return {
        "schema": SUMMARY_SCHEMA,
        "version": 1,
        "split": "validation",
        "testRead": False,
        "manifest": str(manifest_path.resolve()),
        "mode": manifest.get("mode"),
        "variants": list(V4_FORMAL_VARIANTS),
        "seeds": list(V4_SEEDS),
        "poseCount": len(reference["poseIndices"]),
        "poseIndices": reference["poseIndices"],
        "candidateReferenceCount": reference["candidateReferenceCount"],
        "visibleReferenceCount": reference["visibleReferenceCount"],
        "relationArtifacts": manifest.get("relationArtifacts", {}),
        "inputProvenance": manifest.get("inputProvenance", {}),
        "candidateSemantics": "stored native back-camera candidate CSR; no GT union",
        "members": member_summary,
        "memberStatistics": {
            "calibrationSafe": {
                "memberKeys": calibration_safe_members,
                "count": len(calibration_safe_members),
            },
            "calibrationAndValidationSafe": {
                "memberKeys": validation_safe_members,
                "count": len(validation_safe_members),
            },
            "diagnostic_non_safe": {"memberKeys": diagnostic_members, "count": len(diagnostic_members)},
            "safetyDecisionPopulation": (
                "calibration-safe members must additionally pass validation aggregate weighted recall "
                "and its one-sided lower confidence bound; diagnostic members cannot establish safety"
            ),
        },
        "comparisons": comparisons,
        "bootstrap": {
            "replicates": int(replicates),
            "confidence": 0.95,
            "paired": True,
            "clusterUnit": BOOTSTRAP_CLUSTER_UNIT,
            "outerUnit": "seed cluster resampled with replacement",
            "innerUnit": "validation pose IDs resampled with replacement within each selected seed",
        },
        "metricDefinitions": {
            "TP": "P intersect G",
            "FP": "P minus G",
            "FN": "G minus P",
            "TN": "C minus (P union G)",
            "poseMacro": "compute each pose first, then average",
            "aggregate": "sum sufficient statistics after inner pose resampling",
            "weightedRecall": "sum visible weight found by P divided by total visible weight",
            "usefulCull": "TN / candidate",
            "badCull": "FN / candidate",
            "balancedAccuracy": "(recall + specificity) / 2",
            "instanceAccuracy": "(TP + TN) / candidate",
        },
        "imageEvaluation": (
            {
                "status": "available",
                "root": str(image_root.resolve()),
                "memberCount": len(image_members),
                "sampleIdentity": "same ordered view-cell/subpose keys for all variants and seeds",
                "bootstrap": {
                    "replicates": int(replicates),
                    "paired": True,
                    "clusterUnit": BOOTSTRAP_CLUSTER_UNIT,
                },
                "testRead": False,
            }
            if image_members is not None
            else {"status": "not_available", "testRead": False}
        ),
        "resourceEvaluation": resource_evaluation,
        "unavailableMetrics": (
            {"browserWebGpuLatency": "not_available until formal hardware browser replay completes"}
            if image_members is not None
            else {
                "missPixelRate": "not_available until formal hardware Color-ID evaluation completes",
                "wrongIdPixelRate": "not_available until formal hardware Color-ID evaluation completes",
                "extraPixelRate": "not_available until formal hardware Color-ID evaluation completes",
                "browserWebGpuLatency": "not_available until formal hardware browser replay completes",
            }
        ),
    }


def build_summary(
    root: Path,
    replicates: int,
    lcb_replicates: int,
    image_root: Path | None = None,
    resource_root: Path | None = None,
) -> dict[str, Any]:
    return _build_summary(
        root.resolve(),
        int(replicates),
        int(lcb_replicates),
        image_root.resolve() if image_root is not None else None,
        resource_root.resolve() if resource_root is not None else None,
    )


def _build_v3_summary(
    root: Path,
    replicates: int,
    lcb_replicates: int,
    image_root: Path | None = None,
    resource_root: Path | None = None,
) -> dict[str, Any]:
    return _build_summary(root, int(replicates), int(lcb_replicates), image_root, resource_root)


def self_test() -> dict[str, Any]:
    rows = [
        {
            "poseIndex": pose,
            "metrics": {
                **{metric: 0.5 for metric in METRIC_NAMES},
                "tp": 1.0,
                "fp": 1.0,
                "fn": 0.0,
                "tn": 2.0,
                "weightedTp": 1.0,
                "weightedGt": 1.0,
                "candidateCount": 4.0,
            },
        }
        for pose in range(4)
    ]
    summary = _summarize_rows(rows)
    if abs(summary["aggregate"]["accuracy"] - 0.75) > 1e-9:
        raise AssertionError("v4 aggregate accuracy self-test failed")
    return {"status": "passed", "schema": SUMMARY_SCHEMA}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--lcb-replicates", type=int, default=10000)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--resource-root", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if args.root is None or args.output is None:
        raise ValueError("--root and --output are required unless --self-test is used")
    if args.bootstrap_replicates < 10000:
        raise ValueError("--bootstrap-replicates must be at least 10000")
    payload = build_summary(
        args.root,
        int(args.bootstrap_replicates),
        int(args.lcb_replicates),
        args.image_root,
        args.resource_root,
    )
    _write_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "poseCount": payload["poseCount"], "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
