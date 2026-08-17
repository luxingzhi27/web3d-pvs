#!/usr/bin/env python3
"""Summarize validation replays for the 2026-08-13 experiment line.

The input is a matrix manifest plus one calibration-frozen validation replay
per member.  This script performs no model inference and never reads test.
It verifies pose/candidate identity, reports pose-macro and all-pose aggregate
metrics, and computes paired contrasts with the registered hierarchy:
outer resampling of training seeds followed by inner resampling of the same
validation pose IDs within each selected seed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SUMMARY_SCHEMA = "pvs-hierarchical-relation-survival-integrated-validation-summary-v1"
BOOTSTRAP_CLUSTER_UNIT = "outer seed cluster, inner validation pose resampling"
FORMAL_VARIANTS = (
    "full",
    "without_hierarchical_relation",
    "shuffled_relation_source",
    "without_viewcell_integration",
    "without_threshold_aligned_utility",
)
METRIC_NAMES = (
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
    "predOverCandidate",
    "predOverGt",
    "candidateGlbCount",
    "predictedGlbCount",
    "predictedGlbBytes",
    "glbCountReduction",
    "glbByteReduction",
    "downloadUtilityRecall",
    "glbBytesAtAchievedVisualUtility",
    "candidateGlbBytes",
    "requiredGlbCount",
    "requiredGlbBytes",
    "downloadNdcgAt20",
    "downloadNdcgAt50",
    "downloadNdcgAt100",
    "utilityNdcgAt20",
    "utilityNdcgAt50",
    "utilityNdcgAt100",
    "visibilityBaselineNdcgAt20",
    "visibilityBaselineNdcgAt50",
    "visibilityBaselineNdcgAt100",
    "downloadUtilityRecallFraction5pct",
    "downloadUtilityRecallFraction10pct",
    "downloadUtilityRecallFraction20pct",
    "downloadUtilityRecallFraction40pct",
    "downloadUtilityRecallFixed10MiB",
    "downloadUtilityRecallFixed25MiB",
    "downloadUtilityRecallFixed50MiB",
    "downloadRequiredGlbRecallAt100",
)
COUNT_METRICS = ("tp", "fp", "fn", "tn")
IMAGE_METRICS = ("missPixelRate", "wrongIdPixelRate", "extraPixelRate")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
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


def _summarize_rows(
    rows: list[Mapping[str, Any]],
    lcb_replicates: int = 0,
    lcb_seed: int = 0,
    metric_names: Iterable[str] = METRIC_NAMES,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty validation pose list")
    metric_names = tuple(metric_names)
    for row in rows:
        if not isinstance(row.get("metrics"), Mapping):
            raise ValueError("validation pose row has no metrics")
    pose_macro = {
        metric: float(np.mean([_finite(row["metrics"].get(metric, 0.0)) for row in rows]))
        for metric in metric_names
    }
    totals = {name: float(sum(_finite(row["metrics"].get(name, 0.0)) for row in rows)) for name in COUNT_METRICS}
    count = float(len(rows))
    aggregate = {metric: _metric_from_counts(totals, metric) for metric in (
        "precision", "recall", "f1", "jaccard", "accuracy", "balancedAccuracy", "specificity",
        "usefulCull", "badCull", "avgPredCount", "avgCandidateCount", "avgGtCount",
        "predOverCandidate", "predOverGt",
    )}
    # Count-derived metrics above use the pooled TP/FP/FN/TN totals.  The
    # ``avg*Count`` fields are explicitly pose averages, so divide their
    # pooled counts by the number of validation poses before serializing.
    aggregate["avgPredCount"] = _safe_div(totals["tp"] + totals["fp"], count)
    aggregate["avgCandidateCount"] = _safe_div(sum(
        _finite(row["metrics"].get("candidateCount", 0.0))
        for row in rows
    ), count)
    aggregate["avgGtCount"] = _safe_div(totals["tp"] + totals["fn"], count)
    weighted_tp = float(sum(_finite(row["metrics"].get("weightedTp", 0.0)) for row in rows))
    weighted_gt = float(sum(_finite(row["metrics"].get("weightedGt", 0.0)) for row in rows))
    aggregate["weightedRecall"] = _safe_div(weighted_tp, weighted_gt, 1.0)
    for metric in (
        "candidateGlbCount", "predictedGlbCount", "predictedGlbBytes", "glbCountReduction",
        "glbByteReduction", "downloadUtilityRecall",
        "glbBytesAtAchievedVisualUtility",
        "candidateGlbBytes", "requiredGlbCount", "requiredGlbBytes",
        "downloadNdcgAt20", "downloadNdcgAt50", "downloadNdcgAt100",
        "utilityNdcgAt20", "utilityNdcgAt50", "utilityNdcgAt100",
        "visibilityBaselineNdcgAt20", "visibilityBaselineNdcgAt50", "visibilityBaselineNdcgAt100",
        "downloadUtilityRecallFraction5pct", "downloadUtilityRecallFraction10pct",
        "downloadUtilityRecallFraction20pct", "downloadUtilityRecallFraction40pct",
        "downloadUtilityRecallFixed10MiB", "downloadUtilityRecallFixed25MiB",
        "downloadUtilityRecallFixed50MiB", "downloadRequiredGlbRecallAt100",
    ):
            aggregate[metric] = float(np.mean([_finite(row["metrics"].get(metric, 0.0)) for row in rows]))
    aggregate.update({name: value / count for name, value in totals.items()})
    aggregate["weightedTp"] = weighted_tp / count
    aggregate["weightedGt"] = weighted_gt / count
    weighted_lcb = _weighted_lcb_from_rows(rows, int(lcb_replicates), int(lcb_seed)) if lcb_replicates > 0 else None
    pose_macro_lcb = _pose_macro_lcb_from_rows(rows, int(lcb_replicates), int(lcb_seed) + 100000) if lcb_replicates > 0 else None
    for scope in (pose_macro, aggregate):
        scope["weightedRecallLowerConfidenceBound"] = weighted_lcb if scope is aggregate else pose_macro_lcb
    aggregate["aggregateWeightedRecallLowerConfidenceBound"] = weighted_lcb
    aggregate["poseMacroWeightedRecallLowerConfidenceBound"] = pose_macro_lcb
    pose_macro["aggregateWeightedRecallLowerConfidenceBound"] = weighted_lcb
    pose_macro["poseMacroWeightedRecallLowerConfidenceBound"] = pose_macro_lcb
    for scope in (pose_macro, aggregate):
        scope["safetyFactor"] = min(1.0, float(scope.get("recall", 0.0)) / 0.95) * min(1.0, float(scope.get("weightedRecall", 0.0)) / 0.99)
        scope["safetyAdjustedUsefulCull"] = float(scope.get("usefulCull", 0.0)) * scope["safetyFactor"]
    return {
        "poseCount": len(rows),
        "poseMacro": pose_macro,
        "aggregate": aggregate,
        "counts": totals,
        "safety": {
            "weightedRecallFloor": 0.99,
            "weightedRecallLowerConfidenceBoundFloor": 0.99,
            "weightedRecallSatisfied": bool(aggregate["weightedRecall"] > 0.99),
            "weightedRecallLowerConfidenceBoundSatisfied": bool(weighted_lcb is not None and weighted_lcb > 0.99),
            "gateScope": "aggregate",
        },
    }


def _weighted_lcb_from_rows(rows: list[Mapping[str, Any]], replicates: int, seed: int) -> float:
    if not rows:
        return 0.0
    tp = np.asarray([float(row["metrics"].get("weightedTp", 0.0)) for row in rows], dtype=np.float64)
    gt = np.asarray([float(row["metrics"].get("weightedGt", 0.0)) for row in rows], dtype=np.float64)
    valid = np.isfinite(tp) & np.isfinite(gt) & (gt > 1e-12)
    tp = tp[valid]
    gt = gt[valid]
    if tp.size == 0:
        return 1.0
    if replicates <= 0:
        return 0.0
    rng = np.random.default_rng(int(seed))
    batch = max(1, min(int(replicates), 2_000_000 // max(1, len(tp))))
    values = np.empty((int(replicates),), dtype=np.float64)
    for start in range(0, int(replicates), batch):
        end = min(int(replicates), start + batch)
        indices = rng.integers(0, len(tp), size=(end - start, len(tp)), endpoint=False)
        numerator = tp[indices].sum(axis=1)
        denominator = gt[indices].sum(axis=1)
        values[start:end] = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=denominator > 1e-12,
        )
    return float(np.quantile(values, 0.05))


def _pose_macro_lcb_from_rows(rows: list[Mapping[str, Any]], replicates: int, seed: int) -> float:
    values = np.asarray(
        [float(row["metrics"].get("weightedRecall", 0.0)) for row in rows],
        dtype=np.float64,
    )
    if values.size == 0:
        return 0.0
    if replicates <= 0:
        return 0.0
    rng = np.random.default_rng(int(seed))
    batch = max(1, min(int(replicates), 2_000_000 // max(1, values.size)))
    samples = np.empty((int(replicates),), dtype=np.float64)
    for start in range(0, int(replicates), batch):
        end = min(int(replicates), start + batch)
        indices = rng.integers(0, values.size, size=(end - start, values.size), endpoint=False)
        samples[start:end] = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.05))


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "matrix_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("testRead") is not False:
        raise ValueError("matrix manifest is not test-free")
    if manifest.get("mode") != "formal80_validation_replay":
        raise ValueError("matrix manifest is not the formal80 validation replay")
    if int(manifest.get("epochs", -1)) != 80:
        raise ValueError("matrix manifest does not register epoch 80")
    if tuple(manifest.get("variants", {})) != FORMAL_VARIANTS:
        raise ValueError("matrix manifest does not contain exactly the current five variants")
    if tuple(int(value) for value in manifest.get("seeds", [])) != (20260801, 20260802, 20260803):
        raise ValueError("matrix manifest does not contain the registered three seeds")
    if not isinstance(manifest.get("inputProvenance"), Mapping):
        raise ValueError("matrix manifest has no evaluation input provenance")
    return manifest


def _validate_threshold_provenance(payload: Mapping[str, Any], path: Path) -> None:
    source = payload.get("thresholdSource")
    if not isinstance(source, Mapping):
        raise ValueError(f"validation replay has no structured thresholdSource: {path}")
    protocol = str(source.get("protocol", ""))
    if protocol not in {"calibration_ready_pre_test", "diagnostic_calibration_unsafe"}:
        raise ValueError(f"validation replay threshold was not selected on calibration: {path}")
    if str(source.get("selectionSplit", "")) != "calibration":
        raise ValueError(f"validation replay threshold selection split is not calibration: {path}")
    if source.get("selectedFromTest") is not False:
        raise ValueError(f"validation replay threshold claims test selection: {path}")
    try:
        test_evaluations = int(source.get("testEvaluationCount", -1))
    except (TypeError, ValueError):
        test_evaluations = -1
    if test_evaluations != 0:
        raise ValueError(f"validation replay threshold has test evaluation provenance: {path}")
    if "safeWorkpoint" not in source:
        raise ValueError(f"validation replay threshold is missing safeWorkpoint: {path}")
    if protocol == "calibration_ready_pre_test" and source.get("safeWorkpoint") is not True:
        raise ValueError(f"safe calibration protocol is not marked as a safe workpoint: {path}")
    if protocol == "diagnostic_calibration_unsafe" and source.get("safeWorkpoint") is not False:
        raise ValueError(f"diagnostic calibration protocol is incorrectly marked safe: {path}")
    try:
        threshold = float(payload["threshold"])
        selected_threshold = float(source["selectedThreshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"validation replay threshold is incomplete: {path}") from exc
    if not np.isclose(threshold, selected_threshold, rtol=0.0, atol=1e-7):
        raise ValueError(f"validation replay threshold does not match thresholdSource: {path}")


def _validate_checkpoint_provenance(
    payload: Mapping[str, Any],
    path: Path,
    *,
    expected_variant: str,
    expected_seed: int,
    expected_epoch: int,
    manifest: Mapping[str, Any],
) -> None:
    if str(payload.get("variant")) != expected_variant or int(payload.get("seed", -1)) != expected_seed:
        raise ValueError(f"validation replay member identity mismatch: {path}")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"validation replay has no protocol provenance: {path}")
    if protocol.get("testRead") is not False or protocol.get("candidateUnion") is not False:
        raise ValueError(f"validation replay has unsafe protocol provenance: {path}")
    if protocol.get("thresholdSource") != "calibration_only":
        raise ValueError(f"validation replay threshold source is not calibration-only: {path}")
    expected_counts = manifest.get("splitPoseCounts") or {}
    expected_digests = manifest.get("splitCandidateDigests") or {}
    for split in ("train", "calibration", "validation"):
        if split in expected_counts and int(protocol.get(f"{split}PoseCount", -1)) != int(expected_counts[split]):
            raise ValueError(f"validation replay {split} pose count provenance mismatch: {path}")
        if split in expected_digests and str(protocol.get(f"{split}CandidateDigest", "")) != str(expected_digests[split]):
            raise ValueError(f"validation replay {split} candidate provenance mismatch: {path}")
    provenance = payload.get("inputProvenance")
    if provenance != manifest.get("inputProvenance"):
        raise ValueError(f"validation replay input provenance mismatch: {path}")
    configured_epochs = payload.get("trainingEpochs")
    if configured_epochs is not None and int(configured_epochs) != expected_epoch:
        raise ValueError(f"validation replay training epoch provenance mismatch: {path}")
    checkpoint_value = payload.get("checkpoint")
    checkpoint_sha = str(payload.get("checkpointSha256", ""))
    if not checkpoint_value or len(checkpoint_sha) != 64:
        raise ValueError(f"validation replay is missing checkpoint provenance: {path}")
    checkpoint = Path(str(checkpoint_value)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"validation replay checkpoint is missing: {checkpoint}")
    if _sha256_file(checkpoint) != checkpoint_sha:
        raise ValueError(f"validation replay checkpoint hash mismatch: {path}")
    calibration_summary = checkpoint.parent / "calibration_ready_summary.json"
    if not calibration_summary.is_file():
        raise FileNotFoundError(f"checkpoint has no calibration summary: {calibration_summary}")
    try:
        calibration = json.loads(calibration_summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"checkpoint calibration summary is invalid: {calibration_summary}") from exc
    if calibration.get("testRead") is not False:
        raise ValueError(f"checkpoint calibration summary is not test-free: {calibration_summary}")
    recorded_calibration_sha = str(payload.get("calibrationSummarySha256", ""))
    if len(recorded_calibration_sha) != 64 or recorded_calibration_sha != _sha256_file(calibration_summary):
        raise ValueError(f"validation replay calibration summary hash mismatch: {path}")


def _load_members(root: Path, manifest: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    members: dict[tuple[str, int], dict[str, Any]] = {}
    epochs = int(manifest["epochs"])
    expected_pose_count = int((manifest.get("splitPoseCounts") or {}).get("validation", 0))
    expected_candidate_digest = str((manifest.get("splitCandidateDigests") or {}).get("validation", ""))
    if expected_pose_count <= 0:
        raise ValueError("matrix manifest has no validation pose count")
    if len(expected_candidate_digest) != 64:
        raise ValueError("matrix manifest has no validation candidate digest")
    for variant in manifest.get("variants", {}):
        for raw_seed in manifest.get("seeds", []):
            seed = int(raw_seed)
            member_dir = root / f"{variant}_seed{seed}_e{epochs}"
            path = member_dir / "validation_evaluation.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing validation replay: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != "pvs-hierarchical-relation-survival-integrated-evaluation-v1":
                raise ValueError(f"unexpected evaluation schema: {path}")
            if payload.get("split") != "validation" or payload.get("testRead") is not False:
                raise ValueError(f"validation replay has invalid split/test provenance: {path}")
            if str(payload.get("variant")) != str(variant) or int(payload.get("seed")) != seed:
                raise ValueError(f"evaluation identity mismatch: {path}")
            _validate_threshold_provenance(payload, path)
            _validate_checkpoint_provenance(
                payload,
                path,
                expected_variant=str(variant),
                expected_seed=seed,
                expected_epoch=epochs,
                manifest=manifest,
            )
            rows = payload.get("perPose")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"evaluation has no perPose rows: {path}")
            pose_indices = [int(row["poseIndex"]) for row in rows]
            if pose_indices != list(payload.get("poseIndices", [])):
                raise ValueError(f"pose order mismatch: {path}")
            if len(set(pose_indices)) != len(pose_indices):
                raise ValueError(f"duplicate pose IDs: {path}")
            digest = str(payload.get("candidateDigest", ""))
            if len(digest) != 64:
                raise ValueError(f"missing candidate digest: {path}")
            if digest != expected_candidate_digest:
                raise ValueError(f"validation candidate digest disagrees with matrix manifest: {path}")
            if len(rows) != expected_pose_count:
                raise ValueError(
                    f"validation pose count disagrees with matrix manifest: {path}; "
                    f"expected {expected_pose_count}, got {len(rows)}"
                )
            expected_pose_digests = manifest.get("poseCandidateDigests") or {}
            for row in rows:
                if not isinstance(row, Mapping) or not isinstance(row.get("metrics"), Mapping):
                    raise ValueError(f"invalid pose row: {path}")
                pose_key = str(int(row.get("poseIndex", -1)))
                row_digest = str(row.get("candidateDigest", ""))
                if len(row_digest) != 64:
                    raise ValueError(f"missing per-pose candidate digest: {path}")
                if expected_pose_digests and row_digest != str(expected_pose_digests.get(pose_key, "")):
                    raise ValueError(f"per-pose candidate provenance disagrees with matrix manifest: {path}")
                for metric in METRIC_NAMES:
                    if metric not in row["metrics"]:
                        raise ValueError(f"validation pose row is missing formal metric {metric}: {path}")
                    _finite(row["metrics"][metric])
                for metric in COUNT_METRICS + ("weightedTp", "weightedGt"):
                    _finite(row["metrics"].get(metric, 0.0))
                unavailable = payload.get("unavailableMetrics", {})
                if not isinstance(unavailable, Mapping):
                    raise ValueError(f"unavailableMetrics is not an object: {path}")
                for image_metric in IMAGE_METRICS:
                    if image_metric in unavailable and image_metric in row["metrics"]:
                        raise ValueError(f"unavailable image metric was encoded numerically: {path}")
            members[(str(variant), seed)] = {
                "payload": payload,
                "rows": rows,
                "path": str(path.resolve()),
                "poseIndices": pose_indices,
                "candidateDigest": digest,
            }
    if not members:
        raise ValueError("matrix has no members")
    reference = next(iter(members.values()))
    for key, member in members.items():
        if member["poseIndices"] != reference["poseIndices"]:
            raise ValueError(f"pose sequence mismatch for {key}")
        if member["candidateDigest"] != reference["candidateDigest"]:
            raise ValueError(f"candidate digest mismatch for {key}")
        ref_rows = {int(row["poseIndex"]): str(row["candidateDigest"]) for row in reference["rows"]}
        cur_rows = {int(row["poseIndex"]): str(row["candidateDigest"]) for row in member["rows"]}
        if cur_rows != ref_rows:
            raise ValueError(f"per-pose candidate hash mismatch for {key}")
    return members


def _member_arrays(
    member: Mapping[str, Any],
    metric_names: Iterable[str] = METRIC_NAMES,
) -> dict[str, np.ndarray]:
    """Pack one member's pose sufficient statistics as ``[seed, pose]`` data."""
    rows = member["rows"]
    names = (*COUNT_METRICS, "weightedTp", "weightedGt", *tuple(metric_names))
    return {
        name: np.asarray([float(row["metrics"].get(name, 0.0)) for row in rows], dtype=np.float64)
        for name in names
    }


def _gather_pose_field(
    arrays_by_seed: list[dict[str, np.ndarray]],
    field: str,
    selected_seed: np.ndarray,
    pose_indices: np.ndarray,
) -> np.ndarray:
    """Gather one member field for outer seed and inner pose draws."""
    stacked = np.stack([arrays[field] for arrays in arrays_by_seed], axis=0)
    # Advanced indexing with ``[B, seed_position]`` already produces
    # ``[bootstrap, seed_position, pose]``.  Adding a singleton index here
    # creates a fourth dimension and makes the subsequent pose gather invalid.
    selected = stacked[selected_seed]
    return np.take_along_axis(selected, pose_indices, axis=2)


def _scope_bootstrap_values(
    arrays_by_seed: list[dict[str, np.ndarray]],
    scope: str,
    selected_seed: np.ndarray,
    pose_indices: np.ndarray,
    metric_names: Iterable[str] = METRIC_NAMES,
) -> dict[str, np.ndarray]:
    """Return ``[bootstrap draw, seed position]`` values for every metric."""
    gathered = {
        field: _gather_pose_field(arrays_by_seed, field, selected_seed, pose_indices)
        for field in (*COUNT_METRICS, "weightedTp", "weightedGt")
    }
    result: dict[str, np.ndarray] = {}
    metric_names = tuple(metric_names)
    if scope == "poseMacro":
        for metric in metric_names:
            result[metric] = _gather_pose_field(arrays_by_seed, metric, selected_seed, pose_indices).mean(axis=2)
        return result

    totals = {name: values.sum(axis=2) for name, values in gathered.items()}
    tp, fp, fn, tn = (totals[name] for name in COUNT_METRICS)
    candidate = tp + fp + fn + tn
    gt = tp + fn
    pred = tp + fp
    recall = tp / np.maximum(1.0, gt)
    precision = tp / np.maximum(1.0, pred)
    specificity = tn / np.maximum(1.0, tn + fp)
    result.update({
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
    })
    for metric in (
        "candidateGlbCount", "predictedGlbCount", "predictedGlbBytes", "glbCountReduction",
        "glbByteReduction", "downloadUtilityRecall",
        "glbBytesAtAchievedVisualUtility", "candidateGlbBytes", "requiredGlbCount", "requiredGlbBytes",
        "downloadNdcgAt20", "downloadNdcgAt50", "downloadNdcgAt100",
        "utilityNdcgAt20", "utilityNdcgAt50", "utilityNdcgAt100",
        "visibilityBaselineNdcgAt20", "visibilityBaselineNdcgAt50", "visibilityBaselineNdcgAt100",
        "downloadUtilityRecallFraction5pct", "downloadUtilityRecallFraction10pct",
        "downloadUtilityRecallFraction20pct", "downloadUtilityRecallFraction40pct",
        "downloadUtilityRecallFixed10MiB", "downloadUtilityRecallFixed25MiB",
        "downloadUtilityRecallFixed50MiB", "downloadRequiredGlbRecallAt100",
    ):
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
        raise ValueError("formal paired bootstrap requires at least 10000 replicates")
    if tuple(int(value) for value in seeds) != (20260801, 20260802, 20260803):
        raise ValueError("formal paired bootstrap requires the registered three seeds")
    if left == right:
        raise ValueError("paired bootstrap requires two distinct members")
    metric_names = tuple(metrics)
    rng = np.random.default_rng(int(seed))
    pose_count = len(members[(left, seeds[0])]["rows"])
    reference_poses = members[(left, seeds[0])].get("poseIndices")
    reference_digest = members[(left, seeds[0])].get("candidateDigest")
    for variant in (left, right):
        for current_seed in seeds:
            member = members.get((variant, current_seed))
            if member is None:
                raise ValueError(f"paired bootstrap input is missing {variant}/seed{current_seed}")
            if len(member["rows"]) != pose_count:
                raise ValueError("paired bootstrap members have different pose counts")
            if member.get("poseIndices") != reference_poses or member.get("candidateDigest") != reference_digest:
                raise ValueError("paired bootstrap members do not share pose/candidate identity")
    left_arrays = [_member_arrays(members[(left, current_seed)], metric_names) for current_seed in seeds]
    right_arrays = [_member_arrays(members[(right, current_seed)], metric_names) for current_seed in seeds]
    boot = {f"poseMacro.{metric}": np.empty((replicates,), dtype=np.float64) for metric in metric_names}
    boot.update({f"aggregate.{metric}": np.empty((replicates,), dtype=np.float64) for metric in metric_names})
    observed: dict[str, float] = {}
    for scope in ("poseMacro", "aggregate"):
        observed.update({f"{scope}.{metric}": float(np.mean([
            _summarize_rows(members[(right, current_seed)]["rows"])[scope][metric]
            - _summarize_rows(members[(left, current_seed)]["rows"])[scope][metric]
            for current_seed in seeds
        ])) for metric in metric_names})
    batch = 128
    for start in range(0, replicates, batch):
        end = min(replicates, start + batch)
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
        ci = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        effects[key] = {
            "meanDelta": float(observed[key]),
            "ci95": ci,
            "direction": "positive" if observed[key] > 0.0 else "negative" if observed[key] < 0.0 else "zero",
            "crossesZero": bool(ci[0] <= 0.0 <= ci[1]),
            "bootstrapReplicates": int(replicates),
            "clusterUnit": BOOTSTRAP_CLUSTER_UNIT,
            "paired": True,
        }
    return effects


def _comparison_pairs(variants: list[str]) -> list[tuple[str, str, str]]:
    # Formal innovation reports are read as ``full - ablation``: a positive
    # delta therefore means that the complete model improves the metric.  The
    # bootstrap implementation still computes ``right - left``; put the
    # ablation on the left explicitly instead of relying on variant order.
    pairs: list[tuple[str, str, str]] = []
    if "full" in variants:
        for reference in variants:
            if reference == "full":
                continue
            pairs.append((f"full_minus_{reference}", reference, "full"))

    # Keep the remaining pairwise contrasts available for diagnostics, with a
    # name that matches their numeric direction.  They are not used as the
    # primary innovation claims when ``full`` is present.
    for left, right in itertools.combinations(variants, 2):
        if "full" in (left, right):
            continue
        pairs.append((f"{right}_minus_{left}", left, right))
    return pairs


def build_summary(root: Path, replicates: int, lcb_replicates: int) -> dict[str, Any]:
    manifest = _load_manifest(root)
    members = _load_members(root, manifest)
    registered = manifest.get("variants", {})
    if not isinstance(registered, Mapping):
        raise ValueError("validation matrix manifest has no variant registry")
    variants = [name for name in FORMAL_VARIANTS if name in registered]
    if tuple(variants) != FORMAL_VARIANTS:
        raise ValueError(f"validation matrix variants are incomplete: {variants}")
    seeds = [int(value) for value in manifest.get("seeds", [])]
    if set(members) != {(variant, seed) for variant in variants for seed in seeds}:
        raise ValueError("validation matrix member coverage is incomplete")
    reference = members[(variants[0], seeds[0])]
    member_summary: dict[str, Any] = {}
    safe_members: list[str] = []
    diagnostic_members: list[str] = []
    for variant in variants:
        for seed in seeds:
            member = members[(variant, seed)]
            summary = _summarize_rows(member["rows"], lcb_replicates=lcb_replicates, lcb_seed=seed + 7013)
            payload = member["payload"]
            member_key = f"{variant}:{seed}"
            safe = bool((payload.get("thresholdSource") or {}).get("safeWorkpoint", False))
            (safe_members if safe else diagnostic_members).append(member_key)
            member_summary[member_key] = {
                "variant": variant,
                "seed": seed,
                "path": member["path"],
                "checkpoint": payload.get("checkpoint"),
                "threshold": payload.get("threshold"),
                "thresholdSource": payload.get("thresholdSource"),
                "safeWorkpoint": safe,
                "memberClass": "safe" if safe else "diagnostic_non_safe",
                "poseCount": summary["poseCount"],
                "poseMacro": summary["poseMacro"],
                "aggregate": summary["aggregate"],
                "safety": summary["safety"],
                "runtime": payload.get("runtime", {}),
                "imageMetrics": payload.get("imageMetrics", {}),
            }
    metrics = tuple(METRIC_NAMES)
    comparisons: dict[str, Any] = {}
    for index, (name, left, right) in enumerate(_comparison_pairs(variants)):
        effects = _paired_bootstrap(members, left, right, seeds, metrics, replicates, 20260813 + index)
        comparisons[name] = {
            "left": left,
            "right": right,
            "contrast": "full_minus_ablation" if name.startswith("full_minus_") else "right_minus_left",
            "metrics": effects,
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "version": 1,
        "split": "validation",
        "testRead": False,
        "manifest": str((root / "matrix_manifest.json").resolve()),
        "mode": manifest.get("mode"),
        "variants": variants,
        "seeds": seeds,
        "poseCount": len(reference["poseIndices"]),
        "poseIndices": reference["poseIndices"],
        "candidateDigest": reference["candidateDigest"],
        "inputProvenance": manifest["inputProvenance"],
        "candidateSemantics": "stored native back-camera candidate CSR; no GT union, cap, or frontend whitelist",
        "members": member_summary,
        "memberStatistics": {
            "safe": {"memberKeys": safe_members, "count": len(safe_members)},
            "diagnostic_non_safe": {"memberKeys": diagnostic_members, "count": len(diagnostic_members)},
            "safetyDecisionPopulation": "safe members only; diagnostic members remain separate and cannot establish safety",
        },
        "comparisons": comparisons,
        "bootstrap": {
            "replicates": int(replicates),
            "confidence": 0.95,
            "paired": True,
            "clusterUnit": BOOTSTRAP_CLUSTER_UNIT,
        },
        "metricDefinitions": {
            "TP": "P intersect G",
            "FP": "P minus G",
            "FN": "G minus P",
            "TN": "C minus (P union G)",
            "poseMacro": "compute each pose first, then average",
            "aggregate": "sum TP/FP/FN/TN and visible weight across resampled poses before deriving ratios",
            "weightedRecall": "sum visible weight found by P divided by total visible weight",
            "usefulCull": "TN / candidate",
            "badCull": "FN / candidate",
            "balancedAccuracy": "(recall + specificity) / 2",
            "instanceAccuracy": "(TP + TN) / candidate",
        },
        "unavailableMetrics": {
            "missPixelRate": "not_available until hardware Color-ID evaluation is attached; excluded from numeric bootstrap",
            "wrongIdPixelRate": "not_available until hardware Color-ID evaluation is attached; excluded from numeric bootstrap",
            "extraPixelRate": "not_available until hardware Color-ID evaluation is attached; excluded from numeric bootstrap",
        },
    }


def self_test() -> dict[str, Any]:
    rows = []
    for pose in range(4):
        rows.append({"poseIndex": pose, "metrics": {
            **{metric: 0.5 for metric in METRIC_NAMES},
            "tp": 1.0, "fp": 1.0, "fn": 0.0, "tn": 2.0,
            "weightedTp": 1.0, "weightedGt": 1.0,
        }})
    summary = _summarize_rows(rows)
    if abs(summary["aggregate"]["accuracy"] - 0.75) > 1e-9:
        raise AssertionError("aggregate accuracy self-test failed")
    return {"status": "passed", "schema": SUMMARY_SCHEMA}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--lcb-replicates", type=int, default=10000)
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
    payload = build_summary(args.root.resolve(), int(args.bootstrap_replicates), int(args.lcb_replicates))
    _write_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "poseCount": payload["poseCount"], "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
