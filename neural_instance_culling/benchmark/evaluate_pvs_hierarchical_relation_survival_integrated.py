#!/usr/bin/env python3
"""Replay a frozen hierarchical-relation checkpoint on the validation split.

The evaluator is deliberately separate from training.  It reconstructs the
runtime model from one checkpoint, obtains the threshold only from that
checkpoint's calibration artifact, and writes one complete per-pose record for
the stored back-camera candidate CSR.  It never reads the test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402
from common.provenance import sha256_file  # noqa: E402
from common.runtime_meta import load_runtime_meta  # noqa: E402
from current_pvs_utils import evaluate_thresholds  # noqa: E402
from hierarchical_relation_survival_integrated_model import (  # noqa: E402
    GEO_DIM,
    SURVIVAL_PARAMETER_DIM,
    SURVIVAL_RANK,
    HierarchicalRelationSurvivalIntegratedModel,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


EVALUATION_SCHEMA = "pvs-hierarchical-relation-survival-integrated-evaluation-v1"
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


def _select_threshold(
    checkpoint: Mapping[str, Any],
    allow_unsafe: bool,
    calibration_summary: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Read the already-frozen threshold; never reselect from calibration rows.

    The standalone calibration summary is the producer of the threshold.  The
    checkpoint's ``best.selection`` (or its explicitly recorded threshold) is
    a second copy which must agree with that artifact.  Keeping both checks is
    important because a replay must not silently use a changed calibration
    file or recompute a workpoint from rows.
    """
    if not isinstance(calibration_summary, Mapping):
        raise ValueError("checkpoint calibration summary is required; threshold selection is not allowed during replay")
    if calibration_summary.get("testRead") is not False:
        raise ValueError("checkpoint calibration summary claims test was read")
    protocol = str(calibration_summary.get("protocol", ""))
    if protocol not in {"calibration_ready_pre_test", "diagnostic_calibration_unsafe"}:
        raise ValueError(f"unsupported frozen calibration protocol: {protocol!r}")

    selected = calibration_summary.get("selected")
    if isinstance(selected, Mapping):
        if protocol != "calibration_ready_pre_test":
            raise ValueError("diagnostic calibration cannot provide a safe selected workpoint")
        if float(selected.get("aggregateWeightedRecall", -1.0)) <= 0.99:
            raise ValueError("frozen safe workpoint does not pass aggregate weighted recall")
        if float(selected.get("aggregateWeightedRecallLowerConfidenceBound", -1.0)) <= 0.99:
            raise ValueError("frozen safe workpoint does not pass weighted-recall lower bound")
        frozen_selection = selected
        safe_workpoint = True
        source_name = "calibration_ready_summary.selected"
    else:
        if not allow_unsafe:
            raise ValueError("checkpoint has no calibration-qualified weighted-recall workpoint")
        # This is a registered diagnostic fallback, not a fresh optimization:
        # consume only the threshold written by calibration.
        frozen_selection = calibration_summary.get("diagnostic")
        if not isinstance(frozen_selection, Mapping):
            best_diagnostic = calibration_summary.get("bestDiagnostic")
            if isinstance(best_diagnostic, Mapping):
                frozen_selection = best_diagnostic.get("selection", best_diagnostic)
        if not isinstance(frozen_selection, Mapping):
            raise ValueError("diagnostic calibration summary has no frozen diagnostic threshold")
        safe_workpoint = False
        source_name = "calibration_ready_summary.diagnostic"

    try:
        threshold = float(frozen_selection["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen calibration selection has no valid threshold") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("frozen calibration threshold is outside [0, 1]")

    frozen = checkpoint.get("best")
    if not isinstance(frozen, Mapping):
        raise ValueError("checkpoint has no frozen best workpoint")
    checkpoint_selection = frozen.get("selection")
    if not isinstance(checkpoint_selection, Mapping):
        checkpoint_selection = frozen
    try:
        checkpoint_threshold = float(checkpoint_selection["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint best workpoint has no frozen threshold") from exc
    if not np.isclose(threshold, checkpoint_threshold, rtol=0.0, atol=1e-7):
        raise ValueError("checkpoint threshold disagrees with calibration_ready_summary")

    checkpoint_calibration = checkpoint.get("calibration")
    if isinstance(checkpoint_calibration, Mapping):
        embedded = checkpoint_calibration.get("selected")
        if not isinstance(embedded, Mapping):
            embedded = checkpoint_calibration.get("diagnostic")
        if isinstance(embedded, Mapping) and "threshold" in embedded:
            if not np.isclose(float(embedded["threshold"]), threshold, rtol=0.0, atol=1e-7):
                raise ValueError("checkpoint embedded calibration threshold disagrees with frozen summary")

    return threshold, {
        "source": source_name,
        "protocol": protocol,
        "selectionSplit": "calibration",
        "selectedThreshold": threshold,
        "selection": dict(frozen_selection),
        "safeWorkpoint": safe_workpoint,
        "selectedFromTest": False,
        "testEvaluationCount": 0,
    }


def _build_model(
    checkpoint: Mapping[str, Any],
    world_aabbs: np.ndarray,
    instance_to_glb: np.ndarray,
    device: torch.device,
) -> HierarchicalRelationSurvivalIntegratedModel:
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no model config")
    model = HierarchicalRelationSurvivalIntegratedModel(
        num_instances=int(config["numInstances"]),
        num_glbs=int(config.get("numGlbs", int(instance_to_glb.max()) + 1)),
        relation_hidden_dim=int(config.get("relationHiddenDim", 64)),
        query_basis_dim=int(config.get("queryBasisDim", SURVIVAL_RANK)),
        hidden_dim=int(config.get("hiddenDim", 64)),
        spectral_mode=str(config.get("spectralMode", "integrated")),
        relation_source=str(config.get("relationSource", "hierarchical")),
    ).to(device)
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def _runtime_features(checkpoint: Mapping[str, Any], geometry: np.ndarray, device: torch.device) -> torch.Tensor:
    coefficients = checkpoint.get("survivalCoefficients")
    if coefficients is None:
        raise ValueError("checkpoint has no survivalCoefficients")
    coefficients = torch.as_tensor(coefficients, dtype=torch.float32, device=device)
    expected = (geometry.shape[0], SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM)
    if tuple(coefficients.shape) != expected or not bool(torch.isfinite(coefficients).all()):
        raise ValueError(f"checkpoint survival coefficients have shape {tuple(coefficients.shape)}, expected {expected}")
    return torch.cat([torch.from_numpy(geometry).to(device), coefficients.reshape(geometry.shape[0], -1)], dim=-1)


def _candidate_hash(candidate_ids: list[int] | np.ndarray) -> str:
    values = np.asarray(candidate_ids, dtype="<u4").reshape(-1)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def _resource_metrics(
    candidate_ids: np.ndarray,
    predicted_ids: np.ndarray,
    target: Mapping[str, Any],
    instance_to_glb: np.ndarray,
    glb_bytes: np.ndarray,
) -> dict[str, float]:
    candidate_glbs = np.unique(instance_to_glb[candidate_ids.astype(np.int64)]) if candidate_ids.size else np.zeros(0, dtype=np.int64)
    predicted_glbs = np.unique(instance_to_glb[predicted_ids.astype(np.int64)]) if predicted_ids.size else np.zeros(0, dtype=np.int64)
    candidate_byte_count = float(glb_bytes[candidate_glbs].sum()) if candidate_glbs.size else 0.0
    predicted_byte_count = float(glb_bytes[predicted_glbs].sum()) if predicted_glbs.size else 0.0
    gt_mask = np.asarray(target.get("targetIds", []), dtype=np.uint32)
    gt_weights = np.asarray(target.get("targetWeights", []), dtype=np.float64)
    gt_glbs = np.unique(instance_to_glb[gt_mask.astype(np.int64)]) if gt_mask.size else np.zeros(0, dtype=np.int64)
    gt_weight_total = float(gt_weights.sum())
    gt_instance_glbs = instance_to_glb[gt_mask.astype(np.int64)] if gt_mask.size else np.zeros(0, dtype=np.int64)
    predicted_glb_mask = (
        np.isin(gt_instance_glbs, predicted_glbs, assume_unique=False)
        if gt_mask.size
        else np.zeros(0, dtype=bool)
    )
    download_utility = float(gt_weights[predicted_glb_mask].sum()) if gt_weights.size else 0.0
    utility_by_glb: dict[int, float] = {}
    for instance, weight in zip(gt_mask.tolist(), gt_weights.tolist(), strict=True):
        gid = int(instance_to_glb[int(instance)])
        utility_by_glb[gid] = utility_by_glb.get(gid, 0.0) + float(weight)
    equivalent_bytes = 0.0
    if download_utility > 0.0:
        accumulated = 0.0
        for gid in sorted(utility_by_glb, key=lambda value: utility_by_glb[value] / max(1.0, float(glb_bytes[value])), reverse=True):
            equivalent_bytes += float(glb_bytes[gid])
            accumulated += utility_by_glb[gid]
            if accumulated >= download_utility:
                break
    return {
        "predictedGlbCount": float(predicted_glbs.size),
        "candidateGlbCount": float(candidate_glbs.size),
        "candidateGlbBytes": candidate_byte_count,
        "requiredGlbCount": float(gt_glbs.size),
        "requiredGlbBytes": float(glb_bytes[gt_glbs].sum()) if gt_glbs.size else 0.0,
        "predictedGlbBytes": predicted_byte_count,
        "glbCountReduction": 1.0 - _safe_div(float(predicted_glbs.size), float(candidate_glbs.size), 0.0),
        "glbByteReduction": 1.0 - _safe_div(predicted_byte_count, candidate_byte_count, 0.0),
        "downloadUtilityRecall": _safe_div(download_utility, gt_weight_total, 1.0),
        "glbBytesAtAchievedVisualUtility": equivalent_bytes,
    }


def _camel_metrics(row: Mapping[str, Any], resources: Mapping[str, float]) -> dict[str, float]:
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
    result = {
        target: float(row.get(source, row.get(target, 0.0)))
        for source, target in mapping.items()
    }
    result["poseMacroWeightedRecall"] = result["weightedRecall"]
    for key in ("tp", "fp", "fn", "tn"):
        result[key] = float(row.get(key, 0.0))
    result.update({key: float(value) for key, value in resources.items()})
    return result


def _summarize_pose_rows(rows: list[dict[str, Any]], lcb_replicates: int = 0, seed: int = 0) -> dict[str, Any]:
    if not rows:
        raise ValueError("no validation pose rows")
    macro = {
        metric: float(np.mean([float(row["metrics"].get(metric, 0.0)) for row in rows]))
        for metric in CORE_METRICS
        if metric in rows[0]["metrics"]
    }
    macro["poseMacroWeightedRecall"] = float(
        np.mean([float(row["metrics"].get("poseMacroWeightedRecall", row["metrics"].get("weightedRecall", 0.0))) for row in rows])
    )
    totals = {key: float(sum(float(row["metrics"].get(key, 0.0)) for row in rows)) for key in ("tp", "fp", "fn", "tn")}
    candidate = sum(float(row["metrics"].get("avgCandidateCount", 0.0)) for row in rows)
    gt = sum(float(row["metrics"].get("avgGtCount", 0.0)) for row in rows)
    pred = sum(float(row["metrics"].get("avgPredCount", 0.0)) for row in rows)
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    recall = _safe_div(tp, tp + fn, 1.0)
    precision = _safe_div(tp, tp + fp, 1.0)
    specificity = _safe_div(tn, tn + fp, 1.0)
    weighted_tp = sum(float(row["metrics"].get("weightedTp", 0.0)) for row in rows)
    weighted_gt = sum(float(row["metrics"].get("weightedGt", 0.0)) for row in rows)
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
    }
    aggregate["aggregateWeightedRecall"] = aggregate["weightedRecall"]
    aggregate["poseMacroWeightedRecall"] = macro["poseMacroWeightedRecall"]
    for key in (
        "candidateGlbCount",
        "predictedGlbCount",
        "candidateGlbBytes",
        "predictedGlbBytes",
        "glbCountReduction",
        "glbByteReduction",
        "downloadUtilityRecall",
        "glbBytesAtAchievedVisualUtility",
        "requiredGlbCount",
        "requiredGlbBytes",
    ):
        aggregate[key] = float(np.mean([float(row["metrics"].get(key, 0.0)) for row in rows]))
    aggregate["weightedTp"] = weighted_tp / len(rows)
    aggregate["weightedGt"] = weighted_gt / len(rows)
    if int(lcb_replicates) > 0:
        lcb = _weighted_lcb_from_rows(rows, int(lcb_replicates), int(seed))
        macro_lcb = _pose_macro_lcb_from_rows(rows, int(lcb_replicates), int(seed) + 100000)
    else:
        lcb = None
        macro_lcb = None
    aggregate["aggregateWeightedRecallLowerConfidenceBound"] = lcb
    aggregate["poseMacroWeightedRecallLowerConfidenceBound"] = macro_lcb
    aggregate["weightedRecallLowerConfidenceBound"] = lcb
    macro["aggregateWeightedRecallLowerConfidenceBound"] = lcb
    macro["poseMacroWeightedRecallLowerConfidenceBound"] = macro_lcb
    return {
        "poseMacro": macro,
        "aggregate": aggregate,
        "counts": totals,
        "poseCount": len(rows),
        "weightedRecallLowerConfidenceBound": lcb,
        "testRead": False,
    }


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


@torch.no_grad()
def _evaluate_legacy_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    if bool(checkpoint.get("testRead", True)):
        raise ValueError("checkpoint claims test was read")
    runtime_meta_path = Path(args.runtime_meta).resolve()
    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    num_instances = int(world_aabbs.shape[0])
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = _build_model(checkpoint, world_aabbs, instance_to_glb, device)
    geometry_path = Path(args.initial_geo_features).resolve()
    geometry = _load_geometry(geometry_path, num_instances)
    runtime_features = _runtime_features(checkpoint, geometry, device)
    dataset = PoseCSRDataset(Path(args.dataset_dir).resolve(), num_instances=num_instances)
    split_name = str(args.split)
    if split_name.lower() == "test":
        raise ValueError("test is not allowed in this evaluator")
    split = dataset.split(split_name)
    calibration_path = checkpoint_path.parent / "calibration_ready_summary.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"checkpoint has no calibration summary: {calibration_path}")
    calibration_summary = json.loads(calibration_path.read_text(encoding="utf-8"))
    threshold, threshold_source = _select_threshold(
        checkpoint,
        bool(args.allow_unsafe_diagnostic),
        calibration_summary,
    )
    num_glbs = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
    glb_bytes = _load_glb_bytes(Path(args.glb_index).resolve(), Path(args.glb_root).resolve(), num_glbs)
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
        collect_score_stats=False,
        collect_per_pose=True,
    )
    if len(rows) != 1 or not isinstance(rows[0].get("_per_pose"), list):
        raise ValueError("evaluator did not produce per-pose rows")
    raw_rows = sorted(rows[0]["_per_pose"], key=lambda row: int(row["poseIndex"]))
    pose_indices = [int(value) for value in split.pose_indices.tolist()]
    if [int(row["poseIndex"]) for row in raw_rows] != pose_indices:
        raise ValueError("evaluation pose order does not match the validation split")
    per_pose: list[dict[str, Any]] = []
    pose_candidate_digests: dict[str, str] = {}
    for raw in raw_rows:
        candidate_ids = np.asarray(raw.get("candidateIds", []), dtype=np.uint32)
        predicted_ids = np.asarray(raw.get("predictedIds", []), dtype=np.uint32)
        metrics = dict(raw.get("metrics", {}))
        visible_ids, visible_weights = dataset.visible_slice(int(raw["poseIndex"]))
        resources = _resource_metrics(
            candidate_ids,
            predicted_ids,
            {"targetIds": visible_ids, "targetWeights": visible_weights},
            instance_to_glb,
            glb_bytes,
        )
        normalized = _camel_metrics(raw.get("metrics", {}), resources)
        target_mask = np.isin(candidate_ids, visible_ids.astype(np.uint32), assume_unique=False)
        predicted_mask = np.isin(candidate_ids, predicted_ids, assume_unique=False)
        normalized.update({
            "tp": float(np.logical_and(predicted_mask, target_mask).sum()),
            "fp": float(np.logical_and(predicted_mask, ~target_mask).sum()),
            "fn": float(np.logical_and(~predicted_mask, target_mask).sum()),
            "tn": float(np.logical_and(~predicted_mask, ~target_mask).sum()),
        })
        normalized["weightedTp"] = float(np.sum(visible_weights[np.isin(visible_ids, predicted_ids, assume_unique=False)]))
        normalized["weightedGt"] = float(np.sum(visible_weights))
        normalized["candidateCount"] = float(candidate_ids.size)
        normalized["gtCount"] = float(visible_ids.size)
        normalized["predCount"] = float(predicted_ids.size)
        normalized["candidateDigest"] = _candidate_hash(candidate_ids)
        pose_candidate_digests[str(int(raw["poseIndex"]))] = normalized["candidateDigest"]
        per_pose.append({
            "poseIndex": int(raw["poseIndex"]),
            "candidateDigest": normalized["candidateDigest"],
            "candidateCount": int(candidate_ids.size),
            "candidateIds": candidate_ids.astype(int).tolist() if args.persist_ids else None,
            "predictedIds": predicted_ids.astype(int).tolist() if args.persist_ids else None,
            "metrics": normalized,
        })
    summary = _summarize_pose_rows(
        per_pose,
        lcb_replicates=10000 if split_name == "calibration" else 0,
        seed=int(args.seed),
    )
    # The evaluator intentionally does not claim image metrics.  A separate
    # hardware Color-ID pass can attach them later without changing the model
    # or threshold provenance.
    summary["imageMetrics"] = {
        "status": "not_available",
        "reason": "new-model validation replay has no Color-ID render attached",
        "testRead": False,
    }
    result = {
        "schema": EVALUATION_SCHEMA,
        "version": 1,
        "split": split_name,
        "testRead": False,
        "variant": str(checkpoint.get("variant", checkpoint.get("args", {}).get("variant", ""))),
        "variantSpec": checkpoint.get("variantSpec", {}),
        "relationVariant": (checkpoint.get("relationProvenance") or {}).get("variant", {}),
        "seed": int(checkpoint.get("args", {}).get("seed", args.seed)),
        "checkpointSeed": int(checkpoint.get("args", {}).get("seed", args.seed)),
        "checkpoint": str(checkpoint_path),
        "checkpointSha256": sha256_file(checkpoint_path),
        "calibrationSummary": str(calibration_path),
        "calibrationSummarySha256": sha256_file(calibration_path),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("args", {}).get("epochs", 0))),
        "trainingEpochs": int(checkpoint.get("args", {}).get("epochs", checkpoint.get("epochs", 0))),
        "poseIndices": pose_indices,
        "poseCount": len(pose_indices),
        "candidateDigest": candidate_digest_for_pose_sequence(dataset, split.pose_indices),
        "candidateDigestScope": "stored native back-camera candidate CSR in validation split order",
        "poseCandidateDigests": pose_candidate_digests,
        "threshold": threshold,
        "thresholdSource": threshold_source,
        "metrics": summary["aggregate"],
        "aggregateWeightedRecall": summary["aggregate"]["aggregateWeightedRecall"],
        "poseMacroWeightedRecall": summary["poseMacro"]["poseMacroWeightedRecall"],
        "aggregateWeightedRecallLowerConfidenceBound": summary["aggregate"].get("aggregateWeightedRecallLowerConfidenceBound"),
        "poseMacroWeightedRecallLowerConfidenceBound": summary["poseMacro"].get("poseMacroWeightedRecallLowerConfidenceBound"),
        "poseMacro": summary["poseMacro"],
        "aggregate": summary["aggregate"],
        "perPose": per_pose,
        "runtime": {
            "device": str(device),
            "elapsedSeconds": float(time.perf_counter() - started),
            "runtimeFeatureBytes": int(runtime_features.numel() * 2),
            "runtimeFeatureDim": int(runtime_features.shape[1]),
            "inferenceInputDim": int(model.runtime_input_dim),
            "testRead": False,
        },
        "imageMetrics": summary["imageMetrics"],
        "unavailableMetrics": {
            "missPixelRate": "not_available: Color-ID image evaluation not attached",
            "wrongIdPixelRate": "not_available: Color-ID image evaluation not attached",
            "extraPixelRate": "not_available: Color-ID image evaluation not attached",
            "browserWebGpuLatency": "not_available: hardware browser replay not attached",
        },
        "protocol": checkpoint.get("protocol", {}),
        "inputProvenance": {
            "datasetMetaSha256": sha256_file(Path(args.dataset_dir).resolve() / "dataset_meta.json"),
            "runtimeMetaSha256": sha256_file(runtime_meta_path),
            "geometrySha256": sha256_file(geometry_path),
            "glbIndexSha256": sha256_file(Path(args.glb_index).resolve()),
        },
    }
    return result


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Replay the historical hierarchical-relation checkpoint contract."""
    return _evaluate_legacy_checkpoint(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--initial-geo-features", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "validation"), default="validation")
    parser.add_argument("--poses-per-batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-unsafe-diagnostic", action="store_true")
    parser.add_argument("--persist-ids", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = evaluate_checkpoint(args)
    _write_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "poseCount": payload["poseCount"], "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
