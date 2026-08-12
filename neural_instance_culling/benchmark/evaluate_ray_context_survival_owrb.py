#!/usr/bin/env python3
"""Evaluate one frozen ray-context/survival/OWRB checkpoint.

The evaluator reads only the stored back-camera candidate CSR.  Thresholds
are accepted only when they can be reselected from that checkpoint's
calibration rows; validation/test never chooses a threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.occlusion_edges import load_glb_costs  # noqa: E402
from common.candidate_identity import (  # noqa: E402
    audit_native_aabb_candidates,
    candidate_digest_for_pose_sequence,
)
from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from common.threshold_selection import select_weighted_cull_workpoint, weighted_cull_selection_rule  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from ray_context_survival_owrb_model import RayContextSurvivalOWRBModel  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _write_score_artifact(
    path: Path,
    pose_indices: list[int],
    offsets: list[int],
    candidate_ids: list[np.ndarray],
    targets: list[np.ndarray],
    weights: list[np.ndarray],
    scores: list[np.ndarray],
) -> dict[str, Any]:
    """Write raw per-candidate scores without inflating the evaluation JSON.

    The artifact is intentionally a flat CSR-like array.  It preserves the
    exact candidate order used by the evaluator while keeping the normal
    aggregate report backward compatible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.concatenate(candidate_ids, axis=0) if candidate_ids else np.zeros((0,), dtype=np.uint32)
    target = np.concatenate(targets, axis=0) if targets else np.zeros((0,), dtype=np.uint8)
    visible_weights = np.concatenate(weights, axis=0) if weights else np.zeros((0,), dtype=np.float32)
    probability = np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)
    np.savez_compressed(
        path,
        pose_indices=np.asarray(pose_indices, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        candidate_ids=ids.astype(np.uint32, copy=False),
        target=target.astype(np.uint8, copy=False),
        visible_weights=visible_weights.astype(np.float32, copy=False),
        scores=probability.astype(np.float32, copy=False),
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "poseCount": int(len(pose_indices)),
        "candidateCount": int(ids.size),
        "fields": ["pose_indices", "offsets", "candidate_ids", "target", "visible_weights", "scores"],
        "format": "numpy.npz.compressed.flat_pose_csr_v1",
    }


def _load_glb_bytes(glb_index: Path, glb_root: Path, num_glbs: int) -> np.ndarray:
    values = np.zeros((num_glbs,), dtype=np.float64)
    payload = json.loads(glb_index.read_text(encoding="utf-8"))
    for entry in payload.get("entries", []):
        gid = int(entry.get("globalId", -1))
        if 0 <= gid < num_glbs:
            path = glb_root / str(entry.get("path", ""))
            if path.is_file():
                values[gid] = float(path.stat().st_size)
    if np.any(values <= 0.0):
        raise FileNotFoundError("GLB byte index has missing files for the requested scene")
    return values


def _metric_row(
    target: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    glbs: np.ndarray,
    glb_bytes: np.ndarray,
    forward_ms: float,
    pose_index: int,
    candidate_ids: np.ndarray,
    persist_ids: bool,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float32).reshape(-1) > 0.5
    p = np.asarray(scores, dtype=np.float64).reshape(-1) >= float(threshold)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (y.size == p.size == w.size == glbs.size == candidate_ids.size):
        raise ValueError("pose arrays have inconsistent lengths")
    if not np.isfinite(w).all() or np.any(w < 0.0) or not np.isfinite(scores).all():
        raise FloatingPointError("non-finite score/weight in formal evaluation")
    candidate = int(y.size)
    tp = int(np.logical_and(p, y).sum())
    fp = int(np.logical_and(p, ~y).sum())
    fn = int(np.logical_and(~p, y).sum())
    tn = int(np.logical_and(~p, ~y).sum())
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    specificity = tn / max(1, tn + fp)
    weighted_tp = float(w[np.logical_and(p, y)].sum())
    weighted_gt = float(w[y].sum())
    weighted_recall = weighted_tp / weighted_gt if weighted_gt > 0.0 else 1.0
    # Keep the instance-level visual utility metric separate from the raw
    # visible-weight recall and from the GLB-level download utility.  The
    # current CSR stores weak importance evidence, so the registered visual
    # utility teacher is log1p(visible_weights), not pixel coverage.
    visual_utility = np.where(y, np.log1p(np.maximum(w, 0.0)), 0.0)
    visual_utility_tp = float(visual_utility[p].sum())
    visual_utility_gt = float(visual_utility.sum())
    visual_utility_recall = visual_utility_tp / visual_utility_gt if visual_utility_gt > 0.0 else 1.0
    candidate_glbs = np.unique(glbs.astype(np.int64, copy=False))
    predicted_glbs = np.unique(glbs[p].astype(np.int64, copy=False))
    candidate_bytes = float(glb_bytes[candidate_glbs].sum()) if candidate_glbs.size else 0.0
    predicted_bytes = float(glb_bytes[predicted_glbs].sum()) if predicted_glbs.size else 0.0
    # GLB downloads are group-level. Estimate the byte envelope needed to
    # reach the same visible utility as the predicted GLB set. This greedy
    # utility-per-byte value is a resource diagnostic, not a threshold input.
    predicted_glb_mask = np.isin(glbs.astype(np.int64, copy=False), predicted_glbs)
    download_utility_tp = float(w[np.logical_and(y, predicted_glb_mask)].sum())
    download_utility_recall = download_utility_tp / weighted_gt if weighted_gt > 0.0 else 1.0
    utility_by_glb: dict[int, float] = {}
    for glb_id, utility in zip(glbs[y].astype(np.int64, copy=False), w[y]):
        utility_by_glb[int(glb_id)] = utility_by_glb.get(int(glb_id), 0.0) + float(utility)
    equivalent_bytes = 0.0
    if download_utility_tp > 0.0:
        utility_order = sorted(
            (int(glb_id) for glb_id in candidate_glbs if utility_by_glb.get(int(glb_id), 0.0) > 0.0),
            key=lambda glb_id: utility_by_glb[glb_id] / max(float(glb_bytes[glb_id]), 1.0),
            reverse=True,
        )
        accumulated = 0.0
        for glb_id in utility_order:
            equivalent_bytes += float(glb_bytes[glb_id])
            accumulated += utility_by_glb[glb_id]
            if accumulated >= download_utility_tp:
                break
    row: dict[str, Any] = {
        "poseIndex": int(pose_index),
        "threshold": float(threshold),
        "candidateIdSha256": _sha256_array(candidate_ids.astype("<u4", copy=False)),
        "candidateCount": candidate,
        "gtCount": int(y.sum()),
        "predCount": int(p.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "weightedRecall": float(weighted_recall),
        "f1": float(2.0 * precision * recall / max(1e-8, precision + recall)),
        "jaccard": float(tp / max(1, tp + fp + fn)),
        "accuracy": float((tp + tn) / max(1, candidate)),
        "specificity": float(specificity),
        "balancedAccuracy": float(0.5 * (recall + specificity)),
        "usefulCull": float(tn / max(1, candidate)),
        "badCull": float(fn / max(1, candidate)),
        "weightedTp": weighted_tp,
        "weightedGt": weighted_gt,
        "visualUtilityTp": visual_utility_tp,
        "visualUtilityGt": visual_utility_gt,
        "visualUtilityRecall": float(visual_utility_recall),
        "downloadUtilityTp": download_utility_tp,
        "downloadUtilityRecall": float(download_utility_recall),
        "glbBytesAtAchievedVisualUtility": float(equivalent_bytes),
        "predOverCandidate": float(p.sum() / max(1, candidate)),
        "predOverGt": float(p.sum() / max(1, int(y.sum()))),
        "candidateGlbCount": int(candidate_glbs.size),
        "predictedGlbCount": int(predicted_glbs.size),
        "candidateGlbBytes": candidate_bytes,
        "predictedGlbBytes": predicted_bytes,
        "glbCountReduction": float(1.0 - predicted_glbs.size / max(1, candidate_glbs.size)),
        "glbByteReduction": float(1.0 - predicted_bytes / max(1.0, candidate_bytes)),
        "forwardLatencyMs": float(forward_ms),
    }
    if persist_ids:
        row["candidateIds"] = candidate_ids.astype(int).tolist()
        row["predictedIds"] = candidate_ids[p].astype(int).tolist()
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation produced no pose rows")
    macro_keys = (
        "precision", "recall", "weightedRecall", "f1", "jaccard", "accuracy",
        "specificity", "balancedAccuracy", "usefulCull", "badCull", "predCount",
        "tp", "fp", "fn", "tn",
        "candidateCount", "gtCount", "predOverCandidate", "predOverGt",
        "candidateGlbCount", "predictedGlbCount", "candidateGlbBytes",
        "predictedGlbBytes", "glbCountReduction", "glbByteReduction",
        "visualUtilityRecall", "downloadUtilityRecall", "glbBytesAtAchievedVisualUtility",
    )
    # Keep the evaluator usable with compact synthetic rows in unit tests and
    # with older diagnostic rows that predate optional image/GLB metrics.
    macro = {key: float(np.mean([float(row.get(key, 0.0)) for row in rows])) for key in macro_keys}
    sums = {key: int(sum(int(row[key]) for row in rows)) for key in ("tp", "fp", "fn", "tn", "predCount", "candidateCount", "gtCount")}
    weighted_tp = float(sum(float(row.get("weightedTp", 0.0)) for row in rows))
    weighted_gt = float(sum(float(row.get("weightedGt", 0.0)) for row in rows))
    visual_utility_tp = float(sum(float(row.get("visualUtilityTp", 0.0)) for row in rows))
    visual_utility_gt = float(sum(float(row.get("visualUtilityGt", 0.0)) for row in rows))
    download_utility_tp = float(sum(float(row.get("downloadUtilityTp", 0.0)) for row in rows))
    precision = sums["tp"] / max(1, sums["tp"] + sums["fp"])
    recall = sums["tp"] / max(1, sums["tp"] + sums["fn"])
    specificity = sums["tn"] / max(1, sums["tn"] + sums["fp"])
    aggregate = {
        "precision": float(precision),
        "recall": float(recall),
        "weightedRecall": float(weighted_tp / max(1e-12, weighted_gt)),
        "visualUtilityRecall": float(visual_utility_tp / max(1e-12, visual_utility_gt)),
        "downloadUtilityRecall": float(download_utility_tp / max(1e-12, weighted_gt)),
        "f1": float(2.0 * precision * recall / max(1e-8, precision + recall)),
        "jaccard": float(sums["tp"] / max(1, sums["tp"] + sums["fp"] + sums["fn"])),
        "accuracy": float((sums["tp"] + sums["tn"]) / max(1, sums["candidateCount"])),
        "specificity": float(specificity),
        "balancedAccuracy": float(0.5 * (recall + specificity)),
        "usefulCull": float(sums["tn"] / max(1, sums["candidateCount"])),
        "badCull": float(sums["fn"] / max(1, sums["candidateCount"])),
        "avgPredCount": float(sums["predCount"] / len(rows)),
        "avgCandidateCount": float(sums["candidateCount"] / len(rows)),
        "avgGtCount": float(sums["gtCount"] / len(rows)),
        "avgFpCount": float(sums["fp"] / len(rows)),
        "avgFnCount": float(sums["fn"] / len(rows)),
        "avgTnCount": float(sums["tn"] / len(rows)),
        "avgTpCount": float(sums["tp"] / len(rows)),
        "predOverCandidate": float(sums["predCount"] / max(1, sums["candidateCount"])),
        "predOverGt": float(sums["predCount"] / max(1, sums["gtCount"])),
        "candidateGlbCount": float(np.mean([row["candidateGlbCount"] for row in rows])),
        "predictedGlbCount": float(np.mean([row["predictedGlbCount"] for row in rows])),
        "candidateGlbBytes": float(np.mean([row["candidateGlbBytes"] for row in rows])),
        "predictedGlbBytes": float(np.mean([row["predictedGlbBytes"] for row in rows])),
        "glbCountReduction": float(np.mean([row["glbCountReduction"] for row in rows])),
        "glbByteReduction": float(np.mean([row["glbByteReduction"] for row in rows])),
        "glbBytesAtAchievedVisualUtility": float(np.mean([row["glbBytesAtAchievedVisualUtility"] for row in rows])),
    }
    latency = np.asarray([float(row["forwardLatencyMs"]) for row in rows], dtype=np.float64)
    return {
        "poseCount": len(rows),
        "poseMacro": macro,
        "aggregate": aggregate,
        "counts": {
            **sums,
            "weightedTp": weighted_tp,
            "weightedGt": weighted_gt,
            "visualUtilityTp": visual_utility_tp,
            "visualUtilityGt": visual_utility_gt,
            "downloadUtilityTp": download_utility_tp,
        },
        "runtimeLatency": {
            "meanMs": float(latency.mean()),
            "p50Ms": float(np.quantile(latency, 0.50)),
            "p95Ms": float(np.quantile(latency, 0.95)),
            "p99Ms": float(np.quantile(latency, 0.99)),
        },
    }


def _threshold_from_checkpoint(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    allow_unsafe_diagnostic: bool = False,
) -> tuple[float, dict[str, Any]]:
    calibration = checkpoint.get("calibration") or {}
    selected = select_weighted_cull_workpoint(
        calibration.get("rows", []), target_weighted_recall=0.99,
        minimum_lower_confidence_bound=0.99,
    )
    if selected is not None:
        return float(selected["threshold"]), {
            "source": "checkpoint.calibration",
            "protocol": "calibration_ready_pre_test",
            "selectionRule": weighted_cull_selection_rule(0.99, 0.99),
            "selection": selected,
            "safeWorkpoint": True,
            "workpointRole": "safety",
            "testEvaluationCount": 0,
        }
    if not allow_unsafe_diagnostic:
        raise ValueError(f"{checkpoint_path} has no qualified calibration threshold")
    rows = list(calibration.get("rows", []))
    diagnostic = calibration.get("diagnostic")
    if not isinstance(diagnostic, dict):
        if not rows:
            raise ValueError(f"{checkpoint_path} has no calibration rows for diagnostic evaluation")
        diagnostic = max(
            rows,
            key=lambda row: (
                float(row.get("pose_f1", 0.0)),
                float(row.get("pose_precision", 0.0)),
                float(row.get("pose_weighted_recall", 0.0)),
                -float(row.get("avg_pred_count", 0.0)),
            ),
        )
    return float(diagnostic["threshold"]), {
        "source": "checkpoint.calibration",
        "protocol": "calibration_ready_pre_test",
        "selectionRule": weighted_cull_selection_rule(0.99, 0.99),
        "selection": None,
        "diagnosticSelection": diagnostic,
        "safeWorkpoint": False,
        "workpointRole": "diagnostic_only_no_qualified_safety_workpoint",
        "testEvaluationCount": 0,
    }


def _build_model(
    checkpoint: dict[str, Any],
    runtime_meta_path: Path,
    evidence_dir: Path,
    device: torch.device,
) -> tuple[RayContextSurvivalOWRBModel, np.ndarray, np.ndarray, dict[str, Any]]:
    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    config = checkpoint["config"]
    model = RayContextSurvivalOWRBModel(
        num_instances=int(config["numInstances"]),
        num_glbs=int(config["numGlbs"]),
        point_hidden_dim=int(config.get("pointHiddenDim", 160)),
        pointnetpp_centers=int(config.get("pointnetppCenters", 24)),
        pointnetpp_neighbors=int(config.get("pointnetppNeighbors", 12)),
        relation_hidden_dim=int(config.get("relationHiddenDim", 64)),
        context_mode=str(config.get("contextMode", "directional")),
        survival_enabled=bool(config.get("survivalEnabled", True)),
        survival_input_mode=str(config.get("survivalInputMode", "semantic")),
        context_query_dim=int(config.get("contextQueryDim", 8)),
        ray_mode=str(config.get("rayMode", "direct9")),
        survival_parameterization=str(config.get("survivalParameterization", "monotone")),
        scene_size_m=scene_size.tolist(),
    ).to(device)
    model.set_scene_bounds(torch.from_numpy(scene_min).to(device), torch.from_numpy(scene_size).to(device))
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    # Evidence buffers are not needed by the runtime query, but loading them
    # makes the checkpoint schema fully reconstructable and parity-testable.
    from ray_context_survival_owrb_model import load_ray_context_survival_evidence
    evidence = load_ray_context_survival_evidence(evidence_dir, int(world_aabbs.shape[0]))
    model.set_relation_evidence(
        torch.from_numpy(evidence["source_ids"]).to(device),
        torch.from_numpy(evidence["relation_stats"]).to(device),
        torch.from_numpy(evidence["strength"]).to(device),
        torch.from_numpy(evidence["count"]).to(device),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), world_aabbs, instance_to_glb, runtime_meta


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model, world_aabbs, instance_to_glb, runtime_meta = _build_model(
        checkpoint, Path(args.runtime_meta), Path(args.evidence_dir), device
    )
    runtime_path = Path(args.runtime_features) if args.runtime_features else checkpoint_path.with_name("best_instance_runtime_features_fp16.bin")
    if not runtime_path.is_file():
        runtime_path = checkpoint_path.with_name("instance_runtime_features_fp16.bin")
    if not runtime_path.is_file():
        raise FileNotFoundError(f"missing runtime feature table beside checkpoint: {runtime_path}")
    values = np.fromfile(runtime_path, dtype=np.float16)
    expected = int(world_aabbs.shape[0]) * int(checkpoint["config"]["runtimeFeatureDim"])
    if values.size != expected:
        raise ValueError(f"runtime table has {values.size} values; expected {expected}")
    runtime_features = torch.from_numpy(values.reshape(world_aabbs.shape[0], -1).astype(np.float32)).to(device)
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=world_aabbs.shape[0])
    split = dataset.split(args.split)
    candidate_audit = audit_native_aabb_candidates(dataset, world_aabbs, split.pose_indices)
    threshold, threshold_source = _threshold_from_checkpoint(
        checkpoint_path, checkpoint, allow_unsafe_diagnostic=bool(args.allow_unsafe_diagnostic)
    )
    num_glbs = int(instance_to_glb.max()) + 1
    glb_bytes = _load_glb_bytes(Path(args.glb_index), Path(args.glb_root), num_glbs)
    rows: list[dict[str, Any]] = []
    score_pose_indices: list[int] = []
    score_offsets: list[int] = [0]
    score_candidate_ids: list[np.ndarray] = []
    score_targets: list[np.ndarray] = []
    score_weights: list[np.ndarray] = []
    score_values: list[np.ndarray] = []
    pose_indices = split.pose_indices[: int(args.max_poses)] if args.max_poses > 0 else split.pose_indices
    rng = np.random.default_rng(args.seed)
    for pose_index in pose_indices.tolist():
        pose_index = int(pose_index)
        batch = split.build_pose_set_batch(
            np.asarray([pose_index], dtype=np.int64), world_aabbs, rng,
            max_candidates_per_pose=0, allow_candidate_visible_union=False, include_empty=True,
        )
        offsets = np.asarray(batch["pose_offsets"], dtype=np.int64)
        start, end = int(offsets[0]), int(offsets[1])
        candidate_ids = np.asarray(batch["instance"][start:end], dtype=np.int64)
        if candidate_ids.size:
            ids = torch.from_numpy(candidate_ids).to(device)
            camera = torch.from_numpy(batch["camera"][start:end]).to(device)
            camera_world = torch.from_numpy(batch["camera_world"][start:end]).to(device)
            view = torch.from_numpy(batch["camera_view"][start:end]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits = model.compute_visibility_logits(camera, view, camera_world, ids, runtime_features=runtime_features)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_ms = (time.perf_counter() - started) * 1000.0
            scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        else:
            scores = np.zeros((0,), dtype=np.float32)
            forward_ms = 0.0
        glbs = instance_to_glb[candidate_ids] if candidate_ids.size else np.zeros((0,), dtype=np.int64)
        rows.append(_metric_row(
            batch["target"][start:end], scores, batch["visible_weights"][start:end], threshold,
            glbs, glb_bytes, forward_ms, pose_index, candidate_ids, bool(args.persist_ids),
        ))
        if args.persist_scores:
            score_pose_indices.append(pose_index)
            score_candidate_ids.append(candidate_ids.astype(np.uint32, copy=False))
            score_targets.append(np.asarray(batch["target"][start:end], dtype=np.uint8))
            score_weights.append(np.asarray(batch["visible_weights"][start:end], dtype=np.float32))
            score_values.append(np.asarray(scores, dtype=np.float32))
            score_offsets.append(score_offsets[-1] + int(candidate_ids.size))
    payload = _summarize(rows)
    canonical_candidate_digest = candidate_digest_for_pose_sequence(dataset, pose_indices)
    if args.persist_scores:
        artifact_path = Path(args.score_artifact) if args.score_artifact else Path(args.output).with_suffix(".scores.npz")
        score_artifact = _write_score_artifact(
            artifact_path,
            score_pose_indices,
            score_offsets,
            score_candidate_ids,
            score_targets,
            score_weights,
            score_values,
        )
    else:
        score_artifact = None
    payload.update({
        "schema": "ray-context-survival-owrb-evaluation-v1",
        "checkpoint": str(checkpoint_path),
        "checkpointSha256": _sha256_file(checkpoint_path),
        "runtimeFeatures": str(runtime_path.resolve()),
        "runtimeFeaturesSha256": _sha256_file(runtime_path),
        "runtimeFeatureBytes": int(runtime_path.stat().st_size),
        # The fixed table is uploaded/read separately from the cheap query
        # values appended before the visibility head.
        "runtimeFeatureDim": int(checkpoint["config"]["runtimeFeatureDim"]),
        "runtimeFeatureTableDim": int(checkpoint["config"]["runtimeFeatureDim"]),
        "runtimeInputFeatureDim": int(checkpoint["config"]["baseInputDim"]),
        "visibilityHeadInputDim": int(checkpoint["config"]["baseInputDim"]),
        "rayQueryDim": int(checkpoint["config"]["rayDim"]),
        "datasetDir": str(Path(args.dataset_dir).resolve()),
        "runtimeMeta": str(Path(args.runtime_meta).resolve()),
        "evidenceDir": str(Path(args.evidence_dir).resolve()),
        "split": args.split,
        "testRead": bool(args.split == "test"),
        "threshold": threshold,
        "thresholdSource": threshold_source,
        "safeWorkpoint": bool(threshold_source.get("safeWorkpoint", False)),
        "candidateDigest": canonical_candidate_digest,
        "candidateDigestScope": "canonical PoseCSR rows in the evaluated split order",
        "candidateAudit": candidate_audit,
        "seed": int(args.seed),
        "device": str(device),
        "poseIndices": [int(row["poseIndex"]) for row in rows],
        "semantics": {
            "candidate": "native back-camera frustum CSR; no GT-visible union",
            "visibleWeights": dataset.visible_weight_semantics,
            "threshold": "frozen from this checkpoint calibration only",
            "fov": {"modelAndBackCameraDegrees": 66.0, "frontendRenderDegrees": 60.0},
        },
        "imageMetrics": {"status": "not_available", "reason": "this evaluator has no Color-ID image replay input"},
        "scoreArtifact": score_artifact,
        "perPose": rows,
    })
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--glb-index", required=True)
    parser.add_argument("--glb-root", required=True)
    parser.add_argument("--runtime-features", default="")
    parser.add_argument("--split", choices=("validation", "calibration", "test"), default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-poses", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--persist-ids", action="store_true")
    parser.add_argument(
        "--persist-scores",
        action="store_true",
        help="write a compressed sidecar with per-candidate probabilities, labels, weights, and IDs",
    )
    parser.add_argument(
        "--score-artifact",
        default="",
        help="optional path for the compressed score sidecar; defaults beside --output",
    )
    parser.add_argument(
        "--allow-unsafe-diagnostic",
        action="store_true",
        help="evaluate the calibration-frozen diagnostic threshold when no safe workpoint exists; never treats it as safe",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "evaluated", "output": str(Path(args.output).resolve()), "split": args.split, "poseCount": payload["poseCount"], "threshold": payload["threshold"], "testRead": args.split == "test"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
