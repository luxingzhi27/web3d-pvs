#!/usr/bin/env python3
"""Train the relation-prior PVS model with per-instance calibration.

This entry reads train, calibration, and validation only.  The train-owned
relation graph first generates a shared prior.  A zero-initialized train-only
residual then calibrates each instance before both terms are fused into the
fixed 124-value runtime table.  Calibration alone freezes each checkpoint's
threshold, while validation is replayed only at that frozen threshold.  Test
is not a valid argument or fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from bounded_relation_survival_moment_model import (  # noqa: E402
    BoundedRelationSurvivalMomentModel,
    GEO_DIM,
    MODEL_SCHEMA,
    RUNTIME_FEATURE_DIM,
)
from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402
from common.occlusion_edges import glb_priority_loss  # noqa: E402
from common.provenance import relation_artifact_digest  # noqa: E402
from common.runtime_meta import load_runtime_meta  # noqa: E402
from common.safety_reserve_operating_utility_loss import (  # noqa: E402
    OPERATING_THRESHOLD_ANCHORS,
    OPERATING_THRESHOLD_RANGE,
    project_operating_utility_gradient_groups,
    safety_reserve_operating_utility_loss,
    sample_train_operating_thresholds,
)
from common.stratified_survival_sampler import StratifiedSurvivalObservationSampler  # noqa: E402
from common.survival_loss import stratified_survival_censoring_loss  # noqa: E402
from common.threshold_selection import (  # noqa: E402
    aggregate_weighted_cull_selection_rule,
    select_aggregate_weighted_cull_workpoint,
)
from common.train_observed_relation_csr import (  # noqa: E402
    ObservedRelationCSR,
    SCHEMA_V3 as RELATION_SCHEMA_V3,
    load_survival_observations_v3,
    validate_relation_metadata_v3,
    validate_survival_observations_v3,
)
from current_pvs_utils import evaluate_thresholds, threshold_grid, visual_utility_loss  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


EXPERIMENT_PREFIX = "pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4"
CALIBRATION_FLOOR = 0.99
SURVIVAL_SHAPE = (4, 7)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return float(value) if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_value(value), ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_safe_checkpoint_alias(payload: Mapping[str, Any], output_dir: Path) -> None:
    """Write one safe checkpoint and make best.pt its byte-identical alias."""
    safe_path = output_dir / "best_safe.pt"
    alias_path = output_dir / "best.pt"
    torch.save(payload, safe_path)
    shutil.copyfile(safe_path, alias_path)
    if _sha256(safe_path) != _sha256(alias_path):
        raise IOError("best.pt is not a byte-identical copy of best_safe.pt")


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty experiment output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _load_geometry(path: Path, num_instances: int) -> tuple[torch.Tensor, dict[str, Any]]:
    expected = int(num_instances) * GEO_DIM
    if not path.is_file() or path.stat().st_size != expected * 2:
        raise ValueError(f"fixed geometry must be [{num_instances}, {GEO_DIM}] FP16: {path}")
    values = np.fromfile(path, dtype="<f2").reshape(num_instances, GEO_DIM)
    if not bool(np.isfinite(values).all()):
        raise ValueError("fixed geometry contains non-finite values")
    return torch.from_numpy(values.astype(np.float32)), {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "shape": [num_instances, GEO_DIM],
        "dtype": "float16",
    }


def _read_uint32(path: Path, expected: int, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    values = np.fromfile(path, dtype="<u4").astype(np.int64)
    if values.size != int(expected):
        raise ValueError(f"{name} has {values.size} rows, expected {expected}")
    if values.size and (int(values.min()) < 0 or not np.array_equal(np.unique(values), np.arange(int(values.max()) + 1))):
        raise ValueError(f"{name} must use contiguous non-negative IDs")
    return values


def _load_relation_bundle(
    relation_dir: Path,
    dataset: PoseCSRDataset,
    train_split: Any,
    num_instances: int,
) -> tuple[
    ObservedRelationCSR,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    dict[str, np.ndarray],
    dict[str, Any],
]:
    relation = ObservedRelationCSR.load(relation_dir)
    if relation.metadata.get("schema") != RELATION_SCHEMA_V3:
        raise ValueError("v4 training refuses legacy relation artifacts")
    train_digest = candidate_digest_for_pose_sequence(dataset, train_split.pose_indices)
    validate_relation_metadata_v3(
        relation.metadata,
        expected_num_instances=num_instances,
        expected_candidate_digest=train_digest,
    )
    if relation.direction_bins != 12 or relation.depth_shells != 3:
        raise ValueError("v4 requires 12 directions and 3 ordered depth shells")
    hierarchy = relation.metadata["hierarchy"]
    hierarchy_files = hierarchy["files"]
    local = _read_uint32(
        relation_dir / hierarchy_files["localGroupIds"], num_instances, "local group IDs"
    )
    local_count = int(local.max()) + 1
    structural = _read_uint32(
        relation_dir / hierarchy_files["structuralGroupIds"],
        local_count,
        "structural group IDs",
    )
    local_counts = np.bincount(local, minlength=local_count)
    structural_counts = np.bincount(structural, minlength=int(structural.max()) + 1)
    structural_instance_counts = np.bincount(
        structural[local], minlength=int(structural.max()) + 1
    )
    if int(local_counts.max()) > 32 or int(structural_counts.max()) > 64:
        raise ValueError("loaded hierarchy violates v3 capacity bounds")
    if float(structural_instance_counts.max() / max(1, num_instances)) > 0.10 + 1e-9:
        raise ValueError("loaded hierarchy contains an oversized structural group")
    source_k = int(relation.metadata["evidenceTopK"]["k"])
    if np.any(np.diff(relation.row_offsets).astype(np.int64) > source_k):
        raise ValueError("relation CSR declares top-k but contains an oversized row")
    retained_quality = float(relation.metadata["evidenceTopK"].get("minRetainedQuality", 0.0))
    quality_floor = float(relation.metadata["evidenceTopK"].get("qualityFloor", 0.90))
    if retained_quality + 1e-9 < quality_floor:
        raise ValueError("relation evidence top-k retained quality is below its registered floor")

    envelope = relation.metadata.get("survivalObservations")
    if not isinstance(envelope, Mapping) or envelope.get("schema") != "pvs-viewcell-train-observed-survival-censoring-v3":
        raise ValueError("relation artifact lacks corrected v3 event/censor observations")
    files = envelope.get("files")
    if not isinstance(files, Mapping) or "normalizedDepth" not in files:
        raise ValueError("v3 observations must store normalizedDepth, not historical rho")
    loaded_observations = load_survival_observations_v3(relation_dir, relation.metadata)
    observations: dict[str, np.ndarray] = {
        "instance": loaded_observations["instance"],
        "direction": loaded_observations["direction"],
        "normalized_depth": loaded_observations["normalizedDepth"],
        "event": loaded_observations["event"],
        "weight": loaded_observations["weight"],
        "subpose": loaded_observations["subpose"],
        "rawPixelCount": loaded_observations["rawPixelCount"],
    }
    observation_files: dict[str, Any] = {}
    for name, file_key in (
        ("instance", "instance"),
        ("direction", "direction"),
        ("normalized_depth", "normalizedDepth"),
        ("event", "event"),
        ("weight", "weight"),
        ("subpose", "subpose"),
        ("rawPixelCount", "rawPixelCount"),
    ):
        relative = files[file_key]
        path = relation_dir / relative
        observation_files[name] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    validation_alias = {
        "instance": observations["instance"],
        "direction": observations["direction"],
        "normalizedDepth": observations["normalized_depth"],
        "event": observations["event"],
        "weight": observations["weight"],
        "subpose": observations["subpose"],
        "rawPixelCount": observations["rawPixelCount"],
    }
    observation_summary = validate_survival_observations_v3(
        validation_alias,
        num_instances=num_instances,
        direction_bins=relation.direction_bins,
    )
    if observation_summary["eventCount"] <= 0 or observation_summary["rightCensoredCount"] <= 0:
        raise ValueError("v3 survival supervision must contain both events and right-censored rows")
    observations["depth"] = observations["normalized_depth"]
    artifact_digest, artifact_files = relation_artifact_digest(relation_dir, relation.metadata)
    provenance = {
        "schema": relation.metadata["schema"],
        "path": str(relation_dir.resolve()),
        "artifactDigest": artifact_digest,
        "artifactFiles": artifact_files,
        "candidateDigest": train_digest,
        "edgeCount": relation.edge_count,
        "rowCount": relation.row_count,
        "hierarchy": {
            "localGroupCount": local_count,
            "structuralGroupCount": int(structural.max()) + 1,
            "localMaxSize": int(local_counts.max()),
            "structuralMaxSize": int(structural_counts.max()),
            "maxStructuralInstanceFraction": float(
                structural_instance_counts.max() / max(1, num_instances)
            ),
        },
        "observations": {**observation_summary, "files": observation_files},
    }
    return (
        relation,
        relation.to_torch(),
        torch.from_numpy(local),
        torch.from_numpy(structural),
        observations,
        provenance,
    )


def _load_glb_bytes(
    index_path: Path,
    root: Path,
    num_glbs: int,
    instance_to_glb: np.ndarray,
    *,
    allow_missing: bool,
) -> np.ndarray:
    if not index_path.is_file():
        raise FileNotFoundError(f"missing GLB index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    result = np.zeros((num_glbs,), dtype=np.float32)
    for entry in payload.get("entries", []):
        glb_id = int(entry.get("globalId", -1))
        path = root / str(entry.get("path", ""))
        if 0 <= glb_id < num_glbs and path.is_file():
            result[glb_id] = float(path.stat().st_size)
    used = np.unique(instance_to_glb.astype(np.int64))
    missing = used[(used < 0) | (used >= num_glbs) | (result[np.clip(used, 0, num_glbs - 1)] <= 0)]
    if missing.size and not allow_missing:
        raise FileNotFoundError(f"formal training has missing GLB byte costs: {missing[:20].tolist()}")
    if missing.size:
        fallback = float(np.median(result[result > 0])) if np.any(result > 0) else 1.0
        result[result <= 0] = max(1.0, fallback)
    return result


def _resolve_split(dataset: PoseCSRDataset, requested: str, role: str) -> tuple[str, Any]:
    if requested.lower() == "test":
        raise ValueError("test is forbidden during v4 training and threshold selection")
    if requested == "auto":
        choices = {
            "train": ("train",),
            "calibration": ("calibration",),
            "validation": ("validation", "val"),
        }[role]
        requested = next((value for value in choices if value in dataset.split_ids), "")
    if not requested or requested not in dataset.split_ids:
        raise ValueError(f"dataset has no explicit {role} split")
    split = dataset.split(requested)
    if split.pose_indices.size == 0:
        raise ValueError(f"{role} split is empty")
    return requested, split


def _normalized_log_support(counts: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(counts, dtype=np.float64).reshape(-1)
    positive = values[values > 0.0]
    if positive.size == 0:
        return np.zeros(values.shape, dtype=np.float32), 0.0
    reference = max(float(np.percentile(positive, 95.0)), 1.0)
    normalized = np.log1p(values) / math.log1p(reference)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32), reference


def _instance_calibration_reliability(
    dataset: PoseCSRDataset,
    train_split: Any,
    observations: Mapping[str, np.ndarray],
    num_instances: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a train-only reliability prior for regularizing instance residuals."""
    candidate_counts = np.zeros((int(num_instances),), dtype=np.int64)
    for pose_index in np.asarray(train_split.pose_indices, dtype=np.int64).tolist():
        ids = np.asarray(dataset.frustum_slice(int(pose_index)), dtype=np.int64)
        if ids.size:
            candidate_counts += np.bincount(ids, minlength=int(num_instances))
    observation_ids = np.asarray(observations["instance"], dtype=np.int64).reshape(-1)
    if observation_ids.size and (
        int(observation_ids.min()) < 0 or int(observation_ids.max()) >= int(num_instances)
    ):
        raise ValueError("survival observation instance is outside the runtime instance table")
    observation_counts = np.bincount(
        observation_ids, minlength=int(num_instances)
    ).astype(np.int64, copy=False)
    candidate_support, candidate_reference = _normalized_log_support(candidate_counts)
    observation_support, observation_reference = _normalized_log_support(observation_counts)
    reliability = np.maximum(candidate_support, observation_support).astype(np.float32, copy=False)
    if not np.isfinite(reliability).all() or np.any((reliability < 0.0) | (reliability > 1.0)):
        raise FloatingPointError("instance calibration reliability is invalid")
    raw = np.ascontiguousarray(reliability.astype("<f4", copy=False)).tobytes(order="C")
    return reliability, {
        "sourceSplit": "train",
        "definition": "max(log1p(candidate_count)/log1p(candidate_p95), log1p(observation_count)/log1p(observation_p95))",
        "candidateCountP95Reference": candidate_reference,
        "observationCountP95Reference": observation_reference,
        "candidateReferenceCount": int(candidate_counts.sum()),
        "observationReferenceCount": int(observation_counts.sum()),
        "zeroReliabilityCount": int(np.count_nonzero(reliability <= 0.0)),
        "meanReliability": float(reliability.mean()),
        "p50Reliability": float(np.percentile(reliability, 50.0)),
        "p95Reliability": float(np.percentile(reliability, 95.0)),
        "sha256Float32": hashlib.sha256(raw).hexdigest(),
        "testRead": False,
    }


def _instance_calibration_blend(
    completed_steps: int,
    total_steps: int,
    *,
    warmup_fraction: float,
    ramp_fraction: float,
) -> float:
    if total_steps <= 0 or completed_steps < 0:
        raise ValueError("calibration schedule steps must be non-negative with a positive total")
    if not 0.0 <= warmup_fraction < 1.0 or not 0.0 < ramp_fraction <= 1.0:
        raise ValueError("calibration warmup/ramp fractions are invalid")
    warmup_steps = int(math.ceil(float(total_steps) * float(warmup_fraction)))
    ramp_steps = max(1, int(math.ceil(float(total_steps) * float(ramp_fraction))))
    if int(completed_steps) < warmup_steps:
        return 0.0
    return float(min(1.0, (int(completed_steps) - warmup_steps + 1) / ramp_steps))


def _runtime_features(
    model: BoundedRelationSurvivalMomentModel,
    geometry: torch.Tensor,
    relation: Mapping[str, torch.Tensor],
    local_ids: torch.Tensor,
    structural_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    diagnostics = model.offline_encode_survival(
        geometry,
        relation,
        local_ids,
        structural_ids,
        return_diagnostics=True,
    )
    if not isinstance(diagnostics, Mapping):
        raise TypeError("offline encoder did not return calibration diagnostics")
    coefficients = diagnostics.get("survival_coefficients")
    if not isinstance(coefficients, torch.Tensor) or coefficients.shape != (
        geometry.shape[0], *SURVIVAL_SHAPE
    ):
        raise ValueError("offline encoder did not produce [N, 4, 7]")
    runtime = torch.cat([geometry, coefficients.reshape(geometry.shape[0], -1)], dim=-1)
    if runtime.shape != (geometry.shape[0], RUNTIME_FEATURE_DIM) or not bool(torch.isfinite(runtime).all()):
        raise FloatingPointError("v4 fixed runtime table is invalid")
    return runtime, coefficients, dict(diagnostics)


def _gradients(
    objective: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    if not objective.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _assign_projected_gradients(
    parameters: Sequence[torch.nn.Parameter],
    groups: Mapping[str, Sequence[torch.Tensor | None]],
) -> None:
    for index, parameter in enumerate(parameters):
        values = [group[index] for group in groups.values() if group[index] is not None]
        parameter.grad = None if not values else sum(values[1:], values[0].clone())


def _float_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                result[key] = float(value.detach().cpu())
        elif isinstance(value, (float, int, np.generic)):
            result[key] = float(value)
    return result


def _diagnostic_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
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


def _calibration_workpoints(
    calibration_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Choose both workpoints from one checkpoint's calibration rows only."""
    selected = select_aggregate_weighted_cull_workpoint(
        calibration_rows,
        target_weighted_recall=CALIBRATION_FLOOR,
        minimum_lower_confidence_bound=CALIBRATION_FLOOR,
    )
    diagnostic = _diagnostic_row(calibration_rows)
    frozen = selected if selected is not None else diagnostic
    return selected, diagnostic, frozen


def _v4_objective_groups(
    safety: torch.Tensor,
    survival: torch.Tensor,
    relation_consistency: torch.Tensor,
    utility: torch.Tensor,
    download: torch.Tensor,
    regularization: torch.Tensor,
    instance_calibration_regularization: torch.Tensor,
    efficiency: torch.Tensor,
    *,
    survival_weight: float,
    relation_consistency_weight: float,
    utility_weight: float,
    download_weight: float,
    regularization_weight: float,
    instance_calibration_regularization_weight: float,
) -> dict[str, torch.Tensor]:
    """Build the four logged groups and the exact scalar optimized by the step."""
    relation = (
        float(survival_weight) * survival
        + float(relation_consistency_weight) * relation_consistency
        + float(regularization_weight) * regularization
        + float(instance_calibration_regularization_weight)
        * instance_calibration_regularization
    )
    schedule = float(utility_weight) * utility + float(download_weight) * download
    return {
        "safety": safety,
        "relation": relation,
        "schedule": schedule,
        "efficiency": efficiency,
        "total": safety + relation + schedule + efficiency,
    }


def _operating_threshold_metrics(
    train_seed: int,
    global_step: int,
) -> tuple[tuple[float, ...], dict[str, float]]:
    thresholds = sample_train_operating_thresholds(
        train_seed,
        global_step,
        split="train",
    )
    metrics = {
        "operatingThresholdCount": float(len(thresholds)),
        **{
            f"operatingThreshold{index}": float(value)
            for index, value in enumerate(thresholds)
        },
    }
    return thresholds, metrics


def _evaluation_batch_limits(max_poses: int, poses_per_batch: int) -> tuple[int, int | None]:
    """Choose an exact diagnostic pose cap without changing full evaluation."""
    batch_size = int(poses_per_batch)
    pose_limit = int(max_poses)
    if batch_size <= 0 or pose_limit < 0:
        raise ValueError("poses_per_batch must be positive and max_poses non-negative")
    if pose_limit == 0:
        # Zero is the registered full-split value, not one batch.  Passing
        # ``None`` lets PoseCSRSplit iterate every pose deterministically.
        return batch_size, None
    exact_batch_size = math.gcd(batch_size, pose_limit)
    return exact_batch_size, pose_limit // exact_batch_size


def _evaluate(
    model: BoundedRelationSurvivalMomentModel,
    split: Any,
    runtime: torch.Tensor,
    world_aabbs: np.ndarray,
    device: torch.device,
    *,
    thresholds: np.ndarray,
    seed: int,
    poses_per_batch: int,
    max_poses: int,
    bootstrap_replicates: int,
    instance_to_glb: np.ndarray | None = None,
    glb_bytes: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    evaluation_batch_size, max_steps = _evaluation_batch_limits(max_poses, poses_per_batch)
    return evaluate_thresholds(
        model,
        split,
        runtime,
        world_aabbs,
        device,
        poses_per_batch=evaluation_batch_size,
        max_steps=max_steps,
        max_candidates_per_pose=0,
        seed=seed,
        thresholds=thresholds,
        collect_pose_stats=True,
        allow_candidate_visible_union=False,
        bootstrap_replicates=bootstrap_replicates,
        instance_to_glb=instance_to_glb,
        glb_bytes=glb_bytes,
    )


def _checkpoint(
    model: BoundedRelationSurvivalMomentModel,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    protocol: Mapping[str, Any],
    relation_provenance: Mapping[str, Any],
    geometry_meta: Mapping[str, Any],
    coefficients: torch.Tensor,
    coefficient_diagnostics: Mapping[str, Any],
    calibration_reliability_meta: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    best: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prior = coefficient_diagnostics.get("survival_prior_coefficients")
    residual = coefficient_diagnostics.get("instance_calibration_applied_residual")
    if not isinstance(prior, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise ValueError("checkpoint requires prior and applied instance calibration tensors")
    if prior.shape != coefficients.shape or residual.shape != coefficients.shape:
        raise ValueError("checkpoint calibration tensors disagree with fused coefficients")
    if not torch.allclose(coefficients, prior + residual, rtol=1e-5, atol=1e-6):
        raise ValueError("checkpoint fused coefficients do not equal prior plus calibration residual")
    return {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
        "runtimeSchema": MODEL_SCHEMA,
        "experimentName": args.experiment_name,
        "epoch": int(epoch),
        "globalStep": int(global_step),
        "modelConfig": model.config,
        "modelState": model.state_dict(),
        "instanceSurvivalCoefficients": coefficients.detach().cpu().half(),
        "instanceSurvivalPriorCoefficients": prior.detach().cpu().half(),
        "instanceSurvivalCalibrationResidual": residual.detach().cpu().half(),
        "instanceCalibration": {
            "mode": model.instance_calibration_mode,
            "blend": float(model.instance_calibration_blend.detach().cpu()),
            "fusion": "prior_plus_applied_residual",
            "reliability": dict(calibration_reliability_meta),
            "runtimeExport": "fused_coefficients_only",
        },
        "protocol": dict(protocol),
        "relation": dict(relation_provenance),
        "geometry": dict(geometry_meta),
        "calibration": calibration,
        "validationAtCalibration": validation,
        "best": best,
        "testRead": False,
    }


def _save_fp16(path: Path, values: torch.Tensor) -> None:
    values.detach().cpu().numpy().astype("<f2").tofile(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--initial-geo-features", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--subpose-sidecar", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", default=EXPERIMENT_PREFIX)
    parser.add_argument("--variant", default="full")
    parser.add_argument(
        "--relation-source",
        choices=("bounded_hierarchical", "geometry_only"),
        default="bounded_hierarchical",
    )
    parser.add_argument(
        "--spectral-mode", choices=("moment_envelope", "point"), default="moment_envelope"
    )
    parser.add_argument(
        "--instance-calibration-mode",
        choices=("residual", "disabled"),
        default="residual",
    )
    parser.add_argument("--instance-calibration-max-abs", type=float, default=4.0)
    parser.add_argument("--sparse-instance-penalty", type=float, default=3.0)
    parser.add_argument("--instance-calibration-warmup-fraction", type=float, default=0.10)
    parser.add_argument("--instance-calibration-ramp-fraction", type=float, default=0.20)
    parser.add_argument("--loss-variant", choices=("safety_reserve", "normalized_rvl"), default="safety_reserve")
    parser.add_argument("--train-split", default="auto")
    parser.add_argument("--calibration-split", default="auto")
    parser.add_argument("--validation-split", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--observation-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--survival-loss-weight", type=float, default=0.25)
    parser.add_argument("--relation-consistency-weight", type=float, default=0.10)
    parser.add_argument("--utility-loss-weight", type=float, default=0.10)
    parser.add_argument("--download-loss-weight", type=float, default=0.10)
    parser.add_argument("--regularization-weight", type=float, default=1e-5)
    parser.add_argument("--instance-calibration-regularization-weight", type=float, default=0.02)
    parser.add_argument("--boundary-tail-weight", type=float, default=0.30)
    parser.add_argument("--negative-band-weight", type=float, default=0.03)
    parser.add_argument("--negative-band-temperature", type=float, default=0.10)
    parser.add_argument(
        "--negative-band-shape", choices=("sigmoid", "softplus"), default="sigmoid"
    )
    parser.add_argument("--glb-resource-weight", type=float, default=0.015)
    parser.add_argument("--rvl-bce-positive-weight", type=float, default=14.0)
    parser.add_argument("--rvl-tversky-fn-weight", type=float, default=7.0)
    parser.add_argument("--rvl-count-weight", type=float, default=0.10)
    parser.add_argument(
        "--rvl-fp-normalization",
        choices=("positive", "negative", "candidate"),
        default="positive",
    )
    parser.add_argument("--efficiency-warmup-fraction", type=float, default=0.10)
    parser.add_argument("--efficiency-primary-fraction", type=float, default=0.0)
    parser.add_argument("--rvl-rank-weight", type=float, default=0.45)
    parser.add_argument("--rvl-rank-negative-top-k", type=int, default=256)
    parser.add_argument("--relation-gradient-cap", type=float, default=0.25)
    parser.add_argument("--schedule-gradient-cap", type=float, default=0.25)
    parser.add_argument("--efficiency-gradient-cap", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--max-eval-poses", type=int, default=0)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-missing-glb-costs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.steps_per_epoch, args.poses_per_batch, args.observation_batch_size) <= 0:
        raise ValueError("epoch, step, pose-batch, and observation-batch counts must be positive")
    if args.eval_every <= 0 or args.calibration_bootstrap_replicates <= 0:
        raise ValueError("evaluation cadence and calibration bootstrap count must be positive")
    if (
        args.instance_calibration_max_abs <= 0.0
        or args.sparse_instance_penalty < 0.0
        or args.instance_calibration_regularization_weight < 0.0
    ):
        raise ValueError("instance calibration magnitude and regularization arguments are invalid")
    if (
        args.rvl_bce_positive_weight <= 0.0
        or args.rvl_tversky_fn_weight <= 0.0
        or args.rvl_count_weight < 0.0
        or not 0.0 <= args.efficiency_warmup_fraction <= 1.0
        or not 0.0 <= args.efficiency_primary_fraction <= 1.0
        or args.rvl_rank_weight < 0.0
        or args.rvl_rank_negative_top_k <= 0
        or args.negative_band_temperature <= 0.0
    ):
        raise ValueError("RVL diagnostic weights or efficiency warmup are invalid")
    _instance_calibration_blend(
        0,
        int(args.epochs) * int(args.steps_per_epoch),
        warmup_fraction=args.instance_calibration_warmup_fraction,
        ramp_fraction=args.instance_calibration_ramp_fraction,
    )
    if args.allow_missing_glb_costs and args.epochs > 12:
        raise ValueError("synthetic GLB costs are restricted to short diagnostic runs")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    device = _device(args.device)
    world_aabbs, instance_to_glb_np, runtime_meta = load_runtime_meta(args.runtime_meta)
    num_instances = int(world_aabbs.shape[0])
    num_glbs = int(runtime_meta.get("numGlbs", int(instance_to_glb_np.max()) + 1))
    dataset = PoseCSRDataset(
        args.dataset_dir,
        num_instances=num_instances,
        subpose_sidecar=args.subpose_sidecar,
    )
    if dataset.query_centers_world is None or dataset.candidate_cameras_world is None or dataset.viewcell_radii_m is None:
        raise ValueError("v3 PoseCSR must explicitly store query center, candidate camera, and view-cell radius")
    if np.allclose(dataset.query_centers_world, dataset.candidate_cameras_world, rtol=0.0, atol=1e-6):
        raise ValueError("query centers must not be silently replaced by candidate back-camera centers")
    model_fov = float(dataset.meta.get("modelInputFovYDeg", dataset.meta.get("fovYDeg", float("nan"))))
    if not math.isfinite(model_fov) or not math.isclose(model_fov, 66.0, abs_tol=1e-5):
        raise ValueError("v3 candidates and model input must use the registered 66 degree FOV")

    train_name, train_split = _resolve_split(dataset, args.train_split, "train")
    calibration_name, calibration_split = _resolve_split(dataset, args.calibration_split, "calibration")
    validation_name, validation_split = _resolve_split(dataset, args.validation_split, "validation")
    split_pose_sets = [
        set(train_split.pose_indices.tolist()),
        set(calibration_split.pose_indices.tolist()),
        set(validation_split.pose_indices.tolist()),
    ]
    if split_pose_sets[0] & split_pose_sets[1] or split_pose_sets[0] & split_pose_sets[2] or split_pose_sets[1] & split_pose_sets[2]:
        raise ValueError("train, calibration, and validation pose sets must be disjoint")
    allowed_pose_indices = np.concatenate(
        [train_split.pose_indices, calibration_split.pose_indices, validation_split.pose_indices]
    ).astype(np.int64, copy=False)
    viewcell_radii = np.asarray(
        dataset.viewcell_radii_m[allowed_pose_indices], dtype=np.float32
    ).reshape(-1)
    if (
        viewcell_radii.size == 0
        or not np.isfinite(viewcell_radii).all()
        or np.any(viewcell_radii <= 0.0)
    ):
        raise ValueError("permitted PoseCSR view-cell radii must be finite and positive")
    viewcell_radius_m = float(viewcell_radii[0])
    if not np.allclose(viewcell_radii, viewcell_radius_m, rtol=0.0, atol=1e-6):
        raise ValueError("the registered v4 runtime requires one fixed view-cell radius")

    geometry_cpu, geometry_meta = _load_geometry(args.initial_geo_features, num_instances)
    (
        relation,
        relation_cpu,
        local_cpu,
        structural_cpu,
        observations_np,
        relation_provenance,
    ) = _load_relation_bundle(args.relation_dir, dataset, train_split, num_instances)
    depth_meta = relation.metadata["depthNormalization"]
    depth_q01 = float(depth_meta["q01"])
    depth_q99 = float(depth_meta["q99"])
    depth_epsilon = float(depth_meta["epsilon"])
    if not depth_q99 > depth_q01 or depth_epsilon <= 0.0 or depth_meta.get("sourceSplit") != "train":
        raise ValueError("relation depth normalization is not a valid train-frozen contract")

    sampler = StratifiedSurvivalObservationSampler(
        observations_np,
        seed=args.seed,
        metadata={"trainOnly": True, "splitNames": ["train"]},
    )
    required_sampler_steps = math.ceil(
        sampler.unique_instances.size / int(args.observation_batch_size)
    )
    if int(args.steps_per_epoch) < required_sampler_steps:
        raise ValueError(
            f"steps_per_epoch={args.steps_per_epoch} cannot cover all observed instances; "
            f"need at least {required_sampler_steps}"
        )
    calibration_reliability_np, calibration_reliability_meta = (
        _instance_calibration_reliability(
            dataset,
            train_split,
            observations_np,
            num_instances,
        )
    )
    glb_bytes_np = _load_glb_bytes(
        args.glb_index,
        args.glb_root,
        num_glbs,
        instance_to_glb_np,
        allow_missing=args.allow_missing_glb_costs,
    )
    glb_cost_norm_np = np.log1p(glb_bytes_np)
    glb_cost_norm_np /= max(float(glb_cost_norm_np.max()), 1e-6)

    geometry = geometry_cpu.to(device)
    relation_tensors: dict[str, Any] = {
        key: value.to(device) for key, value in relation_cpu.items()
    }
    relation_tensors["metadata"] = relation.metadata
    local_ids = local_cpu.to(device)
    structural_ids = structural_cpu.to(device)
    glb_bytes = torch.from_numpy(glb_bytes_np).to(device)
    glb_cost_norm = torch.from_numpy(glb_cost_norm_np.astype(np.float32)).to(device)
    model = BoundedRelationSurvivalMomentModel(
        num_instances,
        num_glbs,
        relation_source=args.relation_source,
        spectral_mode=args.spectral_mode,
        depth_q01=depth_q01,
        depth_q99=depth_q99,
        depth_epsilon=depth_epsilon,
        instance_calibration_mode=args.instance_calibration_mode,
        instance_calibration_max_abs=args.instance_calibration_max_abs,
        sparse_instance_penalty=args.sparse_instance_penalty,
    ).to(device)
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb_np).to(device))
    model.set_instance_calibration_reliability(
        torch.from_numpy(calibration_reliability_np).to(device)
    )

    protocol = {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4",
        "experiment": args.experiment_name,
        "variant": args.variant,
        "lossVariant": args.loss_variant,
        "rvlDiagnostics": {
            "bcePositiveWeight": float(args.rvl_bce_positive_weight),
            "tverskyFnWeight": float(args.rvl_tversky_fn_weight),
            "countWeight": float(args.rvl_count_weight),
            "fpNormalization": str(args.rvl_fp_normalization),
            "efficiencyWarmupFraction": float(args.efficiency_warmup_fraction),
            "efficiencyPrimaryFraction": float(args.efficiency_primary_fraction),
            "rankWeight": float(args.rvl_rank_weight),
            "rankNegativeTopK": int(args.rvl_rank_negative_top_k),
            "negativeBandTemperature": float(args.negative_band_temperature),
            "negativeBandShape": str(args.negative_band_shape),
        },
        "seed": int(args.seed),
        "testRead": False,
        "thresholdSource": "checkpoint-own-calibration-only",
        "operatingThresholdSampling": {
            "source": "train-seed-and-global-step-only",
            "range": [float(value) for value in OPERATING_THRESHOLD_RANGE],
            "sampleCount": 2,
            "anchors": [float(value) for value in OPERATING_THRESHOLD_ANCHORS],
            "anchorPeriod": 4,
        },
        "candidateUnion": False,
        "fov": {"candidateAndModelDeg": 66.0, "realRenderDeg": 60.0},
        "viewcell": {
            "shape": "horizontal_disk",
            "radiusM": viewcell_radius_m,
            "candidateCameraSemantics": "66-degree back-camera candidate identity only",
            "queryCenterSemantics": "center of the same-direction view-cell visibility union",
        },
        "splitNames": {
            "train": train_name,
            "calibration": calibration_name,
            "validation": validation_name,
        },
        "splitPoseCounts": {
            "train": int(train_split.pose_indices.size),
            "calibration": int(calibration_split.pose_indices.size),
            "validation": int(validation_split.pose_indices.size),
        },
        "candidateDigests": {
            "train": candidate_digest_for_pose_sequence(dataset, train_split.pose_indices),
            "calibration": candidate_digest_for_pose_sequence(dataset, calibration_split.pose_indices),
            "validation": candidate_digest_for_pose_sequence(dataset, validation_split.pose_indices),
        },
        "dataset": {
            "path": str(args.dataset_dir.resolve()),
            "metaSha256": _sha256(args.dataset_dir / "dataset_meta.json"),
        },
        "runtimeMeta": {
            "path": str(args.runtime_meta.resolve()),
            "sha256": _sha256(args.runtime_meta),
        },
        "relation": relation_provenance,
        "sampler": sampler.manifest(),
        "instanceCalibration": {
            "mode": args.instance_calibration_mode,
            "maximumAbsoluteResidual": float(args.instance_calibration_max_abs),
            "sparseInstancePenalty": float(args.sparse_instance_penalty),
            "regularizationWeight": float(
                args.instance_calibration_regularization_weight
            ),
            "warmupFraction": float(args.instance_calibration_warmup_fraction),
            "rampFraction": float(args.instance_calibration_ramp_fraction),
            "reliability": calibration_reliability_meta,
            "runtimeExport": "fused_coefficients_only",
        },
    }
    if protocol["candidateDigests"]["train"] != relation_provenance["candidateDigest"]:
        raise ValueError("relation and PoseCSR train candidate hashes differ")

    _prepare_output(args.output_dir)
    manifest = {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-run-manifest-v4",
        "createdAtUnix": time.time(),
        "arguments": vars(args),
        "modelConfig": model.config,
        "protocol": protocol,
        "geometry": geometry_meta,
        "glbBytes": {
            "count": num_glbs,
            "sum": float(glb_bytes_np.sum()),
            "syntheticMissingAllowed": bool(args.allow_missing_glb_costs),
        },
        "device": {
            "type": str(device),
            "cudaName": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "testRead": False,
    }
    _write_json(args.output_dir / "run_manifest.json", manifest)
    _write_json(args.output_dir / "model_schema.json", model.export_schema())

    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    total_steps = int(args.epochs) * int(args.steps_per_epoch)
    history: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "train_metrics.jsonl"
    best_safe_key: tuple[float, ...] | None = None
    best_diagnostic_key: tuple[float, ...] | None = None
    best_safe: dict[str, Any] | None = None
    best_diagnostic: dict[str, Any] | None = None
    best_safe_calibration: dict[str, Any] | None = None
    best_diagnostic_calibration: dict[str, Any] | None = None
    best_safe_validation: dict[str, Any] | None = None
    best_diagnostic_validation: dict[str, Any] | None = None
    global_step = 0
    started = time.time()

    for epoch in range(int(args.epochs)):
        model.train()
        sampler.start_epoch(epoch)
        rng = np.random.default_rng(int(args.seed) + epoch * 1009)
        epoch_metrics: dict[str, list[float]] = {}
        processed_steps = 0
        pose_batches = train_split.pose_set_batches(
            args.poses_per_batch,
            rng,
            max_steps=args.steps_per_epoch,
            include_empty=False,
        )
        for step, pose_indices in enumerate(pose_batches):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            batch = train_split.build_pose_set_batch(
                pose_indices,
                world_aabbs,
                rng,
                max_candidates_per_pose=0,
                allow_candidate_visible_union=False,
                include_empty=False,
            )
            if batch["instance"].size == 0:
                continue
            processed_steps += 1
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            model.set_instance_calibration_blend(
                _instance_calibration_blend(
                    global_step - 1,
                    total_steps,
                    warmup_fraction=args.instance_calibration_warmup_fraction,
                    ramp_fraction=args.instance_calibration_ramp_fraction,
                )
            )
            runtime_features, coefficients, coefficient_diagnostics = _runtime_features(
                model, geometry, relation_tensors, local_ids, structural_ids
            )
            camera = torch.from_numpy(batch["camera"]).to(device)
            camera_view = torch.from_numpy(batch["camera_view"]).to(device)
            candidate_camera = torch.from_numpy(batch["candidate_camera_world"]).to(device)
            query_center = torch.from_numpy(batch["query_center_world"]).to(device)
            viewcell_radius = torch.from_numpy(batch["viewcell_radius_m"]).to(device)
            instance_ids = torch.from_numpy(batch["instance"]).to(device)
            target = torch.from_numpy(batch["target"]).to(device)
            visible_weights = torch.from_numpy(batch["visible_weights"]).to(device)
            visible_hit_rates = torch.from_numpy(batch["visible_hit_rates"]).to(device)
            pose_offsets = torch.from_numpy(batch["pose_offsets"]).to(device)
            logits, aux = model.compute_logits_with_aux(
                camera,
                camera_view,
                candidate_camera,
                instance_ids,
                runtime_features=runtime_features,
                query_center_world=query_center,
                viewcell_radius_m=viewcell_radius,
            )

            use_safety_reserve = args.loss_variant == "safety_reserve"
            _combined_visibility, visibility_parts = safety_reserve_operating_utility_loss(
                logits,
                target,
                pose_offsets,
                visible_weights,
                visible_hit_rates,
                instance_ids,
                model.instance_to_glb,
                glb_bytes,
                train_seed=args.seed,
                global_step=global_step - 1,
                total_optimizer_steps=total_steps,
                split="train",
                boundary_tail_weight=args.boundary_tail_weight if use_safety_reserve else 0.0,
                negative_band_weight=args.negative_band_weight if use_safety_reserve else 0.0,
                threshold_temperature=args.negative_band_temperature,
                negative_band_shape=args.negative_band_shape,
                glb_resource_weight=args.glb_resource_weight if use_safety_reserve else 0.0,
                warmup_fraction=args.efficiency_warmup_fraction,
                rvl_bce_positive_weight=args.rvl_bce_positive_weight,
                rvl_tversky_fn_weight=args.rvl_tversky_fn_weight,
                rvl_count_weight=args.rvl_count_weight,
                rvl_fp_normalization=args.rvl_fp_normalization,
                rvl_rank_weight=args.rvl_rank_weight,
                rvl_rank_negative_top_k=args.rvl_rank_negative_top_k,
            )
            efficiency_primary_fraction = float(args.efficiency_primary_fraction)
            safety_objective = (
                visibility_parts["lossSafety"]
                + efficiency_primary_fraction
                * visibility_parts["lossEfficiencyUngated"]
            )
            efficiency_objective = (
                (1.0 - efficiency_primary_fraction)
                * visibility_parts["lossEfficiency"]
            )
            assert isinstance(safety_objective, torch.Tensor)
            assert isinstance(efficiency_objective, torch.Tensor)

            sample = sampler.sample_batch(args.observation_batch_size, step=step)
            sampled_observations = sampler.gather(sample, device=device)
            sampled_observations["normalized_depth"] = sampled_observations["depth"]
            survival_loss, survival_parts = stratified_survival_censoring_loss(
                model,
                coefficients,
                sampled_observations,
            )
            relation_loss, relation_parts = model.relation_consistency_loss(
                geometry,
                relation_tensors,
                seed=args.seed + global_step,
                max_edges=args.observation_batch_size,
            )
            utility_loss, utility_parts = visual_utility_loss(
                aux,
                target,
                visible_weights,
                pose_offsets,
                invisible_weight=0.25,
                rank_weight=0.30,
                rank_margin=0.10,
                rank_positive_top_k=32,
                rank_negative_top_k=128,
            )
            download_loss, download_parts = glb_priority_loss(
                aux["download_logits"],
                instance_ids,
                target,
                visible_weights,
                model.instance_to_glb,
                glb_cost_norm,
                pose_offsets,
            )
            regularization = model.regularization()
            instance_calibration_regularization = (
                model.instance_calibration_regularization()
            )
            objective_groups = _v4_objective_groups(
                safety_objective,
                survival_loss,
                relation_loss,
                utility_loss,
                download_loss,
                regularization,
                instance_calibration_regularization,
                efficiency_objective,
                survival_weight=args.survival_loss_weight,
                relation_consistency_weight=args.relation_consistency_weight,
                utility_weight=args.utility_loss_weight,
                download_weight=args.download_loss_weight,
                regularization_weight=args.regularization_weight,
                instance_calibration_regularization_weight=(
                    args.instance_calibration_regularization_weight
                ),
            )
            safety_objective = objective_groups["safety"]
            relation_objective = objective_groups["relation"]
            schedule_objective = objective_groups["schedule"]
            efficiency_objective = objective_groups["efficiency"]
            objective = objective_groups["total"]
            components = (
                objective,
                safety_objective,
                relation_objective,
                schedule_objective,
                efficiency_objective,
            )
            if not bool(torch.isfinite(torch.stack([value.float() for value in components])).all()):
                raise FloatingPointError(f"non-finite v4 objective at epoch {epoch + 1}, step {step + 1}")

            safety_gradients = _gradients(safety_objective, parameters, retain_graph=True)
            relation_gradients = _gradients(relation_objective, parameters, retain_graph=True)
            schedule_gradients = _gradients(schedule_objective, parameters, retain_graph=True)
            efficiency_gradients = _gradients(efficiency_objective, parameters, retain_graph=False)
            projected, gradient_parts = project_operating_utility_gradient_groups(
                safety_gradients,
                relation_gradients,
                schedule_gradients,
                efficiency_gradients,
                relation_norm_cap=args.relation_gradient_cap,
                schedule_norm_cap=args.schedule_gradient_cap,
                efficiency_norm_cap=args.efficiency_gradient_cap,
            )
            _assign_projected_gradients(parameters, projected)
            bad_gradients = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            ]
            if bad_gradients:
                raise FloatingPointError(f"non-finite v4 gradients: {bad_gradients}")
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_started
            training_elapsed = time.time() - started
            remaining_steps = max(0, total_steps - global_step)
            eta_seconds = (
                training_elapsed / max(1, global_step) * remaining_steps
            )
            runtime_metrics: dict[str, float] = {
                "stepSeconds": float(step_seconds),
                "stepsPerSecond": float(1.0 / max(step_seconds, 1e-12)),
                "trainingElapsedSeconds": float(training_elapsed),
                "estimatedRemainingSeconds": float(eta_seconds),
            }
            if device.type == "cuda":
                runtime_metrics.update(
                    {
                        "cudaMemoryAllocatedBytes": float(torch.cuda.memory_allocated(device)),
                        "cudaPeakMemoryAllocatedBytes": float(torch.cuda.max_memory_allocated(device)),
                        "cudaPeakMemoryReservedBytes": float(torch.cuda.max_memory_reserved(device)),
                    }
                )

            operating_thresholds, threshold_metrics = _operating_threshold_metrics(
                args.seed,
                global_step - 1,
            )
            step_metrics = _float_metrics(
                {
                    "loss": objective,
                    "lossSafetyGroup": safety_objective,
                    "lossRelationGroup": relation_objective,
                    "lossScheduleGroup": schedule_objective,
                    "lossEfficiencyGroup": efficiency_objective,
                    "lossSurvivalWeighted": float(args.survival_loss_weight) * survival_loss,
                    "lossRelationConsistencyWeighted": float(args.relation_consistency_weight) * relation_loss,
                    "lossUtilityWeighted": float(args.utility_loss_weight) * utility_loss,
                    "lossDownloadWeighted": float(args.download_loss_weight) * download_loss,
                    "lossRegularizationWeighted": float(args.regularization_weight) * regularization,
                    "lossInstanceCalibrationRegularizationWeighted": (
                        float(args.instance_calibration_regularization_weight)
                        * instance_calibration_regularization
                    ),
                    **visibility_parts,
                    **survival_parts,
                    **relation_parts,
                    **utility_parts,
                    **download_parts,
                    **gradient_parts,
                    **model.instance_calibration_diagnostics(),
                    **threshold_metrics,
                    **runtime_metrics,
                }
            )
            group_sum = (
                step_metrics["lossSafetyGroup"]
                + step_metrics["lossRelationGroup"]
                + step_metrics["lossScheduleGroup"]
                + step_metrics["lossEfficiencyGroup"]
            )
            if not math.isclose(step_metrics["loss"], group_sum, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError("logged v4 loss groups do not sum to the actual objective")
            for key, value in step_metrics.items():
                epoch_metrics.setdefault(key, []).append(value)
            if step == 0 or (step + 1) % max(1, args.steps_per_epoch // 5) == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "globalStep": global_step,
                            "loss": step_metrics["loss"],
                            "safetyGate": step_metrics.get("safetyReserveGate", 0.0),
                            "operatingThresholds": list(operating_thresholds),
                            "observationInstanceCoverage": sampler.epoch_instance_coverage,
                            "stepSeconds": step_metrics["stepSeconds"],
                            "etaSeconds": step_metrics["estimatedRemainingSeconds"],
                            "cudaPeakMemoryAllocatedMiB": (
                                step_metrics.get("cudaPeakMemoryAllocatedBytes", 0.0)
                                / (1024.0 * 1024.0)
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if processed_steps != int(args.steps_per_epoch):
            raise RuntimeError(
                f"processed {processed_steps} non-empty training steps, expected {args.steps_per_epoch}"
            )
        if sampler.epoch_instance_coverage < 1.0:
            raise RuntimeError("stratified observation sampler did not cover every observed instance")
        scheduler.step()
        train_summary = {
            key: {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "count": len(values),
            }
            for key, values in epoch_metrics.items()
            if values
        }
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "globalStep": global_step,
            "lr": float(scheduler.get_last_lr()[0]),
            "trainMetricSummary": train_summary,
            "samplerCoverage": {
                "instances": sampler.epoch_instance_coverage,
                "observations": sampler.epoch_observation_coverage,
                "uniqueObservations": sampler.epoch_unique_observation_count,
                "directions": sampler.epoch_direction_coverage,
                "eventCensor": sampler.epoch_event_censor_coverage,
            },
            "elapsedSeconds": float(time.time() - started),
            "testRead": False,
        }

        calibration_payload: dict[str, Any] | None = None
        validation_row: dict[str, Any] | None = None
        model.eval()
        with torch.no_grad():
            runtime_snapshot, coefficient_snapshot, coefficient_snapshot_diagnostics = _runtime_features(
                model, geometry, relation_tensors, local_ids, structural_ids
            )
        if (epoch + 1) % int(args.eval_every) == 0 or epoch + 1 == int(args.epochs):
            calibration_rows = _evaluate(
                model,
                calibration_split,
                runtime_snapshot,
                world_aabbs,
                device,
                thresholds=threshold_grid(),
                seed=args.seed + epoch,
                poses_per_batch=args.poses_per_batch,
                max_poses=args.max_eval_poses,
                bootstrap_replicates=args.calibration_bootstrap_replicates,
            )
            selected, diagnostic, frozen = _calibration_workpoints(calibration_rows)
            if frozen is not None:
                validation_rows = _evaluate(
                    model,
                    validation_split,
                    runtime_snapshot,
                    world_aabbs,
                    device,
                    thresholds=np.asarray([float(frozen["threshold"])], dtype=np.float32),
                    seed=args.seed + epoch + 10_000,
                    poses_per_batch=args.poses_per_batch,
                    max_poses=args.max_eval_poses,
                    bootstrap_replicates=args.calibration_bootstrap_replicates,
                    instance_to_glb=instance_to_glb_np,
                    glb_bytes=glb_bytes_np,
                )
                validation_row = validation_rows[0] if validation_rows else None
            calibration_payload = {
                "schema": "pvs-bounded-relation-prior-instance-calibrated-calibration-v4",
                "thresholdSource": "this-checkpoint-calibration-only",
                "selectedSafe": selected,
                "diagnostic": diagnostic,
                "thresholdRows": calibration_rows,
                "weightedRecallFloor": CALIBRATION_FLOOR,
                "weightedRecallLowerConfidenceBoundFloor": CALIBRATION_FLOOR,
                "selectionRule": aggregate_weighted_cull_selection_rule(
                    CALIBRATION_FLOOR, CALIBRATION_FLOOR
                ),
                "bootstrapReplicates": int(args.calibration_bootstrap_replicates),
                "validationBootstrapReplicates": int(args.calibration_bootstrap_replicates),
                "testRead": False,
            }
            row["calibration"] = calibration_payload
            row["validationAtFrozenCalibrationThreshold"] = validation_row

            if diagnostic is not None:
                diagnostic_key = (
                    float(diagnostic.get("aggregateWeightedRecallLowerConfidenceBound") or -1.0),
                    float(diagnostic.get("aggregateWeightedRecall") or -1.0),
                    float(diagnostic.get("agg_balanced_accuracy") or 0.0),
                    float(diagnostic.get("agg_useful_cull") or 0.0),
                    float(diagnostic.get("agg_precision") or 0.0),
                    -float(diagnostic.get("avg_pred_count") or 0.0),
                )
                if best_diagnostic_key is None or diagnostic_key > best_diagnostic_key:
                    best_diagnostic_key = diagnostic_key
                    best_diagnostic = {
                        "epoch": epoch + 1,
                        "threshold": float(diagnostic["threshold"]),
                        "safe": selected is not None,
                        "selection": diagnostic,
                    }
                    best_diagnostic_calibration = calibration_payload
                    best_diagnostic_validation = validation_row
                    torch.save(
                        _checkpoint(
                            model,
                            args,
                            epoch + 1,
                            global_step,
                            protocol,
                            relation_provenance,
                            geometry_meta,
                            coefficient_snapshot,
                            coefficient_snapshot_diagnostics,
                            calibration_reliability_meta,
                            calibration_payload,
                            validation_row,
                            best_diagnostic,
                        ),
                        args.output_dir / "best_diagnostic.pt",
                    )
                    _save_fp16(
                        args.output_dir / "best_diagnostic_instance_survival_coefficients_fp16.bin",
                        coefficient_snapshot,
                    )

            if selected is not None:
                safe_key = (
                    float(selected.get("agg_useful_cull") or 0.0),
                    float(selected.get("agg_balanced_accuracy") or 0.0),
                    float(selected.get("agg_precision") or 0.0),
                    -float(selected.get("avg_pred_count") or 0.0),
                )
                if best_safe_key is None or safe_key > best_safe_key:
                    best_safe_key = safe_key
                    best_safe = {
                        "epoch": epoch + 1,
                        "threshold": float(selected["threshold"]),
                        "safe": True,
                        "selection": selected,
                    }
                    best_safe_calibration = calibration_payload
                    best_safe_validation = validation_row
                    payload = _checkpoint(
                        model,
                        args,
                        epoch + 1,
                        global_step,
                        protocol,
                        relation_provenance,
                        geometry_meta,
                        coefficient_snapshot,
                        coefficient_snapshot_diagnostics,
                        calibration_reliability_meta,
                        calibration_payload,
                        validation_row,
                        best_safe,
                    )
                    _save_safe_checkpoint_alias(payload, args.output_dir)
                    _save_fp16(
                        args.output_dir / "best_safe_instance_survival_coefficients_fp16.bin",
                        coefficient_snapshot,
                    )

        if args.snapshot_every > 0 and (epoch + 1) % int(args.snapshot_every) == 0:
            torch.save(
                _checkpoint(
                    model,
                    args,
                    epoch + 1,
                    global_step,
                    protocol,
                    relation_provenance,
                    geometry_meta,
                    coefficient_snapshot,
                    coefficient_snapshot_diagnostics,
                    calibration_reliability_meta,
                    calibration_payload,
                    validation_row,
                    best_safe or best_diagnostic,
                ),
                args.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
            )
        torch.save(
            _checkpoint(
                model,
                args,
                epoch + 1,
                global_step,
                protocol,
                relation_provenance,
                geometry_meta,
                coefficient_snapshot,
                coefficient_snapshot_diagnostics,
                calibration_reliability_meta,
                calibration_payload,
                validation_row,
            ),
            args.output_dir / "last.pt",
        )
        _save_fp16(args.output_dir / "instance_survival_coefficients_fp16.bin", coefficient_snapshot)
        history.append(row)
        _append_jsonl(metrics_path, row)
        _write_json(args.output_dir / "train_history.json", history)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "bestSafe": best_safe,
                    "bestDiagnostic": best_diagnostic,
                    "outputDir": str(args.output_dir),
                    "testRead": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    model.eval()
    with torch.no_grad():
        final_runtime, final_coefficients, final_coefficient_diagnostics = _runtime_features(
            model, geometry, relation_tensors, local_ids, structural_ids
        )
    _save_fp16(args.output_dir / "instance_runtime_features_fp16.bin", final_runtime)
    _save_fp16(args.output_dir / "instance_geo_features_fp16.bin", geometry)
    _save_fp16(args.output_dir / "instance_survival_coefficients_fp16.bin", final_coefficients)
    _save_fp16(
        args.output_dir / "instance_survival_prior_coefficients_fp16.bin",
        final_coefficient_diagnostics["survival_prior_coefficients"],
    )
    _save_fp16(
        args.output_dir / "instance_survival_calibration_residual_fp16.bin",
        final_coefficient_diagnostics["instance_calibration_applied_residual"],
    )
    calibration_summary = {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4",
        "status": "safe" if best_safe is not None else "no_qualified_safety_workpoint",
        "primarySafetyMetric": "aggregateWeightedRecall",
        "weightedRecallFloor": CALIBRATION_FLOOR,
        "weightedRecallLowerConfidenceBoundFloor": CALIBRATION_FLOOR,
        "bestSafe": best_safe,
        "bestDiagnostic": best_diagnostic,
        "calibration": best_safe_calibration or best_diagnostic_calibration,
        "validationAtFrozenThreshold": best_safe_validation or best_diagnostic_validation,
        "safeCheckpoint": str(args.output_dir / "best_safe.pt") if best_safe else None,
        "diagnosticCheckpoint": str(args.output_dir / "best_diagnostic.pt") if best_diagnostic else None,
        "testRead": False,
    }
    _write_json(args.output_dir / "calibration_ready_summary.json", calibration_summary)
    _write_json(
        args.output_dir / "model_meta.json",
        {
            "schema": MODEL_SCHEMA,
            "status": calibration_summary["status"],
            "modelConfig": model.config,
            "protocol": protocol,
            "runtimeFeature": {
                "shape": [num_instances, RUNTIME_FEATURE_DIM],
                "dtype": "float16",
                "file": "instance_runtime_features_fp16.bin",
                "bytes": int(num_instances * RUNTIME_FEATURE_DIM * 2),
            },
            "relationRuntimeExported": False,
            "instanceCalibrationRuntimeExportedSeparately": False,
            "survivalCoefficientFusion": "shared_relation_prior_plus_applied_instance_residual",
            "instanceCalibrationReliability": calibration_reliability_meta,
            "defaultFrontendModified": False,
            "testRead": False,
        },
    )
    print(
        json.dumps(
            {
                "status": calibration_summary["status"],
                "bestSafe": str(args.output_dir / "best_safe.pt") if best_safe else None,
                "bestDiagnostic": str(args.output_dir / "best_diagnostic.pt") if best_diagnostic else None,
                "last": str(args.output_dir / "last.pt"),
                "testRead": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
