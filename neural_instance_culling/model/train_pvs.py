#!/usr/bin/env python3
"""Train the current V4 PVS model without reading test.

The train split optimizes relation/survival and visibility parameters.
Calibration freezes a threshold for each checkpoint; validation ranks epochs
only at that frozen threshold.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.runtime_meta import load_runtime_meta  # noqa: E402
from common.stratified_survival_sampler import StratifiedSurvivalObservationSampler  # noqa: E402
from common.survival_loss import stratified_survival_censoring_loss  # noqa: E402
from common.threshold_selection import (  # noqa: E402
    aggregate_weighted_cull_selection_rule,
    select_aggregate_weighted_cull_workpoint,
)
from common.train_observed_relation_csr import (  # noqa: E402
    ObservedRelationCSR,
    SCHEMA_V3 as RELATION_SCHEMA,
    load_survival_observations_v3,
    validate_survival_observations_v3,
)
from common.visibility_loss import pose_balanced_rvl_contrastive_visibility_loss  # noqa: E402
from pvs_threshold_metrics import evaluate_thresholds, threshold_grid  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from pvs_model import (  # noqa: E402
    BoundedRelationSurvivalMomentModel,
    GEO_DIM,
    MODEL_SCHEMA,
    OCCLUSION_REPRESENTATION_MODES,
    SUPPORTED_SURVIVAL_RANKS,
)


CHECKPOINT_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4"
TRAINING_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4"
CALIBRATION_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-calibration-v4"
CALIBRATION_SUMMARY_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4"
WEIGHTED_RECALL_FLOOR = 0.99


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_value(value), ensure_ascii=False) + "\n")


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def _load_geometry(path: Path, num_instances: int) -> tuple[torch.Tensor, dict[str, Any]]:
    expected = int(num_instances) * GEO_DIM
    if not path.is_file() or path.stat().st_size != expected * 2:
        raise ValueError(f"fixed geometry must be [{num_instances}, {GEO_DIM}] FP16: {path}")
    values = np.fromfile(path, dtype="<f2").reshape(num_instances, GEO_DIM)
    if not bool(np.isfinite(values).all()):
        raise ValueError("fixed geometry contains non-finite values")
    return torch.from_numpy(values.astype(np.float32)), {
        "path": str(path.resolve()),
        "shape": [num_instances, GEO_DIM],
        "dtype": "float16",
    }


def _read_group_ids(path: Path, expected: int, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    values = np.fromfile(path, dtype="<u4").astype(np.int64)
    if values.size != expected or (values.size and int(values.min()) < 0):
        raise ValueError(f"{name} has an invalid shape or value")
    return values


def _load_relation_bundle(
    relation_dir: Path, num_instances: int
) -> tuple[
    ObservedRelationCSR,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    dict[str, np.ndarray],
    dict[str, Any],
]:
    relation = ObservedRelationCSR.load(relation_dir)
    metadata = relation.metadata
    if metadata.get("schema") != RELATION_SCHEMA:
        raise ValueError("training requires the current relation CSR schema")
    if relation.direction_bins != 12 or relation.depth_shells != 3:
        raise ValueError("training requires 12 directions and three depth shells")
    hierarchy = metadata.get("hierarchy")
    files = hierarchy.get("files") if isinstance(hierarchy, Mapping) else None
    if not isinstance(files, Mapping):
        raise ValueError("relation hierarchy files are missing")
    local = _read_group_ids(
        relation_dir / str(files["localGroupIds"]), num_instances, "local group IDs"
    )
    local_count = int(local.max()) + 1
    structural = _read_group_ids(
        relation_dir / str(files["structuralGroupIds"]), local_count, "structural group IDs"
    )
    envelope = metadata.get("survivalObservations")
    if not isinstance(envelope, Mapping):
        raise ValueError("relation artifact has no survival observations")
    loaded = load_survival_observations_v3(relation_dir, metadata)
    observations = {
        "instance": loaded["instance"],
        "direction": loaded["direction"],
        "normalized_depth": loaded["normalizedDepth"],
        "depth": loaded["normalizedDepth"],
        "event": loaded["event"],
        "weight": loaded["weight"],
        "subpose": loaded["subpose"],
        "rawPixelCount": loaded["rawPixelCount"],
    }
    summary = validate_survival_observations_v3(
        {
            "instance": observations["instance"],
            "direction": observations["direction"],
            "normalizedDepth": observations["normalized_depth"],
            "event": observations["event"],
            "weight": observations["weight"],
            "subpose": observations["subpose"],
            "rawPixelCount": observations["rawPixelCount"],
        },
        num_instances=num_instances,
        direction_bins=12,
    )
    if summary["eventCount"] <= 0 or summary["rightCensoredCount"] <= 0:
        raise ValueError("survival supervision needs events and right-censored rows")
    provenance = {
        "schema": metadata["schema"],
        "path": str(relation_dir.resolve()),
        "edgeCount": relation.edge_count,
        "rowCount": relation.row_count,
        "observations": summary,
        "testRead": False,
    }
    relation_tensors = relation.to_torch()
    relation_tensors["metadata"] = metadata
    return (
        relation,
        relation_tensors,
        torch.from_numpy(local),
        torch.from_numpy(structural),
        observations,
        provenance,
    )


def _depth_normalization(relation_dir: Path) -> dict[str, Any]:
    metadata = json.loads((relation_dir / "relation_csr_meta.json").read_text(encoding="utf-8"))
    depth = metadata.get("depthNormalization")
    if not isinstance(depth, Mapping):
        raise ValueError("relation metadata has no depth normalization")
    result = {
        "q01": float(depth["q01"]),
        "q99": float(depth["q99"]),
        "epsilon": float(depth["epsilon"]),
        "sourceSplit": depth.get("sourceSplit"),
    }
    if result["sourceSplit"] != "train" or result["q99"] <= result["q01"] or result["epsilon"] <= 0:
        raise ValueError("depth normalization is not train-owned and finite")
    return result


def _resolve_split(dataset: PoseCSRDataset, requested: str, role: str):
    if requested.lower() == "test":
        raise ValueError("test is forbidden during training and threshold selection")
    if requested == "auto":
        choices = {"train": ("train",), "calibration": ("calibration",), "validation": ("validation", "val")}[role]
        requested = next((name for name in choices if name in dataset.split_ids), "")
    if requested not in dataset.split_ids:
        raise ValueError(f"dataset has no explicit {role} split")
    split = dataset.split(requested)
    if split.pose_indices.size == 0:
        raise ValueError(f"{role} split is empty")
    return requested, split


def _instance_reliability(
    dataset: PoseCSRDataset,
    train_split: Any,
    observations: Mapping[str, np.ndarray],
    num_instances: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_counts = np.zeros(num_instances, dtype=np.int64)
    for pose in np.asarray(train_split.pose_indices, dtype=np.int64):
        ids = np.asarray(dataset.candidate_slice(int(pose)), dtype=np.int64)
        if ids.size:
            candidate_counts += np.bincount(ids, minlength=num_instances)
    observation_counts = np.bincount(
        np.asarray(observations.get("instance", []), dtype=np.int64), minlength=num_instances
    )
    def normalize(values: np.ndarray) -> np.ndarray:
        positive = values[values > 0]
        reference = max(float(np.percentile(positive, 95)), 1.0) if positive.size else 1.0
        return np.clip(np.log1p(values) / math.log1p(reference), 0, 1).astype(np.float32)
    reliability = np.maximum(normalize(candidate_counts), normalize(observation_counts))
    return reliability, {
        "sourceSplit": "train",
        "candidateReferenceCount": int(candidate_counts.sum()),
        "observationReferenceCount": int(observation_counts.sum()),
        "zeroReliabilityCount": int(np.count_nonzero(reliability <= 0)),
        "meanReliability": float(reliability.mean()),
        "testRead": False,
    }


def _load_glb_bytes(index_path: Path, root: Path, num_glbs: int) -> np.ndarray:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    result = np.zeros(num_glbs, dtype=np.float32)
    for entry in payload.get("entries", []):
        glb_id = int(entry.get("globalId", -1))
        path = root / str(entry.get("path", ""))
        if 0 <= glb_id < num_glbs and path.is_file():
            result[glb_id] = float(path.stat().st_size)
    positive = result[result > 0]
    result[result <= 0] = float(np.median(positive)) if positive.size else 1.0
    return result


def _runtime_features(
    model: BoundedRelationSurvivalMomentModel,
    geometry: torch.Tensor,
    relation: Mapping[str, torch.Tensor],
    local_ids: torch.Tensor,
    structural_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    diagnostics = model.offline_encode_survival(
        geometry, relation, local_ids, structural_ids, return_diagnostics=True
    )
    if not isinstance(diagnostics, Mapping):
        raise TypeError("offline encoder did not return diagnostics")
    features = diagnostics.get("occlusion_features")
    if not isinstance(features, torch.Tensor):
        raise ValueError("offline encoder did not produce occlusion features")
    runtime = torch.cat([geometry, features.flatten(1)], dim=-1)
    if runtime.shape != (geometry.shape[0], model.runtime_feature_dim) or not bool(torch.isfinite(runtime).all()):
        raise FloatingPointError("fixed runtime table is invalid")
    return runtime, features, dict(diagnostics)


def _evaluation_batch_limits(max_poses: int, poses_per_batch: int) -> tuple[int, int | None]:
    if max_poses < 0 or poses_per_batch <= 0:
        raise ValueError("evaluation limits are invalid")
    if max_poses == 0:
        return poses_per_batch, None
    batch = math.gcd(max_poses, poses_per_batch)
    return batch, max_poses // batch


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
    batch, steps = _evaluation_batch_limits(max_poses, poses_per_batch)
    return evaluate_thresholds(
        model, split, runtime, world_aabbs, device,
        poses_per_batch=batch, max_steps=steps, max_candidates_per_pose=0,
        seed=seed, thresholds=thresholds, collect_pose_stats=True,
        allow_candidate_visible_union=False,
        bootstrap_replicates=bootstrap_replicates,
        instance_to_glb=instance_to_glb, glb_bytes=glb_bytes,
    )


def _diagnostic_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (
        float(row.get("aggregateWeightedRecallLowerConfidenceBound") or -1),
        float(row.get("aggregateWeightedRecall") or -1),
        float(row.get("agg_balanced_accuracy") or 0),
        float(row.get("agg_precision") or 0),
        -float(row.get("avg_pred_count") or 0),
    ))


def _calibration_workpoints(rows: list[dict[str, Any]]):
    safe = select_aggregate_weighted_cull_workpoint(
        rows,
        target_weighted_recall=WEIGHTED_RECALL_FLOOR,
        minimum_lower_confidence_bound=WEIGHTED_RECALL_FLOOR,
    )
    diagnostic = _diagnostic_row(rows)
    return safe, diagnostic, safe if safe is not None else diagnostic


def _weighted_recall_safety_gate(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    recall = row.get("aggregateWeightedRecall", row.get("agg_weighted_recall"))
    lower = row.get("aggregateWeightedRecallLowerConfidenceBound")
    return recall is not None and lower is not None and float(recall) > 0.99 and float(lower) > 0.99


def _validation_key(validation: Mapping[str, Any], calibration: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(validation.get("agg_balanced_accuracy") or 0),
        float(validation.get("agg_precision") or 0),
        float(validation.get("agg_accuracy") or 0),
        float(validation.get("agg_useful_cull") or 0),
        -float(validation.get("avg_pred_count") or 0),
        float(calibration.get("agg_useful_cull") or 0),
    )


def _calibration_blend(step: int, total: int, warmup: float, ramp: float) -> float:
    progress = step / max(1, total - 1)
    if progress < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return float(np.clip((progress - warmup) / ramp, 0, 1))


def _checkpoint(
    model: BoundedRelationSurvivalMomentModel,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    protocol: Mapping[str, Any],
    relation_meta: Mapping[str, Any],
    geometry_meta: Mapping[str, Any],
    coefficients: torch.Tensor,
    diagnostics: Mapping[str, Any],
    reliability_meta: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    best: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "runtimeSchema": MODEL_SCHEMA,
        "experimentName": args.experiment_name,
        "epoch": epoch,
        "globalStep": global_step,
        "modelConfig": model.config,
        "modelState": model.state_dict(),
        "occlusionRepresentation": model.occlusion_representation,
        "instanceOcclusionFeatures": coefficients.detach().cpu().half(),
        "instanceCalibration": {
            "mode": model.instance_calibration_mode,
            "blend": float(model.instance_calibration_blend.detach().cpu()),
            "fusion": "prior_plus_applied_residual" if model.occlusion_representation == "survival" else "not_applicable",
            "reliability": dict(reliability_meta),
            "runtimeExport": "fused_coefficients_only" if model.occlusion_representation == "survival" else "not_applicable",
        },
        "protocol": dict(protocol),
        "relation": dict(relation_meta),
        "geometry": dict(geometry_meta),
        "viewcell": {"shape": "horizontal_disk", "radiusM": 2.0},
        "calibration": calibration,
        "validationAtCalibration": validation,
        "best": best,
        "testRead": False,
    }
    if model.occlusion_representation == "survival":
        prior = diagnostics["survival_prior_coefficients"]
        residual = diagnostics["instance_calibration_applied_residual"]
        payload.update({
            "instanceSurvivalCoefficients": coefficients.detach().cpu().half(),
            "instanceSurvivalPriorCoefficients": prior.detach().cpu().half(),
            "instanceSurvivalCalibrationResidual": residual.detach().cpu().half(),
        })
    return payload


def _save_fp16(path: Path, values: torch.Tensor) -> None:
    values.detach().cpu().numpy().astype("<f2").tofile(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--initial-geo-features", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--occlusion-representation", choices=OCCLUSION_REPRESENTATION_MODES, default="survival")
    parser.add_argument("--survival-rank", type=int, choices=SUPPORTED_SURVIVAL_RANKS, default=4)
    parser.add_argument("--relation-source", choices=("bounded_hierarchical", "geometry_only", "none"), default="bounded_hierarchical")
    parser.add_argument("--spectral-mode", choices=("moment_envelope", "point"), default="moment_envelope")
    parser.add_argument("--instance-calibration-mode", choices=("residual", "disabled"), default="residual")
    parser.add_argument(
        "--loss-variant",
        choices=("pose_balanced_rvl_contrastive",),
        default="pose_balanced_rvl_contrastive",
    )
    parser.add_argument("--train-split", default="auto")
    parser.add_argument("--calibration-split", default="auto")
    parser.add_argument("--validation-split", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--observation-batch-size", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--snapshot-every", type=int, default=4)
    parser.add_argument("--max-eval-poses", type=int, default=0)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--survival-loss-weight", type=float, default=0.25)
    parser.add_argument("--relation-consistency-weight", type=float, default=0.10)
    parser.add_argument("--regularization-weight", type=float, default=1e-5)
    parser.add_argument("--instance-calibration-regularization-weight", type=float, default=0.02)
    parser.add_argument("--instance-calibration-max-abs", type=float, default=4.0)
    parser.add_argument("--sparse-instance-penalty", type=float, default=3.0)
    parser.add_argument("--instance-calibration-warmup-fraction", type=float, default=0.10)
    parser.add_argument("--instance-calibration-ramp-fraction", type=float, default=0.20)
    parser.add_argument("--relation-gradient-cap", type=float, default=0.25)
    parser.add_argument("--integrated-rvl-recall-guard-weight", type=float, default=0.30)
    parser.add_argument("--integrated-rvl-recall-target", type=float, default=0.99)
    parser.add_argument("--integrated-rvl-recall-temperature", type=float, default=0.05)
    parser.add_argument("--integrated-rvl-pose-cvar-fraction", type=float, default=0.25)
    parser.add_argument("--integrated-rvl-pose-cvar-weight", type=float, default=0.25)
    parser.add_argument("--integrated-separation-weight", type=float, default=0.20)
    parser.add_argument("--integrated-tail-ramp-fraction", type=float, default=0.15)
    parser.add_argument("--frontier-positive-mass-fraction", type=float, default=0.005)
    parser.add_argument("--frontier-positive-count-cap", type=int, default=64)
    parser.add_argument("--frontier-negative-fraction", type=float, default=0.01)
    parser.add_argument("--frontier-negative-count-cap", type=int, default=256)
    parser.add_argument("--frontier-margin", type=float, default=0.50)
    parser.add_argument("--frontier-temperature", type=float, default=0.25)
    parser.add_argument("--frontier-positive-importance-floor", type=float, default=0.5)
    parser.add_argument("--frontier-positive-importance-power", type=float, default=0.5)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.epochs, args.steps_per_epoch, args.poses_per_batch, args.observation_batch_size) <= 0:
        raise ValueError("training counts must be positive")
    if args.eval_every <= 0 or args.calibration_bootstrap_replicates <= 0:
        raise ValueError("evaluation cadence and bootstrap count must be positive")
    if args.occlusion_representation == "survival":
        if args.relation_source not in {"bounded_hierarchical", "geometry_only"}:
            raise ValueError("survival requires a relation source")
    elif args.relation_source != "none" or args.instance_calibration_mode != "disabled":
        raise ValueError("generic28 and none require no relation and disabled calibration")
    values = (
        args.learning_rate, args.weight_decay, args.survival_loss_weight,
        args.relation_consistency_weight, args.regularization_weight,
        args.instance_calibration_regularization_weight,
        args.instance_calibration_max_abs, args.sparse_instance_penalty,
        args.relation_gradient_cap, args.integrated_rvl_recall_guard_weight,
        args.integrated_rvl_recall_temperature, args.integrated_separation_weight,
        args.frontier_margin, args.frontier_temperature,
    )
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("training weights must be finite and non-negative")
    for value in (
        args.instance_calibration_warmup_fraction,
        args.instance_calibration_ramp_fraction,
        args.integrated_rvl_pose_cvar_fraction,
        args.integrated_rvl_pose_cvar_weight,
        args.integrated_tail_ramp_fraction,
        args.frontier_positive_mass_fraction,
        args.frontier_negative_fraction,
        args.frontier_positive_importance_floor,
        args.frontier_positive_importance_power,
    ):
        if not 0 <= value <= 1:
            raise ValueError("fraction arguments must lie in [0, 1]")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    _prepare_output(args.output_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(args.runtime_meta)
    num_instances = int(world_aabbs.shape[0])
    num_glbs = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
    dataset = PoseCSRDataset(args.dataset_dir, num_instances)
    train_name, train_split = _resolve_split(dataset, args.train_split, "train")
    calibration_name, calibration_split = _resolve_split(dataset, args.calibration_split, "calibration")
    validation_name, validation_split = _resolve_split(dataset, args.validation_split, "validation")
    geometry, geometry_meta = _load_geometry(args.initial_geo_features, num_instances)
    depth = _depth_normalization(args.relation_dir)

    if args.occlusion_representation == "survival":
        relation, relation_tensors, local_ids, structural_ids, observations, relation_meta = _load_relation_bundle(
            args.relation_dir, num_instances
        )
        del relation
        sampler: StratifiedSurvivalObservationSampler | None = StratifiedSurvivalObservationSampler(
            observations,
            seed=args.seed,
            metadata={"trainOnly": True, "splitNames": ["train"]},
        )
    else:
        relation_tensors = {}
        local_ids = torch.zeros(num_instances, dtype=torch.long)
        structural_ids = torch.zeros(1, dtype=torch.long)
        observations = {"instance": np.zeros(0, dtype=np.int64)}
        relation_meta = {"enabled": False, "source": "depth_normalization_only", "testRead": False}
        sampler = None

    reliability, reliability_meta = _instance_reliability(
        dataset, train_split, observations, num_instances
    )
    model = BoundedRelationSurvivalMomentModel(
        num_instances, num_glbs,
        survival_rank=args.survival_rank,
        relation_source=args.relation_source,
        occlusion_representation=args.occlusion_representation,
        spectral_mode=args.spectral_mode,
        depth_q01=depth["q01"], depth_q99=depth["q99"], depth_epsilon=depth["epsilon"],
        instance_calibration_mode=args.instance_calibration_mode,
        instance_calibration_max_abs=args.instance_calibration_max_abs,
        sparse_instance_penalty=args.sparse_instance_penalty,
    ).to(device)
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    model.set_instance_calibration_reliability(torch.from_numpy(reliability).to(device))
    geometry = geometry.to(device)
    relation_tensors = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in relation_tensors.items()
    }
    local_ids = local_ids.to(device)
    structural_ids = structural_ids.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    glb_bytes = _load_glb_bytes(args.glb_index, args.glb_root, num_glbs)

    protocol = {
        "schema": TRAINING_SCHEMA,
        "experiment": args.experiment_name,
        "variant": args.variant,
        "seed": int(args.seed),
        "lossVariant": args.loss_variant,
        "initialization": {"mode": "from-scratch"},
        "instanceCalibration": {"mode": args.instance_calibration_mode},
        "splitNames": {"train": train_name, "calibration": calibration_name, "validation": validation_name},
        "splitPoseCounts": {
            "train": int(train_split.pose_indices.size),
            "calibration": int(calibration_split.pose_indices.size),
            "validation": int(validation_split.pose_indices.size),
        },
        "candidateUnion": False,
        "testRead": False,
    }
    _write_json(args.output_dir / "run_manifest.json", {
        "schema": "pvs-v4-training-run-v1",
        "arguments": vars(args),
        "modelConfig": model.config,
        "protocol": protocol,
        "testRead": False,
    })

    history: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "train_metrics.jsonl"
    global_step = 0
    total_steps = args.epochs * args.steps_per_epoch
    best_safe: dict[str, Any] | None = None
    best_diagnostic: dict[str, Any] | None = None
    best_safe_key: tuple[float, ...] | None = None
    best_diagnostic_key: tuple[float, ...] | None = None
    best_safe_calibration = best_diagnostic_calibration = None
    best_safe_validation = best_diagnostic_validation = None
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        if sampler is not None:
            sampler.start_epoch(epoch)
        epoch_values: dict[str, list[float]] = {}
        rng = np.random.default_rng(args.seed + epoch * 1009)
        batches = train_split.pose_set_batches(
            args.poses_per_batch, rng, args.steps_per_epoch
        )
        for step, poses in enumerate(batches):
            batch = train_split.build_pose_set_batch(
                poses, world_aabbs, rng,
                max_candidates_per_pose=0,
                allow_candidate_visible_union=False,
            )
            if batch["instance"].size == 0:
                continue
            global_step += 1
            blend = _calibration_blend(
                global_step - 1, total_steps,
                args.instance_calibration_warmup_fraction,
                args.instance_calibration_ramp_fraction,
            )
            model.set_instance_calibration_blend(blend)
            runtime, coefficients, coefficient_diagnostics = _runtime_features(
                model, geometry, relation_tensors, local_ids, structural_ids
            )
            tensors = {
                key: torch.from_numpy(np.asarray(value)).to(device)
                for key, value in batch.items()
                if isinstance(value, np.ndarray)
            }
            logits, _aux = model.compute_logits_with_aux(
                tensors["camera"].float(),
                tensors["camera_view"].float(),
                tensors["candidate_camera_world"].float(),
                tensors["instance"].long(),
                runtime,
                query_center_world=tensors["query_center_world"].float(),
                viewcell_radius_m=tensors["viewcell_radius_m"].float(),
                pose_offsets=tensors["pose_offsets"].long(),
            )
            separation_scale = min(
                1.0,
                (global_step / max(1, total_steps)) / max(args.integrated_tail_ramp_fraction, 1e-8),
            ) if args.integrated_tail_ramp_fraction > 0 else 1.0
            visibility_loss, visibility_parts = pose_balanced_rvl_contrastive_visibility_loss(
                logits,
                tensors["target"].float(),
                tensors["pose_offsets"].long(),
                tensors["visible_weights"].float(),
                recall_guard_weight=args.integrated_rvl_recall_guard_weight,
                recall_target=args.integrated_rvl_recall_target,
                recall_temperature=args.integrated_rvl_recall_temperature,
                recall_pose_cvar_fraction=args.integrated_rvl_pose_cvar_fraction,
                recall_pose_cvar_weight=args.integrated_rvl_pose_cvar_weight,
                separation_weight=args.integrated_separation_weight,
                separation_scale=separation_scale,
                positive_mass_fraction=args.frontier_positive_mass_fraction,
                positive_count_cap=args.frontier_positive_count_cap,
                negative_top_fraction=args.frontier_negative_fraction,
                negative_count_cap=args.frontier_negative_count_cap,
                margin=args.frontier_margin,
                logit_temperature=args.frontier_temperature,
                positive_importance_floor=args.frontier_positive_importance_floor,
                positive_importance_power=args.frontier_positive_importance_power,
            )
            if sampler is not None:
                sampled = sampler.gather(
                    sampler.sample_batch(args.observation_batch_size, step=step),
                    device=device,
                )
                sampled["normalized_depth"] = sampled["depth"]
                survival_loss, survival_parts = stratified_survival_censoring_loss(
                    model, coefficients, sampled
                )
                relation_loss, relation_parts = model.relation_consistency_loss(
                    geometry, relation_tensors,
                    seed=args.seed + global_step,
                    max_edges=args.observation_batch_size,
                )
            else:
                survival_loss = relation_loss = logits.sum() * 0.0
                survival_parts = {"lossSurvivalCensor": survival_loss}
                relation_parts = {"lossRelationConsistency": relation_loss}
            regularization = model.regularization()
            calibration_regularization = model.instance_calibration_regularization()
            loss = (
                visibility_loss
                + args.survival_loss_weight * survival_loss
                + args.relation_consistency_weight * relation_loss
                + args.regularization_weight * regularization
                + args.instance_calibration_regularization_weight * calibration_regularization
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if model.offline_survival_encoder is not None and args.relation_gradient_cap > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.offline_survival_encoder.parameters(), args.relation_gradient_cap
                )
            optimizer.step()
            values = {
                "loss": loss,
                "lossVisibility": visibility_loss,
                "lossSurvivalWeighted": args.survival_loss_weight * survival_loss,
                "lossRelationWeighted": args.relation_consistency_weight * relation_loss,
                "lossRegularizationWeighted": args.regularization_weight * regularization,
                "lossInstanceCalibrationWeighted": args.instance_calibration_regularization_weight * calibration_regularization,
                "instanceCalibrationBlend": blend,
                **visibility_parts,
                **survival_parts,
                **relation_parts,
            }
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    epoch_values.setdefault(key, []).append(float(value.detach().cpu()))
                elif isinstance(value, (int, float)):
                    epoch_values.setdefault(key, []).append(float(value))
            if step == 0 or (step + 1) % max(1, args.steps_per_epoch // 5) == 0:
                print(json.dumps({
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "globalStep": global_step,
                    "loss": float(loss.detach().cpu()),
                    "elapsedSeconds": time.time() - started,
                }), flush=True)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            runtime, coefficients, coefficient_diagnostics = _runtime_features(
                model, geometry, relation_tensors, local_ids, structural_ids
            )
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "globalStep": global_step,
            "lr": scheduler.get_last_lr()[0],
            "trainMetricSummary": {
                key: {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "count": len(values)}
                for key, values in epoch_values.items() if values
            },
            "elapsedSeconds": time.time() - started,
            "testRead": False,
        }
        calibration_payload = None
        validation_row = None
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            calibration_rows = _evaluate(
                model, calibration_split, runtime, world_aabbs, device,
                thresholds=threshold_grid(), seed=args.seed + 50_000,
                poses_per_batch=args.poses_per_batch, max_poses=args.max_eval_poses,
                bootstrap_replicates=args.calibration_bootstrap_replicates,
            )
            selected, diagnostic, frozen = _calibration_workpoints(calibration_rows)
            if frozen is not None:
                validation_rows = _evaluate(
                    model, validation_split, runtime, world_aabbs, device,
                    thresholds=np.asarray([float(frozen["threshold"])], dtype=np.float32),
                    seed=args.seed + 60_000,
                    poses_per_batch=args.poses_per_batch, max_poses=args.max_eval_poses,
                    bootstrap_replicates=args.calibration_bootstrap_replicates,
                    instance_to_glb=instance_to_glb, glb_bytes=glb_bytes,
                )
                validation_row = validation_rows[0] if validation_rows else None
            calibration_payload = {
                "schema": CALIBRATION_SCHEMA,
                "selected": selected,
                "selectedSafe": selected,
                "diagnostic": diagnostic,
                "thresholdRows": calibration_rows,
                "bootstrapReplicates": args.calibration_bootstrap_replicates,
                "testRead": False,
            }
            row["calibration"] = calibration_payload
            row["validationAtFrozenCalibrationThreshold"] = validation_row
            validation_safe = _weighted_recall_safety_gate(validation_row)
            if diagnostic is not None:
                key = (
                    float(validation_safe),
                    float((validation_row or {}).get("aggregateWeightedRecallLowerConfidenceBound") or -1),
                    float((validation_row or {}).get("agg_balanced_accuracy") or 0),
                    float((validation_row or {}).get("agg_precision") or 0),
                )
                if best_diagnostic_key is None or key > best_diagnostic_key:
                    best_diagnostic_key = key
                    best_diagnostic = {
                        "epoch": epoch + 1,
                        "threshold": float(diagnostic["threshold"]),
                        "selection": diagnostic,
                        "validationSafetyPassed": validation_safe,
                    }
                    best_diagnostic_calibration = calibration_payload
                    best_diagnostic_validation = validation_row
                    torch.save(_checkpoint(
                        model, args, epoch + 1, global_step, protocol, relation_meta,
                        geometry_meta, coefficients, coefficient_diagnostics,
                        reliability_meta, calibration_payload, validation_row, best_diagnostic,
                    ), args.output_dir / "best_diagnostic.pt")
            if selected is not None and validation_safe:
                assert validation_row is not None
                key = _validation_key(validation_row, selected)
                if best_safe_key is None or key > best_safe_key:
                    best_safe_key = key
                    best_safe = {
                        "epoch": epoch + 1,
                        "threshold": float(selected["threshold"]),
                        "safe": True,
                        "selection": selected,
                        "validationSelection": validation_row,
                        "validationSafetyPassed": True,
                    }
                    best_safe_calibration = calibration_payload
                    best_safe_validation = validation_row
                    payload = _checkpoint(
                        model, args, epoch + 1, global_step, protocol, relation_meta,
                        geometry_meta, coefficients, coefficient_diagnostics,
                        reliability_meta, calibration_payload, validation_row, best_safe,
                    )
                    torch.save(payload, args.output_dir / "best_safe.pt")
                    shutil.copyfile(args.output_dir / "best_safe.pt", args.output_dir / "best.pt")

        if args.snapshot_every > 0 and (epoch + 1) % args.snapshot_every == 0:
            torch.save(_checkpoint(
                model, args, epoch + 1, global_step, protocol, relation_meta,
                geometry_meta, coefficients, coefficient_diagnostics,
                reliability_meta, calibration_payload, validation_row, best_safe or best_diagnostic,
            ), args.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt")
        torch.save(_checkpoint(
            model, args, epoch + 1, global_step, protocol, relation_meta,
            geometry_meta, coefficients, coefficient_diagnostics,
            reliability_meta, calibration_payload, validation_row,
        ), args.output_dir / "last.pt")
        history.append(row)
        _append_jsonl(metrics_path, row)
        _write_json(args.output_dir / "train_history.json", history)

    model.eval()
    with torch.no_grad():
        runtime, coefficients, coefficient_diagnostics = _runtime_features(
            model, geometry, relation_tensors, local_ids, structural_ids
        )
    _save_fp16(args.output_dir / "instance_runtime_features_fp16.bin", runtime)
    _save_fp16(args.output_dir / "instance_geo_features_fp16.bin", geometry)
    if model.occlusion_representation == "survival":
        _save_fp16(args.output_dir / "instance_survival_coefficients_fp16.bin", coefficients)
        _save_fp16(args.output_dir / "instance_survival_prior_coefficients_fp16.bin", coefficient_diagnostics["survival_prior_coefficients"])
        _save_fp16(args.output_dir / "instance_survival_calibration_residual_fp16.bin", coefficient_diagnostics["instance_calibration_applied_residual"])
    elif model.occlusion_representation == "generic28":
        _save_fp16(args.output_dir / "instance_generic_occlusion_features_fp16.bin", coefficients)

    calibration_summary = {
        "schema": CALIBRATION_SUMMARY_SCHEMA,
        "status": "safe" if best_safe is not None else "no_qualified_safety_workpoint",
        "primarySafetyMetric": "aggregateWeightedRecall",
        "weightedRecallFloor": WEIGHTED_RECALL_FLOOR,
        "weightedRecallLowerConfidenceBoundFloor": WEIGHTED_RECALL_FLOOR,
        "bestSafe": best_safe,
        "bestDiagnostic": best_diagnostic,
        "calibration": best_safe_calibration or best_diagnostic_calibration,
        "validationAtFrozenThreshold": best_safe_validation or best_diagnostic_validation,
        "safeCheckpoint": str(args.output_dir / "best_safe.pt") if best_safe else None,
        "diagnosticCheckpoint": str(args.output_dir / "best_diagnostic.pt") if best_diagnostic else None,
        "selectionRule": aggregate_weighted_cull_selection_rule(0.99, 0.99),
        "testRead": False,
    }
    _write_json(args.output_dir / "calibration_ready_summary.json", calibration_summary)
    _write_json(args.output_dir / "model_meta.json", {
        "schema": MODEL_SCHEMA,
        "status": calibration_summary["status"],
        "modelConfig": model.config,
        "protocol": protocol,
        "runtimeFeature": {
            "shape": [num_instances, model.runtime_feature_dim],
            "dtype": "float16",
            "file": "instance_runtime_features_fp16.bin",
            "bytes": num_instances * model.runtime_feature_dim * 2,
        },
        "relationRuntimeExported": False,
        "testRead": False,
    })
    print(json.dumps({
        "status": calibration_summary["status"],
        "bestSafe": calibration_summary["safeCheckpoint"],
        "bestDiagnostic": calibration_summary["diagnosticCheckpoint"],
        "last": str(args.output_dir / "last.pt"),
        "testRead": False,
    }), flush=True)


if __name__ == "__main__":
    main()
