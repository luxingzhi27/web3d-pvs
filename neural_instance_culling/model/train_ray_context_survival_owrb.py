#!/usr/bin/env python3
"""Train the independent ray-context/survival/OWRB experiment.

This entry point writes only to the explicitly supplied experiment directory.
It uses the stored back-camera candidate CSR and never unions visible IDs into
the candidates.  Test poses are loaded only to record the split manifest and
are not read during training or threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.glb_points import load_glb_points  # noqa: E402
from common.candidate_identity import (  # noqa: E402
    audit_native_aabb_candidates,
    candidate_digest_for_pose_sequence,
)
from common.occlusion_edges import glb_priority_loss, load_glb_costs  # noqa: E402
from common.owrb_loss import OWRBConfig, OWRBState, owrb_visibility_loss  # noqa: E402
from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from current_pvs_utils import evaluate_thresholds, threshold_grid, validate_training_resources, visual_utility_loss  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from ray_context_survival_owrb_model import (  # noqa: E402
    RayContextSurvivalOWRBModel,
    load_ray_context_survival_evidence,
)
from common.safety_constraint_utility_loss import (  # noqa: E402
    SafetyDualState,
    neutral_rvl_control_evidence,
    rvl_strong_v2_visibility_loss,
    safety_constraint_utility_loss,
)
from common.survival_loss import survival_censoring_loss  # noqa: E402
from common.threshold_selection import select_weighted_cull_workpoint, weighted_cull_selection_rule  # noqa: E402


DEFAULT_INITIAL_GEO = ROOT / (
    "model/out/"
    "pvs_m4_ablation_geometry_context_ray_rvl_strong_v2_hkust_spatial_fov66_seed20260801_full40/"
    "instance_geo_features_fp16.bin"
)
FIXED_GEO_DIM = 96


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _digest_pose_indices(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()[:16]


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu()) if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _protocol_splits(dataset: PoseCSRDataset, seed: int) -> dict[str, Any]:
    required = {"train", "validation", "calibration", "test"}
    if not required.issubset(dataset.split_ids):
        raise ValueError("the OWRB experiment requires train/validation/calibration/test splits")
    return {
        "schema": "ray-context-survival-owrb-four-way-v1",
        "seed": int(seed),
        "trainCount": int(dataset.split("train").pose_indices.size),
        "validationCount": int(dataset.split("validation").pose_indices.size),
        "calibrationCount": int(dataset.split("calibration").pose_indices.size),
        "testCount": int(dataset.split("test").pose_indices.size),
        "trainDigest": _digest_pose_indices(dataset.split("train").pose_indices),
        "validationDigest": _digest_pose_indices(dataset.split("validation").pose_indices),
        "calibrationDigest": _digest_pose_indices(dataset.split("calibration").pose_indices),
        "testDigest": _digest_pose_indices(dataset.split("test").pose_indices),
        "testPolicy": "test split is not read before architecture, checkpoint, and threshold freeze",
    }


def _candidate_digest_for_split(dataset: PoseCSRDataset, split: Any) -> str:
    return candidate_digest_for_pose_sequence(dataset, split.pose_indices)


def _context_targets(
    evidence: dict[str, Any],
    token_dim: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic train-only token and auxiliary targets.

    The first eight channels are the registered compact context target.  The
    optional 16-channel supplement appends direction-local support, shell
    occupancy, and normalized source-score statistics.  All targets come
    from the train-only relation table; no validation or test pose is used.
    """
    if int(token_dim) not in (8, 16):
        raise ValueError("context token dimension must be 8 or 16")
    stats = np.asarray(evidence["relation_stats"], dtype=np.float32)
    strength = np.asarray(evidence["strength"], dtype=np.float32)
    scores = np.clip(stats[..., 0], 0.0, 1.0)
    denom = np.maximum(scores.sum(axis=(2, 3), keepdims=False)[..., None], 1e-6)
    weighted = (stats * scores[..., None]).sum(axis=(2, 3)) / denom
    active_shells = (strength > 0.0).sum(axis=2, keepdims=False).astype(np.float32) / 3.0
    shell_index = np.arange(3, dtype=np.float32).reshape(1, 1, 3, 1)
    shell_denom = np.maximum(scores.sum(axis=3), 1e-6)
    mean_shell = (scores * shell_index).sum(axis=(2, 3)) / np.maximum(scores.sum(axis=(2, 3)), 1e-6)
    base_token = [
        np.clip(strength.sum(axis=2) / 3.0, 0.0, 1.0),
        np.clip(weighted[..., 1], 0.0, 1.0),
        np.clip(weighted[..., 2], 0.0, 1.0),
        np.clip(weighted[..., 3], 0.0, 1.0),
        np.clip(weighted[..., 4], 0.0, 1.0),
        np.clip(weighted[..., 5], 0.0, 1.0),
        np.clip(scores.sum(axis=(2, 3)) / 8.0, 0.0, 1.0),
        np.clip(mean_shell / 2.0, 0.0, 1.0),
    ]
    if int(token_dim) == 16:
        shell_strength = np.clip(strength / np.maximum(strength.max(axis=2, keepdims=True), 1e-6), 0.0, 1.0)
        max_source_score = np.max(scores, axis=(2, 3))
        source_count = np.clip((scores > 0.0).sum(axis=(2, 3)) / 8.0, 0.0, 1.0)
        base_token.extend([
            np.clip(weighted[..., 0], 0.0, 1.0),
            np.clip(max_source_score, 0.0, 1.0),
            source_count,
            shell_strength[..., 0],
            shell_strength[..., 1],
            shell_strength[..., 2],
            np.clip(np.abs(weighted[..., 4] - weighted[..., 5]), 0.0, 1.0),
            np.clip(1.0 - mean_shell / 2.0, 0.0, 1.0),
        ])
    token = np.stack(base_token, axis=-1)
    # The auxiliary head predicts evidence strength, support and depth gap.
    aux = np.stack([
        np.clip(strength.sum(axis=2) / 3.0, 0.0, 1.0),
        np.clip(weighted[..., 2], 0.0, 1.0),
        np.clip(weighted[..., 3], 0.0, 1.0),
    ], axis=-1)
    del shell_denom
    return torch.from_numpy((token * 2.0 - 1.0).astype(np.float32)), torch.from_numpy(aux.astype(np.float32))


def _load_or_compute_geometry(
    model: RayContextSurvivalOWRBModel,
    path: str,
    glb_points_path: Path,
    device: torch.device,
    num_instances: int,
    num_glbs: int,
    points_per_glb: int,
    batch_size: int,
) -> tuple[torch.Tensor, str]:
    if not path:
        raise ValueError(
            "this formal OWRB entry point requires the audited fixed 96-D "
            "geometry table; it never computes a new PointNet++ table"
        )
    values = np.fromfile(path, dtype=np.float16)
    expected = num_instances * model.geo_dim
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} values; expected {expected} FP16 geometry values")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{path} contains non-finite geometry values")
    return torch.from_numpy(values.reshape(num_instances, model.geo_dim).astype(np.float32)).to(device), str(Path(path).resolve())


def _validate_fixed_geometry(path: str, num_instances: int) -> str:
    geometry_path = Path(path)
    if not geometry_path.is_file():
        raise FileNotFoundError(f"missing audited fixed geometry table: {geometry_path}")
    expected_bytes = int(num_instances) * FIXED_GEO_DIM * 2
    actual_bytes = int(geometry_path.stat().st_size)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"fixed geometry table has {actual_bytes} bytes; expected {expected_bytes} "
            f"({num_instances} instances x {FIXED_GEO_DIM} FP16 values)"
        )
    values = np.fromfile(geometry_path, dtype=np.float16)
    if values.size != int(num_instances) * FIXED_GEO_DIM:
        raise ValueError("fixed geometry table value count does not match runtime metadata")
    if not bool(np.isfinite(values).all()):
        raise ValueError("fixed geometry table contains non-finite values")
    return str(geometry_path.resolve())


def _survival_holdout_metrics(
    model: RayContextSurvivalOWRBModel,
    observations: dict[str, np.ndarray],
    holdout_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
) -> dict[str, float]:
    """Measure train-evidence event/censor likelihood on a fixed holdout.

    The holdout is carved out of the train-only relation evidence.  It is not a
    dataset validation split and is never used for visibility thresholding.
    """
    if holdout_indices.size == 0:
        return {"count": 0.0, "nll": 0.0, "brier": 0.0}
    was_training = model.training
    model.eval()
    total_weight = 0.0
    total_nll = 0.0
    total_brier = 0.0
    anchors = model.relation_encoder.anchors.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, int(holdout_indices.size), int(batch_size)):
            local = holdout_indices[start : start + int(batch_size)]
            ids = torch.from_numpy(np.asarray(observations["instance"][local], dtype=np.int64)).to(device)
            direction_ids = torch.from_numpy(np.asarray(observations["direction"][local], dtype=np.int64)).to(device)
            rho = torch.from_numpy(np.asarray(observations["rho"][local], dtype=np.float32)).to(device).view(-1, 1)
            event = torch.from_numpy(np.asarray(observations["event"][local], dtype=np.float32)).to(device).view(-1, 1)
            weight = torch.clamp(
                torch.from_numpy(np.asarray(observations["weight"][local], dtype=np.float32)).to(device).view(-1, 1),
                min=0.1,
            )
            direction = anchors[direction_ids]
            semantic, _ = model.survival_query_from_direction(
                direction, rho, model.survival_coefficients[ids]
            )
            survival = torch.clamp(semantic[:, :1], 1e-5, 1.0 - 1e-5)
            nll = -(event * torch.log(1.0 - survival) + (1.0 - event) * torch.log(survival))
            total_nll += float((nll * weight).sum().cpu())
            total_brier += float(((survival - (1.0 - event)).square() * weight).sum().cpu())
            total_weight += float(weight.sum().cpu())
    if was_training:
        model.train()
    return {
        "count": float(holdout_indices.size),
        "nll": total_nll / max(total_weight, 1e-6),
        "brier": total_brier / max(total_weight, 1e-6),
    }


def _pretrain_context_encoder(
    model: RayContextSurvivalOWRBModel,
    geometry: torch.Tensor,
    token_targets: torch.Tensor,
    aux_targets: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    observations: dict[str, np.ndarray] | None = None,
    observation_batch_size: int = 4096,
) -> dict[str, Any]:
    if steps <= 0 or model.context_mode == "zero":
        return {"steps": 0.0, "loss": 0.0, "survivalTrainNll": 0.0, "survivalHoldout": {"count": 0.0, "nll": 0.0, "brier": 0.0}}
    model.train()
    trainable = (
        list(model.relation_encoder.parameters())
        + list(model.relation_aux_head.parameters())
        + list(model.context_direction_basis.parameters())
        + list(model.survival_direction_basis.parameters())
        + [model.survival_coefficients]
    )
    optimizer = torch.optim.AdamW(trainable, lr=float(lr), weight_decay=1e-5)
    generator = torch.Generator(device=geometry.device).manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    ids_all = torch.arange(model.num_instances, device=geometry.device, dtype=torch.long)
    losses: list[float] = []
    survival_losses: list[float] = []
    token_targets = token_targets.to(geometry.device)
    aux_targets = aux_targets.to(geometry.device)
    observation_tensors: dict[str, torch.Tensor] | None = None
    train_observation_indices = np.zeros((0,), dtype=np.int64)
    holdout_observation_indices = np.zeros((0,), dtype=np.int64)
    if observations is not None and observations.get("instance", np.zeros((0,))).size:
        observation_tensors = {
            key: torch.from_numpy(np.asarray(value)).to(geometry.device)
            for key, value in observations.items()
        }
        all_observation_indices = np.arange(observation_tensors["instance"].numel(), dtype=np.int64)
        holdout_mask = (all_observation_indices % 5) == 0
        holdout_observation_indices = all_observation_indices[holdout_mask]
        train_observation_indices = all_observation_indices[~holdout_mask]
    for _ in range(int(steps)):
        ids = ids_all[torch.randint(ids_all.numel(), (int(batch_size),), generator=generator, device=geometry.device)]
        tokens, _strength = model._relation_tokens(ids, geometry)
        token_loss = F.mse_loss(tokens, token_targets[ids])
        aux_prediction = model.relation_aux_head(tokens)
        aux_loss = F.mse_loss(aux_prediction, aux_targets[ids])
        coefficients = model.fit_context_coefficients(tokens)
        anchors = model.relation_encoder.anchors.to(geometry.device, geometry.dtype)
        phi = model.context_direction_basis(anchors)
        reconstructed = torch.einsum("ma,bak->bmk", phi, coefficients)
        reconstruction_loss = F.mse_loss(reconstructed, tokens)
        loss = token_loss + 0.10 * aux_loss + 0.20 * reconstruction_loss
        if observation_tensors is not None and train_observation_indices.size:
            count = min(int(observation_batch_size), int(train_observation_indices.size))
            picked_np = rng.choice(train_observation_indices, size=count, replace=False)
            picked = torch.from_numpy(picked_np).to(geometry.device)
            survival_batch = {key: value[picked] for key, value in observation_tensors.items()}
            survival_loss, _survival_parts = survival_censoring_loss(model, survival_batch)
            loss = loss + 0.20 * survival_loss
            survival_losses.append(float(survival_loss.detach().cpu()))
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("context pretraining produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    holdout = _survival_holdout_metrics(
        model,
        observations or {},
        holdout_observation_indices,
        geometry.device,
    )
    return {
        "steps": float(steps),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "survivalTrainNll": float(np.mean(survival_losses)) if survival_losses else 0.0,
        "survivalHoldout": holdout,
        "survivalSplit": {
            "source": "train-only relation evidence",
            "trainCount": int(train_observation_indices.size),
            "holdoutCount": int(holdout_observation_indices.size),
            "holdoutRule": "observation index modulo 5 equals zero",
        },
    }


def _runtime_features(model: RayContextSurvivalOWRBModel, geometry: torch.Tensor) -> torch.Tensor:
    return model.runtime_features_from_offline(geometry, model.context_coefficients)


def _set_main_training_stage(
    model: RayContextSurvivalOWRBModel,
) -> tuple[str, list[str], list[str]]:
    """Freeze offline encoders and enable only the compact runtime query."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules = {
        "survival_direction_basis": model.survival_direction_basis,
        "base_trunk": model.base_trunk,
        "base_visibility_head": model.base_visibility_head,
        "utility_head": model.utility_head,
        "download_head": model.download_head,
    }

    trainable_names: list[str] = []
    for module_name, module in modules.items():
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(True)
            trainable_names.append(f"{module_name}.{name}")
    model.survival_coefficients.requires_grad_(True)
    trainable_names.append("survival_coefficients")

    frozen_names = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    return "compact_query", sorted(trainable_names), sorted(frozen_names)


@torch.no_grad()
def _evaluate_rows(
    model: RayContextSurvivalOWRBModel,
    split: Any,
    runtime_features: torch.Tensor,
    world_aabbs: np.ndarray,
    device: torch.device,
    seed: int,
    threshold: float | None,
    max_poses: int,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    effective_split = split
    if max_poses > 0:
        effective_split = split.dataset.subset(f"owrb_eval_{max_poses}", split.pose_indices[: int(max_poses)])
    thresholds = threshold_grid() if threshold is None else np.asarray([threshold], dtype=np.float32)
    return evaluate_thresholds(
        model,
        effective_split,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=1,
        max_steps=None,
        max_candidates_per_pose=0,
        seed=int(seed),
        thresholds=thresholds,
        collect_pose_stats=True,
        allow_candidate_visible_union=False,
        bootstrap_replicates=int(bootstrap_replicates),
    )


def _metrics_to_json(value: Any) -> Any:
    return _json_value(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--runtime-meta", default="hkust-v3/assets/runtimeVisibilityMeta.json")
    parser.add_argument("--glb-points", default="neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin")
    parser.add_argument("--initial-geo-features", default=str(DEFAULT_INITIAL_GEO))
    parser.add_argument("--glb-index", default="hkust-v3/assets/glbIndex.json")
    parser.add_argument("--glb-root", default="hkust-v3/assets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="pvs_ray_context_survival_owrb_v1")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=75)
    parser.add_argument("--pose-set-batch-size", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--max-eval-poses", type=int, default=0)
    parser.add_argument("--glb-train-points", type=int, default=64)
    parser.add_argument("--feature-export-batch-size", type=int, default=512)
    parser.add_argument("--context-query-batch-size", type=int, default=128)
    parser.add_argument("--point-hidden-dim", type=int, default=160)
    parser.add_argument("--pointnetpp-centers", type=int, default=24)
    parser.add_argument("--pointnetpp-neighbors", type=int, default=12)
    parser.add_argument("--relation-hidden-dim", type=int, default=64)
    parser.add_argument("--context-mode", choices=("zero", "pooled", "directional"), default="directional")
    parser.add_argument("--context-query-dim", type=int, choices=(8, 16), default=8)
    parser.add_argument("--ray-mode", choices=("direct9", "fourier117"), default="direct9")
    parser.add_argument(
        "--survival-parameterization",
        choices=("monotone", "unconstrained28"),
        default="monotone",
    )
    parser.add_argument("--context-coefficients", default="")
    parser.add_argument("--context-pretrain-steps", type=int, default=64)
    parser.add_argument("--context-pretrain-batch-size", type=int, default=256)
    parser.add_argument("--context-pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--survival-loss-weight", type=float, default=0.25)
    parser.add_argument("--utility-loss-weight", type=float, default=0.18)
    parser.add_argument("--glb-priority-loss-weight", type=float, default=0.20)
    parser.add_argument("--survival-observation-batch-size", type=int, default=4096)
    parser.add_argument(
        "--loss-mode",
        choices=("owrb", "rvl_strong_v2", "safety_constraint"),
        default="owrb",
        help="registered visibility loss",
    )
    parser.add_argument("--gamma", type=float, default=0.50, help="GLB-byte term mixture for safety_constraint")
    parser.add_argument("--owrb-target-visible-mass", type=float, default=0.995)
    parser.add_argument("--owrb-top-k", type=int, default=128)
    parser.add_argument("--owrb-margin", type=float, default=0.20)
    parser.add_argument("--owrb-positive-margin", type=float, default=0.05)
    parser.add_argument("--owrb-lambda-bce", type=float, default=0.50)
    parser.add_argument("--owrb-lambda-negative", type=float, default=1.0)
    parser.add_argument("--owrb-lambda-positive", type=float, default=0.75)
    parser.add_argument("--owrb-lambda-boundary", type=float, default=0.10)
    parser.add_argument("--owrb-survival-beta", type=float, default=0.50)
    parser.add_argument("--survival-input-mode", choices=("semantic", "raw"), default="semantic")
    parser.add_argument("--disable-survival", action="store_true")
    parser.add_argument("--reg-weight", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--allow-unsafe-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("epochs and steps-per-epoch must be positive")
    if args.calibration_bootstrap_replicates < 1000:
        raise ValueError("calibration bootstrap replicates must be at least 1000")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(args.runtime_meta)
    scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    num_instances = int(world_aabbs.shape[0])
    num_glbs = int(instance_to_glb.max()) + 1
    args.initial_geo_features = _validate_fixed_geometry(args.initial_geo_features, num_instances)
    resource_report = validate_training_resources(
        Path(args.dataset_dir), Path(args.runtime_meta), Path(args.glb_points),
        num_instances, num_glbs, strict_semantics=True,
    )
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=num_instances)
    protocol_split = _protocol_splits(dataset, args.seed)
    train_split = dataset.split("train")
    validation_split = dataset.split("validation")
    calibration_split = dataset.split("calibration")
    candidate_audit = audit_native_aabb_candidates(
        dataset,
        world_aabbs,
        np.concatenate([
            train_split.pose_indices,
            validation_split.pose_indices,
            calibration_split.pose_indices,
        ]),
    )
    evidence = load_ray_context_survival_evidence(args.evidence_dir, num_instances)
    evidence_meta = evidence["meta"]
    if evidence_meta.get("splits") != ["train"]:
        raise ValueError("the OWRB main trainer requires train-only relation evidence")
    train_candidate_digest = _candidate_digest_for_split(dataset, train_split)
    if train_candidate_digest != evidence_meta.get("candidateDigest"):
        raise ValueError("relation evidence candidate hash does not match the native train candidate CSR")
    observations_np = evidence["observations"]
    if observations_np["instance"].size and int(observations_np["instance"].max()) >= num_instances:
        raise ValueError("survival evidence references an instance outside runtime metadata")

    model = RayContextSurvivalOWRBModel(
        num_instances=num_instances,
        num_glbs=num_glbs,
        point_hidden_dim=args.point_hidden_dim,
        pointnetpp_centers=args.pointnetpp_centers,
        pointnetpp_neighbors=args.pointnetpp_neighbors,
        relation_hidden_dim=args.relation_hidden_dim,
        context_mode=args.context_mode,
        survival_enabled=not args.disable_survival,
        survival_input_mode=args.survival_input_mode,
        context_query_dim=args.context_query_dim,
        ray_mode=args.ray_mode,
        survival_parameterization=args.survival_parameterization,
        scene_size_m=scene_size.tolist(),
    ).to(device)
    model.set_scene_bounds(torch.from_numpy(scene_min).to(device), torch.from_numpy(scene_size).to(device))
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    model.set_relation_evidence(
        torch.from_numpy(evidence["source_ids"]).to(device),
        torch.from_numpy(evidence["relation_stats"]).to(device),
        torch.from_numpy(evidence["strength"]).to(device),
        torch.from_numpy(evidence["count"]).to(device),
    )
    geometry, geometry_source = _load_or_compute_geometry(
        model, args.initial_geo_features, Path(args.glb_points), device,
        num_instances, num_glbs, args.glb_train_points, args.feature_export_batch_size,
    )
    geometry = geometry.detach()

    token_targets, aux_targets = _context_targets(evidence, model.context_query_dim)
    context_pretrain = {"steps": 0.0, "loss": 0.0}
    context_source = "zero"
    if args.context_coefficients:
        values = np.fromfile(args.context_coefficients, dtype=np.float16)
        expected_context_values = num_instances * 4 * model.context_query_dim
        if values.size != expected_context_values:
            raise ValueError(
                f"context coefficient file has {values.size} values; "
                f"expected {expected_context_values}"
            )
        context_coefficients = torch.from_numpy(
            values.reshape(num_instances, 4, model.context_query_dim).astype(np.float32)
        ).to(device)
        context_source = str(Path(args.context_coefficients).resolve())
    elif args.context_mode == "zero":
        context_coefficients = torch.zeros(
            (num_instances, 4, model.context_query_dim), device=device
        )
        context_source = "explicit_zero_context_ablation"
    else:
        context_pretrain = _pretrain_context_encoder(
            model, geometry, token_targets, aux_targets, args.context_pretrain_steps,
            args.context_pretrain_batch_size, args.context_pretrain_lr, args.seed,
            observations=observations_np,
            observation_batch_size=args.survival_observation_batch_size,
        )
        context_coefficients = model.compute_offline_context_coefficients(
            geometry, mode=args.context_mode, batch_size=args.context_query_batch_size
        ).detach()
        context_source = "offline_relation_encoder_after_train_only_evidence_pretraining"
    model.set_context_coefficients(context_coefficients)
    context_coefficients.detach().cpu().numpy().astype(np.float16).tofile(output_dir / "context_coefficients_fp16.bin")

    # Geometry and relation encoders are offline-only. The visibility stage
    # trains only the compact browser query and the survival coefficients.
    stage, trainable_names, frozen_names = _set_main_training_stage(model)
    observations = {key: torch.from_numpy(value).to(device) for key, value in observations_np.items()}
    glb_cost_norm = torch.from_numpy(
        load_glb_costs(args.glb_index, args.glb_root, num_glbs, instance_to_glb, allow_missing=False)
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    rng = np.random.default_rng(args.seed)
    owrb_config = OWRBConfig(
        target_visible_mass=args.owrb_target_visible_mass,
        top_k_hard_negatives=args.owrb_top_k,
        boundary_margin=args.owrb_margin,
        positive_margin=args.owrb_positive_margin,
        lambda_balanced_bce=args.owrb_lambda_bce,
        lambda_hard_negative=args.owrb_lambda_negative,
        lambda_positive_protection=args.owrb_lambda_positive,
        lambda_boundary_alignment=args.owrb_lambda_boundary,
        survival_negative_beta=args.owrb_survival_beta,
    )
    owrb_state = OWRBState()
    safety_dual_state = SafetyDualState()
    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    model_meta: dict[str, Any] = {
        "schema": "ray-context-survival-owrb-training-v1",
        "experiment": args.experiment_name,
        "modelConfig": model.config,
        "args": vars(args),
        "protocolSplit": protocol_split,
        "resourceReport": resource_report,
        "evidenceMeta": evidence_meta,
        "geometrySource": geometry_source,
        "contextSource": context_source,
        "contextPretraining": context_pretrain,
        "trainingStagePolicy": {
            "stage": stage,
            "trainableParameters": trainable_names,
            "frozenParameters": frozen_names,
            "geometryEncoderAlwaysFrozen": True,
            "policy": "fixed_offline_tables_then_compact_query_heads",
        },
        "dataSemantics": {
            "candidate": "native frustum_ids.bin/frustum_offsets.bin stored back-camera candidates",
            "candidateVisibleUnion": False,
            "candidateAudit": candidate_audit,
            "modelInputFovYDeg": 66.0,
            "frontendRenderFovYDeg": 60.0,
            "visibleWeights": dataset.visible_weight_semantics,
            "testReadDuringTraining": False,
        },
        "calibrationPolicy": {
            "primarySafetyMetric": "weighted_recall",
            "selectionRule": weighted_cull_selection_rule(0.99, 0.99),
            "poseRecallRole": "diagnostic_only",
        },
        "rvlControlPolicy": {
            "relationEvidenceUsed": False,
            "evidenceTensor": "fixed_zero",
            "reason": "the formal RVL control must not receive the new triangle-depth relation strength",
        },
        "owrbConfig": vars(owrb_config),
        "safetyConstraint": {
            "gamma": float(args.gamma),
            "dualState": safety_dual_state.as_dict(),
            "constraints": {
                "poseMiss": 0.05,
                "weightedMiss": 0.01,
                "weightedMissCvar95": 0.03,
            },
        },
    }
    (output_dir / "model_meta.json").write_text(json.dumps(_json_value(model_meta), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "protocol_split.json").write_text(json.dumps(protocol_split, ensure_ascii=False, indent=2), encoding="utf-8")

    best_key: tuple[float, float, float, float] | None = None
    best_path: Path | None = None
    best_calibration: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        epoch_metric_values: dict[str, list[float]] = {}
        iterator = train_split.pose_set_batches(
            args.pose_set_batch_size,
            rng,
            args.steps_per_epoch,
            include_empty=True,
        )
        for pose_indices in tqdm(
            iterator,
            total=args.steps_per_epoch,
            desc=f"{args.experiment_name} {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            batch = train_split.build_pose_set_batch(
                pose_indices, world_aabbs, rng, max_candidates_per_pose=0,
                allow_candidate_visible_union=False, include_empty=True,
            )
            if batch["instance"].size == 0:
                continue
            ids = torch.from_numpy(batch["instance"]).to(device)
            camera = torch.from_numpy(batch["camera"]).to(device)
            camera_world = torch.from_numpy(batch["camera_world"]).to(device)
            view = torch.from_numpy(batch["camera_view"]).to(device)
            target = torch.from_numpy(batch["target"][:, None]).to(device)
            visible_weights = torch.from_numpy(batch["visible_weights"][:, None]).to(device)
            offsets = torch.from_numpy(batch["pose_offsets"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            runtime = _runtime_features(model, geometry)
            logits, aux = model.compute_logits_with_aux(camera, view, camera_world, ids, runtime_features=runtime)
            if args.loss_mode == "owrb":
                visibility_loss, visibility_parts = owrb_visibility_loss(
                    logits, target, offsets, visible_weights,
                    survival_occlusion_probability=aux["survival_occlusion_probability"],
                    config=owrb_config, state=owrb_state,
                )
            elif args.loss_mode == "safety_constraint":
                visibility_loss, visibility_parts = safety_constraint_utility_loss(
                    logits,
                    target,
                    offsets,
                    visible_weights,
                    ids,
                    model.instance_to_glb,
                    glb_cost_norm,
                    args.gamma,
                    safety_dual_state,
                )
            else:
                visibility_loss, visibility_parts = rvl_strong_v2_visibility_loss(
                    logits,
                    target,
                    offsets,
                    visible_weights,
                    neutral_rvl_control_evidence(logits),
                )
            observation_count = int(observations["instance"].numel())
            if observation_count > args.survival_observation_batch_size:
                picked = torch.from_numpy(rng.choice(observation_count, args.survival_observation_batch_size, replace=False)).to(device)
                survival_batch = {key: value[picked] for key, value in observations.items()}
            else:
                survival_batch = observations
            if args.disable_survival:
                survival_loss = logits.sum() * 0.0
                survival_parts = {"lossSurvivalCensor": survival_loss, "survivalEventRate": 0.0}
            else:
                survival_loss, survival_parts = survival_censoring_loss(model, survival_batch)
            utility_loss, utility_parts = visual_utility_loss(
                aux, target, visible_weights, offsets, invisible_weight=0.25,
                rank_weight=0.3, rank_margin=0.1, rank_positive_top_k=32, rank_negative_top_k=128,
            )
            download_loss, download_parts = glb_priority_loss(
                aux["download_logits"], ids, target, visible_weights,
                model.instance_to_glb, glb_cost_norm, offsets,
            )
            regularization = model.regularization()
            other_loss = (
                args.survival_loss_weight * survival_loss
                + args.utility_loss_weight * utility_loss
                + args.glb_priority_loss_weight * download_loss
                + args.reg_weight * regularization
            )
            loss = visibility_loss + other_loss
            components = {
                "total": loss,
                "visibility": visibility_loss,
                "survival": survival_loss,
                "utility": utility_loss,
                "download": download_loss,
                "regularization": regularization,
            }
            if not bool(torch.isfinite(torch.stack([value.float() for value in components.values()])).all()):
                raise FloatingPointError(f"non-finite OWRB training loss at epoch {epoch + 1}: {_json_value(components)}")
            loss.backward()
            bad_gradient = [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            ]
            if bad_gradient:
                raise FloatingPointError(f"non-finite gradients: {bad_gradient}")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if args.loss_mode == "safety_constraint":
                safety_dual_state.update({
                    key: visibility_parts[key]
                    for key in ("pose_miss", "weighted_miss", "weighted_miss_cvar")
                })
            step_parts = {
                "loss": loss,
                **_json_value(visibility_parts),
                **_json_value(survival_parts),
                **_json_value(utility_parts),
                **_json_value(download_parts),
            }
            for key, value in step_parts.items():
                numeric = _json_value(value)
                if isinstance(numeric, (int, float, np.integer, np.floating)) and np.isfinite(float(numeric)):
                    epoch_metric_values.setdefault(str(key), []).append(float(numeric))
        scheduler.step()
        metric_summary = {
            key: {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "count": int(len(values)),
            }
            for key, values in epoch_metric_values.items()
            if values
        }
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "stage": stage,
            "trainableParameters": trainable_names,
            "frozenParameters": frozen_names,
            "loss": float(metric_summary.get("loss", {}).get("mean", 0.0)),
            "lr": float(scheduler.get_last_lr()[0]),
            "stepsPerEpoch": int(args.steps_per_epoch),
            "owrbState": owrb_state.as_dict(),
            "safetyDualState": safety_dual_state.as_dict(),
            "trainMetricSummary": metric_summary,
            **{f"train_{key}": value["mean"] for key, value in metric_summary.items()},
        }
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            runtime = _runtime_features(model, geometry).detach()
            calibration_rows = _evaluate_rows(
                model, calibration_split, runtime, world_aabbs, device, args.seed + epoch,
                threshold=None, max_poses=args.max_eval_poses,
                bootstrap_replicates=args.calibration_bootstrap_replicates,
            )
            selected = select_weighted_cull_workpoint(
                calibration_rows, target_weighted_recall=0.99,
                minimum_lower_confidence_bound=0.99,
            )
            diagnostic = max(
                calibration_rows,
                key=lambda item: (
                    float(item.get("pose_f1", 0.0)),
                    float(item.get("pose_precision", 0.0)),
                    float(item.get("pose_weighted_recall", 0.0)),
                    -float(item.get("avg_pred_count", 0.0)),
                ),
            )
            threshold = float(selected["threshold"] if selected is not None else diagnostic["threshold"])
            validation_rows = _evaluate_rows(
                model, validation_split, runtime, world_aabbs, device, args.seed + epoch + 1,
                threshold=threshold, max_poses=args.max_eval_poses,
                bootstrap_replicates=args.calibration_bootstrap_replicates,
            )
            row["calibration"] = {
                "selected": selected,
                "diagnostic": diagnostic,
                "diagnosticThreshold": float(diagnostic["threshold"]),
                "rows": calibration_rows,
            }
            row["validationAtCalibration"] = validation_rows[0] if validation_rows else None
            safe = selected is not None
            row["safeCalibrationWorkpoint"] = bool(safe)
            if safe or args.allow_unsafe_final:
                selected_for_key = selected or diagnostic
                key = (
                    float(selected_for_key.get("pose_useful_cull", 0.0)),
                    float(selected_for_key.get("pose_balanced_accuracy", 0.0)),
                    float(selected_for_key.get("pose_precision", 0.0)),
                    -float(selected_for_key.get("avg_pred_count", 0.0)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_path = output_dir / "best.pt"
                    best_calibration = row["calibration"]
                    runtime.detach().cpu().numpy().astype(np.float16).tofile(output_dir / "best_instance_runtime_features_fp16.bin")
                    geometry.detach().cpu().numpy().astype(np.float16).tofile(output_dir / "best_instance_geo_features_fp16.bin")
                    context_coefficients.detach().cpu().numpy().astype(np.float16).tofile(output_dir / "best_instance_context_coefficients_fp16.bin")
                    checkpoint = {
                        "model": model.state_dict(), "config": model.config, "args": vars(args),
                        "epoch": epoch + 1, "calibration": row["calibration"],
                        "validationAtCalibration": row["validationAtCalibration"],
                        "protocolSplit": protocol_split, "evidenceMeta": evidence_meta,
                        "contextPretraining": context_pretrain,
                        "owrbState": owrb_state.as_dict(),
                        "safetyDualState": safety_dual_state.as_dict(),
                        "trainingStage": {
                            "name": stage,
                            "trainableParameters": trainable_names,
                            "frozenParameters": frozen_names,
                        },
                        "best": {"threshold": threshold, **selected_for_key, "safe": bool(safe)},
                    }
                    torch.save(checkpoint, best_path)
            print(json.dumps(_json_value(row), ensure_ascii=False), flush=True)
        else:
            print(json.dumps(_json_value(row), ensure_ascii=False), flush=True)
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_value(row), ensure_ascii=False) + "\n")
        torch.save({
            "model": model.state_dict(), "config": model.config, "args": vars(args),
            "epoch": epoch + 1, "calibration": row.get("calibration"),
            "validationAtCalibration": row.get("validationAtCalibration"),
            "protocolSplit": protocol_split, "evidenceMeta": evidence_meta,
            "contextPretraining": context_pretrain,
            "owrbState": owrb_state.as_dict(),
            "safetyDualState": safety_dual_state.as_dict(),
            "trainingStage": {
                "name": stage,
                "trainableParameters": trainable_names,
                "frozenParameters": frozen_names,
            },
        }, output_dir / "last.pt")

    if best_path is None and not args.allow_unsafe_final:
        raise RuntimeError("no qualified weighted-recall calibration workpoint; use --allow-unsafe-final for pilot diagnostics only")
    model.eval()
    runtime = _runtime_features(model, geometry).detach().cpu().numpy().astype(np.float16)
    runtime.tofile(output_dir / "instance_runtime_features_fp16.bin")
    geometry.detach().cpu().numpy().astype(np.float16).tofile(output_dir / "instance_geo_features_fp16.bin")
    context_coefficients.detach().cpu().numpy().astype(np.float16).tofile(output_dir / "instance_context_coefficients_fp16.bin")
    model_meta["export"] = {
        "schema": "ray-context-survival-scheduler-v1",
        "numInstances": num_instances,
        "runtimeFeatureDim": int(runtime.shape[1]),
        "dtype": "float16",
        "files": {
            "runtime": "instance_runtime_features_fp16.bin",
            "geometry": "instance_geo_features_fp16.bin",
            "contextCoefficients": "instance_context_coefficients_fp16.bin",
        },
        "estimatedBytes": int(runtime.nbytes),
    }
    model_meta["status"] = "calibration_candidate" if best_path is not None else "pilot_diagnostic"
    if best_path is not None:
        model_meta["bestExport"] = {
            "checkpoint": "best.pt",
            "runtime": "best_instance_runtime_features_fp16.bin",
            "geometry": "best_instance_geo_features_fp16.bin",
            "contextCoefficients": "best_instance_context_coefficients_fp16.bin",
        }
    (output_dir / "model_meta.json").write_text(json.dumps(_json_value(model_meta), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "train_history.json").write_text(json.dumps(_json_value(history), ensure_ascii=False, indent=2), encoding="utf-8")
    final_calibration = best_calibration
    if final_calibration is None:
        for item in reversed(history):
            if item.get("calibration"):
                final_calibration = item["calibration"]
                break
    calibration_summary = {
        "schema": "ray-context-survival-owrb-calibration-v1",
        "protocol": "calibration_ready_pre_test",
        "testRead": False,
        "selectionRule": weighted_cull_selection_rule(0.99, 0.99),
        "weightedRecallFloor": 0.99,
        "weightedRecallLowerConfidenceBoundFloor": 0.99,
        "bootstrapReplicates": int(args.calibration_bootstrap_replicates),
        "selected": (final_calibration or {}).get("selected"),
        "diagnostic": (final_calibration or {}).get("diagnostic"),
        "diagnosticThreshold": (final_calibration or {}).get("diagnosticThreshold"),
        "thresholdRows": (final_calibration or {}).get("rows", []),
        "status": "safe" if (final_calibration or {}).get("selected") is not None else "no_qualified_safety_workpoint",
    }
    (output_dir / "calibration_ready_summary.json").write_text(json.dumps(_json_value(calibration_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": model_meta["status"], "outputDir": str(output_dir), "runtimeFeatureBytes": int(runtime.nbytes), "best": str(best_path) if best_path else None, "testRead": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
