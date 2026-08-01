#!/usr/bin/env python3
"""Run inference-time interventions for the directional PVS model.

The runner keeps the checkpoint, stored back-camera candidate CSR, pose order,
and calibration-frozen threshold fixed.  It is a diagnostic experiment: no
model parameters are updated and the sealed test split cannot receive an
explicit threshold override.

The fixed runtime table is laid out as ``geometry | context | proxy``.  The
proxy part is reshaped to ``[instance, direction, depth, channel]`` before an
intervention and flattened again without changing the model checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from directional_occlusion_proxy_encoder_model import DirectionalOcclusionProxyEncoderPVSModel  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from common.runtime_meta import scene_min_max  # noqa: E402


# These names are intentionally descriptive because they appear in the paper
# tables and in the JSON evidence.  The two old names are accepted only as
# compatibility aliases; they do not create additional experiments.
INTERVENTIONS = (
    "baseline",
    "proxy_zero",
    "proxy_random_same_distribution",
    "proxy_direction_mean",
    "proxy_direction_roll",
    "proxy_cross_instance_permutation",
    "context_zero",
    "context_cross_instance_permutation",
    "context_proxy_zero",
)

INTERVENTION_ALIASES = {
    "proxy_instance_permutation": "proxy_cross_instance_permutation",
    "context_instance_permutation": "context_cross_instance_permutation",
}

INTERVENTION_DESCRIPTIONS = {
    "baseline": "unchanged fixed geometry, context, and directional proxy table",
    "proxy_zero": "set every directional/depth proxy channel to zero",
    "proxy_random_same_distribution": "replace proxy values with seeded Gaussian values using the table-wide per-slice mean and standard deviation",
    "proxy_direction_mean": "replace all direction bins by their per-instance mean while retaining depth shells",
    "proxy_direction_roll": "cyclically shift direction bins by one while retaining their marginal values",
    "proxy_cross_instance_permutation": "permute the complete proxy tensor across instance rows",
    "context_zero": "set the fixed context vector to zero while retaining geometry and proxy",
    "context_cross_instance_permutation": "permute the complete context vector across instance rows",
    "context_proxy_zero": "set both context and directional proxy vectors to zero",
}

POSE_METRIC_NAMES = (
    "precision",
    "recall",
    "f1",
    "jaccard",
    "weighted_recall",
    "accuracy",
    "balanced_accuracy",
    "specificity",
    "useful_cull",
    "bad_cull",
    "avg_pred_count",
    "candidate_count",
    "gt_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--runtime-features", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--runtime-meta", type=Path, default=None)
    parser.add_argument("--eval-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", choices=["validation", "calibration", "test"], default="validation")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--poses-per-batch", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--allow-test-diagnostic",
        action="store_true",
        help="Allow a diagnostic test-split run with the already frozen threshold; it is not a formal test evaluation.",
    )
    parser.add_argument(
        "--interventions",
        default=",".join(INTERVENTIONS),
        help="Comma-separated intervention names. The baseline is always included.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a synthetic fixture for intervention transforms and metric definitions, then exit.",
    )
    return parser.parse_args()


def build_model(checkpoint: dict[str, Any], runtime_meta: dict[str, Any], device: torch.device):
    config = checkpoint.get("config") or {}
    world_aabbs, instance_to_glb, _ = load_runtime_meta_from_payload(runtime_meta)
    scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    model = DirectionalOcclusionProxyEncoderPVSModel(
        num_instances=int(config["numInstances"]),
        num_glbs=int(config["numGlbs"]),
        geo_dim=int(config["geoDim"]),
        context_dim=int(config["contextDim"]),
        proxy_dim=int(config["proxyDim"]),
        direction_bins=int(config["directionBins"]),
        depth_shells=int(config["depthShells"]),
        source_k=int(config["sourceK"]),
        point_hidden_dim=int((checkpoint.get("args") or {}).get("point_hidden_dim", 160)),
        pointnetpp_centers=int((checkpoint.get("args") or {}).get("pointnetpp_centers", 24)),
        pointnetpp_neighbors=int((checkpoint.get("args") or {}).get("pointnetpp_neighbors", 12)),
        graph_hidden_dim=int((checkpoint.get("args") or {}).get("graph_hidden_dim", 160)),
        graph_message_dim=int((checkpoint.get("args") or {}).get("graph_message_dim", 96)),
        ray_fourier_bands=int((checkpoint.get("args") or {}).get("ray_fourier_bands", 10)),
        ray_scalar_fourier_bands=int((checkpoint.get("args") or {}).get("ray_scalar_fourier_bands", 4)),
        camera_location_dim=int((checkpoint.get("args") or {}).get("camera_location_dim", 0)),
        mlp_hidden=int((checkpoint.get("args") or {}).get("mlp_hidden", 128)),
        interaction_dim=int((checkpoint.get("args") or {}).get("interaction_dim", 64)),
        scene_size_m=scene_size.tolist(),
        runtime_feature_ablation=str(config.get("runtimeFeatureAblation", "none")),
    ).to(device)
    model.set_scene_bounds(torch.from_numpy(scene_min).to(device), torch.from_numpy(scene_size).to(device))
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, world_aabbs


def load_runtime_meta_from_payload(runtime_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    records = runtime_meta.get("componentRecords") or []
    count = int(runtime_meta.get("instanceCount") or len(records))
    if count <= 0:
        raise ValueError("runtime meta contains no component instances")
    aabbs = np.zeros((count, 6), dtype=np.float32)
    mapping = np.zeros((count,), dtype=np.int64)
    seen = np.zeros((count,), dtype=bool)
    for record in records:
        index = int(record["componentGlobalId"])
        if index < 0 or index >= count:
            raise ValueError(f"componentGlobalId {index} is outside instanceCount={count}")
        bounds = record["bounds"]
        if "min" in bounds and "max" in bounds:
            mn = np.asarray(bounds["min"], dtype=np.float32)
            mx = np.asarray(bounds["max"], dtype=np.float32)
        else:
            center = np.asarray(bounds["center"], dtype=np.float32)
            size = np.asarray(bounds["size"], dtype=np.float32)
            mn = center - size * 0.5
            mx = center + size * 0.5
        if mn.shape != (3,) or mx.shape != (3,) or not np.isfinite(mn).all() or not np.isfinite(mx).all():
            raise ValueError(f"invalid bounds for componentGlobalId={index}")
        if np.any(mx < mn):
            raise ValueError(f"componentGlobalId={index} has max < min")
        aabbs[index, :3] = mn
        aabbs[index, 3:] = mx
        mapping[index] = int(record.get("globalGlbId", record.get("global_glb_id", 0)))
        seen[index] = True
    if not seen.all():
        missing = np.flatnonzero(~seen)[:8].tolist()
        raise ValueError(f"runtime meta has no component record for instance ids {missing}")
    return aabbs, mapping, runtime_meta


def load_threshold(args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    if args.threshold is not None:
        threshold = float(args.threshold)
        if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {args.threshold}")
        return threshold, {"source": "explicit_argument", "testThresholdOverride": False}
    if args.eval_summary is not None:
        summary_path = args.eval_summary
    else:
        # ``--skip-final-test`` deliberately writes a pre-test calibration
        # record instead of an eval_summary containing test data.  Prefer that
        # record so M3 can run without opening the sealed test split.
        candidates = (
            args.checkpoint.with_name("calibration_ready_summary.json"),
            args.checkpoint.with_name("eval_summary.json"),
        )
        summary_path = next((path for path in candidates if path.is_file()), candidates[0])
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"No calibration threshold record found at {summary_path}. "
            "M3 requires calibration_ready_summary.json or a frozen calibration summary."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = str(summary.get("protocol", ""))
    allowed_protocols = {"calibration_ready_pre_test", "frozen_calibration_one_shot_test"}
    if protocol not in allowed_protocols:
        raise ValueError(
            f"{summary_path} is not a calibration threshold record; refusing to infer an intervention threshold."
        )
    test_count = int(summary.get("testEvaluationCount", 0))
    if protocol == "calibration_ready_pre_test" and test_count != 0:
        raise ValueError(f"{summary_path} is marked pre-test but has testEvaluationCount={test_count}.")
    if protocol == "frozen_calibration_one_shot_test" and args.split != "test":
        # A test-derived summary may contain a calibration threshold, but
        # allowing it for a validation/calibration intervention would blur the
        # provenance.  Use the pre-test file for formal M3 instead.
        raise ValueError(
            f"{summary_path} contains a one-shot test result; validation/calibration M3 must use "
            "calibration_ready_summary.json."
        )
    if protocol == "frozen_calibration_one_shot_test" and not args.allow_test_diagnostic:
        raise ValueError("A one-shot test threshold may only be used with --allow-test-diagnostic.")
    if summary.get("frozenThreshold") is None:
        raise ValueError(f"{summary_path} has no frozenThreshold")
    threshold = float(summary["frozenThreshold"])
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"frozenThreshold must be in [0, 1], got {threshold}")
    return threshold, {
        "source": str(summary_path),
        "protocol": protocol,
        "testEvaluationCount": test_count,
        "checkpoint": summary.get("checkpoint"),
        "testThresholdOverride": False,
    }


def canonical_intervention_name(name: str) -> str:
    stripped = str(name).strip()
    return INTERVENTION_ALIASES.get(stripped, stripped)


def normalize_intervention_names(value: str) -> tuple[list[str], list[dict[str, str]]]:
    requested = [item.strip() for item in str(value).split(",") if item.strip()]
    if not requested:
        requested = ["baseline"]
    names: list[str] = []
    aliases_used: list[dict[str, str]] = []
    for requested_name in requested:
        canonical = canonical_intervention_name(requested_name)
        if canonical not in INTERVENTIONS:
            supported = ", ".join((*INTERVENTIONS, *INTERVENTION_ALIASES.keys()))
            raise ValueError(f"Unknown intervention {requested_name!r}; supported names: {supported}")
        if requested_name != canonical:
            aliases_used.append({"requested": requested_name, "canonical": canonical})
        if canonical not in names:
            names.append(canonical)
    if "baseline" not in names:
        names.insert(0, "baseline")
    return names, aliases_used


def _non_identity_permutation(size: int, generator: torch.Generator) -> torch.Tensor:
    permutation = torch.randperm(size, generator=generator, device="cpu")
    if size > 1 and bool(torch.equal(permutation, torch.arange(size, device="cpu"))):
        permutation = torch.roll(permutation, shifts=1, dims=0)
    return permutation


def _permutation_metadata(permutation: torch.Tensor) -> dict[str, Any]:
    values = permutation.detach().cpu().numpy().astype(np.int64, copy=False)
    identity = np.arange(values.size, dtype=np.int64)
    return {
        "size": int(values.size),
        "sha256": sha256_array(values),
        "fixedPointCount": int(np.equal(values, identity).sum()),
    }


def apply_runtime_intervention(
    base: torch.Tensor,
    model: DirectionalOcclusionProxyEncoderPVSModel,
    name: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply one named intervention and return its reproducibility metadata."""
    canonical = canonical_intervention_name(name)
    if canonical not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {name}")
    if base.ndim != 2 or not base.is_floating_point():
        raise ValueError(f"runtime feature table must be a 2-D floating tensor, got {tuple(base.shape)}")
    expected_dim = int(getattr(model, "runtime_feature_dim", base.shape[1]))
    if base.shape[1] != expected_dim:
        raise ValueError(f"runtime feature table has dim {base.shape[1]}, expected model runtime dim {expected_dim}")
    geo, context, proxy = model._split_runtime(base)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    metadata: dict[str, Any] = {
        "requestedName": str(name),
        "name": canonical,
        "seed": int(seed),
        "operation": INTERVENTION_DESCRIPTIONS[canonical],
        "inputShape": [int(value) for value in base.shape],
        "featureLayout": {
            "geometryDim": int(model.geo_dim),
            "contextDim": int(model.context_dim),
            "directionBins": int(model.direction_bins),
            "depthShells": int(model.depth_shells),
            "proxyDim": int(model.proxy_dim),
        },
    }
    if canonical == "baseline":
        output = base
    elif canonical == "proxy_zero":
        proxy = torch.zeros_like(proxy)
        output = torch.cat([geo, context, proxy.reshape(proxy.shape[0], -1)], dim=-1)
    elif canonical == "proxy_random_same_distribution":
        mean = proxy.mean(dim=0, keepdim=True)
        std = proxy.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        random_cpu = torch.randn(proxy.shape, generator=generator, dtype=torch.float32, device="cpu")
        proxy = (random_cpu.to(proxy.device, dtype=proxy.dtype) * std + mean).to(base.dtype)
        metadata["randomMarginal"] = {
            "meanMean": float(mean.float().mean().item()),
            "meanStd": float(std.float().mean().item()),
            "minStd": float(std.float().min().item()),
        }
        output = torch.cat([geo, context, proxy.reshape(proxy.shape[0], -1)], dim=-1)
    elif canonical == "proxy_direction_mean":
        proxy = proxy.mean(dim=1, keepdim=True).expand_as(proxy)
        output = torch.cat([geo, context, proxy.reshape(proxy.shape[0], -1)], dim=-1)
    elif canonical == "proxy_direction_roll":
        proxy = torch.roll(proxy, shifts=1, dims=1)
        metadata["directionShift"] = 1
        output = torch.cat([geo, context, proxy.reshape(proxy.shape[0], -1)], dim=-1)
    elif canonical == "proxy_cross_instance_permutation":
        permutation = _non_identity_permutation(proxy.shape[0], generator)
        proxy = proxy[permutation.to(proxy.device)]
        metadata["permutation"] = _permutation_metadata(permutation)
        output = torch.cat([geo, context, proxy.reshape(proxy.shape[0], -1)], dim=-1)
    elif canonical == "context_zero":
        context = torch.zeros_like(context)
        output = torch.cat([geo, context, proxy.reshape(proxy.shape[0], -1)], dim=-1)
    elif canonical == "context_cross_instance_permutation":
        permutation = _non_identity_permutation(context.shape[0], generator)
        context = context[permutation.to(context.device)]
        metadata["permutation"] = _permutation_metadata(permutation)
        output = torch.cat([geo, context, proxy.reshape(proxy.shape[0], -1)], dim=-1)
    elif canonical == "context_proxy_zero":
        context = torch.zeros_like(context)
        proxy = torch.zeros_like(proxy)
        output = torch.cat([geo, context, proxy.reshape(proxy.shape[0], -1)], dim=-1)
    else:  # pragma: no cover - guarded by the canonical-name check above.
        raise AssertionError(canonical)
    if output.shape != base.shape:
        raise AssertionError(f"intervention {canonical} changed runtime feature shape to {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError(f"intervention {canonical} produced non-finite runtime features")
    metadata["outputShape"] = [int(value) for value in output.shape]
    return output, metadata


def intervene_runtime_features(
    base: torch.Tensor,
    model: DirectionalOcclusionProxyEncoderPVSModel,
    name: str,
    seed: int,
) -> torch.Tensor:
    """Backward-compatible tensor-only wrapper around :func:`apply_runtime_intervention`."""
    output, _metadata = apply_runtime_intervention(base, model, name, seed)
    return output


def _stats(values: np.ndarray | Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    if not np.isfinite(array).all():
        raise FloatingPointError("non-finite diagnostic value encountered")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _metric_row(
    target: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    target_array = np.asarray(target).reshape(-1)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if target_array.size != score_array.size or target_array.size != weight_array.size:
        raise ValueError("target, scores, and visible weights must have the same length")
    if not np.isfinite(score_array).all() or not np.isfinite(weight_array).all():
        raise FloatingPointError("non-finite score or visible weight encountered")
    if np.any(weight_array < 0.0):
        raise ValueError("visible weights must be non-negative for weighted recall")
    n = int(target_array.size)
    if n == 0:
        return {
            "metric_valid": False,
            "weighted_recall_valid": False,
            **{name: 0.0 for name in POSE_METRIC_NAMES},
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
            "weighted_tp": 0.0,
            "weighted_gt": 0.0,
        }
    pred = score_array >= float(threshold)
    yy = target_array.astype(bool, copy=False)
    tp = int(np.logical_and(pred, yy).sum())
    fp = int(np.logical_and(pred, ~yy).sum())
    fn = int(np.logical_and(~pred, yy).sum())
    tn = int(n - tp - fp - fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    specificity = tn / max(1, tn + fp)
    weighted_tp = float(weight_array[np.logical_and(pred, yy)].sum())
    weighted_gt = float(weight_array[yy].sum())
    weighted_valid = weighted_gt > 0.0
    weighted_recall = weighted_tp / weighted_gt if weighted_valid else 0.0
    candidate = float(n)
    return {
        "metric_valid": True,
        "weighted_recall_valid": bool(weighted_valid),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(tp / max(1, tp + fp + fn)),
        "weighted_recall": float(weighted_recall),
        "accuracy": float((tp + tn) / candidate),
        "balanced_accuracy": float((recall + specificity) * 0.5),
        "specificity": float(specificity),
        "useful_cull": float(tn / candidate),
        "bad_cull": float(fn / candidate),
        "avg_pred_count": float(pred.sum()),
        "candidate_count": candidate,
        "gt_count": float(yy.sum()),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "weighted_tp": weighted_tp,
        "weighted_gt": weighted_gt,
    }


def normalized_gate_entropy(gate_logits: torch.Tensor) -> torch.Tensor:
    """Return entropy normalized to [0, 1] over direction-depth gate entries."""
    if gate_logits.ndim != 2 or gate_logits.shape[1] <= 0:
        raise ValueError(f"gate logits must be [N, K] with K>0, got {tuple(gate_logits.shape)}")
    gate = torch.softmax(gate_logits, dim=1)
    entropy = -(gate * torch.log(gate.clamp_min(1e-8))).sum(dim=1)
    if gate.shape[1] == 1:
        return torch.zeros_like(entropy)
    return entropy / float(np.log(gate.shape[1]))


def _diagnostics_for_batch(
    model: DirectionalOcclusionProxyEncoderPVSModel,
    camera_view: torch.Tensor,
    instance_ids: torch.Tensor,
    camera_world: torch.Tensor,
    camera_norm: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    camera_feat = model.ray_features(
        camera_view,
        instance_ids,
        camera_world,
        camera_pos_norm=camera_norm,
    )[3]
    gate_logits = model.ray_proxy_gate(camera_feat)
    gate_entropy = normalized_gate_entropy(gate_logits).float().cpu().numpy().reshape(-1)
    if not np.isfinite(gate_entropy).all():
        raise FloatingPointError("non-finite proxy gate entropy encountered")
    return gate_entropy, camera_feat.detach().float().cpu().numpy()


def build_pose_plan(split, poses_per_batch: int, seed: int) -> list[np.ndarray]:
    """Materialize only the pose-index plan so every intervention is paired exactly."""
    rng = np.random.default_rng(int(seed))
    plan = []
    for pose_indices in split.pose_set_batches(
        max(1, int(poses_per_batch)), rng, max_steps=None, include_empty=True
    ):
        plan.append(np.asarray(pose_indices, dtype=np.int64).reshape(-1).copy())
    return plan


def pose_plan_digest(plan: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for pose_indices in plan:
        values = np.asarray(pose_indices, dtype=np.int64).reshape(-1)
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@torch.no_grad()
def evaluate_variant(
    model: DirectionalOcclusionProxyEncoderPVSModel,
    runtime_features: torch.Tensor,
    split,
    world_aabbs: np.ndarray,
    threshold: float,
    poses_per_batch: int,
    seed: int,
    name: str,
    pose_plan: Sequence[np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # ``pose_plan`` is passed by main.  The fallback preserves the callable API
    # for small external diagnostics while retaining deterministic pairing.
    plan = list(pose_plan) if pose_plan is not None else build_pose_plan(split, poses_per_batch, seed)
    rng = np.random.default_rng(int(seed) + 17)
    totals = {key: 0.0 for key in POSE_METRIC_NAMES}
    tp = fp = fn = tn = 0.0
    weighted_tp = weighted_gt = 0.0
    total_pred = total_candidate = total_gt = 0.0
    poses_seen = 0
    nonempty_poses = 0
    aux_values: dict[str, list[float]] = {
        "base_logit": [],
        "final_logit": [],
        "suppression": [],
        "selected_proxy_abs": [],
        "gate_entropy": [],
    }
    per_pose: list[dict[str, Any]] = []
    for pose_indices in plan:
        batch = split.build_pose_set_batch(
            pose_indices,
            world_aabbs,
            rng,
            max_candidates_per_pose=0,
            allow_candidate_visible_union=False,
            include_empty=True,
        )
        offsets = np.asarray(batch["pose_offsets"], dtype=np.int64)
        if offsets.size != pose_indices.size + 1:
            raise ValueError(
                f"pose-set builder returned {offsets.size - 1} rows for {pose_indices.size} planned poses"
            )
        if batch["instance"].size:
            ids = torch.from_numpy(batch["instance"]).to(model.instance_world_aabbs.device)
            camera = torch.from_numpy(batch["camera"]).to(ids.device)
            camera_world = torch.from_numpy(batch["camera_world"]).to(ids.device)
            camera_view = torch.from_numpy(batch["camera_view"]).to(ids.device)
            logits, aux = model.compute_logits_with_aux(
                camera,
                camera_view,
                camera_world,
                ids,
                runtime_features=runtime_features,
            )
            scores = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
            base_logits = aux["base_logits"].float().cpu().numpy().reshape(-1)
            final_logits = aux["final_logits"].float().cpu().numpy().reshape(-1)
            suppression = aux["inhibition"].float().cpu().numpy().reshape(-1)
            selected_proxy = aux["selected_proxy"].float().cpu().numpy()
            gate_entropy, _camera_feature_cpu = _diagnostics_for_batch(
                model, camera_view, ids, camera_world, camera
            )
            for diagnostic_name, values in (
                ("scores", scores),
                ("base logits", base_logits),
                ("final logits", final_logits),
                ("suppression", suppression),
                ("selected proxy", selected_proxy),
                ("gate entropy", gate_entropy),
            ):
                if not np.isfinite(np.asarray(values)).all():
                    raise FloatingPointError(f"non-finite {diagnostic_name} for intervention {name}")
        else:
            scores = base_logits = final_logits = suppression = gate_entropy = np.zeros((0,), dtype=np.float32)
            selected_proxy = np.zeros((0, int(model.proxy_dim)), dtype=np.float32)

        for local_pose, pose_id in enumerate(pose_indices.tolist()):
            start = int(offsets[local_pose])
            end = int(offsets[local_pose + 1])
            row = _metric_row(
                batch["target"][start:end],
                scores[start:end],
                batch["visible_weights"][start:end],
                threshold,
            )
            row["pose_index"] = int(pose_id)
            row["candidate_id_sha256"] = sha256_array(
                np.asarray(batch["instance"][start:end], dtype=np.uint32)
            )
            row["base_logit"] = _stats(base_logits[start:end])
            row["final_logit"] = _stats(final_logits[start:end])
            row["suppression"] = _stats(suppression[start:end])
            row["gate_entropy"] = _stats(gate_entropy[start:end])
            row["selected_proxy_abs"] = _stats(np.abs(selected_proxy[start:end]))
            # Keep the old scalar fields for readers of the first protocol,
            # while the structured fields above are the authoritative record.
            row["mean_base_logit"] = row["base_logit"]["mean"]
            row["mean_final_logit"] = row["final_logit"]["mean"]
            row["mean_inhibition"] = row["suppression"]["mean"]
            row["mean_proxy_gate_entropy"] = row["gate_entropy"]["mean"]
            row["mean_selected_proxy_abs"] = row["selected_proxy_abs"]["mean"]
            per_pose.append(row)
            poses_seen += 1
            if row["metric_valid"]:
                nonempty_poses += 1
                for key in POSE_METRIC_NAMES:
                    totals[key] += float(row[key])
                tp += row["tp"]
                fp += row["fp"]
                fn += row["fn"]
                tn += row["tn"]
                weighted_tp += row["weighted_tp"]
                weighted_gt += row["weighted_gt"]
                total_pred += row["avg_pred_count"]
                total_candidate += row["candidate_count"]
                total_gt += row["gt_count"]

        aux_values["base_logit"].extend(base_logits.tolist())
        aux_values["final_logit"].extend(final_logits.tolist())
        aux_values["suppression"].extend(suppression.tolist())
        aux_values["selected_proxy_abs"].extend(np.abs(selected_proxy).reshape(-1).tolist())
        aux_values["gate_entropy"].extend(gate_entropy.tolist())

    denominator = max(1.0, total_candidate)
    aggregate_recall = tp / max(1.0, tp + fn)
    aggregate_specificity = tn / max(1.0, tn + fp)
    summary = {
        "name": name,
        "threshold": float(threshold),
        "pose_count": int(poses_seen),
        "nonempty_pose_count": int(nonempty_poses),
        "empty_candidate_pose_count": int(poses_seen - nonempty_poses),
        "pose_metrics": {
            key: float(value / max(1, nonempty_poses)) for key, value in totals.items()
        },
        "aggregate": {
            "precision": float(tp / max(1.0, tp + fp)),
            "recall": float(aggregate_recall),
            "f1": float(2.0 * tp / max(1.0, 2.0 * tp + fp + fn)),
            "jaccard": float(tp / max(1.0, tp + fp + fn)),
            "weighted_recall": float(weighted_tp / weighted_gt) if weighted_gt > 0.0 else 0.0,
            "accuracy": float((tp + tn) / denominator),
            "balanced_accuracy": float((aggregate_recall + aggregate_specificity) * 0.5),
            "specificity": float(aggregate_specificity),
            "useful_cull": float(tn / denominator),
            "bad_cull": float(fn / denominator),
            "avg_pred_count": float(total_pred / max(1, nonempty_poses)),
            "avg_candidate_count": float(total_candidate / max(1, nonempty_poses)),
            "avg_gt_count": float(total_gt / max(1, nonempty_poses)),
            "pred_over_candidate": float(total_pred / max(1.0, total_candidate)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "weighted_tp": float(weighted_tp),
            "weighted_gt": float(weighted_gt),
        },
        "diagnostics": {},
    }
    for key, values in aux_values.items():
        summary["diagnostics"][key] = _stats(values)
    return summary, per_pose


def _index_rows(rows: list[dict[str, Any]], variant_name: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        pose_id = int(row["pose_index"])
        if pose_id in indexed:
            raise ValueError(f"variant {variant_name} emitted duplicate pose index {pose_id}")
        indexed[pose_id] = row
    return indexed


def paired_bootstrap(
    rows_by_name: dict[str, list[dict[str, Any]]],
    baseline_name: str,
    seed: int,
    replicates: int = 10000,
) -> dict[str, Any]:
    if baseline_name not in rows_by_name:
        raise ValueError(f"missing {baseline_name} rows for paired bootstrap")
    if replicates <= 0:
        raise ValueError(f"bootstrap replicates must be positive, got {replicates}")
    base = _index_rows(rows_by_name[baseline_name], baseline_name)
    metrics = (
        "useful_cull",
        "bad_cull",
        "weighted_recall",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "specificity",
        "avg_pred_count",
    )
    out: dict[str, Any] = {}
    rng = np.random.default_rng(int(seed))
    for name, rows in rows_by_name.items():
        if name == baseline_name:
            continue
        current = _index_rows(rows, name)
        if set(current) != set(base):
            missing = sorted(set(base) - set(current))[:8]
            extra = sorted(set(current) - set(base))[:8]
            raise ValueError(
                f"cannot pair {name} with baseline: pose sets differ; missing={missing}, extra={extra}"
            )
        for pose_id in base:
            if (
                base[pose_id]["candidate_id_sha256"] != current[pose_id]["candidate_id_sha256"]
                or base[pose_id]["candidate_count"] != current[pose_id]["candidate_count"]
            ):
                raise ValueError(f"candidate set changed between baseline and {name} at pose {pose_id}")
        paired = [
            (base[pose_id], current[pose_id])
            for pose_id in sorted(base)
            if base[pose_id]["metric_valid"] and current[pose_id]["metric_valid"]
        ]
        variant_out: dict[str, Any] = {
            "paired_pose_count": len(paired),
            "all_pose_count": len(base),
            "excluded_empty_pose_count": len(base) - len(paired),
            "bootstrap_replicates": int(replicates),
        }
        if not paired:
            out[name] = variant_out
            continue
        diffs_by_metric = {
            metric: np.asarray(
                [right[metric] - left[metric] for left, right in paired], dtype=np.float64
            )
            for metric in metrics
        }
        # Keep the same resampled pose indices for every metric, but generate
        # them in chunks so a large split does not allocate a 10,000-by-N
        # matrix in one shot.
        chunk_size = max(1, min(int(replicates), 4_000_000 // max(1, len(paired))))
        bootstrap_means = {metric: np.empty((int(replicates),), dtype=np.float64) for metric in metrics}
        for start in range(0, int(replicates), chunk_size):
            end = min(int(replicates), start + chunk_size)
            sample_indices = rng.integers(
                0,
                len(paired),
                size=(end - start, len(paired)),
                dtype=np.int32,
            )
            for metric in metrics:
                bootstrap_means[metric][start:end] = diffs_by_metric[metric][sample_indices].mean(axis=1)
        variant_out["bootstrap_chunk_size"] = int(chunk_size)
        for metric in metrics:
            diffs = diffs_by_metric[metric]
            boot = bootstrap_means[metric]
            variant_out[metric] = {
                "mean_delta": float(diffs.mean()),
                "ci95": [
                    float(np.quantile(boot, 0.025)),
                    float(np.quantile(boot, 0.975)),
                ],
            }
        out[name] = variant_out
    return out


class _FixtureModel:
    """Minimal model surface used by ``--self-test``; no learned weights."""

    def __init__(self) -> None:
        self.geo_dim = 2
        self.context_dim = 2
        self.direction_bins = 2
        self.depth_shells = 2
        self.proxy_dim = 1
        self.runtime_feature_dim = 2 + 2 + 2 * 2 * 1

    def _split_runtime(self, runtime: torch.Tensor):
        return (
            runtime[:, :2],
            runtime[:, 2:4],
            runtime[:, 4:].reshape(-1, 2, 2, 1),
        )


def run_fixture_self_test() -> dict[str, Any]:
    model = _FixtureModel()
    base = torch.arange(3 * model.runtime_feature_dim, dtype=torch.float32).reshape(
        3, model.runtime_feature_dim
    )
    outputs: dict[str, torch.Tensor] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(INTERVENTIONS):
        outputs[name], manifests[name] = apply_runtime_intervention(base, model, name, 100 + index)
        assert outputs[name].shape == base.shape
        assert torch.isfinite(outputs[name]).all()
    proxy_base = base[:, 4:].reshape(3, 2, 2, 1)
    context_base = base[:, 2:4]
    assert torch.equal(outputs["proxy_zero"][:, 4:], torch.zeros_like(base[:, 4:]))
    proxy_mean = proxy_base.mean(dim=1, keepdim=True).expand_as(proxy_base).reshape(3, -1)
    assert torch.equal(outputs["proxy_direction_mean"][:, 4:], proxy_mean)
    assert torch.equal(
        outputs["proxy_direction_roll"][:, 4:], torch.roll(proxy_base, 1, dims=1).reshape(3, -1)
    )
    assert torch.equal(outputs["context_zero"][:, 2:4], torch.zeros_like(context_base))
    for name in ("proxy_cross_instance_permutation", "context_cross_instance_permutation"):
        assert manifests[name]["permutation"]["fixedPointCount"] < base.shape[0]
    target = np.asarray([1, 0, 1, 0], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.1, 0.1], dtype=np.float32)
    row = _metric_row(target, scores, np.asarray([2, 1, 1, 1], dtype=np.float32), 0.5)
    assert row["tp"] == 1.0 and row["fp"] == 1.0 and row["fn"] == 1.0 and row["tn"] == 1.0
    assert abs(row["weighted_recall"] - (2.0 / 3.0)) < 1e-7
    assert abs(row["accuracy"] - 0.5) < 1e-7
    uniform_entropy = normalized_gate_entropy(torch.zeros((2, 4), dtype=torch.float32))
    assert torch.allclose(uniform_entropy, torch.ones_like(uniform_entropy))
    summary = {
        "interventions": list(outputs),
        "alias": canonical_intervention_name("proxy_instance_permutation"),
        "metricKeys": sorted(row),
        "uniformGateEntropy": uniform_entropy.tolist(),
    }
    return summary


def _require_runtime_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("checkpoint", "runtime_features", "dataset_dir", "runtime_meta", "output")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join('--' + name.replace('_', '-') for name in missing)}")


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps({"status": "passed", **run_fixture_self_test()}, ensure_ascii=False, indent=2))
        return
    _require_runtime_args(args)
    if args.split == "test":
        if not args.allow_test_diagnostic:
            raise ValueError("M3 interventions must use validation or calibration; test is sealed by default.")
        if args.threshold is not None:
            raise ValueError(
                "M3 test diagnostics cannot accept --threshold; use the frozen calibration threshold from eval_summary.json."
            )
    names, aliases_used = normalize_intervention_names(args.interventions)
    if args.poses_per_batch <= 0:
        raise ValueError(f"poses-per-batch must be positive, got {args.poses_per_batch}")

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    runtime_meta = json.loads(args.runtime_meta.read_text(encoding="utf-8"))
    model, world_aabbs = build_model(checkpoint, runtime_meta, device)
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=world_aabbs.shape[0])
    split = dataset.split(args.split)
    threshold, threshold_source = load_threshold(args)
    runtime_dim = int(checkpoint["config"]["runtimeFeatureDim"])
    runtime_np = np.fromfile(args.runtime_features, dtype=np.float16)
    expected = int(world_aabbs.shape[0]) * runtime_dim
    if runtime_np.size != expected:
        raise ValueError(f"Runtime feature table has {runtime_np.size} values; expected {expected}.")
    if not np.isfinite(runtime_np).all():
        raise FloatingPointError("runtime feature table contains non-finite values")
    base_features = torch.from_numpy(runtime_np.astype(np.float32)).reshape(
        world_aabbs.shape[0], runtime_dim
    ).to(device)

    pose_plan = build_pose_plan(split, args.poses_per_batch, args.seed + 1000)
    summaries: dict[str, Any] = {}
    per_pose_rows: dict[str, list[dict[str, Any]]] = {}
    intervention_manifests: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        variant_features, intervention_meta = apply_runtime_intervention(
            base_features, model, name, args.seed + index * 7919
        )
        summary, rows = evaluate_variant(
            model,
            variant_features,
            split,
            world_aabbs,
            threshold,
            args.poses_per_batch,
            args.seed + 1000,
            name,
            pose_plan=pose_plan,
        )
        summaries[name] = summary
        per_pose_rows[name] = rows
        intervention_manifests.append(intervention_meta)
        del variant_features

    output = {
        "schema": "pvs-proxy-intervention-v2",
        "checkpoint": str(args.checkpoint),
        "checkpointSha256": sha256_file(args.checkpoint),
        "runtimeFeatures": str(args.runtime_features),
        "runtimeFeaturesSha256": sha256_file(args.runtime_features),
        "datasetDir": str(args.dataset_dir),
        "runtimeMeta": str(args.runtime_meta),
        "split": args.split,
        "threshold": float(threshold),
        "thresholdSource": threshold_source,
        "seed": int(args.seed),
        "device": str(device),
        "interventions": summaries,
        "interventionManifest": intervention_manifests,
        "interventionAliasesUsed": aliases_used,
        "posePlan": {
            "posesPerBatch": int(args.poses_per_batch),
            "batchCount": len(pose_plan),
            "poseCount": int(sum(len(item) for item in pose_plan)),
            "sha256": pose_plan_digest(pose_plan),
            "includeEmpty": True,
            "candidateSource": "stored frustum_ids.bin; no GT union and no candidate cap",
        },
        "pairedBootstrap": paired_bootstrap(
            per_pose_rows,
            "baseline",
            args.seed + 2000,
            replicates=args.bootstrap_replicates,
        ),
        "perPose": per_pose_rows,
        "semantics": {
            "proxy": "fixed directional occlusion proxy slices selected by the current ray gate",
            "context": "fixed offline context feature shared by all directions",
            "baseLogit": "visibility head logit before the explicit non-negative suppression head",
            "suppression": "softplus inhibition-head output clamped to [0, 4], subtracted from baseLogit",
            "gateEntropy": "normalized Shannon entropy of the direction-depth proxy gate; 0 is concentrated and 1 is uniform",
            "threshold": "one frozen calibration threshold; no test threshold scan or override",
            "metrics": "per-pose metrics are computed on the stored back-camera candidate set; weighted recall uses visible_weights, not pixel coverage",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "split": args.split,
                "threshold": threshold,
                "variants": list(summaries),
                "poseCount": output["posePlan"]["poseCount"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
