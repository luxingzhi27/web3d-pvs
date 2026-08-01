#!/usr/bin/env python3
"""Evaluate zero-shot transfer of the shared PVS query model.

The source checkpoint contributes learnable parameters only.  All scene-sized
buffers and all fixed instance features are rebuilt from the target scene.
Calibration is target-scene-only; the target test split is evaluated once at
the resulting frozen threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.glb_points import load_glb_points  # noqa: E402
from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from current_pvs_utils import evaluate_thresholds  # noqa: E402
from directional_occlusion_proxy_encoder_model import (  # noqa: E402
    DirectionalOcclusionProxyEncoderPVSModel,
    load_directional_occlusion_evidence,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from train_directional_occlusion_proxy_encoder import (  # noqa: E402
    build_protocol_splits,
    evaluate_calibration_and_validation,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config")
    args = checkpoint.get("args")
    if not isinstance(config, dict) or not isinstance(args, dict):
        raise ValueError("source checkpoint must contain config and args")
    required = (
        "geoDim",
        "contextDim",
        "proxyDim",
        "directionBins",
        "depthShells",
        "runtimeFeatureDim",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"source checkpoint config is missing {missing}")
    return config


def _build_target_model(
    checkpoint: dict[str, Any],
    runtime_meta_path: Path,
    evidence_dir: Path,
    device: torch.device,
) -> tuple[DirectionalOcclusionProxyEncoderPVSModel, np.ndarray, np.ndarray, dict[str, Any]]:
    config = _source_config(checkpoint)
    args = checkpoint["args"]
    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    source_direction = int(config["directionBins"])
    source_shells = int(config["depthShells"])
    source_k = int(config.get("sourceK", 8))
    source_ids, source_scores, strength, evidence_meta = load_directional_occlusion_evidence(
        evidence_dir, world_aabbs.shape[0]
    )
    target_shape = (source_direction, source_shells, source_k)
    actual_shape = (
        int(evidence_meta["directionBins"]),
        int(evidence_meta["depthShells"]),
        int(evidence_meta["sourceK"]),
    )
    if actual_shape != target_shape:
        raise ValueError(
            "source and target directional evidence layouts differ: "
            f"source={target_shape}, target={actual_shape}"
        )

    num_glbs = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
    model = DirectionalOcclusionProxyEncoderPVSModel(
        num_instances=int(world_aabbs.shape[0]),
        num_glbs=num_glbs,
        geo_dim=int(config["geoDim"]),
        context_dim=int(config["contextDim"]),
        proxy_dim=int(config["proxyDim"]),
        direction_bins=source_direction,
        depth_shells=source_shells,
        source_k=source_k,
        point_hidden_dim=int(args.get("point_hidden_dim", 160)),
        pointnetpp_centers=int(args.get("pointnetpp_centers", 24)),
        pointnetpp_neighbors=int(args.get("pointnetpp_neighbors", 12)),
        graph_hidden_dim=int(args.get("graph_hidden_dim", 160)),
        graph_message_dim=int(args.get("graph_message_dim", 96)),
        ray_fourier_bands=int(args.get("ray_fourier_bands", 10)),
        ray_scalar_fourier_bands=int(args.get("ray_scalar_fourier_bands", 4)),
        camera_location_dim=int(config.get("cameraLocationFeatureDim", args.get("camera_location_dim", 0))),
        mlp_hidden=int(args.get("mlp_hidden", 128)),
        interaction_dim=int(args.get("interaction_dim", 64)),
        scene_size_m=scene_size.tolist(),
        runtime_feature_ablation=str(config.get("runtimeFeatureAblation", "none")),
        use_explicit_inhibition=bool(config.get("usesExplicitInhibition", True)),
    ).to(device)

    target_parameters = dict(model.named_parameters())
    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("source checkpoint has no model state")
    transferable: dict[str, torch.Tensor] = {}
    mismatched_parameters: list[str] = []
    for name, value in source_state.items():
        if name not in target_parameters:
            continue
        target = target_parameters[name]
        if tuple(value.shape) != tuple(target.shape):
            mismatched_parameters.append(name)
            continue
        transferable[name] = value
    missing_parameters = sorted(set(target_parameters) - set(transferable))
    if mismatched_parameters or missing_parameters:
        raise ValueError(
            "shared learnable parameter schema is incompatible: "
            f"mismatched={mismatched_parameters[:8]}, missing={missing_parameters[:8]}"
        )
    model.load_state_dict(transferable, strict=False)
    model.set_scene_bounds(torch.from_numpy(scene_min).to(device), torch.from_numpy(scene_size).to(device))
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    model.set_evidence(
        torch.from_numpy(source_ids).to(device),
        torch.from_numpy(source_scores).to(device),
        torch.from_numpy(strength).to(device),
    )
    model.eval()
    return model, world_aabbs, instance_to_glb, {
        "targetRuntimeMeta": str(runtime_meta_path.resolve()),
        "targetEvidence": str(evidence_dir.resolve()),
        "targetEvidenceMeta": evidence_meta,
        "transferredParameterCount": len(transferable),
        "targetInstanceCount": int(world_aabbs.shape[0]),
        "targetGlbCount": num_glbs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--target-dataset-dir", required=True, type=Path)
    parser.add_argument("--target-evidence-dir", required=True, type=Path)
    parser.add_argument("--target-glb-points", required=True, type=Path)
    parser.add_argument("--target-runtime-meta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--pose-set-batch-size", type=int, default=2)
    parser.add_argument("--feature-export-batch-size", type=int, default=512)
    parser.add_argument("--glb-train-points", type=int, default=64)
    parser.add_argument("--target-weighted-recall", type=float, default=0.99)
    parser.add_argument("--calibration-point-floor", type=float, default=0.9925)
    parser.add_argument("--calibration-lcb-floor", type=float, default=0.99)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_checkpoint = args.source_checkpoint.resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse transfer output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    checkpoint = torch.load(source_checkpoint, map_location="cpu")
    model, world_aabbs, instance_to_glb, transfer_meta = _build_target_model(
        checkpoint,
        args.target_runtime_meta.resolve(),
        args.target_evidence_dir.resolve(),
        device,
    )
    dataset = PoseCSRDataset(args.target_dataset_dir.resolve(), num_instances=world_aabbs.shape[0])
    if not {"train", "validation", "calibration", "test"}.issubset(dataset.split_ids):
        raise ValueError("target dataset must provide native train/validation/calibration/test splits")
    _train, validation, calibration, test, protocol = build_protocol_splits(
        dataset,
        seed=args.seed,
        calibration_fraction=0.1,
    )
    point_np, glb_meta = load_glb_points(args.target_glb_points.resolve(), max_points=args.glb_train_points)
    required_glbs = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
    if int(point_np.shape[0]) < required_glbs:
        raise ValueError(f"target GLB point cache has {point_np.shape[0]} rows, requires {required_glbs}")
    with torch.no_grad():
        point_cache = torch.from_numpy(point_np).to(device)
        _geo, _context, _proxy, runtime = model.compute_all_runtime_features(
            point_cache,
            batch_size=args.feature_export_batch_size,
        )
    runtime_path = output_dir / "instance_runtime_features_fp16.bin"
    runtime.numpy().astype(np.float16).tofile(runtime_path)
    runtime_device = runtime.to(device)
    feature_meta = {
        "schema": "directional-occlusion-proxy-features-v1",
        "semantics": "target-scene fixed features rebuilt with transferred offline encoder parameters",
        "numInstances": int(runtime.shape[0]),
        "runtimeFeatureDim": int(runtime.shape[1]),
        "dtype": "float16",
        "glbPoints": glb_meta,
        "files": {"instanceRuntime": runtime_path.name},
    }
    write_json(output_dir / "instance_features_meta.json", feature_meta)

    calibration_workpoint, validation_workpoint, calibration_rows, calibration_summary = evaluate_calibration_and_validation(
        model,
        validation,
        calibration,
        runtime_device,
        world_aabbs,
        device,
        poses_per_batch=max(1, min(int(args.pose_set_batch_size), 4)),
        max_candidates_per_pose=0,
        seed=args.seed + 100,
        target_weighted_recall=args.target_weighted_recall,
        calibration_point_floor=args.calibration_point_floor,
        calibration_lcb_floor=args.calibration_lcb_floor,
        calibration_bootstrap_replicates=args.calibration_bootstrap_replicates,
        calibration_bootstrap_confidence=0.95,
        allow_candidate_visible_union=False,
        require_safe_workpoint=False,
    )
    calibration_payload = {
        "schema": "neuralstreamweb3d-cross-scene-transfer-calibration-v1",
        "protocol": "target_calibration_only_pre_test",
        "sourceCheckpoint": str(source_checkpoint),
        "sourceCheckpointSha256": sha256_file(source_checkpoint),
        "targetDataset": str(args.target_dataset_dir.resolve()),
        "targetProtocolSplit": protocol,
        "targetWeightedRecall": float(args.target_weighted_recall),
        "calibration": calibration_summary,
        "validationAtFrozenThreshold": validation_workpoint,
        "transfer": transfer_meta,
    }
    write_json(output_dir / "calibration_ready_summary.json", calibration_payload)

    if calibration_workpoint is None:
        final = dict(calibration_payload)
        final["status"] = "no_safe_target_calibration_workpoint"
        final["testEvaluationCount"] = 0
        write_json(output_dir / "summary.json", final)
        print(json.dumps({"status": final["status"], "outputDir": str(output_dir)}, ensure_ascii=False, indent=2))
        return

    threshold = float(calibration_workpoint["threshold"])
    manifest = {
        "schema": "neuralstreamweb3d-cross-scene-transfer-frozen-test-v1",
        "protocol": "target_calibration_then_one_shot_test",
        "testEvaluationCountBeforeThisRun": 0,
        "threshold": threshold,
        "thresholdSource": "target calibration only",
        "sourceCheckpoint": str(source_checkpoint),
        "sourceCheckpointSha256": sha256_file(source_checkpoint),
        "runtimeFeatures": str(runtime_path),
        "runtimeFeaturesSha256": sha256_file(runtime_path),
        "targetDataset": str(args.target_dataset_dir.resolve()),
        "targetTestPoseCount": int(test.pose_indices.size),
    }
    write_json(output_dir / "frozen_test_manifest.json", manifest)
    test_row = evaluate_thresholds(
        model,
        test,
        runtime_device,
        world_aabbs,
        device,
        poses_per_batch=max(1, min(int(args.pose_set_batch_size), 4)),
        max_steps=None,
        max_candidates_per_pose=0,
        seed=args.seed + 999,
        thresholds=np.asarray([threshold], dtype=np.float32),
        allow_candidate_visible_union=False,
    )[0]
    if int(test_row.get("eval_pose_count", -1)) != int(test.pose_indices.size):
        raise RuntimeError(
            f"target transfer test evaluated {test_row.get('eval_pose_count')} poses; "
            f"expected {test.pose_indices.size}"
        )
    final = dict(calibration_payload)
    final.update(
        {
            "status": "completed",
            "frozenThreshold": threshold,
            "test": test_row,
            "testEvaluationCount": 1,
            "frozenTestManifest": str((output_dir / "frozen_test_manifest.json").resolve()),
        }
    )
    write_json(output_dir / "summary.json", final)
    print(json.dumps({"status": final["status"], "outputDir": str(output_dir), "threshold": threshold}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
