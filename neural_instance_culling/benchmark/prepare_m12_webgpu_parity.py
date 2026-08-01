#!/usr/bin/env python3
"""Prepare deterministic PyTorch references for the M12 WebGPU parity probe.

The browser probe consumes the same camera snapshots and candidate instance IDs
written here.  The FP32 row uses checkpoint weights and the exported feature
table decoded to float32.  The FP16 row first rounds floating tensors and fixed
features and the exported FP16 AABBs to IEEE half precision, then performs
the model arithmetic in float32, matching the frontend asset and WGSL's
``unpack2x16float`` path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from directional_occlusion_proxy_encoder_model import DirectionalOcclusionProxyEncoderPVSModel  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


def _quaternion_from_forward(forward: np.ndarray) -> list[float]:
    """Return a camera quaternion whose local -Z axis is ``forward``."""
    f = np.asarray(forward, dtype=np.float64)
    f /= max(float(np.linalg.norm(f)), 1e-12)
    up_seed = np.asarray([0.0, 0.0, 1.0] if abs(float(f[1])) > 0.98 else [0.0, 1.0, 0.0])
    right = np.cross(f, up_seed)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, f)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    # Local camera axes map to world: +X=right, +Y=up, +Z=-forward.
    m = np.column_stack([right, up, -f])
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q.astype(np.float32).tolist()


def _model_from_checkpoint(checkpoint: dict[str, Any], *, rounded_half: bool) -> DirectionalOcclusionProxyEncoderPVSModel:
    config = dict(checkpoint["config"])
    model = DirectionalOcclusionProxyEncoderPVSModel(
        num_instances=int(config["numInstances"]),
        num_glbs=int(config["numGlbs"]),
        geo_dim=int(config["geoDim"]),
        context_dim=int(config["contextDim"]),
        proxy_dim=int(config["proxyDim"]),
        direction_bins=int(config["directionBins"]),
        depth_shells=int(config["depthShells"]),
        source_k=int(config["sourceK"]),
        point_hidden_dim=int(config.get("pointHiddenDim", 160)),
        pointnetpp_centers=int(config.get("pointNetPPCenters", config.get("pointnetppCenters", 24))),
        pointnetpp_neighbors=int(config.get("pointNetPPNeighbors", config.get("pointnetppNeighbors", 12))),
        graph_hidden_dim=int(config.get("graphHiddenDim", 160)),
        graph_message_dim=int(config.get("graphMessageDim", 96)),
        ray_fourier_bands=int(config.get("rayFourierBands", 10)),
        ray_scalar_fourier_bands=int(config.get("rayScalarFourierBands", 4)),
        camera_location_dim=int(config.get("cameraLocationFeatureDim", 0)),
        mlp_hidden=int(config.get("mlpHiddenDim", 128)),
        interaction_dim=int(config.get("visibilityInteractionDim", 64)),
        scene_size_m=list(config.get("sceneSizeM", [1.0, 1.0, 1.0])),
        runtime_feature_ablation=str(config.get("runtimeFeatureAblation", "none")),
        use_explicit_inhibition=bool(config.get("usesExplicitInhibition", True)),
    )
    state = checkpoint["model"]
    if rounded_half:
        state = {
            key: (value.to(dtype=torch.float16).to(dtype=torch.float32) if torch.is_floating_point(value) else value)
            for key, value in state.items()
        }
    model.load_state_dict(state, strict=True)
    if rounded_half:
        # The frontend packs instance_world_aabbs into the same FP16 asset as
        # the runtime feature table. Round this buffer too, otherwise the
        # reference uses higher-precision camera-relative geometry than WGSL.
        with torch.no_grad():
            model.instance_world_aabbs.copy_(
                model.instance_world_aabbs.to(dtype=torch.float16).to(dtype=torch.float32)
            )
    model.eval()
    return model


def _select_candidates(dataset: PoseCSRDataset, pose_index: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.asarray(dataset.frustum_slice(pose_index), dtype=np.uint32)
    candidates = np.unique(candidates)
    if limit <= 0 or candidates.size <= limit:
        return candidates
    visible, _weights = dataset.visible_slice(pose_index)
    visible = np.unique(np.asarray(visible, dtype=np.uint32))
    visible = visible[np.isin(visible, candidates)]
    if visible.size > limit:
        raise ValueError(f"M12 candidate limit {limit} is smaller than visible set {visible.size} at pose {pose_index}")
    negative = candidates[~np.isin(candidates, visible)]
    take = limit - visible.size
    if negative.size > take:
        negative = rng.choice(negative, size=take, replace=False)
    return np.sort(np.concatenate([visible, negative]).astype(np.uint32))


def _forward_batch(model, cases: list[dict[str, Any]], runtime_features: np.ndarray) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    runtime = torch.from_numpy(runtime_features).to(device=device, dtype=torch.float32)
    output: list[dict[str, Any]] = []
    with torch.no_grad():
        for case in cases:
            ids = torch.from_numpy(np.asarray(case["candidateIds"], dtype=np.int64)).to(device)
            camera = case["camera"]
            camera_norm = torch.from_numpy(np.asarray(camera["cameraNorm"], dtype=np.float32)).expand(ids.numel(), -1).to(device)
            camera_world = torch.from_numpy(np.asarray(camera["position"], dtype=np.float32)).expand(ids.numel(), -1).to(device)
            camera_view = torch.from_numpy(np.asarray(camera["cameraView"], dtype=np.float32)).expand(ids.numel(), -1).to(device)
            logits, aux = model.compute_logits_with_aux(
                camera_norm,
                camera_view,
                camera_world,
                ids,
                runtime_features=runtime,
            )
            output.append({
                "caseId": case["caseId"],
                "visibilityLogit": logits.detach().float().cpu().reshape(-1).tolist(),
                "visibilityProbability": torch.sigmoid(logits).detach().float().cpu().reshape(-1).tolist(),
                "downloadLogit": aux["download_logits"].detach().float().cpu().reshape(-1).tolist(),
                "downloadProbability": torch.sigmoid(aux["download_logits"]).detach().float().cpu().reshape(-1).tolist(),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare M12 PyTorch references and browser probe cases.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-count", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=1024)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint["config"])
    num_instances = int(config["numInstances"])
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=num_instances)
    if args.split not in dataset.split_ids:
        raise ValueError(f"Unknown split {args.split!r}; available={sorted(dataset.split_ids)}")
    split = dataset.split(args.split)
    rng = np.random.default_rng(args.seed)
    eligible = split.pose_indices[dataset.frustum_counts[split.pose_indices] > 0]
    if eligible.size == 0:
        raise ValueError("M12 split has no candidate-bearing poses")
    selected = np.linspace(0, eligible.size - 1, num=max(1, min(args.case_count, eligible.size)), dtype=np.int64)
    pose_indices = eligible[selected]
    cases: list[dict[str, Any]] = []
    for case_id, pose_index in enumerate(pose_indices.tolist()):
        pose = dataset.poses[int(pose_index)]
        forward = np.asarray(pose["camera_forward"], dtype=np.float32)
        camera_view = dataset.camera_view(int(pose_index))
        candidates = _select_candidates(dataset, int(pose_index), int(args.candidate_limit), rng)
        if candidates.size == 0:
            continue
        visible_ids, visible_weights = dataset.visible_slice(int(pose_index))
        tan_x = float(camera_view[3])
        tan_y = float(camera_view[4])
        cases.append({
            "caseId": int(case_id),
            "poseIndex": int(pose_index),
            "candidateIds": candidates.astype(int).tolist(),
            "visibleIds": np.asarray(visible_ids, dtype=np.uint32).tolist(),
            "visibleWeights": np.asarray(visible_weights, dtype=np.float32).tolist(),
            "camera": {
                "position": np.asarray(pose["camera_world"], dtype=np.float32).tolist(),
                "cameraNorm": np.asarray(pose["camera_norm"], dtype=np.float32).tolist(),
                "cameraForward": forward.tolist(),
                "cameraView": camera_view.astype(np.float32).tolist(),
                "quaternion": _quaternion_from_forward(forward),
                "fov": 66.0,
                "aspect": tan_x / max(tan_y, 1e-8),
                "near": 0.1,
                "far": 20000.0,
            },
        })
    if not cases:
        raise ValueError("No M12 cases were constructed")

    runtime_fp16 = np.fromfile(args.runtime_features, dtype=np.float16)
    expected = num_instances * int(config["runtimeFeatureDim"])
    if runtime_fp16.size != expected:
        raise ValueError(f"Runtime feature size {runtime_fp16.size} != expected {expected}")
    runtime_fp16 = runtime_fp16.reshape(num_instances, int(config["runtimeFeatureDim"]))
    runtime_fp32 = runtime_fp16.astype(np.float32)
    reference_fp32 = _forward_batch(_model_from_checkpoint(checkpoint, rounded_half=False), cases, runtime_fp32)
    reference_fp16 = _forward_batch(_model_from_checkpoint(checkpoint, rounded_half=True), cases, runtime_fp16.astype(np.float32))

    cases_payload = {
        "schema": "m12-webgpu-parity-cases-v1",
        "scene": str(Path(args.dataset_dir).name),
        "datasetDir": str(Path(args.dataset_dir).as_posix()),
        "checkpoint": str(Path(args.checkpoint).as_posix()),
        "runtimeFeatures": str(Path(args.runtime_features).as_posix()),
        "split": args.split,
        "seed": int(args.seed),
        "caseCount": len(cases),
        "candidateLimit": int(args.candidate_limit),
        "threshold": float(checkpoint.get("best", {}).get("threshold", 0.02)),
        "cases": cases,
    }
    (output_dir / "m12_cases.json").write_text(json.dumps(cases_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    reference = {
        "schema": "m12-pytorch-reference-v1",
        "precisionDefinition": "FP32 weights/features versus FP16-rounded weights/features with float32 arithmetic",
        "cases": reference_fp32,
        "fp16RoundedCases": reference_fp16,
    }
    (output_dir / "m12_torch_reference.json").write_text(json.dumps(reference, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "cases": str(output_dir / "m12_cases.json"),
        "reference": str(output_dir / "m12_torch_reference.json"),
        "caseCount": len(cases),
        "candidateCount": int(sum(len(case["candidateIds"]) for case in cases)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
