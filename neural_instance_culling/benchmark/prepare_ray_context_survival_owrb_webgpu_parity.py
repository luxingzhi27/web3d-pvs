#!/usr/bin/env python3
"""Create deterministic PyTorch reference cases for the OWRB WebGPU probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from ray_context_survival_owrb_model import RayContextSurvivalOWRBModel  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_model(checkpoint: dict[str, Any], runtime_meta_path: Path) -> tuple[RayContextSurvivalOWRBModel, np.ndarray]:
    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    config = checkpoint["config"]
    _scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    supplement_config = {
        "contextQueryDim": int(config.get("contextQueryDim", 8)),
        "rayMode": str(config.get("rayMode", "direct9")),
        "survivalParameterization": str(config.get("survivalParameterization", "monotone")),
    }
    if supplement_config != {
        "contextQueryDim": 8,
        "rayMode": "direct9",
        "survivalParameterization": "monotone",
    }:
        raise ValueError(
            "WebGPU parity is restricted to the deployable default schema; "
            f"supplement configuration {supplement_config} is offline-only"
        )
    # The browser bundle deliberately stores AABBs as FP16.  The parity
    # reference must run against the same quantized values; using the full
    # precision runtime metadata here makes ray direction, distance, angular
    # size and the survival query disagree with the actual browser input.
    browser_world_aabbs = np.asarray(world_aabbs, dtype=np.float16).astype(np.float32)
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
        context_query_dim=supplement_config["contextQueryDim"],
        ray_mode=supplement_config["rayMode"],
        survival_parameterization=supplement_config["survivalParameterization"],
        scene_size_m=scene_size.tolist(),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    # Checkpoint buffers contain the training-time full-precision metadata;
    # replace them after loading so the reference matches the exported FP16
    # AABB storage buffer rather than silently restoring the checkpoint copy.
    model.set_instance_world_aabbs(torch.from_numpy(browser_world_aabbs))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb))
    return model.eval(), browser_world_aabbs


@torch.no_grad()
def prepare(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    runtime_meta_path = Path(args.runtime_meta).resolve()
    runtime_path = Path(args.runtime_features).resolve() if args.runtime_features else checkpoint_path.with_name("best_instance_runtime_features_fp16.bin")
    if not runtime_path.is_file():
        runtime_path = checkpoint_path.with_name("instance_runtime_features_fp16.bin")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, world_aabbs = _build_model(checkpoint, runtime_meta_path)
    values = np.fromfile(runtime_path, dtype=np.float16)
    expected = int(world_aabbs.shape[0]) * int(checkpoint["config"]["runtimeFeatureDim"])
    if values.size != expected:
        raise ValueError(f"runtime table has {values.size} values; expected {expected}")
    runtime = torch.from_numpy(values.astype(np.float32)).reshape(world_aabbs.shape[0], -1)
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=world_aabbs.shape[0])
    split = dataset.split("validation")
    rng = np.random.default_rng(int(args.seed))
    cases: list[dict[str, Any]] = []
    for pose_index in split.pose_indices.tolist():
        batch = split.build_pose_set_batch(
            np.asarray([int(pose_index)], dtype=np.int64),
            world_aabbs,
            rng,
            max_candidates_per_pose=0,
            allow_candidate_visible_union=False,
            include_empty=True,
        )
        start, end = int(batch["pose_offsets"][0]), int(batch["pose_offsets"][1])
        candidate_ids = np.asarray(batch["instance"][start:end], dtype=np.int64)
        if candidate_ids.size == 0:
            continue
        if candidate_ids.size > int(args.max_candidates):
            # Deterministic coverage across the candidate list rather than a
            # prefix-only smoke, which can hide high-ID feature addressing bugs.
            selected = np.linspace(0, candidate_ids.size - 1, int(args.max_candidates), dtype=np.int64)
            candidate_ids = candidate_ids[selected]
            camera_world = np.asarray(batch["camera_world"][start:end], dtype=np.float32)[selected]
            camera_view = np.asarray(batch["camera_view"][start:end], dtype=np.float32)[selected]
        else:
            camera_world = np.asarray(batch["camera_world"][start:end], dtype=np.float32)
            camera_view = np.asarray(batch["camera_view"][start:end], dtype=np.float32)
        ids = torch.from_numpy(candidate_ids)
        camera_world_t = torch.from_numpy(camera_world)
        camera_view_t = torch.from_numpy(camera_view)
        logits, aux = model.compute_logits_with_aux(
            torch.zeros((candidate_ids.size, 3), dtype=torch.float32),
            camera_view_t,
            camera_world_t,
            ids,
            runtime_features=runtime,
        )
        visible = torch.sigmoid(logits).reshape(-1)
        utility_logits = aux["utility_logits"].reshape(-1)
        download_logits = aux["download_logits"].reshape(-1)
        cases.append({
            "caseId": len(cases),
            "poseIndex": int(pose_index),
            "candidateIds": candidate_ids.astype(np.uint32).tolist(),
            "camera": {
                "position": camera_world[0].astype(np.float32).tolist(),
                "cameraForward": camera_view[0, :3].astype(np.float32).tolist(),
                "cameraView": camera_view[0].astype(np.float32).tolist(),
            },
            "expected": {
                "visibilityLogits": logits.detach().cpu().numpy().reshape(-1).astype(np.float32).tolist(),
                "visibilityScores": visible.detach().cpu().numpy().astype(np.float32).tolist(),
                "utilityLogits": utility_logits.detach().cpu().numpy().astype(np.float32).tolist(),
                "utilityScores": torch.sigmoid(utility_logits).detach().cpu().numpy().astype(np.float32).tolist(),
                "downloadLogits": download_logits.detach().cpu().numpy().astype(np.float32).tolist(),
                "downloadScores": torch.sigmoid(download_logits).detach().cpu().numpy().astype(np.float32).tolist(),
            },
        })
        if len(cases) >= int(args.max_cases):
            break
    if not cases:
        raise ValueError("validation split produced no parity cases")
    return {
        "schema": "ray-context-survival-owrb-webgpu-parity-cases-v1",
        "checkpoint": str(checkpoint_path),
        "checkpointSha256": _sha256_file(checkpoint_path),
        "runtimeFeatures": str(runtime_path),
        "runtimeFeaturesSha256": _sha256_file(runtime_path),
        "runtimeFeatureDim": int(checkpoint["config"]["runtimeFeatureDim"]),
        "instanceAabbPrecision": "float16 export emulation",
        "modelInputFovYDeg": 66.0,
        "frontendRenderFovYDeg": 60.0,
        "candidatePolicy": "native back-camera CSR; no visible union",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-features", default=None)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.max_cases <= 0 or args.max_candidates <= 0:
        parser.error("--max-cases and --max-candidates must be positive")
    payload = prepare(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "output": str(output.resolve()), "caseCount": len(payload["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
