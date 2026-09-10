#!/usr/bin/env python3
"""Export the fixed 96D instance geometry table used by the V4 model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.glb_points import load_glb_points  # noqa: E402
from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from fixed_geometry_encoder import InstancePointNetPPGeoEncoder  # noqa: E402


GEO_DIM = 96
POINT_FEATURE_DIM = 16


def _point_features(
    instance_ids: torch.Tensor,
    glb_points: torch.Tensor,
    instance_to_glb: torch.Tensor,
    world_aabbs: torch.Tensor,
    scene_min: torch.Tensor,
    scene_size: torch.Tensor,
) -> torch.Tensor:
    local = glb_points[instance_to_glb[instance_ids]]
    bounds = world_aabbs[instance_ids]
    minimum = bounds[:, :3]
    maximum = bounds[:, 3:]
    center = (minimum + maximum) * 0.5
    size = (maximum - minimum).clamp_min(1e-4)
    safe_scene_size = scene_size.clamp_min(1e-6)
    center_norm = ((center - scene_min) / safe_scene_size).clamp(0.0, 1.0)
    size_norm = (size / safe_scene_size).clamp(0.0, 4.0)
    uniform_scale = size.amax(dim=-1, keepdim=True)
    world = center[:, None, :] + local * uniform_scale[:, None, :]
    world_norm = ((world - scene_min) / safe_scene_size).clamp(0.0, 1.0)
    scale_ratio = size / uniform_scale.clamp_min(1e-6)
    return torch.cat(
        [
            local,
            world_norm,
            center_norm[:, None, :].expand_as(local),
            size_norm[:, None, :].expand_as(local),
            scale_ratio[:, None, :].expand_as(local),
            torch.linalg.vector_norm(local, dim=-1, keepdim=True),
        ],
        dim=-1,
    )


def _load_encoder(
    checkpoint_path: Path, device: torch.device
) -> tuple[InstancePointNetPPGeoEncoder, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("geometry source checkpoint must be a mapping")
    config = checkpoint.get("config") or checkpoint.get("modelConfig") or {}
    arguments = checkpoint.get("args") or {}
    if not isinstance(config, Mapping) or not isinstance(arguments, Mapping):
        raise ValueError("geometry source checkpoint configuration is invalid")
    state = checkpoint.get("model") or checkpoint.get("modelState")
    if not isinstance(state, Mapping):
        raise ValueError("geometry source checkpoint has no model state")
    encoder_state = {
        str(key).removeprefix("geo_encoder."): value
        for key, value in state.items()
        if str(key).startswith("geo_encoder.")
    }
    if not encoder_state:
        raise ValueError("geometry source checkpoint has no geo_encoder weights")
    encoder = InstancePointNetPPGeoEncoder(
        point_feature_dim=POINT_FEATURE_DIM,
        hidden_dim=int(arguments.get("point_hidden_dim", 160)),
        out_dim=int(config.get("geoDim", GEO_DIM)),
        center_count=int(arguments.get("pointnetpp_centers", 24)),
        neighbor_count=int(arguments.get("pointnetpp_neighbors", 12)),
    ).to(device)
    if encoder.out_dim != GEO_DIM:
        raise ValueError(f"V4 requires {GEO_DIM} geometry features")
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()
    return encoder, {
        "hiddenDim": int(arguments.get("point_hidden_dim", 160)),
        "centerCount": int(arguments.get("pointnetpp_centers", 24)),
        "neighborCount": int(arguments.get("pointnetpp_neighbors", 12)),
    }


@torch.no_grad()
def export_geometry_features(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    world_aabbs_np, instance_to_glb_np, runtime_meta = load_runtime_meta(
        args.runtime_meta
    )
    scene_min_np, _scene_max_np, scene_size_np = scene_min_max(
        runtime_meta["sceneBounds"]
    )
    points_np, points_meta = load_glb_points(args.glb_points, args.points_per_glb)
    num_instances = int(world_aabbs_np.shape[0])
    if instance_to_glb_np.size != num_instances:
        raise ValueError("runtime metadata instance mapping is incomplete")
    if instance_to_glb_np.size and int(instance_to_glb_np.max()) >= points_np.shape[0]:
        raise ValueError("GLB point table does not cover the runtime GLB mapping")

    encoder, encoder_meta = _load_encoder(args.encoder_checkpoint, device)
    world_aabbs = torch.from_numpy(world_aabbs_np).float().to(device)
    instance_to_glb = torch.from_numpy(instance_to_glb_np).long().to(device)
    glb_points = torch.from_numpy(points_np).float().to(device)
    scene_min = torch.from_numpy(scene_min_np).float().to(device)
    scene_size = torch.from_numpy(scene_size_np).float().to(device)
    rows: list[torch.Tensor] = []
    for start in range(0, num_instances, int(args.batch_size)):
        ids = torch.arange(
            start,
            min(num_instances, start + int(args.batch_size)),
            dtype=torch.long,
            device=device,
        )
        features = _point_features(
            ids,
            glb_points,
            instance_to_glb,
            world_aabbs,
            scene_min,
            scene_size,
        )
        rows.append(encoder(features).cpu())
    table = torch.cat(rows, dim=0)
    if table.shape != (num_instances, GEO_DIM) or not bool(torch.isfinite(table).all()):
        raise ValueError("exported geometry table has an invalid shape or value")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.numpy().astype("<f2").tofile(args.output)
    metadata = {
        "schema": "pvs-fixed-instance-geometry-features-v1",
        "shape": [num_instances, GEO_DIM],
        "dtype": "float16",
        "file": args.output.name,
        "encoderCheckpoint": str(args.encoder_checkpoint.expanduser().resolve()),
        "glbPoints": str(args.glb_points.expanduser().resolve()),
        "runtimeMeta": str(args.runtime_meta.expanduser().resolve()),
        "encoder": encoder_meta,
        "pointTable": points_meta,
        "runtimeUse": "fixed table only; PointNet++ is offline",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--glb-points", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-glb", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if min(args.points_per_glb, args.batch_size) <= 0:
        parser.error("points-per-glb and batch-size must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(export_geometry_features(parse_args()), ensure_ascii=False, indent=2))
