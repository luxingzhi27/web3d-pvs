#!/usr/bin/env python3
"""Shared-scene GCOF-PVS V5 training engine and checkpoint contract."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .core import GCOFPVSV5
from .losses import (
    SceneDualState,
    constrained_pose_risks,
    external_hit_nll,
    pbce_objective,
)
from .training_data import PoseBatch, ProbeBatch, V5SceneTrainingData


CHECKPOINT_SCHEMA = "gcof-pvs-v5-training-checkpoint-v1"


@dataclass(frozen=True)
class TrainStepResult:
    scene_id: str
    loss: float
    loss_field: float
    risk_extra: float
    risk_count: float
    risk_visual: float
    lambda_count: float
    lambda_visual: float
    candidate_count: int
    target_unit_count: int
    geometry_unit_count: int


def yaw_rotation(quarter_turns: int, device: torch.device) -> torch.Tensor:
    angle = (int(quarter_turns) % 4) * (math.pi / 2.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float32,
        device=device,
    )


def _rotate_surface(
    points: torch.Tensor,
    size_ratios: torch.Tensor,
    rotation: torch.Tensor,
    quarter_turns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = points.clone()
    result[..., :3] = result[..., :3] @ rotation.T
    result[..., 3:6] = result[..., 3:6] @ rotation.T
    ratios = size_ratios
    if int(quarter_turns) % 2:
        ratios = size_ratios[:, [2, 1, 0]]
    return result, ratios


def _torch_relation(
    relation: Mapping[str, Any],
    rotation: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    features = torch.as_tensor(relation["edge_features"], dtype=torch.float32, device=device).clone()
    features[..., :3] = features[..., :3] @ rotation.T
    return {
        "source_ids": torch.as_tensor(relation["source_ids"], dtype=torch.long, device=device),
        "valid_mask": torch.as_tensor(relation["valid_mask"], dtype=torch.bool, device=device),
        "edge_features": features,
        "metadata": relation["metadata"],
    }


def _encode_without_graph(
    model: GCOFPVSV5,
    scene: V5SceneTrainingData,
    unit_ids: np.ndarray,
    rotation: torch.Tensor,
    quarter_turns: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, unit_ids.size, chunk_size):
            ids = unit_ids[start : start + chunk_size]
            points, ratios = scene.geometry_tensors(ids, device)
            points, ratios = _rotate_surface(points, ratios, rotation, quarter_turns)
            rows.append(model.encode_geometry(points, ratios))
    return torch.cat(rows, dim=0)


def _recompute_geometry_backward(
    model: GCOFPVSV5,
    scene: V5SceneTrainingData,
    unit_ids: np.ndarray,
    cached_gradient: torch.Tensor,
    rotation: torch.Tensor,
    quarter_turns: int,
    chunk_size: int,
    device: torch.device,
) -> None:
    for start in range(0, unit_ids.size, chunk_size):
        ids = unit_ids[start : start + chunk_size]
        points, ratios = scene.geometry_tensors(ids, device)
        points, ratios = _rotate_surface(points, ratios, rotation, quarter_turns)
        encoded = model.encode_geometry(points, ratios)
        torch.autograd.backward(
            encoded,
            grad_tensors=cached_gradient[start : start + ids.size],
        )


def _rotated_pose_tensors(
    batch: PoseBatch,
    rotation: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    query = torch.from_numpy(batch.query_geometry).to(device).clone()
    query[:, :3] = query[:, :3] @ rotation.T
    centers = torch.from_numpy(batch.target_centers).to(device) @ rotation.T
    supports = torch.from_numpy(batch.support_points).to(device) @ rotation.T
    return {
        "candidate_ids": torch.from_numpy(batch.candidate_ids).long().to(device),
        "pose_rows": torch.from_numpy(batch.pose_rows).long().to(device),
        "targets": torch.from_numpy(batch.targets).float().to(device),
        "weights": torch.from_numpy(batch.visible_weights).float().to(device),
        "query": query,
        "centers": centers,
        "supports": supports,
        "radii": torch.from_numpy(batch.target_radii).float().to(device),
    }


def train_step(
    *,
    model: GCOFPVSV5,
    scene: V5SceneTrainingData,
    pose_batch: PoseBatch,
    probe_batch: ProbeBatch | None,
    optimizer: torch.optim.Optimizer,
    dual_state: SceneDualState,
    device: torch.device,
    quarter_turns: int,
    geometry_chunk_size: int = 512,
    objective: str = "FULL",
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> TrainStepResult:
    """Run one exact one-scene optimizer step."""

    if geometry_chunk_size <= 0:
        raise ValueError("geometry_chunk_size must be positive")
    objective_name = str(objective).upper()
    if objective_name not in {"FULL", "PBCE_OBJECTIVE"}:
        raise ValueError("objective must be FULL or PBCE_OBJECTIVE")
    if objective_name == "PBCE_OBJECTIVE" and model.variant != "FULL":
        raise ValueError("PBCE_OBJECTIVE uses the unchanged FULL architecture")

    candidate_targets = np.unique(pose_batch.candidate_ids.astype(np.int64, copy=False))
    if probe_batch is not None and model.variant != "GENERIC_RELATION_28":
        target_ids = np.union1d(candidate_targets, probe_batch.unit_ids.astype(np.int64, copy=False))
    else:
        target_ids = candidate_targets
    if model.variant == "GEOMETRY_FIELD":
        dependency_ids = target_ids
        relation = None
    else:
        dependency_ids = scene.geometry_dependency_ids(target_ids)
        relation = scene.relation_slice(target_ids)

    rotation = yaw_rotation(quarter_turns, device)
    z_cached = _encode_without_graph(
        model,
        scene,
        dependency_ids,
        rotation,
        quarter_turns,
        geometry_chunk_size,
        device,
    )
    z_leaf = z_cached.detach().requires_grad_(True)
    geometry_table = z_leaf.new_zeros((scene.num_units, z_leaf.shape[1])).index_copy(
        0,
        torch.from_numpy(dependency_ids).long().to(device),
        z_leaf,
    )
    relation_tensors = None if relation is None else _torch_relation(relation, rotation, device)
    anchor_directions = None
    if model.relation_compiler is not None:
        anchor_directions = model.relation_compiler.anchors @ rotation.T
    targets_tensor = torch.from_numpy(target_ids).long().to(device)
    compiled = model.compile_outputs(
        geometry_table,
        relation_tensors,
        targets_tensor,
        anchor_directions,
    )
    pose = _rotated_pose_tensors(pose_batch, rotation, device)
    candidate_local = torch.searchsorted(targets_tensor, pose["candidate_ids"])
    geometry_candidates = compiled["geometry"][candidate_local]

    field_loss = geometry_candidates.sum() * 0.0
    if model.variant == "GENERIC_RELATION_28":
        latent = compiled["generic_latent"][candidate_local]
        head_input = torch.cat([geometry_candidates, latent, pose["query"]], dim=-1)
    else:
        fields = compiled["field"]
        candidate_fields = fields[candidate_local]
        stats = model.region_statistics(
            candidate_fields,
            pose["centers"],
            pose["supports"],
            pose["radii"],
        )
        head_input = torch.cat([geometry_candidates, stats, pose["query"]], dim=-1)
        if probe_batch is not None:
            probe_ids = torch.from_numpy(probe_batch.unit_ids).long().to(device)
            probe_local = torch.searchsorted(targets_tensor, probe_ids)
            probe_directions = torch.from_numpy(probe_batch.directions).float().to(device) @ rotation.T
            probe_distances = torch.from_numpy(probe_batch.distances).float().to(device)
            probe_events = torch.from_numpy(probe_batch.events).float().to(device)
            probe_bounds = scene.world_aabbs[probe_batch.unit_ids]
            probe_radii = torch.from_numpy(
                (np.linalg.norm(probe_bounds[:, 3:] - probe_bounds[:, :3], axis=1) * 0.5).astype(np.float32)
            ).to(device)
            field_loss = external_hit_nll(
                fields[probe_local],
                probe_directions,
                probe_distances,
                probe_radii,
                probe_events,
            )
    logits = model.visibility_head(head_input).reshape(-1)
    risks = constrained_pose_risks(
        logits,
        pose["targets"],
        pose["weights"],
        float(pose_batch.denominators.pose_count) / float(pose_batch.pose_ids.size),
        pose_batch.denominators.visible_occurrences,
        pose_batch.denominators.visible_weight_sum,
    )
    if objective_name == "PBCE_OBJECTIVE":
        loss = pbce_objective(
            logits,
            pose["targets"],
            pose["pose_rows"],
            int(pose_batch.pose_ids.size),
            field_loss,
        )
    else:
        loss = dual_state.loss(scene.scene_id, risks, field_loss)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if z_leaf.grad is None or not bool(torch.isfinite(z_leaf.grad).all()):
        raise FloatingPointError("V5 cached geometry gradient is missing or non-finite")
    _recompute_geometry_backward(
        model,
        scene,
        dependency_ids,
        z_leaf.grad.detach(),
        rotation,
        quarter_turns,
        geometry_chunk_size,
        device,
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    # PBCE_OBJECTIVE is the registered objective-only control.  It keeps the
    # Full representation and field supervision, but deliberately does not
    # optimize the constrained objective or advance its dual variables.
    if objective_name != "PBCE_OBJECTIVE":
        dual_state.update(scene.scene_id, risks)
    lambda_index = dual_state.scene_ids.index(scene.scene_id)
    return TrainStepResult(
        scene_id=scene.scene_id,
        loss=float(loss.detach().item()),
        loss_field=float(field_loss.detach().item()),
        risk_extra=float(risks.extra.detach().item()),
        risk_count=float(risks.miss_count.detach().item()),
        risk_visual=float(risks.miss_visual.detach().item()),
        lambda_count=float(dual_state.lambda_count[lambda_index].item()),
        lambda_visual=float(dual_state.lambda_visual[lambda_index].item()),
        candidate_count=int(pose_batch.candidate_ids.size),
        target_unit_count=int(target_ids.size),
        geometry_unit_count=int(dependency_ids.size),
    )


def checkpoint_payload(
    *,
    model: GCOFPVSV5,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dual_state: SceneDualState,
    global_step: int,
    scene_updates: Mapping[str, int],
    rng_states: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "modelConfig": model.config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "dualState": dual_state.state_dict(),
        "globalStep": int(global_step),
        "sceneUpdates": {str(key): int(value) for key, value in scene_updates.items()},
        "rngStates": dict(rng_states),
        "config": dict(config),
        "testRead": False,
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "TrainStepResult",
    "checkpoint_payload",
    "train_step",
    "yaw_rotation",
]
