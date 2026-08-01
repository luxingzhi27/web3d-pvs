#!/usr/bin/env python3
"""Train an independent RankNet GLB-utility score from cold-start metadata.

The ranker is a scheduling baseline, not a replacement for instance
visibility.  It receives only the stored AABB, camera ray/FOV and MVP-derived
projection features.  Training labels are the dataset's weak visible-weight
utility, explicitly normalized and never interpreted as pixel coverage.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "benchmark", ROOT / "model"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aabb_ray_feature_utils import FEATURE_DIM, build_aabb_ray_features  # noqa: E402
from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from utility_ranker import IndependentUtilityRankerMLP  # noqa: E402


UTILITY_SCALE = float(np.log1p(1_000_000.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument("--pairs-per-pose", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def _target_utility(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("visible_weights must be finite and non-negative")
    return np.clip(np.log1p(values) / UTILITY_SCALE, 0.0, 1.0).astype(np.float32, copy=False)


def _sample_pairs(
    dataset: PoseCSRDataset,
    pose_indices: np.ndarray,
    world_aabbs: np.ndarray,
    scene_diagonal: float,
    rng: np.random.Generator,
    pair_limit: int,
    pairs_per_pose: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    positive_targets: list[np.ndarray] = []
    pair_count = 0
    for raw_pose in rng.permutation(np.asarray(pose_indices, dtype=np.int64)).tolist():
        pose_index = int(raw_pose)
        visible_ids, visible_weights = dataset.visible_slice(pose_index)
        candidates = dataset.frustum_slice(pose_index).astype(np.int64, copy=False)
        if candidates.size == 0 or visible_ids.size == 0:
            continue
        visible_ids = np.asarray(visible_ids, dtype=np.int64).reshape(-1)
        visible_weights = np.asarray(visible_weights, dtype=np.float32).reshape(-1)
        if visible_ids.size != visible_weights.size:
            raise ValueError(f"pose {pose_index} has misaligned visible ids and weights")
        if not np.isin(visible_ids, candidates).all():
            raise ValueError(f"pose {pose_index} has visible ids outside the stored candidates")
        positives, positive_indices = np.unique(visible_ids, return_index=True)
        negatives = np.setdiff1d(candidates, positives, assume_unique=False)
        if positives.size == 0 or negatives.size == 0:
            continue
        count = min(int(pairs_per_pose), int(positives.size), int(negatives.size), int(pair_limit - pair_count))
        if count <= 0:
            break
        positive_pick = rng.choice(positives.size, size=count, replace=positives.size < count)
        negative_pick = rng.choice(negatives.size, size=count, replace=negatives.size < count)
        pos_ids = positives[positive_pick]
        neg_ids = negatives[negative_pick]
        pose = dataset.poses[pose_index]
        pos_features = build_aabb_ray_features(
            world_aabbs,
            pos_ids,
            pose["camera_world"],
            dataset.camera_view(pose_index),
            dataset.mvp_slice(pose_index),
            scene_diagonal,
        )
        neg_features = build_aabb_ray_features(
            world_aabbs,
            neg_ids,
            pose["camera_world"],
            dataset.camera_view(pose_index),
            dataset.mvp_slice(pose_index),
            scene_diagonal,
        )
        weight_map = {int(instance): float(weight) for instance, weight in zip(visible_ids.tolist(), visible_weights.tolist())}
        pos_targets = _target_utility(np.asarray([weight_map[int(instance)] for instance in pos_ids.tolist()], dtype=np.float32))
        positive_rows.append(pos_features)
        negative_rows.append(neg_features)
        positive_targets.append(pos_targets)
        pair_count += count
        if pair_count >= int(pair_limit):
            break
    if pair_count == 0:
        raise RuntimeError("could not sample a train pair with both positive and negative candidates")
    return (
        np.concatenate(positive_rows, axis=0)[:pair_count],
        np.concatenate(negative_rows, axis=0)[:pair_count],
        np.concatenate(positive_targets, axis=0)[:pair_count],
    )


@torch.no_grad()
def evaluate_rank_loss(
    model: IndependentUtilityRankerMLP,
    dataset: PoseCSRDataset,
    pose_indices: np.ndarray,
    world_aabbs: np.ndarray,
    scene_diagonal: float,
    device: torch.device,
    max_poses: int = 128,
) -> float:
    losses: list[float] = []
    rng = np.random.default_rng(17)
    selected = np.asarray(pose_indices, dtype=np.int64)[:max_poses]
    for _ in range(max(1, min(8, int(np.ceil(max(1, selected.size) / 16))))):
        if selected.size == 0:
            break
        pos, neg, target = _sample_pairs(dataset, selected, world_aabbs, scene_diagonal, rng, 512, 16)
        pos_score = model(torch.from_numpy(pos).to(device))
        neg_score = model(torch.from_numpy(neg).to(device))
        pair_loss = F.softplus(-(pos_score - neg_score))
        weight = 0.25 + torch.from_numpy(target).to(device)
        losses.append(float((pair_loss * weight).mean().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.steps_per_epoch, args.pairs_per_batch, args.pairs_per_pose) <= 0:
        raise ValueError("epochs, steps-per-epoch, pairs-per-batch and pairs-per-pose must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    world_aabbs, instance_to_glb, runtime = load_runtime_meta(args.runtime_meta)
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=int(world_aabbs.shape[0]))
    validation_name = "validation" if "validation" in dataset.split_ids else "val"
    train = dataset.split("train")
    validation = dataset.split(validation_name)
    _scene_min, _scene_max, scene_size = scene_min_max(runtime["sceneBounds"])
    scene_diagonal = float(np.linalg.norm(scene_size))
    model = IndependentUtilityRankerMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for step in range(args.steps_per_epoch):
            rng = np.random.default_rng(args.seed + epoch * 1_000_003 + step)
            pos, neg, target = _sample_pairs(
                dataset,
                train.pose_indices,
                world_aabbs,
                scene_diagonal,
                rng,
                args.pairs_per_batch,
                args.pairs_per_pose,
            )
            pos_score = model(torch.from_numpy(pos).to(device))
            neg_score = model(torch.from_numpy(neg).to(device))
            target_tensor = torch.from_numpy(target).to(device)
            loss = (F.softplus(-(pos_score - neg_score)) * (0.25 + target_tensor)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite independent RankNet loss at epoch={epoch}, step={step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_loss = evaluate_rank_loss(model, dataset, validation.pose_indices, world_aabbs, scene_diagonal, device)
        record = {
            "epoch": epoch,
            "trainRankNetLoss": float(np.mean(losses)),
            "validationRankNetLoss": validation_loss,
        }
        history.append(record)
        checkpoint = {
            "schema": "neuralstreamweb3d-independent-utility-ranker-v1",
            "config": {
                "featureDim": FEATURE_DIM,
                "numInstances": int(world_aabbs.shape[0]),
                "numGlbs": int(np.max(instance_to_glb)) + 1 if instance_to_glb.size else 0,
                "sceneDiagonal": scene_diagonal,
                "utilityScale": UTILITY_SCALE,
            },
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
            "protocol": {
                "trainOnly": True,
                "candidateSemantics": "stored_frustum_ids_strict",
                "geometryAccess": "instance_aabb_and_mvp_only_no_glb_geometry",
                "loss": "pairwise_ranknet",
                "utilityTeacher": "log1p_visible_weights_not_pixel_coverage",
                "validationPoseCount": int(validation.pose_indices.size),
            },
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if np.isfinite(validation_loss) and validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(record), flush=True)
    summary = {
        "schema": "neuralstreamweb3d-independent-utility-ranker-summary-v1",
        "checkpoint": str((args.output_dir / "best.pt").resolve()),
        "datasetDir": str(args.dataset_dir.resolve()),
        "runtimeMeta": str(args.runtime_meta.resolve()),
        "trainPoseCount": int(train.pose_indices.size),
        "validationPoseCount": int(validation.pose_indices.size),
        "sceneDiagonal": scene_diagonal,
        "featureDim": FEATURE_DIM,
        "utilityScale": UTILITY_SCALE,
        "bestValidationRankNetLoss": best_validation,
        "elapsedSeconds": time.perf_counter() - started,
        "protocol": "train-only pairwise ranker; no visibility input and no test access",
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
