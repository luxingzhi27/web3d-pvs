#!/usr/bin/env python3
"""Train a small candidate-preserving learned AABB-plus-ray baseline.

This is a lightweight comparison model, not the proposed fixed-feature model. It
learns only from stored train candidates and uses no GLB triangles, materials,
or test visibility data.  The output checkpoint is intentionally small and
has an explicit schema consumed by ``model_runners.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"
MODEL_DIR = ROOT / "model"
for path in (BENCHMARK_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aabb_ray_feature_utils import FEATURE_DIM, build_aabb_ray_features  # noqa: E402
from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--positive-samples-per-pose", type=int, default=32)
    parser.add_argument("--negative-samples-per-pose", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


class AabbRayMLP(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _sample_rows(
    dataset: PoseCSRDataset,
    pose_indices: np.ndarray,
    world_aabbs: np.ndarray,
    scene_diagonal: float,
    rng: np.random.Generator,
    batch_size: int,
    positive_samples_per_pose: int,
    negative_samples_per_pose: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rows = 0
    order = rng.permutation(np.asarray(pose_indices, dtype=np.int64))
    for pose_index in order.tolist():
        visible_ids, _weights = dataset.visible_slice(int(pose_index))
        candidates = dataset.candidate_slice(int(pose_index)).astype(np.int64, copy=False)
        if candidates.size == 0 or visible_ids.size == 0:
            continue
        positives = np.intersect1d(candidates, np.unique(visible_ids.astype(np.int64)), assume_unique=False)
        negatives = np.setdiff1d(candidates, positives, assume_unique=False)
        if positives.size == 0 or negatives.size == 0:
            continue
        pos_count = min(int(positive_samples_per_pose), int(positives.size))
        neg_count = min(int(negative_samples_per_pose), int(negatives.size))
        pos = rng.choice(positives, size=pos_count, replace=positives.size < pos_count)
        neg = rng.choice(negatives, size=neg_count, replace=negatives.size < neg_count)
        ids = np.concatenate([pos, neg]).astype(np.int64, copy=False)
        pose = dataset.poses[int(pose_index)]
        features = build_aabb_ray_features(
            world_aabbs,
            ids,
            pose["camera_world"],
            dataset.camera_view(int(pose_index)),
            dataset.mvp_slice(int(pose_index)),
            scene_diagonal,
        )
        feature_rows.append(features)
        targets.append(np.concatenate([np.ones(pos_count, dtype=np.float32), np.zeros(neg_count, dtype=np.float32)]))
        rows += ids.size
        if rows >= int(batch_size):
            break
    if not feature_rows:
        raise RuntimeError("Could not sample train candidate rows with both positive and negative labels")
    features = np.concatenate(feature_rows, axis=0)[:batch_size]
    labels = np.concatenate(targets, axis=0)[:batch_size]
    permutation = rng.permutation(features.shape[0])
    return features[permutation], labels[permutation]


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataset: PoseCSRDataset,
    pose_indices: np.ndarray,
    world_aabbs: np.ndarray,
    scene_diagonal: float,
    device: torch.device,
    max_poses: int = 128,
) -> float:
    losses: list[float] = []
    for pose_index in np.asarray(pose_indices, dtype=np.int64)[:max_poses].tolist():
        visible_ids, _weights = dataset.visible_slice(int(pose_index))
        candidates = dataset.candidate_slice(int(pose_index)).astype(np.int64, copy=False)
        if candidates.size == 0:
            continue
        labels = np.isin(candidates, np.unique(visible_ids.astype(np.int64)), assume_unique=False).astype(np.float32)
        features = build_aabb_ray_features(
            world_aabbs,
            candidates,
            dataset.poses[int(pose_index)]["camera_world"],
            dataset.camera_view(int(pose_index)),
            dataset.mvp_slice(int(pose_index)),
            scene_diagonal,
        )
        logits = model(torch.from_numpy(features).to(device))
        losses.append(float(nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels).to(device)).item()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.steps_per_epoch <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, steps-per-epoch, and batch-size must be positive")
    if args.positive_samples_per_pose <= 0 or args.negative_samples_per_pose <= 0:
        raise ValueError("positive/negative samples per pose must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    world_aabbs, instance_to_glb, runtime = load_runtime_meta(args.runtime_meta)
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=int(world_aabbs.shape[0]))
    scene_min, _scene_max, scene_size = scene_min_max(runtime["sceneBounds"])
    scene_diagonal = float(np.linalg.norm(scene_size))
    train = dataset.split("train")
    validation_name = "validation" if "validation" in dataset.split_ids else "val"
    validation = dataset.split(validation_name)
    model = AabbRayMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Balanced row sampling needs a moderate positive emphasis but not an
    # unbounded inverse-frequency weight; calibration later enforces safety.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0, device=device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for _step in range(args.steps_per_epoch):
            features, labels = _sample_rows(
                dataset,
                train.pose_indices,
                world_aabbs,
                scene_diagonal,
                np.random.default_rng(args.seed + epoch * 1000003 + _step),
                args.batch_size,
                args.positive_samples_per_pose,
                args.negative_samples_per_pose,
            )
            logits = model(torch.from_numpy(features).to(device))
            loss = criterion(logits, torch.from_numpy(labels).to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite AABB-ray baseline loss at epoch={epoch}, step={_step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        train_loss = float(np.mean(losses))
        validation_loss = evaluate_loss(model, dataset, validation.pose_indices, world_aabbs, scene_diagonal, device)
        record = {"epoch": epoch, "trainLoss": train_loss, "validationLoss": validation_loss}
        history.append(record)
        checkpoint = {
            "schema": "neuralstreamweb3d-learned-aabb-ray-v1",
            "config": {
                "featureDim": FEATURE_DIM,
                "numInstances": int(world_aabbs.shape[0]),
                "numGlbs": int(np.max(instance_to_glb)) + 1 if instance_to_glb.size else 0,
                "sceneDiagonal": scene_diagonal,
            },
            "model": model.state_dict(),
            "args": vars(args),
            "sceneBounds": {"min": scene_min.tolist(), "size": scene_size.tolist()},
            "protocol": {
                "trainOnly": True,
                "candidateSemantics": "stored_candidate_ids_strict",
                "geometryAccess": "instance_aabb_only_no_glb_geometry",
                "validationPoseCount": int(validation.pose_indices.size),
            },
            "history": history,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if np.isfinite(validation_loss) and validation_loss < best_val:
            best_val = validation_loss
            torch.save(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(record), flush=True)
    summary = {
        "schema": "neuralstreamweb3d-learned-aabb-ray-summary-v1",
        "checkpoint": str((args.output_dir / "best.pt").resolve()),
        "datasetDir": str(args.dataset_dir.resolve()),
        "runtimeMeta": str(args.runtime_meta.resolve()),
        "trainPoseCount": int(train.pose_indices.size),
        "validationPoseCount": int(validation.pose_indices.size),
        "sceneDiagonal": scene_diagonal,
        "featureDim": FEATURE_DIM,
        "bestValidationLoss": best_val,
        "elapsedSeconds": time.perf_counter() - start,
        "protocol": "train-only checkpoint; threshold must be calibrated by the shared validation/calibration evaluator",
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
