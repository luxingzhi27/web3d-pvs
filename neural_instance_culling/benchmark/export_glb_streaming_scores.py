#!/usr/bin/env python3
"""Export continuous per-candidate scores for the GLB streaming benchmark.

The exporter is intentionally a separate step from the streaming simulator.
It reads only the explicitly supplied CSR/runtime/checkpoint paths and writes a
small aligned score sidecar.  The simulator can then compare threshold-free
ranking methods without rerunning the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
MODEL_DIR = BENCHMARK_DIR.parent / "model"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_runners import load_runner, load_runtime_meta, select_device  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aligned continuous scores for strict cold-cache GLB streaming."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument(
        "--model-spec",
        action="append",
        required=True,
        help=(
            "name|kind|checkpoint|runtime_features|calibration_summary; use '-' "
            "for unused static-runner fields"
        ),
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "validation", "calibration", "train"], default="test")
    parser.add_argument("--pose-limit", type=int, default=0)
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def parse_model_spec(value: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) != 5 or any(not part for part in parts[:2]):
        raise ValueError(
            "--model-spec must use name|kind|checkpoint|runtime_features|calibration_summary"
        )
    name, kind, checkpoint, runtime_features, eval_summary = parts
    spec = {
        "kind": kind,
        "checkpoint": checkpoint,
        "runtime_features": runtime_features,
        "eval_summary": eval_summary,
    }
    return name, spec


def _clean_optional_path(value: str) -> str:
    return "" if value == "-" else value


def main() -> None:
    args = parse_args()
    if args.pose_limit < 0:
        raise ValueError("--pose-limit must be non-negative")
    if args.poses_per_batch < 1:
        raise ValueError("--poses-per-batch must be positive")
    specs: list[tuple[str, dict[str, str]]] = [parse_model_spec(value) for value in args.model_spec]
    names = [name for name, _spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("--model-spec names must be unique")

    runtime_meta = args.runtime_meta.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    device = select_device(args.device)
    world_aabbs, _instance_to_glb, runtime = load_runtime_meta(runtime_meta)
    dataset = PoseCSRDataset(dataset_dir, num_instances=int(world_aabbs.shape[0]))
    split = dataset.split(args.split)
    pose_ids = split.pose_indices.astype(np.int64, copy=False)
    if args.pose_limit > 0:
        pose_ids = pose_ids[: args.pose_limit]
    if pose_ids.size == 0:
        raise ValueError(f"split {args.split} contains no selected poses")

    runners = []
    for name, parsed in specs:
        spec = {key: _clean_optional_path(value) for key, value in parsed.items()}
        runners.append(
            (
                name,
                load_runner(
                    name,
                    spec,
                    runtime_meta,
                    device,
                    fallback_threshold=0.5,
                    dataset_dir=dataset_dir,
                ),
                spec,
            )
        )

    pose_offsets = [0]
    all_candidate_ids: list[np.ndarray] = []
    score_parts: dict[str, list[np.ndarray]] = {name: [] for name, _runner, _spec in runners}
    score_thresholds: dict[str, float] = {}
    threshold_sources: dict[str, str] = {}
    start_time = time.perf_counter()
    rng = np.random.default_rng(20260909)
    for batch_start in range(0, pose_ids.size, args.poses_per_batch):
        batch_pose_ids = pose_ids[batch_start : batch_start + args.poses_per_batch]
        # No candidate cap or GT union is enabled here.  This is the exact
        # stored candidate set used by every later streaming method.
        batch = split.build_pose_set_batch(
            batch_pose_ids,
            world_aabbs,
            rng,
            include_empty=True,
        )
        actual_pose_ids = np.asarray(batch["pose_indices"], dtype=np.int64)
        if actual_pose_ids.tolist() != batch_pose_ids.tolist():
            raise ValueError("pose-set builder changed the selected pose order")
        ids = np.asarray(batch["instance"], dtype=np.uint32)
        all_candidate_ids.append(ids.copy())
        pose_offsets.extend(
            int(pose_offsets[-1] + (int(batch["pose_offsets"][index + 1]) - int(batch["pose_offsets"][index])))
            for index in range(batch_pose_ids.size)
        )
        for name, runner, spec in runners:
            if ids.size == 0:
                result_scores = np.zeros((0,), dtype=np.float32)
            else:
                result = runner.score_batch(batch)
                result_scores = np.asarray(result.scores, dtype=np.float32).reshape(-1)
                if result_scores.shape != ids.shape:
                    raise ValueError(
                        f"runner {name} returned {result_scores.size} scores for {ids.size} rows"
                    )
                if not np.all(np.isfinite(result_scores)):
                    raise ValueError(f"runner {name} returned non-finite scores")
            score_parts[name].append(result_scores.copy())
            if name not in score_thresholds:
                threshold = float(runner.threshold)
                score_thresholds[name] = threshold
                threshold_sources[name] = (
                    "checkpoint calibration" if runner.kind == "bounded_relation_survival_moment_v4" else "runner fallback; not a registered safety workpoint"
                )
        completed = min(pose_ids.size, batch_start + batch_pose_ids.size)
        if args.log_every > 0 and (completed % args.log_every == 0 or completed == pose_ids.size):
            elapsed = time.perf_counter() - start_time
            print(
                f"[streaming-score] {completed}/{pose_ids.size} poses "
                f"rows={pose_offsets[-1]} elapsed={elapsed:.1f}s",
                flush=True,
            )

    candidate_ids = np.concatenate(all_candidate_ids).astype(np.uint32, copy=False)
    arrays: dict[str, Any] = {
        "pose_ids": pose_ids.astype(np.int64, copy=False),
        "pose_offsets": np.asarray(pose_offsets, dtype=np.int64),
        "candidate_ids": candidate_ids,
    }
    for name, parts in score_parts.items():
        values = np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros((0,), dtype=np.float32)
        if values.shape != candidate_ids.shape:
            raise ValueError(f"score field {name} is not aligned with candidate IDs")
        arrays[name] = values

    result_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = result_dir / "scores.npz"
    np.savez(arrays_path, **arrays)
    manifest = {
        "schema": "pvs-glb-streaming-score-sidecar-v1",
        "createdBy": "export_glb_streaming_scores.py",
        "split": args.split,
        "testRead": args.split == "test",
        "poseCount": int(pose_ids.size),
        "candidateRowCount": int(candidate_ids.size),
        "numInstances": int(world_aabbs.shape[0]),
        "scoreFields": list(score_parts),
        "arraysFile": arrays_path.name,
        "thresholds": score_thresholds,
        "thresholdSources": threshold_sources,
        "source": {
            "datasetDir": str(dataset_dir),
            "runtimeMeta": str(runtime_meta),
            "runtimeSchema": runtime.get("schemaVersion"),
            "candidateSemantics": dataset.meta.get("candidateSemantics"),
            "gtSemantics": dataset.meta.get("gtSemantics"),
        },
        "alignment": {
            "poseIds": "pose_ids array in selected split order",
            "candidateIds": "candidate_ids array equals PoseCSR candidate_ids for each pose",
            "offsets": "pose_offsets indexes candidate_ids and every score field",
        },
    }
    (result_dir / "score_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"resultDir": str(result_dir), **{key: manifest[key] for key in ("split", "poseCount", "candidateRowCount", "scoreFields")}}, indent=2))


if __name__ == "__main__":
    main()
