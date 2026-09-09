#!/usr/bin/env python3
"""Materialize a small, explicit plan for the real scheduler replay.

The plan is derived from the same CSR candidate rows and continuous score
sidecar used by ``simulate_glb_streaming.py``.  It contains only a fixed
12-pose test subset, so the Node driver can exercise the real scheduler
without loading Python or discovering paths from its current directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from glb_streaming import (  # noqa: E402
    RANKING_METHODS,
    StreamingContractError,
    ordered_glb_ids_for_pose,
)
from glb_streaming_io import (  # noqa: E402
    attach_geometry_scores,
    load_glb_assets,
    load_pose_inputs,
    load_score_sidecar,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an explicit 12-pose scheduler replay plan.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True, help="continuous score sidecar directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "validation", "calibration", "train"], default="test")
    parser.add_argument("--pose-count", type=int, default=12)
    parser.add_argument(
        "--pose-ids",
        default="",
        help="optional comma-separated dataset pose IDs; otherwise choose evenly spaced split rows",
    )
    parser.add_argument(
        "--methods",
        default="full,aabb",
        help="comma-separated threshold-free ranking methods",
    )
    parser.add_argument("--urgent-fraction", type=float, default=0.10)
    parser.add_argument("--warm-fraction", type=float, default=0.45)
    parser.add_argument("--utility-source", choices=["binary_gt", "visible_weights", "reference_frontmost_pixels"], default="binary_gt")
    parser.add_argument("--reference-frontmost", type=Path, default=None)
    parser.add_argument("--hzb-visible", type=Path, default=None)
    return parser.parse_args()


def parse_pose_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item < 0 for item in result) or len(set(result)) != len(result):
        raise ValueError("--pose-ids must contain unique non-negative integers")
    return result


def select_pose_ids(dataset: PoseCSRDataset, split: str, pose_count: int, explicit: str) -> tuple[list[int], str]:
    if pose_count < 1:
        raise ValueError("--pose-count must be positive")
    requested = parse_pose_ids(explicit)
    split_ids = dataset.split(split).pose_indices.astype(np.int64, copy=False).tolist()
    split_set = set(int(value) for value in split_ids)
    if requested:
        if len(requested) != pose_count:
            raise ValueError(f"--pose-ids must contain exactly {pose_count} IDs")
        if any(value not in split_set for value in requested):
            raise ValueError(f"--pose-ids contains a pose outside split {split}")
        return requested, "explicit_pose_ids"
    if len(split_ids) < pose_count:
        raise ValueError(f"split {split} has only {len(split_ids)} poses, fewer than --pose-count={pose_count}")
    positions = np.rint(np.linspace(0, len(split_ids) - 1, pose_count)).astype(np.int64)
    selected = [int(split_ids[int(position)]) for position in positions.tolist()]
    if len(set(selected)) != pose_count:
        raise ValueError("evenly spaced pose selection produced duplicate IDs")
    return selected, "evenly_spaced_split_order_round_nearest"


def tier_ids(order: tuple[int, ...], urgent_fraction: float, warm_fraction: float) -> dict[str, list[int]]:
    urgent_count = int(round(len(order) * urgent_fraction))
    urgent_count = min(len(order), max(1 if order else 0, urgent_count))
    warm_count = int(round(len(order) * warm_fraction))
    warm_count = min(len(order) - urgent_count, max(0, warm_count))
    return {
        "urgent": list(order[:urgent_count]),
        "warm": list(order[urgent_count : urgent_count + warm_count]),
        "speculative": list(order[urgent_count + warm_count :]),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.urgent_fraction) <= 1.0:
        raise ValueError("--urgent-fraction must lie in (0, 1]")
    if not 0.0 <= float(args.warm_fraction) <= 1.0:
        raise ValueError("--warm-fraction must lie in [0, 1]")
    if float(args.urgent_fraction) + float(args.warm_fraction) > 1.0 + 1e-12:
        raise ValueError("urgent and warm fractions may not exceed one")
    methods = tuple(name.strip() for name in args.methods.split(",") if name.strip())
    if not methods:
        raise ValueError("--methods selected no methods")
    unknown = [name for name in methods if name not in RANKING_METHODS]
    if unknown:
        raise ValueError(f"unknown ranking methods: {unknown}")
    if args.utility_source == "reference_frontmost_pixels" and args.reference_frontmost is None:
        raise ValueError("reference-frontmost-pixels requires --reference-frontmost")

    assets, instance_to_glb, asset_meta = load_glb_assets(
        args.runtime_meta,
        args.glb_index,
        args.glb_root,
    )
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=len(instance_to_glb))
    pose_ids, selection_mode = select_pose_ids(dataset, args.split, args.pose_count, args.pose_ids)
    loaded = load_pose_inputs(
        args.dataset_dir,
        args.runtime_meta,
        assets,
        instance_to_glb,
        split=args.split,
        pose_ids=pose_ids,
        utility_source=args.utility_source,
        reference_frontmost_path=args.reference_frontmost,
        hzb_visible_path=args.hzb_visible,
    )
    score_by_pose, score_manifest = load_score_sidecar(args.result_dir, args.dataset_dir, pose_ids)
    records = attach_geometry_scores(loaded, assets, score_by_pose, instance_to_glb)

    method_plans: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, str] = {}
    for method in methods:
        pose_rows: list[dict[str, Any]] = []
        for pose in records:
            if method == "hzb_visible_first" and pose.hzb_visible_glb_ids is None:
                unavailable[method] = "no --hzb-visible sidecar was supplied"
                continue
            try:
                order = ordered_glb_ids_for_pose(pose, assets, method)
            except StreamingContractError as error:
                if method == "hzb_visible_first":
                    unavailable[method] = str(error)
                    continue
                raise
            tiers = tier_ids(order, float(args.urgent_fraction), float(args.warm_fraction))
            if set(tiers["urgent"] + tiers["warm"] + tiers["speculative"]) != set(pose.candidate_glb_ids):
                raise StreamingContractError(f"method {method} changed pose {pose.pose_id}'s candidate GLB set")
            pose_rows.append(
                {
                    "poseId": int(pose.pose_id),
                    "ordinal": int(pose.ordinal),
                    "candidateGlbIds": list(pose.candidate_glb_ids),
                    "gtGlbIds": list(pose.gt_glb_ids),
                    "orderedGlbIds": list(order),
                    "tiers": tiers,
                }
            )
        method_plans[method] = {
            "method": method,
            "decisionMode": "threshold_free_ranking",
            "thresholdApplied": False,
            "poseCount": len(pose_rows),
            "poses": pose_rows,
            "status": "available" if pose_rows else "unavailable",
            "reason": unavailable.get(method),
        }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "pvs-real-scheduler-plan-v1",
        "scene": asset_meta,
        "split": args.split,
        "testRead": args.split == "test",
        "poseCount": int(args.pose_count),
        "poseIds": pose_ids,
        "poseSelection": selection_mode,
        "methods": method_plans,
        "unavailableMethods": unavailable,
        "tiering": {
            "urgentFraction": float(args.urgent_fraction),
            "warmFraction": float(args.warm_fraction),
            "speculativeRemainder": True,
            "schedulerTiers": ["urgent", "warm", "speculative"],
        },
        "startup100": {
            "enabled": False,
            "startupTierUsed": False,
            "reason": "formal streaming replay has no fixed homepage prefix",
        },
        "cacheMode": "strict_cold_cache_per_pose",
        "candidateSemantics": "the stored CSR candidate instances mapped to one immutable GLB set per pose",
        "source": {
            "datasetDir": str(Path(args.dataset_dir).expanduser().resolve()),
            "runtimeMeta": str(Path(args.runtime_meta).expanduser().resolve()),
            "glbIndex": str(Path(args.glb_index).expanduser().resolve()),
            "glbRoot": str(Path(args.glb_root).expanduser().resolve()),
            "scoreResultDir": str(Path(args.result_dir).expanduser().resolve()),
            "scoreManifest": score_manifest,
            "coverageSource": args.utility_source,
        },
    }
    (output_dir / "scheduler_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outputDir": str(output_dir), "poseIds": pose_ids, "methods": list(method_plans), "unavailableMethods": unavailable}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
