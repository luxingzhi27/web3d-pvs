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
from simulate_glb_streaming import (  # noqa: E402
    _formal_aabb_source,
    attach_formal_region66_visible_glbs,
    hzb_ordering_metadata,
    load_formal_region66_test_result,
    ordered_hzb_glb_ids_for_pose,
)


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
        default="full,aabb,hzb_visible_first",
        help="comma-separated threshold-free ranking methods",
    )
    parser.add_argument("--urgent-fraction", type=float, default=0.10)
    parser.add_argument("--warm-fraction", type=float, default=0.45)
    parser.add_argument("--utility-source", choices=["binary_gt", "visible_weights", "reference_frontmost_pixels"], default="binary_gt")
    parser.add_argument("--reference-frontmost", type=Path, default=None)
    parser.add_argument(
        "--hzb-region66-result",
        dest="hzb_region66_result",
        type=Path,
        default=None,
        help="formal Region66 test result JSON; visibleInstanceIds are mapped to GLBs",
    )
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
    )
    score_by_pose, score_manifest = load_score_sidecar(args.result_dir, args.dataset_dir, pose_ids)
    records = attach_geometry_scores(loaded, assets, score_by_pose, instance_to_glb)

    hzb_source: dict[str, Any] | None = None
    hzb_unavailable_reason: str | None = None
    if args.hzb_region66_result is None:
        hzb_unavailable_reason = "formal Region66 test result was not supplied"
    elif args.split != "test":
        hzb_unavailable_reason = "formal Region66 visible-first input is test-only"
    else:
        try:
            complete_test_pose_ids = dataset.split("test").pose_indices.astype(np.int64, copy=False).tolist()
            expected_candidate_ids = {
                int(pose_id): dataset.candidate_slice(int(pose_id))
                for pose_id in complete_test_pose_ids
            }
            hzb_instances, hzb_source = load_formal_region66_test_result(
                args.hzb_region66_result,
                complete_test_pose_ids,
                expected_scene=str(asset_meta.get("sceneName", "")),
                expected_candidate_ids=expected_candidate_ids,
            )
            records = attach_formal_region66_visible_glbs(
                records,
                loaded,
                hzb_instances,
                instance_to_glb,
            )
        except (FileNotFoundError, OSError, StreamingContractError, ValueError) as error:
            hzb_source = None
            hzb_unavailable_reason = f"formal Region66 result unavailable: {error}"

    method_plans: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, str] = {
        str(name): str(reason)
        for name, reason in (score_manifest.get("unavailableSources", {}) or {}).items()
    }
    score_sources = score_manifest.get("scoreSources", {})
    if "aabb" in methods and not _formal_aabb_source(
        score_sources.get("aabb") if isinstance(score_sources, dict) else None
    ):
        unavailable["aabb"] = (
            "formal AABB test sidecar is unavailable; legacy runner/fallback scores are not accepted"
        )
    if hzb_unavailable_reason is not None:
        unavailable["hzb_visible_first"] = hzb_unavailable_reason
    for method in methods:
        pose_rows: list[dict[str, Any]] = []
        if method == "hzb_visible_first":
            if hzb_source is None:
                method_plans[method] = {
                    "method": method,
                    "decisionMode": "threshold_free_ranking",
                    "thresholdApplied": False,
                    "poseCount": 0,
                    "poses": [],
                    "status": "unavailable",
                    "reason": unavailable.get(method, "formal Region66 visible-first input is unavailable"),
                    "rankingInput": {
                        "kind": "unavailable",
                        "reason": unavailable.get(method, "formal Region66 visible-first input is unavailable"),
                    },
                }
                continue
            try:
                for pose in records:
                    ordered_hzb_glb_ids_for_pose(pose, assets)
            except StreamingContractError as error:
                unavailable[method] = f"formal Region66 ordering unavailable: {error}"
                method_plans[method] = {
                    "method": method,
                    "decisionMode": "threshold_free_ranking",
                    "thresholdApplied": False,
                    "poseCount": 0,
                    "poses": [],
                    "status": "unavailable",
                    "reason": unavailable[method],
                    "rankingInput": {"kind": "unavailable", "reason": unavailable[method]},
                }
                continue
        if method in unavailable:
            method_plans[method] = {
                "method": method,
                "decisionMode": "threshold_free_ranking",
                "thresholdApplied": False,
                "poseCount": 0,
                "poses": [],
                "status": "unavailable",
                "reason": unavailable[method],
                "rankingInput": {
                    "kind": "unavailable",
                    "reason": unavailable[method],
                },
            }
            continue
        if method in {"full", "aabb", "distance", "projected_area", "projected_area_per_byte"} and any(
            method not in pose.rank_scores for pose in records
        ):
            unavailable.setdefault(
                method,
                f"continuous score input for {method} is unavailable for one or more selected poses",
            )
            method_plans[method] = {
                "method": method,
                "decisionMode": "threshold_free_ranking",
                "thresholdApplied": False,
                "poseCount": 0,
                "poses": [],
                "status": "unavailable",
                "reason": unavailable[method],
            }
            continue
        for pose in records:
            try:
                order = (
                    ordered_hzb_glb_ids_for_pose(pose, assets)
                    if method == "hzb_visible_first"
                    else ordered_glb_ids_for_pose(pose, assets, method)
                )
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
                    "rankingInput": hzb_ordering_metadata() if method == "hzb_visible_first" else score_sources.get(method) if isinstance(score_sources, dict) else None,
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
            "rankingInput": hzb_ordering_metadata() if method == "hzb_visible_first" else score_sources.get(method) if isinstance(score_sources, dict) else None,
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
            "scoreSources": score_sources if isinstance(score_sources, dict) else {},
            "hzbRegion66Result": str(args.hzb_region66_result.expanduser().resolve()) if args.hzb_region66_result else None,
            "hzbSource": hzb_source,
        },
    }
    (output_dir / "scheduler_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outputDir": str(output_dir), "poseIds": pose_ids, "methods": list(method_plans), "unavailableMethods": unavailable}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
