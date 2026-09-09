#!/usr/bin/env python3
"""Run the paper's strict cold-cache GLB streaming benchmark.

All input paths are required.  The selected CSR candidate instances are
mapped to GLBs once per pose; every ranking method receives exactly that same
candidate GLB set.  Threshold filtering is emitted in a separate summary and
is never used to compute ranking curves.
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
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from glb_streaming import (  # noqa: E402
    BANDWIDTHS_MBPS,
    RANKING_METHODS,
    TARGET_COVERAGES,
    StreamingContractError,
    build_fractional_utility_byte_lower_bound,
    simulate_ranked_pose,
    simulate_threshold_filter_pose,
    summarize_filter_results,
    summarize_ranking_results,
)
from glb_streaming_io import (  # noqa: E402
    attach_geometry_scores,
    load_glb_assets,
    load_pose_inputs,
    load_score_sidecar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate strict cold-cache GLB streaming.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True, help="continuous score sidecar directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "validation", "calibration", "train"], default="test")
    parser.add_argument("--pose-limit", type=int, default=0)
    parser.add_argument(
        "--utility-source",
        choices=["binary_gt", "visible_weights", "reference_frontmost_pixels"],
        default="binary_gt",
    )
    parser.add_argument(
        "--reference-frontmost",
        type=Path,
        default=None,
        help="JSONL sidecar from build_reference_frontmost_histogram.py",
    )
    parser.add_argument("--hzb-visible", type=Path, default=None, help="per-pose HZB visible GLB sidecar")
    parser.add_argument(
        "--methods",
        default=",".join(RANKING_METHODS),
        help="comma-separated ranking methods; defaults to the complete paper set",
    )
    parser.add_argument(
        "--filter-thresholds",
        type=Path,
        default=None,
        help="optional JSON mapping of frozen threshold-filter fields to thresholds",
    )
    parser.add_argument("--include-runner-fallback-thresholds", action="store_true")
    parser.add_argument("--log-every", type=int, default=100, help="print progress after this many poses; zero disables progress")
    return parser.parse_args()


def _read_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("thresholds", payload) if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        raise StreamingContractError("filter threshold file must be a method-to-threshold object")
    result: dict[str, float] = {}
    for name, value in values.items():
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise StreamingContractError(f"invalid filter threshold for {name}: {value}")
        result[str(name)] = threshold
    return result


def _load_manifest_thresholds(manifest: dict[str, Any], include_fallback: bool) -> dict[str, float]:
    thresholds = manifest.get("thresholds", {})
    sources = manifest.get("thresholdSources", {})
    result: dict[str, float] = {}
    for name, value in thresholds.items():
        source = str(sources.get(name, ""))
        if include_fallback or source == "checkpoint calibration":
            result[str(name)] = float(value)
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _coverage_source_name(utility_source: str) -> str:
    return {
        "binary_gt": "gt_glb_presence",
        "visible_weights": "visible_weights_utility_not_pixel_coverage",
        "reference_frontmost_pixels": "reference-frontmost-pixel-histogram-v1",
    }[utility_source]


def main() -> None:
    args = parse_args()
    if args.pose_limit < 0:
        raise ValueError("--pose-limit must be non-negative")
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative")
    methods = tuple(name.strip() for name in args.methods.split(",") if name.strip())
    unknown = [name for name in methods if name not in RANKING_METHODS]
    if unknown:
        raise ValueError(f"unknown ranking methods: {unknown}")
    if not methods:
        raise ValueError("--methods selected no methods")
    if args.utility_source == "reference_frontmost_pixels" and args.reference_frontmost is None:
        raise ValueError("reference-frontmost-pixels requires --reference-frontmost")

    assets, instance_to_glb, asset_meta = load_glb_assets(
        args.runtime_meta,
        args.glb_index,
        args.glb_root,
    )
    loaded_poses = load_pose_inputs(
        args.dataset_dir,
        args.runtime_meta,
        assets,
        instance_to_glb,
        split=args.split,
        pose_limit=args.pose_limit,
        utility_source=args.utility_source,
        reference_frontmost_path=args.reference_frontmost,
        hzb_visible_path=args.hzb_visible,
    )
    selected_pose_ids = [pose.record.pose_id for pose in loaded_poses]
    score_by_pose, score_manifest = load_score_sidecar(
        args.result_dir,
        args.dataset_dir,
        selected_pose_ids,
    )
    records = attach_geometry_scores(loaded_poses, assets, score_by_pose, instance_to_glb)

    ranking_rows: list[dict[str, Any]] = []
    unavailable_reasons: dict[str, str] = {}
    fractional_bounds: dict[str, list[int | None]] = {
        str(target): [] for target in TARGET_COVERAGES
    }
    simulation_started = time.perf_counter()
    for ordinal, pose in enumerate(records, start=1):
        method_rows: dict[str, dict[str, Any]] = {}
        for method in methods:
            if method == "hzb_visible_first" and pose.hzb_visible_glb_ids is None:
                unavailable_reasons[method] = "no --hzb-visible sidecar was supplied"
                continue
            try:
                method_rows[method] = simulate_ranked_pose(pose, assets, method)
            except StreamingContractError as error:
                if method == "hzb_visible_first":
                    unavailable_reasons[method] = str(error)
                    continue
                raise
        ranking_rows.append(
            {
                "poseId": int(pose.pose_id),
                "ordinal": int(pose.ordinal),
                "candidateGlbIds": list(pose.candidate_glb_ids),
                "gtGlbIds": list(pose.gt_glb_ids),
                "methods": method_rows,
            }
        )
        for target in TARGET_COVERAGES:
            fractional_bounds[str(target)].append(
                build_fractional_utility_byte_lower_bound(pose, assets, float(target))
            )
        if args.log_every > 0 and (ordinal % args.log_every == 0 or ordinal == len(records)):
            elapsed = time.perf_counter() - simulation_started
            print(
                f"[streaming-sim] {ordinal}/{len(records)} poses elapsed={elapsed:.1f}s",
                flush=True,
            )

    ranking_summary = summarize_ranking_results(
        ranking_rows,
        methods=methods,
        target_coverages=TARGET_COVERAGES,
        bandwidths_mbps=BANDWIDTHS_MBPS,
    )
    ranking_summary["scene"] = asset_meta
    ranking_summary["split"] = args.split
    ranking_summary["poseCount"] = len(records)
    ranking_summary["coverageSource"] = _coverage_source_name(args.utility_source)
    ranking_summary["source"] = {
        "datasetDir": str(Path(args.dataset_dir).expanduser().resolve()),
        "runtimeMeta": str(Path(args.runtime_meta).expanduser().resolve()),
        "glbIndex": str(Path(args.glb_index).expanduser().resolve()),
        "glbRoot": str(Path(args.glb_root).expanduser().resolve()),
        "scoreResultDir": str(Path(args.result_dir).expanduser().resolve()),
        "referenceFrontmost": str(args.reference_frontmost.expanduser().resolve()) if args.reference_frontmost else None,
        "hzbVisible": str(args.hzb_visible.expanduser().resolve()) if args.hzb_visible else None,
    }
    ranking_summary["unavailableReasons"] = unavailable_reasons
    ranking_summary["fractionalUtilityByteLowerBound"] = {
        target: {
            "count": int(sum(value is not None for value in values)),
            "mean": float(np.mean([value for value in values if value is not None]))
            if any(value is not None for value in values)
            else None,
            "median": float(np.median([value for value in values if value is not None]))
            if any(value is not None for value in values)
            else None,
        }
        for target, values in fractional_bounds.items()
    }

    filter_thresholds = {}
    if args.filter_thresholds is not None:
        filter_thresholds = _read_thresholds(args.filter_thresholds)
    else:
        filter_thresholds = _load_manifest_thresholds(
            score_manifest,
            args.include_runner_fallback_thresholds,
        )
    filter_rows: list[dict[str, Any]] = []
    filter_methods: list[str] = []
    by_pose_loaded = {pose.record.pose_id: pose for pose in loaded_poses}
    for method, threshold in filter_thresholds.items():
        if not any(method in fields for fields in score_by_pose.values()):
            continue
        filter_methods.append(method)
    for pose in records:
        loaded = by_pose_loaded[pose.pose_id]
        method_rows: dict[str, dict[str, Any]] = {}
        for method in filter_methods:
            scores = score_by_pose[pose.pose_id].get(method)
            if scores is None:
                continue
            method_rows[method] = simulate_threshold_filter_pose(
                pose,
                assets,
                method=method,
                instance_ids=loaded.candidate_instance_ids.tolist(),
                instance_scores=scores.tolist(),
                instance_to_glb=instance_to_glb,
                threshold=filter_thresholds[method],
            )
        filter_rows.append(
            {
                "poseId": int(pose.pose_id),
                "ordinal": int(pose.ordinal),
                "methods": method_rows,
            }
        )
    filtering_summary = summarize_filter_results(
        filter_rows,
        methods=filter_methods,
        target_coverages=TARGET_COVERAGES,
    )
    filtering_summary["scene"] = asset_meta
    filtering_summary["split"] = args.split
    filtering_summary["poseCount"] = len(records)
    filtering_summary["coverageSource"] = _coverage_source_name(args.utility_source)
    filtering_summary["source"] = ranking_summary["source"]
    filtering_summary["thresholdSources"] = score_manifest.get("thresholdSources", {})
    filtering_summary["note"] = (
        "Threshold filtering is a separate decision mode; its predicted subset is not used by ranking metrics."
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "ranking_per_pose.jsonl", ranking_rows)
    _write_jsonl(output_dir / "filtering_per_pose.jsonl", filter_rows)
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(ranking_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "filtering_summary.json").write_text(
        json.dumps(filtering_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "pvs-glb-streaming-experiment-v1",
        "decisionModes": ["threshold_free_ranking", "threshold_filtering"],
        "cacheMode": "strict_cold_cache_per_pose",
        "arrivalSemantics": "GLB bytes and reference-frontmost utility accumulate only after complete GLB arrival",
        "split": args.split,
        "poseCount": len(records),
        "methods": list(methods),
        "filterMethods": filter_methods,
        "filterThresholds": filter_thresholds,
        "coverageSource": _coverage_source_name(args.utility_source),
        "rankingSummary": "ranking_summary.json",
        "filteringSummary": "filtering_summary.json",
        "rankingPerPose": "ranking_per_pose.jsonl",
        "filteringPerPose": "filtering_per_pose.jsonl",
        "source": ranking_summary["source"],
        "scoreManifest": score_manifest,
        "unavailableReasons": unavailable_reasons,
    }
    (output_dir / "streaming_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "outputDir": str(output_dir),
                "split": args.split,
                "poses": len(records),
                "rankingMethods": list(methods),
                "filterMethods": filter_methods,
                "unavailableReasons": unavailable_reasons,
                "coverageSource": _coverage_source_name(args.utility_source),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
