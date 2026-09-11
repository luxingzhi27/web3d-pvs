#!/usr/bin/env python3
"""Generate Table 5 CSV/Markdown and download-coverage figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from glb_streaming import (
    DECISION_MODE_SCHEDULER_REPLAY,
    DECISION_MODE_THRESHOLD_FILTERING,
    DECISION_MODE_THRESHOLD_FREE_RANKING,
    FIXED_RANDOM_METHODS,
    RANKING_METHOD_LABELS,
)


TABLE_FIELDS = [
    "scene",
    "method",
    "label",
    "status",
    "input_kind",
    "availability_reason",
    "decision_mode",
    "ranking_formula",
    "instance_aggregation",
    "cost_alpha",
    "parameter_selection_split",
    "uses_merged_glb_aabb",
    "pose_count",
    "coverage_source",
    "coverage_metric",
    "coverage_unit",
    "coverage_semantics",
    "candidate_glb_mean",
    "candidate_bytes_mean",
    "predicted_glb_mean",
    "predicted_bytes_mean",
    "predicted_byte_ratio_mean",
    "bytes_at_95_mean",
    "bytes_at_99_mean",
    "bytes_at_99_9_mean",
    "bytes_at_100_mean",
    "waste_before_99_mean",
    "required_rank_mean",
    "coverage_upper_bound_mean",
    "unreachable_95_ratio",
    "unreachable_99_ratio",
    "unreachable_99_9_ratio",
    "unreachable_100_ratio",
    "time_at_10mbps_99_mean_s",
    "time_at_25mbps_99_mean_s",
    "time_at_50mbps_99_mean_s",
    "time_at_100mbps_99_mean_s",
]

SCHEDULER_TABLE_FIELDS = [
    "scene",
    "method",
    "bandwidth_mbps",
    "fixed_pose_count",
    "repeat_count",
    "run_count",
    "first_frame_reached_ratio",
    "scheduler_first_frame_mean_ms",
    "scheduler_first_frame_median_ms",
    "scheduler_first_frame_p95_ms",
    "first_frame_glb_bytes_mean",
    "waste_before_first_frame_bytes_mean",
    "startup_asset_status",
    "startup_asset_bytes",
    "startup_asset_local_read_ms",
    "startup_transfer_time_ms",
    "cold_start_bytes_mean",
    "cold_start_network_lower_bound_mean_ms",
    "cold_start_network_lower_bound_median_ms",
    "cold_start_network_lower_bound_p95_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate streaming paper tables and figures.")
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        type=Path,
        help="ranking_summary.json; repeat once per scene",
    )
    parser.add_argument("--filter-summary", action="append", type=Path, default=[])
    parser.add_argument(
        "--scheduler-summary",
        action="append",
        type=Path,
        default=[],
        help="real scheduler replay summary; kept separate from ranking/filtering tables",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def scene_name(summary: dict[str, Any], path: Path) -> str:
    scene = summary.get("scene")
    if isinstance(scene, dict) and scene.get("sceneName"):
        return str(scene["sceneName"])
    source = summary.get("source")
    if isinstance(source, dict) and source.get("runtimeMeta"):
        return Path(str(source["runtimeMeta"])).parent.parent.name
    return path.parent.name


def scene_display_name(scene: str) -> str:
    normalized = scene.lower()
    if "hkust" in normalized:
        return "HKUST"
    if "ifcbench" in normalized or "metropolis" in normalized:
        return "Metropolis"
    return scene


def number_mean(method: dict[str, Any], field: str) -> float | None:
    value = method.get(field)
    if not isinstance(value, dict):
        return None
    return value.get("mean")


def stat_mean(value: Any) -> float | None:
    return value.get("mean") if isinstance(value, dict) else None


PAPER_LABEL_OVERRIDES = {
    "aabb": "AABB MLP",
}

SUMMARY_CONTRACTS = {
    DECISION_MODE_THRESHOLD_FREE_RANKING: (
        "pvs-glb-streaming-ranking-summary-v1",
        False,
    ),
    DECISION_MODE_THRESHOLD_FILTERING: (
        "pvs-glb-streaming-filter-summary-v1",
        True,
    ),
}


def validate_paper_summary(
    summary: dict[str, Any],
    path: Path,
    decision_mode: str,
) -> None:
    if decision_mode == DECISION_MODE_SCHEDULER_REPLAY:
        validate_scheduler_replay_summary(summary, path)
        return
    if decision_mode not in SUMMARY_CONTRACTS:
        raise ValueError(f"{path}: unsupported paper summary decision mode {decision_mode}")
    expected_schema, threshold_applied = SUMMARY_CONTRACTS[decision_mode]
    if summary.get("schema") != expected_schema:
        raise ValueError(f"{path}: expected {expected_schema}")
    if summary.get("decisionMode") != decision_mode:
        raise ValueError(f"{path}: decisionMode must be {decision_mode}")
    if summary.get("thresholdApplied") is not threshold_applied:
        raise ValueError(f"{path}: thresholdApplied disagrees with {decision_mode}")
    if summary.get("split") != "test" or summary.get("testRead") is not True:
        raise ValueError(f"{path}: paper streaming inputs must be frozen test summaries")
    if summary.get("cacheMode") != "strict_cold_cache_per_pose":
        raise ValueError(f"{path}: paper streaming input must use strict cold-cache poses")
    pose_count = summary.get("poseCount")
    if isinstance(pose_count, bool) or not isinstance(pose_count, int) or pose_count <= 0:
        raise ValueError(f"{path}: poseCount must be a positive integer")
    test_pose_count = summary.get("testPoseCount")
    if (
        isinstance(test_pose_count, bool)
        or not isinstance(test_pose_count, int)
        or test_pose_count <= 0
        or test_pose_count != pose_count
    ):
        raise ValueError(f"{path}: complete test summary must declare testPoseCount equal to poseCount")
    selection = summary.get("poseSelection")
    if not isinstance(selection, dict):
        raise ValueError(f"{path}: complete test summary is missing poseSelection")
    if (
        selection.get("split") != "test"
        or selection.get("selectedPoseCount") != pose_count
        or selection.get("limit") not in (None, 0)
        or selection.get("completeTest") is not True
    ):
        raise ValueError(f"{path}: complete test summary has a truncated or incomplete pose selection")
    coverage_source = str(summary.get("coverageSource", ""))
    if coverage_source in {
        "visible_weights",
        "visible_weight_coverage",
        "visible_weights_utility_not_pixel_coverage",
    }:
        semantics = summary.get("coverageSemantics")
        if not isinstance(semantics, str) or "pixel" not in semantics.lower():
            raise ValueError(
                f"{path}: visible-weight coverage must declare that it is not pixel coverage"
            )
    methods = summary.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError(f"{path}: paper streaming summary has no methods")
    for name, method in methods.items():
        if not isinstance(method, dict):
            raise ValueError(f"{path}: method {name} is not an object")
        if method.get("status") == "available":
            if method.get("decisionMode") != decision_mode:
                raise ValueError(f"{path}: method {name} has the wrong decision mode")
            if method.get("poseCount") != pose_count:
                raise ValueError(f"{path}: method {name} does not cover the complete test summary")


def validate_scheduler_replay_summary(
    summary: dict[str, Any],
    path: Path,
) -> None:
    """Validate the separately measured production scheduler replay contract."""

    if summary.get("schema") != "pvs-real-scheduler-streaming-summary-v1":
        raise ValueError(
            f"{path}: scheduler replay must use pvs-real-scheduler-streaming-summary-v1"
        )
    if (
        summary.get("decisionMode") is not None
        and summary.get("decisionMode") != DECISION_MODE_SCHEDULER_REPLAY
    ):
        raise ValueError(f"{path}: scheduler summary has a ranking/filtering decision mode")
    if summary.get("split") != "test" or summary.get("testRead") is not True:
        raise ValueError(f"{path}: scheduler replay must read the test split")
    if summary.get("scheduler") != "GlbResourceScheduler":
        raise ValueError(f"{path}: scheduler replay must use GlbResourceScheduler")
    if summary.get("schedulerTiers") != ["urgent", "warm", "speculative"]:
        raise ValueError(f"{path}: scheduler replay must declare all scheduler tiers")
    if summary.get("startup100Enabled") is not False or summary.get("startupTierUsed") is not False:
        raise ValueError(f"{path}: scheduler replay must disable startup-100")
    pose_count = summary.get("poseCount")
    pose_ids = summary.get("poseIds")
    if pose_count != 12 or not isinstance(pose_ids, list) or len(pose_ids) != 12:
        raise ValueError(f"{path}: scheduler replay must contain exactly 12 fixed test poses")
    if len(set(pose_ids)) != len(pose_ids):
        raise ValueError(f"{path}: scheduler replay pose IDs must be unique")
    if summary.get("cacheMode") != "strict_cold_cache_per_pose":
        raise ValueError(f"{path}: scheduler replay must use strict cold-cache poses")


def scheduler_replay_rows(
    summary: dict[str, Any], path: Path
) -> list[dict[str, Any]]:
    """Build measured scheduler rows plus a startup-transfer lower bound."""

    validate_scheduler_replay_summary(summary, path)
    method_assets = summary.get("methodAssets") or {}
    repeats = int(summary.get("repeats", 0))
    fixed_pose_count = int(summary["poseCount"])
    rows: list[dict[str, Any]] = []
    for measured in summary.get("summaries", []):
        method = str(measured.get("method", ""))
        bandwidth = float(measured.get("bandwidthMbps", 0.0))
        first_frame = measured.get("firstFrameMs") or {}
        first_frame_bytes = measured.get("firstFrameBytes") or {}
        waste = measured.get("wasteBeforeFirstFrameBytes") or {}
        asset = method_assets.get(method) if isinstance(method_assets, dict) else None
        asset_bytes = int(asset.get("byteCount", 0)) if isinstance(asset, dict) else 0
        # A zero-byte AABB entry means no deployable baseline bundle was supplied;
        # it must not become a claimed zero-cost startup result.
        startup_available = asset_bytes > 0
        transfer_ms = (
            asset_bytes * 8.0 / (bandwidth * 1_000_000.0) * 1000.0
            if startup_available and bandwidth > 0
            else None
        )
        row = {
            "scene": scene_name(summary, path),
            "method": method,
            "bandwidth_mbps": bandwidth,
            "fixed_pose_count": fixed_pose_count,
            "repeat_count": repeats,
            "run_count": first_frame.get("count"),
            "first_frame_reached_ratio": measured.get("firstFrameReachedRatio"),
            "scheduler_first_frame_mean_ms": first_frame.get("mean"),
            "scheduler_first_frame_median_ms": first_frame.get("median"),
            "scheduler_first_frame_p95_ms": first_frame.get("p95"),
            "first_frame_glb_bytes_mean": first_frame_bytes.get("mean"),
            "waste_before_first_frame_bytes_mean": waste.get("mean"),
            "startup_asset_status": "available" if startup_available else "unavailable",
            "startup_asset_bytes": asset_bytes if startup_available else None,
            "startup_asset_local_read_ms": asset.get("loadMs") if startup_available else None,
            "startup_transfer_time_ms": transfer_ms,
            "cold_start_bytes_mean": (
                asset_bytes + float(first_frame_bytes["mean"])
                if startup_available and first_frame_bytes.get("mean") is not None
                else None
            ),
        }
        for statistic in ("mean", "median", "p95"):
            elapsed = first_frame.get(statistic)
            row[f"cold_start_network_lower_bound_{statistic}_ms"] = (
                float(elapsed) + float(transfer_ms)
                if elapsed is not None and transfer_ms is not None
                else None
            )
        rows.append({field: row.get(field) for field in SCHEDULER_TABLE_FIELDS})
    return rows


def write_scheduler_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Real scheduler replay",
        "",
        "Scheduler time measures real GLB responses through `GlbResourceScheduler`. The cold-start column adds the visibility startup asset transfer at the same aggregate bandwidth; it is a network lower bound and excludes asset decode, model initialization, HZB construction and final rendering.",
        "",
        "| Scene | Method | Mbps | Scheduler median | Scheduler p95 | GLB mean | Waste mean | Visibility asset | Cold-start median lower bound | Cold-start p95 lower bound |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        asset = row["startup_asset_bytes"]
        cold_median = row["cold_start_network_lower_bound_median_ms"]
        cold_p95 = row["cold_start_network_lower_bound_p95_ms"]
        lines.append(
            "| {scene} | {method} | {bandwidth:.0f} | {median:.3f} s | {p95:.3f} s | {glb:.3f} MiB | {waste:.3f} MiB | {asset} | {cold_median} | {cold_p95} |".format(
                scene=scene_display_name(str(row["scene"])),
                method=row["method"],
                bandwidth=float(row["bandwidth_mbps"]),
                median=float(row["scheduler_first_frame_median_ms"]) / 1000.0,
                p95=float(row["scheduler_first_frame_p95_ms"]) / 1000.0,
                glb=float(row["first_frame_glb_bytes_mean"]) / (1024 * 1024),
                waste=float(row["waste_before_first_frame_bytes_mean"]) / (1024 * 1024),
                asset=(f"{float(asset) / (1024 * 1024):.3f} MiB" if asset is not None else "unavailable"),
                cold_median=(f"{float(cold_median) / 1000.0:.3f} s" if cold_median is not None else "unavailable"),
                cold_p95=(f"{float(cold_p95) / 1000.0:.3f} s" if cold_p95 is not None else "unavailable"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ranking_input(summary: dict[str, Any], method_name: str) -> dict[str, Any] | None:
    inputs = summary.get("rankingInputs")
    if isinstance(inputs, dict) and isinstance(inputs.get(method_name), dict):
        return inputs[method_name]
    sources = summary.get("scoreSources")
    if method_name == "aabb" and isinstance(sources, dict) and isinstance(sources.get("aabb"), dict):
        return sources["aabb"]
    method = summary.get("methods", {}).get(method_name)
    if isinstance(method, dict) and isinstance(method.get("rankingInput"), dict):
        return method["rankingInput"]
    return None


def _paper_method_view(
    summary: dict[str, Any], method_name: str, method: dict[str, Any]
) -> dict[str, Any]:
    """Hide legacy placeholder rows from paper tables and curves."""

    view = dict(method)
    if method_name in PAPER_LABEL_OVERRIDES:
        view["label"] = PAPER_LABEL_OVERRIDES[method_name]
    if method_name == "aabb" and view.get("status") == "available":
        source = _ranking_input(summary, method_name)
        if (
            not isinstance(source, dict)
            or source.get("kind") != "formal_aabb_test_sidecar"
            or source.get("split") != "test"
            or source.get("testRead") is not True
        ):
            view["status"] = "unavailable"
            view["reason"] = (
                "legacy or missing AABB input: a formal test score sidecar is required"
            )
    if method_name == "hzb_visible_first" and view.get("status") == "available":
        source = _ranking_input(summary, method_name)
        if (
            not isinstance(source, dict)
            or source.get("kind") != "formal_region66_test_result"
            or source.get("mode") != "Region66"
            or source.get("split") != "test"
            or source.get("formalReady") is not True
            or source.get("continuousScore") is not False
        ):
            view["status"] = "unavailable"
            view["reason"] = (
                "legacy or missing HZB input: a formal Region66 test visible-instance result is required"
            )
    if method_name == "neural_cost" and view.get("status") == "available":
        source = _ranking_input(summary, method_name)
        manifest = summary.get("neuralCostManifest")
        if (
            not isinstance(source, dict)
            or source.get("formula") != "p_g / bytes_g^alpha"
            or source.get("instanceProbabilityAggregation") != "max"
            or source.get("parameterSelectionSplit") != "validation"
            or not isinstance(manifest, dict)
            or manifest.get("frozen") is not True
            or manifest.get("selectionSplit") != "validation"
            or manifest.get("testRead") is True
        ):
            view["status"] = "unavailable"
            view["reason"] = (
                "missing frozen validation neural-cost alpha manifest"
            )
    return view


def summary_row(summary: dict[str, Any], source_path: Path, method_name: str, method: dict[str, Any]) -> dict[str, Any]:
    method = _paper_method_view(summary, method_name, method)
    filtering = method.get("decisionMode") == "threshold_filtering"
    ranking_input = _ranking_input(summary, method_name)
    coverage_upper_bound = number_mean(method, "visibleWeightCoverageUpperBound")
    if coverage_upper_bound is None:
        coverage_upper_bound = number_mean(method, "coverageUpperBound")
    row: dict[str, Any] = {
        "scene": scene_name(summary, source_path),
        "method": method_name,
        "label": method.get("label", RANKING_METHOD_LABELS.get(method_name, method_name)),
        "status": method.get("status", "unavailable"),
        "input_kind": ranking_input.get("kind") if isinstance(ranking_input, dict) else None,
        "availability_reason": method.get("reason"),
        "decision_mode": method.get("decisionMode", "threshold_free_ranking"),
        "ranking_formula": ranking_input.get("formula") if isinstance(ranking_input, dict) else None,
        "instance_aggregation": (
            ranking_input.get("instanceProbabilityAggregation", ranking_input.get("instanceAabbAggregation"))
            if isinstance(ranking_input, dict)
            else None
        ),
        "cost_alpha": ranking_input.get("costExponent") if isinstance(ranking_input, dict) else None,
        "parameter_selection_split": (
            ranking_input.get("parameterSelectionSplit") if isinstance(ranking_input, dict) else None
        ),
        "uses_merged_glb_aabb": (
            ranking_input.get("usesMergedGlbAabb") if isinstance(ranking_input, dict) else None
        ),
        "pose_count": method.get("poseCount", 0),
        "coverage_source": summary.get("coverageSource", "unknown"),
        "coverage_metric": summary.get("coverageMetric"),
        "coverage_unit": summary.get("coverageUnit"),
        "coverage_semantics": summary.get("coverageSemantics"),
        "candidate_glb_mean": number_mean(method, "candidateGlbCount"),
        "candidate_bytes_mean": number_mean(method, "candidateGlbBytes"),
        "predicted_glb_mean": number_mean(method, "predictedGlbCount") if filtering else None,
        "predicted_bytes_mean": number_mean(method, "predictedGlbBytes") if filtering else None,
        "predicted_byte_ratio_mean": number_mean(method, "predictedGlbByteRatio") if filtering else None,
        "bytes_at_95_mean": None if filtering else number_mean(method.get("bytesAtCoverage", {}), "95"),
        "bytes_at_99_mean": None if filtering else number_mean(method.get("bytesAtCoverage", {}), "99"),
        "bytes_at_99_9_mean": None if filtering else number_mean(method.get("bytesAtCoverage", {}), "99.9"),
        "bytes_at_100_mean": None if filtering else number_mean(method.get("bytesAtCoverage", {}), "100"),
        "waste_before_99_mean": None if filtering else number_mean(method, "wasteBefore99Bytes"),
        "required_rank_mean": None if filtering else number_mean(method, "requiredRank"),
        "coverage_upper_bound_mean": coverage_upper_bound,
    }
    for key in ("95", "99", "99.9", "100"):
        row[f"unreachable_{key.replace('.', '_')}_ratio"] = (
            method.get("unreachable", {}).get(key, {}) or {}
        ).get("ratio")
    time_stats = {} if filtering else method.get("timeSecondsAtCoverage", {}).get("99", {}) or {}
    for bandwidth in (10, 25, 50, 100):
        row[f"time_at_{bandwidth}mbps_99_mean_s"] = stat_mean(time_stats.get(str(bandwidth), {}))
    return {field: row.get(field) for field in TABLE_FIELDS}


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        ("Scene", "scene"),
        ("Method", "label"),
        ("Status", "status"),
        ("Input", "input_kind"),
        ("Reason", "availability_reason"),
        ("Mode", "decision_mode"),
        ("Formula", "ranking_formula"),
        ("Aggregation", "instance_aggregation"),
        ("Alpha", "cost_alpha"),
        ("Pred GLBs", "predicted_glb_mean"),
        ("Pred bytes", "predicted_bytes_mean"),
        ("Pred/cand", "predicted_byte_ratio_mean"),
        ("Bytes@95", "bytes_at_95_mean"),
        ("Bytes@99", "bytes_at_99_mean"),
        ("Bytes@99.9", "bytes_at_99_9_mean"),
        ("Bytes@100", "bytes_at_100_mean"),
        ("Waste<99", "waste_before_99_mean"),
        ("Required rank", "required_rank_mean"),
        ("Coverage cap", "coverage_upper_bound_mean"),
        ("Unreachable@99", "unreachable_99_ratio"),
        ("25 Mbps @99 s", "time_at_25mbps_99_mean_s"),
        ("50 Mbps @99 s", "time_at_50mbps_99_mean_s"),
    ]
    lines = [
        "# Progressive GLB streaming",
        "",
        "Values are per-pose means. Bytes are complete GLB bytes; a GLB contributes only after its full resource arrives.",
        "Ranking rows use the complete fixed candidate GLB set and no visibility threshold. Coverage is visible-weight coverage when the declared utility source is `visible_weights`. `NA` means the method was unavailable.",
        "",
        "| " + " | ".join(label for label, _key in columns) + " |",
        "| " + " | ".join("---" for _label, _key in columns) + " |",
    ]
    for row in rows:
        cells = []
        for label, key in columns:
            value = row.get(key)
            if key in {"predicted_bytes_mean", "bytes_at_95_mean", "bytes_at_99_mean", "bytes_at_99_9_mean", "bytes_at_100_mean", "waste_before_99_mean"}:
                cells.append(fmt(None if value is None else float(value) / (1024 * 1024), 3) + " MiB")
            elif key == "predicted_byte_ratio_mean":
                cells.append(fmt(None if value is None else float(value) * 100.0, 2) + "%")
            elif key.startswith("unreachable_"):
                cells.append(fmt(None if value is None else float(value) * 100.0, 2) + "%")
            elif key == "coverage_upper_bound_mean":
                cells.append(fmt(None if value is None else float(value) * 100.0, 3) + "%")
            elif key.endswith("_s"):
                cells.append(fmt(value, 3))
            else:
                cells.append(fmt(value, 3) if isinstance(value, (float, int)) else str(value or "NA"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "`reference-frontmost-pixel-histogram-v1` is a front-most Color-ID pixel utility proxy. It excludes background and hidden surfaces; it is not a complete download utility or a hidden-visibility count.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def curve_rows(summary: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    scene = scene_name(summary, source_path)
    rows: list[dict[str, Any]] = []
    for method_name, method in summary.get("methods", {}).items():
        method = _paper_method_view(summary, method_name, method)
        if method.get("status") != "available":
            continue
        for point in method.get("coverageCurve", []):
            rows.append(
                {
                    "scene": scene,
                    "method": method_name,
                    "label": method.get("label", RANKING_METHOD_LABELS.get(method_name, method_name)),
                    "rank_fraction": point.get("rankFraction"),
                    "mean_bytes": point.get("meanBytes"),
                    "mean_coverage": point.get(
                        "meanVisibleWeightCoverage",
                        point.get("meanCoverage"),
                    ),
                }
            )
    return rows


def coverage_axis_label(summary: dict[str, Any]) -> str:
    source = str(summary.get("coverageSource", ""))
    if source in {
        "reference_frontmost_pixel_utility",
        "reference-frontmost-pixel-histogram-v1",
    }:
        return "Reference-frontmost pixel utility (%)"
    if source in {
        "visible_weights",
        "visible_weight_coverage",
        "visible_weights_utility_not_pixel_coverage",
    }:
        return "Visible-weight coverage (%)"
    return "GT GLB presence coverage (%)"


def write_curve_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["scene", "method", "label", "rank_fraction", "mean_bytes", "mean_coverage"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path_prefix: Path, summaries: list[tuple[Path, dict[str, Any]]]) -> None:
    scenes = [(source, summary, scene_name(summary, source)) for source, summary in summaries]
    fig, axes = plt.subplots(1, len(scenes), figsize=(6.2 * len(scenes), 4.4), squeeze=False)
    axes = axes[0]
    plot_methods = [
        "full",
        "aabb",
        "original",
        "distance",
        "projected_area",
        "projected_area_per_byte",
        "neural_cost",
        "hzb_visible_first",
        "gt_utility_per_byte_oracle",
    ]
    colors = {
        "full": "#1f77b4",
        "neural_cost": "#17becf",
        "aabb": "#ff7f0e",
        "original": "#7f7f7f",
        "distance": "#2ca02c",
        "projected_area": "#d62728",
        "projected_area_per_byte": "#9467bd",
        "hzb_visible_first": "#8c564b",
        "gt_utility_per_byte_oracle": "#111111",
    }
    for axis, (source, summary, scene) in zip(axes, scenes):
        methods = summary.get("methods", {})
        for method_name in plot_methods:
            method = methods.get(method_name)
            if not isinstance(method, dict):
                continue
            method = _paper_method_view(summary, method_name, method)
            if method.get("status") != "available":
                continue
            points = method.get("coverageCurve", [])
            if not points:
                continue
            x = [float(point.get("meanBytes", 0.0)) / (1024 * 1024) for point in points]
            y = [
                float(
                    point.get(
                        "meanVisibleWeightCoverage",
                        point.get("meanCoverage", 0.0),
                    )
                )
                * 100.0
                for point in points
            ]
            axis.plot(
                x,
                y,
                label=method.get("label", RANKING_METHOD_LABELS.get(method_name, method_name)),
                color=colors.get(method_name),
                linewidth=2.0 if method_name in {"full", "gt_utility_per_byte_oracle"} else 1.25,
                linestyle="--" if method_name == "gt_utility_per_byte_oracle" else "-",
            )
        random_methods = [methods.get(name) for name in FIXED_RANDOM_METHODS]
        random_methods = [method for method in random_methods if isinstance(method, dict) and method.get("status") == "available"]
        if random_methods:
            curves = [method.get("coverageCurve", []) for method in random_methods]
            if all(len(curve) == len(curves[0]) for curve in curves):
                x_values = np.asarray(
                    [[float(point.get("meanBytes", 0.0)) for point in curve] for curve in curves]
                )
                x = x_values.mean(axis=0) / (1024 * 1024)
                y_values = np.asarray([[float(point.get("meanCoverage", 0.0)) * 100.0 for point in curve] for curve in curves])
                axis.plot(x, y_values.mean(axis=0), color="#aaaaaa", linewidth=1.5, label="20 fixed random (mean)")
                axis.fill_between(x, y_values.min(axis=0), y_values.max(axis=0), color="#aaaaaa", alpha=0.18)
        axis.set_title(scene_display_name(scene))
        axis.set_xlabel("Complete GLB bytes downloaded (MiB)")
        axis.set_ylabel(coverage_axis_label(summary))
        axis.set_ylim(0.0, 100.5)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    fig.suptitle("Strict cold-cache GLB streaming")
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(path_prefix.with_suffix(".pdf"))
    fig.savefig(path_prefix.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summaries = [(path.expanduser().resolve(), read_json(path)) for path in args.summary]
    for path, summary in summaries:
        validate_paper_summary(summary, path, "threshold_free_ranking")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []
    for source_path, summary in summaries:
        for method_name, method in summary.get("methods", {}).items():
            rows.append(summary_row(summary, source_path, method_name, method))
        curves.extend(curve_rows(summary, source_path))
    write_csv(output_dir / "table5_streaming_ranking.csv", rows)
    write_markdown(output_dir / "table5_streaming_ranking.md", rows)
    write_curve_csv(output_dir / "streaming_coverage_curve.csv", curves)
    plot_prefix = output_dir / "streaming_coverage_curve"
    plot_curves(plot_prefix, summaries)

    filter_rows = []
    for path in args.filter_summary:
        summary = read_json(path)
        validate_paper_summary(summary, path.expanduser().resolve(), "threshold_filtering")
        for method_name, method in summary.get("methods", {}).items():
            filter_rows.append(summary_row(summary, path, method_name, method))
    if filter_rows:
        write_csv(output_dir / "table5_streaming_filtering.csv", filter_rows)
        write_markdown(output_dir / "table5_streaming_filtering.md", filter_rows)

    scheduler_summaries = []
    scheduler_rows = []
    for path in args.scheduler_summary:
        resolved = path.expanduser().resolve()
        summary = read_json(path)
        validate_paper_summary(summary, resolved, DECISION_MODE_SCHEDULER_REPLAY)
        scheduler_rows.extend(scheduler_replay_rows(summary, resolved))
        scheduler_summaries.append(
            {
                "path": str(resolved),
                "schema": summary["schema"],
                "decisionMode": DECISION_MODE_SCHEDULER_REPLAY,
                "poseCount": summary["poseCount"],
                "methods": summary.get("methods", []),
                "bandwidthsMbps": summary.get("bandwidthsMbps", []),
            }
        )
    if scheduler_summaries:
        (output_dir / "scheduler_replay_inputs.json").write_text(
            json.dumps(
                {
                    "schema": "pvs-glb-streaming-scheduler-replay-inputs-v1",
                    "decisionMode": DECISION_MODE_SCHEDULER_REPLAY,
                    "summaries": scheduler_summaries,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        with (output_dir / "table5_scheduler_replay.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=SCHEDULER_TABLE_FIELDS)
            writer.writeheader()
            writer.writerows(scheduler_rows)
        write_scheduler_markdown(
            output_dir / "table5_scheduler_replay.md", scheduler_rows
        )

    manifest = {
        "schema": "pvs-glb-streaming-paper-output-v1",
        "split": "test",
        "testRead": True,
        "decisionModes": [
            DECISION_MODE_THRESHOLD_FREE_RANKING,
            DECISION_MODE_THRESHOLD_FILTERING,
            DECISION_MODE_SCHEDULER_REPLAY,
        ],
        "rankingSummaries": [str(path) for path, _summary in summaries],
        "filterSummaries": [str(path.expanduser().resolve()) for path in args.filter_summary],
        "schedulerReplaySummaries": [row["path"] for row in scheduler_summaries],
        "rankingTable": "table5_streaming_ranking.csv",
        "rankingTableMarkdown": "table5_streaming_ranking.md",
        "curveSource": "streaming_coverage_curve.csv",
        "figures": [
            "streaming_coverage_curve.png",
            "streaming_coverage_curve.pdf",
            "streaming_coverage_curve.svg",
        ],
        "schedulerReplayInputs": "scheduler_replay_inputs.json" if scheduler_summaries else None,
        "schedulerReplayTable": "table5_scheduler_replay.csv" if scheduler_summaries else None,
        "schedulerReplayTableMarkdown": "table5_scheduler_replay.md" if scheduler_summaries else None,
        "notes": [
            "Ranking is threshold-free and uses the complete per-pose candidate GLB set.",
            "Filtering is emitted separately and uses frozen score thresholds.",
            "Reference-frontmost pixels are a front-surface utility proxy, not hidden-surface coverage or a complete download utility.",
            "Scheduler cold-start values add startup visibility-asset transfer time and are network lower bounds, not measured browser first-frame rendering times.",
        ],
    }
    (output_dir / "paper_output_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
