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

from glb_streaming import FIXED_RANDOM_METHODS, RANKING_METHOD_LABELS


TABLE_FIELDS = [
    "scene",
    "method",
    "label",
    "status",
    "decision_mode",
    "pose_count",
    "coverage_source",
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


def number_mean(method: dict[str, Any], field: str) -> float | None:
    value = method.get(field)
    if not isinstance(value, dict):
        return None
    return value.get("mean")


def stat_mean(value: Any) -> float | None:
    return value.get("mean") if isinstance(value, dict) else None


def summary_row(summary: dict[str, Any], source_path: Path, method_name: str, method: dict[str, Any]) -> dict[str, Any]:
    filtering = method.get("decisionMode") == "threshold_filtering"
    row: dict[str, Any] = {
        "scene": scene_name(summary, source_path),
        "method": method_name,
        "label": method.get("label", RANKING_METHOD_LABELS.get(method_name, method_name)),
        "status": method.get("status", "unavailable"),
        "decision_mode": method.get("decisionMode", "threshold_free_ranking"),
        "pose_count": method.get("poseCount", 0),
        "coverage_source": summary.get("coverageSource", "unknown"),
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
        "coverage_upper_bound_mean": number_mean(method, "coverageUpperBound"),
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
        ("Mode", "decision_mode"),
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
        "Ranking rows use the complete fixed candidate GLB set and no visibility threshold. `NA` means the method was unavailable.",
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
                    "mean_coverage": point.get("meanCoverage"),
                }
            )
    return rows


def coverage_axis_label(summary: dict[str, Any]) -> str:
    source = str(summary.get("coverageSource", ""))
    if source == "reference-frontmost-pixel-histogram-v1":
        return "Reference-frontmost pixel coverage (%)"
    if source == "visible_weights_utility_not_pixel_coverage":
        return "Visible-weight utility coverage (%)"
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
        "hzb_visible_first",
        "gt_utility_per_byte_oracle",
    ]
    colors = {
        "full": "#1f77b4",
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
            if not isinstance(method, dict) or method.get("status") != "available":
                continue
            points = method.get("coverageCurve", [])
            if not points:
                continue
            x = [float(point.get("meanBytes", 0.0)) / (1024 * 1024) for point in points]
            y = [float(point.get("meanCoverage", 0.0)) * 100.0 for point in points]
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
        axis.set_title(scene)
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
        for method_name, method in summary.get("methods", {}).items():
            filter_rows.append(summary_row(summary, path, method_name, method))
    if filter_rows:
        write_csv(output_dir / "table5_streaming_filtering.csv", filter_rows)
        write_markdown(output_dir / "table5_streaming_filtering.md", filter_rows)

    manifest = {
        "schema": "pvs-glb-streaming-paper-output-v1",
        "rankingSummaries": [str(path) for path, _summary in summaries],
        "filterSummaries": [str(path.expanduser().resolve()) for path in args.filter_summary],
        "rankingTable": "table5_streaming_ranking.csv",
        "rankingTableMarkdown": "table5_streaming_ranking.md",
        "curveSource": "streaming_coverage_curve.csv",
        "figures": [
            "streaming_coverage_curve.png",
            "streaming_coverage_curve.pdf",
            "streaming_coverage_curve.svg",
        ],
        "notes": [
            "Ranking is threshold-free and uses the complete per-pose candidate GLB set.",
            "Filtering is emitted separately and uses frozen score thresholds.",
            "Reference-frontmost pixels are a front-surface utility proxy, not hidden-surface coverage or a complete download utility.",
        ],
    }
    (output_dir / "paper_output_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
