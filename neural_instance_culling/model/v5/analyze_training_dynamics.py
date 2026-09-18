#!/usr/bin/env python3
"""Summarize registered V5 risk and dual trajectories by constraint group."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METRICS = ("riskExtra", "riskCount", "riskVisual", "lambdaCount", "lambdaVisual")
REQUIRED_FIELDS = (
    "globalStep",
    "sceneId",
    "sourceKind",
    "dualGroupId",
    "dualGroupUpdate",
    *METRICS,
)


def read_metric_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is missing dynamics fields: {', '.join(missing)}"
                )
            values = [float(row[field]) for field in METRICS]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number} contains non-finite dynamics values")
            normalized = dict(row)
            normalized["run"] = path.parent.name
            rows.append(normalized)
    if not rows:
        raise ValueError("no V5 training metric rows were found")
    return rows


def bin_metric_rows(rows: Sequence[Mapping[str, Any]], bin_width: int) -> list[dict[str, Any]]:
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    grouped: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        update = int(row["dualGroupUpdate"])
        if update <= 0:
            raise ValueError("dualGroupUpdate must be positive")
        bin_index = (update - 1) // int(bin_width)
        key = (str(row["run"]), str(row["sourceKind"]), str(row["dualGroupId"]), bin_index)
        bucket = grouped.setdefault(
            key,
            {
                "run": key[0],
                "sourceKind": key[1],
                "dualGroupId": key[2],
                "dualGroupUpdateStart": bin_index * bin_width + 1,
                "dualGroupUpdateEnd": (bin_index + 1) * bin_width,
                "sampleCount": 0,
                **{field: 0.0 for field in METRICS},
            },
        )
        bucket["sampleCount"] += 1
        for field in METRICS:
            bucket[field] += float(row[field])
    result: list[dict[str, Any]] = []
    for bucket in grouped.values():
        count = int(bucket["sampleCount"])
        for field in METRICS:
            bucket[field] /= count
        result.append(bucket)
    return sorted(
        result,
        key=lambda row: (row["run"], row["sourceKind"], row["dualGroupId"], row["dualGroupUpdateStart"]),
    )


def summarize_groups(rows: Sequence[Mapping[str, Any]], bin_width: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups = sorted({(str(row["run"]), str(row["dualGroupId"])) for row in rows})
    for run, group_id in groups:
        selected = [row for row in rows if row["run"] == run and row["dualGroupId"] == group_id]
        first = selected[0]
        last = selected[-1]
        result[f"{run}:{group_id}"] = {
            "run": run,
            "dualGroupId": group_id,
            "sourceKind": first["sourceKind"],
            "binWidth": int(bin_width),
            "binCount": len(selected),
            "first": {field: float(first[field]) for field in METRICS},
            "last": {field: float(last[field]) for field in METRICS},
            "peak": {field: max(float(row[field]) for row in selected) for field in METRICS},
        }
    return {
        "schema": "gcof-pvs-v5-training-dynamics-v1",
        "metricFields": list(METRICS),
        "groups": result,
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "run",
        "sourceKind",
        "dualGroupId",
        "dualGroupUpdateStart",
        "dualGroupUpdateEnd",
        "sampleCount",
        *METRICS,
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def write_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    flat_axes = axes.reshape(-1)
    groups = sorted({(str(row["run"]), str(row["dualGroupId"])) for row in rows})
    for metric, axis in zip(METRICS, flat_axes):
        for run, group_id in groups:
            selected = [row for row in rows if row["run"] == run and row["dualGroupId"] == group_id]
            synthetic = str(selected[0]["sourceKind"]) == "synthetic"
            axis.plot(
                [row["dualGroupUpdateEnd"] for row in selected],
                [row[metric] for row in selected],
                linestyle="--" if synthetic else "-",
                linewidth=1.2,
                label=f"{run}:{group_id}",
            )
        axis.set_title(metric)
        axis.set_xlabel("dual-group local updates")
        axis.grid(alpha=0.25)
    flat_axes[-1].axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    flat_axes[-1].legend(handles, labels, loc="center", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-width", type=int, default=100)
    args = parser.parse_args(argv)
    raw = read_metric_rows(args.metrics)
    binned = bin_metric_rows(raw, int(args.bin_width))
    output = args.output_dir
    write_csv(binned, output / "training_dynamics_binned.csv")
    (output / "training_dynamics_summary.json").write_text(
        json.dumps(summarize_groups(binned, int(args.bin_width)), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_plot(binned, output / "training_dynamics.png")


if __name__ == "__main__":
    main()
