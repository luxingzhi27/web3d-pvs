#!/usr/bin/env python3
"""Freeze one Geometry-shell HZB configuration from calibration results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SELECTION_SCHEMA = "geometry-shell-hzb-calibration-selection-v1"
METRICS_SCHEMA = "geometry-shell-hzb-metrics-v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _candidate(path: Path) -> dict[str, Any]:
    metrics = _read_json(path.resolve())
    if metrics.get("schema") != METRICS_SCHEMA:
        raise ValueError(f"unsupported HZB metrics schema: {path}")
    if metrics.get("split") != "calibration" or metrics.get("mode") != "Region66":
        raise ValueError(f"HZB selection requires Region66 calibration metrics: {path}")
    if metrics.get("formalReady") is not True or metrics.get("executionClass") != "formal-hardware-gpu":
        raise ValueError(f"HZB calibration result is not a formal hardware run: {path}")
    gate = metrics.get("gpuGate")
    adapter = gate.get("adapter") if isinstance(gate, Mapping) else None
    concurrency = metrics.get("gpuConcurrency")
    if (
        not isinstance(gate, Mapping)
        or gate.get("required") is not True
        or gate.get("hardware") is not True
        or not isinstance(adapter, Mapping)
        or "nvidia" not in str(adapter.get("vendor", "")).lower()
        or not isinstance(concurrency, Mapping)
        or concurrency.get("concurrentComputeDetected") is not False
    ):
        raise ValueError(f"HZB calibration hardware evidence is incomplete: {path}")

    result_path = Path(str(metrics.get("sourceResult", "")))
    if not result_path.is_absolute():
        result_path = path.parent / result_path
    result_path = result_path.resolve()
    result = _read_json(result_path)
    if result.get("formalReady") is not True or result.get("executionClass") != "formal-hardware-gpu":
        raise ValueError(f"HZB source result is not formal: {result_path}")
    workload = result.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError(f"HZB source result has no workload: {result_path}")
    shell_dir = Path(str(result.get("shellDir", ""))).resolve()
    shell_meta = _read_json(shell_dir / "shell_meta.json")
    aggregate = metrics.get("aggregate")
    timing = metrics.get("timing")
    if not isinstance(aggregate, Mapping) or not isinstance(timing, Mapping):
        raise ValueError(f"HZB metrics are incomplete: {path}")
    weighted_recall = _number(metrics.get("weightedRecall"), "weightedRecall")
    lower_bound = _number(metrics.get("weightedRecallLower95"), "weightedRecallLower95")
    useful_cull = _number(aggregate.get("usefulCull"), "aggregate.usefulCull")
    row = {
        "metrics": str(path.resolve()),
        "result": str(result_path),
        "assetVariant": shell_meta.get("variant"),
        "shellDir": str(shell_dir),
        "width": int(workload["width"]),
        "height": int(workload["height"]),
        "depthBiasM": _number(workload.get("depthBiasM"), "workload.depthBiasM"),
        "regionSampleCount": int(
            (workload.get("regionSampling") or {}).get("requestedCount", -1)
        ),
        "weightedRecall": weighted_recall,
        "weightedRecallLower95": lower_bound,
        "usefulCull": useful_cull,
        "badCull": _number(aggregate.get("badCull"), "aggregate.badCull"),
        "balancedAccuracy": _number(
            aggregate.get("balancedAccuracy"), "aggregate.balancedAccuracy"
        ),
        "specificity": _number(aggregate.get("specificity"), "aggregate.specificity"),
        "precision": _number(aggregate.get("precision"), "aggregate.precision"),
        "totalP50Ms": _number(timing.get("totalP50Ms"), "timing.totalP50Ms"),
        "safe": weighted_recall > 0.99 and lower_bound > 0.99,
    }
    if row["regionSampleCount"] != 0:
        raise ValueError("HZB calibration must use all available region subposes")
    return row


def select(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one HZB calibration result is required")
    rows = [_candidate(path) for path in paths]
    safe = [row for row in rows if row["safe"]]
    if safe:
        selected = max(
            safe,
            key=lambda row: (
                row["usefulCull"],
                row["balancedAccuracy"],
                row["specificity"],
                row["precision"],
                -row["totalP50Ms"],
            ),
        )
        status = "safe"
    else:
        selected = max(
            rows,
            key=lambda row: (
                row["weightedRecallLower95"],
                row["weightedRecall"],
                row["usefulCull"],
                row["balancedAccuracy"],
                -row["totalP50Ms"],
            ),
        )
        status = "no_qualified_safety_workpoint"
    return {
        "schema": SELECTION_SCHEMA,
        "selectionSplit": "calibration",
        "selectionRule": (
            "WR and one-sided 95% lower bound strictly above 0.99; "
            "then useful cull, balanced accuracy, specificity, precision and latency"
        ),
        "status": status,
        "selected": selected,
        "rows": rows,
        "testRead": False,
        "testEvaluationCount": 0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = select(args.candidate)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen HZB selection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
