#!/usr/bin/env python3
"""Select frozen-test samples for Reference|Prediction|Difference figures."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .run_test_image_evaluation import (  # type: ignore[import-not-found]
        read_json,
        validate_test_manifest,
    )
except ImportError:
    from run_test_image_evaluation import read_json, validate_test_manifest  # type: ignore[no-redef]


REGISTRATION_SCHEMA = "pvs-qualitative-pose-registration-v1"
SELECTION_SCHEMA = "pvs-qualitative-image-selection-v1"
ROLES = ("fine_component", "occlusion_boundary")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _load_rows(path: Path, manifest_samples: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError(f"sample image metrics must be a non-empty list: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("sampleId"), str):
            raise ValueError("every image metric row must contain sampleId")
        sample_id = item["sampleId"]
        if sample_id in seen or sample_id not in manifest_samples:
            raise ValueError(f"image metrics contain a duplicate or unknown sampleId: {sample_id}")
        metrics = item.get("imageMetrics")
        per = float(metrics["PER"]) if isinstance(metrics, dict) and "PER" in metrics else math.nan
        if not math.isfinite(per) or per < 0.0:
            raise ValueError(f"{sample_id} has an invalid imageMetrics.PER")
        seen.add(sample_id)
        rows.append({
            **manifest_samples[sample_id],
            "sampleId": sample_id,
            "PER": per,
            "errorPixels": metrics.get("errorPixels") if isinstance(metrics, dict) else None,
        })
    if seen != set(manifest_samples):
        raise ValueError("image metrics must cover every sample in the formal manifest")
    return rows


def _load_registration(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema") != REGISTRATION_SCHEMA:
        raise ValueError(f"registration must use {REGISTRATION_SCHEMA}")
    if value.get("registeredBeforeTest") is not True or value.get("selectionSplit") != "pre_test":
        raise ValueError("qualitative registration must be pre-test")
    if value.get("testRead") is not False or value.get("testEvaluationCount") != 0:
        raise ValueError("qualitative registration must precede test evaluation")
    poses = value.get("poses")
    if (
        not isinstance(poses, list)
        or len(poses) != len(ROLES)
        or {item.get("role") for item in poses if isinstance(item, dict)} != set(ROLES)
    ):
        raise ValueError("registration must contain one fine_component and one occlusion_boundary pose")
    return [dict(item) for item in poses]


def _registered(rows: list[dict[str, Any]], item: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in rows
        if (item.get("sampleId") is None or row["sampleId"] == item["sampleId"])
        and (item.get("poseIndex") is None or row.get("poseIndex") == item["poseIndex"])
    ]
    if len(matches) != 1:
        raise ValueError(f"registered pose does not identify one rendered sample: {item}")
    return matches[0]


def _selection(
    role: str,
    row: dict[str, Any],
    method: str,
    target: float | None = None,
    registration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_id = row["sampleId"]
    return {
        "role": role,
        "selectionMethod": method,
        "targetPER": target,
        "observedPER": row["PER"],
        "errorPixels": row.get("errorPixels"),
        "sampleId": sample_id,
        "poseIndex": row.get("poseIndex"),
        "viewcellRow": row.get("viewcellRow"),
        "predictionKey": row.get("predictionKey"),
        "images": {
            "Reference": f"samples/{sample_id}_reference_u32.bin",
            "Prediction": f"samples/{sample_id}_test_u32.bin",
            "Difference": f"samples/{sample_id}_diff_u8.bin",
        },
        "registration": registration,
    }


def select_qualitative_samples(
    render_summary_path: Path,
    sample_metrics_path: Path,
    manifest_path: Path,
    registration_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("formal test image manifest must be an object")
    validation = validate_test_manifest(manifest)
    samples = {str(item["sampleId"]): item for item in manifest.get("samples", [])}
    rows = _load_rows(sample_metrics_path, samples)
    values = [row["PER"] for row in rows]
    median, p95 = _percentile(values, 0.50), _percentile(values, 0.95)
    nearest = lambda target: min(rows, key=lambda row: (abs(row["PER"] - target), row["sampleId"]))
    selected = [
        _selection("median", nearest(median), "nearest_linear_percentile", median),
        _selection("p95", nearest(p95), "nearest_linear_percentile", p95),
        _selection("max_error", min(rows, key=lambda row: (-row["PER"], row["sampleId"])), "maximum_PER"),
    ]
    for item in _load_registration(registration_path):
        selected.append(_selection(item["role"], _registered(rows, item), "pre_registered_pose", registration=item))
    result = {
        "schema": SELECTION_SCHEMA,
        "split": "test",
        "testRead": True,
        "testEvaluationCount": 1,
        "errorMetric": "PER",
        "sampleCount": len(rows),
        "viewcellCount": validation.get("viewcellCount"),
        "percentiles": {"median": median, "p95": p95, "maximum": max(values)},
        "source": {
            "renderSummary": str(render_summary_path.resolve()),
            "sampleImageMetrics": str(sample_metrics_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "registration": str(registration_path.resolve()),
        },
        "selections": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-summary", type=Path, required=True)
    parser.add_argument("--sample-metrics", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = args.render_summary.resolve()
    metrics = (args.sample_metrics or summary.parent / "sample_image_metrics.json").resolve()
    result = select_qualitative_samples(summary, metrics, args.manifest.resolve(), args.registration.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
