#!/usr/bin/env python3
"""Convert a complete HZB Region66 test result to a formal-v2 image manifest."""
from __future__ import annotations

import argparse
import array
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from .instance_id_render_schema import (
        FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA,
        PREDICTION_COMPONENT_IDS_BY_KEY_FIELD,
        PREDICTION_KEY_FIELD,
        validate_formal_instance_render_manifest,
    )
    from .run_test_image_evaluation import validate_test_manifest
except ImportError:
    from instance_id_render_schema import (
        FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA,
        PREDICTION_COMPONENT_IDS_BY_KEY_FIELD,
        PREDICTION_KEY_FIELD,
        validate_formal_instance_render_manifest,
    )
    from run_test_image_evaluation import validate_test_manifest


HZB_RESULT_SCHEMA = "geometry-shell-hzb-browser-result-v1"
HZB_WORKLOAD_SCHEMA = "geometry-shell-hzb-browser-workload-v1"
HZB_REGION_SCHEMA = "geometry-shell-hzb-region-sampling-v1"

def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing JSON input: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer: {value!r}")
    return value


def _ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of component IDs")
    result = [_int(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate component IDs")
    return result


def _base_rows(base: dict[str, Any]) -> dict[int, str]:
    rows: dict[int, str] = {}
    for index, sample in enumerate(base.get("samples") or []):
        row = _int(sample.get("viewcellRow"), f"base sample {index} viewcellRow")
        key = sample.get(PREDICTION_KEY_FIELD)
        if not isinstance(key, str) or not key:
            raise ValueError(f"base sample {index} is missing {PREDICTION_KEY_FIELD}")
        if row in rows and rows[row] != key:
            raise ValueError(f"base view-cell row {row} uses multiple prediction keys")
        rows[row] = key
    if not rows:
        raise ValueError("base formal manifest must contain view-cell samples")
    return rows

def _hzb_rows(result: dict[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if result.get("schema") != HZB_RESULT_SCHEMA:
        raise ValueError(f"unsupported HZB result schema: {result.get('schema')!r}")
    if result.get("mode") != "Region66":
        raise ValueError("HZB image conversion requires mode=Region66")
    if result.get("error"):
        raise ValueError("cannot convert an HZB result that records an execution error")
    workload = result.get("workload")
    if (
        not isinstance(workload, dict)
        or workload.get("schema") != HZB_WORKLOAD_SCHEMA
        or workload.get("split") != "test"
    ):
        raise ValueError("HZB Region66 result must contain a test browser workload")
    samples = result.get("samples")
    if not isinstance(samples, list) or not samples or any(not isinstance(row, dict) for row in samples):
        raise ValueError("HZB result must contain object samples")
    rows: dict[int, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        pose_id = _int(sample.get("poseId"), f"HZB sample {index} poseId")
        if pose_id in rows:
            raise ValueError(f"HZB result has duplicate poseId: {pose_id}")
        rows[pose_id] = sample
    if workload.get("poseCount") != len(rows):
        raise ValueError("HZB workload poseCount does not match result samples")
    return workload, rows

def _read_array(path: Path, typecode: str, label: str) -> array.array:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {label}: {path}") from error
    values = array.array(typecode)
    if len(raw) % values.itemsize:
        raise ValueError(f"{label} has a truncated element: {path}")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values

def _candidate_rows(dataset_dir: Path, pose_ids: Sequence[int]) -> dict[int, list[int]]:
    offsets = _read_array(dataset_dir / "candidate_offsets.bin", "Q", "candidate offsets")
    values = _read_array(dataset_dir / "candidate_ids.bin", "I", "candidate IDs")
    if not offsets or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("candidate CSR offsets do not match candidate IDs")
    if any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("candidate CSR offsets must be monotonic")
    rows: dict[int, list[int]] = {}
    for pose_id in pose_ids:
        if pose_id + 1 >= len(offsets):
            raise ValueError(f"candidate CSR has no row for HZB poseId {pose_id}")
        row = [int(value) for value in values[offsets[pose_id]:offsets[pose_id + 1]]]
        if len(row) != len(set(row)):
            raise ValueError(f"candidate CSR poseId {pose_id} contains duplicate component IDs")
        rows[pose_id] = row
    return rows

def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and non-negative") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result

def _baseline(
    result: dict[str, Any], workload: dict[str, Any], rows: dict[int, dict[str, Any]],
    asset_variant: str, source_result: str | Path,
) -> dict[str, Any]:
    if not isinstance(asset_variant, str) or not asset_variant.strip():
        raise ValueError("assetVariant must be supplied as a non-empty string")
    width = _int(workload.get("width"), "workload.width")
    height = _int(workload.get("height"), "workload.height")
    if not width or not height:
        raise ValueError("workload.width and workload.height must be positive")
    depth_biases = []
    for pose_id, sample in rows.items():
        timings = sample.get("timings")
        if not isinstance(timings, dict):
            raise ValueError(f"HZB poseId {pose_id} is missing timings")
        depth_biases.append(_number(timings.get("depthBiasM"), f"poseId {pose_id} timings.depthBiasM"))
    if any(value != depth_biases[0] for value in depth_biases[1:]):
        raise ValueError("HZB samples record different depthBiasM values")
    region_sampling = result.get("regionSampling")
    if not isinstance(region_sampling, dict) or region_sampling.get("schema") != HZB_REGION_SCHEMA:
        raise ValueError("HZB result is missing regionSampling metadata")
    region_count = _int(region_sampling.get("requestedCount"), "regionSampling.requestedCount")
    return {
        "method": "geometry-shell-hzb",
        "selectionSplit": "calibration",
        "testRead": False,
        "assetVariant": asset_variant.strip(),
        "resolution": [width, height],
        "depthBiasM": depth_biases[0],
        "regionSampleCount": region_count,
        "sourceResult": Path(source_result).name,
    }

def build_hzb_image_manifest(
    base_manifest: dict[str, Any], hzb_result: dict[str, Any], *,
    candidate_dataset_dir: Path, asset_variant: str, source_result: str | Path,
) -> dict[str, Any]:
    """Replace keyed predictions while retaining the base camera/reference contract."""
    validate_test_manifest(base_manifest)
    if base_manifest.get("schema") != FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA:
        raise ValueError("base image manifest must use formal-render-manifest-v2")
    validate_formal_instance_render_manifest(base_manifest)
    base_rows = _base_rows(base_manifest)
    workload, hzb_rows = _hzb_rows(hzb_result)
    missing = sorted(set(base_rows) - set(hzb_rows))
    extra = sorted(set(hzb_rows) - set(base_rows))
    if missing or extra:
        raise ValueError(f"HZB poseIds do not exactly cover base viewcellRow: missing={missing}, extra={extra}")

    bindings = base_manifest.get("instanceBindings") or {}
    component_to_binding = bindings.get("componentToBinding")
    if not isinstance(component_to_binding, dict):
        raise ValueError("base formal manifest has no complete component binding table")
    try:
        component_ids = {int(key) for key in component_to_binding}
    except (TypeError, ValueError) as error:
        raise ValueError("base component binding keys must be component IDs") from error
    candidates = _candidate_rows(candidate_dataset_dir, sorted(hzb_rows))
    predictions: dict[str, list[int]] = {}
    for pose_id in sorted(hzb_rows):
        sample = hzb_rows[pose_id]
        candidate = candidates[pose_id]
        unknown_candidates = sorted(set(candidate) - component_ids)
        if unknown_candidates:
            raise ValueError(f"HZB poseId {pose_id} candidate IDs are outside the instance range: {unknown_candidates}")
        if _int(sample.get("candidateCount"), f"HZB poseId {pose_id} candidateCount") != len(candidate):
            raise ValueError(f"HZB poseId {pose_id} candidateCount does not match candidate IDs")
        predicted = _ids(sample.get("visibleInstanceIds"), f"HZB poseId {pose_id} visibleInstanceIds")
        unknown = sorted(set(predicted) - component_ids)
        if unknown:
            raise ValueError(f"HZB poseId {pose_id} predicts component IDs outside the instance range: {unknown}")
        outside = sorted(set(predicted) - set(candidate))
        if outside:
            raise ValueError(f"HZB poseId {pose_id} predicts component IDs outside its candidate set: {outside}")
        key = base_rows[pose_id]
        if key in predictions and predictions[key] != predicted:
            raise ValueError(f"predictionKey {key!r} maps to conflicting HZB predictions")
        predictions[key] = predicted

    output = copy.deepcopy(base_manifest)
    for field in ("threshold", "thresholdSelection", "thresholdProvenance"):
        output.pop(field, None)
    output["baselineSelection"] = _baseline(hzb_result, workload, hzb_rows, asset_variant, source_result)
    output[PREDICTION_COMPONENT_IDS_BY_KEY_FIELD] = predictions
    validate_test_manifest(output)
    validate_formal_instance_render_manifest(output)
    return output

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--hzb-result", type=Path, required=True)
    parser.add_argument("--candidate-dataset-dir", type=Path, required=True)
    parser.add_argument("--asset-variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result_path = args.hzb_result.resolve()
    output = build_hzb_image_manifest(
        read_json(args.base_manifest.resolve()), read_json(result_path),
        candidate_dataset_dir=args.candidate_dataset_dir.resolve(),
        asset_variant=args.asset_variant, source_result=result_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(output['samples'])} samples)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"build_hzb_image_manifest: {error}", file=sys.stderr)
        raise
