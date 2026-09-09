#!/usr/bin/env python3
"""Run one frozen-test formal-v2 image manifest through the existing renderer."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .instance_id_render_schema import (  # type: ignore[import-not-found]
        FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA,
    )
except ImportError:
    from instance_id_render_schema import (  # type: ignore[no-redef]
        FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENDERER = ROOT / "benchmark" / "render_local_glb_color_id_browser.mjs"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing JSON input: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON input {path}: {error}") from error


def validate_test_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Check only frozen-test fields; the renderer owns the full schema check."""
    if manifest.get("schema") != FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA:
        raise ValueError(
            "test image input must use "
            f"{FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA}"
        )
    if manifest.get("split") != "test":
        raise ValueError("formal test image manifest must declare split=test")
    if manifest.get("testRead") is not True:
        raise ValueError("formal test image manifest must declare testRead=true")
    if manifest.get("testEvaluationCount") != 1:
        raise ValueError("formal test image manifest must declare testEvaluationCount=1")

    has_threshold = "thresholdSelection" in manifest
    has_baseline = "baselineSelection" in manifest
    if has_threshold == has_baseline:
        raise ValueError(
            "formal test image manifest must contain exactly one of "
            "thresholdSelection or baselineSelection"
        )
    if has_threshold:
        threshold = manifest.get("thresholdSelection")
        if not isinstance(threshold, dict):
            raise ValueError("formal test image manifest is missing thresholdSelection")
        if threshold.get("selectionSplit") != "calibration":
            raise ValueError("formal test image threshold must come from calibration")
        if threshold.get("testRead") is not False:
            raise ValueError("calibration threshold provenance must be recorded before test")
        provenance = manifest.get("thresholdProvenance")
        if not isinstance(provenance, dict):
            raise ValueError("formal test image manifest is missing thresholdProvenance")
        if provenance.get("selectionSplit") != "calibration" or provenance.get("testRead") is not False:
            raise ValueError("thresholdProvenance must identify pre-test calibration")
        selection_method = "threshold"
    else:
        if any(field in manifest for field in ("threshold", "thresholdProvenance")):
            raise ValueError("baseline test image manifest cannot contain neural threshold fields")
        baseline = manifest.get("baselineSelection")
        if not isinstance(baseline, dict):
            raise ValueError("formal test image manifest is missing baselineSelection")
        if baseline.get("method") != "geometry-shell-hzb":
            raise ValueError("formal test image baseline must use geometry-shell-hzb")
        if baseline.get("selectionSplit") != "calibration":
            raise ValueError("formal test image baseline must come from calibration")
        if baseline.get("testRead") is not False:
            raise ValueError("calibration baseline provenance must be recorded before test")
        for field in ("assetVariant", "resolution", "depthBiasM", "regionSampleCount", "sourceResult"):
            if field not in baseline:
                raise ValueError(f"formal test image baseline is missing {field}")
        if not isinstance(baseline["assetVariant"], str) or not baseline["assetVariant"]:
            raise ValueError("formal test image baseline assetVariant must be non-empty")
        resolution = baseline["resolution"]
        if isinstance(resolution, dict):
            resolution = [resolution.get("width"), resolution.get("height")]
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resolution)
        ):
            raise ValueError("formal test image baseline resolution must contain positive width and height")
        try:
            depth_bias = float(baseline["depthBiasM"])
        except (TypeError, ValueError) as error:
            raise ValueError("formal test image baseline depthBiasM must be finite and non-negative") from error
        if not math.isfinite(depth_bias) or depth_bias < 0:
            raise ValueError("formal test image baseline depthBiasM must be finite and non-negative")
        if (
            isinstance(baseline["regionSampleCount"], bool)
            or not isinstance(baseline["regionSampleCount"], int)
            or baseline["regionSampleCount"] < 0
        ):
            raise ValueError("formal test image baseline regionSampleCount must be non-negative")
        if not isinstance(baseline["sourceResult"], str) or not baseline["sourceResult"]:
            raise ValueError("formal test image baseline sourceResult must be non-empty")
        selection_method = "geometry-shell-hzb"

    coverage = manifest.get("testCoverage")
    if not isinstance(coverage, dict):
        raise ValueError("formal test image manifest is missing testCoverage")
    if coverage.get("split") != "test":
        raise ValueError("testCoverage must declare split=test")
    if coverage.get("selection") != "all_unique_test_viewcells":
        raise ValueError("formal test image evaluation must use all unique test view-cells")
    if coverage.get("sampledWithReplacement") is not False:
        raise ValueError("formal test image evaluation cannot sample with replacement")
    if coverage.get("maxViewcells") != 0:
        raise ValueError("formal test image evaluation cannot truncate view-cells")
    if coverage.get("subposesPerViewcell") != 0:
        raise ValueError("formal test image evaluation must use all dense subposes")

    subposes = manifest.get("subposeSelection")
    if not isinstance(subposes, dict) or subposes.get("mode") != "all":
        raise ValueError("formal test image manifest must select all dense subposes")
    if subposes.get("requestedPerViewcell") != 0:
        raise ValueError("formal test image manifest cannot select a subpose subset")
    return {
        "schema": "pvs-frozen-test-image-selection-v1",
        "manifestSchema": manifest["schema"],
        "split": "test",
        "testRead": True,
        "testEvaluationCount": 1,
        "selectionSplit": "calibration",
        "selectionMethod": selection_method,
        "viewcellCount": coverage.get("viewcellCount"),
        "sampleCount": coverage.get("sampleCount"),
    }


def renderer_command(
    manifest: Path,
    output_dir: Path,
    renderer: Path,
    chrome_exe: Path | None,
    schema_only: bool,
    timeout_ms: int | None,
) -> list[str]:
    command = [
        "node",
        str(renderer.resolve()),
        "--manifest",
        str(manifest.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--require-hardware-gpu",
    ]
    if schema_only:
        command.append("--validate-only")
    if timeout_ms is not None:
        command.extend(["--timeout-ms", str(timeout_ms)])
    if chrome_exe is not None:
        command.extend(["--chrome-exe", str(chrome_exe.resolve())])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer-script", type=Path, default=DEFAULT_RENDERER)
    parser.add_argument("--chrome-exe", type=Path, default=None)
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument(
        "--require-hardware-gpu",
        action="store_true",
        help="pass the formal renderer's hardware-GPU gate",
    )
    parser.add_argument(
        "--render-schema-only",
        action="store_true",
        help="run the existing renderer validator without starting Chrome",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout_ms is not None and args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")
    if not args.require_hardware_gpu:
        raise ValueError("formal test image evaluation requires --require-hardware-gpu")
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"formal test image manifest must be an object: {manifest_path}")
    validation = validate_test_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = renderer_command(
        manifest_path,
        output_dir,
        args.renderer_script,
        args.chrome_exe,
        args.render_schema_only,
        args.timeout_ms,
    )
    print(
        json.dumps(
            {"manifest": str(manifest_path), "validation": validation, "command": command},
            ensure_ascii=False,
            indent=2,
        )
    )
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"formal image renderer failed with exit code {result.returncode}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"run_test_image_evaluation: {error}", file=sys.stderr)
        raise
