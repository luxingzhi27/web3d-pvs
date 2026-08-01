#!/usr/bin/env python3
"""Run several M5 component-ID image manifests in one browser page.

The ordinary image evaluator already batches samples from one invocation. This
runner extends that boundary across independently produced validation or
calibration manifests, for example manifests produced with different frozen
workpoints. It never predicts IDs, scans thresholds, or changes a threshold;
it only validates and combines existing component-level prediction samples.

Each ``--input`` is ``BATCH_ID=PATH_TO_V2_MANIFEST``. All inputs must refer to
the same local scene inventory, binding table, resolution, and 60/66 degree
camera contract. The Node renderer then loads each selected GLB once in one
page and renders all batches sequentially.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from instance_id_render_schema import (
    INSTANCE_BINDING_SCHEMA,
    INSTANCE_ID_ENCODING,
    INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA,
    INSTANCE_RENDER_MANIFEST_SCHEMA,
    MODEL_INPUT_FOV_Y_DEG,
    RENDER_FOV_Y_DEG,
    InstanceBindingError,
    validate_instance_render_batch_manifest,
    validate_instance_render_manifest,
)


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_RENDERER = BENCHMARK_DIR / "render_local_glb_color_id_browser.mjs"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return value


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_input_spec(value: str) -> tuple[str, Path]:
    batch_id, separator, raw_path = value.partition("=")
    if not separator:
        path = Path(value).expanduser().resolve()
        batch_id = path.stem
    else:
        batch_id = batch_id.strip()
        path = Path(raw_path).expanduser().resolve()
    if not batch_id:
        raise ValueError(f"empty batch id in --input {value!r}")
    if not path.exists():
        raise FileNotFoundError(path)
    return batch_id, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch existing M5 component-ID image manifests in one browser page."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=False,
        help="BATCH_ID=path/to/local-true-component-id-render-manifest-v2.json; repeat for validation/calibration batches.",
    )
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--renderer-script", type=Path, default=DEFAULT_RENDERER)
    parser.add_argument("--chrome-exe", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument("--save-id-buffers", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def minimal_sample(sample_id: str) -> dict[str, Any]:
    return {
        "sampleId": sample_id,
        "cameraPosition": [0.0, 0.0, 0.0],
        "cameraForward": [0.0, 0.0, -1.0],
        "renderFovYDeg": RENDER_FOV_Y_DEG,
        "modelInputFovYDeg": MODEL_INPUT_FOV_Y_DEG,
        "aspect": 16.0 / 9.0,
        "predictionComponentIds": [0],
    }


def self_test() -> None:
    binding = {
        "schema": INSTANCE_BINDING_SCHEMA,
        "byGlobalGlbId": {
            "0": {
                "globalGlbId": 0,
                "componentGlobalIds": [0],
                "instanceCount": 1,
                "renderable": True,
            }
        },
        "componentToBinding": {
            "0": {"componentGlobalId": 0, "globalGlbId": 0, "instanceIndex": 0}
        },
    }
    base = {
        "schema": INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA,
        "idEncoding": INSTANCE_ID_ENCODING,
        "renderFovYDeg": RENDER_FOV_Y_DEG,
        "modelInputFovYDeg": MODEL_INPUT_FOV_Y_DEG,
        "formalImageEvaluationReady": False,
        "selectedGlbs": [0],
        "reference": {"mode": "full_scene_renderable_instances", "idSource": "componentGlobalId"},
        "prediction": {"field": "predictionComponentIds"},
        "instanceBindings": binding,
        "batches": [
            {"batchId": "validation", "samples": [minimal_sample("validation-0")]},
            {"batchId": "calibration", "samples": [minimal_sample("calibration-0")]},
        ],
    }
    validate_instance_render_batch_manifest(base)
    duplicate = json.loads(json.dumps(base))
    duplicate["batches"][1]["samples"][0]["sampleId"] = "validation-0"
    try:
        validate_instance_render_batch_manifest(duplicate)
    except InstanceBindingError:
        pass
    else:
        raise AssertionError("duplicate sample IDs across batches were not rejected")
    print("M5 component-image batch self-test: PASS")


def load_and_validate_inputs(input_specs: list[str]) -> list[tuple[str, Path, dict[str, Any]]]:
    if not input_specs:
        raise ValueError("at least one --input is required unless --self-test is used")
    loaded: list[tuple[str, Path, dict[str, Any]]] = []
    seen_batch_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    for raw in input_specs:
        batch_id, path = parse_input_spec(raw)
        if batch_id in seen_batch_ids:
            raise ValueError(f"duplicate batch id: {batch_id}")
        seen_batch_ids.add(batch_id)
        manifest = read_json(path)
        if manifest.get("schema") != INSTANCE_RENDER_MANIFEST_SCHEMA:
            raise InstanceBindingError(
                f"{path} must be {INSTANCE_RENDER_MANIFEST_SCHEMA}; nested batch manifests are not accepted"
            )
        validate_instance_render_manifest(manifest)
        for sample in manifest.get("samples") or []:
            sample_id = str(sample.get("sampleId") or "")
            if sample_id in seen_sample_ids:
                raise InstanceBindingError(f"duplicate sampleId across input manifests: {sample_id}")
            seen_sample_ids.add(sample_id)
        loaded.append((batch_id, path, manifest))
    if not loaded or not any(manifest.get("samples") for _batch_id, _path, manifest in loaded):
        raise ValueError("input manifests contain no samples")
    return loaded


def assert_same_scene_contract(loaded: list[tuple[str, Path, dict[str, Any]]]) -> None:
    reference = loaded[0][2]
    fields = (
        "width",
        "height",
        "renderFovYDeg",
        "modelInputFovYDeg",
        "glbRoot",
        "glbIndex",
        "runtimeMeta",
        "idEncoding",
        "selectedGlbs",
        "reference",
        "prediction",
        "instanceBindings",
    )
    reference_values = {
        field: sorted(reference[field]) if field == "selectedGlbs" else reference.get(field)
        for field in fields
    }
    reference_digests = {field: canonical_digest(value) for field, value in reference_values.items()}
    for _batch_id, path, manifest in loaded[1:]:
        for field in fields:
            value = sorted(manifest[field]) if field == "selectedGlbs" else manifest.get(field)
            if canonical_digest(value) != reference_digests[field]:
                raise InstanceBindingError(
                    f"input manifest {path} changes the shared M5 scene contract field {field}"
                )


def build_batch_manifest(loaded: list[tuple[str, Path, dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    assert_same_scene_contract(loaded)
    first = loaded[0][2]
    batches = []
    for batch_id, path, manifest in loaded:
        batches.append(
            {
                "batchId": batch_id,
                "sourceManifest": str(path),
                "sourceThreshold": manifest.get("threshold"),
                "sourceModelName": manifest.get("modelName"),
                "samples": manifest.get("samples") or [],
            }
        )
    result = {
        "schema": INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA,
        "created": datetime.now().isoformat(timespec="seconds"),
        "width": int(first["width"]),
        "height": int(first["height"]),
        "renderFovYDeg": float(first["renderFovYDeg"]),
        "modelInputFovYDeg": float(first["modelInputFovYDeg"]),
        "glbRoot": first["glbRoot"],
        "glbIndex": first["glbIndex"],
        "runtimeMeta": first.get("runtimeMeta"),
        "selectedGlbs": sorted(int(value) for value in first["selectedGlbs"]),
        "reference": first["reference"],
        "prediction": first["prediction"],
        "instanceBindings": first["instanceBindings"],
        "batches": batches,
        "previewSamples": max(0, int(args.preview_samples)),
        "saveIdBuffers": bool(args.save_id_buffers),
        "localOnly": True,
        "idEncoding": INSTANCE_ID_ENCODING,
        "formalImageEvaluationReady": False,
        "thresholdPolicy": "inputs_are_frozen_predictions; runner_does_not_scan_or_modify_thresholds",
        "assetReuseContract": {
            "browserPageCount": 1,
            "glbLoadPasses": 1,
            "reuseScope": "all batches and samples in this manifest",
        },
    }
    validate_instance_render_batch_manifest(result)
    return result


def run_renderer(batch_manifest_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        "node",
        str(args.renderer_script.resolve()),
        "--manifest",
        str(batch_manifest_path),
        "--output-dir",
        str(output_dir / "true_glb_render"),
        "--timeout-ms",
        str(max(30, int(args.timeout_sec)) * 1000),
    ]
    if args.validate_only:
        cmd.append("--validate-only")
    if args.chrome_exe is not None:
        cmd.extend(["--chrome-exe", str(args.chrome_exe)])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=max(30, int(args.timeout_sec)) + 30)
    (output_dir / "renderer_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "renderer_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            "M5 batch browser renderer failed; see renderer_stdout.log and renderer_stderr.log; "
            f"return code={proc.returncode}"
        )
    summary_path = output_dir / "true_glb_render" / "render_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return read_json(summary_path)


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    loaded = load_and_validate_inputs(args.input or [])
    output_dir = (args.output_dir or (BENCHMARK_DIR / "out" / "m5_component_image_batch")).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_manifest = build_batch_manifest(loaded, args)
    batch_manifest_path = output_dir / "batch_manifest.json"
    batch_manifest_path.write_text(json.dumps(batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    render_summary = run_renderer(batch_manifest_path, output_dir, args)
    summary = {
        "schema": "m5-component-image-batch-runner-summary-v1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "batchManifest": str(batch_manifest_path),
        "inputManifests": [
            {"batchId": batch_id, "path": str(path), "sampleCount": len(manifest.get("samples") or [])}
            for batch_id, path, manifest in loaded
        ],
        "batchCount": len(loaded),
        "sampleCount": sum(len(manifest.get("samples") or []) for _batch_id, _path, manifest in loaded),
        "thresholdsUntouched": True,
        "renderSummary": render_summary,
        "assetReuse": render_summary.get("assetReuse"),
        "cameraContract": {"renderFovYDeg": RENDER_FOV_Y_DEG, "modelInputFovYDeg": MODEL_INPUT_FOV_Y_DEG},
        "formalTest": False,
    }
    (output_dir / "runner_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
