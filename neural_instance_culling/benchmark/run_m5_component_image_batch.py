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
IMAGE_COUNT_FIELDS = (
    "totalPixels",
    "validReferencePixels",
    "backgroundReferencePixels",
    "errorPixels",
    "missPixels",
    "wrongInstancePixels",
    "extraPixels",
)


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
    parser.add_argument("--chrome-arg", action="append", default=[])
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=0,
        help=(
            "run independent browser pages with at most this many samples per "
            "source batch; zero preserves the one-page runner"
        ),
    )
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


def load_spatial_bounds(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """Materialize audited world-space GLB and component bounds.

    This is a conservative renderer optimization.  It never changes the
    selected GLB inventory or a prediction component list: a GLB is omitted
    from one camera only when its complete world-space AABB is outside that
    camera's render frustum.
    """
    runtime_meta_path = Path(str(manifest.get("runtimeMeta") or "")).expanduser()
    if not runtime_meta_path.is_file():
        raise FileNotFoundError(f"M5 spatial renderer requires runtimeMeta: {runtime_meta_path}")
    runtime_meta = read_json(runtime_meta_path)
    records = {
        int(row["globalGlbId"]): row
        for row in (runtime_meta.get("globalGlbRecords") or [])
        if isinstance(row, dict) and "globalGlbId" in row
    }
    glb_result: dict[str, dict[str, list[float]]] = {}
    missing: list[int] = []
    for raw_gid in manifest.get("selectedGlbs") or []:
        gid = int(raw_gid)
        record = records.get(gid)
        aabb = record.get("aabb") if record else None
        if not isinstance(aabb, dict) or len(aabb.get("min") or []) != 3 or len(aabb.get("max") or []) != 3:
            missing.append(gid)
            continue
        minimum = [float(value) for value in aabb["min"]]
        maximum = [float(value) for value in aabb["max"]]
        if any(not (minimum[index] <= maximum[index]) for index in range(3)):
            raise ValueError(f"invalid runtime GLB AABB for globalGlbId={gid}")
        glb_result[str(gid)] = {"min": minimum, "max": maximum}
    if missing:
        raise ValueError(
            "runtimeMeta is missing conservative AABBs for selected GLBs; "
            f"first missing IDs: {missing[:16]}"
        )
    component_result: dict[str, dict[str, list[float]]] = {}
    for row in runtime_meta.get("componentRecords") or []:
        if not isinstance(row, dict) or "componentGlobalId" not in row:
            continue
        bounds = row.get("bounds")
        if not isinstance(bounds, dict):
            continue
        center = bounds.get("center")
        size = bounds.get("size")
        if not isinstance(center, list) or not isinstance(size, list) or len(center) != 3 or len(size) != 3:
            continue
        minimum = [float(center[index]) - float(size[index]) / 2.0 for index in range(3)]
        maximum = [float(center[index]) + float(size[index]) / 2.0 for index in range(3)]
        if any(not (minimum[index] <= maximum[index]) for index in range(3)):
            raise ValueError(f"invalid runtime component AABB for componentGlobalId={row['componentGlobalId']}")
        component_result[str(int(row["componentGlobalId"]))] = {"min": minimum, "max": maximum}
    return glb_result, component_result


def build_batch_manifest(loaded: list[tuple[str, Path, dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    assert_same_scene_contract(loaded)
    first = loaded[0][2]
    glb_aabbs, component_aabbs = load_spatial_bounds(first)
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
        "glbAabbs": glb_aabbs,
        "componentAabbs": component_aabbs,
        "spatialCulling": {
            "schema": "aabb-frustum-conservative-v1",
            "source": "runtimeMeta.globalGlbRecords[].aabb",
            "purpose": "render_submission_only",
            "renderFovYDeg": RENDER_FOV_Y_DEG,
            "near": 0.05,
            "far": 20000.0,
            "completeInventoryRetained": True,
        },
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
    for chrome_arg in args.chrome_arg:
        cmd.extend(["--chrome-arg", str(chrome_arg)])
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


def _empty_image_totals() -> dict[str, int]:
    return {field: 0 for field in IMAGE_COUNT_FIELDS}


def _merge_image_totals(target: dict[str, int], metrics: dict[str, Any]) -> None:
    for field in IMAGE_COUNT_FIELDS:
        target[field] += int(metrics.get(field, 0))


def _finalize_image_metrics(totals: dict[str, int], sample_count: int) -> dict[str, Any]:
    valid = max(1, totals["validReferencePixels"])
    total = max(1, totals["totalPixels"])
    return {
        "schema": "color-id-per-v1-aggregate",
        **totals,
        "PER": totals["errorPixels"] / valid,
        "missPixelRate": totals["missPixels"] / valid,
        "wrongInstancePixelRate": totals["wrongInstancePixels"] / valid,
        "extraPixelRateOverImage": totals["extraPixels"] / total,
        "evaluatedSubposeCount": sample_count,
        "renderFailedSubposeCount": 0,
        "missingGlbSubposeCount": 0,
    }


def build_chunk_manifests(
    batch_manifest: dict[str, Any],
    output_dir: Path,
    chunk_samples: int,
) -> list[tuple[Path, dict[str, str]]]:
    """Split samples without changing the complete scene contract.

    Each chunk contains the same selected GLB inventory, bindings, AABBs and
    camera contract as the source. Only the browser process lifetime changes;
    samples and their prediction component IDs are copied byte-for-byte.
    """
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    source_batches = batch_manifest.get("batches") or []
    if not source_batches:
        raise ValueError("batch manifest has no batches")
    max_samples = max(len(batch.get("samples") or []) for batch in source_batches)
    if max_samples <= 0:
        raise ValueError("batch manifest has no samples")
    chunk_root = output_dir / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    result: list[tuple[Path, dict[str, str]]] = []
    for start in range(0, max_samples, chunk_samples):
        chunk_number = start // chunk_samples
        chunk_batches: list[dict[str, Any]] = []
        batch_map: dict[str, str] = {}
        for batch in source_batches:
            original_id = str(batch["batchId"])
            samples = (batch.get("samples") or [])[start : start + chunk_samples]
            if not samples:
                continue
            chunk_id = f"{original_id}__chunk{chunk_number:04d}"
            chunk_batch = {
                key: value
                for key, value in batch.items()
                if key != "batchId" and key != "samples"
            }
            chunk_batch.update({"batchId": chunk_id, "samples": samples})
            chunk_batches.append(chunk_batch)
            batch_map[chunk_id] = original_id
        payload = dict(batch_manifest)
        payload["batches"] = chunk_batches
        payload["chunking"] = {
            "schema": "m5-component-image-chunk-v1",
            "sourceBatchManifest": str((output_dir / "batch_manifest.json").resolve()),
            "chunkNumber": chunk_number,
            "sampleStart": start,
            "sampleLimitPerSourceBatch": chunk_samples,
            "completeInventoryRetained": True,
        }
        validate_instance_render_batch_manifest(payload)
        chunk_dir = chunk_root / f"chunk_{chunk_number:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / "batch_manifest.json"
        chunk_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result.append((chunk_path, batch_map))
    return result


def run_chunked_renderer(
    batch_manifest: dict[str, Any],
    batch_manifest_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Render chunks in separate Chrome processes and aggregate pixel counts."""
    chunk_infos = build_chunk_manifests(batch_manifest, output_dir, int(args.chunk_samples))
    total = _empty_image_totals()
    sample_rows: list[dict[str, Any]] = []
    batch_totals: dict[str, dict[str, Any]] = {}
    chunk_summaries: list[dict[str, Any]] = []
    spatial_sum = {
        "sampleGroupCount": 0,
        "referenceRenderCount": 0,
        "referenceReuseCount": 0,
        "requiredGlbCountSum": 0,
        "predictionGlbCountSum": 0,
        "loadedGlbCount": 0,
        "loaderCalls": 0,
    }
    spatial_min: dict[str, int | None] = {
        "requiredGlbCountMin": None,
        "predictionGlbCountMin": None,
    }
    spatial_max = {"requiredGlbCountMax": 0, "predictionGlbCountMax": 0}
    top_missed: dict[int, dict[str, Any]] = {}

    for chunk_path, batch_map in chunk_infos:
        chunk_dir = chunk_path.parent
        summary = run_renderer(chunk_path, chunk_dir, args)
        if summary.get("renderStatus") != "rendered_component_id_buffers":
            raise RuntimeError(
                f"M5 chunk did not complete: {chunk_path}: {summary.get('renderStatus')}"
            )
        chunk_summaries.append({
            "chunkNumber": len(chunk_summaries),
            "manifest": str(chunk_path),
            "outputDir": str(chunk_dir),
            "renderer": summary.get("renderer"),
            "sampleCount": int(summary.get("sampleCount", 0)),
            "renderElapsedMs": float(summary.get("elapsedMs", 0.0)),
            "loadedGlbCount": int(summary.get("loadedGlbCount", 0)),
            "selfConsistencyPER": float(summary.get("selfConsistencyPER", 0.0)),
        })
        _merge_image_totals(total, summary.get("imageMetrics") or {})
        for row in summary.get("batchSummaries") or []:
            chunk_batch_id = str(row.get("batchId") or "")
            original_batch_id = batch_map.get(chunk_batch_id)
            if original_batch_id is None:
                raise ValueError(f"chunk summary contains unknown batch id: {chunk_batch_id}")
            metrics = row.get("imageMetrics") or {}
            aggregate = batch_totals.setdefault(
                original_batch_id,
                {"sampleCount": 0, "totals": _empty_image_totals()},
            )
            aggregate["sampleCount"] += int(row.get("sampleCount", 0))
            _merge_image_totals(aggregate["totals"], metrics)
        chunk_metrics_path = chunk_dir / "true_glb_render" / "sample_image_metrics.json"
        rows = json.loads(chunk_metrics_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"chunk sample metrics must be a list: {chunk_metrics_path}")
        for row in rows:
            chunk_batch_id = str(row.get("batchId") or "")
            original_batch_id = batch_map.get(chunk_batch_id)
            if original_batch_id is None:
                raise ValueError(f"chunk sample metrics contains unknown batch id: {chunk_batch_id}")
            copied = dict(row)
            copied["chunkBatchId"] = chunk_batch_id
            copied["batchId"] = original_batch_id
            sample_rows.append(copied)
        chunk_spatial = summary.get("spatialCulling") or {}
        for field in spatial_sum:
            spatial_sum[field] += int(chunk_spatial.get(field, 0))
        for field in spatial_min:
            value = chunk_spatial.get(field)
            if value is not None:
                numeric = int(value)
                spatial_min[field] = numeric if spatial_min[field] is None else min(spatial_min[field], numeric)
        for field in spatial_max:
            spatial_max[field] = max(spatial_max[field], int(chunk_spatial.get(field, 0)))
        for item in summary.get("topMissedComponents") or []:
            component_id = int(item["componentGlobalId"])
            existing = top_missed.setdefault(component_id, dict(item))
            existing["missPixels"] = int(existing.get("missPixels", 0)) + int(item.get("missPixels", 0))

    sample_count = len(sample_rows)
    rendered = {
        "schema": "local-true-component-id-browser-summary-v3-chunked",
        "renderer": chunk_summaries[0].get("renderer") if chunk_summaries else None,
        "renderStatus": "rendered_component_id_buffers",
        "componentIdShaderImplemented": True,
        "browserInstanceReorderImplemented": False,
        "formalImageEvaluationReady": False,
        "renderFovYDeg": int(batch_manifest["renderFovYDeg"]),
        "selectedGlbCount": len(batch_manifest["selectedGlbs"]),
        "sampleCount": sample_count,
        "batchCount": len(batch_manifest["batches"]),
        "loadedGlbCount": spatial_sum["loadedGlbCount"],
        "spatialCulling": {
            "enabled": True,
            **spatial_sum,
            **spatial_min,
            **spatial_max,
            "requiredGlbCountMax": spatial_max["requiredGlbCountMax"],
            "predictionGlbCountMax": spatial_max["predictionGlbCountMax"],
            "meanRequiredGlbCount": spatial_sum["requiredGlbCountSum"] / max(1, spatial_sum["sampleGroupCount"]),
            "meanPredictionGlbCount": spatial_sum["predictionGlbCountSum"] / max(1, sample_count),
            "chunkCount": len(chunk_infos),
            "completeInventoryRetained": True,
        },
        "elapsedMs": sum(item["renderElapsedMs"] for item in chunk_summaries),
        "assetReuse": {
            "schema": "m5-browser-asset-reuse-v1",
            "browserPageCount": len(chunk_infos),
            "glbLoadPasses": len(chunk_infos),
            "selectedGlbCount": len(batch_manifest["selectedGlbs"]),
            "loadedGlbCount": spatial_sum["loadedGlbCount"],
            "glbLoaderCalls": spatial_sum["loaderCalls"],
            "sampleCount": sample_count,
            "batchCount": len(batch_manifest["batches"]),
            "reuseScope": "same-camera reference reuse within each bounded browser chunk",
            "chunkSampleLimitPerSourceBatch": int(args.chunk_samples),
        },
        "chunkSummaries": chunk_summaries,
        "batchSummaries": [
            {
                "batchId": batch_id,
                "sampleCount": aggregate["sampleCount"],
                "imageMetrics": _finalize_image_metrics(aggregate["totals"], aggregate["sampleCount"]),
            }
            for batch_id, aggregate in sorted(batch_totals.items())
        ],
        "selfConsistencyPER": max(
            (float(item.get("selfConsistencyPER", 0.0)) for item in chunk_summaries),
            default=0.0,
        ),
        "imageMetrics": _finalize_image_metrics(total, sample_count),
        "topMissedComponents": sorted(
            top_missed.values(), key=lambda item: int(item.get("missPixels", 0)), reverse=True
        )[:50],
        "chunking": {
            "schema": "m5-component-image-chunk-v1",
            "sourceBatchManifest": str(batch_manifest_path.resolve()),
            "chunkCount": len(chunk_infos),
            "sampleLimitPerSourceBatch": int(args.chunk_samples),
            "completeInventoryRetained": True,
            "thresholdsUntouched": True,
        },
    }
    render_root = output_dir / "true_glb_render"
    render_root.mkdir(parents=True, exist_ok=True)
    (render_root / "sample_image_metrics.json").write_text(
        json.dumps(sample_rows, ensure_ascii=False), encoding="utf-8"
    )
    (render_root / "render_summary.json").write_text(
        json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rendered


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
    if int(args.chunk_samples) > 0:
        render_summary = run_chunked_renderer(batch_manifest, batch_manifest_path, output_dir, args)
    else:
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
