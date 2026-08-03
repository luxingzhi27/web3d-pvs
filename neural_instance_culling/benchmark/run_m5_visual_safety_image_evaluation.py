#!/usr/bin/env python3
"""Evaluate the registered M5 repair matrix with bounded browser pages.

Each model/split first produces a component-ID manifest without rendering.  The
manifests are then combined into complete-inventory browser batches. With
``--chunk-samples`` each bounded page releases decoded GLBs before the next
chunk while preserving the full local inventory contract. This script never trains,
scans a threshold, changes candidates, or reads the test split.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "neural_instance_culling" / "benchmark"
MODEL_ROOT = ROOT / "neural_instance_culling" / "model" / "out"
DATASET = ROOT / "neural_instance_culling" / "dataset" / "out" / "pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1"
VIEWCELL_DATASET = ROOT / "neural_instance_culling" / "dataset" / "out" / "hkust_v3_viewcell_colorid_fov66_source"
RUNTIME_META = ROOT / "hkust-v3" / "assets" / "runtimeVisibilityMeta.json"
GLB_INDEX = ROOT / "hkust-v3" / "assets" / "glbIndex.json"
GLB_ROOT = ROOT / "hkust-v3" / "assets"
GLB_POINTS_META = ROOT / "neural_instance_culling" / "dataset" / "out" / "glb_points_v3_formal_hkust_fov66_meta.json"

VARIANTS = (
    "m5_visual_mass_linear",
    "m5_visual_mass_tail",
    "m5_visual_mass_soft",
    "m5_control_log1p",
)
SEEDS = (20260801, 20260802, 20260803)
SPLITS = ("validation", "calibration")


def experiment_name(variant: str, seed: int) -> str:
    return f"{variant}_rvl_strong_v2_hkust_spatial_fov66_seed{seed}_full40"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=os.environ.get("SLM_CONDA_ENV", "slm_pvs"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--render-timeout-sec", type=int, default=12 * 60 * 60)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--subposes-per-viewcell", type=int, default=1)
    parser.add_argument("--output-name", default="m5_visual_safety_repair_image_batch_20260802")
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=16,
        help="bound each source batch per browser page; zero uses one page for all samples",
    )
    parser.add_argument("--chrome-arg", action="append", default=[])
    return parser.parse_args()


def run_command(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {stdout_path} and {stderr_path}"
        )


def wait_for_checkpoints(args: argparse.Namespace) -> None:
    expected = [MODEL_ROOT / experiment_name(variant, seed) / "calibration_ready_summary.json" for seed in SEEDS for variant in VARIANTS]
    while True:
        missing = [path for path in expected if not path.exists()]
        if not missing:
            return
        print(f"[m5-image] waiting for {len(missing)}/{len(expected)} calibration-ready checkpoints", flush=True)
        time.sleep(max(1, int(args.poll_seconds)))


def model_spec(experiment: str) -> str:
    model_dir = MODEL_ROOT / experiment
    return "|".join(
        [
            f"{experiment}_formal_image",
            str(model_dir / "best.pt"),
            str(model_dir / "instance_runtime_features_fp16.bin"),
            str(model_dir / "calibration_ready_summary.json"),
        ]
    )


def schema_manifest(args: argparse.Namespace, experiment: str, split: str, output_root: Path) -> Path:
    output_dir = output_root / f"{experiment}_{split}"
    manifest_path = output_dir / "true_glb_render_manifest.json"
    if manifest_path.exists():
        return manifest_path
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to reuse incomplete M5 schema output: {output_dir}")
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.env,
        "python",
        "-u",
        str(BENCHMARK / "evaluate_viewcell_image_per.py"),
        "--model-spec",
        model_spec(experiment),
        "--runtime-meta",
        str(RUNTIME_META),
        "--viewcell-dataset",
        str(VIEWCELL_DATASET),
        "--pose-csr",
        str(DATASET),
        "--glb-root",
        str(GLB_ROOT),
        "--glb-index",
        str(GLB_INDEX),
        "--glb-points-meta",
        str(GLB_POINTS_META),
        "--split",
        split,
        "--split-source",
        "pose_csr",
        "--subposes-per-viewcell",
        str(args.subposes_per_viewcell),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--device",
        "cpu",
        "--image-renderer",
        "true_glb",
        "--render-schema-only",
        "--skip-raw-subpose-gt",
        "--log-every",
        "50",
        "--output-dir",
        str(output_dir),
    ]
    run_command(
        command,
        output_root / f"{experiment}_{split}_schema_stdout.log",
        output_root / f"{experiment}_{split}_schema_stderr.log",
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"schema-only evaluator did not write {manifest_path}")
    return manifest_path


def prefix_manifest(source: Path, destination: Path, batch_id: str) -> Path:
    if destination.exists():
        return destination
    payload = json.loads(source.read_text(encoding="utf-8"))
    samples = []
    seen: set[str] = set()
    for row in payload.get("samples") or []:
        sample = copy.deepcopy(row)
        source_id = str(sample.get("sampleId") or "")
        if not source_id or source_id in seen:
            raise ValueError(f"duplicate or empty source sample id in {source}")
        seen.add(source_id)
        sample["sourceSampleId"] = source_id
        sample["sampleId"] = f"{batch_id}__{source_id}"
        samples.append(sample)
    if not samples:
        raise ValueError(f"schema manifest contains no samples: {source}")
    payload["samples"] = samples
    payload["batchSourceId"] = batch_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def percentile(values: list[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        raise ValueError("cannot calculate a percentile from an empty list")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def load_viewcell_metrics(batch_output: Path, expected_batches: list[str]) -> dict[str, dict[str, Any]]:
    """Summarize per-view-cell image metrics without touching predictions."""
    metrics_path = batch_output / "true_glb_render" / "sample_image_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"sample image metrics must be a list: {metrics_path}")
    grouped: dict[str, list[dict[str, Any]]] = {batch_id: [] for batch_id in expected_batches}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"invalid sample image metric row in {metrics_path}")
        batch_id = str(row.get("batchId") or "")
        if batch_id not in grouped:
            raise ValueError(f"unexpected batch id in {metrics_path}: {batch_id!r}")
        image = row.get("imageMetrics")
        if not isinstance(image, dict):
            raise ValueError(f"sample image metric lacks imageMetrics: {metrics_path}")
        grouped[batch_id].append(image)

    result: dict[str, dict[str, Any]] = {}
    for batch_id, rows in grouped.items():
        if not rows:
            raise ValueError(f"batch {batch_id} has no per-view-cell image metrics")
        miss = [float(row["missPixelRate"]) for row in rows]
        wrong = [float(row["wrongInstancePixelRate"]) for row in rows]
        per = [float(row["PER"]) for row in rows]
        result[batch_id] = {
            "sampleCount": len(rows),
            "viewCellMissPixelRateMean": sum(miss) / len(miss),
            "viewCellMissPixelRateP95": percentile(miss, 0.95),
            "viewCellMissPixelRateMax": max(miss),
            "viewCellWrongInstancePixelRateMean": sum(wrong) / len(wrong),
            "viewCellWrongInstancePixelRateP95": percentile(wrong, 0.95),
            "viewCellPERMean": sum(per) / len(per),
            "viewCellPERP95": percentile(per, 0.95),
        }
    return result


def write_report(batch_output: Path, batch_manifest: Path, output_root: Path, expected_batches: list[str]) -> None:
    render_summary_path = batch_output / "true_glb_render" / "render_summary.json"
    if not render_summary_path.exists():
        raise FileNotFoundError(render_summary_path)
    render_summary = json.loads(render_summary_path.read_text(encoding="utf-8"))
    if render_summary.get("renderStatus") != "rendered_component_id_buffers":
        raise RuntimeError(f"M5 browser render did not complete: {render_summary.get('renderStatus')}")
    batch_rows = {str(row.get("batchId")): row for row in render_summary.get("batchSummaries") or []}
    missing = [batch_id for batch_id in expected_batches if batch_id not in batch_rows]
    if missing:
        raise RuntimeError(f"M5 browser render omitted batches: {missing[:8]}")
    viewcell_metrics = load_viewcell_metrics(batch_output, expected_batches)
    rows: list[dict[str, Any]] = []
    for batch_id in expected_batches:
        row = batch_rows[batch_id]
        metrics = row.get("imageMetrics") or {}
        rows.append({
            "batchId": batch_id,
            "sampleCount": int(row.get("sampleCount", 0)),
            "imageMetrics": metrics,
            "viewCellImageMetrics": viewcell_metrics[batch_id],
        })
    summary = {
        "schema": "m5-visual-safety-repair-image-batch-summary-v1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "split": ["validation", "calibration"],
        "testRead": False,
        "renderFovYDeg": 60,
        "modelInputFovYDeg": 66,
        "scene": "hkust-v3",
        "browserRenderer": render_summary.get("renderer"),
        "browserAssetReuse": render_summary.get("assetReuse"),
        "batches": rows,
        "formalImageEvaluationReady": False,
        "note": "Validation/calibration image evidence only; test remains sealed and the renderer reports non-formal until the registered image gates are reviewed.",
        "qualityGate": {
            "meanMissPixelRateMax": 0.005,
            "viewCellMissPixelRateP95Max": 0.01,
            "selectionUsesTest": False,
            "thresholdsUntouched": True,
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# M5 Visual-Safety Repair Image Batch",
        "",
        "This report contains validation/calibration only. The test split was not read.",
        "",
        "| Batch | Samples | Aggregate miss | View-cell mean miss | View-cell p95 miss | View-cell max miss | Wrong-ID p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["imageMetrics"]
        lines.append(
            f"| `{row['batchId']}` | {row['sampleCount']} | "
            f"{float(metrics.get('missPixelRate', 0.0)):.6f} | "
            f"{row['viewCellImageMetrics']['viewCellMissPixelRateMean']:.6f} | "
            f"{row['viewCellImageMetrics']['viewCellMissPixelRateP95']:.6f} | "
            f"{row['viewCellImageMetrics']['viewCellMissPixelRateMax']:.6f} | "
            f"{row['viewCellImageMetrics']['viewCellWrongInstancePixelRateP95']:.6f} |"
        )
    lines.extend([
        "",
        (
            "The renderer retains the complete local GLB inventory contract and uses "
            "bounded browser pages for asset lifetime; same-camera reference reuse is "
            "local to each page."
        ),
        "`visible_weights` remains an importance proxy, not exact pixel coverage.",
    ])
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "batch_manifest_path.txt").write_text(str(batch_manifest) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    wait_for_checkpoints(args)
    schema_root = BENCHMARK / "out" / "m5_visual_safety_repair_image_manifests_20260802"
    prefixed_root = schema_root / "prefixed"
    batch_output = BENCHMARK / "out" / args.output_name
    if (batch_output / "summary.json").exists():
        print(f"[m5-image] existing completed summary: {batch_output / 'summary.json'}", flush=True)
        return
    if batch_output.exists() and any(batch_output.iterdir()):
        raise RuntimeError(f"refusing to reuse incomplete batch output: {batch_output}")

    inputs: list[str] = []
    expected_batches: list[str] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            experiment = experiment_name(variant, seed)
            for split in SPLITS:
                batch_id = f"{variant}_seed{seed}_{split}"
                source = schema_manifest(args, experiment, split, schema_root)
                destination = prefix_manifest(source, prefixed_root / f"{batch_id}.json", batch_id)
                inputs.extend(["--input", f"{batch_id}={destination}"])
                expected_batches.append(batch_id)

    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.env,
        "python",
        "-u",
        str(BENCHMARK / "run_m5_component_image_batch.py"),
        *inputs,
        "--output-dir",
        str(batch_output),
        "--chrome-exe",
        "/usr/bin/google-chrome",
        "--timeout-sec",
        str(max(30, args.render_timeout_sec)),
        "--preview-samples",
        "8",
    ]
    if args.chunk_samples > 0:
        command.extend(["--chunk-samples", str(args.chunk_samples)])
    for chrome_arg in args.chrome_arg:
        command.extend(["--chrome-arg", str(chrome_arg)])
    run_command(
        command,
        batch_output.parent / f"{batch_output.name}_launcher_stdout.log",
        batch_output.parent / f"{batch_output.name}_launcher_stderr.log",
    )
    write_report(batch_output, batch_output / "batch_manifest.json", batch_output, expected_batches)
    print(json.dumps({"status": "completed", "output": str(batch_output), "batches": len(expected_batches)}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[m5-image] interrupted", file=sys.stderr)
        raise
