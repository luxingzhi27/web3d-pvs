#!/usr/bin/env python3
"""Build per-pose reference-frontmost pixel histograms from Color-ID buffers.

The input buffers are rendered by the existing component-ID renderer.  Each
non-zero pixel stores ``componentGlobalId + 1``; this script maps it to its
owning GLB and sums counts for the requested view-cell/pose.  It does not
render, decode GLBs, or infer hidden visibility.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reference-frontmost-pixel-histogram-v1 sidecar."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--buffer-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_reference_histogram(
    manifest_path: str | Path,
    buffer_dir: str | Path,
    runtime_meta_path: str | Path,
    *,
    width: int = 0,
    height: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build histogram rows and metadata without writing an implicit output.

    Keeping the computation separate from the CLI makes the Color-ID mapping
    contract directly testable with a tiny synthetic fixture.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    runtime_meta_path = Path(runtime_meta_path).expanduser().resolve()
    manifest = read_json(manifest_path, "render manifest")
    runtime = read_json(runtime_meta_path, "runtime metadata")
    width = int(width or manifest.get("width", 0))
    height = int(height or manifest.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("manifest or command line must provide positive width and height")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("render manifest has no samples")
    component_records = runtime.get("componentRecords")
    if not isinstance(component_records, list):
        raise ValueError("runtime metadata has no componentRecords")
    component_to_glb = {}
    for fallback_id, record in enumerate(component_records):
        component_id = int(record.get("componentGlobalId", fallback_id))
        glb_id = int(record.get("globalGlbId", -1))
        if component_id in component_to_glb or component_id < 0 or glb_id < 0:
            raise ValueError(f"invalid component mapping at row {fallback_id}")
        component_to_glb[component_id] = glb_id

    buffer_dir = Path(buffer_dir).expanduser().resolve()
    histograms: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    sample_counts: dict[int, int] = defaultdict(int)
    pixel_counts: dict[int, int] = defaultdict(int)
    seen_sample_ids: set[str] = set()
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"manifest sample {sample_index} is not an object")
        sample_id = str(sample.get("sampleId") or "")
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError(f"manifest sample {sample_index} has a missing or duplicate sampleId")
        seen_sample_ids.add(sample_id)
        pose_id = int(sample.get("viewcellRow", sample.get("poseId", sample.get("poseIndex", -1))))
        if pose_id < 0:
            raise ValueError(f"manifest sample {sample_id} has no view-cell/pose ID")
        relative_buffer = sample.get("referenceBuffer")
        buffer_path = (
            (buffer_dir / str(relative_buffer)).resolve()
            if relative_buffer
            else buffer_dir / f"{sample_id}_reference_u32.bin"
        )
        try:
            buffer_path.relative_to(buffer_dir)
        except ValueError as error:
            raise ValueError(f"reference buffer escapes --buffer-dir: {buffer_path}") from error
        if not buffer_path.is_file():
            raise FileNotFoundError(f"missing reference buffer for {sample_id}: {buffer_path}")
        values = np.fromfile(buffer_path, dtype="<u4")
        expected = width * height
        if values.size != expected:
            raise ValueError(
                f"reference buffer {buffer_path} has {values.size} values, expected {expected}"
            )
        ids, counts = np.unique(values[values != 0], return_counts=True)
        for encoded_component, count in zip(ids.tolist(), counts.tolist()):
            component_id = int(encoded_component) - 1
            if component_id not in component_to_glb:
                raise ValueError(
                    f"reference buffer {sample_id} contains unknown component ID {component_id}"
                )
            glb_id = component_to_glb[component_id]
            histograms[pose_id][glb_id] += int(count)
            pixel_counts[pose_id] += int(count)
        sample_counts[pose_id] += 1

    rows = []
    for pose_id in sorted(sample_counts):
        row = {
            "schema": "reference-frontmost-pixel-histogram-v1",
            "poseId": int(pose_id),
            "sampleCount": int(sample_counts[pose_id]),
            "width": width,
            "height": height,
            "histogram": {
                str(glb_id): int(count)
                for glb_id, count in sorted(histograms[pose_id].items())
                if int(count) > 0
            },
            "frontmostPixelCount": int(pixel_counts[pose_id]),
            "aggregation": "sum of per-sample frontmost Color-ID counts for this view-cell",
        }
        rows.append(row)
    metadata = {
        "schema": "reference-frontmost-pixel-histogram-meta-v1",
        "histogramSchema": "reference-frontmost-pixel-histogram-v1",
        "manifest": str(manifest_path),
        "runtimeMeta": str(runtime_meta_path),
        "bufferDir": str(buffer_dir),
        "width": width,
        "height": height,
        "sampleCount": len(samples),
        "poseCount": len(rows),
        "idEncoding": "componentGlobalId + 1, 0 background",
        "units": "frontmost reference pixels; background excluded",
        "semantics": "per-pose GLB histogram of the front-most component in each reference Color-ID pixel",
        "limitations": [
            "This is not a hidden-surface visibility count and cannot see geometry behind the front-most surface.",
            "It is not a complete download utility model: decode cost, interaction importance, and newly exposed surfaces are not represented.",
            "Counts depend on reference resolution, FOV, aspect, rasterization, and the selected subpose aggregation.",
            "When several subposes are summed, counts are a view-cell utility proxy rather than one physical image pixel total.",
        ],
        "output": "reference_frontmost_histogram.jsonl",
    }
    return rows, metadata


def main() -> None:
    args = parse_args()
    rows, metadata = build_reference_histogram(
        args.manifest,
        args.buffer_dir,
        args.runtime_meta,
        width=args.width,
        height=args.height,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reference_frontmost_histogram.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (output_dir / "histogram_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
