#!/usr/bin/env python3
"""Recover the retained HKUST view-cell locations and emit a corrected FOV-66 plan.

The historical HKUST dataset was generated from the non-uniform view-cell
strategy in the old froxel planner.  The recovered representative pose table
is retained as JSONL, so this tool reuses those locations and category/split
assignments without recreating an evenly spaced grid or depending on the old
CSR dataset.

The old planner skipped collision checks for sky/far samples.  This tool keeps
their horizontal locations, lifts any sample that is inside a component, and
checks every dense subpose before emitting it.  All subposes keep the
representative direction, use a 66 degree sampling/model FOV, and use the
60-degree render FOV only for the conservative back-camera offset formula.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


CATEGORY_NAMES = {
    0: "street_gap",
    1: "near_building",
    2: "plaza",
    3: "perimeter",
    4: "sky",
    5: "far",
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representatives-input",
        type=Path,
        default=Path("neural_instance_culling/sampler/out/hkust_v3_viewcell_fov66/representatives.jsonl"),
        help="Retained view-cell representative JSONL used as the canonical source of HKUST sampling points.",
    )
    parser.add_argument("--runtime-meta", type=Path, default=Path("hkust-v3/assets/runtimeVisibilityMeta.json"))
    parser.add_argument("--representatives-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--low-subposes", type=int, default=32)
    parser.add_argument("--sky-far-subposes", type=int, default=48)
    parser.add_argument("--model-fov-y", type=float, default=66.0)
    parser.add_argument("--frontend-fov-y", type=float, default=60.0)
    parser.add_argument("--aspect", type=float, default=16.0 / 9.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--collision-padding", type=float, default=0.35)
    parser.add_argument("--clearance", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_min_max(scene_bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    if "min" in scene_bounds and "max" in scene_bounds:
        return np.asarray(scene_bounds["min"], dtype=np.float32), np.asarray(scene_bounds["max"], dtype=np.float32)
    center = np.asarray(scene_bounds["center"], dtype=np.float32)
    size = np.asarray(scene_bounds["size"], dtype=np.float32)
    return center - size * 0.5, center + size * 0.5


def bounds_min_max(bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    if "min" in bounds and "max" in bounds:
        return np.asarray(bounds["min"], dtype=np.float32), np.asarray(bounds["max"], dtype=np.float32)
    center = np.asarray(bounds["center"], dtype=np.float32)
    size = np.asarray(bounds["size"], dtype=np.float32)
    return center - size * 0.5, center + size * 0.5


def load_aabbs(runtime_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    records = runtime_meta.get("componentRecords", [])
    count = max((int(record["componentGlobalId"]) for record in records), default=-1) + 1
    mins = np.zeros((count, 3), dtype=np.float32)
    maxs = np.zeros((count, 3), dtype=np.float32)
    for record in records:
        index = int(record["componentGlobalId"])
        mins[index], maxs[index] = bounds_min_max(record["bounds"])
    return mins, maxs


class CollisionIndex:
    def __init__(self, mins: np.ndarray, maxs: np.ndarray, scene_min: np.ndarray, step: float = 15.0):
        self.mins = mins
        self.maxs = maxs
        self.scene_min = scene_min
        self.step = float(step)
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (mn, mx) in enumerate(zip(mins, maxs)):
            if np.any(mx < mn):
                continue
            x0 = math.floor((float(mn[0]) - float(scene_min[0])) / self.step)
            x1 = math.floor((float(mx[0]) - float(scene_min[0])) / self.step)
            z0 = math.floor((float(mn[2]) - float(scene_min[2])) / self.step)
            z1 = math.floor((float(mx[2]) - float(scene_min[2])) / self.step)
            for x in range(x0, x1 + 1):
                for z in range(z0, z1 + 1):
                    self.cells[(x, z)].append(index)

    def hits(self, point: np.ndarray, padding: float) -> list[int]:
        x = math.floor((float(point[0]) - float(self.scene_min[0])) / self.step)
        z = math.floor((float(point[2]) - float(self.scene_min[2])) / self.step)
        result = []
        for index in self.cells.get((x, z), []):
            mn = self.mins[index]
            mx = self.maxs[index]
            if (
                float(point[0]) >= float(mn[0]) - padding
                and float(point[0]) <= float(mx[0]) + padding
                and float(point[1]) >= float(mn[1]) - padding
                and float(point[1]) <= float(mx[1]) + padding
                and float(point[2]) >= float(mn[2]) - padding
                and float(point[2]) <= float(mx[2]) + padding
            ):
                result.append(index)
        return result


def normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < 1e-8:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return (v / length).astype(np.float32, copy=False)


def yaw_pitch(forward: np.ndarray) -> tuple[float, float]:
    f = normalize(forward)
    return math.degrees(math.atan2(-float(f[0]), -float(f[2]))), math.degrees(math.asin(float(np.clip(f[1], -1.0, 1.0))))


def make_rng(seed: int):
    state = int(seed) & 0xFFFFFFFF

    def random() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 4294967296.0

    return random


def lift_out_of_geometry(point: np.ndarray, index: CollisionIndex, padding: float, clearance: float) -> tuple[np.ndarray, int]:
    result = point.astype(np.float32, copy=True)
    lifts = 0
    for _ in range(32):
        hits = index.hits(result, padding)
        if not hits:
            return result, lifts
        result[1] = max(float(result[1]), max(float(index.maxs[i, 1]) for i in hits) + padding + clearance)
        lifts += 1
    raise RuntimeError(f"Could not lift camera out of geometry at {point.tolist()}")


def json_row(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def write_preview(path: Path, scene_min: np.ndarray, scene_max: np.ndarray, mins: np.ndarray, maxs: np.ndarray, reps: list[dict]) -> None:
    width = 1400
    height = max(420, round(width * float(scene_max[2] - scene_min[2]) / max(1.0, float(scene_max[0] - scene_min[0]))))

    def sx(x: float) -> float:
        return (x - float(scene_min[0])) / max(1.0, float(scene_max[0] - scene_min[0])) * width

    def sz(z: float) -> float:
        return (z - float(scene_min[2])) / max(1.0, float(scene_max[2] - scene_min[2])) * height

    colors = {
        "street_gap": "#e53935",
        "near_building": "#fb8c00",
        "plaza": "#43a047",
        "perimeter": "#1e88e5",
        "sky": "#8e24aa",
        "far": "#546e7a",
    }
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="#111"/>']
    for mn, mx in zip(mins, maxs):
        x = sx(float(mn[0]))
        z = sz(float(mn[2]))
        w = max(0.3, sx(float(mx[0])) - x)
        h = max(0.3, sz(float(mx[2])) - z)
        parts.append(f'<rect x="{x:.2f}" y="{z:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#333" opacity="0.18"/>')
    for row in reps:
        center = row["viewcell_center"]
        parts.append(f'<circle cx="{sx(center[0]):.2f}" cy="{sz(center[2]):.2f}" r="2.4" fill="{colors.get(row["sample_category"], "#fff")}"/>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if abs(args.model_fov_y - 66.0) > 1e-6 or abs(args.frontend_fov_y - 60.0) > 1e-6:
        raise RuntimeError("This HKUST plan is fixed to model FOV 66 degrees and frontend FOV 60 degrees.")
    if args.radius <= 0 or args.low_subposes <= 0 or args.sky_far_subposes <= 0:
        raise RuntimeError("radius and subpose counts must be positive")

    runtime_meta = read_json(args.runtime_meta)
    source_records: list[dict] = []
    with args.representatives_input.open("r", encoding="utf-8") as handle:
        source_records = [json.loads(line) for line in handle if line.strip()]
    if not source_records:
        raise RuntimeError(f"Representative input is empty: {args.representatives_input}")

    scene_min, scene_max = scene_min_max(runtime_meta["sceneBounds"])
    mins, maxs = load_aabbs(runtime_meta)
    collision = CollisionIndex(mins, maxs, scene_min)
    model_tan_y = math.tan(math.radians(args.model_fov_y) * 0.5)
    model_fov_x = math.degrees(2.0 * math.atan(model_tan_y * args.aspect))
    back_offset = args.radius / math.tan(math.radians(args.frontend_fov_y) * 0.5)
    random = make_rng(args.seed)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    representatives: list[dict] = []
    rows: list[dict] = []
    category_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    lifted_counts: dict[str, int] = defaultdict(int)
    collision_fallbacks = 0
    dropped = 0

    for source in source_records:
        viewcell_id = int(source.get("viewcell_id", source.get("pose_index", len(representatives))))
        forward_value = source.get("viewcell_forward") or source.get("camera_forward")
        if forward_value is None:
            raise RuntimeError(f"View-cell representative {viewcell_id} has no forward direction.")
        old_forward = normalize(np.asarray(forward_value, dtype=np.float32))
        center = np.asarray(source["viewcell_center"], dtype=np.float32)
        category = str(source.get("sample_category", source.get("viewcell_category", "unknown")))
        category_id = int(source.get("sample_category_id", next((key for key, value in CATEGORY_NAMES.items() if value == category), 255)))
        split = str(source.get("split", "unknown"))
        safe_center, lifts = lift_out_of_geometry(center, collision, args.collision_padding, args.clearance)
        if lifts:
            lifted_counts[category] += 1
        if collision.hits(safe_center, args.collision_padding):
            dropped += 1
            continue
        yaw, pitch = yaw_pitch(old_forward)
        count = args.sky_far_subposes if category in {"sky", "far"} else args.low_subposes
        representative = {
            "pose_index": viewcell_id,
            "viewcell_id": viewcell_id,
            "subpose_id": 0,
            "viewcell_category": category,
            "viewcell_center": safe_center.astype(float).tolist(),
            "viewcell_shape": "horizontal_disk",
            "viewcell_half_extent": [args.radius, args.radius, 0.0],
            "viewcell_radius": args.radius,
            "viewcell_forward": old_forward.astype(float).tolist(),
            "viewcell_yaw_deg": yaw,
            "viewcell_pitch_deg": pitch,
            "pvs_fov_y": args.model_fov_y,
            "pvs_fov_x": model_fov_x,
            "pvs_back_offset": back_offset,
            "split": split,
            "sample_category": category,
            "sample_category_id": category_id,
            "camera_pos": safe_center.astype(float).tolist(),
            "camera_forward": old_forward.astype(float).tolist(),
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "render_fov_y": args.model_fov_y,
            "fov_y": args.model_fov_y,
            "aspect": args.aspect,
            "width": args.width,
            "height": args.height,
        }
        representatives.append(representative)
        category_counts[category] += 1
        split_counts[split] += 1
        for subpose_id in range(count):
            radius = args.radius * math.sqrt((subpose_id + 0.5) / count)
            angle = subpose_id * golden_angle + (random() * 2.0 - 1.0) * 0.075
            position = safe_center.copy()
            position[0] += math.cos(angle) * radius
            position[2] += math.sin(angle) * radius
            if collision.hits(position, args.collision_padding):
                position = safe_center.copy()
                collision_fallbacks += 1
            row = dict(representative)
            row["pose_index"] = len(rows)
            row["subpose_id"] = subpose_id
            row["camera_pos"] = position.astype(float).tolist()
            rows.append(row)

    args.representatives_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.representatives_output.open("w", encoding="utf-8") as handle:
        for row in representatives:
            handle.write(json_row(row) + "\n")
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_row(row) + "\n")

    write_preview(args.preview, scene_min, scene_max, mins, maxs, representatives)
    summary = {
        "schema": "neuralpvs-hkust-fov66-viewcell-subpose-plan-v1",
        "sourceRepresentativesInput": args.representatives_input.as_posix(),
        "runtimeMeta": args.runtime_meta.as_posix(),
        "representativesOutput": args.representatives_output.as_posix(),
        "output": args.output.as_posix(),
        "preview": args.preview.as_posix(),
        "sourceViewcellCount": int(len(source_records)),
        "viewcellCount": int(len(representatives)),
        "subposeCount": int(len(rows)),
        "viewcellRadius": args.radius,
        "lowSubposesPerViewcell": args.low_subposes,
        "skyFarSubposesPerViewcell": args.sky_far_subposes,
        "modelInputFovYDeg": args.model_fov_y,
        "frontendRenderFovYDeg": args.frontend_fov_y,
        "modelFovXDegAtAspect": model_fov_x,
        "pvsBackOffsetM": back_offset,
        "viewcellSemantics": "historical non-uniform HKUST view-cell centers, same-direction horizontal disk subposes",
        "categoryCounts": dict(category_counts),
        "splitCounts": dict(split_counts),
        "liftedOutOfGeometryCounts": dict(lifted_counts),
        "collisionFallbackSubposes": collision_fallbacks,
        "droppedViewcells": dropped,
        "collisionPaddingM": args.collision_padding,
        "clearanceM": args.clearance,
        "seed": args.seed,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
