#!/usr/bin/env python3
"""Materialize canonical view-cell centers and successful subpose positions.

This is a lightweight reconciliation step for a CSR dataset built by an older
aggregator. It does not recalculate visibility or candidates. It restores the
canonical plan center, refreshes the query pose/MVP, and writes geometry arrays
used by the spatial split audit.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from build_rvc_viewcell_pose_csr import DIRECTIONAL_POSE_DTYPE, conservative_mvp, normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--raw-glob", default="*.jsonl")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    meta_path = dataset_dir / "dataset_meta.json"
    meta = read_json(meta_path)
    pose_count = int(meta["poseCount"])
    centers = np.zeros((pose_count, 3), dtype=np.float32)
    forwards = np.zeros((pose_count, 3), dtype=np.float32)
    seen = np.zeros((pose_count,), dtype=np.uint8)
    subpose_positions: list[list[float]] = []
    subpose_offsets = np.zeros((pose_count + 1,), dtype=np.uint64)
    current_viewcell = -1
    current_positions: list[list[float]] = []
    current_center: np.ndarray | None = None
    current_forward: np.ndarray | None = None

    files = sorted(args.raw_dir.glob(args.raw_glob))
    if not files:
        raise FileNotFoundError(f"No raw files matched {args.raw_dir}/{args.raw_glob}")
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("sample_error"):
                    raise RuntimeError(f"Formal geometry reconciliation found sample_error at {file_path}:{line_number}")
                viewcell_id = int(row.get("viewcell_id", -1))
                if viewcell_id < 0 or viewcell_id >= pose_count:
                    raise ValueError(f"viewcell_id {viewcell_id} is outside pose range")
                if current_viewcell != viewcell_id:
                    if current_viewcell >= 0:
                        if not current_positions:
                            raise ValueError(f"viewcell {current_viewcell} has no successful subpose")
                        subpose_positions.extend(current_positions)
                        subpose_offsets[current_viewcell + 1] = len(subpose_positions)
                    if viewcell_id != current_viewcell + 1:
                        raise ValueError(f"viewcell IDs are not contiguous: expected {current_viewcell + 1}, got {viewcell_id}")
                    current_viewcell = viewcell_id
                    current_positions = []
                    current_center = None
                    current_forward = None
                center = np.asarray(row.get("viewcell_center") or row.get("camera_pos"), dtype=np.float32)
                forward = normalize(
                    np.asarray(row.get("viewcell_forward") or row.get("camera_forward") or [0, 0, -1], dtype=np.float32),
                    np.asarray([0, 0, -1], dtype=np.float32),
                )
                if current_center is None:
                    current_center = center
                    current_forward = forward
                elif not np.allclose(center, current_center, atol=1e-4):
                    raise ValueError(f"viewcell {viewcell_id} has inconsistent canonical centers")
                elif not np.allclose(forward, current_forward, atol=1e-4):
                    raise ValueError(f"viewcell {viewcell_id} has inconsistent directions")
                centers[viewcell_id] = center
                forwards[viewcell_id] = forward
                seen[viewcell_id] = 1
                current_positions.append([float(v) for v in row.get("camera_pos", center.tolist())])
    if current_viewcell >= 0:
        subpose_positions.extend(current_positions)
        subpose_offsets[current_viewcell + 1] = len(subpose_positions)
    if not np.all(seen):
        missing = np.flatnonzero(seen == 0)
        raise ValueError(f"Missing view-cell rows: {missing[:10].tolist()}")
    for index in range(1, subpose_offsets.size):
        if subpose_offsets[index] == 0:
            subpose_offsets[index] = subpose_offsets[index - 1]
    if subpose_offsets[-1] != len(subpose_positions):
        raise ValueError("Subpose offsets do not cover the materialized positions")

    poses = np.fromfile(dataset_dir / "poses.bin", dtype=DIRECTIONAL_POSE_DTYPE)
    if poses.size != pose_count:
        raise ValueError(f"Expected {pose_count} directional poses, found {poses.size}")
    poses = poses.copy()
    poses["camera_world"] = centers
    camera_bounds = meta.get("cameraBounds") or {}
    minimum = np.asarray(camera_bounds.get("min", centers.min(axis=0)), dtype=np.float32)
    maximum = np.asarray(camera_bounds.get("max", centers.max(axis=0)), dtype=np.float32)
    size = np.maximum(maximum - minimum, 1e-6)
    poses["camera_norm"] = np.clip((centers - minimum[None, :]) / size[None, :], 0.0, 1.0)
    tan_y = np.asarray(poses["camera_view"][:, 1], dtype=np.float32)
    tan_x = np.asarray(poses["camera_view"][:, 0], dtype=np.float32)
    mvps = np.zeros((pose_count, 16), dtype=np.float32)
    for index in range(pose_count):
        mvps[index] = conservative_mvp(centers[index], forwards[index], float(tan_x[index]), float(tan_y[index]))
    poses.tofile(dataset_dir / "poses.bin")
    mvps.tofile(dataset_dir / "mvp.bin")
    centers.tofile(dataset_dir / "viewcell_centers.bin")
    subpose_offsets.tofile(dataset_dir / "subpose_offsets.bin")
    np.asarray(subpose_positions, dtype="<f4").tofile(dataset_dir / "subpose_camera_pos.bin")

    meta["cameraSemantics"] = "camera_world is the canonical viewcell plan center; candidates are the union over dense subpose positions"
    meta["viewcellGeometrySemantics"] = "viewcell_centers are canonical plan centers; subpose_camera_pos stores every successful dense subpose position"
    meta["viewcellGeometryReconciledFrom"] = str(args.raw_dir)
    meta["viewcellGeometryRawFileCount"] = len(files)
    meta["viewcellGeometrySubposeCount"] = len(subpose_positions)
    meta["files"] = {
        **(meta.get("files") or {}),
        "viewcellCenters": "viewcell_centers.bin",
        "subposeOffsets": "subpose_offsets.bin",
        "subposeCameraPos": "subpose_camera_pos.bin",
    }
    meta["stats"] = {**(meta.get("stats") or {}), "viewcellGeometrySubposeCount": len(subpose_positions)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"datasetDir": str(dataset_dir), "poseCount": pose_count, "subposeCount": len(subpose_positions)}, indent=2))


if __name__ == "__main__":
    main()
