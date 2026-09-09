#!/usr/bin/env python3
"""Create a PoseCSR view with a new split manifest without duplicating data.

All unchanged CSR files are hard-linked when the filesystem permits it.  Only
``poses.bin`` and ``dataset_meta.json`` are newly written, so applying a split
cannot silently regenerate or alter candidate/GT arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .build_directional_viewcell_split import POSE_DTYPE, SPLIT_IDS, sha256_file
except ImportError:  # Direct script execution.
    from build_directional_viewcell_split import POSE_DTYPE, SPLIT_IDS, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--viewcell-source",
        type=Path,
        default=None,
        help=(
            "Optional aligned view-cell source. When supplied, materialize only the V4 "
            "query center, candidate camera and radius arrays without rebuilding candidates or GT."
        ),
    )
    return parser.parse_args()


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    np.ascontiguousarray(values).tofile(temporary)
    temporary.replace(path)


def _materialize_viewcell_query_geometry(
    source_dir: Path,
    output_dir: Path,
    poses: np.ndarray,
) -> dict[str, Any]:
    required = {
        "centers": source_dir / "viewcell_centers.bin",
        "forwards": source_dir / "viewcell_forwards.bin",
        "params": source_dir / "viewcell_params.bin",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"view-cell source is missing required geometry: {missing}")
    pose_count = int(poses.shape[0])
    centers = np.fromfile(required["centers"], dtype="<f4").reshape(-1, 3)
    forwards = np.fromfile(required["forwards"], dtype="<f4").reshape(-1, 3)
    params = np.fromfile(required["params"], dtype="<f4").reshape(-1, 8)
    if centers.shape[0] != pose_count or forwards.shape[0] != pose_count or params.shape[0] != pose_count:
        raise ValueError("view-cell query geometry row count does not match PoseCSR")
    if not np.isfinite(centers).all() or not np.isfinite(forwards).all() or not np.isfinite(params).all():
        raise ValueError("view-cell query geometry contains non-finite values")
    pose_forward = np.asarray(poses["camera_forward"], dtype=np.float32)
    if not np.allclose(forwards, pose_forward, rtol=1e-5, atol=2e-4):
        raise ValueError("view-cell source forward vectors do not align with PoseCSR rows")
    radii = np.asarray(params[:, 0], dtype=np.float32)
    if np.any(radii <= 0.0):
        raise ValueError("view-cell source contains a non-positive radius")
    forward_norm = np.linalg.norm(forwards, axis=1, keepdims=True)
    if np.any(forward_norm <= 1e-6):
        raise ValueError("view-cell source contains a zero camera-forward vector")
    normalized_forward = forwards / forward_norm
    # The real 60-degree view from any point in the horizontal disk is
    # enclosed by backing the 66-degree candidate anchor away from the disk
    # center by radius / tan(60 / 2).  The stored candidate CSR remains the
    # authoritative union over subposes and is never regenerated here.
    back_offsets = radii / math.tan(math.radians(60.0 * 0.5))
    candidate_camera = centers - normalized_forward * back_offsets[:, None]
    if np.allclose(centers, candidate_camera, rtol=0.0, atol=1e-6):
        raise ValueError("candidate camera must be distinct from the view-cell query center")
    if not poses.flags.writeable:
        raise ValueError("view-cell geometry materialization requires a writable pose array")
    poses["camera_world"] = candidate_camera.astype(np.float32, copy=False)
    _write_array_atomic(
        output_dir / "query_center_world.bin", centers.astype("<f4", copy=False)
    )
    _write_array_atomic(
        output_dir / "candidate_camera_world.bin",
        candidate_camera.astype("<f4", copy=False),
    )
    _write_array_atomic(
        output_dir / "viewcell_radius_m.bin", radii.astype("<f4", copy=False)
    )
    return {
        "source": str(source_dir),
        "poseCount": pose_count,
        "radiusRangeM": [float(radii.min()), float(radii.max())],
        "candidateBackOffsetRangeM": [
            float(back_offsets.min()),
            float(back_offsets.max()),
        ],
        "candidateBackOffsetDefinition": "viewcell_radius / tan(frontend_render_fov_y_deg / 2), frontend_render_fov_y_deg=60",
        "candidateAndGtPreserved": True,
        "files": {
            "queryCenterWorld": "query_center_world.bin",
            "candidateCameraWorld": "candidate_camera_world.bin",
            "viewcellRadiusM": "viewcell_radius_m.bin",
        },
    }


def repair_existing_viewcell_query_geometry(
    dataset_dir: Path,
    viewcell_source: Path,
) -> dict[str, Any]:
    """Repair only derived query geometry; candidates, GT and split stay byte-identical."""
    dataset_dir = dataset_dir.resolve()
    viewcell_source = viewcell_source.resolve()
    meta_path = dataset_dir / "dataset_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    pose_count = int(metadata["poseCount"])
    source_poses = np.memmap(
        dataset_dir / "poses.bin",
        dtype=POSE_DTYPE,
        mode="r",
        shape=(pose_count,),
    )
    poses = np.asarray(source_poses, dtype=POSE_DTYPE).copy()
    geometry = _materialize_viewcell_query_geometry(
        viewcell_source, dataset_dir, poses
    )
    _write_array_atomic(dataset_dir / "poses.bin", poses)
    metadata.update(
        {
            "queryCenterSemantics": "canonical center of the aligned same-direction view-cell",
            "candidateCameraSemantics": "66-degree candidate anchor backed from the query center by viewcell_radius/tan(30deg); stored candidate CSR remains the native subpose union",
            "cameraSemantics": "poses.camera_world is candidate_camera_world = query_center_world - normalize(viewcell_forward) * candidate_back_offset",
            "viewcellGeometryMaterialization": geometry,
            "files": {**(metadata.get("files") or {}), **geometry["files"]},
            "stats": {
                **(metadata.get("stats") or {}),
                "pvsBackOffsetRange": geometry["candidateBackOffsetRangeM"],
            },
        }
    )
    temporary_meta = meta_path.with_name(f".{meta_path.name}.partial")
    temporary_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_meta.replace(meta_path)
    return {
        "datasetDir": str(dataset_dir),
        "viewcellSource": str(viewcell_source),
        "poseCount": pose_count,
        "candidateAndGtPreserved": True,
        "geometry": geometry,
    }


def apply_manifest(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_dir = args.split_manifest.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    manifest = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    input_meta_path = input_dir / "dataset_meta.json"
    input_meta = json.loads(input_meta_path.read_text(encoding="utf-8"))
    if int(manifest["poseCount"]) != int(input_meta["poseCount"]):
        raise ValueError("split manifest pose count does not match input dataset")
    expected_meta_hash = str(manifest.get("sourceDatasetMetaSha256", ""))
    actual_meta_hash = sha256_file(input_meta_path)
    if expected_meta_hash and expected_meta_hash != actual_meta_hash:
        raise ValueError(
            f"split manifest source metadata hash mismatch: expected={expected_meta_hash}, actual={actual_meta_hash}"
        )
    split_ids_path = manifest_dir / "pose_split_ids.bin"
    split_ids = np.fromfile(split_ids_path, dtype=np.uint8)
    if split_ids.size != int(input_meta["poseCount"]):
        raise ValueError("pose_split_ids.bin has an unexpected length")
    if np.any(~np.isin(split_ids, np.asarray([0, 1, 2, 3, 254], dtype=np.uint8))):
        raise ValueError("pose_split_ids.bin contains unsupported split ids")

    output_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    copied = 0
    for source in sorted(input_dir.iterdir()):
        if not source.is_file() or source.name in {"dataset_meta.json", "poses.bin"}:
            continue
        result = _link_or_copy(source, output_dir / source.name)
        linked += result == "hardlink"
        copied += result == "copy"
    poses = np.memmap(
        input_dir / "poses.bin",
        dtype=POSE_DTYPE,
        mode="r",
        shape=(int(input_meta["poseCount"]),),
    )
    poses_out = np.asarray(poses, dtype=POSE_DTYPE).copy()
    poses_out["split"] = split_ids
    viewcell_geometry = None
    if args.viewcell_source is not None:
        viewcell_geometry = _materialize_viewcell_query_geometry(
            args.viewcell_source.resolve(), output_dir, poses_out
        )
    _write_array_atomic(output_dir / "poses.bin", poses_out)
    shutil.copy2(manifest_dir / "manifest.json", output_dir / "split_manifest.json")
    if (manifest_dir / "pose_group_ids.bin").exists():
        shutil.copy2(manifest_dir / "pose_group_ids.bin", output_dir / "pose_group_ids.bin")

    split_counts = {
        name: int(np.count_nonzero(split_ids == split_id))
        for name, split_id in SPLIT_IDS.items()
        if name in {"train", "validation", "calibration", "test", "guard"}
    }
    manifest_schema = str(manifest.get("schema", "pose-split-manifest-v1"))
    output_meta = dict(input_meta)
    output_meta.update(
        {
            "schema": "pose-csr-explicit-four-way-split-v1",
            "sourceDataset": str(input_dir),
            "splitManifest": str((manifest_dir / "manifest.json").as_posix()),
            "splitManifestSha256": sha256_file(manifest_dir / "manifest.json"),
            "splitPoseLabelsSha256": hashlib.sha256(split_ids.tobytes()).hexdigest(),
            "splitIds": manifest["splitIds"],
            "splitCounts": split_counts,
            "splitSemantics": manifest["semantics"],
            "splitProtocol": {"schema": manifest_schema, **manifest},
            "files": {**(input_meta.get("files") or {}), "poses": "poses.bin"},
            "stats": {**(input_meta.get("stats") or {}), "explicitSplitPoseCounts": split_counts},
        }
    )
    if viewcell_geometry is not None:
        output_meta.update(
            {
                "queryCenterSemantics": "canonical center of the aligned same-direction view-cell",
                "candidateCameraSemantics": "66-degree candidate anchor backed from the query center by viewcell_radius/tan(30deg); stored candidate CSR remains the native subpose union",
                "cameraSemantics": "poses.camera_world is candidate_camera_world = query_center_world - normalize(viewcell_forward) * candidate_back_offset",
                "viewcellGeometryMaterialization": viewcell_geometry,
                "stats": {
                    **output_meta["stats"],
                    "pvsBackOffsetRange": viewcell_geometry["candidateBackOffsetRangeM"],
                },
                "files": {
                    **output_meta["files"],
                    **viewcell_geometry["files"],
                },
            }
        )
    (output_dir / "dataset_meta.json").write_text(
        json.dumps(output_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "inputDir": str(input_dir),
        "outputDir": str(output_dir),
        "manifest": str(manifest_dir / "manifest.json"),
        "poseCount": int(poses_out.shape[0]),
        "splitCounts": split_counts,
        "hardlinkedFiles": linked,
        "copiedFiles": copied,
        "candidateSemanticsPreserved": output_meta.get("candidateSemantics"),
        "viewcellGeometryMaterialized": viewcell_geometry is not None,
    }


def main() -> None:
    print(json.dumps(apply_manifest(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
