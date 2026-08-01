#!/usr/bin/env python3
"""Create a PoseCSR copy with an immutable spatial four-way split manifest.

The spatial splitter intentionally writes only a manifest and pose labels. This
utility applies those labels to a candidate-complete PoseCSR dataset without
recomputing or changing any visibility/candidate arrays. Files are hard-linked
when possible so a formal copy does not duplicate large CSR payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

POSE_DTYPE = np.dtype(
    {
        "names": ["camera_norm", "camera_world", "split", "category", "reserved0", "reserved1"],
        "formats": [("<f4", (3,)), ("<f4", (3,)), "u1", "u1", "<u2", "<u4"],
        "offsets": [0, 12, 24, 25, 26, 28],
        "itemsize": 32,
    }
)

DIRECTIONAL_POSE_DTYPE = np.dtype(
    {
        "names": [
            "camera_norm",
            "camera_world",
            "camera_forward",
            "camera_view",
            "split",
            "category",
            "reserved0",
            "reserved1",
        ],
        "formats": [
            ("<f4", (3,)),
            ("<f4", (3,)),
            ("<f4", (3,)),
            ("<f4", (2,)),
            "u1",
            "u1",
            "<u2",
            "<u4",
        ],
        "offsets": [0, 12, 24, 36, 44, 45, 46, 48],
        "itemsize": 64,
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def link_tree(source_dir: Path, output_dir: Path, excluded: set[str]) -> None:
    for source in source_dir.iterdir():
        if source.name in excluded:
            continue
        target = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target)
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output_dir}; pass --force only for a deliberate rebuild.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    source_meta = read_json(source_dir / "dataset_meta.json")
    manifest = read_json(manifest_dir / "manifest.json")
    source_meta_sha256 = sha256_file(source_dir / "dataset_meta.json")
    expected_source_meta_sha256 = str(manifest.get("sourceDatasetMetaSha256", ""))
    if expected_source_meta_sha256 and source_meta_sha256 != expected_source_meta_sha256:
        raise ValueError(
            "Spatial split manifest was built from a different dataset metadata file: "
            f"expected {expected_source_meta_sha256}, got {source_meta_sha256}."
        )
    pose_count = int(source_meta["poseCount"])
    split_ids_path = manifest_dir / "pose_split_ids.bin"
    split_ids = np.fromfile(split_ids_path, dtype=np.uint8)
    if split_ids.size != pose_count or int(manifest.get("poseCount", -1)) != pose_count:
        raise ValueError(
            f"Manifest pose count {split_ids.size}/{manifest.get('poseCount')} does not match dataset {pose_count}."
        )
    manifest_split_ids = {
        str(name): int(value) for name, value in (manifest.get("splitIds") or {}).items()
    }
    required_split_ids = {"train": 0, "validation": 1, "calibration": 2, "test": 3, "guard": 254}
    if any(manifest_split_ids.get(name) != value for name, value in required_split_ids.items()):
        raise ValueError(f"Manifest splitIds do not match the formal protocol: {manifest_split_ids}")

    pose_stride = int(source_meta.get("poseStrideBytes", DIRECTIONAL_POSE_DTYPE.itemsize))
    if pose_stride == DIRECTIONAL_POSE_DTYPE.itemsize:
        pose_dtype = DIRECTIONAL_POSE_DTYPE
    elif pose_stride == POSE_DTYPE.itemsize:
        pose_dtype = POSE_DTYPE
    else:
        raise ValueError(f"Unsupported PoseCSR stride: {pose_stride}")
    poses = np.fromfile(source_dir / "poses.bin", dtype=pose_dtype)
    if poses.size != pose_count:
        raise ValueError(f"poses.bin contains {poses.size} rows, expected {pose_count}")

    link_tree(source_dir, output_dir, {"poses.bin", "dataset_meta.json"})
    poses = poses.copy()
    poses["split"] = split_ids
    poses.tofile(output_dir / "poses.bin")

    actual_pose_counts = {
        name: int(np.count_nonzero(split_ids == split_id))
        for name, split_id in manifest_split_ids.items()
        if name != "unknown"
    }
    expected_pose_counts = {
        str(name): int(count) for name, count in (manifest.get("poseCounts") or {}).items()
    }
    if actual_pose_counts != expected_pose_counts:
        raise ValueError(
            "Applied pose labels do not match the immutable manifest counts: "
            f"expected {expected_pose_counts}, got {actual_pose_counts}."
        )

    output_meta = dict(source_meta)
    output_meta.update(
        {
            "schema": "pose-csr-spatial-four-way-v2",
            "sourceDataset": str(source_dir),
            "sourceDatasetMetaSha256": source_meta_sha256,
            "spatialSplitManifest": str(manifest_dir),
            "spatialSplitManifestSha256": sha256_file(manifest_dir / "manifest.json"),
            "spatialSplitPoseLabelsSha256": sha256_file(split_ids_path),
            "splitIds": manifest_split_ids,
            "splitCounts": expected_pose_counts,
            "splitSemantics": "immutable spatial block train/validation/calibration/test labels with a guard band excluded from training",
            "spatialSplit": manifest,
            "files": {**(source_meta.get("files") or {}), "poses": "poses.bin"},
        }
    )
    (output_dir / "dataset_meta.json").write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "outputDir": str(output_dir),
                "poseCount": pose_count,
                "poseCounts": actual_pose_counts,
                "sourceDatasetMetaSha256": output_meta["sourceDatasetMetaSha256"],
                "spatialSplitManifestSha256": output_meta["spatialSplitManifestSha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
