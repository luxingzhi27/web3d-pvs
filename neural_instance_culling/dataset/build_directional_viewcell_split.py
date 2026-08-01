#!/usr/bin/env python3
"""Build a deterministic yaw-sector holdout manifest for a PoseCSR dataset.

The spatial protocol keeps physically separated view cells in different
splits.  This protocol answers a different question: can a model query a
camera direction sector that was not present during fitting?  Whole yaw
sectors are assigned to one split, so nearby poses with the same direction
cannot leak across train, validation, calibration, and test.

The script only writes split labels and a provenance manifest.  It never
changes candidates, visible ids, visible weights, camera poses, or FOV.
``apply_pose_split_manifest.py`` can then create a hard-link based PoseCSR
view with the new labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


POSE_DTYPE = np.dtype(
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

SPLIT_IDS = {
    "train": 0,
    "validation": 1,
    "calibration": 2,
    "test": 3,
    "guard": 254,
    "unknown": 255,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yaw-bins", type=int, default=20)
    parser.add_argument("--yaw-origin-deg", type=float, default=0.0)
    parser.add_argument("--validation-bin", type=int, default=14)
    parser.add_argument("--calibration-bin", type=int, default=16)
    parser.add_argument("--test-bin", type=int, default=18)
    parser.add_argument("--guard-bin", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def _validate_bin(value: int, yaw_bins: int, name: str) -> int:
    if value < 0 or value >= yaw_bins:
        raise ValueError(f"{name}={value} must be in [0, {yaw_bins})")
    return int(value)


def _yaw_deg(forward: np.ndarray, origin_deg: float) -> np.ndarray:
    vectors = np.asarray(forward, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("camera_forward contains a zero-length vector")
    vectors = vectors / norms
    # yaw=0 points along -Z, yaw increases toward +X.  The exact convention is
    # recorded in the manifest so another project cannot silently reinterpret it.
    yaw = np.degrees(np.arctan2(vectors[:, 0], -vectors[:, 2]))
    return (yaw - float(origin_deg)) % 360.0


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if args.yaw_bins < 4:
        raise ValueError("--yaw-bins must be at least 4")
    special = {
        "validation": _validate_bin(args.validation_bin, args.yaw_bins, "validation-bin"),
        "calibration": _validate_bin(args.calibration_bin, args.yaw_bins, "calibration-bin"),
        "test": _validate_bin(args.test_bin, args.yaw_bins, "test-bin"),
        "guard": _validate_bin(args.guard_bin, args.yaw_bins, "guard-bin"),
    }
    if len(set(special.values())) != len(special):
        raise ValueError(f"validation/calibration/test/guard bins must be distinct: {special}")

    meta_path = dataset_dir / "dataset_meta.json"
    poses_path = dataset_dir / "poses.bin"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pose_count = int(meta["poseCount"])
    poses = np.memmap(poses_path, dtype=POSE_DTYPE, mode="r", shape=(pose_count,))
    yaw = _yaw_deg(poses["camera_forward"], args.yaw_origin_deg)
    sector_width = 360.0 / float(args.yaw_bins)
    yaw_bins = np.floor(yaw / sector_width).astype(np.int64) % int(args.yaw_bins)
    labels = np.full((pose_count,), SPLIT_IDS["train"], dtype=np.uint8)
    for name, bin_id in special.items():
        labels[yaw_bins == bin_id] = np.uint8(SPLIT_IDS[name])
    group_ids = yaw_bins.astype(np.uint32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    labels.tofile(output_dir / "pose_split_ids.bin")
    group_ids.tofile(output_dir / "pose_group_ids.bin")

    split_names = ("train", "validation", "calibration", "test", "guard")
    split_counts = {
        name: int(np.count_nonzero(labels == SPLIT_IDS[name])) for name in split_names
    }
    category_counts: dict[str, dict[str, int]] = {}
    for name in split_names:
        selected = poses["category"][labels == SPLIT_IDS[name]]
        values, counts = np.unique(selected, return_counts=True)
        category_counts[name] = {str(int(value)): int(count) for value, count in zip(values, counts)}
    bin_counts = {
        str(bin_id): int(np.count_nonzero(yaw_bins == bin_id))
        for bin_id in range(int(args.yaw_bins))
    }
    yaw_ranges = {
        str(bin_id): [
            float(args.yaw_origin_deg + bin_id * sector_width),
            float(args.yaw_origin_deg + (bin_id + 1) * sector_width),
        ]
        for bin_id in range(int(args.yaw_bins))
    }
    manifest = {
        "schema": "directional-yaw-sector-holdout-v1",
        "sourceDataset": str(dataset_dir),
        "sourceDatasetMetaSha256": sha256_file(meta_path),
        "sourcePosesSha256": sha256_file(poses_path),
        "poseCount": pose_count,
        "groupCount": int(args.yaw_bins),
        "groupMode": "camera_forward_yaw_sector",
        "yawConvention": "degrees, yaw=0 along world -Z, positive toward +X",
        "yawOriginDeg": float(args.yaw_origin_deg),
        "yawBins": int(args.yaw_bins),
        "yawBinWidthDeg": float(sector_width),
        "yawBinRangesDeg": yaw_ranges,
        "specialBins": special,
        "selectionSeed": int(args.seed),
        "splitIds": SPLIT_IDS,
        "poseCounts": split_counts,
        "categoryCounts": category_counts,
        "yawBinCounts": bin_counts,
        "pitchRangeDeg": [float(np.degrees(np.arcsin(np.clip(poses["camera_forward"][:, 1], -1.0, 1.0))).min()),
                          float(np.degrees(np.arcsin(np.clip(poses["camera_forward"][:, 1], -1.0, 1.0))).max())],
        "files": {"poseSplitIds": "pose_split_ids.bin", "poseGroupIds": "pose_group_ids.bin"},
        "semantics": (
            "whole yaw sectors are assigned to one split; the held-out split can share spatial positions "
            "with training poses but contains no camera-forward yaw sector used for training"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(build_manifest(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
