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
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from build_directional_viewcell_split import POSE_DTYPE, SPLIT_IDS, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    return parser.parse_args()


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


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
    poses_out.tofile(output_dir / "poses.bin")
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
    }


def main() -> None:
    print(json.dumps(apply_manifest(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
