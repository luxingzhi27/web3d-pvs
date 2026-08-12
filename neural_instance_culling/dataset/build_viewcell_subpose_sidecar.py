#!/usr/bin/env python3
"""Validate source view-cell supervision against a native candidate CSR.

The source directory stores one row per view-cell plus dense-subpose
supervision.  The candidate CSR stores the same visible union and the native
back-camera candidates.  They are validated together, but the main CSR is
never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "pvs-viewcell-subpose-supervision-sidecar-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path, dtype: str) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")
    return np.fromfile(path, dtype=dtype)


def check_offsets(name: str, offsets: np.ndarray, count: int) -> None:
    if offsets.size != count + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(f"invalid {name}: expected {count + 1} monotonic offsets")


def check_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"{name} mismatch between source and main CSR")


def build_sidecar(source_dir: Path, csr_dir: Path, output_dir: Path) -> dict:
    source_meta_path = source_dir / "dataset_meta.json"
    csr_meta_path = csr_dir / "dataset_meta.json"
    if not source_meta_path.is_file() or not csr_meta_path.is_file():
        raise ValueError("both source and main CSR require dataset_meta.json")
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    csr_meta = json.loads(csr_meta_path.read_text(encoding="utf-8"))

    pose_count = int(source_meta.get("viewcellCount", source_meta.get("poseCount", -1)))
    csr_pose_count = int(csr_meta.get("poseCount", -1))
    if pose_count <= 0 or csr_pose_count != pose_count:
        raise ValueError(f"source/CSR pose count mismatch: {pose_count} != {csr_pose_count}")
    source_viewcell_ids = read(source_dir / "viewcell_ids.bin", "<u4")
    if source_viewcell_ids.size != pose_count:
        raise ValueError("viewcell_ids length does not match source viewcell count")
    if not np.array_equal(source_viewcell_ids, np.arange(pose_count, dtype=np.uint32)):
        raise ValueError("source viewcell IDs are not the canonical row order")

    source_visible_offsets = read(source_dir / "visible_offsets.bin", "<u8")
    csr_visible_offsets = read(csr_dir / "visible_offsets.bin", "<u8")
    source_visible_ids = read(source_dir / "visible_ids.bin", "<u4")
    csr_visible_ids = read(csr_dir / "visible_ids.bin", "<u4")
    csr_candidate_offsets = read(csr_dir / "candidate_offsets.bin", "<u8")
    csr_candidate_ids = read(csr_dir / "candidate_ids.bin", "<u4")
    check_offsets("visible_offsets", source_visible_offsets, pose_count)
    check_offsets("csr_visible_offsets", csr_visible_offsets, pose_count)
    check_offsets("candidate_offsets", csr_candidate_offsets, pose_count)
    for name, left, right in (
        ("visible_offsets", source_visible_offsets, csr_visible_offsets),
        ("visible_ids", source_visible_ids, csr_visible_ids),
    ):
        check_equal(name, left, right)
    source_weights = read(source_dir / "visible_weights.bin", "<f4")
    csr_weights = read(csr_dir / "visible_weights.bin", "<f4")
    check_equal("visible_weights", source_weights, csr_weights)

    hits = read(source_dir / "visible_hit_counts.bin", "<u2")
    subpose_offsets = read(source_dir / "subpose_offsets.bin", "<u8")
    if hits.size != source_visible_ids.size:
        raise ValueError("visible_hit_counts length does not match visible_ids")
    check_offsets("subpose_offsets", subpose_offsets, pose_count)
    if subpose_offsets[-1] == 0:
        raise ValueError("subpose_offsets contains no successful subposes")

    if "sourceViewcellCount" in csr_meta and int(csr_meta["sourceViewcellCount"]) != pose_count:
        raise ValueError("main CSR sourceViewcellCount disagrees with poses")
    source_split_path = source_dir / "viewcell_split_ids.bin"
    source_split = read(source_split_path, "u1") if source_split_path.is_file() else np.full(pose_count, 255, dtype=np.uint8)
    if source_split.size != pose_count:
        raise ValueError("source viewcell split IDs do not match viewcell count")
    csr_pose_bytes = read(csr_dir / "poses.bin", "<u1")
    if csr_pose_bytes.size != pose_count * 64:
        raise ValueError("main CSR poses.bin is not a 64-byte pose array")
    csr_split = csr_pose_bytes.reshape(pose_count, 64)[:, 44].astype(np.uint8, copy=False)

    for row in range(pose_count):
        visible = source_visible_ids[source_visible_offsets[row]:source_visible_offsets[row + 1]]
        candidates = csr_candidate_ids[csr_candidate_offsets[row]:csr_candidate_offsets[row + 1]]
        if not np.isin(visible, candidates, assume_unique=False).all():
            raise ValueError(f"visible_ids is not a candidate subset at pose {row}")

    output_dir.mkdir(parents=True, exist_ok=False)
    np.asarray(hits, dtype="<u2").tofile(output_dir / "visible_hit_counts.bin")
    np.asarray(subpose_offsets, dtype="<u8").tofile(output_dir / "subpose_offsets.bin")
    np.arange(pose_count, dtype="<u4").tofile(output_dir / "pose_indices.bin")
    np.asarray(source_split, dtype="u1").tofile(output_dir / "source_viewcell_split_ids.bin")
    np.asarray(csr_split, dtype="u1").tofile(output_dir / "csr_pose_split_ids.bin")
    manifest = {
        "schema": SCHEMA,
        "sourceDataset": str(source_dir.resolve()),
        "mainCsrDataset": str(csr_dir.resolve()),
        "poseCount": pose_count,
        "visibleCount": int(hits.size),
        "successfulSubposeCount": int(subpose_offsets[-1]),
        "sourceSplitCounts": {str(int(x)): int(np.count_nonzero(source_split == x)) for x in np.unique(source_split)},
        "csrSplitCounts": {str(int(x)): int(np.count_nonzero(csr_split == x)) for x in np.unique(csr_split)},
        "semantics": {
            "label": "h > 0",
            "hitCount": "successful subposes seeing the visible instance",
            "hitRate": "h / H; supervision only, never a candidate or GT rewrite",
        },
        "validatedFiles": {
            "mainCsrPoses": sha256(csr_dir / "poses.bin"),
            "sourceVisibleIds": sha256(source_dir / "visible_ids.bin"),
            "mainCsrVisibleIds": sha256(csr_dir / "visible_ids.bin"),
            "sourceVisibleWeights": sha256(source_dir / "visible_weights.bin"),
            "mainCsrVisibleWeights": sha256(csr_dir / "visible_weights.bin"),
            "mainCsrCandidateIds": sha256(csr_dir / "candidate_ids.bin"),
            "mainCsrCandidateOffsets": sha256(csr_dir / "candidate_offsets.bin"),
            "sourceVisibleHitCounts": sha256(source_dir / "visible_hit_counts.bin"),
            "sourceSubposeOffsets": sha256(source_dir / "subpose_offsets.bin"),
        },
        "files": {
            "visibleHitCounts": "visible_hit_counts.bin",
            "subposeOffsets": "subpose_offsets.bin",
            "poseIndices": "pose_indices.bin",
            "sourceViewcellSplitIds": "source_viewcell_split_ids.bin",
            "csrPoseSplitIds": "csr_pose_split_ids.bin",
        },
    }
    (output_dir / "sidecar_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--csr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_sidecar(args.source_dir, args.csr_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
