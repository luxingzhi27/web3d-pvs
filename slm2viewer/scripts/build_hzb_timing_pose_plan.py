#!/usr/bin/env python3
"""Build a deterministic candidate-count-stratified HZB timing pose plan."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible HZB timing plan stratified by candidate count."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pose-count", type=int, default=120)
    parser.add_argument("--strata-count", type=int, default=6)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def select_evenly(values: np.ndarray, count: int) -> np.ndarray:
    if count < 0 or count > values.size:
        raise ValueError("stratum selection count is out of range")
    if count == 0:
        return np.empty((0,), dtype=values.dtype)
    if count == 1:
        return values[[0]]
    positions = np.rint(np.linspace(0, values.size - 1, count)).astype(np.int64)
    if np.unique(positions).size != count:
        raise ValueError("evenly spaced stratum selection produced duplicate positions")
    return values[positions]


def build_plan(
    dataset_dir: Path,
    output: Path,
    split: str = "test",
    pose_count: int = 120,
    strata_count: int = 6,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    meta = read_json(require_file(dataset_dir / "dataset_meta.json"))
    if pose_count < 1:
        raise ValueError("pose-count must be positive")
    if strata_count < 1:
        raise ValueError("strata-count must be positive")
    split_ids = meta.get("splitIds") or {}
    if split not in split_ids:
        raise ValueError(f"dataset_meta.json has no split {split}")

    pose_count_in_meta = int(meta.get("poseCount", -1))
    poses = np.fromfile(require_file(dataset_dir / "poses.bin"), dtype=POSE_DTYPE)
    if poses.size != pose_count_in_meta:
        raise ValueError("poses.bin does not match dataset_meta.json poseCount")
    offsets = np.fromfile(require_file(dataset_dir / "candidate_offsets.bin"), dtype="<u8")
    candidate_ids_path = require_file(dataset_dir / "candidate_ids.bin")
    candidate_id_bytes = candidate_ids_path.stat().st_size
    if offsets.size != poses.size + 1:
        raise ValueError("candidate_offsets.bin does not match poseCount")
    if (
        offsets[0] != 0
        or candidate_id_bytes % np.dtype("<u4").itemsize != 0
        or offsets[-1] != candidate_id_bytes // np.dtype("<u4").itemsize
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("candidate CSR offsets are invalid")

    eligible = np.flatnonzero(poses["split"] == int(split_ids[split])).astype(np.int64)
    if eligible.size < pose_count:
        raise ValueError(
            f"split {split} has {eligible.size} poses, cannot select {pose_count}"
        )
    strata_count = min(strata_count, int(eligible.size), pose_count)
    candidate_counts = offsets[1:] - offsets[:-1]
    ordered = eligible[np.argsort(candidate_counts[eligible], kind="stable")]
    strata = np.array_split(ordered, strata_count)
    quotas = np.full((strata_count,), pose_count // strata_count, dtype=np.int64)
    quotas[: pose_count % strata_count] += 1

    selected_indices: list[int] = []
    stratum_records: list[dict[str, Any]] = []
    for stratum_index, (source_indices, quota) in enumerate(zip(strata, quotas.tolist())):
        chosen = select_evenly(source_indices, int(quota))
        selected_indices.extend(int(value) for value in chosen.tolist())
        counts = candidate_counts[source_indices]
        stratum_records.append(
            {
                "index": stratum_index,
                "sourcePoseCount": int(source_indices.size),
                "selectedPoseCount": int(chosen.size),
                "candidateCountMin": int(counts.min()),
                "candidateCountMax": int(counts.max()),
                "selectedPoseIndices": [int(value) for value in chosen.tolist()],
                "selectedCandidateCounts": [int(value) for value in candidate_counts[chosen].tolist()],
            }
        )

    selected_poses: list[dict[str, Any]] = []
    for ordinal, pose_index in enumerate(selected_indices):
        tan_x, tan_y = (float(value) for value in poses["camera_view"][pose_index])
        aspect = tan_x / tan_y if np.isfinite(tan_x) and np.isfinite(tan_y) and tan_x > 0 and tan_y > 0 else None
        selected_poses.append(
            {
                "ordinal": ordinal,
                "poseIndex": pose_index,
                "candidateCount": int(candidate_counts[pose_index]),
                "aspect": aspect,
            }
        )

    result = {
        "schema": "geometry-shell-hzb-timing-pose-index-plan-v1",
        "split": split,
        "selection": "candidate-count-stratified-evenly-spaced",
        "targetPoseCount": pose_count,
        "strataCount": strata_count,
        "strata": stratum_records,
        "poseIndices": selected_indices,
        "poses": selected_poses,
        "source": {
            "datasetDir": str(dataset_dir),
            "datasetDirName": dataset_dir.name,
            "poseCount": int(poses.size),
            "splitPoseCount": int(eligible.size),
            "candidateSemantics": meta.get("candidateSemantics"),
        },
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = build_plan(
        args.dataset_dir,
        args.output,
        split=args.split,
        pose_count=args.pose_count,
        strata_count=args.strata_count,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "split": result["split"],
        "poseCount": result["targetPoseCount"],
        "strataCount": result["strataCount"],
        "poseIndices": result["poseIndices"],
    }, indent=2))


if __name__ == "__main__":
    main()
