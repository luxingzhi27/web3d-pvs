#!/usr/bin/env python3
"""Rebuild raw AABB candidates and native four-way splits for a pose CSR dataset."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from pose_csr_dataset import DIRECTIONAL_POSE_DTYPE, frustum_candidate_ids_for_pose  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_aabbs(runtime_meta: dict) -> np.ndarray:
    records = runtime_meta.get("componentRecords") or []
    count = max(int(row["componentGlobalId"]) for row in records) + 1 if records else 0
    aabbs = np.zeros((count, 6), dtype=np.float32)
    for row in records:
        idx = int(row["componentGlobalId"])
        bounds = row["bounds"]
        if "min" in bounds and "max" in bounds:
            minimum = np.asarray(bounds["min"], dtype=np.float32)
            maximum = np.asarray(bounds["max"], dtype=np.float32)
        else:
            center = np.asarray(bounds["center"], dtype=np.float32)
            size = np.asarray(bounds["size"], dtype=np.float32)
            minimum = center - size * 0.5
            maximum = center + size * 0.5
        aabbs[idx, :3] = minimum
        aabbs[idx, 3:] = maximum
    return aabbs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--allow-candidate-visible-union", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_meta = read_json(input_dir / "dataset_meta.json")
    runtime_meta = read_json(args.runtime_meta)
    split_manifest = read_json(args.split_manifest / "manifest.json")
    count = int(input_meta["poseCount"])
    poses = np.memmap(input_dir / "poses.bin", dtype=DIRECTIONAL_POSE_DTYPE, mode="r", shape=(count,))
    split_ids = np.memmap(args.split_manifest / "pose_split_ids.bin", dtype=np.uint8, mode="r")
    if split_ids.size != count:
        raise ValueError(f"split manifest pose count {split_ids.size} != CSR pose count {count}")
    aabbs = load_aabbs(runtime_meta)
    visible_offsets = np.memmap(input_dir / "visible_offsets.bin", dtype=np.uint64, mode="r")
    visible_ids = np.memmap(input_dir / "visible_ids.bin", dtype=np.uint32, mode="r")
    visible_weights_path = input_dir / "visible_weights.bin"
    visible_weights = np.memmap(visible_weights_path, dtype=np.float32, mode="r")

    candidate_offsets = np.zeros((count + 1,), dtype=np.uint64)
    raw_offsets = np.zeros((count + 1,), dtype=np.uint64)
    stats = {"rawCandidateMissVisible": 0, "candidateMissVisible": 0, "maxCandidate": 0, "totalCandidateRefs": 0}
    candidate_path = output_dir / "candidate_ids.bin"
    raw_path = output_dir / "raw_candidate_ids.bin"
    with candidate_path.open("wb") as candidate_file, raw_path.open("wb") as raw_file:
        for index in range(count):
            pose = poses[index]
            raw = frustum_candidate_ids_for_pose(
                np.asarray(pose["camera_world"], dtype=np.float32),
                np.asarray(pose["camera_forward"], dtype=np.float32),
                float(pose["camera_view"][0]),
                float(pose["camera_view"][1]),
                aabbs,
                near=float(args.near),
            )
            raw = np.asarray(raw, dtype=np.uint32)
            raw_file.write(raw.tobytes(order="C"))
            raw_offsets[index + 1] = raw_offsets[index] + int(raw.size)
            visible_start = int(visible_offsets[index])
            visible_end = int(visible_offsets[index + 1])
            visible = np.asarray(visible_ids[visible_start:visible_end], dtype=np.uint32)
            missing = np.setdiff1d(visible, raw, assume_unique=False)
            stats["rawCandidateMissVisible"] += int(missing.size)
            if missing.size and not args.allow_candidate_visible_union:
                candidates = raw
                stats["candidateMissVisible"] += int(missing.size)
            elif args.allow_candidate_visible_union:
                candidates = np.union1d(raw, visible).astype(np.uint32, copy=False)
                stats["candidateMissVisible"] += int(missing.size)
            else:
                candidates = raw
            candidate_file.write(candidates.tobytes(order="C"))
            candidate_offsets[index + 1] = candidate_offsets[index] + int(candidates.size)
            stats["totalCandidateRefs"] += int(candidates.size)
            stats["maxCandidate"] = max(stats["maxCandidate"], int(candidates.size))

    poses_out = np.asarray(poses, dtype=DIRECTIONAL_POSE_DTYPE).copy()
    poses_out["split"] = np.asarray(split_ids, dtype=np.uint8)
    poses_out.tofile(output_dir / "poses.bin")
    for name in ["mvp.bin", "visible_offsets.bin", "visible_ids.bin", "visible_hit_counts.bin"]:
        source = input_dir / name
        if source.exists():
            shutil.copyfile(source, output_dir / name)
    visible_weights.astype("<f4", copy=False).tofile(output_dir / "visible_weights.bin")
    candidate_offsets.tofile(output_dir / "candidate_offsets.bin")
    raw_offsets.tofile(output_dir / "raw_candidate_offsets.bin")
    raw_offsets.tofile(output_dir / "frustum_offsets.bin")
    frustum_path = output_dir / "frustum_ids.bin"
    if frustum_path.exists():
        frustum_path.unlink()
    try:
        os.link(raw_path, frustum_path)
    except OSError:
        shutil.copyfile(raw_path, frustum_path)

    if stats["rawCandidateMissVisible"] and not args.allow_candidate_visible_union:
        raise RuntimeError(
            "Raw AABB candidates miss visible instances; formal dataset was not declared usable. "
            f"missed references={stats['rawCandidateMissVisible']}"
        )

    output_meta = dict(input_meta)
    output_meta.update(
        {
            "schema": "pose-csr-spatial-four-way-raw-candidate-v1",
            "sourceDataset": str(input_dir),
            "runtimeMeta": str(args.runtime_meta),
            "splitManifest": str(args.split_manifest),
            "splitIds": split_manifest["splitIds"],
            "candidateSemantics": (
                "raw back-camera AABB candidates; no GT positive union"
                if not args.allow_candidate_visible_union
                else "raw back-camera AABB candidates plus GT visible positives (exploratory only)"
            ),
            "rawCandidateSemantics": "AABB candidates computed from stored pose camera_world, camera_forward and tanHalfFovX/Y before any union",
            "rawCandidateFile": "raw_candidate_ids.bin",
            "rawCandidateOffsets": "raw_candidate_offsets.bin",
            "stats": {**(input_meta.get("stats") or {}), **stats},
            "files": {
                **(input_meta.get("files") or {}),
                "poses": "poses.bin",
                "candidateIds": "candidate_ids.bin",
                "candidateOffsets": "candidate_offsets.bin",
                "rawCandidateIds": "raw_candidate_ids.bin",
                "rawCandidateOffsets": "raw_candidate_offsets.bin",
                "frustumIds": "frustum_ids.bin",
                "frustumOffsets": "frustum_offsets.bin",
            },
        }
    )
    (output_dir / "dataset_meta.json").write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outputDir": str(output_dir), "stats": stats, "splitIds": split_manifest["splitIds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
