#!/usr/bin/env python3
"""Split one audited triangle-depth render manifest into contiguous shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def shard_manifest(
    input_path: Path,
    output_dir: Path,
    shard_count: int,
    max_poses: int = 0,
) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("shard count must be positive")
    source = read_json(input_path)
    if source.get("schema") != "triangle-depth-layer-render-manifest-v2":
        raise ValueError("input must be a v2 triangle depth-layer manifest")
    if source.get("formalReady") is not True:
        raise ValueError("only a complete audited manifest may be sharded")
    poses = source.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError("input manifest contains no poses")
    source_pose_count = len(poses)
    if max_poses > 0:
        poses = poses[: int(max_poses)]
    dataset_dir = Path(str(source["datasetDir"]))
    if not dataset_dir.is_absolute():
        dataset_dir = (input_path.parent / dataset_dir).resolve()
    runtime_meta = read_json(Path(str(source["runtimeMeta"])))
    dataset = PoseCSRDataset(dataset_dir, num_instances=int(runtime_meta["instanceCount"]))
    ranges = np.array_split(np.arange(len(poses), dtype=np.int64), shard_count)
    if any(indices.size == 0 for indices in ranges):
        raise ValueError("shard count exceeds pose count")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    canonical_identity = dict(source.get("candidateIdentity") or {})
    for shard_index, indices in enumerate(ranges):
        start = int(indices[0])
        end = int(indices[-1]) + 1
        selected = poses[start:end]
        source_pose_indices = np.asarray(
            [int(row["sourcePoseIndex"]) for row in selected], dtype=np.int64
        )
        payload = dict(source)
        payload["formalReady"] = False
        payload["poses"] = selected
        payload["poseOrdinalRange"] = {
            **(source.get("poseOrdinalRange") or {}),
            "start": start,
            "endExclusive": end,
            "selectedRenderPoseCount": len(selected),
        }
        representative = dict(source.get("representativeSubposeSelection") or {})
        representative["records"] = []
        representative["recordsOmittedFromShard"] = True
        representative["completeRecordsManifest"] = str(input_path.resolve())
        payload["representativeSubposeSelection"] = representative
        payload["candidateIdentity"] = {
            **canonical_identity,
            "renderCandidateDigest": candidate_digest_for_pose_sequence(
                dataset, source_pose_indices
            ),
            "renderCandidateDigestScope": (
                "stored candidate CSR rows repeated in this shard renderPose order"
            ),
            "renderPoseCount": len(selected),
        }
        payload["shard"] = {
            "index": shard_index,
            "count": shard_count,
            "fullManifest": str(input_path.resolve()),
            "poseStart": start,
            "poseEndExclusive": end,
        }
        output_path = output_dir / f"shard_{shard_index:02d}.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.append(
            {
                "shard": shard_index,
                "output": str(output_path),
                "poseStart": start,
                "poseEndExclusive": end,
                "poseCount": len(selected),
            }
        )
    return {
        "schema": "triangle-depth-layer-render-manifest-shards-v1",
        "input": str(input_path.resolve()),
        "shardCount": shard_count,
        "poseCount": len(poses),
        "sourcePoseCount": source_pose_count,
        "truncatedForSmoke": bool(max_poses > 0 and len(poses) < source_pose_count),
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--max-poses",
        type=int,
        default=0,
        help="Optional smoke-only prefix limit; zero keeps the complete manifest.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            shard_manifest(
                args.input.resolve(),
                args.output_dir.resolve(),
                args.shards,
                args.max_poses,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
