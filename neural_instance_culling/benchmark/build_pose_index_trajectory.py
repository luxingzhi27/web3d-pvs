#!/usr/bin/env python3
"""Build a deterministic pose-index trajectory for offline replay experiments.

This tool selects rows from an existing native CSR split.  It does not claim
that the resulting order is a real user navigation trace; live traces must be
stored separately and identified as such in M8 reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
import sys

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from pose_csr_dataset import PoseCSRDataset  # noqa: E402


TRAJECTORY_SCHEMA = "neuralstreamweb3d-pose-index-trajectory-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_indices(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_payload(
    dataset_dir: Path,
    split_name: str,
    pose_indices: np.ndarray,
    *,
    step_ms: float,
    bandwidth_bytes_per_sec: float,
    request_latency_ms: float,
    max_concurrent_downloads: int,
    max_concurrent_decode_uploads: int,
    cache_mode: str,
    initial_cache_glb_ids: list[int],
) -> dict[str, Any]:
    meta = json.loads((dataset_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    rows = [
        {"poseIndex": int(pose_index), "timeMs": float(sequence * step_ms)}
        for sequence, pose_index in enumerate(pose_indices.tolist())
    ]
    return {
        "schema": TRAJECTORY_SCHEMA,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "formal": False,
        "status": "deterministic_split_replay_not_live_navigation",
        "datasetDir": str(dataset_dir.resolve()),
        "datasetMetaSha256": sha256_file(dataset_dir / "dataset_meta.json"),
        "split": split_name,
        "selection": {
            "mode": "native_split_ascending_stride",
            "stride": None,
            "poseCount": len(rows),
            "poseIndexSha256": sha256_indices(pose_indices),
            "warning": "This order is a deterministic replay input, not a measured user navigation trace.",
        },
        "camera": {
            "modelInputFovYDeg": meta.get("modelInputFovYDeg"),
            "frontendRenderFovYDeg": meta.get("frontendRenderFovYDeg"),
        },
        "poses": rows,
        "network": {
            "bandwidthBytesPerSec": float(bandwidth_bytes_per_sec),
            "requestLatencyMs": float(request_latency_ms),
            "maxConcurrentDownloads": int(max_concurrent_downloads),
            "maxConcurrentDecodeUploads": int(max_concurrent_decode_uploads),
            "measurementStatus": "assumed_not_device_measurement",
        },
        "cache": {
            "mode": cache_mode,
            "initialGlbIds": [int(value) for value in initial_cache_glb_ids],
        },
    }


def run_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "dataset_meta.json").write_text(
            json.dumps({"modelInputFovYDeg": 66, "frontendRenderFovYDeg": 60}), encoding="utf-8"
        )
        payload = build_payload(
            root,
            "validation",
            np.asarray([4, 9], dtype=np.int64),
            step_ms=250.0,
            bandwidth_bytes_per_sec=1_000_000.0,
            request_latency_ms=20.0,
            max_concurrent_downloads=2,
            max_concurrent_decode_uploads=1,
            cache_mode="cold",
            initial_cache_glb_ids=[],
        )
        assert payload["schema"] == TRAJECTORY_SCHEMA
        assert [row["timeMs"] for row in payload["poses"]] == [0.0, 250.0]
        assert payload["formal"] is False
        assert payload["camera"]["modelInputFovYDeg"] == 66
    return {"status": "passed", "checked": ["schema", "pose timing", "split provenance", "FOV metadata", "non-formal marker"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic pose-index replay trajectory.")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-poses", type=int, default=0, help="0 keeps every selected split pose")
    parser.add_argument("--step-ms", type=float, default=500.0)
    parser.add_argument("--bandwidth-bytes-per-sec", type=float, default=5_000_000.0)
    parser.add_argument("--request-latency-ms", type=float, default=50.0)
    parser.add_argument("--max-concurrent-downloads", type=int, default=6)
    parser.add_argument("--max-concurrent-decode-uploads", type=int, default=2)
    parser.add_argument("--cache-mode", choices=("cold", "warm", "warm-all"), default="cold")
    parser.add_argument("--initial-cache-glbs", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return
    if args.dataset_dir is None or args.output is None:
        parser.error("--dataset-dir and --output are required unless --self-test is used.")
    if args.stride <= 0 or args.max_poses < 0 or args.step_ms <= 0:
        parser.error("--stride and --step-ms must be positive; --max-poses must be non-negative.")
    if args.bandwidth_bytes_per_sec <= 0 or args.request_latency_ms < 0:
        parser.error("Network bandwidth must be positive and request latency must be non-negative.")
    if args.max_concurrent_downloads <= 0 or args.max_concurrent_decode_uploads <= 0:
        parser.error("Concurrency limits must be positive.")
    dataset_dir = args.dataset_dir.resolve()
    meta = json.loads((dataset_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    dataset = PoseCSRDataset(dataset_dir, num_instances=int(meta["numInstances"]))
    if args.split not in dataset.split_ids:
        raise ValueError(f"Split {args.split!r} is unavailable; found {sorted(dataset.split_ids)}")
    indices = np.asarray(dataset.split(args.split).pose_indices, dtype=np.int64)[:: args.stride]
    if args.max_poses:
        indices = indices[: args.max_poses]
    if indices.size == 0:
        raise ValueError("The selected split/stride produced no poses.")
    initial = []
    if args.initial_cache_glbs.strip():
        initial = [int(value.strip()) for value in args.initial_cache_glbs.split(",") if value.strip()]
        if any(value < 0 for value in initial):
            raise ValueError("Initial GLB ids must be non-negative.")
    payload = build_payload(
        dataset_dir,
        args.split,
        indices,
        step_ms=args.step_ms,
        bandwidth_bytes_per_sec=args.bandwidth_bytes_per_sec,
        request_latency_ms=args.request_latency_ms,
        max_concurrent_downloads=args.max_concurrent_downloads,
        max_concurrent_decode_uploads=args.max_concurrent_decode_uploads,
        cache_mode=args.cache_mode,
        initial_cache_glb_ids=initial,
    )
    payload["selection"]["stride"] = int(args.stride)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(output), "poseCount": int(indices.size), "formal": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
