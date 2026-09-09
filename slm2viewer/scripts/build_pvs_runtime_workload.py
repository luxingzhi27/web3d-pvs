#!/usr/bin/env python3
"""Export one scene's frozen test poses and precomputed candidate lists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    root = Path(__file__).resolve().parents[2]
    default_dataset = root / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1"
    default_output = root / "slm2viewer/assets/benchmark/pvs_v4_frontend_inference_latency_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--scene", default="hkust-v3")
    parser.add_argument("--experiment-name", default="pvs_v4_frontend_inference_latency_v1")
    parser.add_argument("--model-asset-url", default="./assets/neural_instance_culling/pvs_mainline_v4")
    parser.add_argument("--wasm-url", default="./assets/wasm/instance_pvs_v4.wasm")
    parser.add_argument("--fov-y-deg", type=float, default=60.0)
    parser.add_argument("--model-fov-y-deg", type=float, default=66.0)
    parser.add_argument("--expected-test-count", type=int, default=0)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    meta = json.loads(require_file(dataset_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    split_ids = meta.get("splitIds") or {}
    test_split_id = int(split_ids.get("test", -1))
    if test_split_id < 0:
        raise ValueError("dataset_meta.json does not define splitIds.test")

    poses = np.fromfile(require_file(dataset_dir / "poses.bin"), dtype=POSE_DTYPE)
    if poses.size != int(meta.get("poseCount", -1)):
        raise ValueError("poses.bin does not match dataset_meta.json poseCount")
    query_centers = np.fromfile(
        require_file(dataset_dir / "query_center_world.bin"), dtype="<f4"
    ).reshape(-1, 3)
    candidate_offsets = np.fromfile(
        require_file(dataset_dir / "candidate_offsets.bin"), dtype="<u8"
    )
    candidate_ids = np.fromfile(
        require_file(dataset_dir / "candidate_ids.bin"), dtype="<u4"
    )
    if query_centers.shape[0] != poses.size or candidate_offsets.size != poses.size + 1:
        raise ValueError("pose, query-center, and candidate offset counts disagree")
    if int(candidate_offsets[0]) != 0 or int(candidate_offsets[-1]) != candidate_ids.size:
        raise ValueError("candidate offsets are not a valid CSR index")
    if np.any(candidate_offsets[1:] < candidate_offsets[:-1]):
        raise ValueError("candidate offsets are not monotonic")

    test_indices = np.flatnonzero(poses["split"] == test_split_id)
    expected_test_count = int((meta.get("splitCounts") or {}).get("test", -1))
    if test_indices.size != expected_test_count:
        raise ValueError(
            f"dataset declares {expected_test_count} test poses, found {test_indices.size}"
        )
    if args.expected_test_count and test_indices.size != args.expected_test_count:
        raise ValueError(
            f"expected {args.expected_test_count} frozen test poses, found {test_indices.size}"
        )

    exported_ids: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    cursor = 0
    for ordinal, pose_index_value in enumerate(test_indices):
        pose_index = int(pose_index_value)
        start = int(candidate_offsets[pose_index])
        end = int(candidate_offsets[pose_index + 1])
        ids = np.asarray(candidate_ids[start:end], dtype="<u4")
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= int(meta["numInstances"])):
            raise ValueError(f"pose {pose_index} has an out-of-range candidate ID")
        forward = np.asarray(poses["camera_forward"][pose_index], dtype=np.float64)
        norm = float(np.linalg.norm(forward))
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"pose {pose_index} has an invalid camera direction")
        forward /= norm
        tangent = np.asarray(poses["camera_view"][pose_index], dtype=np.float64)
        if not bool(np.isfinite(tangent).all()) or tangent[0] <= 0 or tangent[1] <= 0:
            raise ValueError(f"pose {pose_index} has invalid camera tangents")
        rows.append(
            {
                "ordinal": ordinal,
                "poseId": pose_index,
                "position": query_centers[pose_index].astype(float).tolist(),
                "forward": forward.astype(float).tolist(),
                "aspect": float(tangent[0] / tangent[1]),
                "candidateOffset": cursor,
                "candidateCount": int(ids.size),
                "category": int(poses["category"][pose_index]),
            }
        )
        exported_ids.append(ids)
        cursor += int(ids.size)

    flat_ids = np.concatenate(exported_ids).astype("<u4", copy=False)
    if flat_ids.size != cursor:
        raise AssertionError("candidate concatenation count mismatch")
    candidate_counts = np.asarray([row["candidateCount"] for row in rows], dtype=np.int64)
    sorted_ordinals = np.argsort(candidate_counts, kind="stable")
    warmup_positions = np.linspace(0, sorted_ordinals.size - 1, 50).round().astype(np.int64)
    warmup_ordinals = sorted_ordinals[warmup_positions].astype(int).tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_file = "candidate_ids_uint32.bin"
    flat_ids.tofile(output_dir / candidate_file)
    workload = {
        "schema": "pvs-v4-browser-runtime-workload-v1",
        "experimentName": args.experiment_name,
        "scene": args.scene,
        "split": "test",
        "poseCount": int(test_indices.size),
        "candidateCount": int(flat_ids.size),
        "candidateFile": candidate_file,
        "candidateDtype": "uint32-little-endian",
        "fovYDeg": args.fov_y_deg,
        "modelFovYDeg": args.model_fov_y_deg,
        "aspects": sorted({float(row["aspect"]) for row in rows}),
        "near": 0.1,
        "far": 20000,
        "modelAssetUrl": args.model_asset_url,
        "wasmUrl": args.wasm_url,
        "warmupOrdinals": warmup_ordinals,
        "sessionOrderSeed": 20260909,
        "source": {
            "dataset": dataset_dir.name,
            "candidateSemantics": meta.get("candidateSemantics"),
            "queryCenterSemantics": meta.get("queryCenterSemantics"),
        },
        "candidateStats": {
            "minimum": int(candidate_counts.min()),
            "mean": float(candidate_counts.mean()),
            "p50": float(np.quantile(candidate_counts, 0.5)),
            "p95": float(np.quantile(candidate_counts, 0.95)),
            "maximum": int(candidate_counts.max()),
        },
        "poses": rows,
    }
    (output_dir / "workload.json").write_text(
        json.dumps(workload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir),
        **workload["candidateStats"],
        "poses": int(test_indices.size),
    }, indent=2))


if __name__ == "__main__":
    main()
