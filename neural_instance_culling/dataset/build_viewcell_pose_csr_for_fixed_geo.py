#!/usr/bin/env python3
"""Convert proxy viewcell PVS data into directional pose CSR.

Each viewcell becomes one conservative region-PVS pose:
  camera = viewcell center moved backward along the main forward vector
  FOV    = expanded PVS FOV stored in viewcell_params.bin
  GT     = dense subpose visible-id union already stored in the viewcell dataset
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build back-camera pose CSR from proxy viewcell PVS dataset.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Historical proxy viewcell dataset directory; this source is not retained by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output PoseCSR directory. Pass an explicit directory to avoid overwriting retained data.",
    )
    parser.add_argument("--runtime-meta", type=Path, default=Path("hkust-v3/assets/runtimeVisibilityMeta.json"))
    parser.add_argument("--experiment", default="pvs_directional_occlusion_proxy_encoder_rvl_w042_full40")
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional spatial four-way manifest directory containing manifest.json and pose_split_ids.bin.",
    )
    parser.add_argument(
        "--allow-candidate-visible-union",
        action="store_true",
        help="Exploratory-only compatibility mode that adds missed visible IDs to the AABB candidate set.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_min_max(scene_bounds: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "min" in scene_bounds and "max" in scene_bounds:
        mn = np.asarray(scene_bounds["min"], dtype=np.float32)
        mx = np.asarray(scene_bounds["max"], dtype=np.float32)
        return mn, mx, mx - mn
    center = np.asarray(scene_bounds["center"], dtype=np.float32)
    size = np.asarray(scene_bounds["size"], dtype=np.float32)
    return center - size * 0.5, center + size * 0.5, size


def bounds_min_max(bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    if "min" in bounds and "max" in bounds:
        return np.asarray(bounds["min"], dtype=np.float32), np.asarray(bounds["max"], dtype=np.float32)
    center = np.asarray(bounds["center"], dtype=np.float32)
    size = np.asarray(bounds["size"], dtype=np.float32)
    return center - size * 0.5, center + size * 0.5


def load_world_aabbs(runtime_meta: dict) -> np.ndarray:
    records = runtime_meta.get("componentRecords", [])
    count = max(int(r["componentGlobalId"]) for r in records) + 1 if records else 0
    aabbs = np.zeros((count, 6), dtype=np.float32)
    for record in records:
        idx = int(record["componentGlobalId"])
        mn, mx = bounds_min_max(record["bounds"])
        aabbs[idx, :3] = mn
        aabbs[idx, 3:] = mx
    return aabbs


def normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return fallback.astype(np.float32, copy=True)
    return (v / n).astype(np.float32, copy=False)


def camera_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = normalize(forward, np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    up_ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(f, up_ref))) > 0.98:
        up_ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize(np.cross(f, up_ref), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    up = normalize(np.cross(right, f), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    return f, right, up


def conservative_mvp(camera_world: np.ndarray, forward: np.ndarray, tan_x: float, tan_y: float) -> np.ndarray:
    """Build a projection matrix in the same column-major convention used by training code.

    The matrix is only required to provide stable screen coordinates and positive depth
    for candidates. It intentionally uses depth as clip W, matching existing projection
    feature code.
    """
    f, right, up = camera_basis(forward)
    cam = camera_world.astype(np.float32, copy=False)
    tx = max(float(tan_x), 1e-4)
    ty = max(float(tan_y), 1e-4)
    e = np.zeros((16,), dtype=np.float32)
    e[0], e[4], e[8], e[12] = right[0] / tx, right[1] / tx, right[2] / tx, -float(np.dot(right, cam)) / tx
    e[1], e[5], e[9], e[13] = up[0] / ty, up[1] / ty, up[2] / ty, -float(np.dot(up, cam)) / ty
    e[2], e[6], e[10], e[14] = f[0], f[1], f[2], -float(np.dot(f, cam))
    e[3], e[7], e[11], e[15] = f[0], f[1], f[2], -float(np.dot(f, cam))
    return e


def frustum_candidate_ids(
    camera_world: np.ndarray,
    forward: np.ndarray,
    tan_x: float,
    tan_y: float,
    aabbs: np.ndarray,
    near: float,
) -> np.ndarray:
    if aabbs.size == 0:
        return np.zeros((0,), dtype=np.uint32)
    f, right, up = camera_basis(forward)
    mins = aabbs[:, :3]
    maxs = aabbs[:, 3:]
    centers = (mins + maxs) * 0.5
    extents = np.maximum((maxs - mins) * 0.5, 0.0)
    d = centers - camera_world.astype(np.float32, copy=False)[None, :]
    z = d @ f
    x = d @ right
    y = d @ up
    rz = extents @ np.abs(f)
    rx = extents @ np.abs(right)
    ry = extents @ np.abs(up)
    front = z + rz >= float(near)
    x_inside = (np.abs(x) - rx) <= ((z + rz) * max(float(tan_x), 1e-4))
    y_inside = (np.abs(y) - ry) <= ((z + rz) * max(float(tan_y), 1e-4))
    return np.flatnonzero(front & x_inside & y_inside).astype(np.uint32)


def normalize_points(points: np.ndarray, bounds: dict) -> np.ndarray:
    mn, _mx, size = scene_min_max(bounds)
    return np.clip((points - mn[None, :]) / np.maximum(size[None, :], 1e-6), 0.0, 1.0).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_meta = read_json(input_dir / "dataset_meta.json")
    runtime_meta = read_json(args.runtime_meta)
    aabbs = load_world_aabbs(runtime_meta)

    count = int(input_meta["viewcellCount"])
    centers = np.memmap(input_dir / "viewcell_centers.bin", dtype=np.float32, mode="r").reshape(count, 3)
    forwards = np.memmap(input_dir / "viewcell_forwards.bin", dtype=np.float32, mode="r").reshape(count, 3)
    params = np.memmap(input_dir / "viewcell_params.bin", dtype=np.float32, mode="r").reshape(count, 8)
    split_ids = np.memmap(input_dir / "viewcell_split_ids.bin", dtype=np.uint8, mode="r")
    category_ids = np.memmap(input_dir / "viewcell_category_ids.bin", dtype=np.uint8, mode="r")
    visible_offsets = np.memmap(input_dir / "visible_offsets.bin", dtype=np.uint64, mode="r")
    visible_ids = np.memmap(input_dir / "visible_ids.bin", dtype=np.uint32, mode="r")
    visible_weights = np.memmap(input_dir / "visible_weights.bin", dtype=np.float32, mode="r")
    split_manifest = None
    if args.split_manifest is not None:
        split_manifest = read_json(args.split_manifest / "manifest.json")
        manifest_split_ids = np.memmap(args.split_manifest / "pose_split_ids.bin", dtype=np.uint8, mode="r")
        if manifest_split_ids.size != count:
            raise ValueError(
                f"Spatial split pose count {manifest_split_ids.size} does not match source viewcell count {count}"
            )
        split_ids = manifest_split_ids

    back_offsets = params[:, 4]
    normalized_forward = np.asarray([normalize(v, np.asarray([0.0, 0.0, -1.0], dtype=np.float32)) for v in forwards], dtype=np.float32)
    camera_world = centers - normalized_forward * back_offsets[:, None]
    pvs_fov_y = params[:, 2]
    pvs_fov_x = params[:, 3]
    tan_y = np.tan(np.deg2rad(pvs_fov_y) * 0.5).astype(np.float32)
    tan_x = np.tan(np.deg2rad(pvs_fov_x) * 0.5).astype(np.float32)

    scene_min, scene_max, _scene_size = scene_min_max(runtime_meta["sceneBounds"])
    cam_min = np.minimum(scene_min, camera_world.min(axis=0))
    cam_max = np.maximum(scene_max, camera_world.max(axis=0))
    pad = np.maximum((cam_max - cam_min) * 0.05, 1.0)
    camera_bounds = {
        "min": (cam_min - pad).astype(float).tolist(),
        "max": (cam_max + pad).astype(float).tolist(),
        "center": ((cam_min + cam_max) * 0.5).astype(float).tolist(),
        "size": ((cam_max - cam_min) + pad * 2.0).astype(float).tolist(),
    }
    camera_norm = normalize_points(camera_world, camera_bounds)

    poses = np.zeros((count,), dtype=DIRECTIONAL_POSE_DTYPE)
    poses["camera_norm"] = camera_norm
    poses["camera_world"] = camera_world.astype(np.float32, copy=False)
    poses["camera_forward"] = normalized_forward
    poses["camera_view"] = np.stack([tan_x, tan_y], axis=1).astype(np.float32, copy=False)
    poses["split"] = np.asarray(split_ids, dtype=np.uint8)
    poses["category"] = np.asarray(category_ids, dtype=np.uint8)

    mvps = np.zeros((count, 16), dtype=np.float32)
    candidate_offsets = np.zeros((count + 1,), dtype=np.uint64)
    raw_candidate_offsets = np.zeros((count + 1,), dtype=np.uint64)
    stats = {
        "candidateMissVisible": 0,
        "rawCandidateMissVisible": 0,
        "totalCandidateRefs": 0,
        "maxCandidate": 0,
        "emptyCandidate": 0,
        "visibleSubsetFailures": 0,
    }
    candidate_path = output_dir / "candidate_ids.bin"
    raw_candidate_path = output_dir / "raw_candidate_ids.bin"
    with candidate_path.open("wb") as f, raw_candidate_path.open("wb") as raw_f:
        for i in tqdm(range(count), desc="viewcell-back-camera-csr"):
            mvps[i] = conservative_mvp(camera_world[i], normalized_forward[i], float(tan_x[i]), float(tan_y[i]))
            candidates = frustum_candidate_ids(camera_world[i], normalized_forward[i], float(tan_x[i]), float(tan_y[i]), aabbs, args.near)
            raw_candidates = np.asarray(candidates, dtype=np.uint32)
            raw_f.write(raw_candidates.tobytes(order="C"))
            raw_candidate_offsets[i + 1] = raw_candidate_offsets[i] + int(raw_candidates.size)
            v_start = int(visible_offsets[i])
            v_end = int(visible_offsets[i + 1])
            visible = np.asarray(visible_ids[v_start:v_end], dtype=np.uint32)
            before = set(int(v) for v in raw_candidates.tolist())
            missing = np.setdiff1d(visible, raw_candidates, assume_unique=False)
            stats["rawCandidateMissVisible"] += int(missing.size)
            if missing.size and not args.allow_candidate_visible_union:
                stats["candidateMissVisible"] += int(missing.size)
                candidates = raw_candidates
            else:
                for vid in visible.tolist():
                    if int(vid) not in before:
                        stats["candidateMissVisible"] += 1
                        before.add(int(vid))
                candidates = np.asarray(sorted(before), dtype=np.uint32)
            if candidates.size == 0:
                stats["emptyCandidate"] += 1
            if not np.all(np.isin(visible, candidates, assume_unique=False)):
                stats["visibleSubsetFailures"] += 1
            f.write(candidates.tobytes(order="C"))
            candidate_offsets[i + 1] = candidate_offsets[i] + int(candidates.size)
            stats["totalCandidateRefs"] += int(candidates.size)
            stats["maxCandidate"] = max(int(stats["maxCandidate"]), int(candidates.size))

    poses.tofile(output_dir / "poses.bin")
    mvps.tofile(output_dir / "mvp.bin")
    candidate_offsets.tofile(output_dir / "candidate_offsets.bin")
    raw_candidate_offsets.tofile(output_dir / "raw_candidate_offsets.bin")
    candidate_offsets.tofile(output_dir / "frustum_offsets.bin")
    try:
        frustum_path = output_dir / "frustum_ids.bin"
        if frustum_path.exists():
            frustum_path.unlink()
        os.link(candidate_path, frustum_path)
    except OSError:
        shutil.copyfile(candidate_path, output_dir / "frustum_ids.bin")
    shutil.copyfile(input_dir / "visible_offsets.bin", output_dir / "visible_offsets.bin")
    shutil.copyfile(input_dir / "visible_ids.bin", output_dir / "visible_ids.bin")
    visible_weights.astype("<f4", copy=False).tofile(output_dir / "visible_weights.bin")

    if stats["rawCandidateMissVisible"] and not args.allow_candidate_visible_union:
        raise RuntimeError(
            "Raw AABB candidates miss visible instances; refusing to create a formal dataset. "
            f"missed references={stats['rawCandidateMissVisible']}. "
            "Use --allow-candidate-visible-union only for explicitly exploratory output."
        )

    source_sampler = str(input_meta.get("sourceSampler", "rvc"))
    if source_sampler == "three_color_id":
        schema = "pose-csr-viewcell-back-camera-color-id-fov66-v1"
    else:
        schema = "pose-csr-viewcell-back-camera-pvs-fov66-v1"
    meta = {
        "schema": schema,
        "experiment": args.experiment,
        "sourceSampler": source_sampler,
        "sourceDataset": str(input_dir.as_posix()),
        "runtimeMeta": str(args.runtime_meta.as_posix()),
        "poseCount": int(count),
        "numInstances": int(aabbs.shape[0]),
        "poseStrideBytes": int(DIRECTIONAL_POSE_DTYPE.itemsize),
        "mvpStrideBytes": 64,
        "visibleCount": int(visible_ids.size),
        "candidateCount": int(candidate_offsets[-1]),
        "rawCandidateCount": int(raw_candidate_offsets[-1]),
        "candidateSemantics": (
            "back-camera expanded PVS frustum AABB candidates plus all dense-subpose visible positives"
            if args.allow_candidate_visible_union
            else "raw back-camera expanded PVS frustum AABB candidates; no GT positive union"
        ),
        "rawCandidateSemantics": "AABB candidates computed before any visible-positive union",
        "rawCandidateFile": "raw_candidate_ids.bin",
        "rawCandidateOffsets": "raw_candidate_offsets.bin",
        "cameraSemantics": "camera_world = viewcell_center - normalize(viewcell_forward) * pvs_back_offset; FOV uses pvs_fov_x/y",
        "gtSemantics": "visible_ids are dense subpose visible-id unions per viewcell",
        "visibleWeightSemantics": input_meta.get("visibleWeightSemantics", "rvcServer component_weights, not pixel coverage"),
        "visibleWeightDtype": "float32",
        "modelInputFovYDeg": 66.0,
        "frontendRenderFovYDeg": 60.0,
        "cameraBounds": camera_bounds,
        "splitIds": (
            split_manifest.get("splitIds")
            if split_manifest is not None
            else input_meta.get("splitIds", {"train": 0, "val": 1, "test": 2, "unknown": 255})
        ),
        "categoryIds": input_meta.get("categoryIds", {}),
        "sourceViewcellCount": int(input_meta.get("viewcellCount", count)),
        "sourceSubposeCount": int(input_meta.get("subposeCount", 0)),
        "stats": {
            **stats,
            "avgCandidate": float(stats["totalCandidateRefs"] / max(1, count)),
            "avgVisible": float(visible_ids.size / max(1, count)),
            "pvsFovYRange": [float(np.min(pvs_fov_y)), float(np.max(pvs_fov_y))],
            "pvsFovXRange": [float(np.min(pvs_fov_x)), float(np.max(pvs_fov_x))],
            "pvsBackOffsetRange": [float(np.min(back_offsets)), float(np.max(back_offsets))],
        },
        "files": {
            "poses": "poses.bin",
            "mvp": "mvp.bin",
            "visibleOffsets": "visible_offsets.bin",
            "visibleIds": "visible_ids.bin",
            "visibleWeights": "visible_weights.bin",
            "candidateOffsets": "candidate_offsets.bin",
            "candidateIds": "candidate_ids.bin",
            "rawCandidateOffsets": "raw_candidate_offsets.bin",
            "rawCandidateIds": "raw_candidate_ids.bin",
            "frustumOffsets": "frustum_offsets.bin",
            "frustumIds": "frustum_ids.bin",
        },
    }
    (output_dir / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
