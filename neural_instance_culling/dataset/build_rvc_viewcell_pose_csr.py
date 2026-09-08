#!/usr/bin/env python3
"""Build PoseCSR training data from viewcell subpose raw samples.

Each output row is one NeuralPVS-style viewcell.  The visible set is the union
of all successful subpose visibility responses inside the viewcell.  Raw rows
can come from rvcServer or the Three.js color-id sampler.  Candidate ids are
computed independently from every successful subpose and then unioned; this
matches the from-region view-cell semantics.  Ground-truth positive union is
disabled for formal datasets and is available only as an explicit exploratory
diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


CATEGORY_IDS = {
    "street_gap": 0,
    "near_building": 1,
    "plaza": 2,
    "perimeter": 3,
    "sky": 4,
    "far": 5,
    "unknown": 255,
}
SPLIT_IDS = {"train": 0, "val": 1, "test": 2, "unknown": 255}
MODEL_INPUT_FOV_Y_DEG = 66.0
FRONTEND_RENDER_FOV_Y_DEG = 60.0
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
    parser = argparse.ArgumentParser(description="Build viewcell-level PoseCSR from rvcServer raw JSONL.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--raw-glob", default="*.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--experiment", default="viewcell_pvs_fov66_v1")
    parser.add_argument("--source-sampler", choices=["rvc", "three_color_id"], default="rvc")
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--max-raw-rows", type=int, default=0)
    parser.add_argument("--min-success-subposes", type=int, default=1)
    parser.add_argument(
        "--subpose-groups-per-viewcell",
        type=int,
        default=1,
        help=(
            "Split each sampled view cell into this many local cells using the existing subposes. "
            "The default 1 preserves the original view-cell aggregation."
        ),
    )
    parser.add_argument(
        "--stratified-subpose-split",
        action="store_true",
        help="For derived local cells, assign train/val/test with the stable view-cell hash instead of inheriting the source split.",
    )
    parser.add_argument("--default-fov-y", type=float, default=MODEL_INPUT_FOV_Y_DEG)
    parser.add_argument("--default-aspect", type=float, default=16.0 / 9.0)
    parser.add_argument(
        "--allow-candidate-visible-union",
        action="store_true",
        help="Exploratory only: add visible IDs missing from raw AABB candidates. Formal builds must omit this flag.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path_or_dir: Path, raw_glob: str, max_rows: int = 0) -> list[dict[str, Any]]:
    files = sorted(path_or_dir.glob(raw_glob)) if path_or_dir.is_dir() else [path_or_dir]
    if not files:
        raise RuntimeError(f"No raw JSONL files matched {path_or_dir}/{raw_glob}")
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if max_rows > 0 and len(rows) >= max_rows:
                    return rows
    return rows


def scene_min_max(scene_bounds: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "min" in scene_bounds and "max" in scene_bounds:
        mn = np.asarray(scene_bounds["min"], dtype=np.float32)
        mx = np.asarray(scene_bounds["max"], dtype=np.float32)
        return mn, mx, mx - mn
    center = np.asarray(scene_bounds["center"], dtype=np.float32)
    size = np.asarray(scene_bounds["size"], dtype=np.float32)
    return center - size * 0.5, center + size * 0.5, size


def bounds_min_max(bounds: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "min" in bounds and "max" in bounds:
        return np.asarray(bounds["min"], dtype=np.float32), np.asarray(bounds["max"], dtype=np.float32)
    center = np.asarray(bounds["center"], dtype=np.float32)
    size = np.asarray(bounds["size"], dtype=np.float32)
    return center - size * 0.5, center + size * 0.5


def load_world_aabbs(runtime_meta: dict[str, Any]) -> np.ndarray:
    records = runtime_meta.get("componentRecords", [])
    count = max(int(record["componentGlobalId"]) for record in records) + 1 if records else 0
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


def normalize_points(points: np.ndarray, bounds: dict[str, Any]) -> np.ndarray:
    mn, _mx, size = scene_min_max(bounds)
    return np.clip((points - mn[None, :]) / np.maximum(size[None, :], 1e-6), 0.0, 1.0).astype(np.float32, copy=False)


def stable_split(viewcell_id: int, seed: int = 20260623) -> str:
    state = np.uint32(viewcell_id ^ seed)
    state ^= state >> np.uint32(16)
    state = np.uint32(state * np.uint32(2246822507))
    state ^= state >> np.uint32(13)
    state = np.uint32(state * np.uint32(3266489909))
    state ^= state >> np.uint32(16)
    ratio = float(int(state)) / float(2**32)
    if ratio < 0.8:
        return "train"
    if ratio < 0.9:
        return "val"
    return "test"


def aggregate_viewcells(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    group_count = max(1, int(args.subpose_groups_per_viewcell))

    for index, source_row in enumerate(rows):
        row = dict(source_row)
        source_viewcell_id = int(row.get("viewcell_id", row.get("pose_index", index)))
        if group_count == 1:
            group_index = 0
        elif group_count == 4:
            # The four groups are spatial quadrants in the original camera-aligned
            # cell. This keeps each derived cell local while using existing renders.
            center = np.asarray(row.get("viewcell_center") or row.get("camera_pos") or [0, 0, 0], dtype=np.float32)
            forward = normalize(
                np.asarray(row.get("viewcell_forward") or row.get("camera_forward") or [0, 0, -1], dtype=np.float32),
                np.asarray([0, 0, -1], dtype=np.float32),
            )
            position = np.asarray(row.get("camera_pos") or center, dtype=np.float32)
            _f, right, _up = camera_basis(forward)
            offset = position - center
            right_side = 1 if float(np.dot(offset, right)) >= 0.0 else 0
            forward_side = 1 if float(np.dot(offset, forward)) >= 0.0 else 0
            group_index = right_side + 2 * forward_side
        else:
            group_index = int(row.get("subpose_id", index)) % group_count
        viewcell_id = source_viewcell_id * group_count + group_index
        row["source_viewcell_id"] = source_viewcell_id
        row["subpose_group_id"] = group_index
        row["viewcell_id"] = viewcell_id
        if group_count > 1 and args.stratified_subpose_split:
            row["split"] = stable_split(viewcell_id)
        grouped[viewcell_id].append(row)

    viewcells: list[dict[str, Any]] = []
    for viewcell_id in sorted(grouped):
        subposes = grouped[viewcell_id]
        ok = [row for row in subposes if not row.get("sample_error")]
        if len(ok) < int(args.min_success_subposes):
            continue
        first = ok[0]
        subpose_positions = np.asarray(
            [row.get("camera_pos") or row.get("viewcell_center") or [0, 0, 0] for row in ok],
            dtype=np.float32,
        )
        declared_center = first.get("viewcell_center")
        if declared_center is not None:
            # Keep the canonical plan center independent of the random
            # subpose realization. This is the spatial/query anchor.
            center = np.asarray(declared_center, dtype=np.float32)
        else:
            center = subpose_positions.mean(axis=0) if subpose_positions.size else np.asarray(
                first.get("camera_pos") or [0, 0, 0], dtype=np.float32
            )
        forward = normalize(
            np.asarray(first.get("viewcell_forward") or first.get("camera_forward") or [0, 0, -1], dtype=np.float32),
            np.asarray([0, 0, -1], dtype=np.float32),
        )
        fov_y = float(first.get("pvs_fov_y", first.get("fov_y", args.default_fov_y)))
        if abs(fov_y - MODEL_INPUT_FOV_Y_DEG) > 1e-6:
            raise RuntimeError(f"View-cell samples must use the model FOV of {MODEL_INPUT_FOV_Y_DEG} degrees.")
        aspect = float(first.get("aspect", args.default_aspect))
        tan_y = math.tan(math.radians(fov_y) * 0.5)
        tan_x = tan_y * aspect
        weights_by_id: dict[int, float] = {}
        hits_by_id: dict[int, int] = defaultdict(int)
        for row in ok:
            ids = row.get("visible_component_ids") or []
            weights = row.get("component_weights") or []
            for i, raw_id in enumerate(ids):
                cid = int(raw_id)
                weight = float(weights[i]) if i < len(weights) else 1.0
                weights_by_id[cid] = max(weights_by_id.get(cid, 0.0), weight)
                hits_by_id[cid] += 1
        visible = np.asarray(sorted(weights_by_id), dtype=np.uint32)
        viewcells.append(
            {
                "viewcell_id": int(viewcell_id),
                "source_viewcell_id": int(first.get("source_viewcell_id", viewcell_id)),
                "subpose_group_id": int(first.get("subpose_group_id", 0)),
                "center": center,
                "forward": forward,
                "tan_x": float(tan_x),
                "tan_y": float(tan_y),
                "fov_y": float(fov_y),
                "aspect": float(aspect),
                "split": str(first.get("split") or stable_split(int(viewcell_id))),
                "category": str(first.get("viewcell_category") or first.get("sample_category") or "unknown"),
                "category_id": int(first.get("sample_category_id", CATEGORY_IDS.get(str(first.get("sample_category", "unknown")), 255))),
                "subpose_count": int(len(subposes)),
                "success_subpose_count": int(len(ok)),
                "visible": visible,
                "weights": np.asarray([weights_by_id[int(cid)] for cid in visible], dtype=np.float32),
                "hits": np.asarray([hits_by_id[int(cid)] for cid in visible], dtype=np.uint16),
                "subpose_positions": subpose_positions,
            }
        )
    return viewcells


def main() -> None:
    args = parse_args()
    if abs(float(args.default_fov_y) - MODEL_INPUT_FOV_Y_DEG) > 1e-6:
        raise RuntimeError(f"View-cell datasets must use the model FOV of {MODEL_INPUT_FOV_Y_DEG} degrees.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_meta = read_json(args.runtime_meta)
    rows = iter_jsonl(args.raw_dir, args.raw_glob, args.max_raw_rows)
    if not rows:
        raise RuntimeError("No raw JSONL rows found.")
    aabbs = load_world_aabbs(runtime_meta)
    viewcells = aggregate_viewcells(rows, args)
    if not viewcells:
        raise RuntimeError("No valid viewcells after aggregation.")

    centers = np.asarray([vc["center"] for vc in viewcells], dtype=np.float32)
    scene_min, scene_max, _scene_size = scene_min_max(runtime_meta["sceneBounds"])
    cam_min = np.minimum(scene_min, centers.min(axis=0))
    cam_max = np.maximum(scene_max, centers.max(axis=0))
    pad = np.maximum((cam_max - cam_min) * 0.05, 1.0)
    camera_bounds = {
        "min": (cam_min - pad).astype(float).tolist(),
        "max": (cam_max + pad).astype(float).tolist(),
        "center": ((cam_min + cam_max) * 0.5).astype(float).tolist(),
        "size": ((cam_max - cam_min) + pad * 2.0).astype(float).tolist(),
    }

    poses = np.zeros((len(viewcells),), dtype=DIRECTIONAL_POSE_DTYPE)
    poses["camera_norm"] = normalize_points(centers, camera_bounds)
    poses["camera_world"] = centers
    poses["camera_forward"] = np.asarray([vc["forward"] for vc in viewcells], dtype=np.float32)
    poses["camera_view"] = np.asarray([[vc["tan_x"], vc["tan_y"]] for vc in viewcells], dtype=np.float32)
    poses["split"] = np.asarray([SPLIT_IDS.get(vc["split"], SPLIT_IDS["unknown"]) for vc in viewcells], dtype=np.uint8)
    poses["category"] = np.asarray([vc["category_id"] for vc in viewcells], dtype=np.uint8)
    poses.tofile(output_dir / "poses.bin")

    mvps = np.zeros((len(viewcells), 16), dtype=np.float32)
    visible_offsets = np.zeros((len(viewcells) + 1,), dtype=np.uint64)
    subpose_offsets = np.zeros((len(viewcells) + 1,), dtype=np.uint64)
    candidate_offsets = np.zeros((len(viewcells) + 1,), dtype=np.uint64)
    visible_ids_all: list[int] = []
    visible_weights_all: list[float] = []
    visible_hits_all: list[int] = []
    subpose_positions_all: list[list[float]] = []
    stats = {
        "rawRows": int(len(rows)),
        "viewcellCount": int(len(viewcells)),
        "candidateMissVisible": 0,
        "candidateMissVisibleViewcells": 0,
        "candidateVisibleUnionAdded": 0,
        "visibleSubsetFailures": 0,
        "emptyCandidate": 0,
        "emptyVisible": 0,
        "totalCandidateRefs": 0,
        "maxCandidate": 0,
        "subposeCount": int(sum(vc["subpose_count"] for vc in viewcells)),
        "successSubposeCount": int(sum(vc["success_subpose_count"] for vc in viewcells)),
    }

    candidate_path = output_dir / "candidate_ids.bin"
    with candidate_path.open("wb") as candidate_file:
        for i, vc in enumerate(tqdm(viewcells, desc="pack viewcell csr")):
            mvps[i] = conservative_mvp(vc["center"], vc["forward"], vc["tan_x"], vc["tan_y"])
            visible = vc["visible"]
            weights = vc["weights"]
            hits = vc["hits"]
            visible_ids_all.extend(int(v) for v in visible.tolist())
            visible_weights_all.extend(float(v) for v in weights.tolist())
            visible_hits_all.extend(int(v) for v in hits.tolist())
            visible_offsets[i + 1] = len(visible_ids_all)
            subpose_positions_all.extend(np.asarray(vc["subpose_positions"], dtype=np.float32).tolist())
            subpose_offsets[i + 1] = len(subpose_positions_all)
            if visible.size == 0:
                stats["emptyVisible"] += 1

            candidate_set: set[int] = set()
            for subpose_position in vc.get("subpose_positions", np.asarray([vc["center"]], dtype=np.float32)):
                candidates = frustum_candidate_ids(
                    np.asarray(subpose_position, dtype=np.float32),
                    vc["forward"],
                    vc["tan_x"],
                    vc["tan_y"],
                    aabbs,
                    args.near,
                )
                candidate_set.update(int(v) for v in candidates.tolist())
            missing = np.setdiff1d(visible, np.asarray(sorted(candidate_set), dtype=np.uint32), assume_unique=False)
            if missing.size:
                stats["candidateMissVisible"] += int(missing.size)
                stats["candidateMissVisibleViewcells"] += 1
                if args.allow_candidate_visible_union:
                    candidate_set.update(int(v) for v in missing.tolist())
                    stats["candidateVisibleUnionAdded"] += int(missing.size)
            candidates = np.asarray(sorted(candidate_set), dtype=np.uint32)
            if candidates.size == 0:
                stats["emptyCandidate"] += 1
            if np.setdiff1d(visible, candidates, assume_unique=False).size:
                stats["visibleSubsetFailures"] += 1
            candidate_file.write(candidates.tobytes(order="C"))
            candidate_offsets[i + 1] = candidate_offsets[i] + int(candidates.size)
            stats["totalCandidateRefs"] += int(candidates.size)
            stats["maxCandidate"] = max(int(stats["maxCandidate"]), int(candidates.size))

    mvps.tofile(output_dir / "mvp.bin")
    visible_offsets.tofile(output_dir / "visible_offsets.bin")
    np.asarray([vc["center"] for vc in viewcells], dtype="<f4").tofile(output_dir / "viewcell_centers.bin")
    subpose_offsets.tofile(output_dir / "subpose_offsets.bin")
    np.asarray(subpose_positions_all, dtype="<f4").tofile(output_dir / "subpose_camera_pos.bin")
    np.asarray(visible_ids_all, dtype="<u4").tofile(output_dir / "visible_ids.bin")
    np.asarray(visible_weights_all, dtype="<f4").tofile(output_dir / "visible_weights.bin")
    np.asarray(visible_hits_all, dtype="<u2").tofile(output_dir / "visible_hit_counts.bin")
    candidate_offsets.tofile(output_dir / "candidate_offsets.bin")
    # In formal mode candidate_ids is already the raw AABB result because GT
    # positive union is disabled. Keep an explicit alias so an audit can verify
    # that this claim is backed by a separately named artifact.
    if not args.allow_candidate_visible_union:
        raw_candidate_path = output_dir / "raw_candidate_ids.bin"
        raw_offsets_path = output_dir / "raw_candidate_offsets.bin"
        try:
            os.link(candidate_path, raw_candidate_path)
            os.link(output_dir / "candidate_offsets.bin", raw_offsets_path)
        except OSError:
            shutil.copyfile(candidate_path, raw_candidate_path)
            shutil.copyfile(output_dir / "candidate_offsets.bin", raw_offsets_path)

    category_counts: dict[str, int] = {}
    for vc in viewcells:
        category_counts[vc["category"]] = category_counts.get(vc["category"], 0) + 1
    split_counts = {
        name: int(np.count_nonzero(poses["split"] == sid))
        for name, sid in SPLIT_IDS.items()
        if name != "unknown"
    }
    if args.source_sampler == "three_color_id":
        schema = "viewcell-csr-color-id-fov66-v2"
        gt_semantics = "visible_ids are unioned Three.js color-id visible component ids over all same-direction subposes in the viewcell"
        weight_semantics = "Three.js color-id screen coverage in parts per million, max-pooled over subposes"
    else:
        schema = "viewcell-csr-rvc-fov66-v2"
        gt_semantics = "visible_ids are unioned rvcServer visible component ids over all successful same-direction subposes in the viewcell"
        weight_semantics = "rvcServer component_weights max-pooled over subposes"
    meta = {
        "schema": schema,
        "experiment": args.experiment,
        "sourceSampler": args.source_sampler,
        "runtimeMeta": args.runtime_meta.as_posix(),
        "viewcellCount": int(len(viewcells)),
        "poseCount": int(len(viewcells)),
        "numInstances": int(aabbs.shape[0]),
        "poseStrideBytes": int(DIRECTIONAL_POSE_DTYPE.itemsize),
        "mvpStrideBytes": 64,
        "visibleCount": int(len(visible_ids_all)),
        "candidateCount": int(candidate_offsets[-1]),
        "candidateSemantics": (
            "Union of full AABB candidates computed independently for every successful subpose; no GT positive union"
            if not args.allow_candidate_visible_union
            else "Union of full AABB candidates over all sampled subpose positions plus unioned visible positives (exploratory only)"
        ),
        "candidateVisibleUnionAllowed": bool(args.allow_candidate_visible_union),
        "rawCandidateFile": "raw_candidate_ids.bin" if not args.allow_candidate_visible_union else None,
        "rawCandidateOffsets": "raw_candidate_offsets.bin" if not args.allow_candidate_visible_union else None,
        "rawCandidateSemantics": "AABB candidate union computed before any visible-positive union",
        "cameraSemantics": "camera_world is the canonical viewcell plan center; candidates are the union over dense subpose positions",
        "viewcellGeometrySemantics": "viewcell_centers are canonical plan centers; subpose_camera_pos stores every successful dense subpose position",
        "gtSemantics": gt_semantics,
        "visibleWeightSemantics": weight_semantics,
        "visibleWeightDtype": "float32",
        "modelInputFovYDeg": MODEL_INPUT_FOV_Y_DEG,
        "frontendRenderFovYDeg": FRONTEND_RENDER_FOV_Y_DEG,
        "cameraBounds": camera_bounds,
        "splitIds": SPLIT_IDS,
        "categoryIds": CATEGORY_IDS,
        "categoryCounts": category_counts,
        "splitCounts": split_counts,
        "stats": {
            **stats,
            "avgCandidate": float(candidate_offsets[-1] / max(1, len(viewcells))),
            "avgVisible": float(len(visible_ids_all) / max(1, len(viewcells))),
            "avgSubposesPerViewcell": float(stats["subposeCount"] / max(1, len(viewcells))),
            "avgSuccessfulSubposesPerViewcell": float(stats["successSubposeCount"] / max(1, len(viewcells))),
            "subposeGroupsPerSourceViewcell": int(args.subpose_groups_per_viewcell),
        },
        "files": {
            "poses": "poses.bin",
            "mvp": "mvp.bin",
            "viewcellCenters": "viewcell_centers.bin",
            "subposeOffsets": "subpose_offsets.bin",
            "subposeCameraPos": "subpose_camera_pos.bin",
            "visibleOffsets": "visible_offsets.bin",
            "visibleIds": "visible_ids.bin",
            "visibleWeights": "visible_weights.bin",
            "visibleHitCounts": "visible_hit_counts.bin",
            "candidateOffsets": "candidate_offsets.bin",
            "candidateIds": "candidate_ids.bin",
            "rawCandidateOffsets": "raw_candidate_offsets.bin" if not args.allow_candidate_visible_union else None,
            "rawCandidateIds": "raw_candidate_ids.bin" if not args.allow_candidate_visible_union else None,
        },
    }
    (output_dir / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if stats["candidateMissVisible"] and not args.allow_candidate_visible_union:
        raise RuntimeError(
            "Formal raw-subpose AABB candidates miss visible instances; refusing to declare the dataset usable. "
            f"missing references={stats['candidateMissVisible']} across {stats['candidateMissVisibleViewcells']} viewcells. "
            "Use --allow-candidate-visible-union only for exploratory diagnostics."
        )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
