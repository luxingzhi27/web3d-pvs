#!/usr/bin/env python3
"""Build a deterministic spatial train/validation/calibration/test manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
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

SPLIT_IDS = {"train": 0, "validation": 1, "calibration": 2, "test": 3, "guard": 254, "unknown": 255}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ratios(value: str) -> np.ndarray:
    ratios = np.asarray([float(part.strip()) for part in value.split(",") if part.strip()], dtype=np.float64)
    if ratios.size != 4 or np.any(ratios <= 0.0) or not np.isclose(float(ratios.sum()), 1.0):
        raise ValueError("--split-ratios must contain four positive values summing to 1")
    return ratios


def load_viewcell_geometry(input_dir: Path, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load canonical centers and optional dense subpose support positions."""
    pose_count = int(poses.size)
    centers_path = input_dir / "viewcell_centers.bin"
    offsets_path = input_dir / "subpose_offsets.bin"
    positions_path = input_dir / "subpose_camera_pos.bin"
    if centers_path.exists() and offsets_path.exists() and positions_path.exists():
        centers = np.fromfile(centers_path, dtype="<f4").reshape(-1, 3)
        offsets = np.fromfile(offsets_path, dtype="<u8")
        positions = np.fromfile(positions_path, dtype="<f4").reshape(-1, 3)
        if centers.shape[0] != pose_count or offsets.size != pose_count + 1:
            raise ValueError("view-cell geometry arrays do not match pose count")
        if int(offsets[-1]) != positions.shape[0] or np.any(np.diff(offsets) <= 0):
            raise ValueError("subpose offsets are invalid or contain empty view-cells")
        pose_ids = np.repeat(np.arange(pose_count, dtype=np.int64), np.diff(offsets).astype(np.int64))
        return centers, positions, pose_ids, "canonical centers plus dense subpose support positions"
    centers = np.asarray(poses["camera_world"], dtype=np.float32).copy()
    positions = centers.copy()
    pose_ids = np.arange(pose_count, dtype=np.int64)
    return centers, positions, pose_ids, "pose camera_world fallback; dense subpose geometry unavailable"


def group_pose_indices(anchors_world: np.ndarray, mode: str, quantization: float) -> tuple[np.ndarray, np.ndarray]:
    if mode == "viewcell":
        # In this mode each canonical anchor is already one independent group.
        # Use the passed array rather than a module-global pose table so the
        # helper remains deterministic and usable from tests/callers.
        group_ids = np.arange(anchors_world.shape[0], dtype=np.int64)
    elif mode == "xz":
        if quantization <= 0.0:
            raise ValueError("anchor quantization must be positive")
        xz = np.round(np.asarray(anchors_world[:, [0, 2]], dtype=np.float64) / quantization).astype(np.int64)
        _, group_ids = np.unique(xz, axis=0, return_inverse=True)
        group_ids = group_ids.astype(np.int64, copy=False)
    else:
        raise ValueError(f"unsupported group mode: {mode}")
    return group_ids, np.unique(group_ids)


def assign_blocks(
    anchors: np.ndarray,
    block_size: float,
    ratios: np.ndarray,
    seed: int,
    assignment_mode: str,
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    minimum = anchors.min(axis=0)
    block_coords = np.floor((anchors - minimum[None, :]) / float(block_size)).astype(np.int64)
    unique_blocks, inverse = np.unique(block_coords, axis=0, return_inverse=True)
    if assignment_mode == "random_blocks":
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(unique_blocks.shape[0])
    elif assignment_mode == "spatial_stripes":
        block_anchor = np.zeros((unique_blocks.shape[0], 2), dtype=np.float64)
        for block_index in range(unique_blocks.shape[0]):
            block_anchor[block_index] = anchors[inverse == block_index].mean(axis=0)
        # Lexicographic ordering produces contiguous X stripes with a stable Z
        # tie-breaker.  The four sets are therefore separated by physical
        # boundaries rather than interleaved random blocks.
        order = np.lexsort((block_anchor[:, 1], block_anchor[:, 0]))
    else:
        raise ValueError(f"unsupported assignment mode: {assignment_mode}")
    block_labels = np.full((unique_blocks.shape[0],), SPLIT_IDS["train"], dtype=np.uint8)
    split_order = [SPLIT_IDS["train"], SPLIT_IDS["validation"], SPLIT_IDS["calibration"], SPLIT_IDS["test"]]
    if assignment_mode == "spatial_stripes":
        block_weights = np.bincount(inverse, minlength=unique_blocks.shape[0]).astype(np.float64)
        total_weight = float(block_weights.sum())
        cumulative = 0.0
        split_index = 0
        boundary = float(ratios[0]) * total_weight
        for block_index in order.tolist():
            if split_index < len(split_order) - 1 and cumulative >= boundary:
                split_index += 1
                boundary = float(ratios[: split_index + 1].sum()) * total_weight
            block_labels[int(block_index)] = np.uint8(split_order[split_index])
            cumulative += float(block_weights[int(block_index)])
    else:
        counts = np.floor(ratios * unique_blocks.shape[0]).astype(np.int64)
        counts[0] += unique_blocks.shape[0] - int(counts.sum())
        cursor = 0
        for split_id, count in zip(split_order, counts.tolist()):
            selected = order[cursor:cursor + int(count)]
            block_labels[selected] = np.uint8(split_id)
            cursor += int(count)
    pose_labels = block_labels[inverse]
    block_map = {
        (int(coord[0]), int(coord[1])): int(label)
        for coord, label in zip(unique_blocks.tolist(), block_labels.tolist())
    }
    return pose_labels, block_map


def apply_guard_band(
    labels: np.ndarray,
    anchors: np.ndarray,
    guard_distance: float,
    support_points: np.ndarray | None = None,
    support_group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | str]]:
    result = labels.copy()
    if guard_distance <= 0.0:
        return result, {"guardBasis": "disabled", "crossSplitSupportPairs": 0, "guardGroupCount": 0}
    if support_points is not None and support_group_ids is not None:
        points = np.asarray(support_points, dtype=np.float64)[:, [0, 2]]
        group_ids = np.asarray(support_group_ids, dtype=np.int64)
        if points.shape[0] != group_ids.size:
            raise ValueError("support points and support group ids have different lengths")
        cell_size = float(guard_distance)
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, point in enumerate(points):
            cell = (int(np.floor(point[0] / cell_size)), int(np.floor(point[1] / cell_size)))
            grid[cell].append(index)
        radius_sq = float(guard_distance) ** 2
        guard_groups: set[int] = set()
        cross_pairs = 0
        for index, point in enumerate(points):
            group = int(group_ids[index])
            label = int(labels[group])
            if label == SPLIT_IDS["guard"]:
                continue
            cell = (int(np.floor(point[0] / cell_size)), int(np.floor(point[1] / cell_size)))
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in grid.get((cell[0] + dx, cell[1] + dz), []):
                        other_group = int(group_ids[other])
                        if other_group == group or labels[other_group] == label:
                            continue
                        delta = point - points[other]
                        if float(delta @ delta) < radius_sq:
                            guard_groups.add(group)
                            guard_groups.add(other_group)
                            cross_pairs += 1
        if guard_groups:
            result[np.asarray(sorted(guard_groups), dtype=np.int64)] = np.uint8(SPLIT_IDS["guard"])
        return result, {
            "guardBasis": "dense subpose XZ support positions",
            "crossSplitSupportPairs": int(cross_pairs),
            "guardGroupCount": int(len(guard_groups)),
        }
    cell_size = float(guard_distance)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(anchors):
        cell = (int(np.floor(point[0] / cell_size)), int(np.floor(point[1] / cell_size)))
        grid[cell].append(index)
    radius_sq = float(guard_distance) ** 2
    for index, point in enumerate(anchors):
        cell = (int(np.floor(point[0] / cell_size)), int(np.floor(point[1] / cell_size)))
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for other in grid.get((cell[0] + dx, cell[1] + dz), []):
                    if other <= index or result[other] == result[index]:
                        continue
                    delta = point - anchors[other]
                    if float(delta @ delta) < radius_sq:
                        result[index] = np.uint8(SPLIT_IDS["guard"])
                        result[other] = np.uint8(SPLIT_IDS["guard"])
    return result, {"guardBasis": "canonical XZ anchors", "crossSplitSupportPairs": 0, "guardGroupCount": int(np.count_nonzero(result == SPLIT_IDS["guard"]))}


def nearest_cross_split_distances(
    labels: np.ndarray,
    anchors: np.ndarray,
    support_points: np.ndarray | None = None,
    support_group_ids: np.ndarray | None = None,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return {"available": None}
    if support_points is not None and support_group_ids is not None:
        source_points = np.asarray(support_points, dtype=np.float64)[:, [0, 2]]
        source_labels = labels[np.asarray(support_group_ids, dtype=np.int64)]
    else:
        source_points = np.asarray(anchors, dtype=np.float64)
        source_labels = labels
    for left_name, left_id in [("train", 0), ("validation", 1), ("calibration", 2), ("test", 3)]:
        left = source_points[source_labels == left_id]
        if left.size == 0:
            result[f"{left_name}ToOtherMin"] = None
            continue
        minimum = float("inf")
        for right_name, right_id in [("train", 0), ("validation", 1), ("calibration", 2), ("test", 3)]:
            if right_name == left_name:
                continue
            right = source_points[source_labels == right_id]
            if right.size == 0:
                continue
            distance = cKDTree(right).query(left, k=1)[0]
            minimum = min(minimum, float(np.min(distance)))
        result[f"{left_name}ToOtherMin"] = None if not np.isfinite(minimum) else minimum
    return result


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((input_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    pose_count = int(meta["poseCount"])
    poses = np.memmap(input_dir / "poses.bin", dtype=POSE_DTYPE, mode="r", shape=(pose_count,))
    centers_world, support_world, support_pose_ids, geometry_semantics = load_viewcell_geometry(input_dir, poses)
    group_ids, unique_groups = group_pose_indices(centers_world, args.group_mode, args.anchor_quantization)
    group_anchors = np.zeros((unique_groups.size, 2), dtype=np.float64)
    for group_id in unique_groups.tolist():
        members = np.flatnonzero(group_ids == int(group_id))
        positions = np.asarray(centers_world[members][:, [0, 2]], dtype=np.float64)
        group_anchors[int(group_id)] = positions.mean(axis=0)
    ratios = parse_ratios(args.split_ratios)
    group_labels, block_map = assign_blocks(
        group_anchors,
        args.block_size,
        ratios,
        args.seed,
        args.assignment_mode,
    )
    support_group_ids = group_ids[support_pose_ids]
    group_labels, guard_stats = apply_guard_band(
        group_labels,
        group_anchors,
        args.guard_distance,
        support_points=support_world,
        support_group_ids=support_group_ids,
    )
    pose_labels = group_labels[group_ids]
    pose_split_ids = np.asarray(pose_labels, dtype=np.uint8)
    pose_split_ids.tofile(output_dir / "pose_split_ids.bin")
    np.asarray(group_ids, dtype=np.uint32).tofile(output_dir / "pose_group_ids.bin")

    counts = {
        name: int(np.count_nonzero(pose_split_ids == split_id))
        for name, split_id in SPLIT_IDS.items()
        if name in {"train", "validation", "calibration", "test", "guard"}
    }
    group_counts = {
        name: int(np.count_nonzero(group_labels == split_id))
        for name, split_id in SPLIT_IDS.items()
        if name in {"train", "validation", "calibration", "test", "guard"}
    }
    category_counts: dict[str, dict[str, int]] = {}
    for name, split_id in SPLIT_IDS.items():
        if name not in {"train", "validation", "calibration", "test", "guard"}:
            continue
        category_counts[name] = {
            str(category): int(np.count_nonzero(poses["category"][pose_split_ids == split_id] == category))
            for category in np.unique(poses["category"][pose_split_ids == split_id]).tolist()
        }
    group_categories = np.zeros((unique_groups.size,), dtype=np.uint8)
    for group_id in unique_groups.tolist():
        members = np.flatnonzero(group_ids == int(group_id))
        values, value_counts = np.unique(poses["category"][members], return_counts=True)
        group_categories[int(group_id)] = np.uint8(values[int(np.argmax(value_counts))])
    category_group_counts: dict[str, dict[str, int]] = {}
    for name, split_id in SPLIT_IDS.items():
        if name not in {"train", "validation", "calibration", "test", "guard"}:
            continue
        selected = group_categories[group_labels == split_id]
        category_group_counts[name] = {
            str(category): int(np.count_nonzero(selected == category))
            for category in np.unique(selected).tolist()
        }
    block_coords = np.floor((group_anchors - group_anchors.min(axis=0)[None, :]) / float(args.block_size)).astype(np.int64)
    post_guard_block_counts: dict[str, int] = defaultdict(int)
    for coord, label in zip(block_coords.tolist(), group_labels.tolist()):
        post_guard_block_counts[str(int(label))] += 1
    manifest = {
        "schema": "spatial-viewcell-four-way-split-v2",
        "sourceDataset": str(input_dir),
        "sourceDatasetMetaSha256": sha256_file(input_dir / "dataset_meta.json"),
        "sourcePosesSha256": sha256_file(input_dir / "poses.bin"),
        "poseCount": pose_count,
        "groupMode": args.group_mode,
        "anchorQuantizationM": float(args.anchor_quantization),
        "groupCount": int(unique_groups.size),
        "blockSizeM": float(args.block_size),
        "guardDistanceM": float(args.guard_distance),
        "splitRatios": ratios.tolist(),
        "selectionSeed": int(args.seed),
        "assignmentMode": args.assignment_mode,
        "splitIds": SPLIT_IDS,
        "poseCounts": counts,
        "groupCounts": group_counts,
        "categoryCounts": category_counts,
        "categoryGroupCounts": category_group_counts,
        "blockCounts": {
            str(split_id): int(sum(1 for label in block_map.values() if label == split_id))
            for split_id in [0, 1, 2, 3]
        },
        "postGuardGroupCounts": {
            name: int(np.count_nonzero(group_labels == split_id))
            for name, split_id in SPLIT_IDS.items()
            if name in {"train", "validation", "calibration", "test", "guard"}
        },
        "postGuardBlockCounts": dict(post_guard_block_counts),
        "guardStats": guard_stats,
        "nearestCrossSplitDistancesM": nearest_cross_split_distances(
            group_labels,
            group_anchors,
            support_points=support_world,
            support_group_ids=support_group_ids,
        ),
        "geometrySemantics": geometry_semantics,
        "supportPointCount": int(support_world.shape[0]),
        "files": {"poseSplitIds": "pose_split_ids.bin", "poseGroupIds": "pose_group_ids.bin"},
        "semantics": "all poses sharing a physical XZ anchor are assigned together; dense subpose support positions are used for guard exclusion and cross-split distance",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-mode", choices=["viewcell", "xz"], default="xz")
    parser.add_argument("--anchor-quantization", type=float, default=0.01)
    parser.add_argument("--block-size", type=float, default=64.0)
    parser.add_argument("--guard-distance", type=float, default=16.0)
    parser.add_argument("--assignment-mode", choices=["random_blocks", "spatial_stripes"], default="spatial_stripes")
    parser.add_argument("--split-ratios", default="0.7,0.1,0.1,0.1")
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
