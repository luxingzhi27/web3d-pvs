#!/usr/bin/env python3
"""Measure nested view-cell GT-union convergence from real subpose samples."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


POSE_DTYPE = np.dtype(
    {
        "names": ["camera_norm", "camera_world", "camera_forward", "camera_view",
                  "split", "category", "reserved0", "reserved1"],
        "formats": [("<f4", (3,)), ("<f4", (3,)), ("<f4", (3,)), ("<f4", (2,)),
                    "u1", "u1", "<u2", "<u4"],
        "offsets": [0, 12, 24, 36, 44, 45, 46, 48],
        "itemsize": 64,
    }
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def farthest_point_order(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if not points.size:
        return np.zeros(0, dtype=np.int64)
    first = int(np.argmin(np.linalg.norm(points - center[None, :], axis=1)))
    selected = [first]
    remaining = np.ones(points.shape[0], dtype=bool)
    remaining[first] = False
    minimum_distance = np.linalg.norm(points - points[first], axis=1)
    while np.any(remaining):
        score = np.where(remaining, minimum_distance, -1.0)
        index = int(np.argmax(score))
        selected.append(index)
        remaining[index] = False
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(points - points[index], axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def nested_union(
    ordered_pose_ids: np.ndarray,
    visibility: dict[int, tuple[np.ndarray, np.ndarray]],
    count: int,
) -> tuple[set[int], dict[int, float]]:
    ids: set[int] = set()
    weights: dict[int, float] = {}
    for pose_id in ordered_pose_ids[:count].tolist():
        pose_ids, pose_weights = visibility[int(pose_id)]
        for component_id, weight in zip(pose_ids.tolist(), pose_weights.tolist()):
            component_id = int(component_id)
            ids.add(component_id)
            weights[component_id] = max(weights.get(component_id, 0.0), float(weight))
    return ids, weights


def representative_rows(
    validation_rows: np.ndarray,
    categories: np.ndarray,
    candidate_counts: np.ndarray,
    gt_counts: np.ndarray,
    limit: int,
) -> np.ndarray:
    rows = np.asarray(validation_rows, dtype=np.int64)
    if rows.size <= limit:
        return rows
    candidate_rank = np.argsort(np.argsort(candidate_counts[rows], kind="stable"), kind="stable")
    gt_rank = np.argsort(np.argsort(gt_counts[rows], kind="stable"), kind="stable")
    candidate_bin = np.minimum(4, candidate_rank * 5 // rows.size)
    gt_bin = np.minimum(4, gt_rank * 5 // rows.size)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, row in enumerate(rows.tolist()):
        key = (int(categories[row]), int(candidate_bin[index]), int(gt_bin[index]))
        groups.setdefault(key, []).append(row)
    selected: list[int] = []
    group_rows = [groups[key] for key in sorted(groups)]
    depth = 0
    while len(selected) < limit:
        added = False
        for group in group_rows:
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def parse_counts(value: str) -> list[int]:
    counts = sorted({int(item) for item in value.split(",") if item.strip()})
    if not counts or counts[0] <= 0:
        raise ValueError("sample counts must be positive")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--viewcell-dataset", type=Path, required=True)
    parser.add_argument("--pose-csr", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--sample-counts", default="1,2,4,8,16,32,64,128")
    args = parser.parse_args()

    source = args.viewcell_dataset.resolve()
    pose_root = args.pose_csr.resolve()
    source_meta = json.loads((source / "dataset_meta.json").read_text(encoding="utf-8"))
    pose_meta = json.loads((pose_root / "dataset_meta.json").read_text(encoding="utf-8"))
    viewcell_count = int(source_meta.get("viewcellCount", source_meta.get("poseCount", 0)))
    poses = np.fromfile(pose_root / "poses.bin", dtype=POSE_DTYPE)
    if poses.size != viewcell_count:
        raise ValueError("view-cell source and Pose CSR row counts differ")
    test_split_id = int((pose_meta.get("splitIds") or {})["validation"])
    validation = np.flatnonzero(poses["split"] == test_split_id)
    categories = np.fromfile(source / "viewcell_category_ids.bin", dtype="u1")
    centers = np.fromfile(source / "viewcell_centers.bin", dtype="<f4").reshape(-1, 3)
    subpose_offsets = np.fromfile(source / "subpose_offsets.bin", dtype="<u8")
    subpose_ids = np.fromfile(source / "subpose_pose_indices.bin", dtype="<u4")
    subpose_positions = np.fromfile(source / "subpose_camera_pos.bin", dtype="<f4").reshape(-1, 3)
    candidate_offsets = np.fromfile(pose_root / "candidate_offsets.bin", dtype="<u8")
    visible_offsets = np.fromfile(pose_root / "visible_offsets.bin", dtype="<u8")
    candidate_counts = np.diff(candidate_offsets)
    gt_counts = np.diff(visible_offsets)
    chosen = representative_rows(validation, categories, candidate_counts, gt_counts, args.cells)

    required_pose_ids: set[int] = set()
    for row in chosen.tolist():
        start, end = int(subpose_offsets[row]), int(subpose_offsets[row + 1])
        required_pose_ids.update(int(value) for value in subpose_ids[start:end].tolist())
    visibility: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    remaining_pose_ids = set(required_pose_ids)
    for record in iter_jsonl(args.raw_dir.resolve()):
        pose_id = record.get("pose_index")
        if pose_id is None or int(pose_id) not in remaining_pose_ids:
            continue
        ids = np.asarray(record.get("visible_component_ids") or [], dtype=np.uint32)
        raw_weights = record.get("component_weights") or []
        weights = np.ones(ids.size, dtype=np.float32)
        if raw_weights:
            weights[: min(ids.size, len(raw_weights))] = np.asarray(raw_weights[: ids.size], dtype=np.float32)
        visibility[int(pose_id)] = (ids, weights)
        remaining_pose_ids.remove(int(pose_id))
        if not remaining_pose_ids:
            break
    missing = sorted(required_pose_ids - set(visibility))
    if missing:
        raise ValueError(f"raw visibility is missing {len(missing)} selected subposes")

    requested_counts = parse_counts(args.sample_counts)
    rows: list[dict[str, Any]] = []
    for row in chosen.tolist():
        start, end = int(subpose_offsets[row]), int(subpose_offsets[row + 1])
        local_order = farthest_point_order(subpose_positions[start:end], centers[row])
        ordered_pose_ids = subpose_ids[start:end][local_order]
        reference_ids, reference_weights = nested_union(
            ordered_pose_ids, visibility, ordered_pose_ids.size
        )
        reference_mass = sum(reference_weights.values())
        unions: dict[int, set[int]] = {}
        for count in requested_counts:
            if count > ordered_pose_ids.size:
                continue
            union, _weights = nested_union(ordered_pose_ids, visibility, count)
            unions[count] = union
            missed = reference_ids - union
            rows.append({
                "scene": args.scene,
                "viewcellRow": row,
                "availableSubposes": int(ordered_pose_ids.size),
                "sampleCount": count,
                "referenceVisibleCount": len(reference_ids),
                "visibleCoverage": (
                    len(union) / len(reference_ids) if reference_ids else 1.0
                ),
                "weightedCoverage": 1.0 - sum(reference_weights[item] for item in missed)
                / max(1e-12, reference_mass),
                "newAtDoubleFraction": None,
            })
        for item in rows:
            if item["viewcellRow"] != row:
                continue
            count = int(item["sampleCount"])
            if count * 2 in unions:
                doubled = unions[count * 2]
                item["newAtDoubleFraction"] = len(doubled - unions[count]) / max(1, len(doubled))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "per_viewcell.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for count in requested_counts:
        subset = [item for item in rows if item["sampleCount"] == count]
        if not subset:
            continue
        summary_rows.append({
            "scene": args.scene,
            "sampleCount": count,
            "viewcellCount": len(subset),
            "visibleCoverageMean": float(np.mean([item["visibleCoverage"] for item in subset])),
            "visibleCoverageP05": float(np.quantile([item["visibleCoverage"] for item in subset], 0.05)),
            "weightedCoverageMean": float(np.mean([item["weightedCoverage"] for item in subset])),
            "weightedCoverageP05": float(np.quantile([item["weightedCoverage"] for item in subset], 0.05)),
            "newAtDoubleFractionMean": (
                float(np.mean([item["newAtDoubleFraction"] for item in subset
                               if item["newAtDoubleFraction"] is not None]))
                if any(item["newAtDoubleFraction"] is not None for item in subset) else None
            ),
        })
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {
        "schema": "pvs-viewcell-gt-convergence-v1",
        "scene": args.scene,
        "split": "validation",
        "selectedViewcellCount": int(chosen.size),
        "selection": "category/candidate-count/GT-count deterministic strata",
        "nestedOrder": "center-nearest then farthest-point spatial coverage",
        "requestedSampleCounts": requested_counts,
        "availableSubposeRange": [
            int(min(item["availableSubposes"] for item in rows)),
            int(max(item["availableSubposes"] for item in rows)),
        ],
        "reference": "union of every currently sampled real subpose in each selected view-cell",
        "summary": summary_rows,
        "testRead": False,
    }
    (output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
