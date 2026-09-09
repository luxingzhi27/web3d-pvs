#!/usr/bin/env python3
"""Build a stratified, nested subpose plan for view-cell GT convergence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _normalize(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= 1e-8:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    return value.astype(np.float64, copy=False) / length


def _camera_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalize(forward)
    up_seed = np.asarray([0.0, 0.0, 1.0] if abs(float(forward[1])) > 0.98 else [0.0, 1.0, 0.0])
    right = _normalize(np.cross(forward, up_seed))
    up = _normalize(np.cross(right, forward))
    return forward, right, up


def nested_offsets(
    count: int,
    *,
    shape: str,
    radius: float,
    half_up: float,
    forward: np.ndarray,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("subpose count must be positive")
    result = np.zeros((count, 3), dtype=np.float64)
    _forward, right, up = _camera_basis(forward)
    if shape == "horizontal_disk":
        for index in range(1, count):
            radial = radius * math.sqrt(_van_der_corput(index, 2))
            angle = 2.0 * math.pi * _van_der_corput(index, 3)
            result[index, 0] = radial * math.cos(angle)
            result[index, 2] = radial * math.sin(angle)
        return result
    if shape != "camera_aligned_box":
        raise ValueError(f"unsupported view-cell shape: {shape}")
    for index in range(1, count):
        right_scale = (2.0 * _van_der_corput(index, 2) - 1.0) * radius
        forward_scale = (2.0 * _van_der_corput(index, 3) - 1.0) * radius
        up_scale = (2.0 * _van_der_corput(index, 5) - 1.0) * half_up
        result[index] = right * right_scale + _forward * forward_scale + up * up_scale
    return result


def _quantile_bins(values: np.ndarray, bin_count: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.int16)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bin_count + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right").astype(np.int16)


def select_stratified_poses(
    pose_indices: np.ndarray,
    categories: np.ndarray,
    forwards: np.ndarray,
    candidate_counts: np.ndarray,
    gt_counts: np.ndarray,
    weight_sums: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("selection count must be positive")
    if pose_indices.size <= count:
        return np.sort(pose_indices.astype(np.int64, copy=False))
    selected_forwards = forwards[pose_indices]
    yaw = np.arctan2(-selected_forwards[:, 0], -selected_forwards[:, 2])
    yaw_bin = np.floor((yaw + math.pi) / (2.0 * math.pi) * 8.0).astype(np.int16) % 8
    category_values = categories[pose_indices].astype(np.int16, copy=False)
    feature_columns = (
        _quantile_bins(np.log1p(candidate_counts[pose_indices]), 4),
        _quantile_bins(np.log1p(gt_counts[pose_indices]), 4),
        _quantile_bins(np.log1p(weight_sums[pose_indices]), 4),
        yaw_bin,
    )
    strata_by_category: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for local_index, pose_index in enumerate(pose_indices.tolist()):
        key = tuple(int(column[local_index]) for column in feature_columns)
        category = int(category_values[local_index])
        strata_by_category.setdefault(category, {}).setdefault(key, []).append(int(pose_index))
    rng = np.random.default_rng(seed)
    category_sizes = {
        category: sum(len(members) for members in strata.values())
        for category, strata in strata_by_category.items()
    }
    exact_quotas = {category: count * size / pose_indices.size for category, size in category_sizes.items()}
    quotas = {category: min(size, int(math.floor(exact_quotas[category]))) for category, size in category_sizes.items()}
    if count >= len(quotas):
        for category in quotas:
            quotas[category] = max(1, quotas[category])
    while sum(quotas.values()) < count:
        available = [category for category in quotas if quotas[category] < category_sizes[category]]
        if not available:
            break
        category = max(available, key=lambda value: (exact_quotas[value] - quotas[value], category_sizes[value], -value))
        quotas[category] += 1
    while sum(quotas.values()) > count:
        removable = [category for category in quotas if quotas[category] > 1]
        category = min(removable, key=lambda value: (exact_quotas[value] - quotas[value], category_sizes[value], -value))
        quotas[category] -= 1

    chosen: list[int] = []
    for category in sorted(strata_by_category):
        strata = strata_by_category[category]
        for members in strata.values():
            rng.shuffle(members)
        keys = sorted(strata)
        category_chosen = 0
        while category_chosen < quotas[category]:
            added = False
            for key in keys:
                members = strata[key]
                if not members:
                    continue
                chosen.append(members.pop())
                category_chosen += 1
                added = True
                if category_chosen == quotas[category]:
                    break
            if not added:
                break
    return np.asarray(sorted(chosen), dtype=np.int64)


def build_plan(
    dataset_dir: Path,
    representative_plan: Path,
    output: Path,
    *,
    scene: str,
    viewcell_shape: str,
    radius: float,
    half_up: float,
    viewcell_count: int,
    subpose_count: int,
    seed: int,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    meta = _read_json(dataset_dir / "dataset_meta.json")
    poses = np.fromfile(dataset_dir / "poses.bin", dtype=POSE_DTYPE)
    representatives = _read_jsonl(representative_plan.resolve())
    if len(representatives) != poses.size:
        raise ValueError("representative plan row count does not match the pose CSR")
    candidate_offsets = np.fromfile(dataset_dir / "candidate_offsets.bin", dtype="<u8")
    visible_offsets = np.fromfile(dataset_dir / "visible_offsets.bin", dtype="<u8")
    visible_weights = np.fromfile(dataset_dir / "visible_weights.bin", dtype="<f4")
    if candidate_offsets.size != poses.size + 1 or visible_offsets.size != poses.size + 1:
        raise ValueError("candidate/visible offsets do not match the pose count")
    candidate_counts = np.diff(candidate_offsets).astype(np.int64, copy=False)
    gt_counts = np.diff(visible_offsets).astype(np.int64, copy=False)
    weight_prefix = np.empty(visible_weights.size + 1, dtype=np.float64)
    weight_prefix[0] = 0.0
    np.cumsum(visible_weights, dtype=np.float64, out=weight_prefix[1:])
    weight_sums = weight_prefix[visible_offsets[1:]] - weight_prefix[visible_offsets[:-1]]
    validation_id = int((meta.get("splitIds") or {}).get("validation", 1))
    validation = np.flatnonzero(poses["split"] == validation_id)
    selected = select_stratified_poses(
        validation,
        poses["category"],
        poses["camera_forward"],
        candidate_counts,
        gt_counts,
        weight_sums,
        count=viewcell_count,
        seed=seed,
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pose_index = 0
    with output.open("w", encoding="utf-8") as handle:
        for output_viewcell_id, source_pose_index in enumerate(selected.tolist()):
            representative = dict(representatives[source_pose_index])
            center = np.asarray(
                representative.get("viewcell_center") or representative.get("camera_pos"),
                dtype=np.float64,
            )
            forward = _normalize(np.asarray(representative.get("viewcell_forward") or representative.get("camera_forward")))
            offsets = nested_offsets(
                subpose_count,
                shape=viewcell_shape,
                radius=radius,
                half_up=half_up,
                forward=forward,
            )
            for subpose_id, offset in enumerate(offsets):
                row = dict(representative)
                row.update(
                    {
                        "pose_index": pose_index,
                        "viewcell_id": output_viewcell_id,
                        "source_pose_index": source_pose_index,
                        "subpose_id": subpose_id,
                        "split": "validation",
                        "viewcell_center": center.tolist(),
                        "viewcell_forward": forward.tolist(),
                        "viewcell_shape": viewcell_shape,
                        "viewcell_half_extent": [radius, radius, 0.0 if viewcell_shape == "horizontal_disk" else half_up],
                        "viewcell_radius": radius,
                        "camera_pos": (center + offset).tolist(),
                        "camera_forward": forward.tolist(),
                        "render_fov_y": 66.0,
                        "fov_y": 66.0,
                    }
                )
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                pose_index += 1
    summary = {
        "schema": "pvs-gt-convergence-nested-subpose-plan-v1",
        "scene": scene,
        "datasetDir": str(dataset_dir),
        "sourceSplit": "validation",
        "selection": "category, yaw, candidate count, GT count and visible-weight stratified round-robin",
        "selectionSeed": seed,
        "sourcePoseIndices": selected.tolist(),
        "viewcellCount": int(selected.size),
        "subposesPerViewcell": subpose_count,
        "poseCount": pose_index,
        "nestedPrefixCounts": [value for value in (1, 2, 4, 8, 16, 32, 64, 128) if value <= subpose_count],
        "viewcellShape": viewcell_shape,
        "viewcellRadiusM": radius,
        "viewcellHalfUpM": half_up,
        "sampling": "center followed by a deterministic nested low-discrepancy sequence",
        "modelAndSamplingFovYDeg": 66.0,
        "frontendRenderFovYDeg": 60.0,
        "testRead": False,
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--representative-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--viewcell-shape", choices=("horizontal_disk", "camera_aligned_box"), required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--half-up", type=float, default=0.0)
    parser.add_argument("--viewcell-count", type=int, default=100)
    parser.add_argument("--subpose-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_plan(
        args.dataset_dir,
        args.representative_plan,
        args.output,
        scene=args.scene,
        viewcell_shape=args.viewcell_shape,
        radius=args.radius,
        half_up=args.half_up,
        viewcell_count=args.viewcell_count,
        subpose_count=args.subpose_count,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
