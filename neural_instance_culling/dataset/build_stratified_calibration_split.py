#!/usr/bin/env python3
"""Build the in-scene four-way split with a stratified calibration holdout.

The historical validation and test pose IDs stay fixed.  Calibration poses are
selected only from the historical training set while matching scene category,
view direction, candidate count, GT count, and total visible-weight marginals.
"""
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

SPLIT_IDS = {
    "train": 0,
    "validation": 1,
    "calibration": 2,
    "test": 3,
    "guard": 254,
    "unknown": 255,
}
SOURCE_SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


def _quantile_bins(values: np.ndarray, requested_bins: int) -> tuple[np.ndarray, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0,), dtype=np.int64), []
    quantiles = np.linspace(0.0, 1.0, max(1, int(requested_bins)) + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantiles))
    return np.searchsorted(edges, values, side="right").astype(np.int64), edges.astype(float).tolist()


def _visible_weight_sums(offsets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.int64)
    values = np.asarray(weights, dtype=np.float64)
    if (
        offsets.ndim != 1
        or offsets.size == 0
        or int(offsets[0]) != 0
        or np.any(offsets[1:] < offsets[:-1])
        or int(offsets[-1]) != int(values.size)
    ):
        raise ValueError("visible weight offsets are not a valid CSR index")
    prefix = np.empty((values.size + 1,), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=prefix[1:])
    return prefix[offsets[1:]] - prefix[offsets[:-1]]


def _label_columns(
    categories: np.ndarray,
    forward: np.ndarray,
    candidate_counts: np.ndarray,
    visible_counts: np.ndarray,
    visible_weight_sums: np.ndarray,
    train_indices: np.ndarray,
    yaw_bins: int,
    pitch_bins: int,
    numeric_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_forward = np.asarray(forward[train_indices], dtype=np.float64)
    yaw = np.mod(np.arctan2(train_forward[:, 0], -train_forward[:, 2]), 2.0 * np.pi)
    yaw_id = np.floor(yaw / (2.0 * np.pi) * int(yaw_bins)).astype(np.int64)
    pitch = np.arcsin(np.clip(train_forward[:, 1], -1.0, 1.0))
    pitch_id, pitch_edges = _quantile_bins(pitch, pitch_bins)
    candidate_id, candidate_edges = _quantile_bins(np.log1p(candidate_counts[train_indices]), numeric_bins)
    visible_id, visible_edges = _quantile_bins(np.log1p(visible_counts[train_indices]), numeric_bins)
    weight_id, weight_edges = _quantile_bins(np.log1p(visible_weight_sums[train_indices]), numeric_bins)
    category_values, category_id = np.unique(categories[train_indices], return_inverse=True)

    groups = [category_id, yaw_id, pitch_id, candidate_id, visible_id, weight_id]
    widths = [int(group.max()) + 1 if group.size else 0 for group in groups]
    offsets: list[int] = []
    cursor = 0
    columns = np.empty((train_indices.size, len(groups)), dtype=np.int64)
    for column, (group, width) in enumerate(zip(groups, widths)):
        offsets.append(cursor)
        columns[:, column] = group + cursor
        cursor += width
    definition = {
        "categoryValues": category_values.astype(int).tolist(),
        "yawBins": int(yaw_bins),
        "pitchEdgesRadians": pitch_edges,
        "candidateLog1pEdges": candidate_edges,
        "visibleCountLog1pEdges": visible_edges,
        "visibleWeightSumLog1pEdges": weight_edges,
        "labelGroupOffsets": offsets,
        "labelGroupWidths": widths,
        "labelCount": int(cursor),
    }
    return columns, definition


def _objective(counts: np.ndarray, target: np.ndarray) -> float:
    scale = np.maximum(target, 1.0)
    return float(np.square((counts - target) / scale).sum())


def select_stratified_calibration(
    label_columns: np.ndarray,
    count: int,
    seed: int,
    swap_iterations: int,
) -> np.ndarray:
    """Select rows whose feature-bin marginals match the full training pool."""
    row_count = int(label_columns.shape[0])
    if not 0 < int(count) < row_count:
        raise ValueError("calibration count must be between zero and the training row count")
    label_count = int(label_columns.max()) + 1
    population = np.zeros((label_count,), dtype=np.float64)
    for column in range(label_columns.shape[1]):
        population += np.bincount(label_columns[:, column], minlength=label_count)
    target = population * (float(count) / float(row_count))
    rng = np.random.default_rng(int(seed))
    selected = np.zeros((row_count,), dtype=bool)
    selected_counts = np.zeros_like(population)
    jitter = rng.random(row_count) * 1e-8

    for _ in range(int(count)):
        deficit = (target - selected_counts) / np.maximum(target, 1.0)
        scores = deficit[label_columns].sum(axis=1) + jitter
        scores[selected] = -np.inf
        chosen = int(np.argmax(scores))
        selected[chosen] = True
        np.add.at(selected_counts, label_columns[chosen], 1.0)

    selected_rows = np.flatnonzero(selected)
    unselected_rows = np.flatnonzero(~selected)
    current_objective = _objective(selected_counts, target)
    for _ in range(max(0, int(swap_iterations))):
        left = int(selected_rows[rng.integers(0, selected_rows.size)])
        right = int(unselected_rows[rng.integers(0, unselected_rows.size)])
        changed = np.unique(np.concatenate([label_columns[left], label_columns[right]]))
        before = selected_counts[changed].copy()
        trial = before.copy()
        remove = {int(value) for value in label_columns[left].tolist()}
        add = {int(value) for value in label_columns[right].tolist()}
        for index, label in enumerate(changed.tolist()):
            trial[index] += float(label in add) - float(label in remove)
        scale = np.maximum(target[changed], 1.0)
        delta = float(
            np.square((trial - target[changed]) / scale).sum()
            - np.square((before - target[changed]) / scale).sum()
        )
        if delta >= -1e-15:
            continue
        selected_counts[changed] = trial
        selected[left] = False
        selected[right] = True
        selected_rows = np.flatnonzero(selected)
        unselected_rows = np.flatnonzero(~selected)
        current_objective += delta

    if int(selected.sum()) != int(count):
        raise RuntimeError("stratified selector returned the wrong calibration count")
    return np.flatnonzero(selected)


def _describe(
    indices: np.ndarray,
    categories: np.ndarray,
    candidate_counts: np.ndarray,
    visible_counts: np.ndarray,
    weight_sums: np.ndarray,
) -> dict[str, Any]:
    if indices.size == 0:
        return {"poseCount": 0}
    return {
        "poseCount": int(indices.size),
        "averageCandidateCount": float(candidate_counts[indices].mean()),
        "averageVisibleCount": float(visible_counts[indices].mean()),
        "visibleCandidateRatio": float(visible_counts[indices].sum() / max(1, candidate_counts[indices].sum())),
        "averageVisibleWeightSum": float(weight_sums[indices].mean()),
        "categoryCounts": {
            str(int(value)): int(np.count_nonzero(categories[indices] == value))
            for value in np.unique(categories[indices]).tolist()
        },
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    meta = json.loads((dataset_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    pose_count = int(meta["poseCount"])
    poses = np.fromfile(dataset_dir / "poses.bin", dtype=POSE_DTYPE)
    source_split_path = args.source_split_ids.resolve() if args.source_split_ids else None
    source_split = (
        np.fromfile(source_split_path, dtype=np.uint8)
        if source_split_path is not None
        else np.asarray(poses["split"], dtype=np.uint8)
    )
    if poses.size != pose_count or source_split.size != pose_count:
        raise ValueError("dataset poses and historical split labels must match poseCount")
    if np.any(~np.isin(source_split, np.asarray(list(SOURCE_SPLIT_IDS.values()), dtype=np.uint8))):
        raise ValueError("historical split file contains unsupported labels")

    candidate_offsets = np.fromfile(dataset_dir / "candidate_offsets.bin", dtype="<u8")
    visible_offsets = np.fromfile(dataset_dir / "visible_offsets.bin", dtype="<u8")
    visible_weights = np.fromfile(dataset_dir / "visible_weights.bin", dtype="<f4")
    if candidate_offsets.size != pose_count + 1 or visible_offsets.size != pose_count + 1:
        raise ValueError("CSR offsets do not match poseCount")
    candidate_counts = np.diff(candidate_offsets).astype(np.int64)
    visible_counts = np.diff(visible_offsets).astype(np.int64)
    weight_sums = _visible_weight_sums(visible_offsets, visible_weights)
    historical_train = np.flatnonzero(source_split == SOURCE_SPLIT_IDS["train"])
    calibration_count = int(np.floor(historical_train.size * float(args.calibration_fraction) + 0.5))
    labels, feature_definition = _label_columns(
        poses["category"],
        poses["camera_forward"],
        candidate_counts,
        visible_counts,
        weight_sums,
        historical_train,
        args.yaw_bins,
        args.pitch_bins,
        args.numeric_bins,
    )
    selected_local = select_stratified_calibration(
        labels,
        calibration_count,
        args.seed,
        args.swap_iterations,
    )
    calibration_indices = np.sort(historical_train[selected_local])
    split_ids = np.full((pose_count,), SPLIT_IDS["unknown"], dtype=np.uint8)
    split_ids[source_split == SOURCE_SPLIT_IDS["train"]] = SPLIT_IDS["train"]
    split_ids[source_split == SOURCE_SPLIT_IDS["validation"]] = SPLIT_IDS["validation"]
    split_ids[source_split == SOURCE_SPLIT_IDS["test"]] = SPLIT_IDS["test"]
    split_ids[calibration_indices] = SPLIT_IDS["calibration"]
    if np.any(split_ids == SPLIT_IDS["unknown"]):
        raise RuntimeError("some poses were not assigned to the main split")

    counts = {
        name: int(np.count_nonzero(split_ids == split_id))
        for name, split_id in SPLIT_IDS.items()
        if name in {"train", "validation", "calibration", "test", "guard"}
    }
    expected = {
        "train": int(historical_train.size - calibration_count),
        "validation": int(np.count_nonzero(source_split == SOURCE_SPLIT_IDS["validation"])),
        "calibration": int(calibration_count),
        "test": int(np.count_nonzero(source_split == SOURCE_SPLIT_IDS["test"])),
        "guard": 0,
    }
    if counts != expected:
        raise RuntimeError(f"unexpected split counts: expected={expected}, actual={counts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_ids.tofile(output_dir / "pose_split_ids.bin")
    manifest = {
        "schema": "main-in-scene-stratified-calibration-split-v1",
        "sourceDataset": str(dataset_dir),
        "sourceHistoricalSplitIds": (
            str(source_split_path)
            if source_split_path is not None
            else "poses.bin:split"
        ),
        "poseCount": pose_count,
        "selectionSeed": int(args.seed),
        "calibrationFractionOfHistoricalTrain": float(args.calibration_fraction),
        "splitIds": SPLIT_IDS,
        "sourceSplitIds": SOURCE_SPLIT_IDS,
        "poseCounts": counts,
        "featureDefinition": feature_definition,
        "distribution": {
            name: _describe(
                np.flatnonzero(split_ids == split_id),
                poses["category"],
                candidate_counts,
                visible_counts,
                weight_sums,
            )
            for name, split_id in SPLIT_IDS.items()
            if name in {"train", "validation", "calibration", "test"}
        },
        "files": {"poseSplitIds": "pose_split_ids.bin"},
        "semantics": (
            "historical validation and test pose IDs remain fixed; calibration is a stratified holdout "
            "from historical train; all subposes stay inside their aggregated view-cell"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--source-split-ids",
        type=Path,
        default=None,
        help=(
            "Optional historical uint8 split file. By default, read the train/validation/test "
            "labels directly from the input dataset's poses.bin split field."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--yaw-bins", type=int, default=12)
    parser.add_argument("--pitch-bins", type=int, default=4)
    parser.add_argument("--numeric-bins", type=int, default=8)
    parser.add_argument("--swap-iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if not 0.08 <= args.calibration_fraction <= 0.10:
        parser.error("--calibration-fraction must be between 0.08 and 0.10 for this protocol")
    return args


def main() -> None:
    print(json.dumps(build_manifest(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
