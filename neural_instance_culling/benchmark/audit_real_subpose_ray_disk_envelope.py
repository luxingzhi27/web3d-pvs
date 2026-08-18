#!/usr/bin/env python3
"""Audit real subpose coverage of the first-order view-cell ray disk.

The audit keeps the stored PoseCSR candidate and GT semantics intact.  For a
sampled ``(pose, instance)`` pair it builds the center feature ``c`` and the
two-axis Jacobian ``B`` at the registered view-cell center, then evaluates the
nonlinear nine-dimensional feature at every real subpose position.  The
observed feature is covered by the first-order interval when
``abs(feature - c) <= ||B[row]||_2`` for every row.

This is a diagnostic-only benchmark.  It accepts train, calibration, and
validation, and deliberately refuses the test split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_instance_culling.model.common.viewcell_ray_space import (  # noqa: E402
    VIEW_DIM,
    build_horizontal_disk_ray_query,
    export_ray_space_schema,
)
from neural_instance_culling.model.pose_csr_dataset import PoseCSRDataset  # noqa: E402


AUDIT_SCHEMA = "pvs-real-subpose-ray-disk-envelope-audit-v1"
ALLOWED_SPLITS = ("train", "calibration", "validation")
RARE_POSITIVE_HIT_RATE = 0.05
EXCEEDANCE_TOLERANCE = 1e-7
RADIUS_EPSILON = 1e-8


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_file(
    root: Path,
    meta: dict[str, Any],
    key: str,
    defaults: Iterable[str],
    *,
    required: bool,
) -> Path | None:
    files = meta.get("files")
    candidates: list[Path] = []
    if isinstance(files, dict) and files.get(key):
        candidates.append(root / str(files[key]))
    candidates.extend(root / name for name in defaults)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    if required:
        names = ", ".join(str(path.name) for path in candidates)
        raise FileNotFoundError(f"source viewcell is missing {key}; tried {names}")
    return None


def _read_array(path: Path, dtype: str | np.dtype[Any], *, label: str) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype)
    if values.size == 0 and path.stat().st_size != 0:
        raise ValueError(f"{label} could not be decoded: {path}")
    return values


def _check_finite(values: np.ndarray, label: str) -> None:
    if not bool(np.isfinite(np.asarray(values)).all()):
        raise ValueError(f"{label} contains non-finite values")


@dataclass(frozen=True)
class SourceViewcellData:
    root: Path
    meta: dict[str, Any]
    viewcell_count: int
    subpose_offsets: np.ndarray
    subpose_camera_pos: np.ndarray
    centers: np.ndarray | None
    radii: np.ndarray | None
    visible_offsets: np.ndarray | None
    visible_ids: np.ndarray | None
    visible_hit_counts: np.ndarray | None
    center_source: str
    radius_source: str
    hit_count_source: str

    def subposes(self, pose_index: int) -> np.ndarray:
        start = int(self.subpose_offsets[pose_index])
        end = int(self.subpose_offsets[pose_index + 1])
        return np.asarray(self.subpose_camera_pos[start:end], dtype=np.float32)

    def hit_counts_for_row(self, pose_index: int, visible_ref_count: int) -> np.ndarray | None:
        if self.visible_hit_counts is None:
            return None
        if self.visible_offsets is None:
            return None
        start = int(self.visible_offsets[pose_index])
        end = int(self.visible_offsets[pose_index + 1])
        values = np.asarray(self.visible_hit_counts[start:end], dtype=np.uint16)
        if values.size != int(visible_ref_count):
            raise ValueError(
                f"source visible hit count row {pose_index} has {values.size} entries, "
                f"expected {visible_ref_count}"
            )
        return values


def load_source_viewcell(source_dir: Path) -> SourceViewcellData:
    """Load and validate the source view-cell geometry needed by the audit."""

    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source viewcell directory does not exist: {source_dir}")
    meta_path = source_dir / "dataset_meta.json"
    meta = _read_json(meta_path) if meta_path.is_file() else {}

    offsets_path = _resolve_file(
        source_dir,
        meta,
        "subposeOffsets",
        ("subpose_offsets.bin",),
        required=True,
    )
    positions_path = _resolve_file(
        source_dir,
        meta,
        "subposeCameraPos",
        ("subpose_camera_pos.bin",),
        required=True,
    )
    assert offsets_path is not None and positions_path is not None
    offsets = _read_array(offsets_path, "<u8", label="subpose_offsets.bin")
    if offsets.size < 2 or int(offsets[0]) != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("source subpose_offsets.bin must be monotone and start at zero")
    viewcell_count = int(offsets.size - 1)
    declared_count = meta.get("viewcellCount", meta.get("poseCount"))
    if declared_count is not None and int(declared_count) != viewcell_count:
        raise ValueError(
            f"source viewcell count {viewcell_count} disagrees with metadata {declared_count}"
        )

    position_values = _read_array(positions_path, "<f4", label="subpose_camera_pos.bin")
    if position_values.size != int(offsets[-1]) * 3:
        raise ValueError(
            "source subpose_camera_pos.bin length does not match the terminal subpose offset"
        )
    positions = position_values.reshape(-1, 3)
    _check_finite(positions, "source subpose_camera_pos.bin")

    centers_path = _resolve_file(
        source_dir,
        meta,
        "viewcellCenters",
        ("viewcell_centers.bin", "query_center_world.bin"),
        required=False,
    )
    centers: np.ndarray | None = None
    center_source = "not_available"
    if centers_path is not None:
        center_values = _read_array(centers_path, "<f4", label="source viewcell centers")
        if center_values.size != viewcell_count * 3:
            raise ValueError("source viewcell centers do not match viewcell count")
        centers = center_values.reshape(viewcell_count, 3)
        _check_finite(centers, "source viewcell centers")
        center_source = centers_path.name

    radius_path = _resolve_file(
        source_dir,
        meta,
        "viewcellRadiusM",
        ("viewcell_radius_m.bin",),
        required=False,
    )
    radii: np.ndarray | None = None
    radius_source = "not_available"
    if radius_path is not None:
        radius_values = _read_array(radius_path, "<f4", label="source viewcell radii")
        if radius_values.size != viewcell_count:
            raise ValueError("source viewcell radii do not match viewcell count")
        radii = radius_values.astype(np.float32, copy=False)
        radius_source = radius_path.name
    else:
        params_path = _resolve_file(
            source_dir,
            meta,
            "viewcellParams",
            ("viewcell_params.bin",),
            required=False,
        )
        if params_path is not None:
            params = _read_array(params_path, "<f4", label="source viewcell params")
            if params.size != viewcell_count * 8:
                raise ValueError("source viewcell params do not match viewcell count")
            params = params.reshape(viewcell_count, 8)
            _check_finite(params, "source viewcell params")
            radii = params[:, 0].astype(np.float32, copy=False)
            radius_source = f"{params_path.name}[...,0]"
    if radii is not None:
        _check_finite(radii, "source viewcell radii")
        if bool((radii < 0.0).any()):
            raise ValueError("source viewcell radii must be non-negative")

    visible_offsets_path = _resolve_file(
        source_dir,
        meta,
        "visibleOffsets",
        ("visible_offsets.bin",),
        required=False,
    )
    visible_ids_path = _resolve_file(
        source_dir,
        meta,
        "visibleIds",
        ("visible_ids.bin",),
        required=False,
    )
    hit_counts_path = _resolve_file(
        source_dir,
        meta,
        "visibleHitCounts",
        ("visible_hit_counts.bin",),
        required=False,
    )
    visible_offsets: np.ndarray | None = None
    visible_ids: np.ndarray | None = None
    visible_hit_counts: np.ndarray | None = None
    hit_count_source = "not_available"
    if visible_offsets_path is not None:
        visible_offsets = _read_array(visible_offsets_path, "<u8", label="source visible_offsets.bin")
        if visible_offsets.size != viewcell_count + 1:
            raise ValueError("source visible_offsets.bin does not match viewcell count")
        if int(visible_offsets[0]) != 0 or np.any(visible_offsets[1:] < visible_offsets[:-1]):
            raise ValueError("source visible_offsets.bin must be monotone and start at zero")
    if visible_ids_path is not None:
        visible_ids = _read_array(visible_ids_path, "<u4", label="source visible_ids.bin")
        if visible_offsets is not None and int(visible_offsets[-1]) != visible_ids.size:
            raise ValueError("source visible_offsets.bin does not match visible_ids.bin")
    if hit_counts_path is not None:
        visible_hit_counts = _read_array(
            hit_counts_path,
            "<u2",
            label="source visible_hit_counts.bin",
        )
        if visible_offsets is not None and int(visible_offsets[-1]) != visible_hit_counts.size:
            raise ValueError("source visible_hit_counts.bin does not match visible_offsets.bin")
        hit_count_source = hit_counts_path.name
    if visible_ids is not None and visible_offsets is None:
        raise ValueError("source visible_ids.bin requires source visible_offsets.bin")
    if visible_hit_counts is not None and visible_offsets is None:
        # A source hit-count array without source visible offsets is still
        # usable when its row order is exactly the PoseCSR visible CSR order.
        hit_count_source = f"{hit_counts_path.name} (PoseCSR row order)" if hit_counts_path else hit_count_source

    return SourceViewcellData(
        root=source_dir,
        meta=meta,
        viewcell_count=viewcell_count,
        subpose_offsets=offsets,
        subpose_camera_pos=positions,
        centers=centers,
        radii=radii,
        visible_offsets=visible_offsets,
        visible_ids=visible_ids,
        visible_hit_counts=visible_hit_counts,
        center_source=center_source,
        radius_source=radius_source,
        hit_count_source=hit_count_source,
    )


def load_runtime_aabbs(runtime_meta_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the instance AABBs indexed by componentGlobalId."""

    runtime_meta_path = Path(runtime_meta_path).resolve()
    payload = _read_json(runtime_meta_path)
    records = payload.get("componentRecords") or []
    if not records:
        raise ValueError(f"runtimeVisibilityMeta has no componentRecords: {runtime_meta_path}")
    ids = [int(record["componentGlobalId"]) for record in records]
    if min(ids) < 0 or len(set(ids)) != len(ids):
        raise ValueError("runtime componentGlobalId values must be unique and non-negative")
    count = max(ids) + 1
    if set(ids) != set(range(count)):
        raise ValueError("runtime componentGlobalId values must form a contiguous instance table")
    aabbs = np.empty((count, 6), dtype=np.float32)
    for record in records:
        instance_id = int(record["componentGlobalId"])
        bounds = record.get("bounds")
        if not isinstance(bounds, dict):
            raise ValueError(f"runtime component {instance_id} has no bounds")
        if "min" in bounds and "max" in bounds:
            minimum = np.asarray(bounds["min"], dtype=np.float32)
            maximum = np.asarray(bounds["max"], dtype=np.float32)
        elif "center" in bounds and "size" in bounds:
            center = np.asarray(bounds["center"], dtype=np.float32)
            size = np.asarray(bounds["size"], dtype=np.float32)
            minimum = center - size * 0.5
            maximum = center + size * 0.5
        else:
            raise ValueError(f"runtime component {instance_id} has unsupported bounds")
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError(f"runtime component {instance_id} bounds must be three-dimensional")
        if not bool(np.isfinite(minimum).all() and np.isfinite(maximum).all()):
            raise ValueError(f"runtime component {instance_id} bounds are non-finite")
        if bool((maximum < minimum).any()):
            raise ValueError(f"runtime component {instance_id} bounds have negative extent")
        aabbs[instance_id, :3] = minimum
        aabbs[instance_id, 3:] = maximum
    return aabbs, payload


def _sample_without_replacement(
    values: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if int(limit) < 0:
        raise ValueError("sampling limits must be non-negative; zero means no cap")
    if int(limit) == 0 or values.size <= int(limit):
        return values.copy()
    return np.sort(rng.choice(values, size=int(limit), replace=False).astype(np.int64, copy=False))


def _sample_candidates(
    candidates: np.ndarray,
    visible: np.ndarray,
    limit: int,
    rng: np.random.Generator,
    *,
    pose_index: int,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64).reshape(-1)
    visible = np.unique(np.asarray(visible, dtype=np.int64).reshape(-1))
    if candidates.size != np.unique(candidates).size:
        raise ValueError(f"candidate IDs contain duplicates at pose {pose_index}")
    missing = np.setdiff1d(visible, candidates, assume_unique=False)
    if missing.size:
        raise ValueError(
            f"visible IDs are absent from stored candidates at pose {pose_index}: {missing[:8].tolist()}"
        )
    if int(limit) < 0:
        raise ValueError("sampling limits must be non-negative; zero means no cap")
    if int(limit) == 0 or candidates.size <= int(limit):
        return candidates.copy()
    if visible.size > int(limit):
        raise ValueError(
            f"max_candidates={limit} would drop {visible.size} GT instances at pose {pose_index}"
        )
    negatives = np.setdiff1d(candidates, visible, assume_unique=True)
    chosen_count = int(limit) - int(visible.size)
    if chosen_count > negatives.size:
        chosen_count = int(negatives.size)
    chosen = _sample_without_replacement(negatives, chosen_count, rng)
    return np.sort(np.concatenate([visible, chosen]).astype(np.int64, copy=False))


def _as_numpy_features(value: torch.Tensor) -> np.ndarray:
    result = value.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 2 or result.shape[1] != VIEW_DIM:
        raise RuntimeError(f"ray feature shape is not [B, {VIEW_DIM}]: {result.shape}")
    _check_finite(result, "ray feature")
    return result


def compute_nonlinear_ray_features(
    camera_positions: np.ndarray,
    camera_view: np.ndarray,
    instance_aabbs: np.ndarray,
) -> np.ndarray:
    """Evaluate the existing nonlinear ray feature at camera positions."""

    camera_positions = np.asarray(camera_positions, dtype=np.float32).reshape(-1, 3)
    camera_view = np.asarray(camera_view, dtype=np.float32).reshape(-1, 5)
    instance_aabbs = np.asarray(instance_aabbs, dtype=np.float32).reshape(-1, 6)
    if not (camera_positions.shape[0] == camera_view.shape[0] == instance_aabbs.shape[0]):
        raise ValueError("camera positions, camera views, and AABBs must have the same row count")
    if camera_positions.shape[0] == 0:
        return np.zeros((0, VIEW_DIM), dtype=np.float32)
    with torch.no_grad():
        query = build_horizontal_disk_ray_query(
            torch.from_numpy(camera_positions),
            torch.from_numpy(camera_view),
            torch.from_numpy(instance_aabbs),
            0.0,
        )
    return _as_numpy_features(query.center_view)


@dataclass
class _MetricAccumulator:
    pair_count: int = 0
    sample_count: int = 0
    exceed_count: np.ndarray | None = None
    pair_exceed_count: np.ndarray | None = None
    sum_excess: np.ndarray | None = None
    max_excess: np.ndarray | None = None
    max_abs_deviation: np.ndarray | None = None
    max_radius: np.ndarray | None = None
    sum_signed_slack: np.ndarray | None = None
    min_signed_slack: np.ndarray | None = None
    max_multiplier: np.ndarray | None = None
    unbounded_count: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.exceed_count = np.zeros((VIEW_DIM,), dtype=np.int64)
        self.pair_exceed_count = np.zeros((VIEW_DIM,), dtype=np.int64)
        self.sum_excess = np.zeros((VIEW_DIM,), dtype=np.float64)
        self.max_excess = np.zeros((VIEW_DIM,), dtype=np.float64)
        self.max_abs_deviation = np.zeros((VIEW_DIM,), dtype=np.float64)
        self.max_radius = np.zeros((VIEW_DIM,), dtype=np.float64)
        self.sum_signed_slack = np.zeros((VIEW_DIM,), dtype=np.float64)
        self.min_signed_slack = np.full((VIEW_DIM,), np.inf, dtype=np.float64)
        self.max_multiplier = np.ones((VIEW_DIM,), dtype=np.float64)
        self.unbounded_count = np.zeros((VIEW_DIM,), dtype=np.int64)

    def update(
        self,
        features: np.ndarray,
        centers: np.ndarray,
        row_norms: np.ndarray,
        *,
        pair_count: int,
        subpose_count: int,
    ) -> None:
        features = np.asarray(features, dtype=np.float64).reshape(-1, VIEW_DIM)
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, VIEW_DIM)
        row_norms = np.asarray(row_norms, dtype=np.float64).reshape(-1, VIEW_DIM)
        if not (features.shape == centers.shape == row_norms.shape):
            raise ValueError("feature, center, and interval-radius shapes must match")
        expected = int(pair_count) * int(subpose_count)
        if features.shape[0] != expected:
            raise ValueError(f"expected {expected} subpose rows, got {features.shape[0]}")
        if expected == 0:
            return
        absolute = np.abs(features - centers)
        raw_excess = np.maximum(absolute - row_norms, 0.0)
        exceeded = raw_excess > EXCEEDANCE_TOLERANCE
        excess = np.where(exceeded, raw_excess, 0.0)
        assert self.exceed_count is not None
        assert self.pair_exceed_count is not None
        assert self.sum_excess is not None
        assert self.max_excess is not None
        assert self.max_abs_deviation is not None
        assert self.max_radius is not None
        assert self.sum_signed_slack is not None
        assert self.min_signed_slack is not None
        assert self.max_multiplier is not None
        assert self.unbounded_count is not None
        self.pair_count += int(pair_count)
        self.sample_count += int(expected)
        self.exceed_count += exceeded.sum(axis=0).astype(np.int64, copy=False)
        reshaped_exceeded = exceeded.reshape(int(pair_count), int(subpose_count), VIEW_DIM)
        self.pair_exceed_count += reshaped_exceeded.any(axis=1).sum(axis=0).astype(np.int64, copy=False)
        self.sum_excess += excess.sum(axis=0)
        self.max_excess = np.maximum(self.max_excess, excess.max(axis=0))
        self.max_abs_deviation = np.maximum(self.max_abs_deviation, absolute.max(axis=0))
        self.max_radius = np.maximum(self.max_radius, row_norms.max(axis=0))
        signed_slack = row_norms - absolute
        self.sum_signed_slack += signed_slack.sum(axis=0)
        self.min_signed_slack = np.minimum(self.min_signed_slack, signed_slack.min(axis=0))

        has_radius = row_norms > RADIUS_EPSILON
        ratio = np.ones_like(absolute)
        np.divide(absolute, row_norms, out=ratio, where=has_radius)
        unbounded = (~has_radius) & (absolute > EXCEEDANCE_TOLERANCE)
        self.unbounded_count += unbounded.sum(axis=0).astype(np.int64, copy=False)
        finite_ratio = np.where(unbounded, 1.0, ratio)
        self.max_multiplier = np.maximum(self.max_multiplier, finite_ratio.max(axis=0))

    def to_dict(self) -> dict[str, Any]:
        assert self.exceed_count is not None
        assert self.pair_exceed_count is not None
        assert self.sum_excess is not None
        assert self.max_excess is not None
        assert self.max_abs_deviation is not None
        assert self.max_radius is not None
        assert self.sum_signed_slack is not None
        assert self.min_signed_slack is not None
        assert self.max_multiplier is not None
        assert self.unbounded_count is not None
        sample_count = max(1, int(self.sample_count))
        pair_count = max(1, int(self.pair_count))
        dimensions: list[dict[str, Any]] = []
        exceed_rates: list[float] = []
        max_excess: list[float] = []
        interval_slack: list[float | None] = []
        min_signed_slack: list[float] = []
        mean_signed_slack: list[float] = []
        for dimension in range(VIEW_DIM):
            rate = float(self.exceed_count[dimension] / sample_count) if self.sample_count else 0.0
            signed_min = (
                float(self.min_signed_slack[dimension])
                if self.sample_count
                else 0.0
            )
            signed_mean = (
                float(self.sum_signed_slack[dimension] / sample_count)
                if self.sample_count
                else 0.0
            )
            if self.unbounded_count[dimension] > 0:
                multiplier: float | None = None
                relative_slack: float | None = None
            else:
                multiplier = float(self.max_multiplier[dimension]) if self.sample_count else 1.0
                relative_slack = max(0.0, multiplier - 1.0)
            exceed_rates.append(rate)
            max_excess.append(float(self.max_excess[dimension]))
            interval_slack.append(signed_min)
            min_signed_slack.append(signed_min)
            mean_signed_slack.append(signed_mean)
            dimensions.append(
                {
                    "index": dimension,
                    "sampleCount": int(self.sample_count),
                    "pairCount": int(self.pair_count),
                    "exceedCount": int(self.exceed_count[dimension]),
                    "exceedRate": rate,
                    "pairExceedCount": int(self.pair_exceed_count[dimension]),
                    "pairExceedRate": float(self.pair_exceed_count[dimension] / pair_count)
                    if self.pair_count
                    else 0.0,
                    "maxExcess": float(self.max_excess[dimension]),
                    "meanExcess": float(self.sum_excess[dimension] / sample_count)
                    if self.sample_count
                    else 0.0,
                    "maxAbsoluteDeviation": float(self.max_abs_deviation[dimension]),
                    "maxEnvelopeRadius": float(self.max_radius[dimension]),
                    "intervalSlack": signed_min,
                    "minSignedIntervalSlack": signed_min,
                    "meanSignedIntervalSlack": signed_mean,
                    "relativeIntervalRelaxation": relative_slack,
                    "requiredRadiusMultiplier": multiplier,
                    "unboundedRelaxationSampleCount": int(self.unbounded_count[dimension]),
                }
            )
        return {
            "pairCount": int(self.pair_count),
            "subposeSampleCount": int(self.sample_count),
            "exceedCountByDimension": [int(value) for value in self.exceed_count.tolist()],
            "exceedRateByDimension": exceed_rates,
            "pairExceedCountByDimension": [int(value) for value in self.pair_exceed_count.tolist()],
            "pairExceedRateByDimension": [
                float(value / pair_count) if self.pair_count else 0.0
                for value in self.pair_exceed_count.tolist()
            ],
            "maxExcessByDimension": max_excess,
            "maxAdditiveIntervalSlackByDimension": max_excess,
            "meanExcessByDimension": [
                float(value / sample_count) if self.sample_count else 0.0
                for value in self.sum_excess.tolist()
            ],
            "maxAbsoluteDeviationByDimension": [float(value) for value in self.max_abs_deviation.tolist()],
            "maxEnvelopeRadiusByDimension": [float(value) for value in self.max_radius.tolist()],
            "intervalSlackByDimension": interval_slack,
            "minSignedIntervalSlackByDimension": min_signed_slack,
            "meanSignedIntervalSlackByDimension": mean_signed_slack,
            "maxRelativeIntervalRelaxationByDimension": [
                None
                if self.unbounded_count[dimension] > 0
                else (max(0.0, float(self.max_multiplier[dimension] - 1.0)) if self.sample_count else 0.0)
                for dimension in range(VIEW_DIM)
            ],
            "requiredRadiusMultiplierByDimension": [
                None
                if self.unbounded_count[dimension] > 0
                else (float(self.max_multiplier[dimension]) if self.sample_count else 1.0)
                for dimension in range(VIEW_DIM)
            ],
            "unboundedRelaxationSampleCountByDimension": [
                int(value) for value in self.unbounded_count.tolist()
            ],
            "intervalSlackDefinition": (
                "min(row_norm(B)-abs(actual-c)) over observed subpose samples; "
                "negative means the first-order interval is exceeded"
            ),
            "relativeIntervalRelaxationDefinition": (
                "max(max(abs(actual-c)/row_norm(B)-1, 0)); null means a positive deviation was "
                "observed where row_norm(B) was zero"
            ),
            "exceedanceDefinition": "max(abs(actual-c)-row_norm(B), 0) > 1e-7",
            "dimensions": dimensions,
        }


def _query_contract(
    dataset: PoseCSRDataset,
    source: SourceViewcellData,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pose_count = int(dataset.poses.size)
    if source.viewcell_count != pose_count:
        raise ValueError(
            f"source viewcell count {source.viewcell_count} does not match PoseCSR pose count {pose_count}"
        )
    dataset_centers = (
        np.asarray(dataset.query_centers_world, dtype=np.float32)
        if dataset.query_centers_world is not None
        else None
    )
    dataset_radii = (
        np.asarray(dataset.viewcell_radii_m, dtype=np.float32)
        if dataset.viewcell_radii_m is not None
        else None
    )
    centers = dataset_centers if dataset_centers is not None else source.centers
    radii = dataset_radii if dataset_radii is not None else source.radii
    if centers is None:
        raise ValueError(
            "the audit needs a registered view-cell center from PoseCSR query_center_world.bin "
            "or source viewcell_centers.bin"
        )
    if radii is None:
        raise ValueError(
            "the audit needs a registered view-cell radius from PoseCSR viewcell_radius_m.bin "
            "or source viewcell_radius_m.bin/viewcell_params.bin"
        )
    centers = np.asarray(centers, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    if centers.shape != (pose_count, 3) or radii.shape != (pose_count,):
        raise ValueError("view-cell center/radius arrays do not match PoseCSR pose count")
    _check_finite(centers, "view-cell centers")
    _check_finite(radii, "view-cell radii")
    if bool((radii < 0.0).any()):
        raise ValueError("view-cell radii must be non-negative")
    center_residual = None
    radius_residual = None
    if dataset_centers is not None and source.centers is not None:
        center_residual = float(np.max(np.abs(dataset_centers - source.centers)))
        if center_residual > 1e-4:
            raise ValueError(
                f"PoseCSR and source view-cell centers disagree; max residual={center_residual}"
            )
    if dataset_radii is not None and source.radii is not None:
        radius_residual = float(np.max(np.abs(dataset_radii - source.radii)))
        if radius_residual > 1e-4:
            raise ValueError(
                f"PoseCSR and source view-cell radii disagree; max residual={radius_residual}"
            )
    return centers, radii, {
        "centerSource": "PoseCSR.query_center_world" if dataset_centers is not None else source.center_source,
        "radiusSource": "PoseCSR.viewcell_radius_m" if dataset_radii is not None else source.radius_source,
        "sourceCenterMaxResidual": center_residual,
        "sourceRadiusMaxResidual": radius_residual,
    }


def _validate_source_visible_rows(
    dataset: PoseCSRDataset,
    source: SourceViewcellData,
) -> None:
    if source.visible_offsets is None or source.visible_ids is None:
        return
    for pose_index in range(int(dataset.poses.size)):
        source_start = int(source.visible_offsets[pose_index])
        source_end = int(source.visible_offsets[pose_index + 1])
        source_ids = np.asarray(source.visible_ids[source_start:source_end], dtype=np.uint32)
        dataset_ids = np.asarray(dataset.visible_slice(pose_index)[0], dtype=np.uint32)
        if not np.array_equal(source_ids, dataset_ids):
            raise ValueError(f"source and PoseCSR visible IDs disagree at pose {pose_index}")


def _hit_counts_for_pose(
    dataset: PoseCSRDataset,
    source: SourceViewcellData,
    pose_index: int,
    visible_ref_count: int,
) -> np.ndarray:
    if int(visible_ref_count) == 0:
        return np.zeros((0,), dtype=np.int64)
    source_hits = source.hit_counts_for_row(pose_index, visible_ref_count)
    if source_hits is not None:
        return np.asarray(source_hits, dtype=np.int64)
    if source.visible_hit_counts is not None and source.visible_offsets is None:
        dataset_start = int(dataset.visible_offsets[pose_index])
        dataset_end = int(dataset.visible_offsets[pose_index + 1])
        values = np.asarray(source.visible_hit_counts[dataset_start:dataset_end], dtype=np.int64)
        if values.size != int(visible_ref_count):
            raise ValueError(
                f"source hit-count row {pose_index} has {values.size} entries, expected {visible_ref_count}"
            )
        return values
    if dataset.visible_hit_counts is not None:
        values = np.asarray(dataset.visible_hit_count_slice(pose_index), dtype=np.int64)
        if values.size != int(visible_ref_count):
            raise ValueError(
                f"PoseCSR hit-count row {pose_index} has {values.size} entries, expected {visible_ref_count}"
            )
        return values
    raise ValueError(
        "rare-positive classification requires visible_hit_counts.bin in the source viewcell "
        "or PoseCSR dataset"
    )


def audit_real_subpose_ray_disk_envelope(
    pose_csr: Path,
    source_viewcell_dir: Path,
    runtime_meta: Path,
    split: str,
    *,
    max_poses: int = 0,
    max_candidates: int = 0,
    seed: int = 20260818,
) -> dict[str, Any]:
    """Run the subpose coverage audit and return a JSON-serializable payload."""

    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"split {split!r} is not allowed; choose only train, calibration, or validation; test is refused"
        )
    if int(max_poses) < 0 or int(max_candidates) < 0:
        raise ValueError("sampling limits must be non-negative; zero means no cap")

    world_aabbs, runtime_payload = load_runtime_aabbs(Path(runtime_meta))
    dataset = PoseCSRDataset(Path(pose_csr), num_instances=int(world_aabbs.shape[0]))
    declared_instances = dataset.meta.get("numInstances")
    if declared_instances is not None and int(declared_instances) != int(world_aabbs.shape[0]):
        raise ValueError(
            f"PoseCSR numInstances={declared_instances} disagrees with runtime metadata {world_aabbs.shape[0]}"
        )
    if split not in dataset.split_ids:
        raise ValueError(f"PoseCSR has no explicit {split!r} split")
    source = load_source_viewcell(Path(source_viewcell_dir))
    centers, radii, contract = _query_contract(dataset, source)
    _validate_source_visible_rows(dataset, source)

    rng = np.random.default_rng(int(seed))
    all_pose_indices = np.asarray(dataset.split(split).pose_indices, dtype=np.int64)
    selected_pose_indices = _sample_without_replacement(all_pose_indices, int(max_poses), rng)
    accumulators = {
        "all": _MetricAccumulator(),
        "rarePositive": _MetricAccumulator(),
        "regularPositive": _MetricAccumulator(),
        "negative": _MetricAccumulator(),
    }
    candidate_reference_count = 0
    gt_reference_count = 0
    subpose_reference_count = 0

    for pose_index_raw in selected_pose_indices.tolist():
        pose_index = int(pose_index_raw)
        subposes = source.subposes(pose_index)
        subpose_count = int(subposes.shape[0])
        if subpose_count <= 0:
            raise ValueError(f"selected pose {pose_index} has no real subpose positions")
        visible_ids, _visible_weights = dataset.visible_slice(pose_index)
        visible_ids = np.unique(np.asarray(visible_ids, dtype=np.int64))
        hit_counts = _hit_counts_for_pose(dataset, source, pose_index, int(visible_ids.size))
        if hit_counts.size != visible_ids.size:
            raise ValueError(f"hit count and visible ID lengths disagree at pose {pose_index}")
        if bool((hit_counts < 0).any()) or bool((hit_counts > subpose_count).any()):
            raise ValueError(f"hit counts are outside [0, subpose_count] at pose {pose_index}")
        hit_rate_by_id = {
            int(instance_id): float(hit) / float(subpose_count)
            for instance_id, hit in zip(visible_ids.tolist(), hit_counts.tolist(), strict=True)
        }

        stored_candidates = np.asarray(dataset.frustum_slice(pose_index), dtype=np.int64)
        candidate_ids = _sample_candidates(
            stored_candidates,
            visible_ids,
            int(max_candidates),
            rng,
            pose_index=pose_index,
        )
        if candidate_ids.size and int(candidate_ids.max()) >= world_aabbs.shape[0]:
            raise ValueError(f"candidate ID is outside runtime instance table at pose {pose_index}")
        if visible_ids.size and int(visible_ids.max()) >= world_aabbs.shape[0]:
            raise ValueError(f"visible ID is outside runtime instance table at pose {pose_index}")
        candidate_count = int(candidate_ids.size)
        candidate_reference_count += candidate_count
        gt_reference_count += int(visible_ids.size)
        subpose_reference_count += subpose_count
        if candidate_count == 0:
            continue

        aabbs = np.asarray(world_aabbs[candidate_ids], dtype=np.float32)
        camera_view = np.asarray(dataset.camera_view(pose_index), dtype=np.float32)
        center_world_batch = np.repeat(
            np.asarray(centers[pose_index : pose_index + 1], dtype=np.float32),
            candidate_count,
            axis=0,
        ).copy()
        center_view_batch = np.repeat(camera_view.reshape(1, 5), candidate_count, axis=0)
        with torch.no_grad():
            center_query = build_horizontal_disk_ray_query(
                torch.from_numpy(center_world_batch),
                torch.from_numpy(center_view_batch),
                torch.from_numpy(aabbs),
                float(radii[pose_index]),
            )
        center_features = _as_numpy_features(center_query.center_view)
        disk_axes = center_query.disk_axes.detach().cpu().numpy().astype(np.float32, copy=False)
        row_norms = np.sqrt(np.sum(np.square(disk_axes), axis=-1)).astype(np.float32, copy=False)
        if center_features.shape != (candidate_count, VIEW_DIM) or row_norms.shape != (candidate_count, VIEW_DIM):
            raise RuntimeError("center/disk ray query returned an unexpected shape")

        actual_positions = np.tile(subposes, (candidate_count, 1))
        actual_aabbs = np.repeat(aabbs, subpose_count, axis=0)
        actual_views = np.repeat(camera_view.reshape(1, 5), candidate_count * subpose_count, axis=0)
        actual_features = compute_nonlinear_ray_features(actual_positions, actual_views, actual_aabbs)
        repeated_centers = np.repeat(center_features, subpose_count, axis=0)
        repeated_row_norms = np.repeat(row_norms, subpose_count, axis=0)

        positive = np.isin(candidate_ids, visible_ids, assume_unique=False)
        rare_positive = np.asarray(
            [positive[i] and hit_rate_by_id[int(instance_id)] <= RARE_POSITIVE_HIT_RATE
             for i, instance_id in enumerate(candidate_ids)],
            dtype=bool,
        )
        regular_positive = positive & ~rare_positive
        negative = ~positive
        masks = {
            "all": np.ones((candidate_count,), dtype=bool),
            "rarePositive": rare_positive,
            "regularPositive": regular_positive,
            "negative": negative,
        }
        for name, pair_mask in masks.items():
            pair_indices = np.flatnonzero(pair_mask)
            if pair_indices.size == 0:
                continue
            sample_indices = (
                pair_indices[:, None] * subpose_count + np.arange(subpose_count, dtype=np.int64)[None, :]
            ).reshape(-1)
            accumulators[name].update(
                actual_features[sample_indices],
                repeated_centers[sample_indices],
                repeated_row_norms[sample_indices],
                pair_count=int(pair_indices.size),
                subpose_count=subpose_count,
            )

    category_summaries = {name: accumulator.to_dict() for name, accumulator in accumulators.items()}
    return {
        "schema": AUDIT_SCHEMA,
        "testRead": False,
        "split": split,
        "poseCsr": {
            "path": str(Path(pose_csr).resolve()),
            "poseCount": int(dataset.poses.size),
            "selectedPoseCount": int(selected_pose_indices.size),
            "selectedPoseIndices": [int(value) for value in selected_pose_indices.tolist()],
            "datasetSchema": dataset.meta.get("schema"),
        },
        "sourceViewcell": {
            "path": str(source.root),
            "viewcellCount": int(source.viewcell_count),
            "subposeCount": int(source.subpose_camera_pos.shape[0]),
            "centerSource": source.center_source,
            "radiusSource": source.radius_source,
            "hitCountSource": source.hit_count_source,
        },
        "runtimeMeta": {
            "path": str(Path(runtime_meta).resolve()),
            "instanceCount": int(world_aabbs.shape[0]),
            "sceneName": runtime_payload.get("sceneName") or runtime_payload.get("source"),
        },
        "viewcellContract": contract,
        "sampling": {
            "seed": int(seed),
            "requestedMaxPoses": int(max_poses),
            "requestedMaxCandidatesPerPose": int(max_candidates),
            "poseCountAvailableInSplit": int(all_pose_indices.size),
            "selectedPoseCount": int(selected_pose_indices.size),
            "candidateReferenceCount": int(candidate_reference_count),
            "gtReferenceCount": int(gt_reference_count),
            "subposeReferenceCount": int(subpose_reference_count),
            "candidateCapPreservesVisibleIds": True,
        },
        "feature": {
            **export_ray_space_schema(),
            "actualFeatureEvaluation": "nonlinear ray feature at each source subpose camera position",
            "envelope": "per-feature c +/- row_norm(B), where B is [9,2] disk_axes",
            "subposeCameraOrientation": "PoseCSR camera_forward and tanHalfFovX/Y are reused for same-direction subposes",
            "exceedanceTolerance": EXCEEDANCE_TOLERANCE,
        },
        "categories": category_summaries,
        "categorySemantics": {
            "rarePositive": "PoseCSR visible instance with source hit rate <= 5%",
            "regularPositive": "PoseCSR visible instance with source hit rate > 5%",
            "negative": "Stored candidate instance absent from PoseCSR visible IDs",
            "all": "All sampled stored candidate instances",
            "aggregation": "Each sampled pose-instance pair contributes one row per real source subpose",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-csr", type=Path, required=True, help="PoseCSR dataset directory")
    parser.add_argument(
        "--source-viewcell-dir",
        type=Path,
        required=True,
        help="Original source viewcell directory containing subpose_offsets.bin and subpose_camera_pos.bin",
    )
    parser.add_argument("--runtime-meta", type=Path, required=True, help="runtimeVisibilityMeta.json")
    parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    parser.add_argument("--max-poses", type=int, default=0, help="Maximum sampled poses; zero means all")
    parser.add_argument(
        "--max-candidates",
        "--max-candidates-per-pose",
        dest="max_candidates",
        type=int,
        default=0,
        help="Maximum sampled candidates per pose; zero means all and all GT positives are retained",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_real_subpose_ray_disk_envelope(
        args.pose_csr,
        args.source_viewcell_dir,
        args.runtime_meta,
        args.split,
        max_poses=args.max_poses,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
