#!/usr/bin/env python3
"""Replay GLB download schedules along an offline pose-index trajectory.

This module is deliberately independent from model training and the browser
runtime.  For every pose it uses the stored CSR candidate set, calls an
existing benchmark runner, aggregates the returned instance scores to GLB
scores, and then replays a deterministic download/decode/upload queue.

The replay does not fabricate image pixels.  The current CSR data only gives
``visible_weights`` (historically also called ``component_weights``), so all
visual utility values are explicitly labelled as the weak
``log1p(visible_weights)`` utility.  A GLB contributes utility only after its
decode/upload completion event, not when its network transfer merely starts.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from evaluate_visual_utility_metrics import (  # noqa: E402
    GLB_AGGREGATIONS,
    SCORE_MODES,
    _aggregate_glb_scores,
    _score_mode_result,
    load_glb_byte_costs,
    load_glb_time_costs,
    parse_learned_model_specs,
)
from model_runners import (  # noqa: E402
    load_runner,
    selected_default_specs,
    select_device,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


TRAJECTORY_SCHEMA = "neuralstreamweb3d-pose-index-trajectory-v1"
REPLAY_SCHEMA = "neuralstreamweb3d-download-trajectory-replay-v1"
UTILITY_TEACHER = "weak_log1p_visible_weights_not_pixel_coverage"
_EPS = 1e-7


@dataclass(frozen=True)
class TrajectoryPose:
    """One offline CSR pose row requested at an absolute trajectory time."""

    sequence: int
    pose_index: int
    time_ms: float


@dataclass(frozen=True)
class TrajectorySpec:
    """Normalized trajectory input plus optional network/cache defaults."""

    path: str
    schema: str
    poses: tuple[TrajectoryPose, ...]
    network: dict[str, Any]
    cache: dict[str, Any]
    source: dict[str, Any]


@dataclass(frozen=True)
class ReplayConfig:
    """Deterministic queue and cache configuration for one replay."""

    bandwidth_bytes_per_sec: float
    request_latency_ms: float
    max_concurrent_downloads: int
    max_concurrent_decode_uploads: int
    initial_cache_glb_ids: frozenset[int]
    cache_mode: str
    drain_after_last_pose_ms: float = 0.0


@dataclass
class PosePlan:
    """Runner output and weak GT utility for one trajectory pose."""

    sequence: int
    pose_index: int
    time_ms: float
    candidate_count: int
    candidate_glb_count: int
    ranked_glbs: np.ndarray
    ranked_scores: np.ndarray
    utility_by_glb: dict[int, float]
    instance_scores: np.ndarray | None = None
    instance_glbs: np.ndarray | None = None

    @property
    def total_utility(self) -> float:
        return float(sum(max(0.0, value) for value in self.utility_by_glb.values()))

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sequence": int(self.sequence),
            "poseIndex": int(self.pose_index),
            "timeMs": float(self.time_ms),
            "candidateCount": int(self.candidate_count),
            "candidateGlbCount": int(self.candidate_glb_count),
            "rankedGlbs": [int(value) for value in self.ranked_glbs.tolist()],
            "rankedScores": [float(value) for value in self.ranked_scores.tolist()],
            "weakUtilityByGlb": {
                str(int(key)): float(value) for key, value in sorted(self.utility_by_glb.items()) if value > 0.0
            },
            "weakUtilityTotal": float(self.total_utility),
        }
        if self.instance_scores is not None and self.instance_glbs is not None:
            payload["instanceGlbs"] = [int(value) for value in self.instance_glbs.tolist()]
            payload["instanceScores"] = [float(value) for value in self.instance_scores.tolist()]
        return payload


def _finite_float(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {result}.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {result}.")
    return result


def _int_value(value: Any, name: str, *, nonnegative: bool = False, positive: bool = False) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if positive and result <= 0:
        raise ValueError(f"{name} must be > 0, got {result}.")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be >= 0, got {result}.")
    return result


def _field(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _parse_pose_rows(payload: dict[str, Any]) -> tuple[TrajectoryPose, ...]:
    rows = payload.get("poses")
    if rows is not None:
        if not isinstance(rows, list) or not rows:
            raise ValueError("trajectory 'poses' must be a non-empty list.")
        parsed: list[TrajectoryPose] = []
        for sequence, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"trajectory poses[{sequence}] must be an object.")
            pose_index_raw = _field(row, "poseIndex", "pose_index")
            time_raw = _field(row, "timeMs", "time_ms")
            if pose_index_raw is None or time_raw is None:
                raise ValueError(f"trajectory poses[{sequence}] needs poseIndex and timeMs.")
            pose_index = _int_value(pose_index_raw, f"poses[{sequence}].poseIndex", nonnegative=True)
            time_ms = _finite_float(time_raw, f"poses[{sequence}].timeMs", nonnegative=True)
            parsed.append(TrajectoryPose(sequence, pose_index, time_ms))
    else:
        raw_indices = payload.get("poseIndices", payload.get("pose_indices"))
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError("trajectory needs a non-empty 'poses' or 'poseIndices' list.")
        step_raw = _field(payload, "stepMs", "stepDurationMs", "step_duration_ms")
        if step_raw is None:
            step_ms = 0.0 if len(raw_indices) == 1 else None
        else:
            step_ms = _finite_float(step_raw, "trajectory stepMs", positive=len(raw_indices) > 1)
        if step_ms is None:
            raise ValueError("poseIndices with more than one pose requires stepMs or stepDurationMs.")
        parsed = []
        for sequence, raw_index in enumerate(raw_indices):
            pose_index = _int_value(raw_index, f"poseIndices[{sequence}]", nonnegative=True)
            parsed.append(TrajectoryPose(sequence, pose_index, float(sequence) * step_ms))

    previous = -math.inf
    for row in parsed:
        if row.time_ms + _EPS < previous:
            raise ValueError("trajectory pose times must be non-decreasing; the evaluator does not reorder navigation.")
        previous = row.time_ms
    return tuple(parsed)


def load_trajectory(path: str | Path) -> TrajectorySpec:
    """Load and normalize an offline pose-index trajectory JSON."""

    trajectory_path = Path(path)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"trajectory JSON does not exist: {trajectory_path}")
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trajectory JSON must contain an object.")
    schema = str(payload.get("schema", ""))
    if schema != TRAJECTORY_SCHEMA:
        raise ValueError(f"Unsupported trajectory schema {schema!r}; expected {TRAJECTORY_SCHEMA!r}.")
    network = payload.get("network", {})
    cache = payload.get("cache", {})
    if not isinstance(network, dict) or not isinstance(cache, dict):
        raise ValueError("trajectory 'network' and 'cache' fields must be objects when present.")
    return TrajectorySpec(
        path=str(trajectory_path),
        schema=schema,
        poses=_parse_pose_rows(payload),
        network=dict(network),
        cache=dict(cache),
        source=payload,
    )


def _parse_id_list(value: str | None) -> frozenset[int] | None:
    if value is None or not value.strip():
        return None
    values: set[int] = set()
    for raw in value.split(","):
        if not raw.strip():
            continue
        values.add(_int_value(raw.strip(), "GLB id", nonnegative=True))
    return frozenset(values)


def resolve_replay_config(
    trajectory: TrajectorySpec,
    *,
    bandwidth_bytes_per_sec: float | None = None,
    request_latency_ms: float | None = None,
    max_concurrent_downloads: int | None = None,
    max_concurrent_decode_uploads: int | None = None,
    cache_mode: str = "trajectory",
    initial_cache_glb_ids: frozenset[int] | None = None,
    all_glb_ids: Iterable[int] = (),
    drain_after_last_pose_ms: float = 0.0,
) -> ReplayConfig:
    """Resolve CLI overrides without inventing missing network parameters."""

    network = trajectory.network
    bandwidth = bandwidth_bytes_per_sec
    if bandwidth is None:
        bandwidth = _field(network, "bandwidthBytesPerSec", "bandwidth_bytes_per_sec")
    if bandwidth is None:
        raise ValueError("A positive bandwidthBytesPerSec is required in the trajectory or CLI.")
    latency = request_latency_ms
    if latency is None:
        latency = _field(network, "requestLatencyMs", "request_latency_ms", default=0.0)
    downloads = max_concurrent_downloads
    if downloads is None:
        downloads = _field(network, "maxConcurrentDownloads", "max_concurrent_downloads", default=1)
    decodes = max_concurrent_decode_uploads
    if decodes is None:
        decodes = _field(network, "maxConcurrentDecodeUploads", "max_concurrent_decode_uploads", default=1)

    resolved_mode = str(cache_mode).lower()
    if resolved_mode not in {"trajectory", "cold", "warm", "warm-all"}:
        raise ValueError("cache mode must be trajectory, cold, warm, or warm-all.")
    trajectory_cache = trajectory.cache
    if resolved_mode == "trajectory":
        resolved_mode = str(trajectory_cache.get("mode", "cold")).lower()
        if resolved_mode not in {"cold", "warm", "warm-all"}:
            raise ValueError("trajectory cache.mode must be cold, warm, or warm-all.")
    ids = initial_cache_glb_ids
    if ids is None:
        raw_ids = _field(trajectory_cache, "initialGlbIds", "initial_glb_ids", default=[])
        if not isinstance(raw_ids, list):
            raise ValueError("cache.initialGlbIds must be a list.")
        ids = frozenset(_int_value(value, "cache.initialGlbIds value", nonnegative=True) for value in raw_ids)
    if resolved_mode == "cold":
        ids = frozenset()
    elif resolved_mode == "warm-all":
        ids = frozenset(_int_value(value, "all GLB id", nonnegative=True) for value in all_glb_ids)
    elif not ids:
        raise ValueError("warm cache mode requires cache.initialGlbIds or --initial-cache-glbs.")

    return ReplayConfig(
        bandwidth_bytes_per_sec=_finite_float(bandwidth, "bandwidthBytesPerSec", positive=True),
        request_latency_ms=_finite_float(latency, "requestLatencyMs", nonnegative=True),
        max_concurrent_downloads=_int_value(downloads, "maxConcurrentDownloads", positive=True),
        max_concurrent_decode_uploads=_int_value(decodes, "maxConcurrentDecodeUploads", positive=True),
        initial_cache_glb_ids=ids,
        cache_mode=resolved_mode,
        drain_after_last_pose_ms=_finite_float(
            drain_after_last_pose_ms,
            "drainAfterLastPoseMs",
            nonnegative=True,
        ),
    )


def _pose_camera_arrays(dataset: Any, pose_index: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    camera_norm = np.asarray(dataset.poses["camera_norm"][pose_index], dtype=np.float32)
    camera_world = np.asarray(dataset.poses["camera_world"][pose_index], dtype=np.float32)
    camera_view = np.asarray(dataset.camera_view(pose_index), dtype=np.float32)
    mvp = None
    if getattr(dataset, "mvp", None) is not None:
        mvp = np.repeat(np.asarray(dataset.mvp_slice(pose_index), dtype=np.float32)[None, :], count, axis=0)
    return (
        np.repeat(camera_norm[None, :], count, axis=0),
        np.repeat(camera_world[None, :], count, axis=0),
        np.repeat(camera_view[None, :], count, axis=0),
        mvp,
    )


def build_pose_plans(
    runner: Any,
    dataset: Any,
    trajectory: TrajectorySpec,
    *,
    score_mode: str,
    aggregation: str,
    aggregation_top_k: int = 2,
    include_instance_scores: bool = False,
) -> tuple[list[PosePlan], dict[str, Any]]:
    """Call one retained runner per trajectory pose and aggregate its scores.

    The stored CSR candidate set is authoritative.  A visible id outside that
    set is a data error and is rejected; it is never silently added to the
    candidate list.
    """

    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported score mode {score_mode!r}; allowed={SCORE_MODES}.")
    if aggregation not in GLB_AGGREGATIONS:
        raise ValueError(f"Unsupported GLB aggregation {aggregation!r}; allowed={GLB_AGGREGATIONS}.")
    if aggregation_top_k <= 0:
        raise ValueError("aggregation_top_k must be positive.")
    num_poses = int(dataset.poses.shape[0])
    instance_to_glb = np.asarray(runner.instance_to_glb, dtype=np.int64).reshape(-1)
    plans: list[PosePlan] = []
    mode_status: dict[str, Any] | None = None

    for item in trajectory.poses:
        pose_index = int(item.pose_index)
        if pose_index < 0 or pose_index >= num_poses:
            raise IndexError(f"trajectory poseIndex {pose_index} is outside dataset pose range [0, {num_poses}).")
        candidate_ids = np.asarray(dataset.frustum_slice(pose_index), dtype=np.int64).reshape(-1)
        if candidate_ids.size:
            if int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= instance_to_glb.size:
                raise ValueError(f"pose {pose_index} has an instance id outside runner.instance_to_glb.")
            if np.unique(candidate_ids).size != candidate_ids.size:
                raise ValueError(f"pose {pose_index} has duplicate stored candidate instance ids.")
        visible_ids, visible_weights = dataset.visible_slice(pose_index)
        visible_ids = np.asarray(visible_ids, dtype=np.int64).reshape(-1)
        visible_weights = np.asarray(visible_weights, dtype=np.float64).reshape(-1)
        if visible_ids.size != visible_weights.size:
            raise ValueError(f"pose {pose_index} has misaligned visible ids and visible weights.")
        if visible_ids.size and np.unique(visible_ids).size != visible_ids.size:
            raise ValueError(f"pose {pose_index} has duplicate visible instance ids.")
        if not np.all(np.isfinite(visible_weights)) or np.any(visible_weights < 0.0):
            raise ValueError(f"pose {pose_index} has invalid visible weights.")
        if visible_ids.size and not np.isin(visible_ids, candidate_ids).all():
            missing = np.setdiff1d(visible_ids, candidate_ids, assume_unique=True)
            raise ValueError(
                f"pose {pose_index} has visible ids outside the stored candidate set: {missing[:20].tolist()}. "
                "Trajectory replay refuses GT candidate repair."
            )

        instance_scores: np.ndarray | None = None
        instance_glbs: np.ndarray | None = None
        if candidate_ids.size:
            camera, world, view, mvp = _pose_camera_arrays(dataset, pose_index, candidate_ids.size)
            result = runner.score_arrays(camera, world, view, candidate_ids, mvp)
            scores, status = _score_mode_result(result, score_mode)
            if scores is None:
                raise RuntimeError(
                    f"runner {getattr(runner, 'name', '<unnamed>')} cannot provide {score_mode}: "
                    f"{status.get('reason', status)}"
                )
            if scores.size != candidate_ids.size:
                raise ValueError(
                    f"runner returned {scores.size} scores for {candidate_ids.size} candidate instances at pose {pose_index}."
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError(f"runner returned non-finite {score_mode} scores at pose {pose_index}.")
            mode_status = status
            instance_glb_values = instance_to_glb[candidate_ids]
            unique_glbs, inverse = np.unique(instance_glb_values, return_inverse=True)
            glb_scores = _aggregate_glb_scores(inverse, scores, aggregation, aggregation_top_k)
            order = np.lexsort((unique_glbs.astype(np.int64, copy=False), -glb_scores))
            ranked_glbs = unique_glbs[order].astype(np.int64, copy=False)
            ranked_scores = glb_scores[order].astype(np.float64, copy=False)
            if include_instance_scores:
                instance_scores = np.asarray(scores, dtype=np.float64)
                instance_glbs = np.asarray(instance_glb_values, dtype=np.int64)
        else:
            ranked_glbs = np.zeros((0,), dtype=np.int64)
            ranked_scores = np.zeros((0,), dtype=np.float64)

        utility_by_glb: dict[int, float] = {}
        if visible_ids.size:
            visible_glbs = instance_to_glb[visible_ids]
            weak_utility = np.log1p(np.maximum(visible_weights, 0.0))
            for glb, value in zip(visible_glbs.tolist(), weak_utility.tolist()):
                utility_by_glb[int(glb)] = utility_by_glb.get(int(glb), 0.0) + float(value)
        plans.append(
            PosePlan(
                sequence=int(item.sequence),
                pose_index=pose_index,
                time_ms=float(item.time_ms),
                candidate_count=int(candidate_ids.size),
                candidate_glb_count=int(ranked_glbs.size),
                ranked_glbs=ranked_glbs,
                ranked_scores=ranked_scores,
                utility_by_glb=utility_by_glb,
                instance_scores=instance_scores,
                instance_glbs=instance_glbs,
            )
        )

    if mode_status is None:
        mode_status = {
            "status": "implemented_empty_trajectory_candidates",
            "definition": score_mode,
            "reason": "All trajectory poses had empty candidate sets; no runner score call was needed.",
        }
    return plans, mode_status


def _covered_utility(plan: PosePlan | None, cache: set[int]) -> float:
    if plan is None:
        return 0.0
    return float(sum(value for glb, value in plan.utility_by_glb.items() if glb in cache))


def _curve_area(samples: list[dict[str, Any]], x_key: str) -> dict[str, float | None]:
    if not samples:
        return {"area": None, "normalizedArea": None, "maxX": None}
    grouped: dict[float, float] = {}
    for sample in samples:
        x = float(sample[x_key])
        y = float(sample["utilityRecall"])
        grouped[x] = max(grouped.get(x, 0.0), y)
    xs = np.asarray(sorted(grouped), dtype=np.float64)
    ys = np.asarray([grouped[value] for value in xs.tolist()], dtype=np.float64)
    area = float(np.trapz(ys, xs)) if xs.size > 1 else 0.0
    max_x = float(xs[-1]) if xs.size else 0.0
    return {"area": area, "normalizedArea": (area / max_x if max_x > 0.0 else 0.0), "maxX": max_x}


def _budget_rows(samples: list[dict[str, Any]], x_key: str, budgets: Iterable[float]) -> dict[str, dict[str, float | int]]:
    ordered = sorted(samples, key=lambda row: (float(row[x_key]), float(row["timeMs"])))
    output: dict[str, dict[str, float | int]] = {}
    for raw_budget in budgets:
        budget = _finite_float(raw_budget, f"{x_key} budget", nonnegative=True)
        eligible = [row for row in ordered if float(row[x_key]) <= budget + _EPS]
        if not eligible:
            output[str(budget)] = {
                "budget": budget,
                "utilityRecall": 0.0,
                "weakUtilityServed": 0.0,
                "resourceAtPoint": 0.0,
                "timeMs": 0.0,
            }
            continue
        best = max(eligible, key=lambda row: (float(row["utilityRecall"]), -float(row["timeMs"])))
        output[str(budget)] = {
            "budget": budget,
            "utilityRecall": float(best["utilityRecall"]),
            "weakUtilityServed": float(best["weakUtilityServed"]),
            "resourceAtPoint": float(best[x_key]),
            "timeMs": float(best["timeMs"]),
        }
    return output


def simulate_download_trajectory(
    plans: list[PosePlan],
    glb_byte_costs: np.ndarray,
    glb_time_costs: np.ndarray,
    config: ReplayConfig,
    *,
    byte_budgets: Iterable[float] = (),
    time_budgets_ms: Iterable[float] = (),
) -> dict[str, Any]:
    """Replay requests, network transfers, and decode/upload completion events.

    Bandwidth is divided into a fixed equal share among the configured network
    slots.  This is intentionally a deterministic approximation, not a claim
    about any browser's TCP scheduler.  A request's network finish time is
    ``latency + bytes / (bandwidth / maxConcurrentDownloads)``.  Decode/upload
    jobs use their measured per-GLB time cost and a separate FIFO concurrency
    limit.
    """

    if not plans:
        raise ValueError("trajectory must contain at least one pose plan.")
    byte_costs = np.asarray(glb_byte_costs, dtype=np.float64).reshape(-1)
    time_costs = np.asarray(glb_time_costs, dtype=np.float64).reshape(-1)
    if byte_costs.size != time_costs.size:
        raise ValueError("GLB byte and decode/upload cost arrays must have equal length.")
    if not np.all(np.isfinite(byte_costs)) or np.any(byte_costs <= 0.0):
        raise ValueError("All GLB byte costs must be finite and > 0 for trajectory replay.")
    if not np.all(np.isfinite(time_costs)) or np.any(time_costs <= 0.0):
        raise ValueError("All GLB decode/upload costs must be finite and > 0 for trajectory replay.")
    if any(plan.time_ms + _EPS < plans[0].time_ms for plan in plans):
        raise ValueError("pose plan times must be non-decreasing.")

    used_glbs = sorted({int(glb) for plan in plans for glb in plan.ranked_glbs.tolist()})
    if any(glb < 0 or glb >= byte_costs.size for glb in used_glbs):
        raise ValueError("A trajectory plan references a GLB outside the cost arrays.")
    initial_cache = set(int(glb) for glb in config.initial_cache_glb_ids)
    if any(glb < 0 or glb >= byte_costs.size for glb in initial_cache):
        raise ValueError("Initial cache contains a GLB outside the cost arrays.")

    relative_plans = plans
    start_absolute_ms = float(plans[0].time_ms)
    last_pose_time = float(plans[-1].time_ms - start_absolute_ms)
    horizon_ms = last_pose_time + float(config.drain_after_last_pose_ms)
    useful_glbs = {
        int(glb)
        for plan in plans
        for glb, value in plan.utility_by_glb.items()
        if float(value) > 0.0
    }
    total_weak_utility = float(sum(plan.total_utility for plan in plans))
    useful_window_end_by_glb: dict[int, float] = {}
    for index, plan in enumerate(plans):
        window_end = (
            float(plans[index + 1].time_ms - start_absolute_ms)
            if index + 1 < len(plans)
            else float(horizon_ms)
        )
        for glb, value in plan.utility_by_glb.items():
            if float(value) > 0.0:
                useful_window_end_by_glb[int(glb)] = max(
                    useful_window_end_by_glb.get(int(glb), -math.inf), window_end
                )

    state: dict[int, str] = {glb: ("cached" if glb in initial_cache else "unseen") for glb in used_glbs}
    cache = set(initial_cache)
    pending: list[tuple[float, int, int, int]] = []
    pending_version: dict[int, int] = {}
    priority_by_glb: dict[int, float] = {}
    enqueue_sequence = 0
    event_sequence = 0
    event_heap: list[tuple[float, int, int, str, int]] = []
    active_downloads: dict[int, float] = {}
    active_decodes: dict[int, float] = {}
    decode_waiting: list[tuple[int, int]] = []

    download_events: list[dict[str, Any]] = []
    decode_upload_events: list[dict[str, Any]] = []
    downloaded_bytes = 0.0
    requested_bytes = 0.0
    invalid_download_bytes = 0.0
    invalid_download_glbs: set[int] = set()
    late_useful_download_bytes = 0.0
    completed_download_glbs: set[int] = set()
    completed_upload_glbs: set[int] = set()

    def push_event(time_ms: float, order: int, kind: str, glb: int) -> None:
        nonlocal event_sequence
        event_sequence += 1
        heapq.heappush(event_heap, (float(time_ms), order, event_sequence, kind, int(glb)))

    def start_downloads(now_ms: float) -> None:
        nonlocal requested_bytes
        while len(active_downloads) < config.max_concurrent_downloads:
            selected: int | None = None
            while pending:
                _negative_priority, _sequence, glb, version = heapq.heappop(pending)
                if state.get(glb) != "queued" or pending_version.get(glb) != version:
                    continue
                selected = glb
                break
            if selected is None:
                return
            glb = selected
            state[glb] = "downloading"
            bytes_for_glb = float(byte_costs[glb])
            requested_bytes += bytes_for_glb
            per_slot_bandwidth = config.bandwidth_bytes_per_sec / float(config.max_concurrent_downloads)
            duration_ms = config.request_latency_ms + bytes_for_glb / per_slot_bandwidth * 1000.0
            finish_ms = float(now_ms + duration_ms)
            active_downloads[glb] = finish_ms
            download_events.append(
                {
                    "glbId": glb,
                    "startMs": float(now_ms),
                    "networkCompleteMs": finish_ms,
                    "bytes": bytes_for_glb,
                }
            )
            push_event(finish_ms, 0, "download_complete", glb)

    def start_decodes(now_ms: float) -> None:
        while len(active_decodes) < config.max_concurrent_decode_uploads and decode_waiting:
            _ready_sequence, glb = heapq.heappop(decode_waiting)
            if state.get(glb) != "decode_queued":
                continue
            state[glb] = "decoding"
            duration_ms = float(time_costs[glb])
            finish_ms = float(now_ms + duration_ms)
            active_decodes[glb] = finish_ms
            decode_upload_events.append(
                {
                    "glbId": glb,
                    "startMs": float(now_ms),
                    "completeMs": finish_ms,
                    "decodeUploadMs": duration_ms,
                }
            )
            push_event(finish_ms, 1, "decode_complete", glb)

    def enqueue_plan(plan: PosePlan) -> None:
        nonlocal enqueue_sequence
        for glb, score in zip(plan.ranked_glbs.tolist(), plan.ranked_scores.tolist()):
            glb = int(glb)
            priority = float(score)
            if not np.isfinite(priority):
                raise ValueError(f"non-finite GLB priority for GLB {glb} at pose {plan.pose_index}.")
            previous = priority_by_glb.get(glb, -math.inf)
            if priority <= previous + _EPS:
                continue
            priority_by_glb[glb] = priority
            if state.get(glb) in {"cached", "downloading", "decode_queued", "decoding", "completed"}:
                continue
            state[glb] = "queued"
            enqueue_sequence += 1
            version = pending_version.get(glb, 0) + 1
            pending_version[glb] = version
            heapq.heappush(pending, (-priority, enqueue_sequence, glb, version))

    def cover_now(plan: PosePlan | None) -> tuple[float, float]:
        covered = _covered_utility(plan, cache)
        total = plan.total_utility if plan is not None else 0.0
        return covered, total

    current_plan: PosePlan | None = None
    active_covered = 0.0
    achieved_weak_utility = 0.0
    current_pose_index: int | None = None
    first_useful: dict[str, Any] | None = None
    missing_integral_weak_utility_ms = 0.0
    missing_integral_ratio_ms = 0.0
    current_time_ms = 0.0
    samples: list[dict[str, Any]] = []

    def update_first_useful(now_ms: float) -> None:
        nonlocal first_useful
        if first_useful is not None or current_plan is None:
            return
        covered, total = cover_now(current_plan)
        if covered > _EPS:
            first_useful = {
                "status": "available",
                "timeMs": float(now_ms),
                "poseIndex": int(current_plan.pose_index),
                "sequence": int(current_plan.sequence),
                "weakUtilityServed": float(covered),
                "weakUtilityRecall": float(covered / total) if total > _EPS else 0.0,
                "utilityBasis": UTILITY_TEACHER,
            }

    def record_sample(now_ms: float, reason: str) -> None:
        update_first_useful(now_ms)
        nonlocal active_covered
        active_covered, active_total = cover_now(current_plan)
        achieved = achieved_weak_utility + active_covered
        recall = achieved / total_weak_utility if total_weak_utility > _EPS else 1.0
        samples.append(
            {
                "timeMs": float(now_ms),
                "downloadedBytes": float(downloaded_bytes),
                "weakUtilityServed": float(achieved),
                "utilityRecall": float(np.clip(recall, 0.0, 1.0)),
                "currentPoseIndex": current_pose_index,
                "event": reason,
                "cacheGlbCount": int(len(cache)),
            }
        )

    def integrate_until(next_time_ms: float) -> None:
        nonlocal current_time_ms, missing_integral_weak_utility_ms, missing_integral_ratio_ms
        duration = float(next_time_ms - current_time_ms)
        if duration < -_EPS:
            raise RuntimeError("internal replay event time moved backwards.")
        if duration > 0.0 and current_plan is not None:
            covered, total = cover_now(current_plan)
            missing = max(0.0, total - covered)
            missing_integral_weak_utility_ms += missing * duration
            missing_integral_ratio_ms += (missing / total if total > _EPS else 0.0) * duration
        current_time_ms = float(next_time_ms)

    def process_event(kind: str, glb: int, now_ms: float) -> None:
        nonlocal downloaded_bytes, invalid_download_bytes, late_useful_download_bytes, active_covered
        if kind == "download_complete":
            if state.get(glb) != "downloading":
                return
            active_downloads.pop(glb, None)
            state[glb] = "decode_queued"
            bytes_for_glb = float(byte_costs[glb])
            downloaded_bytes += bytes_for_glb
            completed_download_glbs.add(glb)
            if glb not in useful_glbs:
                invalid_download_bytes += bytes_for_glb
                invalid_download_glbs.add(glb)
            heapq.heappush(decode_waiting, (len(decode_upload_events), glb))
            start_decodes(now_ms)
            start_downloads(now_ms)
        elif kind == "decode_complete":
            if state.get(glb) != "decoding":
                return
            active_decodes.pop(glb, None)
            state[glb] = "completed"
            cache.add(glb)
            completed_upload_glbs.add(glb)
            last_useful_window_end = useful_window_end_by_glb.get(glb, -math.inf)
            if glb in useful_glbs and now_ms > last_useful_window_end + _EPS:
                late_useful_download_bytes += float(byte_costs[glb])
            start_decodes(now_ms)
            start_downloads(now_ms)
        else:
            raise RuntimeError(f"unknown replay event kind {kind!r}")
        active_covered, _active_total = cover_now(current_plan)

    for plan_index, plan in enumerate(relative_plans):
        push_event(float(plan.time_ms - start_absolute_ms), 2, "pose", plan_index)

    # The first point includes no newly transferred bytes; a warm cache can
    # still contribute utility as soon as the first pose is entered.
    record_sample(0.0, "replay_start")

    while event_heap:
        next_time = float(event_heap[0][0])
        if next_time > horizon_ms + _EPS:
            integrate_until(horizon_ms)
            break
        integrate_until(next_time)
        event_had_change = False
        # Completions have lower event order than a pose at the same time.  A
        # zero-duration decode/network fixture is therefore fully settled
        # before the pose observes its cache.
        while event_heap and float(event_heap[0][0]) <= current_time_ms + _EPS:
            _event_time, _order, _sequence, kind, value = heapq.heappop(event_heap)
            event_had_change = True
            if kind == "pose":
                plan = relative_plans[value]
                if current_plan is not None:
                    active_covered, _active_total = cover_now(current_plan)
                    achieved_weak_utility += active_covered
                current_plan = plan
                current_pose_index = int(plan.pose_index)
                active_covered, _active_total = cover_now(current_plan)
                enqueue_plan(plan)
                start_decodes(current_time_ms)
                start_downloads(current_time_ms)
            else:
                process_event(kind, value, current_time_ms)
        if event_had_change:
            record_sample(current_time_ms, "event_batch")

    if current_time_ms < horizon_ms - _EPS:
        integrate_until(horizon_ms)
    if current_plan is not None:
        active_covered, _active_total = cover_now(current_plan)
    record_sample(horizon_ms, "replay_end")
    if current_plan is not None:
        achieved_weak_utility += active_covered

    if first_useful is None:
        if useful_glbs:
            first_useful = {
                "status": "not_available",
                "reason": "No weak-visible GLB completed before the active pose window ended.",
                "utilityBasis": UTILITY_TEACHER,
            }
        else:
            first_useful = {
                "status": "not_available",
                "reason": "Trajectory has no positive weak visible-weight utility; image utility is unavailable.",
                "utilityBasis": UTILITY_TEACHER,
            }

    initial_cache_bytes = float(sum(byte_costs[glb] for glb in initial_cache))
    unfinished = sorted(glb for glb, value in state.items() if value in {"queued", "downloading", "decode_queued", "decoding"})
    utility_at_bytes = {
        "status": "available",
        "utilityBasis": UTILITY_TEACHER,
        "curve": samples,
        "budgets": _budget_rows(samples, "downloadedBytes", byte_budgets),
        "area": _curve_area(samples, "downloadedBytes"),
    }
    utility_at_time = {
        "status": "available",
        "budgetUnit": "milliseconds",
        "utilityBasis": UTILITY_TEACHER,
        "curve": samples,
        "budgets": _budget_rows(samples, "timeMs", time_budgets_ms),
        "area": _curve_area(samples, "timeMs"),
    }
    final_recall = achieved_weak_utility / total_weak_utility if total_weak_utility > _EPS else 1.0
    return {
        "status": "completed" if not unfinished else "horizon_reached_with_pending_events",
        "utilityBasis": UTILITY_TEACHER,
        "utilityWarning": "Weak visible-weight utility only; no pixel coverage or image utility is fabricated.",
        "navigationStartAbsoluteMs": start_absolute_ms,
        "navigationDurationMs": float(last_pose_time),
        "replayHorizonMs": float(horizon_ms),
        "network": {
            "bandwidthBytesPerSec": float(config.bandwidth_bytes_per_sec),
            "requestLatencyMs": float(config.request_latency_ms),
            "maxConcurrentDownloads": int(config.max_concurrent_downloads),
            "maxConcurrentDecodeUploads": int(config.max_concurrent_decode_uploads),
            "bandwidthModel": "equal_fixed_share_per_download_slot",
        },
        "cache": {
            "mode": config.cache_mode,
            "initialGlbIds": sorted(initial_cache),
            "initialCacheBytes": initial_cache_bytes,
            "finalCachedGlbCount": int(len(cache)),
        },
        "trajectoryDemand": {
            "poseCount": int(len(plans)),
            "weakUtilityTotal": total_weak_utility,
            "trajectoryUsefulGlbIds": sorted(useful_glbs),
            "trajectoryUsefulGlbCount": int(len(useful_glbs)),
        },
        "utilityAtBytes": utility_at_bytes,
        "utilityAtTime": utility_at_time,
        "missingUtility": {
            "utilityBasis": UTILITY_TEACHER,
            "missingWeakUtilityIntegralMs": float(missing_integral_weak_utility_ms),
            "missingUtilityRatioIntegralMs": float(missing_integral_ratio_ms),
            "meanMissingUtilityRatio": float(missing_integral_ratio_ms / horizon_ms) if horizon_ms > _EPS else 0.0,
        },
        "firstUsefulFrame": first_useful,
        "finalTrajectoryUtilityRecall": float(np.clip(final_recall, 0.0, 1.0)),
        "resourceAccounting": {
            "requestedBytes": float(requested_bytes),
            "downloadedBytes": float(downloaded_bytes),
            "inflightOrQueuedBytesAtHorizon": float(sum(byte_costs[glb] for glb in unfinished)),
            "completedDownloadGlbCount": int(len(completed_download_glbs)),
            "completedUploadGlbCount": int(len(completed_upload_glbs)),
            "invalidDownloadBytes": float(invalid_download_bytes),
            "invalidDownloadGlbIds": sorted(invalid_download_glbs),
            "invalidDownloadDefinition": "Downloaded GLB has zero weak visible-weight utility anywhere in this trajectory.",
            "lateUsefulDownloadBytes": float(late_useful_download_bytes),
            "unfinishedGlbIds": unfinished,
        },
        "events": {
            "downloads": download_events,
            "decodeUploads": decode_upload_events,
        },
    }


def evaluate_model_trajectory(
    runner: Any,
    dataset: Any,
    trajectory: TrajectorySpec,
    *,
    runtime_meta_path: str | Path,
    glb_index_path: str | Path,
    glb_root: str | Path | None,
    glb_time_index_path: str | Path,
    config: ReplayConfig,
    score_mode: str,
    aggregation: str,
    aggregation_top_k: int,
    byte_budgets: Iterable[float],
    time_budgets_ms: Iterable[float],
    include_instance_scores: bool = False,
) -> dict[str, Any]:
    """Evaluate one runner and attach the complete reproducible replay trace."""

    plans, score_status = build_pose_plans(
        runner,
        dataset,
        trajectory,
        score_mode=score_mode,
        aggregation=aggregation,
        aggregation_top_k=aggregation_top_k,
        include_instance_scores=include_instance_scores,
    )
    mapping = np.asarray(runner.instance_to_glb, dtype=np.int64).reshape(-1)
    if mapping.size == 0:
        raise ValueError("runner has an empty instance_to_glb mapping.")
    num_glbs = int(mapping.max()) + 1
    byte_costs = load_glb_byte_costs(glb_index_path, glb_root, num_glbs, mapping, allow_missing=False)
    time_costs = load_glb_time_costs(glb_time_index_path, num_glbs, mapping, allow_missing=False)
    if time_costs is None:
        raise ValueError("A strict GLB decode/upload time index is required for completion-event replay.")
    replay = simulate_download_trajectory(
        plans,
        byte_costs,
        time_costs,
        config,
        byte_budgets=byte_budgets,
        time_budgets_ms=time_budgets_ms,
    )
    return {
        "model": str(getattr(runner, "name", "unnamed_runner")),
        "runnerKind": str(getattr(runner, "kind", "unknown")),
        "runnerInfo": {
            "informationLevel": str(getattr(runner, "information_level", "unspecified")),
            "decisionMode": str(getattr(runner, "decision_mode", "unspecified")),
            "thresholdLoadedButUnusedForRankingReplay": float(getattr(runner, "threshold", 0.5)),
        },
        "scoreMode": score_mode,
        "scoreModeStatus": score_status,
        "glbAggregation": aggregation,
        "aggregationTopK": int(aggregation_top_k),
        "trajectory": {
            "schema": trajectory.schema,
            "path": trajectory.path,
            "poseCount": int(len(plans)),
            "poses": [plan.to_json() for plan in plans],
        },
        "costSources": {
            "runtimeMeta": str(runtime_meta_path),
            "glbIndex": str(glb_index_path),
            "glbRoot": str(glb_root) if glb_root is not None else None,
            "glbTimeIndex": str(glb_time_index_path),
            "byteSemantics": "filesystem bytes from glbIndex paths",
            "timeSemantics": "explicit per-GLB decode plus upload completion time",
        },
        "replay": replay,
    }


def _parse_float_list(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    result = tuple(_finite_float(part.strip(), "budget", nonnegative=True) for part in value.split(",") if part.strip())
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate budgets are not allowed: {result}")
    return result


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _self_test() -> dict[str, Any]:
    """Run a model-free deterministic fixture; no repository data is touched."""

    plans = [
        PosePlan(
            sequence=0,
            pose_index=10,
            time_ms=0.0,
            candidate_count=3,
            candidate_glb_count=3,
            ranked_glbs=np.asarray([0, 2, 1], dtype=np.int64),
            ranked_scores=np.asarray([0.9, 0.8, 0.7], dtype=np.float64),
            utility_by_glb={0: 1.0},
        ),
        PosePlan(
            sequence=1,
            pose_index=11,
            time_ms=300.0,
            candidate_count=3,
            candidate_glb_count=3,
            ranked_glbs=np.asarray([1, 0, 2], dtype=np.int64),
            ranked_scores=np.asarray([0.95, 0.6, 0.1], dtype=np.float64),
            utility_by_glb={1: 1.0},
        ),
    ]
    config = ReplayConfig(
        bandwidth_bytes_per_sec=1000.0,
        request_latency_ms=0.0,
        max_concurrent_downloads=2,
        max_concurrent_decode_uploads=1,
        initial_cache_glb_ids=frozenset(),
        cache_mode="cold",
        drain_after_last_pose_ms=500.0,
    )
    result = simulate_download_trajectory(
        plans,
        np.asarray([100.0, 200.0, 50.0], dtype=np.float64),
        np.asarray([20.0, 10.0, 5.0], dtype=np.float64),
        config,
        byte_budgets=(0.0, 100.0, 350.0),
        time_budgets_ms=(0.0, 220.0, 1000.0),
    )
    assert result["status"] == "completed", result
    assert result["resourceAccounting"]["downloadedBytes"] == 350.0, result
    assert result["resourceAccounting"]["invalidDownloadBytes"] == 50.0, result
    assert result["firstUsefulFrame"]["status"] == "available", result
    assert result["firstUsefulFrame"]["timeMs"] == 220.0, result
    assert result["utilityAtBytes"]["budgets"]["350.0"]["utilityRecall"] == 1.0, result

    warm = simulate_download_trajectory(
        plans[:1],
        np.asarray([100.0, 200.0, 50.0], dtype=np.float64),
        np.asarray([20.0, 10.0, 5.0], dtype=np.float64),
        ReplayConfig(
            bandwidth_bytes_per_sec=1000.0,
            request_latency_ms=0.0,
            max_concurrent_downloads=1,
            max_concurrent_decode_uploads=1,
            initial_cache_glb_ids=frozenset({0}),
            cache_mode="warm",
            drain_after_last_pose_ms=0.0,
        ),
    )
    assert warm["firstUsefulFrame"]["timeMs"] == 0.0, warm
    return {"status": "passed", "checks": ["cold_queue", "warm_cache", "completion_events", "weak_utility_label"]}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay GLB downloads along an offline pose-index trajectory.")
    parser.add_argument("--self-test", action="store_true", help="Run a model-free fixture without reading project data.")
    parser.add_argument("--trajectory", help=f"Trajectory JSON with schema {TRAJECTORY_SCHEMA}.")
    parser.add_argument("--dataset-dir", help="Pose CSR dataset directory.")
    parser.add_argument("--runtime-meta", help="runtimeVisibilityMeta.json.")
    parser.add_argument("--glb-index", help="glbIndex.json used to resolve filesystem byte costs.")
    parser.add_argument("--glb-root", help="Root used to resolve paths in glbIndex.json.")
    parser.add_argument("--glb-time-index", help="Strict per-GLB decode/upload time cost index JSON.")
    parser.add_argument("--models", help="Comma-separated retained runner names.")
    parser.add_argument(
        "--learned-model-spec",
        action="append",
        default=[],
        help="Explicit learned runner: name|checkpoint|runtime_features|calibration_summary.",
    )
    parser.add_argument("--independent-ranker-spec", action="append", default=[], help="name|checkpoint for an independent RankNet utility runner.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--score-mode", default="current-cascade", choices=SCORE_MODES)
    parser.add_argument("--glb-aggregation", default="max", choices=GLB_AGGREGATIONS)
    parser.add_argument("--aggregation-top-k", type=int, default=2)
    parser.add_argument("--include-instance-scores", action="store_true")
    parser.add_argument("--bandwidth-bytes-per-sec", type=float)
    parser.add_argument("--request-latency-ms", type=float)
    parser.add_argument("--max-concurrent-downloads", type=int)
    parser.add_argument("--max-concurrent-decode-uploads", type=int)
    parser.add_argument("--cache-mode", default="trajectory", choices=("trajectory", "cold", "warm", "warm-all"))
    parser.add_argument("--initial-cache-glbs", default=None, help="Comma-separated GLB ids; overrides trajectory cache ids.")
    parser.add_argument("--drain-after-last-pose-ms", type=float, default=0.0)
    parser.add_argument("--byte-budgets", default="", help="Comma-separated strict downloaded-byte budgets.")
    parser.add_argument("--time-budgets-ms", default="", help="Comma-separated elapsed-time budgets in milliseconds.")
    parser.add_argument("--output", help="Output JSON path. Without it, only a concise result is printed.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2))
        return
    required = ("trajectory", "dataset_dir", "runtime_meta", "glb_index", "glb_time_index")
    missing = [f"--{name.replace('_', '-')}" for name in required if not getattr(args, name)]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    try:
        trajectory = load_trajectory(args.trajectory)
        byte_budgets = _parse_float_list(args.byte_budgets)
        time_budgets = _parse_float_list(args.time_budgets_ms)
        initial_ids = _parse_id_list(args.initial_cache_glbs)
        specs = selected_default_specs(args.models or "")
        explicit_specs = parse_learned_model_specs(args.learned_model_spec)
        duplicate_specs = sorted(set(specs).intersection(explicit_specs))
        if duplicate_specs:
            raise ValueError(f"explicit learned model specs duplicate retained models: {duplicate_specs}")
        specs.update(explicit_specs)
        for raw in args.independent_ranker_spec:
            parts = str(raw).split("|")
            if len(parts) != 2 or any(not part.strip() for part in parts):
                raise ValueError("--independent-ranker-spec must use name|checkpoint")
            name, checkpoint = (part.strip() for part in parts)
            if name in specs:
                raise ValueError(f"independent ranker duplicates model name {name!r}")
            specs[name] = {"kind": "independent_utility_ranker", "checkpoint": checkpoint}
        if not specs:
            raise ValueError("--models must contain at least one retained runner.")
        device = select_device(args.device)
        # The runner supplies the authoritative instance-to-GLB mapping.  We
        # resolve the cache after loading the first runner, then require all
        # selected models to share the same scene mapping.
        runners: list[Any] = []
        datasets: list[Any] = []
        first_mapping: np.ndarray | None = None
        for name, spec in specs.items():
            runner = load_runner(name, spec, args.runtime_meta, device, dataset_dir=args.dataset_dir)
            mapping = np.asarray(runner.instance_to_glb, dtype=np.int64).reshape(-1)
            if first_mapping is None:
                first_mapping = mapping
            elif not np.array_equal(first_mapping, mapping):
                raise ValueError("selected runners do not share the same instance_to_glb mapping.")
            runners.append(runner)
        assert first_mapping is not None
        num_instances = int(first_mapping.size)
        dataset = PoseCSRDataset(args.dataset_dir, num_instances=num_instances)
        num_glbs = int(first_mapping.max()) + 1 if first_mapping.size else 0
        config = resolve_replay_config(
            trajectory,
            bandwidth_bytes_per_sec=args.bandwidth_bytes_per_sec,
            request_latency_ms=args.request_latency_ms,
            max_concurrent_downloads=args.max_concurrent_downloads,
            max_concurrent_decode_uploads=args.max_concurrent_decode_uploads,
            cache_mode=args.cache_mode,
            initial_cache_glb_ids=initial_ids,
            all_glb_ids=range(num_glbs),
            drain_after_last_pose_ms=args.drain_after_last_pose_ms,
        )
        if any(glb >= num_glbs for glb in config.initial_cache_glb_ids):
            raise ValueError(f"initial cache contains an id >= runtime GLB count {num_glbs}.")

        model_results = []
        for runner in runners:
            model_results.append(
                evaluate_model_trajectory(
                    runner,
                    dataset,
                    trajectory,
                    runtime_meta_path=args.runtime_meta,
                    glb_index_path=args.glb_index,
                    glb_root=args.glb_root,
                    glb_time_index_path=args.glb_time_index,
                    config=config,
                    score_mode=args.score_mode,
                    aggregation=args.glb_aggregation,
                    aggregation_top_k=args.aggregation_top_k,
                    byte_budgets=byte_budgets,
                    time_budgets_ms=time_budgets,
                    include_instance_scores=args.include_instance_scores,
                )
            )
        payload = {
            "schema": REPLAY_SCHEMA,
            "createdBy": "neural_instance_culling/benchmark/evaluate_download_trajectory.py",
            "utilityTeacher": UTILITY_TEACHER,
            "utilityWarning": "No pixel coverage is available in this replay; weak visible-weight utility is reported explicitly.",
            "protocol": {
                "candidateSemantics": "stored_back_camera_frustum_candidate_set_strict",
                "gtCandidateRepair": False,
                "thresholdScanning": False,
                "testEvaluation": False,
                "cacheAndNetwork": "deterministic_offline_event_replay",
            },
            "models": model_results,
        }
        if args.output:
            _write_json(args.output, payload)
        summary = {
            "output": str(args.output) if args.output else None,
            "models": [result["model"] for result in model_results],
            "poseCount": len(trajectory.poses),
            "utilityBasis": UTILITY_TEACHER,
            "replayStatuses": [result["replay"]["status"] for result in model_results],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (FileNotFoundError, IndexError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
