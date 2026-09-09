"""Strict cold-cache GLB streaming simulation primitives.

The simulator models a GLB as an atomic resource.  Bytes are added only when
the complete resource arrives, so a partially downloaded GLB cannot increase
coverage.  Candidate GLB membership is fixed once per pose and is shared by
all ranking methods.

Two decision modes are intentionally separate:

* ``threshold_filtering`` selects a subset with a frozen score threshold.
* ``threshold_free_ranking`` orders the complete candidate set and never
  applies a visibility threshold.

This module is deliberately independent of model training, PVS evaluation,
and the HZB implementation.  HZB contributes only a per-pose visible-first
  sidecar when one is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TARGET_COVERAGES: tuple[float, ...] = (0.95, 0.99, 0.999, 1.0)
BANDWIDTHS_MBPS: tuple[int, ...] = (10, 25, 50, 100)
FIXED_RANDOM_SEEDS: tuple[int, ...] = tuple(20260909 + index for index in range(20))
FIXED_RANDOM_METHODS: tuple[str, ...] = tuple(
    f"fixed_random_{index:02d}" for index in range(len(FIXED_RANDOM_SEEDS))
)
RANKING_METHODS: tuple[str, ...] = (
    "full",
    "aabb",
    "original",
    "distance",
    "projected_area",
    "projected_area_per_byte",
    *FIXED_RANDOM_METHODS,
    "hzb_visible_first",
    "gt_utility_per_byte_oracle",
)
RANKING_METHOD_LABELS: Mapping[str, str] = {
    "full": "Full",
    "aabb": "AABB + ray",
    "original": "Original order",
    "distance": "Distance",
    "projected_area": "Projected area",
    "projected_area_per_byte": "Projected area / byte",
    "hzb_visible_first": "HZB visible-first",
    "gt_utility_per_byte_oracle": "GT utility / byte oracle",
}


class StreamingContractError(ValueError):
    """Raised when a streaming input violates the experiment contract."""


@dataclass(frozen=True)
class GlbAsset:
    """One complete downloadable GLB and its geometry metadata."""

    global_id: int
    byte_size: int
    original_rank: int
    aabb: np.ndarray | None = None


@dataclass(frozen=True)
class PoseRecord:
    """The immutable per-pose input shared by all ranking methods."""

    pose_id: int
    ordinal: int
    candidate_glb_ids: tuple[int, ...]
    gt_glb_ids: tuple[int, ...]
    utility_by_glb: Mapping[int, float] = field(default_factory=dict)
    rank_scores: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    hzb_visible_glb_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        candidate = _validate_id_tuple(self.candidate_glb_ids, "candidate_glb_ids")
        gt = _validate_id_tuple(self.gt_glb_ids, "gt_glb_ids")
        if len(candidate) != len(set(candidate)):
            raise StreamingContractError(
                f"pose {self.pose_id} has duplicate candidate GLB IDs; ranking needs a set"
            )
        if len(gt) != len(set(gt)):
            raise StreamingContractError(f"pose {self.pose_id} has duplicate GT GLB IDs")
        utility = {int(key): float(value) for key, value in self.utility_by_glb.items()}
        if any(not np.isfinite(value) or value < 0.0 for value in utility.values()):
            raise StreamingContractError(f"pose {self.pose_id} has invalid GLB utility")
        object.__setattr__(self, "candidate_glb_ids", candidate)
        object.__setattr__(self, "gt_glb_ids", gt)
        object.__setattr__(self, "utility_by_glb", utility)
        if self.hzb_visible_glb_ids is not None:
            hzb = _validate_id_tuple(self.hzb_visible_glb_ids, "hzb_visible_glb_ids")
            if len(hzb) != len(set(hzb)):
                raise StreamingContractError(f"pose {self.pose_id} has duplicate HZB GLB IDs")
            object.__setattr__(self, "hzb_visible_glb_ids", hzb)


def _validate_id_tuple(values: Iterable[int], label: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        try:
            numeric = int(value)
        except (TypeError, ValueError) as error:
            raise StreamingContractError(f"{label} contains a non-integer ID: {value!r}") from error
        if numeric < 0:
            raise StreamingContractError(f"{label} contains a negative ID: {numeric}")
        result.append(numeric)
    return tuple(result)


def _finite_score(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _asset_bytes(assets: Mapping[int, GlbAsset], glb_id: int) -> int:
    asset = assets.get(int(glb_id))
    if asset is None:
        raise StreamingContractError(f"no byte record for candidate GLB {glb_id}")
    if int(asset.byte_size) <= 0:
        raise StreamingContractError(f"GLB {glb_id} has a non-positive byte size")
    return int(asset.byte_size)


def validate_pose_assets(pose: PoseRecord, assets: Mapping[int, GlbAsset]) -> dict[str, Any]:
    """Validate candidate bytes and return candidate/GT mismatch diagnostics.

    A missing GT resource is retained as a measured coverage ceiling rather
    than silently unioned into the candidate set.  Formal inputs should have
    no missing IDs; keeping the mismatch visible makes an unreachable result
    auditable.
    """

    for glb_id in pose.candidate_glb_ids:
        _asset_bytes(assets, glb_id)
    candidate = set(pose.candidate_glb_ids)
    gt = set(pose.gt_glb_ids)
    missing_gt = tuple(sorted(gt - candidate))
    return {
        "candidateGlbCount": len(candidate),
        "candidateGlbBytes": int(sum(_asset_bytes(assets, glb_id) for glb_id in candidate)),
        "gtGlbCount": len(gt),
        "missingGtGlbIds": list(missing_gt),
        "missingGtGlbCount": len(missing_gt),
        "candidateGtSubset": not missing_gt,
    }


def _utility_vector(pose: PoseRecord) -> tuple[dict[int, float], float]:
    """Return GT utility and its denominator without inventing pixel meaning."""

    gt = set(pose.gt_glb_ids)
    if not gt:
        return {}, 0.0
    if pose.utility_by_glb:
        utility = {
            glb_id: max(0.0, _finite_score(pose.utility_by_glb.get(glb_id), 0.0))
            for glb_id in gt
        }
        return utility, float(sum(utility.values()))
    utility = {glb_id: 1.0 for glb_id in gt}
    return utility, float(len(gt))


def _coverage_from_utility(loaded_utility: float, total_utility: float) -> float:
    if total_utility <= 0.0:
        return 1.0
    return float(np.clip(loaded_utility / total_utility, 0.0, 1.0))


def _target_key(target: float) -> str:
    if np.isclose(target, 0.999):
        return "99.9"
    if np.isclose(target, 1.0):
        return "100"
    return str(int(round(target * 100)))


def _summary_stats(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    finite = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.quantile(finite, 0.5)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _ordered_candidate_ids(
    pose: PoseRecord,
    assets: Mapping[int, GlbAsset],
    method: str,
) -> tuple[int, ...]:
    candidate = tuple(pose.candidate_glb_ids)
    if method == "original":
        return tuple(sorted(candidate, key=lambda glb_id: (assets[glb_id].original_rank, glb_id)))
    if method in {"full", "aabb"} or method in pose.rank_scores:
        scores = pose.rank_scores.get(method)
        if scores is None:
            raise StreamingContractError(f"pose {pose.pose_id} has no score map for {method}")
        return tuple(sorted(candidate, key=lambda glb_id: (-_finite_score(scores.get(glb_id)), glb_id)))
    if method == "distance":
        scores = pose.rank_scores.get(method)
        if scores is None:
            raise StreamingContractError(f"pose {pose.pose_id} has no distance score map")
        return tuple(sorted(candidate, key=lambda glb_id: (-_finite_score(scores.get(glb_id)), glb_id)))
    if method in {"projected_area", "projected_area_per_byte"}:
        scores = pose.rank_scores.get(method)
        if scores is None:
            raise StreamingContractError(f"pose {pose.pose_id} has no projected-area score map")
        return tuple(sorted(candidate, key=lambda glb_id: (-_finite_score(scores.get(glb_id)), glb_id)))
    if method in FIXED_RANDOM_METHODS:
        random_index = FIXED_RANDOM_METHODS.index(method)
        seed = FIXED_RANDOM_SEEDS[random_index]
        base = np.asarray(sorted(candidate), dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(pose.ordinal)]))
        return tuple(int(value) for value in rng.permutation(base).tolist())
    if method == "hzb_visible_first":
        if pose.hzb_visible_glb_ids is None:
            raise StreamingContractError(f"pose {pose.pose_id} has no HZB visible-first sidecar")
        visible = set(pose.hzb_visible_glb_ids)
        unknown = visible - set(candidate)
        if unknown:
            raise StreamingContractError(
                f"pose {pose.pose_id} HZB sidecar contains non-candidate GLBs: {sorted(unknown)[:8]}"
            )
        return tuple(
            sorted(
                candidate,
                key=lambda glb_id: (0 if glb_id in visible else 1, assets[glb_id].original_rank, glb_id),
            )
        )
    if method == "gt_utility_per_byte_oracle":
        utility, _total = _utility_vector(pose)
        return tuple(
            sorted(
                candidate,
                key=lambda glb_id: (
                    -utility.get(glb_id, 0.0) / max(1, _asset_bytes(assets, glb_id)),
                    -utility.get(glb_id, 0.0),
                    _asset_bytes(assets, glb_id),
                    glb_id,
                ),
            )
        )
    raise StreamingContractError(f"unknown streaming ranking method: {method}")


def ordered_glb_ids_for_pose(
    pose: PoseRecord,
    assets: Mapping[int, GlbAsset],
    method: str,
) -> tuple[int, ...]:
    """Return one method's order while preserving the immutable candidate set.

    The real scheduler driver uses this small bridge to replay the exact
    threshold-free order without copying the full per-pose simulation row.
    """

    if method not in RANKING_METHODS:
        raise StreamingContractError(f"unsupported ranking method: {method}")
    validate_pose_assets(pose, assets)
    order = _ordered_candidate_ids(pose, assets, method)
    if set(order) != set(pose.candidate_glb_ids) or len(order) != len(pose.candidate_glb_ids):
        raise StreamingContractError(f"method {method} changed pose {pose.pose_id}'s candidate GLB set")
    return order


def simulate_ranked_pose(
    pose: PoseRecord,
    assets: Mapping[int, GlbAsset],
    method: str,
    *,
    target_coverages: Sequence[float] = TARGET_COVERAGES,
    bandwidths_mbps: Sequence[int] = BANDWIDTHS_MBPS,
) -> dict[str, Any]:
    """Simulate one complete cold-cache ranked download for one pose."""

    if method not in RANKING_METHODS:
        raise StreamingContractError(f"unsupported ranking method: {method}")
    asset_check = validate_pose_assets(pose, assets)
    order = ordered_glb_ids_for_pose(pose, assets, method)

    utility, total_utility = _utility_vector(pose)
    required = set(pose.gt_glb_ids)
    bytes_by_rank = np.asarray([_asset_bytes(assets, glb_id) for glb_id in order], dtype=np.float64)
    utility_by_rank = np.asarray([utility.get(glb_id, 0.0) for glb_id in order], dtype=np.float64)
    required_by_rank = np.asarray([glb_id in required for glb_id in order], dtype=bool)
    cumulative_bytes = np.cumsum(bytes_by_rank)
    cumulative_utility = np.cumsum(utility_by_rank)
    coverage_by_rank = np.asarray(
        [_coverage_from_utility(value, total_utility) for value in cumulative_utility],
        dtype=np.float64,
    )
    coverage_ceiling = _coverage_from_utility(float(cumulative_utility[-1]) if order else 0.0, total_utility)

    bytes_at: dict[str, int | None] = {}
    rank_at: dict[str, int | None] = {}
    time_at: dict[str, dict[str, float | None]] = {}
    for target in target_coverages:
        key = _target_key(float(target))
        if total_utility <= 0.0:
            rank = 0
            bytes_value = 0
        else:
            hits = np.flatnonzero(coverage_by_rank >= float(target) - 1e-12)
            rank = int(hits[0] + 1) if hits.size else None
            bytes_value = int(round(cumulative_bytes[rank - 1])) if rank is not None else None
        bytes_at[key] = bytes_value
        rank_at[key] = rank
        time_at[key] = {
            str(int(bandwidth)): (
                None
                if bytes_value is None
                else float(bytes_value * 8.0 / (float(bandwidth) * 1_000_000.0))
            )
            for bandwidth in bandwidths_mbps
        }

    if not required:
        required_rank: int | None = 0
    elif required.issubset(set(order)):
        required_rank = max(order.index(glb_id) + 1 for glb_id in required)
    else:
        required_rank = None

    curve: list[dict[str, float | int]] = []
    for fraction in np.linspace(0.0, 1.0, 21):
        rank = int(np.ceil(float(fraction) * len(order))) if fraction > 0.0 else 0
        rank = min(len(order), max(0, rank))
        curve.append(
            {
                "rankFraction": float(fraction),
                "rank": rank,
                "bytes": int(round(cumulative_bytes[rank - 1])) if rank > 0 else 0,
                "coverage": (
                    float(coverage_by_rank[rank - 1])
                    if rank > 0
                    else (1.0 if total_utility <= 0.0 else 0.0)
                ),
            }
        )

    rank_99 = rank_at.get(_target_key(0.99))
    prefix_rank_99 = len(order) if rank_99 is None else int(rank_99)
    if prefix_rank_99 > 0:
        useful_bytes = float(bytes_by_rank[:prefix_rank_99][required_by_rank[:prefix_rank_99]].sum())
        waste_before_99 = max(0.0, float(cumulative_bytes[prefix_rank_99 - 1]) - useful_bytes)
    else:
        waste_before_99 = 0.0

    return {
        "poseId": int(pose.pose_id),
        "ordinal": int(pose.ordinal),
        "method": method,
        "decisionMode": "threshold_free_ranking",
        "thresholdApplied": False,
        "candidateGlbCount": int(asset_check["candidateGlbCount"]),
        "candidateGlbBytes": int(asset_check["candidateGlbBytes"]),
        "gtGlbCount": int(asset_check["gtGlbCount"]),
        "missingGtGlbIds": asset_check["missingGtGlbIds"],
        "missingGtGlbBytes": int(
            sum(
                _asset_bytes(assets, glb_id)
                for glb_id in asset_check["missingGtGlbIds"]
                if glb_id in assets
            )
        ),
        "coverageCeiling": float(coverage_ceiling),
        "coverageUtilityTotal": float(total_utility),
        "bytesAtCoverage": bytes_at,
        "rankAtCoverage": rank_at,
        "timeSecondsAtCoverage": time_at,
        "wasteBefore99Bytes": int(round(waste_before_99)),
        "requiredRank": required_rank,
        "requiredRankReachable": required_rank is not None,
        "coverageCurve": curve,
        "coverageUnreachable": {
            _target_key(float(target)): bool(bytes_at[_target_key(float(target))] is None)
            for target in target_coverages
        },
    }


def _filter_coverage(
    gt_glb_ids: Iterable[int],
    selected_glb_ids: Iterable[int],
    utility_by_glb: Mapping[int, float],
) -> tuple[float, float, list[int]]:
    gt = set(int(value) for value in gt_glb_ids)
    selected = set(int(value) for value in selected_glb_ids)
    if not gt:
        return 1.0, 0.0, []
    if utility_by_glb:
        utility = {glb_id: max(0.0, _finite_score(utility_by_glb.get(glb_id), 0.0)) for glb_id in gt}
    else:
        utility = {glb_id: 1.0 for glb_id in gt}
    total = float(sum(utility.values()))
    loaded = float(sum(value for glb_id, value in utility.items() if glb_id in selected))
    missing = sorted(gt - selected)
    return _coverage_from_utility(loaded, total), total, missing


def simulate_threshold_filter_pose(
    pose: PoseRecord,
    assets: Mapping[int, GlbAsset],
    *,
    method: str,
    instance_ids: Sequence[int],
    instance_scores: Sequence[float],
    instance_to_glb: Mapping[int, int],
    threshold: float,
    target_coverages: Sequence[float] = TARGET_COVERAGES,
) -> dict[str, Any]:
    """Evaluate a frozen threshold-selected GLB subset without ranking it."""

    if len(instance_ids) != len(instance_scores):
        raise StreamingContractError("instance score sidecar row count does not match instance IDs")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise StreamingContractError(f"invalid threshold for {method}: {threshold}")
    numeric_instances = [int(value) for value in instance_ids]
    if len(numeric_instances) != len(set(numeric_instances)):
        raise StreamingContractError(f"{method} contains duplicate instance score rows")
    if any(value < 0 for value in numeric_instances):
        raise StreamingContractError(f"{method} contains a negative instance ID")
    missing_mappings = [value for value in numeric_instances if value not in instance_to_glb]
    if missing_mappings:
        raise StreamingContractError(
            f"{method} has no instance-to-GLB mapping for {missing_mappings[:8]}"
        )
    candidate_instances = set(numeric_instances)
    candidate_glbs = set(pose.candidate_glb_ids)
    selected_glbs: set[int] = set()
    for numeric_instance, score in zip(numeric_instances, instance_scores):
        if numeric_instance not in candidate_instances:
            raise StreamingContractError(f"duplicate or invalid instance score row: {numeric_instance}")
        if float(score) >= threshold_value:
            if numeric_instance not in instance_to_glb:
                raise StreamingContractError(f"no instance-to-GLB mapping for instance {numeric_instance}")
            selected_glbs.add(int(instance_to_glb[numeric_instance]))
    if not selected_glbs.issubset(candidate_glbs):
        raise StreamingContractError(f"threshold filter {method} introduced a non-candidate GLB")
    selected_bytes = int(sum(_asset_bytes(assets, glb_id) for glb_id in selected_glbs))
    coverage, total_utility, missing = _filter_coverage(
        pose.gt_glb_ids,
        selected_glbs,
        pose.utility_by_glb,
    )
    return {
        "poseId": int(pose.pose_id),
        "ordinal": int(pose.ordinal),
        "method": method,
        "decisionMode": "threshold_filtering",
        "thresholdApplied": True,
        "threshold": threshold_value,
        "candidateGlbCount": int(len(candidate_glbs)),
        "candidateGlbBytes": int(sum(_asset_bytes(assets, glb_id) for glb_id in candidate_glbs)),
        "predictedGlbCount": int(len(selected_glbs)),
        "predictedGlbBytes": selected_bytes,
        "predictedGlbIds": sorted(selected_glbs),
        "gtGlbCount": int(len(pose.gt_glb_ids)),
        "coverageCeiling": float(coverage),
        "coverageUtilityTotal": float(total_utility),
        "missingRequiredGlbIds": missing,
        "missingRequiredGlbCount": int(len(missing)),
        "missingRequiredGlbBytes": int(
            sum(_asset_bytes(assets, glb_id) for glb_id in missing)
        ),
        "coverageUnreachable": {
            _target_key(float(target)): bool(coverage + 1e-12 < float(target))
            for target in target_coverages
        },
    }


def summarize_ranking_results(
    pose_rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = RANKING_METHODS,
    target_coverages: Sequence[float] = TARGET_COVERAGES,
    bandwidths_mbps: Sequence[int] = BANDWIDTHS_MBPS,
) -> dict[str, Any]:
    """Aggregate per-pose ranking rows and produce compact figure curves."""

    summaries: dict[str, Any] = {}
    curve_fractions = np.linspace(0.0, 1.0, 21)
    for method in methods:
        rows = [row["methods"][method] for row in pose_rows if method in row.get("methods", {})]
        if not rows:
            summaries[method] = {
                "method": method,
                "label": RANKING_METHOD_LABELS.get(method, method),
                "status": "unavailable",
                "decisionMode": "threshold_free_ranking",
                "thresholdApplied": False,
                "reason": "no usable per-pose ranking rows",
            }
            continue
        bytes_stats: dict[str, Any] = {}
        rank_stats: dict[str, Any] = {}
        time_stats: dict[str, Any] = {}
        unreachable: dict[str, Any] = {}
        for target in target_coverages:
            key = _target_key(float(target))
            values = [row.get("bytesAtCoverage", {}).get(key) for row in rows]
            bytes_stats[key] = _summary_stats(values)
            rank_stats[key] = _summary_stats([row.get("rankAtCoverage", {}).get(key) for row in rows])
            unreachable[key] = {
                "count": int(sum(bool(row.get("coverageUnreachable", {}).get(key)) for row in rows)),
                "ratio": float(sum(bool(row.get("coverageUnreachable", {}).get(key)) for row in rows) / max(1, len(rows))),
            }
            time_stats[key] = {
                str(int(bandwidth)): _summary_stats(
                    [row.get("timeSecondsAtCoverage", {}).get(key, {}).get(str(int(bandwidth))) for row in rows]
                )
                for bandwidth in bandwidths_mbps
            }
        curves: list[dict[str, float]] = []
        for curve_index, fraction in enumerate(curve_fractions):
            curve_rows = [row.get("coverageCurve", [])[curve_index] for row in rows]
            curves.append(
                {
                    "rankFraction": float(fraction),
                    "meanCoverage": float(np.mean([float(row.get("coverage", 0.0)) for row in curve_rows])),
                    "meanBytes": float(np.mean([float(row.get("bytes", 0.0)) for row in curve_rows])),
                }
            )
        summaries[method] = {
            "method": method,
            "label": RANKING_METHOD_LABELS.get(method, method),
            "status": "available",
            "decisionMode": "threshold_free_ranking",
            "thresholdApplied": False,
            "poseCount": len(rows),
            "candidateGlbCount": _summary_stats([row.get("candidateGlbCount") for row in rows]),
            "candidateGlbBytes": _summary_stats([row.get("candidateGlbBytes") for row in rows]),
            "gtGlbCount": _summary_stats([row.get("gtGlbCount") for row in rows]),
            "coverageUpperBound": _summary_stats([row.get("coverageCeiling") for row in rows]),
            "missingGtGlbCount": _summary_stats([len(row.get("missingGtGlbIds", [])) for row in rows]),
            "missingGtGlbBytes": _summary_stats([row.get("missingGtGlbBytes") for row in rows]),
            "bytesAtCoverage": bytes_stats,
            "rankAtCoverage": rank_stats,
            "timeSecondsAtCoverage": time_stats,
            "wasteBefore99Bytes": _summary_stats([row.get("wasteBefore99Bytes") for row in rows]),
            "requiredRank": _summary_stats([row.get("requiredRank") for row in rows]),
            "unreachable": unreachable,
            "coverageCurve": curves,
        }
    return {
        "schema": "pvs-glb-streaming-ranking-summary-v1",
        "decisionMode": "threshold_free_ranking",
        "thresholdApplied": False,
        "cacheMode": "strict_cold_cache_per_pose",
        "arrivalSemantics": "GLB bytes and coverage are accumulated only at complete resource arrival",
        "methods": summaries,
        "targetCoverages": [float(value) for value in target_coverages],
        "bandwidthsMbps": [int(value) for value in bandwidths_mbps],
        "fixedRandomSeeds": list(FIXED_RANDOM_SEEDS),
    }


def summarize_filter_results(
    pose_rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    target_coverages: Sequence[float] = TARGET_COVERAGES,
) -> dict[str, Any]:
    """Aggregate threshold filtering without mixing it into ranking metrics."""

    summaries: dict[str, Any] = {}
    for method in methods:
        rows = [row["methods"][method] for row in pose_rows if method in row.get("methods", {})]
        if not rows:
            summaries[method] = {
                "method": method,
                "status": "unavailable",
                "decisionMode": "threshold_filtering",
                "thresholdApplied": True,
            }
            continue
        candidate_bytes = [row.get("candidateGlbBytes", 0) for row in rows]
        predicted_bytes = [row.get("predictedGlbBytes", 0) for row in rows]
        summaries[method] = {
            "method": method,
            "label": RANKING_METHOD_LABELS.get(method, method),
            "status": "available",
            "decisionMode": "threshold_filtering",
            "thresholdApplied": True,
            "threshold": float(rows[0].get("threshold", 0.0)),
            "poseCount": len(rows),
            "candidateGlbCount": _summary_stats([row.get("candidateGlbCount") for row in rows]),
            "candidateGlbBytes": _summary_stats(candidate_bytes),
            "predictedGlbCount": _summary_stats([row.get("predictedGlbCount") for row in rows]),
            "predictedGlbBytes": _summary_stats(predicted_bytes),
            "predictedGlbByteRatio": _summary_stats(
                [float(predicted) / max(1.0, float(candidate)) for predicted, candidate in zip(predicted_bytes, candidate_bytes)]
            ),
            "coverageUpperBound": _summary_stats([row.get("coverageCeiling") for row in rows]),
            "missingRequiredGlbCount": _summary_stats([row.get("missingRequiredGlbCount") for row in rows]),
            "missingRequiredGlbBytes": _summary_stats(
                [row.get("missingRequiredGlbBytes") for row in rows]
            ),
            "unreachable": {
                _target_key(float(target)): {
                    "count": int(sum(bool(row.get("coverageUnreachable", {}).get(_target_key(float(target)))) for row in rows)),
                    "ratio": float(sum(bool(row.get("coverageUnreachable", {}).get(_target_key(float(target)))) for row in rows) / max(1, len(rows))),
                }
                for target in target_coverages
            },
        }
    return {
        "schema": "pvs-glb-streaming-filter-summary-v1",
        "decisionMode": "threshold_filtering",
        "thresholdApplied": True,
        "cacheMode": "strict_cold_cache_per_pose",
        "methods": summaries,
        "targetCoverages": [float(value) for value in target_coverages],
    }


def build_fractional_utility_byte_lower_bound(
    pose: PoseRecord,
    assets: Mapping[int, GlbAsset],
    target: float,
) -> int | None:
    """Return a fractional utility/byte lower bound for a target coverage.

    This is a byte lower bound, not an order and not the greedy oracle.  A
    fractional GLB is allowed only for the bound, never for reported arrival
    coverage.
    """

    utility, total = _utility_vector(pose)
    if total <= 0.0:
        return 0
    candidate = set(pose.candidate_glb_ids)
    items = [
        (utility.get(glb_id, 0.0), _asset_bytes(assets, glb_id))
        for glb_id in candidate
        if utility.get(glb_id, 0.0) > 0.0
    ]
    items.sort(key=lambda item: (-(item[0] / max(1, item[1])), -item[0], item[1]))
    required_utility = float(target) * total
    obtained = 0.0
    bytes_value = 0.0
    for item_utility, item_bytes in items:
        take = min(item_utility, required_utility - obtained)
        if take <= 0.0:
            break
        bytes_value += item_bytes * take / item_utility
        obtained += take
    if obtained + 1e-12 < required_utility:
        return None
    return int(np.ceil(bytes_value))
