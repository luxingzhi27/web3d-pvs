"""Strict cold-cache GLB streaming simulation primitives.

The simulator models a GLB as an atomic resource.  Bytes are added only when
the complete resource arrives, so a partially downloaded GLB cannot increase
coverage.  Candidate GLB membership is fixed once per pose and is shared by
all ranking methods.

Three decision modes are intentionally separate:

* ``threshold_filtering`` selects a subset with a frozen score threshold.
* ``threshold_free_ranking`` orders the complete candidate set and never
  applies a visibility threshold.
* ``scheduler_replay`` describes the separately measured urgent/warm/
  speculative production scheduler; it is not a ranking curve.

This module is deliberately independent of model training, PVS evaluation,
and the HZB implementation.  HZB contributes only a per-pose visible-first
  sidecar when one is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TARGET_COVERAGES: tuple[float, ...] = (0.95, 0.99, 0.999, 1.0)
BANDWIDTHS_MBPS: tuple[int, ...] = (10, 25, 50, 100)
FIXED_RANDOM_SEEDS: tuple[int, ...] = tuple(20260909 + index for index in range(20))
NEURAL_COST_ALPHA_CANDIDATES: tuple[float, ...] = (0.0, 0.5, 1.0)
NEURAL_COST_METHOD = "neural_cost"
NEURAL_PROBABILITY_METHOD = "neural_probability"
NEURAL_COST_MANIFEST_SCHEMA = "pvs-glb-streaming-neural-cost-manifest-v1"
DECISION_MODE_THRESHOLD_FILTERING = "threshold_filtering"
DECISION_MODE_THRESHOLD_FREE_RANKING = "threshold_free_ranking"
DECISION_MODE_SCHEDULER_REPLAY = "scheduler_replay"
# Descriptive aliases used by the standalone simulator and downstream checks.
ALLOWED_COST_ALPHAS = NEURAL_COST_ALPHA_CANDIDATES
COST_ALPHA_MANIFEST_SCHEMA = NEURAL_COST_MANIFEST_SCHEMA
THRESHOLD_FILTERING_MODE = DECISION_MODE_THRESHOLD_FILTERING
THRESHOLD_FREE_RANKING_MODE = DECISION_MODE_THRESHOLD_FREE_RANKING
SCHEDULER_REPLAY_MODE = DECISION_MODE_SCHEDULER_REPLAY
FIXED_RANDOM_METHODS: tuple[str, ...] = tuple(
    f"fixed_random_{index:02d}" for index in range(len(FIXED_RANDOM_SEEDS))
)
NEURAL_COST_METHODS: tuple[str, ...] = tuple(
    f"{NEURAL_COST_METHOD}_alpha_{'0_5' if np.isclose(alpha, 0.5) else int(alpha)}"
    for alpha in NEURAL_COST_ALPHA_CANDIDATES
)
RANKING_METHODS: tuple[str, ...] = (
    "full",
    NEURAL_COST_METHOD,
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
    "full": "Neural p_g",
    NEURAL_PROBABILITY_METHOD: "Neural p_g",
    NEURAL_COST_METHOD: "Neural p_g / bytes^alpha",
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


def validate_neural_cost_alpha(value: Any, label: str = "alpha") -> float:
    """Validate the only three cost exponents registered by the paper protocol."""

    if isinstance(value, bool):
        raise StreamingContractError(
            f"{label} must be one of {NEURAL_COST_ALPHA_CANDIDATES}"
        )
    try:
        alpha = float(value)
    except (TypeError, ValueError) as error:
        raise StreamingContractError(
            f"{label} must be one of {NEURAL_COST_ALPHA_CANDIDATES}"
        ) from error
    if not np.isfinite(alpha) or not any(
        np.isclose(alpha, candidate, rtol=0.0, atol=1e-12)
        for candidate in NEURAL_COST_ALPHA_CANDIDATES
    ):
        raise StreamingContractError(
            f"{label} must be one of {NEURAL_COST_ALPHA_CANDIDATES}, got {value!r}"
        )
    return float(
        next(
            candidate
            for candidate in NEURAL_COST_ALPHA_CANDIDATES
            if np.isclose(alpha, candidate, rtol=0.0, atol=1e-12)
        )
    )


def neural_cost_method_for_alpha(alpha: Any) -> str:
    """Return the private evaluation name for one registered alpha candidate."""

    value = validate_neural_cost_alpha(alpha)
    suffix = "0_5" if np.isclose(value, 0.5, rtol=0.0, atol=1e-12) else str(int(round(value)))
    return f"{NEURAL_COST_METHOD}_alpha_{suffix}"


def _is_supported_ranking_method(method: str) -> bool:
    return method in set(RANKING_METHODS) | {NEURAL_PROBABILITY_METHOD} | {
        neural_cost_method_for_alpha(alpha) for alpha in NEURAL_COST_ALPHA_CANDIDATES
    }


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
    neural_cost_alpha: float | None = None
    coverage_source: str = "gt_glb_presence"
    coverage_semantics: str = "equal utility for each GT GLB present in the pose"
    coverage_unit: str = "binary_glb_presence"

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
        if self.neural_cost_alpha is not None:
            object.__setattr__(
                self,
                "neural_cost_alpha",
                validate_neural_cost_alpha(self.neural_cost_alpha),
            )
        if not isinstance(self.coverage_source, str) or not self.coverage_source:
            raise StreamingContractError("pose coverage_source must be a non-empty string")
        if not isinstance(self.coverage_semantics, str) or not self.coverage_semantics:
            raise StreamingContractError("pose coverage_semantics must be a non-empty string")
        if not isinstance(self.coverage_unit, str) or not self.coverage_unit:
            raise StreamingContractError("pose coverage_unit must be a non-empty string")
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


def coverage_metadata(
    source: str,
    *,
    visible_weight_semantics: str | None = None,
    source_sampler: str | None = None,
) -> dict[str, str]:
    """Describe the denominator and unit of a streaming coverage value."""

    normalized = str(source)
    if normalized in {
        "visible_weights",
        "visible_weight_coverage",
        "visible_weights_utility_not_pixel_coverage",
    }:
        semantics = str(visible_weight_semantics or "").strip()
        sampler = str(source_sampler or "").strip().lower()
        if sampler == "three_color_id" and "screen coverage" in semantics.lower():
            return {
                "source": "visible_weight_coverage",
                "metric": "visible_weight_coverage",
                "unit": "max_pooled_color_id_screen_coverage_ppm",
                "semantics": (
                    "sum of dataset-provided Three.js Color-ID screen-coverage weights for "
                    "GT-visible instances grouped by GLB; weights are max-pooled over view-cell "
                    "subposes and therefore are not a union pixel count"
                ),
            }
        return {
            "source": "visible_weight_coverage",
            "metric": "visible_weight_coverage",
            "unit": "dataset_visible_weight",
            "semantics": (
                f"sum of dataset visible weights for GT-visible instances grouped by GLB; "
                f"dataset semantics: {semantics or 'unspecified nonnegative visible weight'}"
            ),
        }
    if normalized in {"reference_frontmost_pixels", "reference-frontmost-pixel-histogram-v1"}:
        return {
            "source": "reference-frontmost-pixel-histogram-v1",
            "metric": "reference_frontmost_pixel_utility",
            "unit": "frontmost_Color_ID_pixel_count",
            "semantics": (
                "front-most Color-ID pixels aggregated by GLB as a visible-surface utility proxy; "
                "not complete hidden-surface visibility"
            ),
        }
    return {
        "source": "gt_glb_presence_coverage",
        "metric": "gt_glb_presence_coverage",
        "unit": "binary_glb_presence",
        "semantics": "equal utility for each GT GLB present in the pose",
    }


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


def _neural_probability_map(pose: PoseRecord) -> Mapping[int, float] | None:
    """Return the per-GLB ``p_g`` map produced by max instance probability."""

    for name in (NEURAL_PROBABILITY_METHOD, "full", "neural"):
        values = pose.rank_scores.get(name)
        if isinstance(values, Mapping):
            return values
    return None


def _validate_probability(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise StreamingContractError(f"{label} is not a probability in [0, 1]") from error
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise StreamingContractError(f"{label} is not a probability in [0, 1]")
    return numeric


def _neural_cost_score_map(
    pose: PoseRecord,
    assets: Mapping[int, GlbAsset],
    method: str,
) -> dict[int, float]:
    """Build ``p_g / bytes_g^alpha`` without allowing test-time alpha selection."""

    if method == NEURAL_COST_METHOD:
        if pose.neural_cost_alpha is None:
            raise StreamingContractError(
                f"pose {pose.pose_id} has no frozen alpha for {NEURAL_COST_METHOD}"
            )
        alpha = validate_neural_cost_alpha(pose.neural_cost_alpha)
    else:
        prefix = f"{NEURAL_COST_METHOD}_alpha_"
        if not method.startswith(prefix):
            raise StreamingContractError(f"unsupported neural ranking method: {method}")
        suffix = method[len(prefix) :]
        alpha_by_suffix = {neural_cost_method_for_alpha(value).split(prefix, 1)[1]: value for value in NEURAL_COST_ALPHA_CANDIDATES}
        if suffix not in alpha_by_suffix:
            raise StreamingContractError(f"unsupported neural cost alpha method: {method}")
        alpha = validate_neural_cost_alpha(alpha_by_suffix[suffix])
    probabilities = _neural_probability_map(pose)
    if probabilities is None:
        raise StreamingContractError(f"pose {pose.pose_id} has no neural instance probability map")
    result: dict[int, float] = {}
    for glb_id in pose.candidate_glb_ids:
        probability = _validate_probability(
            probabilities.get(glb_id, 0.0),
            f"pose {pose.pose_id} GLB {glb_id} p_g",
        )
        byte_size = _asset_bytes(assets, glb_id)
        result[int(glb_id)] = float(probability / (float(byte_size) ** alpha))
    return result


def ranking_input_metadata(
    pose: PoseRecord,
    method: str,
    *,
    score_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the score input so summaries cannot hide the aggregation rule."""

    if method in {"full", NEURAL_PROBABILITY_METHOD, "neural"}:
        metadata: dict[str, Any] = {
            "kind": "neural_glb_probability",
            "formula": "p_g=max_i(p_i)",
            "instanceProbabilityAggregation": "max",
            "costExponent": None,
            "usesProjectedAabbArea": False,
            "bytesSource": None,
            "parameterSelectionSplit": None,
            "continuousScore": True,
        }
    elif method == NEURAL_COST_METHOD or method.startswith(f"{NEURAL_COST_METHOD}_alpha_"):
        if method == NEURAL_COST_METHOD:
            alpha = pose.neural_cost_alpha
            if alpha is None:
                raise StreamingContractError(
                    f"pose {pose.pose_id} has no frozen alpha for {NEURAL_COST_METHOD}"
                )
        else:
            alpha_by_method = {
                neural_cost_method_for_alpha(value): value for value in NEURAL_COST_ALPHA_CANDIDATES
            }
            alpha = alpha_by_method.get(method)
            if alpha is None:
                raise StreamingContractError(f"unsupported neural cost alpha method: {method}")
        metadata = {
            "kind": "neural_glb_cost_probability",
            "formula": "p_g / bytes_g^alpha",
            "instanceProbabilityAggregation": "max",
            "costExponent": validate_neural_cost_alpha(alpha),
            "usesProjectedAabbArea": False,
            "bytesSource": "complete GLB file bytes",
            "parameterSelectionSplit": "validation" if method == NEURAL_COST_METHOD else None,
            "frozenForTest": method == NEURAL_COST_METHOD,
            "continuousScore": True,
        }
    elif method == "distance":
        metadata = {
            "kind": "candidate_instance_aabb_distance",
            "formula": "D_g=min_i(D_i)",
            "instanceAabbAggregation": "min_distance",
            "input": "each candidate instance AABB",
            "usesMergedGlbAabb": False,
            "continuousScore": True,
        }
    elif method in {"projected_area", "projected_area_per_byte"}:
        metadata = {
            "kind": "candidate_instance_aabb_projected_area",
            "formula": "A_g=max_i(A_i)" if method == "projected_area" else "A_g / bytes_g",
            "instanceAabbAggregation": "max_projected_area",
            "input": "each candidate instance AABB eight-corner projection",
            "usesMergedGlbAabb": False,
            "bytesSource": "complete GLB file bytes" if method == "projected_area_per_byte" else None,
            "continuousScore": True,
        }
    elif method == "aabb":
        metadata = {
            "kind": "formal_aabb_score_sidecar",
            "instanceScore": "aabb model score",
            "glbAggregation": "max_i_in_glb(aabb_instance_score)",
            "continuousScore": True,
        }
    elif method == "hzb_visible_first":
        metadata = {
            "kind": "formal_region66_visible_set",
            "continuousScore": False,
        }
    else:
        metadata = {"kind": method, "continuousScore": True}
    if score_source:
        metadata["scoreSource"] = dict(score_source)
    return metadata


def _ordered_candidate_ids(
    pose: PoseRecord,
    assets: Mapping[int, GlbAsset],
    method: str,
) -> tuple[int, ...]:
    candidate = tuple(pose.candidate_glb_ids)
    if method == "original":
        return tuple(sorted(candidate, key=lambda glb_id: (assets[glb_id].original_rank, glb_id)))
    if method in {NEURAL_COST_METHOD} or method.startswith(f"{NEURAL_COST_METHOD}_alpha_"):
        scores = _neural_cost_score_map(pose, assets, method)
        return tuple(sorted(candidate, key=lambda glb_id: (-scores[glb_id], glb_id)))
    if method in {NEURAL_PROBABILITY_METHOD, "neural"}:
        scores = _neural_probability_map(pose)
        if scores is None:
            raise StreamingContractError(f"pose {pose.pose_id} has no neural instance probability map")
        return tuple(
            sorted(
                candidate,
                key=lambda glb_id: (-_validate_probability(scores.get(glb_id, 0.0), f"pose {pose.pose_id} GLB {glb_id} p_g"), glb_id),
            )
        )
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

    if not _is_supported_ranking_method(method):
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

    if not _is_supported_ranking_method(method):
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
    coverage_info = coverage_metadata(pose.coverage_source)
    coverage_info["semantics"] = pose.coverage_semantics
    coverage_info["unit"] = pose.coverage_unit
    visible_weight_source = coverage_info["metric"] == "visible_weight_coverage"
    for fraction in np.linspace(0.0, 1.0, 21):
        rank = int(np.ceil(float(fraction) * len(order))) if fraction > 0.0 else 0
        rank = min(len(order), max(0, rank))
        visible_weight_coverage = (
            float(coverage_by_rank[rank - 1])
            if rank > 0
            else (1.0 if total_utility <= 0.0 else 0.0)
        )
        curve_point: dict[str, float | int] = {
            "rankFraction": float(fraction),
            "rank": rank,
            "bytes": int(round(cumulative_bytes[rank - 1])) if rank > 0 else 0,
            "coverage": visible_weight_coverage,
        }
        if visible_weight_source:
            curve_point["visibleWeightCoverage"] = visible_weight_coverage
        curve.append(curve_point)

    rank_99 = rank_at.get(_target_key(0.99))
    prefix_rank_99 = len(order) if rank_99 is None else int(rank_99)
    if prefix_rank_99 > 0:
        useful_bytes = float(bytes_by_rank[:prefix_rank_99][required_by_rank[:prefix_rank_99]].sum())
        waste_before_99 = max(0.0, float(cumulative_bytes[prefix_rank_99 - 1]) - useful_bytes)
    else:
        waste_before_99 = 0.0

    ranking_input = ranking_input_metadata(pose, method)
    result: dict[str, Any] = {
        "poseId": int(pose.pose_id),
        "ordinal": int(pose.ordinal),
        "method": method,
        "decisionMode": DECISION_MODE_THRESHOLD_FREE_RANKING,
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
        "rankingInput": ranking_input,
        "coverageSource": coverage_info["source"],
        "coverageMetric": coverage_info["metric"],
        "coverageUnit": coverage_info["unit"],
        "coverageSemantics": coverage_info["semantics"],
        "coverageDenominator": float(total_utility),
    }
    if visible_weight_source:
        result["visibleWeightCoverageCeiling"] = float(coverage_ceiling)
        result["visibleWeightUtilityTotal"] = float(total_utility)
        result["bytesAtVisibleWeightCoverage"] = dict(bytes_at)
        result["rankAtVisibleWeightCoverage"] = dict(rank_at)
        result["timeSecondsAtVisibleWeightCoverage"] = {
            key: dict(value) for key, value in time_at.items()
        }
        result["visibleWeightCoverageCurve"] = [
            {
                "rankFraction": point["rankFraction"],
                "rank": point["rank"],
                "bytes": point["bytes"],
                "visibleWeightCoverage": point["visibleWeightCoverage"],
            }
            for point in curve
        ]
        result["visibleWeightCoverageUnreachable"] = {
            _target_key(float(target)): bool(bytes_at[_target_key(float(target))] is None)
            for target in target_coverages
        }
    return result


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
        if not np.isfinite(float(score)):
            raise StreamingContractError(f"{method} contains a non-finite instance score")
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
    coverage_info = coverage_metadata(pose.coverage_source)
    coverage_info["semantics"] = pose.coverage_semantics
    coverage_info["unit"] = pose.coverage_unit
    result: dict[str, Any] = {
        "poseId": int(pose.pose_id),
        "ordinal": int(pose.ordinal),
        "method": method,
        "decisionMode": DECISION_MODE_THRESHOLD_FILTERING,
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
        "coverageSource": coverage_info["source"],
        "coverageMetric": coverage_info["metric"],
        "coverageUnit": coverage_info["unit"],
        "coverageSemantics": coverage_info["semantics"],
        "coverageDenominator": float(total_utility),
    }
    if coverage_info["metric"] == "visible_weight_coverage":
        result["visibleWeightCoverageCeiling"] = float(coverage)
        result["visibleWeightUtilityTotal"] = float(total_utility)
        result["visibleWeightCoverageUnreachable"] = {
            _target_key(float(target)): bool(coverage + 1e-12 < float(target))
            for target in target_coverages
        }
    return result


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
                "decisionMode": DECISION_MODE_THRESHOLD_FREE_RANKING,
                "thresholdApplied": False,
                "reason": "no usable per-pose ranking rows",
            }
            continue
        coverage_source = str(rows[0].get("coverageSource", "gt_glb_presence"))
        coverage_info = coverage_metadata(coverage_source)
        coverage_info["semantics"] = str(
            rows[0].get("coverageSemantics", coverage_info["semantics"])
        )
        coverage_info["unit"] = str(rows[0].get("coverageUnit", coverage_info["unit"]))
        visible_weight_source = coverage_info["metric"] == "visible_weight_coverage"
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
            curve_point: dict[str, float] = {
                "rankFraction": float(fraction),
                "meanCoverage": float(
                    np.mean([float(row.get("coverage", 0.0)) for row in curve_rows])
                ),
                "meanBytes": float(np.mean([float(row.get("bytes", 0.0)) for row in curve_rows])),
            }
            if visible_weight_source:
                curve_point["meanVisibleWeightCoverage"] = float(
                    np.mean(
                        [
                            float(row.get("visibleWeightCoverage", row.get("coverage", 0.0)))
                            for row in curve_rows
                        ]
                    )
                )
            curves.append(curve_point)
        method_summary: dict[str, Any] = {
            "method": method,
            "label": RANKING_METHOD_LABELS.get(method, method),
            "status": "available",
            "decisionMode": DECISION_MODE_THRESHOLD_FREE_RANKING,
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
            "rankingInput": rows[0].get("rankingInput"),
            "coverageSource": coverage_info["source"],
            "coverageMetric": coverage_info["metric"],
            "coverageUnit": coverage_info["unit"],
            "coverageSemantics": coverage_info["semantics"],
        }
        if visible_weight_source:
            method_summary["visibleWeightCoverageUpperBound"] = _summary_stats(
                [row.get("visibleWeightCoverageCeiling", row.get("coverageCeiling")) for row in rows]
            )
            method_summary["visibleWeightCoverageCurve"] = curves
        summaries[method] = method_summary
    result: dict[str, Any] = {
        "schema": "pvs-glb-streaming-ranking-summary-v1",
        "decisionMode": DECISION_MODE_THRESHOLD_FREE_RANKING,
        "thresholdApplied": False,
        "cacheMode": "strict_cold_cache_per_pose",
        "arrivalSemantics": "GLB bytes and coverage are accumulated only at complete resource arrival",
        "methods": summaries,
        "targetCoverages": [float(value) for value in target_coverages],
        "bandwidthsMbps": [int(value) for value in bandwidths_mbps],
        "fixedRandomSeeds": list(FIXED_RANDOM_SEEDS),
        "coverageSource": (
            next(
                (
                    str(row.get("coverageSource"))
                    for pose_row in pose_rows
                    for row in pose_row.get("methods", {}).values()
                    if row.get("coverageSource") is not None
                ),
                "gt_glb_presence",
            )
        ),
    }
    summary_coverage = coverage_metadata(result["coverageSource"])
    result.update(
        {
            "coverageMetric": summary_coverage["metric"],
            "coverageUnit": summary_coverage["unit"],
            "coverageSemantics": summary_coverage["semantics"],
        }
    )
    return result


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
                "decisionMode": DECISION_MODE_THRESHOLD_FILTERING,
                "thresholdApplied": True,
            }
            continue
        candidate_bytes = [row.get("candidateGlbBytes", 0) for row in rows]
        predicted_bytes = [row.get("predictedGlbBytes", 0) for row in rows]
        coverage_source = str(rows[0].get("coverageSource", "gt_glb_presence"))
        coverage_info = coverage_metadata(coverage_source)
        coverage_info["semantics"] = str(
            rows[0].get("coverageSemantics", coverage_info["semantics"])
        )
        coverage_info["unit"] = str(rows[0].get("coverageUnit", coverage_info["unit"]))
        visible_weight_source = coverage_info["metric"] == "visible_weight_coverage"
        method_summary: dict[str, Any] = {
            "method": method,
            "label": RANKING_METHOD_LABELS.get(method, method),
            "status": "available",
            "decisionMode": DECISION_MODE_THRESHOLD_FILTERING,
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
            "coverageSource": coverage_info["source"],
            "coverageMetric": coverage_info["metric"],
            "coverageUnit": coverage_info["unit"],
            "coverageSemantics": coverage_info["semantics"],
        }
        if visible_weight_source:
            method_summary["visibleWeightCoverageUpperBound"] = _summary_stats(
                [row.get("visibleWeightCoverageCeiling", row.get("coverageCeiling")) for row in rows]
            )
            method_summary["visibleWeightCoverageUnreachable"] = {
                _target_key(float(target)): {
                    "count": int(
                        sum(
                            bool(
                                row.get("visibleWeightCoverageUnreachable", {}).get(
                                    _target_key(float(target))
                                )
                            )
                            for row in rows
                        )
                    ),
                    "ratio": float(
                        sum(
                            bool(
                                row.get("visibleWeightCoverageUnreachable", {}).get(
                                    _target_key(float(target))
                                )
                            )
                            for row in rows
                        )
                        / max(1, len(rows))
                    ),
                }
                for target in target_coverages
            }
        summaries[method] = method_summary
    result: dict[str, Any] = {
        "schema": "pvs-glb-streaming-filter-summary-v1",
        "decisionMode": DECISION_MODE_THRESHOLD_FILTERING,
        "thresholdApplied": True,
        "cacheMode": "strict_cold_cache_per_pose",
        "methods": summaries,
        "targetCoverages": [float(value) for value in target_coverages],
    }
    result["coverageSource"] = next(
        (
            str(row.get("coverageSource"))
            for pose_row in pose_rows
            for row in pose_row.get("methods", {}).values()
            if row.get("coverageSource") is not None
        ),
        "gt_glb_presence",
    )
    summary_coverage = coverage_metadata(result["coverageSource"])
    result.update(
        {
            "coverageMetric": summary_coverage["metric"],
            "coverageUnit": summary_coverage["unit"],
            "coverageSemantics": summary_coverage["semantics"],
        }
    )
    return result


def apply_neural_cost_alpha(
    poses: Sequence[PoseRecord],
    alpha: Any,
) -> list[PoseRecord]:
    """Attach one already-selected alpha to every pose for a frozen replay."""

    selected = validate_neural_cost_alpha(alpha)
    return [replace_pose_neural_cost_alpha(pose, selected) for pose in poses]


def replace_pose_neural_cost_alpha(pose: PoseRecord, alpha: Any) -> PoseRecord:
    """Return a pose carrying the immutable alpha used by ``neural_cost``."""

    selected = validate_neural_cost_alpha(alpha)
    return PoseRecord(
        pose_id=pose.pose_id,
        ordinal=pose.ordinal,
        candidate_glb_ids=pose.candidate_glb_ids,
        gt_glb_ids=pose.gt_glb_ids,
        utility_by_glb=pose.utility_by_glb,
        rank_scores=pose.rank_scores,
        hzb_visible_glb_ids=pose.hzb_visible_glb_ids,
        neural_cost_alpha=selected,
        coverage_source=pose.coverage_source,
        coverage_semantics=pose.coverage_semantics,
        coverage_unit=pose.coverage_unit,
    )


def select_neural_cost_alpha(
    poses: Sequence[PoseRecord],
    assets: Mapping[int, GlbAsset],
    *,
    selection_split: str = "validation",
    target_coverage: float = 0.99,
) -> dict[str, Any]:
    """Select alpha on validation only and return a test-free frozen manifest.

    The selection objective is the mean complete-GLB bytes needed to reach the
    declared visible-weight coverage target.  Unreachable poses contribute an
    infinite objective, then lower target bytes and lower alpha break ties.
    """

    if selection_split != "validation":
        raise StreamingContractError(
            "neural cost alpha selection is allowed only on the validation split"
        )
    if not poses:
        raise StreamingContractError("neural cost alpha selection requires validation poses")
    coverage_sources = {pose.coverage_source for pose in poses}
    coverage_units = {pose.coverage_unit for pose in poses}
    coverage_semantics = {pose.coverage_semantics for pose in poses}
    if len(coverage_sources) != 1 or len(coverage_units) != 1 or len(coverage_semantics) != 1:
        raise StreamingContractError(
            "neural cost alpha selection requires one consistent coverage definition"
        )
    target = float(target_coverage)
    if not np.isfinite(target) or not 0.0 < target <= 1.0:
        raise StreamingContractError(f"invalid neural cost alpha selection target: {target_coverage}")

    candidates: list[dict[str, Any]] = []
    for alpha in NEURAL_COST_ALPHA_CANDIDATES:
        prepared = apply_neural_cost_alpha(poses, alpha)
        rows = [simulate_ranked_pose(pose, assets, NEURAL_COST_METHOD) for pose in prepared]
        key = _target_key(target)
        bytes_values = [row.get("bytesAtCoverage", {}).get(key) for row in rows]
        finite_values = [float(value) for value in bytes_values if value is not None]
        unreachable = len(bytes_values) - len(finite_values)
        objective = float(np.mean(finite_values)) if not unreachable else float("inf")
        candidates.append(
            {
                "alpha": float(alpha),
                "targetCoverage": target,
                "targetKey": key,
                "poseCount": len(rows),
                "reachablePoseCount": len(finite_values),
                "unreachablePoseCount": unreachable,
                "meanBytesAtTarget": None if not np.isfinite(objective) else objective,
                "meanBytesAt95": float(
                    np.mean(
                        [
                            float(value)
                            for value in (
                                row.get("bytesAtCoverage", {}).get(_target_key(0.95))
                                for row in rows
                            )
                            if value is not None
                        ]
                    )
                )
                if any(
                    row.get("bytesAtCoverage", {}).get(_target_key(0.95)) is not None
                    for row in rows
                )
                else None,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            float("inf") if row["meanBytesAtTarget"] is None else float(row["meanBytesAtTarget"]),
            float("inf") if row["meanBytesAt95"] is None else float(row["meanBytesAt95"]),
            float(row["alpha"]),
        ),
    )
    return {
        "schema": NEURAL_COST_MANIFEST_SCHEMA,
        "frozen": True,
        "eligibleForTest": True,
        "selectionSplit": "validation",
        "testRead": False,
        "candidateAlphas": [float(value) for value in NEURAL_COST_ALPHA_CANDIDATES],
        "selectedAlpha": float(selected["alpha"]),
        "selectedMethod": neural_cost_method_for_alpha(selected["alpha"]),
        "formula": "p_g / bytes_g^alpha",
        "instanceProbabilityAggregation": "max",
        "bytesSource": "complete GLB file bytes",
        "selectionTargetCoverage": target,
        "selectionMetric": "mean_complete_glb_bytes_at_target_coverage",
        "selectionDirection": "minimize",
        "coverageSource": next(iter(coverage_sources)),
        "coverageUnit": next(iter(coverage_units)),
        "coverageSemantics": next(iter(coverage_semantics)),
        "poseCount": len(poses),
        "selectedCandidate": selected,
        "candidates": candidates,
    }


def simulate_scheduler_replay_pose(
    pose: PoseRecord,
    assets: Mapping[int, GlbAsset],
    *,
    method: str,
    tiers: Mapping[str, Sequence[int]],
    threshold_applied: bool = False,
    target_coverages: Sequence[float] = TARGET_COVERAGES,
    bandwidths_mbps: Sequence[int] = BANDWIDTHS_MBPS,
) -> dict[str, Any]:
    """Replay one explicit scheduler tier partition as a separate mode.

    The tier concatenation is the production arrival order.  It must still be
    a permutation of the complete candidate GLB set; no scheduler replay is
    allowed to silently repair or truncate the ranking input.
    """

    names = ("urgent", "warm", "speculative")
    if not isinstance(tiers, Mapping) or any(name not in tiers for name in names):
        raise StreamingContractError(
            "scheduler replay must declare urgent, warm, and speculative tiers"
        )
    tier_values: dict[str, tuple[int, ...]] = {}
    flattened: list[int] = []
    for name in names:
        values = _validate_id_tuple(tiers[name], f"scheduler {name} tier")
        if len(values) != len(set(values)):
            raise StreamingContractError(f"scheduler {name} tier contains duplicate GLB IDs")
        tier_values[name] = values
        flattened.extend(values)
    candidate = set(pose.candidate_glb_ids)
    if len(flattened) != len(set(flattened)) or set(flattened) != candidate:
        raise StreamingContractError(
            f"scheduler replay for pose {pose.pose_id} must partition the complete candidate GLB set"
        )
    if any(glb_id not in assets for glb_id in flattened):
        raise StreamingContractError("scheduler replay references an unknown GLB")

    permutation_assets = {
        int(glb_id): GlbAsset(
            global_id=assets[int(glb_id)].global_id,
            byte_size=assets[int(glb_id)].byte_size,
            original_rank=index,
            aabb=assets[int(glb_id)].aabb,
        )
        for index, glb_id in enumerate(flattened)
    }
    result = simulate_ranked_pose(
        pose,
        permutation_assets,
        "original",
        target_coverages=target_coverages,
        bandwidths_mbps=bandwidths_mbps,
    )
    result["method"] = str(method)
    result["decisionMode"] = DECISION_MODE_SCHEDULER_REPLAY
    result["thresholdApplied"] = bool(threshold_applied)
    result["schedulerReplay"] = {
        "scheduler": "GlbResourceScheduler",
        "tiers": {name: list(tier_values[name]) for name in names},
        "tierOrder": list(names),
        "arrivalOrder": list(flattened),
        "thresholdApplied": bool(threshold_applied),
    }
    result["rankingInput"] = {
        "kind": "scheduler_tier_replay",
        "decisionMode": DECISION_MODE_SCHEDULER_REPLAY,
        "continuousScore": False,
        "thresholdApplied": bool(threshold_applied),
    }
    return result


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
