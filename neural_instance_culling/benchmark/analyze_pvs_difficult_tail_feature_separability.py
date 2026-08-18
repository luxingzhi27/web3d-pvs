#!/usr/bin/env python3
"""Measure input-feature separability in the difficult PVS score tails.

The evaluator's --persist-scores output contains scores, labels, and visible
weights, but it deliberately does not persist the per-candidate query inputs.
This tool therefore has two input modes:

* a small, self-contained capture whose perPose rows also contain a
  featureFamilies mapping; or
* the normal evaluator capture plus --checkpoint, --dataset-dir,
  --runtime-meta, and --initial-geo-features.  In the latter mode the fixed v4
  runtime table and the registered view-cell ray/moment query are replayed to
  recover the features.

Only calibration, train, and validation are accepted.  The low-score-positive
and high-score-negative cutoffs are quantiles fitted on calibration scores and
then reused unchanged on every other split.  Probe weights may be fitted on
train or calibration, while the probe decision threshold is always selected
on calibration.  Validation is used only for the reported ranking metrics.

The four feature families are:

center
    The nine-dimensional center ray-space query.  It describes the instance
    ray, distance, screen location, and angular extent at the view-cell center.
mean_std
    The sixteen-frequency Fourier moment query, in its existing per-frequency
    mean-sine, mean-cosine, standard-sine, standard-cosine order.
boundary
    The existing learned eight-dimensional boundary spectral summary emitted
    by the v4 query head.
region_extrema
    A directly computable candidate feature: for each center feature, the disk
    bound center +/- row_norm(disk_axes) and its span.  It does not add an
    online model or subpose expansion.

The probe is a ridge-regularized linear regression used as a ranking probe.
When fitted on train, it also emits the score distribution of the exact rows
used by the fit and finite-sample train-only rescue/suppression certificates.
Optional nonzero sample-weight power emphasizes visible mass in the low-score
positive tail; power zero preserves the original unweighted ridge semantics.
ROC-AUC and average precision are computed without optional sklearn/scipy
dependencies.  This script is diagnostic only and never changes a checkpoint,
threshold, model, training file, or frontend asset.

The optional --subpose-sidecar supplies visible_hit_counts.bin and
subpose_offsets.bin for the rare-positive diagnostic.  Its hit rate is h/H and
is supervision only: it never rewrites candidates or the main GT.  Without the
sidecar, rare-positive output is explicitly unavailable rather than filled with
zero hit rates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

try:
    import torch
except ImportError:  # pragma: no cover - statistics-only users can still import the module
    torch = None  # type: ignore[assignment]


SCHEMA = "pvs-difficult-tail-feature-separability-v1"
FEATURE_FAMILIES = ("center", "mean_std", "boundary", "region_extrema")
GROUP_LOW_POSITIVE = "low_score_positive"
GROUP_HIGH_NEGATIVE = "high_score_negative"
RARE_POSITIVE_UNAVAILABLE_REASON = (
    "visible_hit_counts/subpose_offsets sidecar was not provided"
)
DEFAULT_RARE_POSITIVE_HIT_RATE_MAX = 0.05
QUANTILE_LEVELS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
SUPPRESSION_RISK_LEVELS = (0.001, 0.005, 0.01)
RESCUE_RISK_LEVELS = (0.01, 0.05)
DEFAULT_PROBE_SAMPLE_WEIGHT_POWER = 0.0
DEFAULT_PROBE_SAMPLE_WEIGHT_EPSILON = 1e-6
CENTER_FEATURE_NAMES = (
    "ray_x",
    "ray_y",
    "ray_z",
    "log_distance",
    "dot_forward",
    "screen_u",
    "screen_v",
    "angular_horizontal",
    "angular_vertical",
)


@dataclass
class ScoreRow:
    """One persisted evaluator row, normalized to candidate-aligned arrays."""

    pose_index: int
    candidate_ids: np.ndarray
    scores: np.ndarray | None
    targets: np.ndarray | None
    visible_weights: np.ndarray | None
    feature_families: Mapping[str, Any] | None = None


@dataclass
class FeatureExamples:
    """Candidate-aligned arrays used by the pure statistics path."""

    split: str
    scores: np.ndarray
    targets: np.ndarray
    features: dict[str, np.ndarray]
    visible_weights: np.ndarray | None = None
    pose_indices: np.ndarray | None = None
    instance_ids: np.ndarray | None = None
    # Candidate-aligned h/H values. NaN means undefined, never an imputed zero.
    hit_rates: np.ndarray | None = None
    subpose_supervision_available: bool = False


def _finite_array(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _optional_array(value: Any, *, name: str, dtype: Any) -> np.ndarray | None:
    if value is None:
        return None
    return _finite_array(value, name=name, dtype=dtype)


def _optional_hit_rates(value: Any, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if bool(np.isinf(result).any()) or bool(np.any((result < 0.0) | (result > 1.0))):
        raise ValueError(f"{name} must contain values in [0, 1] or NaN for undefined rates")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"capture must contain a JSON object: {path}")
    return dict(value)


def load_capture(path: Path, expected_split: str) -> dict[str, Any]:
    """Load one evaluator capture and reject test provenance immediately."""

    if expected_split == "test":
        raise ValueError("test capture analysis is forbidden")
    payload = _load_json(path)
    if payload.get("testRead") is not False:
        raise ValueError(f"capture must explicitly declare testRead=false: {path}")
    if payload.get("split") != expected_split:
        raise ValueError(
            f"capture {path} has split={payload.get('split')!r}, expected {expected_split!r}"
        )
    rows = payload.get("perPose")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"capture has no perPose rows: {path}")
    normalized: list[ScoreRow] = []
    seen: set[int] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"perPose[{index}] is not an object")
        try:
            pose_index = int(raw["poseIndex"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"perPose[{index}] has no integer poseIndex") from exc
        if pose_index in seen:
            raise ValueError(f"capture contains duplicate poseIndex={pose_index}")
        seen.add(pose_index)
        candidate_ids = _optional_array(
            raw.get("candidateIds"), name=f"perPose[{index}].candidateIds", dtype=np.int64
        )
        if candidate_ids is None:
            candidate_ids = np.zeros((0,), dtype=np.int64)
        normalized.append(
            ScoreRow(
                pose_index=pose_index,
                candidate_ids=candidate_ids,
                scores=_optional_array(
                    raw.get("candidateScores"),
                    name=f"perPose[{index}].candidateScores",
                    dtype=np.float64,
                ),
                targets=_optional_array(
                    raw.get("targets"),
                    name=f"perPose[{index}].targets",
                    dtype=np.float64,
                ),
                visible_weights=_optional_array(
                    raw.get("visibleWeights"),
                    name=f"perPose[{index}].visibleWeights",
                    dtype=np.float64,
                ),
                feature_families=(
                    raw.get("featureFamilies")
                    or raw.get("features")
                    or raw.get("inputFeatures")
                ),
            )
        )
    normalized.sort(key=lambda row: row.pose_index)
    result = dict(payload)
    result["_normalizedRows"] = normalized
    return result


def _canonical_family_name(name: str) -> str | None:
    key = "".join(character for character in str(name) if character.isalnum()).lower()
    aliases = {
        "center": "center",
        "centerview": "center",
        "centerquery": "center",
        "meanstd": "mean_std",
        "moments": "mean_std",
        "spectralfeatures": "mean_std",
        "spectralmoment": "mean_std",
        "boundary": "boundary",
        "boundarysummary": "boundary",
        "boundaryspectralsummary": "boundary",
        "regionextrema": "region_extrema",
        "extrema": "region_extrema",
    }
    return aliases.get(key)


def _canonical_feature_mapping(value: Any, *, name: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping of feature family to arrays")
    result: dict[str, np.ndarray] = {}
    for key, raw in value.items():
        family = _canonical_family_name(str(key))
        if family is None:
            continue
        array = np.asarray(raw, dtype=np.float64)
        if family == "mean_std" and array.ndim == 3:
            if array.shape[-1] != 4:
                raise ValueError(f"{name}.{key} moments must end in dimension 4")
            array = array.reshape(array.shape[0], -1)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or not bool(np.isfinite(array).all()):
            raise ValueError(f"{name}.{key} must be a finite [candidate, feature] array")
        if family in result:
            raise ValueError(f"{name} contains duplicate aliases for {family}")
        result[family] = array
    missing = [family for family in FEATURE_FAMILIES if family not in result]
    if missing:
        raise ValueError(f"{name} is missing feature families: {', '.join(missing)}")
    row_count = {int(array.shape[0]) for array in result.values()}
    if len(row_count) != 1:
        raise ValueError(f"{name} feature families have different candidate counts")
    return result


def _resolve_row_arrays(
    rows: Sequence[ScoreRow],
    *,
    dataset: Any | None,
    split: str,
) -> list[ScoreRow]:
    """Fill omitted candidate/label/weight arrays from the PoseCSR schema."""

    if dataset is not None:
        split_view = dataset.split(split)
        expected_pose_indices = [int(value) for value in split_view.pose_indices.tolist()]
        actual_pose_indices = [row.pose_index for row in rows]
        if actual_pose_indices != expected_pose_indices:
            raise ValueError(
                f"{split} capture pose indices do not match dataset split: "
                f"capture={len(actual_pose_indices)}, dataset={len(expected_pose_indices)}"
            )

    result: list[ScoreRow] = []
    for row in rows:
        candidate_ids = row.candidate_ids
        if dataset is not None:
            stored_candidates = np.asarray(
                dataset.frustum_slice(row.pose_index), dtype=np.int64
            )
            if candidate_ids.size == 0:
                candidate_ids = stored_candidates
            elif not np.array_equal(candidate_ids, stored_candidates):
                raise ValueError(
                    f"pose {row.pose_index} candidateIds disagree with stored CSR candidates"
                )
            visible_ids, visible_weights = dataset.visible_slice(row.pose_index)
            expected_targets = np.isin(
                candidate_ids.astype(np.int64), np.asarray(visible_ids, dtype=np.int64)
            ).astype(np.float64)
            weight_map = {
                int(instance_id): float(weight)
                for instance_id, weight in zip(visible_ids.tolist(), visible_weights.tolist())
            }
            expected_weights = np.asarray(
                [weight_map.get(int(instance_id), 0.0) for instance_id in candidate_ids.tolist()],
                dtype=np.float64,
            )
        else:
            expected_targets = None
            expected_weights = None

        lengths = [
            array.size
            for array in (row.scores, row.targets, row.visible_weights)
            if array is not None
        ]
        if lengths and any(length != lengths[0] for length in lengths):
            raise ValueError(f"pose {row.pose_index} persisted arrays have different lengths")
        if lengths and candidate_ids.size not in (0, lengths[0]):
            raise ValueError(f"pose {row.pose_index} candidate count does not match persisted arrays")
        if candidate_ids.size == 0 and lengths:
            # A self-contained feature capture can omit IDs, but it still has a
            # stable row-local identity for alignment and probe diagnostics.
            candidate_ids = np.arange(lengths[0], dtype=np.int64)
        count = int(candidate_ids.size)
        targets = row.targets
        if targets is None and expected_targets is not None:
            targets = expected_targets
        weights = row.visible_weights
        if weights is None and expected_weights is not None:
            weights = expected_weights
        if targets is not None and targets.size != count:
            raise ValueError(f"pose {row.pose_index} targets length does not match candidates")
        if weights is not None and weights.size != count:
            raise ValueError(f"pose {row.pose_index} visibleWeights length does not match candidates")
        if expected_targets is not None and targets is not None and not np.array_equal(
            targets > 0.5, expected_targets > 0.5
        ):
            raise ValueError(f"pose {row.pose_index} targets disagree with dataset visible_ids")
        if expected_weights is not None and weights is not None and not np.allclose(
            weights, expected_weights, rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"pose {row.pose_index} visibleWeights disagree with dataset")
        if row.scores is not None:
            if row.scores.size != count:
                raise ValueError(f"pose {row.pose_index} scores length does not match candidates")
            if bool(np.any((row.scores < 0.0) | (row.scores > 1.0))):
                raise ValueError(f"pose {row.pose_index} scores must be probabilities")
        result.append(
            ScoreRow(
                pose_index=row.pose_index,
                candidate_ids=candidate_ids,
                scores=row.scores,
                targets=targets,
                visible_weights=weights,
                feature_families=row.feature_families,
            )
        )
    return result


def _flatten_rows(rows: Sequence[ScoreRow], *, split: str) -> FeatureExamples:
    score_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    visible_weight_values: list[np.ndarray] = []
    pose_values: list[np.ndarray] = []
    instance_values: list[np.ndarray] = []
    feature_values: dict[str, list[np.ndarray]] = {family: [] for family in FEATURE_FAMILIES}
    for row in rows:
        if row.scores is None or row.targets is None or row.visible_weights is None:
            raise ValueError(
                f"{split} capture lacks candidateScores/targets/visibleWeights; "
                "provide checkpoint+dataset for extraction"
            )
        score_values.append(_finite_array(row.scores, name=f"{split}/scores"))
        target_values.append(_finite_array(row.targets, name=f"{split}/targets"))
        visible_weight_values.append(
            _finite_array(row.visible_weights, name=f"{split}/visibleWeights")
        )
        pose_values.append(np.full(row.scores.size, row.pose_index, dtype=np.int64))
        instance_values.append(np.asarray(row.candidate_ids, dtype=np.int64))
        if row.feature_families is None:
            raise ValueError(f"{split} capture does not contain embedded input features")
        features = _canonical_feature_mapping(
            row.feature_families, name=f"{split}/pose{row.pose_index}/featureFamilies"
        )
        for family, values in features.items():
            if values.shape[0] != row.scores.size:
                raise ValueError(
                    f"{split} pose {row.pose_index} {family} feature count does not match scores"
                )
            feature_values[family].append(values)
    if not score_values:
        raise ValueError(f"{split} capture has no candidate rows")
    return FeatureExamples(
        split=split,
        scores=np.concatenate(score_values),
        targets=np.concatenate(target_values) > 0.5,
        features={family: np.concatenate(values) for family, values in feature_values.items()},
        visible_weights=np.concatenate(visible_weight_values),
        pose_indices=np.concatenate(pose_values),
        instance_ids=np.concatenate(instance_values),
    )


def _coerce_examples(value: FeatureExamples | Mapping[str, Any], *, split: str) -> FeatureExamples:
    if isinstance(value, FeatureExamples):
        examples = FeatureExamples(
            split=split,
            scores=_finite_array(value.scores, name=f"{split}/scores"),
            targets=_finite_array(value.targets, name=f"{split}/targets") > 0.5,
            features=_canonical_feature_mapping(value.features, name=f"{split}/features"),
            visible_weights=(
                None
                if value.visible_weights is None
                else _finite_array(
                    value.visible_weights, name=f"{split}/visibleWeights"
                )
            ),
            pose_indices=value.pose_indices,
            instance_ids=value.instance_ids,
        )
    else:
        features_value = value.get("features", value.get("featureFamilies"))
        features = _canonical_feature_mapping(features_value, name=f"{split}/features")
        examples = FeatureExamples(
            split=split,
            scores=_finite_array(value.get("scores"), name=f"{split}/scores"),
            targets=_finite_array(value.get("targets"), name=f"{split}/targets") > 0.5,
            features=features,
            visible_weights=(
                None
                if value.get("visibleWeights") is None
                else _finite_array(
                    value["visibleWeights"], name=f"{split}/visibleWeights"
                )
            ),
            pose_indices=(
                None
                if value.get("poseIndices") is None
                else np.asarray(value["poseIndices"], dtype=np.int64).reshape(-1)
            ),
            instance_ids=(
                None
                if value.get("instanceIds") is None
                else np.asarray(value["instanceIds"], dtype=np.int64).reshape(-1)
            ),
        )
    count = examples.scores.size
    if examples.targets.size != count or any(
        values.shape[0] != count for values in examples.features.values()
    ):
        raise ValueError(f"{split} scores, targets, and feature rows are not aligned")
    if bool(np.any((examples.scores < 0.0) | (examples.scores > 1.0))):
        raise ValueError(f"{split} scores must be probabilities")
    if examples.pose_indices is not None and examples.pose_indices.size != count:
        raise ValueError(f"{split} poseIndices do not align with scores")
    if examples.instance_ids is not None and examples.instance_ids.size != count:
        raise ValueError(f"{split} instanceIds do not align with scores")
    if examples.visible_weights is not None:
        if examples.visible_weights.size != count:
            raise ValueError(f"{split} visibleWeights do not align with scores")
        if bool(np.any(examples.visible_weights < 0.0)):
            raise ValueError(f"{split} visibleWeights must be non-negative")
    return examples


def freeze_tail_cutoffs(
    calibration_scores: np.ndarray,
    calibration_targets: np.ndarray,
    *,
    visible_weights: np.ndarray | None = None,
    positive_tail_mode: str = "count",
    source_split: str = "calibration",
    positive_quantile: float = 0.01,
    negative_quantile: float = 0.99,
) -> dict[str, Any]:
    """Freeze low-positive/high-negative score boundaries on train or calibration."""

    if not 0.0 < float(positive_quantile) < 1.0:
        raise ValueError("positive_quantile must lie in (0, 1)")
    if not 0.0 < float(negative_quantile) < 1.0:
        raise ValueError("negative_quantile must lie in (0, 1)")
    scores = _finite_array(calibration_scores, name="calibration scores")
    targets = _finite_array(calibration_targets, name="calibration targets") > 0.5
    if scores.size != targets.size:
        raise ValueError("calibration scores and targets are not aligned")
    if positive_tail_mode not in {"count", "weighted_mass"}:
        raise ValueError("positive_tail_mode must be count or weighted_mass")
    if source_split not in {"train", "calibration"}:
        raise ValueError("tail cutoff source must be train or calibration")
    positive = scores[targets]
    negative = scores[~targets]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("calibration must contain both positive and negative candidates")
    if positive_tail_mode == "weighted_mass":
        if visible_weights is None:
            raise ValueError("weighted_mass positive tails require visible_weights")
        weights = _finite_array(visible_weights, name="calibration visible weights")
        if weights.size != scores.size or bool(np.any(weights < 0.0)):
            raise ValueError("calibration visible weights are invalid or misaligned")
        positive_weights = weights[targets]
        if float(positive_weights.sum()) <= 0.0:
            raise ValueError("positive visible weights must have positive mass")
        order = np.argsort(positive, kind="mergesort")
        cumulative = np.cumsum(positive_weights[order])
        target_mass = float(positive_quantile) * float(cumulative[-1])
        selected = min(
            int(np.searchsorted(cumulative, target_mass, side="left")),
            int(order.size) - 1,
        )
        positive_cutoff = float(positive[order[selected]])
    else:
        positive_cutoff = float(np.quantile(positive, float(positive_quantile)))
    negative_cutoff = float(np.quantile(negative, float(negative_quantile)))
    return {
        "positiveLowScoreQuantile": float(positive_quantile),
        "positiveTailMode": positive_tail_mode,
        "negativeHighScoreQuantile": float(negative_quantile),
        "lowPositiveScoreCutoff": positive_cutoff,
        "highNegativeScoreCutoff": negative_cutoff,
        "sourceSplit": source_split,
        "selectedFromValidation": False,
        "testRead": False,
    }


def _tail_masks(
    examples: FeatureExamples,
    cutoffs: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    low_positive = examples.targets & (
        examples.scores <= float(cutoffs["lowPositiveScoreCutoff"])
    )
    high_negative = (~examples.targets) & (
        examples.scores >= float(cutoffs["highNegativeScoreCutoff"])
    )
    return low_positive, high_negative


def _tail_examples(examples: FeatureExamples, cutoffs: Mapping[str, Any]) -> FeatureExamples:
    low_positive, high_negative = _tail_masks(examples, cutoffs)
    selected = low_positive | high_negative
    # The probe label is deliberately the difficult negative class: 0 is a
    # low-score positive and 1 is a high-score negative.
    labels = high_negative[selected].astype(np.float64)
    return FeatureExamples(
        split=examples.split,
        scores=examples.scores[selected],
        targets=labels,
        features={family: values[selected] for family, values in examples.features.items()},
        visible_weights=(
            None
            if examples.visible_weights is None
            else examples.visible_weights[selected]
        ),
        pose_indices=None if examples.pose_indices is None else examples.pose_indices[selected],
        instance_ids=None if examples.instance_ids is None else examples.instance_ids[selected],
    )


def _float_or_none(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _feature_names(family: str, dimension: int) -> list[str]:
    if family == "center":
        if dimension != len(CENTER_FEATURE_NAMES):
            return [f"center_{index}" for index in range(dimension)]
        return list(CENTER_FEATURE_NAMES)
    if family == "mean_std":
        if dimension % 4 == 0:
            names: list[str] = []
            for frequency in range(dimension // 4):
                names.extend(
                    [
                        f"frequency_{frequency:02d}_mean_sin",
                        f"frequency_{frequency:02d}_mean_cos",
                        f"frequency_{frequency:02d}_std_sin",
                        f"frequency_{frequency:02d}_std_cos",
                    ]
                )
            return names
        return [f"mean_std_{index}" for index in range(dimension)]
    if family == "boundary":
        return [f"boundary_{index}" for index in range(dimension)]
    if family == "region_extrema":
        base_dimension = dimension // 3
        base = _feature_names("center", base_dimension)
        if dimension != len(base) * 3:
            return [f"region_extrema_{index}" for index in range(dimension)]
        return (
            [f"region_min_{name}" for name in base]
            + [f"region_max_{name}" for name in base]
            + [f"region_span_{name}" for name in base]
        )
    return [f"{family}_{index}" for index in range(dimension)]


def _group_summary(values: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != scores.size:
        raise ValueError("group feature values and scores are not aligned")
    quantiles = {
        f"q{int(round(level * 100)):02d}": [
            _float_or_none(float(item)) for item in np.quantile(values, level, axis=0)
        ]
        for level in QUANTILE_LEVELS
    }
    score_quantiles = {
        f"q{int(round(level * 100)):02d}": _float_or_none(
            float(np.quantile(scores, level))
        )
        for level in QUANTILE_LEVELS
    }
    return {
        "count": int(values.shape[0]),
        "scoreQuantiles": score_quantiles,
        "mean": [_float_or_none(float(item)) for item in np.mean(values, axis=0)],
        "std": [_float_or_none(float(item)) for item in np.std(values, axis=0)],
        "quantiles": quantiles,
    }


def _standardized_mean_difference(low_positive: np.ndarray, high_negative: np.ndarray) -> list[float | None]:
    if low_positive.ndim != 2 or high_negative.ndim != 2:
        raise ValueError("standardized mean difference expects two matrices")
    if low_positive.shape[1] != high_negative.shape[1]:
        raise ValueError("standardized mean difference feature dimensions differ")
    if low_positive.shape[0] == 0 or high_negative.shape[0] == 0:
        return [None] * low_positive.shape[1]
    mean_difference = np.mean(low_positive, axis=0) - np.mean(high_negative, axis=0)
    variance_low = np.var(low_positive, axis=0)
    variance_high = np.var(high_negative, axis=0)
    pooled_scale = np.sqrt(0.5 * (variance_low + variance_high))
    result = np.full(mean_difference.shape, np.nan, dtype=np.float64)
    nonzero = pooled_scale > 1e-12
    result[nonzero] = mean_difference[nonzero] / pooled_scale[nonzero]
    result[~nonzero & (np.abs(mean_difference) <= 1e-12)] = 0.0
    return [_float_or_none(float(item)) for item in result]


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, scores.size + 1, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[labels[order]].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    cumulative_positive = np.cumsum(sorted_labels)
    ranks = np.arange(1, scores.size + 1, dtype=np.float64)
    return float(np.sum((sorted_labels * cumulative_positive) / ranks) / positive_count)


def _score_distribution(values: np.ndarray) -> dict[str, Any]:
    """Summarize only the scores supplied by the caller, including empty groups."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not bool(np.isfinite(values).all()):
        raise ValueError("score distribution contains non-finite values")
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            **{f"q{int(round(level * 100)):02d}": None for level in QUANTILE_LEVELS},
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        **{
            f"q{int(round(level * 100)):02d}": float(np.quantile(values, level))
            for level in QUANTILE_LEVELS
        },
    }


def _probe_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    source_split: str,
    fit_rows_only: bool,
) -> dict[str, Any]:
    """Report the fitted probe score distribution with explicit label semantics."""

    values = _finite_array(scores, name="probe scores")
    probe_labels = _finite_array(labels, name="probe labels") > 0.5
    if values.size != probe_labels.size:
        raise ValueError("probe scores and labels are not aligned")
    return {
        "sourceSplit": str(source_split),
        "fitRowsOnly": bool(fit_rows_only),
        "probeLabel": "1=high_score_negative, 0=low_score_positive",
        "lowScorePositive": _score_distribution(values[~probe_labels]),
        "highScoreNegative": _score_distribution(values[probe_labels]),
    }


def _probe_tail_importance_weights(
    labels: np.ndarray,
    visible_weights: np.ndarray | None,
    *,
    power: float,
    epsilon: float,
) -> np.ndarray:
    """Return visibility-mass weights for the low-score-positive tail."""

    probe_labels = _finite_array(labels, name="probe labels") > 0.5
    if not np.isfinite(power) or float(power) < 0.0:
        raise ValueError("sample weight power must be finite and non-negative")
    if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
        raise ValueError("sample weight epsilon must be finite and positive")
    if visible_weights is None:
        weights = np.ones(probe_labels.size, dtype=np.float64)
    else:
        weights = _finite_array(visible_weights, name="probe visible weights")
        if weights.size != probe_labels.size or bool(np.any(weights < 0.0)):
            raise ValueError("probe visible weights are invalid or misaligned")
    if float(power) == 0.0:
        result = np.ones(probe_labels.size, dtype=np.float64)
    else:
        result = np.ones(probe_labels.size, dtype=np.float64)
        result[~probe_labels] = np.maximum(
            weights[~probe_labels], float(epsilon)
        ) ** float(power)
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("probe tail weights are non-finite")
    return result


def _probe_fit_sample_weights(
    labels: np.ndarray,
    visible_weights: np.ndarray | None,
    *,
    power: float,
    epsilon: float,
) -> np.ndarray:
    """Build train-only ridge weights while preserving power=0 old semantics."""

    probe_labels = _finite_array(labels, name="probe labels") > 0.5
    if float(power) == 0.0:
        return np.ones(probe_labels.size, dtype=np.float64)
    tail_weights = _probe_tail_importance_weights(
        probe_labels,
        visible_weights,
        power=power,
        epsilon=epsilon,
    )
    low_positive = ~probe_labels
    high_negative = probe_labels
    positive_mass = float(tail_weights[low_positive].sum())
    negative_count = int(high_negative.sum())
    if negative_count <= 0:
        return tail_weights
    if positive_mass <= 0.0:
        raise ValueError("probe sample weighting requires positive tail mass")
    # Keep both class totals equal and keep the overall loss scale near one
    # sample per row.  The positive relative weights retain the visible-mass
    # ordering; the negative tail has uniform weights after class balancing.
    target_class_mass = 0.5 * float(probe_labels.size)
    result = np.zeros(probe_labels.size, dtype=np.float64)
    result[low_positive] = target_class_mass * tail_weights[low_positive] / positive_mass
    result[high_negative] = target_class_mass / float(negative_count)
    if not bool(np.isfinite(result).all()) or bool(np.any(result <= 0.0)):
        raise FloatingPointError("probe fit sample weights are invalid")
    return result


def _weight_summary(values: np.ndarray) -> dict[str, float]:
    """Return ESS and maximum normalized weight share without clipping weights."""

    weights = _finite_array(values, name="fit sample weights")
    if weights.size == 0 or bool(np.any(weights < 0.0)):
        raise ValueError("fit sample weights must be non-empty and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("fit sample weights must have positive total mass")
    return {
        "weightTotal": total,
        "effectiveSampleSize": float(total * total / np.dot(weights, weights)),
        "maximumWeightShare": float(np.max(weights) / total),
    }


def _weighted_rank_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> float | None:
    """Weighted AUC where label 1 (high negative) should have higher scores."""

    values = _finite_array(scores, name="weighted AUC scores")
    probe_labels = _finite_array(labels, name="weighted AUC labels") > 0.5
    weights = _finite_array(sample_weights, name="weighted AUC sample weights")
    if values.size != probe_labels.size or values.size != weights.size:
        raise ValueError("weighted AUC arrays are not aligned")
    if bool(np.any(weights < 0.0)):
        raise ValueError("weighted AUC weights must be non-negative")
    high = probe_labels
    low = ~probe_labels
    high_mass = float(weights[high].sum())
    low_mass = float(weights[low].sum())
    if high_mass <= 0.0 or low_mass <= 0.0:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_scores = values[order]
    sorted_labels = probe_labels[order]
    sorted_weights = weights[order]
    low_before = 0.0
    numerator = 0.0
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_labels = sorted_labels[start:end]
        group_weights = sorted_weights[start:end]
        low_group = float(group_weights[~group_labels].sum())
        high_group = float(group_weights[group_labels].sum())
        numerator += high_group * (low_before + 0.5 * low_group)
        low_before += low_group
        start = end
    return float(numerator / (high_mass * low_mass))


def _probe_direction_diagnostic(
    scores: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> dict[str, Any]:
    """Check whether high-score negatives rank above low-score positives."""

    values = _finite_array(scores, name="direction scores")
    probe_labels = _finite_array(labels, name="direction labels") > 0.5
    weights = _finite_array(sample_weights, name="direction weights")
    if not (values.size == probe_labels.size == weights.size):
        raise ValueError("direction diagnostic arrays are not aligned")
    high = probe_labels
    low = ~probe_labels
    high_mass = float(weights[high].sum())
    low_mass = float(weights[low].sum())
    high_mean = (
        float(np.average(values[high], weights=weights[high])) if high_mass > 0.0 else None
    )
    low_mean = (
        float(np.average(values[low], weights=weights[low])) if low_mass > 0.0 else None
    )
    difference = (
        None if high_mean is None or low_mean is None else float(high_mean - low_mean)
    )
    return {
        "expected": "high_score_negative_probe_score_above_low_score_positive",
        "weightedHighScoreNegativeMean": high_mean,
        "weightedLowScorePositiveMean": low_mean,
        "weightedHighMinusLow": difference,
        "correctDirection": None if difference is None else bool(difference > 0.0),
        "weightedRocAuc": _weighted_rank_auc(values, probe_labels, weights),
    }


def _risk_record(
    *,
    risk: float,
    threshold: float,
    selected: np.ndarray,
    labels: np.ndarray,
    positive_weights: np.ndarray,
    kind: str,
) -> dict[str, Any]:
    probe_labels = _finite_array(labels, name="certificate labels") > 0.5
    weights = _finite_array(positive_weights, name="certificate positive weights")
    low_positive = ~probe_labels
    high_negative = probe_labels
    total_positive_weight = float(weights[low_positive].sum())
    positive_selected_weight = float(weights[low_positive & selected].sum())
    high_negative_count = int(high_negative.sum())
    selected_high_negative = int((high_negative & selected).sum())
    if kind == "suppression":
        observed_risk = (
            positive_selected_weight / total_positive_weight
            if total_positive_weight > 0.0
            else 0.0
        )
        high_negative_coverage = (
            selected_high_negative / high_negative_count
            if high_negative_count > 0
            else 0.0
        )
        return {
            "registeredRiskUpperBound": float(risk),
            "threshold": float(threshold),
            "applyWhen": "probe_score >= threshold",
            "observedRisk": float(observed_risk),
            "riskDefinition": "weighted low-score-positive mass selected for suppression / all low-score-positive mass",
            "riskWithinRegisteredBound": bool(observed_risk <= float(risk)),
            "lowScorePositiveWeightedCoverage": float(
                1.0 - observed_risk
            ),
            "highScoreNegativeCoverage": float(high_negative_coverage),
            "selectedHighScoreNegativeCount": selected_high_negative,
            "highScoreNegativeCount": high_negative_count,
            "tiePolicy": "inclusive observed threshold; all equal scores are selected",
        }
    if kind == "rescue":
        observed_risk = (
            selected_high_negative / high_negative_count
            if high_negative_count > 0
            else 0.0
        )
        low_positive_coverage = (
            positive_selected_weight / total_positive_weight
            if total_positive_weight > 0.0
            else 0.0
        )
        return {
            "registeredRiskUpperBound": float(risk),
            "threshold": float(threshold),
            "applyWhen": "probe_score <= threshold",
            "observedRisk": float(observed_risk),
            "riskDefinition": "high-score-negative count selected for rescue / all high-score-negative count",
            "riskWithinRegisteredBound": bool(observed_risk <= float(risk)),
            "highScoreNegativeCoverage": float(1.0 - observed_risk),
            "lowScorePositiveWeightedCoverage": float(low_positive_coverage),
            "selectedHighScoreNegativeCount": selected_high_negative,
            "highScoreNegativeCount": high_negative_count,
            "tiePolicy": "inclusive observed threshold; all equal scores are selected",
        }
    raise ValueError(f"unsupported certificate kind {kind!r}")


def build_train_risk_certificates(
    probe_scores: np.ndarray,
    labels: np.ndarray,
    visible_weights: np.ndarray | None,
    *,
    sample_weight_power: float = DEFAULT_PROBE_SAMPLE_WEIGHT_POWER,
    sample_weight_epsilon: float = DEFAULT_PROBE_SAMPLE_WEIGHT_EPSILON,
) -> dict[str, Any]:
    """Build finite-sample train-only rescue/suppression certificates.

    Thresholds are inclusive observed-score cut points.  The selected tail is
    evaluated including every tie, so a certificate is never made optimistic
    by splitting equal probe scores.
    """

    scores = _finite_array(probe_scores, name="certificate probe scores")
    probe_labels = _finite_array(labels, name="certificate labels") > 0.5
    if scores.size != probe_labels.size:
        raise ValueError("certificate scores and labels are not aligned")
    if np.unique(probe_labels).size < 2:
        raise ValueError("certificates require both low positives and high negatives")
    positive_weights = _probe_tail_importance_weights(
        probe_labels,
        visible_weights,
        power=sample_weight_power,
        epsilon=sample_weight_epsilon,
    )
    low_positive = ~probe_labels
    high_negative = probe_labels
    positive_total = float(positive_weights[low_positive].sum())
    negative_count = int(high_negative.sum())
    if positive_total <= 0.0 or negative_count <= 0:
        raise ValueError("certificates require non-empty positive mass and negative count")

    suppression: dict[str, Any] = {}
    suppression_candidates = sorted(float(value) for value in np.unique(scores))
    above_max = float(np.nextafter(float(np.max(scores)), np.inf))
    if not np.isfinite(above_max):
        raise ValueError("cannot construct a finite suppression no-selection threshold")
    suppression_candidates.append(above_max)
    for risk in SUPPRESSION_RISK_LEVELS:
        chosen_threshold = above_max
        chosen_selected = scores >= above_max
        for threshold in suppression_candidates:
            selected = scores >= float(threshold)
            observed = float(positive_weights[low_positive & selected].sum()) / positive_total
            if observed <= float(risk):
                chosen_threshold = float(threshold)
                chosen_selected = selected
                break
        record = _risk_record(
            risk=float(risk),
            threshold=chosen_threshold,
            selected=chosen_selected,
            labels=probe_labels,
            positive_weights=positive_weights,
            kind="suppression",
        )
        if not bool(record["riskWithinRegisteredBound"]):
            raise AssertionError("suppression certificate exceeds its registered risk")
        suppression[str(risk)] = record

    below_min = float(np.nextafter(float(np.min(scores)), -np.inf))
    if not np.isfinite(below_min):
        raise ValueError("cannot construct a finite rescue no-selection threshold")
    rescue_candidates = [below_min] + sorted(float(value) for value in np.unique(scores))
    rescue: dict[str, Any] = {}
    for risk in RESCUE_RISK_LEVELS:
        chosen_threshold = below_min
        chosen_selected = scores <= below_min
        for threshold in rescue_candidates:
            selected = scores <= float(threshold)
            observed = float((high_negative & selected).sum()) / float(negative_count)
            if observed <= float(risk):
                chosen_threshold = float(threshold)
                chosen_selected = selected
        record = _risk_record(
            risk=float(risk),
            threshold=chosen_threshold,
            selected=chosen_selected,
            labels=probe_labels,
            positive_weights=positive_weights,
            kind="rescue",
        )
        if not bool(record["riskWithinRegisteredBound"]):
            raise AssertionError("rescue certificate exceeds its registered risk")
        rescue[str(risk)] = record
    return {
        "sourceSplit": "train",
        "fitRowsOnly": True,
        "sampleWeightPower": float(sample_weight_power),
        "sampleWeightEpsilon": float(sample_weight_epsilon),
        "probeLabel": "1=high_score_negative, 0=low_score_positive",
        "suppression": suppression,
        "rescue": rescue,
    }


def _threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = np.asarray(scores, dtype=np.float64) >= float(threshold)
    labels = np.asarray(labels, dtype=bool)
    tp = float(np.logical_and(predicted, labels).sum())
    fp = float(np.logical_and(predicted, ~labels).sum())
    fn = float(np.logical_and(~predicted, labels).sum())
    tn = float(np.logical_and(~predicted, ~labels).sum())
    recall = tp / max(1.0, tp + fn)
    specificity = tn / max(1.0, tn + fp)
    return {
        "threshold": float(threshold),
        "balancedAccuracy": float(0.5 * (recall + specificity)),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float((tp + tn) / max(1.0, tp + fp + fn + tn)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _freeze_probe_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any] | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    if scores.size == 0 or labels.size != scores.size or labels.all() or (~labels).all():
        return None
    candidates = np.unique(scores)
    metrics = [_threshold_metrics(scores, labels, float(value)) for value in candidates]
    chosen = max(
        metrics,
        key=lambda item: (float(item["balancedAccuracy"]), float(item["threshold"])),
    )
    return {
        **chosen,
        "sourceSplit": "calibration",
        "selectedFromValidation": False,
        "testRead": False,
    }


def _probe_fit(
    fit_examples: FeatureExamples,
    calibration_examples: FeatureExamples,
    validation_examples: FeatureExamples,
    family: str,
    *,
    ridge: float,
    max_samples: int,
    seed: int,
    sample_weight_power: float = DEFAULT_PROBE_SAMPLE_WEIGHT_POWER,
    sample_weight_epsilon: float = DEFAULT_PROBE_SAMPLE_WEIGHT_EPSILON,
) -> dict[str, Any]:
    fit_values = fit_examples.features[family]
    calibration_values = calibration_examples.features[family]
    validation_values = validation_examples.features[family]
    fit_labels = fit_examples.targets.astype(np.float64)
    if fit_values.shape[0] == 0 or np.unique(fit_labels).size < 2:
        return {
            "status": "insufficient_fit_classes",
            "fitSplit": fit_examples.split,
            "featureCount": int(fit_values.shape[1]),
            "fitCount": int(fit_values.shape[0]),
            "validation": {"rocAuc": None, "averagePrecision": None},
            "fitProbeScoreDistribution": None,
            "trainFitProbeScoreDistribution": None,
            "riskCertificates": None,
            "weighting": {
                "sampleWeightPower": float(sample_weight_power),
                "sampleWeightEpsilon": float(sample_weight_epsilon),
                "fitWeightingApplied": False,
            },
            "thresholdSource": {
                "sourceSplit": "calibration",
                "selectedFromValidation": False,
                "testRead": False,
            },
        }
    if float(ridge) <= 0.0:
        raise ValueError("ridge must be positive")
    if not np.isfinite(sample_weight_power) or float(sample_weight_power) < 0.0:
        raise ValueError("sample_weight_power must be finite and non-negative")
    if not np.isfinite(sample_weight_epsilon) or float(sample_weight_epsilon) <= 0.0:
        raise ValueError("sample_weight_epsilon must be finite and positive")
    if fit_values.shape[1] != calibration_values.shape[1] or fit_values.shape[1] != validation_values.shape[1]:
        raise ValueError(f"{family} feature dimensions differ across splits")
    rng = np.random.default_rng(int(seed))
    fit_indices = np.arange(fit_values.shape[0], dtype=np.int64)
    if int(max_samples) > 0 and fit_indices.size > int(max_samples):
        fit_indices = np.sort(rng.choice(fit_indices, size=int(max_samples), replace=False))
    fit_values = fit_values[fit_indices].astype(np.float64, copy=False)
    fit_labels = fit_labels[fit_indices]
    fit_visible_weights = (
        None
        if fit_examples.visible_weights is None
        else _finite_array(
            fit_examples.visible_weights[fit_indices],
            name=f"{fit_examples.split} visible weights used for fit",
        )
    )
    fit_sample_weights = _probe_fit_sample_weights(
        fit_labels,
        fit_visible_weights,
        power=float(sample_weight_power),
        epsilon=float(sample_weight_epsilon),
    )
    if float(sample_weight_power) == 0.0:
        # Keep the original unweighted path numerically unchanged.
        mean = np.mean(fit_values, axis=0)
        scale = np.std(fit_values, axis=0)
    else:
        weight_total = float(fit_sample_weights.sum())
        mean = np.sum(fit_values * fit_sample_weights[:, None], axis=0) / weight_total
        centered_values = fit_values - mean[None, :]
        scale = np.sqrt(
            np.sum(
                centered_values * centered_values * fit_sample_weights[:, None],
                axis=0,
            )
            / weight_total
        )
    scale = np.where(scale > 1e-8, scale, 1.0)

    def design(values: np.ndarray) -> np.ndarray:
        standardized = (values - mean) / scale
        return np.concatenate(
            [np.ones((standardized.shape[0], 1), dtype=np.float64), standardized], axis=1
        )

    design_fit = design(fit_values)
    regularizer = np.eye(design_fit.shape[1], dtype=np.float64) * float(ridge)
    regularizer[0, 0] = 0.0
    weighted_design = design_fit * fit_sample_weights[:, None]
    normal = design_fit.T @ weighted_design + regularizer
    rhs = design_fit.T @ (fit_sample_weights * fit_labels)
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]

    def predict(values: np.ndarray) -> np.ndarray:
        return design(values) @ coefficients

    calibration_scores = predict(calibration_values)
    validation_scores = predict(validation_values)
    fit_probe_scores = predict(fit_values)
    calibration_sample_weights = _probe_fit_sample_weights(
        calibration_examples.targets,
        calibration_examples.visible_weights,
        power=float(sample_weight_power),
        epsilon=float(sample_weight_epsilon),
    )
    validation_sample_weights = _probe_fit_sample_weights(
        validation_examples.targets,
        validation_examples.visible_weights,
        power=float(sample_weight_power),
        epsilon=float(sample_weight_epsilon),
    )
    frozen_threshold = _freeze_probe_threshold(
        calibration_scores, calibration_examples.targets.astype(bool)
    )
    validation_threshold_metrics = (
        _threshold_metrics(
            validation_scores,
            validation_examples.targets.astype(bool),
            float(frozen_threshold["threshold"]),
        )
        if frozen_threshold is not None
        else None
    )
    return {
        "status": "ok",
        "probeType": "standardized_ridge_linear",
        "fitSplit": fit_examples.split,
        "fitCount": int(fit_values.shape[0]),
        "featureCount": int(fit_values.shape[1]),
        "ridge": float(ridge),
        "coefficients": [float(value) for value in coefficients],
        "coefficientOrder": ["intercept", "standardized_features"],
        "fitProbeScoreDistribution": _probe_score_distribution(
            fit_probe_scores,
            fit_labels,
            source_split=fit_examples.split,
            fit_rows_only=True,
        ),
        "trainFitProbeScoreDistribution": (
            _probe_score_distribution(
                fit_probe_scores,
                fit_labels,
                source_split="train",
                fit_rows_only=True,
            )
            if fit_examples.split == "train"
            else None
        ),
        "fit": {
            "count": int(fit_probe_scores.size),
            "rocAuc": _rank_auc(fit_probe_scores, fit_labels),
            "weightedRocAuc": _weighted_rank_auc(
                fit_probe_scores,
                fit_labels,
                fit_sample_weights,
            ),
            "direction": _probe_direction_diagnostic(
                fit_probe_scores,
                fit_labels,
                fit_sample_weights,
            ),
        },
        "weighting": {
            "sampleWeightPower": float(sample_weight_power),
            "sampleWeightEpsilon": float(sample_weight_epsilon),
            "fitWeightingApplied": bool(float(sample_weight_power) != 0.0),
            "classTotalPolicy": (
                "original_unweighted_all_rows"
                if float(sample_weight_power) == 0.0
                else "low_positive_visibility_mass_and_equal_class_total"
            ),
            "fitLowScorePositiveWeightTotal": float(
                fit_sample_weights[fit_labels <= 0.5].sum()
            ),
            "fitHighScoreNegativeWeightTotal": float(
                fit_sample_weights[fit_labels > 0.5].sum()
            ),
            "standardization": (
                "unweighted_train_fit_mean_std"
                if float(sample_weight_power) == 0.0
                else "train_fit_sample_weighted_mean_std"
            ),
            "fitEffectiveSampleSize": _weight_summary(fit_sample_weights)[
                "effectiveSampleSize"
            ],
            "fitMaximumWeightShare": _weight_summary(fit_sample_weights)[
                "maximumWeightShare"
            ],
            "fitByProbeClass": {
                "lowScorePositive": _weight_summary(
                    fit_sample_weights[fit_labels <= 0.5]
                ),
                "highScoreNegative": _weight_summary(
                    fit_sample_weights[fit_labels > 0.5]
                ),
            },
        },
        "calibration": {
            "count": int(calibration_values.shape[0]),
            "rocAuc": _rank_auc(calibration_scores, calibration_examples.targets),
            "averagePrecision": _average_precision(
                calibration_scores, calibration_examples.targets
            ),
            "weightedRocAuc": _weighted_rank_auc(
                calibration_scores,
                calibration_examples.targets,
                calibration_sample_weights,
            ),
            "direction": _probe_direction_diagnostic(
                calibration_scores,
                calibration_examples.targets,
                calibration_sample_weights,
            ),
        },
        "validation": {
            "count": int(validation_values.shape[0]),
            "rocAuc": _rank_auc(validation_scores, validation_examples.targets),
            "averagePrecision": _average_precision(
                validation_scores, validation_examples.targets
            ),
            "weightedRocAuc": _weighted_rank_auc(
                validation_scores,
                validation_examples.targets,
                validation_sample_weights,
            ),
            "direction": _probe_direction_diagnostic(
                validation_scores,
                validation_examples.targets,
                validation_sample_weights,
            ),
            "thresholdMetrics": validation_threshold_metrics,
        },
        "frozenThreshold": frozen_threshold,
        "riskCertificates": (
            build_train_risk_certificates(
                fit_probe_scores,
                fit_labels,
                fit_visible_weights,
                sample_weight_power=float(sample_weight_power),
                sample_weight_epsilon=float(sample_weight_epsilon),
            )
            if fit_examples.split == "train"
            else None
        ),
        "standardization": {
            "mean": [_float_or_none(float(item)) for item in mean],
            "scale": [_float_or_none(float(item)) for item in scale],
        },
        "testRead": False,
    }


def _family_report(
    calibration_tail: FeatureExamples,
    validation_tail: FeatureExamples,
    family: str,
) -> dict[str, Any]:
    calibration_values = calibration_tail.features[family]
    validation_values = validation_tail.features[family]
    dimension = int(calibration_values.shape[1])
    names = _feature_names(family, dimension)

    def split_report(examples: FeatureExamples) -> dict[str, Any]:
        low = examples.targets == 0
        high = examples.targets == 1
        return {
            "lowScorePositive": _group_summary(
                examples.features[family][low], examples.scores[low]
            ),
            "highScoreNegative": _group_summary(
                examples.features[family][high], examples.scores[high]
            ),
            "standardizedMeanDifference": _standardized_mean_difference(
                examples.features[family][low], examples.features[family][high]
            ),
            "standardizedMeanDifferenceDefinition": (
                "mean(low_score_positive) - mean(high_score_negative) divided by "
                "sqrt((variance_low + variance_high) / 2)"
            ),
        }

    return {
        "dimension": dimension,
        "featureNames": names,
        "calibration": split_report(calibration_tail),
        "validation": split_report(validation_tail),
    }


def _combine_tail_features(examples: FeatureExamples) -> FeatureExamples:
    return FeatureExamples(
        split=examples.split,
        scores=examples.scores,
        targets=examples.targets,
        features={
            "combined": np.concatenate(
                [examples.features[family] for family in FEATURE_FAMILIES], axis=1
            )
        },
        visible_weights=examples.visible_weights,
        pose_indices=examples.pose_indices,
        instance_ids=examples.instance_ids,
    )


def analyze_tail_examples(
    calibration_tail: FeatureExamples,
    validation_tail: FeatureExamples,
    *,
    train: FeatureExamples | None,
    cutoffs: Mapping[str, Any],
    fit_cutoffs: Mapping[str, Any] | None = None,
    probe_fit_split: str,
    ridge: float,
    max_probe_samples: int,
    seed: int,
    sample_weight_power: float = DEFAULT_PROBE_SAMPLE_WEIGHT_POWER,
    sample_weight_epsilon: float = DEFAULT_PROBE_SAMPLE_WEIGHT_EPSILON,
) -> dict[str, Any]:
    """Analyze already-selected tails; used by the CLI and unit tests."""

    calibration_tail = _coerce_examples(calibration_tail, split="calibration")
    validation_tail = _coerce_examples(validation_tail, split="validation")
    train_tail = None if train is None else _coerce_examples(train, split="train")
    tail_by_split = {"calibration": calibration_tail, "validation": validation_tail}
    if train_tail is not None:
        tail_by_split["train"] = train_tail
    if probe_fit_split not in {"calibration", "train"}:
        raise ValueError("probe_fit_split must be calibration or train")
    if probe_fit_split == "train" and train_tail is None:
        raise ValueError("probe_fit_split=train requires train tail examples")
    fit_tail = tail_by_split[probe_fit_split]
    families = {
        family: _family_report(calibration_tail, validation_tail, family)
        for family in FEATURE_FAMILIES
    }
    probe_reports = {
        family: _probe_fit(
            fit_tail,
            calibration_tail,
            validation_tail,
            family,
            ridge=ridge,
            max_samples=max_probe_samples,
            seed=seed + FEATURE_FAMILIES.index(family),
            sample_weight_power=sample_weight_power,
            sample_weight_epsilon=sample_weight_epsilon,
        )
        for family in FEATURE_FAMILIES
    }
    combined_fit = _combine_tail_features(fit_tail)
    combined_calibration = _combine_tail_features(calibration_tail)
    combined_validation = _combine_tail_features(validation_tail)
    probe_reports["combined"] = _probe_fit(
        combined_fit,
        combined_calibration,
        combined_validation,
        "combined",
        ridge=ridge,
        max_samples=max_probe_samples,
        seed=seed + len(FEATURE_FAMILIES),
        sample_weight_power=sample_weight_power,
        sample_weight_epsilon=sample_weight_epsilon,
    )
    combined_names = []
    for family in FEATURE_FAMILIES:
        combined_names.extend(
            [
                f"{family}.{name}"
                for name in _feature_names(family, calibration_tail.features[family].shape[1])
            ]
        )
    return {
        "schema": SCHEMA,
        "testRead": False,
        "splitsRead": ["calibration"]
        + (["train"] if train_tail is not None else [])
        + ["validation"],
        "tailDefinition": {
            "lowScorePositive": "target=1 and score <= calibration lowPositiveScoreCutoff",
            "highScoreNegative": "target=0 and score >= calibration highNegativeScoreCutoff",
            "probeLabel": "1=high_score_negative, 0=low_score_positive",
            "cutoffs": dict(cutoffs),
            "fitCutoffs": dict(fit_cutoffs or cutoffs),
        },
        "groups": {
            split: {
                "lowScorePositive": int(np.count_nonzero(tail.targets == 0)),
                "highScoreNegative": int(np.count_nonzero(tail.targets == 1)),
                "total": int(tail.scores.size),
            }
            for split, tail in tail_by_split.items()
        },
        "features": families,
        "probe": {
            "fitSplit": probe_fit_split,
            "thresholdSource": "calibration",
            "selectedFromValidation": False,
            "families": probe_reports,
            "combinedFeatureNames": combined_names,
        },
        "configuration": {
            "positiveTailQuantile": cutoffs.get("positiveLowScoreQuantile"),
            "negativeTailQuantile": cutoffs.get("negativeHighScoreQuantile"),
            "ridge": float(ridge),
            "maxProbeSamples": int(max_probe_samples),
            "probeSampleWeightPower": float(sample_weight_power),
            "probeSampleWeightEpsilon": float(sample_weight_epsilon),
            "seed": int(seed),
        },
    }


def analyze_feature_tables(
    calibration: FeatureExamples | Mapping[str, Any],
    validation: FeatureExamples | Mapping[str, Any],
    *,
    train: FeatureExamples | Mapping[str, Any] | None = None,
    positive_tail_mode: str = "count",
    probe_fit_tail_source: str = "calibration",
    positive_tail_quantile: float = 0.01,
    negative_tail_quantile: float = 0.99,
    probe_fit_split: str = "calibration",
    ridge: float = 1e-3,
    max_probe_samples: int = 100000,
    seed: int = 20260818,
    probe_sample_weight_power: float = DEFAULT_PROBE_SAMPLE_WEIGHT_POWER,
    probe_sample_weight_epsilon: float = DEFAULT_PROBE_SAMPLE_WEIGHT_EPSILON,
) -> dict[str, Any]:
    """Freeze calibration cutoffs, select tails, and run the full report."""

    calibration_examples = _coerce_examples(calibration, split="calibration")
    validation_examples = _coerce_examples(validation, split="validation")
    train_examples = None if train is None else _coerce_examples(train, split="train")
    if probe_fit_tail_source not in {"calibration", "train"}:
        raise ValueError("probe_fit_tail_source must be calibration or train")
    if probe_fit_tail_source == "train" and train_examples is None:
        raise ValueError("train tail cutoffs require train examples")
    cutoffs = freeze_tail_cutoffs(
        calibration_examples.scores,
        calibration_examples.targets,
        visible_weights=calibration_examples.visible_weights,
        positive_tail_mode=positive_tail_mode,
        positive_quantile=positive_tail_quantile,
        negative_quantile=negative_tail_quantile,
    )
    fit_cutoffs = cutoffs
    if probe_fit_tail_source == "train":
        assert train_examples is not None
        fit_cutoffs = freeze_tail_cutoffs(
            train_examples.scores,
            train_examples.targets,
            visible_weights=train_examples.visible_weights,
            positive_tail_mode=positive_tail_mode,
            positive_quantile=positive_tail_quantile,
            negative_quantile=negative_tail_quantile,
            source_split="train",
        )
    return analyze_tail_examples(
        _tail_examples(calibration_examples, cutoffs),
        _tail_examples(validation_examples, cutoffs),
        train=(
            None
            if train_examples is None
            else _tail_examples(train_examples, fit_cutoffs)
        ),
        cutoffs=cutoffs,
        fit_cutoffs=fit_cutoffs,
        probe_fit_split=probe_fit_split,
        ridge=ridge,
        max_probe_samples=max_probe_samples,
        seed=seed,
        sample_weight_power=probe_sample_weight_power,
        sample_weight_epsilon=probe_sample_weight_epsilon,
    )


def region_extrema_features(center_view: np.ndarray, disk_axes: np.ndarray) -> np.ndarray:
    """Return exact per-coordinate unit-disk bounds and spans."""

    center = np.asarray(center_view, dtype=np.float64)
    axes = np.asarray(disk_axes, dtype=np.float64)
    if center.ndim != 2 or axes.ndim != 3 or axes.shape[:2] != center.shape:
        raise ValueError("center_view must be [B, D] and disk_axes must be [B, D, 2]")
    if axes.shape[-1] != 2 or not bool(np.isfinite(center).all() and np.isfinite(axes).all()):
        raise ValueError("center_view and disk_axes must be finite with two disk axes")
    extent = np.linalg.norm(axes, axis=-1)
    lower = center - extent
    upper = center + extent
    return np.concatenate([lower, upper, upper - lower], axis=1)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("checkpoint extraction requires PyTorch")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return dict(value)


def _load_geometry(path: Path, num_instances: int) -> np.ndarray:
    from bounded_relation_survival_moment_model import GEO_DIM

    expected = int(num_instances) * int(GEO_DIM)
    if not path.is_file() or path.stat().st_size != expected * 2:
        raise ValueError(f"fixed geometry must be [{num_instances}, {GEO_DIM}] FP16: {path}")
    values = np.fromfile(path, dtype="<f2").reshape(num_instances, GEO_DIM)
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"fixed geometry contains non-finite values: {path}")
    return values.astype(np.float32)


def _view_residual_values(config: Mapping[str, Any]) -> tuple[float, int]:
    residual = config.get("viewResidual")
    if not isinstance(residual, Mapping) or not bool(residual.get("enabled", False)):
        return 0.0, 16
    maximum = float(residual.get("maximumAbsoluteResidual", 0.0))
    hidden_dim = int(residual.get("hiddenDim", 0))
    if not np.isfinite(maximum) or maximum <= 0.0 or hidden_dim <= 0:
        raise ValueError("checkpoint viewResidual configuration is invalid")
    return maximum, hidden_dim


def _boundary_opportunity_values(
    config: Mapping[str, Any],
) -> tuple[int, int, float, float]:
    opportunity = config.get("boundaryOpportunity")
    if not isinstance(opportunity, Mapping) or not bool(
        opportunity.get("enabled", False)
    ):
        return 0, 24, -6.0, 6.0
    hidden_dim = int(opportunity.get("hiddenDim", 0))
    projection_dim = int(opportunity.get("projectionDim", 0))
    initial_logit = float(opportunity.get("initialLogit", float("nan")))
    maximum_uplift = float(
        opportunity.get("maximumLogitUplift", float("nan"))
    )
    if (
        hidden_dim <= 0
        or projection_dim <= 0
        or not np.isfinite(initial_logit)
        or not np.isfinite(maximum_uplift)
        or maximum_uplift <= 0.0
    ):
        raise ValueError("checkpoint boundaryOpportunity configuration is invalid")
    return hidden_dim, projection_dim, initial_logit, maximum_uplift


def _boundary_tail_residual_values(
    config: Mapping[str, Any],
) -> tuple[int, int, float, str, str, str, float]:
    residual = config.get("boundaryTailResidual")
    if not isinstance(residual, Mapping) or not bool(residual.get("enabled", False)):
        return 0, 24, 1.0, "pose_mean", "none", "product", 0.0
    hidden_dim = int(residual.get("hiddenDim", 0))
    projection_dim = int(residual.get("projectionDim", 0))
    maximum = float(residual.get("maximumAbsoluteResidual", float("nan")))
    centering = str(residual.get("centering", "pose_mean"))
    shortcut = str(residual.get("shortcut", "none"))
    fusion = str(residual.get("fusionMode", "product"))
    output_init_std = float(residual.get("outputInitializationStd", 0.0))
    if (
        hidden_dim <= 0
        or projection_dim <= 0
        or not np.isfinite(maximum)
        or maximum <= 0.0
        or centering not in {"pose_mean", "none"}
        or shortcut not in {"none", "region_linear"}
        or fusion not in {"product", "affine_region"}
        or not np.isfinite(output_init_std)
        or output_init_std < 0.0
    ):
        raise ValueError("checkpoint boundaryTailResidual configuration is invalid")
    return (
        hidden_dim,
        projection_dim,
        maximum,
        centering,
        shortcut,
        fusion,
        output_init_std,
    )


def _build_model_bundle(
    checkpoint_path: Path,
    runtime_meta_path: Path,
    geometry_path: Path,
    dataset: Any,
    device_name: str,
) -> tuple[Any, Any, Any]:
    if torch is None:
        raise RuntimeError("checkpoint extraction requires PyTorch")
    from bounded_relation_survival_moment_model import (
        BoundedRelationSurvivalMomentModel,
        RUNTIME_FEATURE_DIM,
    )
    from common.runtime_meta import load_runtime_meta

    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint.get("testRead") is not False:
        raise ValueError("checkpoint must explicitly declare testRead=false")
    config = checkpoint.get("modelConfig")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no modelConfig")
    world_aabbs, instance_to_glb, _meta = load_runtime_meta(runtime_meta_path)
    num_instances = int(config.get("numInstances", -1))
    if num_instances <= 0 or world_aabbs.shape != (num_instances, 6):
        raise ValueError("runtime metadata instance count disagrees with checkpoint")
    if int(config.get("runtimeFeatureDim", -1)) != int(RUNTIME_FEATURE_DIM):
        raise ValueError("checkpoint runtimeFeatureDim is not the registered v4 dimension")
    if int(dataset.num_instances) != num_instances:
        raise ValueError("dataset num_instances disagrees with checkpoint")
    depth = config.get("depthNormalization")
    instance_calibration = config.get("instanceCalibration")
    if not isinstance(depth, Mapping) or not isinstance(instance_calibration, Mapping):
        raise ValueError("checkpoint depthNormalization or instanceCalibration is missing")
    frequency = config.get("frequency")
    if not isinstance(frequency, Mapping):
        raise ValueError("checkpoint frequency configuration is missing")
    certificate = config.get("cullCertificate")
    certificate_enabled = isinstance(certificate, Mapping) and bool(certificate.get("enabled"))
    view_residual_max_abs, view_residual_hidden_dim = _view_residual_values(config)
    (
        boundary_opportunity_hidden_dim,
        boundary_opportunity_projection_dim,
        boundary_opportunity_initial_logit,
        boundary_opportunity_max_logit_uplift,
    ) = _boundary_opportunity_values(config)
    (
        boundary_tail_residual_hidden_dim,
        boundary_tail_residual_projection_dim,
        boundary_tail_residual_max_abs,
        boundary_tail_residual_centering,
        boundary_tail_residual_shortcut,
        boundary_tail_residual_fusion,
        boundary_tail_residual_output_init_std,
    ) = _boundary_tail_residual_values(config)
    device = torch.device(
        "cuda"
        if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    num_glbs = int(config.get("numGlbs", int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0))
    model = BoundedRelationSurvivalMomentModel(
        num_instances=num_instances,
        num_glbs=num_glbs,
        relation_source=str(config.get("relationSource")),
        spectral_mode=str(config.get("spectralMode")),
        depth_q01=float(depth["q01"]),
        depth_q99=float(depth["q99"]),
        depth_epsilon=float(depth["epsilon"]),
        max_frequency_norm_cycles=float(frequency["maxNormCycles"]),
        instance_calibration_mode=str(instance_calibration.get("mode")),
        instance_calibration_max_abs=float(instance_calibration["maximumAbsoluteResidual"]),
        sparse_instance_penalty=float(instance_calibration["sparseInstancePenalty"]),
        cull_certificate_max_suppression=(
            float(certificate.get("maximumSuppressionLogit"))
            if certificate_enabled
            else 0.0
        ),
        cull_certificate_initial_suppression=(
            float(certificate.get("initialSuppressionLogit", 0.05))
            if certificate_enabled
            else 0.05
        ),
        cull_certificate_hidden_dim=(
            int(certificate.get("hiddenDim", 0)) if certificate_enabled else 0
        ),
        cull_certificate_input_mode=(
            str(certificate.get("inputMode", "hidden")) if certificate_enabled else "hidden"
        ),
        view_residual_max_abs=view_residual_max_abs,
        view_residual_hidden_dim=view_residual_hidden_dim,
        boundary_opportunity_hidden_dim=boundary_opportunity_hidden_dim,
        boundary_opportunity_projection_dim=boundary_opportunity_projection_dim,
        boundary_opportunity_initial_logit=boundary_opportunity_initial_logit,
        boundary_opportunity_max_logit_uplift=(
            boundary_opportunity_max_logit_uplift
        ),
        boundary_tail_residual_hidden_dim=boundary_tail_residual_hidden_dim,
        boundary_tail_residual_projection_dim=(
            boundary_tail_residual_projection_dim
        ),
        boundary_tail_residual_max_abs=boundary_tail_residual_max_abs,
        boundary_tail_residual_centering=boundary_tail_residual_centering,
        boundary_tail_residual_shortcut=boundary_tail_residual_shortcut,
        boundary_tail_residual_fusion=boundary_tail_residual_fusion,
        boundary_tail_residual_output_init_std=(
            boundary_tail_residual_output_init_std
        ),
    ).to(device)
    state = checkpoint.get("modelState")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint has no modelState")
    model.load_state_dict(state, strict=True)
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    coefficients = torch.as_tensor(
        checkpoint.get("instanceSurvivalCoefficients"), dtype=torch.float32, device=device
    )
    if tuple(coefficients.shape) != (num_instances, 4, 7) or not bool(torch.isfinite(coefficients).all()):
        raise ValueError("checkpoint instanceSurvivalCoefficients has invalid shape")
    geometry = torch.from_numpy(_load_geometry(geometry_path, num_instances)).to(device)
    runtime_features = torch.cat([geometry, coefficients.reshape(num_instances, -1)], dim=-1)
    if runtime_features.shape != (num_instances, RUNTIME_FEATURE_DIM):
        raise ValueError("constructed runtime feature table has invalid shape")
    model.eval()
    return model, runtime_features, device


def _row_dataset_inputs(dataset: Any, row: ScoreRow) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    pose = int(row.pose_index)
    camera_norm = np.asarray(dataset.poses["camera_norm"][pose], dtype=np.float32)
    candidate_camera = np.asarray(dataset.candidate_camera_world(pose), dtype=np.float32)
    query_center = np.asarray(dataset.query_center_world(pose, required=True), dtype=np.float32)
    camera_view = np.asarray(dataset.camera_view(pose), dtype=np.float32)
    radius = float(dataset.viewcell_radius_m(pose, required=True))
    return camera_norm, candidate_camera, query_center, camera_view, radius


def _model_scores_for_rows(
    rows: Sequence[ScoreRow],
    dataset: Any,
    model: Any,
    runtime_features: Any,
    device: Any,
    batch_size: int,
) -> None:
    if torch is None:
        raise RuntimeError("model extraction requires PyTorch")
    with torch.no_grad():
        for row in rows:
            if row.scores is not None:
                continue
            if row.targets is None:
                raise ValueError(f"pose {row.pose_index} has no targets for model extraction")
            scores: list[np.ndarray] = []
            camera_norm, candidate_camera, query_center, camera_view, radius = _row_dataset_inputs(
                dataset, row
            )
            for start in range(0, row.candidate_ids.size, max(1, int(batch_size))):
                ids = row.candidate_ids[start : start + max(1, int(batch_size))]
                logits = model.compute_visibility_logits(
                    torch.from_numpy(camera_norm).to(device),
                    torch.from_numpy(camera_view).to(device),
                    torch.from_numpy(candidate_camera).to(device),
                    torch.from_numpy(ids.astype(np.int64)).to(device),
                    runtime_features=runtime_features,
                    query_center_world=torch.from_numpy(query_center).to(device),
                    viewcell_radius_m=torch.tensor(radius, dtype=torch.float32, device=device),
                )
                scores.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
            row.scores = np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float64)


def _model_features_for_tail_rows(
    rows: Sequence[ScoreRow],
    masks: Sequence[np.ndarray],
    dataset: Any,
    model: Any,
    runtime_features: Any,
    device: Any,
    batch_size: int,
) -> FeatureExamples:
    if torch is None:
        raise RuntimeError("model extraction requires PyTorch")
    values_by_family: dict[str, list[np.ndarray]] = {family: [] for family in FEATURE_FAMILIES}
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    visible_weights: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    instances: list[np.ndarray] = []
    with torch.no_grad():
        for row, selected in zip(rows, masks):
            if row.scores is None or row.targets is None or row.visible_weights is None:
                raise ValueError(
                    "model feature extraction requires scores, targets, and visible weights"
                )
            indices = np.flatnonzero(selected)
            if indices.size == 0:
                continue
            camera_norm, candidate_camera, query_center, camera_view, radius = _row_dataset_inputs(
                dataset, row
            )
            for start in range(0, indices.size, max(1, int(batch_size))):
                chosen = indices[start : start + max(1, int(batch_size))]
                ids = row.candidate_ids[chosen]
                _logits, aux = model.compute_logits_with_aux(
                    torch.from_numpy(camera_norm).to(device),
                    torch.from_numpy(camera_view).to(device),
                    torch.from_numpy(candidate_camera).to(device),
                    torch.from_numpy(ids.astype(np.int64)).to(device),
                    runtime_features=runtime_features,
                    query_center_world=torch.from_numpy(query_center).to(device),
                    viewcell_radius_m=torch.tensor(radius, dtype=torch.float32, device=device),
                )
                center = aux["center_view"].detach().cpu().numpy().astype(np.float64)
                spectral = aux["spectral_features"].detach().cpu().numpy().astype(np.float64)
                boundary = aux["boundary_spectral_summary"].detach().cpu().numpy().astype(np.float64)
                axes = aux["disk_axes"].detach().cpu().numpy().astype(np.float64)
                extrema = region_extrema_features(center, axes)
                batches = {
                    "center": center,
                    "mean_std": spectral,
                    "boundary": boundary,
                    "region_extrema": extrema,
                }
                for family, batch_values in batches.items():
                    if not bool(np.isfinite(batch_values).all()):
                        raise FloatingPointError(f"model-generated {family} features are non-finite")
                    values_by_family[family].append(batch_values)
                scores.append(row.scores[chosen])
                labels.append((~row.targets[chosen].astype(bool)).astype(np.float64))
                visible_weights.append(row.visible_weights[chosen].astype(np.float64))
                poses.append(np.full(chosen.size, row.pose_index, dtype=np.int64))
                instances.append(row.candidate_ids[chosen].astype(np.int64))
    if not scores:
        raise ValueError("no difficult-tail candidates selected; check calibration cutoffs and captures")
    return FeatureExamples(
        split="tail",
        scores=np.concatenate(scores),
        targets=np.concatenate(labels),
        features={family: np.concatenate(values) for family, values in values_by_family.items()},
        visible_weights=np.concatenate(visible_weights),
        pose_indices=np.concatenate(poses),
        instance_ids=np.concatenate(instances),
    )


def _embedded_features_for_tail_rows(
    rows: Sequence[ScoreRow], masks: Sequence[np.ndarray], split: str
) -> FeatureExamples:
    values_by_family: dict[str, list[np.ndarray]] = {family: [] for family in FEATURE_FAMILIES}
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    visible_weights: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    instances: list[np.ndarray] = []
    for row, selected in zip(rows, masks):
        if row.scores is None or row.targets is None or row.visible_weights is None:
            raise ValueError(
                f"{split} embedded feature capture has no scores/targets/visibleWeights"
            )
        if row.feature_families is None:
            raise ValueError(f"{split} capture has no embedded features for pose {row.pose_index}")
        features = _canonical_feature_mapping(
            row.feature_families, name=f"{split}/pose{row.pose_index}/featureFamilies"
        )
        indices = np.flatnonzero(selected)
        for family in FEATURE_FAMILIES:
            if features[family].shape[0] != row.scores.size:
                raise ValueError(f"{split} pose {row.pose_index} feature rows are not score-aligned")
            values_by_family[family].append(features[family][indices])
        scores.append(row.scores[indices])
        labels.append((~row.targets[indices].astype(bool)).astype(np.float64))
        visible_weights.append(row.visible_weights[indices].astype(np.float64))
        poses.append(np.full(indices.size, row.pose_index, dtype=np.int64))
        instances.append(row.candidate_ids[indices].astype(np.int64))
    if not scores or not any(values.shape[0] for values in values_by_family["center"]):
        raise ValueError(f"{split} has no difficult-tail candidates")
    return FeatureExamples(
        split=split,
        scores=np.concatenate(scores),
        targets=np.concatenate(labels),
        features={family: np.concatenate(values) for family, values in values_by_family.items()},
        visible_weights=np.concatenate(visible_weights),
        pose_indices=np.concatenate(poses),
        instance_ids=np.concatenate(instances),
    )


def _capture_examples(
    payload: Mapping[str, Any],
    *,
    split: str,
    dataset: Any | None,
    model_bundle: tuple[Any, Any, Any] | None,
    cutoffs: Mapping[str, Any],
    batch_size: int,
) -> FeatureExamples:
    rows = _resolve_row_arrays(payload["_normalizedRows"], dataset=dataset, split=split)
    if any(row.scores is None for row in rows):
        if model_bundle is None or dataset is None:
            raise ValueError(
                f"{split} capture lacks persisted scores; provide checkpoint+dataset extraction inputs"
            )
        _model_scores_for_rows(rows, dataset, *model_bundle, batch_size)
    masks = [
        (
            row.targets.astype(bool)
            & (row.scores <= float(cutoffs["lowPositiveScoreCutoff"]))
        )
        | (
            ~row.targets.astype(bool)
            & (row.scores >= float(cutoffs["highNegativeScoreCutoff"]))
        )
        for row in rows
    ]
    if all(row.feature_families is not None for row in rows):
        return _embedded_features_for_tail_rows(rows, masks, split)
    if model_bundle is None or dataset is None:
        raise ValueError(
            f"{split} persisted scores have no input features; provide --checkpoint, --dataset-dir, "
            "--runtime-meta, and --initial-geo-features"
        )
    model, runtime_features, device = model_bundle
    return _model_features_for_tail_rows(
        rows, masks, dataset, model, runtime_features, device, batch_size
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-capture", type=Path, required=True)
    parser.add_argument("--validation-capture", type=Path, required=True)
    parser.add_argument("--train-capture", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--runtime-meta", type=Path, default=None)
    parser.add_argument("--initial-geo-features", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--positive-tail-quantile", type=float, default=0.01)
    parser.add_argument(
        "--positive-tail-mode",
        choices=("count", "weighted_mass"),
        default="count",
    )
    parser.add_argument("--negative-tail-quantile", type=float, default=0.99)
    parser.add_argument("--probe-fit-split", choices=("calibration", "train"), default="calibration")
    parser.add_argument(
        "--probe-fit-tail-source",
        choices=("calibration", "train"),
        default="calibration",
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--max-probe-samples", type=int, default=100000)
    parser.add_argument(
        "--probe-sample-weight-power",
        "--sample-weight-power",
        type=float,
        default=DEFAULT_PROBE_SAMPLE_WEIGHT_POWER,
        help="train-only low-positive visible-mass power; 0 preserves unweighted ridge",
    )
    parser.add_argument(
        "--probe-sample-weight-epsilon",
        "--sample-weight-epsilon",
        type=float,
        default=DEFAULT_PROBE_SAMPLE_WEIGHT_EPSILON,
    )
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.batch_size) <= 0 or int(args.max_probe_samples) == 0:
        raise ValueError("batch-size must be positive and max-probe-samples must be non-zero")
    if (
        not np.isfinite(args.probe_sample_weight_power)
        or float(args.probe_sample_weight_power) < 0.0
        or not np.isfinite(args.probe_sample_weight_epsilon)
        or float(args.probe_sample_weight_epsilon) <= 0.0
    ):
        raise ValueError("probe sample weighting parameters must be finite and valid")
    calibration_payload = load_capture(args.calibration_capture.resolve(), "calibration")
    validation_payload = load_capture(args.validation_capture.resolve(), "validation")
    train_payload = (
        None
        if args.train_capture is None
        else load_capture(args.train_capture.resolve(), "train")
    )
    provided_runtime_args = [
        args.checkpoint,
        args.dataset_dir,
        args.runtime_meta,
        args.initial_geo_features,
    ]
    if any(value is not None for value in provided_runtime_args) and not all(
        value is not None for value in provided_runtime_args
    ):
        raise ValueError(
            "checkpoint extraction requires all of --checkpoint, --dataset-dir, "
            "--runtime-meta, and --initial-geo-features"
        )

    dataset = None
    model_bundle = None
    if all(value is not None for value in provided_runtime_args):
        from pose_csr_dataset import PoseCSRDataset

        checkpoint = _load_checkpoint(args.checkpoint.resolve())
        config = checkpoint.get("modelConfig")
        if not isinstance(config, Mapping):
            raise ValueError("checkpoint has no modelConfig")
        dataset = PoseCSRDataset(args.dataset_dir.resolve(), num_instances=int(config["numInstances"]))
        model_bundle = _build_model_bundle(
            args.checkpoint.resolve(),
            args.runtime_meta.resolve(),
            args.initial_geo_features.resolve(),
            dataset,
            args.device,
        )

    calibration_rows = _resolve_row_arrays(
        calibration_payload["_normalizedRows"], dataset=dataset, split="calibration"
    )
    if any(row.scores is None for row in calibration_rows):
        if model_bundle is None or dataset is None:
            raise ValueError("calibration capture lacks scores and no checkpoint extraction inputs were supplied")
        _model_scores_for_rows(calibration_rows, dataset, *model_bundle, int(args.batch_size))
    calibration_scores = np.concatenate(
        [row.scores for row in calibration_rows if row.scores is not None]
    )
    calibration_targets = np.concatenate(
        [row.targets for row in calibration_rows if row.targets is not None]
    )
    calibration_visible_weights = np.concatenate(
        [
            row.visible_weights
            for row in calibration_rows
            if row.visible_weights is not None
        ]
    )
    cutoffs = freeze_tail_cutoffs(
        calibration_scores,
        calibration_targets,
        visible_weights=calibration_visible_weights,
        positive_tail_mode=args.positive_tail_mode,
        positive_quantile=args.positive_tail_quantile,
        negative_quantile=args.negative_tail_quantile,
    )

    fit_cutoffs = cutoffs
    train_rows: list[ScoreRow] | None = None
    if train_payload is not None:
        train_rows = _resolve_row_arrays(
            train_payload["_normalizedRows"], dataset=dataset, split="train"
        )
        if any(row.scores is None for row in train_rows):
            if model_bundle is None or dataset is None:
                raise ValueError(
                    "train capture lacks scores and no checkpoint extraction inputs were supplied"
                )
            _model_scores_for_rows(
                train_rows, dataset, *model_bundle, int(args.batch_size)
            )
    if args.probe_fit_tail_source == "train":
        if train_rows is None:
            raise ValueError("probe-fit-tail-source=train requires --train-capture")
        fit_cutoffs = freeze_tail_cutoffs(
            np.concatenate([row.scores for row in train_rows if row.scores is not None]),
            np.concatenate([row.targets for row in train_rows if row.targets is not None]),
            visible_weights=np.concatenate(
                [
                    row.visible_weights
                    for row in train_rows
                    if row.visible_weights is not None
                ]
            ),
            positive_tail_mode=args.positive_tail_mode,
            positive_quantile=args.positive_tail_quantile,
            negative_quantile=args.negative_tail_quantile,
            source_split="train",
        )

    def table(payload: Mapping[str, Any], split: str) -> FeatureExamples:
        return _capture_examples(
            payload,
            split=split,
            dataset=dataset,
            model_bundle=model_bundle,
            cutoffs=cutoffs,
            batch_size=int(args.batch_size),
        )

    calibration_tail = table(calibration_payload, "calibration")
    validation_tail = table(validation_payload, "validation")
    train_tail = (
        None
        if train_payload is None
        else _capture_examples(
            train_payload,
            split="train",
            dataset=dataset,
            model_bundle=model_bundle,
            cutoffs=fit_cutoffs,
            batch_size=int(args.batch_size),
        )
    )
    payload = analyze_tail_examples(
        calibration_tail,
        validation_tail,
        train=train_tail,
        cutoffs=cutoffs,
        fit_cutoffs=fit_cutoffs,
        probe_fit_split=args.probe_fit_split,
        ridge=args.ridge,
        max_probe_samples=args.max_probe_samples,
        seed=args.seed,
        sample_weight_power=args.probe_sample_weight_power,
        sample_weight_epsilon=args.probe_sample_weight_epsilon,
    )
    payload["inputProvenance"] = {
        "calibrationCapture": str(args.calibration_capture.resolve()),
        "validationCapture": str(args.validation_capture.resolve()),
        "trainCapture": None if args.train_capture is None else str(args.train_capture.resolve()),
        "featureSource": "checkpoint_dataset_v4_query" if model_bundle is not None else "persisted_featureFamilies",
        "testRead": False,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
