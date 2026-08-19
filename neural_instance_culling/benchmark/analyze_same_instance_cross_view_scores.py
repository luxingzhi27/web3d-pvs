#!/usr/bin/env python3
"""Diagnose whether one instance is separated across visible/invisible views.

The input is a validation capture produced by
``evaluate_pvs_bounded_relation_survival_moment_v4.py`` with both
``--persist-ids`` and ``--persist-scores``.  The analysis is read-only and
rejects test provenance.  It measures score ordering only between opposite
labels of the same instance, so static per-instance score biases cannot make
the result look better.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "pvs-same-instance-cross-view-score-diagnostic-v1"


def _load_capture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation capture must contain a JSON object")
    if payload.get("testRead") is not False or payload.get("split") == "test":
        raise ValueError("test captures are forbidden")
    if payload.get("split") != "validation":
        raise ValueError("same-instance formal diagnostics require validation")
    rows = payload.get("perPose")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation capture has no perPose rows")
    return dict(payload)


def _aligned_row_arrays(
    row: Mapping[str, Any], row_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = []
    for key, dtype in (
        ("candidateIds", np.int64),
        ("candidateScores", np.float64),
        ("targets", np.float64),
        ("visibleWeights", np.float64),
    ):
        if row.get(key) is None:
            raise ValueError(
                f"perPose[{row_index}].{key} is missing; rerun with "
                "--persist-ids --persist-scores"
            )
        array = np.asarray(row[key], dtype=dtype).reshape(-1)
        if array.size == 0 or not bool(np.isfinite(array).all()):
            raise ValueError(f"perPose[{row_index}].{key} is empty or non-finite")
        values.append(array)
    ids, scores, targets, weights = values
    if not (ids.size == scores.size == targets.size == weights.size):
        raise ValueError(f"perPose[{row_index}] candidate arrays are not aligned")
    if bool(np.any(ids < 0)) or not bool(np.all((targets == 0.0) | (targets == 1.0))):
        raise ValueError(f"perPose[{row_index}] has invalid IDs or targets")
    if bool(np.any(weights < 0.0)):
        raise ValueError(f"perPose[{row_index}] has negative visible weights")
    return ids, scores, targets, weights


def _pair_credit(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> np.ndarray:
    sorted_negative = np.sort(negative_scores)
    lower = np.searchsorted(sorted_negative, positive_scores, side="left")
    upper = np.searchsorted(sorted_negative, positive_scores, side="right")
    return lower.astype(np.float64) + 0.5 * (upper - lower)


def analyze_capture(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload.get("perPose")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation capture has no perPose rows")
    threshold = float(payload.get("threshold"))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("evaluation threshold must be finite and in [0, 1]")

    ids_parts = []
    score_parts = []
    target_parts = []
    weight_parts = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"perPose[{index}] is not an object")
        ids, scores, targets, weights = _aligned_row_arrays(raw, index)
        ids_parts.append(ids)
        score_parts.append(scores)
        target_parts.append(targets)
        weight_parts.append(weights)
    ids = np.concatenate(ids_parts)
    scores = np.concatenate(score_parts)
    targets = np.concatenate(target_parts)
    weights = np.concatenate(weight_parts)

    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    scores = scores[order]
    targets = targets[order]
    weights = weights[order]
    boundaries = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])

    pair_credit = 0.0
    pair_count = 0
    weighted_credit = 0.0
    weighted_pair_mass = 0.0
    mixed_count = 0
    ordered_mean_count = 0
    mean_gaps = []
    positive_instance_ids: set[int] = set()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group_targets = targets[start:end] > 0.5
        instance_id = int(ids[start])
        if bool(group_targets.any()):
            positive_instance_ids.add(instance_id)
        if not bool(group_targets.any()) or bool(group_targets.all()):
            continue
        positive = scores[start:end][group_targets]
        negative = scores[start:end][~group_targets]
        positive_weights = np.sqrt(
            np.maximum(weights[start:end][group_targets], 1e-12)
        )
        credit = _pair_credit(positive, negative)
        pairs = int(positive.size * negative.size)
        pair_credit += float(credit.sum())
        pair_count += pairs
        weighted_credit += float(np.dot(credit, positive_weights))
        weighted_pair_mass += float(negative.size * positive_weights.sum())
        gap = float(positive.mean() - negative.mean())
        mean_gaps.append(gap)
        mixed_count += 1
        ordered_mean_count += int(gap > 0.0)

    if pair_count <= 0 or weighted_pair_mass <= 0.0 or mixed_count <= 0:
        raise ValueError("capture contains no instances with opposite labels")

    false_positive = (targets < 0.5) & (scores >= threshold)
    if bool(false_positive.any()):
        other_view_visible = np.fromiter(
            (int(value) in positive_instance_ids for value in ids[false_positive]),
            dtype=bool,
            count=int(false_positive.sum()),
        )
        other_view_fraction = float(other_view_visible.mean())
    else:
        other_view_fraction = 0.0
    gaps = np.asarray(mean_gaps, dtype=np.float64)
    return {
        "schema": SCHEMA,
        "split": "validation",
        "testRead": False,
        "threshold": threshold,
        "poseCount": len(rows),
        "candidateRowCount": int(ids.size),
        "uniqueInstanceCount": int(boundaries.size - 1),
        "mixedLabelInstanceCount": mixed_count,
        "sameInstancePairCount": pair_count,
        "sameInstancePairAuc": pair_credit / pair_count,
        "visibleWeightSameInstancePairAuc": weighted_credit / weighted_pair_mass,
        "meanPositiveMinusNegativeScore": float(gaps.mean()),
        "medianPositiveMinusNegativeScore": float(np.median(gaps)),
        "meanOrderedInstanceFraction": ordered_mean_count / mixed_count,
        "meanOrderViolationInstanceFraction": 1.0 - ordered_mean_count / mixed_count,
        "falsePositiveCount": int(false_positive.sum()),
        "falsePositiveFromOtherVisibleViewFraction": other_view_fraction,
        "falsePositiveNeverVisibleInCaptureFraction": 1.0 - other_view_fraction,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = analyze_capture(_load_capture(args.capture.resolve()))
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
