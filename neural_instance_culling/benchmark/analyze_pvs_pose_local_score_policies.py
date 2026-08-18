#!/usr/bin/env python3
"""Diagnose whether pose-local score normalization can repair a global PVS threshold.

The input captures must come from calibration and validation only.  Every
deployable policy freezes one scalar operating point on calibration, then
replays validation once.  Oracle rows use validation labels and are reported
only as ranking upper bounds; they are never eligible operating policies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from current_pvs_utils import weighted_recall_lower_confidence_bound  # noqa: E402


SCHEMA = "pvs-pose-local-score-policy-diagnosis-v1"


def _load_capture(path: Path, split: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("testRead") is not False or payload.get("split") != split:
        raise ValueError(f"capture must be an explicit {split} split with testRead=false")
    rows = payload.get("perPose")
    if not isinstance(rows, list) or not rows:
        raise ValueError("capture has no per-pose rows")
    return payload


def _pose_arrays(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(row.get("candidateScores"), dtype=np.float64).reshape(-1)
    targets = np.asarray(row.get("targets"), dtype=np.uint8).reshape(-1).astype(bool)
    weights = np.asarray(row.get("visibleWeights"), dtype=np.float64).reshape(-1)
    if not (scores.size == targets.size == weights.size):
        raise ValueError("per-pose score, target, and weight rows must align")
    if not bool(np.isfinite(scores).all() and np.isfinite(weights).all()):
        raise ValueError("per-pose scores and weights must be finite")
    if bool(np.any((scores < 0.0) | (scores > 1.0))) or bool(np.any(weights < 0.0)):
        raise ValueError("scores must be probabilities and weights non-negative")
    return scores, targets, weights


def _logits(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-8, 1.0 - 1e-8)
    return np.log(clipped) - np.log1p(-clipped)


def _transform_pose_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    if scores.size == 0:
        return scores.copy()
    logits = _logits(scores)
    if mode == "raw_logit":
        return logits
    if mode == "robust_logit":
        q10, median, q90 = np.quantile(logits, [0.10, 0.50, 0.90])
        scale = max(float(q90 - q10), 1e-3)
        return (logits - float(median)) / scale
    if mode == "rank_percentile":
        order = np.argsort(-logits, kind="stable")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(order.size, dtype=np.float64)
        return 1.0 - ranks / max(1.0, float(order.size - 1))
    raise ValueError(f"unknown score policy mode: {mode}")


def _prepared_rows(
    capture: Mapping[str, Any], mode: str
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result = []
    for row in capture["perPose"]:
        scores, targets, weights = _pose_arrays(row)
        result.append((_transform_pose_scores(scores, mode), targets, weights))
    return result


def _metrics(
    rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
    *,
    bootstrap_replicates: int = 0,
    seed: int = 20260801,
) -> dict[str, float | int | None]:
    tp = fp = fn = tn = 0
    weighted_tp_rows: list[float] = []
    weighted_gt_rows: list[float] = []
    pred_count = candidate_count = gt_count = 0
    for scores, targets, weights in rows:
        predicted = scores >= float(threshold)
        tp += int(np.logical_and(predicted, targets).sum())
        fp += int(np.logical_and(predicted, ~targets).sum())
        fn += int(np.logical_and(~predicted, targets).sum())
        tn += int(np.logical_and(~predicted, ~targets).sum())
        weighted_tp_rows.append(float(weights[np.logical_and(predicted, targets)].sum()))
        weighted_gt_rows.append(float(weights[targets].sum()))
        pred_count += int(predicted.sum())
        candidate_count += int(targets.size)
        gt_count += int(targets.sum())
    total = max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    weighted_tp = float(np.sum(weighted_tp_rows))
    weighted_gt = float(np.sum(weighted_gt_rows))
    weighted_recall = weighted_tp / weighted_gt if weighted_gt > 0.0 else 1.0
    lcb = (
        weighted_recall_lower_confidence_bound(
            np.asarray(weighted_tp_rows, dtype=np.float64),
            np.asarray(weighted_gt_rows, dtype=np.float64),
            replicates=int(bootstrap_replicates),
            seed=int(seed),
        )
        if int(bootstrap_replicates) > 0
        else None
    )
    pose_count = max(1, len(rows))
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "weightedRecall": float(weighted_recall),
        "weightedRecallLowerConfidenceBound": None if lcb is None else float(lcb),
        "accuracy": float((tp + tn) / total),
        "balancedAccuracy": float(0.5 * (recall + specificity)),
        "specificity": float(specificity),
        "usefulCull": float(tn / total),
        "badCull": float(fn / total),
        "avgPredCount": float(pred_count / pose_count),
        "avgCandidateCount": float(candidate_count / pose_count),
        "avgGtCount": float(gt_count / pose_count),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _calibrate(
    rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    pooled = np.concatenate([scores for scores, _targets, _weights in rows if scores.size])
    quantiles = np.linspace(0.0, 1.0, 513, dtype=np.float64)
    thresholds = np.unique(
        np.concatenate(
            ([float("-inf")], np.quantile(pooled, quantiles), [float("inf")])
        )
    )[::-1]
    for threshold in thresholds:
        point = _metrics(rows, float(threshold))
        if float(point["weightedRecall"]) <= 0.99:
            continue
        checked = _metrics(
            rows,
            float(threshold),
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        if float(checked["weightedRecallLowerConfidenceBound"] or 0.0) > 0.99:
            return checked
    raise RuntimeError("score policy has no qualified calibration safety workpoint")


def _oracle_pose_weighted_rows(
    capture: Mapping[str, Any], target_weighted_recall: float
) -> dict[str, float | int | None]:
    if not 0.0 < float(target_weighted_recall) <= 1.0:
        raise ValueError("oracle target weighted recall must lie in (0, 1]")
    transformed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for row in capture["perPose"]:
        scores, targets, weights = _pose_arrays(row)
        predicted = np.zeros(scores.size, dtype=np.float64)
        weighted_gt = float(weights[targets].sum())
        if scores.size and weighted_gt > 0.0:
            order = np.argsort(-scores, kind="stable")
            cumulative = np.cumsum(np.where(targets[order], weights[order], 0.0))
            needed = float(target_weighted_recall) * weighted_gt
            k = min(scores.size, int(np.searchsorted(cumulative, needed, side="left")) + 1)
            predicted[order[:k]] = 1.0
        transformed.append((predicted, targets, weights))
    metrics = _metrics(transformed, 0.5)
    metrics["oracleTargetPoseWeightedRecall"] = float(target_weighted_recall)
    return metrics


def analyze(
    calibration: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for mode in ("raw_logit", "robust_logit", "rank_percentile"):
        calibration_rows = _prepared_rows(calibration, mode)
        validation_rows = _prepared_rows(validation, mode)
        selected = _calibrate(
            calibration_rows,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        policies[mode] = {
            "deployable": True,
            "calibration": selected,
            "validation": _metrics(validation_rows, float(selected["threshold"])),
        }
    return {
        "schema": SCHEMA,
        "selectionRule": "calibration WR > 0.99 and one-sided 95% pose-bootstrap LCB > 0.99; choose highest policy threshold",
        "bootstrapReplicates": int(bootstrap_replicates),
        "policies": policies,
        "oracleUpperBounds": {
            "poseWeightedRecall99": _oracle_pose_weighted_rows(validation, 0.99),
            "poseWeightedRecall100": _oracle_pose_weighted_rows(validation, 1.0),
        },
        "calibrationPoseCount": len(calibration["perPose"]),
        "validationPoseCount": len(validation["perPose"]),
        "testRead": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-capture", type=Path, required=True)
    parser.add_argument("--validation-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    calibration = _load_capture(args.calibration_capture.resolve(), "calibration")
    validation = _load_capture(args.validation_capture.resolve(), "validation")
    payload = analyze(
        calibration,
        validation,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "testRead": False}))


if __name__ == "__main__":
    main()
