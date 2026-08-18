#!/usr/bin/env python3
"""Summarize the three-seed nonlinear tail-posterior ablation.

The input scans already contain calibration-frozen workpoints.  This script
does not select thresholds.  It aligns validation poses, reports pooled
pose/aggregate metrics, and performs a paired clustered bootstrap: seeds are
sampled first, then the same pose indices are sampled within each selected seed
for every compared posterior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "pvs-train-owned-nonlinear-tail-posterior-longtrain-summary-v1"
FORMAL_SEEDS = (20260801, 20260802, 20260803)
VARIANTS = ("no_posterior", "linear_dual_probe", "hinge_mlp_dual_probe")
COMPARISONS = (
    ("linear_minus_none", "linear_dual_probe", "no_posterior"),
    ("nonlinear_minus_none", "hinge_mlp_dual_probe", "no_posterior"),
    ("nonlinear_minus_linear", "hinge_mlp_dual_probe", "linear_dual_probe"),
)
COUNT_FIELDS = ("tp", "fp", "fn", "tn", "weightedTp", "weightedGt")
POSE_FIELDS = (
    "precision",
    "recall",
    "weightedRecall",
    "f1",
    "jaccard",
    "accuracy",
    "balancedAccuracy",
    "specificity",
    "usefulCull",
    "badCull",
)
MEAN_FIELDS = (
    "predCount",
    "candidateCount",
    "gtCount",
    "predictedGlbCount",
    "predictedGlbBytes",
    "glbCountReduction",
    "glbByteReduction",
    "downloadUtilityRecall",
)
BOOTSTRAP_METRICS = (
    "posePrecision",
    "poseRecall",
    "poseWeightedRecall",
    "poseF1",
    "poseAccuracy",
    "poseBalancedAccuracy",
    "poseSpecificity",
    "poseUsefulCull",
    "poseBadCull",
    "aggregatePrecision",
    "aggregateRecall",
    "aggregateWeightedRecall",
    "aggregateF1",
    "aggregateAccuracy",
    "aggregateBalancedAccuracy",
    "aggregateSpecificity",
    "aggregateUsefulCull",
    "aggregateBadCull",
    "avgPredCount",
    "glbCountReduction",
    "glbByteReduction",
    "predictedGlbBytes",
)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("testRead") is not False:
        raise ValueError(f"scan is not a test-free JSON object: {path}")
    return payload


def _select_member(payload: Mapping[str, Any], *, alpha: float) -> dict[str, Any]:
    members = payload.get("members")
    if not isinstance(members, list):
        raise ValueError("scan has no members")
    selected = [
        member
        for member in members
        if isinstance(member, Mapping)
        and member.get("mode") == "dual_rescue"
        and np.isclose(float(member.get("alpha", -1.0)), float(alpha))
        and np.isclose(float(member.get("temperature", -1.0)), 0.05)
        and np.isclose(float(member.get("coverageWeight", -1.0)), 1.0)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one dual-rescue alpha={alpha} member")
    member = dict(selected[0])
    calibration = member.get("calibration")
    validation = member.get("validation")
    if not isinstance(calibration, Mapping) or calibration.get("status") != "safe":
        raise ValueError(f"member has no qualified calibration safety workpoint: alpha={alpha}")
    if not isinstance(validation, Mapping):
        raise ValueError(f"safe member has no validation replay: alpha={alpha}")
    return member


def _pose_table(member: Mapping[str, Any]) -> dict[str, np.ndarray]:
    validation = member["validation"]
    rows = validation.get("perPose")
    if not isinstance(rows, list) or not rows:
        raise ValueError("validation member lacks perPose statistics")
    pose_ids = np.asarray([int(row["poseIndex"]) for row in rows], dtype=np.int64)
    if np.unique(pose_ids).size != pose_ids.size:
        raise ValueError("validation pose indices are not unique")
    table: dict[str, np.ndarray] = {"poseIndex": pose_ids}
    for field in COUNT_FIELDS + POSE_FIELDS + MEAN_FIELDS:
        values = np.asarray([row.get(field, np.nan) for row in rows], dtype=np.float64)
        if values.shape != pose_ids.shape or not bool(np.isfinite(values).all()):
            raise ValueError(f"validation perPose field is missing or non-finite: {field}")
        table[field] = values
    return table


def load_members(input_root: Path) -> tuple[dict[str, list[dict[str, np.ndarray]]], list[dict[str, Any]]]:
    tables = {variant: [] for variant in VARIANTS}
    records: list[dict[str, Any]] = []
    reference_pose_ids: np.ndarray | None = None
    for seed in FORMAL_SEEDS:
        seed_root = input_root.resolve() / f"seed{seed}"
        linear_payload = _load_json(seed_root / "linear_dual_probe_formal.json")
        nonlinear_payload = _load_json(seed_root / "nonlinear_dual_probe_formal.json")
        members = {
            "no_posterior": _select_member(nonlinear_payload, alpha=0.0),
            "linear_dual_probe": _select_member(linear_payload, alpha=0.5),
            "hinge_mlp_dual_probe": _select_member(nonlinear_payload, alpha=0.5),
        }
        seed_record: dict[str, Any] = {"seed": seed, "variants": {}, "testRead": False}
        for variant, member in members.items():
            table = _pose_table(member)
            if reference_pose_ids is None:
                reference_pose_ids = table["poseIndex"]
            elif not np.array_equal(table["poseIndex"], reference_pose_ids):
                raise ValueError(f"validation pose alignment differs for seed={seed}, variant={variant}")
            tables[variant].append(table)
            seed_record["variants"][variant] = {
                "threshold": float(member["validation"]["threshold"]),
                "calibration": dict(member["calibration"]["selected"]),
                "validation": {
                    "poseMacro": dict(member["validation"]["poseMacro"]),
                    "aggregate": dict(member["validation"]["aggregate"]),
                    "weightedRecallLowerConfidenceBound": float(
                        member["validation"]["weightedRecallLowerConfidenceBound"]
                    ),
                },
                "testRead": False,
            }
        records.append(seed_record)
    return tables, records


def _empty_accumulator() -> dict[str, float]:
    return {field: 0.0 for field in COUNT_FIELDS + POSE_FIELDS + MEAN_FIELDS} | {
        "poseCount": 0.0
    }


def _accumulate(
    tables: Sequence[Mapping[str, np.ndarray]],
    selections: Sequence[tuple[int, np.ndarray]],
) -> dict[str, float]:
    result = _empty_accumulator()
    for seed_index, pose_indices in selections:
        table = tables[int(seed_index)]
        count = int(pose_indices.size)
        result["poseCount"] += count
        for field in COUNT_FIELDS + POSE_FIELDS + MEAN_FIELDS:
            result[field] += float(np.asarray(table[field])[pose_indices].sum())
    return result


def metrics_from_accumulator(values: Mapping[str, float]) -> dict[str, float]:
    count = float(values["poseCount"])
    tp, fp, fn, tn = (float(values[field]) for field in ("tp", "fp", "fn", "tn"))
    candidate = tp + fp + fn + tn
    recall = _safe_div(tp, tp + fn, 1.0)
    precision = _safe_div(tp, tp + fp, 1.0)
    specificity = _safe_div(tn, tn + fp, 1.0)
    aggregate = {
        "aggregatePrecision": precision,
        "aggregateRecall": recall,
        "aggregateWeightedRecall": _safe_div(
            float(values["weightedTp"]), float(values["weightedGt"]), 1.0
        ),
        "aggregateF1": _safe_div(2.0 * precision * recall, precision + recall),
        "aggregateJaccard": _safe_div(tp, tp + fp + fn),
        "aggregateAccuracy": _safe_div(tp + tn, candidate, 1.0),
        "aggregateBalancedAccuracy": 0.5 * (recall + specificity),
        "aggregateSpecificity": specificity,
        "aggregateUsefulCull": _safe_div(tn, candidate),
        "aggregateBadCull": _safe_div(fn, candidate),
    }
    pose = {
        f"pose{field[0].upper()}{field[1:]}": _safe_div(float(values[field]), count)
        for field in POSE_FIELDS
    }
    means = {
        "avgPredCount": _safe_div(float(values["predCount"]), count),
        "avgCandidateCount": _safe_div(float(values["candidateCount"]), count),
        "avgGtCount": _safe_div(float(values["gtCount"]), count),
        "predictedGlbCount": _safe_div(float(values["predictedGlbCount"]), count),
        "predictedGlbBytes": _safe_div(float(values["predictedGlbBytes"]), count),
        "glbCountReduction": _safe_div(float(values["glbCountReduction"]), count),
        "glbByteReduction": _safe_div(float(values["glbByteReduction"]), count),
        "downloadUtilityRecall": _safe_div(float(values["downloadUtilityRecall"]), count),
    }
    return {**pose, **aggregate, **means}


def observed_metrics(tables: Sequence[Mapping[str, np.ndarray]]) -> dict[str, float]:
    selections = [
        (seed_index, np.arange(table["poseIndex"].size, dtype=np.int64))
        for seed_index, table in enumerate(tables)
    ]
    return metrics_from_accumulator(_accumulate(tables, selections))


def paired_cluster_bootstrap(
    tables: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(int(seed))
    samples = {
        comparison_id: {metric: np.empty(replicates, dtype=np.float64) for metric in BOOTSTRAP_METRICS}
        for comparison_id, _left, _right in COMPARISONS
    }
    seed_count = len(FORMAL_SEEDS)
    pose_count = int(tables[VARIANTS[0]][0]["poseIndex"].size)
    for replicate in range(replicates):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        selections = [
            (
                int(seed_index),
                rng.integers(0, pose_count, size=pose_count, dtype=np.int64),
            )
            for seed_index in sampled_seeds
        ]
        replicate_metrics = {
            variant: metrics_from_accumulator(_accumulate(tables[variant], selections))
            for variant in VARIANTS
        }
        for comparison_id, left, right in COMPARISONS:
            for metric in BOOTSTRAP_METRICS:
                samples[comparison_id][metric][replicate] = (
                    replicate_metrics[left][metric] - replicate_metrics[right][metric]
                )

    result: dict[str, Any] = {}
    observed = {variant: observed_metrics(tables[variant]) for variant in VARIANTS}
    for comparison_id, left, right in COMPARISONS:
        metric_rows: dict[str, Any] = {}
        for metric in BOOTSTRAP_METRICS:
            difference = float(observed[left][metric] - observed[right][metric])
            lower, upper = np.quantile(samples[comparison_id][metric], [0.025, 0.975])
            metric_rows[metric] = {
                "difference": difference,
                "confidenceInterval95": [float(lower), float(upper)],
                "direction": "positive" if difference > 0.0 else "negative" if difference < 0.0 else "zero",
                "crossesZero": bool(lower <= 0.0 <= upper),
            }
        result[comparison_id] = {
            "left": left,
            "right": right,
            "metrics": metric_rows,
        }
    return {
        "replicates": int(replicates),
        "confidenceInterval": 0.95,
        "method": "sample seeds with replacement, then paired validation poses within each sampled seed",
        "comparisons": result,
    }


def summarize(input_root: Path, *, replicates: int, seed: int) -> dict[str, Any]:
    tables, seed_records = load_members(input_root)
    return {
        "schema": SCHEMA,
        "experiment": "pvs_train_owned_nonlinear_tail_posterior_longtrain_v1",
        "inputRoot": str(input_root.resolve()),
        "variants": {
            variant: {"pooledValidation": observed_metrics(tables[variant])}
            for variant in VARIANTS
        },
        "seedMembers": seed_records,
        "pairedBootstrap": paired_cluster_bootstrap(
            tables, replicates=int(replicates), seed=int(seed)
        ),
        "safetyGate": {
            "selectionSplit": "calibration",
            "weightedRecall": ">0.99",
            "oneSided95LowerConfidenceBound": ">0.99",
            "poseRecallRole": "diagnostic_only",
        },
        "imageMetrics": {
            "status": "not_available",
            "reason": "Color-ID image evaluation is a separate formal stage",
        },
        "candidateSetChanged": False,
        "groundTruthChanged": False,
        "testRead": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = summarize(
        args.input_root,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
