#!/usr/bin/env python3
"""Summarize the three-seed joint query-tail separator ablation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from run_pvs_joint_108d_query_tail_separator_from_scratch_v1 import (
    FORMAL_SEEDS,
    OUTPUT_TAG,
    ROOT,
    VARIANT_FAMILIES,
    _member_name,
)


SCHEMA = "pvs-joint-108d-query-tail-separator-from-scratch-summary-v1"
BOOTSTRAP_REPLICATES = 10_000
BASELINE = "without_query_tail_separator"
METRICS = (
    "precision",
    "recall",
    "weightedRecall",
    "accuracy",
    "balancedAccuracy",
    "specificity",
    "f1",
    "usefulCull",
    "badCull",
    "avgPredCount",
    "predictedGlbBytes",
    "glbByteReduction",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_metrics(row: Mapping[str, Any], path: Path) -> np.ndarray:
    try:
        values = np.asarray([float(row[name]) for name in METRICS], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing metric in {path}") from exc
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"non-finite metrics in {path}")
    return values


def load_matrix(benchmark_root: Path) -> tuple[dict[str, dict[int, dict[str, Any]]], list[int]]:
    matrix: dict[str, dict[int, dict[str, Any]]] = {}
    expected_poses: list[int] | None = None
    candidate_reference: list[list[int]] | None = None
    gt_count_reference: np.ndarray | None = None
    weighted_gt_reference: np.ndarray | None = None
    for variant, _ in VARIANT_FAMILIES:
        matrix[variant] = {}
        for seed in FORMAL_SEEDS:
            path = (
                benchmark_root
                / "members"
                / _member_name(variant, seed)
                / "validation_evaluation.json"
            )
            payload = _load_json(path)
            if (
                payload.get("split") != "validation"
                or payload.get("testRead") is not False
                or str(payload.get("variant")) != variant
                or int(payload.get("seed", -1)) != seed
            ):
                raise ValueError(f"evaluation identity mismatch: {path}")
            rows = payload.get("perPose")
            if not isinstance(rows, list) or len(rows) != 213:
                raise ValueError(f"evaluation must contain all 213 validation poses: {path}")
            poses = [int(row["poseIndex"]) for row in rows]
            if expected_poses is None:
                expected_poses = poses
            elif poses != expected_poses:
                raise ValueError(f"validation pose order mismatch: {path}")
            candidates = [list(map(int, row["candidateIds"])) for row in rows]
            gt_counts = np.asarray(
                [float(row["metrics"]["tp"]) + float(row["metrics"]["fn"]) for row in rows],
                dtype=np.float64,
            )
            weighted_gt = np.asarray(
                [float(row["metrics"]["weightedGt"]) for row in rows],
                dtype=np.float64,
            )
            if candidate_reference is None:
                candidate_reference = candidates
                gt_count_reference = gt_counts
                weighted_gt_reference = weighted_gt
            elif (
                candidates != candidate_reference
                or not np.array_equal(gt_counts, gt_count_reference)
                or not np.allclose(weighted_gt, weighted_gt_reference, rtol=0.0, atol=0.0)
            ):
                raise ValueError(f"candidate or GT semantics changed across members: {path}")
            for row in rows:
                _finite_metrics(row["metrics"], path)
            matrix[variant][seed] = payload
    assert expected_poses is not None
    return matrix, expected_poses


def _paired_cluster_bootstrap(
    variant_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if variant_values.shape != baseline_values.shape or variant_values.ndim != 3:
        raise ValueError("paired bootstrap arrays must have shape [seed, pose, metric]")
    seed_count, pose_count, metric_count = variant_values.shape
    if metric_count != len(METRICS):
        raise ValueError("paired bootstrap metric width mismatch")
    differences = variant_values - baseline_values
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(replicates), metric_count), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        cluster_means = []
        for sampled_seed in sampled_seeds.tolist():
            sampled_poses = rng.integers(0, pose_count, size=pose_count)
            cluster_means.append(differences[sampled_seed, sampled_poses].mean(axis=0))
        samples[replicate] = np.stack(cluster_means).mean(axis=0)
    point = differences.mean(axis=(0, 1))
    lower = np.quantile(samples, 0.025, axis=0)
    upper = np.quantile(samples, 0.975, axis=0)
    return {
        metric: {
            "difference": float(point[index]),
            "confidenceInterval95": [float(lower[index]), float(upper[index])],
            "direction": "increase" if point[index] > 0.0 else "decrease" if point[index] < 0.0 else "no_change",
            "crossesZero": bool(lower[index] <= 0.0 <= upper[index]),
        }
        for index, metric in enumerate(METRICS)
    }


def summarize(benchmark_root: Path, *, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    matrix, poses = load_matrix(benchmark_root)
    member_rows: list[dict[str, Any]] = []
    variant_arrays: dict[str, np.ndarray] = {}
    variant_summaries: dict[str, Any] = {}
    for variant, _ in VARIANT_FAMILIES:
        seed_arrays = []
        aggregates = []
        safe_count = 0
        for seed in FORMAL_SEEDS:
            payload = matrix[variant][seed]
            per_pose = np.stack(
                [_finite_metrics(row["metrics"], Path(payload["checkpoint"])) for row in payload["perPose"]]
            )
            seed_arrays.append(per_pose)
            aggregate = payload["aggregate"]
            aggregate_metrics = {metric: float(aggregate[metric]) for metric in METRICS}
            aggregates.append(aggregate_metrics)
            safe = bool(payload["thresholdSource"].get("safeWorkpoint", False))
            safe_count += int(safe)
            member_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "epoch": int(payload.get("epoch", 0)),
                    "threshold": float(payload["threshold"]),
                    "safeWorkpoint": safe,
                    "metrics": aggregate_metrics,
                    "weightedRecallLowerConfidenceBound": payload["aggregate"].get(
                        "weightedRecallLowerConfidenceBound"
                    ),
                    "testRead": False,
                }
            )
        variant_arrays[variant] = np.stack(seed_arrays)
        variant_summaries[variant] = {
            "safeSeedCount": safe_count,
            "seedCount": len(FORMAL_SEEDS),
            "meanAggregateMetrics": {
                metric: float(np.mean([row[metric] for row in aggregates]))
                for metric in METRICS
            },
            "seedAggregateMetrics": aggregates,
        }
    comparisons = {
        variant: _paired_cluster_bootstrap(
            variant_arrays[variant],
            variant_arrays[BASELINE],
            replicates=int(replicates),
            seed=20260819 + index,
        )
        for index, (variant, _) in enumerate(VARIANT_FAMILIES)
        if variant != BASELINE
    }
    return {
        "schema": SCHEMA,
        "experiment": OUTPUT_TAG,
        "split": "validation",
        "poseCount": len(poses),
        "poseIndices": poses,
        "seedCount": len(FORMAL_SEEDS),
        "seeds": list(FORMAL_SEEDS),
        "variantCount": len(VARIANT_FAMILIES),
        "variants": [variant for variant, _ in VARIANT_FAMILIES],
        "memberCount": len(member_rows),
        "members": member_rows,
        "variantSummaries": variant_summaries,
        "pairedClusterBootstrap": {
            "replicates": int(replicates),
            "clusterOrder": "resample seeds, then paired validation poses within each sampled seed",
            "baseline": BASELINE,
            "comparisons": comparisons,
        },
        "candidateAndGtContract": {
            "samePoseOrder": True,
            "sameCandidateIds": True,
            "sameGtCounts": True,
            "sameWeightedGt": True,
            "gtSource": "same registered validation PoseCSR",
        },
        "testRead": False,
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# 108 维视角尾部分离器从头联合长训消融结果",
        "",
        "本报告只使用各 checkpoint 自己的 calibration 冻结阈值，并在完整 213 个 validation pose 上回放；test split 未读取。安全性以 weighted recall 及其 calibration 单侧置信下界为主门。",
        "",
        "| 变体 | 安全种子 | Precision | Weighted recall | Accuracy | Balanced accuracy | Useful cull | Avg pred | GLB byte reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    variants = summary["variantSummaries"]
    for variant, _ in VARIANT_FAMILIES:
        row = variants[variant]
        metrics = row["meanAggregateMetrics"]
        lines.append(
            f"| `{variant}` | {row['safeSeedCount']}/3 | {_format_percent(metrics['precision'])} | "
            f"{_format_percent(metrics['weightedRecall'])} | {_format_percent(metrics['accuracy'])} | "
            f"{_format_percent(metrics['balancedAccuracy'])} | {_format_percent(metrics['usefulCull'])} | "
            f"{metrics['avgPredCount']:.2f} | {_format_percent(metrics['glbByteReduction'])} |"
        )
    lines.extend(["", "## 配对比较", ""])
    comparisons = summary["pairedClusterBootstrap"]["comparisons"]
    for variant, comparison in comparisons.items():
        lines.append(f"### `{variant}` 相对无分离器基线")
        lines.append("")
        lines.append("| 指标 | 差值 | 95% CI | 跨零 |")
        lines.append("|---|---:|---:|---:|")
        for metric in ("precision", "weightedRecall", "accuracy", "balancedAccuracy", "usefulCull", "badCull", "avgPredCount", "predictedGlbBytes"):
            result = comparison[metric]
            lower, upper = result["confidenceInterval95"]
            lines.append(
                f"| {metric} | {result['difference']:.6f} | [{lower:.6f}, {upper:.6f}] | "
                f"{'是' if result['crossesZero'] else '否'} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    baseline = np.zeros((3, 5, len(METRICS)), dtype=np.float64)
    variant = baseline + 0.25
    result = _paired_cluster_bootstrap(
        variant, baseline, replicates=100, seed=1
    )
    for metric in METRICS:
        if not math.isclose(result[metric]["difference"], 0.25):
            raise AssertionError("paired bootstrap point estimate is incorrect")
        if result[metric]["crossesZero"]:
            raise AssertionError("constant positive effect cannot cross zero")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out" / OUTPUT_TAG,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        print(json.dumps({"selfTest": "passed"}))
        return
    output = args.output or args.benchmark_root / "summary.json"
    report = args.report or ROOT / "docs/evaluation/pvs_joint_108d_query_tail_separator_from_scratch_v1_2026-08-19.md"
    summary = summarize(args.benchmark_root.resolve(), replicates=args.bootstrap_replicates)
    _write_json(output.resolve(), summary)
    write_report(report.resolve(), summary)
    print(json.dumps({"output": str(output.resolve()), "report": str(report.resolve()), "memberCount": summary["memberCount"], "testRead": False}))


if __name__ == "__main__":
    main()
