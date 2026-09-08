#!/usr/bin/env python3
"""Summarize the fixed PVS core ablations with paired hierarchical bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SEEDS = (20260801, 20260802, 20260803)
VARIANTS = (
    "no_relation",
    "no_survival",
    "generic28",
    "no_moment",
    "no_recall_guard",
    "no_tail_margin",
)
REFERENCE = "full"
POSE_COUNT = 730
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260823

SET_METRICS = (
    "Precision",
    "Recall",
    "WeightedRecall",
    "Accuracy",
    "BalancedAccuracy",
    "Specificity",
    "F1",
    "Jaccard",
    "UsefulCull",
    "BadCull",
)
METRICS = (
    *(f"pose{name}" for name in SET_METRICS),
    *(f"aggregate{name}" for name in SET_METRICS),
    "avgCandidateCount",
    "avgGtCount",
    "avgPredCount",
    "avgTp",
    "avgFp",
    "avgFn",
    "avgTn",
    "predOverCandidate",
    "predOverGt",
    "predictedGlbCount",
    "predictedGlbBytes",
    "glbCountReduction",
    "glbByteReduction",
    "glbBytesAtAchievedVisualUtility",
)
COUNT_FIELDS = (
    "tp",
    "fp",
    "fn",
    "tn",
    "weightedTp",
    "weightedGt",
    "predCount",
    "candidateGlbCount",
    "predictedGlbCount",
    "candidateGlbBytes",
    "predictedGlbBytes",
    "glbCountReduction",
    "glbByteReduction",
    "glbBytesAtAchievedVisualUtility",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _evaluation_path(root: Path, variant: str, seed: int) -> Path:
    member = f"paper_{variant}_seed{seed}_e40"
    return root / "members" / member / "validation_evaluation.json"


def _full_reference_path(root: Path, seed: int) -> Path:
    return _evaluation_path(root, REFERENCE, seed)


def _validate_evaluation(
    payload: Mapping[str, Any], path: Path, seed: int
) -> list[dict[str, Any]]:
    if payload.get("split") != "validation" or payload.get("testRead") is not False:
        raise ValueError(f"evaluation is not a test-free validation replay: {path}")
    if int(payload.get("checkpointSeed", -1)) != int(seed):
        raise ValueError(f"evaluation seed disagrees with its member: {path}")
    rows = payload.get("perPose")
    if not isinstance(rows, list) or len(rows) != POSE_COUNT:
        raise ValueError(f"evaluation must contain {POSE_COUNT} per-pose rows: {path}")
    pose_indices = [int(row["poseIndex"]) for row in rows]
    if pose_indices != sorted(pose_indices) or len(set(pose_indices)) != POSE_COUNT:
        raise ValueError(f"evaluation pose order is invalid: {path}")
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"evaluation pose has no metrics: {path}")
        values = [float(metrics[field]) for field in COUNT_FIELDS]
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"evaluation pose contains non-finite metrics: {path}")
    return rows


def _validate_pair(
    reference_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    label: str,
) -> None:
    for reference, variant in zip(reference_rows, variant_rows):
        if int(reference["poseIndex"]) != int(variant["poseIndex"]):
            raise ValueError(f"paired pose order differs: {label}")
        if int(reference["candidateCount"]) != int(variant["candidateCount"]):
            raise ValueError(f"paired candidate count differs: {label}")
        reference_ids = reference.get("candidateIds")
        variant_ids = variant.get("candidateIds")
        if not isinstance(reference_ids, list) or not isinstance(variant_ids, list):
            raise ValueError(f"paired evaluation must persist candidate IDs: {label}")
        if reference_ids != variant_ids:
            raise ValueError(f"paired candidate IDs differ: {label}")


def _row_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(
            [float(row["metrics"][field]) for row in rows], dtype=np.float64
        )
        for field in COUNT_FIELDS
    }


def _set_metrics(
    values: Mapping[str, np.ndarray], *, prefix: str
) -> dict[str, np.ndarray]:
    tp, fp, fn, tn = (values[field] for field in ("tp", "fp", "fn", "tn"))
    candidate = np.maximum(1.0, tp + fp + fn + tn)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    specificity = tn / np.maximum(1.0, tn + fp)
    return {
        f"{prefix}Precision": precision,
        f"{prefix}Recall": recall,
        f"{prefix}WeightedRecall": values["weightedTp"]
        / np.maximum(1e-12, values["weightedGt"]),
        f"{prefix}Accuracy": (tp + tn) / candidate,
        f"{prefix}BalancedAccuracy": 0.5 * (recall + specificity),
        f"{prefix}Specificity": specificity,
        f"{prefix}F1": 2.0 * tp / np.maximum(1.0, 2.0 * tp + fp + fn),
        f"{prefix}Jaccard": tp / np.maximum(1.0, tp + fp + fn),
        f"{prefix}UsefulCull": tn / candidate,
        f"{prefix}BadCull": fn / candidate,
    }


def _metrics_from_values(
    values: Mapping[str, np.ndarray], pose_count: int
) -> dict[str, np.ndarray]:
    aggregate_values = {
        field: value.sum(axis=-1) for field, value in values.items()
    }
    aggregate = _set_metrics(aggregate_values, prefix="aggregate")
    pose = {
        metric: value.mean(axis=-1)
        for metric, value in _set_metrics(values, prefix="pose").items()
    }
    tp, fp, fn, tn = (
        aggregate_values[field] for field in ("tp", "fp", "fn", "tn")
    )
    candidate = np.maximum(1.0, tp + fp + fn + tn)
    gt = np.maximum(1.0, tp + fn)
    efficiency = {
        "avgCandidateCount": candidate / max(1, pose_count),
        "avgGtCount": (tp + fn) / max(1, pose_count),
        "avgPredCount": aggregate_values["predCount"] / max(1, pose_count),
        "avgTp": tp / max(1, pose_count),
        "avgFp": fp / max(1, pose_count),
        "avgFn": fn / max(1, pose_count),
        "avgTn": tn / max(1, pose_count),
        "predOverCandidate": aggregate_values["predCount"] / candidate,
        "predOverGt": aggregate_values["predCount"] / gt,
        "predictedGlbCount": aggregate_values["predictedGlbCount"]
        / max(1, pose_count),
        "predictedGlbBytes": aggregate_values["predictedGlbBytes"]
        / max(1, pose_count),
        "glbCountReduction": aggregate_values["glbCountReduction"]
        / max(1, pose_count),
        "glbByteReduction": aggregate_values["glbByteReduction"]
        / max(1, pose_count),
        "glbBytesAtAchievedVisualUtility": aggregate_values[
            "glbBytesAtAchievedVisualUtility"
        ]
        / max(1, pose_count),
    }
    return {**pose, **aggregate, **efficiency}


def _seed_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {
        metric: float(value)
        for metric, value in _metrics_from_values(arrays, POSE_COUNT).items()
    }


def paired_hierarchical_bootstrap(
    reference: Mapping[int, Mapping[str, np.ndarray]],
    variant: Mapping[int, Mapping[str, np.ndarray]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    if int(replicates) < 1:
        raise ValueError("bootstrap replicates must be positive")
    seeds = sorted(reference)
    if seeds != sorted(variant) or not seeds:
        raise ValueError("paired bootstrap members do not have the same seeds")
    for current_seed in seeds:
        if any(
            reference[current_seed][field].shape != (POSE_COUNT,)
            or variant[current_seed][field].shape != (POSE_COUNT,)
            for field in COUNT_FIELDS
        ):
            raise ValueError("paired bootstrap members have invalid pose arrays")

    reference_stack = np.stack(
        [
            np.stack([reference[current_seed][field] for current_seed in seeds])
            for field in COUNT_FIELDS
        ]
    )
    variant_stack = np.stack(
        [
            np.stack([variant[current_seed][field] for current_seed in seeds])
            for field in COUNT_FIELDS
        ]
    )
    observed_reference = {
        metric: np.mean(
            [_seed_metrics(reference[current_seed])[metric] for current_seed in seeds]
        )
        for metric in METRICS
    }
    observed_variant = {
        metric: np.mean(
            [_seed_metrics(variant[current_seed])[metric] for current_seed in seeds]
        )
        for metric in METRICS
    }
    samples = {
        metric: np.empty(int(replicates), dtype=np.float64) for metric in METRICS
    }
    rng = np.random.default_rng(int(seed))
    batch_size = 64
    for start in range(0, int(replicates), batch_size):
        end = min(int(replicates), start + batch_size)
        count = end - start
        seed_indices = rng.integers(0, len(seeds), size=(count, len(seeds)))
        pose_indices = rng.integers(
            0, POSE_COUNT, size=(count, len(seeds), POSE_COUNT)
        )

        def sample_metrics(stack: np.ndarray) -> dict[str, np.ndarray]:
            selected_seeds = stack[:, seed_indices, :]
            selected = np.take_along_axis(
                selected_seeds, pose_indices[None, ...], axis=3
            )
            values = {
                field: selected[index]
                for index, field in enumerate(COUNT_FIELDS)
            }
            return _metrics_from_values(values, POSE_COUNT)

        reference_metrics = sample_metrics(reference_stack)
        variant_metrics = sample_metrics(variant_stack)
        for metric in METRICS:
            samples[metric][start:end] = (
                reference_metrics[metric] - variant_metrics[metric]
            ).mean(axis=1)

    result: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        values = samples[metric]
        delta = float(observed_reference[metric] - observed_variant[metric])
        interval = [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        result[metric] = {
            "fullMinusAblation": delta,
            "ci95": interval,
            "direction": "positive" if delta > 0.0 else "negative" if delta < 0.0 else "zero",
            "crossesZero": bool(interval[0] <= 0.0 <= interval[1]),
        }
    return result


def _aggregate_row(payload: Mapping[str, Any], source: Path) -> dict[str, Any]:
    aggregate = payload.get("aggregate")
    runtime = payload.get("runtime")
    if not isinstance(aggregate, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError(f"evaluation summary is incomplete: {source}")
    per_pose = payload.get("perPose")
    if not isinstance(per_pose, list):
        raise ValueError(f"evaluation has no per-pose rows: {source}")
    member_metrics = _seed_metrics(_row_arrays(per_pose))
    return {
        "source": str(source),
        "threshold": float(payload["threshold"]),
        "epoch": int(payload["epoch"]),
        "validationWeightedRecallLowerConfidenceBound": float(
            aggregate["aggregateWeightedRecallLowerConfidenceBound"]
        ),
        "runtimeFeatureBytes": int(runtime["runtimeFeatureBytes"]),
        "runtimeFeatureDim": int(runtime["runtimeFeatureDim"]),
        "metrics": member_metrics,
        "testRead": False,
    }


def summarize(
    root: Path,
    output: Path,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    arrays: dict[str, dict[int, dict[str, np.ndarray]]] = {
        name: {} for name in (REFERENCE, *VARIANTS)
    }
    rows: dict[str, dict[str, Any]] = {name: {} for name in arrays}
    for current_seed in SEEDS:
        full_path = _full_reference_path(root, current_seed)
        full_payload = _load_json(full_path)
        full_rows = _validate_evaluation(full_payload, full_path, current_seed)
        arrays[REFERENCE][current_seed] = _row_arrays(full_rows)
        rows[REFERENCE][str(current_seed)] = _aggregate_row(full_payload, full_path)
        for variant in VARIANTS:
            path = _evaluation_path(root, variant, current_seed)
            payload = _load_json(path)
            variant_rows = _validate_evaluation(payload, path, current_seed)
            _validate_pair(full_rows, variant_rows, f"seed={current_seed} {variant}")
            arrays[variant][current_seed] = _row_arrays(variant_rows)
            rows[variant][str(current_seed)] = _aggregate_row(payload, path)

    comparisons = {
        variant: paired_hierarchical_bootstrap(
            arrays[REFERENCE],
            arrays[variant],
            replicates=replicates,
            seed=int(seed) + index,
        )
        for index, variant in enumerate(VARIANTS)
    }
    payload = {
        "schema": "pvs-mainline-core-ablation-summary-v2",
        "split": "validation",
        "testRead": False,
        "reference": "three-seed V4 full model",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "poseCountPerSeed": POSE_COUNT,
        "thresholdProtocol": "each checkpoint uses only its own calibration-frozen threshold",
        "candidateProtocol": "identical persisted back-camera candidate IDs per paired pose; no GT union",
        "bootstrap": {
            "replicates": int(replicates),
            "confidence": 0.95,
            "clusterUnit": "outer seed cluster, inner validation pose resampling",
            "paired": True,
        },
        "contrastDirection": (
            "full minus ablation; positive favors full for accuracy/recall/precision/"
            "useful-cull/reduction metrics, while negative favors full for bad-cull, "
            "prediction count, false-positive/false-negative count, and byte-cost metrics"
        ),
        "metrics": list(METRICS),
        "rows": rows,
        "pairedComparisons": comparisons,
        "imageMetrics": {
            "status": "pending_color_id_backfill",
            "testRead": False,
        },
        "browserMetrics": {
            "status": "pending_export_and_hardware_adapter_gate",
            "testRead": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("neural_instance_culling/benchmark/out/pvs_v4_integrated_visibility_mainline_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "neural_instance_culling/benchmark/out/"
            "pvs_v4_integrated_visibility_mainline_v1/paper_core_ablation_summary.json"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(
        args.root.resolve(),
        args.output.resolve(),
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "variants": len(payload["variants"]),
                "seeds": len(payload["seeds"]),
                "bootstrapReplicates": payload["bootstrap"]["replicates"],
                "testRead": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
