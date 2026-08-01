#!/usr/bin/env python3
"""Summarize the registered M4 validation matrix without opening test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = (
    "aabb_ray",
    "geometry_ray",
    "geometry_context_ray",
    "geometry_context_proxy_ray_no_inhibition",
    "full",
)
SEEDS = (20260801, 20260802, 20260803)
REFERENCE = "geometry_context_ray"
METRICS = ("useful_cull", "bad_cull", "weighted_recall", "precision", "recall", "f1", "avg_pred_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path("neural_instance_culling/benchmark/out"))
    parser.add_argument("--evaluation-prefix", default="m4_formal_baseline_")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def experiment_name(variant: str, seed: int) -> str:
    return f"pvs_m4_ablation_{variant}_rvl_strong_v2_hkust_spatial_fov66_seed{seed}_full40"


def input_path(root: Path, prefix: str, variant: str, seed: int) -> Path:
    return root / f"{prefix}{experiment_name(variant, seed)}_validation" / "interventions.json"


def read_input(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "validation":
        raise ValueError(f"M4 matrix input must be validation-only: {path}")
    if int(payload.get("posePlan", {}).get("poseCount", 0)) <= 0:
        raise ValueError(f"M4 matrix input has no validation poses: {path}")
    threshold_source = json.dumps(payload.get("thresholdSource", {}), ensure_ascii=False).lower()
    if "test" in threshold_source:
        raise ValueError(f"M4 validation input has test-derived threshold provenance: {path}")
    baseline = payload.get("interventions", {}).get("baseline")
    rows = payload.get("perPose", {}).get("baseline")
    if not isinstance(baseline, dict) or not isinstance(rows, list) or not rows:
        raise ValueError(f"M4 matrix input lacks baseline per-pose records: {path}")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        pose_id = int(row["pose_index"])
        if pose_id in indexed:
            raise ValueError(f"duplicate pose index {pose_id} in {path}")
        indexed[pose_id] = row
    return {"path": str(path), "payload": payload, "aggregate": baseline["aggregate"], "rows": indexed}


def validate_pair(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]], label: str) -> None:
    if set(left) != set(right):
        raise ValueError(f"M4 pose sets differ for {label}")
    for pose_id in left:
        lrow, rrow = left[pose_id], right[pose_id]
        if lrow.get("candidate_id_sha256") != rrow.get("candidate_id_sha256"):
            raise ValueError(f"M4 candidate hash differs at pose {pose_id} for {label}")
        if int(lrow.get("candidate_count", -1)) != int(rrow.get("candidate_count", -1)):
            raise ValueError(f"M4 candidate count differs at pose {pose_id} for {label}")


def stable_seed(*parts: str | int) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def hierarchical_bootstrap(
    rows_by_seed: dict[int, dict[int, tuple[dict[str, Any], dict[str, Any]]]],
    metric: str,
    seed: int,
    replicates: int,
) -> dict[str, float | list[float]]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    seed_ids = sorted(rows_by_seed)
    differences = []
    for seed_id in seed_ids:
        pairs = rows_by_seed[seed_id]
        values = np.asarray(
            [float(right[metric]) - float(left[metric]) for left, right in pairs.values()], dtype=np.float64
        )
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"invalid {metric} values in seed {seed_id}")
        differences.append(values)
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(replicates),), dtype=np.float64)
    for index in range(int(replicates)):
        selected_seeds = rng.integers(0, len(seed_ids), size=len(seed_ids))
        cluster_means = []
        for cluster_index in selected_seeds.tolist():
            values = differences[cluster_index]
            cluster_means.append(float(values[rng.integers(0, values.size, size=values.size)].mean()))
        samples[index] = float(np.mean(cluster_means))
    observed = float(np.mean([values.mean() for values in differences]))
    return {
        "mean_delta": observed,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    records: dict[tuple[int, str], dict[str, Any]] = {}
    missing: list[str] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            path = input_path(args.benchmark_root, args.evaluation_prefix, variant, seed)
            if not path.is_file():
                missing.append(str(path))
                continue
            records[(seed, variant)] = read_input(path)
    if missing:
        raise FileNotFoundError("M4 matrix is incomplete; missing inputs:\n" + "\n".join(missing))

    rows: dict[str, Any] = {}
    for seed in SEEDS:
        rows[str(seed)] = {}
        for variant in VARIANTS:
            record = records[(seed, variant)]
            aggregate = record["aggregate"]
            rows[str(seed)][variant] = {
                "input": record["path"],
                "threshold": float(record["payload"]["threshold"]),
                "poseCount": len(record["rows"]),
                "aggregate": {metric: aggregate.get(metric) for metric in METRICS},
            }
        reference_rows = records[(seed, REFERENCE)]["rows"]
        for variant in VARIANTS:
            validate_pair(reference_rows, records[(seed, variant)]["rows"], f"seed={seed} variant={variant}")

    comparisons: dict[str, Any] = {}
    for variant in VARIANTS:
        if variant == REFERENCE:
            continue
        comparisons[variant] = {}
        for metric in METRICS:
            pair_rows: dict[int, dict[int, tuple[dict[str, Any], dict[str, Any]]]] = {}
            for seed in SEEDS:
                reference_rows = records[(seed, REFERENCE)]["rows"]
                candidate_rows = records[(seed, variant)]["rows"]
                pair_rows[seed] = {
                    pose_id: (reference_rows[pose_id], candidate_rows[pose_id]) for pose_id in sorted(reference_rows)
                }
            comparisons[variant][metric] = hierarchical_bootstrap(
                pair_rows, metric, stable_seed(args.seed, variant, metric), args.bootstrap_replicates
            )

    payload = {
        "schema": "neuralstreamweb3d-formal-m4-matrix-summary-v1",
        "split": "validation",
        "candidateSemantics": "stored back-camera candidates; no GT union and no candidate cap",
        "thresholdSemantics": "each checkpoint threshold frozen from its own calibration; no validation threshold scan",
        "referenceVariant": REFERENCE,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "metrics": list(METRICS),
        "bootstrap": {
            "replicates": int(args.bootstrap_replicates),
            "clusterUnit": "seed, then validation pose within seed",
            "confidence": 0.95,
        },
        "rows": rows,
        "pairedComparisonsVariantMinusReference": comparisons,
        "status": "validation_summary_only; formal test not read",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": payload["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
