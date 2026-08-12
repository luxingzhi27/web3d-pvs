#!/usr/bin/env python3
"""Summarize the ray-context/survival/OWRB matrix with paired bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


POSE_METRICS = {
    "precision": "precision",
    "recall": "recall",
    "weighted_recall": "weightedRecall",
    "accuracy": "accuracy",
    "balanced_accuracy": "balancedAccuracy",
    "specificity": "specificity",
    "f1": "f1",
    "jaccard": "jaccard",
    "useful_cull": "usefulCull",
    "bad_cull": "badCull",
    "avg_pred_count": "predCount",
    "pred_over_candidate": "predOverCandidate",
    "pred_over_gt": "predOverGt",
    "candidate_glb_count": "candidateGlbCount",
    "predicted_glb_bytes": "predictedGlbBytes",
    "predicted_glb_count": "predictedGlbCount",
    "candidate_glb_bytes": "candidateGlbBytes",
    "glb_count_reduction": "glbCountReduction",
    "glb_byte_reduction": "glbByteReduction",
    "download_utility_recall": "downloadUtilityRecall",
    "glb_bytes_at_achieved_visual_utility": "glbBytesAtAchievedVisualUtility",
    "candidate_count": "candidateCount",
    "gt_count": "gtCount",
    "avg_fp_count": "fp",
    "avg_fn_count": "fn",
    "avg_tn_count": "tn",
    "avg_tp_count": "tp",
}
AGG_METRICS = {
    "precision": "precision",
    "recall": "recall",
    "weighted_recall": "weightedRecall",
    "accuracy": "accuracy",
    "balanced_accuracy": "balancedAccuracy",
    "specificity": "specificity",
    "f1": "f1",
    "jaccard": "jaccard",
    "useful_cull": "usefulCull",
    "bad_cull": "badCull",
    "avg_pred_count": "avgPredCount",
    "avg_fp_count": "avgFpCount",
    "avg_fn_count": "avgFnCount",
    "avg_tn_count": "avgTnCount",
    "avg_tp_count": "avgTpCount",
    "pred_over_candidate": "predOverCandidate",
    "pred_over_gt": "predOverGt",
    "candidate_glb_count": "candidateGlbCount",
    "predicted_glb_bytes": "predictedGlbBytes",
    "predicted_glb_count": "predictedGlbCount",
    "candidate_glb_bytes": "candidateGlbBytes",
    "glb_count_reduction": "glbCountReduction",
    "glb_byte_reduction": "glbByteReduction",
    "download_utility_recall": "downloadUtilityRecall",
    "glb_bytes_at_achieved_visual_utility": "glbBytesAtAchievedVisualUtility",
}
BASE_METRICS = tuple(
    [f"pose_{name}" for name in POSE_METRICS]
    + [f"aggregate_{name}" for name in AGG_METRICS]
)
IMAGE_POSE_METRICS = {
    "image_per": "PER",
    "image_miss_pixel_rate": "missPixelRate",
    "image_wrong_id_pixel_rate": "wrongInstancePixelRate",
    "image_extra_pixel_rate": "extraPixelRateOverImage",
}
IMAGE_AGG_METRICS = dict(IMAGE_POSE_METRICS)
IMAGE_METRICS = tuple(
    [f"pose_{name}" for name in IMAGE_POSE_METRICS]
    + [f"aggregate_{name}" for name in IMAGE_AGG_METRICS]
)
ALL_METRICS = BASE_METRICS + IMAGE_METRICS

# The core formal matrix is the registered 2x2x2 design.  The supplement is a
# separate, five-member mechanism audit; keeping its registry here makes it
# impossible for a malformed manifest to silently masquerade as the core
# factorial experiment.
CORE_FORMAL_VARIANT_COUNT = 8
SUPPLEMENT_FORMAL_VARIANTS = (
    "triangle_context32_direct9_monotone",
    "triangle_context64_direct9_monotone",
    "triangle_context32_fourier117_monotone",
    "triangle_context32_direct9_unconstrained28",
    "aabb_context32_direct9_monotone",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_webgpu_parity(path: Path | None) -> dict[str, Any]:
    """Load WebGPU parity without turning a software adapter into a GPU claim."""
    if path is None:
        return {
            "status": "not_registered",
            "passed": False,
            "formalReady": False,
            "hardwareClaim": "none",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ray-context-survival-owrb-webgpu-parity-validation-v1":
        raise ValueError(f"unexpected WebGPU parity validation schema: {path}")
    passed = bool(payload.get("passed", False))
    formal_ready = bool(payload.get("formalReady", False))
    if not passed:
        raise ValueError(f"WebGPU parity validation did not pass: {path}")
    if formal_ready:
        status = "hardware_numeric_parity_passed"
        claim = "NVIDIA WebGPU hardware parity"
    else:
        status = "software_numeric_parity_passed"
        claim = "software WebGPU numeric parity only; no hardware WebGPU claim"
    return {
        "schema": payload["schema"],
        "status": status,
        "passed": passed,
        "formalReady": formal_ready,
        "hardwareClaim": claim,
        "backend": "webgpu",
        "validationPath": str(path.resolve()),
        "capture": payload.get("capture"),
        "cases": payload.get("cases"),
        "caseCount": int(payload.get("caseCount", 0)),
        "maxAbs": float(payload.get("maxAbs", 0.0)),
        "maxRelative": float(payload.get("maxRelative", 0.0)),
        "atol": float(payload.get("atol", 0.0)),
        "rtol": float(payload.get("rtol", 0.0)),
        "gpuGate": payload.get("gpuGate"),
    }


def _resolve_manifest_path(raw_value: str, manifest_path: Path) -> Path:
    """Resolve both legacy repository-relative and current absolute entries.

    Early pilot manifests stored paths relative to the repository root while
    the loader interpreted every relative path relative to the manifest.  A
    formal manifest now stores absolute paths, but the pilot artifacts must
    remain readable without rewriting or moving their results.
    """
    raw_path = Path(str(raw_value)).expanduser()
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"manifest member does not exist: {candidate}")
    candidates = (
        (manifest_path.parent / raw_path).resolve(),
        (REPO_ROOT / raw_path).resolve(),
        (Path.cwd() / raw_path).resolve(),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"manifest member does not exist: {raw_value}; tried {tried}")


def _load_manifest(path: Path) -> tuple[dict[str, dict[int, Path]], list[dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ray-context-survival-owrb-matrix-manifest-v1":
        raise ValueError("unexpected OWRB matrix manifest schema")
    root = path.parent
    variants: dict[str, dict[int, Path]] = {}
    for variant, members in payload.get("variants", {}).items():
        if not isinstance(members, dict) or not members:
            raise ValueError(f"variant {variant} has no members")
        variants[variant] = {}
        for seed, value in members.items():
            variants[variant][int(seed)] = _resolve_manifest_path(str(value), path)
    comparisons = []
    for item in payload.get("comparisons", []):
        comparisons.append({"name": str(item["name"]), "left": str(item["left"]), "right": str(item["right"])})
    if not variants:
        raise ValueError("matrix manifest contains no variants")
    return variants, comparisons


def _load_members(
    variants: dict[str, dict[int, Path]],
    require_image_metrics: bool = False,
) -> dict[str, dict[int, dict[str, Any]]]:
    loaded: dict[str, dict[int, dict[str, Any]]] = {}
    for variant, paths in variants.items():
        loaded[variant] = {}
        for seed, path in paths.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != "ray-context-survival-owrb-evaluation-v1":
                raise ValueError(f"{path}: unexpected evaluator schema")
            if payload.get("thresholdSource", {}).get("protocol") != "calibration_ready_pre_test":
                raise ValueError(f"{path}: threshold was not calibration-frozen")
            if int(payload.get("thresholdSource", {}).get("testEvaluationCount", -1)) != 0:
                raise ValueError(f"{path}: threshold source read test data")
            if len(str(payload.get("candidateDigest", ""))) != 64:
                raise ValueError(f"{path}: evaluation has no canonical candidate digest")
            if require_image_metrics:
                image = payload.get("imageMetrics") or {}
                if image.get("status") != "formal_ready" or image.get("perPoseAvailable") is not True:
                    raise ValueError(f"{path}: formal summary requires a completed Color-ID image evaluation")
                rows = payload.get("perPose") or []
                if not rows or any(not isinstance(row.get("imageMetrics"), dict) for row in rows):
                    raise ValueError(f"{path}: formal summary requires image metrics for every pose")
            loaded[variant][int(seed)] = {"path": path, "payload": payload}
    return loaded


def _validate_alignment(members: dict[str, dict[int, dict[str, Any]]]) -> tuple[list[int], list[int], dict[int, dict[int, str]]]:
    seed_sets = [set(values) for values in members.values()]
    if not seed_sets or any(values != seed_sets[0] for values in seed_sets[1:]):
        raise ValueError("all variants must contain the same seed set")
    seeds = sorted(seed_sets[0])
    reference_variant = next(iter(members))
    pose_indices_by_seed: dict[int, list[int]] = {}
    hashes_by_seed: dict[int, dict[int, str]] = {}
    candidate_digests_by_seed: dict[int, str] = {}
    for seed in seeds:
        candidate_digests_by_seed[seed] = str(members[reference_variant][seed]["payload"]["candidateDigest"])
        reference_rows = members[reference_variant][seed]["payload"].get("perPose", [])
        reference_pose_indices = [int(row["poseIndex"]) for row in reference_rows]
        if not reference_pose_indices:
            raise ValueError("a member contains no pose rows")
        pose_indices_by_seed[seed] = reference_pose_indices
        hashes_by_seed[seed] = {int(row["poseIndex"]): str(row["candidateIdSha256"]) for row in reference_rows}
        for variant, by_seed in members.items():
            if str(by_seed[seed]["payload"].get("candidateDigest", "")) != candidate_digests_by_seed[seed]:
                raise ValueError(f"canonical candidate digest mismatch at variant {variant}, seed {seed}")
            rows = by_seed[seed]["payload"].get("perPose", [])
            indices = [int(row["poseIndex"]) for row in rows]
            if indices != reference_pose_indices:
                raise ValueError(f"pose order mismatch between {reference_variant} and {variant} at seed {seed}")
            for row in rows:
                pose = int(row["poseIndex"])
                if str(row["candidateIdSha256"]) != hashes_by_seed[seed][pose]:
                    raise ValueError(f"candidate hash mismatch at variant {variant}, seed {seed}, pose {pose}")
    reference_seed = seeds[0]
    reference_digest = candidate_digests_by_seed[reference_seed]
    reference_pose_indices = pose_indices_by_seed[reference_seed]
    for seed in seeds[1:]:
        if candidate_digests_by_seed[seed] != reference_digest:
            raise ValueError("formal members use different canonical candidate digests across seeds")
        if pose_indices_by_seed[seed] != reference_pose_indices:
            raise ValueError("formal members use different validation pose sequences across seeds")
    pose_counts = {len(values) for values in pose_indices_by_seed.values()}
    if len(pose_counts) != 1:
        raise ValueError("seed members have different pose counts")
    return seeds, pose_indices_by_seed[seeds[0]], hashes_by_seed


def _metric_from_rows(rows: list[dict[str, Any]], metric: str) -> float:
    if metric.startswith("pose_image_"):
        field = IMAGE_POSE_METRICS[metric[len("pose_"):]]
        return float(np.mean([float((row.get("imageMetrics") or {})[field]) for row in rows]))
    if metric.startswith("aggregate_image_"):
        field = IMAGE_AGG_METRICS[metric[len("aggregate_"):]]
        return float(_aggregate_image_sample(rows)[field])
    if metric.startswith("pose_"):
        field = POSE_METRICS[metric[len("pose_"):]]
        return float(np.mean([float(row[field]) for row in rows]))
    if not metric.startswith("aggregate_"):
        raise ValueError(f"unknown metric {metric}")
    field = AGG_METRICS[metric[len("aggregate_"):]]
    # Aggregate metrics are recomputed from TP/FP/FN/TN and weighted masses;
    # averaging per-pose ratios would produce a different estimand.
    return float(_aggregate_sample(rows)[field])


def _aggregate_sample(rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = float(sum(row["tp"] for row in rows))
    fp = float(sum(row["fp"] for row in rows))
    fn = float(sum(row["fn"] for row in rows))
    tn = float(sum(row["tn"] for row in rows))
    candidate = max(1.0, tp + fp + fn + tn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    specificity = tn / max(1.0, tn + fp)
    weighted_tp = float(sum(row["weightedTp"] for row in rows))
    weighted_gt = float(sum(row["weightedGt"] for row in rows))
    download_utility_tp = float(sum(row["downloadUtilityTp"] for row in rows))
    return {
        "precision": precision,
        "recall": recall,
        "weightedRecall": weighted_tp / max(1e-12, weighted_gt),
        "downloadUtilityRecall": download_utility_tp / max(1e-12, weighted_gt),
        "accuracy": (tp + tn) / candidate,
        "balancedAccuracy": 0.5 * (recall + specificity),
        "specificity": specificity,
        "f1": 2.0 * precision * recall / max(1e-8, precision + recall),
        "jaccard": tp / max(1.0, tp + fp + fn),
        "usefulCull": tn / candidate,
        "badCull": fn / candidate,
        "avgPredCount": float(sum(row["predCount"] for row in rows) / max(1, len(rows))),
        "avgFpCount": float(fp / max(1, len(rows))),
        "avgFnCount": float(fn / max(1, len(rows))),
        "avgTnCount": float(tn / max(1, len(rows))),
        "avgTpCount": float(tp / max(1, len(rows))),
        "predOverCandidate": float(sum(row["predCount"] for row in rows) / max(1.0, candidate)),
        "predOverGt": float(sum(row["predCount"] for row in rows) / max(1.0, tp + fn)),
        "candidateGlbCount": float(np.mean([row["candidateGlbCount"] for row in rows])),
        "predictedGlbBytes": float(np.mean([row["predictedGlbBytes"] for row in rows])),
        "predictedGlbCount": float(np.mean([row["predictedGlbCount"] for row in rows])),
        "candidateGlbBytes": float(np.mean([row["candidateGlbBytes"] for row in rows])),
        "glbCountReduction": float(np.mean([row["glbCountReduction"] for row in rows])),
        "glbByteReduction": float(np.mean([row["glbByteReduction"] for row in rows])),
        "glbBytesAtAchievedVisualUtility": float(np.mean([row["glbBytesAtAchievedVisualUtility"] for row in rows])),
    }


def _aggregate_image_sample(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate Color-ID errors from pixel counts rather than mean ratios."""
    fields = (
        "totalPixels", "validReferencePixels", "errorPixels", "missPixels",
        "wrongInstancePixels", "extraPixels",
    )
    totals = {
        field: float(sum(float((row.get("imageMetrics") or {}).get(field, 0.0)) for row in rows))
        for field in fields
    }
    valid = max(1.0, totals["validReferencePixels"])
    total = max(1.0, totals["totalPixels"])
    return {
        "PER": totals["errorPixels"] / valid,
        "missPixelRate": totals["missPixels"] / valid,
        "wrongInstancePixelRate": totals["wrongInstancePixels"] / valid,
        "extraPixelRateOverImage": totals["extraPixels"] / total,
    }


def _sample_metric(rows: list[dict[str, Any]], metric: str, indices: np.ndarray) -> float:
    sampled = [rows[int(index)] for index in np.asarray(indices, dtype=np.int64).tolist()]
    if metric.startswith("pose_"):
        return _metric_from_rows(sampled, metric)
    if metric.startswith("aggregate_image_"):
        return _aggregate_image_sample(sampled)[IMAGE_AGG_METRICS[metric[len("aggregate_"):]]]
    return _aggregate_sample(sampled)[AGG_METRICS[metric[len("aggregate_"):]]]


_BOOTSTRAP_AGG_FIELDS = (
    "tp", "fp", "fn", "tn", "weightedTp", "weightedGt", "downloadUtilityTp", "predCount",
    "candidateGlbCount", "predictedGlbCount", "candidateGlbBytes",
    "predictedGlbBytes", "glbCountReduction", "glbByteReduction", "glbBytesAtAchievedVisualUtility",
)
_BOOTSTRAP_IMAGE_FIELDS = (
    "totalPixels", "validReferencePixels", "errorPixels", "missPixels",
    "wrongInstancePixels", "extraPixels",
)


def _bootstrap_row_arrays(rows: list[dict[str, Any]], metric: str) -> dict[str, np.ndarray]:
    """Materialize sufficient statistics for vectorized pose resampling."""
    if not rows:
        raise ValueError("bootstrap cannot resample an empty pose sequence")
    if metric.startswith("pose_"):
        key = metric[len("pose_"):]
        if key in IMAGE_POSE_METRICS:
            field = IMAGE_POSE_METRICS[key]
            values = [float((row.get("imageMetrics") or {}).get(field, 0.0)) for row in rows]
        else:
            field = POSE_METRICS[key]
            values = [float(row[field]) for row in rows]
        return {"value": np.asarray(values, dtype=np.float64)}
    if metric.startswith("aggregate_image_"):
        return {
            field: np.asarray([
                float((row.get("imageMetrics") or {}).get(field, 0.0)) for row in rows
            ], dtype=np.float64)
            for field in _BOOTSTRAP_IMAGE_FIELDS
        }
    if not metric.startswith("aggregate_"):
        raise ValueError(f"unknown bootstrap metric: {metric}")
    return {
        field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in _BOOTSTRAP_AGG_FIELDS
    }


def _bootstrap_aggregate_metric(
    metric: str,
    sums: dict[str, np.ndarray],
    sample_count: int,
) -> np.ndarray:
    """Evaluate an aggregate metric from sampled per-pose sums."""
    if metric.startswith("aggregate_image_"):
        key = metric[len("aggregate_"):]
        total = np.maximum(1.0, sums["totalPixels"])
        valid = np.maximum(1.0, sums["validReferencePixels"])
        values = {
            "image_per": sums["errorPixels"] / valid,
            "image_miss_pixel_rate": sums["missPixels"] / valid,
            "image_wrong_id_pixel_rate": sums["wrongInstancePixels"] / valid,
            "image_extra_pixel_rate": sums["extraPixels"] / total,
        }
        return values[key]

    key = metric[len("aggregate_"):]
    tp, fp, fn, tn = (sums[name] for name in ("tp", "fp", "fn", "tn"))
    candidate = np.maximum(1.0, tp + fp + fn + tn)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    specificity = tn / np.maximum(1.0, tn + fp)
    values = {
        "precision": precision,
        "recall": recall,
        "weighted_recall": sums["weightedTp"] / np.maximum(1e-12, sums["weightedGt"]),
        "download_utility_recall": sums["downloadUtilityTp"] / np.maximum(1e-12, sums["weightedGt"]),
        "accuracy": (tp + tn) / candidate,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "specificity": specificity,
        "f1": 2.0 * precision * recall / np.maximum(1e-8, precision + recall),
        "jaccard": tp / np.maximum(1.0, tp + fp + fn),
        "useful_cull": tn / candidate,
        "bad_cull": fn / candidate,
        "avg_pred_count": sums["predCount"] / max(1, sample_count),
        "avg_fp_count": fp / max(1, sample_count),
        "avg_fn_count": fn / max(1, sample_count),
        "avg_tn_count": tn / max(1, sample_count),
        "avg_tp_count": tp / max(1, sample_count),
        "pred_over_candidate": sums["predCount"] / candidate,
        "pred_over_gt": sums["predCount"] / np.maximum(1.0, tp + fn),
        "candidate_glb_count": sums["candidateGlbCount"] / max(1, sample_count),
        "predicted_glb_count": sums["predictedGlbCount"] / max(1, sample_count),
        "candidate_glb_bytes": sums["candidateGlbBytes"] / max(1, sample_count),
        "predicted_glb_bytes": sums["predictedGlbBytes"] / max(1, sample_count),
        "glb_count_reduction": sums["glbCountReduction"] / max(1, sample_count),
        "glb_byte_reduction": sums["glbByteReduction"] / max(1, sample_count),
        "glb_bytes_at_achieved_visual_utility": sums["glbBytesAtAchievedVisualUtility"] / max(1, sample_count),
    }
    return values[key]


def _bootstrap_effect(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    return _bootstrap_linear_effect(
        {"left": (1.0, left), "right": (-1.0, right)},
        metric,
        replicates,
        seed,
        label="left_minus_right",
    )


def _bootstrap_linear_effect(
    terms: dict[str, tuple[float, dict[int, dict[str, Any]]]],
    metric: str,
    replicates: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Bootstrap a paired linear contrast over the same seed/pose samples.

    The sampling scheme is unchanged, but the inner pose calculations are
    vectorized in batches.  This is necessary for the registered 10,000
    replicate matrix without weakening the seed-clustered paired protocol.
    """
    if not terms:
        raise ValueError("linear effect needs at least one term")
    members = [member for _name, (_coefficient, member) in terms.items()]
    seeds = sorted(next(iter(members)))
    if any(sorted(member) != seeds for member in members[1:]):
        raise ValueError("paired effect members do not have the same seeds")
    pose_rows = {
        name: {current_seed: member[current_seed]["payload"]["perPose"] for current_seed in seeds}
        for name, (_coefficient, member) in terms.items()
    }
    observed = float(np.mean([
        sum(
            float(coefficient) * _metric_from_rows(pose_rows[name][current_seed], metric)
            for name, (coefficient, _member) in terms.items()
        )
        for current_seed in seeds
    ]))
    pose_count = len(next(iter(pose_rows.values()))[seeds[0]])
    if pose_count <= 0 or any(
        len(pose_rows[name][current_seed]) != pose_count
        for name in pose_rows
        for current_seed in seeds
    ):
        raise ValueError("paired bootstrap members have inconsistent pose counts")
    prepared = []
    for name, (coefficient, _member) in terms.items():
        arrays_by_seed = [
            _bootstrap_row_arrays(pose_rows[name][current_seed], metric)
            for current_seed in seeds
        ]
        prepared.append((float(coefficient), arrays_by_seed))

    rng = np.random.default_rng(int(seed))
    values = np.empty((int(replicates),), dtype=np.float64)
    batch_size = 256
    for batch_start in range(0, int(replicates), batch_size):
        batch_end = min(int(replicates), batch_start + batch_size)
        batch_count = batch_end - batch_start
        chosen_seed_indices = rng.integers(
            0, len(seeds), size=(batch_count, len(seeds)), endpoint=False
        )
        sampled_pose_indices = rng.integers(
            0, pose_count, size=(batch_count, len(seeds), pose_count), endpoint=False
        )
        deltas = np.zeros((batch_count, len(seeds)), dtype=np.float64)
        for coefficient, arrays_by_seed in prepared:
            field_names = tuple(arrays_by_seed[0])
            stacked = np.stack([
                np.stack([arrays[field] for arrays in arrays_by_seed], axis=0)
                for field in field_names
            ], axis=0)
            selected = np.take_along_axis(
                stacked[:, chosen_seed_indices, :],
                sampled_pose_indices[None, ...],
                axis=3,
            )
            if metric.startswith("pose_"):
                term_values = selected[0].mean(axis=2)
            else:
                sums = {
                    field: selected[field_index].sum(axis=2)
                    for field_index, field in enumerate(field_names)
                }
                term_values = _bootstrap_aggregate_metric(metric, sums, pose_count)
            deltas += coefficient * term_values
        values[batch_start:batch_end] = deltas.mean(axis=1)
    ci = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    safe_members = {
        name: all(bool(member[current_seed]["payload"].get("safeWorkpoint", False)) for current_seed in seeds)
        for name, (_coefficient, member) in terms.items()
    }
    return {
        "meanDelta": float(observed),
        "ci95": ci,
        "direction": "positive" if observed > 0 else "negative" if observed < 0 else "zero",
        "crossesZero": bool(ci[0] <= 0.0 <= ci[1]),
        "bootstrapReplicates": int(replicates),
        "clusterUnit": "outer seed cluster, inner pose resampling",
        "paired": True,
        "contrast": label,
        "safeWorkpoints": {
            "membersAllSafe": safe_members,
            "bothAllSafe": bool(all(safe_members.values())),
        },
    }


def _variant_name(context: str, survival: str, loss: str) -> str:
    return f"context_{context}_survival_{survival}_{loss}"


def _factor_contrasts(
    members: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, tuple[float, dict[int, dict[str, Any]]]]]:
    """Return registered main-effect and interaction contrasts for 2x2x2."""
    def terms(pairs: list[tuple[float, str]]) -> dict[str, tuple[float, dict[int, dict[str, Any]]]] | None:
        if any(name not in members for _coefficient, name in pairs):
            return None
        return {name: (coefficient, members[name]) for coefficient, name in pairs}

    contrasts: dict[str, dict[str, tuple[float, dict[int, dict[str, Any]]]]] = {}
    context_terms: list[tuple[float, str]] = []
    survival_terms: list[tuple[float, str]] = []
    loss_terms: list[tuple[float, str]] = []
    for survival in ("off", "on"):
        for loss in ("rvl", "safety"):
            context_terms.extend([
                (0.25, _variant_name("on", survival, loss)),
                (-0.25, _variant_name("off", survival, loss)),
            ])
    for context in ("off", "on"):
        for loss in ("rvl", "safety"):
            survival_terms.extend([
                (0.25, _variant_name(context, "on", loss)),
                (-0.25, _variant_name(context, "off", loss)),
            ])
    for context in ("off", "on"):
        for survival in ("off", "on"):
            loss_terms.extend([
                (0.25, _variant_name(context, survival, "safety")),
                (-0.25, _variant_name(context, survival, "rvl")),
            ])
    for name, values in (
        ("context_main_effect", context_terms),
        ("survival_main_effect", survival_terms),
        ("loss_main_effect", loss_terms),
    ):
        value = terms(values)
        if value is not None:
            contrasts[name] = value

    for loss in ("rvl", "safety"):
        value = terms([
            (1.0, _variant_name("on", "on", loss)),
            (-1.0, _variant_name("on", "off", loss)),
            (-1.0, _variant_name("off", "on", loss)),
            (1.0, _variant_name("off", "off", loss)),
        ])
        if value is not None:
            contrasts[f"context_survival_interaction_{loss}"] = value
    for survival in ("off", "on"):
        value = terms([
            (1.0, _variant_name("on", survival, "safety")),
            (-1.0, _variant_name("on", survival, "rvl")),
            (-1.0, _variant_name("off", survival, "safety")),
            (1.0, _variant_name("off", survival, "rvl")),
        ])
        if value is not None:
            contrasts[f"context_loss_interaction_{survival}"] = value
    for context in ("off", "on"):
        value = terms([
            (1.0, _variant_name(context, "on", "safety")),
            (-1.0, _variant_name(context, "on", "rvl")),
            (-1.0, _variant_name(context, "off", "safety")),
            (1.0, _variant_name(context, "off", "rvl")),
        ])
        if value is not None:
            contrasts[f"survival_loss_interaction_{context}"] = value
    value = terms([
        (1.0, _variant_name("on", "on", "safety")),
        (-1.0, _variant_name("off", "on", "safety")),
        (-1.0, _variant_name("on", "off", "safety")),
        (1.0, _variant_name("off", "off", "safety")),
        (-1.0, _variant_name("on", "on", "rvl")),
        (1.0, _variant_name("off", "on", "rvl")),
        (1.0, _variant_name("on", "off", "rvl")),
        (-1.0, _variant_name("off", "off", "rvl")),
    ])
    if value is not None:
        contrasts["context_survival_loss_three_way_interaction"] = value
    return contrasts


def _validation_member_safety(
    members: dict[str, dict[int, dict[str, Any]]],
    contrast: dict[str, tuple[float, dict[int, dict[str, Any]]]],
) -> dict[str, bool]:
    """Check the validation weighted-recall guard for every contrast term.

    ``contrast`` maps variant names to ``(coefficient, member-table)``.  Keep
    the variant name as the key here; using the tuple value as a lookup key
    silently fails with ``unhashable type: 'dict'`` and would prevent the
    formal summary from being generated.
    """
    return {
        term: all(
            float(members[term][seed]["payload"].get("poseMacro", {}).get("weightedRecall", 0.0)) > 0.99
            and float(members[term][seed]["payload"].get("aggregate", {}).get("weightedRecall", 0.0)) > 0.99
            for seed in sorted(members[term])
        )
        for term, (_coefficient, _member) in contrast.items()
    }


def build_summary(
    manifest_path: Path,
    output_path: Path,
    report_path: Path | None,
    replicates: int,
    webgpu_parity_path: Path | None = None,
) -> dict[str, Any]:
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants, comparisons = _load_manifest(manifest_path)
    formal = manifest_payload.get("mode") == "formal"
    supplement = bool(manifest_payload.get("supplement", False))
    members = _load_members(variants, require_image_metrics=formal)
    seeds, pose_indices, hashes = _validate_alignment(members)
    image_available = all(
        (member["payload"].get("imageMetrics") or {}).get("status") == "formal_ready"
        and (member["payload"].get("imageMetrics") or {}).get("perPoseAvailable") is True
        for by_seed in members.values()
        for member in by_seed.values()
    )
    if formal:
        if supplement:
            if tuple(variants) != SUPPLEMENT_FORMAL_VARIANTS:
                raise ValueError(
                    "formal OWRB supplement must contain the registered five variants "
                    f"in order, got {list(variants)}"
                )
        elif len(variants) != CORE_FORMAL_VARIANT_COUNT:
            raise ValueError(
                f"formal OWRB core matrix must contain {CORE_FORMAL_VARIANT_COUNT} variants, "
                f"got {len(variants)}"
            )
        if len(seeds) != 3:
            raise ValueError(f"formal OWRB matrix must contain 3 seeds, got {len(seeds)}")
        if int(replicates) < 10000:
            raise ValueError("formal OWRB summary requires at least 10,000 paired bootstrap replicates")
    elif int(replicates) < 2000:
        raise ValueError("OWRB bootstrap requires at least 2,000 replicates")
    metric_names = ALL_METRICS if image_available else BASE_METRICS
    member_summaries: dict[str, Any] = {}
    for variant, by_seed in members.items():
        member_summaries[variant] = {}
        for seed in seeds:
            payload = by_seed[seed]["payload"]
            member_summaries[variant][str(seed)] = {
                "path": str(by_seed[seed]["path"]),
                "seed": int(seed),
                "checkpoint": payload["checkpoint"],
                "runtimeFeatures": payload["runtimeFeatures"],
                "threshold": payload["threshold"],
                "thresholdSource": payload["thresholdSource"],
                "safeWorkpoint": bool(payload.get("safeWorkpoint", payload["thresholdSource"].get("safeWorkpoint", False))),
                "poseMacro": payload["poseMacro"],
                "aggregate": payload["aggregate"],
                "runtimeLatency": payload["runtimeLatency"],
                "runtimeFeatureBytes": payload.get("runtimeFeatureBytes"),
                "runtimeFeatureDim": payload.get("runtimeFeatureDim", payload.get("runtimeFeatureTableDim")),
                "runtimeFeatureTableDim": payload.get("runtimeFeatureTableDim", payload.get("runtimeFeatureDim")),
                "runtimeInputFeatureDim": payload.get("runtimeInputFeatureDim"),
                "visibilityHeadInputDim": payload.get("visibilityHeadInputDim", payload.get("runtimeInputFeatureDim")),
                "rayQueryDim": payload.get("rayQueryDim"),
                "imageMetrics": payload.get("imageMetrics"),
                "safetyFactor": float(
                    min(1.0, float(payload["poseMacro"].get("recall", 0.0)) / 0.95)
                    * min(1.0, float(payload["poseMacro"].get("weightedRecall", 0.0)) / 0.99)
                ),
                "safetyAdjustedUsefulCull": float(
                    payload["poseMacro"].get("usefulCull", 0.0)
                    * min(1.0, float(payload["poseMacro"].get("recall", 0.0)) / 0.95)
                    * min(1.0, float(payload["poseMacro"].get("weightedRecall", 0.0)) / 0.99)
                ),
                "validationSafeWorkpoint": bool(
                    float(payload["poseMacro"].get("weightedRecall", 0.0)) > 0.99
                    and float(payload["aggregate"].get("weightedRecall", 0.0)) > 0.99
                ),
            }
    effects: dict[str, Any] = {}
    for comparison in comparisons:
        left_name, right_name = comparison["left"], comparison["right"]
        if left_name not in members or right_name not in members:
            raise ValueError(f"comparison references unknown variant: {comparison}")
        effects[comparison["name"]] = {
            "left": left_name,
            "right": right_name,
            "metrics": {
                metric: _bootstrap_effect(members[left_name], members[right_name], metric, replicates, 20260810 + index)
                for index, metric in enumerate(metric_names)
            },
        }

    factor_effects: dict[str, Any] = {}
    for contrast_index, (name, contrast) in enumerate(_factor_contrasts(members).items()):
        metrics = {
            metric: _bootstrap_linear_effect(
                contrast, metric, replicates, 20260810 + 1000 + contrast_index,
                label=name,
            )
            for metric in metric_names
        }
        # Route selection consumes safety qualification at the factor-effect
        # level.  Keep the member qualification map beside the contrast so a
        # full-factorial effect cannot accidentally look unsafe merely because
        # its per-metric bootstrap records are nested one level deeper.
        first_metric = next(iter(metrics.values()), {})
        validation_member_safe = _validation_member_safety(members, contrast)
        factor_effects[name] = {
            "terms": {term: coefficient for term, (coefficient, _member) in contrast.items()},
            "metrics": metrics,
            "safeWorkpoints": first_metric.get("safeWorkpoints", {}),
            "validationSafeWorkpoints": {
                "membersAllSafe": validation_member_safe,
                "bothAllSafe": bool(all(validation_member_safe.values())),
            },
        }
    summary: dict[str, Any] = {
        "schema": "ray-context-survival-owrb-matrix-summary-v1",
        "manifest": str(manifest_path.resolve()),
        "mode": manifest_payload.get("mode"),
        "supplement": supplement,
        "variants": list(members),
        "seeds": seeds,
        "poseCount": len(pose_indices),
        "poseIndices": pose_indices,
        "candidateHashDigest": _sha256_text(json.dumps(hashes, sort_keys=True)),
        "members": member_summaries,
        "comparisons": effects,
        "factorEffects": factor_effects,
        "metricNames": list(metric_names),
        "bootstrap": {
            "replicates": int(replicates),
            "confidence": 0.95,
            "paired": True,
            "clusterUnit": "outer seed cluster, inner pose resampling",
        },
        "matrixDesign": (
            {
                "type": "registered_supplement_mechanism_audit",
                "registeredVariantSet": "ray_context_survival_owrb_supplement_v1",
                "variantDefinitions": manifest_payload.get("data", {}).get("variantDefinitions", {}),
                "variantCount": len(members),
                "derivedContrastCount": len(factor_effects),
            }
            if supplement
            else {
                "type": "registered_2x2x2_factorial",
                "factors": {
                    "context": {"off": "pooled global relation aggregation", "on": "directional spherical relation aggregation"},
                    "survival": {"off": "disabled", "on": "semantic monotone survival field"},
                    "loss": {"rvl": "rvl_strong_v2", "safety": "safety_constraint"},
                },
                "variantCount": len(members),
                "derivedContrastCount": len(factor_effects),
            }
        ),
        "testRead": False,
        "safetyPolicy": "weighted recall and its checkpoint calibration lower bound are primary; pose recall is diagnostic",
        "imageEvaluation": {
            "available": bool(image_available),
            "status": "formal_ready" if image_available else "not_available",
            "metrics": list(IMAGE_METRICS) if image_available else [],
        },
        "webgpuParity": _load_webgpu_parity(webgpu_parity_path),
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if report_path is not None:
        write_report(summary, report_path)
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    metric_names = tuple(summary.get("metricNames", BASE_METRICS))
    image_available = bool((summary.get("imageEvaluation") or {}).get("available", False))
    webgpu = summary.get("webgpuParity") or {}
    webgpu_status = str(webgpu.get("status", "not_registered"))
    if webgpu_status == "hardware_numeric_parity_passed":
        webgpu_line = (
            f"WebGPU 数值 parity：已通过 NVIDIA 硬件门，最大绝对误差 "
            f"{_fmt(webgpu.get('maxAbs'), 8)}，最大相对误差 {_fmt(webgpu.get('maxRelative'), 8)}。"
        )
    elif webgpu_status == "software_numeric_parity_passed":
        webgpu_line = (
            f"WebGPU 数值 parity：已通过软件后端数值校验，最大绝对误差 "
            f"{_fmt(webgpu.get('maxAbs'), 8)}，最大相对误差 {_fmt(webgpu.get('maxRelative'), 8)}；"
            "当前适配器未通过 NVIDIA WebGPU 硬件门，不能据此报告硬件 WebGPU 延迟。"
        )
    else:
        webgpu_line = "WebGPU 数值 parity：未登记；不能据此报告浏览器 WebGPU 结果。"
    supplement = bool(summary.get("supplement", False))
    title = "视线关系场、生存场与 OWRB 补充矩阵评价" if supplement else "视线关系场、生存场与 OWRB 矩阵评价"
    lines = [
        f"# {title}",
        "",
        (
            "本报告只汇总 calibration 冻结阈值后的 validation 结果。补充矩阵是独立的五变体机制对照，"
            "不生成核心 2×2×2 因子主效应；所有成员使用同一 pose 顺序、后退相机候选集合、实例 GT 和候选哈希。"
            if supplement
            else "本报告只汇总 calibration 冻结阈值后的 validation 结果。所有成员使用同一 pose 顺序、后退相机候选集合、实例 GT 和候选哈希。"
        ),
        "paired bootstrap 先按 seed 重采样，再在每个 seed 内按 pose 重采样。",
        "",
        f"- 变体：{len(summary['variants'])}；seed：{len(summary['seeds'])}；pose：{summary['poseCount']}。",
        f"- bootstrap：{summary['bootstrap']['replicates']} 次；test split 读取：否。",
        "- weighted recall 是画面安全主指标；useful cull 必须与 bad cull、recall 和 weighted recall 一起解释。",
        f"- Color-ID 图像评价：{'已完成硬件 GPU 正式评价' if image_available else '未完成，不能用集合指标替代'}。",
        f"- {webgpu_line}",
        "",
        "## 画面安全指标",
        "",
        "| 变体 | seed | 阈值 | pose recall | aggregate recall | pose weighted recall | aggregate weighted recall | pose useful cull | aggregate useful cull | pose bad cull | aggregate bad cull | safety-adjusted useful cull |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in summary["variants"]:
        for seed in summary["seeds"]:
            member = summary["members"][variant][str(seed)]
            pose = member["poseMacro"]
            aggregate = member["aggregate"]
            lines.append(
                f"| `{variant}` | {seed} | {_fmt(member.get('threshold'), 6)} | {_fmt(pose.get('recall'))} | {_fmt(aggregate.get('recall'))} | {_fmt(pose.get('weightedRecall'))} | {_fmt(aggregate.get('weightedRecall'))} | {_fmt(pose.get('usefulCull'))} | {_fmt(aggregate.get('usefulCull'))} | {_fmt(pose.get('badCull'))} | {_fmt(aggregate.get('badCull'))} | {_fmt(member.get('safetyAdjustedUsefulCull'))} |"
            )
    lines.extend([
        "",
        "## 分类诊断指标",
        "",
        "| 变体 | seed | pose precision | aggregate precision | pose F1 | aggregate F1 | pose Jaccard | aggregate Jaccard | pose accuracy | aggregate accuracy | pose balanced acc. | aggregate balanced acc. | pose specificity | aggregate specificity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in summary["variants"]:
        for seed in summary["seeds"]:
            member = summary["members"][variant][str(seed)]
            pose = member["poseMacro"]
            aggregate = member["aggregate"]
            lines.append(
                f"| `{variant}` | {seed} | {_fmt(pose.get('precision'))} | {_fmt(aggregate.get('precision'))} | {_fmt(pose.get('f1'))} | {_fmt(aggregate.get('f1'))} | {_fmt(pose.get('jaccard'))} | {_fmt(aggregate.get('jaccard'))} | {_fmt(pose.get('accuracy'))} | {_fmt(aggregate.get('accuracy'))} | {_fmt(pose.get('balancedAccuracy'))} | {_fmt(aggregate.get('balancedAccuracy'))} | {_fmt(pose.get('specificity'))} | {_fmt(aggregate.get('specificity'))} |"
            )
    lines.extend([
        "",
        "## 剔除、资源与运行成本",
        "",
        "| 变体 | seed | 平均预测数 | pose 平均 FP | pose 平均 FN | pose 平均 TN | aggregate 平均 FP | aggregate 平均 FN | aggregate 平均 TN | 预测/候选 | 预测/GT | 候选 GLB 数 | 预测 GLB 数 | 候选 GLB 字节 | 预测 GLB 字节 | GLB 数量削减 | GLB 字节削减 | 下载效用召回 | 达到同等视觉效用所需字节 | 前向 p95 ms | 特征表字节 | 固定表维度 | 头输入维度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in summary["variants"]:
        for seed in summary["seeds"]:
            member = summary["members"][variant][str(seed)]
            pose = member["poseMacro"]
            aggregate = member["aggregate"]
            latency = member.get("runtimeLatency", {})
            lines.append(
                f"| `{variant}` | {seed} | {_fmt(aggregate.get('avgPredCount'))} | {_fmt(pose.get('fp'))} | {_fmt(pose.get('fn'))} | {_fmt(pose.get('tn'))} | {_fmt(aggregate.get('avgFpCount'))} | {_fmt(aggregate.get('avgFnCount'))} | {_fmt(aggregate.get('avgTnCount'))} | {_fmt(aggregate.get('predOverCandidate'))} | {_fmt(aggregate.get('predOverGt'))} | {_fmt(aggregate.get('candidateGlbCount'))} | {_fmt(aggregate.get('predictedGlbCount'))} | {_fmt(aggregate.get('candidateGlbBytes'))} | {_fmt(aggregate.get('predictedGlbBytes'))} | {_fmt(aggregate.get('glbCountReduction'))} | {_fmt(aggregate.get('glbByteReduction'))} | {_fmt(aggregate.get('downloadUtilityRecall'))} | {_fmt(aggregate.get('glbBytesAtAchievedVisualUtility'))} | {_fmt(latency.get('p95Ms'), 3)} | {member.get('runtimeFeatureBytes', 'n/a')} | {member.get('runtimeFeatureTableDim', 'n/a')} | {member.get('visibilityHeadInputDim', 'n/a')} |"
            )
    if image_available:
        lines.extend([
            "",
            "## 图像质量",
            "",
            "| 变体 | seed | PER | mean miss-pixel | p95 miss-pixel | wrong-ID pixel | extra-pixel |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for variant in summary["variants"]:
            for seed in summary["seeds"]:
                image = summary["members"][variant][str(seed)].get("imageMetrics") or {}
                lines.append(
                    f"| `{variant}` | {seed} | {_fmt(image.get('PER'))} | {_fmt(image.get('meanMissPixelRate'))} | {_fmt(image.get('p95MissPixelRate'))} | {_fmt(image.get('meanWrongInstancePixelRate'))} | {_fmt(image.get('meanExtraPixelRateOverImage'))} |"
                )
    else:
        lines.extend(["", "## 图像质量", "", "Color-ID miss-pixel、wrong-ID pixel 和 extra-pixel 尚未完成；本报告不对图像质量作结论。"])

    lines.extend([
        "",
        "## 配对差值和因子效应",
        "",
        (
            "差值定义为左侧减右侧；补充矩阵只报告注册的单因素机制对照，不把它们包装成完整因子主效应。"
            if supplement
            else "差值定义为左侧减右侧；派生因子效应使用完整 2×2×2 对比并保持同一 seed/pose 重采样。"
        ),
        "所有区间均保持同一 seed/pose 配对重采样；区间跨零时不宣称稳定改善。",
        "",
        "| 比较 | 指标 | 差值 | 95% 区间 | 方向 | 跨零 |",
        "|---|---|---:|---|---|---|",
    ])
    report_effect_metrics = [
        "pose_precision", "aggregate_precision", "pose_recall", "aggregate_recall",
        "pose_weighted_recall", "aggregate_weighted_recall", "pose_accuracy", "aggregate_accuracy",
        "pose_balanced_accuracy", "aggregate_balanced_accuracy", "pose_f1", "aggregate_f1",
        "pose_useful_cull", "aggregate_useful_cull", "pose_bad_cull", "aggregate_bad_cull",
        "pose_avg_fp_count", "pose_avg_fn_count", "pose_avg_tn_count",
        "aggregate_avg_fp_count", "aggregate_avg_fn_count", "aggregate_avg_tn_count",
        "pose_avg_pred_count", "aggregate_avg_pred_count", "aggregate_predicted_glb_bytes",
        "pose_download_utility_recall", "aggregate_download_utility_recall",
        "pose_glb_bytes_at_achieved_visual_utility", "aggregate_glb_bytes_at_achieved_visual_utility",
        "aggregate_glb_byte_reduction",
    ]
    if image_available:
        report_effect_metrics.extend([
            "pose_image_per", "aggregate_image_per", "pose_image_miss_pixel_rate",
            "aggregate_image_miss_pixel_rate", "pose_image_wrong_id_pixel_rate",
            "aggregate_image_extra_pixel_rate",
        ])
    effect_items = list(summary["comparisons"].items()) + list(summary.get("factorEffects", {}).items())
    for name, effect in effect_items:
        for metric in report_effect_metrics:
            if metric not in metric_names or metric not in effect.get("metrics", {}):
                continue
            item = effect["metrics"][metric]
            ci = item.get("ci95", [None, None])
            lines.append(
                f"| `{name}` | `{metric}` | {_fmt(item.get('meanDelta'))} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | {item.get('direction', 'n/a')} | {'是' if item.get('crossesZero') else '否'} |"
            )
    lines.extend([
        "",
        "## useful cull 的解释边界",
        "",
        "例如某个 pose 有 1000 个候选实例、100 个真实可见实例，模型只预测 10 个且其中 10 个恰好正确，则 TP=10、FP=0、FN=90、TN=900。此时 useful cull=0.90，看起来很高，但 recall=0.10、bad cull=0.09；若漏掉的实例具有较大 visible weight，weighted recall 也会明显下降。这种结果是错误剔除风险，不能解释为模型性能更好。",
        "",
        "方向代理、上下文和损失只有在 weighted recall 安全性不恶化，并且 precision、balanced accuracy、F1、有效剔除、图像质量或资源成本至少一项的配对置信区间稳定改善时，才可写成预测或系统贡献。",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> dict[str, Any]:
    rows = [
        {"poseIndex": 0, "candidateIdSha256": "a", "precision": 1.0, "recall": 1.0, "weightedRecall": 1.0, "accuracy": 1.0, "balancedAccuracy": 1.0, "f1": 1.0, "usefulCull": 0.5, "badCull": 0.0, "predCount": 1, "predictedGlbBytes": 10.0, "predictedGlbCount": 1, "candidateGlbCount": 2, "candidateGlbBytes": 30.0, "glbCountReduction": 0.5, "glbByteReduction": 2.0 / 3.0, "downloadUtilityTp": 1.0, "glbBytesAtAchievedVisualUtility": 10.0, "tp": 1, "fp": 0, "fn": 0, "tn": 1, "weightedTp": 1.0, "weightedGt": 1.0},
        {"poseIndex": 1, "candidateIdSha256": "b", "precision": 0.5, "recall": 1.0, "weightedRecall": 1.0, "accuracy": 0.5, "balancedAccuracy": 0.5, "f1": 2.0 / 3.0, "usefulCull": 0.0, "badCull": 0.0, "predCount": 2, "predictedGlbBytes": 20.0, "predictedGlbCount": 2, "candidateGlbCount": 2, "candidateGlbBytes": 30.0, "glbCountReduction": 0.0, "glbByteReduction": 0.0, "downloadUtilityTp": 1.0, "glbBytesAtAchievedVisualUtility": 20.0, "tp": 1, "fp": 1, "fn": 0, "tn": 0, "weightedTp": 1.0, "weightedGt": 1.0},
    ]
    aggregate = _aggregate_sample(rows)
    assert abs(aggregate["precision"] - 0.6666666667) < 1e-5
    assert abs(_metric_from_rows(rows, "pose_precision") - 0.75) < 1e-6
    return {"status": "ok", "aggregatePrecision": aggregate["precision"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--webgpu-parity",
        type=Path,
        default=None,
        help="separate WebGPU parity validation JSON; hardware status is reported without inference",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
        return
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required unless --self-test is used")
    summary = build_summary(
        args.manifest,
        args.output,
        args.report,
        args.bootstrap_replicates,
        args.webgpu_parity,
    )
    print(json.dumps({"status": "summarized", "output": str(args.output.resolve()), "variantCount": len(summary["variants"]), "seedCount": len(summary["seeds"]), "poseCount": summary["poseCount"], "bootstrapReplicates": args.bootstrap_replicates, "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
