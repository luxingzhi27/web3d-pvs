#!/usr/bin/env python3
"""Shared validation, metric, and bootstrap helpers for M4-v2.

This module deliberately operates on the stored per-pose intervention records.
It never builds a candidate set by unioning GT IDs and never reads the test
split.  The two metric scopes are kept separate:

* ``pose_macro`` averages the metric computed independently for each pose.
* ``aggregate`` first sums TP/FP/FN/TN (and visible weights) over poses.

The distinction matters for PVS because candidate counts vary considerably
between view cells.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SEEDS = (20260801, 20260802, 20260803)
STAGED_VARIANTS = ("aabb_ray", "geometry_ray")
FACTOR_VARIANTS = (
    "geometry_context_ray_no_inhibition",
    "geometry_context_proxy_ray_no_inhibition",
    "geometry_context_ray",
    "full",
)
ALL_VARIANTS = STAGED_VARIANTS + FACTOR_VARIANTS
FACTOR_CODES = {
    "geometry_context_ray_no_inhibition": "A",
    "geometry_context_proxy_ray_no_inhibition": "B",
    "geometry_context_ray": "C",
    "full": "D",
}
FACTOR_EFFECTS = {
    "B_minus_A": {"geometry_context_proxy_ray_no_inhibition": 1.0, "geometry_context_ray_no_inhibition": -1.0},
    "C_minus_A": {"geometry_context_ray": 1.0, "geometry_context_ray_no_inhibition": -1.0},
    "D_minus_C": {"full": 1.0, "geometry_context_ray": -1.0},
    "D_minus_B": {"full": 1.0, "geometry_context_proxy_ray_no_inhibition": -1.0},
    "interaction_D_minus_B_minus_C_plus_A": {
        "full": 1.0,
        "geometry_context_proxy_ray_no_inhibition": -1.0,
        "geometry_context_ray": -1.0,
        "geometry_context_ray_no_inhibition": 1.0,
    },
}
STAGED_EFFECTS = {
    "geometry_minus_aabb": {"geometry_ray": 1.0, "aabb_ray": -1.0},
    "context_minus_geometry": {"geometry_context_ray": 1.0, "geometry_ray": -1.0},
}
ALL_EFFECTS = {**STAGED_EFFECTS, **FACTOR_EFFECTS}

COUNT_FIELDS = ("tp", "fp", "fn", "tn")
CORE_METRICS = (
    "precision",
    "recall",
    "weighted_recall",
    "f1",
    "jaccard",
    "accuracy",
    "balanced_accuracy",
    "specificity",
    "useful_cull",
    "bad_cull",
)
DIAGNOSTIC_METRICS = CORE_METRICS + (
    "avg_tp_count",
    "avg_fp_count",
    "avg_fn_count",
    "avg_tn_count",
    "avg_pred_count",
    "avg_candidate_count",
    "avg_gt_count",
    "pred_over_candidate",
    "pred_over_gt",
)
UNAVAILABLE_METRICS = {
    "visual_utility_recall": "not_available: interventions.json has visible_weights but no unified visual-utility target",
    "miss_pixel_rate": "not_available: no same-pose 60-degree instance-ID render attached to this intervention",
    "wrong_id_pixel_rate": "not_available: no same-pose 60-degree instance-ID render attached to this intervention",
    "extra_pixel_rate": "not_available: no same-pose 60-degree instance-ID render attached to this intervention",
    "glb_count_reduction": "not_available: intervention rows do not persist predicted/candidate GLB sets",
    "glb_byte_reduction": "not_available: intervention rows do not persist predicted/candidate GLB sets",
    "equal_visual_utility_glb_bytes": "not_available: no paired visual-utility/GLB budget curve attached",
    "forward_latency_ms": "not_available: stored interventions contain no forward timing",
    "webgpu_latency_ms": "not_available: no browser timing attached to this matrix",
    "main_thread_ms": "not_available: no browser timing attached to this matrix",
    "runtime_memory_bytes": "not_available: no browser memory sample attached to this matrix",
}


def experiment_name(variant: str, seed: int, v2: bool = False) -> str:
    prefix = "pvs_m4_v2_ablation" if v2 else "pvs_m4_ablation"
    return f"{prefix}_{variant}_rvl_strong_v2_hkust_spatial_fov66_seed{seed}_full40"


def old_intervention_path(root: Path, variant: str, seed: int) -> Path:
    return root / f"m4_formal_baseline_{experiment_name(variant, seed)}_validation" / "interventions.json"


def v2_intervention_path(root: Path, variant: str, seed: int) -> Path:
    return root / "members" / variant / f"seed{seed}" / "interventions.json"


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def validate_threshold_source(payload: Mapping[str, Any], path: Path) -> None:
    source = payload.get("thresholdSource")
    if not isinstance(source, Mapping):
        raise ValueError(f"missing structured thresholdSource: {path}")
    if source.get("protocol") != "calibration_ready_pre_test":
        raise ValueError(f"non-calibration threshold protocol in {path}: {source.get('protocol')!r}")
    if int(source.get("testEvaluationCount", -1)) != 0:
        raise ValueError(f"test-derived threshold provenance in {path}")
    if bool(source.get("testThresholdOverride", False)):
        raise ValueError(f"test threshold override in {path}")


def validate_row(row: Mapping[str, Any], path: Path, expected_pose_count: int | None = None) -> None:
    required = {"pose_index", "candidate_id_sha256", "candidate_count", "gt_count", *COUNT_FIELDS}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{path} pose {row.get('pose_index')} missing fields {missing}")
    pose = int(row["pose_index"])
    candidate = int(row["candidate_count"])
    gt = int(row["gt_count"])
    counts = [int(row[name]) for name in COUNT_FIELDS]
    if pose < 0 or candidate < 0 or gt < 0 or min(counts) < 0:
        raise ValueError(f"negative count in {path}, pose {pose}")
    if sum(counts) != candidate:
        raise ValueError(f"candidate confusion counts do not close in {path}, pose {pose}")
    if counts[0] + counts[2] != gt:
        raise ValueError(f"GT confusion counts do not close in {path}, pose {pose}")
    if gt > candidate:
        raise ValueError(f"GT is not a subset of candidates in {path}, pose {pose}")
    candidate_hash = str(row["candidate_id_sha256"])
    if len(candidate_hash) != 64:
        raise ValueError(f"invalid candidate hash in {path}, pose {pose}")
    for key, value in row.items():
        if key in {"pose_index", "candidate_id_sha256", "base_logit", "final_logit", "suppression", "gate_entropy", "selected_proxy_abs"}:
            continue
        if isinstance(value, (int, float)):
            finite(value, f"{path}:{pose}:{key}")
    # ``pose_index`` is the original dataset pose id, not the zero-based row
    # position in the frozen validation subset.  A 664-pose evaluation can
    # therefore legitimately contain ids such as 2991 and 193.  Cardinality
    # and uniqueness are checked by ``read_member`` after all rows are read;
    # this row-level validator must not confuse the two domains.


def _check_dependency_hash(payload: Mapping[str, Any], path: Path, repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, sha_field in (("checkpoint", "checkpointSha256"), ("runtimeFeatures", "runtimeFeaturesSha256")):
        value = payload.get(field)
        if value is None:
            raise ValueError(f"{path} has no {field}")
        dependency = resolve_repo_path(value, repo_root)
        if not dependency.is_file():
            raise FileNotFoundError(f"{path} dependency is missing: {dependency}")
        actual = sha256_file(dependency)
        if str(payload.get(sha_field)) != actual:
            raise ValueError(f"{path} {field} SHA mismatch")
        result[field] = str(dependency.resolve())
        result[f"{field}Sha256"] = actual
        result[f"{field}Bytes"] = dependency.stat().st_size
    return result


def read_member(path: Path, variant: str, seed: int, repo_root: Path, expected_pose_count: int = 664) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pvs-proxy-intervention-v2":
        raise ValueError(f"unexpected intervention schema in {path}: {payload.get('schema')!r}")
    if payload.get("split") != "validation":
        raise ValueError(f"non-validation intervention in {path}")
    validate_threshold_source(payload, path)
    if int(payload.get("posePlan", {}).get("poseCount", -1)) != expected_pose_count:
        raise ValueError(f"pose count mismatch in {path}")
    rows = payload.get("perPose", {}).get("baseline")
    if not isinstance(rows, list) or len(rows) != expected_pose_count:
        raise ValueError(f"baseline rows mismatch in {path}")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"non-object pose row in {path}")
        validate_row(row, path, expected_pose_count)
        pose = int(row["pose_index"])
        if pose in indexed:
            raise ValueError(f"duplicate pose {pose} in {path}")
        indexed[pose] = row
    if len(indexed) != expected_pose_count:
        raise ValueError(f"unique pose count mismatch in {path}")
    dependencies = _check_dependency_hash(payload, path, repo_root)
    return {
        "variant": variant,
        "factorCode": FACTOR_CODES.get(variant),
        "seed": int(seed),
        "path": str(path.resolve()),
        "artifactSha256": sha256_file(path),
        "payload": payload,
        "rows": indexed,
        **dependencies,
    }


def candidate_identity(member: Mapping[str, Any]) -> dict[int, tuple[str, int]]:
    return {
        int(pose): (str(row["candidate_id_sha256"]), int(row["candidate_count"]))
        for pose, row in member["rows"].items()
    }


def candidate_digest(identity: Mapping[int, tuple[str, int]]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_candidate_matrix(members: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
    if not members:
        raise ValueError("M4-v2 matrix is empty")
    reference_key = sorted(members)[0]
    reference = candidate_identity(members[reference_key])
    for key, member in members.items():
        current = candidate_identity(member)
        if current != reference:
            for pose in sorted(set(reference) | set(current)):
                if reference.get(pose) != current.get(pose):
                    raise ValueError(f"candidate mismatch at pose {pose}: {reference_key} vs {key}")
    pose_sets = {tuple(sorted(member["rows"])) for member in members.values()}
    if len(pose_sets) != 1:
        raise ValueError("M4-v2 members do not share the same pose set")
    return {
        "poseCount": len(reference),
        "candidateDigest": candidate_digest(reference),
        "source": "stored back-camera candidate hash/count; no GT union, candidate cap, or frontend whitelist",
    }


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def _metrics_from_counts(tp: float, fp: float, fn: float, tn: float, weighted_tp: float, weighted_gt: float, pred: float, candidate: float, gt: float) -> dict[str, float]:
    recall = _safe_div(tp, tp + fn, 1.0)
    specificity = _safe_div(tn, tn + fp, 1.0)
    precision = _safe_div(tp, tp + fp, 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "weighted_recall": _safe_div(weighted_tp, weighted_gt),
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "jaccard": _safe_div(tp, tp + fp + fn),
        "accuracy": _safe_div(tp + tn, candidate),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "specificity": specificity,
        "useful_cull": _safe_div(tn, candidate),
        "bad_cull": _safe_div(fn, candidate),
        "avg_tp_count": tp,
        "avg_fp_count": fp,
        "avg_fn_count": fn,
        "avg_tn_count": tn,
        "avg_pred_count": pred,
        "avg_candidate_count": candidate,
        "avg_gt_count": gt,
        "pred_over_candidate": _safe_div(pred, candidate),
        "pred_over_gt": _safe_div(pred, gt),
    }


def _row_arrays(rows: Iterable[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("cannot summarize an empty pose set")
    arrays: dict[str, np.ndarray] = {}
    for name in COUNT_FIELDS + ("weighted_tp", "weighted_gt", "avg_pred_count", "candidate_count", "gt_count"):
        arrays[name] = np.asarray([float(row[name]) for row in rows_list], dtype=np.float64)
    for name in CORE_METRICS:
        arrays[name] = np.asarray([float(row[name]) for row in rows_list], dtype=np.float64)
    return arrays


def summarize_rows(rows: Iterable[Mapping[str, Any]], lcb_replicates: int = 10000, lcb_seed: int = 0) -> dict[str, Any]:
    arrays = _row_arrays(rows)
    count = int(arrays["candidate_count"].size)
    pose_macro = {name: float(arrays[name].mean()) for name in CORE_METRICS}
    for name in ("tp", "fp", "fn", "tn", "weighted_tp", "weighted_gt", "avg_pred_count", "candidate_count", "gt_count"):
        key = {
            "tp": "avg_tp_count",
            "fp": "avg_fp_count",
            "fn": "avg_fn_count",
            "tn": "avg_tn_count",
            "candidate_count": "avg_candidate_count",
            "gt_count": "avg_gt_count",
        }.get(name, name)
        pose_macro[key] = float(arrays[name].mean())
    pose_macro["pred_over_candidate"] = _safe_div(float(arrays["avg_pred_count"].mean()), float(arrays["candidate_count"].mean()))
    pose_macro["pred_over_gt"] = _safe_div(float(arrays["avg_pred_count"].mean()), float(arrays["gt_count"].mean()))
    aggregate = _metrics_from_counts(
        *(float(arrays[name].sum()) for name in COUNT_FIELDS),
        float(arrays["weighted_tp"].sum()),
        float(arrays["weighted_gt"].sum()),
        float(arrays["avg_pred_count"].sum()),
        float(arrays["candidate_count"].sum()),
        float(arrays["gt_count"].sum()),
    )
    # Ratios use all-pose totals, while count diagnostics follow the existing
    # benchmark convention and are reported per pose.
    for name in ("avg_tp_count", "avg_fp_count", "avg_fn_count", "avg_tn_count", "avg_pred_count", "avg_candidate_count", "avg_gt_count"):
        aggregate[name] /= float(count)
    weighted_lcb = weighted_recall_lcb(arrays["weighted_tp"], arrays["weighted_gt"], lcb_replicates, lcb_seed)
    for scope in (pose_macro, aggregate):
        scope["weighted_recall_lower_confidence_bound"] = float(weighted_lcb)
        scope["safety_factor"] = min(1.0, scope["recall"] / 0.95) * min(1.0, scope["weighted_recall"] / 0.99)
        scope["safety_adjusted_useful_cull"] = scope["useful_cull"] * scope["safety_factor"]
    return {
        "poseCount": count,
        "poseMacro": pose_macro,
        "aggregate": aggregate,
        "safety": {
            "poseRecallFloor": 0.95,
            "weightedRecallFloor": 0.99,
            "weightedRecallLowerConfidenceBoundFloor": 0.99,
            "poseRecallSatisfied": pose_macro["recall"] >= 0.95,
            "weightedRecallSatisfied": pose_macro["weighted_recall"] > 0.99,
            "weightedRecallLowerConfidenceBoundSatisfied": weighted_lcb > 0.99,
            "qualifiedSafetyWorkpoint": bool(
                pose_macro["recall"] >= 0.95 and pose_macro["weighted_recall"] > 0.99 and weighted_lcb > 0.99
            ),
        },
    }


def weighted_recall_lcb(weighted_tp: np.ndarray, weighted_gt: np.ndarray, replicates: int, seed: int) -> float:
    if replicates <= 0:
        return float(np.sum(weighted_tp) / max(1e-12, np.sum(weighted_gt)))
    if weighted_tp.size == 0 or weighted_tp.size != weighted_gt.size:
        return 0.0
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, weighted_tp.size, size=(int(replicates), weighted_tp.size), endpoint=False)
    numerator = weighted_tp[indices].sum(axis=1)
    denominator = weighted_gt[indices].sum(axis=1)
    values = numerator / np.maximum(1e-12, denominator)
    return float(np.quantile(values, 0.05))


def _scope_metric_arrays(rows: Mapping[int, Mapping[str, Any]], scope: str, replicates: int, seed: int) -> dict[str, np.ndarray]:
    ordered = [rows[index] for index in sorted(rows)]
    arrays = _row_arrays(ordered)
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(ordered), size=(int(replicates), len(ordered)), endpoint=False)
    result: dict[str, np.ndarray] = {}
    if scope == "pose_macro":
        for name in DIAGNOSTIC_METRICS:
            if name in arrays:
                result[name] = arrays[name][sampled].mean(axis=1)
        result["avg_tp_count"] = arrays["tp"][sampled].mean(axis=1)
        result["avg_fp_count"] = arrays["fp"][sampled].mean(axis=1)
        result["avg_fn_count"] = arrays["fn"][sampled].mean(axis=1)
        result["avg_tn_count"] = arrays["tn"][sampled].mean(axis=1)
        result["avg_pred_count"] = arrays["avg_pred_count"][sampled].mean(axis=1)
        result["avg_candidate_count"] = arrays["candidate_count"][sampled].mean(axis=1)
        result["avg_gt_count"] = arrays["gt_count"][sampled].mean(axis=1)
        result["pred_over_candidate"] = result["avg_pred_count"] / np.maximum(1e-12, result["avg_candidate_count"])
        result["pred_over_gt"] = result["avg_pred_count"] / np.maximum(1e-12, result["avg_gt_count"])
        return result
    if scope != "aggregate":
        raise ValueError(f"unknown bootstrap metric scope: {scope}")
    sums = {name: arrays[name][sampled].sum(axis=1) for name in ("tp", "fp", "fn", "tn", "weighted_tp", "weighted_gt", "avg_pred_count", "candidate_count", "gt_count")}
    tp, fp, fn, tn = (sums[name] for name in COUNT_FIELDS)
    recall = tp / np.maximum(1.0, tp + fn)
    specificity = tn / np.maximum(1.0, tn + fp)
    precision = tp / np.maximum(1.0, tp + fp)
    result.update({
        "precision": precision,
        "recall": recall,
        "weighted_recall": sums["weighted_tp"] / np.maximum(1e-12, sums["weighted_gt"]),
        "f1": 2.0 * precision * recall / np.maximum(1e-8, precision + recall),
        "jaccard": tp / np.maximum(1.0, tp + fp + fn),
        "accuracy": (tp + tn) / np.maximum(1.0, sums["candidate_count"]),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "specificity": specificity,
        "useful_cull": tn / np.maximum(1.0, sums["candidate_count"]),
        "bad_cull": fn / np.maximum(1.0, sums["candidate_count"]),
        "avg_tp_count": sums["tp"] / len(ordered),
        "avg_fp_count": sums["fp"] / len(ordered),
        "avg_fn_count": sums["fn"] / len(ordered),
        "avg_tn_count": sums["tn"] / len(ordered),
        "avg_pred_count": sums["avg_pred_count"] / len(ordered),
        "avg_candidate_count": sums["candidate_count"] / len(ordered),
        "avg_gt_count": sums["gt_count"] / len(ordered),
    })
    result["pred_over_candidate"] = sums["avg_pred_count"] / np.maximum(1e-12, sums["candidate_count"])
    result["pred_over_gt"] = sums["avg_pred_count"] / np.maximum(1e-12, sums["gt_count"])
    return result


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def bootstrap_effects_by_scope(
    members: Mapping[tuple[str, int], Mapping[str, Any]],
    effect_terms: Mapping[str, float],
    metric_names: Iterable[str],
    replicates: int = 10000,
    seed: int = 20260803,
) -> dict[str, Any]:
    """Return both metric scopes with hierarchical paired bootstrap CIs."""
    if replicates < 10000:
        raise ValueError("M4-v2 formal bootstrap requires at least 10000 replicates")
    output: dict[str, Any] = {}
    metric_names = tuple(metric_names)
    for scope in ("pose_macro", "aggregate"):
        per_metric: dict[str, Any] = {}
        # Shape: [original seed, outer seed-draw position, replicate].
        # The second axis gives independent inner pose resamples when the
        # same seed is selected more than once by the outer cluster bootstrap.
        per_variant_sample: dict[str, dict[str, np.ndarray]] = {}
        observed: dict[str, dict[str, float]] = {}
        for variant in effect_terms:
            per_variant_sample[variant] = {metric: np.empty((len(SEEDS), len(SEEDS), replicates), dtype=np.float64) for metric in metric_names}
            observed[variant] = {}
            for seed_index, model_seed in enumerate(SEEDS):
                rows = members[(variant, model_seed)]["rows"]
                # The pose resample stream is shared by every variant in this
                # seed.  Including ``variant`` here would turn a paired
                # bootstrap into independent resampling and inflate the CI.
                sampled = _scope_metric_arrays(rows, scope, replicates * len(SEEDS), stable_seed(seed, scope, model_seed))
                full = summarize_rows(rows.values(), lcb_replicates=0)["poseMacro" if scope == "pose_macro" else "aggregate"]
                for metric in metric_names:
                    per_variant_sample[variant][metric][seed_index] = sampled[metric].reshape(len(SEEDS), replicates)
                    observed[variant][metric] = float(full[metric])
        cluster_rng = np.random.default_rng(stable_seed(seed, scope, "seed_cluster"))
        selected = cluster_rng.integers(0, len(SEEDS), size=(replicates, len(SEEDS)), endpoint=False)
        for metric in metric_names:
            boot_values: dict[str, np.ndarray] = {}
            for variant in effect_terms:
                values = per_variant_sample[variant][metric]
                # ``selected[r, s]`` chooses the original seed for outer
                # draw position s.  The position axis and replicate axis pick
                # an independently generated inner pose resample.
                boot_values[variant] = values[
                    selected,
                    np.arange(len(SEEDS))[None, :],
                    np.arange(replicates)[:, None],
                ].mean(axis=1)
            samples = sum(float(coefficient) * boot_values[variant] for variant, coefficient in effect_terms.items())
            mean_delta = sum(float(coefficient) * observed[variant][metric] for variant, coefficient in effect_terms.items())
            ci = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
            per_metric[metric] = {
                "mean_delta": float(mean_delta),
                "ci95": ci,
                "direction": "positive" if mean_delta > 0.0 else "negative" if mean_delta < 0.0 else "zero",
                "crosses_zero": bool(ci[0] <= 0.0 <= ci[1]),
                "scope": scope,
                "bootstrap_replicates": int(replicates),
                "cluster_unit": "seed, then validation pose within seed",
            }
        output[scope] = per_metric
    return output


def bootstrap_all_effects_by_scope(
    members: Mapping[tuple[str, int], Mapping[str, Any]],
    effects: Mapping[str, Mapping[str, float]],
    metric_names: Iterable[str],
    replicates: int = 10000,
    seed: int = 20260803,
) -> dict[str, Any]:
    """Bootstrap all registered factor effects while sampling each member once."""
    if replicates < 10000:
        raise ValueError("M4-v2 formal bootstrap requires at least 10000 replicates")
    metric_names = tuple(metric_names)
    variants = sorted({variant for terms in effects.values() for variant in terms})
    output: dict[str, Any] = {name: {} for name in effects}
    for scope in ("pose_macro", "aggregate"):
        sampled_values: dict[str, dict[str, np.ndarray]] = {}
        observed: dict[str, dict[str, float]] = {}
        for variant in variants:
            sampled_values[variant] = {metric: np.empty((len(SEEDS), len(SEEDS), replicates), dtype=np.float64) for metric in metric_names}
            observed[variant] = {}
            for seed_index, model_seed in enumerate(SEEDS):
                rows = members[(variant, model_seed)]["rows"]
                sampled = _scope_metric_arrays(rows, scope, replicates * len(SEEDS), stable_seed(seed, scope, model_seed))
                full = summarize_rows(rows.values(), lcb_replicates=0)["poseMacro" if scope == "pose_macro" else "aggregate"]
                for metric in metric_names:
                    sampled_values[variant][metric][seed_index] = sampled[metric].reshape(len(SEEDS), replicates)
                    observed[variant][metric] = float(full[metric])
        cluster_rng = np.random.default_rng(stable_seed(seed, scope, "seed_cluster"))
        selected = cluster_rng.integers(0, len(SEEDS), size=(replicates, len(SEEDS)), endpoint=False)
        for effect_name, terms in effects.items():
            for metric in metric_names:
                samples = np.zeros((replicates,), dtype=np.float64)
                mean_delta = 0.0
                for variant, coefficient in terms.items():
                    values = sampled_values[variant][metric]
                    samples += float(coefficient) * values[
                        selected,
                        np.arange(len(SEEDS))[None, :],
                        np.arange(replicates)[:, None],
                    ].mean(axis=1)
                    mean_delta += float(coefficient) * observed[variant][metric]
                ci = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
                output[effect_name].setdefault("comparisons", {}).setdefault(scope, {})[metric] = {
                    "mean_delta": float(mean_delta),
                    "ci95": ci,
                    "direction": "positive" if mean_delta > 0.0 else "negative" if mean_delta < 0.0 else "zero",
                    "crosses_zero": bool(ci[0] <= 0.0 <= ci[1]),
                    "scope": scope,
                    "bootstrap_replicates": int(replicates),
                    "cluster_unit": "seed, then validation pose within seed",
                }
    for effect_name, terms in effects.items():
        output[effect_name]["formula"] = dict(terms)
    return output


def safety_effect_is_acceptable(effect: Mapping[str, Any], max_drop: float) -> bool:
    ci = effect.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        return False
    return float(ci[0]) >= -float(max_drop)
