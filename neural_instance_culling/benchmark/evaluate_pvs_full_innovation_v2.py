#!/usr/bin/env python3
"""Evaluate one full-innovation v2 matrix on the validation split.

The training runner deliberately keeps training orchestration separate from
formal evaluation.  This entry point replays every formal member at the
member's own calibration-frozen threshold and evaluates both output
granularities:

* instance visibility, using the stored back-camera candidate CSR; and
* the independent GLB download head, using max-logit aggregation per GLB.

It never reads the test split, never adds visible IDs to candidates, and does
not select a threshold from validation.  The output is intentionally shaped
like the existing validation replay so the established summary and paired
bootstrap tools can consume it, while adding a separate ``downloadHead``
section to each pose and matrix result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "neural_instance_culling" / "benchmark"
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
for path in (BENCHMARK_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402
from common.runtime_meta import load_runtime_meta  # noqa: E402
from current_pvs_utils import evaluate_thresholds  # noqa: E402
from evaluate_pvs_hierarchical_relation_survival_integrated import (  # noqa: E402
    _build_model,
    _load_geometry,
    _load_glb_bytes,
    _load_checkpoint,
    _resource_metrics,
    _runtime_features,
    _select_threshold,
    _summarize_pose_rows,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


EVALUATION_SCHEMA = "pvs-full-innovation-v2-validation-evaluation-v1"
VISIBILITY_SCHEMA = "pvs-hierarchical-relation-survival-integrated-evaluation-v1"
FORMAL_VARIANTS = (
    "full",
    "without_hierarchical_relation",
    "shuffled_relation_source",
    "without_viewcell_integration",
    "without_threshold_aligned_utility",
)
DEFAULT_DATASET = ROOT / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1"
DEFAULT_RUNTIME_META = ROOT / "hkust-v3/assets/runtimeVisibilityMeta.json"
DEFAULT_GLB_INDEX = ROOT / "hkust-v3/assets/glbIndex.json"
DEFAULT_GLB_ROOT = ROOT / "hkust-v3/assets"
BYTE_BUDGET_FRACTIONS = (0.05, 0.10, 0.20, 0.40)
FIXED_BYTE_BUDGETS = (10 * 1024**2, 25 * 1024**2, 50 * 1024**2)
DOWNLOAD_KS = (20, 50, 100)
DOWNLOAD_METRIC_NAMES = (
    "downloadNdcgAt20",
    "downloadNdcgAt50",
    "downloadNdcgAt100",
    "utilityNdcgAt20",
    "utilityNdcgAt50",
    "utilityNdcgAt100",
    "visibilityBaselineNdcgAt20",
    "visibilityBaselineNdcgAt50",
    "visibilityBaselineNdcgAt100",
    "downloadUtilityRecallFraction5pct",
    "downloadUtilityRecallFraction10pct",
    "downloadUtilityRecallFraction20pct",
    "downloadUtilityRecallFraction40pct",
    "downloadUtilityRecallFixed10MiB",
    "downloadUtilityRecallFixed25MiB",
    "downloadUtilityRecallFixed50MiB",
    "downloadRequiredGlbRecallAt100",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def _discounted_gain(relevance: np.ndarray, k: int) -> float:
    values = np.asarray(relevance, dtype=np.float64).reshape(-1)[: int(k)]
    if values.size == 0:
        return 0.0
    positions = np.arange(2, values.size + 2, dtype=np.float64)
    return float(np.sum((2.0 ** np.clip(values, 0.0, 1.0) - 1.0) / np.log2(positions)))


def _ndcg(ranked_ids: np.ndarray, relevance_by_glb: Mapping[int, float], k: int) -> float:
    ranked_relevance = np.asarray(
        [float(relevance_by_glb.get(int(gid), 0.0)) for gid in ranked_ids[: int(k)]],
        dtype=np.float64,
    )
    ideal = np.asarray(sorted(relevance_by_glb.values(), reverse=True), dtype=np.float64)
    denominator = _discounted_gain(ideal, int(k))
    return _safe_div(_discounted_gain(ranked_relevance, int(k)), denominator, 1.0)


def _ranked_budget_ids(
    ranked_ids: np.ndarray,
    glb_bytes: np.ndarray,
    budget: float,
) -> np.ndarray:
    selected: list[int] = []
    used = 0.0
    for gid in ranked_ids.tolist():
        cost = float(glb_bytes[int(gid)])
        if cost <= 0.0:
            continue
        if used + cost > float(budget):
            continue
        selected.append(int(gid))
        used += cost
    return np.asarray(selected, dtype=np.int64)


def _download_head_metrics(
    candidate_ids: np.ndarray,
    download_logits: np.ndarray,
    utility_logits: np.ndarray,
    visibility_logits: np.ndarray,
    visible_ids: np.ndarray,
    visible_weights: np.ndarray,
    instance_to_glb: np.ndarray,
    glb_bytes: np.ndarray,
) -> dict[str, Any]:
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    glb_for_instance = instance_to_glb[candidate_ids] if candidate_ids.size else np.zeros(0, dtype=np.int64)
    candidate_glbs = np.unique(glb_for_instance)
    download_values: dict[int, float] = {}
    utility_values: dict[int, float] = {}
    visibility_values: dict[int, float] = {}
    for gid in candidate_glbs.tolist():
        local = glb_for_instance == int(gid)
        download_values[int(gid)] = float(np.max(download_logits[local]))
        utility_values[int(gid)] = float(np.max(utility_logits[local]))
    visible_ids = np.asarray(visible_ids, dtype=np.int64).reshape(-1)
    visible_weights = np.asarray(visible_weights, dtype=np.float64).reshape(-1)
    visible_glbs = instance_to_glb[visible_ids] if visible_ids.size else np.zeros(0, dtype=np.int64)
    relevance_raw: dict[int, float] = {}
    for gid, weight in zip(visible_glbs.tolist(), visible_weights.tolist(), strict=True):
        relevance_raw[int(gid)] = relevance_raw.get(int(gid), 0.0) + math.log1p(max(0.0, float(weight)))
    max_relevance = max(relevance_raw.values(), default=1.0)
    relevance = {gid: min(1.0, value / max(1e-12, max_relevance)) for gid, value in relevance_raw.items()}
    ranked_download = np.asarray(
        sorted(candidate_glbs.tolist(), key=lambda gid: (-download_values[int(gid)], int(gid))),
        dtype=np.int64,
    )
    ranked_utility = np.asarray(
        sorted(candidate_glbs.tolist(), key=lambda gid: (-utility_values[int(gid)], int(gid))),
        dtype=np.int64,
    )
    ranked_visibility = np.asarray(
        sorted(
            candidate_glbs.tolist(),
            key=lambda gid: (-float(np.max(1.0 / (1.0 + np.exp(-visibility_logits[glb_for_instance == int(gid)])))), int(gid)),
        ),
        dtype=np.int64,
    )
    candidate_bytes = float(glb_bytes[candidate_glbs].sum()) if candidate_glbs.size else 0.0
    required = set(relevance)

    def budget_row(ranked: np.ndarray, budget: float) -> dict[str, float]:
        selected = _ranked_budget_ids(ranked, glb_bytes, budget)
        selected_set = set(int(value) for value in selected.tolist())
        utility = sum(relevance.get(gid, 0.0) for gid in selected_set)
        total = sum(relevance.values())
        return {
            "budgetBytes": float(budget),
            "selectedGlbCount": float(selected.size),
            "selectedBytes": float(glb_bytes[selected].sum()) if selected.size else 0.0,
            "utilityRecall": _safe_div(utility, total, 1.0),
            "requiredGlbRecall": _safe_div(float(len(selected_set & required)), float(len(required)), 1.0),
        }

    ndcg = {f"ndcgAt{k}": _ndcg(ranked_download, relevance, k) for k in DOWNLOAD_KS}
    fraction_budgets = {
        f"fraction_{int(fraction * 100)}pct": budget_row(ranked_download, candidate_bytes * fraction)
        for fraction in BYTE_BUDGET_FRACTIONS
    }
    fixed_budgets = {
        f"fixed_{int(budget / 1024**2)}MiB": budget_row(ranked_download, budget)
        for budget in FIXED_BYTE_BUDGETS
    }
    visibility_ndcg = {f"ndcgAt{k}": _ndcg(ranked_visibility, relevance, k) for k in DOWNLOAD_KS}
    utility_ndcg = {f"ndcgAt{k}": _ndcg(ranked_utility, relevance, k) for k in DOWNLOAD_KS}
    return {
        "candidateGlbCount": float(candidate_glbs.size),
        "candidateGlbBytes": candidate_bytes,
        "requiredGlbCount": float(len(required)),
        "requiredGlbBytes": float(glb_bytes[np.asarray(sorted(required), dtype=np.int64)].sum()) if required else 0.0,
        "downloadHead": {**ndcg, "topGlbIds": ranked_download[:100].astype(int).tolist()},
        "utilityHead": utility_ndcg,
        "visibilityAggregateBaseline": visibility_ndcg,
        "downloadBudgetUtility": {**fraction_budgets, **fixed_budgets},
        "downloadHeadRequiredGlbRecallAt100": _safe_div(float(len(set(ranked_download[:100].tolist()) & required)), float(len(required)), 1.0),
    }


def _flatten_download_metrics(download: Mapping[str, Any]) -> dict[str, float]:
    """Expose the independent GLB-head measurements as per-pose scalars.

    The nested record remains available for inspection, while these stable
    fields are what the paired bootstrap consumes.  They are intentionally
    separate from instance visibility metrics.
    """
    download_head = download.get("downloadHead")
    utility_head = download.get("utilityHead")
    visibility_baseline = download.get("visibilityAggregateBaseline")
    budget = download.get("downloadBudgetUtility")
    if not isinstance(download_head, Mapping) or not isinstance(utility_head, Mapping) or not isinstance(visibility_baseline, Mapping):
        raise ValueError("download evaluation is missing one of the three ranking heads")
    if not isinstance(budget, Mapping):
        raise ValueError("download evaluation is missing budget utility metrics")

    result = {
        "candidateGlbCount": float(download["candidateGlbCount"]),
        "candidateGlbBytes": float(download["candidateGlbBytes"]),
        "requiredGlbCount": float(download["requiredGlbCount"]),
        "requiredGlbBytes": float(download["requiredGlbBytes"]),
        "downloadNdcgAt20": float(download_head["ndcgAt20"]),
        "downloadNdcgAt50": float(download_head["ndcgAt50"]),
        "downloadNdcgAt100": float(download_head["ndcgAt100"]),
        "utilityNdcgAt20": float(utility_head["ndcgAt20"]),
        "utilityNdcgAt50": float(utility_head["ndcgAt50"]),
        "utilityNdcgAt100": float(utility_head["ndcgAt100"]),
        "visibilityBaselineNdcgAt20": float(visibility_baseline["ndcgAt20"]),
        "visibilityBaselineNdcgAt50": float(visibility_baseline["ndcgAt50"]),
        "visibilityBaselineNdcgAt100": float(visibility_baseline["ndcgAt100"]),
        "downloadRequiredGlbRecallAt100": float(download["downloadHeadRequiredGlbRecallAt100"]),
    }
    fraction_names = ("5pct", "10pct", "20pct", "40pct")
    for name in fraction_names:
        result[f"downloadUtilityRecallFraction{name}"] = float(
            budget[f"fraction_{name}"]["utilityRecall"]
        )
    for mib in (10, 25, 50):
        result[f"downloadUtilityRecallFixed{mib}MiB"] = float(
            budget[f"fixed_{mib}MiB"]["utilityRecall"]
        )
    if set(result) != set(DOWNLOAD_METRIC_NAMES):
        raise AssertionError("download metric flattening schema drift")
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("download evaluation contains a non-finite metric")
    return result


@torch.no_grad()
def evaluate_member(
    args: argparse.Namespace,
    checkpoint_path: Path,
    output: Path,
    *,
    device_index: int | None = None,
) -> dict[str, Any]:
    checkpoint = _load_checkpoint(checkpoint_path)
    if bool(checkpoint.get("testRead", True)):
        raise ValueError(f"checkpoint claims test was read: {checkpoint_path}")
    world_aabbs, instance_to_glb, _runtime = load_runtime_meta(Path(args.runtime_meta).resolve())
    num_instances = int(world_aabbs.shape[0])
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda:
        if device_index is None:
            device = torch.device("cuda")
        else:
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {device_index} is unavailable")
            device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
    model = _build_model(checkpoint, world_aabbs, instance_to_glb, device)
    geometry_path = Path(args.initial_geo_features).resolve()
    geometry = _load_geometry(geometry_path, num_instances)
    runtime_features = _runtime_features(checkpoint, geometry, device)
    dataset = PoseCSRDataset(Path(args.dataset_dir).resolve(), num_instances=num_instances)
    split = dataset.split("validation")
    calibration_path = checkpoint_path.parent / "calibration_ready_summary.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"checkpoint has no calibration summary: {calibration_path}")
    calibration_summary = json.loads(calibration_path.read_text(encoding="utf-8"))
    threshold, threshold_source = _select_threshold(
        checkpoint,
        bool(args.allow_unsafe_diagnostic),
        calibration_summary,
    )
    num_glbs = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
    glb_bytes = _load_glb_bytes(Path(args.glb_index).resolve(), Path(args.glb_root).resolve(), num_glbs)
    started = time.perf_counter()

    # Reuse the trusted visibility evaluator for exact TP/FP/FN/TN semantics.
    visibility_rows = evaluate_thresholds(
        model,
        split,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=max(1, int(args.poses_per_batch)),
        max_steps=None,
        max_candidates_per_pose=0,
        seed=int(args.seed),
        thresholds=np.asarray([threshold], dtype=np.float32),
        collect_pose_stats=True,
        allow_candidate_visible_union=False,
        bootstrap_replicates=0,
        collect_score_stats=False,
        collect_per_pose=True,
    )
    if len(visibility_rows) != 1 or not isinstance(visibility_rows[0].get("_per_pose"), list):
        raise ValueError("visibility evaluator did not return per-pose rows")
    raw_by_pose = {int(row["poseIndex"]): row for row in visibility_rows[0]["_per_pose"]}
    per_pose: list[dict[str, Any]] = []
    download_sums: dict[str, list[float]] = {}
    download_nested_sums: dict[str, list[float]] = {}
    rng = np.random.default_rng(int(args.seed) + 9137)
    for pose_index in split.pose_indices.tolist():
        pose_index = int(pose_index)
        raw = raw_by_pose.get(pose_index)
        if raw is None:
            raise ValueError(f"visibility evaluator omitted validation pose {pose_index}")
        batch = split.build_pose_set_batch(
            np.asarray([pose_index], dtype=np.int64),
            world_aabbs,
            rng,
            max_candidates_per_pose=0,
            allow_candidate_visible_union=False,
            include_empty=True,
        )
        candidate_ids = np.asarray(raw.get("candidateIds", []), dtype=np.int64)
        batch_ids = np.asarray(batch["instance"], dtype=np.int64).reshape(-1)
        if not np.array_equal(batch_ids, candidate_ids):
            raise ValueError(f"candidate order changed while evaluating pose {pose_index}")
        if candidate_ids.size:
            ids = torch.from_numpy(batch["instance"].astype(np.int64, copy=False)).to(device)
            camera = torch.from_numpy(batch["camera"]).to(device)
            camera_world = torch.from_numpy(batch["camera_world"]).to(device)
            view = torch.from_numpy(batch["camera_view"]).to(device)
            visibility_logits, aux = model.compute_logits_with_aux(
                camera, view, camera_world, ids, runtime_features=runtime_features
            )
            download_logits = aux["download_logits"].detach().float().cpu().numpy().reshape(-1)
            utility_logits = aux["utility_logits"].detach().float().cpu().numpy().reshape(-1)
            visibility_values = visibility_logits.detach().float().cpu().numpy().reshape(-1)
        else:
            download_logits = np.zeros(0, dtype=np.float32)
            utility_logits = np.zeros(0, dtype=np.float32)
            visibility_values = np.zeros(0, dtype=np.float32)
        visible_ids, visible_weights = dataset.visible_slice(pose_index)
        download = _download_head_metrics(
            candidate_ids,
            download_logits,
            utility_logits,
            visibility_values,
            visible_ids,
            visible_weights,
            instance_to_glb,
            glb_bytes,
        )
        for key, value in download.items():
            if isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, Mapping):
                        for budget_key, budget_value in nested_value.items():
                            if isinstance(budget_value, Mapping):
                                for metric_key, metric_value in budget_value.items():
                                    if isinstance(metric_value, (int, float)) and np.isfinite(float(metric_value)):
                                        download_nested_sums.setdefault(
                                            f"{key}.{nested_key}.{budget_key}.{metric_key}", []
                                        ).append(float(metric_value))
                            elif isinstance(budget_value, (int, float)) and np.isfinite(float(budget_value)):
                                download_nested_sums.setdefault(f"{key}.{nested_key}.{budget_key}", []).append(float(budget_value))
                    elif isinstance(nested_value, (int, float)) and np.isfinite(float(nested_value)):
                        download_nested_sums.setdefault(f"{key}.{nested_key}", []).append(float(nested_value))
                continue
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                download_sums.setdefault(key, []).append(float(value))
        predicted_ids = np.asarray(raw.get("predictedIds", []), dtype=np.uint32)
        candidate_mask = np.isin(candidate_ids, visible_ids, assume_unique=False)
        predicted_mask = np.isin(candidate_ids, predicted_ids, assume_unique=False)
        tp = float(np.logical_and(candidate_mask, predicted_mask).sum())
        fp = float(np.logical_and(~candidate_mask, predicted_mask).sum())
        fn = float(np.logical_and(candidate_mask, ~predicted_mask).sum())
        tn = float(np.logical_and(~candidate_mask, ~predicted_mask).sum())
        weighted_tp = float(np.sum(visible_weights[np.isin(visible_ids, predicted_ids, assume_unique=False)]))
        weighted_gt = float(np.sum(visible_weights))
        recall = _safe_div(tp, tp + fn, 1.0)
        precision = _safe_div(tp, tp + fp, 1.0)
        specificity = _safe_div(tn, tn + fp, 1.0)
        raw_metrics = dict(raw.get("metrics", {}))
        resources = _resource_metrics(
            candidate_ids.astype(np.uint32, copy=False),
            predicted_ids,
            {"targetIds": visible_ids, "targetWeights": visible_weights},
            instance_to_glb,
            glb_bytes,
        )
        download_metrics = _flatten_download_metrics(download)
        raw_metrics.update(resources)
        raw_metrics.update(download_metrics)
        for key in ("missPixelRate", "wrongIdPixelRate", "extraPixelRate"):
            raw_metrics.pop(key, None)
        raw_metrics.update({
            "candidateCount": float(candidate_ids.size),
            "gtCount": float(visible_ids.size),
            "predCount": float(predicted_ids.size),
            "avgCandidateCount": float(candidate_ids.size),
            "avgGtCount": float(visible_ids.size),
            "avgPredCount": float(predicted_ids.size),
            "weightedTp": weighted_tp,
            "weightedGt": weighted_gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "weightedRecall": _safe_div(weighted_tp, weighted_gt, 1.0),
            "f1": _safe_div(2.0 * precision * recall, precision + recall),
            "jaccard": _safe_div(tp, tp + fp + fn),
            "accuracy": _safe_div(tp + tn, candidate_ids.size, 1.0),
            "balancedAccuracy": 0.5 * (recall + specificity),
            "specificity": specificity,
            "usefulCull": _safe_div(tn, candidate_ids.size),
            "badCull": _safe_div(fn, candidate_ids.size),
            "predOverCandidate": _safe_div(predicted_ids.size, candidate_ids.size),
            "predOverGt": _safe_div(predicted_ids.size, visible_ids.size),
        })
        per_pose.append({
            "poseIndex": pose_index,
            "candidateDigest": hashlib.sha256(np.asarray(candidate_ids, dtype="<u4").tobytes()).hexdigest(),
            "candidateCount": int(candidate_ids.size),
            "candidateIds": candidate_ids.astype(int).tolist(),
            "predictedIds": predicted_ids.astype(int).tolist(),
            "metrics": raw_metrics,
            "downloadHead": download,
        })
    visibility_summary = _summarize_pose_rows(per_pose, lcb_replicates=0, seed=int(args.seed))
    # The shared visibility summarizer predates the independent GLB head. Add
    # the resource and ranking scalars to both scopes so the top-level replay
    # and the per-pose bootstrap input expose the same measurements.
    for metric in (
        "candidateGlbBytes",
        "requiredGlbCount",
        "requiredGlbBytes",
        *DOWNLOAD_METRIC_NAMES,
    ):
        values = [float(row["metrics"][metric]) for row in per_pose]
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"non-finite formal metric: {metric}")
        mean_value = float(np.mean(values))
        visibility_summary["poseMacro"][metric] = mean_value
        visibility_summary["aggregate"][metric] = mean_value
    download_macro = {
        key: float(np.mean(values)) for key, values in download_sums.items() if values
    }
    pose_candidate_digests = {
        str(int(row["poseIndex"])): str(row["candidateDigest"]) for row in per_pose
    }
    result = {
        # Keep the established visibility evaluation schema so the existing
        # validator and paired-bootstrap summarizer can consume this replay.
        # The additional section below carries the independent download-head
        # evaluation without changing instance-level metric semantics.
        "schema": VISIBILITY_SCHEMA,
        "fullInnovationEvaluationSchema": EVALUATION_SCHEMA,
        "visibilitySchema": VISIBILITY_SCHEMA,
        "version": 1,
        "split": "validation",
        "testRead": False,
        "variant": str(checkpoint.get("variant", checkpoint.get("args", {}).get("variant", ""))),
        "variantSpec": checkpoint.get("variantSpec", {}),
        "seed": int(checkpoint.get("args", {}).get("seed", args.seed)),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpointSha256": _sha256(checkpoint_path),
        "calibrationSummary": str(calibration_path.resolve()),
        "calibrationSummarySha256": _sha256(calibration_path),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("args", {}).get("epochs", 80))),
        "trainingEpochs": int(checkpoint.get("args", {}).get("epochs", 80)),
        "poseIndices": [int(value) for value in split.pose_indices.tolist()],
        "poseCount": int(split.pose_indices.size),
        "candidateDigest": candidate_digest_for_pose_sequence(dataset, split.pose_indices),
        "poseCandidateDigests": pose_candidate_digests,
        "threshold": float(threshold),
        "thresholdSource": threshold_source,
        "metrics": visibility_summary["aggregate"],
        "aggregate": visibility_summary["aggregate"],
        "poseMacro": visibility_summary["poseMacro"],
        "perPose": per_pose,
        "downloadHead": {
            "poseMacro": {**download_macro, **{
                key: float(np.mean(values)) for key, values in download_nested_sums.items() if values
            }},
            "aggregation": "max download logit per candidate GLB",
            "target": "normalized sum of log1p visible weights per required GLB",
        },
        "runtime": {
            "device": str(device),
            "elapsedSeconds": float(time.perf_counter() - started),
            "runtimeFeatureBytes": int(runtime_features.numel() * 2),
            "runtimeFeatureDim": int(runtime_features.shape[1]),
            "testRead": False,
        },
        "unavailableMetrics": {
            "missPixelRate": "not_available: hardware Color-ID replay is not attached",
            "wrongIdPixelRate": "not_available: hardware Color-ID replay is not attached",
            "extraPixelRate": "not_available: hardware Color-ID replay is not attached",
            "browserWebGpuLatency": "not_available: browser replay is not attached",
        },
        "protocol": checkpoint.get("protocol", {}),
        "inputProvenance": {
            "datasetMetaSha256": _sha256(Path(args.dataset_dir).resolve() / "dataset_meta.json"),
            "runtimeMetaSha256": _sha256(Path(args.runtime_meta).resolve()),
            "geometrySha256": _sha256(geometry_path),
            "glbIndexSha256": _sha256(Path(args.glb_index).resolve()),
        },
    }
    _write_json(output, result)
    return {"variant": result["variant"], "seed": result["seed"], "output": str(output), "testRead": False}


def _member_jobs(args: argparse.Namespace, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs")
    if jobs is None:
        jobs = manifest.get("plannedMembers")
    if not isinstance(jobs, list) or len(jobs) != 15:
        raise ValueError("formal80 manifest must contain exactly 15 jobs")
    if int(manifest.get("epochs", -1)) != 80:
        raise ValueError("formal80 manifest must register epoch 80")
    registered_variants = manifest.get("variants")
    if isinstance(registered_variants, list):
        if tuple(registered_variants) != FORMAL_VARIANTS:
            raise ValueError("formal80 manifest variant registry is not the current five-variant design")
        registered_variant_specs: Mapping[str, Any] = {}
    elif isinstance(registered_variants, Mapping) and tuple(registered_variants) == FORMAL_VARIANTS:
        registered_variant_specs = registered_variants
    else:
        raise ValueError("formal80 manifest variant registry is not the current five-variant design")
    expected = {(variant, seed) for variant in FORMAL_VARIANTS for seed in (20260801, 20260802, 20260803)}
    actual = {(str(job.get("variant")), int(job.get("seed", -1))) for job in jobs}
    if actual != expected:
        raise ValueError(f"formal80 member set mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    output_root = Path(args.output_root).resolve()
    result: list[dict[str, Any]] = []
    for job in jobs:
        member_dir = Path(str(job["outputDir"])) if job.get("outputDir") else Path(args.formal_root) / str(job["member"])
        if not member_dir.is_absolute():
            member_dir = (Path(args.formal_root) / member_dir).resolve()
        if member_dir.parent != Path(args.formal_root).resolve():
            raise ValueError(f"formal member escapes formal root: {member_dir}")
        history_path = member_dir / "train_history.json"
        if not history_path.is_file():
            raise FileNotFoundError(f"formal member has no train history: {member_dir}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list) or not history:
            raise ValueError(f"formal member train history is empty: {history_path}")
        last_epoch = history[-1].get("epoch") if isinstance(history[-1], Mapping) else None
        if int(last_epoch or -1) != 80:
            raise ValueError(f"formal member did not complete 80 epochs: {member_dir}")
        if not (member_dir / "calibration_ready_summary.json").is_file():
            raise FileNotFoundError(f"formal member has no calibration summary: {member_dir}")
        checkpoint = member_dir / "best.pt"
        if not checkpoint.is_file():
            checkpoint = member_dir / "best_diagnostic.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"formal member has neither safe nor diagnostic checkpoint: {member_dir}")
        checkpoint_payload = _load_checkpoint(checkpoint)
        expected_variant = str(job["variant"])
        expected_seed = int(job["seed"])
        actual_variant = str(checkpoint_payload.get("variant", checkpoint_payload.get("args", {}).get("variant", "")))
        actual_seed = int(checkpoint_payload.get("args", {}).get("seed", -1))
        if actual_variant != expected_variant or actual_seed != expected_seed:
            raise ValueError(
                f"formal checkpoint identity mismatch for {member_dir}: "
                f"expected {expected_variant}/{expected_seed}, got {actual_variant}/{actual_seed}"
            )
        if "seed" in checkpoint_payload and int(checkpoint_payload["seed"]) != expected_seed:
            raise ValueError(f"formal checkpoint top-level seed mismatch: {member_dir}")
        configured_epochs = checkpoint_payload.get("args", {}).get("epochs", checkpoint_payload.get("epochs"))
        if configured_epochs is not None and int(configured_epochs) != 80:
            raise ValueError(f"formal checkpoint training epoch configuration mismatch: {member_dir}")
        expected_spec = job.get("variantSpec")
        if not isinstance(expected_spec, Mapping):
            expected_spec = registered_variant_specs.get(expected_variant, {})
        expected_loss = str(expected_spec.get("lossVariant", ""))
        actual_loss = str((checkpoint_payload.get("variantSpec") or {}).get("lossVariant", ""))
        if expected_loss and actual_loss != expected_loss:
            raise ValueError(
                f"formal loss identity mismatch for {member_dir}: expected {expected_loss}, got {actual_loss}"
            )
        actual_spec = checkpoint_payload.get("variantSpec") or {}
        for spec_key in ("relationSource", "spectralMode", "lossVariant"):
            expected_value = expected_spec.get(spec_key)
            if expected_value is not None and actual_spec.get(spec_key) != expected_value:
                raise ValueError(f"formal {spec_key} identity mismatch for {member_dir}")
        protocol = checkpoint_payload.get("protocol")
        expected_digests = manifest.get("splitCandidateDigests") or {}
        if not isinstance(protocol, Mapping):
            raise ValueError(f"formal checkpoint has no protocol provenance: {member_dir}")
        if protocol.get("schema") != "pvs-hierarchical-relation-survival-integrated-training-v1":
            raise ValueError(f"formal checkpoint protocol schema mismatch: {member_dir}")
        for split_name in ("train", "calibration", "validation"):
            actual_digest = str(protocol.get(f"{split_name}CandidateDigest", ""))
            expected_digest = str(expected_digests.get(split_name, ""))
            if not expected_digest or actual_digest != expected_digest:
                raise ValueError(f"formal checkpoint {split_name} candidate digest mismatch for {member_dir}")
            expected_count = (manifest.get("splitPoseCounts") or {}).get(split_name)
            actual_count = protocol.get(f"{split_name}PoseCount")
            if expected_count is not None and int(actual_count) != int(expected_count):
                raise ValueError(f"formal checkpoint {split_name} pose count mismatch for {member_dir}")
        if protocol.get("testRead") is not False or protocol.get("candidateUnion") is not False:
            raise ValueError(f"formal checkpoint has unsafe candidate/test protocol: {member_dir}")
        if protocol.get("thresholdSource") != "calibration_only":
            raise ValueError(f"formal checkpoint threshold source is not calibration-only: {member_dir}")
        calibration_summary = json.loads(
            (member_dir / "calibration_ready_summary.json").read_text(encoding="utf-8")
        )
        if calibration_summary.get("testRead") is not False:
            raise ValueError(f"formal calibration summary claims test was read: {member_dir}")
        result.append({
            "job": job,
            "checkpoint": checkpoint.resolve(),
            "output": output_root / f"{job['variant']}_seed{int(job['seed'])}_e80" / "validation_evaluation.json",
        })
    return result


def _run_job(args: argparse.Namespace, job: Mapping[str, Any], gpu: int) -> dict[str, Any]:
    output = Path(str(job["output"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    try:
        payload = evaluate_member(
            args,
            Path(str(job["checkpoint"])),
            output,
            device_index=int(gpu),
        )
        return {**payload, "gpu": int(gpu), "elapsedSeconds": time.time() - started, "returnCode": 0}
    except Exception as exc:  # keep the matrix record auditable and fail later
        error_path = output.with_suffix(".error.json")
        _write_json(error_path, {"error": repr(exc), "checkpoint": str(job["checkpoint"]), "testRead": False})
        return {"variant": job["job"]["variant"], "seed": int(job["job"]["seed"]), "gpu": int(gpu), "returnCode": 1, "error": repr(exc), "testRead": False}


def evaluate_matrix(args: argparse.Namespace) -> dict[str, Any]:
    formal_root = Path(args.formal_root).resolve()
    manifest_path = formal_root / "matrix_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "formal80" or manifest.get("testRead") is not False:
        raise ValueError("formal root is not a test-free formal80 matrix")
    registered_root = manifest.get("outputRoot")
    if registered_root and Path(str(registered_root)).resolve() != formal_root:
        raise ValueError("formal root does not match the manifest outputRoot")
    input_provenance = manifest.get("evaluationInputs") or manifest.get("inputProvenance")
    if input_provenance is None:
        # The v2 training runner registers these four blocks directly.  Keep
        # the normalized consumer contract local to this evaluator.
        input_provenance = {
            "dataset": manifest.get("dataset"),
            "runtimeMeta": manifest.get("runtimeMeta"),
            "geometry": manifest.get("geometry"),
            "glb": manifest.get("glb"),
        }
    if not isinstance(input_provenance, Mapping):
        raise ValueError("formal manifest has no evaluation input provenance")
    input_checks = (
        ("dataset", Path(args.dataset_dir).resolve() / "dataset_meta.json", ("datasetMetaSha256", "sha256")),
        ("runtimeMeta", Path(args.runtime_meta).resolve(), ("runtimeMetaSha256", "sha256")),
        ("geometry", Path(args.initial_geo_features).resolve(), ("geometrySha256", "sha256")),
        ("glb", Path(args.glb_index).resolve(), ("glbIndexSha256", "indexSha256")),
    )
    for manifest_key, path, digest_keys in input_checks:
        registered = input_provenance.get(manifest_key)
        if isinstance(registered, Mapping):
            expected_digest = next((registered.get(key) for key in digest_keys if registered.get(key)), None)
        else:
            expected_digest = next((input_provenance.get(key) for key in digest_keys if key in input_provenance), None)
        if not expected_digest or _sha256(path) != str(expected_digest):
            raise ValueError(f"formal input digest mismatch or missing provenance for {manifest_key}: {path}")
    jobs = _member_jobs(args, manifest)
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty evaluation output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    pending = iter(jobs)
    with ThreadPoolExecutor(max_workers=len(args.gpu_ids)) as executor:
        active: dict[Future[dict[str, Any]], int] = {}
        for gpu in args.gpu_ids:
            try:
                job = next(pending)
            except StopIteration:
                break
            active[executor.submit(_run_job, args, job, int(gpu))] = int(gpu)
        while active:
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                gpu = active.pop(future)
                result = future.result()
                results.append(result)
                try:
                    job = next(pending)
                except StopIteration:
                    continue
                active[executor.submit(_run_job, args, job, gpu)] = gpu
    results.sort(key=lambda row: (str(row.get("variant", "")), int(row.get("seed", -1))))
    if any(int(row.get("returnCode", 1)) != 0 for row in results):
        raise RuntimeError("one or more formal validation members failed")
    payload = {
        "schema": "pvs-full-innovation-v2-validation-matrix-v1",
        "formalRoot": str(formal_root),
        "formalManifestSha256": _sha256(manifest_path),
        "split": "validation",
        "mode": "formal80_validation_replay",
        "variants": {
            variant: {"registered": True}
            for variant in FORMAL_VARIANTS
        },
        "seeds": [20260801, 20260802, 20260803],
        "epochs": 80,
        "splitPoseCounts": manifest.get("splitPoseCounts", {}),
        "splitCandidateDigests": manifest.get("splitCandidateDigests", {}),
        "inputProvenance": {
            "datasetMetaSha256": _sha256(Path(args.dataset_dir).resolve() / "dataset_meta.json"),
            "runtimeMetaSha256": _sha256(Path(args.runtime_meta).resolve()),
            "geometrySha256": _sha256(Path(args.initial_geo_features).resolve()),
            "glbIndexSha256": _sha256(Path(args.glb_index).resolve()),
        },
        "candidateSemantics": "stored native back-camera candidate CSR; no GT union, cap, or frontend whitelist",
        "members": results,
        "memberCount": len(results),
        "gpuIds": [int(value) for value in args.gpu_ids],
        "testRead": False,
    }
    _write_json(output_root / "validation_matrix_summary.json", payload)
    _write_json(
        output_root / "matrix_manifest.json",
        {
            "schema": "pvs-full-innovation-v2-validation-matrix-manifest-v1",
            "mode": "formal80_validation_replay",
            "variants": payload["variants"],
            "seeds": payload["seeds"],
            "epochs": payload["epochs"],
            "splitPoseCounts": payload["splitPoseCounts"],
            "splitCandidateDigests": payload["splitCandidateDigests"],
            "inputProvenance": payload["inputProvenance"],
            "candidateSemantics": payload["candidateSemantics"],
            "testRead": False,
        },
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=False)
    parser.add_argument("--output-root", type=Path, required=False)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--runtime-meta", type=Path, default=DEFAULT_RUNTIME_META)
    parser.add_argument("--initial-geo-features", type=Path, required=False)
    parser.add_argument("--glb-index", type=Path, default=DEFAULT_GLB_INDEX)
    parser.add_argument("--glb-root", type=Path, default=DEFAULT_GLB_ROOT)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--poses-per-batch", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-unsafe-diagnostic", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def self_test() -> dict[str, Any]:
    relevance = {1: 1.0, 2: 0.5}
    ranked = np.asarray([1, 2, 3], dtype=np.int64)
    if not math.isclose(_ndcg(ranked, relevance, 2), 1.0):
        raise AssertionError("NDCG self-test failed")
    bytes_ = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    ranked_with_valid_ids = np.asarray([0, 1, 2], dtype=np.int64)
    if _ranked_budget_ids(ranked_with_valid_ids, bytes_, 100.0).tolist() != [0]:
        raise AssertionError("budget self-test failed")
    return {"status": "passed", "schema": EVALUATION_SCHEMA, "testRead": False}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if args.formal_root is None or args.output_root is None or args.initial_geo_features is None:
        raise ValueError("--formal-root, --output-root, and --initial-geo-features are required")
    if not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("gpu-ids must be non-empty and unique")
    if args.device != "cpu":
        if len(args.gpu_ids) != 4:
            raise ValueError("formal validation requires all four registered GPU IDs")
        if not torch.cuda.is_available() or max(args.gpu_ids) >= torch.cuda.device_count():
            raise ValueError("formal validation requires four available CUDA devices")
    result = evaluate_matrix(args)
    print(json.dumps({"memberCount": result["memberCount"], "outputRoot": str(args.output_root.resolve()), "testRead": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
