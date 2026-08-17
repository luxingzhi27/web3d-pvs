from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common.threshold_selection import (
    select_weighted_precision_workpoint,
    target_weighted_recall_from_payload,
)


_FOV_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "config" / "neuralpvs_viewcell_protocol.json"
_FOV_PROTOCOL = json.loads(_FOV_PROTOCOL_PATH.read_text(encoding="utf-8"))
RENDER_FOV_Y_DEG = float(_FOV_PROTOCOL["renderFovYDeg"])
MODEL_INPUT_FOV_Y_DEG = float(_FOV_PROTOCOL["modelFovYDeg"])
if RENDER_FOV_Y_DEG != 60.0 or MODEL_INPUT_FOV_Y_DEG != 66.0:
    raise ValueError(
        "The current PVS FOV protocol must define frontend rendering at 60 degrees "
        "and model input at 66 degrees."
    )


def fourier_features(values: torch.Tensor, bands: int) -> torch.Tensor:
    features = [values]
    for band in range(max(0, int(bands))):
        freq = float(2 ** band) * torch.pi
        features.append(torch.sin(values * freq))
        features.append(torch.cos(values * freq))
    return torch.cat(features, dim=-1)


def fourier_dim(input_dim: int, bands: int) -> int:
    return int(input_dim) * (1 + 2 * max(0, int(bands)))


def mlp(dims: list[int], last_relu: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_relu:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class InstancePointNetPPGeoEncoder(nn.Module):
    """Offline instance point-cloud encoder used by the current PVS model."""

    def __init__(
        self,
        point_feature_dim: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 96,
        center_count: int = 16,
        neighbor_count: int = 8,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.center_count = int(center_count)
        self.neighbor_count = int(neighbor_count)
        self.local_mlp = mlp([point_feature_dim + 3, hidden_dim, hidden_dim], last_relu=True)
        self.local_pool = mlp([hidden_dim * 2, hidden_dim, hidden_dim], last_relu=True)
        self.global_mlp = mlp([hidden_dim * 2, hidden_dim, out_dim], last_relu=True)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        b, p, c = point_features.shape
        center_count = min(max(1, self.center_count), p)
        neighbor_count = min(max(1, self.neighbor_count), p)
        center_idx = torch.linspace(0, p - 1, steps=center_count, device=point_features.device).round().long()
        xyz = point_features[..., :3]
        centers = xyz[:, center_idx, :]
        d2 = (centers[:, :, None, :] - xyz[:, None, :, :]).pow(2).sum(dim=-1)
        neighbor_idx = torch.topk(d2, k=neighbor_count, dim=-1, largest=False).indices
        expanded = point_features[:, None, :, :].expand(b, center_count, p, c)
        grouped = torch.gather(expanded, 2, neighbor_idx[..., None].expand(b, center_count, neighbor_count, c))
        center_features = point_features[:, center_idx, :]
        relative_xyz = grouped[..., :3] - center_features[:, :, None, :3]
        local_input = torch.cat([grouped, relative_xyz], dim=-1)
        encoded = self.local_mlp(local_input.reshape(b * center_count * neighbor_count, c + 3))
        encoded = encoded.reshape(b, center_count, neighbor_count, -1)
        pooled = torch.cat([encoded.mean(dim=2), encoded.max(dim=2).values], dim=-1)
        local = self.local_pool(pooled.reshape(b * center_count, -1)).reshape(b, center_count, -1)
        global_pooled = torch.cat([local.mean(dim=1), local.max(dim=1).values], dim=-1)
        return self.global_mlp(global_pooled)


class VisibilityMLP(nn.Module):
    """Runtime visibility head shared by current training and export paths."""

    def __init__(
        self,
        camera_dim: int = 27,
        latent_dim: int = 32,
        geo_dim: int = 16,
        hidden_dim: int = 64,
        interaction_dim: int = 64,
    ):
        super().__init__()
        self.camera_dim = int(camera_dim)
        self.latent_dim = int(latent_dim)
        self.geo_dim = int(geo_dim)
        self.interaction_dim = int(interaction_dim)
        if self.interaction_dim > 0:
            self.camera_interaction = nn.Linear(camera_dim, self.interaction_dim)
            self.instance_interaction = nn.Linear(latent_dim + geo_dim, self.interaction_dim)
        else:
            self.camera_interaction = None
            self.instance_interaction = None
        input_dim = camera_dim + latent_dim + geo_dim + max(0, self.interaction_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def logits(self, spatial_feat: torch.Tensor, latent_code: torch.Tensor, geo_summary: torch.Tensor) -> torch.Tensor:
        features = [spatial_feat, latent_code, geo_summary]
        if self.interaction_dim > 0:
            instance_feat = torch.cat([latent_code, geo_summary], dim=-1)
            cam_i = self.camera_interaction(spatial_feat)
            inst_i = self.instance_interaction(instance_feat)
            features.append(cam_i * inst_i)
        return self.net(torch.cat(features, dim=-1))

    def forward(self, spatial_feat: torch.Tensor, latent_code: torch.Tensor, geo_summary: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(spatial_feat, latent_code, geo_summary))


def threshold_grid() -> np.ndarray:
    # Spatially held-out positives can occupy the low-score tail even when
    # calibration still has a safe workpoint.  The grid must cover that tail
    # without turning threshold selection into test-time tuning.
    low = np.geomspace(1e-8, 1e-3, 17, dtype=np.float32)
    low = np.concatenate([np.asarray([0.0], dtype=np.float32), low])
    mid = np.asarray([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2], dtype=np.float32)
    high = np.linspace(0.22, 0.9, 35, dtype=np.float32)
    return np.unique(np.concatenate([low, mid, high]))


def weighted_recall_lower_confidence_bound(
    weighted_tp: np.ndarray,
    weighted_gt: np.ndarray,
    replicates: int = 10000,
    seed: int = 0,
) -> float:
    """Return the percentile lower bound used by calibration safety gates.

    The resampling unit is a pose.  ``weighted_tp`` and ``weighted_gt`` are
    therefore per-pose weighted hit and GT mass, not already aggregated
    ratios.  A zero replicate count is reserved for lightweight diagnostics;
    formal calibration always passes at least 10,000.
    """
    tp = np.asarray(weighted_tp, dtype=np.float64).reshape(-1)
    gt = np.asarray(weighted_gt, dtype=np.float64).reshape(-1)
    if tp.size == 0 or tp.size != gt.size:
        return 0.0
    # Empty-candidate poses have no visible weight and must not affect the
    # aggregate bootstrap denominator.  Keeping them in the raw pose stream
    # is useful for pose-count diagnostics, but resampling them as successful
    # recall observations would make the lower bound depend on empty poses.
    valid = np.isfinite(tp) & np.isfinite(gt) & (gt > 1e-12)
    tp = tp[valid]
    gt = gt[valid]
    if tp.size == 0:
        return 1.0
    if int(replicates) <= 0:
        denominator = float(gt.sum())
        return 1.0 if denominator <= 1e-12 else float(tp.sum() / denominator)
    rng = np.random.default_rng(int(seed))
    # Keep memory bounded for larger calibration sets while retaining exactly
    # the requested number of bootstrap draws.
    values = np.empty((int(replicates),), dtype=np.float64)
    chunk = max(1, min(int(replicates), 1_000_000 // max(1, tp.size)))
    for start in range(0, int(replicates), chunk):
        end = min(int(replicates), start + chunk)
        indices = rng.integers(0, tp.size, size=(end - start, tp.size), endpoint=False)
        numerator = tp[indices].sum(axis=1)
        denominator = gt[indices].sum(axis=1)
        values[start:end] = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=denominator > 1e-12,
        )
    return float(np.quantile(values, 0.05))


def pose_macro_weighted_recall_lower_confidence_bound(
    pose_weighted_recall: np.ndarray,
    replicates: int = 10000,
    seed: int = 0,
) -> float:
    """Return the one-sided bootstrap bound for the pose-macro diagnostic.

    This is intentionally separate from ``weighted_recall_lower_confidence_bound``:
    the latter resamples weighted TP/GT sufficient statistics and therefore
    estimates the aggregate safety quantity.  A macro bound resamples already
    computed per-pose ratios and is diagnostic only.
    """
    values = np.asarray(pose_weighted_recall, dtype=np.float64).reshape(-1)
    if values.size == 0 or not bool(np.isfinite(values).all()):
        return 0.0
    if int(replicates) <= 0:
        return float(np.mean(values))
    rng = np.random.default_rng(int(seed))
    result = np.empty((int(replicates),), dtype=np.float64)
    chunk = max(1, min(int(replicates), 1_000_000 // max(1, values.size)))
    for start in range(0, int(replicates), chunk):
        end = min(int(replicates), start + chunk)
        indices = rng.integers(0, values.size, size=(end - start, values.size), endpoint=False)
        result[start:end] = values[indices].mean(axis=1)
    return float(np.quantile(result, 0.05))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return a finite weighted quantile for score-distribution diagnostics."""
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value_array.size == 0 or value_array.size != weight_array.size:
        return float("nan")
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    value_array = value_array[valid]
    weight_array = weight_array[valid]
    if value_array.size == 0:
        return float("nan")
    order = np.argsort(value_array, kind="mergesort")
    ordered_values = value_array[order]
    ordered_weights = weight_array[order]
    cumulative = np.cumsum(ordered_weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def score_distribution_summary(
    scores: np.ndarray,
    target: np.ndarray,
    visible_weights: np.ndarray,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Summarize score separation without selecting a threshold.

    Positive scores use their visible-weight mass; negative scores use unit
    candidate mass.  This keeps the summary aligned with the safety metric
    while still exposing whether the model only becomes safe after moving the
    threshold into the numerical tail.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    visible = np.asarray(visible_weights, dtype=np.float64).reshape(-1)
    if not (values.size == labels.size == visible.size):
        raise ValueError("score, target, and visible weight arrays must have equal length")
    if values.size == 0:
        return {
            "scoreCount": 0,
            "positiveCount": 0,
            "negativeCount": 0,
            "positiveWeightedQ05": None,
            "negativeQ95": None,
            "positiveNegativeGapQ05Q95": None,
            "brierScore": None,
            "expectedCalibrationError": None,
        }
    if not np.isfinite(values).all() or not np.isfinite(labels).all() or not np.isfinite(visible).all():
        raise FloatingPointError("score distribution contains non-finite values")
    positive = labels > 0.5
    negative = ~positive
    positive_weights = np.maximum(visible, 1e-6)
    positive_q05 = _weighted_quantile(values[positive], positive_weights[positive], 0.05)
    negative_q95 = _weighted_quantile(values[negative], np.ones(np.count_nonzero(negative)), 0.95)
    gap = positive_q05 - negative_q95 if np.isfinite(positive_q05) and np.isfinite(negative_q95) else float("nan")

    evaluation_weights = np.where(positive, positive_weights, 1.0)
    brier = float(
        np.sum(evaluation_weights * np.square(values - labels))
        / max(1e-12, float(np.sum(evaluation_weights)))
    )
    bin_count = max(1, int(calibration_bins))
    bin_index = np.minimum((np.clip(values, 0.0, 1.0) * bin_count).astype(np.int64), bin_count - 1)
    ece = 0.0
    for current_bin in range(bin_count):
        selected = bin_index == current_bin
        if not np.any(selected):
            continue
        bin_weights = evaluation_weights[selected]
        total = float(np.sum(bin_weights))
        ece += (total / max(1e-12, float(np.sum(evaluation_weights)))) * abs(
            float(np.sum(bin_weights * values[selected]) / max(1e-12, total))
            - float(np.sum(bin_weights * labels[selected]) / max(1e-12, total))
        )
    return {
        "scoreCount": int(values.size),
        "positiveCount": int(np.count_nonzero(positive)),
        "negativeCount": int(np.count_nonzero(negative)),
        "positiveWeightedQ05": None if not np.isfinite(positive_q05) else float(positive_q05),
        "negativeQ95": None if not np.isfinite(negative_q95) else float(negative_q95),
        "positiveNegativeGapQ05Q95": None if not np.isfinite(gap) else float(gap),
        "positiveMean": float(np.mean(values[positive])) if np.any(positive) else None,
        "negativeMean": float(np.mean(values[negative])) if np.any(negative) else None,
        "brierScore": brier,
        "expectedCalibrationError": float(ece),
    }


def validate_training_resources(
    dataset_dir: Path,
    runtime_meta: Path,
    glb_points: Path,
    num_instances: int,
    num_glbs: int,
    strict_semantics: bool = False,
) -> dict[str, Any]:
    required = [
        dataset_dir / "dataset_meta.json",
        dataset_dir / "poses.bin",
        dataset_dir / "visible_offsets.bin",
        dataset_dir / "visible_ids.bin",
        dataset_dir / "frustum_offsets.bin",
        dataset_dir / "frustum_ids.bin",
        runtime_meta,
        glb_points,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required resources: {missing}")
    visible_ids = np.memmap(dataset_dir / "visible_ids.bin", dtype=np.uint32, mode="r")
    frustum_ids = np.memmap(dataset_dir / "frustum_ids.bin", dtype=np.uint32, mode="r")
    if visible_ids.size and int(visible_ids.max()) >= int(num_instances):
        raise ValueError("visible_ids contains instance id outside runtimeVisibilityMeta component count")
    if frustum_ids.size and int(frustum_ids.max()) >= int(num_instances):
        raise ValueError("frustum_ids contains instance id outside runtimeVisibilityMeta component count")
    meta_path = glb_points.with_name(glb_points.stem + "_meta.json")
    glb_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    # glb_points_v3 is deliberately deduplicated by downloadable GLB. Several
    # component instances may share one row, so validation must compare the
    # cache row count with the number of unique GLBs, not the instance count.
    if int(glb_meta.get("numGlbs", num_glbs)) < int(num_glbs):
        raise ValueError(
            f"GLB point cache row count {glb_meta.get('numGlbs')} < runtime GLB count {num_glbs}"
        )
    dataset_meta = json.loads((dataset_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    strict_failures: list[str] = []
    if strict_semantics:
        point_count = int(glb_meta.get("numGlbs", -1))
        if point_count != int(num_glbs):
            strict_failures.append(
                f"point cache rows ({point_count}) do not exactly match runtime GLB count ({int(num_glbs)})"
            )
        if int(glb_meta.get("failed", 0)) != 0:
            strict_failures.append(f"point cache reports {int(glb_meta.get('failed', 0))} decode failures")
        if int(glb_meta.get("fallback", 0)) != 0:
            strict_failures.append(f"point cache reports {int(glb_meta.get('fallback', 0))} fallback rows")
        candidate_semantics = str(dataset_meta.get("candidateSemantics", "")).lower()
        has_raw_candidate = bool(
            dataset_meta.get("rawCandidateFile")
            or dataset_meta.get("rawCandidateIds")
            or (dataset_dir / "raw_candidate_ids.bin").exists()
            or ((dataset_dir / "frustum_ids.bin").exists() and (dataset_dir / "frustum_offsets.bin").exists())
        )
        if bool(dataset_meta.get("candidateVisibleUnionAllowed", False)):
            strict_failures.append("dataset metadata explicitly allows GT-visible candidate union")
        if int(dataset_meta.get("stats", {}).get("candidateVisibleUnionAdded", 0)) > 0:
            strict_failures.append("dataset metadata reports GT-visible candidates were added")
        if "union" in candidate_semantics and "no gt" not in candidate_semantics and not has_raw_candidate:
            strict_failures.append("candidate semantics describe a GT-visible union rather than raw candidates")
        if "plus" in candidate_semantics and "visible" in candidate_semantics and not has_raw_candidate:
            strict_failures.append(
                "candidate file is documented as post-union with visible positives, but raw AABB candidates are absent"
            )
    if strict_failures:
        raise ValueError(
            "Formal training resource semantics failed: " + "; ".join(strict_failures) + ". "
            "Use --allow-invalid-resource-semantics only for explicitly exploratory runs."
        )
    subset_status = "not_checked_no_candidate_offsets"
    candidate_offsets_path = dataset_dir / "candidate_offsets.bin"
    candidate_ids_path = dataset_dir / "candidate_ids.bin"
    if candidate_offsets_path.exists() and candidate_ids_path.exists():
        visible_offsets = np.memmap(dataset_dir / "visible_offsets.bin", dtype=np.uint64, mode="r")
        candidate_offsets = np.memmap(candidate_offsets_path, dtype=np.uint64, mode="r")
        candidate_ids = np.memmap(candidate_ids_path, dtype=np.uint32, mode="r")
        bad_pose = None
        for pose_index in range(min(visible_offsets.size, candidate_offsets.size) - 1):
            vs = int(visible_offsets[pose_index])
            ve = int(visible_offsets[pose_index + 1])
            cs = int(candidate_offsets[pose_index])
            ce = int(candidate_offsets[pose_index + 1])
            if ve <= vs:
                continue
            visible = np.asarray(visible_ids[vs:ve], dtype=np.uint32)
            candidate = np.asarray(candidate_ids[cs:ce], dtype=np.uint32)
            if np.setdiff1d(visible, candidate, assume_unique=False).size:
                bad_pose = int(pose_index)
                break
        if bad_pose is not None:
            raise ValueError(f"visible_ids is not subset of candidate_ids at pose {bad_pose}")
        subset_status = "visible_ids_subset_candidate_ids"
    return {
        "datasetDir": str(dataset_dir.as_posix()),
        "runtimeMeta": str(runtime_meta.as_posix()),
        "glbPoints": str(glb_points.as_posix()),
        "glbPointsMeta": glb_meta,
        "glbPointsSemantics": "Deduplicated point cache indexed by global GLB; instance features resolve through instance_to_glb.",
        "runtimeGlbCount": int(num_glbs),
        "visibleCandidateSubset": subset_status,
        "strictSemantics": bool(strict_semantics),
    }


@torch.no_grad()
def evaluate_thresholds(
    model,
    split,
    runtime_features: torch.Tensor,
    world_aabbs: np.ndarray,
    device: torch.device,
    poses_per_batch: int,
    max_steps: int | None,
    max_candidates_per_pose: int,
    seed: int,
    thresholds: np.ndarray,
    collect_pose_stats: bool = False,
    allow_candidate_visible_union: bool = False,
    bootstrap_replicates: int = 0,
    collect_score_stats: bool = False,
    collect_per_pose: bool = False,
    instance_to_glb: np.ndarray | None = None,
    glb_bytes: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    rng = np.random.default_rng(seed)
    th = thresholds.astype(np.float32)
    n_th = int(th.size)
    pose_recalls = np.zeros((n_th,), dtype=np.float64)
    pose_precisions = np.zeros((n_th,), dtype=np.float64)
    pose_f1s = np.zeros((n_th,), dtype=np.float64)
    pose_jaccards = np.zeros((n_th,), dtype=np.float64)
    pose_weighted_recalls = np.zeros((n_th,), dtype=np.float64)
    pose_specificities = np.zeros((n_th,), dtype=np.float64)
    pose_accuracies = np.zeros((n_th,), dtype=np.float64)
    pose_balanced_accuracies = np.zeros((n_th,), dtype=np.float64)
    pose_useful_culls = np.zeros((n_th,), dtype=np.float64)
    pose_bad_culls = np.zeros((n_th,), dtype=np.float64)
    pred_counts = np.zeros((n_th,), dtype=np.float64)
    predicted_glb_counts = np.zeros((n_th,), dtype=np.float64)
    predicted_glb_bytes = np.zeros((n_th,), dtype=np.float64)
    candidate_glb_count_sum = 0.0
    candidate_glb_bytes_sum = 0.0
    gt_count_sum = 0.0
    candidate_count_sum = 0.0
    agg_tp = np.zeros((n_th,), dtype=np.float64)
    agg_fp = np.zeros((n_th,), dtype=np.float64)
    agg_fn = np.zeros((n_th,), dtype=np.float64)
    agg_tn = np.zeros((n_th,), dtype=np.float64)
    agg_weighted_tp = np.zeros((n_th,), dtype=np.float64)
    agg_weighted_gt = 0.0
    pose_weighted_recall_values: list[list[float]] = [[] for _ in range(n_th)]
    pose_weighted_tp_values: list[list[float]] = [[] for _ in range(n_th)]
    pose_weighted_gt_values: list[list[float]] = [[] for _ in range(n_th)]
    score_values: list[np.ndarray] = []
    score_targets: list[np.ndarray] = []
    score_weights: list[np.ndarray] = []
    per_pose_values: list[list[dict[str, Any]]] = [[] for _ in range(n_th)]
    if (instance_to_glb is None) != (glb_bytes is None):
        raise ValueError("instance_to_glb and glb_bytes must be supplied together")
    resource_mapping = None
    resource_bytes = None
    if instance_to_glb is not None and glb_bytes is not None:
        resource_mapping = np.asarray(instance_to_glb, dtype=np.int64).reshape(-1)
        resource_bytes = np.asarray(glb_bytes, dtype=np.float64).reshape(-1)
        if resource_mapping.size != int(world_aabbs.shape[0]):
            raise ValueError("instance_to_glb does not match the runtime instance count")
        if resource_mapping.size and (
            int(resource_mapping.min()) < 0 or int(resource_mapping.max()) >= resource_bytes.size
        ):
            raise ValueError("instance_to_glb contains an id outside glb_bytes")
        if not bool(np.isfinite(resource_bytes).all()) or bool(np.any(resource_bytes < 0.0)):
            raise ValueError("glb_bytes must contain finite non-negative costs")
    pose_count = 0
    for pose_indices in split.pose_set_batches(poses_per_batch, rng, max_steps, include_empty=True):
        batch = split.build_pose_set_batch(
            pose_indices,
            world_aabbs,
            rng,
            max_candidates_per_pose=max_candidates_per_pose,
            allow_candidate_visible_union=allow_candidate_visible_union,
            include_empty=True,
        )
        batch_offsets = np.asarray(batch["pose_offsets"], dtype=np.int64)
        batch_visible_counts = np.asarray(batch.get("visible_counts", []), dtype=np.int64)
        empty_rows = np.flatnonzero(np.diff(batch_offsets) == 0)
        for row_index in empty_rows.tolist():
            if row_index < batch_visible_counts.size and int(batch_visible_counts[row_index]) != 0:
                raise ValueError(
                    f"Stored candidate semantics are invalid at empty-candidate pose row {row_index}: "
                    "the pose has visible GT instances but no candidate references."
                )
            pose_precisions += 1.0
            pose_recalls += 1.0
            pose_f1s += 1.0
            pose_jaccards += 1.0
            pose_weighted_recalls += 1.0
            pose_specificities += 1.0
            pose_accuracies += 1.0
            pose_balanced_accuracies += 1.0
            pose_useful_culls += 1.0
            if collect_pose_stats:
                for threshold_index in range(n_th):
                    pose_weighted_recall_values[threshold_index].append(1.0)
                    pose_weighted_tp_values[threshold_index].append(0.0)
                    pose_weighted_gt_values[threshold_index].append(0.0)
            if collect_per_pose:
                empty_metrics = {
                    "precision": 1.0, "recall": 1.0, "weightedRecall": 1.0,
                    "f1": 1.0, "jaccard": 1.0, "accuracy": 1.0,
                    "balancedAccuracy": 1.0, "specificity": 1.0,
                    "usefulCull": 1.0, "badCull": 0.0,
                    "avgPredCount": 0.0, "avgCandidateCount": 0.0,
                    "avgGtCount": 0.0, "predictedGlbCount": 0.0,
                    "candidateGlbCount": 0.0, "candidateGlbBytes": 0.0,
                    "predictedGlbBytes": 0.0, "glbByteReduction": 0.0,
                    "missPixelRate": 0.0, "wrongIdPixelRate": 0.0,
                    "extraPixelRate": 0.0,
                }
                for threshold_index in range(n_th):
                    per_pose_values[threshold_index].append({
                        "poseIndex": int(pose_indices[row_index]),
                        "candidateIds": [], "predictedIds": [],
                        "metrics": dict(empty_metrics),
                    })
            pose_count += 1
        if batch["instance"].size == 0:
            continue
        camera = torch.from_numpy(batch["camera"]).to(device)
        camera_world = torch.from_numpy(batch["camera_world"]).to(device)
        view = torch.from_numpy(batch["camera_view"]).to(device)
        ids = torch.from_numpy(batch["instance"]).to(device)
        query_center_world = torch.from_numpy(batch["query_center_world"]).to(device)
        viewcell_radius_m = torch.from_numpy(batch["viewcell_radius_m"]).to(device)
        logits = model.compute_visibility_logits(
            camera,
            view,
            camera_world,
            ids,
            runtime_features=runtime_features,
            query_center_world=query_center_world,
            viewcell_radius_m=viewcell_radius_m,
        )
        scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        target = batch["target"].astype(bool, copy=False)
        weights = batch.get("visible_weights", np.zeros_like(batch["target"], dtype=np.float32)).astype(np.float64, copy=False)
        if collect_score_stats:
            score_values.append(scores.astype(np.float64, copy=True))
            score_targets.append(batch["target"].astype(np.float64, copy=True))
            score_weights.append(weights.astype(np.float64, copy=True))
        offsets = batch["pose_offsets"]
        for i in range(offsets.size - 1):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            if end <= start:
                continue
            pred = scores[start:end, None] >= th[None, :]
            yy = target[start:end, None]
            ww = weights[start:end, None]
            local_tp = np.logical_and(pred, yy).sum(axis=0).astype(np.float64)
            local_fp = np.logical_and(pred, ~yy).sum(axis=0).astype(np.float64)
            local_fn = np.logical_and(~pred, yy).sum(axis=0).astype(np.float64)
            local_tn = np.logical_and(~pred, ~yy).sum(axis=0).astype(np.float64)
            precision = np.divide(
                local_tp,
                local_tp + local_fp,
                out=np.ones_like(local_tp),
                where=(local_tp + local_fp) > 0.0,
            )
            recall = np.divide(
                local_tp,
                local_tp + local_fn,
                out=np.ones_like(local_tp),
                where=(local_tp + local_fn) > 0.0,
            )
            specificity = np.divide(
                local_tn,
                local_tn + local_fp,
                out=np.ones_like(local_tn),
                where=(local_tn + local_fp) > 0.0,
            )
            accuracy = (local_tp + local_tn) / max(1.0, float(end - start))
            f1 = 2.0 * precision * recall / np.maximum(1e-8, precision + recall)
            pose_precisions += precision
            pose_recalls += recall
            pose_f1s += f1
            pose_jaccards += np.divide(
                local_tp,
                local_tp + local_fp + local_fn,
                out=np.ones_like(local_tp),
                where=(local_tp + local_fp + local_fn) > 0.0,
            )
            weighted_tp = (np.logical_and(pred, yy) * ww).sum(axis=0)
            # The GT weight mass does not depend on the threshold.  Expand
            # the scalar to the threshold axis so pose-level bootstrap rows
            # have the same shape for both one and many thresholds.
            weighted_gt_scalar = float((yy * ww).sum())
            weighted_gt = np.full((n_th,), weighted_gt_scalar, dtype=np.float64)
            weighted_recall = np.divide(
                weighted_tp,
                weighted_gt,
                out=np.ones_like(weighted_tp),
                where=weighted_gt > 0.0,
            )
            pose_weighted_recalls += weighted_recall
            agg_weighted_tp += weighted_tp
            agg_weighted_gt += weighted_gt_scalar
            pose_specificities += specificity
            pose_accuracies += accuracy
            pose_balanced_accuracies += 0.5 * (recall + specificity)
            pose_useful_culls += local_tn / max(1.0, float(end - start))
            pose_bad_culls += local_fn / max(1.0, float(end - start))
            local_predicted_glb_counts = np.zeros((n_th,), dtype=np.float64)
            local_predicted_glb_bytes = np.zeros((n_th,), dtype=np.float64)
            local_candidate_glb_count = 0.0
            local_candidate_glb_bytes = 0.0
            candidate_ids = np.asarray(batch["instance"][start:end], dtype=np.uint32)
            if resource_mapping is not None and resource_bytes is not None:
                candidate_glbs = np.unique(resource_mapping[candidate_ids.astype(np.int64, copy=False)])
                local_candidate_glb_count = float(candidate_glbs.size)
                local_candidate_glb_bytes = (
                    float(resource_bytes[candidate_glbs].sum()) if candidate_glbs.size else 0.0
                )
                candidate_glb_count_sum += local_candidate_glb_count
                candidate_glb_bytes_sum += local_candidate_glb_bytes
                instance_glbs = resource_mapping[candidate_ids.astype(np.int64, copy=False)]
                for threshold_index in range(n_th):
                    selected_glbs = np.unique(instance_glbs[pred[:, threshold_index]])
                    local_predicted_glb_counts[threshold_index] = float(selected_glbs.size)
                    local_predicted_glb_bytes[threshold_index] = (
                        float(resource_bytes[selected_glbs].sum()) if selected_glbs.size else 0.0
                    )
                predicted_glb_counts += local_predicted_glb_counts
                predicted_glb_bytes += local_predicted_glb_bytes
            if collect_pose_stats:
                for threshold_index in range(n_th):
                    pose_weighted_recall_values[threshold_index].append(float(weighted_recall[threshold_index]))
                    pose_weighted_tp_values[threshold_index].append(float(weighted_tp[threshold_index]))
                    pose_weighted_gt_values[threshold_index].append(float(weighted_gt[threshold_index]))
            if collect_per_pose:
                for threshold_index in range(n_th):
                    local_pred_ids = candidate_ids[pred[:, threshold_index]].astype(np.uint32, copy=False)
                    per_pose_values[threshold_index].append({
                        "poseIndex": int(pose_indices[i]),
                        "candidateIds": candidate_ids.tolist(),
                        "predictedIds": local_pred_ids.tolist(),
                        "metrics": {
                            "precision": float(precision[threshold_index]),
                            "recall": float(recall[threshold_index]),
                            "weightedRecall": float(weighted_recall[threshold_index]),
                            "f1": float(f1[threshold_index]),
                            "jaccard": float(local_tp[threshold_index] / max(1.0, local_tp[threshold_index] + local_fp[threshold_index] + local_fn[threshold_index])),
                            "accuracy": float(accuracy[threshold_index]),
                            "balancedAccuracy": float(0.5 * (recall[threshold_index] + specificity[threshold_index])),
                            "specificity": float(specificity[threshold_index]),
                            "usefulCull": float(local_tn[threshold_index] / max(1.0, end - start)),
                            "badCull": float(local_fn[threshold_index] / max(1.0, end - start)),
                            "avgPredCount": float(pred[:, threshold_index].sum()),
                            "avgCandidateCount": float(end - start),
                            "avgGtCount": float(yy.sum()),
                            "candidateGlbCount": local_candidate_glb_count,
                            "candidateGlbBytes": local_candidate_glb_bytes,
                            "predictedGlbCount": float(local_predicted_glb_counts[threshold_index]),
                            "predictedGlbBytes": float(local_predicted_glb_bytes[threshold_index]),
                            "glbByteReduction": float(
                                1.0 - local_predicted_glb_bytes[threshold_index] / local_candidate_glb_bytes
                                if local_candidate_glb_bytes > 0.0
                                else 0.0
                            ),
                            "missPixelRate": 0.0,
                            "wrongIdPixelRate": 0.0,
                            "extraPixelRate": 0.0,
                        },
                    })
            pred_counts += pred.sum(axis=0).astype(np.float64)
            gt_count_sum += float(yy.sum())
            candidate_count_sum += float(end - start)
            agg_tp += local_tp
            agg_fp += local_fp
            agg_fn += local_fn
            agg_tn += local_tn
            pose_count += 1
    rows = []
    distribution = None
    if collect_score_stats:
        distribution = score_distribution_summary(
            np.concatenate(score_values) if score_values else np.zeros((0,), dtype=np.float64),
            np.concatenate(score_targets) if score_targets else np.zeros((0,), dtype=np.float64),
            np.concatenate(score_weights) if score_weights else np.zeros((0,), dtype=np.float64),
        )
    for i, value in enumerate(th):
        agg_precision = agg_tp[i] / max(1.0, agg_tp[i] + agg_fp[i])
        agg_recall = agg_tp[i] / max(1.0, agg_tp[i] + agg_fn[i])
        agg_specificity = agg_tn[i] / max(1.0, agg_tn[i] + agg_fp[i])
        avg_candidate = float(candidate_count_sum / max(1, pose_count))
        avg_pred = float(pred_counts[i] / max(1, pose_count))
        row = {
            "threshold": float(value),
            "pose_precision": float(pose_precisions[i] / max(1, pose_count)),
            "pose_recall": float(pose_recalls[i] / max(1, pose_count)),
            "pose_f1": float(pose_f1s[i] / max(1, pose_count)),
            "pose_jaccard": float(pose_jaccards[i] / max(1, pose_count)),
            "pose_weighted_recall": float(pose_weighted_recalls[i] / max(1, pose_count)),
            "pose_specificity": float(pose_specificities[i] / max(1, pose_count)),
            "pose_accuracy": float(pose_accuracies[i] / max(1, pose_count)),
            "pose_balanced_accuracy": float(pose_balanced_accuracies[i] / max(1, pose_count)),
            "pose_useful_cull": float(pose_useful_culls[i] / max(1, pose_count)),
            "pose_bad_cull": float(pose_bad_culls[i] / max(1, pose_count)),
            "agg_precision": float(agg_precision),
            "agg_recall": float(agg_recall),
            "agg_weighted_recall": float(agg_weighted_tp[i] / max(1e-12, agg_weighted_gt)),
            "agg_specificity": float(agg_specificity),
            "agg_accuracy": float((agg_tp[i] + agg_tn[i]) / max(1.0, agg_tp[i] + agg_fp[i] + agg_fn[i] + agg_tn[i])),
            "agg_balanced_accuracy": float(0.5 * (agg_recall + agg_specificity)),
            "agg_useful_cull": float(agg_tn[i] / max(1.0, agg_tp[i] + agg_fp[i] + agg_fn[i] + agg_tn[i])),
            "agg_bad_cull": float(agg_fn[i] / max(1.0, agg_tp[i] + agg_fp[i] + agg_fn[i] + agg_tn[i])),
            "agg_f1": float(2.0 * agg_precision * agg_recall / max(1e-8, agg_precision + agg_recall)),
            "avg_pred_count": avg_pred,
            "avg_gt_count": float(gt_count_sum / max(1, pose_count)),
            "avg_candidate_count": avg_candidate,
            "avg_pred_glb_count": float(predicted_glb_counts[i] / max(1, pose_count)),
            "avg_pred_glb_bytes": float(predicted_glb_bytes[i] / max(1, pose_count)),
            "avg_candidate_glb_count": float(candidate_glb_count_sum / max(1, pose_count)),
            "avg_candidate_glb_bytes": float(candidate_glb_bytes_sum / max(1, pose_count)),
            "candidate_reduction_ratio": float(1.0 - avg_pred / max(1.0, avg_candidate)),
            "tp": int(agg_tp[i]),
            "fp": int(agg_fp[i]),
            "fn": int(agg_fn[i]),
            "tn": int(agg_tn[i]),
            "eval_pose_count": int(pose_count),
        }
        aggregate_weighted_recall = float(
            agg_weighted_tp[i] / agg_weighted_gt if agg_weighted_gt > 1e-12 else 1.0
        )
        pose_macro_weighted_recall = float(pose_weighted_recalls[i] / max(1, pose_count))
        aggregate_lcb = None
        pose_macro_lcb = None
        if collect_pose_stats and int(bootstrap_replicates) > 0:
            aggregate_lcb = weighted_recall_lower_confidence_bound(
                np.asarray(pose_weighted_tp_values[i]),
                np.asarray(pose_weighted_gt_values[i]),
                replicates=bootstrap_replicates,
                seed=int(seed) + i,
            )
            pose_macro_lcb = pose_macro_weighted_recall_lower_confidence_bound(
                np.asarray(pose_weighted_recall_values[i]),
                replicates=bootstrap_replicates,
                seed=int(seed) + 100000 + i,
            )
        if collect_pose_stats:
            row["_pose_weighted_recall_values"] = pose_weighted_recall_values[i]
            # A zero-replicate evaluation has only a point estimate.  Do not
            # serialize that estimate under a confidence-bound field.
            row["weighted_recall_lower_confidence_bound"] = (
                None
                if int(bootstrap_replicates) <= 0
                else weighted_recall_lower_confidence_bound(
                    np.asarray(pose_weighted_tp_values[i]),
                    np.asarray(pose_weighted_gt_values[i]),
                    replicates=bootstrap_replicates,
                    seed=int(seed) + i,
                )
            )
        row["aggregateWeightedRecall"] = aggregate_weighted_recall
        row["aggregate_weighted_recall"] = aggregate_weighted_recall
        row["poseMacroWeightedRecall"] = pose_macro_weighted_recall
        row["aggregateWeightedRecallLowerConfidenceBound"] = aggregate_lcb
        row["poseMacroWeightedRecallLowerConfidenceBound"] = pose_macro_lcb
        row["weightedRecallBootstrapReplicates"] = int(max(0, bootstrap_replicates))
        if distribution is not None:
            row["scoreDistribution"] = distribution
        if collect_per_pose:
            row["_per_pose"] = per_pose_values[i]
        rows.append(row)
    return rows


def visual_utility_loss(
    aux: dict[str, torch.Tensor],
    target: torch.Tensor,
    visible_weights: torch.Tensor,
    pose_offsets: torch.Tensor,
    invisible_weight: float,
    rank_weight: float,
    rank_margin: float,
    rank_positive_top_k: int,
    rank_negative_top_k: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = aux["utility_logits"].float().view(-1)
    pred = torch.sigmoid(logits)
    y = target.float().view(-1)
    weights = visible_weights.float().view(-1)
    target_utility = torch.zeros_like(pred)
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        local_y = y[start:end]
        local_raw = torch.log1p(torch.clamp(weights[start:end], min=0.0)) * local_y
        denom = torch.clamp(local_raw.max(), min=1e-6)
        target_utility[start:end] = local_raw / denom

    visible = y > 0.5
    invisible = ~visible
    reg_visible = F.smooth_l1_loss(pred[visible], target_utility[visible], reduction="mean") if visible.any() else torch.zeros((), device=pred.device)
    reg_invisible = torch.square(pred[invisible]).mean() if invisible.any() else torch.zeros((), device=pred.device)

    rank_terms = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        local_pred = pred[start:end]
        local_target = target_utility[start:end]
        pos_mask = local_target > 0.02
        neg_mask = local_target <= 0.0
        if not (pos_mask.any() and neg_mask.any()):
            continue
        pos_pred = local_pred[pos_mask]
        pos_target = local_target[pos_mask]
        neg_pred = local_pred[neg_mask]
        if pos_pred.numel() > int(rank_positive_top_k):
            top = torch.topk(pos_target, k=int(rank_positive_top_k)).indices
            pos_pred = pos_pred[top]
            pos_target = pos_target[top]
        if neg_pred.numel() > int(rank_negative_top_k):
            neg_pred = torch.topk(neg_pred, k=int(rank_negative_top_k)).values
        pair = F.softplus(neg_pred[:, None] - pos_pred[None, :] + float(rank_margin))
        pair_weight = torch.clamp(pos_target[None, :] / torch.clamp(pos_target.max(), min=1e-6), 0.1, 1.0)
        rank_terms.append((pair * pair_weight).mean())
    rank = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=pred.device)
    loss = reg_visible + float(invisible_weight) * reg_invisible + float(rank_weight) * rank
    return loss, {
        "lossUtilityVisibleReg": float(reg_visible.detach().cpu()),
        "lossUtilityInvisibleReg": float(reg_invisible.detach().cpu()),
        "lossUtilityRank": float(rank.detach().cpu()),
        "utilityPredMean": float(pred.detach().mean().cpu()),
        "utilityTargetMean": float(target_utility.detach().mean().cpu()),
        "utilityVisibleTargetMean": float(target_utility[visible].detach().mean().cpu()) if visible.any() else 0.0,
    }


def load_threshold(eval_summary: str | None, fallback: float, model_name: str) -> float:
    if not eval_summary:
        return float(fallback)
    path = Path(eval_summary)
    if not path.exists():
        return float(fallback)
    data = json.loads(path.read_text(encoding="utf-8"))
    frozen = data.get("frozenThreshold")
    if frozen is not None:
        return float(frozen)
    rows = data.get("thresholdRows") or []
    selected = select_weighted_precision_workpoint(
        rows,
        target_weighted_recall_from_payload(data),
    )
    if selected is not None:
        return float(selected.get("threshold", fallback))
    if rows:
        raise RuntimeError(
            f"{path} has no threshold satisfying the strict weighted-recall rule; "
            "refusing to load an unsafe threshold."
        )
    return float(fallback)


def infer_prediction_camera_fields(
    checkpoint: dict,
    config: dict,
    dataset_meta_arg: str | None,
    prediction_camera_mode_arg: str | None,
    pvs_back_offset_arg: float | None,
) -> dict[str, Any]:
    args = checkpoint.get("args") or {}
    dataset_meta_path = Path(dataset_meta_arg) if dataset_meta_arg else None
    if dataset_meta_path is None:
        dataset_dir = args.get("dataset_dir") or args.get("dataset") or config.get("datasetDir")
        if dataset_dir:
            dataset_meta_path = Path(dataset_dir) / "dataset_meta.json"

    dataset_meta = {}
    if dataset_meta_path and dataset_meta_path.exists():
        dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))

    dataset_hint = " ".join(
        str(value)
        for value in [
            dataset_meta.get("schema"),
            dataset_meta.get("cameraSemantics"),
            args.get("dataset_dir"),
            args.get("dataset"),
        ]
        if value
    ).lower()
    is_viewcell_back_camera = "viewcell" in dataset_hint and "back" in dataset_hint
    mode = prediction_camera_mode_arg or config.get("predictionCameraMode")
    if not mode:
        mode = "viewcell-back-camera" if is_viewcell_back_camera else "active-camera"

    stats = dataset_meta.get("stats") or {}

    def first_range_value(name: str, fallback: float | None) -> float | None:
        value = stats.get(name)
        if isinstance(value, list) and value:
            return float(value[0])
        if value is not None:
            return float(value)
        return fallback

    pvs_back_offset = pvs_back_offset_arg
    if pvs_back_offset is None:
        pvs_back_offset = config.get("pvsBackOffsetM")
    if pvs_back_offset is None:
        pvs_back_offset = first_range_value("pvsBackOffsetRange", 0.0)

    return {
        "predictionCameraMode": str(mode),
        "pvsBackOffsetM": float(pvs_back_offset if pvs_back_offset is not None else 0.0),
        "modelInputFovYDeg": MODEL_INPUT_FOV_Y_DEG,
        "predictionCameraMetaSource": "neural_instance_culling/config/neuralpvs_viewcell_protocol.json",
    }


def as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


class AssetBuilder:
    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.layout: list[dict[str, Any]] = []
        self.byte_offset = 0

    def _pad_to(self, alignment: int) -> None:
        pad = (-self.byte_offset) % int(alignment)
        if pad:
            self.parts.append(bytes(pad))
            self.byte_offset += pad

    def append_fp16(self, name: str, array: torch.Tensor | np.ndarray) -> None:
        self._pad_to(2)
        arr = np.ascontiguousarray(as_numpy(array).astype(np.float16, copy=False))
        data = arr.reshape(-1).tobytes()
        self.layout.append({
            "name": name,
            "type": "fp16",
            "halfOffset": int(self.byte_offset // 2),
            "floatOffset": int(self.byte_offset // 2),
            "wordOffset": int(self.byte_offset // 4),
            "byteOffset": int(self.byte_offset),
            "elementCount": int(arr.size),
            "byteSize": int(len(data)),
            "shape": list(arr.shape),
        })
        self.parts.append(data)
        self.byte_offset += len(data)

    def append_u32(self, name: str, array: torch.Tensor | np.ndarray) -> None:
        self._pad_to(4)
        arr = np.ascontiguousarray(as_numpy(array).astype(np.uint32, copy=False))
        data = arr.reshape(-1).tobytes()
        self.layout.append({
            "name": name,
            "type": "u32",
            "halfOffset": int(self.byte_offset // 2),
            "floatOffset": int(self.byte_offset // 2),
            "wordOffset": int(self.byte_offset // 4),
            "byteOffset": int(self.byte_offset),
            "elementCount": int(arr.size),
            "byteSize": int(len(data)),
            "shape": list(arr.shape),
        })
        self.parts.append(data)
        self.byte_offset += len(data)

    def write(self, path: Path) -> None:
        path.write_bytes(b"".join(self.parts))
