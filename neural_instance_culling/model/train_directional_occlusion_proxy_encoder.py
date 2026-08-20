#!/usr/bin/env python3
"""Train the directional occlusion proxy encoder PVS model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from common.glb_points import load_glb_points
from common.occlusion_edges import glb_priority_loss, load_glb_costs
from common.pose_set_loss import (
    pose_set_visibility_loss_with_calibration,
    pose_set_visibility_loss_with_importance,
    pose_subpose_robust_safety_loss,
    pose_visual_safety_loss,
)
from common.runtime_meta import load_runtime_meta, scene_min_max
from directional_occlusion_proxy_encoder_model import DirectionalOcclusionProxyEncoderPVSModel, load_directional_occlusion_evidence
from pose_csr_dataset import PoseCSRDataset
from current_pvs_utils import evaluate_thresholds, threshold_grid, validate_training_resources, visual_utility_loss
from common.threshold_selection import (
    attach_viewcell_weighted_recall_bounds,
    select_weighted_precision_workpoint,
    weighted_precision_selection_key,
    weighted_precision_selection_rule,
)


class PoseContrastiveProjectionHead(nn.Module):
    """Training-only projection head for pose-local contrastive supervision."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features.float()), dim=-1, eps=1e-6)


def pose_hard_negative_contrastive_loss(
    embeddings: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    *,
    temperature: float,
    positive_top_k: int,
    negative_top_k: int,
    importance_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Separate important visible queries from hard invisible queries per pose.

    Positives are not contrasted across unrelated poses. Within each pose, the
    most important visible instances are supervised as one positive set while
    the highest-scoring invisible instances form the denominator's hard
    negatives. The projection head is discarded after training.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = F.normalize(embeddings.float(), dim=-1, eps=1e-6)
    scores = logits.float().view(-1)
    y = target.float().view(-1)
    raw_weights = torch.clamp(visible_weights.float().view(-1), min=0.0)
    importance_denom = torch.log1p(
        torch.tensor(1024.0, device=raw_weights.device, dtype=raw_weights.dtype)
    )
    importance = torch.clamp(
        torch.log1p(raw_weights) / torch.clamp(importance_denom, min=1e-6),
        0.0,
        1.0,
    )

    pose_losses: list[torch.Tensor] = []
    positive_similarities: list[torch.Tensor] = []
    hard_negative_similarities: list[torch.Tensor] = []
    anchor_count = 0
    hard_negative_count = 0
    max_pos = max(2, int(positive_top_k))
    max_neg = max(1, int(negative_top_k))
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        local_y = y[start:end]
        pos_indices = torch.nonzero(local_y > 0.5, as_tuple=False).view(-1)
        neg_indices = torch.nonzero(local_y <= 0.5, as_tuple=False).view(-1)
        if pos_indices.numel() < 2 or neg_indices.numel() == 0:
            continue
        local_importance = importance[start:end]
        if pos_indices.numel() > max_pos:
            selected = torch.topk(local_importance[pos_indices], k=max_pos).indices
            pos_indices = pos_indices[selected]
        if neg_indices.numel() > max_neg:
            selected = torch.topk(scores[start:end][neg_indices], k=max_neg).indices
            neg_indices = neg_indices[selected]

        pos_z = z[start:end][pos_indices]
        neg_z = z[start:end][neg_indices]
        candidates = torch.cat([pos_z, neg_z], dim=0)
        similarities = pos_z @ candidates.transpose(0, 1)
        scaled = similarities / float(temperature)
        pos_count = int(pos_z.shape[0])
        self_mask = torch.zeros_like(scaled, dtype=torch.bool)
        self_mask[:, :pos_count] = torch.eye(pos_count, device=z.device, dtype=torch.bool)
        denominator = torch.logsumexp(scaled.masked_fill(self_mask, float("-inf")), dim=1)
        positive_logits = scaled[:, :pos_count].masked_fill(
            torch.eye(pos_count, device=z.device, dtype=torch.bool),
            float("-inf"),
        )
        numerator = torch.logsumexp(positive_logits, dim=1)
        anchor_loss = denominator - numerator
        anchor_weights = 1.0 + float(importance_scale) * local_importance[pos_indices]
        pose_losses.append(
            (anchor_loss * anchor_weights).sum() / torch.clamp(anchor_weights.sum(), min=1.0)
        )

        off_diagonal = ~torch.eye(pos_count, device=z.device, dtype=torch.bool)
        positive_similarities.append(similarities[:, :pos_count][off_diagonal].mean())
        hard_negative_similarities.append(similarities[:, pos_count:].mean())
        anchor_count += pos_count
        hard_negative_count += int(neg_z.shape[0])

    if not pose_losses:
        zero = embeddings.sum() * 0.0
        return zero, {
            "lossContrastive": 0.0,
            "contrastiveAnchorCount": 0.0,
            "contrastiveHardNegativeCount": 0.0,
            "contrastivePositiveSimilarity": 0.0,
            "contrastiveHardNegativeSimilarity": 0.0,
        }
    loss = torch.stack(pose_losses).mean()
    return loss, {
        "lossContrastive": float(loss.detach().cpu()),
        "contrastiveAnchorCount": float(anchor_count),
        "contrastiveHardNegativeCount": float(hard_negative_count),
        "contrastivePositiveSimilarity": float(torch.stack(positive_similarities).mean().detach().cpu()),
        "contrastiveHardNegativeSimilarity": float(torch.stack(hard_negative_similarities).mean().detach().cpu()),
    }


def select_target_recall_workpoints(rows: list[dict[str, Any]], target_recall: float, target_weighted_recall: float) -> dict[str, Any]:
    if not rows:
        return {}
    best_f1 = max(rows, key=lambda row: row["pose_f1"])
    recall_safe = [row for row in rows if row["pose_recall"] >= float(target_recall)]
    primary_weighted = select_weighted_precision_workpoint(rows, target_weighted_recall)
    primary_recall = (
        max(recall_safe, key=lambda row: (row["pose_precision"], row["pose_weighted_recall"], row["pose_f1"], -row["avg_pred_count"]))
        if recall_safe
        else None
    )
    fallback = max(rows, key=lambda row: (row["pose_weighted_recall"], row["pose_precision"], row["pose_f1"], -row["avg_pred_count"]))
    return {
        "bestF1": best_f1,
        "primaryWeightedPrecision": primary_weighted,
        "primaryWeightedTargetRecall": primary_weighted,
        "primaryTargetRecall": primary_recall,
        "primaryHighRecall": primary_weighted or fallback,
        "metTargetRecall": primary_recall is not None,
        "metTargetWeightedRecall": primary_weighted is not None,
        "selectionStatus": "weighted_recall_safe" if primary_weighted is not None else "weighted_recall_target_unmet",
        "selectionRule": weighted_precision_selection_rule(
            target_weighted_recall,
            minimum_pose_recall=target_recall,
        ),
        "targetRecall": float(target_recall),
        "targetWeightedRecall": float(target_weighted_recall),
    }


def target_recall_selection_score(row: dict[str, Any], target_recall: float, target_weighted_recall: float) -> float:
    """Compatibility score: unsafe rows are never eligible for best.pt."""
    key = weighted_precision_selection_key(row, target_weighted_recall)
    if key is None:
        return float("-inf")
    precision, f1, weighted, neg_avg_pred = key
    return precision + 1e-6 * f1 + 1e-9 * weighted + 1e-12 * neg_avg_pred


def relative_checkpoint_selection_key(
    row: dict[str, Any],
    target_weighted_recall: float,
    *,
    calibration_safe: bool,
) -> tuple[float, float, float, float, float]:
    """Rank every completed checkpoint without weakening the safety label.

    A checkpoint that is safe on both calibration and validation always ranks
    above an unsafe checkpoint.  When a pilot has no safe member, retaining the
    relatively best unsafe checkpoint preserves the completed trajectory for
    the mandatory follow-up run instead of turning the safety gate into an
    experiment-cancellation gate.
    """
    safe_key = weighted_precision_selection_key(row, target_weighted_recall)
    if calibration_safe and safe_key is not None:
        precision, f1, weighted, neg_avg_pred = safe_key
        return (1.0, precision, f1, weighted, neg_avg_pred)
    return (
        0.0,
        float(row.get("pose_weighted_recall", 0.0)),
        float(row.get("pose_balanced_accuracy", 0.0)),
        float(row.get("pose_precision", 0.0)),
        -float(row.get("avg_pred_count", 0.0)),
    )


def select_diagnostic_calibration_workpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose an unsafe monitoring row without treating it as a safe workpoint.

    Intermediate optimization checkpoints may temporarily miss the registered
    safety target.  The returned row is only used to keep validation telemetry
    continuous; callers must not use it to select ``best.pt`` or freeze a
    threshold.
    """
    if not rows:
        raise ValueError("Cannot select a diagnostic calibration row from an empty list.")
    return max(
        rows,
        key=lambda row: (
            float(row.get("pose_weighted_recall", 0.0)),
            float(row.get("pose_precision", 0.0)),
            float(row.get("pose_f1", 0.0)),
            -float(row.get("avg_pred_count", 0.0)),
        ),
    )


def parse_float_list(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v.strip()]


def _pose_index_digest(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def build_protocol_splits(
    dataset: PoseCSRDataset,
    seed: int,
    calibration_fraction: float,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Build deterministic train/validation/calibration/test views.

    The dataset's original test split is never touched.  Calibration poses are
    held out from the original training split using a seed-stable permutation;
    the complete original validation split remains the fixed checkpoint
    selection set.  All views share the original memory-mapped CSR arrays.
    """
    # A rebuilt spatial dataset may already contain immutable four-way split
    # IDs.  Prefer those native boundaries; never carve calibration poses out
    # of a formal validation/test manifest.
    if "calibration" in dataset.split_ids:
        validation_name = "validation" if "validation" in dataset.split_ids else "val"
        train_split = dataset.split("train")
        validation_split = dataset.split(validation_name)
        calibration_split = dataset.split("calibration")
        test_split = dataset.split("test")
        native = {
            "schema": "native-four-way-dataset-split-v1",
            "source": "dataset_meta.splitIds",
            "selectionSeed": int(seed),
            "calibrationFractionOfOriginalTrain": None,
            "originalTrainCount": int(train_split.pose_indices.size),
            "trainFitCount": int(train_split.pose_indices.size),
            "fixedValidationCount": int(validation_split.pose_indices.size),
            "calibrationCount": int(calibration_split.pose_indices.size),
            "frozenTestCount": int(test_split.pose_indices.size),
            "trainFitDigest": _pose_index_digest(train_split.pose_indices),
            "fixedValidationDigest": _pose_index_digest(validation_split.pose_indices),
            "calibrationDigest": _pose_index_digest(calibration_split.pose_indices),
            "frozenTestDigest": _pose_index_digest(test_split.pose_indices),
            "validationSemantics": "complete native validation split; no random pose truncation",
            "calibrationSemantics": "native spatial calibration split; not used for parameter updates",
            "testSemantics": "native frozen spatial test split; evaluated once at the frozen calibration threshold",
        }
        return train_split, validation_split, calibration_split, test_split, native

    fraction = float(calibration_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("calibration_fraction must be strictly between 0 and 0.5")

    original_train = np.asarray(dataset.split("train").pose_indices, dtype=np.int64)
    fixed_validation = np.asarray(dataset.split("val").pose_indices, dtype=np.int64)
    frozen_test = np.asarray(dataset.split("test").pose_indices, dtype=np.int64)
    if original_train.size < 2:
        raise ValueError("The original training split is too small to hold out calibration poses")

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(original_train)
    calibration_count = max(1, int(round(permutation.size * fraction)))
    calibration_count = min(calibration_count, permutation.size - 1)
    calibration_indices = np.sort(permutation[:calibration_count])
    train_indices = np.sort(permutation[calibration_count:])

    train_split = dataset.subset("train_fit", train_indices)
    validation_split = dataset.subset("validation_fixed", fixed_validation)
    calibration_split = dataset.subset("calibration_fixed", calibration_indices)
    test_split = dataset.subset("test_frozen", frozen_test)
    protocol = {
        "schema": "fixed-validation-independent-calibration-v1",
        "selectionSeed": int(seed),
        "calibrationFractionOfOriginalTrain": fraction,
        "originalTrainCount": int(original_train.size),
        "trainFitCount": int(train_indices.size),
        "fixedValidationCount": int(fixed_validation.size),
        "calibrationCount": int(calibration_indices.size),
        "frozenTestCount": int(frozen_test.size),
        "trainFitDigest": _pose_index_digest(train_indices),
        "fixedValidationDigest": _pose_index_digest(fixed_validation),
        "calibrationDigest": _pose_index_digest(calibration_indices),
        "frozenTestDigest": _pose_index_digest(frozen_test),
        "validationSemantics": "complete original val split; no random pose truncation",
        "calibrationSemantics": "held out from original train and used only for threshold selection",
        "testSemantics": "frozen original test split; evaluated once at the frozen calibration threshold",
    }
    return train_split, validation_split, calibration_split, test_split, protocol


def apply_train_pose_fraction(
    dataset: PoseCSRDataset,
    train_split,
    protocol: dict[str, Any],
    fraction: float,
    seed: int,
):
    """Select a deterministic subset of native train poses for M11 adaptation.

    Validation, calibration and test remain untouched.  The selected indices
    are recorded in the checkpoint protocol so a later frozen-test audit can
    distinguish few-shot adaptation from full-data training.
    """
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train_pose_fraction must be in (0, 1], got {fraction}")
    indices = np.asarray(train_split.pose_indices, dtype=np.int64)
    if fraction >= 1.0:
        return train_split, protocol
    if indices.size == 0:
        raise ValueError("Cannot select a few-shot training subset from an empty train split")
    count = max(1, int(round(indices.size * fraction)))
    rng = np.random.default_rng(int(seed) + 3109)
    selected = np.sort(rng.choice(indices, size=count, replace=False).astype(np.int64, copy=False))
    selected_split = dataset.subset("train_fit_fraction", selected)
    updated = dict(protocol)
    updated.update(
        {
            "trainFitSelectionFraction": fraction,
            "trainFitSelectionSeed": int(seed) + 3109,
            "trainFitCount": int(selected.size),
            "trainFitDigest": _pose_index_digest(selected),
            "trainFitSemantics": "deterministic subset of native train poses; validation/calibration/test unchanged",
        }
    )
    return selected_split, updated


def load_initial_checkpoint(
    model: DirectionalOcclusionProxyEncoderPVSModel,
    checkpoint_path: Path,
    allow_scene_transfer: bool,
) -> dict[str, Any]:
    """Load a normal checkpoint or only shared parameters for scene transfer."""
    checkpoint = torch.load(checkpoint_path, map_location=model.scene_min.device)
    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError(f"{checkpoint_path} has no model state")
    if not allow_scene_transfer:
        model.load_state_dict(source_state, strict=True)
        return {
            "path": str(checkpoint_path.as_posix()),
            "mode": "strict_same_scene_checkpoint",
            "best": checkpoint.get("best"),
            "config": checkpoint.get("config"),
        }

    target_parameters = dict(model.named_parameters())
    transferable: dict[str, torch.Tensor] = {}
    mismatched: list[str] = []
    for name, value in source_state.items():
        if name not in target_parameters:
            continue
        if tuple(value.shape) != tuple(target_parameters[name].shape):
            mismatched.append(name)
            continue
        transferable[name] = value
    missing = sorted(set(target_parameters) - set(transferable))
    if mismatched or missing:
        raise ValueError(
            "scene-transfer checkpoint is incompatible with the target model: "
            f"mismatched={mismatched[:8]}, missing={missing[:8]}"
        )
    model.load_state_dict(transferable, strict=False)
    return {
        "path": str(checkpoint_path.as_posix()),
        "mode": "shared_learnable_parameters_only_scene_buffers_rebuilt",
        "transferredParameterCount": len(transferable),
        "sourceConfig": checkpoint.get("config"),
        "sourceBest": checkpoint.get("best"),
    }


def evaluate_calibration_and_validation(
    model,
    validation_split,
    calibration_split,
    runtime_features: torch.Tensor,
    world_aabbs: np.ndarray,
    device: torch.device,
    poses_per_batch: int,
    max_candidates_per_pose: int,
    seed: int,
    target_weighted_recall: float,
    calibration_point_floor: float,
    calibration_pose_recall_floor: float | None = None,
    calibration_lcb_floor: float | None = None,
    calibration_bootstrap_replicates: int = 0,
    calibration_bootstrap_confidence: float = 0.95,
    allow_candidate_visible_union: bool = False,
    require_safe_workpoint: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Select a threshold on calibration, then evaluate fixed validation once.

    The returned validation row is measured only at the calibration-selected
    threshold.  Consequently validation never participates in threshold
    scanning, and the later test pass can use the same frozen scalar.
    """
    calibration_rows = evaluate_thresholds(
        model,
        calibration_split,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=poses_per_batch,
        max_steps=None,
        max_candidates_per_pose=max_candidates_per_pose,
        seed=int(seed),
        thresholds=threshold_grid(),
        collect_pose_stats=bool(calibration_bootstrap_replicates > 0),
        allow_candidate_visible_union=allow_candidate_visible_union,
    )
    attach_viewcell_weighted_recall_bounds(
        calibration_rows,
        bootstrap_replicates=calibration_bootstrap_replicates,
        confidence=calibration_bootstrap_confidence,
        seed=int(seed) + 17,
    )
    calibration_workpoint = select_weighted_precision_workpoint(
        calibration_rows,
        target_weighted_recall,
        minimum_point_estimate=calibration_point_floor,
        minimum_lower_confidence_bound=calibration_lcb_floor if calibration_bootstrap_replicates > 0 else None,
        minimum_pose_recall=calibration_pose_recall_floor,
    )
    diagnostic_fallback = None
    if calibration_workpoint is None:
        if require_safe_workpoint:
            raise RuntimeError(
                "No calibration threshold satisfies the strict weighted-recall rule "
                f"pose_weighted_recall > {float(target_weighted_recall):.3f}."
            )
        if not calibration_rows:
            raise RuntimeError("Calibration produced no threshold rows for diagnostic evaluation.")
        # Intermediate epochs are allowed to be unsafe while optimization
        # continues.  This row is diagnostic only and can never select
        # best.pt; final calibration keeps the strict failure behavior above.
        diagnostic_fallback = select_diagnostic_calibration_workpoint(calibration_rows)
        frozen_threshold = float(diagnostic_fallback["threshold"])
        selection_status = "unsafe_diagnostic_fallback"
    else:
        frozen_threshold = float(calibration_workpoint["threshold"])
        selection_status = "safe"
    validation_row = evaluate_thresholds(
        model,
        validation_split,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=poses_per_batch,
        max_steps=None,
        max_candidates_per_pose=max_candidates_per_pose,
        seed=int(seed) + 1,
        thresholds=np.asarray([frozen_threshold], dtype=np.float32),
        allow_candidate_visible_union=allow_candidate_visible_union,
    )[0]
    validation_workpoint = {
        "threshold": frozen_threshold,
        **validation_row,
    }
    calibration_summary = {
        "selected": calibration_workpoint,
        "diagnosticFallback": diagnostic_fallback,
        "selectionStatus": selection_status,
        "requiresSafeWorkpoint": bool(require_safe_workpoint),
        "thresholdRows": calibration_rows,
        "selectionRule": weighted_precision_selection_rule(
            target_weighted_recall,
            minimum_point_estimate=calibration_point_floor,
            minimum_lower_confidence_bound=calibration_lcb_floor if calibration_bootstrap_replicates > 0 else None,
            minimum_pose_recall=calibration_pose_recall_floor,
        ),
        "bootstrap": {
            "replicates": int(calibration_bootstrap_replicates),
            "confidence": float(calibration_bootstrap_confidence),
            "lowerBoundFloor": calibration_lcb_floor if calibration_bootstrap_replicates > 0 else None,
        },
    }
    return calibration_workpoint, validation_workpoint, calibration_rows, calibration_summary


def proxy_evidence_loss(
    aux: dict[str, torch.Tensor],
    target: torch.Tensor,
    evidence_weight: float,
    visible_guard_weight: float,
    sparsity_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    y = target.float().view(-1, 1)
    evidence = torch.clamp(aux["evidence_target"].float().view(-1, 1), 0.0, 1.0)
    inhibition = aux["inhibition"].float().view(-1, 1)
    inhibition_norm = torch.clamp(inhibition / 4.0, 0.0, 1.0)
    invisible = (y <= 0.5).float()
    visible = (y > 0.5).float()
    evidence_loss = (F.smooth_l1_loss(inhibition_norm, evidence, reduction="none") * invisible * (0.25 + evidence)).sum()
    evidence_loss = evidence_loss / torch.clamp((invisible * (0.25 + evidence)).sum(), min=1.0)
    visible_guard = (inhibition * visible).sum() / torch.clamp(visible.sum(), min=1.0)
    sparsity = (inhibition * invisible * (1.0 - evidence)).sum() / torch.clamp((invisible * (1.0 - evidence)).sum(), min=1.0)
    loss = float(evidence_weight) * evidence_loss + float(visible_guard_weight) * visible_guard + float(sparsity_weight) * sparsity
    return loss, {
        "lossProxyEvidence": float(evidence_loss.detach().cpu()),
        "lossProxyVisibleGuard": float(visible_guard.detach().cpu()),
        "lossProxySparsity": float(sparsity.detach().cpu()),
        "proxyEvidenceMean": float(evidence.detach().mean().cpu()),
        "proxyInhibitionMean": float(inhibition.detach().mean().cpu()),
    }


def prediction_budget_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    safety_multiplier: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = torch.sigmoid(logits.float()).view(-1)
    y = target.float().view(-1)
    terms = []
    ratios = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        pred_count = pred[start:end].sum()
        gt_count = torch.clamp(y[start:end].sum(), min=1.0)
        budget = gt_count * float(safety_multiplier)
        terms.append(torch.square(F.relu(pred_count - budget) / torch.clamp(torch.tensor(float(end - start), device=pred.device), min=1.0)))
        ratios.append(pred_count.detach() / gt_count.detach())
    if not terms:
        return torch.zeros((), device=logits.device), {"lossPredBudget": 0.0, "predGtRatio": 0.0}
    loss = torch.stack(terms).mean()
    ratio = torch.stack(ratios).mean()
    return loss, {"lossPredBudget": float(loss.detach().cpu()), "predGtRatio": float(ratio.detach().cpu())}


def proxy_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    evidence: torch.Tensor,
    pose_offsets: torch.Tensor,
    margin: float,
    max_pos: int = 64,
    max_neg: int = 256,
) -> tuple[torch.Tensor, dict[str, float]]:
    scores = logits.float().view(-1)
    y = target.float().view(-1)
    ev = evidence.float().view(-1)
    terms = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        pos = y[start:end] > 0.5
        hard_neg = (y[start:end] <= 0.5) & (ev[start:end] > 0.05)
        if not (pos.any() and hard_neg.any()):
            continue
        pos_scores = scores[start:end][pos]
        neg_scores = scores[start:end][hard_neg]
        neg_ev = ev[start:end][hard_neg]
        if pos_scores.numel() > max_pos:
            pos_scores = torch.topk(pos_scores, k=max_pos).values
        if neg_scores.numel() > max_neg:
            top = torch.topk(neg_ev, k=max_neg).indices
            neg_scores = neg_scores[top]
            neg_ev = neg_ev[top]
        pair = F.softplus(neg_scores[:, None] - pos_scores[None, :] + float(margin))
        weight = torch.clamp(neg_ev[:, None] / torch.clamp(neg_ev.max(), min=1e-6), 0.1, 1.0)
        terms.append((pair * weight).mean())
    loss = torch.stack(terms).mean() if terms else torch.zeros((), device=logits.device)
    return loss, {"lossProxyRank": float(loss.detach().cpu())}


def repulsive_visibility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    evidence: torch.Tensor,
    instance_ids: torch.Tensor,
    instance_to_glb: torch.Tensor,
    glb_cost_norm: torch.Tensor,
    mode: str,
    fn_weight: float,
    fp_weight: float,
    evidence_scale: float,
    cost_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if mode == "off":
        return torch.zeros((), device=logits.device), {
            "lossRvl": 0.0,
            "lossRvlFn": 0.0,
            "lossRvlFp": 0.0,
            "rvlNegWeightMean": 0.0,
        }
    prob = torch.sigmoid(logits.float()).view(-1)
    y = target.float().view(-1)
    weights = torch.clamp(visible_weights.float().view(-1), min=0.0)
    evidence_flat = torch.clamp(evidence.float().view(-1), 0.0, 1.0)
    ids = instance_ids.long().view(-1)
    importance = torch.clamp(torch.log1p(weights) / torch.log1p(torch.tensor(1024.0, device=weights.device, dtype=weights.dtype)), 0.0, 1.0)
    positive_weight = torch.where(y > 0.5, 1.0 + importance, torch.ones_like(y))
    neg_weight = torch.ones_like(y)
    if mode in {"evidence", "cost_aware"}:
        neg_weight = neg_weight + float(evidence_scale) * evidence_flat
    if mode == "cost_aware":
        glb_ids = torch.clamp(instance_to_glb[ids], 0, glb_cost_norm.numel() - 1)
        cost = torch.clamp(glb_cost_norm[glb_ids].float(), min=0.0)
        if cost.numel():
            cost = cost / torch.clamp(cost.mean().detach(), min=1e-6)
        neg_weight = neg_weight * (1.0 + float(cost_scale) * torch.clamp(cost, 0.0, 4.0))
    neg_weight = torch.where(y <= 0.5, neg_weight, torch.zeros_like(neg_weight))

    fn_terms = []
    fp_terms = []
    neg_weight_means = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        local_y = y[start:end]
        local_p = prob[start:end]
        local_pos_w = positive_weight[start:end]
        local_neg_w = neg_weight[start:end]
        pos_mask = local_y > 0.5
        neg_mask = ~pos_mask
        if not pos_mask.any():
            continue
        visible_mass = torch.clamp((local_y * local_pos_w).sum(), min=1.0)
        visible_count = torch.clamp(local_y.sum(), min=1.0)
        soft_fn = ((1.0 - local_p) * local_y * local_pos_w).sum() / visible_mass
        soft_fp = (local_p * (1.0 - local_y) * local_neg_w).sum() / visible_count
        fn_terms.append(soft_fn)
        fp_terms.append(soft_fp)
        if neg_mask.any():
            neg_weight_means.append(local_neg_w[neg_mask].mean())
    loss_fn = torch.stack(fn_terms).mean() if fn_terms else torch.zeros((), device=logits.device)
    loss_fp = torch.stack(fp_terms).mean() if fp_terms else torch.zeros((), device=logits.device)
    loss = float(fn_weight) * loss_fn + float(fp_weight) * loss_fp
    neg_mean = torch.stack(neg_weight_means).mean() if neg_weight_means else torch.zeros((), device=logits.device)
    return loss, {
        "lossRvl": float(loss.detach().cpu()),
        "lossRvlFn": float(loss_fn.detach().cpu()),
        "lossRvlFp": float(loss_fp.detach().cpu()),
        "rvlNegWeightMean": float(neg_mean.detach().cpu()),
    }


def visibility_supervision_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pose_offsets: torch.Tensor,
    visible_weights: torch.Tensor,
    visible_hit_rates: torch.Tensor | None,
    evidence: torch.Tensor,
    instance_ids: torch.Tensor,
    instance_to_glb: torch.Tensor,
    glb_cost_norm: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if args.visibility_loss_mode == "calibrated":
        set_loss, set_parts = pose_set_visibility_loss_with_calibration(
            logits,
            target,
            pose_offsets,
            visible_weights=visible_weights,
            bce_weight=args.set_bce_weight,
            tversky_weight=args.set_tversky_weight,
            count_weight=args.set_count_weight,
            rank_weight=args.set_rank_weight,
            positive_margin_weight=args.positive_margin_weight,
            hard_negative_margin_weight=args.hard_negative_margin_weight,
            pos_class_weight=args.pos_class_weight,
            neg_class_weight=args.neg_class_weight,
            tversky_alpha=args.pose_tversky_alpha,
            tversky_beta=args.pose_tversky_beta,
            rank_margin=args.calibration_rank_margin,
            positive_logit_margin=args.positive_logit_margin,
            hard_negative_logit_margin=args.hard_negative_logit_margin,
            hard_negative_top_k=args.hard_negative_top_k,
            importance_weight_scale=args.importance_weight_scale,
        )
    else:
        set_loss, set_parts = pose_set_visibility_loss_with_importance(
            logits,
            target,
            pose_offsets,
            visible_weights=visible_weights,
            false_negative_weight=args.false_negative_weight,
            bce_weight=args.set_bce_weight,
            tversky_weight=args.set_tversky_weight,
            count_weight=args.set_count_weight,
            rank_weight=args.set_rank_weight,
            tversky_alpha=args.pose_tversky_alpha,
            tversky_beta=args.pose_tversky_beta,
            importance_weight_scale=args.importance_weight_scale,
        )
    rvl_loss, rvl_parts = repulsive_visibility_loss(
        logits,
        target,
        pose_offsets,
        visible_weights,
        evidence,
        instance_ids,
        instance_to_glb,
        glb_cost_norm,
        mode=args.rvl_mode,
        fn_weight=args.rvl_fn_weight,
        fp_weight=args.rvl_fp_weight,
        evidence_scale=args.rvl_evidence_scale,
        cost_scale=args.rvl_cost_scale,
    )
    scaled_rvl = float(args.rvl_loss_weight) * rvl_loss
    budget_loss, budget_parts = prediction_budget_loss(logits, target, pose_offsets, args.budget_safety_multiplier)
    proxy_rank_loss_value, proxy_rank_parts = proxy_rank_loss(logits, target, evidence, pose_offsets, args.proxy_rank_margin)
    scaled_budget = float(args.budget_loss_weight) * budget_loss
    scaled_proxy_rank = float(args.proxy_rank_weight) * proxy_rank_loss_value
    visual_safety_value, visual_safety_parts = pose_visual_safety_loss(
        logits,
        target,
        pose_offsets,
        visible_weights,
        weight_power=args.visual_safety_weight_power,
        tail_k=args.visual_safety_tail_k,
        tail_margin=args.visual_safety_tail_margin,
        tail_weight=args.visual_safety_tail_weight,
    )
    scaled_visual_safety = float(args.visual_safety_loss_weight) * visual_safety_value
    subpose_robust_value, subpose_robust_parts = pose_subpose_robust_safety_loss(
        logits,
        target,
        pose_offsets,
        visible_weights,
        visible_hit_rates,
        rare_weight=args.subpose_robust_rare_weight,
        frequency_power=args.subpose_robust_frequency_power,
        tail_k=args.subpose_robust_tail_k,
        tail_margin=args.subpose_robust_tail_margin,
        tail_weight=args.subpose_robust_tail_weight,
    )
    scaled_subpose_robust = float(args.subpose_robust_loss_weight) * subpose_robust_value
    loss = set_loss + scaled_rvl + scaled_budget + scaled_proxy_rank + scaled_visual_safety + scaled_subpose_robust
    return loss, {
        **set_parts,
        **rvl_parts,
        **budget_parts,
        **proxy_rank_parts,
        "lossVisibilitySet": float(set_loss.detach().cpu()),
        "lossVisibilityRvlScaled": float(scaled_rvl.detach().cpu()),
        "lossVisibilityBudgetScaled": float(scaled_budget.detach().cpu()),
        "lossVisibilityProxyRankScaled": float(scaled_proxy_rank.detach().cpu()),
        **visual_safety_parts,
        "lossVisibilityVisualSafetyScaled": float(scaled_visual_safety.detach().cpu()),
        **subpose_robust_parts,
        "lossVisibilitySubposeRobustScaled": float(scaled_subpose_robust.detach().cpu()),
        "lossVisibility": float(loss.detach().cpu()),
    }


def apply_loss_profile(args: argparse.Namespace) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {
        "legacy": {},
        "balanced_v2": {
            "set_bce_weight": 0.28,
            "set_tversky_weight": 1.35,
            "set_count_weight": 0.10,
            "set_rank_weight": 0.45,
            "false_negative_weight": 14.0,
            "pose_tversky_alpha": 1.0,
            "pose_tversky_beta": 7.0,
            "rvl_mode": "evidence",
            "rvl_loss_weight": 0.08,
            "rvl_fn_weight": 0.25,
            "rvl_fp_weight": 1.0,
            "budget_loss_weight": 0.18,
            "budget_safety_multiplier": 2.4,
            "proxy_evidence_weight": 0.18,
            "proxy_visible_guard_weight": 0.06,
            "proxy_sparsity_weight": 0.015,
            "proxy_rank_weight": 0.16,
            "utility_loss_weight": 0.18,
            "glb_priority_loss_weight": 0.20,
        },
        "rvl_strong_v2": {
            "set_bce_weight": 0.28,
            "set_tversky_weight": 1.35,
            "set_count_weight": 0.10,
            "set_rank_weight": 0.45,
            "false_negative_weight": 14.0,
            "pose_tversky_alpha": 1.0,
            "pose_tversky_beta": 7.0,
            "rvl_mode": "evidence",
            "rvl_loss_weight": 0.12,
            "rvl_fn_weight": 0.25,
            "rvl_fp_weight": 1.0,
            "budget_loss_weight": 0.18,
            "budget_safety_multiplier": 2.4,
            "proxy_evidence_weight": 0.18,
            "proxy_visible_guard_weight": 0.06,
            "proxy_sparsity_weight": 0.015,
            "proxy_rank_weight": 0.16,
            "utility_loss_weight": 0.18,
            "glb_priority_loss_weight": 0.20,
        },
        "rvl_calibrated_v3": {
            "visibility_loss_mode": "calibrated",
            "set_bce_weight": 0.70,
            "set_tversky_weight": 0.85,
            "set_count_weight": 0.06,
            "set_rank_weight": 0.35,
            "positive_margin_weight": 0.45,
            "hard_negative_margin_weight": 0.55,
            "pos_class_weight": 1.0,
            "neg_class_weight": 1.0,
            "false_negative_weight": 14.0,
            "pose_tversky_alpha": 1.0,
            "pose_tversky_beta": 6.0,
            "calibration_rank_margin": 0.50,
            "positive_logit_margin": 1.50,
            "hard_negative_logit_margin": -4.00,
            "hard_negative_top_k": 1024,
            "rvl_mode": "evidence",
            "rvl_loss_weight": 0.12,
            "rvl_fn_weight": 0.25,
            "rvl_fp_weight": 1.0,
            "budget_loss_weight": 0.30,
            "budget_safety_multiplier": 2.15,
            "proxy_evidence_weight": 0.16,
            "proxy_visible_guard_weight": 0.06,
            "proxy_sparsity_weight": 0.015,
            "proxy_rank_weight": 0.16,
            "utility_loss_weight": 0.18,
            "glb_priority_loss_weight": 0.20,
        },
        "budget_tight_v2": {
            "set_bce_weight": 0.28,
            "set_tversky_weight": 1.30,
            "set_count_weight": 0.08,
            "set_rank_weight": 0.42,
            "false_negative_weight": 13.0,
            "pose_tversky_alpha": 1.0,
            "pose_tversky_beta": 6.5,
            "rvl_mode": "evidence",
            "rvl_loss_weight": 0.10,
            "rvl_fn_weight": 0.25,
            "rvl_fp_weight": 1.0,
            "budget_loss_weight": 0.30,
            "budget_safety_multiplier": 2.15,
            "proxy_evidence_weight": 0.16,
            "proxy_visible_guard_weight": 0.06,
            "proxy_sparsity_weight": 0.015,
            "proxy_rank_weight": 0.16,
            "utility_loss_weight": 0.18,
            "glb_priority_loss_weight": 0.20,
        },
        "proxy_light_v2": {
            "set_bce_weight": 0.30,
            "set_tversky_weight": 1.35,
            "set_count_weight": 0.10,
            "set_rank_weight": 0.45,
            "false_negative_weight": 14.0,
            "pose_tversky_alpha": 1.0,
            "pose_tversky_beta": 7.0,
            "rvl_mode": "evidence",
            "rvl_loss_weight": 0.10,
            "rvl_fn_weight": 0.25,
            "rvl_fp_weight": 1.0,
            "budget_loss_weight": 0.20,
            "budget_safety_multiplier": 2.35,
            "proxy_evidence_weight": 0.10,
            "proxy_visible_guard_weight": 0.04,
            "proxy_sparsity_weight": 0.010,
            "proxy_rank_weight": 0.12,
            "utility_loss_weight": 0.18,
            "glb_priority_loss_weight": 0.20,
        },
        "m5_subpose_robust_v1": {
            "visibility_loss_mode": "calibrated",
            "set_bce_weight": 0.70,
            "set_tversky_weight": 0.85,
            "set_count_weight": 0.06,
            "set_rank_weight": 0.35,
            "positive_margin_weight": 0.45,
            "hard_negative_margin_weight": 0.55,
            "pos_class_weight": 1.0,
            "neg_class_weight": 1.0,
            "pose_tversky_alpha": 1.0,
            "pose_tversky_beta": 6.0,
            "calibration_rank_margin": 0.50,
            "positive_logit_margin": 1.50,
            "hard_negative_logit_margin": -4.00,
            "hard_negative_top_k": 1024,
            "rvl_mode": "evidence",
            "rvl_loss_weight": 0.12,
            "rvl_fn_weight": 0.25,
            "rvl_fp_weight": 1.0,
            "budget_loss_weight": 0.30,
            "budget_safety_multiplier": 2.15,
            "proxy_evidence_weight": 0.16,
            "proxy_visible_guard_weight": 0.06,
            "proxy_sparsity_weight": 0.015,
            "proxy_rank_weight": 0.16,
            "utility_loss_weight": 0.18,
            "glb_priority_loss_weight": 0.20,
            "subpose_robust_loss_weight": 0.35,
            "subpose_robust_rare_weight": 1.0,
            "subpose_robust_frequency_power": 0.5,
            "subpose_robust_tail_k": 8,
            "subpose_robust_tail_margin": 1.5,
            "subpose_robust_tail_weight": 0.5,
        },
    }
    overrides = profiles.get(args.loss_profile, {})
    for key, value in overrides.items():
        setattr(args, key, value)
    return overrides


def resolve_loss_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Apply a named profile while preserving explicit loss-mode overrides."""

    explicit_rvl_mode = getattr(args, "rvl_mode", None)
    overrides = apply_loss_profile(args)
    if explicit_rvl_mode is not None:
        args.rvl_mode = explicit_rvl_mode
    elif getattr(args, "rvl_mode", None) is None:
        args.rvl_mode = "evidence"
    return overrides


def freeze_offline_encoder(model: DirectionalOcclusionProxyEncoderPVSModel) -> dict[str, int]:
    modules = [model.geo_encoder, model.edge_encoder, model.context_head, model.proxy_head]
    frozen = 0
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                frozen += int(param.numel())
            param.requires_grad_(False)
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in model.parameters())
    return {"frozenOfflineParameters": frozen, "trainableParameters": trainable, "totalParameters": total}


def apply_runtime_feature_ablation_loss(args: argparse.Namespace) -> dict[str, float]:
    """Disable auxiliary supervision for branches hidden from the query head."""
    if args.runtime_feature_ablation == "none":
        return {}
    overrides = {
        "proxy_evidence_weight": 0.0,
        "proxy_visible_guard_weight": 0.0,
        "proxy_sparsity_weight": 0.0,
        "proxy_rank_weight": 0.0,
    }
    for key, value in overrides.items():
        setattr(args, key, value)
    return overrides


def save_proxy_runtime_features(
    model: DirectionalOcclusionProxyEncoderPVSModel,
    point_cache: torch.Tensor,
    output_dir: Path,
    glb_meta: dict[str, Any],
    resource_report: dict[str, Any],
    evidence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    geo, context, proxy, runtime = model.compute_all_runtime_features(point_cache, batch_size=args.feature_export_batch_size)
    paths = {
        "instanceGeo": output_dir / "instance_geo_features_fp16.bin",
        "instanceContext": output_dir / "instance_context_features_fp16.bin",
        "instanceOcclusionProxy": output_dir / "instance_occlusion_proxy_features_fp16.bin",
        "instanceRuntime": output_dir / "instance_runtime_features_fp16.bin",
    }
    geo.numpy().astype(np.float16).tofile(paths["instanceGeo"])
    context.numpy().astype(np.float16).tofile(paths["instanceContext"])
    proxy.numpy().astype(np.float16).tofile(paths["instanceOcclusionProxy"])
    runtime.numpy().astype(np.float16).tofile(paths["instanceRuntime"])
    meta = {
        "schema": "directional-occlusion-proxy-features-v1",
        "experiment": args.experiment_name,
        "numInstances": int(runtime.shape[0]),
        "geoDim": int(model.geo_dim),
        "contextDim": int(model.context_dim),
        "proxyDim": int(model.proxy_dim),
        "directionBins": int(model.direction_bins),
        "depthShells": int(model.depth_shells),
        "runtimeFeatureDim": int(model.runtime_feature_dim),
        "dtype": "float16",
        "runtimeFeatureAblation": model.runtime_feature_ablation,
        "runtimeFeatureAblationSemantics": (
            "full fixed table is exported for schema/byte accounting; the named branches are "
            "zeroed before the runtime query head for registered M4 input ablations"
        ),
        "geoSource": "GLB point cloud plus instance AABB attributes, encoded offline.",
        "contextSource": "Directional evidence graph encoder context head.",
        "proxySource": "Directional evidence graph encoder occlusion proxy head.",
        "runtimeUsesPointNet": False,
        "runtimeUsesGraphPropagation": False,
        "evidenceMeta": evidence_meta,
        "glbPointsMeta": glb_meta,
        "resources": resource_report,
        "files": {key: path.name for key, path in paths.items()},
    }
    (output_dir / "instance_features_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pvs_directional_occlusion_proxy_encoder.")
    parser.add_argument("--dataset-dir", default="neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66")
    parser.add_argument("--evidence-dir", default="neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_fov66")
    parser.add_argument("--glb-points", default="neural_instance_culling/dataset/out/glb_points_v3.bin")
    parser.add_argument("--runtime-meta", default="hkust-v3/assets/runtimeVisibilityMeta.json")
    parser.add_argument("--glb-index", default="hkust-v3/assets/glbIndex.json")
    parser.add_argument("--glb-root", default="hkust-v3/assets")
    parser.add_argument("--output-dir", default="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66")
    parser.add_argument("--experiment-name", default="pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--steps-per-epoch", type=int, default=900)
    parser.add_argument("--pose-set-batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument(
        "--eval-pose-steps",
        type=int,
        default=0,
        help="Deprecated compatibility option. Formal validation/calibration always evaluate all poses; zero means all.",
    )
    parser.add_argument("--max-candidates-per-pose", type=int, default=0)
    parser.add_argument("--glb-train-points", type=int, default=64)
    parser.add_argument("--geo-dim", type=int, default=96)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--proxy-dim", type=int, default=8)
    parser.add_argument("--point-hidden-dim", type=int, default=160)
    parser.add_argument("--pointnetpp-centers", type=int, default=24)
    parser.add_argument("--pointnetpp-neighbors", type=int, default=12)
    parser.add_argument("--graph-hidden-dim", type=int, default=160)
    parser.add_argument("--graph-message-dim", type=int, default=96)
    parser.add_argument("--ray-fourier-bands", type=int, default=10)
    parser.add_argument("--ray-scalar-fourier-bands", type=int, default=4)
    parser.add_argument(
        "--camera-location-dim",
        type=int,
        default=0,
        help="Optional low-dimensional normalized scene-location auxiliary feature appended after ray-space features.",
    )
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--interaction-dim", type=int, default=64)
    parser.add_argument(
        "--runtime-feature-ablation",
        choices=["none", "proxy_zero", "context_proxy_zero", "geo_context_proxy_zero"],
        default="none",
        help=(
            "Registered M4 input ablation. proxy_zero is geometry+context+ray; "
            "context_proxy_zero is geometry+ray; geo_context_proxy_zero is AABB+ray."
        ),
    )
    parser.add_argument(
        "--disable-explicit-inhibition",
        action="store_true",
        help="Registered M4 control: keep the directional proxy input but replace the learned inhibition output with zero.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--loss-profile",
        choices=["legacy", "balanced_v2", "rvl_strong_v2", "rvl_calibrated_v3", "budget_tight_v2", "proxy_light_v2", "m5_subpose_robust_v1"],
        default="balanced_v2",
        help="Named loss preset. Use legacy to keep raw CLI weights unchanged.",
    )
    parser.add_argument("--false-negative-weight", type=float, default=16.0)
    parser.add_argument("--set-bce-weight", type=float, default=0.3)
    parser.add_argument("--set-tversky-weight", type=float, default=1.4)
    parser.add_argument("--set-count-weight", type=float, default=0.12)
    parser.add_argument("--set-rank-weight", type=float, default=0.5)
    parser.add_argument(
        "--visibility-loss-mode",
        choices=["set_importance", "calibrated"],
        default="set_importance",
        help="Visibility score objective. calibrated explicitly separates positive and hard-negative logit ranges.",
    )
    parser.add_argument("--pose-tversky-alpha", type=float, default=1.0)
    parser.add_argument("--pose-tversky-beta", type=float, default=8.0)
    parser.add_argument("--positive-margin-weight", type=float, default=0.55)
    parser.add_argument("--hard-negative-margin-weight", type=float, default=0.20)
    parser.add_argument("--pos-class-weight", type=float, default=2.0)
    parser.add_argument("--neg-class-weight", type=float, default=1.0)
    parser.add_argument("--calibration-rank-margin", type=float, default=0.5)
    parser.add_argument("--positive-logit-margin", type=float, default=1.0)
    parser.add_argument("--hard-negative-logit-margin", type=float, default=-1.0)
    parser.add_argument("--hard-negative-top-k", type=int, default=512)
    parser.add_argument("--importance-weight-scale", type=float, default=2.5)
    parser.add_argument(
        "--visual-safety-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the pose-level soft missed-coverage loss; zero keeps the registered baseline.",
    )
    parser.add_argument(
        "--visual-safety-weight-power",
        type=float,
        default=1.0,
        help="Power applied to positive visible weights before per-pose normalization.",
    )
    parser.add_argument(
        "--visual-safety-tail-k",
        type=int,
        default=8,
        help="Number of highest-weight visible positives receiving the visual safety margin.",
    )
    parser.add_argument(
        "--visual-safety-tail-margin",
        type=float,
        default=1.0,
        help="Target positive logit margin for the highest-weight visible positives.",
    )
    parser.add_argument(
        "--visual-safety-tail-weight",
        type=float,
        default=0.5,
        help="Relative weight of the high-contribution positive margin inside the visual safety loss.",
    )
    parser.add_argument(
        "--subpose-robust-loss-weight",
        type=float,
        default=0.0,
        help="Weight for dense view-cell subpose robust safety loss; zero preserves the registered baseline.",
    )
    parser.add_argument("--subpose-robust-rare-weight", type=float, default=1.0)
    parser.add_argument("--subpose-robust-frequency-power", type=float, default=0.5)
    parser.add_argument("--subpose-robust-tail-k", type=int, default=8)
    parser.add_argument("--subpose-robust-tail-margin", type=float, default=1.5)
    parser.add_argument("--subpose-robust-tail-weight", type=float, default=0.5)
    parser.add_argument("--proxy-evidence-weight", type=float, default=0.25)
    parser.add_argument("--proxy-visible-guard-weight", type=float, default=0.08)
    parser.add_argument("--proxy-sparsity-weight", type=float, default=0.02)
    parser.add_argument("--budget-loss-weight", type=float, default=0.25)
    parser.add_argument("--budget-safety-multiplier", type=float, default=2.5)
    parser.add_argument("--proxy-rank-weight", type=float, default=0.20)
    parser.add_argument("--proxy-rank-margin", type=float, default=0.30)
    parser.add_argument("--utility-loss-weight", type=float, default=0.20)
    parser.add_argument("--glb-priority-loss-weight", type=float, default=0.20)
    parser.add_argument(
        "--rvl-mode",
        choices=["off", "uniform", "evidence", "cost_aware"],
        default=None,
        help="Override the loss profile's RVL mode; omitted uses the selected profile default.",
    )
    parser.add_argument("--rvl-loss-weight", type=float, default=0.08)
    parser.add_argument("--rvl-fn-weight", type=float, default=0.25)
    parser.add_argument("--rvl-fp-weight", type=float, default=1.0)
    parser.add_argument("--rvl-evidence-scale", type=float, default=1.0)
    parser.add_argument("--rvl-cost-scale", type=float, default=0.25)
    parser.add_argument(
        "--contrastive-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the training-only pose-local hard-negative contrastive objective.",
    )
    parser.add_argument("--contrastive-projection-dim", type=int, default=32)
    parser.add_argument("--contrastive-hidden-dim", type=int, default=64)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--contrastive-positive-top-k", type=int, default=64)
    parser.add_argument("--contrastive-negative-top-k", type=int, default=256)
    parser.add_argument("--contrastive-importance-scale", type=float, default=2.5)
    parser.add_argument("--reg-weight", type=float, default=1e-5)
    parser.add_argument("--feature-export-batch-size", type=int, default=512)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--target-weighted-recall", type=float, default=0.99)
    parser.add_argument(
        "--calibration-pose-recall-floor",
        type=float,
        default=0.95,
        help="Minimum ordinary pose recall required for a calibration safety workpoint.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.10,
        help="Fraction of the original training poses held out as an independent threshold-calibration split.",
    )
    parser.add_argument(
        "--calibration-point-floor",
        type=float,
        default=0.9925,
        help="Pre-registered calibration weighted-recall point-estimate floor.",
    )
    parser.add_argument(
        "--calibration-lcb-floor",
        type=float,
        default=0.99,
        help="One-sided view-cell bootstrap lower-bound floor used in the final calibration pass.",
    )
    parser.add_argument(
        "--calibration-bootstrap-replicates",
        type=int,
        default=10000,
        help="Bootstrap replicates for the final calibration pass; intermediate checkpoint evaluations use point estimates only.",
    )
    parser.add_argument("--calibration-bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--init-checkpoint", default="", help="Optional checkpoint used to initialize model weights before training.")
    parser.add_argument(
        "--allow-scene-transfer",
        action="store_true",
        help="Load only shape-compatible learnable parameters from --init-checkpoint and rebuild target-scene buffers.",
    )
    parser.add_argument(
        "--train-pose-fraction",
        type=float,
        default=1.0,
        help="Deterministic fraction of native train poses for few-shot adaptation; validation/calibration/test are unchanged.",
    )
    parser.add_argument("--fixed-runtime-features", default="", help="Optional exported full-instance runtime feature table used during training/eval instead of recomputing offline encoders.")
    parser.add_argument("--freeze-offline-encoder", action="store_true", help="Freeze PointNet++ geo encoder and context/proxy evidence encoder; train only runtime query heads.")
    parser.add_argument(
        "--allow-invalid-resource-semantics",
        action="store_true",
        help="Exploratory-only escape hatch for legacy point caches or post-union candidate files; formal runs must leave this disabled.",
    )
    parser.add_argument(
        "--export-eval-checkpoint",
        default="",
        help="Load this checkpoint, export runtime features, calibrate, and optionally run the one-shot test.",
    )
    parser.add_argument("--eval-summary-name", default="eval_summary.json")
    parser.add_argument("--checkpoint-alias", default="", help="Optional checkpoint copy written under output-dir before export/eval.")
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Export the checkpoint and final calibration provenance without reading test; use evaluate_frozen_test.py for the one-shot test.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not 0.0 < float(args.train_pose_fraction) <= 1.0:
        parser.error("--train-pose-fraction must be in (0, 1]")
    if args.allow_scene_transfer and not args.init_checkpoint:
        parser.error("--allow-scene-transfer requires --init-checkpoint")
    if args.contrastive_loss_weight < 0:
        parser.error("--contrastive-loss-weight must be non-negative")
    if args.contrastive_projection_dim <= 0 or args.contrastive_hidden_dim <= 0:
        parser.error("contrastive projection dimensions must be positive")
    if args.contrastive_temperature <= 0:
        parser.error("--contrastive-temperature must be positive")
    loss_profile_overrides = resolve_loss_profile(args)
    runtime_ablation_loss_overrides = apply_runtime_feature_ablation_loss(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(args.runtime_meta)
    scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    num_glbs = int(instance_to_glb.max()) + 1
    resource_report = validate_training_resources(
        Path(args.dataset_dir),
        Path(args.runtime_meta),
        Path(args.glb_points),
        world_aabbs.shape[0],
        num_glbs,
        strict_semantics=not args.allow_invalid_resource_semantics,
    )
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=world_aabbs.shape[0])
    if args.subpose_robust_loss_weight > 0 and not dataset.has_subpose_robust_labels:
        raise ValueError(
            "subpose-robust supervision requires visible_hit_counts.bin and subpose_offsets.bin; "
            "refusing to silently train without dense view-cell labels"
        )
    train_split, val_split, calibration_split, test_split, protocol_split = build_protocol_splits(
        dataset,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    train_split, protocol_split = apply_train_pose_fraction(
        dataset,
        train_split,
        protocol_split,
        args.train_pose_fraction,
        args.seed,
    )
    camera_bounds = dataset.meta.get("cameraBounds", runtime_meta["sceneBounds"])
    point_np, glb_meta = load_glb_points(args.glb_points, max_points=args.glb_train_points)
    if int(point_np.shape[0]) < int(num_glbs):
        raise ValueError(
            f"GLB point cache contains {point_np.shape[0]} rows, runtime requires {num_glbs} unique GLBs"
        )
    source_ids_np, source_scores_np, strength_np, evidence_meta = load_directional_occlusion_evidence(args.evidence_dir, world_aabbs.shape[0])
    if int(evidence_meta["directionBins"]) <= 0 or int(evidence_meta["depthShells"]) <= 0:
        raise ValueError("Invalid evidence metadata.")
    point_cache = torch.from_numpy(point_np).to(device)
    model = DirectionalOcclusionProxyEncoderPVSModel(
        num_instances=world_aabbs.shape[0],
        num_glbs=num_glbs,
        geo_dim=args.geo_dim,
        context_dim=args.context_dim,
        proxy_dim=args.proxy_dim,
        direction_bins=int(evidence_meta["directionBins"]),
        depth_shells=int(evidence_meta["depthShells"]),
        source_k=int(evidence_meta["sourceK"]),
        point_hidden_dim=args.point_hidden_dim,
        pointnetpp_centers=args.pointnetpp_centers,
        pointnetpp_neighbors=args.pointnetpp_neighbors,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_message_dim=args.graph_message_dim,
        ray_fourier_bands=args.ray_fourier_bands,
        ray_scalar_fourier_bands=args.ray_scalar_fourier_bands,
        camera_location_dim=args.camera_location_dim,
        mlp_hidden=args.mlp_hidden,
        interaction_dim=args.interaction_dim,
        scene_size_m=scene_size.tolist(),
        runtime_feature_ablation=args.runtime_feature_ablation,
        use_explicit_inhibition=not args.disable_explicit_inhibition,
    ).to(device)
    model.set_scene_bounds(torch.from_numpy(scene_min).to(device), torch.from_numpy(scene_size).to(device))
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    model.set_evidence(
        torch.from_numpy(source_ids_np).to(device),
        torch.from_numpy(source_scores_np).to(device),
        torch.from_numpy(strength_np).to(device),
    )
    contrastive_projector = None
    if args.contrastive_loss_weight > 0:
        contrastive_projector = PoseContrastiveProjectionHead(
            input_dim=model.query_feature_dim + model.camera_feature_dim,
            hidden_dim=args.contrastive_hidden_dim,
            output_dim=args.contrastive_projection_dim,
        ).to(device)
    init_checkpoint_meta: dict[str, Any] | None = None
    if args.init_checkpoint:
        init_checkpoint_path = Path(args.init_checkpoint)
        init_checkpoint_meta = load_initial_checkpoint(
            model,
            init_checkpoint_path,
            allow_scene_transfer=bool(args.allow_scene_transfer),
        )
    freeze_report = freeze_offline_encoder(model) if args.freeze_offline_encoder else None
    fixed_runtime_features = None
    fixed_runtime_meta = None
    if args.fixed_runtime_features:
        fixed_runtime_path = Path(args.fixed_runtime_features)
        data = np.fromfile(fixed_runtime_path, dtype=np.float16)
        expected = int(world_aabbs.shape[0]) * int(model.runtime_feature_dim)
        if data.size != expected:
            raise ValueError(f"{fixed_runtime_path} has {data.size} fp16 values, expected {expected}")
        fixed_runtime_features = torch.from_numpy(data.reshape(world_aabbs.shape[0], model.runtime_feature_dim).astype(np.float32)).to(device)
        fixed_runtime_meta = {
            "path": str(fixed_runtime_path.as_posix()),
            "numInstances": int(world_aabbs.shape[0]),
            "runtimeFeatureDim": int(model.runtime_feature_dim),
            "dtype": "float16",
            "semantics": "fixed view-independent instance feature table; training updates only query/output heads when offline encoder is frozen",
        }
    glb_cost_norm = torch.from_numpy(
        load_glb_costs(
            args.glb_index,
            args.glb_root,
            num_glbs,
            instance_to_glb,
            allow_missing=args.allow_invalid_resource_semantics,
        )
    ).to(device)

    model_meta_payload = {
        "schema": "directional-occlusion-proxy-encoder-train-v1",
        "experiment": args.experiment_name,
        "route": "offline heavy point encoder with context head and directional occlusion proxy head; runtime uses fixed features and ray query only",
        "args": vars(args),
        "modelConfig": model.config,
        "lossProfile": args.loss_profile,
        "lossProfileOverrides": loss_profile_overrides,
        "runtimeFeatureAblationLossOverrides": runtime_ablation_loss_overrides,
        "initCheckpoint": init_checkpoint_meta,
        "freezeOfflineEncoder": freeze_report,
        "fixedRuntimeFeatures": fixed_runtime_meta,
        "contrastiveTraining": {
            "enabled": contrastive_projector is not None,
            "runtimeExported": False,
            "semantics": (
                "training-only pose-local separation of important visible queries and hard invisible queries; "
                "the projection head is omitted from runtime assets"
            ),
        },
        "datasetMeta": dataset.meta,
        "protocolSplit": protocol_split,
        "evidenceMeta": evidence_meta,
        "resources": resource_report,
        "dataSemantics": {
            "visibleIds": "pose-level GT visible instance set",
            "visibleWeights": dataset.visible_weight_semantics,
            "subposeRobustLabels": dataset.subpose_robust_label_semantics,
            "occlusionEvidence": "projection-geometry weak supervision from visible sources and invisible candidate targets; no dynamic-pool teacher",
        },
        "imageEvaluation": {
            "requiredForFormalReport": True,
            "script": "neural_instance_culling/benchmark/evaluate_viewcell_image_per.py",
        },
    }
    (output_dir / "model_meta.json").write_text(
        json.dumps(model_meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "protocol_split.json").write_text(
        json.dumps(protocol_split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.export_eval_checkpoint:
        checkpoint_path = Path(args.export_eval_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        if args.checkpoint_alias:
            alias_path = output_dir / args.checkpoint_alias
            shutil.copy2(checkpoint_path, alias_path)
        feature_meta = save_proxy_runtime_features(model, point_cache, output_dir, glb_meta, resource_report, evidence_meta, args)
        runtime_np = np.fromfile(output_dir / "instance_runtime_features_fp16.bin", dtype=np.float16).reshape(
            world_aabbs.shape[0],
            model.runtime_feature_dim,
        )
        runtime_features = torch.from_numpy(runtime_np.astype(np.float32)).to(device)
        calibration_workpoint, validation_workpoint, calibration_rows, calibration_summary = evaluate_calibration_and_validation(
            model,
            val_split,
            calibration_split,
            runtime_features,
            world_aabbs,
            device,
            poses_per_batch=max(1, min(args.pose_set_batch_size, 4)),
            max_candidates_per_pose=args.max_candidates_per_pose,
            seed=args.seed + 100,
            target_weighted_recall=args.target_weighted_recall,
            calibration_pose_recall_floor=args.calibration_pose_recall_floor,
            calibration_point_floor=args.calibration_point_floor,
            calibration_lcb_floor=args.calibration_lcb_floor,
            calibration_bootstrap_replicates=args.calibration_bootstrap_replicates,
            calibration_bootstrap_confidence=args.calibration_bootstrap_confidence,
            allow_candidate_visible_union=args.allow_invalid_resource_semantics,
        )
        if args.checkpoint_alias:
            # Keep checkpoint selection (which used the complete validation
            # history) separate from the final calibration record.  The
            # latter may choose a different threshold after the registered
            # bootstrap safety margin is applied.
            checkpoint_selection = dict(checkpoint.get("best") or {})
            checkpoint["checkpointSelection"] = checkpoint_selection
            checkpoint["best"] = {
                "epoch": int(checkpoint_selection.get("epoch", checkpoint.get("epoch", 0))),
                "threshold": float(calibration_workpoint["threshold"]),
                **validation_workpoint,
                "selectionMetrics": checkpoint_selection,
            }
            checkpoint["workpoints"] = {
                "calibration": calibration_summary,
                "validationAtFrozenThreshold": validation_workpoint,
                "frozenThreshold": float(calibration_workpoint["threshold"]),
                "selectionRule": weighted_precision_selection_rule(
                    args.target_weighted_recall,
                    minimum_point_estimate=args.calibration_point_floor,
                    minimum_lower_confidence_bound=args.calibration_lcb_floor,
                    minimum_pose_recall=args.calibration_pose_recall_floor,
                ),
            }
            checkpoint["finalCalibration"] = calibration_summary
            torch.save(checkpoint, output_dir / args.checkpoint_alias)
        if args.skip_final_test:
            if not args.checkpoint_alias:
                raise RuntimeError(
                    "--skip-final-test with --export-eval-checkpoint requires --checkpoint-alias "
                    "so the independent calibration bundle has a checkpoint."
                )
            calibration_ready = {
                "protocol": "calibration_ready_pre_test",
                "frozenThreshold": float(calibration_workpoint["threshold"]),
                "calibration": calibration_summary,
                "validationAtFrozenThreshold": validation_workpoint,
                "calibrationThresholdRows": calibration_rows,
                "testEvaluationCount": 0,
                "protocolSplit": protocol_split,
                "args": vars(args),
                "featureMeta": feature_meta,
                "selectionRule": weighted_precision_selection_rule(
                    args.target_weighted_recall,
                    minimum_point_estimate=args.calibration_point_floor,
                    minimum_lower_confidence_bound=args.calibration_lcb_floor,
                    minimum_pose_recall=args.calibration_pose_recall_floor,
                ),
                "testPolicy": (
                    "Use evaluate_frozen_test.py prepare/evaluate exactly once; "
                    "this file contains no test result."
                ),
                "provenance": {
                    "sourceCheckpoint": str(checkpoint_path.resolve()),
                    "mode": "independent calibration-only export",
                    "testRead": False,
                    "candidateSetChanged": False,
                    "gtChanged": False,
                },
            }
            ready_path = output_dir / "calibration_ready_summary.json"
            ready_path.write_text(json.dumps(calibration_ready, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "outputDir": str(output_dir),
                        "calibrationReadySummary": str(ready_path),
                        "frozenThreshold": float(calibration_workpoint["threshold"]),
                        "testEvaluationCount": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            return
        frozen_threshold = float(calibration_workpoint["threshold"])
        test_row = evaluate_thresholds(
            model,
            test_split,
            runtime_features,
            world_aabbs,
            device,
            poses_per_batch=max(1, min(args.pose_set_batch_size, 4)),
            max_steps=None,
            max_candidates_per_pose=args.max_candidates_per_pose,
            seed=args.seed + 999,
            thresholds=np.asarray([frozen_threshold], dtype=np.float32),
            allow_candidate_visible_union=args.allow_invalid_resource_semantics,
        )[0]
        summary = {
            "checkpoint": str(checkpoint_path.as_posix()),
            "checkpointAlias": args.checkpoint_alias or None,
            "protocol": "frozen_calibration_one_shot_test",
            "best": test_row,
            "frozenThreshold": frozen_threshold,
            "calibration": calibration_summary,
            "validationAtFrozenThreshold": validation_workpoint,
            "test": test_row,
            "testEvaluationCount": 1,
            "protocolSplit": protocol_split,
            "args": vars(args),
            "featureMeta": feature_meta,
            "dataSemantics": {
                "visibleIds": "pose-level GT visible instance set",
                "visibleWeights": dataset.visible_weight_semantics,
                "occlusionEvidence": "derived from GT visibility and current-view projection geometry; no teacher model",
            },
            "imageEvaluation": {
                "status": "not_run",
                "requiredForFormalReport": True,
                "script": "neural_instance_culling/benchmark/evaluate_viewcell_image_per.py",
            },
            "selectionRule": weighted_precision_selection_rule(
                args.target_weighted_recall,
                minimum_point_estimate=args.calibration_point_floor,
                minimum_lower_confidence_bound=args.calibration_lcb_floor,
                minimum_pose_recall=args.calibration_pose_recall_floor,
            ),
        }
        summary_path = output_dir / args.eval_summary_name
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "best": summary["best"],
                    "summary": str(summary_path),
                    "outputDir": str(output_dir),
                    "imageEvaluation": summary["imageEvaluation"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if contrastive_projector is not None:
        trainable_params.extend(contrastive_projector.parameters())
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying freeze options.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    steps_per_epoch = int(args.steps_per_epoch or math.ceil(train_split.pose_indices_with_visible.size / max(1, args.pose_set_batch_size)))
    best_selection_key: tuple[float, float, float, float, float] | None = None
    best_saved = False
    history: list[dict[str, Any]] = []
    metrics_log = output_dir / "train_metrics.jsonl"
    if metrics_log.exists():
        metrics_log.unlink()

    for epoch in range(args.epochs):
        model.train()
        if contrastive_projector is not None:
            contrastive_projector.train()
        losses = []
        loss_parts: dict[str, float] = {}
        skipped_nonfinite_loss = 0
        skipped_nonfinite_grad = 0
        last_grad_norm = 0.0
        for pose_indices in tqdm(
            train_split.pose_set_batches(args.pose_set_batch_size, rng, steps_per_epoch),
            total=steps_per_epoch,
            desc=f"{args.experiment_name} {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            batch = train_split.build_pose_set_batch(
                pose_indices,
                world_aabbs,
                rng,
                max_candidates_per_pose=args.max_candidates_per_pose,
                allow_candidate_visible_union=args.allow_invalid_resource_semantics,
            )
            if batch["instance"].size == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=bool(args.amp and device.type == "cuda")):
                ids = torch.from_numpy(batch["instance"]).to(device)
                camera = torch.from_numpy(batch["camera"]).to(device)
                camera_world = torch.from_numpy(batch["camera_world"]).to(device)
                view = torch.from_numpy(batch["camera_view"]).to(device)
                offsets = torch.from_numpy(batch["pose_offsets"]).to(device)
                target = torch.from_numpy(batch["target"][:, None]).to(device)
                visible_weights = torch.from_numpy(batch["visible_weights"][:, None]).to(device)
                visible_hit_rates = torch.from_numpy(batch["visible_hit_rates"][:, None]).to(device)
                logits, aux = model.compute_logits_with_aux(
                    camera,
                    view,
                    camera_world,
                    ids,
                    glb_points=point_cache if fixed_runtime_features is None else None,
                    runtime_features=fixed_runtime_features,
                )
                visibility_loss, visibility_parts = visibility_supervision_loss(
                    logits,
                    target,
                    offsets,
                    visible_weights,
                    visible_hit_rates,
                    aux["evidence_target"],
                    ids,
                    model.instance_to_glb,
                    glb_cost_norm,
                    args,
                )
                proxy_loss, proxy_parts = proxy_evidence_loss(
                    aux,
                    target,
                    evidence_weight=args.proxy_evidence_weight,
                    visible_guard_weight=args.proxy_visible_guard_weight,
                    sparsity_weight=args.proxy_sparsity_weight,
                )
                utility_loss, utility_parts = visual_utility_loss(
                    aux,
                    target,
                    visible_weights,
                    offsets,
                    invisible_weight=0.25,
                    rank_weight=0.3,
                    rank_margin=0.1,
                    rank_positive_top_k=32,
                    rank_negative_top_k=128,
                )
                glb_loss, glb_parts = glb_priority_loss(
                    aux["download_logits"],
                    ids,
                    target,
                    visible_weights,
                    model.instance_to_glb,
                    glb_cost_norm,
                    offsets,
                )
                if contrastive_projector is not None:
                    contrastive_input = torch.cat(
                        [aux["query_features_for_ids"], aux["camera_features_for_ids"]],
                        dim=-1,
                    )
                    contrastive_embeddings = contrastive_projector(contrastive_input)
                    contrastive_loss, contrastive_parts = pose_hard_negative_contrastive_loss(
                        contrastive_embeddings,
                        logits,
                        target,
                        offsets,
                        visible_weights,
                        temperature=args.contrastive_temperature,
                        positive_top_k=args.contrastive_positive_top_k,
                        negative_top_k=args.contrastive_negative_top_k,
                        importance_scale=args.contrastive_importance_scale,
                    )
                else:
                    contrastive_loss = logits.sum() * 0.0
                    contrastive_parts = {
                        "lossContrastive": 0.0,
                        "contrastiveAnchorCount": 0.0,
                        "contrastiveHardNegativeCount": 0.0,
                        "contrastivePositiveSimilarity": 0.0,
                        "contrastiveHardNegativeSimilarity": 0.0,
                    }
                contrastive_parts["lossContrastiveScaled"] = float(
                    (float(args.contrastive_loss_weight) * contrastive_loss).detach().cpu()
                )
                loss = (
                    visibility_loss
                    + proxy_loss
                    + float(args.utility_loss_weight) * utility_loss
                    + float(args.glb_priority_loss_weight) * glb_loss
                    + float(args.contrastive_loss_weight) * contrastive_loss
                    + float(args.reg_weight) * model.regularization()
                )
            if not torch.isfinite(loss):
                skipped_nonfinite_loss += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            if not torch.isfinite(grad_norm):
                skipped_nonfinite_grad += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            last_grad_norm = float(grad_norm.detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            loss_parts = {
                **visibility_parts,
                **proxy_parts,
                **utility_parts,
                **glb_parts,
                **contrastive_parts,
            }

        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "lr": float(scheduler.get_last_lr()[0]),
            "stepsPerEpoch": int(steps_per_epoch),
            "trainSkippedNonFiniteLoss": int(skipped_nonfinite_loss),
            "trainSkippedNonFiniteGrad": int(skipped_nonfinite_grad),
            "trainLastGradNorm": float(last_grad_norm),
            **{f"train_{k}": v for k, v in loss_parts.items()},
        }
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            if fixed_runtime_features is None:
                _geo, _context, _proxy, eval_runtime = model.compute_all_runtime_features(point_cache, batch_size=args.feature_export_batch_size)
                eval_runtime = eval_runtime.to(device)
            else:
                eval_runtime = fixed_runtime_features
            calibration_workpoint, validation_workpoint, _calibration_rows, calibration_summary = evaluate_calibration_and_validation(
                model,
                val_split,
                calibration_split,
                eval_runtime,
                world_aabbs,
                device,
                poses_per_batch=max(1, min(args.pose_set_batch_size, 4)),
                max_candidates_per_pose=args.max_candidates_per_pose,
                seed=args.seed + 100,
                target_weighted_recall=args.target_weighted_recall,
                calibration_pose_recall_floor=args.calibration_pose_recall_floor,
                calibration_point_floor=args.calibration_point_floor,
                calibration_lcb_floor=None,
                calibration_bootstrap_replicates=0,
                calibration_bootstrap_confidence=args.calibration_bootstrap_confidence,
                allow_candidate_visible_union=args.allow_invalid_resource_semantics,
                require_safe_workpoint=False,
            )
            metrics = validation_workpoint
            workpoints = {
                "calibration": calibration_summary,
                "validationAtCalibrationThreshold": validation_workpoint,
                "thresholdUsedForValidation": float(validation_workpoint["threshold"]),
                "selectionStatus": calibration_summary["selectionStatus"],
                "selectionRule": weighted_precision_selection_rule(
                    args.target_weighted_recall,
                    minimum_point_estimate=args.calibration_point_floor,
                    minimum_pose_recall=args.calibration_pose_recall_floor,
                ),
            }
            if calibration_summary["selectionStatus"] == "safe":
                workpoints["frozenThreshold"] = float(calibration_workpoint["threshold"])
                workpoints["validationAtFrozenThreshold"] = validation_workpoint
            row.update({"val": metrics, "valWorkpoints": workpoints})
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "contrastiveProjector": (
                        contrastive_projector.state_dict() if contrastive_projector is not None else None
                    ),
                    "config": model.config,
                    "sceneBounds": runtime_meta["sceneBounds"],
                    "cameraBounds": camera_bounds,
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "val": metrics,
                    "workpoints": workpoints,
                    "protocolSplit": protocol_split,
                    "resources": resource_report,
                    "glbMeta": glb_meta,
                    "evidenceMeta": evidence_meta,
                },
                output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
            )
            calibration_safe = (
                calibration_summary["selectionStatus"] == "safe"
                and calibration_workpoint is not None
            )
            selection_key = relative_checkpoint_selection_key(
                metrics,
                args.target_weighted_recall,
                calibration_safe=calibration_safe,
            )
            if best_selection_key is None or selection_key > best_selection_key:
                best_selection_key = selection_key
                best_saved = True
                checkpoint_safe = bool(selection_key[0] > 0.5)
                best = {
                    "epoch": epoch + 1,
                    "threshold": float(metrics["threshold"]),
                    "selectionStatus": "safe" if checkpoint_safe else "unsafe_relative_best",
                    **metrics,
                }
                torch.save(
                    {
                        "model": model.state_dict(),
                        "contrastiveProjector": (
                            contrastive_projector.state_dict() if contrastive_projector is not None else None
                        ),
                        "config": model.config,
                        "sceneBounds": runtime_meta["sceneBounds"],
                        "cameraBounds": camera_bounds,
                        "args": vars(args),
                        "best": best,
                        "workpoints": workpoints,
                        "protocolSplit": protocol_split,
                        "resources": resource_report,
                        "glbMeta": glb_meta,
                        "evidenceMeta": evidence_meta,
                    },
                    output_dir / "best.pt",
                )
        else:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        history.append(row)
        with metrics_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "contrastiveProjector": (
                    contrastive_projector.state_dict() if contrastive_projector is not None else None
                ),
                "config": model.config,
                "sceneBounds": runtime_meta["sceneBounds"],
                "cameraBounds": camera_bounds,
                "args": vars(args),
                "protocolSplit": protocol_split,
                "resources": resource_report,
                "glbMeta": glb_meta,
                "evidenceMeta": evidence_meta,
            },
            output_dir / "last.pt",
        )
        (output_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    if not best_saved:
        raise RuntimeError(
            "Training completed without an evaluable checkpoint; best.pt was not written."
        )

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    feature_meta = save_proxy_runtime_features(model, point_cache, output_dir, glb_meta, resource_report, evidence_meta, args)
    runtime_np = np.fromfile(output_dir / "instance_runtime_features_fp16.bin", dtype=np.float16).reshape(world_aabbs.shape[0], model.runtime_feature_dim)
    runtime_features = torch.from_numpy(runtime_np.astype(np.float32)).to(device)
    calibration_workpoint, validation_workpoint, calibration_rows, calibration_summary = evaluate_calibration_and_validation(
        model,
        val_split,
        calibration_split,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=max(1, min(args.pose_set_batch_size, 4)),
        max_candidates_per_pose=args.max_candidates_per_pose,
        seed=args.seed + 100,
        target_weighted_recall=args.target_weighted_recall,
        calibration_pose_recall_floor=args.calibration_pose_recall_floor,
        calibration_point_floor=args.calibration_point_floor,
        calibration_lcb_floor=args.calibration_lcb_floor,
        calibration_bootstrap_replicates=args.calibration_bootstrap_replicates,
        calibration_bootstrap_confidence=args.calibration_bootstrap_confidence,
        allow_candidate_visible_union=args.allow_invalid_resource_semantics,
        require_safe_workpoint=False,
    )
    if calibration_workpoint is None:
        checkpoint_selection = dict(checkpoint.get("best") or {})
        checkpoint["checkpointSelection"] = checkpoint_selection
        checkpoint["workpoints"] = {
            "calibration": calibration_summary,
            "validationAtDiagnosticThreshold": validation_workpoint,
            "frozenThreshold": None,
            "selectionStatus": "no_safe_calibration_workpoint",
            "selectionRule": weighted_precision_selection_rule(
                args.target_weighted_recall,
                minimum_point_estimate=args.calibration_point_floor,
                minimum_lower_confidence_bound=args.calibration_lcb_floor,
                minimum_pose_recall=args.calibration_pose_recall_floor,
            ),
        }
        checkpoint["finalCalibration"] = calibration_summary
        torch.save(checkpoint, output_dir / "best.pt")
        unsafe_summary = {
            "protocol": "training_complete_no_safe_calibration_workpoint",
            "selectionStatus": "no_safe_calibration_workpoint",
            "frozenThreshold": None,
            "calibration": calibration_summary,
            "validationAtDiagnosticThreshold": validation_workpoint,
            "calibrationThresholdRows": calibration_rows,
            "testEvaluationCount": 0,
            "protocolSplit": protocol_split,
            "args": vars(args),
            "featureMeta": feature_meta,
            "checkpointSelection": checkpoint_selection,
            "testPolicy": "No test inference is allowed because calibration did not produce a safe frozen threshold.",
        }
        unsafe_path = output_dir / "training_complete_unsafe_summary.json"
        unsafe_path.write_text(json.dumps(unsafe_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "outputDir": str(output_dir),
                    "trainingCompleteUnsafeSummary": str(unsafe_path),
                    "selectionStatus": "no_safe_calibration_workpoint",
                    "testEvaluationCount": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if not isinstance(validation_workpoint, dict) or "threshold" not in validation_workpoint:
        raise RuntimeError(
            "Final calibration returned an invalid validation workpoint; "
            "refusing to export an ambiguous threshold."
        )
    frozen_threshold = float(calibration_workpoint["threshold"])
    if args.skip_final_test:
        # Keep the validation-selected epoch separate from the final
        # calibration threshold.  The strict frozen-test entry point will
        # consume this checkpoint and perform the only test inference.
        checkpoint_selection = dict(checkpoint.get("best") or {})
        checkpoint["checkpointSelection"] = checkpoint_selection
        checkpoint["best"] = {
            "epoch": int(checkpoint_selection.get("epoch", checkpoint.get("epoch", 0))),
            "threshold": frozen_threshold,
            **validation_workpoint,
            "selectionMetrics": checkpoint_selection,
        }
        checkpoint["workpoints"] = {
            "calibration": calibration_summary,
            "validationAtFrozenThreshold": validation_workpoint,
            "frozenThreshold": frozen_threshold,
            "selectionRule": weighted_precision_selection_rule(
                args.target_weighted_recall,
                minimum_point_estimate=args.calibration_point_floor,
                minimum_lower_confidence_bound=args.calibration_lcb_floor,
                minimum_pose_recall=args.calibration_pose_recall_floor,
            ),
        }
        checkpoint["finalCalibration"] = calibration_summary
        torch.save(checkpoint, output_dir / "best.pt")
        calibration_ready = {
            "protocol": "calibration_ready_pre_test",
            "frozenThreshold": frozen_threshold,
            "calibration": calibration_summary,
            "validationAtFrozenThreshold": validation_workpoint,
            "calibrationThresholdRows": calibration_rows,
            "testEvaluationCount": 0,
            "protocolSplit": protocol_split,
            "args": vars(args),
            "featureMeta": feature_meta,
            "selectionRule": weighted_precision_selection_rule(
                args.target_weighted_recall,
                minimum_point_estimate=args.calibration_point_floor,
                minimum_lower_confidence_bound=args.calibration_lcb_floor,
                minimum_pose_recall=args.calibration_pose_recall_floor,
            ),
            "testPolicy": "Use evaluate_frozen_test.py prepare/evaluate exactly once; this file contains no test result.",
        }
        ready_path = output_dir / "calibration_ready_summary.json"
        ready_path.write_text(json.dumps(calibration_ready, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "outputDir": str(output_dir),
                    "calibrationReadySummary": str(ready_path),
                    "frozenThreshold": frozen_threshold,
                    "testEvaluationCount": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    test_row = evaluate_thresholds(
        model,
        test_split,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=max(1, min(args.pose_set_batch_size, 4)),
        max_steps=None,
        max_candidates_per_pose=args.max_candidates_per_pose,
        seed=args.seed + 999,
        thresholds=np.asarray([frozen_threshold], dtype=np.float32),
        allow_candidate_visible_union=args.allow_invalid_resource_semantics,
    )[0]
    # The checkpoint was selected using complete validation records during
    # training.  Store that selection separately from the final calibration
    # threshold so the frozen-test audit can verify both decisions.
    checkpoint_selection = dict(checkpoint.get("best") or {})
    checkpoint["checkpointSelection"] = checkpoint_selection
    checkpoint["best"] = {
        "epoch": int(checkpoint_selection.get("epoch", checkpoint.get("epoch", 0))),
        "threshold": frozen_threshold,
        **validation_workpoint,
        "selectionMetrics": checkpoint_selection,
    }
    checkpoint["workpoints"] = {
        "calibration": calibration_summary,
        "validationAtFrozenThreshold": validation_workpoint,
        "frozenThreshold": frozen_threshold,
        "selectionRule": weighted_precision_selection_rule(
            args.target_weighted_recall,
            minimum_point_estimate=args.calibration_point_floor,
            minimum_lower_confidence_bound=args.calibration_lcb_floor,
            minimum_pose_recall=args.calibration_pose_recall_floor,
        ),
    }
    checkpoint["finalCalibration"] = calibration_summary
    torch.save(checkpoint, output_dir / "best.pt")
    summary = {
        "protocol": "frozen_calibration_one_shot_test",
        "best": test_row,
        "frozenThreshold": frozen_threshold,
        "calibration": calibration_summary,
        "validationAtFrozenThreshold": validation_workpoint,
        "test": test_row,
        "testEvaluationCount": 1,
        "protocolSplit": protocol_split,
        "args": vars(args),
        "featureMeta": feature_meta,
        "dataSemantics": {
            "visibleIds": "pose-level GT visible instance set",
            "visibleWeights": dataset.visible_weight_semantics,
            "occlusionEvidence": "derived from GT visibility and current-view projection geometry; no teacher model",
        },
        "imageEvaluation": {
            "status": "not_run",
            "requiredForFormalReport": True,
            "script": "neural_instance_culling/benchmark/evaluate_viewcell_image_per.py",
        },
        "selectionRule": weighted_precision_selection_rule(
            args.target_weighted_recall,
            minimum_point_estimate=args.calibration_point_floor,
            minimum_lower_confidence_bound=args.calibration_lcb_floor,
            minimum_pose_recall=args.calibration_pose_recall_floor,
        ),
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": summary["best"], "outputDir": str(output_dir), "imageEvaluation": summary["imageEvaluation"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
