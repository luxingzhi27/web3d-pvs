"""Bounded v3 hierarchical relation encoder for offline survival fields.

The relation artifact consumed here is deliberately narrow.  It must be the
train-only ``SCHEMA_V3`` artifact produced by ``train_observed_relation_csr``
and its edge features must have ``RELATION_FEATURE_DIM_V3`` columns.  The
browser never receives this graph; this module is an offline encoder used to
produce one fixed ``[N, 4, 7]`` coefficient field per instance.

Each edge is assigned to a target/direction/depth cell.  Within that cell the
learned attention logit is combined with the evidence confidence prior before
the segment softmax.  Empty cells stay exactly zero.  Depth cells remain
ordered in a fixed sequence, while instance, local-group, and structural-group
summaries all use bounded attention pools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .train_observed_relation_csr import (
    CONFIDENCE_FEATURE,
    RELATION_FEATURE_DIM_V3,
    RELATION_FEATURE_NAMES_V3,
    SCHEMA_V3,
    ObservedRelationCSR,
    degree_preserving_directed_edge_swap,
)


_EPS = 1.0e-6
_V3_CONFIDENCE_FEATURE = RELATION_FEATURE_NAMES_V3.index("evidence_confidence")
_V3_RELATIVE_CENTER_SLICE = slice(
    RELATION_FEATURE_NAMES_V3.index("relative_center_x"),
    RELATION_FEATURE_NAMES_V3.index("relative_center_z") + 1,
)
_V3_RELATIVE_LOG_SCALE_FEATURE = RELATION_FEATURE_NAMES_V3.index("relative_log_scale")
if CONFIDENCE_FEATURE != _V3_CONFIDENCE_FEATURE:
    raise RuntimeError("v3 confidence feature must retain the registered column")


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, *, layers: int = 2) -> nn.Sequential:
    if min(int(input_dim), int(hidden_dim), int(output_dim), int(layers)) <= 0:
        raise ValueError("MLP dimensions and layers must be positive")
    modules: list[nn.Module] = []
    current = int(input_dim)
    for _ in range(int(layers) - 1):
        modules.extend((nn.Linear(current, int(hidden_dim)), nn.SiLU()))
        current = int(hidden_dim)
    modules.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*modules)


def _segment_weight_sums(
    weights: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    sums = weights.new_zeros((int(num_segments),))
    if weights.numel():
        sums.index_add_(0, segment_ids, weights)
    return sums


def segment_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int | None = None,
    evidence_confidence: torch.Tensor | None = None,
    *,
    eps: float = _EPS,
    valid_mask: torch.Tensor | None = None,
    return_segment_sums: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Softmax over segments with the v3 evidence-confidence prior.

    ``logits`` contains the learned score for each edge.  The effective score
    is ``logits + log(evidence_confidence + eps)``.  ``valid_mask`` is useful
    for higher-level pools: masked entries do not participate at all.  The
    returned edge weights have one sum per non-empty segment; segments with no
    entries have a diagnostic sum of zero.
    """
    values = torch.as_tensor(logits)
    if values.ndim != 1 or not values.is_floating_point():
        raise ValueError("logits must be a floating-point vector")
    segments = torch.as_tensor(segment_ids, dtype=torch.long, device=values.device).reshape(-1)
    if segments.numel() != values.numel():
        raise ValueError("segment_ids and logits must have the same length")
    if segments.numel() and int(segments.min()) < 0:
        raise ValueError("segment_ids must be non-negative")
    if num_segments is None:
        count = int(segments.max().item()) + 1 if segments.numel() else 0
    else:
        count = int(num_segments)
    if count < 0 or (segments.numel() and int(segments.max()) >= count):
        raise ValueError("segment_ids exceed num_segments")
    if not torch.isfinite(values).all():
        raise ValueError("logits must be finite")
    if evidence_confidence is None:
        confidence = torch.ones_like(values)
    else:
        confidence = torch.as_tensor(
            evidence_confidence,
            dtype=values.dtype,
            device=values.device,
        ).reshape(-1)
        if confidence.numel() != values.numel():
            raise ValueError("evidence_confidence and logits must have the same length")
        if not torch.isfinite(confidence).all() or bool((confidence < 0).any()):
            raise ValueError("evidence_confidence must be finite and non-negative")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    if valid_mask is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    else:
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=values.device).reshape(-1)
        if valid.numel() != values.numel():
            raise ValueError("valid_mask and logits must have the same length")

    if values.numel() == 0:
        weights = values.new_empty((0,))
        sums = values.new_zeros((count,))
        return (weights, sums) if return_segment_sums else weights

    adjusted = values + torch.log(confidence + float(eps))
    adjusted = torch.where(valid, adjusted, torch.full_like(adjusted, -torch.inf))
    maxima = torch.full((count,), -torch.inf, dtype=values.dtype, device=values.device)
    if segments.numel():
        maxima.scatter_reduce_(0, segments, adjusted, reduce="amax", include_self=True)
    finite_segments = torch.isfinite(maxima)
    shifted = adjusted - maxima[segments]
    exponent = torch.where(valid & finite_segments[segments], torch.exp(shifted), torch.zeros_like(values))
    denominators = values.new_zeros((count,))
    denominators.index_add_(0, segments, exponent)
    weights = exponent / denominators[segments].clamp_min(float(eps))
    weights = torch.where(valid & (denominators[segments] > 0.0), weights, torch.zeros_like(weights))
    sums = _segment_weight_sums(weights, segments, count)
    return (weights, sums) if return_segment_sums else weights


def segment_attention(
    values: torch.Tensor,
    learned_logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int | None = None,
    evidence_confidence: torch.Tensor | None = None,
    *,
    eps: float = _EPS,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool ``values`` by segment and return pooled values, weights, and sums."""
    tensor = torch.as_tensor(values)
    if tensor.ndim != 2 or not tensor.is_floating_point():
        raise ValueError("values must be a floating-point matrix [E, C]")
    segments = torch.as_tensor(segment_ids, dtype=torch.long, device=tensor.device).reshape(-1)
    if segments.numel() != tensor.shape[0]:
        raise ValueError("segment_ids and values must have the same first dimension")
    count = int(num_segments) if num_segments is not None else (
        int(segments.max().item()) + 1 if segments.numel() else 0
    )
    weights, sums = segment_softmax(
        learned_logits,
        segments,
        count,
        evidence_confidence,
        eps=eps,
        valid_mask=valid_mask,
        return_segment_sums=True,
    )
    pooled = tensor.new_zeros((count, tensor.shape[1]))
    if tensor.shape[0]:
        pooled.index_add_(0, segments, weights.unsqueeze(-1) * tensor)
    return pooled, weights, sums


# The longer name makes call sites self-documenting while keeping the short
# helper above convenient in tests and small offline controls.
bounded_attention_pool = segment_attention


def _contiguous_segment_ids(
    values: torch.Tensor,
    *,
    expected_size: int,
    name: str,
) -> tuple[torch.Tensor, int]:
    ids = torch.as_tensor(values, dtype=torch.long).reshape(-1)
    if ids.numel() != int(expected_size):
        raise ValueError(f"{name} must have shape [{expected_size}]")
    if ids.numel() == 0:
        return ids, 0
    if int(ids.min()) < 0:
        raise ValueError(f"{name} must be non-negative")
    count = int(ids.max().item()) + 1
    expected = torch.arange(count, dtype=torch.long, device=ids.device)
    if not torch.equal(torch.unique(ids, sorted=True), expected):
        raise ValueError(f"{name} must contain contiguous IDs from zero")
    return ids, count


@dataclass(frozen=True)
class BoundedHierarchySpec:
    """Contiguous instance-to-local and local-to-structural group mappings."""

    local_group_ids: torch.Tensor
    structural_group_ids: torch.Tensor
    local_group_count: int
    structural_group_count: int


def identity_hierarchy(
    num_instances: int,
    *,
    local_group_size: int = 1,
    structural_group_size: int | None = None,
    device: torch.device | None = None,
) -> BoundedHierarchySpec:
    """Create a deterministic bounded hierarchy for controls and smoke tests."""
    if int(num_instances) <= 0 or int(local_group_size) <= 0:
        raise ValueError("num_instances and local_group_size must be positive")
    local = torch.arange(int(num_instances), dtype=torch.long, device=device) // int(local_group_size)
    local_count = int(local.max().item()) + 1
    parent_size = int(structural_group_size or local_count)
    if parent_size <= 0:
        raise ValueError("structural_group_size must be positive")
    structural = torch.arange(local_count, dtype=torch.long, device=device) // parent_size
    return BoundedHierarchySpec(local, structural, local_count, int(structural.max().item()) + 1)


def _relation_metadata(relation: ObservedRelationCSR | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(relation, ObservedRelationCSR):
        return relation.metadata
    if not isinstance(relation, Mapping):
        return {}
    metadata = relation.get("metadata", relation.get("relation_metadata", {}))
    if isinstance(metadata, Mapping):
        return metadata
    return {}


def _relation_schema(relation: ObservedRelationCSR | Mapping[str, Any]) -> Any:
    metadata = _relation_metadata(relation)
    if isinstance(relation, Mapping) and "schema" in relation:
        return relation["schema"]
    return metadata.get("schema")


def _relation_tensors(
    relation: ObservedRelationCSR | Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load and strictly validate the tensors needed by the v3 encoder."""
    if _relation_schema(relation) != SCHEMA_V3:
        raise ValueError(f"bounded relation encoder requires {SCHEMA_V3}")
    if isinstance(relation, ObservedRelationCSR):
        relation.validate()
        raw = relation.to_torch(device=device)
    elif isinstance(relation, Mapping):
        required = ("source_ids", "target_ids", "direction_ids", "depth_shell_ids", "edge_features")
        if any(key not in relation for key in required):
            raise ValueError(f"v3 relation mapping needs {required}")
        raw = {key: torch.as_tensor(relation[key], device=device) for key in required}
    else:
        raise TypeError("relation must be ObservedRelationCSR or a v3 tensor mapping")
    result = {
        "source_ids": torch.as_tensor(raw["source_ids"], dtype=torch.long, device=device).reshape(-1),
        "target_ids": torch.as_tensor(raw["target_ids"], dtype=torch.long, device=device).reshape(-1),
        "direction_ids": torch.as_tensor(raw["direction_ids"], dtype=torch.long, device=device).reshape(-1),
        "depth_shell_ids": torch.as_tensor(raw["depth_shell_ids"], dtype=torch.long, device=device).reshape(-1),
        "edge_features": torch.as_tensor(raw["edge_features"], dtype=torch.float32, device=device),
    }
    edge_count = int(result["source_ids"].numel())
    if result["edge_features"].ndim != 2 or tuple(result["edge_features"].shape) != (
        edge_count,
        RELATION_FEATURE_DIM_V3,
    ):
        raise ValueError(
            f"v3 relation edge_features must have shape [{edge_count}, {RELATION_FEATURE_DIM_V3}]"
        )
    if any(int(result[key].numel()) != edge_count for key in ("target_ids", "direction_ids", "depth_shell_ids")):
        raise ValueError("v3 relation index arrays have different edge counts")
    if edge_count:
        if torch.any(result["source_ids"] == result["target_ids"]):
            raise ValueError("v3 relation edges may not contain self-loops")
        if not torch.isfinite(result["edge_features"]).all():
            raise ValueError("v3 relation edge_features must be finite")
        confidence = result["edge_features"][:, _V3_CONFIDENCE_FEATURE]
        if bool((confidence < 0.0).any()) or bool((confidence > 1.0).any()):
            raise ValueError("v3 evidence_confidence must be in [0, 1]")
    return result


def _validate_relation_indices(
    relation: Mapping[str, torch.Tensor],
    *,
    num_instances: int,
    direction_bins: int,
    depth_shells: int,
) -> None:
    for name in ("source_ids", "target_ids"):
        values = relation[name]
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= int(num_instances)):
            raise ValueError(f"relation {name} is outside geometry features")
    if relation["direction_ids"].numel() and (
        int(relation["direction_ids"].min()) < 0
        or int(relation["direction_ids"].max()) >= int(direction_bins)
    ):
        raise ValueError("relation direction ID exceeds encoder direction_bins")
    if relation["depth_shell_ids"].numel() and (
        int(relation["depth_shell_ids"].min()) < 0
        or int(relation["depth_shell_ids"].max()) >= int(depth_shells)
    ):
        raise ValueError("relation depth shell exceeds encoder depth_shells")


def _empty_like_ids(reference: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.long, device=reference.device)


def deterministic_negative_edges(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    direction_ids: torch.Tensor,
    depth_shell_ids: torch.Tensor,
    *,
    distance_bucket_ids: torch.Tensor | None = None,
    seed: int = 0,
    split: str = "train",
) -> dict[str, torch.Tensor]:
    """Create deterministic degree-preserving negatives inside relation strata.

    The sampled discrete arrays are transferred to CPU once, shuffled with the
    same directed two-edge swap used by the formal relation control, and then
    transferred back once. ``feature_row_ids`` identifies the positive
    evidence template for every changed edge so the caller can gather and
    recompute its target-dependent geometry without a per-edge table scan.
    """
    if split != "train":
        raise ValueError("relation negatives are train-only")
    source = torch.as_tensor(source_ids, dtype=torch.long).reshape(-1)
    target = torch.as_tensor(target_ids, dtype=torch.long, device=source.device).reshape(-1)
    direction = torch.as_tensor(direction_ids, dtype=torch.long, device=source.device).reshape(-1)
    shell = torch.as_tensor(depth_shell_ids, dtype=torch.long, device=source.device).reshape(-1)
    arrays = (target, direction, shell)
    if any(item.numel() != source.numel() for item in arrays):
        raise ValueError("negative-edge index arrays have different lengths")
    if source.numel() and bool((source == target).any()):
        raise ValueError("positive relation input contains a self-loop")
    if distance_bucket_ids is None:
        bucket = torch.zeros_like(direction)
    else:
        bucket = torch.as_tensor(distance_bucket_ids, dtype=torch.long, device=source.device).reshape(-1)
        if bucket.numel() != source.numel():
            raise ValueError("distance_bucket_ids and edge arrays have different lengths")

    source_cpu = source.detach().cpu().numpy().astype(np.int64, copy=False)
    target_cpu = target.detach().cpu().numpy().astype(np.int64, copy=False)
    direction_cpu = direction.detach().cpu().numpy().astype(np.int64, copy=False)
    shell_cpu = shell.detach().cpu().numpy().astype(np.int64, copy=False)
    bucket_cpu = bucket.detach().cpu().numpy().astype(np.int64, copy=False)
    shuffled_source, shuffled_target = degree_preserving_directed_edge_swap(
        source_cpu,
        target_cpu,
        direction_cpu,
        shell_cpu,
        distance_bucket_ids=bucket_cpu,
        seed=int(seed),
        swap_fraction=1.0,
    )
    changed = np.flatnonzero(shuffled_target.astype(np.int64, copy=False) != target_cpu)
    if changed.size == 0:
        return {
            "source_ids": _empty_like_ids(source),
            "target_ids": _empty_like_ids(source),
            "direction_ids": _empty_like_ids(source),
            "depth_shell_ids": _empty_like_ids(source),
            "feature_row_ids": _empty_like_ids(source),
        }
    rows = torch.as_tensor(changed, dtype=torch.long, device=source.device)
    return {
        "source_ids": torch.as_tensor(shuffled_source[changed], dtype=torch.long, device=source.device),
        "target_ids": torch.as_tensor(shuffled_target[changed], dtype=torch.long, device=source.device),
        "direction_ids": direction[rows],
        "depth_shell_ids": shell[rows],
        "feature_row_ids": rows,
    }


def _confidence_ranking_loss(
    scores: torch.Tensor,
    segment_ids: torch.Tensor,
    evidence_confidence: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if scores.numel() < 2:
        return scores.sum() * 0.0
    pieces: list[torch.Tensor] = []
    for segment in torch.unique(segment_ids, sorted=True).tolist():
        indices = torch.nonzero(segment_ids == int(segment), as_tuple=False).reshape(-1)
        if indices.numel() < 2:
            continue
        order = torch.argsort(evidence_confidence[indices], descending=True, stable=True)
        high = indices[order[:-1]]
        low = indices[order[1:]]
        pieces.append(F.softplus(float(margin) - (scores[high] - scores[low])))
    if not pieces:
        return scores.sum() * 0.0
    return torch.cat(pieces).mean()


def _relation_consistency_from_scores(
    positive_edge_logits: torch.Tensor,
    negative_edge_logits: torch.Tensor,
    positive_depth_order_logits: torch.Tensor,
    positive_confidence_logits: torch.Tensor,
    positive_segment_ids: torch.Tensor,
    evidence_confidence: torch.Tensor,
    *,
    negative_depth_order_logits: torch.Tensor | None = None,
    margin: float = 0.1,
    classification_weight: float = 1.0,
    depth_order_weight: float = 1.0,
    confidence_rank_weight: float = 1.0,
    split: str = "train",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the train-only relation classification/order/ranking loss.

    Negative edge construction is intentionally separate and deterministic;
    callers should use :func:`deterministic_negative_edges` within the train
    split before scoring those edges.  The depth term requires true source to
    target ordering to score above a positive margin.  The ranking term orders
    learned confidence scores in each target/direction/depth cell according to
    the observed evidence confidence.
    """
    if split != "train":
        raise ValueError("relation consistency loss is train-only")
    positive = torch.as_tensor(positive_edge_logits)
    negative = torch.as_tensor(negative_edge_logits, dtype=positive.dtype, device=positive.device).reshape(-1)
    depth = torch.as_tensor(positive_depth_order_logits, dtype=positive.dtype, device=positive.device).reshape(-1)
    confidence_scores = torch.as_tensor(
        positive_confidence_logits,
        dtype=positive.dtype,
        device=positive.device,
    ).reshape(-1)
    segments = torch.as_tensor(positive_segment_ids, dtype=torch.long, device=positive.device).reshape(-1)
    evidence = torch.as_tensor(evidence_confidence, dtype=positive.dtype, device=positive.device).reshape(-1)
    if positive.ndim != 1 or not positive.is_floating_point():
        raise ValueError("positive_edge_logits must be a floating-point vector")
    if depth.numel() != positive.numel() or confidence_scores.numel() != positive.numel():
        raise ValueError("positive relation score arrays have different lengths")
    if segments.numel() != positive.numel() or evidence.numel() != positive.numel():
        raise ValueError("positive relation metadata arrays have different lengths")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all() or not torch.isfinite(depth).all():
        raise ValueError("relation consistency scores must be finite")
    if not torch.isfinite(confidence_scores).all() or not torch.isfinite(evidence).all():
        raise ValueError("relation confidence scores must be finite")
    if bool((evidence < 0.0).any()):
        raise ValueError("evidence_confidence must be non-negative")

    zero = positive.sum() * 0.0 + negative.sum() * 0.0
    if positive.numel():
        positive_classification = F.binary_cross_entropy_with_logits(
            positive,
            torch.ones_like(positive),
        )
    else:
        positive_classification = zero
    if negative.numel():
        negative_classification = F.binary_cross_entropy_with_logits(
            negative,
            torch.zeros_like(negative),
        )
    else:
        negative_classification = zero
    classification = 0.5 * (positive_classification + negative_classification)

    if depth.numel():
        depth_order = F.softplus(float(margin) - depth).mean()
    else:
        depth_order = zero
    if negative_depth_order_logits is not None:
        negative_depth = torch.as_tensor(
            negative_depth_order_logits,
            dtype=positive.dtype,
            device=positive.device,
        ).reshape(-1)
        if not torch.isfinite(negative_depth).all():
            raise ValueError("negative depth-order scores must be finite")
        if negative_depth.numel():
            depth_order = depth_order + 0.25 * F.softplus(float(margin) + negative_depth).mean()
    ranking = _confidence_ranking_loss(
        confidence_scores,
        segments,
        evidence,
        margin=float(margin),
    )
    total = (
        float(classification_weight) * classification
        + float(depth_order_weight) * depth_order
        + float(confidence_rank_weight) * ranking
    )
    parts = {
        "lossRelationClassification": classification,
        "lossDepthOrder": depth_order,
        "lossConfidenceRanking": ranking,
        "lossRelationConsistency": total,
        "positiveEdgeCount": positive.new_tensor(float(positive.numel())),
        "negativeEdgeCount": positive.new_tensor(float(negative.numel())),
    }
    return total, parts


class BoundedHierarchicalRelationSurvivalEncoder(nn.Module):
    """Encode v3 observed relations into bounded per-instance survival fields."""

    def __init__(
        self,
        *,
        geo_dim: int = 96,
        hidden_dim: int = 64,
        direction_bins: int = 12,
        depth_shells: int = 3,
        survival_rank: int = 4,
        survival_parameter_dim: int = 7,
        local_max_group_size: int = 32,
        structural_max_group_size: int = 64,
    ) -> None:
        super().__init__()
        if min(int(geo_dim), int(hidden_dim), int(direction_bins), int(depth_shells)) <= 0:
            raise ValueError("encoder dimensions must be positive")
        if int(survival_rank) != 4 or int(survival_parameter_dim) != 7:
            raise ValueError("v3 survival coefficients are fixed at [4, 7]")
        if int(local_max_group_size) <= 0 or int(structural_max_group_size) <= 0:
            raise ValueError("hierarchy group caps must be positive")
        self.geo_dim = int(geo_dim)
        self.hidden_dim = int(hidden_dim)
        self.direction_bins = int(direction_bins)
        self.depth_shells = int(depth_shells)
        self.survival_rank = int(survival_rank)
        self.survival_parameter_dim = int(survival_parameter_dim)
        self.local_max_group_size = int(local_max_group_size)
        self.structural_max_group_size = int(structural_max_group_size)

        self.geometry_encoder = _mlp(self.geo_dim, self.hidden_dim, self.hidden_dim)
        self.direction_embedding = nn.Parameter(torch.empty(self.direction_bins, self.hidden_dim))
        self.depth_embedding = nn.Parameter(torch.empty(self.depth_shells, self.hidden_dim))
        nn.init.normal_(self.direction_embedding, std=0.04)
        nn.init.normal_(self.depth_embedding, std=0.04)

        self.edge_message = _mlp(
            2 * self.hidden_dim + RELATION_FEATURE_DIM_V3 + 2 * self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
        )
        self.edge_attention_score = _mlp(self.hidden_dim, self.hidden_dim, 1)
        self.depth_sequence = _mlp(
            self.depth_shells * (2 * self.hidden_dim + 1),
            self.hidden_dim,
            self.hidden_dim,
        )
        self.instance_attention_score = _mlp(2 * self.hidden_dim, self.hidden_dim, 1)
        self.instance_update = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.local_attention_score = _mlp(2 * self.hidden_dim, self.hidden_dim, 1)
        self.local_update = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.structural_attention_score = _mlp(2 * self.hidden_dim, self.hidden_dim, 1)
        self.structural_update = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)

        self.relation_decode = _mlp(3 * self.hidden_dim + 1, self.hidden_dim, self.hidden_dim)
        self.geometry_base_head = nn.Linear(self.hidden_dim, self.survival_rank * self.survival_parameter_dim)
        self.relation_delta_head = nn.Linear(self.hidden_dim, self.survival_rank * self.survival_parameter_dim)
        self.relation_gate_head = nn.Linear(2 * self.hidden_dim + 1, self.survival_rank)
        nn.init.normal_(self.geometry_base_head.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.geometry_base_head.bias)
        nn.init.normal_(self.relation_delta_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.relation_delta_head.bias)
        nn.init.constant_(self.relation_gate_head.bias, -1.5)

        self.relation_edge_classifier = nn.Linear(self.hidden_dim, 1)
        self.relation_depth_order_head = nn.Linear(self.hidden_dim, 1)
        self.relation_confidence_head = nn.Linear(self.hidden_dim, 1)

    def _prepare_hierarchy(
        self,
        local_group_ids: torch.Tensor,
        structural_group_ids: torch.Tensor,
        *,
        num_instances: int,
        device: torch.device,
    ) -> BoundedHierarchySpec:
        local, local_count = _contiguous_segment_ids(
            torch.as_tensor(local_group_ids, dtype=torch.long, device=device),
            expected_size=num_instances,
            name="local_group_ids",
        )
        structural_input = torch.as_tensor(structural_group_ids, dtype=torch.long, device=device).reshape(-1)
        if structural_input.numel() == num_instances and structural_input.numel() != local_count:
            # Accept an instance-level spelling only when every local group has
            # one consistent structural parent, then canonicalize it.
            structural_per_local = []
            for local_id in range(local_count):
                values = structural_input[local == local_id]
                if values.numel() == 0 or not bool((values == values[0]).all()):
                    raise ValueError("structural_group_ids must be constant within each local group")
                structural_per_local.append(values[0])
            structural_input = torch.stack(structural_per_local) if structural_per_local else structural_input[:0]
        structural, structural_count = _contiguous_segment_ids(
            structural_input,
            expected_size=local_count,
            name="structural_group_ids",
        )
        local_sizes = torch.bincount(local, minlength=local_count)
        structural_sizes = torch.bincount(structural, minlength=structural_count)
        if local_sizes.numel() and int(local_sizes.max()) > self.local_max_group_size:
            raise ValueError("local hierarchy group exceeds the bounded instance cap")
        if structural_sizes.numel() and int(structural_sizes.max()) > self.structural_max_group_size:
            raise ValueError("structural hierarchy group exceeds the bounded local-group cap")
        return BoundedHierarchySpec(local, structural, local_count, structural_count)

    def _edge_messages(
        self,
        node_state: torch.Tensor,
        relation: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = relation["source_ids"]
        target = relation["target_ids"]
        direction = relation["direction_ids"]
        shell = relation["depth_shell_ids"]
        features = relation["edge_features"]
        if source.numel() == 0:
            empty = node_state.new_zeros((0, self.hidden_dim))
            return empty, node_state.new_zeros((0,))
        edge_input = torch.cat(
            [
                node_state[source],
                node_state[target],
                features,
                self.direction_embedding[direction],
                self.depth_embedding[shell],
            ],
            dim=-1,
        )
        message = self.edge_message(edge_input)
        learned_logits = self.edge_attention_score(message).squeeze(-1)
        return message, learned_logits

    @staticmethod
    def _scatter_selected(
        values: torch.Tensor,
        selected_indices: torch.Tensor,
        weights: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        output = values.new_zeros((int(num_segments), values.shape[1]))
        if selected_indices.numel():
            output.index_add_(
                0,
                segment_ids[selected_indices],
                weights.unsqueeze(-1) * values[selected_indices],
            )
        return output

    def _selected_attention(
        self,
        values: torch.Tensor,
        learned_logits: torch.Tensor,
        segment_ids: torch.Tensor,
        valid: torch.Tensor,
        num_segments: int,
        evidence_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if selected.numel() == 0:
            pooled = values.new_zeros((int(num_segments), values.shape[1]))
            full_weights = values.new_zeros((values.shape[0],))
            sums = values.new_zeros((int(num_segments),))
            return pooled, full_weights, sums, selected
        confidence = None if evidence_confidence is None else evidence_confidence[selected]
        pooled, weights, sums = segment_attention(
            values[selected],
            learned_logits[selected],
            segment_ids[selected],
            int(num_segments),
            confidence,
        )
        full_weights = values.new_zeros((values.shape[0],))
        full_weights[selected] = weights
        return pooled, full_weights, sums, selected

    def _encode_core(
        self,
        geo: torch.Tensor,
        relation: Mapping[str, torch.Tensor],
        hierarchy: BoundedHierarchySpec,
    ) -> dict[str, torch.Tensor | BoundedHierarchySpec]:
        num_instances = int(geo.shape[0])
        node_state = self.geometry_encoder(geo)
        edge_message, edge_learned_logits = self._edge_messages(node_state, relation)
        source = relation["source_ids"]
        target = relation["target_ids"]
        direction = relation["direction_ids"]
        shell = relation["depth_shell_ids"]
        features = relation["edge_features"]
        cell_ids = target * (self.direction_bins * self.depth_shells) + direction * self.depth_shells + shell
        cell_count = num_instances * self.direction_bins * self.depth_shells
        edge_attention_confidence = (
            features[:, _V3_CONFIDENCE_FEATURE] if features.shape[0] else features.new_empty((0,))
        )
        cell_values, edge_attention, cell_weight_sums = segment_attention(
            edge_message,
            edge_learned_logits,
            cell_ids,
            cell_count,
            edge_attention_confidence,
        )
        cell_values = cell_values.reshape(num_instances, self.direction_bins, self.depth_shells, self.hidden_dim)
        cell_weight_grid = cell_weight_sums.reshape(num_instances, self.direction_bins, self.depth_shells)
        depth_present = cell_weight_grid > 0.0
        depth_embedding = self.depth_embedding.view(1, 1, self.depth_shells, self.hidden_dim).expand(
            num_instances, self.direction_bins, -1, -1
        )
        depth_input = torch.cat(
            [
                cell_values,
                depth_embedding * depth_present.unsqueeze(-1).to(cell_values.dtype),
                depth_present.unsqueeze(-1).to(cell_values.dtype),
            ],
            dim=-1,
        )
        ordered_direction = self.depth_sequence(depth_input.reshape(num_instances * self.direction_bins, -1))
        ordered_direction = ordered_direction.reshape(num_instances, self.direction_bins, self.hidden_dim)
        direction_present = depth_present.any(dim=-1)
        ordered_direction = ordered_direction * direction_present.unsqueeze(-1).to(ordered_direction.dtype)

        direction_values = ordered_direction.reshape(num_instances * self.direction_bins, self.hidden_dim)
        direction_segments = torch.arange(num_instances, dtype=torch.long, device=geo.device).repeat_interleave(
            self.direction_bins
        )
        direction_valid = direction_present.reshape(-1)
        direction_evidence = cell_weight_grid.sum(dim=-1).reshape(-1)
        instance_pooled, instance_attention, instance_attention_sums, _ = self._selected_attention(
            direction_values,
            self.instance_attention_score(
                torch.cat(
                    [
                        direction_values,
                        node_state.repeat_interleave(self.direction_bins, dim=0),
                    ],
                    dim=-1,
                )
            ).squeeze(-1),
            direction_segments,
            direction_valid,
            num_instances,
            direction_evidence,
        )
        instance_present = instance_attention_sums > 0.0
        instance_relation = self.instance_update(torch.cat([node_state, instance_pooled], dim=-1))
        instance_relation = instance_relation * instance_present.unsqueeze(-1).to(instance_relation.dtype)

        local_ids = hierarchy.local_group_ids.to(geo.device)
        local_logits = self.local_attention_score(torch.cat([instance_relation, node_state], dim=-1)).squeeze(-1)
        local_pooled, local_attention, local_attention_sums, local_selected = self._selected_attention(
            instance_relation,
            local_logits,
            local_ids,
            instance_present,
            hierarchy.local_group_count,
            instance_present.to(instance_relation.dtype),
        )
        local_geometry = self._scatter_selected(
            node_state,
            local_selected,
            local_attention[local_selected],
            local_ids,
            hierarchy.local_group_count,
        )
        local_present = local_attention_sums > 0.0
        local_state = self.local_update(torch.cat([local_pooled, local_geometry], dim=-1))
        local_state = local_state * local_present.unsqueeze(-1).to(local_state.dtype)

        structural_ids = hierarchy.structural_group_ids.to(geo.device)
        structural_logits = self.structural_attention_score(
            torch.cat([local_state, local_geometry], dim=-1)
        ).squeeze(-1)
        structural_pooled, structural_attention, structural_attention_sums, structural_selected = self._selected_attention(
            local_state,
            structural_logits,
            structural_ids,
            local_present,
            hierarchy.structural_group_count,
            local_present.to(local_state.dtype),
        )
        structural_geometry = self._scatter_selected(
            local_geometry,
            structural_selected,
            structural_attention[structural_selected],
            structural_ids,
            hierarchy.structural_group_count,
        )
        structural_present = structural_attention_sums > 0.0
        structural_state = self.structural_update(torch.cat([structural_pooled, structural_geometry], dim=-1))
        structural_state = structural_state * structural_present.unsqueeze(-1).to(structural_state.dtype)

        relation_context = torch.cat(
            [
                instance_relation,
                local_state[local_ids],
                structural_state[structural_ids[local_ids]],
                instance_present.unsqueeze(-1).to(instance_relation.dtype),
            ],
            dim=-1,
        )
        relation_hidden = self.relation_decode(relation_context)
        base = self.geometry_base_head(node_state).reshape(
            num_instances,
            self.survival_rank,
            self.survival_parameter_dim,
        )
        delta = torch.tanh(self.relation_delta_head(relation_hidden)).reshape(
            num_instances,
            self.survival_rank,
            self.survival_parameter_dim,
        )
        delta = delta * instance_present.view(num_instances, 1, 1).to(delta.dtype)
        gate = torch.sigmoid(
            self.relation_gate_head(
                torch.cat([node_state, relation_hidden, instance_present.unsqueeze(-1).to(node_state.dtype)], dim=-1)
            )
        ).reshape(num_instances, self.survival_rank, 1)
        coefficients = base + gate * delta
        delta_base_norm = delta.flatten(1).norm(dim=-1) / base.flatten(1).norm(dim=-1).clamp_min(_EPS)
        return {
            "survival_coefficients": coefficients,
            "coefficients": coefficients,
            "geometry_base": base,
            "geometry_base_coefficients": base,
            "base": base,
            "relation_delta": delta,
            "relation_delta_coefficients": delta,
            "delta": delta,
            "relation_gate": gate,
            "relation_gate_probability": gate,
            "gate": gate,
            "delta_base_norm": delta_base_norm,
            "delta_over_base_norm": delta_base_norm,
            "edge_attention": edge_attention,
            "edge_attention_grid_sum": cell_weight_grid,
            "edge_attention_weight_sums": cell_weight_grid,
            "depth_cell_tokens": cell_values,
            "depth_present": depth_present,
            "instance_attention": instance_attention,
            "instance_attention_sums": instance_attention_sums,
            "local_attention": local_attention,
            "local_attention_sums": local_attention_sums,
            "structural_attention": structural_attention,
            "structural_attention_sums": structural_attention_sums,
            "instance_features": instance_relation,
            "local_features": local_state,
            "structural_features": structural_state,
            "edge_messages": edge_message,
            "edge_learned_logits": edge_learned_logits,
            "hierarchy": hierarchy,
        }

    def forward(
        self,
        geo_features: torch.Tensor,
        relation: ObservedRelationCSR | Mapping[str, Any],
        local_group_ids: torch.Tensor,
        structural_group_ids: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Return ``[N, 4, 7]`` coefficients, optionally with offline diagnostics."""
        geo = torch.as_tensor(geo_features, dtype=torch.float32)
        if geo.ndim != 2 or geo.shape[1] != self.geo_dim:
            raise ValueError(f"geo_features must have shape [N, {self.geo_dim}]")
        if geo.shape[0] <= 0:
            raise ValueError("geo_features must contain at least one instance")
        tensors = _relation_tensors(relation, device=geo.device)
        _validate_relation_indices(
            tensors,
            num_instances=int(geo.shape[0]),
            direction_bins=self.direction_bins,
            depth_shells=self.depth_shells,
        )
        hierarchy = self._prepare_hierarchy(
            local_group_ids,
            structural_group_ids,
            num_instances=int(geo.shape[0]),
            device=geo.device,
        )
        result = self._encode_core(geo, tensors, hierarchy)
        return result if return_diagnostics else result["survival_coefficients"]

    def _negative_feature_rows(
        self,
        features: torch.Tensor,
        feature_row_ids: torch.Tensor,
        negative_source_ids: torch.Tensor,
        negative_target_ids: torch.Tensor,
        instance_world_aabbs: torch.Tensor,
    ) -> torch.Tensor:
        """Gather evidence templates and recompute target-dependent geometry."""
        if feature_row_ids.numel() == 0:
            return features.new_zeros((0, RELATION_FEATURE_DIM_V3))
        rows = features[feature_row_ids].clone()
        aabbs = torch.as_tensor(
            instance_world_aabbs,
            dtype=features.dtype,
            device=features.device,
        )
        if aabbs.ndim != 2 or aabbs.shape[1] != 6:
            raise ValueError("instance_world_aabbs must have shape [N, 6]")
        if not torch.isfinite(aabbs).all() or bool((aabbs[:, 3:] < aabbs[:, :3]).any()):
            raise ValueError("instance_world_aabbs must be finite ordered bounds")
        centers = (aabbs[:, :3] + aabbs[:, 3:]) * 0.5
        radii = torch.linalg.vector_norm((aabbs[:, 3:] - aabbs[:, :3]).clamp_min(1e-6), dim=-1) * 0.5
        scene_size = (aabbs[:, 3:].amax(dim=0) - aabbs[:, :3].amin(dim=0)).clamp_min(1e-6)
        rows[:, _V3_RELATIVE_CENTER_SLICE] = (
            (centers[negative_source_ids] - centers[negative_target_ids]) / scene_size
        ).clamp(-8.0, 8.0)
        rows[:, _V3_RELATIVE_LOG_SCALE_FEATURE] = torch.log(
            radii[negative_source_ids].clamp_min(1e-6)
            / radii[negative_target_ids].clamp_min(1e-6)
        ).clamp(-8.0, 8.0)
        return rows

    @staticmethod
    def _limit_relation_edges(
        relation: Mapping[str, torch.Tensor],
        *,
        max_edges: int,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        if int(max_edges) <= 0:
            raise ValueError("max_edges must be positive")
        edge_count = int(relation["source_ids"].numel())
        if edge_count <= int(max_edges):
            return dict(relation)
        # Keep the permutation on the relation device.  In particular, do not
        # materialize edge IDs as Python scalars: this path runs every train
        # step and a CUDA tensor -> Python conversion synchronizes the device.
        device = relation["source_ids"].device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        selected = torch.randperm(edge_count, generator=generator, device=device)[: int(max_edges)]
        return {
            "source_ids": relation["source_ids"][selected],
            "target_ids": relation["target_ids"][selected],
            "direction_ids": relation["direction_ids"][selected],
            "depth_shell_ids": relation["depth_shell_ids"][selected],
            "edge_features": relation["edge_features"][selected],
        }

    def relation_consistency_loss(
        self,
        geo_features: torch.Tensor,
        relation: ObservedRelationCSR | Mapping[str, Any],
        *,
        instance_world_aabbs: torch.Tensor,
        seed: int,
        max_edges: int = 8192,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score bounded train-only true and deterministically permuted edges.

        This is the trainer-facing public contract.  The relation artifact is
        always the train-only v3 CSR; validation and test relations are
        rejected before any edge is scored.  ``max_edges`` is applied with a
        stable target/direction/depth/source ordering so repeated steps do not
        depend on Python hash order.
        """
        metadata = _relation_metadata(relation)
        if not metadata or metadata.get("trainOnly") is not True or metadata.get("splitNames") != ["train"]:
            raise ValueError("relation consistency loss requires an explicitly train-only v3 relation")
        geo = torch.as_tensor(geo_features, dtype=torch.float32)
        if geo.ndim != 2 or geo.shape[1] != self.geo_dim or geo.shape[0] <= 0:
            raise ValueError(f"geo_features must have shape [N, {self.geo_dim}] with N > 0")
        tensors = _relation_tensors(relation, device=geo.device)
        tensors = self._limit_relation_edges(tensors, max_edges=int(max_edges), seed=int(seed))
        _validate_relation_indices(
            tensors,
            num_instances=int(geo.shape[0]),
            direction_bins=self.direction_bins,
            depth_shells=self.depth_shells,
        )
        node_state = self.geometry_encoder(geo)
        positive_messages, _ = self._edge_messages(node_state, tensors)
        positive_edge_logits = self.relation_edge_classifier(positive_messages).squeeze(-1)
        positive_depth_order = self.relation_depth_order_head(positive_messages).squeeze(-1)
        positive_confidence_score = self.relation_confidence_head(positive_messages).squeeze(-1)
        positive_segment_ids = (
            tensors["target_ids"] * (self.direction_bins * self.depth_shells)
            + tensors["direction_ids"] * self.depth_shells
            + tensors["depth_shell_ids"]
        )
        aabbs = torch.as_tensor(instance_world_aabbs, dtype=geo.dtype, device=geo.device)
        if aabbs.shape != (geo.shape[0], 6):
            raise ValueError(f"instance_world_aabbs must have shape [{geo.shape[0]}, 6]")
        centers = (aabbs[:, :3] + aabbs[:, 3:]) * 0.5
        edge_distance = torch.linalg.vector_norm(
            centers[tensors["source_ids"]] - centers[tensors["target_ids"]],
            dim=-1,
        )
        if edge_distance.numel():
            quantiles = torch.quantile(
                edge_distance.detach(),
                torch.linspace(0.125, 0.875, 7, device=edge_distance.device),
            )
            distance_bucket_ids = torch.bucketize(edge_distance.detach(), quantiles)
        else:
            distance_bucket_ids = torch.empty_like(tensors["direction_ids"])
        negative = deterministic_negative_edges(
            tensors["source_ids"],
            tensors["target_ids"],
            tensors["direction_ids"],
            tensors["depth_shell_ids"],
            distance_bucket_ids=distance_bucket_ids,
            seed=int(seed),
            split="train",
        )
        negative_features = self._negative_feature_rows(
            tensors["edge_features"],
            negative["feature_row_ids"],
            negative["source_ids"],
            negative["target_ids"],
            aabbs,
        )
        negative_relation = {
            "source_ids": negative["source_ids"],
            "target_ids": negative["target_ids"],
            "direction_ids": negative["direction_ids"],
            "depth_shell_ids": negative["depth_shell_ids"],
            "edge_features": negative_features,
        }
        negative_messages, _ = self._edge_messages(node_state, negative_relation)
        negative_edge_logits = self.relation_edge_classifier(negative_messages).squeeze(-1)
        negative_depth_order = self.relation_depth_order_head(negative_messages).squeeze(-1)
        total, parts = _relation_consistency_from_scores(
            positive_edge_logits,
            negative_edge_logits,
            positive_depth_order,
            positive_confidence_score,
            positive_segment_ids,
            tensors["edge_features"][:, _V3_CONFIDENCE_FEATURE],
            negative_depth_order_logits=negative_depth_order,
            split="train",
        )
        parts.update(
            {
                "positiveEdgeLogits": positive_edge_logits,
                "negativeEdgeLogits": negative_edge_logits,
                "positiveDepthOrderLogits": positive_depth_order,
                "positiveConfidenceLogits": positive_confidence_score,
            }
        )
        parts["positive_edge_count"] = parts["positiveEdgeCount"]
        parts["negative_edge_count"] = parts["negativeEdgeCount"]
        return total, parts


BoundedHierarchicalOcclusionSurvivalEncoder = BoundedHierarchicalRelationSurvivalEncoder


__all__ = [
    "SCHEMA_V3",
    "RELATION_FEATURE_DIM_V3",
    "BoundedHierarchySpec",
    "identity_hierarchy",
    "segment_softmax",
    "segment_attention",
    "bounded_attention_pool",
    "deterministic_negative_edges",
    "BoundedHierarchicalRelationSurvivalEncoder",
    "BoundedHierarchicalOcclusionSurvivalEncoder",
]
