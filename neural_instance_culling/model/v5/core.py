"""GCOF-PVS V5 geometry-compiled visibility core.

The module has one offline path and one lightweight query path:

* local surface samples become shared 32D geometry descriptors;
* a geometry-only proxy relation is compiled once with one masked attention
  pool per target/anchor cell;
* twelve seven-value anchor responses are projected with a fixed order-one
  directional pseudoinverse into a 4x7 field;
* a nine-support survival summary and a 16D query vector feed the visibility
  head.

``GEOMETRY_FIELD`` removes the relation input while keeping the structured
field and query head.  ``GENERIC_RELATION_28`` keeps the proxy compiler but
uses a same-capacity 28D latent directly in its query head.  Neither control
contains a scene identifier or a learned per-unit table.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from neural_instance_culling.dataset.v5.directions import icosahedron12_directions

try:  # Support both package imports and the repository's path-based tests.
    from .geometry_encoder import (
        GEOMETRY_DIM,
        LocalSurfaceGeometryEncoder,
        POINT_COUNT,
    )
    from .survival import (
        DoubleTruncatedLogisticSurvival,
        FIELD_RANK,
        SUPPORT_POINT_COUNT,
        SURVIVAL_PARAMETER_DIM,
        first_order_direction_basis,
        region_survival_statistics,
    )
except ImportError:  # pragma: no cover - exercised by path-based imports.
    from geometry_encoder import (  # type: ignore[no-redef]
        GEOMETRY_DIM,
        LocalSurfaceGeometryEncoder,
        POINT_COUNT,
    )
    from survival import (  # type: ignore[no-redef]
        DoubleTruncatedLogisticSurvival,
        FIELD_RANK,
        SUPPORT_POINT_COUNT,
        SURVIVAL_PARAMETER_DIM,
        first_order_direction_basis,
        region_survival_statistics,
    )


MODEL_SCHEMA = "gcof-pvs-v5"
GEOMETRY_RELATION_SCHEMA = "pvs-geometry-proxy-relation-csr-v1"
VARIANTS = ("FULL", "GEOMETRY_FIELD", "GENERIC_RELATION_28")
ANCHOR_COUNT = 12
TOP_K = 8
EDGE_FEATURE_DIM = 8
EDGE_HIDDEN_DIM = 64
MESSAGE_DIM = 32
FIELD_PARAMETER_DIM = SURVIVAL_PARAMETER_DIM
FIELD_SHAPE = (FIELD_RANK, FIELD_PARAMETER_DIM)
FIELD_DIM = FIELD_RANK * FIELD_PARAMETER_DIM
QUERY_GEOMETRY_DIM = 16
REGION_STATS_DIM = 4
FULL_HEAD_INPUT_DIM = GEOMETRY_DIM + REGION_STATS_DIM + QUERY_GEOMETRY_DIM
GENERIC_HEAD_INPUT_DIM = GEOMETRY_DIM + FIELD_DIM + QUERY_GEOMETRY_DIM
FULL_HEAD_HIDDEN_DIM = 32
GENERIC_HEAD_HIDDEN_DIM = 22
GEOMETRY_FIELD_HIDDEN_DIM = 157
QUERY_GEOMETRY_CHANNELS = (
    "target_to_region_world_direction_xyz",
    "region_to_target_camera_right_up_forward",
    "log_distance_over_radius",
    "radius_over_distance_plus_radius",
    "normalized_region_half_axes",
    "tan_half_fov_xy",
    "region_type",
    "near_over_distance_plus_radius",
    "log_one_plus_far_over_distance_plus_radius",
)


def icosahedron_anchors(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the canonical ordered unit-norm icosahedron anchors."""
    return torch.as_tensor(
        icosahedron12_directions(dtype=np.float64),
        dtype=dtype,
        device=device,
    )


def fixed_direction_projection(
    anchors: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the fixed Moore-Penrose projection ``[12, 4] -> [4, 12]``."""
    directions = (
        icosahedron_anchors(dtype=torch.float64, device=device)
        if anchors is None
        else torch.as_tensor(anchors, dtype=torch.float64, device=device)
    )
    if directions.shape != (ANCHOR_COUNT, 3):
        raise ValueError(f"anchors must have shape [{ANCHOR_COUNT}, 3]")
    basis = first_order_direction_basis(directions)
    projection = torch.linalg.pinv(basis)
    return projection.to(dtype=dtype)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _finite_float(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _metadata(relation: Any) -> Mapping[str, Any]:
    if isinstance(relation, Mapping):
        raw = relation.get("metadata", relation.get("relation_metadata", {}))
    else:
        raw = getattr(relation, "metadata", {})
    return raw if isinstance(raw, Mapping) else {}


def _metadata_value(metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return None


def _relation_value(relation: Any, *names: str) -> Any:
    if isinstance(relation, Mapping):
        for name in names:
            if name in relation:
                return relation[name]
        return None
    for name in names:
        if hasattr(relation, name):
            return getattr(relation, name)
    return None


def _validate_relation_provenance(relation: Any) -> None:
    metadata = _metadata(relation)
    schema = _relation_value(relation, "schema")
    if schema is None:
        schema = _metadata_value(metadata, "schema")
    if schema is not None and schema != GEOMETRY_RELATION_SCHEMA:
        raise ValueError(
            f"V5 requires geometry-only relation schema {GEOMETRY_RELATION_SCHEMA}"
        )
    uses_labels = _metadata_value(metadata, "usesVisibilityLabels", "uses_visibility_labels")
    if uses_labels is not False:
        raise ValueError("V5 geometry relation must declare usesVisibilityLabels=false")
    source = _metadata_value(metadata, "source", "relationSource", "relation_source")
    if source is not None and str(source).lower() not in {
        "geometry_only",
        "geometry-only",
        "aabb_orthographic_overlap",
    }:
        raise ValueError("V5 relation source must be geometry-only")
    relation_directions = np.asarray(metadata.get("anchorDirections"), dtype=np.float64)
    canonical_directions = icosahedron12_directions(dtype=np.float64)
    if relation_directions.shape != canonical_directions.shape or not bool(
        np.allclose(relation_directions, canonical_directions, atol=1.0e-7, rtol=0.0)
    ):
        raise ValueError("V5 relation anchorDirections disagree with the canonical ordered table")


def _relation_tensors(
    relation: Mapping[str, Any] | Any,
    *,
    num_instances: int,
    target_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Normalize a full or target-sliced geometry proxy relation.

    Source IDs always index the global geometry table.  Dense relations may
    have either a full first dimension ``N`` or the selected first dimension
    ``T``.  Sparse target IDs are global unless metadata explicitly declares a
    selected/local ID space.
    """
    _validate_relation_provenance(relation)
    selected_targets = torch.as_tensor(target_ids, dtype=torch.long, device=device).reshape(-1)
    if (
        selected_targets.numel()
        and (
            bool((selected_targets < 0).any())
            or bool((selected_targets >= num_instances).any())
            or torch.unique(selected_targets).numel() != selected_targets.numel()
        )
    ):
        raise ValueError("target_ids must be unique global geometry-table IDs")
    target_count = selected_targets.numel()
    source_raw = _relation_value(relation, "source_ids", "sourceIds")
    edge_raw = _relation_value(relation, "edge_features", "edgeFeatures", "features")
    if source_raw is None or edge_raw is None:
        raise ValueError("geometry relation requires source_ids and edge_features")
    source = torch.as_tensor(source_raw, dtype=torch.long, device=device)
    edge = torch.as_tensor(edge_raw, dtype=torch.float32, device=device)
    valid_raw = _relation_value(relation, "valid_mask", "validMask", "mask")

    if source.ndim == 3:
        if source.shape[1] != ANCHOR_COUNT:
            raise ValueError(f"dense source_ids must have shape [N, {ANCHOR_COUNT}, K]")
        if source.shape[0] == num_instances:
            source = source[selected_targets]
            if edge.ndim != 4 or edge.shape[0] != num_instances:
                raise ValueError("full dense edge_features must have the geometry-table first dimension")
            edge = edge[selected_targets]
            if valid_raw is not None:
                valid_raw = torch.as_tensor(valid_raw, dtype=torch.bool, device=device)[selected_targets]
        elif source.shape[0] == target_count:
            if edge.ndim != 4 or tuple(edge.shape[:3]) != tuple(source.shape):
                raise ValueError("selected dense edge_features must align with source_ids")
        else:
            raise ValueError("dense relation must contain all targets or exactly selected targets")
        if edge.ndim != 4 or tuple(edge.shape[:3]) != tuple(source.shape) or edge.shape[3] != EDGE_FEATURE_DIM:
            raise ValueError("dense edge_features must have shape [N, 12, K, 8]")
        target = torch.arange(target_count, device=device)[:, None, None].expand_as(source)
        anchor = torch.arange(ANCHOR_COUNT, device=device)[None, :, None].expand_as(source)
        if valid_raw is None:
            valid = source >= 0
        else:
            valid = torch.as_tensor(valid_raw, dtype=torch.bool, device=device)
            if valid.shape != source.shape:
                raise ValueError("valid_mask must have the same shape as source_ids")
        source = source.reshape(-1)
        target = target.reshape(-1)
        anchor = anchor.reshape(-1)
        edge = edge.reshape(-1, EDGE_FEATURE_DIM)
        valid = valid.reshape(-1)
    elif source.ndim == 1:
        if edge.ndim != 2 or tuple(edge.shape) != (source.numel(), EDGE_FEATURE_DIM):
            raise ValueError("sparse edge_features must have shape [E, 8]")
        target_raw = _relation_value(relation, "target_ids", "targetIds")
        anchor_raw = _relation_value(relation, "anchor_ids", "anchorIds", "direction_ids")
        if target_raw is None or anchor_raw is None:
            raise ValueError("sparse relations require target_ids and anchor_ids")
        target_raw_tensor = torch.as_tensor(target_raw, dtype=torch.long, device=device).reshape(-1)
        anchor = torch.as_tensor(anchor_raw, dtype=torch.long, device=device).reshape(-1)
        if target_raw_tensor.shape != source.shape or anchor.shape != source.shape:
            raise ValueError("sparse relation index arrays must have equal length")
        if valid_raw is None:
            valid = torch.ones_like(source, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid_raw, dtype=torch.bool, device=device).reshape(-1)
            if valid.shape != source.shape:
                raise ValueError("valid_mask must align with sparse relation edges")
        metadata = _metadata(relation)
        target_space = str(
            _metadata_value(metadata, "targetIdSpace", "target_id_space") or "global"
        ).lower()
        if target_space in {"local", "selected", "slice"}:
            if bool((target_raw_tensor[valid] < 0).any()) or bool(
                (target_raw_tensor[valid] >= target_count).any()
            ):
                raise ValueError("local sparse target_ids are outside the selected target slice")
            target = torch.full_like(target_raw_tensor, -1)
            target[valid] = target_raw_tensor[valid]
            valid = valid & (target >= 0)
        else:
            if bool((target_raw_tensor[valid] < 0).any()) or bool(
                (target_raw_tensor[valid] >= num_instances).any()
            ):
                raise ValueError("global sparse target_ids are outside the geometry table")
            inverse = torch.full((num_instances,), -1, dtype=torch.long, device=device)
            inverse[selected_targets] = torch.arange(target_count, device=device)
            target = inverse[target_raw_tensor.clamp_min(0)]
            valid = valid & (target >= 0)
    else:
        raise ValueError("source_ids must be sparse [E] or dense [N, 12, K]")

    if not bool(torch.isfinite(edge).all()):
        raise ValueError("edge_features contain non-finite values")
    if bool((target[valid] < 0).any()) or bool((target[valid] >= target_count).any()):
        raise ValueError("target_ids are outside the geometry table")
    if bool((anchor[valid] < 0).any()) or bool((anchor[valid] >= ANCHOR_COUNT).any()):
        raise ValueError("anchor_ids are outside the twelve fixed anchors")
    if bool((source[valid] < 0).any()) or bool((source[valid] >= num_instances).any()):
        raise ValueError("source_ids are outside the geometry table")
    return {
        "source_ids": source,
        "target_ids": target,
        "anchor_ids": anchor,
        "edge_features": edge,
        "valid_mask": valid,
    }


def _segment_softmax(logits: torch.Tensor, groups: torch.Tensor, group_count: int) -> torch.Tensor:
    if logits.ndim != 1 or groups.ndim != 1 or logits.numel() != groups.numel():
        raise ValueError("segment attention inputs must be aligned vectors")
    if logits.numel() == 0:
        return logits.new_empty((0,))
    maxima = torch.full(
        (group_count,), -torch.inf, dtype=logits.dtype, device=logits.device
    )
    maxima.scatter_reduce_(0, groups, logits, reduce="amax", include_self=True)
    exponent = torch.exp(logits - maxima[groups])
    denominator = logits.new_zeros((group_count,))
    denominator.index_add_(0, groups, exponent)
    return exponent / denominator[groups].clamp_min(torch.finfo(logits.dtype).tiny)


class SingleLayerRelationFieldCompiler(nn.Module):
    """Compile one geometry-only proxy graph into a structured directional field."""

    def __init__(self) -> None:
        super().__init__()
        self.edge_message = _mlp(GEOMETRY_DIM * 2 + EDGE_FEATURE_DIM, EDGE_HIDDEN_DIM, MESSAGE_DIM)
        self.attention_logit = nn.Linear(MESSAGE_DIM, 1)
        self.base_head = _mlp(GEOMETRY_DIM + 3, MESSAGE_DIM, FIELD_PARAMETER_DIM)
        self.delta_head = _mlp(MESSAGE_DIM + 2 + 3, MESSAGE_DIM, FIELD_PARAMETER_DIM)
        anchors = icosahedron_anchors()
        self.register_buffer("anchors", anchors, persistent=True)
        self.register_buffer(
            "direction_projection",
            fixed_direction_projection(anchors),
            persistent=True,
        )

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def compile_anchor_responses(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        target_ids: torch.Tensor | None = None,
        anchor_directions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``q[T,12,7]`` for all or selected global target rows.

        ``geometry`` remains the global ``[N,32]`` table so source descriptors
        can be gathered by global ID.  Only selected targets allocate the
        anchor pooling tensors, which avoids an ``N x 12`` workspace for a
        large scene when a step touches a small target batch.
        """
        z = _finite_float(geometry, "geometry")
        if z.ndim != 2 or z.shape[1] != GEOMETRY_DIM:
            raise ValueError(f"geometry must have shape [N, {GEOMETRY_DIM}]")
        device = z.device
        if target_ids is None:
            selected_target_ids = torch.arange(z.shape[0], dtype=torch.long, device=device)
        else:
            selected_target_ids = torch.as_tensor(
                target_ids, dtype=torch.long, device=device
            ).reshape(-1)
        if (
            selected_target_ids.numel()
            and (
                bool((selected_target_ids < 0).any())
                or bool((selected_target_ids >= z.shape[0]).any())
                or torch.unique(selected_target_ids).numel() != selected_target_ids.numel()
            )
        ):
            raise ValueError("target_ids must be unique global geometry-table IDs")
        selected_geometry = z[selected_target_ids]
        count = selected_target_ids.numel()
        anchors = self.anchors
        if anchor_directions is not None:
            anchors = _finite_float(anchor_directions, "anchor_directions").to(device)
            if anchors.shape != (ANCHOR_COUNT, 3):
                raise ValueError(f"anchor_directions must have shape [{ANCHOR_COUNT}, 3]")
            anchors = F.normalize(anchors, dim=-1)
        group_count = count * ANCHOR_COUNT
        pooled = selected_geometry.new_zeros((group_count, MESSAGE_DIM))
        neighbor_count = selected_geometry.new_zeros((group_count,))
        overlap_sum = selected_geometry.new_zeros((group_count,))
        attention_grid = selected_geometry.new_zeros((group_count,))
        source_ids = selected_target_ids.new_empty((0,))
        local_target_ids = selected_target_ids.new_empty((0,))
        anchor_ids = selected_target_ids.new_empty((0,))
        if relation is not None:
            tensors = _relation_tensors(
                relation,
                num_instances=z.shape[0],
                target_ids=selected_target_ids,
                device=device,
            )
            valid = tensors["valid_mask"]
            source_ids = tensors["source_ids"][valid]
            local_target_ids = tensors["target_ids"][valid]
            anchor_ids = tensors["anchor_ids"][valid]
            edge = tensors["edge_features"][valid]
            if source_ids.numel():
                messages = self.edge_message(
                    torch.cat(
                        [
                            selected_geometry[local_target_ids],
                            z[source_ids],
                            edge,
                        ],
                        dim=-1,
                    )
                )
                logits = self.attention_logit(messages).squeeze(-1)
                groups = local_target_ids * ANCHOR_COUNT + anchor_ids
                weights = _segment_softmax(logits, groups, group_count)
                pooled.index_add_(0, groups, weights[:, None] * messages)
                neighbor_count.index_add_(0, groups, torch.ones_like(logits))
                overlap_sum.index_add_(0, groups, edge[:, 5].clamp_min(0.0))
                attention_grid.index_add_(0, groups, weights)
            else:
                messages = z.new_empty((0, MESSAGE_DIM))
                weights = z.new_empty((0,))
        else:
            messages = z.new_empty((0, MESSAGE_DIM))
            weights = z.new_empty((0,))

        h = pooled.reshape(count, ANCHOR_COUNT, MESSAGE_DIM)
        counts = neighbor_count.reshape(count, ANCHOR_COUNT)
        overlaps = overlap_sum.reshape(count, ANCHOR_COUNT)
        base_input = torch.cat(
            [
                selected_geometry[:, None, :].expand(-1, ANCHOR_COUNT, -1),
                anchors[None, :, :].expand(count, -1, -1),
            ],
            dim=-1,
        )
        base = self.base_head(base_input)
        delta_input = torch.cat(
            [
                h,
                torch.log1p(counts).unsqueeze(-1),
                overlaps.unsqueeze(-1),
                anchors[None, :, :].expand(count, -1, -1),
            ],
            dim=-1,
        )
        delta = self.delta_head(delta_input)
        has_neighbor = (counts > 0).unsqueeze(-1).to(delta.dtype)
        responses = base + has_neighbor * delta
        diagnostics = {
            "anchor_responses": responses,
            "anchor_messages": h,
            "neighbor_count": counts,
            "overlap_sum": overlaps,
            "attention_weights": weights,
            "attention_grid_sum": attention_grid.reshape(count, ANCHOR_COUNT),
            "source_ids": source_ids,
            "target_ids": selected_target_ids,
            "local_target_ids": local_target_ids,
            "anchor_ids": anchor_ids,
        }
        return responses, diagnostics

    def project_anchor_responses(
        self,
        responses: torch.Tensor,
        anchor_directions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = _finite_float(responses, "anchor_responses")
        if values.ndim != 3 or tuple(values.shape[1:]) != (ANCHOR_COUNT, FIELD_PARAMETER_DIM):
            raise ValueError(f"anchor_responses must have shape [N, {ANCHOR_COUNT}, 7]")
        projection = self.direction_projection.to(values)
        if anchor_directions is not None:
            projection = fixed_direction_projection(
                anchor_directions,
                dtype=values.dtype,
                device=values.device,
            )
        field = torch.einsum("rk,nkp->nrp", projection, values)
        if not bool(torch.isfinite(field).all()):
            raise FloatingPointError("directional field projection produced non-finite values")
        return field

    def forward(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        *,
        target_ids: torch.Tensor | None = None,
        anchor_directions: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        responses, diagnostics = self.compile_anchor_responses(
            geometry, relation, target_ids, anchor_directions
        )
        field = self.project_anchor_responses(responses, anchor_directions)
        if return_diagnostics:
            return {"field": field, **diagnostics}
        return field


def _capacity_matched_geometry_hidden(relation_parameters: int) -> int:
    """Match the Full relation compiler with a two-layer geometry MLP."""
    # Parameters for Linear(32,h), Linear(h,28), including biases.
    width = round((relation_parameters - FIELD_DIM) / (GEOMETRY_DIM + 1 + FIELD_DIM))
    candidates = range(max(1, width - 4), width + 5)
    return min(
        candidates,
        key=lambda candidate: abs(
            (GEOMETRY_DIM + FIELD_DIM + 1) * candidate + FIELD_DIM
            - relation_parameters
        ),
    )


def _select_rows(
    values: torch.Tensor,
    batch: int,
    instance_ids: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    tensor = values
    if instance_ids is not None:
        ids = torch.as_tensor(instance_ids, dtype=torch.long, device=tensor.device).reshape(-1)
        if ids.numel() != batch or bool((ids < 0).any()) or bool((ids >= tensor.shape[0]).any()):
            raise ValueError(f"{name} instance_ids do not select {batch} valid rows")
        return tensor[ids]
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(f"{name} has {tensor.shape[0]} rows but query batch is {batch}")


def _as_query_geometry(query_geometry: torch.Tensor) -> torch.Tensor:
    query = _finite_float(query_geometry, "query_geometry")
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if query.ndim != 2 or query.shape[1] != QUERY_GEOMETRY_DIM:
        raise ValueError(f"query_geometry must have shape [B, {QUERY_GEOMETRY_DIM}]")
    return query


def build_query_geometry(
    target_to_region_world_direction: torch.Tensor,
    camera_right_up_forward: torch.Tensor,
    distance: torch.Tensor,
    target_radius: torch.Tensor,
    region_half_axes: torch.Tensor,
    fov_tangent_xy: torch.Tensor,
    region_type: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
) -> torch.Tensor:
    """Build the exact V6.1 query geometry layout.

    ``target_to_region_world_direction`` is the target-to-region direction.
    The second channel group is computed from its reverse direction in the
    supplied camera ``[right, up, forward]`` basis.  FOV inputs are already
    ``tan(FOV_x/2), tan(FOV_y/2)``; the function does not substitute scene
    constants or a camera hash.
    """
    world = torch.as_tensor(target_to_region_world_direction).float()
    if world.ndim == 1:
        world = world.unsqueeze(0)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError("target_to_region_world_direction must have shape [B, 3]")
    batch = world.shape[0]
    if not bool(torch.isfinite(world).all()):
        raise ValueError("target_to_region_world_direction contains non-finite values")
    world_norm = torch.linalg.vector_norm(world, dim=-1, keepdim=True)
    eps = torch.finfo(world.dtype).eps
    world_unit = torch.where(
        world_norm > eps,
        world / world_norm.clamp_min(eps),
        torch.zeros_like(world),
    )

    basis = torch.as_tensor(camera_right_up_forward, device=world.device).float()
    if basis.ndim == 2 and basis.shape == (3, 3):
        basis = basis.unsqueeze(0)
    if basis.ndim != 3 or basis.shape[1:] != (3, 3) or basis.shape[0] not in (1, batch):
        raise ValueError("camera_right_up_forward must have shape [B, 3, 3]")
    if basis.shape[0] == 1 and batch != 1:
        basis = basis.expand(batch, -1, -1)
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("camera_right_up_forward contains non-finite values")
    basis_norm = torch.linalg.vector_norm(basis, dim=-1, keepdim=True)
    if bool((basis_norm <= eps).any()):
        raise ValueError("camera right/up/forward axes must be non-zero")
    basis = basis / basis_norm
    reverse_camera_components = torch.einsum(
        "bi,bji->bj", -world_unit, basis
    )

    def rows(value: torch.Tensor, width: int, name: str) -> torch.Tensor:
        result = torch.as_tensor(value, device=world.device).float()
        if result.ndim == 0:
            result = result.reshape(1, 1)
        elif result.ndim == 1:
            if result.numel() == width:
                result = result.reshape(1, width)
            else:
                result = result.reshape(-1, 1)
        if result.ndim != 2 or result.shape[1] != width or result.shape[0] not in (1, batch):
            raise ValueError(f"{name} must have shape [{width}] or [B, {width}]")
        if result.shape[0] == 1 and batch != 1:
            result = result.expand(batch, -1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} contains non-finite values")
        return result

    d = rows(distance, 1, "distance")
    radius = rows(target_radius, 1, "target_radius")
    axes = rows(region_half_axes, 3, "region_half_axes")
    fov = rows(fov_tangent_xy, 2, "fov_tangent_xy")
    region = rows(region_type, 1, "region_type")
    near_value = rows(near, 1, "near")
    far_value = rows(far, 1, "far")
    if bool((d < 0).any()) or bool((radius <= 0).any()):
        raise ValueError("distance must be non-negative and target_radius positive")
    if bool((axes < 0).any()) or bool((fov <= 0).any()):
        raise ValueError("region_half_axes must be non-negative and FOV tangents positive")
    if bool(((region != 0) & (region != 1)).any()):
        raise ValueError("region_type must be 0 for disk or 1 for oriented box")
    if bool((near_value < 0).any()) or bool((far_value < 0).any()):
        raise ValueError("near and far must be non-negative")

    denominator = d + radius
    axis_scale = denominator + axes.abs().amax(dim=-1, keepdim=True)
    normalized_axes = axes / axis_scale
    log_distance = torch.log1p(d / radius)
    radius_fraction = radius / denominator
    near_fraction = near_value / denominator
    log_far = torch.log1p(far_value / denominator)
    result = torch.cat(
        [
            world_unit,
            reverse_camera_components,
            log_distance,
            radius_fraction,
            normalized_axes,
            fov,
            region,
            near_fraction,
            log_far,
        ],
        dim=-1,
    )
    if result.shape[1] != QUERY_GEOMETRY_DIM:
        raise RuntimeError(f"query geometry packing produced width {result.shape[1]}")
    return _finite_float(result, "query_geometry")


class GCOFPVSV5(nn.Module):
    """Full GCOF-PVS V5 and its two registered representation controls."""

    def __init__(self, variant: str = "FULL") -> None:
        super().__init__()
        normalized_variant = str(variant).upper()
        if normalized_variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = normalized_variant
        self.geometry_encoder = LocalSurfaceGeometryEncoder()
        self.survival = DoubleTruncatedLogisticSurvival()

        if normalized_variant in {"FULL", "GENERIC_RELATION_28"}:
            self.relation_compiler: SingleLayerRelationFieldCompiler | None = (
                SingleLayerRelationFieldCompiler()
            )
        else:
            self.relation_compiler = None

        if normalized_variant == "GEOMETRY_FIELD":
            assert self.relation_compiler is None
            matched_width = _capacity_matched_geometry_hidden(
                SingleLayerRelationFieldCompiler().parameter_count
            )
            self.geometry_field = _mlp(GEOMETRY_DIM, matched_width, FIELD_DIM)
            self.geometry_field_hidden_dim = matched_width
        else:
            self.geometry_field = None
            self.geometry_field_hidden_dim = None

        if normalized_variant == "GENERIC_RELATION_28":
            # The unrestricted control receives the complete ordered 12x7
            # evidence. Its runtime latent remains exactly 28 values.
            self.generic_projection = nn.Linear(
                ANCHOR_COUNT * FIELD_PARAMETER_DIM, FIELD_DIM, bias=False
            )
            head_input_dim = GENERIC_HEAD_INPUT_DIM
            head_hidden_dim = GENERIC_HEAD_HIDDEN_DIM
        else:
            self.generic_projection = None
            head_input_dim = FULL_HEAD_INPUT_DIM
            head_hidden_dim = FULL_HEAD_HIDDEN_DIM
        self.head_input_dim = head_input_dim
        self.visibility_head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    @property
    def runtime_field_shape(self) -> tuple[int, int] | None:
        return FIELD_SHAPE if self.variant != "GENERIC_RELATION_28" else None

    @property
    def config(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SCHEMA,
            "variant": self.variant,
            "pointsPerUnit": POINT_COUNT,
            "geometryDim": GEOMETRY_DIM,
            "anchors": ANCHOR_COUNT,
            "relationTopK": TOP_K,
            "edgeFeatureDim": EDGE_FEATURE_DIM,
            "fieldShape": list(FIELD_SHAPE),
            "supportPointCount": SUPPORT_POINT_COUNT,
            "queryGeometryDim": QUERY_GEOMETRY_DIM,
            "headShape": [self.head_input_dim, self.visibility_head[0].out_features, 1],
        }

    def encode_geometry(
        self,
        points: torch.Tensor,
        size_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.geometry_encoder(points, size_ratios)

    def _geometry_table(
        self,
        geometry_or_points: torch.Tensor,
        size_ratios: torch.Tensor | None,
    ) -> torch.Tensor:
        values = torch.as_tensor(geometry_or_points)
        if values.ndim == 2 and values.shape[1] == GEOMETRY_DIM:
            return _finite_float(values, "geometry")
        return self.encode_geometry(values, size_ratios)

    def compile_outputs(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        target_ids: torch.Tensor | None = None,
        anchor_directions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self._geometry_table(geometry, None)
        selected_ids = (
            torch.arange(z.shape[0], dtype=torch.long, device=z.device)
            if target_ids is None
            else torch.as_tensor(target_ids, dtype=torch.long, device=z.device).reshape(-1)
        )
        if self.variant == "GEOMETRY_FIELD":
            if relation is not None:
                raise ValueError("GEOMETRY_FIELD does not consume relation input")
            assert self.geometry_field is not None
            selected_z = _select_rows(z, selected_ids.numel(), selected_ids, "geometry")
            field = self.geometry_field(selected_z).reshape(-1, *FIELD_SHAPE)
            return {"geometry": selected_z, "field": field, "target_ids": selected_ids}

        assert self.relation_compiler is not None
        responses, diagnostics = self.relation_compiler.compile_anchor_responses(
            z, relation, selected_ids, anchor_directions
        )
        if self.variant == "FULL":
            field = self.relation_compiler.project_anchor_responses(
                responses, anchor_directions
            )
            return {"geometry": z[selected_ids], "field": field, **diagnostics}

        assert self.generic_projection is not None
        latent = self.generic_projection(responses.flatten(start_dim=1))
        return {"geometry": z[selected_ids], "generic_latent": latent, **diagnostics}

    def compile_field(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compile Full/GEOMETRY_FIELD coefficients; reject generic latent use."""
        outputs = self.compile_outputs(geometry, relation, target_ids)
        if "field" not in outputs:
            raise ValueError("GENERIC_RELATION_28 does not expose an analytic survival field")
        return outputs["field"]

    def relation_latent(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.variant != "GENERIC_RELATION_28":
            raise ValueError("relation_latent is only defined for GENERIC_RELATION_28")
        return self.compile_outputs(geometry, relation, target_ids)["generic_latent"]

    def region_statistics(
        self,
        field: torch.Tensor,
        target_center: torch.Tensor,
        support_points: torch.Tensor,
        target_radius: torch.Tensor | float,
    ) -> torch.Tensor:
        return region_survival_statistics(field, target_center, support_points, target_radius)

    def query(
        self,
        geometry: torch.Tensor,
        query_geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        *,
        instance_ids: torch.Tensor | None = None,
        field: torch.Tensor | None = None,
        generic_latent: torch.Tensor | None = None,
        field_stats: torch.Tensor | None = None,
        support_points: torch.Tensor | None = None,
        target_center: torch.Tensor | None = None,
        target_radius: torch.Tensor | float | None = None,
        size_ratios: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        query = _as_query_geometry(query_geometry)
        batch = query.shape[0]
        z = self._geometry_table(geometry, size_ratios)
        selected_z = _select_rows(z, batch, instance_ids, "geometry")

        outputs: dict[str, torch.Tensor]
        if field is not None or generic_latent is not None:
            outputs = {"geometry": z}
            if field is not None:
                values = _finite_float(field, "field")
                if values.ndim == 2:
                    values = values.reshape(-1, *FIELD_SHAPE)
                if values.ndim != 3 or tuple(values.shape[1:]) != FIELD_SHAPE:
                    raise ValueError(f"field must have shape [N, {FIELD_RANK}, {FIELD_PARAMETER_DIM}]")
                outputs["field"] = values
            if generic_latent is not None:
                latent = _finite_float(generic_latent, "generic_latent")
                if latent.ndim == 1:
                    latent = latent.unsqueeze(0)
                if latent.ndim != 2 or latent.shape[1] != FIELD_DIM:
                    raise ValueError(f"generic_latent must have shape [N, {FIELD_DIM}]")
                outputs["generic_latent"] = latent
        else:
            outputs = self.compile_outputs(z, relation)

        if self.variant == "GENERIC_RELATION_28":
            if "generic_latent" not in outputs:
                raise ValueError("GENERIC_RELATION_28 requires generic_latent or relation input")
            latent = _select_rows(outputs["generic_latent"], batch, instance_ids, "generic_latent")
            head_input = torch.cat([selected_z, latent, query], dim=-1)
            stats = None
        else:
            if "field" not in outputs:
                raise ValueError("structured variants require a 4x7 field")
            selected_field = _select_rows(outputs["field"], batch, instance_ids, "field")
            if field_stats is not None:
                stats = _finite_float(field_stats, "field_stats")
                if stats.ndim == 1:
                    stats = stats.unsqueeze(0)
                if stats.shape != (batch, REGION_STATS_DIM):
                    raise ValueError(f"field_stats must have shape [B, {REGION_STATS_DIM}]")
            else:
                if support_points is None or target_center is None or target_radius is None:
                    raise ValueError(
                        "structured variants require field_stats or nine support points, target_center, and target_radius"
                    )
                centers = _finite_float(target_center, "target_center")
                points = _finite_float(support_points, "support_points")
                centers = _select_rows(centers if centers.ndim == 2 else centers.unsqueeze(0), batch, instance_ids, "target_center")
                points = _select_rows(points if points.ndim == 3 else points.unsqueeze(0), batch, instance_ids, "support_points")
                radii = torch.as_tensor(target_radius).float()
                if radii.ndim == 0:
                    radii = radii.reshape(1)
                if radii.ndim == 2 and radii.shape[1] == 1:
                    radii = radii[:, 0]
                radii = _select_rows(radii.reshape(-1, 1), batch, instance_ids, "target_radius").reshape(-1)
                stats = region_survival_statistics(selected_field, centers, points, radii)
            head_input = torch.cat([selected_z, stats, query], dim=-1)

        if head_input.shape[1] != self.head_input_dim:
            raise RuntimeError(f"visibility head expected {self.head_input_dim} inputs")
        logits = self.visibility_head(head_input)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("V5 visibility head produced non-finite logits")
        if return_details:
            result: dict[str, torch.Tensor] = {
                "logits": logits,
                "geometry": selected_z,
                "query_geometry": query,
                "query_input": head_input,
            }
            if stats is not None:
                result["field_stats"] = stats
            if "field" in outputs and self.variant != "GENERIC_RELATION_28":
                result["field"] = _select_rows(outputs["field"], batch, instance_ids, "field")
            if "generic_latent" in outputs:
                result["generic_latent"] = _select_rows(
                    outputs["generic_latent"], batch, instance_ids, "generic_latent"
                )
            return result
        return logits

    def forward(
        self,
        geometry: torch.Tensor,
        query_geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        return self.query(geometry, query_geometry, relation, **kwargs)


GCOFPVS = GCOFPVSV5
V5Model = GCOFPVSV5


__all__ = [
    "ANCHOR_COUNT",
    "EDGE_FEATURE_DIM",
    "FIELD_DIM",
    "FIELD_SHAPE",
    "FULL_HEAD_INPUT_DIM",
    "GCOFPVS",
    "GCOFPVSV5",
    "GEOMETRY_FIELD_HIDDEN_DIM",
    "GEOMETRY_RELATION_SCHEMA",
    "GENERIC_HEAD_INPUT_DIM",
    "MODEL_SCHEMA",
    "QUERY_GEOMETRY_DIM",
    "SingleLayerRelationFieldCompiler",
    "TOP_K",
    "V5Model",
    "VARIANTS",
    "build_query_geometry",
    "fixed_direction_projection",
    "icosahedron_anchors",
]
