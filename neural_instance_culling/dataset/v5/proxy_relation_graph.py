"""Pure-geometry AABB proxy relations for GCOF-PVS V5."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import (
    ANCHOR_SCHEMA,
    EDGE_FEATURE_DIM,
    RELATION_ANCHOR_COUNT,
    RELATION_SCHEMA,
    RELATION_TOP_K,
    SchemaError,
    read_json_manifest,
    validate_relation_manifest,
    write_json_manifest,
)


_PHI = (1.0 + np.sqrt(5.0)) * 0.5
_ANCHOR_COORDINATES = np.asarray(
    [
        (0.0, 1.0, _PHI),
        (0.0, 1.0, -_PHI),
        (0.0, -1.0, _PHI),
        (0.0, -1.0, -_PHI),
        (1.0, _PHI, 0.0),
        (1.0, -_PHI, 0.0),
        (-1.0, _PHI, 0.0),
        (-1.0, -_PHI, 0.0),
        (_PHI, 0.0, 1.0),
        (_PHI, 0.0, -1.0),
        (-_PHI, 0.0, 1.0),
        (-_PHI, 0.0, -1.0),
    ],
    dtype=np.float64,
)
_ICOSAHEDRON12 = (_ANCHOR_COORDINATES / np.linalg.norm(_ANCHOR_COORDINATES, axis=1, keepdims=True)).astype(
    np.float32
)


def icosahedron12_directions() -> np.ndarray:
    """Return the immutable-order normalized 12-direction anchor table."""

    return _ICOSAHEDRON12.copy()


def _screen_basis(anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = np.eye(3, dtype=np.float64)
    least_parallel = int(np.argmin(np.abs(axes @ anchor)))
    u = np.cross(axes[least_parallel], anchor)
    u /= np.linalg.norm(u)
    v = np.cross(anchor, u)
    v /= np.linalg.norm(v)
    return u.astype(np.float32), v.astype(np.float32)


def icosahedron12_screen_bases() -> tuple[np.ndarray, np.ndarray]:
    right = np.empty((RELATION_ANCHOR_COUNT, 3), dtype=np.float32)
    up = np.empty((RELATION_ANCHOR_COUNT, 3), dtype=np.float32)
    for index, anchor in enumerate(_ICOSAHEDRON12):
        right[index], up[index] = _screen_basis(anchor)
    return right, up


def _coerce_aabbs(aabbs: Any) -> np.ndarray:
    value = np.asarray(aabbs, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (2, 3) or value.shape[0] == 0:
        raise ValueError("aabbs must have shape [N,2,3] and contain at least one unit")
    if not np.isfinite(value).all() or np.any(value[:, 0, :] > value[:, 1, :]):
        raise ValueError("AABBs must be finite and min <= max")
    return value


def _aabb_corners(aabbs: np.ndarray) -> np.ndarray:
    signs = np.asarray(
        [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ],
        dtype=np.int64,
    )
    return aabbs[:, None, 0, :] + signs[None, :, :] * (
        aabbs[:, None, 1, :] - aabbs[:, None, 0, :]
    )


def _projected_overlap_candidates(
    u_min: np.ndarray,
    u_max: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
) -> list[list[int]]:
    """Return every positive-area 2D overlap using a deterministic u-sweep.

    The sweep emits each pair exactly once.  It is a broad phase only: the
    caller still applies the exact depth and center-side predicates.  No
    candidate limit is applied here, so a dense projection remains exact.
    """

    count = int(u_min.size)
    order = np.lexsort((np.arange(count, dtype=np.int64), u_min))
    candidates: list[list[int]] = [[] for _ in range(count)]
    active: dict[int, None] = {}
    expiry: list[tuple[float, int]] = []
    for current in order.tolist():
        start = float(u_min[current])
        while expiry and expiry[0][0] <= start:
            _end, expired = heapq.heappop(expiry)
            active.pop(expired, None)
        for other in active:
            if v_min[other] < v_max[current] and v_max[other] > v_min[current]:
                candidates[current].append(other)
                candidates[other].append(current)
        active[current] = None
        heapq.heappush(expiry, (float(u_max[current]), current))
    return candidates


@dataclass(frozen=True)
class ProxyRelationGraph:
    unit_ids: np.ndarray
    source_ids: np.ndarray
    valid_mask: np.ndarray
    edge_features: np.ndarray
    manifest: dict[str, Any]

    @property
    def target_ids(self) -> np.ndarray:
        return self.unit_ids[:, None, None]

    @property
    def anchor_ids(self) -> np.ndarray:
        return np.broadcast_to(
            np.arange(RELATION_ANCHOR_COUNT, dtype=np.int16)[None, :, None],
            self.source_ids.shape,
        )

    @property
    def edge_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    def flat_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return valid target/source/anchor/feature rows in deterministic order."""

        target = np.broadcast_to(self.unit_ids[:, None, None], self.source_ids.shape)
        anchor = self.anchor_ids
        mask = self.valid_mask
        return (
            target[mask].astype(np.uint64, copy=False),
            self.source_ids[mask].astype(np.uint64, copy=False),
            anchor[mask].astype(np.int16, copy=False),
            self.edge_features[mask].astype(np.float32, copy=False),
        )


def _make_relation_manifest(unit_ids: np.ndarray) -> dict[str, Any]:
    return {
        "schema": RELATION_SCHEMA,
        "version": 1,
        "source": "geometry_only",
        "usesVisibilityLabels": False,
        "anchors": ANCHOR_SCHEMA,
        "anchorCount": RELATION_ANCHOR_COUNT,
        "K": RELATION_TOP_K,
        "projection": "orthographic_aabb_overlap",
        "depthPredicate": "source_center_forward_and_depth_interval_camera_side",
        "edgeFeatureSchema": "relative_direction_log_distance_log_radius_overlap_log_gap",
        "edgeFeatureDim": EDGE_FEATURE_DIM,
        "storage": "dense_topk",
        "numUnits": int(unit_ids.size),
    }


def build_proxy_relation_graph(
    aabbs: Any,
    *,
    unit_ids: Any | None = None,
    top_k: int = RELATION_TOP_K,
) -> ProxyRelationGraph:
    """Construct geometry-only top-K potential occluders per target/anchor.

    The depth test is interval based: source ``d_max`` must be in front of the
    target ``d_min``.  The selected edge uses only AABB projections, centers,
    and target-relative scale-free quantities.  No labels or observed edge
    table is accepted by this API.
    """

    if top_k != RELATION_TOP_K:
        raise ValueError("V5 relation top_k is fixed at 8")
    boxes = _coerce_aabbs(aabbs)
    count = boxes.shape[0]
    if unit_ids is None:
        ids = np.arange(count, dtype=np.uint64)
    else:
        ids = np.asarray(unit_ids, dtype=np.uint64)
        if ids.shape != (count,) or len(np.unique(ids)) != count:
            raise ValueError("unit_ids must be unique and have shape [N]")
    centers = (boxes[:, 0, :] + boxes[:, 1, :]) * 0.5
    extents = boxes[:, 1, :] - boxes[:, 0, :]
    radii = np.linalg.norm(extents, axis=1) * 0.5
    valid_units = radii > 1e-12
    safe_radii = np.maximum(radii, 1e-12)
    corners = _aabb_corners(boxes)
    anchors = _ICOSAHEDRON12.astype(np.float64)
    right, up = icosahedron12_screen_bases()
    source_ids = np.full((count, RELATION_ANCHOR_COUNT, RELATION_TOP_K), -1, dtype=np.int64)
    valid_mask = np.zeros((count, RELATION_ANCHOR_COUNT, RELATION_TOP_K), dtype=bool)
    edge_features = np.zeros(
        (count, RELATION_ANCHOR_COUNT, RELATION_TOP_K, EDGE_FEATURE_DIM), dtype=np.float32
    )

    for anchor_index, anchor in enumerate(anchors):
        # Each projection is made from all eight corners, so rotated AABBs are
        # represented by their exact orthogonal projected intervals.
        u_values = corners @ right[anchor_index].astype(np.float64)
        v_values = corners @ up[anchor_index].astype(np.float64)
        d_values = corners @ anchor
        u_min, u_max = u_values.min(axis=1), u_values.max(axis=1)
        v_min, v_max = v_values.min(axis=1), v_values.max(axis=1)
        d_min, d_max = d_values.min(axis=1), d_values.max(axis=1)
        target_u_width = np.maximum(u_max - u_min, 1e-12)
        target_v_width = np.maximum(v_max - v_min, 1e-12)
        projected_area = target_u_width * target_v_width
        overlap_candidates = _projected_overlap_candidates(u_min, u_max, v_min, v_max)

        for target_index in range(count):
            candidate_indices = np.asarray(overlap_candidates[target_index], dtype=np.int64)
            if candidate_indices.size == 0:
                continue
            overlap_u = np.minimum(u_max[candidate_indices], u_max[target_index]) - np.maximum(
                u_min[candidate_indices], u_min[target_index]
            )
            overlap_v = np.minimum(v_max[candidate_indices], v_max[target_index]) - np.maximum(
                v_min[candidate_indices], v_min[target_index]
            )
            overlap_area = overlap_u * overlap_v
            center_forward = (centers[candidate_indices] - centers[target_index]) @ anchor
            target_center_depth = float(centers[target_index] @ anchor)
            gap = np.maximum(0.0, d_min[candidate_indices] - d_max[target_index])
            candidate = (
                (candidate_indices != target_index)
                & valid_units[candidate_indices]
                & valid_units[target_index]
                & (center_forward > 0.0)
                & (d_max[candidate_indices] > target_center_depth)
                & (d_max[candidate_indices] > d_min[target_index])
            )
            if not bool(candidate.any()):
                continue
            candidate_indices = candidate_indices[candidate]
            overlap_area = overlap_area[candidate]
            gap = gap[candidate]
            target_ratio = overlap_area / projected_area[target_index]
            score = target_ratio / (1.0 + gap / safe_radii[target_index])
            # np.lexsort uses the last key as primary.  This is the frozen
            # geometry-only order: score, smaller gap, larger source area, ID.
            order = np.lexsort(
                (
                    ids[candidate_indices],
                    -projected_area[candidate_indices],
                    gap,
                    -score,
                )
            )[:RELATION_TOP_K]
            selected = candidate_indices[order]
            slots = selected.size
            source_ids[target_index, anchor_index, :slots] = ids[selected].astype(np.int64)
            valid_mask[target_index, anchor_index, :slots] = True

            relative = centers[selected] - centers[target_index]
            distances = np.linalg.norm(relative, axis=1)
            direction = np.zeros_like(relative)
            nonzero = distances > 1e-12
            direction[nonzero] = relative[nonzero] / distances[nonzero, None]
            selected_overlap = overlap_area[order]
            selected_gap = gap[order]
            overlap_target = selected_overlap / projected_area[target_index]
            overlap_source = selected_overlap / projected_area[selected]
            feature_values = np.column_stack(
                [
                    direction,
                    np.log1p(distances / safe_radii[target_index]),
                    np.log(safe_radii[selected] / safe_radii[target_index]),
                    overlap_target,
                    overlap_source,
                    np.log1p(selected_gap / safe_radii[target_index]),
                ]
            )
            edge_features[target_index, anchor_index, :slots] = feature_values.astype(np.float32)

    manifest = _make_relation_manifest(ids)
    validate_relation_manifest(manifest)
    return ProxyRelationGraph(
        ids.astype(np.uint64, copy=False),
        source_ids,
        valid_mask,
        edge_features,
        manifest,
    )


def validate_proxy_relation_graph(graph: ProxyRelationGraph) -> None:
    validate_relation_manifest(graph.manifest)
    count = graph.unit_ids.size
    expected_ids = (count, RELATION_ANCHOR_COUNT, RELATION_TOP_K)
    if graph.source_ids.shape != expected_ids or graph.valid_mask.shape != expected_ids:
        raise SchemaError("relation arrays must have shape [N,12,8]")
    if graph.edge_features.shape != expected_ids + (EDGE_FEATURE_DIM,):
        raise SchemaError("relation edge features must have shape [N,12,8,8]")
    if not np.isfinite(graph.edge_features).all():
        raise SchemaError("relation edge features contain non-finite values")
    if np.any(graph.source_ids[~graph.valid_mask] != -1):
        raise SchemaError("invalid relation slots must use source ID -1")
    if np.any(graph.source_ids[graph.valid_mask] < 0):
        raise SchemaError("valid relation slots must have non-negative source IDs")


def write_proxy_relation_graph(graph: ProxyRelationGraph, output_dir: str | Path) -> Path:
    validate_proxy_relation_graph(graph)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "source_ids_int64.npy", graph.source_ids)
    np.save(output / "valid_mask_bool.npy", graph.valid_mask)
    np.save(output / "edge_features_fp32.npy", graph.edge_features)
    np.save(output / "unit_ids_uint64.npy", graph.unit_ids)
    write_json_manifest(output / "relation_manifest.json", graph.manifest)
    return output / "relation_manifest.json"


def load_proxy_relation_graph(output_dir: str | Path) -> ProxyRelationGraph:
    output = Path(output_dir)
    manifest = validate_relation_manifest(read_json_manifest(output / "relation_manifest.json"))
    unit_ids = np.load(output / "unit_ids_uint64.npy", allow_pickle=False)
    source_ids = np.load(output / "source_ids_int64.npy", allow_pickle=False)
    valid_mask = np.load(output / "valid_mask_bool.npy", allow_pickle=False)
    edge_features = np.load(output / "edge_features_fp32.npy", allow_pickle=False)
    graph = ProxyRelationGraph(
        unit_ids.astype(np.uint64, copy=False),
        source_ids.astype(np.int64, copy=False),
        valid_mask.astype(bool, copy=False),
        edge_features.astype(np.float32, copy=False),
        manifest,
    )
    validate_proxy_relation_graph(graph)
    if graph.unit_ids.size != manifest["numUnits"]:
        raise SchemaError("relation unit count disagrees with manifest")
    return graph
