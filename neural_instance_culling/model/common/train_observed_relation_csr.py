"""Training-only observed occlusion relation CSR schema and validators.

This module deliberately defines a new contract for the hierarchical relation
experiment.  It does not read or write the historical ``sourceK`` relation
table.  A row is indexed by ``target instance -> direction bin -> depth
shell`` and stores every observed source within the registered train render
coverage.  The source is the front instance and the target is the deeper
instance.

The CSR is an offline training artifact.  Its metadata carries enough
provenance to reject a relation graph built from another split, candidate
sequence, FOV, or evidence cache before a model is trained.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# v1 is retained only so historical artifacts and their tests remain readable.
# New builders must use the bounded, evidence-aware v3 contract below.
LEGACY_SCHEMA = "pvs-viewcell-train-observed-relation-csr-v1"
SCHEMA = LEGACY_SCHEMA
SCHEMA_V3 = "pvs-viewcell-train-observed-relation-csr-v3"
ROLE = "train-only-observed-occlusion-relation"
INDEX_ORDER = ("target_instance", "direction_bin", "depth_shell")
SOURCE_TRUNCATION = "none_within_observed_coverage"
DEPTH_ORDERING = "source_front_to_target_deep"
SOURCE_TYPES = {
    "depth_peeling": 0,
    "surface_fallback": 1,
    "merged": 2,
}
SOURCE_TYPE_NAMES = tuple(SOURCE_TYPES)

# The feature vector is deliberately explicit.  A future builder must update
# the schema instead of silently changing the meaning of an existing column.
RELATION_FEATURE_NAMES = (
    "pose_support_count",
    "pose_support_rate",
    "pixel_support",
    "pixel_fraction",
    "depth_gap_mean",
    "depth_gap_std",
    "relative_depth_gap_mean",
    "relative_depth_gap_std",
    "relative_center_x",
    "relative_center_y",
    "relative_center_z",
    "relative_log_scale",
    "evidence_confidence",
)
RELATION_FEATURE_DIM = len(RELATION_FEATURE_NAMES)
DEPTH_GAP_FEATURE = RELATION_FEATURE_NAMES.index("depth_gap_mean")
RELATIVE_DEPTH_GAP_FEATURE = RELATION_FEATURE_NAMES.index("relative_depth_gap_mean")
POSE_SUPPORT_COUNT_FEATURE = RELATION_FEATURE_NAMES.index("pose_support_count")
POSE_SUPPORT_RATE_FEATURE = RELATION_FEATURE_NAMES.index("pose_support_rate")
PIXEL_SUPPORT_FEATURE = RELATION_FEATURE_NAMES.index("pixel_support")
PIXEL_FRACTION_FEATURE = RELATION_FEATURE_NAMES.index("pixel_fraction")
CONFIDENCE_FEATURE = RELATION_FEATURE_NAMES.index("evidence_confidence")

RELATION_FEATURE_NAMES_V3 = (
    "pose_support_count",
    "pose_support_rate",
    "pixel_support",
    "pixel_fraction",
    "depth_gap_mean",
    "depth_gap_std",
    "relative_depth_gap_mean",
    "relative_depth_gap_std",
    "relative_center_x",
    "relative_center_y",
    "relative_center_z",
    "relative_log_scale",
    "evidence_confidence",
    "retained_topk_quality",
    "evidence_level",
    "evidence_layer_count",
    "topk_truncated",
    "normalized_log_depth_mean",
    "normalized_log_depth_std",
    "source_count",
)
RELATION_FEATURE_DIM_V3 = len(RELATION_FEATURE_NAMES_V3)


def _feature_schema(metadata: Mapping[str, Any]) -> tuple[tuple[str, ...], int]:
    """Return the feature contract selected by an explicit schema."""
    if metadata.get("schema") == SCHEMA_V3:
        return RELATION_FEATURE_NAMES_V3, RELATION_FEATURE_DIM_V3
    return RELATION_FEATURE_NAMES, RELATION_FEATURE_DIM


class RelationSchemaError(ValueError):
    """Raised when a training relation artifact violates its contract."""


def _canonical_array(values: Any, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(-1))


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def candidate_digest(
    candidate_ids: Any,
    pose_indices: Any,
    offsets: Any,
) -> str:
    """Hash a candidate CSR without sorting or normalizing its row order."""
    poses = _canonical_array(pose_indices, "<i8")
    # Keep the byte contract identical to common.candidate_identity, whose
    # formal candidate digest uses signed little-endian int64 offsets.
    csr_offsets = _canonical_array(offsets, "<i8")
    ids = _canonical_array(candidate_ids, "<u4")
    if csr_offsets.size != poses.size + 1:
        raise RelationSchemaError(
            "candidate offsets must have one more entry than pose indices"
        )
    if csr_offsets.size and int(csr_offsets[0]) != 0:
        raise RelationSchemaError("candidate offsets must start at zero")
    if np.any(np.diff(csr_offsets) < 0):
        raise RelationSchemaError("candidate offsets must be non-decreasing")
    if int(csr_offsets[-1]) != ids.size:
        raise RelationSchemaError("candidate offsets do not cover candidate IDs")
    return _sha256_arrays(poses, csr_offsets, ids)


@dataclass(frozen=True)
class CandidateSummary:
    """Digest and size summary for one explicit candidate pose sequence."""

    digest: str
    pose_count: int
    candidate_reference_count: int
    min_candidates: int
    max_candidates: int
    mean_candidates: float
    unique_candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "poseCount": self.pose_count,
            "candidateReferenceCount": self.candidate_reference_count,
            "candidateCountMin": self.min_candidates,
            "candidateCountMax": self.max_candidates,
            "candidateCountMean": self.mean_candidates,
            "uniqueCandidateCount": self.unique_candidate_count,
        }


def summarize_candidate_csr(
    candidate_ids: Any,
    pose_indices: Any,
    offsets: Any,
    num_instances: int,
) -> CandidateSummary:
    """Validate and summarize a candidate CSR in its stored pose order.

    Candidate IDs are the native frustum candidates.  This helper never reads
    visible IDs and never unions a ground-truth positive into the candidate
    set.
    """
    poses = _canonical_array(pose_indices, "<i8")
    csr_offsets = _canonical_array(offsets, "<i8")
    ids = _canonical_array(candidate_ids, "<u4")
    if int(num_instances) <= 0:
        raise RelationSchemaError("num_instances must be positive")
    if csr_offsets.size != poses.size + 1:
        raise RelationSchemaError("candidate CSR row count does not match poses")
    if poses.size and np.any(poses < 0):
        raise RelationSchemaError("candidate pose indices must be non-negative")
    if ids.size and int(ids.max()) >= int(num_instances):
        raise RelationSchemaError("candidate ID is outside num_instances")
    if csr_offsets.size and int(csr_offsets[-1]) != ids.size:
        raise RelationSchemaError("candidate CSR offsets do not cover candidate IDs")
    counts = np.diff(csr_offsets).astype(np.int64, copy=False)
    for pose_index, (start, end) in enumerate(
        zip(csr_offsets[:-1].tolist(), csr_offsets[1:].tolist(), strict=True)
    ):
        row = ids[int(start):int(end)]
        if row.size != np.unique(row).size:
            raise RelationSchemaError(
                f"candidate CSR row {pose_index} contains duplicate instance IDs"
            )
    return CandidateSummary(
        digest=candidate_digest(ids, poses, csr_offsets),
        pose_count=int(poses.size),
        candidate_reference_count=int(ids.size),
        min_candidates=int(counts.min()) if counts.size else 0,
        max_candidates=int(counts.max()) if counts.size else 0,
        mean_candidates=float(counts.mean()) if counts.size else 0.0,
        unique_candidate_count=int(np.unique(ids).size),
    )


def validate_candidate_summary(
    candidate_ids: Any,
    pose_indices: Any,
    offsets: Any,
    num_instances: int,
    expected: Mapping[str, Any],
) -> CandidateSummary:
    """Recompute a candidate summary and compare every recorded size field."""
    actual = summarize_candidate_csr(
        candidate_ids, pose_indices, offsets, num_instances
    )
    expected_digest = str(expected.get("digest", ""))
    if actual.digest != expected_digest:
        raise RelationSchemaError(
            f"candidate digest mismatch: expected {expected_digest}, got {actual.digest}"
        )
    checks = {
        "poseCount": actual.pose_count,
        "candidateReferenceCount": actual.candidate_reference_count,
        "candidateCountMin": actual.min_candidates,
        "candidateCountMax": actual.max_candidates,
        "uniqueCandidateCount": actual.unique_candidate_count,
    }
    for key, value in checks.items():
        if int(expected.get(key, -1)) != int(value):
            raise RelationSchemaError(
                f"candidate summary field {key} mismatch: "
                f"expected {expected.get(key)!r}, got {value!r}"
            )
    expected_mean = float(expected.get("candidateCountMean", float("nan")))
    if not np.isfinite(expected_mean) or not np.isclose(
        expected_mean, actual.mean_candidates, rtol=1e-6, atol=1e-6
    ):
        raise RelationSchemaError(
            "candidate summary field candidateCountMean does not match the CSR"
        )
    return actual


def _valid_sha256_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for digest in value.values():
        if not isinstance(digest, str) or len(digest) != 64:
            return False
        try:
            int(digest, 16)
        except ValueError:
            return False
    return True


def make_relation_metadata(
    *,
    num_instances: int,
    direction_bins: int,
    depth_shells: int,
    canonical_candidate_summary: CandidateSummary | Mapping[str, Any],
    render_candidate_summary: CandidateSummary | Mapping[str, Any],
    input_sha256: Mapping[str, str],
    cache_schema: str,
    max_layers: int,
    model_input_fov_y_deg: float,
    surface_fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the strict metadata envelope for a new relation CSR."""
    if int(num_instances) <= 0 or int(direction_bins) <= 0 or int(depth_shells) <= 0:
        raise RelationSchemaError("relation dimensions must be positive")
    canonical = (
        canonical_candidate_summary.to_dict()
        if isinstance(canonical_candidate_summary, CandidateSummary)
        else dict(canonical_candidate_summary)
    )
    render = (
        render_candidate_summary.to_dict()
        if isinstance(render_candidate_summary, CandidateSummary)
        else dict(render_candidate_summary)
    )
    if not _valid_sha256_mapping(input_sha256):
        raise RelationSchemaError("input_sha256 must contain SHA-256 values")
    if int(max_layers) < 2:
        raise RelationSchemaError("observed relation evidence needs at least two layers")
    if not np.isfinite(float(model_input_fov_y_deg)) or float(model_input_fov_y_deg) <= 0:
        raise RelationSchemaError("model_input_fov_y_deg must be positive")
    return {
        "schema": SCHEMA,
        "role": ROLE,
        "trainOnly": True,
        "splitNames": ["train"],
        "numInstances": int(num_instances),
        "directionBins": int(direction_bins),
        "depthShells": int(depth_shells),
        "indexOrder": list(INDEX_ORDER),
        "edgeDirectionSemantics": DEPTH_ORDERING,
        "depthOrdering": "strict_positive_source_to_target_gap",
        "sourceTopK": None,
        "sourceTruncation": SOURCE_TRUNCATION,
        "relationFeatureDim": RELATION_FEATURE_DIM,
        "relationFeatureNames": list(RELATION_FEATURE_NAMES),
        "sourceTypeNames": list(SOURCE_TYPE_NAMES),
        "canonicalCandidateSummary": canonical,
        "renderCandidateSummary": render,
        "candidateIdentity": {
            "canonicalCandidateDigest": str(canonical.get("digest", "")),
            "renderCandidateDigest": str(render.get("digest", "")),
            "canonicalPoseCount": int(canonical.get("poseCount", -1)),
            "renderPoseCount": int(render.get("poseCount", -1)),
        },
        "inputSha256": dict(input_sha256),
        "renderEvidence": {
            "cacheSchema": str(cache_schema),
            "maxLayers": int(max_layers),
            "modelInputFovYDeg": float(model_input_fov_y_deg),
            "surfaceFallback": dict(surface_fallback or {"included": False}),
        },
        "files": {
            "rowOffsets": "relation_row_offsets_uint64.bin",
            "sourceIds": "relation_source_ids_uint32.bin",
            "edgeFeatures": "relation_edge_features_fp32.bin",
            "sourceTypes": "relation_source_types_uint8.bin",
        },
    }


def make_relation_metadata_v3(
    *,
    num_instances: int,
    direction_bins: int,
    depth_shells: int,
    canonical_candidate_summary: CandidateSummary | Mapping[str, Any],
    render_candidate_summary: CandidateSummary | Mapping[str, Any],
    input_sha256: Mapping[str, str],
    cache_schema: str,
    max_layers: int,
    model_input_fov_y_deg: float,
    depth_normalization: Mapping[str, Any],
    hierarchy: Mapping[str, Any],
    evidence_topk: Mapping[str, Any],
    surface_fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the bounded v3 train-only relation contract.

    The explicit arguments are intentionally required: a v3 artifact cannot
    silently fall back to the unbounded v1 relation or to validation-derived
    depth statistics.
    """
    if int(num_instances) <= 0 or int(direction_bins) <= 0 or int(depth_shells) <= 0:
        raise RelationSchemaError("relation dimensions must be positive")
    canonical = canonical_candidate_summary.to_dict() if isinstance(canonical_candidate_summary, CandidateSummary) else dict(canonical_candidate_summary)
    render = render_candidate_summary.to_dict() if isinstance(render_candidate_summary, CandidateSummary) else dict(render_candidate_summary)
    if not _valid_sha256_mapping(input_sha256):
        raise RelationSchemaError("input_sha256 must contain SHA-256 values")
    if int(max_layers) < 2:
        raise RelationSchemaError("observed relation evidence needs at least two layers")
    if not np.isfinite(float(model_input_fov_y_deg)) or float(model_input_fov_y_deg) <= 0:
        raise RelationSchemaError("model_input_fov_y_deg must be positive")
    normalized = dict(depth_normalization)
    for key in ("q01", "q99", "epsilon"):
        if not np.isfinite(float(normalized.get(key, float("nan")))):
            raise RelationSchemaError(f"depth normalization is missing finite {key}")
    if float(normalized["q99"]) < float(normalized["q01"]):
        raise RelationSchemaError("depth normalization q99 must not be below q01")
    return {
        "schema": SCHEMA_V3,
        "role": ROLE,
        "trainOnly": True,
        "splitNames": ["train"],
        "numInstances": int(num_instances),
        "directionBins": int(direction_bins),
        "depthShells": int(depth_shells),
        "indexOrder": list(INDEX_ORDER),
        "edgeDirectionSemantics": DEPTH_ORDERING,
        "depthOrdering": "strict_positive_source_to_target_gap",
        "sourceTopK": int(evidence_topk.get("k", 0)),
        "sourceTruncation": "formal_score_topk_per_target_direction_depth",
        "relationFeatureDim": RELATION_FEATURE_DIM_V3,
        "relationFeatureNames": list(RELATION_FEATURE_NAMES_V3),
        "sourceTypeNames": list(SOURCE_TYPE_NAMES),
        "canonicalCandidateSummary": canonical,
        "renderCandidateSummary": render,
        "candidateIdentity": {
            "canonicalCandidateDigest": str(canonical.get("digest", "")),
            "renderCandidateDigest": str(render.get("digest", "")),
            "canonicalPoseCount": int(canonical.get("poseCount", -1)),
            "renderPoseCount": int(render.get("poseCount", -1)),
        },
        "inputSha256": dict(input_sha256),
        "renderEvidence": {
            "cacheSchema": str(cache_schema),
            "maxLayers": int(max_layers),
            "modelInputFovYDeg": float(model_input_fov_y_deg),
            "surfaceFallback": dict(surface_fallback or {"included": False}),
            "eventAndCensorMayCoexist": True,
            "pixelWeightPolicy": "per_render_subpose_sum_to_one",
        },
        "depthNormalization": normalized,
        "hierarchy": dict(hierarchy),
        "evidenceTopK": dict(evidence_topk),
        "files": {
            "rowOffsets": "relation_v3_row_offsets_uint64.bin",
            "sourceIds": "relation_v3_source_ids_uint32.bin",
            "edgeFeatures": "relation_v3_edge_features_fp32.bin",
            "sourceTypes": "relation_v3_source_types_uint8.bin",
        },
    }


def validate_relation_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_num_instances: int | None = None,
    expected_candidate_digest: str | None = None,
    expected_render_candidate_digest: str | None = None,
) -> None:
    """Validate schema, provenance, split ownership, and feature semantics."""
    if not isinstance(metadata, Mapping):
        raise RelationSchemaError("relation metadata must be a mapping")
    if metadata.get("schema") != SCHEMA:
        raise RelationSchemaError(
            f"expected {SCHEMA}, got {metadata.get('schema')!r}"
        )
    if metadata.get("role") != ROLE or metadata.get("trainOnly") is not True:
        raise RelationSchemaError("relation CSR must be explicitly marked train-only")
    if metadata.get("splitNames") != ["train"]:
        raise RelationSchemaError("relation CSR may only contain the train split")
    for key in ("numInstances", "directionBins", "depthShells"):
        if int(metadata.get(key, 0)) <= 0:
            raise RelationSchemaError(f"metadata field {key} must be positive")
    if expected_num_instances is not None and int(metadata["numInstances"]) != int(
        expected_num_instances
    ):
        raise RelationSchemaError("relation numInstances does not match the model")
    if tuple(metadata.get("indexOrder", ())) != INDEX_ORDER:
        raise RelationSchemaError("relation CSR index order is not target/direction/depth")
    if metadata.get("edgeDirectionSemantics") != DEPTH_ORDERING:
        raise RelationSchemaError("relation edge direction semantics are invalid")
    if metadata.get("depthOrdering") != "strict_positive_source_to_target_gap":
        raise RelationSchemaError("relation depth ordering is not strict source-to-target")
    if metadata.get("sourceTopK") is not None:
        raise RelationSchemaError("the training relation CSR must not use source top-k truncation")
    if metadata.get("sourceTruncation") != SOURCE_TRUNCATION:
        raise RelationSchemaError("relation source truncation policy is invalid")
    if int(metadata.get("relationFeatureDim", -1)) != RELATION_FEATURE_DIM:
        raise RelationSchemaError("relation feature dimension does not match schema v1")
    if tuple(metadata.get("relationFeatureNames", ())) != RELATION_FEATURE_NAMES:
        raise RelationSchemaError("relation feature names do not match schema v1")
    if tuple(metadata.get("sourceTypeNames", ())) != SOURCE_TYPE_NAMES:
        raise RelationSchemaError("relation source type names do not match schema v1")
    if not _valid_sha256_mapping(metadata.get("inputSha256")):
        raise RelationSchemaError("relation metadata is missing input SHA-256 provenance")
    evidence = metadata.get("renderEvidence")
    if not isinstance(evidence, Mapping):
        raise RelationSchemaError("relation metadata is missing render evidence provenance")
    if int(evidence.get("maxLayers", 0)) < 2:
        raise RelationSchemaError("render evidence must contain at least two depth layers")
    if not np.isfinite(float(evidence.get("modelInputFovYDeg", float("nan")))):
        raise RelationSchemaError("render evidence FOV is not finite")

    canonical = metadata.get("canonicalCandidateSummary")
    render = metadata.get("renderCandidateSummary")
    if not isinstance(canonical, Mapping) or not isinstance(render, Mapping):
        raise RelationSchemaError("relation metadata is missing candidate summaries")
    identity = metadata.get("candidateIdentity")
    if not isinstance(identity, Mapping):
        raise RelationSchemaError("relation metadata is missing candidate identity")
    canonical_digest = str(canonical.get("digest", ""))
    render_digest = str(render.get("digest", ""))
    if len(canonical_digest) != 64 or len(render_digest) != 64:
        raise RelationSchemaError("candidate summaries must contain SHA-256 digests")
    if identity.get("canonicalCandidateDigest") != canonical_digest:
        raise RelationSchemaError("canonical candidate identity does not match its summary")
    if identity.get("renderCandidateDigest") != render_digest:
        raise RelationSchemaError("render candidate identity does not match its summary")
    if expected_candidate_digest is not None and canonical_digest != expected_candidate_digest:
        raise RelationSchemaError("canonical candidate digest does not match expectation")
    if (
        expected_render_candidate_digest is not None
        and render_digest != expected_render_candidate_digest
    ):
        raise RelationSchemaError("render candidate digest does not match expectation")
    files = metadata.get("files")
    if not isinstance(files, Mapping) or not all(
        isinstance(files.get(key), str)
        for key in ("rowOffsets", "sourceIds", "edgeFeatures", "sourceTypes")
    ):
        raise RelationSchemaError("relation metadata has incomplete binary file names")


def validate_relation_metadata_v3(
    metadata: Mapping[str, Any],
    *,
    expected_num_instances: int | None = None,
    expected_candidate_digest: str | None = None,
    expected_render_candidate_digest: str | None = None,
) -> None:
    """Validate the new bounded evidence-aware schema without accepting v1."""
    if metadata.get("schema") != SCHEMA_V3:
        raise RelationSchemaError(
            f"expected {SCHEMA_V3}, got {metadata.get('schema')!r}"
        )
    if metadata.get("role") != ROLE or metadata.get("trainOnly") is not True:
        raise RelationSchemaError("v3 relation CSR must be explicitly train-only")
    if metadata.get("splitNames") != ["train"]:
        raise RelationSchemaError("v3 relation CSR may only contain train")
    for key in ("numInstances", "directionBins", "depthShells"):
        if int(metadata.get(key, 0)) <= 0:
            raise RelationSchemaError(f"metadata field {key} must be positive")
    if expected_num_instances is not None and int(metadata["numInstances"]) != int(expected_num_instances):
        raise RelationSchemaError("v3 numInstances does not match the model")
    if tuple(metadata.get("indexOrder", ())) != INDEX_ORDER:
        raise RelationSchemaError("v3 index order is not target/direction/depth")
    if metadata.get("edgeDirectionSemantics") != DEPTH_ORDERING:
        raise RelationSchemaError("v3 edge direction semantics are invalid")
    if metadata.get("depthOrdering") != "strict_positive_source_to_target_gap":
        raise RelationSchemaError("v3 depth ordering is not strict source-to-target")
    if int(metadata.get("sourceTopK", 0)) <= 0 or metadata.get("sourceTruncation") != "formal_score_topk_per_target_direction_depth":
        raise RelationSchemaError("v3 relation rows must declare formal top-k truncation")
    if int(metadata.get("relationFeatureDim", -1)) != RELATION_FEATURE_DIM_V3:
        raise RelationSchemaError("v3 relation feature dimension is invalid")
    if tuple(metadata.get("relationFeatureNames", ())) != RELATION_FEATURE_NAMES_V3:
        raise RelationSchemaError("v3 relation feature names are invalid")
    if tuple(metadata.get("sourceTypeNames", ())) != SOURCE_TYPE_NAMES:
        raise RelationSchemaError("v3 source type names are invalid")
    if not _valid_sha256_mapping(metadata.get("inputSha256")):
        raise RelationSchemaError("v3 relation metadata is missing input provenance")
    evidence = metadata.get("renderEvidence")
    if not isinstance(evidence, Mapping) or evidence.get("eventAndCensorMayCoexist") is not True:
        raise RelationSchemaError("v3 evidence must declare event/censor coexistence")
    if evidence.get("pixelWeightPolicy") != "per_render_subpose_sum_to_one":
        raise RelationSchemaError("v3 evidence must use per-subpose conserved weights")
    normalization = metadata.get("depthNormalization")
    if not isinstance(normalization, Mapping):
        raise RelationSchemaError("v3 relation metadata is missing depth normalization")
    for key in ("q01", "q99", "epsilon"):
        if not np.isfinite(float(normalization.get(key, float("nan")))):
            raise RelationSchemaError(f"v3 depth normalization lacks finite {key}")
    if float(normalization["q99"]) < float(normalization["q01"]):
        raise RelationSchemaError("v3 depth normalization q99 < q01")
    hierarchy = metadata.get("hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise RelationSchemaError("v3 relation metadata is missing bounded hierarchy")
    if not 0 < int(hierarchy.get("localMaxSize", 0)) <= 32:
        raise RelationSchemaError("v3 local hierarchy cap is invalid")
    if not 0 < int(hierarchy.get("structuralMaxLocalGroups", 0)) <= 64:
        raise RelationSchemaError("v3 hierarchy exceeds the declared group caps")
    if not 0.0 < float(hierarchy.get("maxStructuralFraction", 1.0)) <= 0.10 + 1e-9:
        raise RelationSchemaError("v3 structural groups may not exceed ten percent")
    for key in ("localDiameter", "structuralDiameter", "maxLocalDiameter", "maxStructuralDiameter"):
        if not np.isfinite(float(hierarchy.get(key, float("nan")))) or float(hierarchy[key]) < 0.0:
            raise RelationSchemaError(f"v3 hierarchy has invalid {key}")
    if float(hierarchy["localDiameter"]) <= 0.0 or float(hierarchy["structuralDiameter"]) <= 0.0:
        raise RelationSchemaError("v3 hierarchy diameter limits must be positive")
    if float(hierarchy["maxLocalDiameter"]) > float(hierarchy["localDiameter"]) + 1e-5:
        raise RelationSchemaError("v3 local hierarchy exceeds its spatial diameter")
    if float(hierarchy["maxStructuralDiameter"]) > float(hierarchy["structuralDiameter"]) + 1e-5:
        raise RelationSchemaError("v3 structural hierarchy exceeds its spatial diameter")
    if int(hierarchy.get("maxLocalInstanceCount", 0)) > int(hierarchy["localMaxSize"]):
        raise RelationSchemaError("v3 local hierarchy exceeds its instance cap")
    if int(hierarchy.get("maxStructuralLocalGroupCount", 0)) > int(hierarchy["structuralMaxLocalGroups"]):
        raise RelationSchemaError("v3 structural hierarchy exceeds its local-group cap")
    if float(hierarchy.get("maxStructuralInstanceFraction", 1.0)) > float(hierarchy["maxStructuralFraction"]) + 1e-9:
        raise RelationSchemaError("v3 structural hierarchy exceeds its instance fraction cap")
    topk = metadata.get("evidenceTopK")
    if not isinstance(topk, Mapping) or int(topk.get("k", 0)) <= 0:
        raise RelationSchemaError("v3 evidence top-k metadata is missing")
    canonical = metadata.get("canonicalCandidateSummary")
    render = metadata.get("renderCandidateSummary")
    identity = metadata.get("candidateIdentity")
    if not isinstance(canonical, Mapping) or not isinstance(render, Mapping) or not isinstance(identity, Mapping):
        raise RelationSchemaError("v3 relation metadata is missing candidate identity")
    canonical_digest = str(canonical.get("digest", ""))
    render_digest = str(render.get("digest", ""))
    if len(canonical_digest) != 64 or len(render_digest) != 64:
        raise RelationSchemaError("v3 candidate summaries must contain SHA-256 digests")
    if identity.get("canonicalCandidateDigest") != canonical_digest or identity.get("renderCandidateDigest") != render_digest:
        raise RelationSchemaError("v3 candidate identity does not match summaries")
    if expected_candidate_digest is not None and canonical_digest != expected_candidate_digest:
        raise RelationSchemaError("v3 canonical candidate digest mismatch")
    if expected_render_candidate_digest is not None and render_digest != expected_render_candidate_digest:
        raise RelationSchemaError("v3 render candidate digest mismatch")
    files = metadata.get("files")
    if not isinstance(files, Mapping) or not all(isinstance(files.get(k), str) for k in ("rowOffsets", "sourceIds", "edgeFeatures", "sourceTypes")):
        raise RelationSchemaError("v3 relation metadata has incomplete binary files")


def validate_survival_observations_v3(
    observations: Mapping[str, Any],
    *,
    num_instances: int,
    direction_bins: int,
    require_conservation: bool = True,
) -> dict[str, int | float]:
    """Validate v3 event/censor records and conserved per-subpose weights."""
    required = ("instance", "direction", "normalizedDepth", "event", "weight", "subpose", "rawPixelCount")
    if not isinstance(observations, Mapping) or any(k not in observations for k in required):
        raise RelationSchemaError("v3 observations need subpose, rawPixelCount and event/censor fields")
    arrays = {k: np.asarray(observations[k]).reshape(-1) for k in required}
    if len({a.size for a in arrays.values()}) != 1:
        raise RelationSchemaError("v3 observation arrays have different lengths")
    instance = arrays["instance"].astype(np.int64, copy=False)
    direction = arrays["direction"].astype(np.int64, copy=False)
    normalized_depth = arrays["normalizedDepth"].astype(np.float64, copy=False)
    event = arrays["event"].astype(np.int64, copy=False)
    weight = arrays["weight"].astype(np.float64, copy=False)
    subpose = arrays["subpose"].astype(np.int64, copy=False)
    pixels = arrays["rawPixelCount"].astype(np.float64, copy=False)
    if instance.size and (int(instance.min()) < 0 or int(instance.max()) >= int(num_instances)):
        raise RelationSchemaError("v3 observation instance is outside num_instances")
    if direction.size and (int(direction.min()) < 0 or int(direction.max()) >= int(direction_bins)):
        raise RelationSchemaError("v3 observation direction is outside direction_bins")
    if np.any(~np.isfinite(normalized_depth)) or np.any((normalized_depth < 0) | (normalized_depth > 1)):
        raise RelationSchemaError("v3 normalizedDepth must be finite and in [0, 1]")
    if np.any((event != 0) & (event != 1)):
        raise RelationSchemaError("v3 event must be 0 or 1")
    if np.any(subpose < 0) or np.any(pixels <= 0) or np.any(~np.isfinite(pixels)):
        raise RelationSchemaError("v3 subpose and raw pixels are invalid")
    if np.any(weight < 0) or np.any(~np.isfinite(weight)):
        raise RelationSchemaError("v3 pixel weights must be finite and non-negative")
    if require_conservation and subpose.size:
        order = np.argsort(subpose, kind="stable")
        sorted_pose = subpose[order]
        sorted_weight = weight[order]
        starts = np.r_[
            0,
            np.flatnonzero(sorted_pose[1:] != sorted_pose[:-1]) + 1,
        ]
        pose_weights = np.add.reduceat(sorted_weight, starts)
        valid = np.isclose(pose_weights, 1.0, rtol=1e-4, atol=2e-3)
        if not bool(np.all(valid)):
            failed = int(np.flatnonzero(~valid)[0])
            raise RelationSchemaError(
                "v3 pixel weights do not conserve subpose "
                f"{int(sorted_pose[starts[failed]])}: {float(pose_weights[failed])}"
            )
        unique_subpose_count = int(starts.size)
    else:
        unique_subpose_count = int(np.unique(subpose).size)
    return {
        "observationCount": int(instance.size),
        "eventCount": int(np.sum(event == 1)),
        "rightCensoredCount": int(np.sum(event == 0)),
        "subposeCount": unique_subpose_count,
        "weightSum": float(weight.sum()),
    }


def load_survival_observations_v3(directory: str | Path, metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Load the v3 observation sidecar; never reinterpret normalized depth as rho."""
    root = Path(directory)
    envelope = metadata.get("survivalObservations") if isinstance(metadata, Mapping) else None
    if not isinstance(envelope, Mapping) or envelope.get("schema") != "pvs-viewcell-train-observed-survival-censoring-v3":
        raise RelationSchemaError("not a v3 train-only survival observation envelope")
    files = envelope.get("files")
    required = ("instance", "direction", "normalizedDepth", "event", "weight", "subpose", "rawPixelCount")
    if not isinstance(files, Mapping) or any(not isinstance(files.get(key), str) for key in required):
        raise RelationSchemaError("v3 survival envelope is missing normalizedDepth files")
    arrays = {
        "instance": np.fromfile(root / files["instance"], dtype="<u4"),
        "direction": np.fromfile(root / files["direction"], dtype="<u1"),
        "normalizedDepth": np.fromfile(root / files["normalizedDepth"], dtype="<f2").astype(np.float32),
        "event": np.fromfile(root / files["event"], dtype="<u1"),
        "weight": np.fromfile(root / files["weight"], dtype="<f2").astype(np.float32),
        "subpose": np.fromfile(root / files["subpose"], dtype="<u4"),
        "rawPixelCount": np.fromfile(root / files["rawPixelCount"], dtype="<f4"),
    }
    if "confidence" in files:
        arrays["confidence"] = np.fromfile(root / files["confidence"], dtype="<f2").astype(np.float32)
    if "evidenceLevel" in files:
        arrays["evidenceLevel"] = np.fromfile(root / files["evidenceLevel"], dtype="<u1")
    if "rawDepth" in files:
        arrays["rawDepth"] = np.fromfile(root / files["rawDepth"], dtype="<f4")
    validate_survival_observations_v3(
        arrays,
        num_instances=int(metadata["numInstances"]),
        direction_bins=int(metadata["directionBins"]),
    )
    return arrays


def radius_relative_log_depth(depth: Any, radius: Any, *, epsilon: float = 1e-6) -> np.ndarray:
    """Return log1p(depth / radius), the train-only normalization primitive."""
    d = np.asarray(depth, dtype=np.float64)
    r = np.maximum(np.asarray(radius, dtype=np.float64), float(epsilon))
    if np.any(~np.isfinite(d)) or np.any(d < 0):
        raise RelationSchemaError("depth must be finite and non-negative")
    return np.log1p(d / r)


def train_depth_quantiles(values: Any, *, epsilon: float = 1e-6) -> dict[str, float | str]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise RelationSchemaError("train depth values must be finite and non-negative")
    return {
        "definition": "log1p(depth / max(target_radius, epsilon))",
        "q01": float(np.quantile(values, 0.01)),
        "q99": float(np.quantile(values, 0.99)),
        "epsilon": float(epsilon),
        "sourceSplit": "train",
        "clipRange": [0.0, 1.0],
    }


def normalize_radius_relative_log_depth(values: Any, stats: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    q01, q99 = float(stats["q01"]), float(stats["q99"])
    return np.clip((values - q01) / max(q99 - q01, 1e-12), 0.0, 1.0).astype(np.float32)


def recompute_relative_geometry(
    source_ids: Any,
    target_ids: Any,
    centers: Any,
    radii: Any,
    depths: Any,
    depth_stats: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Recompute directed geometry after an edge permutation.

    This function is deliberately separate from edge swapping so callers
    cannot accidentally retain the old source-target geometry after a
    degree-preserving shuffle.
    """
    source = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    target = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    points = np.asarray(centers, dtype=np.float64)
    radius = np.maximum(np.asarray(radii, dtype=np.float64), 1e-6)
    depth = np.asarray(depths, dtype=np.float64).reshape(-1)
    if source.size != target.size or points.ndim != 2 or points.shape[1] != 3:
        raise RelationSchemaError("relative geometry arrays have incompatible shapes")
    delta = points[source] - points[target]
    gap = np.maximum(depth[target] - depth[source], 0.0)
    relative_log_depth = radius_relative_log_depth(depth[target], radius[target])
    result = {
        "relative_center": delta.astype(np.float32),
        "relative_log_scale": np.log(radius[source] / radius[target]).astype(np.float32),
        "depth_gap": gap.astype(np.float32),
        "relative_log_depth": relative_log_depth.astype(np.float32),
    }
    if depth_stats is not None:
        result["normalized_log_depth"] = normalize_radius_relative_log_depth(relative_log_depth, depth_stats)
    return result


def degree_preserving_directed_edge_swap(
    source_ids: Any,
    target_ids: Any,
    direction_ids: Any,
    depth_shell_ids: Any,
    *,
    distance_bucket_ids: Any | None = None,
    seed: int = 0,
    strata: tuple[int, int] | None = None,
    swap_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Swap directed targets within strata while preserving in/out degrees."""
    sources = np.asarray(source_ids, dtype=np.int64).reshape(-1).copy()
    targets = np.asarray(target_ids, dtype=np.int64).reshape(-1).copy()
    directions = np.asarray(direction_ids, dtype=np.int64).reshape(-1)
    shells = np.asarray(depth_shell_ids, dtype=np.int64).reshape(-1)
    distance_buckets = (
        np.zeros_like(directions)
        if distance_bucket_ids is None
        else np.asarray(distance_bucket_ids, dtype=np.int64).reshape(-1)
    )
    if not (sources.size == targets.size == directions.size == shells.size == distance_buckets.size):
        raise RelationSchemaError("edge swap arrays have different lengths")
    if not np.isfinite(float(swap_fraction)) or not 0.0 <= float(swap_fraction) <= 1.0:
        raise RelationSchemaError("swap_fraction must be in [0, 1]")
    if np.any(distance_buckets < 0):
        raise RelationSchemaError("distance buckets must be non-negative")
    if np.any(sources == targets):
        raise RelationSchemaError("edge swap input contains self-loop")
    if np.unique(np.column_stack((sources, targets, directions, shells)), axis=0).shape[0] != sources.size:
        raise RelationSchemaError("edge swap input contains duplicate stratified edges")
    rng = np.random.default_rng(int(seed))
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (direction, shell, bucket) in enumerate(
        zip(directions.tolist(), shells.tolist(), distance_buckets.tolist(), strict=True)
    ):
        pair = (direction, shell)
        if strata is None or pair == tuple(strata):
            groups[(direction, shell, bucket)].append(index)
    desired = max(0, int(round(float(swap_fraction) * sum(len(v) for v in groups.values()) / 2.0)))
    swaps = 0
    edge_set = set(zip(sources.tolist(), targets.tolist(), directions.tolist(), shells.tolist()))
    for indices in groups.values():
        if swaps >= desired or len(indices) < 2:
            continue
        order = list(indices)
        rng.shuffle(order)
        for left, right in zip(order[::2], order[1::2]):
            if swaps >= desired:
                break
            new_left, new_right = int(targets[right]), int(targets[left])
            if sources[left] == new_left or sources[right] == new_right:
                continue
            old_a = (int(sources[left]), int(targets[left]), int(directions[left]), int(shells[left]))
            old_b = (int(sources[right]), int(targets[right]), int(directions[right]), int(shells[right]))
            a = (int(sources[left]), new_left, int(directions[left]), int(shells[left]))
            b = (int(sources[right]), new_right, int(directions[right]), int(shells[right]))
            edge_set.remove(old_a)
            edge_set.remove(old_b)
            if a in edge_set or b in edge_set or a == b:
                edge_set.add(old_a)
                edge_set.add(old_b)
                continue
            targets[left], targets[right] = new_left, new_right
            edge_set.add(a)
            edge_set.add(b)
            swaps += 1
    if np.any(sources == targets) or np.unique(np.column_stack((sources, targets, directions, shells)), axis=0).shape[0] != sources.size:
        raise RelationSchemaError("edge swap produced an invalid directed edge set")
    return sources.astype(np.uint32), targets.astype(np.uint32)


def bounded_hierarchy_ids(
    num_instances: int,
    target_ids: Any,
    source_ids: Any,
    confidence: Any,
    centers: Any | None = None,
    *,
    local_max_size: int = 32,
    local_diameter: float,
    structural_max_size: int = 64,
    structural_diameter: float,
    max_structural_fraction: float = 0.10,
    local_threshold: float = 0.0,
    structural_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically pack evidence edges into bounded contiguous groups."""
    n = int(num_instances)
    if n <= 0 or local_max_size <= 0 or structural_max_size <= 0:
        raise RelationSchemaError("hierarchy dimensions must be positive")
    target = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    source = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    score = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if not (target.size == source.size == score.size):
        raise RelationSchemaError("hierarchy edge arrays have different lengths")
    if np.any(target < 0) or np.any(source < 0) or np.any(target >= n) or np.any(source >= n):
        raise RelationSchemaError("hierarchy edge ID is outside num_instances")
    points = None if centers is None else np.asarray(centers, dtype=np.float64)
    if points is not None and points.shape != (n, 3):
        raise RelationSchemaError("hierarchy centers must have shape (num_instances, 3)")
    if points is None:
        raise RelationSchemaError("bounded hierarchy requires instance centers")
    if not np.isfinite(float(local_diameter)) or float(local_diameter) <= 0.0:
        raise RelationSchemaError("local hierarchy diameter must be finite and positive")
    if not np.isfinite(float(structural_diameter)) or float(structural_diameter) <= 0.0:
        raise RelationSchemaError("structural hierarchy diameter must be finite and positive")
    if float(structural_diameter) < float(local_diameter):
        raise RelationSchemaError("structural hierarchy diameter must not be below local diameter")
    if not 0.0 < float(max_structural_fraction) <= 0.10:
        raise RelationSchemaError("max_structural_fraction must be in (0, 0.10]")
    max_structural_instances = max(1, int(np.floor(float(n) * float(max_structural_fraction))))
    effective_local_max = min(int(local_max_size), max_structural_instances)
    parent = np.arange(n, dtype=np.int64)
    member_count: dict[int, int] = {i: 1 for i in range(n)}
    lower: dict[int, np.ndarray] = {i: points[i].copy() for i in range(n)}
    upper: dict[int, np.ndarray] = {i: points[i].copy() for i in range(n)}
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    def merged_diameter(left: int, right: int) -> float:
        merged_lower = np.minimum(lower[left], lower[right])
        merged_upper = np.maximum(upper[left], upper[right])
        return float(np.linalg.norm(merged_upper - merged_lower))
    order = np.argsort(-score, kind="stable")
    for index in order.tolist():
        if score[index] < float(local_threshold):
            continue
        left, right = find(int(target[index])), find(int(source[index]))
        if left == right:
            continue
        if member_count[left] + member_count[right] > effective_local_max:
            continue
        if merged_diameter(left, right) > float(local_diameter):
            continue
        parent[right] = left
        member_count[left] += member_count[right]
        lower[left] = np.minimum(lower[left], lower[right])
        upper[left] = np.maximum(upper[left], upper[right])
        member_count.pop(right, None)
        lower.pop(right, None)
        upper.pop(right, None)
    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    _, local = np.unique(roots, return_inverse=True)
    local = local.astype(np.uint32)
    local_count = int(local.max()) + 1 if local.size else 0
    local_sizes = np.bincount(local, minlength=local_count).astype(np.int64, copy=False)
    lparent = np.arange(local_count, dtype=np.int64)
    lgroup_count: dict[int, int] = {i: 1 for i in range(local_count)}
    linstance_count: dict[int, int] = {i: int(local_sizes[i]) for i in range(local_count)}
    local_lower = np.full((local_count, 3), np.inf, dtype=np.float64)
    local_upper = np.full((local_count, 3), -np.inf, dtype=np.float64)
    np.minimum.at(local_lower, local.astype(np.int64, copy=False), points)
    np.maximum.at(local_upper, local.astype(np.int64, copy=False), points)
    llower: dict[int, np.ndarray] = {i: local_lower[i] for i in range(local_count)}
    lupper: dict[int, np.ndarray] = {i: local_upper[i] for i in range(local_count)}
    for index in order.tolist():
        if score[index] < float(structural_threshold):
            continue
        left, right = int(local[int(target[index])]), int(local[int(source[index])])
        while lparent[left] != left:
            lparent[left] = lparent[lparent[left]]; left = int(lparent[left])
        while lparent[right] != right:
            lparent[right] = lparent[lparent[right]]; right = int(lparent[right])
        if left == right:
            continue
        if lgroup_count[left] + lgroup_count[right] > int(structural_max_size):
            continue
        if linstance_count[left] + linstance_count[right] > max_structural_instances:
            continue
        merged_lower = np.minimum(llower[left], llower[right])
        merged_upper = np.maximum(lupper[left], lupper[right])
        if float(np.linalg.norm(merged_upper - merged_lower)) > float(structural_diameter):
            continue
        lparent[right] = left
        lgroup_count[left] += lgroup_count[right]
        linstance_count[left] += linstance_count[right]
        llower[left] = merged_lower
        lupper[left] = merged_upper
        for table in (lgroup_count, linstance_count, llower, lupper):
            table.pop(right, None)
    structural_roots = np.arange(local_count, dtype=np.int64)
    for i in range(local_count):
        root = i
        while lparent[root] != root:
            lparent[root] = lparent[lparent[root]]
            root = int(lparent[root])
        structural_roots[i] = root
    _, local_to_structural = np.unique(structural_roots, return_inverse=True)
    return local, local_to_structural.astype(np.uint32)


def summarize_bounded_hierarchy(
    local_group_ids: Any,
    local_to_structural_group_ids: Any,
    centers: Any,
) -> dict[str, int | float]:
    """Return measured hierarchy bounds for metadata and hard validation."""
    local = np.asarray(local_group_ids, dtype=np.int64).reshape(-1)
    structural = np.asarray(local_to_structural_group_ids, dtype=np.int64).reshape(-1)
    points = np.asarray(centers, dtype=np.float64)
    if points.shape != (local.size, 3) or local.size == 0:
        raise RelationSchemaError("hierarchy summary requires one center and local ID per instance")
    local_count = int(local.max()) + 1
    if np.unique(local).tolist() != list(range(local_count)):
        raise RelationSchemaError("local hierarchy IDs must be contiguous")
    if structural.size != local_count:
        raise RelationSchemaError("structural hierarchy must contain one ID per local group")
    structural_count = int(structural.max()) + 1
    if np.unique(structural).tolist() != list(range(structural_count)):
        raise RelationSchemaError("structural hierarchy IDs must be contiguous")

    local_sizes = np.bincount(local, minlength=local_count)
    local_lower = np.full((local_count, 3), np.inf, dtype=np.float64)
    local_upper = np.full((local_count, 3), -np.inf, dtype=np.float64)
    np.minimum.at(local_lower, local, points)
    np.maximum.at(local_upper, local, points)
    local_diameters = np.linalg.norm(local_upper - local_lower, axis=1)
    structural_local_sizes = np.bincount(structural, minlength=structural_count)
    instance_structural = structural[local]
    structural_instance_sizes = np.bincount(instance_structural, minlength=structural_count)
    structural_lower = np.full((structural_count, 3), np.inf, dtype=np.float64)
    structural_upper = np.full((structural_count, 3), -np.inf, dtype=np.float64)
    np.minimum.at(structural_lower, instance_structural, points)
    np.maximum.at(structural_upper, instance_structural, points)
    structural_diameters = np.linalg.norm(structural_upper - structural_lower, axis=1)
    return {
        "localGroupCount": local_count,
        "structuralGroupCount": structural_count,
        "maxLocalInstanceCount": int(local_sizes.max()),
        "maxLocalDiameter": float(local_diameters.max()),
        "maxStructuralLocalGroupCount": int(structural_local_sizes.max()),
        "maxStructuralInstanceCount": int(structural_instance_sizes.max()),
        "maxStructuralInstanceFraction": float(structural_instance_sizes.max() / local.size),
        "maxStructuralDiameter": float(structural_diameters.max()),
    }


def validate_survival_observations(
    observations: Mapping[str, Any],
    *,
    num_instances: int,
    direction_bins: int,
) -> dict[str, int]:
    """Validate event/right-censor records copied beside the relation CSR."""
    required = ("instance", "direction", "rho", "event", "weight")
    if not isinstance(observations, Mapping) or any(key not in observations for key in required):
        raise RelationSchemaError(
            "survival observations need instance/direction/rho/event/weight arrays"
        )
    arrays = {
        key: np.asarray(observations[key]).reshape(-1)
        for key in required
    }
    lengths = {int(value.size) for value in arrays.values()}
    if len(lengths) != 1:
        raise RelationSchemaError("survival observation arrays have different lengths")
    instance = arrays["instance"].astype(np.int64, copy=False)
    direction = arrays["direction"].astype(np.int64, copy=False)
    rho = arrays["rho"].astype(np.float64, copy=False)
    event = arrays["event"].astype(np.int64, copy=False)
    weight = arrays["weight"].astype(np.float64, copy=False)
    if instance.size and (int(instance.min()) < 0 or int(instance.max()) >= int(num_instances)):
        raise RelationSchemaError("survival observation instance is outside num_instances")
    if direction.size and (int(direction.min()) < 0 or int(direction.max()) >= int(direction_bins)):
        raise RelationSchemaError("survival observation direction exceeds direction bins")
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0) or np.any(rho > 1.0):
        raise RelationSchemaError("survival rho must be finite and in [0, 1]")
    if np.any((event != 0) & (event != 1)):
        raise RelationSchemaError("survival event must be 0 (right-censored) or 1 (event)")
    if np.any(~np.isfinite(weight)) or np.any(weight <= 0.0):
        raise RelationSchemaError("survival observation weights must be positive")
    event_count = int(np.sum(event == 1))
    return {
        "observationCount": int(instance.size),
        "eventCount": event_count,
        "rightCensoredCount": int(instance.size - event_count),
    }


@dataclass
class ObservedRelationCSR:
    """In-memory representation of the v1 training relation CSR."""

    num_instances: int
    direction_bins: int
    depth_shells: int
    row_offsets: np.ndarray
    source_ids: np.ndarray
    edge_features: np.ndarray
    source_types: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.num_instances = int(self.num_instances)
        self.direction_bins = int(self.direction_bins)
        self.depth_shells = int(self.depth_shells)
        self.row_offsets = _canonical_array(self.row_offsets, "<u8")
        self.source_ids = _canonical_array(self.source_ids, "<u4")
        self.edge_features = np.ascontiguousarray(
            np.asarray(self.edge_features, dtype="<f4")
        )
        self.source_types = _canonical_array(self.source_types, "<u1")
        self.metadata = dict(self.metadata)

    @property
    def row_count(self) -> int:
        return self.num_instances * self.direction_bins * self.depth_shells

    @property
    def edge_count(self) -> int:
        return int(self.source_ids.size)

    @classmethod
    def from_rows(
        cls,
        *,
        num_instances: int,
        direction_bins: int,
        depth_shells: int,
        target_ids: Any,
        direction_ids: Any,
        depth_shell_ids: Any,
        source_ids: Any,
        edge_features: Any,
        source_types: Any | None,
        metadata: Mapping[str, Any],
    ) -> "ObservedRelationCSR":
        """Pack unsorted edge rows without silently merging duplicates."""
        targets = _canonical_array(target_ids, "<i8")
        directions = _canonical_array(direction_ids, "<i8")
        shells = _canonical_array(depth_shell_ids, "<i8")
        sources = _canonical_array(source_ids, "<u4")
        features = np.ascontiguousarray(np.asarray(edge_features, dtype="<f4"))
        types = np.zeros((sources.size,), dtype="<u1") if source_types is None else _canonical_array(source_types, "<u1")
        lengths = {targets.size, directions.size, shells.size, sources.size, types.size}
        if len(lengths) != 1 or features.ndim != 2 or features.shape[0] != sources.size:
            raise RelationSchemaError("relation edge row arrays have inconsistent lengths")
        expected_names, expected_dim = _feature_schema(metadata)
        if features.shape[1] != expected_dim:
            raise RelationSchemaError(
                f"relation edge features do not match {metadata.get('schema')}: {expected_dim}"
            )
        if targets.size:
            if int(targets.min()) < 0 or int(targets.max()) >= int(num_instances):
                raise RelationSchemaError("relation target is outside num_instances")
            if int(directions.min()) < 0 or int(directions.max()) >= int(direction_bins):
                raise RelationSchemaError("relation direction is outside direction_bins")
            if int(shells.min()) < 0 or int(shells.max()) >= int(depth_shells):
                raise RelationSchemaError("relation depth shell is outside depth_shells")
            if np.any(sources >= int(num_instances)):
                raise RelationSchemaError("relation source is outside num_instances")
            if np.any(sources.astype(np.int64) == targets):
                raise RelationSchemaError("relation source and target must differ")
        order = np.lexsort((sources, shells, directions, targets))
        targets, directions, shells = targets[order], directions[order], shells[order]
        sources, features, types = sources[order], features[order], types[order]
        keys = np.column_stack((targets, directions, shells, sources))
        if keys.shape[0] > 1 and np.any(np.all(keys[1:] == keys[:-1], axis=1)):
            raise RelationSchemaError("duplicate target/direction/shell/source relation row")
        row_ids = targets * (int(direction_bins) * int(depth_shells)) + directions * int(depth_shells) + shells
        offsets = np.zeros((int(num_instances) * int(direction_bins) * int(depth_shells) + 1,), dtype="<u8")
        if row_ids.size:
            counts = np.bincount(row_ids.astype(np.int64), minlength=offsets.size - 1)
            offsets[1:] = np.cumsum(counts, dtype=np.uint64)
        result = cls(
            num_instances=int(num_instances),
            direction_bins=int(direction_bins),
            depth_shells=int(depth_shells),
            row_offsets=offsets,
            source_ids=sources,
            edge_features=features,
            source_types=types,
            metadata=dict(metadata),
        )
        result.validate()
        return result

    def row_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return target, direction, and shell IDs for every stored edge."""
        if self.row_offsets.size != self.row_count + 1:
            raise RelationSchemaError("relation offsets have an invalid row count")
        row_ids = np.repeat(
            np.arange(self.row_count, dtype=np.int64),
            np.diff(self.row_offsets).astype(np.int64, copy=False),
        )
        target = row_ids // (self.direction_bins * self.depth_shells)
        rem = row_ids % (self.direction_bins * self.depth_shells)
        direction = rem // self.depth_shells
        shell = rem % self.depth_shells
        return target, direction, shell

    def validate(self, *, expected_candidate_digest: str | None = None) -> None:
        validator = validate_relation_metadata_v3 if self.metadata.get("schema") == SCHEMA_V3 else validate_relation_metadata
        validator(
            self.metadata,
            expected_num_instances=self.num_instances,
            expected_candidate_digest=expected_candidate_digest,
        )
        if int(self.metadata["directionBins"]) != self.direction_bins:
            raise RelationSchemaError("relation direction_bins disagree with metadata")
        if int(self.metadata["depthShells"]) != self.depth_shells:
            raise RelationSchemaError("relation depth_shells disagree with metadata")
        if self.row_offsets.size != self.row_count + 1:
            raise RelationSchemaError("relation offsets must have row_count + 1 entries")
        if int(self.row_offsets[0]) != 0 or np.any(np.diff(self.row_offsets) < 0):
            raise RelationSchemaError("relation offsets must be non-decreasing from zero")
        if int(self.row_offsets[-1]) != self.source_ids.size:
            raise RelationSchemaError("relation offsets do not cover source IDs")
        _feature_names, feature_dim = _feature_schema(self.metadata)
        if self.edge_features.shape != (self.edge_count, feature_dim):
            raise RelationSchemaError("relation edge feature shape is invalid")
        if self.source_types.size != self.edge_count:
            raise RelationSchemaError("relation source type count is invalid")
        if self.edge_count and int(self.source_ids.max()) >= self.num_instances:
            raise RelationSchemaError("relation source ID is outside num_instances")
        if self.source_types.size and np.any(self.source_types >= len(SOURCE_TYPES)):
            raise RelationSchemaError("relation source type ID is invalid")
        if np.any(~np.isfinite(self.edge_features)):
            raise RelationSchemaError("relation edge features contain non-finite values")
        if np.any(self.edge_features[:, POSE_SUPPORT_COUNT_FEATURE] < 1.0):
            raise RelationSchemaError("every relation needs positive pose support")
        for index in (POSE_SUPPORT_RATE_FEATURE, PIXEL_FRACTION_FEATURE, CONFIDENCE_FEATURE):
            if np.any(self.edge_features[:, index] < 0.0) or np.any(self.edge_features[:, index] > 1.0):
                raise RelationSchemaError(f"relation feature {RELATION_FEATURE_NAMES[index]} is outside [0, 1]")
        for index in (PIXEL_SUPPORT_FEATURE, DEPTH_GAP_FEATURE, RELATIVE_DEPTH_GAP_FEATURE):
            if np.any(self.edge_features[:, index] < 0.0):
                raise RelationSchemaError(f"relation feature {RELATION_FEATURE_NAMES[index]} must be non-negative")
        for index in (5, 7):
            if np.any(self.edge_features[:, index] < 0.0):
                raise RelationSchemaError(f"relation feature {RELATION_FEATURE_NAMES[index]} must be non-negative")
        for index in (DEPTH_GAP_FEATURE, RELATIVE_DEPTH_GAP_FEATURE):
            if np.any(self.edge_features[:, index] <= 0.0):
                raise RelationSchemaError(
                    f"relation feature {RELATION_FEATURE_NAMES[index]} must be strictly positive"
                )
        if self.metadata.get("schema") == SCHEMA_V3:
            if np.any(self.edge_features[:, 13:20] < 0.0):
                raise RelationSchemaError("v3 evidence quality features must be non-negative")
            if np.any(self.edge_features[:, 13] > 1.0) or np.any(self.edge_features[:, 14] > 1.0) or np.any(self.edge_features[:, 16] > 1.0):
                raise RelationSchemaError("v3 bounded evidence features are outside [0, 1]")
        target, _direction, _shell = self.row_indices()
        if self.edge_count and np.any(self.source_ids.astype(np.int64) == target):
            raise RelationSchemaError("relation source and target must differ")
        for row_id, (start, end) in enumerate(
            zip(self.row_offsets[:-1].tolist(), self.row_offsets[1:].tolist(), strict=True)
        ):
            row_sources = self.source_ids[int(start):int(end)]
            if row_sources.size != np.unique(row_sources).size:
                raise RelationSchemaError(f"relation row {row_id} contains duplicate sources")

    def to_torch(self, device: Any = None) -> dict[str, Any]:
        """Convert the CSR to tensors used by the offline relation encoder."""
        import torch

        self.validate()
        target, direction, shell = self.row_indices()
        return {
            "row_offsets": torch.as_tensor(self.row_offsets, dtype=torch.long, device=device),
            "source_ids": torch.as_tensor(self.source_ids, dtype=torch.long, device=device),
            "edge_features": torch.as_tensor(self.edge_features, dtype=torch.float32, device=device),
            "source_types": torch.as_tensor(self.source_types, dtype=torch.long, device=device),
            "target_ids": torch.as_tensor(target, dtype=torch.long, device=device),
            "direction_ids": torch.as_tensor(direction, dtype=torch.long, device=device),
            "depth_shell_ids": torch.as_tensor(shell, dtype=torch.long, device=device),
        }

    def save(self, directory: str | Path) -> None:
        """Write the schema metadata and binary arrays to a dedicated directory."""
        self.validate()
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        files = self.metadata["files"]
        self.row_offsets.astype("<u8", copy=False).tofile(output / files["rowOffsets"])
        self.source_ids.astype("<u4", copy=False).tofile(output / files["sourceIds"])
        self.edge_features.astype("<f4", copy=False).tofile(output / files["edgeFeatures"])
        self.source_types.astype("<u1", copy=False).tofile(output / files["sourceTypes"])
        (output / "relation_csr_meta.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path, *, allow_legacy: bool = False) -> "ObservedRelationCSR":
        """Load v3 by default; v1 requires an explicit historical opt-in."""
        root = Path(directory)
        metadata = json.loads((root / "relation_csr_meta.json").read_text(encoding="utf-8"))
        if metadata.get("schema") != SCHEMA_V3 and not allow_legacy:
            raise RelationSchemaError(
                "historical relation schema requires load(..., allow_legacy=True)"
            )
        (validate_relation_metadata_v3 if metadata.get("schema") == SCHEMA_V3 else validate_relation_metadata)(metadata)
        files = metadata["files"]
        offsets = np.fromfile(root / files["rowOffsets"], dtype="<u8")
        sources = np.fromfile(root / files["sourceIds"], dtype="<u4")
        feature_values = np.fromfile(root / files["edgeFeatures"], dtype="<f4")
        feature_dim = int(metadata["relationFeatureDim"])
        if feature_values.size % feature_dim:
            raise RelationSchemaError("relation feature binary has a partial record")
        features = feature_values.reshape(-1, feature_dim)
        source_types = np.fromfile(root / files["sourceTypes"], dtype="<u1")
        result = cls(
            num_instances=int(metadata["numInstances"]),
            direction_bins=int(metadata["directionBins"]),
            depth_shells=int(metadata["depthShells"]),
            row_offsets=offsets,
            source_ids=sources,
            edge_features=features,
            source_types=source_types,
            metadata=metadata,
        )
        result.validate()
        return result


def build_observed_relation_csr(**kwargs: Any) -> ObservedRelationCSR:
    """Named convenience wrapper used by future train-only builders."""
    return ObservedRelationCSR.from_rows(**kwargs)


__all__ = [
    "SCHEMA",
    "LEGACY_SCHEMA",
    "SCHEMA_V3",
    "ROLE",
    "INDEX_ORDER",
    "SOURCE_TYPES",
    "RELATION_FEATURE_NAMES",
    "RELATION_FEATURE_DIM",
    "RELATION_FEATURE_NAMES_V3",
    "RELATION_FEATURE_DIM_V3",
    "RelationSchemaError",
    "CandidateSummary",
    "candidate_digest",
    "summarize_candidate_csr",
    "validate_candidate_summary",
    "make_relation_metadata",
    "make_relation_metadata_v3",
    "validate_relation_metadata",
    "validate_relation_metadata_v3",
    "validate_survival_observations",
    "validate_survival_observations_v3",
    "load_survival_observations_v3",
    "radius_relative_log_depth",
    "train_depth_quantiles",
    "normalize_radius_relative_log_depth",
    "recompute_relative_geometry",
    "degree_preserving_directed_edge_swap",
    "bounded_hierarchy_ids",
    "summarize_bounded_hierarchy",
    "ObservedRelationCSR",
    "build_observed_relation_csr",
]
