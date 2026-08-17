#!/usr/bin/env python3
"""Train the train-only hierarchical relation survival PVS model.

This entry point is intentionally independent from the historical ray-context
trainer.  It consumes four explicit offline artifacts:

* a fixed 96-dimensional instance geometry table;
* a train-only :class:`ObservedRelationCSR` directory;
* the hierarchy mappings stored beside that relation CSR; and
* a pose CSR dataset containing train, calibration, and validation rows.

The relation encoder is run offline at every optimization step and produces a
``[N, 4, 7]`` survival coefficient table.  The pose-set path then queries a
batch of candidates with one view-cell query and trains the visibility,
visual-utility, and GLB-download heads together.  Test is deliberately not a
valid split argument and is never instantiated or sliced by this file.

The output directory is explicit and must be empty.  This prevents a pilot
from replacing a default model directory or silently mixing two experiments.
For formal safety selection, ``best.pt`` is written only for a checkpoint that
has a calibration threshold with aggregate weighted recall and its bootstrap
lower bound strictly above the registered floor.  Every numerically valid run
also writes ``best_diagnostic.pt`` so a failed safety member can still be
compared after a long train; safety status never controls whether training
finishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402
from common.occlusion_edges import glb_priority_loss, load_glb_costs  # noqa: E402
from common.provenance import protocol_digest, relation_artifact_digest  # noqa: E402
from common.runtime_meta import load_runtime_meta  # noqa: E402
from common.survival_loss import integrated_relation_survival_censoring_loss  # noqa: E402
from common.threshold_selection import (  # noqa: E402
    aggregate_weighted_cull_selection_rule,
    select_aggregate_weighted_cull_workpoint,
)
from common.train_observed_relation_csr import (  # noqa: E402
    ObservedRelationCSR,
    validate_survival_observations,
)
from common.viewcell_quality_rvl_loss import quality_tail_rvl_loss  # noqa: E402
from current_pvs_utils import (  # noqa: E402
    evaluate_thresholds,
    threshold_grid,
    visual_utility_loss,
)
from hierarchical_relation_survival_integrated_model import (  # noqa: E402
    GEO_DIM,
    HierarchicalRelationSurvivalIntegratedModel,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


EXPERIMENT_PREFIX = "pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1"
SURVIVAL_SHAPE = (4, 7)
FIXED_GEO_DTYPE = np.float16
DEFAULT_DATASET = ROOT / "dataset/out/pose_csr_hkust_v3_spatial_fov66_v1"
DEFAULT_RELATION = ROOT / (
    "dataset/out/"
    "pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/relation_csr"
)
DEFAULT_RUNTIME_META = ROOT.parent / "hkust-v3/assets/runtimeVisibilityMeta.json"
DEFAULT_INITIAL_GEO = ROOT / (
    "model/out/"
    "pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_"
    "hkust_spatial_fov66_seed20260801_protocolfix_retry2/"
    "instance_geo_features_fp16.bin"
)
DEFAULT_SUBPOSE_SIDECAR = ROOT / (
    "dataset/out/"
    "pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/"
    "subpose_quality_sidecar"
)
CALIBRATION_FLOOR = 0.99
QUALITY_LOSS_VARIANTS = frozenset({"quality", "threshold_aligned_utility"})
RESOURCE_LOSS_VARIANTS = frozenset({"threshold_aligned_utility"})


def _json_value(value: Any) -> Any:
    """Convert tensors and numpy values into stable JSON values."""

    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return float(detached) if detached.numel() == 1 else detached.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    """Write JSON atomically enough that an interrupted epoch leaves no half file."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_value(value), ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _finite_or(value: Any, fallback: float) -> float:
    """Read optional diagnostic metrics without treating JSON null as numeric."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return result if np.isfinite(result) else float(fallback)


def _prepare_output(output_dir: Path) -> None:
    """Create a new experiment directory and refuse to reuse partial output."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty experiment directory: {output_dir}; "
            "choose a new explicit experiment name"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _load_fixed_geometry(path: Path, num_instances: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and audit the fixed FP16 ``[N, 96]`` geometry table."""

    if not path.is_file():
        raise FileNotFoundError(f"missing fixed 96-D geometry table: {path}")
    expected_values = int(num_instances) * int(GEO_DIM)
    expected_bytes = expected_values * np.dtype(FIXED_GEO_DTYPE).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes; expected {expected_bytes} "
            f"({num_instances} instances x {GEO_DIM} FP16 values)"
        )
    values = np.fromfile(path, dtype=FIXED_GEO_DTYPE)
    if values.size != expected_values or not bool(np.isfinite(values).all()):
        raise ValueError("fixed geometry table has an invalid size or non-finite values")
    values = values.reshape(int(num_instances), int(GEO_DIM))
    return torch.from_numpy(values.astype(np.float32, copy=False)), {
        "path": str(path.resolve()),
        "dtype": "float16",
        "shape": [int(num_instances), int(GEO_DIM)],
        "sha256": _sha256_file(path),
    }


def _read_uint32_ids(path: Path, expected_size: int, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    values = np.fromfile(path, dtype="<u4")
    if values.size != int(expected_size):
        raise ValueError(f"{name} has {values.size} entries; expected {expected_size}")
    if values.size and int(values.min()) < 0:
        raise ValueError(f"{name} contains a negative ID")
    return values.astype(np.int64, copy=False)


def _validate_contiguous_ids(values: np.ndarray, expected_size: int, name: str) -> None:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if values.size != int(expected_size):
        raise ValueError(f"{name} has {values.size} entries; expected {expected_size}")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    expected = np.arange(int(values.max()) + 1, dtype=np.int64)
    if not np.array_equal(np.unique(values), expected):
        raise ValueError(f"{name} must contain contiguous IDs from zero")


def _load_hierarchy_ids(
    relation_dir: Path,
    metadata: Mapping[str, Any],
    num_instances: int,
    local_path_arg: str,
    structural_path_arg: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hierarchy = metadata.get("hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise ValueError("relation metadata is missing train-only hierarchy mappings")
    files = hierarchy.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("relation metadata has no hierarchy file mapping")
    local_path = Path(local_path_arg) if local_path_arg else relation_dir / str(files.get("localGroupIds", ""))
    structural_path = (
        Path(structural_path_arg)
        if structural_path_arg
        else relation_dir / str(files.get("structuralGroupIds", ""))
    )
    local = _read_uint32_ids(local_path, num_instances, "instance local group IDs")
    local_count = int(local.max()) + 1
    declared_local_count = hierarchy.get("localGroupCount")
    if declared_local_count is not None and int(declared_local_count) != local_count:
        raise ValueError("local group ID count disagrees with relation metadata")
    structural = _read_uint32_ids(
        structural_path,
        local_count,
        "local structural group IDs",
    )
    structural_count = int(structural.max()) + 1
    declared_structural_count = hierarchy.get("structureGroupCount", hierarchy.get("structuralGroupCount"))
    if declared_structural_count is not None and int(declared_structural_count) != structural_count:
        raise ValueError("structural group ID count disagrees with relation metadata")
    _validate_contiguous_ids(local, num_instances, "instance local group IDs")
    _validate_contiguous_ids(structural, local_count, "local structural group IDs")
    return local, structural, {
        "localGroupCount": local_count,
        "structuralGroupCount": structural_count,
        "localGroupIds": str(local_path.resolve()),
        "structuralGroupIds": str(structural_path.resolve()),
        "localGroupIdsSha256": _sha256_file(local_path),
        "structuralGroupIdsSha256": _sha256_file(structural_path),
    }


def _load_survival_observations(
    relation_dir: Path,
    metadata: Mapping[str, Any],
    num_instances: int,
    direction_bins: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load only the train-owned event/right-censor arrays beside the CSR."""

    envelope = metadata.get("survivalObservations")
    if not isinstance(envelope, Mapping):
        raise ValueError("relation metadata is missing train-only survival observations")
    files = envelope.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("survival observation metadata has no file mapping")
    specifications = {
        "instance": ("instance", "<u4"),
        "direction": ("direction", "<u1"),
        "rho": ("rho", "<f2"),
        "event": ("event", "<u1"),
        "weight": ("weight", "<f2"),
    }
    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {
        "schema": str(envelope.get("schema", "")),
        "count": int(envelope.get("count", -1)),
        "files": {},
    }
    for key, (file_key, dtype) in specifications.items():
        relative = files.get(file_key)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"survival observation metadata has no {file_key} file")
        path = relation_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing survival observation file: {path}")
        arrays[key] = np.fromfile(path, dtype=dtype)
        provenance["files"][key] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "count": int(arrays[key].size),
        }
    summary = validate_survival_observations(
        arrays,
        num_instances=int(num_instances),
        direction_bins=int(direction_bins),
    )
    if provenance["count"] >= 0 and provenance["count"] != summary["observationCount"]:
        raise ValueError("survival observation count disagrees with relation metadata")
    provenance.update(summary)
    return arrays, provenance


def _load_relation_resources(
    relation_dir: Path,
    dataset: PoseCSRDataset,
    train_split: Any,
    num_instances: int,
    local_group_ids: str,
    structural_group_ids: str,
) -> tuple[ObservedRelationCSR, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Load and cross-check the train-only relation, hierarchy, and censor data."""

    relation = ObservedRelationCSR.load(relation_dir)
    if relation.num_instances != int(num_instances):
        raise ValueError("relation CSR numInstances does not match runtime metadata")
    if relation.direction_bins != 12 or relation.depth_shells != 3:
        raise ValueError("the registered integrated model requires 12 directions and 3 depth shells")
    train_digest = candidate_digest_for_pose_sequence(dataset, train_split.pose_indices)
    # ``ObservedRelationCSR.validate`` derives the instance-count check from
    # the loaded object; the only external identity that must be matched here
    # is the native train candidate digest.
    relation.validate(expected_candidate_digest=train_digest)
    if relation.metadata.get("splitNames") != ["train"] or relation.metadata.get("trainOnly") is not True:
        raise ValueError("observed relation CSR must be explicitly train-only")
    local_np, structural_np, hierarchy_meta = _load_hierarchy_ids(
        relation_dir,
        relation.metadata,
        num_instances,
        local_group_ids,
        structural_group_ids,
    )
    observations_np, observation_meta = _load_survival_observations(
        relation_dir,
        relation.metadata,
        num_instances,
        relation.direction_bins,
    )
    artifact_digest, artifact_files = relation_artifact_digest(relation_dir, relation.metadata)
    return (
        relation,
        relation.to_torch(),
        torch.from_numpy(local_np),
        torch.from_numpy(structural_np),
        hierarchy_meta,
        observations_np,
        {
            "relationDir": str(relation_dir.resolve()),
            "relationMeta": relation.metadata,
            "relationArtifactDigest": artifact_digest,
            "relationArtifactFiles": artifact_files,
            "canonicalCandidateDigest": relation.metadata["candidateIdentity"]["canonicalCandidateDigest"],
            "renderCandidateDigest": relation.metadata["candidateIdentity"]["renderCandidateDigest"],
            "trainCandidateDigest": train_digest,
            "edgeCount": relation.edge_count,
            "rowCount": relation.row_count,
            "hierarchy": hierarchy_meta,
            "survivalObservations": observation_meta,
        },
    )


def _relation_variant(
    relation_tensors: Mapping[str, torch.Tensor],
    local_group_ids: torch.Tensor,
    structural_group_ids: torch.Tensor,
    source: str,
    num_instances: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Construct the registered relation controls without changing candidates."""
    source = str(source)
    identity = torch.arange(int(num_instances), device=device, dtype=torch.long)
    if source in {"free", "geometry_only"}:
        # The free control is intentionally relation-free.  Keep this branch
        # before touching the supplied mapping so a caller cannot accidentally
        # make the control depend on CSR loading or graph tensor contents.
        empty = {
            "source_ids": torch.zeros(0, dtype=torch.long, device=device),
            "target_ids": torch.zeros(0, dtype=torch.long, device=device),
            "direction_ids": torch.zeros(0, dtype=torch.long, device=device),
            "depth_shell_ids": torch.zeros(0, dtype=torch.long, device=device),
            "edge_features": torch.zeros(0, 13, dtype=torch.float32, device=device),
        }
        return empty, identity, identity, {
            "source": source,
            "shuffled": False,
            "hierarchy": "geometry_only" if source == "geometry_only" else "none_free_per_instance",
        }

    tensors = {key: value.to(device) for key, value in relation_tensors.items()}
    if source == "hierarchical":
        return tensors, local_group_ids, structural_group_ids, {
            "source": source, "shuffled": False, "hierarchy": "three_scale_observed"
        }
    if source == "single_scale":
        return tensors, identity, identity, {
            "source": source, "shuffled": False, "hierarchy": "identity"
        }
    if source == "shuffled":
        shifted = (tensors["source_ids"].long() + 7919) % int(num_instances)
        same = shifted == tensors["target_ids"].long()
        shifted = torch.where(same, (shifted + 1) % int(num_instances), shifted)
        tensors["source_ids"] = shifted
        return tensors, local_group_ids, structural_group_ids, {
            "source": source, "shuffled": True, "hierarchy": "three_scale_source_shuffled"
        }
    raise ValueError(f"unknown relation source: {source}")


def _resolve_split(dataset: PoseCSRDataset, requested: str, role: str) -> tuple[str, Any]:
    """Resolve a named split without ever allowing ``test``."""

    if requested == "auto":
        candidates = {
            "validation": ("validation", "val"),
            "calibration": ("calibration",),
            "train": ("train",),
        }[role]
        name = next((candidate for candidate in candidates if candidate in dataset.split_ids), None)
        if name is None:
            raise ValueError(f"dataset has no usable {role} split; test is not a fallback")
    else:
        name = requested
    if name.lower() == "test":
        raise ValueError("test is forbidden during training, checkpoint selection, and calibration")
    if name not in dataset.split_ids:
        raise ValueError(f"dataset has no split named {name!r} for {role}")
    split = dataset.split(name)
    if split.pose_indices.size == 0:
        raise ValueError(f"{role} split {name!r} is empty")
    return name, split


def _protocol_splits(
    train_name: str,
    train_split: Any,
    validation_name: str,
    validation_split: Any,
    calibration_name: str,
    calibration_split: Any,
    seed: int,
    subpose_quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def digest(split: Any) -> str:
        return candidate_digest_for_pose_sequence(split.dataset, split.pose_indices)

    protocol = {
        "schema": "pvs-hierarchical-relation-survival-integrated-training-v1",
        "seed": int(seed),
        "testRead": False,
        "thresholdSource": "calibration_only",
        "candidateUnion": False,
        "trainPoseCount": int(train_split.pose_indices.size),
        "calibrationPoseCount": int(calibration_split.pose_indices.size),
        "validationPoseCount": int(validation_split.pose_indices.size),
        "trainCandidateDigest": digest(train_split),
        "calibrationCandidateDigest": digest(calibration_split),
        "validationCandidateDigest": digest(validation_split),
        "splitNames": {
            "train": train_name,
            "calibration": calibration_name,
            "validation": validation_name,
        },
        "subposeQuality": dict(subpose_quality or {
            "required": False,
            "available": False,
            "semantics": "not used by this loss variant",
        }),
        "policy": "train optimizes train poses; calibration freezes the threshold; validation is recorded only; test is not read",
    }
    protocol["protocolDigest"] = protocol_digest(protocol)
    return protocol


def _make_runtime_features(
    model: HierarchicalRelationSurvivalIntegratedModel,
    geometry: torch.Tensor,
    relation: Mapping[str, torch.Tensor],
    local_group_ids: torch.Tensor,
    structural_group_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the offline fixed table and assert the registered shape."""

    coefficients = model.offline_encode_survival(
        geometry,
        relation,
        local_group_ids,
        structural_group_ids,
    )
    if not isinstance(coefficients, torch.Tensor) or tuple(coefficients.shape[1:]) != SURVIVAL_SHAPE:
        raise ValueError(f"offline survival coefficients must have shape [N, 4, 7], got {getattr(coefficients, 'shape', None)}")
    if coefficients.shape[0] != geometry.shape[0] or not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError("offline survival coefficient table is invalid")
    runtime_features = torch.cat([geometry, coefficients.reshape(geometry.shape[0], -1)], dim=-1)
    if tuple(runtime_features.shape) != (geometry.shape[0], 124):
        raise ValueError("runtime feature table must contain 96 geometry values and 28 survival values")
    return runtime_features, coefficients


def _variant_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Return the effective semantic tuple recorded in every artifact."""

    loss_variant = str(args.loss_variant)
    quality_weight = float(args.quality_loss_weight) if loss_variant in QUALITY_LOSS_VARIANTS else 0.0
    resource_weight = float(args.resource_loss_weight) if loss_variant in RESOURCE_LOSS_VARIANTS else 0.0
    if quality_weight < 0.0 or resource_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if loss_variant in QUALITY_LOSS_VARIANTS and quality_weight <= 0.0:
        raise ValueError(f"{loss_variant} requires a positive --quality-loss-weight")
    if loss_variant in RESOURCE_LOSS_VARIANTS and resource_weight <= 0.0:
        raise ValueError(f"{loss_variant} requires a positive --resource-loss-weight")
    return {
        "variant": str(args.variant),
        "relationSource": str(args.relation_source),
        "spectralMode": str(args.spectral_mode),
        "lossVariant": loss_variant,
        "effectiveQualityWeight": quality_weight,
        "effectiveResourceWeight": resource_weight,
        "qualityWeight": float(args.quality_loss_weight),
        "resourceWeight": float(args.resource_loss_weight),
    }


def _runtime_features_from_coefficients(
    geometry: torch.Tensor,
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack an already-owned coefficient table for the free-field control."""
    if tuple(coefficients.shape) != (int(geometry.shape[0]), 4, 7):
        raise ValueError("free survival coefficients must have shape [N, 4, 7]")
    if not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError("free survival coefficient table is non-finite")
    runtime = torch.cat([geometry, coefficients.reshape(geometry.shape[0], -1)], dim=-1)
    return runtime, coefficients


def _sample_observations(
    observations: Mapping[str, torch.Tensor],
    max_observations: int,
) -> dict[str, torch.Tensor]:
    """Keep observation selection deterministic; no validation/test signal is involved."""

    if max_observations <= 0:
        return dict(observations)
    count = int(observations["instance"].numel())
    if count <= int(max_observations):
        return dict(observations)
    indices = torch.linspace(
        0,
        count - 1,
        steps=int(max_observations),
        device=observations["instance"].device,
    ).long()
    return {key: value[indices] for key, value in observations.items()}


def _gradient_tuple(
    objective: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    if not bool(objective.requires_grad):
        return tuple(None for _ in parameters)
    return torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _project_resource_gradients(
    safety_gradients: tuple[torch.Tensor | None, ...],
    resource_gradients: tuple[torch.Tensor | None, ...],
    *,
    enabled: bool,
    maximum_ratio: float = 0.25,
) -> tuple[list[torch.Tensor | None], dict[str, float | bool]]:
    """Project resource gradients away from a conflicting safety gradient."""

    safety_norm_sq = torch.zeros((), device=next(
        (value.device for value in safety_gradients if value is not None),
        torch.device("cpu"),
    ))
    resource_norm_sq = torch.zeros_like(safety_norm_sq)
    dot = torch.zeros_like(safety_norm_sq)
    for safety, resource in zip(safety_gradients, resource_gradients, strict=True):
        if safety is not None:
            safety_norm_sq = safety_norm_sq + safety.detach().float().square().sum()
        if resource is not None:
            resource_norm_sq = resource_norm_sq + resource.detach().float().square().sum()
        if safety is not None and resource is not None:
            dot = dot + (safety.detach().float() * resource.detach().float()).sum()

    safety_norm = torch.sqrt(safety_norm_sq.clamp_min(0.0))
    resource_norm_before = torch.sqrt(resource_norm_sq.clamp_min(0.0))
    projected = False
    limited = False
    projected_resources: list[torch.Tensor | None] = []
    if enabled and bool((dot < 0.0).item()) and bool((safety_norm_sq > 1e-20).item()):
        coefficient = dot / safety_norm_sq.clamp_min(1e-20)
        projected = True
        for safety, resource in zip(safety_gradients, resource_gradients, strict=True):
            if resource is None:
                projected_resources.append(None)
            elif safety is None:
                projected_resources.append(resource.detach().clone())
            else:
                projected_resources.append((resource.detach() - coefficient * safety.detach()).clone())
    else:
        projected_resources = [None if value is None else value.detach().clone() for value in resource_gradients]

    projected_norm_sq = torch.zeros_like(safety_norm_sq)
    for value in projected_resources:
        if value is not None:
            projected_norm_sq = projected_norm_sq + value.float().square().sum()
    projected_norm = torch.sqrt(projected_norm_sq.clamp_min(0.0))
    maximum_norm = float(maximum_ratio) * safety_norm
    if enabled and bool((maximum_norm > 0.0).item()) and bool((projected_norm > maximum_norm).item()):
        scale = (maximum_norm / projected_norm.clamp_min(1e-20)).detach()
        projected_resources = [None if value is None else value * scale for value in projected_resources]
        projected_norm = maximum_norm
        limited = True

    combined: list[torch.Tensor | None] = []
    for safety, resource in zip(safety_gradients, projected_resources, strict=True):
        if safety is None and resource is None:
            combined.append(None)
        elif safety is None:
            combined.append(resource)
        elif resource is None:
            combined.append(safety.detach().clone())
        else:
            combined.append(safety.detach().clone() + resource)
    return combined, {
        "resourceGradientDotSafety": _scalar(dot),
        "safetyGradientNorm": _scalar(safety_norm),
        "resourceGradientNormBefore": _scalar(resource_norm_before),
        "resourceGradientNormAfter": _scalar(projected_norm),
        "resourceGradientProjection": bool(projected),
        "resourceGradientLimited": bool(limited),
    }


def _assign_gradients(
    parameters: list[torch.nn.Parameter],
    gradients: list[torch.Tensor | None],
) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.to(device=parameter.device, dtype=parameter.dtype)


def _metric_values(*groups: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in groups:
        for key, value in group.items():
            try:
                numeric = _scalar(value)
            except (TypeError, ValueError, RuntimeError):
                continue
            if np.isfinite(numeric):
                result[str(key)] = numeric
    return result


def _eval_split(
    model: HierarchicalRelationSurvivalIntegratedModel,
    split: Any,
    runtime_features: torch.Tensor,
    world_aabbs: np.ndarray,
    device: torch.device,
    *,
    seed: int,
    thresholds: np.ndarray,
    max_poses: int,
    bootstrap_replicates: int,
    label: str,
    poses_per_batch: int,
    max_candidates_per_pose: int,
) -> list[dict[str, Any]]:
    if max_poses > 0:
        effective = split.dataset.subset(
            f"{label}_limited_pre_test",
            split.pose_indices[: int(max_poses)],
        )
    else:
        effective = split
    return evaluate_thresholds(
        model,
        effective,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=max(1, int(poses_per_batch)),
        max_steps=None,
        max_candidates_per_pose=max(0, int(max_candidates_per_pose)),
        seed=int(seed),
        thresholds=np.asarray(thresholds, dtype=np.float32),
        collect_pose_stats=True,
        allow_candidate_visible_union=False,
        bootstrap_replicates=int(bootstrap_replicates),
        collect_score_stats=label == "calibration",
    )


def _diagnostic_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("aggregateWeightedRecall", -1.0)),
            float(row.get("agg_balanced_accuracy", 0.0)),
            float(row.get("agg_precision", 0.0)),
            -float(row.get("avg_pred_count", 0.0)),
        ),
    )


def _checkpoint_payload(
    model: HierarchicalRelationSurvivalIntegratedModel,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    protocol: Mapping[str, Any],
    relation_meta: Mapping[str, Any],
    relation_variant: Mapping[str, Any],
    geometry_meta: Mapping[str, Any],
    coefficients: torch.Tensor,
    calibration: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    *,
    best: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "pvs-hierarchical-relation-survival-integrated-training-v1",
        "modelSchema": "pvs-hierarchical-relation-survival-integrated-v1",
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": model.config,
        "args": vars(args),
        "variant": str(args.variant),
        "variantSpec": _variant_spec(args),
        "epoch": int(epoch),
        "globalStep": int(global_step),
        "protocol": protocol,
        "protocolDigest": protocol.get("protocolDigest"),
        "relationProvenance": {
            **dict(relation_meta),
            "variant": dict(relation_variant),
        },
        "relation": {
            "schema": relation_meta.get("relationMeta", {}).get("schema"),
            "trainCandidateDigest": relation_meta.get("trainCandidateDigest"),
            "canonicalCandidateDigest": relation_meta.get("canonicalCandidateDigest"),
            "renderCandidateDigest": relation_meta.get("renderCandidateDigest"),
            "relationArtifactDigest": relation_meta.get("relationArtifactDigest"),
            "edgeCount": relation_meta.get("edgeCount"),
            "rowCount": relation_meta.get("rowCount"),
        },
        "geometryMeta": dict(geometry_meta),
        "survivalCoefficients": coefficients.detach().cpu(),
        "survivalCoefficientShape": list(SURVIVAL_SHAPE),
        "calibration": calibration,
        "validationAtCalibration": validation,
        "best": best,
        "testRead": False,
    }


def _save_table(path: Path, values: torch.Tensor) -> None:
    values.detach().cpu().numpy().astype(np.float16, copy=False).tofile(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--relation-dir", default=str(DEFAULT_RELATION))
    parser.add_argument("--runtime-meta", default=str(DEFAULT_RUNTIME_META))
    parser.add_argument("--initial-geo-features", default=str(DEFAULT_INITIAL_GEO))
    parser.add_argument(
        "--subpose-sidecar",
        default="",
        help=(
            "Validated dense-subpose supervision sidecar. Required for quality "
            "and threshold-aligned utility losses; omitted for RVL-only controls."
        ),
    )
    parser.add_argument("--local-group-ids", default="")
    parser.add_argument("--structural-group-ids", default="")
    parser.add_argument("--glb-index", default="")
    parser.add_argument("--glb-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default=EXPERIMENT_PREFIX)
    parser.add_argument("--variant", default="full")
    parser.add_argument(
        "--relation-source",
        choices=("hierarchical", "single_scale", "geometry_only", "free", "shuffled"),
        default="hierarchical",
    )
    parser.add_argument(
        "--spectral-mode",
        choices=("integrated", "learned_point", "fourier117"),
        default="integrated",
    )
    parser.add_argument(
        "--loss-variant",
        choices=("rvl", "quality", "threshold_aligned_utility"),
        default="threshold_aligned_utility",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="auto")
    parser.add_argument("--calibration-split", default="calibration")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=75)
    parser.add_argument("--pose-set-batch-size", type=int, default=2)
    parser.add_argument("--max-candidates-per-pose", type=int, default=0)
    parser.add_argument("--max-eval-poses", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--relation-hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--query-basis-dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--survival-loss-weight", type=float, default=0.25)
    parser.add_argument("--utility-loss-weight", type=float, default=0.18)
    parser.add_argument("--download-loss-weight", type=float, default=0.20)
    parser.add_argument("--reg-weight", type=float, default=1e-5)
    parser.add_argument("--max-survival-observations", type=int, default=4096)
    parser.add_argument("--quality-loss-weight", type=float, default=0.5)
    parser.add_argument("--resource-loss-weight", type=float, default=0.05)
    parser.add_argument("--quality-temperature", type=float, default=0.10)
    parser.add_argument("--quality-tail-fraction", type=float, default=0.01)
    parser.add_argument("--rare-positive-weight", type=float, default=1.0)
    parser.add_argument("--resource-warmup-steps", type=int, default=0)
    parser.add_argument("--resource-gate-quality", type=float, default=0.0)
    parser.add_argument("--disable-resource-gradient-projection", action="store_true")
    parser.add_argument("--allow-missing-glb-costs", action="store_true")
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Save an auditable checkpoint_epoch_NNN.pt at every Nth epoch; zero disables periodic snapshots.",
    )
    parser.add_argument("--allow-unsafe-final", action="store_true")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("epochs and steps-per-epoch must be positive")
    if args.pose_set_batch_size <= 0:
        raise ValueError("pose-set-batch-size must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if args.calibration_bootstrap_replicates < 0:
        raise ValueError("calibration bootstrap replicates must be non-negative")
    if args.snapshot_every < 0:
        raise ValueError("snapshot-every must be non-negative")
    if args.max_candidates_per_pose < 0 or args.max_eval_poses < 0:
        raise ValueError("candidate and evaluation limits must be non-negative")
    if not str(args.variant).strip():
        raise ValueError("variant must not be empty")
    device = _resolve_device(args.device)
    variant_spec = _variant_spec(args)
    output_dir = Path(args.output_dir).resolve()
    _prepare_output(output_dir)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
        torch.set_float32_matmul_precision("high")

    dataset_dir = Path(args.dataset_dir).resolve()
    runtime_meta_path = Path(args.runtime_meta).resolve()
    relation_dir = Path(args.relation_dir).resolve()
    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    num_instances = int(world_aabbs.shape[0])
    if num_instances <= 0:
        raise ValueError("runtime metadata contains no instances")
    num_glbs = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
    if num_glbs <= 0:
        raise ValueError("runtime metadata contains no GLB mapping")

    quality_loss_enabled = str(args.loss_variant) in QUALITY_LOSS_VARIANTS
    subpose_sidecar = Path(args.subpose_sidecar).resolve() if str(args.subpose_sidecar).strip() else None
    if quality_loss_enabled and subpose_sidecar is None:
        raise ValueError(
            "quality and threshold-aligned utility losses require --subpose-sidecar; "
            "do not silently replace dense subpose supervision with zero hit rates"
        )
    dataset = PoseCSRDataset(
        dataset_dir,
        num_instances=num_instances,
        subpose_sidecar=subpose_sidecar,
    )
    if quality_loss_enabled and not dataset.has_subpose_robust_labels:
        raise ValueError("quality loss requires validated visible_hit_counts and subpose_offsets in the sidecar")
    train_name, train_split = _resolve_split(dataset, args.train_split, "train")
    validation_name, validation_split = _resolve_split(dataset, args.validation_split, "validation")
    calibration_name, calibration_split = _resolve_split(dataset, args.calibration_split, "calibration")
    protocol = _protocol_splits(
        train_name,
        train_split,
        validation_name,
        validation_split,
        calibration_name,
        calibration_split,
        args.seed,
        {
            "required": quality_loss_enabled,
            "available": bool(dataset.has_subpose_robust_labels),
            "sidecar": str(subpose_sidecar) if subpose_sidecar is not None else None,
            "manifestSha256": (
                _sha256_file(subpose_sidecar / "sidecar_manifest.json")
                if subpose_sidecar is not None else None
            ),
            "semantics": dataset.subpose_robust_label_semantics,
        },
    )

    relation, relation_tensors, local_group_ids_cpu, structural_group_ids_cpu, hierarchy_meta, observations_np, relation_meta = _load_relation_resources(
        relation_dir,
        dataset,
        train_split,
        num_instances,
        args.local_group_ids,
        args.structural_group_ids,
    )
    geometry_cpu, geometry_meta = _load_fixed_geometry(
        Path(args.initial_geo_features).resolve(),
        num_instances,
    )
    geometry = geometry_cpu.to(device)
    local_group_ids = local_group_ids_cpu.to(device)
    structural_group_ids = structural_group_ids_cpu.to(device)
    relation_tensors = {key: value.to(device) for key, value in relation_tensors.items()}
    relation_tensors, local_group_ids, structural_group_ids, relation_variant_meta = _relation_variant(
        relation_tensors,
        local_group_ids,
        structural_group_ids,
        str(args.relation_source),
        num_instances,
        device,
    )
    observations = {
        "instance": torch.from_numpy(observations_np["instance"].astype(np.int64, copy=False)).to(device),
        "direction": torch.from_numpy(observations_np["direction"].astype(np.int64, copy=False)).to(device),
        "rho": torch.from_numpy(observations_np["rho"].astype(np.float32, copy=False)).to(device),
        "event": torch.from_numpy(observations_np["event"].astype(np.float32, copy=False)).to(device),
        "weight": torch.from_numpy(observations_np["weight"].astype(np.float32, copy=False)).to(device),
    }

    glb_index = Path(args.glb_index).resolve() if args.glb_index else runtime_meta_path.with_name("glbIndex.json")
    glb_root = Path(args.glb_root).resolve() if args.glb_root else runtime_meta_path.parent
    glb_cost_norm = torch.from_numpy(
        load_glb_costs(
            glb_index,
            glb_root,
            num_glbs,
            instance_to_glb,
            allow_missing=bool(args.allow_missing_glb_costs),
        )
    ).to(device)

    model = HierarchicalRelationSurvivalIntegratedModel(
        num_instances=num_instances,
        num_glbs=num_glbs,
        relation_hidden_dim=int(args.relation_hidden_dim),
        query_basis_dim=int(args.query_basis_dim),
        hidden_dim=int(args.hidden_dim),
        spectral_mode=str(args.spectral_mode),
        relation_source=str(args.relation_source),
    ).to(device)
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    rng = np.random.default_rng(int(args.seed))

    model_meta: dict[str, Any] = {
        "schema": "pvs-hierarchical-relation-survival-integrated-training-v1",
        "modelSchema": "pvs-hierarchical-relation-survival-integrated-v1",
        "experiment": str(args.experiment_name),
        "modelConfig": model.config,
        "args": vars(args),
        "device": str(device),
        "protocol": protocol,
        "protocolDigest": protocol.get("protocolDigest"),
        "variant": str(args.variant),
        "variantSpec": variant_spec,
        "relationVariant": {
            "source": str(args.relation_source),
            "shuffled": str(args.relation_source) == "shuffled",
            "hierarchy": (
                "identity" if str(args.relation_source) == "single_scale"
                else "geometry_only" if str(args.relation_source) == "geometry_only"
                else "none_free_per_instance" if str(args.relation_source) == "free"
                else "three_scale_source_shuffled" if str(args.relation_source) == "shuffled"
                else "three_scale_observed"
            ),
        },
        "testRead": False,
        "geometry": geometry_meta,
        "relation": {
            "schema": relation_meta["relationMeta"].get("schema"),
            "trainCandidateDigest": relation_meta.get("trainCandidateDigest"),
            "canonicalCandidateDigest": relation_meta.get("canonicalCandidateDigest"),
            "renderCandidateDigest": relation_meta.get("renderCandidateDigest"),
            "relationArtifactDigest": relation_meta.get("relationArtifactDigest"),
            "inputSha256": relation_meta["relationMeta"].get("inputSha256", {}),
            "edgeCount": relation_meta.get("edgeCount"),
            "rowCount": relation_meta.get("rowCount"),
        },
        "relationProvenance": relation_meta,
        "hierarchy": hierarchy_meta,
        "runtimeMeta": {
            "path": str(runtime_meta_path),
            "sha256": _sha256_file(runtime_meta_path),
            "numInstances": num_instances,
            "numGlbs": num_glbs,
        },
        "glbCosts": {
            "index": str(glb_index),
            "root": str(glb_root),
            "allowMissing": bool(args.allow_missing_glb_costs),
        },
        "dataSemantics": {
            "candidate": "stored native frustum candidate CSR; visible IDs are never unioned into candidates",
            "visibleWeights": dataset.visible_weight_semantics,
            "relation": "train-only observed directed relation CSR",
            "survivalCoefficients": "offline [N, 4, 7] generated from fixed geometry, relation CSR, and hierarchy IDs",
            "poseSet": "one batch contains complete stored candidate sets for selected poses",
            "testRead": False,
        },
        "trainingStage": {
            "offlineRelationEncoderTrainable": True,
            "fixedGeometryTrainable": False,
            "runtimeUses": ["fixed geometry table", "fixed survival coefficient table", "one integrated spectral query"],
        },
        "calibrationPolicy": {
            "primarySafetyMetric": "aggregateWeightedRecall",
            "diagnosticSafetyMetric": "poseMacroWeightedRecall",
            "weightedRecallFloor": CALIBRATION_FLOOR,
            "lowerConfidenceBoundFloor": CALIBRATION_FLOOR,
            "selectionRule": aggregate_weighted_cull_selection_rule(CALIBRATION_FLOOR, CALIBRATION_FLOOR),
            "poseRecallRole": "diagnostic_only",
        },
        "outputs": {
            "best": "best.pt",
            "bestSafe": "best_safe.pt",
            "bestDiagnostic": "best_diagnostic.pt",
            "last": "last.pt",
            "history": "train_history.json",
            "metrics": "train_metrics.jsonl",
            "calibration": "calibration_ready_summary.json",
        },
    }
    _write_json(output_dir / "model_meta.json", model_meta)
    _write_json(output_dir / "protocol_split.json", protocol)
    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    history: list[dict[str, Any]] = []
    best_safe_key: tuple[float, float, float, float, float] | None = None
    best_safe_calibration: dict[str, Any] | None = None
    best_safe_validation: dict[str, Any] | None = None
    best_safe_checkpoint_info: dict[str, Any] | None = None
    best_diagnostic_key: tuple[float, float, float, float, float, float] | None = None
    best_diagnostic_calibration: dict[str, Any] | None = None
    best_diagnostic_validation: dict[str, Any] | None = None
    best_diagnostic_checkpoint_info: dict[str, Any] | None = None
    latest_coefficients: torch.Tensor | None = None
    global_step = 0
    quality_ema: float | None = None
    total_steps = int(args.epochs) * int(args.steps_per_epoch)
    started = time.time()
    formal_calibration = int(args.max_eval_poses) <= 0 and int(args.calibration_bootstrap_replicates) >= 10000

    for epoch in range(int(args.epochs)):
        model.train()
        epoch_metrics: dict[str, list[float]] = {}
        processed_steps = 0
        resource_gate_count = 0
        projection_count = 0
        iterator = train_split.pose_set_batches(
            int(args.pose_set_batch_size),
            rng,
            int(args.steps_per_epoch),
            include_empty=True,
        )
        for step, pose_indices in enumerate(iterator):
            batch = train_split.build_pose_set_batch(
                pose_indices,
                world_aabbs,
                rng,
                max_candidates_per_pose=int(args.max_candidates_per_pose),
                allow_candidate_visible_union=False,
                include_empty=True,
            )
            if int(batch["instance"].size) == 0:
                continue
            processed_steps += 1
            global_step += 1
            ids = torch.from_numpy(batch["instance"]).to(device)
            camera = torch.from_numpy(batch["camera"]).to(device)
            camera_world = torch.from_numpy(batch["camera_world"]).to(device)
            view = torch.from_numpy(batch["camera_view"]).to(device)
            target = torch.from_numpy(batch["target"][:, None]).to(device)
            visible_weights = torch.from_numpy(batch["visible_weights"][:, None]).to(device)
            visible_hit_rates = torch.from_numpy(batch["visible_hit_rates"][:, None]).to(device)
            pose_offsets = torch.from_numpy(batch["pose_offsets"]).to(device)

            optimizer.zero_grad(set_to_none=True)
            runtime_features, coefficients = _make_runtime_features(
                model,
                geometry,
                relation_tensors,
                local_group_ids,
                structural_group_ids,
            )
            latest_coefficients = coefficients.detach()
            logits, aux = model.compute_logits_with_aux(
                camera,
                view,
                camera_world,
                ids,
                runtime_features=runtime_features,
            )
            quality_total, quality_parts = quality_tail_rvl_loss(
                logits,
                target,
                pose_offsets,
                visible_weights,
                visible_hit_rates,
                ids,
                model.instance_to_glb,
                glb_cost_norm,
                quality_weight=float(variant_spec["effectiveQualityWeight"]),
                resource_weight=float(variant_spec["effectiveResourceWeight"]),
                quality_temperature=float(args.quality_temperature),
                tail_fraction=float(args.quality_tail_fraction),
                rare_positive_weight=float(args.rare_positive_weight),
            )
            quality_recall = _scalar(quality_parts.get("qualityRecall", 0.0))
            quality_ema = quality_recall if quality_ema is None else 0.9 * quality_ema + 0.1 * quality_recall
            resource_gate = bool(
                str(args.loss_variant) in RESOURCE_LOSS_VARIANTS
                and
                global_step > int(args.resource_warmup_steps)
                and quality_ema >= float(args.resource_gate_quality)
            )
            if resource_gate:
                resource_gate_count += 1
            safety_core = quality_parts["lossSafetyRvl"]
            if str(args.loss_variant) in QUALITY_LOSS_VARIANTS:
                safety_core = safety_core + (
                    float(variant_spec["effectiveQualityWeight"])
                    * quality_parts["lossQualityTail"]
                )
            resource_core = (
                float(variant_spec["effectiveResourceWeight"])
                * quality_parts["lossResourceRepulsion"]
                * float(resource_gate and str(args.loss_variant) in RESOURCE_LOSS_VARIANTS)
            )
            survival_batch = _sample_observations(observations, int(args.max_survival_observations))
            survival_loss, survival_parts = integrated_relation_survival_censoring_loss(
                model,
                coefficients,
                survival_batch,
                max_observations=int(args.max_survival_observations) if args.max_survival_observations > 0 else None,
            )
            utility_loss, utility_parts = visual_utility_loss(
                aux,
                target,
                visible_weights,
                pose_offsets,
                invisible_weight=0.25,
                rank_weight=0.3,
                rank_margin=0.1,
                rank_positive_top_k=32,
                rank_negative_top_k=128,
            )
            download_loss, download_parts = glb_priority_loss(
                aux["download_logits"],
                ids,
                target,
                visible_weights,
                model.instance_to_glb,
                glb_cost_norm,
                pose_offsets,
            )
            regularization = model.regularization()
            safety_objective = (
                safety_core
                + float(args.survival_loss_weight) * survival_loss
                + float(args.utility_loss_weight) * utility_loss
                + float(args.download_loss_weight) * download_loss
                + float(args.reg_weight) * regularization
            )
            objective = safety_objective + resource_core
            components = [objective, quality_total, safety_objective, resource_core, survival_loss, utility_loss, download_loss, regularization]
            if not bool(torch.isfinite(torch.stack([value.float() for value in components])).all()):
                raise FloatingPointError(f"non-finite training loss at epoch={epoch + 1} step={step + 1}")

            safety_gradients = _gradient_tuple(safety_objective, parameters, retain_graph=True)
            resource_gradients = _gradient_tuple(resource_core, parameters, retain_graph=False)
            gradients, gradient_parts = _project_resource_gradients(
                safety_gradients,
                resource_gradients,
                enabled=not bool(args.disable_resource_gradient_projection),
            )
            _assign_gradients(parameters, gradients)
            bad_gradients = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            ]
            if bad_gradients:
                raise FloatingPointError(f"non-finite gradients: {bad_gradients}")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            gradient_parts["resourceGate"] = float(resource_gate)
            gradient_parts["qualityRecallEma"] = float(quality_ema)
            if gradient_parts["resourceGradientProjection"]:
                projection_count += 1
            step_metrics = _metric_values(
                {
                    "loss": objective,
                    "lossQualityResourceRvl": quality_total,
                    "lossSafetyCore": safety_core,
                    "lossResourceEffective": resource_core,
                    "lossSurvivalWeighted": float(args.survival_loss_weight) * survival_loss,
                    "lossUtilityWeighted": float(args.utility_loss_weight) * utility_loss,
                    "lossDownloadWeighted": float(args.download_loss_weight) * download_loss,
                    "lossRegularizationWeighted": float(args.reg_weight) * regularization,
                    **quality_parts,
                    **survival_parts,
                    **utility_parts,
                    **download_parts,
                },
                gradient_parts,
            )
            for key, value in step_metrics.items():
                epoch_metrics.setdefault(key, []).append(value)
            if (step + 1) % max(1, int(args.steps_per_epoch) // 5) == 0 or step == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "globalStep": global_step,
                            "loss": step_metrics.get("loss", 0.0),
                            "survival": step_metrics.get("lossSurvivalCensor", 0.0),
                            "qualityRecall": step_metrics.get("qualityRecall", 0.0),
                            "resourceGate": step_metrics.get("resourceGate", 0.0),
                            "progress": f"{global_step}/{total_steps}",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        scheduler.step()
        train_summary = {
            key: {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "count": int(len(values)),
            }
            for key, values in epoch_metrics.items()
            if values
        }
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "globalStep": global_step,
            "stage": "joint_offline_relation_and_pose_set",
            "trainableParameterCount": int(sum(parameter.numel() for parameter in parameters)),
            "processedSteps": int(processed_steps),
            "configuredSteps": int(args.steps_per_epoch),
            "lr": float(scheduler.get_last_lr()[0]),
            "resourceGateRate": float(resource_gate_count / max(1, processed_steps)),
            "resourceProjectionRate": float(projection_count / max(1, processed_steps)),
            "trainMetricSummary": train_summary,
            **{f"train_{key}": value["mean"] for key, value in train_summary.items()},
            "testRead": False,
        }

        calibration_payload: dict[str, Any] | None = None
        validation_row: dict[str, Any] | None = None
        coefficient_snapshot: torch.Tensor
        if (epoch + 1) % int(args.eval_every) == 0 or epoch == int(args.epochs) - 1:
            model.eval()
            with torch.no_grad():
                runtime_eval, coefficient_eval = _make_runtime_features(
                    model,
                    geometry,
                    relation_tensors,
                    local_group_ids,
                    structural_group_ids,
                )
            coefficient_snapshot = coefficient_eval.detach()
            calibration_rows = _eval_split(
                model,
                calibration_split,
                runtime_eval.detach(),
                world_aabbs,
                device,
                seed=int(args.seed) + epoch,
                thresholds=threshold_grid(),
                max_poses=int(args.max_eval_poses),
                bootstrap_replicates=int(args.calibration_bootstrap_replicates),
                label="calibration",
                poses_per_batch=int(args.pose_set_batch_size),
                max_candidates_per_pose=int(args.max_candidates_per_pose),
            )
            selected = (
                select_aggregate_weighted_cull_workpoint(
                    calibration_rows,
                    target_weighted_recall=CALIBRATION_FLOOR,
                    minimum_lower_confidence_bound=CALIBRATION_FLOOR,
                )
                if formal_calibration
                else None
            )
            diagnostic = _diagnostic_row(calibration_rows)
            chosen = selected or diagnostic
            threshold = None if chosen is None else float(chosen["threshold"])
            if threshold is not None:
                validation_rows = _eval_split(
                    model,
                    validation_split,
                    runtime_eval.detach(),
                    world_aabbs,
                    device,
                    seed=int(args.seed) + epoch + 1,
                    thresholds=np.asarray([threshold], dtype=np.float32),
                    max_poses=int(args.max_eval_poses),
                    bootstrap_replicates=int(args.calibration_bootstrap_replicates),
                    label="validation",
                    poses_per_batch=int(args.pose_set_batch_size),
                    max_candidates_per_pose=int(args.max_candidates_per_pose),
                )
                validation_row = validation_rows[0] if validation_rows else None
            calibration_payload = {
                "selected": selected,
                "diagnostic": diagnostic,
                "diagnosticThreshold": threshold,
                "rows": calibration_rows,
                "safe": selected is not None,
                "targetWeightedRecall": CALIBRATION_FLOOR,
                "lowerConfidenceBoundFloor": CALIBRATION_FLOOR,
                "testRead": False,
                "schema": "pvs-hierarchical-relation-survival-integrated-calibration-v2",
                "primarySafetyMetric": "aggregateWeightedRecall",
                "diagnosticSafetyMetric": "poseMacroWeightedRecall",
            }
            row["calibration"] = calibration_payload
            row["validationAtCalibration"] = validation_row
            row["safeCalibrationWorkpoint"] = selected is not None
            # Always keep the best diagnostic checkpoint.  It is selected only
            # from calibration and is explicitly marked unsafe when no formal
            # aggregate weighted-recall workpoint exists.
            if diagnostic is not None:
                diagnostic_key = (
                    _finite_or(diagnostic.get("aggregateWeightedRecallLowerConfidenceBound"), -1.0),
                    _finite_or(diagnostic.get("aggregateWeightedRecall"), -1.0),
                    _finite_or(diagnostic.get("agg_balanced_accuracy"), 0.0),
                    _finite_or(diagnostic.get("agg_useful_cull"), 0.0),
                    _finite_or(diagnostic.get("agg_precision"), 0.0),
                    -_finite_or(diagnostic.get("avg_pred_count"), 0.0),
                )
                if best_diagnostic_key is None or diagnostic_key > best_diagnostic_key:
                    best_diagnostic_key = diagnostic_key
                    best_diagnostic_calibration = calibration_payload
                    best_diagnostic_validation = validation_row
                    best_diagnostic_checkpoint_info = {
                        "threshold": float(diagnostic["threshold"]),
                        "safe": selected is not None,
                        "selection": diagnostic,
                    }
                    diagnostic_payload = _checkpoint_payload(
                        model,
                        args,
                        epoch + 1,
                        global_step,
                        protocol,
                        relation_meta,
                        relation_variant_meta,
                        geometry_meta,
                        coefficient_snapshot,
                        calibration_payload,
                        validation_row,
                        best=best_diagnostic_checkpoint_info,
                    )
                    torch.save(diagnostic_payload, output_dir / "best_diagnostic.pt")
                    _save_table(
                        output_dir / "best_diagnostic_instance_survival_coefficients_fp16.bin",
                        coefficient_snapshot,
                    )

            # A safe checkpoint is a separate artifact and is also exposed as
            # the historical ``best.pt`` only when the aggregate safety gate
            # is actually met.
            if selected is not None:
                candidate = selected
                candidate_key = (
                    float(candidate.get("agg_useful_cull", 0.0)),
                    float(candidate.get("agg_balanced_accuracy", 0.0)),
                    float(candidate.get("agg_precision", 0.0)),
                    -float(candidate.get("avg_pred_count", 0.0)),
                    float(candidate.get("aggregateWeightedRecall", 0.0)),
                )
                if best_safe_key is None or candidate_key > best_safe_key:
                    best_safe_key = candidate_key
                    best_safe_calibration = calibration_payload
                    best_safe_validation = validation_row
                    best_safe_checkpoint_info = {
                        "threshold": float(candidate["threshold"]),
                        "safe": True,
                        "selection": candidate,
                    }
                    safe_payload = _checkpoint_payload(
                        model,
                        args,
                        epoch + 1,
                        global_step,
                        protocol,
                        relation_meta,
                        relation_variant_meta,
                        geometry_meta,
                        coefficient_snapshot,
                        calibration_payload,
                        validation_row,
                        best=best_safe_checkpoint_info,
                    )
                    torch.save(safe_payload, output_dir / "best_safe.pt")
                    torch.save(safe_payload, output_dir / "best.pt")
                    _save_table(
                        output_dir / "best_safe_instance_survival_coefficients_fp16.bin",
                        coefficient_snapshot,
                    )
                    _save_table(
                        output_dir / "best_instance_survival_coefficients_fp16.bin",
                        coefficient_snapshot,
                    )

        else:
            with torch.no_grad():
                coefficient_snapshot = _make_runtime_features(
                    model,
                    geometry,
                    relation_tensors,
                    local_group_ids,
                    structural_group_ids,
                )[1].detach()
        latest_coefficients = coefficient_snapshot
        if int(args.snapshot_every) > 0 and (epoch + 1) % int(args.snapshot_every) == 0:
            torch.save(
                _checkpoint_payload(
                    model,
                    args,
                    epoch + 1,
                    global_step,
                    protocol,
                    relation_meta,
                    relation_variant_meta,
                    geometry_meta,
                    coefficient_snapshot,
                    calibration_payload,
                    validation_row,
                    best=best_safe_checkpoint_info or best_diagnostic_checkpoint_info,
                ),
                output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
            )
        row["elapsedSeconds"] = float(time.time() - started)
        history.append(row)
        _write_jsonl(metrics_path, row)
        _write_json(output_dir / "train_history.json", history)
        torch.save(
            _checkpoint_payload(
                model,
                args,
                epoch + 1,
                global_step,
                protocol,
                relation_meta,
                relation_variant_meta,
                geometry_meta,
                coefficient_snapshot,
                calibration_payload,
                validation_row,
            ),
            output_dir / "last.pt",
        )
        _save_table(output_dir / "instance_survival_coefficients_fp16.bin", coefficient_snapshot)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": row.get("train_loss", 0.0),
                    "safeCalibrationWorkpoint": row.get("safeCalibrationWorkpoint"),
                    "bestSafe": best_safe_checkpoint_info,
                    "bestDiagnostic": best_diagnostic_checkpoint_info,
                    "outputDir": str(output_dir),
                    "testRead": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if latest_coefficients is None:
        raise RuntimeError("no non-empty pose-set batch was processed")
    model.eval()
    with torch.no_grad():
        final_runtime, final_coefficients = _make_runtime_features(
            model,
            geometry,
            relation_tensors,
            local_group_ids,
            structural_group_ids,
        )
    _save_table(output_dir / "instance_runtime_features_fp16.bin", final_runtime)
    _save_table(output_dir / "instance_survival_coefficients_fp16.bin", final_coefficients)
    _save_table(output_dir / "instance_geo_features_fp16.bin", geometry)

    final_calibration = best_safe_calibration or best_diagnostic_calibration
    if final_calibration is None:
        for item in reversed(history):
            if item.get("calibration") is not None:
                final_calibration = item["calibration"]
                break
    calibration_summary = {
        "schema": "pvs-hierarchical-relation-survival-integrated-calibration-v2",
                "protocol": "calibration_ready_pre_test" if formal_calibration else "diagnostic_calibration_unsafe",
        "testRead": False,
        "primarySafetyMetric": "aggregateWeightedRecall",
        "diagnosticSafetyMetric": "poseMacroWeightedRecall",
        "selectionRule": aggregate_weighted_cull_selection_rule(CALIBRATION_FLOOR, CALIBRATION_FLOOR),
        "weightedRecallFloor": CALIBRATION_FLOOR,
        "weightedRecallLowerConfidenceBoundFloor": CALIBRATION_FLOOR,
        "bootstrapReplicates": int(args.calibration_bootstrap_replicates),
        "selected": (final_calibration or {}).get("selected"),
        "diagnostic": (final_calibration or {}).get("diagnostic"),
        "diagnosticThreshold": (final_calibration or {}).get("diagnosticThreshold"),
        "thresholdRows": (final_calibration or {}).get("rows", []),
        "validationAtCalibration": best_safe_validation or best_diagnostic_validation,
        "status": "safe" if (final_calibration or {}).get("selected") is not None else "no_qualified_safety_workpoint",
        "best": best_safe_checkpoint_info or best_diagnostic_checkpoint_info,
        "bestSafe": best_safe_checkpoint_info,
        "bestDiagnostic": best_diagnostic_checkpoint_info,
        "safeCheckpoint": str(output_dir / "best_safe.pt") if best_safe_checkpoint_info else None,
        "diagnosticCheckpoint": str(output_dir / "best_diagnostic.pt") if best_diagnostic_checkpoint_info else None,
    }
    _write_json(output_dir / "calibration_ready_summary.json", calibration_summary)
    model_meta["status"] = calibration_summary["status"]
    model_meta["export"] = {
        "runtimeFeatureShape": [num_instances, 124],
        "survivalCoefficientShape": [num_instances, 4, 7],
        "dtype": "float16",
        "files": {
            "geometry": "instance_geo_features_fp16.bin",
            "survivalCoefficients": "instance_survival_coefficients_fp16.bin",
            "runtime": "instance_runtime_features_fp16.bin",
            "bestSurvivalCoefficients": "best_instance_survival_coefficients_fp16.bin" if best_safe_checkpoint_info else None,
            "bestSafeSurvivalCoefficients": "best_safe_instance_survival_coefficients_fp16.bin" if best_safe_checkpoint_info else None,
            "bestDiagnosticSurvivalCoefficients": "best_diagnostic_instance_survival_coefficients_fp16.bin" if best_diagnostic_checkpoint_info else None,
        },
    }
    _write_json(output_dir / "model_meta.json", model_meta)
    _write_json(output_dir / "train_history.json", history)
    print(
        json.dumps(
            {
                "status": model_meta["status"],
                "outputDir": str(output_dir),
                "best": str(output_dir / "best.pt") if best_safe_checkpoint_info else None,
                "bestSafe": str(output_dir / "best_safe.pt") if best_safe_checkpoint_info else None,
                "bestDiagnostic": str(output_dir / "best_diagnostic.pt") if best_diagnostic_checkpoint_info else None,
                "last": str(output_dir / "last.pt"),
                "calibration": str(output_dir / "calibration_ready_summary.json"),
                "testRead": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
