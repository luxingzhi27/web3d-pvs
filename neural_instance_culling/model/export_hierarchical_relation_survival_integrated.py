#!/usr/bin/env python3
"""Export the runtime bundle for the hierarchical relation-survival model.

The trainer has two deliberately separate phases.  The relation encoder is an
offline training component; the browser only needs fixed geometry, the baked
survival field, scene AABBs, the instance-to-GLB table, and the lightweight
view-cell query network.  This exporter uses an explicit allow-list for query
weights so the relation encoder, relation CSR, hierarchy tables, and source
IDs cannot accidentally become deployment assets.

The command requires an explicit output directory and never overwrites one.
``--dry-run`` performs the same input and schema checks as a real export but
does not create the directory or write any files.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from hierarchical_relation_survival_integrated_model import (  # noqa: E402
    GEO_DIM,
    HierarchicalRelationSurvivalIntegratedModel,
    MODEL_SCHEMA,
    RELATION_CONDITION_DIM,
    SURVIVAL_PARAMETER_DIM,
    SURVIVAL_RANK,
    SURVIVAL_SEMANTIC_DIM,
    VIEW_DIM,
)
from common.provenance import protocol_digest  # noqa: E402
from common.viewcell_integrated_spectral_query import (  # noqa: E402
    FREQUENCY_COUNT,
    HIGH_FREQUENCY_COUNT,
    INTEGRATED_COMPONENTS_PER_FREQUENCY,
    LOW_FREQUENCY_COUNT,
    MAX_QUERY_BASIS_DIM,
    MIDDLE_FREQUENCY_COUNT,
    SPECTRAL_FEATURE_DIM,
)


TRAINING_SCHEMA = "pvs-hierarchical-relation-survival-integrated-training-v1"
EXPORT_SCHEMA = "pvs-hierarchical-relation-survival-integrated-runtime-v1"
QUERY_WEIGHTS_SCHEMA = "pvs-hierarchical-relation-survival-integrated-query-weights-v1"
TARGET_WEIGHTED_RECALL = 0.99
MINIMUM_WEIGHTED_RECALL_LCB = 0.99
RUNTIME_FEATURE_DIM = GEO_DIM + SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM
RUNTIME_INPUT_DIM = (
    GEO_DIM + SURVIVAL_RANK + SURVIVAL_SEMANTIC_DIM + VIEW_DIM + 2 + 1
)
SELF_CHECK_SCHEMA = "pvs-hierarchical-relation-survival-integrated-self-check-v1"
RUNTIME_CONSTANTS_SCHEMA = "pvs-hierarchical-relation-survival-integrated-runtime-constants-v1"
SELF_CHECK_MAX_ABS_ERROR = 5e-2

def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    if isinstance(value, float) and value != result:
        raise ValueError(f"{name} must be an integer")
    return result


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - numpy supplies the detail
        raise ValueError(f"cannot convert {name} to an array") from exc


def _to_fp16(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = _to_numpy(value, name)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(array.shape)}")
    try:
        values = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} contains non-finite values")
    result = np.ascontiguousarray(values.astype("<f2", copy=False))
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} cannot be represented as finite FP16 values")
    return result


def _to_fp32(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Validate a non-neural table that must retain its boundary precision."""

    array = _to_numpy(value, name)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(array.shape)}")
    try:
        values = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(values.astype("<f4", copy=False))


def _read_fp16_table(path: Path, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name} table: {path}")
    values = np.fromfile(path, dtype="<f2")
    expected = int(np.prod(shape, dtype=np.int64))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} FP16 values; expected {expected}")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} table contains non-finite FP16 values: {path}")
    return np.ascontiguousarray(values.reshape(shape))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    return dict(payload)


def _validate_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    if checkpoint.get("schema") != TRAINING_SCHEMA:
        raise ValueError(
            f"unexpected checkpoint schema: {checkpoint.get('schema')!r}; "
            f"expected {TRAINING_SCHEMA!r}"
        )
    if checkpoint.get("modelSchema") != MODEL_SCHEMA:
        raise ValueError(
            f"unexpected modelSchema: {checkpoint.get('modelSchema')!r}; "
            f"expected {MODEL_SCHEMA!r}"
        )
    if checkpoint.get("testRead") is not False:
        raise ValueError("refusing to export a checkpoint without testRead=false")

    config = dict(_as_mapping(checkpoint.get("config"), "checkpoint.config"))
    if config.get("runtimeSchema") != MODEL_SCHEMA:
        raise ValueError(
            f"config.runtimeSchema must be {MODEL_SCHEMA!r}, got "
            f"{config.get('runtimeSchema')!r}"
        )
    num_instances = _positive_int(config.get("numInstances"), "config.numInstances")
    num_glbs = _positive_int(config.get("numGlbs"), "config.numGlbs", allow_zero=True)
    geo_dim = _positive_int(config.get("geoDim"), "config.geoDim")
    coefficient_shape = config.get("survivalCoefficientShape")
    if list(coefficient_shape or ()) != [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM]:
        raise ValueError(
            "config.survivalCoefficientShape must be "
            f"[{SURVIVAL_RANK}, {SURVIVAL_PARAMETER_DIM}]"
        )
    if _positive_int(config.get("survivalCoefficientDim"), "config.survivalCoefficientDim") != SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM:
        raise ValueError("config.survivalCoefficientDim does not match [4, 7]")
    expected_dimensions = {
        "geoDim": GEO_DIM,
        "viewCellCenterDim": VIEW_DIM,
        "viewCellVarianceDim": VIEW_DIM,
        "survivalSemanticDim": SURVIVAL_SEMANTIC_DIM,
        "relationConditionDim": RELATION_CONDITION_DIM,
        "queryBasisDim": SURVIVAL_RANK,
        "runtimeInputDim": RUNTIME_INPUT_DIM,
        "runtimeFeatureDim": RUNTIME_FEATURE_DIM,
    }
    for key, expected in expected_dimensions.items():
        actual = _positive_int(config.get(key), f"config.{key}")
        if actual != expected:
            raise ValueError(f"config.{key}={actual} does not match expected {expected}")
    hidden_dim = _positive_int(config.get("hiddenDim"), "config.hiddenDim")
    relation_hidden_dim = _positive_int(config.get("relationHiddenDim"), "config.relationHiddenDim")
    spectral_mode = str(config.get("spectralMode", ""))
    if spectral_mode not in {"integrated", "learned_point", "fourier117"}:
        raise ValueError(f"unsupported config.spectralMode: {spectral_mode!r}")

    heads = _as_mapping(config.get("heads"), "config.heads")
    expected_heads = {
        "visibility": [hidden_dim, 1],
        "utility": [hidden_dim + 1, 32, 1],
        "download": [hidden_dim + 1, 32, 1],
    }
    for name, expected in expected_heads.items():
        if list(heads.get(name) or ()) != expected:
            raise ValueError(f"config.heads.{name} must be {expected}")

    # Only this small runtime subset is copied into model_meta.  In
    # particular, do not serialize config.offlineOnly or any trainer args.
    runtime_config = {
        "runtimeSchema": MODEL_SCHEMA,
        "numInstances": num_instances,
        "numGlbs": num_glbs,
        "geoDim": geo_dim,
        "survivalCoefficientShape": [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
        "survivalCoefficientDim": SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM,
        "viewCellCenterDim": VIEW_DIM,
        "viewCellVarianceDim": VIEW_DIM,
        "survivalSemanticDim": SURVIVAL_SEMANTIC_DIM,
        "relationConditionDim": RELATION_CONDITION_DIM,
        "queryBasisDim": SURVIVAL_RANK,
        "runtimeInputDim": RUNTIME_INPUT_DIM,
        "runtimeFeatureDim": RUNTIME_FEATURE_DIM,
        "hiddenDim": hidden_dim,
        "relationHiddenDim": relation_hidden_dim,
        "spectralMode": spectral_mode,
        "heads": expected_heads,
    }
    return runtime_config


def _training_provenance(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and retain the immutable training provenance for export."""
    protocol = _as_mapping(checkpoint.get("protocol"), "checkpoint.protocol")
    recorded_protocol_digest = checkpoint.get("protocolDigest")
    if recorded_protocol_digest != protocol_digest(protocol):
        raise ValueError("checkpoint protocolDigest does not match canonical protocol")
    if protocol.get("testRead") is not False or "testPoseCount" in protocol:
        raise ValueError("checkpoint protocol is not train/calibration/validation-only")

    relation_provenance = _as_mapping(
        checkpoint.get("relationProvenance"), "checkpoint.relationProvenance"
    )
    relation_meta = _as_mapping(
        relation_provenance.get("relationMeta"),
        "checkpoint.relationProvenance.relationMeta",
    )
    if relation_meta.get("schema") != "pvs-viewcell-train-observed-relation-csr-v1":
        raise ValueError("checkpoint relation provenance has an unexpected schema")
    identity = _as_mapping(relation_meta.get("candidateIdentity"), "relation candidateIdentity")
    canonical_digest = str(identity.get("canonicalCandidateDigest", ""))
    render_digest = str(identity.get("renderCandidateDigest", ""))
    if len(canonical_digest) != 64 or len(render_digest) != 64:
        raise ValueError("relation candidate provenance is missing SHA-256 digests")
    if canonical_digest != str(protocol.get("trainCandidateDigest", "")):
        raise ValueError("relation canonical candidate digest disagrees with protocol")
    artifact_digest = str(relation_provenance.get("relationArtifactDigest", ""))
    if len(artifact_digest) != 64:
        raise ValueError("checkpoint relation artifact digest is missing")
    variant = str(checkpoint.get("variant", ""))
    variant_spec = _as_mapping(checkpoint.get("variantSpec"), "checkpoint.variantSpec")
    relation_variant = _as_mapping(
        relation_provenance.get("variant"), "checkpoint.relationProvenance.variant"
    )
    return {
        "protocolDigest": str(recorded_protocol_digest),
        "splitDigests": {
            "train": str(protocol["trainCandidateDigest"]),
            "calibration": str(protocol["calibrationCandidateDigest"]),
            "validation": str(protocol["validationCandidateDigest"]),
        },
        "variant": variant,
        "variantSpec": dict(variant_spec),
        "relationVariant": dict(relation_variant),
        "relation": {
            "schema": str(relation_meta["schema"]),
            "canonicalCandidateDigest": canonical_digest,
            "renderCandidateDigest": render_digest,
            "relationArtifactDigest": artifact_digest,
            "inputSha256": dict(relation_meta.get("inputSha256", {})),
        },
    }


def _linear_specs(prefix: str, dimensions: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    specs: list[tuple[str, tuple[int, ...]]] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
        module_index = index * 2
        specs.extend([
            (f"{prefix}.{module_index}.weight", (output_dim, input_dim)),
            (f"{prefix}.{module_index}.bias", (output_dim,)),
        ])
    return specs


def _query_weight_specs(config: Mapping[str, Any]) -> list[tuple[str, tuple[int, ...]]]:
    hidden_dim = int(config["hiddenDim"])
    mode = str(config["spectralMode"])
    specs: list[tuple[str, tuple[int, ...]]] = []
    if mode in {"integrated", "learned_point"}:
        specs.append(("integrated_query.frequency_vectors", (16, VIEW_DIM)))
        specs.extend(_linear_specs("integrated_query.low_frequency_body", (45, 24, 16)))
        specs.extend(_linear_specs("integrated_query.high_frequency_body", (14, 16, 12)))
        specs.extend(_linear_specs("integrated_query.relation_body", (RELATION_CONDITION_DIM, 12, 8)))
        specs.extend(_linear_specs("integrated_query.high_frequency_gate", (12, 12, 1)))
        specs.extend([
            ("integrated_query.query_fuse.weight", (SURVIVAL_RANK, 36)),
            ("integrated_query.query_fuse.bias", (SURVIVAL_RANK,)),
        ])
    else:
        specs.extend(_linear_specs("fourier_query", (125, 32, SURVIVAL_RANK)))
    specs.extend(_linear_specs("relation_condition_head", (28, 16, RELATION_CONDITION_DIM)))
    specs.extend(_linear_specs("shared_trunk", (RUNTIME_INPUT_DIM, hidden_dim, hidden_dim)))
    specs.extend([
        ("visibility_head.weight", (1, hidden_dim)),
        ("visibility_head.bias", (1,)),
    ])
    specs.extend(_linear_specs("utility_head", (hidden_dim + 1, 32, 1)))
    specs.extend(_linear_specs("download_head", (hidden_dim + 1, 32, 1)))
    return specs


def _pack_query_weights(
    state: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[bytes, list[dict[str, Any]]]:
    specs = _query_weight_specs(config)
    chunks: list[bytes] = []
    layout: list[dict[str, Any]] = []
    offset_elements = 0
    for name, shape in specs:
        if name not in state:
            raise ValueError(f"checkpoint is missing runtime query weight: {name}")
        array = _to_numpy(state[name], name)
        if tuple(array.shape) != shape:
            raise ValueError(f"{name} has shape {list(array.shape)}; expected {list(shape)}")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} must be a floating-point tensor")
        value = _to_fp16(array, name, shape)
        raw = value.tobytes(order="C")
        chunks.append(raw)
        layout.append({
            "name": name,
            "offsetElements": int(offset_elements),
            "offsetBytes": int(offset_elements * 2),
            "shape": list(shape),
            "dtype": "float16",
            "elementCount": int(value.size),
            "byteLength": len(raw),
        })
        offset_elements += int(value.size)
    return b"".join(chunks), layout


def _resolve_input_path(value: str | None, *, checkpoint_dir: Path, name: str) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {name}: {candidate}")
    return candidate


def _load_geometry(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    explicit_path: str | None,
    num_instances: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    geometry_meta = checkpoint.get("geometryMeta")
    geometry_meta = dict(geometry_meta) if isinstance(geometry_meta, Mapping) else {}
    candidates: list[Path] = []
    explicit = _resolve_input_path(explicit_path, checkpoint_dir=checkpoint_path.parent, name="geometry")
    if explicit is not None:
        candidates.append(explicit)
    metadata_path = geometry_meta.get("path")
    if explicit is None and metadata_path:
        candidate = Path(str(metadata_path)).expanduser()
        if not candidate.is_absolute():
            candidate = checkpoint_path.parent / candidate
        candidates.append(candidate.resolve())
    if explicit is None:
        candidates.extend([
            checkpoint_path.parent / "best_instance_geo_features_fp16.bin",
            checkpoint_path.parent / "instance_geo_features_fp16.bin",
        ])
    path: Path | None = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"cannot locate FP16 geometry table; searched: {searched}")
    values = _read_fp16_table(path, (num_instances, GEO_DIM), "geometry")
    expected_hash = geometry_meta.get("sha256")
    if expected_hash and _sha256_file(path) != str(expected_hash):
        raise ValueError(f"geometry table hash does not match checkpoint.geometryMeta: {path}")
    if geometry_meta.get("numInstances") is not None and int(geometry_meta["numInstances"]) != num_instances:
        raise ValueError("checkpoint.geometryMeta.numInstances does not match config.numInstances")
    if geometry_meta.get("dim") is not None and int(geometry_meta["dim"]) != GEO_DIM:
        raise ValueError("checkpoint.geometryMeta.dim does not match the registered geometry schema")
    return values, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "shape": [num_instances, GEO_DIM],
        "dtype": "float16",
    }


def _load_survival_coefficients(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    explicit_path: str | None,
    num_instances: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_shape = (num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM)
    explicit = _resolve_input_path(
        explicit_path, checkpoint_dir=checkpoint_path.parent, name="survival coefficients"
    )
    if "survivalCoefficients" in checkpoint:
        values = _to_fp16(
            checkpoint["survivalCoefficients"],
            "checkpoint.survivalCoefficients",
            expected_shape,
        )
        if explicit is not None:
            external = _read_fp16_table(explicit, expected_shape, "survival coefficients")
            if not bool(np.array_equal(external, values)):
                raise ValueError(
                    "explicit survival coefficient table disagrees with "
                    "checkpoint.survivalCoefficients after FP16 conversion"
                )
        return values, {
            "source": "checkpoint.survivalCoefficients",
            "shape": list(expected_shape),
            "dtype": "float16",
        }

    if explicit is not None:
        values = _read_fp16_table(explicit, expected_shape, "survival coefficients")
        return values, {
            "path": str(explicit),
            "sha256": _sha256_file(explicit),
            "shape": list(expected_shape),
            "dtype": "float16",
            "source": "explicit_fp16_table",
        }

    for name in ("best_instance_survival_coefficients_fp16.bin", "instance_survival_coefficients_fp16.bin"):
        candidate = checkpoint_path.parent / name
        if candidate.is_file():
            values = _read_fp16_table(
                candidate,
                expected_shape,
                "survival coefficients",
            )
            return values, {
                "path": str(candidate),
                "sha256": _sha256_file(candidate),
                "shape": list(expected_shape),
                "dtype": "float16",
                "source": "sibling_fp16_table",
            }
    raise ValueError("checkpoint has no survivalCoefficients tensor or FP16 coefficient table")


def _bounds_min_max(bounds: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    if "min" in bounds and "max" in bounds:
        minimum = np.asarray(bounds["min"], dtype=np.float32)
        maximum = np.asarray(bounds["max"], dtype=np.float32)
    elif "center" in bounds and "size" in bounds:
        center = np.asarray(bounds["center"], dtype=np.float32)
        size = np.asarray(bounds["size"], dtype=np.float32)
        if center.shape != (3,) or size.shape != (3,):
            raise ValueError(f"{name} center/size must each have shape [3]")
        minimum = center - size * 0.5
        maximum = center + size * 0.5
    else:
        raise ValueError(f"{name} must contain min/max or center/size")
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError(f"{name} bounds must each have shape [3]")
    if not bool(np.isfinite(minimum).all() and np.isfinite(maximum).all()):
        raise ValueError(f"{name} contains non-finite bounds")
    if bool((minimum > maximum).any()):
        raise ValueError(f"{name} has min greater than max")
    return minimum, maximum


def _load_runtime_tables(
    path: Path, num_instances: int, num_glbs: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing runtime meta: {path}")
    try:
        runtime_meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid runtime meta JSON: {path}") from exc
    runtime_meta = dict(_as_mapping(runtime_meta, "runtime meta"))
    if runtime_meta.get("schemaVersion") is not None:
        schema_version = _positive_int(
            runtime_meta["schemaVersion"], "runtime meta.schemaVersion", allow_zero=True
        )
        if schema_version != 2:
            raise ValueError(f"unsupported runtime meta schemaVersion: {schema_version}")
    id_spaces = runtime_meta.get("idSpaces")
    if id_spaces is not None:
        if not isinstance(id_spaces, list) or not {"component", "global-glb"}.issubset(set(id_spaces)):
            raise ValueError("runtime meta.idSpaces must include component and global-glb")
    for key, expected in (
        ("componentCount", num_instances),
        ("instanceCount", num_instances),
        ("globalGlbCount", num_glbs),
    ):
        if runtime_meta.get(key) is not None:
            actual = _positive_int(runtime_meta[key], f"runtime meta.{key}")
            if actual != expected:
                raise ValueError(
                    f"runtime meta.{key}={actual} does not match expected {expected}"
                )
    records = runtime_meta.get("componentRecords")
    if not isinstance(records, list) or len(records) != num_instances:
        raise ValueError(
            "runtime meta componentRecords must contain exactly "
            f"{num_instances} records"
        )
    aabbs = np.empty((num_instances, 6), dtype=np.float32)
    mapping = np.empty((num_instances,), dtype=np.int64)
    seen: set[int] = set()
    for record_index, raw_record in enumerate(records):
        record = _as_mapping(raw_record, f"componentRecords[{record_index}]")
        component_id = _positive_int(
            record.get("componentGlobalId"),
            f"componentRecords[{record_index}].componentGlobalId",
            allow_zero=True,
        )
        if component_id >= num_instances or component_id in seen:
            raise ValueError("runtime meta componentGlobalId values must be a unique 0..N-1 permutation")
        seen.add(component_id)
        glb_id = _positive_int(
            record.get("globalGlbId"),
            f"componentRecords[{record_index}].globalGlbId",
            allow_zero=True,
        )
        if glb_id >= 2**32:
            raise ValueError("globalGlbId does not fit the exported uint32 mapping")
        minimum, maximum = _bounds_min_max(
            _as_mapping(record.get("bounds"), f"componentRecords[{record_index}].bounds"),
            f"componentRecords[{record_index}].bounds",
        )
        aabbs[component_id, :3] = minimum
        aabbs[component_id, 3:] = maximum
        mapping[component_id] = glb_id
    if seen != set(range(num_instances)):
        raise ValueError("runtime meta componentGlobalId values must cover every instance")
    if num_glbs == 0 and num_instances:
        raise ValueError("config.numGlbs is zero but runtime meta contains instances")
    if mapping.size and int(mapping.max()) >= num_glbs:
        raise ValueError("runtime meta globalGlbId exceeds config.numGlbs")
    if mapping.size and int(mapping.max()) != num_glbs - 1:
        raise ValueError("runtime meta globalGlbId range does not cover config.numGlbs")

    scene_bounds = _as_mapping(runtime_meta.get("sceneBounds"), "runtime meta.sceneBounds")
    scene_min, scene_max = _bounds_min_max(scene_bounds, "runtime meta.sceneBounds")
    scene = {
        "min": scene_min.astype(np.float32).tolist(),
        "max": scene_max.astype(np.float32).tolist(),
    }
    # Keep the spatial predicate in FP32.  Neural tables are quantized below,
    # but rounding an AABB at a page/frustum boundary can change candidate
    # identity and is unrelated to the model's FP16 quantization.
    aabb_fp32 = _to_fp32(aabbs, "instance AABB", (num_instances, 6))
    mapping_u32 = np.ascontiguousarray(mapping.astype("<u4", copy=False))
    return aabb_fp32, mapping_u32, scene, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "sceneBounds": scene,
    }


def _threshold_value(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in row and row[name] is not None:
            return _finite_float(row[name], f"calibration.{name}")
    raise ValueError(f"calibration workpoint is missing one of: {', '.join(names)}")


def _compact_workpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "threshold",
        "aggregateWeightedRecall",
        "aggregateWeightedRecallLowerConfidenceBound",
        "poseMacroWeightedRecall",
        "poseMacroWeightedRecallLowerConfidenceBound",
        "pose_weighted_recall",
        "weighted_recall_lower_confidence_bound",
        "pose_recall",
        "pose_precision",
        "pose_f1",
        "pose_balanced_accuracy",
        "pose_specificity",
        "pose_useful_cull",
        "pose_bad_cull",
        "avg_pred_count",
        "avg_gt_count",
        "avg_candidate_count",
    )
    result: dict[str, Any] = {}
    for key in keys:
        if key in row and row[key] is not None:
            result[key] = _finite_float(row[key], f"calibration.{key}")
    return result


def _resolve_threshold(
    checkpoint: Mapping[str, Any], requested: float | None, allow_unsafe: bool
) -> tuple[float, dict[str, Any]]:
    calibration = _as_mapping(checkpoint.get("calibration"), "checkpoint.calibration")
    selected = calibration.get("selected")
    if not isinstance(selected, Mapping):
        if not allow_unsafe:
            raise ValueError(
                "checkpoint.calibration.selected is missing; refusing to export without "
                "a calibration-frozen threshold"
            )
        selected = calibration.get("diagnostic") or checkpoint.get("best")
    if not isinstance(selected, Mapping):
        raise ValueError("checkpoint has no calibration workpoint containing a threshold")

    threshold = _threshold_value(selected, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"calibration threshold must be in [0, 1], got {threshold}")
    # These are the evaluator's canonical pooled-candidate fields.  Pose-macro
    # values remain diagnostics and must never become an export safety gate.
    weighted_recall = _threshold_value(selected, "aggregateWeightedRecall")
    lower_bound = _threshold_value(selected, "aggregateWeightedRecallLowerConfidenceBound")
    safe = weighted_recall > TARGET_WEIGHTED_RECALL and lower_bound > MINIMUM_WEIGHTED_RECALL_LCB
    if not safe and not allow_unsafe:
        raise ValueError(
            "checkpoint calibration threshold is unsafe: "
            f"weighted recall={weighted_recall:.8f}, lower bound={lower_bound:.8f}; "
            "pass --allow-unsafe-threshold only for a diagnostic export"
        )
    if requested is not None:
        requested_value = _finite_float(requested, "--threshold")
        if not 0.0 <= requested_value <= 1.0:
            raise ValueError("--threshold must be in [0, 1]")
        if abs(requested_value - threshold) > 1e-7:
            raise ValueError(
                "--threshold must equal the checkpoint's calibration workpoint; "
                "threshold scanning and unrecorded overrides are not allowed"
            )
    best = checkpoint.get("best")
    if isinstance(best, Mapping) and best.get("threshold") is not None:
        best_threshold = _finite_float(best["threshold"], "checkpoint.best.threshold")
        if abs(best_threshold - threshold) > 1e-7:
            raise ValueError("checkpoint.best.threshold disagrees with calibration workpoint")
    if isinstance(best, Mapping) and best.get("safe") is not None:
        if bool(best["safe"]) != safe:
            raise ValueError("checkpoint.best.safe disagrees with calibration workpoint")
    return threshold, {
        "source": "checkpoint.calibration.selected",
        "protocol": "calibration_ready_pre_test" if safe else "diagnostic_calibration",
        "targetWeightedRecall": TARGET_WEIGHTED_RECALL,
        "minimumWeightedRecallLowerConfidenceBound": MINIMUM_WEIGHTED_RECALL_LCB,
        "weightedRecallField": "aggregateWeightedRecall",
        "weightedRecallLowerConfidenceBoundField": "aggregateWeightedRecallLowerConfidenceBound",
        "safe": safe,
        "selected": _compact_workpoint(selected),
        "testEvaluationCount": 0,
    }


def _file_descriptor(
    name: str, dtype: str, shape: list[int], raw: bytes, *, source: str | None = None
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "file": name,
        "dtype": dtype,
        "shape": shape,
        "byteLength": len(raw),
        "sha256": _sha256_bytes(raw),
    }
    if source is not None:
        descriptor["source"] = source
    return descriptor


def _build_model_meta(
    *,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    geometry: np.ndarray,
    coefficient: np.ndarray,
    aabb: np.ndarray,
    instance_to_glb: np.ndarray,
    query_weights: bytes,
    query_layout: list[dict[str, Any]],
    scene_bounds: dict[str, Any],
    runtime_meta_info: Mapping[str, Any],
    geometry_info: Mapping[str, Any],
    coefficient_info: Mapping[str, Any],
    threshold: float,
    threshold_info: Mapping[str, Any],
    output_files: Mapping[str, bytes],
    training_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    files = {
        "geometry": _file_descriptor(
            "instance_geo_features_fp16.bin",
            "float16",
            list(geometry.shape),
            output_files["instance_geo_features_fp16.bin"],
            source=str(geometry_info.get("source", geometry_info.get("path", "checkpoint"))),
        ),
        "survivalCoefficients": _file_descriptor(
            "instance_survival_coefficients_fp16.bin",
            "float16",
            list(coefficient.shape),
            output_files["instance_survival_coefficients_fp16.bin"],
            source=str(coefficient_info.get("source", coefficient_info.get("path", "checkpoint"))),
        ),
        "instanceAabb": _file_descriptor(
            "instance_aabb_fp32.bin", "float32", list(aabb.shape), output_files["instance_aabb_fp32.bin"]
        ),
        "instanceToGlb": _file_descriptor(
            "instance_to_glb_uint32.bin", "uint32", list(instance_to_glb.shape), output_files["instance_to_glb_uint32.bin"]
        ),
        "queryWeights": {
            **_file_descriptor("query_weights_fp16.bin", "float16", [sum(item["elementCount"] for item in query_layout)], query_weights),
            "schema": QUERY_WEIGHTS_SCHEMA,
            "layout": query_layout,
        },
    }
    payload_bytes = sum(int(item["byteLength"]) for item in files.values() if "byteLength" in item)
    if runtime_config["spectralMode"] in {"integrated", "learned_point"}:
        query_input_layout = {
            "centerView": {
                "dim": VIEW_DIM,
                "components": [
                    "rayDirectionX",
                    "rayDirectionY",
                    "rayDirectionZ",
                    "normalizedLogDistance",
                    "dotForward",
                    "screenU",
                    "screenV",
                    "horizontalAngularSize",
                    "verticalAngularSize",
                ],
            },
            "diagonalVariance": {
                "dim": VIEW_DIM,
                "nonNegative": True,
                "source": "analytic_viewcell_diagonal_variance",
            },
            "rho": {"dim": 1, "range": [0.0, 1.0]},
            "integratedSpectrum": {
                "frequencyCount": 16,
                "frequencyVectorShape": [16, VIEW_DIM],
                "componentsPerFrequency": 3,
                "featureDim": 57,
                "featureOrder": "centerView_then_frequency_major_triplets",
                "tripletOrder": ["attenuatedSin", "attenuatedCos", "unresolvedPhaseEnergy"],
                "frequencyParameterization": "bounded_l2_norm",
                "maximumFrequencyNorm": 8.0,
            },
            "relationConditionDim": RELATION_CONDITION_DIM,
            "oneCenterQueryPerCandidateBatch": True,
        }
    else:
        query_input_layout = {
            "centerView": {
                "dim": VIEW_DIM,
                "components": [
                    "rayDirectionX",
                    "rayDirectionY",
                    "rayDirectionZ",
                    "normalizedLogDistance",
                    "dotForward",
                    "screenU",
                    "screenV",
                    "horizontalAngularSize",
                    "verticalAngularSize",
                ],
            },
            "diagonalVariance": {
                "dim": VIEW_DIM,
                "acceptedButUnusedByFourier117Basis": True,
            },
            "rho": {"dim": 1, "range": [0.0, 1.0]},
            "fourier117": {
                "dim": 117,
                "encoding": "values_then_each_band_sin_cos",
                "segments": [
                    {
                        "name": "centerViewDirection",
                        "offset": 0,
                        "dim": 63,
                        "sourceDim": 3,
                        "bands": 10,
                    },
                    {
                        "name": "centerViewScalars",
                        "offset": 63,
                        "dim": 54,
                        "sourceDim": 6,
                        "bands": 4,
                    },
                ],
            },
            "relationConditionDim": RELATION_CONDITION_DIM,
            "oneCenterQueryPerCandidateBatch": True,
        }
    spectral_query: dict[str, Any] = {
        "mode": runtime_config["spectralMode"],
        "centerViewDim": VIEW_DIM,
        "diagonalVarianceDim": VIEW_DIM,
        "rhoDim": 1,
        "queryBasisDim": SURVIVAL_RANK,
        "survivalSemanticDim": SURVIVAL_SEMANTIC_DIM,
        "survivalParameterization": "monotone_two_logistic_survival",
        "oneCenterQueryPerCandidateBatch": True,
        "subposeExpansion": False,
    }
    if runtime_config["spectralMode"] in {"integrated", "learned_point"}:
        spectral_query.update({
            "integratedFrequencyCount": 16,
            "integratedSpectralFeatureDim": 57,
        })
    else:
        spectral_query.update({
            "fourierFeatureDim": 117,
            "fourierEncoding": "values_then_each_band_sin_cos",
        })
    meta: dict[str, Any] = {
        "schema": EXPORT_SCHEMA,
        "checkpointSchema": checkpoint.get("schema"),
        "modelSchema": MODEL_SCHEMA,
        "checkpointSha256": _sha256_file(checkpoint_path),
        "checkpointPathForAudit": str(checkpoint_path),
        "numInstances": int(runtime_config["numInstances"]),
        "numGlbs": int(runtime_config["numGlbs"]),
        "runtimeFeatureDim": RUNTIME_FEATURE_DIM,
        "runtimeFeatureLayout": {
            "geometry": {"offset": 0, "dim": GEO_DIM, "shape": ["N", GEO_DIM]},
            "survivalCoefficients": {
                "offset": GEO_DIM,
                "dim": SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM,
                "shape": ["N", SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
            },
        },
        "modelConfig": dict(runtime_config),
        "queryInputLayout": query_input_layout,
        "spectralQuery": spectral_query,
        "queryOutputs": {
            "visibilityLogits": ["B", 1],
            "utilityLogits": ["B", 1],
            "downloadLogits": ["B", 1],
            "survivalSemantic": ["B", SURVIVAL_SEMANTIC_DIM],
        },
        "runtimeTables": {
            "geometry": {"required": True, "lookup": "instanceIds"},
            "survivalCoefficients": {"required": True, "lookup": "instanceIds"},
            "instanceAabb": {"required": True, "lookup": "instanceIds"},
            "instanceToGlb": {"required": True, "lookup": "instanceIds"},
        },
        "files": files,
        "instanceToGlobalGlb": instance_to_glb.astype(np.uint32, copy=False).tolist(),
        "threshold": float(threshold),
        "thresholdSpace": "visibility_probability",
        "thresholdSelection": dict(threshold_info),
        "trainingProvenance": dict(training_provenance),
        "sceneBounds": scene_bounds,
        "runtimeMetaPathForAudit": str(runtime_meta_info["path"]),
        "runtimeMetaSha256": runtime_meta_info["sha256"],
        "geometryInput": dict(geometry_info),
        "survivalCoefficientInput": dict(coefficient_info),
        "runtimeSemantics": "fixed offline tables; one lightweight view-cell query per candidate batch",
        "trainingArtifactsIncluded": False,
        "relationGraphDataIncluded": False,
        "onlineGraphPropagation": False,
        "onlineNeighborSearch": False,
        "onlinePointEncoder": False,
        "onlineAabbCornerProjection": False,
        "bytes": {
            "runtimeTablesAndQueryWeights": int(payload_bytes),
            "modelMeta": "written_after_validation",
        },
    }
    return meta


def _assert_planned_files_are_runtime_only(files: Mapping[str, bytes]) -> None:
    forbidden = ("relation_csr", "source_ids", "local_group_ids", "structural_group_ids", "hierarchy_ids")
    for name in files:
        lowered = name.lower()
        if any(token in lowered for token in forbidden):
            raise AssertionError(f"forbidden offline artifact in export plan: {name}")


def _prepare_export(args: argparse.Namespace) -> tuple[Path, dict[str, bytes], dict[str, Any], float]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    runtime_meta_path = Path(args.runtime_meta).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == checkpoint_path.parent or output_dir == runtime_meta_path.parent:
        raise ValueError("output-dir must be an independent directory, not an input directory")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to use existing output directory: {output_dir}; choose a new independent directory"
        )

    checkpoint = _load_checkpoint(checkpoint_path)
    runtime_config = _validate_model_config(checkpoint)
    training_provenance = _training_provenance(checkpoint)
    num_instances = int(runtime_config["numInstances"])
    num_glbs = int(runtime_config["numGlbs"])
    geometry, geometry_info = _load_geometry(
        checkpoint, checkpoint_path, args.geometry, num_instances
    )
    coefficients, coefficient_info = _load_survival_coefficients(
        checkpoint, checkpoint_path, args.survival_coefficients, num_instances
    )
    aabb, instance_to_glb, scene_bounds, runtime_meta_info = _load_runtime_tables(
        runtime_meta_path, num_instances, num_glbs
    )
    state = _as_mapping(checkpoint.get("model"), "checkpoint.model")
    query_weights, query_layout = _pack_query_weights(state, runtime_config)
    threshold, threshold_info = _resolve_threshold(
        checkpoint, args.threshold, bool(args.allow_unsafe_threshold)
    )

    output_files: dict[str, bytes] = {
        "instance_geo_features_fp16.bin": geometry.tobytes(order="C"),
        "instance_survival_coefficients_fp16.bin": coefficients.tobytes(order="C"),
        "instance_aabb_fp32.bin": aabb.tobytes(order="C"),
        "instance_to_glb_uint32.bin": instance_to_glb.tobytes(order="C"),
        "query_weights_fp16.bin": query_weights,
    }
    _assert_planned_files_are_runtime_only(output_files)
    meta = _build_model_meta(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        runtime_config=runtime_config,
        geometry=geometry,
        coefficient=coefficients,
        aabb=aabb,
        instance_to_glb=instance_to_glb,
        query_weights=query_weights,
        query_layout=query_layout,
        scene_bounds=scene_bounds,
        runtime_meta_info=runtime_meta_info,
        geometry_info=geometry_info,
        coefficient_info=coefficient_info,
        threshold=threshold,
        threshold_info=threshold_info,
        output_files=output_files,
        training_provenance=training_provenance,
    )
    return output_dir, output_files, meta, threshold


def _write_export(output_dir: Path, output_files: Mapping[str, bytes], meta: Mapping[str, Any]) -> None:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(parent)))
    try:
        for name, raw in output_files.items():
            (temporary / name).write_bytes(raw)
        meta_text = json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        (temporary / "model_meta.json").write_text(meta_text, encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Validate inputs and export one independent runtime directory."""
    output_dir, output_files, meta, threshold = _prepare_export(args)
    meta_text = json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    planned = {
        name: int(len(raw)) for name, raw in output_files.items()
    }
    planned["model_meta.json"] = len(meta_text.encode("utf-8"))
    if bool(args.dry_run):
        return {
            "status": "dry-run",
            "schema": EXPORT_SCHEMA,
            "outputDir": str(output_dir),
            "threshold": float(threshold),
            "files": planned,
            "meta": meta,
        }
    _write_export(output_dir, output_files, meta)
    return {
        "status": "exported",
        "schema": EXPORT_SCHEMA,
        "outputDir": str(output_dir),
        "metadata": str(output_dir / "model_meta.json"),
        "threshold": float(threshold),
        "files": planned,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--geometry",
        "--geometry-features",
        "--initial-geo-features",
        dest="geometry",
        default=None,
        help="Optional FP16 N x 96 geometry table; otherwise use checkpoint metadata or its sibling table.",
    )
    parser.add_argument(
        "--survival-coefficients",
        default=None,
        help="Optional FP16 N x 4 x 7 coefficient table; otherwise use checkpoint.survivalCoefficients.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional assertion of the checkpoint calibration threshold; it cannot override it.",
    )
    parser.add_argument(
        "--allow-unsafe-threshold",
        action="store_true",
        help="Allow a diagnostic export when the recorded calibration workpoint misses the safety gate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the export plan without creating output files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = export(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
