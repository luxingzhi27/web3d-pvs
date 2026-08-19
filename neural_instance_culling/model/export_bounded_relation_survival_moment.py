#!/usr/bin/env python3
"""Export the calibrated relation-prior runtime bundle.

The checkpoint contains both offline training state and the small online query
network.  This exporter has an explicit runtime allow-list.  The bundle gets a
single ``[N, 124]`` FP16 table, scene lookup tables, the online dense weights,
the learned frequency vectors, and the fixed FP32 disk-transfer table.  It
never copies the checkpoint, relation CSR, hierarchy IDs, observations, or
offline encoder state into the output directory.

An output directory is required to be new.  ``--dry-run`` performs the same
validation and budget checks without creating it.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch


MODEL_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-envelope-v4"
CHECKPOINT_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4"
TRAINING_PROTOCOL_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4"
CALIBRATION_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-calibration-v4"
EXPORT_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4"
QUERY_WEIGHTS_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-query-weights-v4"

GEO_DIM = 96
SURVIVAL_RANK = 4
SURVIVAL_PARAMETER_DIM = 7
SURVIVAL_DIM = SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM
RUNTIME_FEATURE_DIM = GEO_DIM + SURVIVAL_DIM
VIEW_DIM = 9
DISK_AXIS_DIM = 2
SPECTRAL_FREQUENCY_COUNT = 16
SPECTRAL_MOMENT_DIM = SPECTRAL_FREQUENCY_COUNT * 4
BOUNDARY_SUMMARY_DIM = 8
LOW_RANK_SUMMARY_DIM = 4
SURVIVAL_SEMANTIC_DIM = 8
RELATION_CONDITION_DIM = 8
RUNTIME_HEAD_INPUT_DIM = (
    GEO_DIM
    + SURVIVAL_RANK
    + SURVIVAL_SEMANTIC_DIM
    + BOUNDARY_SUMMARY_DIM
    + VIEW_DIM
    + LOW_RANK_SUMMARY_DIM
    + 1
)

CHI_TABLE_SIZE = 8192
CHI_TABLE_MAX_ARGUMENT = 320.0
CHI_TABLE_DTYPE = "float32"
CHI_TABLE_BYTES = CHI_TABLE_SIZE * np.dtype("<f4").itemsize
CHI_TABLE_MAX_ABS_ERROR = 1e-4
RAY_SPACE_SCHEMA = "viewcell-ray-space-horizontal-disk-v3"
FEATURE_DOMAIN_ABS_MAX = 1.0
DISK_AXIS_BOUND = "per-feature row norm <= 1 - abs(center feature)"
REGISTERED_MAX_NORM_CYCLES = 8.0
RELATION_SCHEMA_V3 = "pvs-viewcell-train-observed-relation-csr-v3"

TARGET_WEIGHTED_RECALL = 0.99
MINIMUM_WEIGHTED_RECALL_LCB = 0.99
MAX_NEURAL_ASSET_BYTES = 7 * 1024 * 1024
# Prior, applied residual, and fused coefficient tables are each serialized
# as FP16; validate their FP32 reconstruction with the registered bound.
MAX_FP16_FUSION_ABS_ERROR = 0.02
DEFAULT_VIEWCELL_SHAPE = "horizontal_disk"
V4_VARIANT_CONTRACTS: dict[str, tuple[str, str, bool]] = {
    "full": ("safety_reserve", "residual", False),
    "without_bounded_relation": ("safety_reserve", "residual", False),
    "without_viewcell_moment_envelope": ("safety_reserve", "residual", False),
    "without_safety_reserve_utility": ("normalized_rvl", "residual", False),
    "without_instance_calibration_residual": ("safety_reserve", "disabled", False),
    "pose_balanced_frontier_without_exposure": (
        "pose_balanced_frontier",
        "residual",
        False,
    ),
    "pose_balanced_frontier_with_exposure": (
        "pose_balanced_frontier",
        "residual",
        True,
    ),
    "cross_pose_operating_without_exposure": (
        "cross_pose_operating",
        "residual",
        False,
    ),
    "cross_pose_operating_with_exposure": (
        "cross_pose_operating",
        "residual",
        True,
    ),
}

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
        kind = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {kind}")
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


def _read_binary(path: Path, dtype: str, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    values = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape, dtype=np.int64))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} values; expected {expected}")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} contains non-finite values: {path}")
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


def _validate_checkpoint_schema(checkpoint: Mapping[str, Any]) -> None:
    """Reject old models even when they happen to expose similar tensors."""

    top_schema = checkpoint.get("schema")
    if top_schema != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unexpected checkpoint schema: {top_schema!r}; expected {CHECKPOINT_SCHEMA!r}"
        )
    runtime_schema = checkpoint.get("runtimeSchema")
    if runtime_schema != MODEL_SCHEMA:
        raise ValueError(
            f"unexpected checkpoint runtimeSchema: {runtime_schema!r}; expected {MODEL_SCHEMA!r}"
        )
    declared_model_schema = checkpoint.get("modelSchema")
    if declared_model_schema is not None and declared_model_schema != MODEL_SCHEMA:
        raise ValueError(
            f"unexpected checkpoint modelSchema: {declared_model_schema!r}; expected {MODEL_SCHEMA!r}"
        )
    if checkpoint.get("testRead") is not False:
        raise ValueError("refusing to export a checkpoint without testRead=false")
    protocol = _as_mapping(checkpoint.get("protocol"), "checkpoint.protocol")
    if protocol.get("schema") != TRAINING_PROTOCOL_SCHEMA:
        raise ValueError(
            "checkpoint.protocol.schema must be "
            f"{TRAINING_PROTOCOL_SCHEMA!r}"
        )
    if protocol.get("testRead") is not False:
        raise ValueError("checkpoint.protocol must declare testRead=false")
    if protocol.get("candidateUnion") is not False:
        raise ValueError("checkpoint.protocol must declare candidateUnion=false")
    variant = str(protocol.get("variant", ""))
    if variant not in V4_VARIANT_CONTRACTS:
        raise ValueError(f"checkpoint.protocol.variant is not registered: {variant!r}")
    (
        expected_loss,
        expected_calibration_mode,
        expected_exposure_enabled,
    ) = V4_VARIANT_CONTRACTS[variant]
    if protocol.get("lossVariant") != expected_loss:
        raise ValueError("checkpoint.protocol.lossVariant disagrees with the registered variant")
    protocol_calibration = _as_mapping(
        protocol.get("instanceCalibration"), "checkpoint.protocol.instanceCalibration"
    )
    if protocol_calibration.get("mode") != expected_calibration_mode:
        raise ValueError(
            "checkpoint.protocol.instanceCalibration.mode disagrees with the registered variant"
        )
    protocol_exposure_value = protocol.get("viewcellExposureSupervision")
    protocol_exposure = (
        {}
        if protocol_exposure_value is None
        else _as_mapping(
            protocol_exposure_value,
            "checkpoint.protocol.viewcellExposureSupervision",
        )
    )
    if bool(protocol_exposure.get("enabled", False)) != expected_exposure_enabled:
        raise ValueError(
            "checkpoint.protocol.viewcellExposureSupervision.enabled disagrees with "
            "the registered variant"
        )


def _validate_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    _validate_checkpoint_schema(checkpoint)
    config_value = checkpoint.get("config", checkpoint.get("modelConfig"))
    config = _as_mapping(config_value, "checkpoint.config")
    if config.get("runtimeSchema") != MODEL_SCHEMA:
        raise ValueError(
            f"config.runtimeSchema must be {MODEL_SCHEMA!r}, got "
            f"{config.get('runtimeSchema')!r}"
        )

    num_instances = _positive_int(config.get("numInstances"), "config.numInstances")
    num_glbs = _positive_int(config.get("numGlbs"), "config.numGlbs", allow_zero=True)
    if _positive_int(config.get("geometryDim"), "config.geometryDim") != GEO_DIM:
        raise ValueError("config.geometryDim must be 96")
    if list(config.get("survivalCoefficientShape") or ()) != [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM]:
        raise ValueError("config.survivalCoefficientShape must be [4, 7]")
    if _positive_int(config.get("runtimeFeatureDim"), "config.runtimeFeatureDim") != RUNTIME_FEATURE_DIM:
        raise ValueError("config.runtimeFeatureDim must be 124")
    if _positive_int(config.get("runtimeHeadInputDim"), "config.runtimeHeadInputDim") != RUNTIME_HEAD_INPUT_DIM:
        raise ValueError("config.runtimeHeadInputDim must be 130")
    if _positive_int(config.get("boundarySummaryDim"), "config.boundarySummaryDim") != BOUNDARY_SUMMARY_DIM:
        raise ValueError("config.boundarySummaryDim must be 8")
    if _positive_int(config.get("lowRankSummaryDim"), "config.lowRankSummaryDim") != LOW_RANK_SUMMARY_DIM:
        raise ValueError("config.lowRankSummaryDim must be 4")
    hidden_dim = _positive_int(config.get("hiddenDim"), "config.hiddenDim")
    instance_calibration = _as_mapping(
        config.get("instanceCalibration"), "config.instanceCalibration"
    )
    calibration_mode = str(instance_calibration.get("mode", ""))
    if calibration_mode not in {"residual", "disabled"}:
        raise ValueError("config.instanceCalibration.mode must be residual or disabled")
    if list(instance_calibration.get("shape") or ()) != [
        SURVIVAL_RANK,
        SURVIVAL_PARAMETER_DIM,
    ]:
        raise ValueError("config.instanceCalibration.shape must be [4, 7]")
    calibration_max_abs = _finite_float(
        instance_calibration.get("maximumAbsoluteResidual"),
        "config.instanceCalibration.maximumAbsoluteResidual",
    )
    sparse_instance_penalty = _finite_float(
        instance_calibration.get("sparseInstancePenalty"),
        "config.instanceCalibration.sparseInstancePenalty",
    )
    if calibration_max_abs <= 0.0 or sparse_instance_penalty < 0.0:
        raise ValueError("config.instanceCalibration bounds are invalid")
    if instance_calibration.get("runtimeExport") != "fused coefficients only":
        raise ValueError("runtime must export only fused survival coefficients")
    protocol = _as_mapping(checkpoint.get("protocol"), "checkpoint.protocol")
    expected_calibration_mode = V4_VARIANT_CONTRACTS[str(protocol["variant"])][1]
    if calibration_mode != expected_calibration_mode:
        raise ValueError("config.instanceCalibration.mode disagrees with checkpoint variant")
    spectral_mode = str(config.get("spectralMode", ""))
    if spectral_mode not in {"moment_envelope", "point"}:
        raise ValueError("config.spectralMode must be moment_envelope or point")

    depth = _as_mapping(config.get("depthNormalization"), "config.depthNormalization")
    depth_q01 = _finite_float(depth.get("q01"), "config.depthNormalization.q01")
    depth_q99 = _finite_float(depth.get("q99"), "config.depthNormalization.q99")
    depth_epsilon = _finite_float(depth.get("epsilon"), "config.depthNormalization.epsilon")
    if depth_q99 <= depth_q01:
        raise ValueError("config.depthNormalization.q99 must be greater than q01")
    if depth_epsilon <= 0.0:
        raise ValueError("config.depthNormalization.epsilon must be positive")
    if depth.get("sourceSplit") not in (None, "train"):
        raise ValueError("depth normalization must be frozen from the train split")

    frequency = _as_mapping(config.get("frequency"), "config.frequency")
    frequency_count = _positive_int(frequency.get("count"), "config.frequency.count")
    if frequency_count != SPECTRAL_FREQUENCY_COUNT:
        raise ValueError("config.frequency.count must be 16")
    if frequency.get("units") != "cycles":
        raise ValueError("config.frequency.units must be 'cycles'")
    max_frequency_norm = _finite_float(
        frequency.get("maxNormCycles"), "config.frequency.maxNormCycles"
    )
    if max_frequency_norm <= 0.0:
        raise ValueError("config.frequency.maxNormCycles must be positive")
    if max_frequency_norm > REGISTERED_MAX_NORM_CYCLES + 1e-7:
        raise ValueError(
            "config.frequency.maxNormCycles exceeds the horizontal-disk range contract "
            f"of {REGISTERED_MAX_NORM_CYCLES}"
        )

    exposure = config.get("viewcellExposureSupervision")
    exposure_enabled = False
    exposure_hidden_dim = 0
    if exposure is not None:
        exposure = _as_mapping(
            exposure, "config.viewcellExposureSupervision"
        )
        exposure_enabled = bool(exposure.get("enabled", False))
        if exposure_enabled:
            exposure_hidden_dim = _positive_int(
                exposure.get("hiddenDim"),
                "config.viewcellExposureSupervision.hiddenDim",
            )
            if (
                _positive_int(
                    exposure.get("inputDim"),
                    "config.viewcellExposureSupervision.inputDim",
                )
                != hidden_dim
                or exposure.get("trainingOnly") is not True
                or exposure.get("runtimeExport") is not False
            ):
                raise ValueError(
                    "view-cell exposure supervision must be train-only and excluded from runtime"
                )
    expected_exposure_enabled = V4_VARIANT_CONTRACTS[str(protocol["variant"])][2]
    if exposure_enabled != expected_exposure_enabled:
        raise ValueError(
            "config.viewcellExposureSupervision.enabled disagrees with checkpoint variant"
        )

    return {
        "runtimeSchema": MODEL_SCHEMA,
        "numInstances": num_instances,
        "numGlbs": num_glbs,
        "geometryDim": GEO_DIM,
        "survivalCoefficientShape": [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
        "survivalCoefficientDim": SURVIVAL_DIM,
        "runtimeFeatureDim": RUNTIME_FEATURE_DIM,
        "runtimeHeadInputDim": RUNTIME_HEAD_INPUT_DIM,
        "boundarySummaryDim": BOUNDARY_SUMMARY_DIM,
        "lowRankSummaryDim": LOW_RANK_SUMMARY_DIM,
        "hiddenDim": hidden_dim,
        "relationSource": str(config.get("relationSource", "bounded_hierarchical")),
        "spectralMode": spectral_mode,
        "instanceCalibration": {
            "mode": calibration_mode,
            "shape": [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
            "maximumAbsoluteResidual": calibration_max_abs,
            "sparseInstancePenalty": sparse_instance_penalty,
            "runtimeExport": "fused coefficients only",
        },
        "depthNormalization": {
            "definition": str(
                depth.get(
                    "definition",
                    "clip((log1p(distance/(radius+epsilon))-q01)/(q99-q01),0,1)",
                )
            ),
            "q01": depth_q01,
            "q99": depth_q99,
            "epsilon": depth_epsilon,
            "sourceSplit": "train",
        },
        "frequency": {
            "count": frequency_count,
            "units": "cycles",
            "maxNormCycles": max_frequency_norm,
        },
        "trainingOnlyExposureSupervision": {
            "enabledInCheckpoint": exposure_enabled,
            "hiddenDim": exposure_hidden_dim,
            "runtimeExported": False,
        },
    }


def _resolve_input_path(value: str | Path | None, *, base_dir: Path, name: str) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {name}: {candidate}")
    return candidate


def _load_runtime_features(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    num_instances: int,
    instance_calibration_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild the only valid runtime table from checkpoint-owned inputs.

    The training directory also contains a last-epoch 124-value table.  It is
    intentionally ignored because the selected ``best.pt`` may come from an
    earlier epoch.  The checkpoint records the geometry table's source,
    shape, and dtype, while storing its own 28 survival coefficients.  These
    structural fields are the accepted sources for a formal bundle; file
    fingerprints are deliberately not part of the export contract.
    """

    geometry_meta = _as_mapping(checkpoint.get("geometry"), "checkpoint.geometry")
    if list(geometry_meta.get("shape") or ()) != [num_instances, GEO_DIM]:
        raise ValueError(f"checkpoint.geometry.shape must be [{num_instances}, {GEO_DIM}]")
    if geometry_meta.get("dtype") != "float16":
        raise ValueError("checkpoint.geometry.dtype must be 'float16'")
    geometry_path = _resolve_input_path(
        geometry_meta.get("path"),
        base_dir=checkpoint_path.parent,
        name="checkpoint geometry",
    )
    if geometry_path is None:
        raise ValueError("checkpoint.geometry.path is required")
    geometry = _read_binary(
        geometry_path,
        "<f2",
        (num_instances, GEO_DIM),
        "checkpoint geometry features",
    )

    coefficient_value = checkpoint.get("instanceSurvivalCoefficients")
    if coefficient_value is None:
        raise ValueError("checkpoint.instanceSurvivalCoefficients is required")
    coefficients = _to_fp16(
        coefficient_value,
        "checkpoint.instanceSurvivalCoefficients",
        (num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM),
    )
    prior = _to_fp16(
        checkpoint.get("instanceSurvivalPriorCoefficients"),
        "checkpoint.instanceSurvivalPriorCoefficients",
        (num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM),
    )
    residual = _to_fp16(
        checkpoint.get("instanceSurvivalCalibrationResidual"),
        "checkpoint.instanceSurvivalCalibrationResidual",
        (num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM),
    )
    fusion_error = np.abs(
        coefficients.astype(np.float32)
        - prior.astype(np.float32)
        - residual.astype(np.float32)
    )
    max_fusion_error = float(fusion_error.max(initial=0.0))
    if max_fusion_error > MAX_FP16_FUSION_ABS_ERROR:
        raise ValueError(
            "checkpoint fused survival coefficients disagree with prior plus residual"
        )
    calibration = _as_mapping(
        checkpoint.get("instanceCalibration"), "checkpoint.instanceCalibration"
    )
    calibration_mode = str(instance_calibration_mode)
    if calibration.get("mode") != calibration_mode:
        raise ValueError("checkpoint instance calibration mode disagrees with model config")
    if calibration.get("fusion") != "prior_plus_applied_residual":
        raise ValueError("checkpoint instance calibration fusion contract is invalid")
    if calibration.get("runtimeExport") != "fused_coefficients_only":
        raise ValueError("checkpoint attempts to export an unfused calibration table")
    blend = _finite_float(calibration.get("blend"), "checkpoint.instanceCalibration.blend")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("checkpoint instance calibration blend must lie in [0, 1]")
    model_state = _as_mapping(
        checkpoint.get("modelState", checkpoint.get("model")), "checkpoint model state"
    )
    residual_parameter_name = "instance_calibration_residual_raw"
    if calibration_mode == "disabled":
        if max_fusion_error > MAX_FP16_FUSION_ABS_ERROR or float(np.max(np.abs(residual), initial=0.0)) > 1e-7:
            raise ValueError("disabled instance calibration checkpoint has a non-zero residual")
        if float(np.max(np.abs(coefficients.astype(np.float32) - prior.astype(np.float32)), initial=0.0)) > MAX_FP16_FUSION_ABS_ERROR:
            raise ValueError("disabled instance calibration checkpoint does not equal its shared prior")
        if abs(blend) > 1e-7:
            raise ValueError("disabled instance calibration checkpoint has a non-zero blend")
        if residual_parameter_name in model_state:
            raise ValueError("disabled instance calibration checkpoint contains residual parameters")
    elif residual_parameter_name not in model_state:
        raise ValueError("residual instance calibration checkpoint has no residual parameters")
    values = np.ascontiguousarray(
        np.concatenate([geometry, coefficients.reshape(num_instances, SURVIVAL_DIM)], axis=1),
        dtype="<f2",
    )
    return values, {
        "source": "checkpoint.geometryFeatures_plus_survivalCoefficients",
        "geometryDtype": "float16",
        "survivalCoefficientDtype": "float16",
        "geometry": {
            "source": "checkpoint.geometry.path",
            "path": str(geometry_path),
            "shape": [num_instances, GEO_DIM],
            "dtype": "float16",
        },
        "survivalCoefficients": {
            "source": "checkpoint.instanceSurvivalCoefficients",
            "shape": [num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
            "dtype": "float16",
        },
        "offlineCoefficientFusion": {
            "definition": "shared_relation_prior + applied_per_instance_calibration_residual",
            "maximumFp16FusionError": max_fusion_error,
            "runtimeIncludesSeparateResidual": False,
        },
    }


def _linear_specs(prefix: str, dimensions: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    specs: list[tuple[str, tuple[int, ...]]] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
        module_index = index * 2
        specs.extend(
            [
                (f"{prefix}.{module_index}.weight", (output_dim, input_dim)),
                (f"{prefix}.{module_index}.bias", (output_dim,)),
            ]
        )
    return specs


def _runtime_weight_specs(hidden_dim: int) -> list[tuple[str, tuple[int, ...]]]:
    spectral_fuse_dim = SPECTRAL_MOMENT_DIM + VIEW_DIM + LOW_RANK_SUMMARY_DIM + RELATION_CONDITION_DIM
    return (
        _linear_specs("relation_condition_head", (SURVIVAL_DIM, 16, RELATION_CONDITION_DIM))
        + _linear_specs("boundary_summary_head", (spectral_fuse_dim, 48, BOUNDARY_SUMMARY_DIM))
        + _linear_specs(
            "direction_basis_head",
            (BOUNDARY_SUMMARY_DIM + RELATION_CONDITION_DIM + 3, 24, SURVIVAL_RANK),
        )
        + _linear_specs("shared_trunk", (RUNTIME_HEAD_INPUT_DIM, hidden_dim, hidden_dim))
        + [
            ("visibility_head.weight", (1, hidden_dim)),
            ("visibility_head.bias", (1,)),
        ]
        + _linear_specs("utility_head", (hidden_dim + 1, 32, 1))
        + _linear_specs("download_head", (hidden_dim + 1, 32, 1))
    )


def _pack_query_weights(
    state: Mapping[str, Any], hidden_dim: int
) -> tuple[bytes, list[dict[str, Any]]]:
    specs = _runtime_weight_specs(hidden_dim)
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
        layout.append(
            {
                "name": name,
                "offsetElements": int(offset_elements),
                "offsetBytes": int(offset_elements * 2),
                "shape": list(shape),
                "dtype": "float16",
                "elementCount": int(value.size),
                "byteLength": len(raw),
            }
        )
        offset_elements += int(value.size)
    return b"".join(chunks), layout


def _state_table(state: Mapping[str, Any], names: tuple[str, ...], name: str) -> Any:
    for key in names:
        if key in state:
            return state[key]
    raise ValueError(f"checkpoint is missing {name}")


def _pack_frequency_and_chi(
    state: Mapping[str, Any], max_norm_cycles: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_frequency = _state_table(
        state,
        ("moment_query.frequency_cycles", "frequency_cycles"),
        "learned frequency_cycles",
    )
    frequency = _to_fp32(
        raw_frequency,
        "moment_query.frequency_cycles",
        (SPECTRAL_FREQUENCY_COUNT, VIEW_DIM),
    )
    norms = np.linalg.norm(frequency, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm_cycles / np.maximum(norms, 1e-12))
    frequency = np.ascontiguousarray(frequency * scale, dtype="<f4")

    chi = _to_fp32(
        _state_table(state, ("moment_query.chi_table", "chi_table"), "chi_table"),
        "moment_query.chi_table",
        (CHI_TABLE_SIZE,),
    )
    if abs(float(chi[0]) - 1.0) > 1e-5:
        raise ValueError("chi_table[0] must equal the zero-limit value 1")
    return frequency, chi, {
        "frequencyWasNormBounded": bool(np.any(scale < 1.0)),
        "frequencyMaxNormCycles": float(np.linalg.norm(frequency, axis=1).max()),
    }


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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid runtime meta JSON: {path}") from exc
    runtime_meta = _as_mapping(payload, "runtime meta")
    records = runtime_meta.get("componentRecords")
    if not isinstance(records, list) or len(records) != num_instances:
        raise ValueError(f"runtime meta componentRecords must contain exactly {num_instances} records")

    aabbs = np.empty((num_instances, 6), dtype=np.float32)
    mapping = np.empty((num_instances,), dtype=np.uint32)
    seen: set[int] = set()
    for record_index, raw_record in enumerate(records):
        record = _as_mapping(raw_record, f"componentRecords[{record_index}]")
        component_id = _positive_int(
            record.get("componentGlobalId"),
            f"componentRecords[{record_index}].componentGlobalId",
            allow_zero=True,
        )
        if component_id >= num_instances or component_id in seen:
            raise ValueError("runtime componentGlobalId values must be a unique 0..N-1 permutation")
        glb_id = _positive_int(
            record.get("globalGlbId"),
            f"componentRecords[{record_index}].globalGlbId",
            allow_zero=True,
        )
        if glb_id >= 2**32 or (num_glbs and glb_id >= num_glbs):
            raise ValueError("runtime globalGlbId is outside config.numGlbs or uint32")
        minimum, maximum = _bounds_min_max(
            _as_mapping(record.get("bounds"), f"componentRecords[{record_index}].bounds"),
            f"componentRecords[{record_index}].bounds",
        )
        seen.add(component_id)
        aabbs[component_id, :3] = minimum
        aabbs[component_id, 3:] = maximum
        mapping[component_id] = glb_id
    if seen != set(range(num_instances)):
        raise ValueError("runtime componentGlobalId values must cover every instance")
    if num_glbs == 0 and num_instances:
        raise ValueError("config.numGlbs is zero but runtime meta contains instances")
    if num_glbs and int(mapping.max()) != num_glbs - 1:
        raise ValueError("runtime globalGlbId range does not cover config.numGlbs")
    scene_min, scene_max = _bounds_min_max(
        _as_mapping(runtime_meta.get("sceneBounds"), "runtime meta.sceneBounds"),
        "runtime meta.sceneBounds",
    )
    scene_bounds = {"min": scene_min.tolist(), "max": scene_max.tolist()}
    return (
        np.ascontiguousarray(aabbs.astype("<f4", copy=False)),
        np.ascontiguousarray(mapping.astype("<u4", copy=False)),
        scene_bounds,
        {
            "path": str(path),
            "schema": runtime_meta.get("schema"),
            "schemaVersion": runtime_meta.get("schemaVersion"),
            "componentRecordCount": len(records),
            "numInstances": int(num_instances),
            "numGlbs": int(num_glbs),
        },
    )


def _find_nested_value(containers: list[Mapping[str, Any]], keys: tuple[str, ...]) -> Any:
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value is not None:
                return value
    return None


def _training_provenance(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate checkpoint provenance with explainable structural fields only.

    Old checkpoints may still carry digest keys.  They are intentionally
    ignored here.  Reuse safety is established by the explicit paths, schema,
    split counts, relation counts, and the binary shape checks performed by
    the exporter and evaluator.
    """

    protocol = _as_mapping(checkpoint.get("protocol"), "checkpoint.protocol")
    relation = _as_mapping(checkpoint.get("relation"), "checkpoint.relation")
    if relation.get("schema") != RELATION_SCHEMA_V3:
        raise ValueError(
            "checkpoint relation provenance must use the corrected v3 relation schema"
        )
    relation_path_value = relation.get("path")
    if not relation_path_value:
        raise ValueError("checkpoint relation provenance must include relation.path")
    relation_path = Path(str(relation_path_value)).expanduser().resolve()
    if not relation_path.is_dir():
        raise FileNotFoundError(f"missing checkpoint relation directory: {relation_path}")
    relation_meta_path = relation_path / "relation_csr_meta.json"
    if not relation_meta_path.is_file():
        raise FileNotFoundError(f"missing checkpoint relation metadata: {relation_meta_path}")
    try:
        relation_meta = json.loads(relation_meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid relation metadata: {relation_meta_path}") from exc
    relation_meta = _as_mapping(relation_meta, "relation metadata")
    if relation_meta.get("schema") != RELATION_SCHEMA_V3:
        raise ValueError("checkpoint relation metadata schema is not v3")

    def _declared_count(container: Mapping[str, Any], *names: str) -> int | None:
        for name in names:
            if container.get(name) is not None:
                return _positive_int(container[name], f"relation.{name}", allow_zero=True)
        return None

    relation_stats = relation_meta.get("stats")
    if not isinstance(relation_stats, Mapping):
        relation_stats = relation_meta
    relation_counts = {
        "edgeCount": _declared_count(relation, "edgeCount") or _declared_count(relation_stats, "edgeCount"),
        "rowCount": _declared_count(relation, "rowCount") or _declared_count(relation_stats, "rowCount"),
        "observationCount": _declared_count(relation, "observationCount") or _declared_count(relation_stats, "survivalObservationCount", "observationCount"),
    }
    for key, value in relation_counts.items():
        if value is None or value <= 0:
            raise ValueError(f"checkpoint relation provenance is missing positive {key}")
    for key, value in relation_counts.items():
        metadata_value = _declared_count(relation_stats, key, "survivalObservationCount" if key == "observationCount" else key)
        if metadata_value is not None and metadata_value != value:
            raise ValueError(f"checkpoint relation {key} disagrees with relation metadata")

    split_counts = protocol.get("splitPoseCounts")
    if not isinstance(split_counts, Mapping):
        raise ValueError("checkpoint protocol.splitPoseCounts is required")
    expected_split_counts = {key: _positive_int(value, f"protocol.splitPoseCounts.{key}") for key, value in split_counts.items()}
    if expected_split_counts.get("train") != 2772 or expected_split_counts.get("calibration") != 168 or expected_split_counts.get("validation") != 213:
        raise ValueError("checkpoint protocol split pose counts do not match v4 contract")
    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping) or not dataset.get("path"):
        raise ValueError("checkpoint protocol.dataset.path is required")
    dataset_path = Path(str(dataset["path"])).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"missing checkpoint dataset directory: {dataset_path}")
    runtime_meta = protocol.get("runtimeMeta")
    runtime_meta_path = None
    if isinstance(runtime_meta, Mapping) and runtime_meta.get("path"):
        runtime_meta_path = Path(str(runtime_meta["path"])).expanduser().resolve()
        if not runtime_meta_path.is_file():
            raise FileNotFoundError(f"missing checkpoint runtime metadata: {runtime_meta_path}")
    return {
        "dataset": {
            "path": str(dataset_path),
            "splitPoseCounts": expected_split_counts,
            "testRead": False,
        },
        "relation": {
            "path": str(relation_path),
            "schema": RELATION_SCHEMA_V3,
            "edgeCount": int(relation_counts["edgeCount"]),
            "rowCount": int(relation_counts["rowCount"]),
            "observationCount": int(relation_counts["observationCount"]),
            "testRead": False,
        },
        "runtimeMeta": {
            "path": str(runtime_meta_path) if runtime_meta_path is not None else None,
            "testRead": False,
        },
        "candidateSemantics": "native split candidates recorded in the checkpoint protocol; replay validates IDs structurally",
        "testRead": False,
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
        "precision",
        "balancedAccuracy",
        "specificity",
        "usefulCull",
        "badCull",
        "avgPredCount",
        "avgCandidateCount",
    )
    result: dict[str, Any] = {}
    for key in keys:
        if key in row and row[key] is not None:
            result[key] = _finite_float(row[key], f"calibration.{key}")
    return result


def _resolve_threshold(
    checkpoint: Mapping[str, Any], requested: float | None = None, allow_unsafe: bool = False
) -> tuple[float, dict[str, Any]]:
    calibration = _as_mapping(checkpoint.get("calibration"), "checkpoint.calibration")
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(
            f"checkpoint.calibration.schema must be {CALIBRATION_SCHEMA!r}"
        )
    if calibration.get("testRead") is not False:
        raise ValueError("checkpoint.calibration must declare testRead=false")
    best = checkpoint.get("best")
    selected = calibration.get("selected", calibration.get("selectedSafe"))
    source = (
        "checkpoint.calibration.selected"
        if calibration.get("selected") is not None
        else "checkpoint.calibration.selectedSafe"
    )
    if (
        allow_unsafe
        and isinstance(best, Mapping)
        and best.get("safe") is False
        and isinstance(calibration.get("diagnostic"), Mapping)
    ):
        selected = calibration["diagnostic"]
        source = "checkpoint.calibration.diagnostic"
    if not isinstance(selected, Mapping):
        if not allow_unsafe:
            raise ValueError("checkpoint.calibration.selected/selectedSafe is missing")
        selected = calibration.get("diagnostic")
        source = "checkpoint.calibration.diagnostic"
    if not isinstance(selected, Mapping):
        raise ValueError("checkpoint has no calibration workpoint containing a threshold")

    threshold = _threshold_value(selected, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"calibration threshold must be in [0, 1], got {threshold}")
    weighted_recall = _threshold_value(selected, "aggregateWeightedRecall")
    lower_bound = _threshold_value(
        selected, "aggregateWeightedRecallLowerConfidenceBound"
    )
    calibration_safe = (
        weighted_recall > TARGET_WEIGHTED_RECALL
        and lower_bound > MINIMUM_WEIGHTED_RECALL_LCB
    )
    checkpoint_safe = not isinstance(best, Mapping) or best.get("safe") is not False
    safe = calibration_safe and checkpoint_safe
    if not safe and not allow_unsafe:
        raise ValueError(
            "checkpoint calibration threshold is unsafe: "
            f"weighted recall={weighted_recall:.8f}, lower bound={lower_bound:.8f}"
        )
    if requested is not None:
        requested_value = _finite_float(requested, "--threshold")
        if not 0.0 <= requested_value <= 1.0:
            raise ValueError("--threshold must be in [0, 1]")
        if abs(requested_value - threshold) > 1e-7:
            raise ValueError(
                "--threshold must equal the checkpoint calibration threshold; "
                "threshold overrides are not allowed"
            )
    if calibration.get("selectedFromTest") is True or calibration.get("testEvaluationCount", 0) not in (0, None):
        raise ValueError("calibration threshold must be selected before test evaluation")
    declared_safe = selected.get("safe")
    if declared_safe is not None and bool(declared_safe) != safe:
        raise ValueError("calibration selected.safe disagrees with weighted-recall safety fields")
    if isinstance(best, Mapping):
        if best.get("threshold") is not None:
            best_threshold = _finite_float(best["threshold"], "checkpoint.best.threshold")
            if abs(best_threshold - threshold) > 1e-7:
                raise ValueError("checkpoint.best.threshold disagrees with calibration threshold")
        if best.get("safe") is not None and bool(best["safe"]) != safe:
            raise ValueError("checkpoint.best.safe disagrees with calibration safety")
    return threshold, {
        "source": source,
        "protocol": "calibration_ready_pre_test" if safe else "diagnostic_calibration",
        "thresholdSpace": "visibility_probability",
        "targetWeightedRecall": TARGET_WEIGHTED_RECALL,
        "minimumWeightedRecallLowerConfidenceBound": MINIMUM_WEIGHTED_RECALL_LCB,
        "weightedRecallField": "aggregateWeightedRecall",
        "weightedRecallLowerConfidenceBoundField": "aggregateWeightedRecallLowerConfidenceBound",
        "calibrationSafe": calibration_safe,
        "safe": safe,
        "status": "safe" if safe else "unsafe_diagnostic",
        "selected": _compact_workpoint(selected),
        "testEvaluationCount": 0,
    }


def _viewcell_contract(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    candidates: list[Mapping[str, Any]] = []
    protocol = checkpoint.get("protocol")
    for value in (
        checkpoint.get("viewcell"),
        checkpoint.get("viewCell"),
        checkpoint.get("queryContract"),
        config.get("viewcell") if isinstance(config, Mapping) else None,
        config.get("viewCell") if isinstance(config, Mapping) else None,
        protocol.get("viewcell") if isinstance(protocol, Mapping) else None,
        protocol.get("viewCell") if isinstance(protocol, Mapping) else None,
        checkpoint.get("datasetMeta"),
        protocol,
    ):
        if isinstance(value, Mapping):
            candidates.append(value)
    shape = _find_nested_value(candidates, ("viewcellShape", "viewCellShape", "shape"))
    if shape is not None and str(shape) != DEFAULT_VIEWCELL_SHAPE:
        raise ValueError("v4 export requires viewcell shape horizontal_disk")
    radius_value = _find_nested_value(
        candidates, ("viewcellRadiusM", "viewCellRadiusM", "radiusM", "radius")
    )
    if radius_value is None:
        raise ValueError("v4 export requires an explicit checkpoint view-cell radius")
    radius_source = "checkpoint.viewcell_contract"
    radius = _finite_float(radius_value, "viewcell radius")
    if radius <= 0.0:
        raise ValueError("viewcell radius must be positive")
    candidate_semantics = _find_nested_value(candidates, ("candidateCameraSemantics",))
    query_semantics = _find_nested_value(candidates, ("queryCenterSemantics",))
    expected_candidate = "66-degree back-camera candidate identity only"
    expected_query = "center of the same-direction view-cell visibility union"
    if candidate_semantics is not None and str(candidate_semantics) != expected_candidate:
        raise ValueError("checkpoint candidateCameraSemantics does not match v3")
    if query_semantics is not None and str(query_semantics) != expected_query:
        raise ValueError("checkpoint queryCenterSemantics does not match v3")
    return {
        "shape": DEFAULT_VIEWCELL_SHAPE,
        "radiusM": radius,
        "radiusSource": radius_source,
        "candidateCameraSemantics": expected_candidate,
        "queryCenterSemantics": expected_query,
        "candidateCameraFormula": "candidate_camera_world = query_center_world - forward * back_offset",
        "queryCenterFormula": "query_center_world = view-cell center",
        "horizontalPlane": "world XZ; no world Y displacement",
    }


def _ray_space_contract(max_norm_cycles: float) -> dict[str, Any]:
    """Describe the bounded Jacobian contract consumed by the frontend."""

    max_norm = float(max_norm_cycles)
    max_two_s = 4.0 * math.pi * math.sqrt(VIEW_DIM) * max_norm
    return {
        "schema": RAY_SPACE_SCHEMA,
        "centerDim": VIEW_DIM,
        "viewcellShape": DEFAULT_VIEWCELL_SHAPE,
        "diskAxisShape": [VIEW_DIM, DISK_AXIS_DIM],
        "diskAxisBound": DISK_AXIS_BOUND,
        "featureDomain": [-FEATURE_DOMAIN_ABS_MAX, FEATURE_DOMAIN_ABS_MAX],
        "rangeGuarantee": {
            "enforcedBy": "build_horizontal_disk_ray_query",
            "diskCoordinate": "uniform unit disk; ||u||_2 <= 1",
            "rowFormula": "||B[i,:]||_2 <= 1 - abs(centerFeature[i])",
            "featureFormula": "abs(centerFeature[i] + dot(B[i,:], u)) <= 1",
            "frequencyNorm": "||f||_2 <= maxNormCycles in cycles",
            "maxNormCycles": max_norm,
            "sUpperBoundFormula": "2*pi*sqrt(9)*maxNormCycles",
            "twoSUpperBoundFormula": "4*pi*sqrt(9)*maxNormCycles",
            "maxTwoS": max_two_s,
            "chiLookupMaxArgument": CHI_TABLE_MAX_ARGUMENT,
            "twoSStrictlyInsideChiRange": bool(max_two_s < CHI_TABLE_MAX_ARGUMENT),
        },
        "candidateCameraRole": "candidate identity only",
        "queryCenterRole": "center of same-direction view-cell visibility union",
        "depth": "train-frozen radius-relative log1p q01/q99 normalization with exported epsilon",
        "onlineSubposeExpansion": False,
    }


def _file_descriptor(name: str, dtype: str, shape: list[Any], raw: bytes) -> dict[str, Any]:
    return {
        "file": name,
        "dtype": dtype,
        "shape": shape,
        "byteLength": len(raw),
    }


def _check_neural_asset_budget(asset_bytes: Mapping[str, bytes]) -> int:
    neural_names = {
        "instance_runtime_features_fp16.bin",
        "query_weights_fp16.bin",
        "frequency_cycles_fp32.bin",
        "chi_table_fp32.bin",
    }
    unknown = set(asset_bytes) - neural_names - {
        "instance_aabb_fp32.bin",
        "instance_to_glb_uint32.bin",
    }
    if unknown:
        raise ValueError(f"unclassified export assets: {sorted(unknown)}")
    used = sum(len(asset_bytes[name]) for name in neural_names if name in asset_bytes)
    if used > MAX_NEURAL_ASSET_BYTES:
        raise ValueError(
            f"neural asset budget exceeded: {used} bytes > {MAX_NEURAL_ASSET_BYTES} bytes"
        )
    return int(used)


def _build_model_meta(
    *,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    viewcell: Mapping[str, Any],
    runtime_features: np.ndarray,
    aabb: np.ndarray,
    instance_to_glb: np.ndarray,
    scene_bounds: Mapping[str, Any],
    runtime_meta_info: Mapping[str, Any],
    query_weights: bytes,
    query_layout: list[dict[str, Any]],
    frequency: np.ndarray,
    chi_table: np.ndarray,
    threshold: float,
    threshold_info: Mapping[str, Any],
    provenance: Mapping[str, Any],
    output_files: Mapping[str, bytes],
    frequency_info: Mapping[str, Any],
) -> dict[str, Any]:
    neural_bytes = _check_neural_asset_budget(output_files)
    runtime_raw = output_files["instance_runtime_features_fp16.bin"]
    aabb_raw = output_files["instance_aabb_fp32.bin"]
    mapping_raw = output_files["instance_to_glb_uint32.bin"]
    frequency_raw = output_files["frequency_cycles_fp32.bin"]
    chi_raw = output_files["chi_table_fp32.bin"]
    ray_space = _ray_space_contract(float(runtime_config["frequency"]["maxNormCycles"]))
    if not ray_space["rangeGuarantee"]["twoSStrictlyInsideChiRange"]:
        raise ValueError("v3 ray-space range guarantee does not fit the chi lookup range")
    return {
        "schema": EXPORT_SCHEMA,
        "testRead": False,
        "checkpointSchema": checkpoint.get("schema", MODEL_SCHEMA),
        "modelSchema": MODEL_SCHEMA,
        "numInstances": int(runtime_config["numInstances"]),
        "numGlbs": int(runtime_config["numGlbs"]),
        "modelConfig": dict(runtime_config),
        "fixedTable": {
            "file": "instance_runtime_features_fp16.bin",
            "shape": [int(runtime_config["numInstances"]), RUNTIME_FEATURE_DIM],
            "dtype": "float16",
            "layout": [
                {"name": "geometry", "offset": 0, "dim": GEO_DIM},
                {
                    "name": "survivalCoefficients",
                    "offset": GEO_DIM,
                    "dim": SURVIVAL_DIM,
                    "shape": ["N", SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
                },
            ],
            "byteLength": len(runtime_raw),
        },
        "candidateCameraSemantics": viewcell["candidateCameraSemantics"],
        "queryCenterSemantics": viewcell["queryCenterSemantics"],
        "viewcellShape": DEFAULT_VIEWCELL_SHAPE,
        "viewcell": dict(viewcell),
        "raySpace": ray_space,
        "diskAxisBound": ray_space["diskAxisBound"],
        "featureDomain": list(ray_space["featureDomain"]),
        "rangeGuarantee": dict(ray_space["rangeGuarantee"]),
        "query": {
            "candidateCameraSemantics": viewcell["candidateCameraSemantics"],
            "queryCenterSemantics": viewcell["queryCenterSemantics"],
            "viewcellShape": DEFAULT_VIEWCELL_SHAPE,
            "viewcellRadiusM": float(viewcell["radiusM"]),
            "raySpaceDim": VIEW_DIM,
            "diskAxisShape": [VIEW_DIM, DISK_AXIS_DIM],
            "diskAxisBound": ray_space["diskAxisBound"],
            "featureDomain": list(ray_space["featureDomain"]),
            "rangeGuarantee": dict(ray_space["rangeGuarantee"]),
            "raySpace": ray_space,
            "oneCenterQueryPerCandidateBatch": True,
            "subposeExpansion": False,
        },
        "frequency": {
            "file": "frequency_cycles_fp32.bin",
            "shape": [SPECTRAL_FREQUENCY_COUNT, VIEW_DIM],
            "dtype": "float32",
            "units": "cycles",
            "phase": "phi = 2*pi*f^T*mu",
            "phaseFactor": "2*pi",
            "twoPi": 2.0 * math.pi,
            "radialArgument": "s = 2*pi*norm(B^T*f)",
            "secondMomentRadialArgument": "2*s",
            "maxNormCycles": float(runtime_config["frequency"]["maxNormCycles"]),
            "boundedAtExport": bool(frequency_info["frequencyWasNormBounded"]),
            "byteLength": len(frequency_raw),
        },
        "chiLookup": {
            "function": "chi(s) = 2*J1(s)/s with chi(0)=1",
            "range": [0.0, CHI_TABLE_MAX_ARGUMENT],
            "interpolation": "piecewise_linear",
            "rangePolicy": "strict_error_for_s_and_2s_outside_range",
            "table": {
                "file": "chi_table_fp32.bin",
                "dtype": CHI_TABLE_DTYPE,
                "count": CHI_TABLE_SIZE,
                "bytes": len(chi_raw),
                "maxAbsoluteFunctionError": CHI_TABLE_MAX_ABS_ERROR,
            },
            "bytes": len(chi_raw),
            "sAnd2s": True,
        },
        "depthNormalization": dict(runtime_config["depthNormalization"]),
        "depth": {
            "q01": float(runtime_config["depthNormalization"]["q01"]),
            "q99": float(runtime_config["depthNormalization"]["q99"]),
            "epsilon": float(runtime_config["depthNormalization"]["epsilon"]),
            "formula": runtime_config["depthNormalization"]["definition"],
        },
        "networkWeights": {
            "schema": QUERY_WEIGHTS_SCHEMA,
            "file": "query_weights_fp16.bin",
            "dtype": "float16",
            "layout": query_layout,
            "byteLength": len(query_weights),
        },
        "runtimeTables": {
            "instanceAabb": _file_descriptor(
                "instance_aabb_fp32.bin", "float32", ["N", 6], aabb_raw
            ),
            "instanceToGlb": _file_descriptor(
                "instance_to_glb_uint32.bin", "uint32", ["N"], mapping_raw
            ),
        },
        "files": {
            "runtimeFeatures": _file_descriptor(
                "instance_runtime_features_fp16.bin",
                "float16",
                ["N", RUNTIME_FEATURE_DIM],
                runtime_raw,
            ),
            "queryWeights": _file_descriptor(
                "query_weights_fp16.bin", "float16", ["variable"], query_weights
            ),
            "frequency": _file_descriptor(
                "frequency_cycles_fp32.bin",
                "float32",
                [SPECTRAL_FREQUENCY_COUNT, VIEW_DIM],
                frequency_raw,
            ),
            "chiTable": _file_descriptor(
                "chi_table_fp32.bin", "float32", [CHI_TABLE_SIZE], chi_raw
            ),
            "instanceAabb": _file_descriptor(
                "instance_aabb_fp32.bin", "float32", ["N", 6], aabb_raw
            ),
            "instanceToGlb": _file_descriptor(
                "instance_to_glb_uint32.bin", "uint32", ["N"], mapping_raw
            ),
            "modelMeta": {"file": "model_meta.json"},
        },
        "sceneBounds": dict(scene_bounds),
        "instanceToGlobalGlb": instance_to_glb.astype(np.uint32, copy=False).tolist(),
        "threshold": float(threshold),
        "thresholdSpace": "visibility_probability",
        "calibrationFrozenThreshold": float(threshold),
        "calibration": dict(threshold_info),
        "safety": {
            "status": "safe" if bool(threshold_info["safe"]) else "unsafe_diagnostic",
            "safe": bool(threshold_info["safe"]),
            "source": threshold_info["source"],
        },
        "provenance": {
            "dataset": dict(provenance["dataset"]),
            "relation": dict(provenance["relation"]),
            "runtimeMeta": dict(provenance["runtimeMeta"]),
            "candidateSemantics": provenance["candidateSemantics"],
            "testRead": False,
        },
        "neuralAssetBudget": {
            "usedBytes": neural_bytes,
            "limitBytes": MAX_NEURAL_ASSET_BYTES,
            "limitMiB": 7,
            "withinLimit": neural_bytes <= MAX_NEURAL_ASSET_BYTES,
            "countedFiles": [
                "instance_runtime_features_fp16.bin",
                "query_weights_fp16.bin",
                "frequency_cycles_fp32.bin",
                "chi_table_fp32.bin",
            ],
        },
        "runtimeSemantics": "fixed 124D offline table; one horizontal-disk view-cell query per candidate batch",
        "trainingCheckpointIncluded": False,
        "relationGraphDataIncluded": False,
        "offlineResourcesExcluded": [
            "relation CSR",
            "group IDs",
            "event/censor observations",
            "depth-layer evidence",
            "offline relation encoder",
            "per-instance calibration residual parameters",
            "per-instance calibration reliability",
        ],
        "onlineOperators": {
            "featureLookup": "FP16 [N,124] storage-buffer lookup",
            "frequencyQuery": "16 cycles frequencies with 2*pi phase and radial arguments",
            "chiLookup": "FP32 piecewise-linear table; no Bessel evaluation at runtime",
            "onlineRelationPropagation": False,
            "onlineNeighborSearch": False,
            "onlineSubposeExpansion": False,
            "onlinePointEncoder": False,
            "onlineAabbCornerProjection": False,
        },
    }


def _prepare_export(args: argparse.Namespace) -> tuple[Path, dict[str, bytes], dict[str, Any], float]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    runtime_meta_path = Path(args.runtime_meta).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir in {checkpoint_path.parent, runtime_meta_path.parent}:
        raise ValueError("output-dir must be independent of checkpoint and runtime-meta directories")
    if output_dir.exists():
        raise FileExistsError(f"refusing to use existing output directory: {output_dir}")

    checkpoint = _load_checkpoint(checkpoint_path)
    runtime_config = _validate_model_config(checkpoint)
    viewcell = _viewcell_contract(
        checkpoint, checkpoint.get("config", checkpoint.get("modelConfig", {}))
    )
    provenance = _training_provenance(checkpoint)
    threshold, threshold_info = _resolve_threshold(
        checkpoint, args.threshold, bool(args.allow_unsafe_threshold)
    )
    num_instances = int(runtime_config["numInstances"])
    num_glbs = int(runtime_config["numGlbs"])
    runtime_features, runtime_feature_info = _load_runtime_features(
        checkpoint,
        checkpoint_path,
        num_instances,
        str(runtime_config["instanceCalibration"]["mode"]),
    )
    aabb, instance_to_glb, scene_bounds, runtime_meta_info = _load_runtime_tables(
        runtime_meta_path, num_instances, num_glbs
    )
    state = _as_mapping(
        checkpoint.get("model", checkpoint.get("modelState")), "checkpoint.model/modelState"
    )
    query_weights, query_layout = _pack_query_weights(state, int(runtime_config["hiddenDim"]))
    frequency, chi_table, frequency_info = _pack_frequency_and_chi(
        state, float(runtime_config["frequency"]["maxNormCycles"])
    )
    output_files: dict[str, bytes] = {
        "instance_runtime_features_fp16.bin": runtime_features.tobytes(order="C"),
        "instance_aabb_fp32.bin": aabb.tobytes(order="C"),
        "instance_to_glb_uint32.bin": instance_to_glb.tobytes(order="C"),
        "query_weights_fp16.bin": query_weights,
        "frequency_cycles_fp32.bin": frequency.tobytes(order="C"),
        "chi_table_fp32.bin": chi_table.tobytes(order="C"),
    }
    _check_neural_asset_budget(output_files)
    meta = _build_model_meta(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        runtime_config=runtime_config,
        viewcell=viewcell,
        runtime_features=runtime_features,
        aabb=aabb,
        instance_to_glb=instance_to_glb,
        scene_bounds=scene_bounds,
        runtime_meta_info=runtime_meta_info,
        query_weights=query_weights,
        query_layout=query_layout,
        frequency=frequency,
        chi_table=chi_table,
        threshold=threshold,
        threshold_info=threshold_info,
        provenance=provenance,
        output_files=output_files,
        frequency_info=frequency_info,
    )
    meta["runtimeFeatureSource"] = runtime_feature_info
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
    """Validate inputs and export one independent v4 runtime directory."""

    output_dir, output_files, meta, threshold = _prepare_export(args)
    meta_text = json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    planned = {name: len(raw) for name, raw in output_files.items()}
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
        "--threshold",
        type=float,
        default=None,
        help="Optional assertion of the checkpoint calibration threshold; it cannot override it.",
    )
    parser.add_argument(
        "--allow-unsafe-threshold",
        action="store_true",
        help="Allow a diagnostic bundle when the recorded calibration workpoint misses the safety gate.",
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
