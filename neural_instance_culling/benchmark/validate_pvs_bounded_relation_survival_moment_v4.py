#!/usr/bin/env python3
"""Validate calibrated relation-prior training artifacts.

This validator owns only the v4 checkpoint contract.  It checks the
checkpoint, its model metadata, and the checkpoint-owned calibration summary;
it never accepts the historical integrated-relation schema or discovers a
threshold from another split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CHECKPOINT_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4"
RUNTIME_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-envelope-v4"
TRAINING_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4"
MODEL_SCHEMA = RUNTIME_SCHEMA
CALIBRATION_SUMMARY_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4"
CALIBRATION_ROW_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-calibration-v4"
RELATION_SCHEMA = "pvs-viewcell-train-observed-relation-csr-v3"
CALIBRATION_FLOOR = 0.99
RUNTIME_FEATURE_DIM = 124
GEOMETRY_DIM = 96
SURVIVAL_COEFFICIENT_SHAPE = (4, 7)
SPECTRAL_FREQUENCY_COUNT = 16
MAX_FREQUENCY_NORM_CYCLES = 8.0
EXPECTED_SPLIT_POSE_COUNTS = {"train": 2772, "calibration": 168, "validation": 213}
# Checkpoint survival tensors are stored independently as FP16.  The fused
# equation is exact before export; this bound covers their separate rounding.
MAX_FP16_FUSION_ABS_ERROR = 0.02

# These are the only variant names emitted by the v4 runner.  Keeping the
# registry here makes a v4 checkpoint self-describing without importing the
# orchestration script.
V4_VARIANT_SPECS: dict[str, dict[str, str]] = {
    "full": {
        "relationSource": "bounded_hierarchical",
        "spectralMode": "moment_envelope",
        "lossVariant": "safety_reserve",
        "relationArtifact": "native",
    },
    "without_bounded_relation": {
        "relationSource": "geometry_only",
        "spectralMode": "moment_envelope",
        "lossVariant": "safety_reserve",
        "relationArtifact": "native",
    },
    "without_viewcell_moment_envelope": {
        "relationSource": "bounded_hierarchical",
        "spectralMode": "point",
        "lossVariant": "safety_reserve",
        "relationArtifact": "native",
    },
    "without_safety_reserve_utility": {
        "relationSource": "bounded_hierarchical",
        "spectralMode": "moment_envelope",
        "lossVariant": "normalized_rvl",
        "relationArtifact": "native",
    },
    "without_instance_calibration_residual": {
        "relationSource": "bounded_hierarchical",
        "spectralMode": "moment_envelope",
        "lossVariant": "safety_reserve",
        "relationArtifact": "native",
    },
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _int(value: Any, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        _fail(f"{label} must be an integer")


def _tensor_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            shape = np.asarray(value).shape
        except Exception as exc:  # pragma: no cover - defensive error text
            _fail(f"instanceSurvivalCoefficients has no readable shape: {exc}")
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        _fail("instanceSurvivalCoefficients has an invalid shape")


def _finite_array(value: Any) -> bool:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return bool(np.isfinite(np.asarray(value)).all())
    except Exception:
        return False


def _validate_variant(
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    expected_variant: str | None,
) -> tuple[str, dict[str, str]]:
    variant = str(protocol.get("variant", ""))
    _require(variant in V4_VARIANT_SPECS, f"unknown v4 variant: {variant!r}")
    if expected_variant is not None:
        _require(str(expected_variant) in V4_VARIANT_SPECS, f"unknown expected v4 variant: {expected_variant!r}")
        _require(variant == str(expected_variant), f"v4 checkpoint variant mismatch: expected {expected_variant}, got {variant}")
    spec = V4_VARIANT_SPECS[variant]
    _require(config.get("relationSource") == spec["relationSource"], "v4 model relation source disagrees with variant")
    _require(config.get("spectralMode") == spec["spectralMode"], "v4 model spectral mode disagrees with variant")
    _require(protocol.get("lossVariant") == spec["lossVariant"], "v4 loss variant disagrees with registered variant")
    calibration = config.get("instanceCalibration")
    _require(isinstance(calibration, Mapping), "instance calibration config is missing")
    expected_mode = "disabled" if variant == "without_instance_calibration_residual" else "residual"
    _require(
        calibration.get("mode") == expected_mode,
        "instance calibration mode disagrees with variant",
    )
    protocol_calibration = protocol.get("instanceCalibration")
    _require(isinstance(protocol_calibration, Mapping), "protocol instance calibration contract is missing")
    _require(
        protocol_calibration.get("mode") == expected_mode,
        "protocol instance calibration mode disagrees with variant",
    )
    return variant, dict(spec)


def _selection_from_calibration(
    summary: Mapping[str, Any],
    *,
    allow_unsafe: bool,
) -> tuple[dict[str, Any], bool]:
    _require(summary.get("schema") == CALIBRATION_SUMMARY_SCHEMA, "unexpected v4 calibration summary schema")
    _require(summary.get("testRead") is False, "v4 calibration summary claims test was read")
    status = str(summary.get("status", ""))
    if status == "safe":
        best = summary.get("bestSafe")
        safe = True
    elif status == "no_qualified_safety_workpoint":
        _require(allow_unsafe, "v4 calibration has no safe workpoint")
        best = summary.get("bestDiagnostic")
        safe = False
    else:
        raise ValueError(f"unknown v4 calibration status: {status!r}")

    _require(isinstance(best, Mapping), "v4 calibration summary has no selected workpoint")
    selection = best.get("selection", best)
    _require(isinstance(selection, Mapping), "v4 calibration workpoint has no selection")
    _require(
        _int(selection.get("eval_pose_count"), label="v4 calibration evaluation pose count")
        == EXPECTED_SPLIT_POSE_COUNTS["calibration"],
        "v4 calibration workpoint does not cover the full calibration split",
    )
    _require(_finite(selection.get("threshold")) and 0.0 <= float(selection["threshold"]) <= 1.0, "v4 frozen threshold is invalid")
    if safe:
        _require(best.get("safe") is True, "v4 safe calibration workpoint is not marked safe")

    calibration = summary.get("calibration")
    _require(isinstance(calibration, Mapping), "v4 calibration summary has no calibration payload")
    _require(calibration.get("schema") == CALIBRATION_ROW_SCHEMA, "v4 calibration payload schema is invalid")
    _require(calibration.get("testRead") is False, "v4 calibration payload claims test was read")
    rows = calibration.get("thresholdRows")
    _require(isinstance(rows, list) and rows, "v4 calibration payload has no threshold rows")
    thresholds: list[float] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"v4 calibration threshold row {index} is not an object")
        _require(_finite(row.get("threshold")) and 0.0 <= float(row["threshold"]) <= 1.0, f"v4 calibration threshold row {index} has an invalid threshold")
        thresholds.append(round(float(row["threshold"]), 7))
    _require(round(float(selection["threshold"]), 7) in thresholds, "v4 selected threshold is absent from calibration rows")
    if safe:
        _require(float(selection.get("aggregateWeightedRecall", -1.0)) > CALIBRATION_FLOOR, "v4 safe weighted recall fails the safety gate")
        _require(float(selection.get("aggregateWeightedRecallLowerConfidenceBound", -1.0)) > CALIBRATION_FLOOR, "v4 safe weighted-recall lower bound fails the safety gate")
    return dict(selection), safe


def validate_v4_training_artifact(
    checkpoint: Mapping[str, Any],
    *,
    model_meta: Mapping[str, Any] | None = None,
    calibration_summary: Mapping[str, Any] | None = None,
    expected_variant: str | None = None,
    expected_seed: int | None = None,
    allow_unsafe: bool = True,
) -> dict[str, Any]:
    """Validate one v4 checkpoint and its own calibration summary."""
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "unexpected v4 checkpoint schema")
    _require(checkpoint.get("runtimeSchema") == RUNTIME_SCHEMA, "v4 checkpoint runtime schema mismatch")
    _require(checkpoint.get("testRead") is False, "v4 checkpoint claims test was read")

    protocol = checkpoint.get("protocol")
    _require(isinstance(protocol, Mapping), "v4 checkpoint protocol is missing")
    _require(protocol.get("schema") == TRAINING_SCHEMA, "v4 checkpoint training protocol schema mismatch")
    _require(protocol.get("testRead") is False, "v4 protocol claims test was read")
    _require(protocol.get("candidateUnion") is False, "v4 protocol permits candidate/GT union")
    _require(protocol.get("thresholdSource") == "checkpoint-own-calibration-only", "v4 protocol threshold source is invalid")
    _require(
        protocol.get("splitPoseCounts") == EXPECTED_SPLIT_POSE_COUNTS,
        "v4 protocol split pose counts disagree with the registered dataset",
    )
    split_names = protocol.get("splitNames")
    _require(isinstance(split_names, Mapping) and set(split_names) == {"train", "calibration", "validation"}, "v4 protocol split names are incomplete")
    _require("test" not in split_names and "test" not in protocol, "v4 protocol must not contain test provenance")

    seed = _int(protocol.get("seed"), label="v4 checkpoint seed")
    if expected_seed is not None:
        _require(seed == int(expected_seed), f"v4 checkpoint seed mismatch: expected {expected_seed}, got {seed}")
    if "seed" in checkpoint:
        _require(_int(checkpoint["seed"], label="v4 top-level seed") == seed, "v4 checkpoint seed fields disagree")

    config = checkpoint.get("modelConfig")
    _require(isinstance(config, Mapping), "v4 checkpoint modelConfig is missing")
    variant, variant_spec = _validate_variant(protocol, config, expected_variant=expected_variant)
    num_instances = _int(config.get("numInstances"), label="v4 modelConfig.numInstances")
    _require(num_instances > 0, "v4 modelConfig.numInstances must be positive")
    _require(_int(config.get("geometryDim"), label="v4 geometryDim") == GEOMETRY_DIM, "v4 geometry dimension is invalid")
    _require(_int(config.get("runtimeFeatureDim"), label="v4 runtimeFeatureDim") == RUNTIME_FEATURE_DIM, "v4 runtime feature dimension is invalid")
    _require(tuple(config.get("survivalCoefficientShape", ())) == SURVIVAL_COEFFICIENT_SHAPE, "v4 survival coefficient shape is invalid")
    _require(config.get("runtimeSchema") == RUNTIME_SCHEMA, "v4 modelConfig runtime schema mismatch")
    frequency = config.get("frequency")
    _require(isinstance(frequency, Mapping), "v4 frequency config is missing")
    _require(
        _int(frequency.get("count"), label="v4 frequency count") == SPECTRAL_FREQUENCY_COUNT,
        "v4 frequency count is invalid",
    )
    _require(frequency.get("units") == "cycles", "v4 frequency units must be cycles")
    _require(_finite(frequency.get("maxNormCycles")), "v4 maximum frequency norm is invalid")
    max_frequency_norm = float(frequency["maxNormCycles"])
    _require(
        0.0 < max_frequency_norm <= MAX_FREQUENCY_NORM_CYCLES,
        "v4 maximum frequency norm exceeds the registered range",
    )
    viewcell = protocol.get("viewcell")
    _require(isinstance(viewcell, Mapping), "v4 protocol view-cell contract is missing")
    _require(viewcell.get("shape") == "horizontal_disk", "v4 view-cell shape is invalid")
    _require(
        _finite(viewcell.get("radiusM")) and float(viewcell["radiusM"]) > 0.0,
        "v4 view-cell radius is invalid",
    )
    _require(
        viewcell.get("candidateCameraSemantics") == "66-degree back-camera candidate identity only",
        "v4 candidate-camera semantics are invalid",
    )
    _require(
        viewcell.get("queryCenterSemantics") == "center of the same-direction view-cell visibility union",
        "v4 query-center semantics are invalid",
    )
    coefficients = checkpoint.get("instanceSurvivalCoefficients")
    _require(_tensor_shape(coefficients) == (num_instances, *SURVIVAL_COEFFICIENT_SHAPE), "v4 instance survival coefficient shape mismatch")
    _require(_finite_array(coefficients), "v4 instance survival coefficients contain non-finite values")
    prior = checkpoint.get("instanceSurvivalPriorCoefficients")
    residual = checkpoint.get("instanceSurvivalCalibrationResidual")
    _require(_tensor_shape(prior) == (num_instances, *SURVIVAL_COEFFICIENT_SHAPE), "survival prior coefficient shape mismatch")
    _require(_tensor_shape(residual) == (num_instances, *SURVIVAL_COEFFICIENT_SHAPE), "instance calibration residual shape mismatch")
    _require(_finite_array(prior) and _finite_array(residual), "instance calibration tensors contain non-finite values")
    coefficients_np = np.asarray(coefficients.detach().cpu() if hasattr(coefficients, "detach") else coefficients, dtype=np.float32)
    prior_np = np.asarray(prior.detach().cpu() if hasattr(prior, "detach") else prior, dtype=np.float32)
    residual_np = np.asarray(residual.detach().cpu() if hasattr(residual, "detach") else residual, dtype=np.float32)
    _require(
        float(np.max(np.abs(coefficients_np - prior_np - residual_np), initial=0.0)) <= MAX_FP16_FUSION_ABS_ERROR,
        "fused survival coefficients disagree with prior plus residual",
    )
    checkpoint_calibration = checkpoint.get("instanceCalibration")
    _require(isinstance(checkpoint_calibration, Mapping), "v4 checkpoint instance calibration is missing")
    expected_calibration_mode = "disabled" if variant == "without_instance_calibration_residual" else "residual"
    _require(
        checkpoint_calibration.get("mode") == expected_calibration_mode,
        "v4 checkpoint instance calibration mode disagrees with variant",
    )
    _require(
        checkpoint_calibration.get("fusion") == "prior_plus_applied_residual",
        "v4 checkpoint instance calibration fusion is invalid",
    )
    _require(
        checkpoint_calibration.get("runtimeExport") == "fused_coefficients_only",
        "v4 checkpoint instance calibration export contract is invalid",
    )
    _require(
        _finite(checkpoint_calibration.get("blend"))
        and 0.0 <= float(checkpoint_calibration["blend"]) <= 1.0,
        "v4 checkpoint instance calibration blend is invalid",
    )
    model_state = checkpoint.get("modelState")
    _require(isinstance(model_state, Mapping), "v4 checkpoint modelState is missing")
    if expected_calibration_mode == "disabled":
        _require(
            float(np.max(np.abs(residual_np), initial=0.0)) <= 1e-7,
            "disabled instance calibration checkpoint has a non-zero residual",
        )
        _require(
            float(np.max(np.abs(coefficients_np - prior_np), initial=0.0)) <= MAX_FP16_FUSION_ABS_ERROR,
            "disabled instance calibration checkpoint does not equal its shared prior",
        )
        _require(
            abs(float(checkpoint_calibration["blend"])) <= 1e-7,
            "disabled instance calibration checkpoint has a non-zero blend",
        )
        _require(
            "instance_calibration_residual_raw" not in model_state,
            "disabled instance calibration checkpoint still contains residual parameters",
        )
    else:
        _require(
            "instance_calibration_residual_raw" in model_state,
            "residual instance calibration checkpoint has no residual parameters",
        )

    relation = checkpoint.get("relation")
    _require(isinstance(relation, Mapping), "v4 checkpoint relation provenance is missing")
    _require(relation.get("schema") == RELATION_SCHEMA, "v4 relation schema mismatch")
    _require(_int(relation.get("edgeCount"), label="v4 relation edge count") > 0, "v4 relation edge count is invalid")
    _require(_int(relation.get("rowCount"), label="v4 relation row count") > 0, "v4 relation row count is invalid")
    hierarchy = relation.get("hierarchy")
    _require(isinstance(hierarchy, Mapping), "v4 relation hierarchy declaration is missing")
    observations = relation.get("observations")
    _require(isinstance(observations, Mapping), "v4 relation observations declaration is missing")
    observation_count = _int(observations.get("observationCount"), label="v4 relation observation count")
    event_count = _int(observations.get("eventCount"), label="v4 relation event count")
    censored_count = _int(observations.get("rightCensoredCount"), label="v4 relation right-censored count")
    _require(observation_count > 0 and event_count >= 0 and censored_count >= 0, "v4 relation observation counts are invalid")
    _require(event_count + censored_count == observation_count, "v4 relation observation counts do not close")

    geometry = checkpoint.get("geometry")
    _require(isinstance(geometry, Mapping), "v4 checkpoint geometry provenance is missing")
    _require(list(geometry.get("shape", [])) == [num_instances, GEOMETRY_DIM], "v4 geometry shape is invalid")
    _require(geometry.get("dtype") == "float16", "v4 geometry dtype is invalid")

    _require(calibration_summary is not None, "v4 calibration_ready_summary is required")
    validation_at_frozen = calibration_summary.get("validationAtFrozenThreshold")
    _require(
        isinstance(validation_at_frozen, Mapping)
        and _int(validation_at_frozen.get("eval_pose_count"), label="v4 validation evaluation pose count")
        == EXPECTED_SPLIT_POSE_COUNTS["validation"],
        "v4 validation summary does not cover the full validation split",
    )
    selection, safe = _selection_from_calibration(calibration_summary, allow_unsafe=allow_unsafe)
    best = checkpoint.get("best")
    _require(isinstance(best, Mapping), "v4 checkpoint frozen best workpoint is missing")
    checkpoint_selection = best.get("selection", best)
    _require(isinstance(checkpoint_selection, Mapping), "v4 checkpoint best selection is missing")
    _require(_finite(checkpoint_selection.get("threshold")), "v4 checkpoint frozen threshold is invalid")
    _require(abs(float(checkpoint_selection["threshold"]) - float(selection["threshold"])) <= 1e-7, "v4 checkpoint threshold disagrees with calibration summary")
    checkpoint_calibration = checkpoint.get("calibration")
    if isinstance(checkpoint_calibration, Mapping):
        _require(checkpoint_calibration.get("schema") == CALIBRATION_ROW_SCHEMA, "v4 embedded calibration schema is invalid")
        _require(checkpoint_calibration.get("testRead") is False, "v4 embedded calibration claims test was read")
        embedded = checkpoint_calibration.get("selectedSafe") or checkpoint_calibration.get("diagnostic")
        if isinstance(embedded, Mapping):
            _require(_finite(embedded.get("threshold")), "v4 embedded calibration threshold is invalid")
            _require(abs(float(embedded["threshold"]) - float(selection["threshold"])) <= 1e-7, "v4 embedded calibration threshold disagrees")

    if model_meta is not None:
        _require(model_meta.get("schema") == MODEL_SCHEMA, "v4 model_meta schema mismatch")
        _require(model_meta.get("testRead") is False, "v4 model_meta claims test was read")
        _require(model_meta.get("modelConfig") == dict(config), "v4 model_meta modelConfig disagrees with checkpoint")
        _require(model_meta.get("protocol") == dict(protocol), "v4 model_meta protocol disagrees with checkpoint")
        runtime_feature = model_meta.get("runtimeFeature")
        _require(isinstance(runtime_feature, Mapping), "v4 model_meta runtimeFeature is missing")
        _require(list(runtime_feature.get("shape", [])) == [num_instances, RUNTIME_FEATURE_DIM], "v4 model_meta runtimeFeature shape is invalid")
        _require(runtime_feature.get("dtype") == "float16", "v4 model_meta runtimeFeature dtype is invalid")

    return {
        "status": "passed",
        "schema": CHECKPOINT_SCHEMA,
        "variant": variant,
        "variantSpec": variant_spec,
        "seed": seed,
        "geometry": {
            "shape": list(geometry.get("shape", [])),
            "dtype": str(geometry.get("dtype")),
        },
        "relation": {
            "schema": str(relation.get("schema")),
            "edgeCount": int(relation.get("edgeCount")),
            "rowCount": int(relation.get("rowCount")),
            "observationCount": observation_count,
            "eventCount": event_count,
            "rightCensoredCount": censored_count,
        },
        "threshold": float(selection["threshold"]),
        "safeWorkpoint": bool(safe),
        "epoch": _int(checkpoint.get("epoch"), label="v4 checkpoint epoch"),
        "testRead": False,
    }


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    return value


def self_test() -> dict[str, Any]:
    """Run a schema rejection test without project assets."""
    try:
        validate_v4_training_artifact({"schema": "pvs-hierarchical-relation-survival-integrated-training-v1"})
    except ValueError:
        return {"status": "passed", "schema": CHECKPOINT_SCHEMA}
    _fail("v4 validator accepted a historical checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-meta", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--expected-variant")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--strict-safe", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --self-test is used")
    if args.model_meta is None or args.calibration is None:
        parser.error("--model-meta and --calibration are required for a v4 checkpoint")
    checkpoint = _load_checkpoint(args.checkpoint)
    model_meta = json.loads(args.model_meta.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    output = validate_v4_training_artifact(
        checkpoint,
        model_meta=model_meta,
        calibration_summary=calibration,
        expected_variant=args.expected_variant,
        expected_seed=args.expected_seed,
        allow_unsafe=not args.strict_safe,
    )
    output["checkpoint"] = str(args.checkpoint.resolve())
    output["modelMeta"] = str(args.model_meta.resolve())
    output["calibrationSummary"] = str(args.calibration.resolve())
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
