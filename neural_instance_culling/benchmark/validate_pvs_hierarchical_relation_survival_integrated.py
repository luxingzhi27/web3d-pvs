#!/usr/bin/env python3
"""Validate the isolated hierarchical-relation PVS experiment contract.

This validator is intentionally independent from the training and evaluation
entry points.  It checks the provenance that a runner must record before a
result can enter a pilot, confirmation, or formal matrix:

* the four relation-source variants have distinct, auditable semantics;
* a checkpoint uses only train/calibration/validation for selection;
* candidate digests, pose order, variant identity, and seed identity agree;
* validation results use a threshold frozen on calibration;
* a one-shot test result can only consume that already-frozen threshold; and
* a matrix cannot silently mix candidates, poses, variants, or seeds.

The module does not read the test split and does not depend on the new runner.
It accepts decoded JSON-like payloads so unit tests can exercise rejection
paths without model weights or benchmark assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from common.provenance import protocol_digest, relation_artifact_digest  # noqa: E402


PREFIX = "pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1"
TRAINING_SCHEMA = "pvs-hierarchical-relation-survival-integrated-training-v1"
MODEL_SCHEMA = "pvs-hierarchical-relation-survival-integrated-v1"
CALIBRATION_SCHEMA = "pvs-hierarchical-relation-survival-integrated-calibration-v2"
EVALUATION_SCHEMA = "pvs-hierarchical-relation-survival-integrated-evaluation-v1"
MATRIX_SCHEMA = "pvs-hierarchical-relation-survival-integrated-matrix-manifest-v1"
CALIBRATION_PROTOCOL = "calibration_ready_pre_test"
DIAGNOSTIC_CALIBRATION_PROTOCOL = "diagnostic_calibration_unsafe"
TEST_PROTOCOL = "frozen_calibration_one_shot_test"

RELATION_SOURCES = ("hierarchical", "single_scale", "geometry_only", "free", "shuffled")
SPECTRAL_MODES = ("integrated", "learned_point", "fourier117")
LOSS_VARIANTS = ("rvl", "quality", "threshold_aligned_utility", "quality_resource")
FORMAL_SEEDS = (20260801, 20260802, 20260803)

# The formal seven-member design from the experiment plan.  The values use
# the training CLI spelling; ``validate_variant_semantics`` also accepts the
# camelCase spelling used in result manifests.
FORMAL_VARIANTS: dict[str, dict[str, Any]] = {
    "historical_strong_reference": {
        "relationSource": "free",
        "spectralMode": "fourier117",
        "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
    "full": {
        "relationSource": "hierarchical",
        "spectralMode": "integrated",
        "lossVariant": "threshold_aligned_utility",
        "effectiveResourceWeight": 0.05,
    },
    "without_hierarchical_relation": {
        "relationSource": "free",
        "spectralMode": "integrated",
        "lossVariant": "threshold_aligned_utility",
        "effectiveResourceWeight": 0.05,
    },
    "without_integrated_spectral": {
        "relationSource": "hierarchical",
        "spectralMode": "fourier117",
        "lossVariant": "threshold_aligned_utility",
        "effectiveResourceWeight": 0.05,
    },
    "without_quality_resource_loss": {
        "relationSource": "hierarchical",
        "spectralMode": "integrated",
        "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
    "shuffled_occlusion_source": {
        "relationSource": "shuffled",
        "spectralMode": "integrated",
        "lossVariant": "threshold_aligned_utility",
        "effectiveResourceWeight": 0.05,
    },
    "quality_risk_only": {
        "relationSource": "hierarchical",
        "spectralMode": "integrated",
        "lossVariant": "quality",
        "effectiveResourceWeight": 0.0,
    },
}

PILOT_VARIANTS: dict[str, dict[str, Any]] = {
    "R0_free_survival": {
        "relationSource": "free", "spectralMode": "fourier117", "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
    "R1_single_scale_relation": {
        "relationSource": "single_scale", "spectralMode": "fourier117", "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
    "R2_hierarchical_relation": {
        "relationSource": "hierarchical", "spectralMode": "fourier117", "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
    "R3_hierarchical_shuffled_source": {
        "relationSource": "shuffled", "spectralMode": "fourier117", "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
}

SPECTRAL_PILOT_VARIANTS: dict[str, dict[str, Any]] = {
    "S0_fourier117": {
        "relationSource": "hierarchical", "spectralMode": "fourier117", "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
    "S1_learned_spectral_point": {
        "relationSource": "hierarchical", "spectralMode": "learned_point", "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
    "S2_integrated_spectral": {
        "relationSource": "hierarchical", "spectralMode": "integrated", "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
}

LOSS_PILOT_VARIANTS: dict[str, dict[str, Any]] = {
    "L0_rvl_strong_v2": {
        "relationSource": "hierarchical", "spectralMode": "integrated", "lossVariant": "rvl",
        "effectiveResourceWeight": 0.0,
    },
    "L1_quality_risk": {
        "relationSource": "hierarchical", "spectralMode": "integrated", "lossVariant": "quality",
        "effectiveResourceWeight": 0.0,
    },
    "L2_quality_resource_r005": {
        "relationSource": "hierarchical", "spectralMode": "integrated", "lossVariant": "quality_resource",
        "effectiveResourceWeight": 0.05,
    },
    "L3_quality_resource_r010": {
        "relationSource": "hierarchical", "spectralMode": "integrated", "lossVariant": "quality_resource",
        "effectiveResourceWeight": 0.10,
    },
}

# Complete stage-C rechecks use a distinct registry entry.  The names and
# configurations match the module-recheck protocol, while remaining separate
# from the earlier eight-epoch pilot outputs.
MODULE_RECHECK_VARIANTS: dict[str, dict[str, Any]] = {
    "R1_single_scale_relation": PILOT_VARIANTS["R1_single_scale_relation"],
    "R3_hierarchical_shuffled_source": PILOT_VARIANTS["R3_hierarchical_shuffled_source"],
    "S1_learned_spectral_point": SPECTRAL_PILOT_VARIANTS["S1_learned_spectral_point"],
    "S2_integrated_spectral": SPECTRAL_PILOT_VARIANTS["S2_integrated_spectral"],
    "L1_quality_risk": LOSS_PILOT_VARIANTS["L1_quality_risk"],
}

REGISTERED_VARIANTS = {
    **PILOT_VARIANTS,
    **SPECTRAL_PILOT_VARIANTS,
    **LOSS_PILOT_VARIANTS,
    **MODULE_RECHECK_VARIANTS,
    **FORMAL_VARIANTS,
}

REQUIRED_METRICS = (
    "precision",
    "recall",
    "weightedRecall",
    "f1",
    "jaccard",
    "accuracy",
    "balancedAccuracy",
    "specificity",
    "usefulCull",
    "badCull",
    "avgPredCount",
    "avgCandidateCount",
    "avgGtCount",
    "predictedGlbCount",
    "predictedGlbBytes",
    "glbByteReduction",
    "requiredGlbCount",
    "requiredGlbBytes",
)
IMAGE_METRICS = ("missPixelRate", "wrongIdPixelRate", "extraPixelRate")

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


def _digest(value: Any, *, label: str) -> str:
    text = str(value)
    _require(len(text) == 64, f"{label} must be a 64-character SHA-256 digest")
    _require(all(char in "0123456789abcdef" for char in text.lower()), f"{label} is not hexadecimal SHA-256")
    return text


def _int(value: Any, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        _fail(f"{label} must be an integer")
    return result


def _same_sequence(actual: Any, expected: Iterable[int], *, label: str) -> list[int]:
    try:
        values = [_int(item, label=f"{label} item") for item in actual]
    except TypeError:
        _fail(f"{label} must be a sequence")
    wanted = [_int(item, label=f"{label} expected item") for item in expected]
    _require(values == wanted, f"{label} order/content mismatch: expected {wanted}, got {values}")
    _require(len(values) == len(set(values)), f"{label} contains duplicate pose IDs")
    return values


def _field(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _variant_fields(spec: Mapping[str, Any]) -> dict[str, Any]:
    relation_source = _field(spec, "relationSource", "relation_source", "source")
    spectral_mode = _field(spec, "spectralMode", "spectral_mode")
    loss_variant = _field(spec, "lossVariant", "loss_variant")
    resource_weight = _field(
        spec,
        "effectiveResourceWeight",
        "effective_resource_weight",
        "resourceWeight",
        "resource_weight",
        "lambdaResource",
        "lambda_resource",
        default=None,
    )
    quality_weight = _field(spec, "qualityWeight", "quality_weight", default=None)
    return {
        "relationSource": relation_source,
        "spectralMode": spectral_mode,
        "lossVariant": loss_variant,
        "effectiveResourceWeight": resource_weight,
        "qualityWeight": quality_weight,
    }


def validate_variant_semantics(
    variant: str,
    spec: Mapping[str, Any],
    relation_variant: Mapping[str, Any] | None = None,
    *,
    require_registered: bool = False,
) -> dict[str, Any]:
    """Validate the semantic tuple for one model member.

    ``relation_variant`` is the trainer's recorded ``relationVariant`` block.
    It is checked separately from the CLI tuple so a mislabeled checkpoint
    cannot claim to be hierarchical while the offline path used ``free`` or a
    source shuffle.
    """
    name = str(variant)
    normalized = _variant_fields(spec)
    relation_source = normalized["relationSource"]
    spectral_mode = normalized["spectralMode"]
    loss_variant = normalized["lossVariant"]
    _require(relation_source in RELATION_SOURCES, f"{name}: unknown relation source {relation_source!r}")
    _require(spectral_mode in SPECTRAL_MODES, f"{name}: unknown spectral mode {spectral_mode!r}")
    _require(loss_variant in LOSS_VARIANTS, f"{name}: unknown loss variant {loss_variant!r}")

    if require_registered:
        _require(name in REGISTERED_VARIANTS, f"{name}: variant is not registered")
    registered = REGISTERED_VARIANTS.get(name)
    if registered is not None:
        for key in ("relationSource", "spectralMode", "lossVariant"):
            _require(
                normalized[key] == registered[key],
                f"{name}: {key}={normalized[key]!r} does not match registered {registered[key]!r}",
            )

    raw_resource_weight = normalized["effectiveResourceWeight"]
    if raw_resource_weight is None:
        # The trainer's CLI keeps a default lambda for all modes, but it is
        # effective only for quality_resource.  Recording the effective value
        # avoids mistaking an ignored CLI default for an active loss term.
        effective_resource_weight = 0.05 if loss_variant in {"threshold_aligned_utility", "quality_resource"} else 0.0
    else:
        _require(_finite(raw_resource_weight), f"{name}: resource weight is non-finite")
        effective_resource_weight = float(raw_resource_weight) if loss_variant in {"threshold_aligned_utility", "quality_resource"} else 0.0
    _require(effective_resource_weight >= 0.0, f"{name}: resource weight must be non-negative")
    if loss_variant in {"threshold_aligned_utility", "quality_resource"}:
        _require(effective_resource_weight > 0.0, f"{name}: {loss_variant} requires a positive resource weight")
    else:
        _require(effective_resource_weight == 0.0, f"{name}: non-resource loss has an active resource weight")
    if registered is not None:
        _require(
            abs(effective_resource_weight - float(registered["effectiveResourceWeight"])) <= 1e-12,
            f"{name}: effective resource weight does not match the registered design",
        )

    if normalized["qualityWeight"] is not None:
        _require(_finite(normalized["qualityWeight"]) and float(normalized["qualityWeight"]) >= 0.0, f"{name}: invalid quality weight")
        if loss_variant in {"quality", "threshold_aligned_utility", "quality_resource"}:
            _require(float(normalized["qualityWeight"]) > 0.0, f"{name}: quality loss requires a positive quality weight")

    if relation_variant is not None:
        recorded_source = _field(relation_variant, "source", "relationSource")
        _require(recorded_source == relation_source, f"{name}: relationVariant source disagrees with variant spec")
        recorded_shuffled = bool(_field(relation_variant, "shuffled", default=False))
        _require(recorded_shuffled is (relation_source == "shuffled"), f"{name}: relationVariant shuffled flag is inconsistent")
        if relation_source == "single_scale":
            _require(
                _field(relation_variant, "hierarchy", default=None) == "identity",
                f"{name}: single_scale must record identity hierarchy",
            )
        if relation_source != "single_scale":
            hierarchy = _field(relation_variant, "hierarchy", default=None)
            _require(hierarchy != "identity", f"{name}: only single_scale may record identity hierarchy")

    return {
        "variant": name,
        "relationSource": relation_source,
        "spectralMode": spectral_mode,
        "lossVariant": loss_variant,
        "effectiveResourceWeight": float(effective_resource_weight),
    }


def _validate_pose_protocol(
    protocol: Mapping[str, Any],
    *,
    expected_pose_counts: Mapping[str, int] | None = None,
    expected_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    _require(
        protocol.get("schema") == "pvs-hierarchical-relation-survival-integrated-training-v1",
        "unexpected training protocol schema",
    )
    _require(
        protocol.get("protocolDigest") == protocol_digest(protocol),
        "training protocol digest is missing or does not match canonical protocol",
    )
    _require(protocol.get("testRead") is False, "training protocol claims test was read")
    _require(protocol.get("thresholdSource") == "calibration_only", "threshold source is not calibration_only")
    _require(protocol.get("candidateUnion") is False, "training protocol allows candidate/GT union")
    result: dict[str, Any] = {}
    for split in ("train", "calibration", "validation"):
        count_key = f"{split}PoseCount"
        digest_key = f"{split}CandidateDigest"
        count = _int(protocol.get(count_key), label=count_key)
        _require(count > 0, f"{count_key} must be positive")
        digest = _digest(protocol.get(digest_key), label=digest_key)
        if expected_pose_counts is not None and split in expected_pose_counts:
            _require(count == int(expected_pose_counts[split]), f"{count_key} does not match the dataset")
        if expected_digests is not None and split in expected_digests:
            _require(digest == _digest(expected_digests[split], label=f"expected {split} digest"), f"{digest_key} does not match the dataset")
        result[split] = {"poseCount": count, "candidateDigest": digest}
    _require(isinstance(protocol.get("splitNames"), Mapping), "training protocol splitNames is missing")
    _require(
        set(protocol["splitNames"]) == {"train", "calibration", "validation"},
        "training protocol splitNames must contain exactly train/calibration/validation",
    )
    for forbidden in ("testPoseCount", "testCandidateDigest", "testPoseIndices", "test"):
        _require(forbidden not in protocol, f"training protocol must not record {forbidden}")
    return result


def _validate_subpose_quality_protocol(
    protocol: Mapping[str, Any],
    *,
    loss_variant: str,
) -> dict[str, Any]:
    """Validate the dense-subpose supervision claim independently of files.

    The sidecar is allowed to be present for an RVL control so all members can
    share one staged dataset.  Only quality-aware losses may claim that the
    hit-rate supervision affected optimization, and those members must record
    the manifest digest and semantic description in the immutable protocol.
    """

    value = protocol.get("subposeQuality")
    # RVL-only controls predate the dense-subpose quality sidecar.  They are
    # valid controls precisely because the quality term is disabled; accepting
    # a missing block here preserves that provenance instead of inventing a
    # sidecar claim after training.  Quality-aware variants remain strict.
    if value is None:
        _require(
            loss_variant == "rvl",
            "training protocol subposeQuality is required for quality-aware losses",
        )
        return {
            "required": False,
            "available": False,
            "manifestSha256": None,
            "semantics": None,
        }
    _require(isinstance(value, Mapping), "training protocol subposeQuality must be an object")
    required = bool(value.get("required", False))
    available = bool(value.get("available", False))
    if loss_variant in {"quality", "threshold_aligned_utility", "quality_resource"}:
        _require(required, f"{loss_variant} must require subpose quality supervision")
        _require(available, f"{loss_variant} must record available subpose quality supervision")
        manifest_sha = value.get("manifestSha256")
        _digest(manifest_sha, label="subposeQuality.manifestSha256")
        semantics = value.get("semantics")
        _require(isinstance(semantics, str) and semantics.strip(), "subposeQuality semantics are missing")
    else:
        _require(not required, "RVL-only control cannot claim required subpose quality supervision")
    return {
        "required": required,
        "available": available,
        "manifestSha256": value.get("manifestSha256"),
        "semantics": value.get("semantics"),
    }


def _tensor_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            shape = np.asarray(value).shape
        except Exception as exc:  # pragma: no cover - defensive error text
            _fail(f"survivalCoefficients has no readable shape: {exc}")
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        _fail("survivalCoefficients has an invalid shape")


def _finite_array(value: Any) -> bool:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return bool(np.isfinite(np.asarray(value)).all())
    except Exception:
        return False


def validate_calibration_summary(
    summary: Mapping[str, Any],
    *,
    expected_threshold: float | None = None,
    allow_unsafe: bool = True,
) -> dict[str, Any]:
    """Validate the calibration-only threshold artifact."""
    _require(summary.get("schema") == CALIBRATION_SCHEMA, "unexpected calibration summary schema")
    protocol = str(summary.get("protocol", ""))
    _require(
        protocol in {CALIBRATION_PROTOCOL, DIAGNOSTIC_CALIBRATION_PROTOCOL},
        "calibration summary has an unknown protocol",
    )
    _require(summary.get("testRead") is False, "calibration summary claims test was read")
    _require(
        isinstance(summary.get("selectionRule"), str)
        and "aggregateWeightedRecall" in summary["selectionRule"],
        "calibration summary lacks the aggregate weighted-recall selection rule",
    )
    rows = summary.get("thresholdRows")
    _require(isinstance(rows, list) and rows, "calibration summary has no threshold rows")
    row_by_threshold: dict[float, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"calibration threshold row {index} is not an object")
        threshold = row.get("threshold")
        _require(_finite(threshold) and 0.0 <= float(threshold) <= 1.0, f"calibration threshold row {index} has an invalid threshold")
        for key in (
            "aggregateWeightedRecall",
            "poseMacroWeightedRecall",
        ):
            _require(_finite(row.get(key)), f"calibration threshold row {index} lacks {key}")
        replicates = int(row.get("weightedRecallBootstrapReplicates", 0))
        if protocol == CALIBRATION_PROTOCOL:
            _require(replicates >= 10000, f"calibration threshold row {index} lacks formal bootstrap replicates")
            _require(
                _finite(row.get("aggregateWeightedRecallLowerConfidenceBound")),
                f"calibration threshold row {index} lacks aggregate lower confidence bound",
            )
        row_by_threshold[round(float(threshold), 7)] = row

    selected = summary.get("selected")
    status = summary.get("status")
    if selected is None:
        _require(status == "no_qualified_safety_workpoint", "unsafe calibration summary has the wrong status")
        _require(allow_unsafe or protocol == DIAGNOSTIC_CALIBRATION_PROTOCOL, "calibration has no qualified safety workpoint")
        return {"safe": False, "threshold": None, "selected": None}
    _require(isinstance(selected, Mapping), "calibration selected workpoint is not an object")
    _require(status == "safe", "calibration selected a workpoint but is not marked safe")
    _require(_finite(selected.get("threshold")) and 0.0 <= float(selected["threshold"]) <= 1.0, "selected calibration threshold is invalid")
    _require(protocol == CALIBRATION_PROTOCOL, "unsafe diagnostic protocol cannot be marked safe")
    _require(float(selected.get("aggregateWeightedRecall", -1.0)) > 0.99, "selected calibration aggregate weighted recall does not pass the safety gate")
    _require(float(selected.get("aggregateWeightedRecallLowerConfidenceBound", -1.0)) > 0.99, "selected calibration aggregate weighted-recall lower bound does not pass the safety gate")
    key = round(float(selected["threshold"]), 7)
    _require(key in row_by_threshold, "selected calibration threshold is absent from thresholdRows")
    if expected_threshold is not None:
        _require(abs(float(selected["threshold"]) - float(expected_threshold)) <= 1e-6, "selected calibration threshold mismatch")
    return {"safe": True, "threshold": float(selected["threshold"]), "selected": dict(selected)}


def validate_training_artifact(
    checkpoint: Mapping[str, Any],
    *,
    model_meta: Mapping[str, Any] | None = None,
    calibration_summary: Mapping[str, Any] | None = None,
    expected_variant: str | None = None,
    expected_seed: int | None = None,
    expected_pose_counts: Mapping[str, int] | None = None,
    expected_digests: Mapping[str, str] | None = None,
    allow_unsafe: bool = True,
) -> dict[str, Any]:
    """Validate one new-model checkpoint and its pre-test provenance."""
    _require(checkpoint.get("schema") == TRAINING_SCHEMA, "unexpected hierarchical training checkpoint schema")
    _require(checkpoint.get("modelSchema") == MODEL_SCHEMA, "checkpoint model schema mismatch")
    _require(checkpoint.get("testRead") is False, "checkpoint claims test was read")
    args = checkpoint.get("args")
    _require(isinstance(args, Mapping), "checkpoint args are missing")
    seed = _int(args.get("seed"), label="checkpoint seed")
    if expected_seed is not None:
        _require(seed == int(expected_seed), f"checkpoint seed mismatch: expected {expected_seed}, got {seed}")
    if "seed" in checkpoint:
        _require(_int(checkpoint["seed"], label="checkpoint top-level seed") == seed, "checkpoint seed fields disagree")

    variant = str(
        _field(
            checkpoint,
            "variant",
            default=_field(args, "variant", "variant_name", default=expected_variant or ""),
        )
    )
    if expected_variant is not None:
        variant = str(expected_variant)
    _require(variant, "checkpoint variant is missing")
    variant_spec = checkpoint.get("variantSpec")
    if not isinstance(variant_spec, Mapping):
        variant_spec = {
            "relationSource": _field(args, "relation_source", "relationSource"),
            "spectralMode": _field(args, "spectral_mode", "spectralMode"),
            "lossVariant": _field(args, "loss_variant", "lossVariant"),
            "lambdaResource": _field(args, "lambda_resource", "lambdaResource", default=None),
            "qualityWeight": _field(args, "quality_weight", "qualityWeight", default=None),
        }
    relation_variant = checkpoint.get("relationProvenance", {}).get("variant")
    _require(isinstance(relation_variant, Mapping), "checkpoint relationVariant provenance is missing")
    normalized_variant = validate_variant_semantics(variant, variant_spec, relation_variant, require_registered=False)
    protocol = checkpoint.get("protocol")
    _require(isinstance(protocol, Mapping), "checkpoint protocol is missing")
    split_protocol = _validate_pose_protocol(
        protocol,
        expected_pose_counts=expected_pose_counts,
        expected_digests=expected_digests,
    )
    subpose_quality = _validate_subpose_quality_protocol(
        protocol,
        loss_variant=normalized_variant["lossVariant"],
    )
    relation_provenance = checkpoint.get("relationProvenance")
    _require(isinstance(relation_provenance, Mapping), "checkpoint relation provenance is missing")
    relation_meta = relation_provenance.get("relationMeta")
    _require(isinstance(relation_meta, Mapping), "checkpoint relation metadata is missing")
    _require(relation_meta.get("schema") == "pvs-viewcell-train-observed-relation-csr-v1", "checkpoint relation schema is invalid")
    identity = relation_meta.get("candidateIdentity")
    _require(isinstance(identity, Mapping), "checkpoint relation candidate identity is missing")
    canonical_digest = _digest(identity.get("canonicalCandidateDigest"), label="checkpoint canonical relation digest")
    render_digest = _digest(identity.get("renderCandidateDigest"), label="checkpoint render relation digest")
    _require(canonical_digest == protocol["trainCandidateDigest"], "checkpoint relation canonical digest disagrees with training protocol")
    _require(relation_provenance.get("trainCandidateDigest") == protocol["trainCandidateDigest"], "checkpoint relation train digest disagrees with training protocol")
    _digest(relation_provenance.get("relationArtifactDigest"), label="checkpoint relation artifact digest")

    config = checkpoint.get("config")
    _require(isinstance(config, Mapping), "checkpoint config is missing")
    _require(_int(config.get("numInstances"), label="config.numInstances") > 0, "config.numInstances must be positive")
    _require(_int(config.get("runtimeFeatureDim"), label="config.runtimeFeatureDim") == 124, "runtime feature dimension must be 124")
    _require(list(config.get("survivalCoefficientShape", [])) == [4, 7], "survival coefficient shape must be [4, 7]")
    _require(config.get("spectralMode") == normalized_variant["spectralMode"], "config spectral mode disagrees with args")
    coefficients = checkpoint.get("survivalCoefficients")
    _require(_tensor_shape(coefficients) == (_int(config.get("numInstances"), label="config.numInstances"), 4, 7), "survivalCoefficients shape mismatch")
    _require(_finite_array(coefficients), "survivalCoefficients contains non-finite values")

    calibration = checkpoint.get("calibration")
    calibration_result = None
    if calibration_summary is not None:
        calibration_result = validate_calibration_summary(calibration_summary, allow_unsafe=allow_unsafe)
    elif isinstance(calibration, Mapping):
        # The checkpoint keeps the same selected/diagnostic/rows payload but
        # does not repeat the standalone summary schema.
        selected = calibration.get("selected")
        if selected is not None:
            _require(float(selected.get("aggregateWeightedRecall", -1.0)) > 0.99, "checkpoint calibration aggregate weighted recall is unsafe")
            _require(float(selected.get("aggregateWeightedRecallLowerConfidenceBound", -1.0)) > 0.99, "checkpoint calibration aggregate lower bound is unsafe")
            calibration_result = {"safe": True, "threshold": float(selected["threshold"]), "selected": dict(selected)}
        else:
            calibration_result = {"safe": False, "threshold": None, "selected": None}
    if not allow_unsafe and calibration_result is not None:
        _require(calibration_result["safe"], "checkpoint has no qualified calibration workpoint")

    if model_meta is not None:
        _require(model_meta.get("schema") == TRAINING_SCHEMA, "model_meta schema mismatch")
        _require(model_meta.get("modelSchema") == MODEL_SCHEMA, "model_meta model schema mismatch")
        _require(model_meta.get("testRead") is False, "model_meta claims test was read")
        _require("protocolSplit" not in model_meta, "obsolete protocolSplit field is not accepted")
        _require(model_meta.get("protocolDigest") == protocol.get("protocolDigest"), "model_meta protocol digest disagrees with checkpoint")
        model_protocol = model_meta.get("protocol")
        _require(model_protocol == dict(protocol), "model_meta protocol disagrees with checkpoint")
        meta_args = model_meta.get("args")
        _require(isinstance(meta_args, Mapping), "model_meta args are missing")
        _require(_int(meta_args.get("seed"), label="model_meta seed") == seed, "model_meta seed disagrees with checkpoint")
        validate_variant_semantics(
            variant,
            model_meta.get("variantSpec", meta_args),
            model_meta.get("relationVariant"),
            require_registered=False,
        )
        relation_meta = model_meta.get("relation")
        _require(isinstance(relation_meta, Mapping), "model_meta relation provenance is missing")
        _require(relation_meta.get("schema") == "pvs-viewcell-train-observed-relation-csr-v1", "model_meta relation schema is invalid")
        _require(relation_meta.get("trainCandidateDigest") == protocol["trainCandidateDigest"], "model_meta relation train digest disagrees with protocol")
        _require(relation_meta.get("canonicalCandidateDigest") == canonical_digest, "model_meta canonical relation digest disagrees with checkpoint")
        _require(relation_meta.get("renderCandidateDigest") == render_digest, "model_meta render relation digest disagrees with checkpoint")
        _require(relation_meta.get("relationArtifactDigest") == relation_provenance.get("relationArtifactDigest"), "model_meta relation artifact digest disagrees with checkpoint")

    return {
        "status": "passed",
        "variant": variant,
        "variantSpec": normalized_variant,
        "seed": seed,
        "protocol": split_protocol,
        "subposeQuality": subpose_quality,
        "calibration": calibration_result,
        "testRead": False,
    }


def _validate_metrics(metrics: Any, *, label: str) -> None:
    _require(isinstance(metrics, Mapping), f"{label} metrics are missing")
    for name in REQUIRED_METRICS:
        _require(_finite(metrics.get(name)), f"{label} metric {name} is missing or non-finite")
    for name in ("tp", "fp", "fn", "tn", "weightedTp", "weightedGt"):
        _require(_finite(metrics.get(name)), f"{label} sufficient statistic {name} is missing or non-finite")


def _validate_image_metric_contract(result: Mapping[str, Any]) -> None:
    """Image metrics are optional; unavailable values must never be zero-filled."""
    unavailable = result.get("unavailableMetrics", {})
    _require(isinstance(unavailable, Mapping), "evaluation unavailableMetrics must be an object")
    metrics = result.get("metrics", {})
    for name in IMAGE_METRICS:
        if name in unavailable:
            _require(name not in metrics or metrics[name] is None, f"{name} is unavailable but has a numeric aggregate value")
        elif name in metrics:
            _require(_finite(metrics[name]), f"available image metric {name} is non-finite")


def _validate_pose_digest_map(
    values: Any,
    pose_indices: list[int],
    *,
    label: str,
) -> dict[int, str]:
    _require(isinstance(values, Mapping), f"{label} must be a pose-to-digest object")
    result: dict[int, str] = {}
    for pose in pose_indices:
        raw = values.get(str(pose), values.get(pose))
        result[pose] = _digest(raw, label=f"{label}[{pose}]")
    _require(set(str(key) for key in values) == {str(pose) for pose in pose_indices}, f"{label} contains an unexpected pose set")
    return result


def validate_evaluation_result(
    result: Mapping[str, Any],
    *,
    expected_variant: str | None = None,
    expected_seed: int | None = None,
    expected_split: str = "validation",
    expected_pose_indices: Iterable[int] | None = None,
    expected_candidate_digest: str | None = None,
    expected_threshold: float | None = None,
    allow_test: bool = False,
) -> dict[str, Any]:
    """Validate one calibration/validation/test result from the new runner."""
    _require(result.get("schema") == EVALUATION_SCHEMA, "unexpected new-model evaluation schema")
    split = str(result.get("split", ""))
    _require(split in {"calibration", "validation", "test"}, f"unsupported evaluation split {split!r}")
    _require(split == str(expected_split), f"evaluation split mismatch: expected {expected_split}, got {split}")
    test_read = result.get("testRead")
    _require(isinstance(test_read, bool), "evaluation testRead must be boolean")
    _require(test_read is (split == "test"), "evaluation testRead does not agree with split")
    if split == "test":
        _require(allow_test, "test evaluation is not authorized by this validator call")

    variant = str(result.get("variant", ""))
    _require(variant, "evaluation variant is missing")
    if expected_variant is not None:
        _require(variant == str(expected_variant), f"evaluation variant mismatch: expected {expected_variant}, got {variant}")
    spec = result.get("variantSpec")
    _require(isinstance(spec, Mapping), "evaluation variantSpec is missing")
    normalized = validate_variant_semantics(variant, spec, result.get("relationVariant"), require_registered=False)
    seed = _int(result.get("seed"), label="evaluation seed")
    if expected_seed is not None:
        _require(seed == int(expected_seed), f"evaluation seed mismatch: expected {expected_seed}, got {seed}")
    if "checkpointSeed" in result:
        _require(_int(result["checkpointSeed"], label="evaluation checkpointSeed") == seed, "evaluation checkpointSeed disagrees with seed")

    pose_indices = _same_sequence(result.get("poseIndices"), expected_pose_indices if expected_pose_indices is not None else result.get("poseIndices"), label="evaluation poseIndices")
    _require(_int(result.get("poseCount"), label="evaluation poseCount") == len(pose_indices), "evaluation poseCount does not match poseIndices")
    candidate_digest = _digest(result.get("candidateDigest"), label="evaluation candidateDigest")
    if expected_candidate_digest is not None:
        _require(candidate_digest == _digest(expected_candidate_digest, label="expected candidateDigest"), "evaluation candidate digest mismatch")
    threshold = result.get("threshold")
    _require(_finite(threshold) and 0.0 <= float(threshold) <= 1.0, "evaluation threshold is invalid")
    if expected_threshold is not None:
        _require(abs(float(threshold) - float(expected_threshold)) <= 1e-6, "evaluation threshold mismatch")

    source = result.get("thresholdSource")
    _require(isinstance(source, Mapping), "evaluation thresholdSource is missing")
    if split == "test":
        _require(source.get("protocol") == TEST_PROTOCOL, "test result does not use the one-shot frozen-threshold protocol")
        _require(_int(source.get("testEvaluationCount"), label="thresholdSource.testEvaluationCount") == 1, "test result must record exactly one test evaluation")
    else:
        _require(
            source.get("protocol") in {CALIBRATION_PROTOCOL, DIAGNOSTIC_CALIBRATION_PROTOCOL},
            "non-test result threshold is not calibration-frozen or explicitly diagnostic",
        )
        _require(_int(source.get("testEvaluationCount"), label="thresholdSource.testEvaluationCount") == 0, "non-test result threshold source read test")
    _require(source.get("selectionSplit") == "calibration", "threshold selection did not come from calibration")
    _require(source.get("selectedFromTest") is False, "threshold was selected from test")
    _require(_finite(source.get("selectedThreshold")) and abs(float(source["selectedThreshold"]) - float(threshold)) <= 1e-6, "thresholdSource selectedThreshold mismatch")

    _validate_metrics(result.get("metrics"), label="evaluation")
    _validate_image_metric_contract(result)
    pose_digests = _validate_pose_digest_map(result.get("poseCandidateDigests"), pose_indices, label="poseCandidateDigests")
    rows = result.get("perPose")
    _require(isinstance(rows, list) and len(rows) == len(pose_indices), "evaluation perPose count mismatch")
    row_indices: list[int] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"evaluation perPose[{index}] is not an object")
        pose = _int(row.get("poseIndex"), label=f"perPose[{index}].poseIndex")
        row_indices.append(pose)
        _require(pose in pose_digests, f"perPose[{index}] contains an unexpected pose")
        _require(_digest(row.get("candidateDigest"), label=f"perPose[{index}].candidateDigest") == pose_digests[pose], f"perPose[{index}] candidate digest mismatch")
        _validate_metrics(row.get("metrics"), label=f"perPose[{index}]")
    _require(row_indices == pose_indices, "perPose order does not match poseIndices")

    return {
        "status": "passed",
        "variant": variant,
        "variantSpec": normalized,
        "seed": seed,
        "split": split,
        "poseCount": len(pose_indices),
        "poseIndices": pose_indices,
        "candidateDigest": candidate_digest,
        "poseCandidateDigests": pose_digests,
        "threshold": float(threshold),
        "testRead": test_read,
    }


def validate_matrix_results(
    results: Iterable[Mapping[str, Any]],
    *,
    expected_variants: Iterable[str],
    expected_seeds: Iterable[int],
    expected_pose_indices: Iterable[int],
    expected_candidate_digest: str,
    split: str = "validation",
) -> dict[str, Any]:
    """Validate complete member coverage and cross-member identity."""
    variants = [str(value) for value in expected_variants]
    seeds = [_int(value, label="expected seed") for value in expected_seeds]
    poses = [_int(value, label="expected pose") for value in expected_pose_indices]
    _require(variants and len(variants) == len(set(variants)), "expected variant set is empty or duplicated")
    _require(seeds and len(seeds) == len(set(seeds)), "expected seed set is empty or duplicated")
    _digest(expected_candidate_digest, label="expected matrix candidateDigest")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for result in results:
        variant = str(result.get("variant", ""))
        seed = _int(result.get("seed"), label="matrix result seed")
        key = (variant, seed)
        _require(key not in indexed, f"duplicate matrix member {variant}/seed{seed}")
        indexed[key] = result
    expected_keys = {(variant, seed) for variant in variants for seed in seeds}
    _require(set(indexed) == expected_keys, f"matrix member set mismatch: expected {sorted(expected_keys)}, got {sorted(indexed)}")

    reference_pose_digests: dict[int, str] | None = None
    for variant in variants:
        for seed in seeds:
            checked = validate_evaluation_result(
                indexed[(variant, seed)],
                expected_variant=variant,
                expected_seed=seed,
                expected_split=split,
                expected_pose_indices=poses,
                expected_candidate_digest=expected_candidate_digest,
            )
            current = checked["poseCandidateDigests"]
            if reference_pose_digests is None:
                reference_pose_digests = current
            else:
                _require(current == reference_pose_digests, f"{variant}/seed{seed}: per-pose candidate digests differ")
    _require(reference_pose_digests is not None, "matrix has no members")
    return {
        "status": "passed",
        "schema": MATRIX_SCHEMA,
        "variantCount": len(variants),
        "seedCount": len(seeds),
        "poseCount": len(poses),
        "candidateDigest": _digest(expected_candidate_digest, label="expected matrix candidateDigest"),
        "testRead": False,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def self_test() -> dict[str, Any]:
    """Run a tiny schema self-test without accessing project assets."""
    bad = {"schema": EVALUATION_SCHEMA, "split": "validation", "testRead": True}
    try:
        validate_evaluation_result(bad)
    except ValueError:
        return {"status": "passed", "schema": EVALUATION_SCHEMA}
    _fail("validator self-test accepted an incomplete/test-reading result")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
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
    if args.result is None and args.checkpoint is None:
        parser.error("--result or --checkpoint is required unless --self-test is used")
    if args.result is not None:
        payload = json.loads(args.result.read_text(encoding="utf-8"))
        output = validate_evaluation_result(
            payload,
            expected_variant=args.expected_variant,
            expected_seed=args.expected_seed,
            expected_split=str(payload.get("split", "validation")),
            allow_test=False,
        )
    else:
        try:
            import torch
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must contain a mapping")
        model_meta = json.loads(args.model_meta.read_text(encoding="utf-8")) if args.model_meta else None
        calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
        output = validate_training_artifact(
            checkpoint,
            model_meta=model_meta,
            calibration_summary=calibration,
            expected_variant=args.expected_variant,
            expected_seed=args.expected_seed,
            allow_unsafe=not args.strict_safe,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
