#!/usr/bin/env python3
"""Build a compact dual-probe rescue specification from train-fitted probes.

The source probe reports may contain calibration and validation diagnostics,
but this builder copies only train-fitted standardization, ridge coefficients,
and train-only rescue certificates.  Fusion hyperparameters must be supplied
explicitly and are recorded as calibration-selected configuration; no metric
or validation row is copied into the runtime specification.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any


SOURCE_SCHEMA = "pvs-difficult-tail-feature-separability-v1"
OUTPUT_SCHEMA = "pvs-dual-probe-rescue-init-v1"
FEATURE_FAMILY = "combined"
FEATURE_DIM = 108
RAW_FEATURE_LAYOUT = [
    {"name": "center_view", "offset": 0, "dimension": 9},
    {"name": "spectral_features", "offset": 9, "dimension": 64},
    {"name": "boundary_spectral_summary", "offset": 73, "dimension": 8},
    {"name": "region_extrema_lower_upper_span", "offset": 81, "dimension": 27},
]


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != int(length):
        raise ValueError(f"{name} must contain {length} values")
    return [_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _risk_key(section: Mapping[str, Any], risk: float, name: str) -> Mapping[str, Any]:
    for key, value in section.items():
        try:
            matches = math.isclose(float(key), float(risk), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            matches = False
        if matches and isinstance(value, Mapping):
            return value
    available = ", ".join(str(key) for key in section)
    raise ValueError(f"{name} has no rescue certificate for risk {risk:g}; available: {available}")


def _probe_record(
    payload: Mapping[str, Any],
    *,
    role: str,
    expected_weight_power: float,
    rescue_risk: float,
) -> dict[str, Any]:
    if payload.get("schema") != SOURCE_SCHEMA or bool(payload.get("testRead", True)):
        raise ValueError(f"{role} probe must be a no-test {SOURCE_SCHEMA} report")
    probe = payload.get("probe")
    if not isinstance(probe, Mapping):
        raise ValueError(f"{role} probe report has no probe section")
    families = probe.get("families")
    family = families.get(FEATURE_FAMILY) if isinstance(families, Mapping) else None
    if not isinstance(family, Mapping) or family.get("status") != "ok":
        raise ValueError(f"{role} probe has no successful {FEATURE_FAMILY} family")
    if family.get("fitSplit") != "train" or bool(family.get("testRead", True)):
        raise ValueError(f"{role} probe coefficients must be fitted on train only")
    if int(family.get("featureCount", -1)) != FEATURE_DIM:
        raise ValueError(f"{role} probe feature dimension must be {FEATURE_DIM}")
    weighting = family.get("weighting")
    if not isinstance(weighting, Mapping):
        raise ValueError(f"{role} probe has no weighting provenance")
    weight_power = _finite_float(weighting.get("sampleWeightPower"), f"{role} weight power")
    if not math.isclose(weight_power, float(expected_weight_power), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{role} probe weight power {weight_power:g} does not match expected "
            f"{float(expected_weight_power):g}"
        )
    standardization = family.get("standardization")
    if not isinstance(standardization, Mapping):
        raise ValueError(f"{role} probe has no train standardization")
    mean = _finite_vector(standardization.get("mean"), FEATURE_DIM, f"{role} mean")
    scale = _finite_vector(standardization.get("scale"), FEATURE_DIM, f"{role} scale")
    if any(value <= 0.0 for value in scale):
        raise ValueError(f"{role} standardization scale must be positive")
    coefficients = _finite_vector(
        family.get("coefficients"), FEATURE_DIM + 1, f"{role} coefficients"
    )
    certificates = family.get("riskCertificates")
    if (
        not isinstance(certificates, Mapping)
        or certificates.get("sourceSplit") != "train"
        or not bool(certificates.get("fitRowsOnly", False))
    ):
        raise ValueError(f"{role} probe has no train-only risk certificates")
    rescue = certificates.get("rescue")
    if not isinstance(rescue, Mapping):
        raise ValueError(f"{role} probe has no rescue certificate section")
    certificate = _risk_key(rescue, float(rescue_risk), role)
    threshold = _finite_float(certificate.get("threshold"), f"{role} rescue threshold")
    observed_risk = _finite_float(certificate.get("observedRisk"), f"{role} observed risk")
    registered_risk = _finite_float(
        certificate.get("registeredRiskUpperBound"), f"{role} registered risk"
    )
    if observed_risk > registered_risk or not bool(certificate.get("riskWithinRegisteredBound", False)):
        raise ValueError(f"{role} rescue certificate exceeds its registered train risk")
    return {
        "role": role,
        "fitSplit": "train",
        "featureFamily": FEATURE_FAMILY,
        "featureDimension": FEATURE_DIM,
        "probeType": family.get("probeType"),
        "labelSemantics": "1=high_score_negative, 0=low_score_positive",
        "sampleWeightPower": weight_power,
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
        "coefficientOrder": ["intercept", "standardized_features"],
        "threshold": threshold,
        "riskCertificate": {
            "sourceSplit": "train",
            "fitRowsOnly": True,
            "risk": registered_risk,
            "observedRisk": observed_risk,
            "registeredRiskUpperBound": registered_risk,
            "riskWithinRegisteredBound": True,
            "threshold": threshold,
            "applyWhen": "probe_score <= threshold",
            "tiePolicy": certificate.get("tiePolicy"),
        },
    }


def build_spec(
    primary_payload: Mapping[str, Any],
    coverage_payload: Mapping[str, Any],
    *,
    alpha: float,
    temperature: float,
    coverage_weight: float,
    primary_rescue_risk: float,
    coverage_rescue_risk: float,
) -> dict[str, Any]:
    alpha_value = _finite_float(alpha, "alpha")
    temperature_value = _finite_float(temperature, "temperature")
    coverage_weight_value = _finite_float(coverage_weight, "coverage weight")
    if alpha_value <= 0.0 or temperature_value <= 0.0 or coverage_weight_value < 0.0:
        raise ValueError("alpha and temperature must be positive; coverage weight must be non-negative")
    primary = _probe_record(
        primary_payload,
        role="weighted_importance_rescue",
        expected_weight_power=0.5,
        rescue_risk=primary_rescue_risk,
    )
    coverage = _probe_record(
        coverage_payload,
        role="ordinary_positive_coverage_rescue",
        expected_weight_power=0.0,
        rescue_risk=coverage_rescue_risk,
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "version": 1,
        "enabled": True,
        "testRead": False,
        "selectedFromValidation": False,
        "selectionSplit": "calibration",
        "rawFeatureLayout": RAW_FEATURE_LAYOUT,
        "rawFeatureDimension": FEATURE_DIM,
        "primary": primary,
        "coverage": coverage,
        "alpha": alpha_value,
        "temperature": temperature_value,
        "coverageWeight": coverage_weight_value,
        "fusion": {
            "alpha": alpha_value,
            "temperature": temperature_value,
            "coverageWeight": coverage_weight_value,
            "formula": (
                "alpha * max(primary_attenuation * max(0,tanh((primary_threshold-primary_score)/temperature)), "
                "coverageWeight * coverage_attenuation * max(0,tanh((coverage_threshold-coverage_score)/temperature)))"
            ),
            "poseCentering": False,
            "allowsNegativeResidual": False,
            "attenuationRange": [0.0, 1.0],
        },
        "runtime": {
            "noAdditionalPerInstanceAssets": True,
            "sharedParameterPrecision": "float32",
            "onlineNeighborQuery": False,
            "onlineSubposeExpansion": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-probe", type=Path, required=True)
    parser.add_argument("--coverage-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--coverage-weight", type=float, default=2.0)
    parser.add_argument("--primary-rescue-risk", type=float, default=0.01)
    parser.add_argument("--coverage-rescue-risk", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    primary = json.loads(args.primary_probe.resolve().read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage_probe.resolve().read_text(encoding="utf-8"))
    if not isinstance(primary, Mapping) or not isinstance(coverage, Mapping):
        raise ValueError("probe inputs must contain JSON objects")
    result = build_spec(
        primary,
        coverage,
        alpha=args.alpha,
        temperature=args.temperature,
        coverage_weight=args.coverage_weight,
        primary_rescue_risk=args.primary_rescue_risk,
        coverage_rescue_risk=args.coverage_rescue_risk,
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
