#!/usr/bin/env python3
"""Scan a train-owned, bounded tail residual without retraining a model.

The scan is deliberately a posterior feasibility experiment.  It reads three
score captures (train, calibration, and validation), uses the score in each
capture as the authoritative base score, and asks the checkpoint only for the
input features needed by the train-fitted ridge probe.  No model probability
is substituted for a capture probability and no candidate or ground-truth
set is changed.

The probe label is ``1 = high-score negative`` and ``0 = low-score positive``.
Consequently, a low probe value receives a positive logit correction and a
high probe value receives a negative correction.  The symmetric correction is
bounded by ``alpha`` and centered by subtracting its mean separately for every
pose.  One-sided rescue and suppression corrections use train-only
finite-sample certificates and are not pose-centered: rescue is never
negative and suppression is never positive.
The optional ``dual_rescue`` mode combines a primary high-visual-weight rescue
probe with a separately certified ordinary-coverage rescue probe by taking the
pointwise maximum of their non-negative gates; it never reads a test split.
Calibration freezes one threshold per (alpha, temperature) member.  A member
is safe only when both its calibration weighted recall and its one-sided
pose-bootstrap 95% lower confidence bound are strictly greater than 0.99.
Validation is evaluated only after that calibration decision.

The command intentionally has no test-capture argument and rejects any test
provenance in an input.  This is a scan, not a training entry point: the only
learned quantities are the train-owned probe coefficients and
standardization values already stored in the probe JSON.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:  # Package imports are used by unittest; script imports are used by CLI.
    from .analyze_pvs_difficult_tail_feature_separability import (
        ScoreRow,
        _build_model_bundle,
        _load_checkpoint,
        _resolve_row_arrays,
        _row_dataset_inputs,
        load_capture as _load_analyzer_capture,
        region_extrema_features,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution
    from analyze_pvs_difficult_tail_feature_separability import (  # type: ignore
        ScoreRow,
        _build_model_bundle,
        _load_checkpoint,
        _resolve_row_arrays,
        _row_dataset_inputs,
        load_capture as _load_analyzer_capture,
        region_extrema_features,
    )

try:
    import torch
except ImportError:  # pragma: no cover - pure function tests do not need torch
    torch = None  # type: ignore[assignment]

try:
    from common.runtime_meta import load_runtime_meta
    from current_pvs_utils import threshold_grid, weighted_recall_lower_confidence_bound
    from pose_csr_dataset import PoseCSRDataset
except ImportError:  # pragma: no cover - package import fallback
    from neural_instance_culling.model.common.runtime_meta import load_runtime_meta
    from neural_instance_culling.model.current_pvs_utils import (
        threshold_grid,
        weighted_recall_lower_confidence_bound,
    )
    from neural_instance_culling.model.pose_csr_dataset import PoseCSRDataset


SCHEMA = "pvs-train-owned-tail-residual-posterior-scan-v1"
PROBE_SCHEMA = "pvs-difficult-tail-feature-separability-v1"
ALLOWED_SPLITS = ("train", "calibration", "validation")
TEST_SPLIT = "test"
SUPPORTED_FAMILIES = ("region_extrema", "combined")
SUPPORTED_RESIDUAL_MODES = (
    "symmetric",
    "rescue_only",
    "suppress_only",
    "dual",
    "dual_rescue",
)
PROBE_LABEL_SEMANTICS = "1=high_score_negative, 0=low_score_positive"
FEATURE_FAMILY_ORDER = ("center", "mean_std", "boundary", "region_extrema")
DEFAULT_ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50)
DEFAULT_TEMPERATURES = (0.05, 0.10, 0.25)
DEFAULT_GATE_TEMPERATURES: tuple[float | None, ...] = (None,)
DEFAULT_RESIDUAL_SIGNS = (1.0, -1.0)
DEFAULT_RESIDUAL_MODES = ("symmetric",)
DEFAULT_SUPPRESSION_RISK = 0.01
DEFAULT_RESCUE_RISK = 0.05
WEIGHTED_RECALL_FLOOR = 0.99
BOOTSTRAP_CONFIDENCE_QUANTILE = 0.05

__all__ = [
    "ALLOWED_SPLITS",
    "CandidateSplit",
    "ProbeSpec",
    "SCHEMA",
    "SUPPORTED_FAMILIES",
    "SUPPORTED_RESIDUAL_MODES",
    "apply_bounded_centered_residual",
    "build_probe_feature_matrix",
    "certificate_threshold_from_probe",
    "frontier_center_from_probe_payload",
    "freeze_calibration_workpoint",
    "load_probe_spec",
    "load_score_rows",
    "probe_linear_scores",
    "residual_diagnostics",
    "score_probe_families_from_aux",
    "scan_members",
]


@dataclass(frozen=True)
class ProbeSpec:
    """Train-owned lightweight probe parameters for one feature family."""

    family: str
    feature_count: int
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    fit_split: str
    ridge: float
    frontier_center_logit: float | None = None
    risk_certificates: Mapping[str, Any] | None = None
    probe_type: str = "standardized_ridge_linear"
    hinge_knots: np.ndarray | None = None
    hidden_weight: np.ndarray | None = None
    hidden_bias: np.ndarray | None = None
    output_weight: np.ndarray | None = None
    output_bias: float | None = None
    activation: str | None = None


@dataclass
class CandidateSplit:
    """Resolved candidate-aligned rows for one allowed split."""

    split: str
    rows: list[ScoreRow]
    probe_scores: list[np.ndarray] | None = None
    coverage_probe_scores: list[np.ndarray] | None = None


def _finite_vector(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_values = np.exp(values[~positive])
    result[~positive] = negative_values / (1.0 + negative_values)
    return result


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not bool(np.isfinite(values).all()) or bool(np.any((values < 0.0) | (values > 1.0))):
        raise ValueError("base scores must be finite probabilities in [0, 1]")
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    return np.log(clipped) - np.log1p(-clipped)


def frontier_center_from_probe_payload(payload: Mapping[str, Any]) -> float | None:
    """Return the train-owned frontier midpoint in base-score logit space.

    ``fitCutoffs`` are not probabilities produced by the ridge probe.  They
    are two *base visibility score* cutoffs fitted on train.  Only these
    base-score values are converted with the probability logit function; the
    ridge output remains an unbounded regression value throughout the residual
    formula.
    """

    tail_definition = payload.get("tailDefinition")
    if not isinstance(tail_definition, Mapping):
        raise ValueError("probe has no tailDefinition")
    fit_cutoffs = tail_definition.get("fitCutoffs")
    if fit_cutoffs is None:
        return None
    if not isinstance(fit_cutoffs, Mapping) or fit_cutoffs.get("sourceSplit") != "train":
        raise ValueError("probe fitCutoffs must be explicitly train-owned")
    try:
        low = float(fit_cutoffs["lowPositiveScoreCutoff"])
        high = float(fit_cutoffs["highNegativeScoreCutoff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("train-owned fitCutoffs lack the two base-score cutoffs") from exc
    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or not 0.0 < low < 1.0
        or not 0.0 < high < 1.0
        or low >= high
    ):
        raise ValueError("train-owned fitCutoffs must satisfy 0 < low < high < 1")
    return float(0.5 * (_logit(np.asarray([low]))[0] + _logit(np.asarray([high]))[0]))


def _parse_float_list(value: str, *, name: str, minimum: float, strict: bool) -> tuple[float, ...]:
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        raise ValueError(f"{name} must contain at least one number")
    parsed: list[float] = []
    for piece in pieces:
        try:
            number = float(piece)
        except ValueError as exc:
            raise ValueError(f"{name} contains an invalid number: {piece!r}") from exc
        if not np.isfinite(number) or (strict and number <= minimum) or (not strict and number < minimum):
            relation = ">" if strict else ">="
            raise ValueError(f"{name} values must be finite and {relation} {minimum}")
        parsed.append(number)
    return tuple(dict.fromkeys(parsed))


def _parse_gate_temperature_list(value: str) -> tuple[float | None, ...]:
    """Parse positive gate temperatures; ``0`` and ``none`` disable gating."""

    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        raise ValueError("gate-temperatures must contain none, 0, or a positive number")
    parsed: list[float | None] = []
    for piece in pieces:
        if piece.lower() in {"none", "null", "off"} or piece == "0":
            candidate: float | None = None
        else:
            try:
                candidate = float(piece)
            except ValueError as exc:
                raise ValueError(f"gate-temperatures contains an invalid value: {piece!r}") from exc
            if not np.isfinite(candidate) or candidate <= 0.0:
                raise ValueError("active gate temperatures must be finite and positive")
        if candidate not in parsed:
            parsed.append(candidate)
    return tuple(parsed)


def _parse_residual_signs(value: str) -> tuple[float, ...]:
    parsed = _parse_float_list(value, name="residual-signs", minimum=-1.0, strict=False)
    if any(not np.isclose(abs(number), 1.0) for number in parsed):
        raise ValueError("residual-signs may contain only +1 and -1")
    return parsed


def _assert_test_free_payload(payload: Mapping[str, Any], *, name: str) -> None:
    if payload.get("testRead") is not False:
        raise ValueError(f"{name} must explicitly declare testRead=false")
    splits = payload.get("splitsRead")
    if splits is not None:
        if not isinstance(splits, list) or any(str(item) == TEST_SPLIT for item in splits):
            raise ValueError(f"{name} declares a forbidden test split")
    provenance = payload.get("inputProvenance")
    if isinstance(provenance, Mapping) and provenance.get("testRead") is not False:
        raise ValueError(f"{name} input provenance is not test-free")


def load_probe_spec(path: Path, family: str) -> tuple[ProbeSpec, dict[str, Any]]:
    """Read only a train-fitted probe family and validate its provenance."""

    family = str(family)
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported probe family {family!r}; use one of {SUPPORTED_FAMILIES}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("probe JSON must contain an object")
    _assert_test_free_payload(payload, name="probe")
    if payload.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"probe schema must be {PROBE_SCHEMA!r}")
    splits_read = payload.get("splitsRead")
    if not isinstance(splits_read, list) or not set(ALLOWED_SPLITS).issubset(
        {str(item) for item in splits_read}
    ):
        raise ValueError("probe must declare train, calibration, and validation splits")
    tail_definition = payload.get("tailDefinition")
    if not isinstance(tail_definition, Mapping) or tail_definition.get("probeLabel") != PROBE_LABEL_SEMANTICS:
        raise ValueError("probe label semantics are not 1=high_score_negative, 0=low_score_positive")
    probe = payload.get("probe")
    if not isinstance(probe, Mapping):
        raise ValueError("probe JSON has no probe section")
    if probe.get("fitSplit") != "train" or probe.get("thresholdSource") != "calibration":
        raise ValueError("probe must be fitted on train and thresholded from calibration")
    if bool(probe.get("selectedFromValidation", True)):
        raise ValueError("probe records validation-based selection")
    families = probe.get("families")
    if not isinstance(families, Mapping) or not isinstance(families.get(family), Mapping):
        raise ValueError(f"probe has no family {family!r}")
    record = families[family]
    if record.get("status") != "ok" or record.get("fitSplit") != "train":
        raise ValueError(f"probe family {family!r} is not a train-fitted usable probe")
    try:
        feature_count = int(record["featureCount"])
        standardization = record["standardization"]
        mean = _finite_vector(standardization["mean"], name=f"probe.{family}.mean")
        scale = _finite_vector(standardization["scale"], name=f"probe.{family}.scale")
        ridge = float(record["ridge"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"probe family {family!r} has incomplete parameters") from exc
    probe_type = str(record.get("probeType", "standardized_ridge_linear"))
    coefficients = np.zeros((0,), dtype=np.float64)
    hinge_knots = None
    hidden_weight = None
    hidden_bias = None
    output_weight = None
    output_bias = None
    activation = None
    if probe_type == "standardized_ridge_linear":
        coefficients = _finite_vector(
            record.get("coefficients"), name=f"probe.{family}.coefficients"
        )
        if coefficients.size != feature_count + 1:
            raise ValueError(f"probe family {family!r} coefficient dimension is invalid")
    elif probe_type == "standardized_ridge_hinge":
        coefficients = _finite_vector(
            record.get("coefficients"), name=f"probe.{family}.coefficients"
        )
        hinge_knots = _finite_vector(
            record.get("hingeKnots"), name=f"probe.{family}.hingeKnots"
        )
        if hinge_knots.size == 0 or np.unique(hinge_knots).size != hinge_knots.size:
            raise ValueError(f"probe family {family!r} hinge knots must be unique")
        expected = 1 + feature_count * (1 + int(hinge_knots.size))
        if coefficients.size != expected:
            raise ValueError(f"probe family {family!r} hinge coefficient dimension is invalid")
    elif probe_type == "standardized_shallow_mlp":
        parameters = record.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"probe family {family!r} MLP parameters are missing")
        hidden_weight = np.asarray(parameters.get("hiddenWeight"), dtype=np.float64)
        hidden_bias = _finite_vector(
            parameters.get("hiddenBias"), name=f"probe.{family}.hiddenBias"
        )
        output_weight = _finite_vector(
            parameters.get("outputWeight"), name=f"probe.{family}.outputWeight"
        )
        try:
            output_bias = float(parameters["outputBias"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"probe family {family!r} MLP output bias is invalid") from exc
        activation = str(record.get("activation", "relu"))
        if (
            hidden_weight.ndim != 2
            or hidden_weight.shape[1] != feature_count
            or hidden_weight.shape[0] != hidden_bias.size
            or output_weight.size != hidden_bias.size
            or not bool(np.isfinite(hidden_weight).all())
            or not np.isfinite(output_bias)
            or activation not in {"relu", "tanh"}
        ):
            raise ValueError(f"probe family {family!r} MLP parameter dimensions are invalid")
    else:
        raise ValueError(f"probe family {family!r} has unsupported probeType={probe_type!r}")
    if feature_count <= 0:
        raise ValueError(f"probe family {family!r} feature dimension is invalid")
    if mean.size != feature_count or scale.size != feature_count:
        raise ValueError(f"probe family {family!r} standardization dimension is invalid")
    if bool(np.any(scale <= 0.0)) or not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError(f"probe family {family!r} has invalid scale or ridge")
    combined_names = probe.get("combinedFeatureNames")
    if family == "combined" and (
        not isinstance(combined_names, list) or len(combined_names) != feature_count
    ):
        raise ValueError("combined probe feature names do not match its coefficient dimension")
    risk_certificates = record.get("riskCertificates")
    if risk_certificates is not None:
        if not isinstance(risk_certificates, Mapping):
            raise ValueError("probe riskCertificates must be an object")
        if risk_certificates.get("sourceSplit") != "train" or not bool(
            risk_certificates.get("fitRowsOnly", False)
        ):
            raise ValueError("probe riskCertificates must be train-only fit-row certificates")
        for kind in ("suppression", "rescue"):
            section = risk_certificates.get(kind)
            if not isinstance(section, Mapping):
                raise ValueError(f"probe riskCertificates has no {kind} section")
            for risk_key, certificate in section.items():
                if not isinstance(certificate, Mapping):
                    raise ValueError(f"probe {kind} certificate {risk_key!r} is invalid")
                try:
                    risk_value = float(risk_key)
                    threshold = float(certificate["threshold"])
                    observed = float(certificate["observedRisk"])
                    registered = float(certificate["registeredRiskUpperBound"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"probe {kind} certificate {risk_key!r} is incomplete") from exc
                if (
                    not np.isfinite(risk_value)
                    or not np.isfinite(threshold)
                    or not np.isfinite(observed)
                    or not np.isfinite(registered)
                    or risk_value < 0.0
                    or observed > registered
                    or registered < 0.0
                    or not bool(certificate.get("riskWithinRegisteredBound", False))
                ):
                    raise ValueError(f"probe {kind} certificate {risk_key!r} violates its risk bound")
    return (
        ProbeSpec(
            family=family,
            feature_count=feature_count,
            coefficients=coefficients,
            mean=mean,
            scale=scale,
            fit_split="train",
            ridge=ridge,
            frontier_center_logit=frontier_center_from_probe_payload(payload),
            risk_certificates=(
                dict(risk_certificates) if isinstance(risk_certificates, Mapping) else None
            ),
            probe_type=probe_type,
            hinge_knots=hinge_knots,
            hidden_weight=hidden_weight,
            hidden_bias=hidden_bias,
            output_weight=output_weight,
            output_bias=output_bias,
            activation=activation,
        ),
        dict(payload),
    )


def certificate_threshold_from_probe(
    probe: ProbeSpec,
    *,
    kind: str,
    risk: float,
) -> float:
    """Return a threshold from a train-only certificate, never calibration."""

    if kind not in {"suppression", "rescue"}:
        raise ValueError("certificate kind must be suppression or rescue")
    if not np.isfinite(risk) or float(risk) < 0.0:
        raise ValueError("certificate risk must be finite and non-negative")
    certificates = probe.risk_certificates
    if not isinstance(certificates, Mapping) or certificates.get("sourceSplit") != "train":
        raise ValueError(f"probe has no train-only {kind} certificates")
    section = certificates.get(kind)
    if not isinstance(section, Mapping):
        raise ValueError(f"probe has no train-only {kind} certificates")
    for key, certificate in section.items():
        try:
            matches = bool(np.isclose(float(key), float(risk)))
        except (TypeError, ValueError):
            matches = False
        if matches:
            if not isinstance(certificate, Mapping):
                break
            threshold = float(certificate.get("threshold", float("nan")))
            observed = float(certificate.get("observedRisk", float("nan")))
            registered = float(certificate.get("registeredRiskUpperBound", float("nan")))
            if (
                not np.isfinite(threshold)
                or not np.isfinite(observed)
                or not np.isfinite(registered)
                or observed > registered
            ):
                raise ValueError(f"{kind} certificate for risk {risk:g} is invalid")
            return threshold
    available = ", ".join(str(key) for key in section.keys())
    raise ValueError(
        f"probe has no {kind} certificate for risk {float(risk):g}; available: {available}"
    )


def load_score_rows(path: Path, split: str, dataset: Any) -> CandidateSplit:
    """Load one score capture and recover IDs only from the fixed dataset CSR."""

    split = str(split)
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split {split!r} is not allowed; test is forbidden")
    payload = _load_analyzer_capture(Path(path).resolve(), split)
    rows = _resolve_row_arrays(payload["_normalizedRows"], dataset=dataset, split=split)
    if not rows:
        raise ValueError(f"{split} score capture has no poses")
    for row in rows:
        if row.scores is None or row.targets is None or row.visible_weights is None:
            raise ValueError(
                f"{split} pose {row.pose_index} must contain candidateScores, targets, and visibleWeights"
            )
        if row.candidate_ids.size != row.scores.size:
            raise ValueError(f"{split} pose {row.pose_index} candidate count disagrees with scores")
        if bool(np.any(row.targets < 0.0) or np.any(row.targets > 1.0)):
            raise ValueError(f"{split} pose {row.pose_index} targets are not binary")
    return CandidateSplit(split=split, rows=rows)


def probe_linear_scores(features: np.ndarray, probe: ProbeSpec) -> np.ndarray:
    """Apply a train-owned lightweight probe without fitting.

    The historical public name is retained because callers already import it,
    but the function now dispatches over linear, additive hinge, and one-hidden-
    layer probes.  All three consume the same fixed 108-dimensional runtime
    query and add no per-instance asset.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != probe.feature_count:
        raise ValueError(
            f"features for {probe.family} must have shape [N, {probe.feature_count}]"
        )
    if not bool(np.isfinite(values).all()):
        raise ValueError("probe features contain non-finite values")
    standardized = (values - probe.mean[None, :]) / probe.scale[None, :]
    if probe.probe_type == "standardized_ridge_linear":
        result = probe.coefficients[0] + standardized @ probe.coefficients[1:]
    elif probe.probe_type == "standardized_ridge_hinge":
        if probe.hinge_knots is None:
            raise ValueError("hinge probe has no knots")
        pieces = [standardized]
        pieces.extend(
            np.maximum(standardized - float(knot), 0.0)
            for knot in probe.hinge_knots.tolist()
        )
        design = np.concatenate(pieces, axis=1)
        result = probe.coefficients[0] + design @ probe.coefficients[1:]
    elif probe.probe_type == "standardized_shallow_mlp":
        if (
            probe.hidden_weight is None
            or probe.hidden_bias is None
            or probe.output_weight is None
            or probe.output_bias is None
        ):
            raise ValueError("shallow MLP probe parameters are incomplete")
        hidden = standardized @ probe.hidden_weight.T + probe.hidden_bias[None, :]
        if probe.activation == "relu":
            hidden = np.maximum(hidden, 0.0)
        elif probe.activation == "tanh":
            hidden = np.tanh(hidden)
        else:
            raise ValueError(f"unsupported shallow MLP activation {probe.activation!r}")
        result = hidden @ probe.output_weight + float(probe.output_bias)
    else:
        raise ValueError(f"unsupported probe type {probe.probe_type!r}")
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("probe score is non-finite")
    return result


def build_probe_feature_matrix(aux: Mapping[str, Any], family: str) -> np.ndarray:
    """Build the requested candidate-aligned family from model auxiliary output."""

    def array(name: str) -> np.ndarray:
        value = aux.get(name)
        if value is None:
            raise ValueError(f"model auxiliary output has no {name}")
        if torch is not None and isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        result = np.asarray(value, dtype=np.float64)
        if result.ndim != 2 or not bool(np.isfinite(result).all()):
            raise ValueError(f"model auxiliary {name} must be a finite matrix")
        return result

    center = array("center_view")
    spectral = array("spectral_features")
    boundary = array("boundary_spectral_summary")
    axes_value = aux.get("disk_axes")
    if axes_value is None:
        raise ValueError("model auxiliary output has no disk_axes")
    if torch is not None and isinstance(axes_value, torch.Tensor):
        axes_value = axes_value.detach().cpu().numpy()
    axes = np.asarray(axes_value, dtype=np.float64)
    extrema = region_extrema_features(center, axes)
    if family == "region_extrema":
        result = extrema
    elif family == "combined":
        if not (center.shape[0] == spectral.shape[0] == boundary.shape[0] == extrema.shape[0]):
            raise ValueError("model auxiliary feature families have different row counts")
        result = np.concatenate([center, spectral, boundary, extrema], axis=1)
    else:
        raise ValueError(f"unsupported probe family {family!r}")
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("constructed probe features are non-finite")
    return result


def score_probe_families_from_aux(
    aux: Mapping[str, Any],
    probes: Mapping[str, ProbeSpec],
) -> dict[str, np.ndarray]:
    """Score several train-owned probes while caching each feature family once.

    A primary and coverage probe may use different ridge coefficients but share
    the same reconstructed model feature family.  Reconstructing that family
    once keeps the diagnostic path faithful to the intended runtime cost and
    makes the sharing explicit for tests and provenance.
    """

    if not isinstance(probes, Mapping) or not probes:
        raise ValueError("probes must be a non-empty mapping")
    feature_cache: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for name, probe in probes.items():
        if not isinstance(probe, ProbeSpec):
            raise TypeError(f"probe {name!r} is not a ProbeSpec")
        if probe.family not in feature_cache:
            feature_cache[probe.family] = build_probe_feature_matrix(aux, probe.family)
        scores[str(name)] = probe_linear_scores(feature_cache[probe.family], probe)
    return scores


def _bounded_residual_components(
    base_scores: np.ndarray,
    probe_scores: np.ndarray,
    pose_indices: np.ndarray,
    *,
    alpha: float,
    temperature: float,
    mode: str = "symmetric",
    residual_sign: float = 1.0,
    gate_temperature: float | None = None,
    frontier_center_logit: float | None = None,
    suppression_threshold: float | None = None,
    rescue_threshold: float | None = None,
    coverage_probe_scores: np.ndarray | None = None,
    coverage_threshold: float | None = None,
    coverage_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and actual logit residuals for one candidate-aligned pose batch."""

    mode = str(mode)
    if mode not in SUPPORTED_RESIDUAL_MODES:
        raise ValueError(f"unsupported residual mode {mode!r}")

    base = _finite_vector(base_scores, name="base scores")
    probe = _finite_vector(probe_scores, name="probe scores")
    poses = np.asarray(pose_indices, dtype=np.int64).reshape(-1)
    if base.size != probe.size or base.size != poses.size:
        raise ValueError("base scores, probe scores, and pose indices are not aligned")
    if not np.isfinite(alpha) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not np.isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(residual_sign) or not np.isclose(abs(float(residual_sign)), 1.0):
        raise ValueError("residual_sign must be +1 or -1")
    if mode != "symmetric" and not np.isclose(float(residual_sign), 1.0):
        raise ValueError("one-sided residual modes have a fixed positive orientation")
    if mode in {"rescue_only", "dual", "dual_rescue"} and (
        rescue_threshold is None or not np.isfinite(rescue_threshold)
    ):
        raise ValueError(f"{mode} requires a finite train-only rescue threshold")
    if mode == "dual_rescue":
        if coverage_probe_scores is None:
            raise ValueError("dual_rescue requires coverage probe scores")
        if coverage_threshold is None or not np.isfinite(coverage_threshold):
            raise ValueError("dual_rescue requires a finite train-only coverage rescue threshold")
        if not np.isfinite(coverage_weight) or float(coverage_weight) < 0.0:
            raise ValueError("dual_rescue coverage_weight must be finite and non-negative")
        if gate_temperature is not None and float(gate_temperature) != 0.0:
            raise ValueError("dual_rescue does not support a frontier gate")
    if mode in {"suppress_only", "dual"} and (
        suppression_threshold is None or not np.isfinite(suppression_threshold)
    ):
        raise ValueError(f"{mode} requires a finite train-only suppression threshold")
    if mode == "dual_rescue" or gate_temperature is None or float(gate_temperature) == 0.0:
        gate = np.ones_like(base)
    else:
        if (
            frontier_center_logit is None
            or not np.isfinite(frontier_center_logit)
            or not np.isfinite(gate_temperature)
            or float(gate_temperature) <= 0.0
        ):
            raise ValueError(
                "an active frontier gate requires a finite positive gate_temperature "
                "and train-owned frontier_center_logit"
            )
        gate = np.exp(
            -np.abs(_logit(base) - float(frontier_center_logit)) / float(gate_temperature)
        )
    if mode == "symmetric":
        # The ridge value is deliberately used directly here.  The fixed 0.5
        # is the train probe's label-space midpoint, not a probability cutoff.
        raw_residual = (
            float(residual_sign)
            * float(alpha)
            * np.tanh((0.5 - probe) / float(temperature))
            * gate
        )
        actual_residual = raw_residual.copy()
        if actual_residual.size:
            _, inverse = np.unique(poses, return_inverse=True)
            sums = np.bincount(inverse, weights=raw_residual)
            counts = np.bincount(inverse)
            actual_residual -= sums[inverse] / np.maximum(counts[inverse], 1)
    elif mode == "dual_rescue":
        coverage = _finite_vector(coverage_probe_scores, name="coverage probe scores")
        if coverage.size != probe.size:
            raise ValueError("primary and coverage probe scores are not aligned")
        primary_gate = np.maximum(
            0.0,
            np.tanh((float(rescue_threshold) - probe) / float(temperature)),
        )
        coverage_gate = np.maximum(
            0.0,
            np.tanh((float(coverage_threshold) - coverage) / float(temperature)),
        )
        # The max keeps the stronger of the high-importance rescue probe and
        # the ordinary-coverage probe without ever signing a negative logit
        # correction.  It is deliberately not pose-centered.
        raw_residual = float(alpha) * np.maximum(
            primary_gate,
            float(coverage_weight) * coverage_gate,
        )
        actual_residual = raw_residual.copy()
        if bool(np.any(actual_residual < -1e-12)):
            raise AssertionError("dual_rescue residual became negative")
    else:
        rescue = np.zeros_like(probe)
        suppress = np.zeros_like(probe)
        if mode in {"rescue_only", "dual"}:
            rescue = np.maximum(
                0.0,
                np.tanh((float(rescue_threshold) - probe) / float(temperature)),
            )
        if mode in {"suppress_only", "dual"}:
            suppress = -np.maximum(
                0.0,
                np.tanh((probe - float(suppression_threshold)) / float(temperature)),
            )
        raw_residual = float(alpha) * (rescue + suppress) * gate
        # A one-sided certificate is a pointwise statement.  Centering would
        # turn a certified non-negative/non-positive correction into the
        # opposite sign for some candidates, so it is intentionally disabled.
        actual_residual = raw_residual.copy()
        if mode == "rescue_only" and bool(np.any(actual_residual < -1e-12)):
            raise AssertionError("rescue-only residual became negative")
        if mode == "suppress_only" and bool(np.any(actual_residual > 1e-12)):
            raise AssertionError("suppress-only residual became positive")
    if not bool(np.isfinite(raw_residual).all() and np.isfinite(actual_residual).all()):
        raise FloatingPointError("bounded residual produced non-finite values")
    return raw_residual, actual_residual


def apply_bounded_centered_residual(
    base_scores: np.ndarray,
    probe_scores: np.ndarray,
    pose_indices: np.ndarray,
    *,
    alpha: float,
    temperature: float,
    mode: str = "symmetric",
    residual_sign: float = 1.0,
    gate_temperature: float | None = None,
    frontier_center_logit: float | None = None,
    suppression_threshold: float | None = None,
    rescue_threshold: float | None = None,
    coverage_probe_scores: np.ndarray | None = None,
    coverage_threshold: float | None = None,
    coverage_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return corrected probabilities and the actual logit residual.

    ``symmetric`` preserves the original pose-centered behavior.  One-sided
    modes use inclusive train-owned certificate cutoffs and never center the
    residual.  ``dual_rescue`` takes the pointwise maximum of two non-negative
    rescue gates and therefore has no residual sign reversal.
    """

    base = _finite_vector(base_scores, name="base scores")
    raw_residual, actual_residual = _bounded_residual_components(
        base,
        probe_scores,
        pose_indices,
        alpha=alpha,
        temperature=temperature,
        mode=mode,
        residual_sign=residual_sign,
        gate_temperature=gate_temperature,
        frontier_center_logit=frontier_center_logit,
        suppression_threshold=suppression_threshold,
        rescue_threshold=rescue_threshold,
        coverage_probe_scores=coverage_probe_scores,
        coverage_threshold=coverage_threshold,
        coverage_weight=coverage_weight,
    )
    corrected = _sigmoid(_logit(base) + actual_residual)
    if not bool(np.isfinite(corrected).all() and np.isfinite(actual_residual).all()):
        raise FloatingPointError("bounded residual produced non-finite values")
    return corrected, actual_residual


def _residual_group_diagnostic(
    raw: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    *,
    expected_sign: int | None,
) -> dict[str, Any]:
    selected_raw = np.asarray(raw, dtype=np.float64).reshape(-1)[mask]
    selected_actual = np.asarray(actual, dtype=np.float64).reshape(-1)[mask]
    if selected_raw.size == 0:
        return {
            "count": 0,
            "rawMean": None,
            "actualMean": None,
            "rawCorrectDirectionProportion": None,
            "actualCorrectDirectionProportion": None,
        }
    if expected_sign is None:
        raw_direction = None
        actual_direction = None
    elif expected_sign > 0:
        raw_direction = float(np.mean(selected_raw > 0.0))
        actual_direction = float(np.mean(selected_actual > 0.0))
    else:
        raw_direction = float(np.mean(selected_raw < 0.0))
        actual_direction = float(np.mean(selected_actual < 0.0))
    return {
        "count": int(selected_raw.size),
        "rawMean": float(np.mean(selected_raw)),
        "actualMean": float(np.mean(selected_actual)),
        "rawCorrectDirectionProportion": raw_direction,
        "actualCorrectDirectionProportion": actual_direction,
    }


def residual_diagnostics(
    rows: Sequence[ScoreRow],
    probe_scores: Sequence[np.ndarray],
    raw_residuals: Sequence[np.ndarray],
    actual_residuals: Sequence[np.ndarray],
    cutoffs: Mapping[str, Any],
    *,
    cutoff_source: str,
) -> dict[str, Any]:
    """Summarize residual direction in difficult tails and all GT classes."""

    if not (
        len(rows)
        == len(probe_scores)
        == len(raw_residuals)
        == len(actual_residuals)
    ):
        raise ValueError("residual diagnostic rows and arrays are not aligned")
    try:
        low_cutoff = float(cutoffs["lowPositiveScoreCutoff"])
        high_cutoff = float(cutoffs["highNegativeScoreCutoff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("residual diagnostics require both score cutoffs") from exc
    if not np.isfinite(low_cutoff) or not np.isfinite(high_cutoff):
        raise ValueError("residual diagnostic cutoffs must be finite")
    all_scores: list[np.ndarray] = []
    all_raw: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for row, probe, raw, actual in zip(
        rows, probe_scores, raw_residuals, actual_residuals, strict=True
    ):
        if row.targets is None or row.scores is None:
            raise ValueError("residual diagnostics require candidate scores and targets")
        score_values = _finite_vector(row.scores, name="diagnostic base scores")
        probe_values = _finite_vector(probe, name="diagnostic probe scores")
        raw_values = _finite_vector(raw, name="diagnostic raw residuals")
        actual_values = _finite_vector(actual, name="diagnostic actual residuals")
        targets = _finite_vector(row.targets, name="diagnostic targets") > 0.5
        if not (
            score_values.size
            == probe_values.size
            == raw_values.size
            == actual_values.size
            == targets.size
        ):
            raise ValueError(f"pose {row.pose_index} residual diagnostic arrays disagree")
        # Tail membership is defined by the captured base visibility score.
        # The positive and negative class-conditional cutoffs may overlap;
        # that overlap is the failure mode this diagnostic is intended to
        # measure.  Probe scores only determine the applied correction.
        all_scores.append(score_values)
        all_raw.append(raw_values)
        all_actual.append(actual_values)
        all_targets.append(targets)
    scores = np.concatenate(all_scores) if all_scores else np.zeros((0,), dtype=np.float64)
    raw = np.concatenate(all_raw) if all_raw else np.zeros((0,), dtype=np.float64)
    actual = np.concatenate(all_actual) if all_actual else np.zeros((0,), dtype=np.float64)
    targets = np.concatenate(all_targets) if all_targets else np.zeros((0,), dtype=bool)
    low_positive = targets & (scores <= low_cutoff)
    high_negative = (~targets) & (scores >= high_cutoff)
    return {
        "cutoffSource": str(cutoff_source),
        "cutoffs": {
            "sourceSplit": cutoffs.get("sourceSplit"),
            "lowPositiveScoreCutoff": low_cutoff,
            "highNegativeScoreCutoff": high_cutoff,
        },
        "lowScorePositive": _residual_group_diagnostic(
            raw, actual, low_positive, expected_sign=1
        ),
        "highScoreNegative": _residual_group_diagnostic(
            raw, actual, high_negative, expected_sign=-1
        ),
        "fullCandidate": {
            "positive": _residual_group_diagnostic(
                raw, actual, targets, expected_sign=None
            ),
            "negative": _residual_group_diagnostic(
                raw, actual, ~targets, expected_sign=None
            ),
        },
        "correctDirectionDefinition": (
            "low-score-positive residual > 0 and high-score-negative residual < 0; zero is not counted"
        ),
    }


def _extract_probe_scores(
    rows: Sequence[ScoreRow],
    dataset: Any,
    model: Any,
    runtime_features: Any,
    device: Any,
    probe: ProbeSpec,
    batch_size: int,
) -> list[np.ndarray]:
    """Query one train-owned probe; capture scores remain authoritative."""

    return _extract_probe_score_sets(
        rows,
        dataset,
        model,
        runtime_features,
        device,
        {"primary": probe},
        batch_size,
    )["primary"]


def _extract_probe_score_sets(
    rows: Sequence[ScoreRow],
    dataset: Any,
    model: Any,
    runtime_features: Any,
    device: Any,
    probes: Mapping[str, ProbeSpec],
    batch_size: int,
) -> dict[str, list[np.ndarray]]:
    """Extract several probe scores while rebuilding each family once per batch."""

    if torch is None:
        raise RuntimeError("feature extraction requires PyTorch")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(probes, Mapping) or not probes:
        raise ValueError("probes must be a non-empty mapping")
    output: dict[str, list[np.ndarray]] = {str(name): [] for name in probes}
    model.eval()
    with torch.no_grad():
        for row in rows:
            if row.scores is None:
                raise ValueError(f"pose {row.pose_index} has no authoritative candidate scores")
            if row.candidate_ids.size == 0:
                for name in output:
                    output[name].append(np.zeros((0,), dtype=np.float64))
                continue
            camera_norm, candidate_camera, query_center, camera_view, radius = _row_dataset_inputs(
                dataset, row
            )
            chunks: dict[str, list[np.ndarray]] = {str(name): [] for name in probes}
            for start in range(0, row.candidate_ids.size, int(batch_size)):
                chosen = row.candidate_ids[start : start + int(batch_size)]
                _model_logits, aux = model.compute_logits_with_aux(
                    torch.from_numpy(camera_norm).to(device),
                    torch.from_numpy(camera_view).to(device),
                    torch.from_numpy(candidate_camera).to(device),
                    torch.from_numpy(chosen.astype(np.int64)).to(device),
                    runtime_features=runtime_features,
                    query_center_world=torch.from_numpy(query_center).to(device),
                    viewcell_radius_m=torch.tensor(radius, dtype=torch.float32, device=device),
                )
                batch_scores = score_probe_families_from_aux(aux, probes)
                for name, values in batch_scores.items():
                    chunks[name].append(values)
            for name, values in chunks.items():
                combined = np.concatenate(values) if values else np.zeros((0,), dtype=np.float64)
                if combined.size != row.scores.size:
                    raise ValueError(
                        f"pose {row.pose_index} extracted feature rows disagree with scores"
                    )
                output[name].append(combined)
    return output


def _pose_metrics(
    row: ScoreRow,
    predicted: np.ndarray,
    instance_to_glb: np.ndarray | None = None,
    glb_bytes: np.ndarray | None = None,
) -> dict[str, Any]:
    if row.targets is None or row.visible_weights is None or row.scores is None:
        raise ValueError("score row is incomplete")
    target = np.asarray(row.targets, dtype=np.float64).reshape(-1) > 0.5
    weights = _finite_vector(row.visible_weights, name="visible weights")
    ids = np.asarray(row.candidate_ids, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted, dtype=bool).reshape(-1)
    if not (target.size == weights.size == ids.size == predicted.size == row.scores.size):
        raise ValueError(f"pose {row.pose_index} arrays are not candidate-aligned")
    if bool(np.any(weights < 0.0)):
        raise ValueError("visible weights must be non-negative")
    tp = int(np.logical_and(predicted, target).sum())
    fp = int(np.logical_and(predicted, ~target).sum())
    fn = int(np.logical_and(~predicted, target).sum())
    tn = int(np.logical_and(~predicted, ~target).sum())
    weighted_tp = float(weights[np.logical_and(predicted, target)].sum())
    weighted_gt = float(weights[target].sum())
    recall = _safe_div(tp, tp + fn, 1.0)
    precision = _safe_div(tp, tp + fp, 1.0)
    specificity = _safe_div(tn, tn + fp, 1.0)
    result: dict[str, Any] = {
        "poseIndex": int(row.pose_index),
        "candidateCount": int(ids.size),
        "gtCount": int(target.sum()),
        "predCount": int(predicted.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "weightedRecall": _safe_div(weighted_tp, weighted_gt, 1.0),
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "jaccard": _safe_div(tp, tp + fp + fn),
        "accuracy": _safe_div(tp + tn, ids.size, 1.0),
        "specificity": specificity,
        "balancedAccuracy": 0.5 * (recall + specificity),
        "usefulCull": _safe_div(tn, ids.size),
        "badCull": _safe_div(fn, ids.size),
        "weightedTp": weighted_tp,
        "weightedGt": weighted_gt,
        "predOverCandidate": _safe_div(predicted.sum(), ids.size),
        "predOverGt": _safe_div(predicted.sum(), int(target.sum())),
    }
    if instance_to_glb is None or glb_bytes is None:
        for key in (
            "candidateGlbCount", "predictedGlbCount", "requiredGlbCount",
            "candidateGlbBytes", "predictedGlbBytes", "requiredGlbBytes",
            "glbCountReduction", "glbByteReduction", "downloadUtilityRecall",
        ):
            result[key] = None
        return result
    mapping = np.asarray(instance_to_glb, dtype=np.int64).reshape(-1)
    costs = np.asarray(glb_bytes, dtype=np.float64).reshape(-1)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= mapping.size):
        raise ValueError("candidate instance id is outside instance_to_glb")
    if not bool(np.isfinite(costs).all()) or bool(np.any(costs < 0.0)):
        raise ValueError("glb bytes must be finite and non-negative")
    candidate_glbs = np.unique(mapping[ids]) if ids.size else np.zeros((0,), dtype=np.int64)
    predicted_glbs = np.unique(mapping[ids[predicted]]) if bool(predicted.any()) else np.zeros((0,), dtype=np.int64)
    visible_ids = ids[target]
    visible_glbs = np.unique(mapping[visible_ids]) if visible_ids.size else np.zeros((0,), dtype=np.int64)
    candidate_bytes = float(costs[candidate_glbs].sum()) if candidate_glbs.size else 0.0
    predicted_bytes = float(costs[predicted_glbs].sum()) if predicted_glbs.size else 0.0
    required_bytes = float(costs[visible_glbs].sum()) if visible_glbs.size else 0.0
    predicted_glb_mask = (
        np.isin(mapping[ids], predicted_glbs) if predicted_glbs.size else np.zeros(ids.size, dtype=bool)
    )
    download_utility = float(weights[np.logical_and(target, predicted_glb_mask)].sum())
    result.update(
        {
            "candidateGlbCount": int(candidate_glbs.size),
            "predictedGlbCount": int(predicted_glbs.size),
            "requiredGlbCount": int(visible_glbs.size),
            "candidateGlbBytes": candidate_bytes,
            "predictedGlbBytes": predicted_bytes,
            "requiredGlbBytes": required_bytes,
            "glbCountReduction": 1.0 - _safe_div(predicted_glbs.size, candidate_glbs.size),
            "glbByteReduction": 1.0 - _safe_div(predicted_bytes, candidate_bytes),
            "downloadUtilityRecall": _safe_div(download_utility, weighted_gt, 1.0),
        }
    )
    return result


def summarize_threshold(
    rows: Sequence[ScoreRow],
    corrected_scores: Sequence[np.ndarray],
    threshold: float,
    *,
    instance_to_glb: np.ndarray | None = None,
    glb_bytes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return pose-macro and aggregate metrics at one frozen threshold."""

    if len(rows) != len(corrected_scores) or not rows:
        raise ValueError("rows and corrected scores are not aligned or empty")
    pose_rows = [
        _pose_metrics(
            row,
            np.asarray(scores, dtype=np.float64) >= float(threshold),
            instance_to_glb,
            glb_bytes,
        )
        for row, scores in zip(rows, corrected_scores, strict=True)
    ]
    metric_names = (
        "precision", "recall", "weightedRecall", "f1", "jaccard", "accuracy",
        "balancedAccuracy", "specificity", "usefulCull", "badCull", "predCount",
        "candidateCount", "gtCount", "tp", "fp", "fn", "tn", "predOverCandidate",
        "predOverGt",
    )
    macro = {name: float(np.mean([float(row[name]) for row in pose_rows])) for name in metric_names}
    tp = int(sum(int(row["tp"]) for row in pose_rows))
    fp = int(sum(int(row["fp"]) for row in pose_rows))
    fn = int(sum(int(row["fn"]) for row in pose_rows))
    tn = int(sum(int(row["tn"]) for row in pose_rows))
    candidate_count = tp + fp + fn + tn
    gt_count = tp + fn
    pred_count = tp + fp
    aggregate_recall = _safe_div(tp, gt_count, 1.0)
    aggregate_precision = _safe_div(tp, tp + fp, 1.0)
    aggregate_specificity = _safe_div(tn, tn + fp, 1.0)
    weighted_tp = float(sum(float(row["weightedTp"]) for row in pose_rows))
    weighted_gt = float(sum(float(row["weightedGt"]) for row in pose_rows))
    aggregate: dict[str, Any] = {
        "precision": aggregate_precision,
        "recall": aggregate_recall,
        "weightedRecall": _safe_div(weighted_tp, weighted_gt, 1.0),
        "f1": _safe_div(2.0 * aggregate_precision * aggregate_recall, aggregate_precision + aggregate_recall),
        "jaccard": _safe_div(tp, tp + fp + fn),
        "accuracy": _safe_div(tp + tn, candidate_count, 1.0),
        "specificity": aggregate_specificity,
        "balancedAccuracy": 0.5 * (aggregate_recall + aggregate_specificity),
        "usefulCull": _safe_div(tn, candidate_count),
        "badCull": _safe_div(fn, candidate_count),
        "avgPredCount": _safe_div(pred_count, len(pose_rows)),
        "avgCandidateCount": _safe_div(candidate_count, len(pose_rows)),
        "avgGtCount": _safe_div(gt_count, len(pose_rows)),
        "avgFpCount": _safe_div(fp, len(pose_rows)),
        "avgFnCount": _safe_div(fn, len(pose_rows)),
        "avgTnCount": _safe_div(tn, len(pose_rows)),
        "avgTpCount": _safe_div(tp, len(pose_rows)),
        "predOverCandidate": _safe_div(pred_count, candidate_count),
        "predOverGt": _safe_div(pred_count, gt_count),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "candidateCount": candidate_count,
        "gtCount": gt_count,
        "predCount": pred_count,
        "weightedTp": weighted_tp,
        "weightedGt": weighted_gt,
    }
    for key in (
        "candidateGlbCount", "predictedGlbCount", "requiredGlbCount",
        "candidateGlbBytes", "predictedGlbBytes", "requiredGlbBytes",
        "glbCountReduction", "glbByteReduction", "downloadUtilityRecall",
    ):
        values = [row[key] for row in pose_rows]
        aggregate[key] = None if any(value is None for value in values) else float(np.mean(values))
        macro[key] = aggregate[key]
    return {
        "threshold": float(threshold),
        "poseCount": len(pose_rows),
        "poseMacro": macro,
        "aggregate": aggregate,
        "_weightedTpByPose": np.asarray([row["weightedTp"] for row in pose_rows], dtype=np.float64),
        "_weightedGtByPose": np.asarray([row["weightedGt"] for row in pose_rows], dtype=np.float64),
    }


def freeze_calibration_workpoint(
    rows: Sequence[ScoreRow],
    corrected_scores: Sequence[np.ndarray],
    *,
    bootstrap_replicates: int = 10000,
    seed: int = 20260818,
) -> dict[str, Any]:
    """Choose a safe threshold using calibration only."""

    if int(bootstrap_replicates) < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    threshold_rows: list[dict[str, Any]] = []
    bootstrap_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    for index, threshold in enumerate(threshold_grid().astype(np.float64)):
        summary = summarize_threshold(rows, corrected_scores, float(threshold))
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "weightedRecall": float(summary["aggregate"]["weightedRecall"]),
                "weightedRecallLowerConfidenceBound": None,
                "bootstrapEvaluated": False,
                "usefulCull": float(summary["aggregate"]["usefulCull"]),
                "badCull": float(summary["aggregate"]["badCull"]),
                "precision": float(summary["aggregate"]["precision"]),
                "balancedAccuracy": float(summary["aggregate"]["balancedAccuracy"]),
                "avgPredCount": float(summary["aggregate"]["avgPredCount"]),
            }
        )
        bootstrap_inputs.append(
            (
                summary["_weightedTpByPose"],
                summary["_weightedGtByPose"],
            )
        )

    # Bootstrap only thresholds that can still win the registered ranking.
    # Weighted recall and its pose-bootstrap lower bound are monotone as the
    # threshold is lowered, so the first qualified row in useful-cull order is
    # exactly the same choice as bootstrapping all 142 grid values.
    point_eligible = [
        (index, row)
        for index, row in enumerate(threshold_rows)
        if row["weightedRecall"] > WEIGHTED_RECALL_FLOOR
    ]
    point_eligible.sort(
        key=lambda item: (
            item[1]["usefulCull"],
            item[1]["precision"],
            item[1]["balancedAccuracy"],
            -item[1]["avgPredCount"],
            item[1]["threshold"],
        ),
        reverse=True,
    )
    selected = None
    for index, row in point_eligible:
        weighted_tp, weighted_gt = bootstrap_inputs[index]
        lcb = weighted_recall_lower_confidence_bound(
            weighted_tp,
            weighted_gt,
            replicates=int(bootstrap_replicates),
            seed=int(seed) + index,
        )
        row["weightedRecallLowerConfidenceBound"] = float(lcb)
        row["bootstrapEvaluated"] = True
        if float(lcb) > WEIGHTED_RECALL_FLOOR:
            selected = dict(row)
            break
    return {
        "status": "safe" if selected is not None else "no_qualified_safety_workpoint",
        "selected": None if selected is None else dict(selected),
        "selectionSplit": "calibration",
        "selectedFromValidation": False,
        "weightedRecallFloor": WEIGHTED_RECALL_FLOOR,
        "weightedRecallLcbFloor": WEIGHTED_RECALL_FLOOR,
        "bootstrapReplicates": int(bootstrap_replicates),
        "bootstrapLowerQuantile": BOOTSTRAP_CONFIDENCE_QUANTILE,
        "thresholdRows": threshold_rows,
        "testRead": False,
    }


def _flatten_probe_data(split: CandidateSplit) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split.probe_scores is None or len(split.probe_scores) != len(split.rows):
        raise ValueError(f"{split.split} has no candidate-aligned probe scores")
    base: list[np.ndarray] = []
    probe: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for row, values in zip(split.rows, split.probe_scores, strict=True):
        if row.scores is None:
            raise ValueError(f"{split.split} pose {row.pose_index} has no scores")
        base.append(_finite_vector(row.scores, name=f"{split.split}/scores"))
        probe.append(_finite_vector(values, name=f"{split.split}/probeScores"))
        poses.append(np.full(row.scores.size, int(row.pose_index), dtype=np.int64))
    return (
        np.concatenate(base) if base else np.zeros((0,), dtype=np.float64),
        np.concatenate(probe) if probe else np.zeros((0,), dtype=np.float64),
        np.concatenate(poses) if poses else np.zeros((0,), dtype=np.int64),
        np.asarray([int(row.pose_index) for row in split.rows], dtype=np.int64),
    )


def _corrected_scores_by_pose(
    split: CandidateSplit,
    alpha: float,
    temperature: float,
    *,
    mode: str = "symmetric",
    residual_sign: float,
    gate_temperature: float | None,
    frontier_center_logit: float | None,
    suppression_threshold: float | None = None,
    rescue_threshold: float | None = None,
    coverage_threshold: float | None = None,
    coverage_weight: float = 1.0,
) -> list[np.ndarray]:
    corrected, _raw, _actual = _residual_components_by_pose(
        split,
        alpha,
        temperature,
        mode=mode,
        residual_sign=residual_sign,
        gate_temperature=gate_temperature,
        frontier_center_logit=frontier_center_logit,
        suppression_threshold=suppression_threshold,
        rescue_threshold=rescue_threshold,
        coverage_threshold=coverage_threshold,
        coverage_weight=coverage_weight,
    )
    return corrected


def _residual_components_by_pose(
    split: CandidateSplit,
    alpha: float,
    temperature: float,
    *,
    mode: str = "symmetric",
    residual_sign: float,
    gate_temperature: float | None,
    frontier_center_logit: float | None,
    suppression_threshold: float | None = None,
    rescue_threshold: float | None = None,
    coverage_threshold: float | None = None,
    coverage_weight: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return corrected scores plus raw and actual residuals per pose."""

    if split.probe_scores is None:
        raise ValueError(f"{split.split} has no probe scores")
    if mode == "dual_rescue":
        if split.coverage_probe_scores is None:
            raise ValueError(f"{split.split} has no coverage probe scores")
        if len(split.coverage_probe_scores) != len(split.rows):
            raise ValueError(f"{split.split} coverage probe scores are not pose-aligned")
    result: list[np.ndarray] = []
    raw_result: list[np.ndarray] = []
    actual_result: list[np.ndarray] = []
    coverage_values_iter = (
        split.coverage_probe_scores
        if mode == "dual_rescue"
        else [None] * len(split.rows)
    )
    for row, probe_values, coverage_values in zip(
        split.rows,
        split.probe_scores,
        coverage_values_iter,
        strict=True,
    ):
        if row.scores is None:
            raise ValueError(f"{split.split} pose {row.pose_index} has no scores")
        pose_ids = np.full(row.scores.size, int(row.pose_index), dtype=np.int64)
        raw_residual, actual_residual = _bounded_residual_components(
            row.scores,
            probe_values,
            pose_ids,
            alpha=alpha,
            temperature=temperature,
            mode=mode,
            residual_sign=residual_sign,
            gate_temperature=gate_temperature,
            frontier_center_logit=frontier_center_logit,
            suppression_threshold=suppression_threshold,
            rescue_threshold=rescue_threshold,
            coverage_probe_scores=coverage_values,
            coverage_threshold=coverage_threshold,
            coverage_weight=coverage_weight,
        )
        result.append(_sigmoid(_logit(row.scores) + actual_residual))
        raw_result.append(raw_residual)
        actual_result.append(actual_residual)
    return result, raw_result, actual_result


def _dual_rescue_branch_components_by_pose(
    split: CandidateSplit,
    alpha: float,
    temperature: float,
    *,
    primary_threshold: float,
    coverage_threshold: float,
    coverage_weight: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return the two uncentered rescue branches for diagnostics only."""

    if split.probe_scores is None or split.coverage_probe_scores is None:
        raise ValueError(f"{split.split} has no dual-rescue probe scores")
    if len(split.probe_scores) != len(split.rows) or len(split.coverage_probe_scores) != len(
        split.rows
    ):
        raise ValueError(f"{split.split} dual-rescue probe scores are not pose-aligned")
    if not np.isfinite(alpha) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not np.isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(primary_threshold) or not np.isfinite(coverage_threshold):
        raise ValueError("dual-rescue thresholds must be finite")
    if not np.isfinite(coverage_weight) or float(coverage_weight) < 0.0:
        raise ValueError("coverage_weight must be finite and non-negative")
    primary_result: list[np.ndarray] = []
    coverage_result: list[np.ndarray] = []
    for primary_values, coverage_values in zip(
        split.probe_scores,
        split.coverage_probe_scores,
        strict=True,
    ):
        primary = _finite_vector(primary_values, name="primary probe scores")
        coverage = _finite_vector(coverage_values, name="coverage probe scores")
        if primary.size != coverage.size:
            raise ValueError("primary and coverage probe scores are not aligned")
        primary_gate = np.maximum(
            0.0,
            np.tanh((float(primary_threshold) - primary) / float(temperature)),
        )
        coverage_gate = np.maximum(
            0.0,
            np.tanh((float(coverage_threshold) - coverage) / float(temperature)),
        )
        primary_result.append(float(alpha) * primary_gate)
        coverage_result.append(float(alpha) * float(coverage_weight) * coverage_gate)
    return primary_result, coverage_result


def _probe_residual_diagnostics(
    rows: Sequence[ScoreRow],
    probe_scores: Sequence[np.ndarray] | None,
    raw_residuals: Sequence[np.ndarray],
    actual_residuals: Sequence[np.ndarray],
    train_fit_cutoffs: Mapping[str, Any] | None,
    calibration_cutoffs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the two report-only cutoff diagnostics for one probe."""

    if probe_scores is None:
        raise ValueError("probe residual diagnostics require probe scores")
    result: dict[str, Any] = {}
    if train_fit_cutoffs is not None:
        result["trainFitCutoffs"] = residual_diagnostics(
            rows,
            probe_scores,
            raw_residuals,
            actual_residuals,
            train_fit_cutoffs,
            cutoff_source="trainFitCutoffs",
        )
    if calibration_cutoffs is not None:
        result["calibrationCutoffs"] = residual_diagnostics(
            rows,
            probe_scores,
            raw_residuals,
            actual_residuals,
            calibration_cutoffs,
            cutoff_source="calibration_cutoffs_report_only",
        )
    return result


def scan_members(
    calibration: CandidateSplit,
    validation: CandidateSplit,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    gate_temperatures: Sequence[float | None] = DEFAULT_GATE_TEMPERATURES,
    residual_signs: Sequence[float] = DEFAULT_RESIDUAL_SIGNS,
    residual_modes: Sequence[str] = DEFAULT_RESIDUAL_MODES,
    frontier_center_logit: float | None = None,
    suppression_threshold: float | None = None,
    rescue_threshold: float | None = None,
    coverage_rescue_threshold: float | None = None,
    suppression_risk: float = DEFAULT_SUPPRESSION_RISK,
    rescue_risk: float = DEFAULT_RESCUE_RISK,
    coverage_rescue_risk: float = DEFAULT_RESCUE_RISK,
    coverage_rescue_weights: Sequence[float] = (1.0,),
    train_fit_cutoffs: Mapping[str, Any] | None = None,
    calibration_cutoffs: Mapping[str, Any] | None = None,
    coverage_train_fit_cutoffs: Mapping[str, Any] | None = None,
    coverage_calibration_cutoffs: Mapping[str, Any] | None = None,
    bootstrap_replicates: int = 10000,
    seed: int = 20260818,
    instance_to_glb: np.ndarray | None = None,
    glb_bytes: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Scan members; every validation threshold comes from its own calibration."""

    if calibration.split != "calibration" or validation.split != "validation":
        raise ValueError("scan_members requires calibration and validation splits")
    normalized_modes = tuple(str(value) for value in residual_modes)
    if not normalized_modes or any(value not in SUPPORTED_RESIDUAL_MODES for value in normalized_modes):
        raise ValueError(
            f"residual_modes must be drawn from {SUPPORTED_RESIDUAL_MODES}"
        )
    normalized_signs = tuple(float(value) for value in residual_signs)
    if not normalized_signs or any(not np.isclose(abs(value), 1.0) for value in normalized_signs):
        raise ValueError("residual_signs must contain +1 and/or -1")
    normalized_gates = tuple(
        None if value is None or float(value) == 0.0 else float(value)
        for value in gate_temperatures
    )
    if not normalized_gates or any(value is not None and (not np.isfinite(value) or value <= 0.0) for value in normalized_gates):
        raise ValueError("gate_temperatures must contain positive values or None/0")
    if any(value is not None for value in normalized_gates) and frontier_center_logit is None:
        raise ValueError("an active gate requires the train-owned frontier center")
    if not np.isfinite(suppression_risk) or float(suppression_risk) < 0.0:
        raise ValueError("suppression_risk must be finite and non-negative")
    if not np.isfinite(rescue_risk) or float(rescue_risk) < 0.0:
        raise ValueError("rescue_risk must be finite and non-negative")
    normalized_coverage_weights = tuple(float(value) for value in coverage_rescue_weights)
    if not normalized_coverage_weights or any(
        not np.isfinite(value) or value < 0.0 for value in normalized_coverage_weights
    ):
        raise ValueError("coverage_rescue_weights must contain finite non-negative values")
    if not np.isfinite(coverage_rescue_risk) or float(coverage_rescue_risk) < 0.0:
        raise ValueError("coverage_rescue_risk must be finite and non-negative")
    if any(value in {"suppress_only", "dual"} for value in normalized_modes):
        if suppression_threshold is None or not np.isfinite(suppression_threshold):
            raise ValueError("suppressing modes require a train-only suppression threshold")
    if any(value in {"rescue_only", "dual", "dual_rescue"} for value in normalized_modes):
        if rescue_threshold is None or not np.isfinite(rescue_threshold):
            raise ValueError("rescuing modes require a train-only rescue threshold")
    if "dual_rescue" in normalized_modes:
        if coverage_rescue_threshold is None or not np.isfinite(coverage_rescue_threshold):
            raise ValueError("dual_rescue requires a train-only coverage rescue threshold")
        if validation.coverage_probe_scores is None or calibration.coverage_probe_scores is None:
            raise ValueError("dual_rescue requires coverage probe scores for both splits")
    members: list[dict[str, Any]] = []
    for mode in normalized_modes:
        mode_signs = normalized_signs if mode == "symmetric" else (1.0,)
        mode_weights = normalized_coverage_weights if mode == "dual_rescue" else (None,)
        for alpha in alphas:
            for temperature in temperatures:
                for gate_temperature in normalized_gates:
                    for residual_sign in mode_signs:
                        for coverage_weight in mode_weights:
                            coverage_weight_value = (
                                1.0 if coverage_weight is None else float(coverage_weight)
                            )
                            corrected_calibration, raw_calibration, actual_calibration = (
                                _residual_components_by_pose(
                                    calibration,
                                    float(alpha),
                                    float(temperature),
                                    mode=mode,
                                    residual_sign=float(residual_sign),
                                    gate_temperature=gate_temperature,
                                    frontier_center_logit=frontier_center_logit,
                                    suppression_threshold=suppression_threshold,
                                    rescue_threshold=rescue_threshold,
                                    coverage_threshold=coverage_rescue_threshold,
                                    coverage_weight=coverage_weight_value,
                                )
                            )
                            workpoint = freeze_calibration_workpoint(
                                calibration.rows,
                                corrected_calibration,
                                bootstrap_replicates=int(bootstrap_replicates),
                                # All members use the same bootstrap seed.  The
                                # existing winner-only LCB optimization remains
                                # inside freeze_calibration_workpoint.
                                seed=int(seed),
                            )
                            selected = workpoint.get("selected")
                            corrected_validation, raw_validation, actual_validation = (
                                _residual_components_by_pose(
                                    validation,
                                    float(alpha),
                                    float(temperature),
                                    mode=mode,
                                    residual_sign=float(residual_sign),
                                    gate_temperature=gate_temperature,
                                    frontier_center_logit=frontier_center_logit,
                                    suppression_threshold=suppression_threshold,
                                    rescue_threshold=rescue_threshold,
                                    coverage_threshold=coverage_rescue_threshold,
                                    coverage_weight=coverage_weight_value,
                                )
                            )
                            validation_summary = None
                            if isinstance(selected, Mapping):
                                validation_summary = summarize_threshold(
                                    validation.rows,
                                    corrected_validation,
                                    float(selected["threshold"]),
                                    instance_to_glb=instance_to_glb,
                                    glb_bytes=glb_bytes,
                                )
                                validation_lcb = weighted_recall_lower_confidence_bound(
                                    validation_summary["_weightedTpByPose"],
                                    validation_summary["_weightedGtByPose"],
                                    replicates=int(bootstrap_replicates),
                                    seed=int(seed),
                                )
                                validation_summary["weightedRecallLowerConfidenceBound"] = float(
                                    validation_lcb
                                )
                                validation_summary.pop("_weightedTpByPose", None)
                                validation_summary.pop("_weightedGtByPose", None)

                            if mode == "dual_rescue":
                                primary_calibration_branch, coverage_calibration_branch = (
                                    _dual_rescue_branch_components_by_pose(
                                        calibration,
                                        float(alpha),
                                        float(temperature),
                                        primary_threshold=float(rescue_threshold),
                                        coverage_threshold=float(coverage_rescue_threshold),
                                        coverage_weight=coverage_weight_value,
                                    )
                                )
                                primary_validation_branch, coverage_validation_branch = (
                                    _dual_rescue_branch_components_by_pose(
                                        validation,
                                        float(alpha),
                                        float(temperature),
                                        primary_threshold=float(rescue_threshold),
                                        coverage_threshold=float(coverage_rescue_threshold),
                                        coverage_weight=coverage_weight_value,
                                    )
                                )
                                calibration_diagnostics = {
                                    "primaryProbe": _probe_residual_diagnostics(
                                        calibration.rows,
                                        calibration.probe_scores,
                                        primary_calibration_branch,
                                        primary_calibration_branch,
                                        train_fit_cutoffs,
                                        calibration_cutoffs,
                                    ),
                                    "coverageProbe": _probe_residual_diagnostics(
                                        calibration.rows,
                                        calibration.coverage_probe_scores,
                                        coverage_calibration_branch,
                                        coverage_calibration_branch,
                                        coverage_train_fit_cutoffs,
                                        coverage_calibration_cutoffs,
                                    ),
                                }
                                validation_diagnostics = {
                                    "primaryProbe": _probe_residual_diagnostics(
                                        validation.rows,
                                        validation.probe_scores,
                                        primary_validation_branch,
                                        primary_validation_branch,
                                        train_fit_cutoffs,
                                        calibration_cutoffs,
                                    ),
                                    "coverageProbe": _probe_residual_diagnostics(
                                        validation.rows,
                                        validation.coverage_probe_scores,
                                        coverage_validation_branch,
                                        coverage_validation_branch,
                                        coverage_train_fit_cutoffs,
                                        coverage_calibration_cutoffs,
                                    ),
                                }
                            else:
                                validation_diagnostics = None
                                calibration_diagnostics = None
                                if train_fit_cutoffs is not None or calibration_cutoffs is not None:
                                    validation_diagnostics = _probe_residual_diagnostics(
                                        validation.rows,
                                        validation.probe_scores,
                                        raw_validation,
                                        actual_validation,
                                        train_fit_cutoffs,
                                        calibration_cutoffs,
                                    )
                                    calibration_diagnostics = _probe_residual_diagnostics(
                                        calibration.rows,
                                        calibration.probe_scores,
                                        raw_calibration,
                                        actual_calibration,
                                        train_fit_cutoffs,
                                        calibration_cutoffs,
                                    )
                            gate_label = "none" if gate_temperature is None else f"{float(gate_temperature):g}"
                            member_id = (
                                f"{mode}_alpha{float(alpha):g}_temperature{float(temperature):g}"
                                f"_gate{gate_label}"
                            )
                            if mode == "symmetric":
                                member_id += f"_sign{float(residual_sign):g}"
                            if mode == "dual_rescue":
                                member_id += f"_coverageWeight{coverage_weight_value:g}"
                            certificate: dict[str, Any] = {
                                "source": "probe JSON train-only risk certificate"
                                if mode != "symmetric"
                                else "not used by symmetric mode",
                                "thresholdSource": (
                                    "train_probe_risk_certificate"
                                    if mode != "symmetric"
                                    else None
                                ),
                                "suppressionRisk": float(suppression_risk),
                                "rescueRisk": float(rescue_risk),
                                "suppressionThreshold": suppression_threshold,
                                "rescueThreshold": rescue_threshold,
                            }
                            if mode == "dual_rescue":
                                certificate.update(
                                    {
                                        "primary": {
                                            "source": "primary probe JSON train-only rescue certificate",
                                            "thresholdSource": "train_probe_risk_certificate",
                                            "risk": float(rescue_risk),
                                            "threshold": float(rescue_threshold),
                                            "role": "high_visual_weight_low_score_positive_rescue",
                                        },
                                        "coverage": {
                                            "source": "coverage probe JSON train-only rescue certificate",
                                            "thresholdSource": "coverage_probe_train_risk_certificate",
                                            "risk": float(coverage_rescue_risk),
                                            "threshold": float(coverage_rescue_threshold),
                                            "role": "ordinary_low_score_positive_rescue",
                                        },
                                        "coverageRescueRisk": float(coverage_rescue_risk),
                                        "coverageRescueThreshold": float(coverage_rescue_threshold),
                                    }
                                )
                            members.append(
                                {
                                    "memberId": member_id,
                                    "mode": mode,
                                    "alpha": float(alpha),
                                    "temperature": float(temperature),
                                    "gateTemperature": gate_temperature,
                                    "residualSign": float(residual_sign),
                                    "coverageWeight": (
                                        coverage_weight_value if mode == "dual_rescue" else None
                                    ),
                                    "certificate": certificate,
                                    "residual": {
                                        "mode": mode,
                                        "formula": (
                                            "sign * alpha * tanh((0.5 - train_probe_score) / temperature) * gate"
                                            if mode == "symmetric"
                                            else (
                                                "alpha * max(primary_rescue_gate, coverageWeight * coverage_rescue_gate)"
                                                if mode == "dual_rescue"
                                                else "alpha * (positive rescue gate + negative suppression gate) * gate"
                                            )
                                        ),
                                        "probeMidpoint": 0.5 if mode == "symmetric" else None,
                                        "probeMidpointSemantics": (
                                            "train 0/1 ridge-target midpoint, not a probability threshold"
                                            if mode == "symmetric"
                                            else None
                                        ),
                                        "gate": (
                                            "none"
                                            if gate_temperature is None
                                            else "exp(-abs(base_logit - train_fit_frontier_center_logit) / gate_temperature)"
                                        ),
                                        "frontierCenterLogit": frontier_center_logit,
                                        "boundedBy": "alpha before any symmetric pose centering",
                                        "poseCentering": (
                                            "subtract mean raw residual over every candidate in each pose"
                                            if mode == "symmetric"
                                            else "disabled for certificate sign preservation"
                                        ),
                                        "probeLabel": PROBE_LABEL_SEMANTICS,
                                        "coverageWeight": (
                                            coverage_weight_value if mode == "dual_rescue" else None
                                        ),
                                    },
                                    "calibration": workpoint,
                                    "validation": validation_summary,
                                    "residualDiagnostics": {
                                        "calibration": calibration_diagnostics,
                                        "validation": validation_diagnostics,
                                    },
                                    "testRead": False,
                                }
                            )
    return members


def _load_glb_bytes(index_path: Path, root: Path, num_glbs: int) -> np.ndarray:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("GLB index must contain an object")
    values = np.zeros((int(num_glbs),), dtype=np.float64)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("GLB index has no entries list")
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        gid = int(entry.get("globalId", -1))
        if 0 <= gid < values.size:
            asset = Path(root) / str(entry.get("path", ""))
            if asset.is_file():
                values[gid] = float(asset.stat().st_size)
    if values.size and bool(np.any(values <= 0.0)):
        raise FileNotFoundError("GLB index/root does not provide every requested GLB")
    return values


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-capture", type=Path, required=True)
    parser.add_argument("--calibration-capture", type=Path, required=True)
    parser.add_argument("--validation-capture", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-family", choices=SUPPORTED_FAMILIES, default="combined")
    parser.add_argument("--coverage-probe", type=Path, default=None)
    parser.add_argument("--coverage-probe-family", choices=SUPPORTED_FAMILIES, default="combined")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--initial-geo-features", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, default=None)
    parser.add_argument("--glb-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS))
    parser.add_argument("--temperatures", default=",".join(str(value) for value in DEFAULT_TEMPERATURES))
    parser.add_argument("--gate-temperatures", default="none")
    parser.add_argument("--residual-signs", default="1,-1")
    parser.add_argument(
        "--residual-modes",
        default=",".join(DEFAULT_RESIDUAL_MODES),
        help="comma-separated symmetric,rescue_only,suppress_only,dual,dual_rescue",
    )
    parser.add_argument("--suppression-risk", type=float, default=DEFAULT_SUPPRESSION_RISK)
    parser.add_argument("--rescue-risk", type=float, default=DEFAULT_RESCUE_RISK)
    parser.add_argument("--coverage-rescue-risk", type=float, default=DEFAULT_RESCUE_RISK)
    parser.add_argument("--coverage-rescue-weight", default="1")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.batch_size) <= 0 or int(args.bootstrap_replicates) < 0:
        raise ValueError("batch-size must be positive and bootstrap-replicates must be non-negative")
    alphas = _parse_float_list(args.alphas, name="alphas", minimum=0.0, strict=False)
    temperatures = _parse_float_list(args.temperatures, name="temperatures", minimum=0.0, strict=True)
    gate_temperatures = _parse_gate_temperature_list(args.gate_temperatures)
    residual_signs = _parse_residual_signs(args.residual_signs)
    coverage_rescue_weights = _parse_float_list(
        args.coverage_rescue_weight,
        name="coverage-rescue-weight",
        minimum=0.0,
        strict=False,
    )
    residual_modes = tuple(
        dict.fromkeys(piece.strip() for piece in str(args.residual_modes).split(",") if piece.strip())
    )
    if not residual_modes or any(mode not in SUPPORTED_RESIDUAL_MODES for mode in residual_modes):
        raise ValueError(f"residual-modes must be drawn from {SUPPORTED_RESIDUAL_MODES}")
    if (
        not np.isfinite(args.suppression_risk)
        or float(args.suppression_risk) < 0.0
        or not np.isfinite(args.rescue_risk)
        or float(args.rescue_risk) < 0.0
        or not np.isfinite(args.coverage_rescue_risk)
        or float(args.coverage_rescue_risk) < 0.0
    ):
        raise ValueError("certificate risks must be finite and non-negative")
    if (args.glb_index is None) != (args.glb_root is None):
        raise ValueError("--glb-index and --glb-root must be supplied together")
    dual_rescue_requested = "dual_rescue" in residual_modes
    if dual_rescue_requested and args.coverage_probe is None:
        raise ValueError("--coverage-probe is required when residual mode=dual_rescue")
    probe, probe_payload = load_probe_spec(args.probe.resolve(), args.probe_family)
    coverage_probe = None
    coverage_probe_payload: dict[str, Any] | None = None
    if dual_rescue_requested:
        coverage_probe, coverage_probe_payload = load_probe_spec(
            args.coverage_probe.resolve(), args.coverage_probe_family
        )
    suppression_threshold = None
    rescue_threshold = None
    if any(mode in {"suppress_only", "dual"} for mode in residual_modes):
        suppression_threshold = certificate_threshold_from_probe(
            probe, kind="suppression", risk=float(args.suppression_risk)
        )
    if any(mode in {"rescue_only", "dual", "dual_rescue"} for mode in residual_modes):
        rescue_threshold = certificate_threshold_from_probe(
            probe, kind="rescue", risk=float(args.rescue_risk)
        )
    tail_definition = probe_payload.get("tailDefinition")
    if not isinstance(tail_definition, Mapping):
        raise ValueError("probe has no tailDefinition for residual diagnostics")
    train_fit_cutoffs = tail_definition.get("fitCutoffs")
    calibration_cutoffs = tail_definition.get("cutoffs")
    if train_fit_cutoffs is not None and not isinstance(train_fit_cutoffs, Mapping):
        raise ValueError("probe train fitCutoffs must be an object")
    if calibration_cutoffs is not None and not isinstance(calibration_cutoffs, Mapping):
        raise ValueError("probe calibration cutoffs must be an object")
    coverage_train_fit_cutoffs = None
    coverage_calibration_cutoffs = None
    coverage_rescue_threshold = None
    if dual_rescue_requested:
        coverage_rescue_threshold = certificate_threshold_from_probe(
            coverage_probe, kind="rescue", risk=float(args.coverage_rescue_risk)
        )
        if not isinstance(coverage_probe_payload, Mapping):
            raise ValueError("coverage probe has no payload")
        coverage_tail_definition = coverage_probe_payload.get("tailDefinition")
        if not isinstance(coverage_tail_definition, Mapping):
            raise ValueError("coverage probe has no tailDefinition for residual diagnostics")
        coverage_train_fit_cutoffs = coverage_tail_definition.get("fitCutoffs")
        coverage_calibration_cutoffs = coverage_tail_definition.get("cutoffs")
        if coverage_train_fit_cutoffs is not None and not isinstance(
            coverage_train_fit_cutoffs, Mapping
        ):
            raise ValueError("coverage probe train fitCutoffs must be an object")
        if coverage_calibration_cutoffs is not None and not isinstance(
            coverage_calibration_cutoffs, Mapping
        ):
            raise ValueError("coverage probe calibration cutoffs must be an object")
    checkpoint = _load_checkpoint(args.checkpoint.resolve())
    config = checkpoint.get("modelConfig")
    if not isinstance(config, Mapping) or int(config.get("numInstances", -1)) <= 0:
        raise ValueError("checkpoint has no valid modelConfig.numInstances")
    dataset = PoseCSRDataset(
        args.dataset_dir.resolve(),
        num_instances=int(config["numInstances"]),
    )
    model, runtime_features, device = _build_model_bundle(
        args.checkpoint.resolve(),
        args.runtime_meta.resolve(),
        args.initial_geo_features.resolve(),
        dataset,
        args.device,
    )
    _world_aabbs, instance_to_glb, _runtime_meta = load_runtime_meta(args.runtime_meta.resolve())
    glb_bytes = None
    if args.glb_index is not None and args.glb_root is not None:
        num_glbs = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
        glb_bytes = _load_glb_bytes(args.glb_index.resolve(), args.glb_root.resolve(), num_glbs)

    train = load_score_rows(args.train_capture, "train", dataset)
    calibration = load_score_rows(args.calibration_capture, "calibration", dataset)
    validation = load_score_rows(args.validation_capture, "validation", dataset)
    probes: dict[str, ProbeSpec] = {"primary": probe}
    if dual_rescue_requested and coverage_probe is not None:
        probes["coverage"] = coverage_probe
    calibration_probe_scores = _extract_probe_score_sets(
        calibration.rows,
        dataset,
        model,
        runtime_features,
        device,
        probes,
        int(args.batch_size),
    )
    validation_probe_scores = _extract_probe_score_sets(
        validation.rows,
        dataset,
        model,
        runtime_features,
        device,
        probes,
        int(args.batch_size),
    )
    calibration.probe_scores = calibration_probe_scores["primary"]
    validation.probe_scores = validation_probe_scores["primary"]
    if dual_rescue_requested:
        calibration.coverage_probe_scores = calibration_probe_scores["coverage"]
        validation.coverage_probe_scores = validation_probe_scores["coverage"]
    members = scan_members(
        calibration,
        validation,
        alphas=alphas,
        temperatures=temperatures,
        gate_temperatures=gate_temperatures,
        residual_signs=residual_signs,
        residual_modes=residual_modes,
        frontier_center_logit=probe.frontier_center_logit,
        suppression_threshold=suppression_threshold,
        rescue_threshold=rescue_threshold,
        coverage_rescue_threshold=coverage_rescue_threshold,
        suppression_risk=float(args.suppression_risk),
        rescue_risk=float(args.rescue_risk),
        coverage_rescue_risk=float(args.coverage_rescue_risk),
        coverage_rescue_weights=coverage_rescue_weights,
        train_fit_cutoffs=train_fit_cutoffs,
        calibration_cutoffs=calibration_cutoffs,
        coverage_train_fit_cutoffs=coverage_train_fit_cutoffs,
        coverage_calibration_cutoffs=coverage_calibration_cutoffs,
        bootstrap_replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
        instance_to_glb=instance_to_glb if glb_bytes is not None else None,
        glb_bytes=glb_bytes,
    )
    payload = {
        "schema": SCHEMA,
        "version": 1,
        "testRead": False,
        "splitsRead": list(ALLOWED_SPLITS),
        "probe": {
            "path": str(args.probe.resolve()),
            "schema": PROBE_SCHEMA,
            "family": probe.family,
            "featureCount": probe.feature_count,
            "probeType": probe.probe_type,
            "fitSplit": probe.fit_split,
            "ridge": probe.ridge,
            "labelSemantics": PROBE_LABEL_SEMANTICS,
            "standardizationSource": "probe.families[family].standardization from train",
            "parametersSource": "probe.families[family] train-owned parameters",
            "frontierCenterLogit": probe.frontier_center_logit,
            "frontierCenterSource": (
                "tailDefinition.fitCutoffs from train"
                if probe.frontier_center_logit is not None
                else "unavailable; gate must remain none/0"
            ),
            "riskCertificatesSource": (
                "probe.families[family].riskCertificates; sourceSplit=train; fitRowsOnly=true"
                if probe.risk_certificates is not None
                else "unavailable; one-sided modes are not runnable"
            ),
        },
        "coverageProbe": (
            None
            if coverage_probe is None
            else {
                "path": str(args.coverage_probe.resolve()),
                "schema": PROBE_SCHEMA,
                "family": coverage_probe.family,
                "featureCount": coverage_probe.feature_count,
                "probeType": coverage_probe.probe_type,
                "fitSplit": coverage_probe.fit_split,
                "ridge": coverage_probe.ridge,
                "labelSemantics": PROBE_LABEL_SEMANTICS,
                "role": "ordinary_low_score_positive_rescue",
                "risk": float(args.coverage_rescue_risk),
                "threshold": coverage_rescue_threshold,
                "thresholdSource": "coverage probe JSON train-only rescue certificate",
                "standardizationSource": "coverage probe.families[family].standardization from train",
                "parametersSource": "coverage probe.families[family] train-owned parameters",
                "riskCertificatesSource": (
                    "coverage probe.families[family].riskCertificates; sourceSplit=train; fitRowsOnly=true"
                ),
            }
        ),
        "configuration": {
            "alphas": list(alphas),
            "temperatures": list(temperatures),
            "gateTemperatures": list(gate_temperatures),
            "residualSigns": list(residual_signs),
            "residualModes": list(residual_modes),
            "suppressionRisk": float(args.suppression_risk),
            "rescueRisk": float(args.rescue_risk),
            "coverageProbeFamily": args.coverage_probe_family if dual_rescue_requested else None,
            "coverageRescueRisk": float(args.coverage_rescue_risk),
            "coverageRescueWeights": list(coverage_rescue_weights),
            "certificateThresholds": {
                "suppression": suppression_threshold,
                "rescue": rescue_threshold,
                "coverageRescue": coverage_rescue_threshold,
            },
            "bootstrapReplicates": int(args.bootstrap_replicates),
            "bootstrapLowerQuantile": BOOTSTRAP_CONFIDENCE_QUANTILE,
            "weightedRecallFloor": WEIGHTED_RECALL_FLOOR,
            "scoreAuthority": "candidateScores in each capture",
            "modelUse": "compute_logits_with_aux only for candidate-aligned probe features",
            "parametersTrained": False,
            "candidateSetChanged": False,
            "groundTruthChanged": False,
            "correctionParametersSource": (
                "CLI pre-registered scan; calibration freezes only visibility threshold"
            ),
        },
        "inputProvenance": {
            "trainCapture": str(args.train_capture.resolve()),
            "calibrationCapture": str(args.calibration_capture.resolve()),
            "validationCapture": str(args.validation_capture.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "datasetDir": str(args.dataset_dir.resolve()),
            "runtimeMeta": str(args.runtime_meta.resolve()),
            "initialGeoFeatures": str(args.initial_geo_features.resolve()),
            "coverageProbe": (
                None if args.coverage_probe is None else str(args.coverage_probe.resolve())
            ),
            "glbMetrics": glb_bytes is not None,
            "testRead": False,
        },
        "splitCounts": {
            "trainPoseCount": len(train.rows),
            "calibrationPoseCount": len(calibration.rows),
            "validationPoseCount": len(validation.rows),
        },
        "members": members,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "memberCount": len(members), "testRead": False}))


if __name__ == "__main__":
    main()
