#!/usr/bin/env python3
"""Fit a tiny nonlinear difficult-tail probe without retraining the PVS model.

The probe sees only the existing 108-dimensional view-cell query.  Its weights,
normalization, nonlinear basis, and finite-sample rescue certificate are fitted
from train.  Calibration freezes only the diagnostic probe threshold; validation
is replay-only and test is rejected by construction.

Two exportable probe families are supported:

``hinge``
    A ridge-regularized additive piecewise-linear model.  It adds train-fixed
    ReLU hinges to each standardized feature and has no feature interactions.

``mlp``
    A one-hidden-layer ReLU classifier with at most a few thousand shared
    parameters.  It adds no per-instance feature table or online neighbor query.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_instance_culling.benchmark.analyze_pvs_difficult_tail_feature_separability import (
    _average_precision,
    _build_model_bundle,
    _capture_examples,
    _combine_tail_features,
    _freeze_probe_threshold,
    _load_checkpoint,
    _model_scores_for_rows,
    _probe_direction_diagnostic,
    _probe_fit_sample_weights,
    _probe_score_distribution,
    _rank_auc,
    _resolve_row_arrays,
    _threshold_metrics,
    _weighted_rank_auc,
    build_train_risk_certificates,
    freeze_tail_cutoffs,
    load_capture,
)
from neural_instance_culling.model.pose_csr_dataset import PoseCSRDataset


SCHEMA = "pvs-difficult-tail-feature-separability-v1"
PROBE_TYPES = ("hinge", "mlp")
PROBE_LABEL_SEMANTICS = "1=high_score_negative, 0=low_score_positive"


def _finite_matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def _fit_rows(
    features: np.ndarray,
    labels: np.ndarray,
    visible_weights: np.ndarray,
    *,
    max_samples: int,
    sample_weight_power: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = _finite_matrix(features, "train features")
    targets = np.asarray(labels, dtype=bool).reshape(-1)
    visual_weights = np.asarray(visible_weights, dtype=np.float64).reshape(-1)
    if values.shape[0] != targets.size or targets.size != visual_weights.size:
        raise ValueError("train probe arrays are not aligned")
    if np.unique(targets).size != 2:
        raise ValueError("train probe rows require both difficult-tail classes")
    indices = np.arange(targets.size, dtype=np.int64)
    if max_samples > 0 and indices.size > max_samples:
        rng = np.random.default_rng(int(seed))
        low = indices[~targets]
        high = indices[targets]
        low_count = min(low.size, max(1, max_samples // 2))
        high_count = min(high.size, max(1, max_samples - low_count))
        if low_count + high_count < max_samples:
            if low_count < low.size:
                low_count = min(low.size, max_samples - high_count)
            elif high_count < high.size:
                high_count = min(high.size, max_samples - low_count)
        indices = np.sort(
            np.concatenate(
                [
                    rng.choice(low, size=low_count, replace=False),
                    rng.choice(high, size=high_count, replace=False),
                ]
            )
        )
    values = values[indices]
    targets = targets[indices]
    visual_weights = visual_weights[indices]
    sample_weights = _probe_fit_sample_weights(
        targets,
        visual_weights,
        power=float(sample_weight_power),
        epsilon=1e-6,
    )
    total = float(sample_weights.sum())
    mean = np.sum(values * sample_weights[:, None], axis=0) / total
    centered = values - mean[None, :]
    scale = np.sqrt(
        np.sum(centered * centered * sample_weights[:, None], axis=0) / total
    )
    scale = np.where(scale > 1e-8, scale, 1.0)
    return values, targets, visual_weights, sample_weights, mean, scale


def _hinge_design(standardized: np.ndarray, knots: np.ndarray) -> np.ndarray:
    pieces = [standardized]
    pieces.extend(np.maximum(standardized - float(knot), 0.0) for knot in knots)
    return np.concatenate(pieces, axis=1)


def _fit_hinge(
    values: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    knots: np.ndarray,
    ridge: float,
) -> tuple[dict[str, Any], Any]:
    standardized = (values - mean[None, :]) / scale[None, :]
    basis = _hinge_design(standardized, knots)
    design = np.concatenate(
        [np.ones((basis.shape[0], 1), dtype=np.float64), basis], axis=1
    )
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    regularizer[0, 0] = 0.0
    weighted_design = design * sample_weights[:, None]
    normal = design.T @ weighted_design + regularizer
    rhs = design.T @ (sample_weights * labels.astype(np.float64))
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]

    def predict(raw: np.ndarray) -> np.ndarray:
        standardized_raw = (raw - mean[None, :]) / scale[None, :]
        raw_basis = _hinge_design(standardized_raw, knots)
        return coefficients[0] + raw_basis @ coefficients[1:]

    record = {
        "probeType": "standardized_ridge_hinge",
        "ridge": float(ridge),
        "hingeKnots": [float(value) for value in knots],
        "coefficients": [float(value) for value in coefficients],
        "coefficientOrder": ["intercept", "standardized_features", "featurewise_hinges"],
        "parameterCount": int(coefficients.size),
    }
    return record, predict


def _fit_mlp(
    values: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], Any]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    features = torch.from_numpy(
        ((values - mean[None, :]) / scale[None, :]).astype(np.float32)
    )
    targets = torch.from_numpy(labels.astype(np.float32))
    weights = torch.from_numpy(sample_weights.astype(np.float32))
    model = torch.nn.Sequential(
        torch.nn.Linear(values.shape[1], int(hidden_dim)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(hidden_dim), 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for _epoch in range(int(epochs)):
        order = torch.randperm(features.shape[0], generator=generator)
        for start in range(0, features.shape[0], int(batch_size)):
            chosen = order[start : start + int(batch_size)]
            x = features[chosen].to(device)
            y = targets[chosen].to(device)
            w = weights[chosen].to(device)
            logits = model(x).reshape(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, weight=w, reduction="sum"
            ) / torch.clamp(w.sum(), min=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    first = model[0]
    last = model[2]
    hidden_weight = first.weight.detach().cpu().numpy().astype(np.float64)
    hidden_bias = first.bias.detach().cpu().numpy().astype(np.float64)
    output_weight = last.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    output_bias = float(last.bias.detach().cpu().item())

    def predict(raw: np.ndarray) -> np.ndarray:
        standardized_raw = (raw - mean[None, :]) / scale[None, :]
        hidden = np.maximum(
            standardized_raw @ hidden_weight.T + hidden_bias[None, :], 0.0
        )
        return hidden @ output_weight + output_bias

    record = {
        "probeType": "standardized_shallow_mlp",
        "ridge": float(weight_decay),
        "activation": "relu",
        "hiddenDim": int(hidden_dim),
        "epochs": int(epochs),
        "learningRate": float(learning_rate),
        "parameterCount": int(
            hidden_weight.size + hidden_bias.size + output_weight.size + 1
        ),
        "parameters": {
            "hiddenWeight": hidden_weight.tolist(),
            "hiddenBias": hidden_bias.tolist(),
            "outputWeight": output_weight.tolist(),
            "outputBias": output_bias,
        },
    }
    return record, predict


def fit_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_visible_weights: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    probe_type: str,
    sample_weight_power: float,
    max_samples: int,
    ridge: float,
    hinge_knots: Sequence[float] = (-1.0, 0.0, 1.0),
    hidden_dim: int = 8,
    epochs: int = 40,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    seed: int = 20260818,
    device: str = "auto",
) -> dict[str, Any]:
    """Fit one train-owned probe and return a loader-compatible record."""

    if probe_type not in PROBE_TYPES:
        raise ValueError(f"probe_type must be one of {PROBE_TYPES}")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge/weight decay must be finite and positive")
    train = _finite_matrix(train_features, "train features")
    calibration = _finite_matrix(calibration_features, "calibration features")
    validation = _finite_matrix(validation_features, "validation features")
    if train.shape[1] != calibration.shape[1] or train.shape[1] != validation.shape[1]:
        raise ValueError("probe feature dimensions differ across splits")
    values, labels, visual_weights, sample_weights, mean, scale = _fit_rows(
        train,
        train_labels,
        train_visible_weights,
        max_samples=int(max_samples),
        sample_weight_power=float(sample_weight_power),
        seed=int(seed),
    )
    if probe_type == "hinge":
        knots = np.unique(np.asarray(tuple(hinge_knots), dtype=np.float64))
        if knots.size == 0 or not bool(np.isfinite(knots).all()):
            raise ValueError("hinge knots must be finite and non-empty")
        parameters, predict = _fit_hinge(
            values,
            labels,
            sample_weights,
            mean=mean,
            scale=scale,
            knots=knots,
            ridge=float(ridge),
        )
    else:
        if hidden_dim <= 0 or epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
            raise ValueError("MLP dimensions and optimization settings must be positive")
        torch_device = torch.device(
            "cuda"
            if device == "cuda" or (device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        parameters, predict = _fit_mlp(
            values,
            labels,
            sample_weights,
            mean=mean,
            scale=scale,
            hidden_dim=int(hidden_dim),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(ridge),
            seed=int(seed),
            device=torch_device,
        )
    train_scores = predict(values)
    calibration_scores = predict(calibration)
    validation_scores = predict(validation)
    frozen = _freeze_probe_threshold(calibration_scores, calibration_labels)
    validation_threshold = (
        None
        if frozen is None
        else _threshold_metrics(
            validation_scores, validation_labels, float(frozen["threshold"])
        )
    )
    return {
        "status": "ok",
        "fitSplit": "train",
        "featureCount": int(train.shape[1]),
        **parameters,
        "standardization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "fitCount": int(values.shape[0]),
        "weighting": {
            "sampleWeightPower": float(sample_weight_power),
            "classTotalPolicy": "low_positive_visibility_mass_and_equal_class_total",
        },
        "fit": {
            "rocAuc": _rank_auc(train_scores, labels),
            "weightedRocAuc": _weighted_rank_auc(train_scores, labels, sample_weights),
            "direction": _probe_direction_diagnostic(train_scores, labels, sample_weights),
        },
        "calibration": {
            "rocAuc": _rank_auc(calibration_scores, calibration_labels),
            "averagePrecision": _average_precision(calibration_scores, calibration_labels),
        },
        "validation": {
            "rocAuc": _rank_auc(validation_scores, validation_labels),
            "averagePrecision": _average_precision(validation_scores, validation_labels),
            "thresholdMetrics": validation_threshold,
        },
        "frozenThreshold": frozen,
        "fitProbeScoreDistribution": _probe_score_distribution(
            train_scores, labels, source_split="train", fit_rows_only=True
        ),
        "trainFitProbeScoreDistribution": _probe_score_distribution(
            train_scores, labels, source_split="train", fit_rows_only=True
        ),
        "riskCertificates": build_train_risk_certificates(
            train_scores,
            labels,
            visual_weights,
            sample_weight_power=float(sample_weight_power),
            sample_weight_epsilon=1e-6,
        ),
        "testRead": False,
    }


def _parse_knots(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not result or not all(np.isfinite(item) for item in result):
        raise ValueError("hinge-knots must contain finite values")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-capture", type=Path, required=True)
    parser.add_argument("--calibration-capture", type=Path, required=True)
    parser.add_argument("--validation-capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--initial-geo-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-type", choices=PROBE_TYPES, required=True)
    parser.add_argument("--positive-tail-mode", choices=("count", "weighted_mass"), default="weighted_mass")
    parser.add_argument("--positive-tail-quantile", type=float, default=0.01)
    parser.add_argument("--negative-tail-quantile", type=float, default=0.99)
    parser.add_argument("--sample-weight-power", type=float, default=0.5)
    parser.add_argument("--max-probe-samples", type=int, default=30000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--hinge-knots", default="-1,0,1")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    captures = {
        split: load_capture(Path(getattr(args, f"{split}_capture")).resolve(), split)
        for split in ("train", "calibration", "validation")
    }
    checkpoint = _load_checkpoint(args.checkpoint.resolve())
    config = checkpoint.get("modelConfig")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no modelConfig")
    dataset = PoseCSRDataset(
        args.dataset_dir.resolve(), num_instances=int(config["numInstances"])
    )
    model_bundle = _build_model_bundle(
        args.checkpoint.resolve(),
        args.runtime_meta.resolve(),
        args.initial_geo_features.resolve(),
        dataset,
        args.device,
    )
    rows = {
        split: _resolve_row_arrays(payload["_normalizedRows"], dataset=dataset, split=split)
        for split, payload in captures.items()
    }
    for split_rows in rows.values():
        if any(row.scores is None for row in split_rows):
            _model_scores_for_rows(
                split_rows, dataset, *model_bundle, int(args.batch_size)
            )

    def arrays(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate([row.scores for row in rows[split] if row.scores is not None]),
            np.concatenate([row.targets for row in rows[split] if row.targets is not None]),
            np.concatenate(
                [row.visible_weights for row in rows[split] if row.visible_weights is not None]
            ),
        )

    calibration_arrays = arrays("calibration")
    train_arrays = arrays("train")
    calibration_cutoffs = freeze_tail_cutoffs(
        calibration_arrays[0],
        calibration_arrays[1],
        visible_weights=calibration_arrays[2],
        positive_tail_mode=args.positive_tail_mode,
        positive_quantile=float(args.positive_tail_quantile),
        negative_quantile=float(args.negative_tail_quantile),
        source_split="calibration",
    )
    train_cutoffs = freeze_tail_cutoffs(
        train_arrays[0],
        train_arrays[1],
        visible_weights=train_arrays[2],
        positive_tail_mode=args.positive_tail_mode,
        positive_quantile=float(args.positive_tail_quantile),
        negative_quantile=float(args.negative_tail_quantile),
        source_split="train",
    )

    def tail(split: str, cutoffs: Mapping[str, Any]):
        return _combine_tail_features(
            _capture_examples(
                captures[split],
                split=split,
                dataset=dataset,
                model_bundle=model_bundle,
                cutoffs=cutoffs,
                batch_size=int(args.batch_size),
            )
        )

    train_tail = tail("train", train_cutoffs)
    calibration_tail = tail("calibration", calibration_cutoffs)
    validation_tail = tail("validation", calibration_cutoffs)
    record = fit_probe(
        train_tail.features["combined"],
        train_tail.targets,
        train_tail.visible_weights,
        calibration_tail.features["combined"],
        calibration_tail.targets,
        validation_tail.features["combined"],
        validation_tail.targets,
        probe_type=args.probe_type,
        sample_weight_power=float(args.sample_weight_power),
        max_samples=int(args.max_probe_samples),
        ridge=float(args.ridge),
        hinge_knots=_parse_knots(args.hinge_knots),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        device=args.device,
    )
    payload = {
        "schema": SCHEMA,
        "testRead": False,
        "splitsRead": ["train", "calibration", "validation"],
        "tailDefinition": {
            "lowScorePositive": "target=1 and score <= frozen lowPositiveScoreCutoff",
            "highScoreNegative": "target=0 and score >= frozen highNegativeScoreCutoff",
            "probeLabel": PROBE_LABEL_SEMANTICS,
            "cutoffs": calibration_cutoffs,
            "fitCutoffs": train_cutoffs,
        },
        "groups": {
            "train": {"total": int(train_tail.scores.size)},
            "calibration": {"total": int(calibration_tail.scores.size)},
            "validation": {"total": int(validation_tail.scores.size)},
        },
        "probe": {
            "fitSplit": "train",
            "thresholdSource": "calibration",
            "selectedFromValidation": False,
            "combinedFeatureNames": [
                f"combined.feature{index}"
                for index in range(int(train_tail.features["combined"].shape[1]))
            ],
            "families": {"combined": record},
        },
        "configuration": {
            "probeType": args.probe_type,
            "positiveTailMode": args.positive_tail_mode,
            "positiveTailQuantile": float(args.positive_tail_quantile),
            "negativeTailQuantile": float(args.negative_tail_quantile),
            "sampleWeightPower": float(args.sample_weight_power),
            "maxProbeSamples": int(args.max_probe_samples),
            "seed": int(args.seed),
        },
        "inputProvenance": {
            "trainCapture": str(args.train_capture.resolve()),
            "calibrationCapture": str(args.calibration_capture.resolve()),
            "validationCapture": str(args.validation_capture.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "datasetDir": str(args.dataset_dir.resolve()),
            "featureSource": "existing_v4_108d_runtime_query",
            "candidateSetChanged": False,
            "groundTruthChanged": False,
            "testRead": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "probeType": record["probeType"],
                "parameterCount": record["parameterCount"],
                "validationRocAuc": record["validation"]["rocAuc"],
                "testRead": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
