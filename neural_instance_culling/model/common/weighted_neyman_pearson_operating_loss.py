"""Train-only weighted Neyman-Pearson operating-point loss.

The trainable boundary is represented in logit space.  The loss has no
calibration, validation, or test threshold input and is not imported by a
runtime inference path::

    soft_fpr + multiplier * (recall_target - soft_weighted_recall)

``visible_weights`` affect only the positive recall constraint.  False
positive rate is normalized by the number of negative candidates so that its
scale does not change with positive visibility mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc


def _flat_values(
    value: Any,
    name: str,
    device: torch.device,
    *,
    allow_bool: bool = False,
) -> torch.Tensor:
    tensor = _as_tensor(value, name)
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if torch.is_complex(tensor) or (tensor.dtype == torch.bool and not allow_bool):
        raise ValueError(f"{name} must be real-valued")
    result = tensor.to(device=device, dtype=torch.float32).reshape(-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    tensor = _as_tensor(value, name)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    if torch.is_complex(tensor) or tensor.dtype == torch.bool:
        raise ValueError(f"{name} must be a real scalar")
    detached = tensor.detach().reshape(-1)
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{name} must be finite")
    result = float(detached.cpu().item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boundary_tensor(value: Any, device: torch.device) -> torch.Tensor:
    tensor = _as_tensor(value, "boundary_logit")
    if tensor.numel() != 1:
        raise ValueError("boundary_logit must be a scalar")
    if torch.is_complex(tensor) or tensor.dtype == torch.bool:
        raise ValueError("boundary_logit must be a real scalar")
    boundary = tensor.to(device=device, dtype=torch.float32).reshape(())
    if not bool(torch.isfinite(boundary).all()):
        raise ValueError("boundary_logit must be finite")
    return boundary


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} became non-finite")


@dataclass
class WeightedNeymanPearsonDualState:
    """Autograd-independent dual state for the recall constraint."""

    multiplier: float = 0.0
    learning_rate: float = 0.05
    maximum: float = 20.0

    def __post_init__(self) -> None:
        self.multiplier = _finite_scalar(self.multiplier, "multiplier")
        self.learning_rate = _finite_scalar(self.learning_rate, "learning_rate")
        self.maximum = _finite_scalar(self.maximum, "maximum")
        if self.multiplier < 0.0:
            raise ValueError("multiplier must be non-negative")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        if self.maximum < 0.0:
            raise ValueError("maximum must be non-negative")
        if self.multiplier > self.maximum:
            raise ValueError("multiplier must not exceed maximum")

    def update(self, constraint_violation: torch.Tensor | float) -> None:
        """Perform detached projected dual ascent for ``target - recall``."""
        violation = _finite_scalar(constraint_violation, "constraint_violation")
        updated = self.multiplier + self.learning_rate * violation
        if not math.isfinite(updated):
            raise FloatingPointError("dual multiplier update became non-finite")
        self.multiplier = min(self.maximum, max(0.0, updated))

    def as_dict(self) -> dict[str, float]:
        return {
            "multiplier": float(self.multiplier),
            "learning_rate": float(self.learning_rate),
            "maximum": float(self.maximum),
        }


# Keep the shorter name available for callers that describe the constraint
# without repeating the weighted operating-point prefix.
NeymanPearsonDualState = WeightedNeymanPearsonDualState


def weighted_neyman_pearson_operating_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    visible_weights: torch.Tensor,
    boundary_logit: torch.Tensor,
    temperature: float,
    dual_multiplier: float,
    weighted_recall_target: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Return the train-only weighted Neyman-Pearson operating loss.

    ``logits`` and ``boundary_logit`` produce a soft keep probability with
    ``sigmoid((logit - boundary_logit) / temperature)``.  The false-positive
    rate is the mean soft keep probability over negative candidates.  The
    recall term is the raw ratio of weighted soft true positives to weighted
    positive mass; no calibration or runtime threshold is consulted.

    A batch must contain at least one positive and one negative candidate, and
    the positive candidates must have strictly positive total visible weight.
    This makes both operating-point quantities defined instead of silently
    turning a zero denominator into NaN or an arbitrary fallback.
    """
    logits_tensor = _as_tensor(logits, "logits")
    if logits_tensor.numel() == 0:
        raise ValueError("logits must be non-empty")
    if torch.is_complex(logits_tensor) or logits_tensor.dtype == torch.bool:
        raise ValueError("logits must be real-valued")
    logits_flat = logits_tensor.to(dtype=torch.float32).reshape(-1)
    if not bool(torch.isfinite(logits_flat).all()):
        raise ValueError("logits must contain only finite values")

    device = logits_flat.device
    labels = _flat_values(target, "target", device, allow_bool=True)
    weights = _flat_values(visible_weights, "visible_weights", device)
    if labels.numel() != logits_flat.numel():
        raise ValueError("target must align with logits")
    if weights.numel() != logits_flat.numel():
        raise ValueError("visible_weights must align with logits")
    if bool((weights < 0.0).any()):
        raise ValueError("visible_weights must be non-negative")
    binary = (labels == 0.0) | (labels == 1.0)
    if not bool(binary.all()):
        raise ValueError("target must contain only binary values 0 and 1")

    boundary = _boundary_tensor(boundary_logit, device)
    temperature_value = _finite_scalar(temperature, "temperature")
    dual_value = _finite_scalar(dual_multiplier, "dual_multiplier")
    recall_target_value = _finite_scalar(
        weighted_recall_target, "weighted_recall_target"
    )
    if temperature_value <= 0.0:
        raise ValueError("temperature must be positive")
    if dual_value < 0.0:
        raise ValueError("dual_multiplier must be non-negative")
    if not 0.0 <= recall_target_value <= 1.0:
        raise ValueError("weighted_recall_target must lie in [0, 1]")

    # Labels and supervision weights are constants for this train-only loss.
    labels = labels.detach()
    weights = weights.detach()
    positive = labels
    negative = 1.0 - labels
    positive_count = int((positive > 0.5).sum().item())
    negative_count = int((negative > 0.5).sum().item())
    if positive_count == 0:
        raise ValueError("target must contain at least one positive")
    if negative_count == 0:
        raise ValueError("target must contain at least one negative")

    positive_weight_mass = (positive * weights).sum()
    _require_finite(positive_weight_mass, "positive visible weight mass")
    if float(positive_weight_mass.detach().cpu()) <= 0.0:
        raise ValueError("positive visible weights must have positive mass")

    scaled_logits = (logits_flat - boundary) / temperature_value
    _require_finite(scaled_logits, "scaled logits")
    soft_keep = torch.sigmoid(scaled_logits)
    _require_finite(soft_keep, "soft keep probabilities")

    soft_false_positive_mass = (soft_keep * negative).sum()
    soft_fpr = soft_false_positive_mass / float(negative_count)
    soft_weighted_true_positive_mass = (soft_keep * positive * weights).sum()
    soft_weighted_recall = (
        soft_weighted_true_positive_mass / positive_weight_mass
    )
    constraint_violation = recall_target_value - soft_weighted_recall
    dual_penalty = dual_value * constraint_violation
    loss = soft_fpr + dual_penalty

    for name, value in (
        ("soft false-positive mass", soft_false_positive_mass),
        ("soft FPR", soft_fpr),
        ("soft weighted-positive true-positive mass", soft_weighted_true_positive_mass),
        ("soft weighted recall", soft_weighted_recall),
        ("constraint violation", constraint_violation),
        ("dual penalty", dual_penalty),
        ("loss", loss),
    ):
        _require_finite(value, name)

    diagnostics: dict[str, torch.Tensor | float] = {
        "loss": loss,
        "soft_fpr": soft_fpr,
        "soft_weighted_recall": soft_weighted_recall,
        "weighted_recall_violation": constraint_violation,
        "dual_penalty": dual_penalty,
        "soft_false_positive_mass": soft_false_positive_mass,
        "soft_weighted_true_positive_mass": soft_weighted_true_positive_mass,
        "positive_weight_mass": positive_weight_mass,
        "boundary_logit": boundary.detach(),
        "temperature": temperature_value,
        "dual_multiplier": dual_value,
        "weighted_recall_target": recall_target_value,
        "positive_count": float(positive_count),
        "negative_count": float(negative_count),
    }
    # The camel-case names match the metric naming used by adjacent training
    # losses while the snake-case names above remain the primary API.
    diagnostics.update(
        {
            "lossWeightedNeymanPearsonOperating": loss,
            "lossSoftFpr": soft_fpr,
            "softWeightedRecall": soft_weighted_recall,
            "weightedRecallViolation": constraint_violation,
            "dualPenalty": dual_penalty,
            "boundaryLogit": boundary.detach(),
            "boundaryProbability": torch.sigmoid(boundary.detach()),
            "temperatureLogit": temperature_value,
            "dualMultiplier": dual_value,
            "weightedRecallTarget": recall_target_value,
            "positiveWeightMass": positive_weight_mass,
            "positiveCount": float(positive_count),
            "negativeCount": float(negative_count),
        }
    )
    return loss, diagnostics


__all__ = [
    "NeymanPearsonDualState",
    "WeightedNeymanPearsonDualState",
    "weighted_neyman_pearson_operating_loss",
]
