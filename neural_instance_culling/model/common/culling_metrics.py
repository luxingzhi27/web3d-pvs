"""Shared culling metrics derived from per-pose confusion counts."""
from __future__ import annotations

import numpy as np


def candidate_normalized_occlusion_recall(
    true_negative: np.ndarray,
    false_positive: np.ndarray,
    candidate_count: np.ndarray,
) -> float:
    """Return candidate-normalized occlusion recall (CNOR).

    Poses contribute in proportion to their negative opportunity divided by
    their candidate count. Empty candidate poses and poses without negative
    candidates contribute no opportunity.
    """
    tn = np.asarray(true_negative, dtype=np.float64)
    fp = np.asarray(false_positive, dtype=np.float64)
    candidates = np.asarray(candidate_count, dtype=np.float64)
    if tn.shape != fp.shape or tn.shape != candidates.shape:
        raise ValueError("CNOR inputs must have identical shapes")
    valid = candidates > 0.0
    normalized_tn = np.divide(tn, candidates, out=np.zeros_like(tn), where=valid)
    normalized_negative = np.divide(
        tn + fp,
        candidates,
        out=np.zeros_like(tn),
        where=valid,
    )
    denominator = float(normalized_negative.sum())
    return float(normalized_tn.sum() / denominator) if denominator > 0.0 else 1.0
