"""Select spatially covering dense subposes for offline depth evidence.

The selector is deliberately independent of visibility labels.  It uses the
view-cell center as the first representative and then greedily maximizes the
distance to the already selected camera positions.  This gives a deterministic
center-plus-spatial-coverage set without adding GT instances or changing the
candidate CSR.
"""
from __future__ import annotations

from typing import Any

import numpy as np


SCHEMA = "viewcell-center-spatial-coverage-subpose-selection-v1"


def select_representative_subposes(
    viewcell_center: np.ndarray,
    subpose_positions: np.ndarray,
    *,
    count: int = 5,
) -> np.ndarray:
    """Return local subpose indices for center plus farthest-point coverage."""
    center = np.asarray(viewcell_center, dtype=np.float64).reshape(3)
    positions = np.asarray(subpose_positions, dtype=np.float64).reshape(-1, 3)
    if int(count) <= 0:
        raise ValueError("representative subpose count must be positive")
    if positions.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    target = min(int(count), int(positions.shape[0]))
    distance_to_center = np.sum((positions - center[None, :]) ** 2, axis=1)
    first = int(np.argmin(distance_to_center))
    selected = [first]
    selected_mask = np.zeros((positions.shape[0],), dtype=bool)
    selected_mask[first] = True
    min_distance = np.sum((positions - positions[first][None, :]) ** 2, axis=1)
    while len(selected) < target:
        scores = np.where(selected_mask, -np.inf, min_distance)
        next_index = int(np.argmax(scores))
        if not np.isfinite(scores[next_index]):
            break
        selected.append(next_index)
        selected_mask[next_index] = True
        distance = np.sum((positions - positions[next_index][None, :]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return np.asarray(selected, dtype=np.int64)


def describe_selection(
    viewcell_id: int,
    center: np.ndarray,
    subpose_positions: np.ndarray,
    selected_local_indices: np.ndarray,
) -> dict[str, Any]:
    positions = np.asarray(subpose_positions, dtype=np.float64).reshape(-1, 3)
    selected = np.asarray(selected_local_indices, dtype=np.int64).reshape(-1)
    return {
        "schema": SCHEMA,
        "viewcellId": int(viewcell_id),
        "requestedCount": int(selected.size),
        "availableSubposeCount": int(positions.shape[0]),
        "selectedLocalIndices": selected.astype(int).tolist(),
        "centerLocalIndex": int(selected[0]) if selected.size else None,
        "selectedCameraPositions": positions[selected].astype(np.float32).tolist() if selected.size else [],
        "viewcellCenter": np.asarray(center, dtype=np.float32).reshape(3).tolist(),
        "selectionRule": "nearest-to-viewcell-center then greedy farthest-point spatial coverage",
        "usesVisibilityLabels": False,
    }
