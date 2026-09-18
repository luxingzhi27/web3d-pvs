"""Single-source fixed direction contract shared by V5 data and model code."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DIRECTION_SCHEMA = "gcof-pvs-v5-fixed-directions-v1"
DIRECTION_RESOURCE = Path(__file__).with_name("directions.json")


def _load_anchor_directions() -> np.ndarray:
    payload = json.loads(DIRECTION_RESOURCE.read_text(encoding="utf-8"))
    if payload.get("schema") != DIRECTION_SCHEMA:
        raise ValueError("V5 fixed direction resource has an unexpected schema")
    values = np.asarray(payload.get("anchorDirections"), dtype=np.float64)
    if values.shape != (12, 3) or not bool(np.isfinite(values).all()):
        raise ValueError("V5 fixed direction resource must contain twelve finite vectors")
    norms = np.linalg.norm(values, axis=1)
    if not bool(np.allclose(norms, 1.0, atol=1.0e-7, rtol=0.0)):
        raise ValueError("V5 fixed anchor directions must have unit norm")
    if np.unique(np.round(values, decimals=12), axis=0).shape[0] != 12:
        raise ValueError("V5 fixed anchor directions must be unique")
    values.setflags(write=False)
    return values


_ANCHOR_DIRECTIONS = _load_anchor_directions()


def icosahedron12_directions(*, dtype: np.dtype | type = np.float32) -> np.ndarray:
    return _ANCHOR_DIRECTIONS.astype(dtype, copy=True)


__all__ = ["DIRECTION_RESOURCE", "DIRECTION_SCHEMA", "icosahedron12_directions"]
