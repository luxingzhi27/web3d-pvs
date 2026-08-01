from __future__ import annotations

from pathlib import Path

import numpy as np


def load_glb_points(path: str | Path, max_points: int | None = None) -> tuple[np.ndarray, dict]:
    """Load the GLB surface point cache produced by `generate_glb_points_v3.mjs`.

    File layout:
      - 16-byte header as little-endian uint32: num_glbs, points_per_glb, dims, reserved.
      - FP32 payload with shape [num_glbs, points_per_glb, 3].
    """
    path = Path(path)
    with path.open("rb") as f:
        header = np.frombuffer(f.read(16), dtype="<u4")
    num_glbs, points_per_glb, dims = int(header[0]), int(header[1]), int(header[2])
    if dims != 3:
        raise ValueError(f"Expected GLB point dim=3, got {dims}")
    data = np.memmap(path, mode="r", dtype="<f4", offset=16, shape=(num_glbs, points_per_glb, 3))
    if max_points is not None and max_points > 0:
        data = data[:, :min(max_points, points_per_glb), :]
    return np.array(data, dtype=np.float32, copy=True), {
        "numGlbs": num_glbs,
        "pointsPerGlb": points_per_glb,
        "dims": dims,
    }
