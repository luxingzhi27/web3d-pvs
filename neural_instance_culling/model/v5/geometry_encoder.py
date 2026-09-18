"""The scene-independent local surface encoder used by GCOF-PVS V5.

The encoder consumes a fixed 256 point sample for one renderable unit.  The
first six point channels are normalized local position and unit normal.  The
three size ratios are unit-level inputs and are concatenated after symmetric
max/mean pooling.  There is deliberately no scene or unit lookup table here:
the same module can compile geometry from a new scene without fitting any
parameters.
"""
from __future__ import annotations

import torch
from torch import nn


POINT_COUNT = 256
POINT_INPUT_DIM = 6
SIZE_RATIO_DIM = 3
GEOMETRY_DIM = 32


def _require_finite(value: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


class LocalSurfaceGeometryEncoder(nn.Module):
    """Encode one 256-point local surface sample into a 32D descriptor.

    Parameters are shared across all units.  ``points`` may contain either
    the required six point channels and a separate ``size_ratios`` tensor, or
    the same nine values with the three ratios repeated in the last point
    row.  The separate form is the canonical representation and is used by
    the training/export path.
    """

    point_count: int = POINT_COUNT
    point_input_dim: int = POINT_INPUT_DIM
    size_ratio_dim: int = SIZE_RATIO_DIM
    out_dim: int = GEOMETRY_DIM

    def __init__(self) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(POINT_INPUT_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.unit_mlp = nn.Sequential(
            nn.Linear(64 * 2 + SIZE_RATIO_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, GEOMETRY_DIM),
            nn.Tanh(),
        )

    @staticmethod
    def _split_inputs(
        points: torch.Tensor,
        size_ratios: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        values = torch.as_tensor(points)
        squeezed = False
        if values.ndim == 2:
            values = values.unsqueeze(0)
            squeezed = True
        if values.ndim != 3 or values.shape[1] != POINT_COUNT:
            raise ValueError(f"points must have shape [B, {POINT_COUNT}, 6]")

        if values.shape[2] == POINT_INPUT_DIM:
            if size_ratios is None:
                raise ValueError("size_ratios are required for six-channel points")
            ratios = torch.as_tensor(size_ratios, device=values.device, dtype=values.dtype)
        elif values.shape[2] == POINT_INPUT_DIM + SIZE_RATIO_DIM:
            if size_ratios is not None:
                raise ValueError("size_ratios must be omitted when points contain nine channels")
            ratios = values[:, 0, POINT_INPUT_DIM:]
            values = values[:, :, :POINT_INPUT_DIM]
        else:
            raise ValueError(f"points must have six or nine channels, got {values.shape[2]}")

        if ratios.ndim == 1:
            ratios = ratios.unsqueeze(0)
        if ratios.shape != (values.shape[0], SIZE_RATIO_DIM):
            raise ValueError(
                f"size_ratios must have shape [{values.shape[0]}, {SIZE_RATIO_DIM}]"
            )
        return values.float(), ratios.float(), squeezed

    def forward(
        self,
        points: torch.Tensor,
        size_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        point_values, ratios, squeezed = self._split_inputs(points, size_ratios)
        _require_finite(point_values, "points")
        _require_finite(ratios, "size_ratios")
        encoded = self.point_mlp(point_values)
        pooled = torch.cat(
            [encoded.amax(dim=1), encoded.mean(dim=1), ratios],
            dim=-1,
        )
        descriptor = self.unit_mlp(pooled)
        if not bool(torch.isfinite(descriptor).all()):
            raise FloatingPointError("local geometry encoder produced non-finite values")
        return descriptor[0] if squeezed else descriptor


# The shorter name is convenient for callers while retaining the explicit
# class name in exported model metadata.
LocalGeometryEncoder = LocalSurfaceGeometryEncoder


__all__ = [
    "GEOMETRY_DIM",
    "LocalGeometryEncoder",
    "LocalSurfaceGeometryEncoder",
    "POINT_COUNT",
    "POINT_INPUT_DIM",
    "SIZE_RATIO_DIM",
]
