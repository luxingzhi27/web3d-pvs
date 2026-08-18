"""Analytic extrema of the linearized horizontal view-cell query.

The registered view-cell query has the form ``x = center + axes @ u`` with
``||u|| <= 1``.  Each scalar feature therefore has the exact interval
``center[j] +/- ||axes[j]||`` inside that linearized disk.  These extrema keep
the rare boundary opportunity required by a view-cell visibility union; no
runtime subpose expansion is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


VIEW_DIM = 9
DISK_AXIS_DIM = 2
VIEWCELL_EXTREME_ENVELOPE_DIM = 17
VIEWCELL_EXTREME_ENVELOPE_SCHEMA = "view-cell-analytic-extreme-envelope-v1"
VIEWCELL_SUPPORT_FREQUENCY_COUNT = 16
VIEWCELL_SUPPORT_ENVELOPE_DIM = VIEWCELL_SUPPORT_FREQUENCY_COUNT * 2
VIEWCELL_SUPPORT_ENVELOPE_SCHEMA = "view-cell-learned-direction-support-envelope-v1"


@dataclass(frozen=True)
class ViewCellExtremeEnvelope:
    half_range: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    overlap_margins: torch.Tensor
    features: torch.Tensor


@dataclass(frozen=True)
class ViewCellSupportEnvelope:
    lower: torch.Tensor
    upper: torch.Tensor
    features: torch.Tensor


def _floating(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def viewcell_extreme_envelope(
    center_view: Any,
    disk_axes: Any,
) -> ViewCellExtremeEnvelope:
    """Return exact scalar extrema for the registered linearized disk.

    Feature order is nine per-coordinate half-ranges, minimum/maximum forward
    cosine, minimum/maximum normalized log distance, then best/worst horizontal
    and vertical frustum-overlap margins.  Positive overlap margin means that
    the instance can overlap that screen axis at the represented location.
    """

    center = _floating(center_view, "center_view")
    axes = _floating(disk_axes, "disk_axes").to(
        device=center.device, dtype=center.dtype
    )
    if center.ndim != 2 or center.shape[1] != VIEW_DIM:
        raise ValueError(f"center_view must have shape [B, {VIEW_DIM}]")
    if axes.shape != (center.shape[0], VIEW_DIM, DISK_AXIS_DIM):
        raise ValueError(
            f"disk_axes must have shape [B, {VIEW_DIM}, {DISK_AXIS_DIM}]"
        )

    half_range = torch.linalg.vector_norm(axes, dim=-1)
    lower = (center - half_range).clamp(-1.0, 1.0)
    upper = (center + half_range).clamp(-1.0, 1.0)

    def axis_overlap(screen_index: int, extent_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        center_screen = center[:, screen_index : screen_index + 1]
        screen_radius = half_range[:, screen_index : screen_index + 1]
        minimum_abs_screen = torch.relu(center_screen.abs() - screen_radius)
        maximum_abs_screen = (center_screen.abs() + screen_radius).clamp(max=1.0)

        # screen_u/v are raw projected coordinates divided by four.  The
        # angular extent feature maps raw half extent h to h / 2 - 1, so its
        # matching screen-space half extent is (feature + 1) / 2.
        minimum_extent = (lower[:, extent_index : extent_index + 1] + 1.0) * 0.5
        maximum_extent = (upper[:, extent_index : extent_index + 1] + 1.0) * 0.5
        best = 0.25 + maximum_extent - minimum_abs_screen
        worst = 0.25 + minimum_extent - maximum_abs_screen
        return best.clamp(-1.0, 1.0), worst.clamp(-1.0, 1.0)

    best_horizontal, worst_horizontal = axis_overlap(5, 7)
    best_vertical, worst_vertical = axis_overlap(6, 8)
    overlap_margins = torch.cat(
        [best_horizontal, best_vertical, worst_horizontal, worst_vertical],
        dim=-1,
    )
    features = torch.cat(
        [
            half_range,
            lower[:, 4:5],
            upper[:, 4:5],
            lower[:, 3:4],
            upper[:, 3:4],
            overlap_margins,
        ],
        dim=-1,
    )
    if features.shape != (center.shape[0], VIEWCELL_EXTREME_ENVELOPE_DIM):
        raise RuntimeError("view-cell extreme-envelope layout drifted")
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("view-cell extreme envelope is non-finite")
    return ViewCellExtremeEnvelope(
        half_range=half_range,
        lower=lower,
        upper=upper,
        overlap_margins=overlap_margins,
        features=features,
    )


def viewcell_support_envelope(
    center_view: Any,
    disk_axes: Any,
    directions: Any,
) -> ViewCellSupportEnvelope:
    """Return exact support intervals along learned joint feature directions.

    Unlike independent coordinate bounds, each interval preserves correlation
    across all nine ray-space coordinates.  Direction magnitude is removed so
    the feature describes orientation rather than duplicating Fourier scale.
    """

    center = _floating(center_view, "center_view")
    axes = _floating(disk_axes, "disk_axes").to(
        device=center.device, dtype=center.dtype
    )
    direction = _floating(directions, "directions").to(
        device=center.device, dtype=center.dtype
    )
    if center.ndim != 2 or center.shape[1] != VIEW_DIM:
        raise ValueError(f"center_view must have shape [B, {VIEW_DIM}]")
    if axes.shape != (center.shape[0], VIEW_DIM, DISK_AXIS_DIM):
        raise ValueError(
            f"disk_axes must have shape [B, {VIEW_DIM}, {DISK_AXIS_DIM}]"
        )
    if direction.ndim == 2:
        if direction.shape != (VIEWCELL_SUPPORT_FREQUENCY_COUNT, VIEW_DIM):
            raise ValueError(
                "directions must have shape [16, 9] or [B, 16, 9]"
            )
        center_projection = center @ direction.transpose(0, 1)
        axis_projection = torch.einsum("bda,fd->bfa", axes, direction)
        direction_norm = torch.linalg.vector_norm(direction, dim=-1).unsqueeze(0)
    elif direction.ndim == 3:
        if direction.shape != (
            center.shape[0],
            VIEWCELL_SUPPORT_FREQUENCY_COUNT,
            VIEW_DIM,
        ):
            raise ValueError(
                "directions must have shape [16, 9] or [B, 16, 9]"
            )
        center_projection = torch.einsum("bd,bfd->bf", center, direction)
        axis_projection = torch.einsum("bda,bfd->bfa", axes, direction)
        direction_norm = torch.linalg.vector_norm(direction, dim=-1)
    else:
        raise ValueError("directions must have shape [16, 9] or [B, 16, 9]")

    valid_norm = direction_norm > 1e-6
    safe_norm = direction_norm.clamp_min(1e-6)
    center_unit = center_projection / safe_norm
    radius_unit = torch.linalg.vector_norm(axis_projection, dim=-1) / safe_norm
    normalization = math.sqrt(float(VIEW_DIM))
    lower = ((center_unit - radius_unit) / normalization).clamp(-1.0, 1.0)
    upper = ((center_unit + radius_unit) / normalization).clamp(-1.0, 1.0)
    lower = torch.where(valid_norm, lower, torch.zeros_like(lower))
    upper = torch.where(valid_norm, upper, torch.zeros_like(upper))
    features = torch.stack([lower, upper], dim=-1).reshape(center.shape[0], -1)
    if features.shape != (center.shape[0], VIEWCELL_SUPPORT_ENVELOPE_DIM):
        raise RuntimeError("view-cell support-envelope layout drifted")
    return ViewCellSupportEnvelope(lower=lower, upper=upper, features=features)


def export_viewcell_extreme_envelope_schema() -> dict[str, Any]:
    return {
        "schema": VIEWCELL_EXTREME_ENVELOPE_SCHEMA,
        "input": {
            "centerView": ["B", VIEW_DIM],
            "diskAxes": ["B", VIEW_DIM, DISK_AXIS_DIM],
        },
        "outputDim": VIEWCELL_EXTREME_ENVELOPE_DIM,
        "interval": "center[j] +/- row_norm(diskAxes[j])",
        "runtimeSubposeExpansion": False,
        "additionalPerInstanceAssets": False,
    }


def export_viewcell_support_envelope_schema() -> dict[str, Any]:
    return {
        "schema": VIEWCELL_SUPPORT_ENVELOPE_SCHEMA,
        "directionCount": VIEWCELL_SUPPORT_FREQUENCY_COUNT,
        "outputDim": VIEWCELL_SUPPORT_ENVELOPE_DIM,
        "perDirectionOrder": ["minimumSupport", "maximumSupport"],
        "directionSource": "same learned directions as Fourier moment query",
        "runtimeSubposeExpansion": False,
        "additionalPerInstanceAssets": False,
    }


__all__ = [
    "VIEWCELL_EXTREME_ENVELOPE_DIM",
    "VIEWCELL_EXTREME_ENVELOPE_SCHEMA",
    "VIEWCELL_SUPPORT_ENVELOPE_DIM",
    "VIEWCELL_SUPPORT_ENVELOPE_SCHEMA",
    "ViewCellExtremeEnvelope",
    "ViewCellSupportEnvelope",
    "export_viewcell_extreme_envelope_schema",
    "export_viewcell_support_envelope_schema",
    "viewcell_extreme_envelope",
    "viewcell_support_envelope",
]
