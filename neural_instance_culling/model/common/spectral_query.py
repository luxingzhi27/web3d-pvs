"""Uniform-disk moment-envelope Fourier queries for the v3 view-cell contract.

The query represents a view-cell as a two-dimensional disk embedded in the
nine-dimensional view feature space.  For a frequency expressed in cycles,
the disk characteristic function is evaluated at
``s = 2*pi*||B^T f||``.  The characteristic-function table is fixed FP32
data; the forward path only performs piecewise-linear interpolation so the
query remains differentiable with respect to the input and frequency values.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn


VIEWCELL_MOMENT_ENVELOPE_SPECTRAL_SCHEMA = "view-cell-moment-envelope-spectral-query-v3"
MOMENT_ENVELOPE_SPECTRAL_SCHEMA = VIEWCELL_MOMENT_ENVELOPE_SPECTRAL_SCHEMA

VIEW_DIM = 9
DISK_AXIS_DIM = 2
DEFAULT_FREQUENCY_COUNT = 16

# The table stores chi(s), while the moment envelope also queries chi(2s).
# The 320 range covers the registered 8-cycle frequency norm because the
# bounded disk axes imply 2s <= 4*pi*sqrt(9)*8 ~= 301.6.  8192 FP32 values
# occupy exactly 32 KiB; dense-reference validation keeps the interpolation
# error below 1e-4 across the full range.
CHI_TABLE_SIZE = 8192
CHI_TABLE_MAX_ARGUMENT = 320.0
CHI_TABLE_BYTES = CHI_TABLE_SIZE * 4
CHI_TABLE_DTYPE = "float32"


def _as_floating(value: torch.Tensor | Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    elif tensor.dtype not in (torch.float32, torch.float64):
        tensor = tensor.to(dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _validate_center_and_axes(
    center_view: torch.Tensor | Any,
    disk_axes: torch.Tensor | Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = _as_floating(center_view, "center_view")
    axes = _as_floating(disk_axes, "disk_axes")
    if center.ndim != 2 or tuple(center.shape[-1:]) != (VIEW_DIM,):
        raise ValueError(f"center_view must have shape [B, {VIEW_DIM}]")
    if axes.ndim != 3 or tuple(axes.shape[-2:]) != (VIEW_DIM, DISK_AXIS_DIM):
        raise ValueError(f"disk_axes must have shape [B, {VIEW_DIM}, {DISK_AXIS_DIM}]")
    if int(center.shape[0]) != int(axes.shape[0]):
        raise ValueError("center_view and disk_axes must have the same batch size")
    axes = axes.to(device=center.device, dtype=center.dtype)
    return center, axes


def _validate_frequency_cycles(
    frequency_cycles: torch.Tensor | Any,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    frequencies = _as_floating(frequency_cycles, "frequency_cycles")
    if frequencies.ndim == 1:
        if tuple(frequencies.shape) != (VIEW_DIM,):
            raise ValueError(f"frequency_cycles must have shape [F, {VIEW_DIM}] or [B, F, {VIEW_DIM}]")
        frequencies = frequencies.unsqueeze(0)
    elif frequencies.ndim == 2:
        if frequencies.shape[-1] != VIEW_DIM or frequencies.shape[0] <= 0:
            raise ValueError(f"frequency_cycles must have shape [F, {VIEW_DIM}] or [B, F, {VIEW_DIM}]")
    elif frequencies.ndim == 3:
        if frequencies.shape[-1] != VIEW_DIM or frequencies.shape[1] <= 0:
            raise ValueError(f"frequency_cycles must have shape [F, {VIEW_DIM}] or [B, F, {VIEW_DIM}]")
        if frequencies.shape[0] not in (1, batch):
            raise ValueError("batched frequency_cycles must have batch size 1 or B")
        if frequencies.shape[0] == 1 and batch != 1:
            frequencies = frequencies.expand(batch, -1, -1)
    else:
        raise ValueError(f"frequency_cycles must have shape [F, {VIEW_DIM}] or [B, F, {VIEW_DIM}]")
    return frequencies.to(device=device, dtype=dtype)


def _default_frequency_cycles() -> torch.Tensor:
    """Return deterministic, non-collapsed frequencies measured in cycles."""

    index = torch.arange(1, DEFAULT_FREQUENCY_COUNT + 1, dtype=torch.float32).unsqueeze(1)
    dimension = torch.arange(1, VIEW_DIM + 1, dtype=torch.float32).unsqueeze(0)
    directions = torch.sin(index * dimension * 0.71) + 0.5 * torch.cos(index * dimension * 0.37)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    magnitudes = torch.logspace(math.log10(0.05), math.log10(8.0), DEFAULT_FREQUENCY_COUNT)
    return directions * magnitudes.unsqueeze(1)


def _build_reference_chi_table() -> torch.Tensor:
    """Build the detached FP32 reference table once at module construction."""

    grid = torch.linspace(
        0.0,
        CHI_TABLE_MAX_ARGUMENT,
        CHI_TABLE_SIZE,
        dtype=torch.float32,
    )
    with torch.no_grad():
        table = torch.ones_like(grid)
        nonzero = grid != 0.0
        table[nonzero] = 2.0 * torch.special.bessel_j1(grid[nonzero]) / grid[nonzero]
    return table.detach().contiguous()


def _linear_lookup_chi(
    argument: torch.Tensor,
    table: torch.Tensor,
) -> torch.Tensor:
    """Interpolate chi from a uniform table while retaining argument grads."""

    if not argument.is_floating_point():
        argument = argument.to(dtype=torch.float32)
    if not bool(torch.isfinite(argument).all()):
        raise ValueError("chi lookup argument contains non-finite values")
    if bool((argument < 0.0).any()) or bool((argument > CHI_TABLE_MAX_ARGUMENT).any()):
        raise ValueError(
            "chi lookup argument is outside the strict range "
            f"[0, {CHI_TABLE_MAX_ARGUMENT}]"
        )

    values = argument.reshape(-1)
    step = CHI_TABLE_MAX_ARGUMENT / float(CHI_TABLE_SIZE - 1)
    position = values / step
    # The bin selection is discrete by definition.  Detaching only the index
    # leaves the interpolation fraction, and therefore the argument gradient,
    # on the original tensor.
    lower = torch.floor(position.detach()).to(dtype=torch.long)
    lower = lower.clamp(max=CHI_TABLE_SIZE - 2)
    fraction = position - lower.to(dtype=position.dtype)
    table_values = table.to(device=values.device, dtype=values.dtype)
    lower_value = table_values[lower]
    upper_value = table_values[lower + 1]
    result = lower_value + fraction * (upper_value - lower_value)
    return result.reshape(argument.shape)


def _stable_standard_deviation(variance: torch.Tensor) -> torch.Tensor:
    """Return an exact zero with a finite gradient at degenerate moments."""

    floor = torch.finfo(variance.dtype).eps
    safe = torch.sqrt(torch.clamp(variance, min=floor))
    return torch.where(variance > floor, safe, torch.zeros_like(safe))


@dataclass(frozen=True)
class MomentEnvelopeSpectralQueryResult:
    """Named outputs for one batched uniform-disk Fourier query."""

    center_view: torch.Tensor
    disk_axes: torch.Tensor
    frequencies_cycles: torch.Tensor
    phase: torch.Tensor
    s: torch.Tensor
    chi_s: torch.Tensor
    chi_2s: torch.Tensor
    mean_sin: torch.Tensor
    mean_cos: torch.Tensor
    std_sin: torch.Tensor
    std_cos: torch.Tensor
    moments: torch.Tensor
    spectral_features: torch.Tensor
    zero_radius: torch.Tensor

    @property
    def mean_sin_cos(self) -> torch.Tensor:
        return torch.stack((self.mean_sin, self.mean_cos), dim=-1)

    @property
    def std_sin_cos(self) -> torch.Tensor:
        return torch.stack((self.std_sin, self.std_cos), dim=-1)

    @property
    def expected_sin_cos(self) -> torch.Tensor:
        return self.mean_sin_cos

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "centerView": self.center_view,
            "diskAxes": self.disk_axes,
            "frequencyCycles": self.frequencies_cycles,
            "phase": self.phase,
            "s": self.s,
            "chiS": self.chi_s,
            "chi2S": self.chi_2s,
            "meanSin": self.mean_sin,
            "meanCos": self.mean_cos,
            "stdSin": self.std_sin,
            "stdCos": self.std_cos,
            "meanSinCos": self.mean_sin_cos,
            "stdSinCos": self.std_sin_cos,
            "moments": self.moments,
            "spectralFeatures": self.spectral_features,
            "zeroRadius": self.zero_radius,
        }

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.as_dict()[key]


class ViewCellMomentEnvelopeSpectralQuery(nn.Module):
    """Differentiable Fourier moments of a uniform embedded disk.

    ``disk_axes[b]`` is the 9-by-2 matrix ``B`` in
    ``x = center_view[b] + B @ u`` for a uniformly distributed unit-disk
    coordinate ``u``.  Frequencies are expressed in cycles, not radians.
    The returned per-frequency order is mean sine, mean cosine, standard
    deviation of sine, and standard deviation of cosine.
    """

    def __init__(
        self,
        cycles: torch.Tensor | Any | None = None,
        *,
        frequency_cycles: torch.Tensor | Any | None = None,
        learnable_frequencies: bool = True,
    ) -> None:
        super().__init__()
        if cycles is not None and frequency_cycles is not None:
            raise ValueError("provide only one of cycles and frequency_cycles")
        initial = frequency_cycles if frequency_cycles is not None else cycles
        if initial is None:
            initial = _default_frequency_cycles()
        initial_tensor = _as_floating(initial, "frequency_cycles")
        if initial_tensor.ndim == 1:
            initial_tensor = initial_tensor.unsqueeze(0)
        if initial_tensor.ndim != 2 or initial_tensor.shape[-1] != VIEW_DIM or initial_tensor.shape[0] <= 0:
            raise ValueError(f"frequency_cycles must have shape [F, {VIEW_DIM}]")
        initial_tensor = initial_tensor.detach().clone()
        if learnable_frequencies:
            self.frequency_cycles = nn.Parameter(initial_tensor)
        else:
            self.register_buffer("frequency_cycles", initial_tensor)
        self.learnable_frequencies = bool(learnable_frequencies)
        self.register_buffer("chi_table", _build_reference_chi_table())

    @property
    def frequency_count(self) -> int:
        return int(self.frequency_cycles.shape[0])

    @property
    def chi_table_size(self) -> int:
        return CHI_TABLE_SIZE

    @property
    def chi_table_max_argument(self) -> float:
        return CHI_TABLE_MAX_ARGUMENT

    @property
    def chi_table_bytes(self) -> int:
        return CHI_TABLE_BYTES

    def lookup_chi(self, argument: torch.Tensor | Any) -> torch.Tensor:
        """Evaluate the fixed table with strict range checking."""

        tensor = _as_floating(argument, "chi_argument")
        return _linear_lookup_chi(tensor, self.chi_table)

    def export_schema(self) -> dict[str, Any]:
        return {
            "schema": VIEWCELL_MOMENT_ENVELOPE_SPECTRAL_SCHEMA,
            "version": 3,
            "inputs": {
                "centerView": {"shape": ["B", VIEW_DIM], "dtype": "float32"},
                "diskAxes": {"shape": ["B", VIEW_DIM, DISK_AXIS_DIM], "dtype": "float32"},
                "frequencyCycles": {"shape": ["F", VIEW_DIM], "dtype": "float32", "units": "cycles"},
            },
            "uniformDisk": {
                "parameterization": "x = centerView + diskAxes @ u",
                "coordinate": "u is uniform on the unit disk",
                "argument": "s = 2*pi*norm(diskAxes^T @ frequencyCycles)",
            },
            "chiLookup": {
                "function": "chi(s) = 2*J1(s)/s",
                "zeroLimit": 1.0,
                "interpolation": "piecewise_linear",
                "range": [0.0, CHI_TABLE_MAX_ARGUMENT],
                "rangePolicy": "strict_error_for_s_and_2s_outside_range",
                "table": {
                    "dtype": CHI_TABLE_DTYPE,
                    "count": CHI_TABLE_SIZE,
                    "bytes": CHI_TABLE_BYTES,
                    "torchSpecial": "detached_reference_generation_only",
                },
            },
            "outputs": {
                "perFrequencyOrder": ["meanSin", "meanCos", "stdSin", "stdCos"],
                "momentsShape": ["B", "F", 4],
                "spectralFeatureShape": ["B", "F*4"],
                "zeroRadius": "point_fourier",
            },
            "runtime": {"subposeExpansion": False, "dtype": "float32"},
        }

    def forward(
        self,
        center_view: torch.Tensor | Any,
        disk_axes: torch.Tensor | Any,
        frequency_cycles: torch.Tensor | Any | None = None,
        *,
        cycles: torch.Tensor | Any | None = None,
    ) -> MomentEnvelopeSpectralQueryResult:
        if frequency_cycles is not None and cycles is not None:
            raise ValueError("provide only one of frequency_cycles and cycles")
        center, axes = _validate_center_and_axes(center_view, disk_axes)
        requested = frequency_cycles if frequency_cycles is not None else cycles
        if requested is None:
            requested = self.frequency_cycles
        frequencies = _validate_frequency_cycles(
            requested,
            int(center.shape[0]),
            center.device,
            center.dtype,
        )

        if frequencies.ndim == 2:
            phase = 2.0 * math.pi * torch.matmul(center, frequencies.transpose(0, 1))
            projected_axes = torch.einsum("bda,fd->bfa", axes, frequencies)
        else:
            phase = 2.0 * math.pi * torch.einsum("bd,bfd->bf", center, frequencies)
            projected_axes = torch.einsum("bda,bfd->bfa", axes, frequencies)
        s = 2.0 * math.pi * torch.linalg.vector_norm(projected_axes, dim=-1)
        chi_s = self.lookup_chi(s)
        chi_2s = self.lookup_chi(2.0 * s)

        mean_sin = torch.sin(phase) * chi_s
        mean_cos = torch.cos(phase) * chi_s
        phase_double = 2.0 * phase
        second_sin = 0.5 * (1.0 - torch.cos(phase_double) * chi_2s)
        second_cos = 0.5 * (1.0 + torch.cos(phase_double) * chi_2s)
        variance_sin = torch.clamp(second_sin - mean_sin.square(), min=0.0)
        variance_cos = torch.clamp(second_cos - mean_cos.square(), min=0.0)
        std_sin = _stable_standard_deviation(variance_sin)
        std_cos = _stable_standard_deviation(variance_cos)

        # At s=0 the disk perturbation has no phase variation for that
        # frequency.  Enforce the exact degenerate value instead of allowing
        # floating-point cancellation in the second-moment equations to leave
        # a tiny nonzero standard deviation.
        zero_s = s == 0.0
        zero_value = torch.zeros_like(std_sin)
        std_sin = torch.where(zero_s, zero_value, std_sin)
        std_cos = torch.where(zero_s, zero_value, std_cos)
        zero_radius = torch.linalg.vector_norm(axes, dim=(-2, -1)) == 0.0
        moments = torch.stack((mean_sin, mean_cos, std_sin, std_cos), dim=-1)
        spectral_features = moments.reshape(int(center.shape[0]), -1)
        return MomentEnvelopeSpectralQueryResult(
            center_view=center,
            disk_axes=axes,
            frequencies_cycles=frequencies,
            phase=phase,
            s=s,
            chi_s=chi_s,
            chi_2s=chi_2s,
            mean_sin=mean_sin,
            mean_cos=mean_cos,
            std_sin=std_sin,
            std_cos=std_cos,
            moments=moments,
            spectral_features=spectral_features,
            zero_radius=zero_radius,
        )

    def encode(
        self,
        center_view: torch.Tensor | Any,
        disk_axes: torch.Tensor | Any,
        frequency_cycles: torch.Tensor | Any | None = None,
        *,
        cycles: torch.Tensor | Any | None = None,
    ) -> MomentEnvelopeSpectralQueryResult:
        return self.forward(center_view, disk_axes, frequency_cycles, cycles=cycles)


def spectral_query(
    center_view: torch.Tensor | Any,
    disk_axes: torch.Tensor | Any,
    frequency_cycles: torch.Tensor | Any,
) -> MomentEnvelopeSpectralQueryResult:
    """Functional convenience wrapper using fixed, non-learnable frequencies."""

    query = ViewCellMomentEnvelopeSpectralQuery(
        frequency_cycles=frequency_cycles,
        learnable_frequencies=False,
    )
    return query(center_view, disk_axes)


compute_moment_envelope = spectral_query


__all__ = [
    "CHI_TABLE_BYTES",
    "CHI_TABLE_DTYPE",
    "CHI_TABLE_MAX_ARGUMENT",
    "CHI_TABLE_SIZE",
    "DISK_AXIS_DIM",
    "MOMENT_ENVELOPE_SPECTRAL_SCHEMA",
    "MomentEnvelopeSpectralQueryResult",
    "VIEWCELL_MOMENT_ENVELOPE_SPECTRAL_SCHEMA",
    "VIEW_DIM",
    "ViewCellMomentEnvelopeSpectralQuery",
    "compute_moment_envelope",
    "spectral_query",
]
