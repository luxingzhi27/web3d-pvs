from __future__ import annotations

import sys
import math
from pathlib import Path
import unittest

import torch

MODEL = Path(__file__).resolve().parents[2]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.spectral_query import (  # noqa: E402
    CHI_TABLE_BYTES,
    CHI_TABLE_MAX_ARGUMENT,
    CHI_TABLE_SIZE,
    VIEWCELL_MOMENT_ENVELOPE_SPECTRAL_SCHEMA,
    ViewCellMomentEnvelopeSpectralQuery,
)


class ViewCellMomentEnvelopeSpectralQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260814)
        self.center = torch.tensor(
            [[0.31, -0.22, 0.92, -0.45, 0.88, 0.12, -0.07, -0.82, -0.76]],
            dtype=torch.float32,
        )
        self.axes = torch.tensor(
            [[[0.13, -0.04], [0.02, 0.10], [-0.07, 0.03], [0.04, 0.02],
              [-0.03, 0.05], [0.06, -0.02], [0.01, 0.04], [-0.05, 0.01],
              [0.03, 0.02]]],
            dtype=torch.float32,
        )
        self.cycles = torch.tensor(
            [[0.17, -0.11, 0.08, 0.02, -0.05, 0.04, 0.03, -0.07, 0.06],
             [-0.09, 0.05, 0.12, -0.03, 0.04, -0.02, 0.07, 0.01, -0.06],
             [0.03, 0.08, -0.04, 0.06, 0.02, 0.05, -0.01, 0.09, 0.04]],
            dtype=torch.float32,
        )
        self.query = ViewCellMomentEnvelopeSpectralQuery(
            frequency_cycles=self.cycles,
            learnable_frequencies=False,
        )

    def test_export_schema_and_table_budget(self) -> None:
        schema = self.query.export_schema()
        self.assertEqual(schema["schema"], VIEWCELL_MOMENT_ENVELOPE_SPECTRAL_SCHEMA)
        self.assertEqual(schema["inputs"]["centerView"]["shape"], ["B", 9])
        self.assertEqual(schema["inputs"]["diskAxes"]["shape"], ["B", 9, 2])
        self.assertEqual(schema["inputs"]["frequencyCycles"]["units"], "cycles")
        self.assertEqual(schema["chiLookup"]["table"]["count"], CHI_TABLE_SIZE)
        self.assertEqual(schema["chiLookup"]["table"]["bytes"], CHI_TABLE_BYTES)
        self.assertLessEqual(CHI_TABLE_BYTES, 32 * 1024)
        self.assertEqual(self.query.chi_table.dtype, torch.float32)
        self.assertFalse(self.query.chi_table.requires_grad)

    def test_lookup_error_stays_below_registered_bound(self) -> None:
        argument = torch.linspace(0.0, CHI_TABLE_MAX_ARGUMENT, 200001)
        reference = torch.ones_like(argument)
        nonzero = argument != 0.0
        reference[nonzero] = 2.0 * torch.special.bessel_j1(argument[nonzero]) / argument[nonzero]
        actual = self.query.lookup_chi(argument)
        self.assertLess(float((actual - reference).abs().max()), 1e-4)

    def test_monte_carlo_matches_uniform_disk_moments(self) -> None:
        result = self.query(self.center, self.axes)
        sample_count = 100000
        radius = torch.sqrt(torch.rand(1, sample_count))
        theta = 2.0 * torch.pi * torch.rand(1, sample_count)
        unit_disk = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)
        points = self.center[:, None, :] + torch.einsum("bda,bna->bnd", self.axes, unit_disk)
        phase = 2.0 * torch.pi * torch.einsum("bnd,fd->bnf", points, self.cycles)
        sampled_sin = torch.sin(phase)
        sampled_cos = torch.cos(phase)
        expected_mean = torch.stack((sampled_sin.mean(dim=1), sampled_cos.mean(dim=1)), dim=-1)
        expected_std = torch.stack((sampled_sin.std(dim=1), sampled_cos.std(dim=1)), dim=-1)
        self.assertTrue(torch.allclose(result.mean_sin_cos, expected_mean[0], atol=0.018, rtol=0.0))
        self.assertTrue(torch.allclose(result.std_sin_cos, expected_std[0], atol=0.018, rtol=0.0))

    def test_finite_difference_gradient_through_frequency_and_lookup(self) -> None:
        frequency = self.cycles[:1].clone().requires_grad_(True)
        result = self.query(self.center, self.axes, frequency_cycles=frequency)
        loss = result.mean_sin[0, 0] + 0.4 * result.std_cos[0, 0]
        loss.backward()
        self.assertIsNotNone(frequency.grad)
        analytic = float(frequency.grad[0, 0])

        step = 1e-4
        plus = frequency.detach().clone()
        minus = frequency.detach().clone()
        plus[0, 0] += step
        minus[0, 0] -= step
        loss_plus = self.query(self.center, self.axes, frequency_cycles=plus)
        loss_minus = self.query(self.center, self.axes, frequency_cycles=minus)
        value_plus = float(loss_plus.mean_sin[0, 0] + 0.4 * loss_plus.std_cos[0, 0])
        value_minus = float(loss_minus.mean_sin[0, 0] + 0.4 * loss_minus.std_cos[0, 0])
        finite_difference = (value_plus - value_minus) / (2.0 * step)
        self.assertAlmostEqual(analytic, finite_difference, delta=0.02)

    def test_frequency_phase_uses_two_pi_cycles(self) -> None:
        center = torch.zeros((1, 9), dtype=torch.float32)
        center[0, 0] = 0.37
        frequency = torch.zeros((1, 9), dtype=torch.float32)
        frequency[0, 0] = 0.25
        result = self.query(center, torch.zeros((1, 9, 2)), frequency_cycles=frequency)
        phase = 2.0 * torch.pi * 0.37 * 0.25
        self.assertAlmostEqual(float(result.mean_sin[0, 0]), math.sin(float(phase)), places=6)
        self.assertAlmostEqual(float(result.mean_cos[0, 0]), math.cos(float(phase)), places=6)

    def test_zero_radius_is_exact_point_fourier(self) -> None:
        result = self.query(self.center, torch.zeros_like(self.axes))
        phase = 2.0 * torch.pi * torch.matmul(self.center, self.cycles.transpose(0, 1))
        self.assertTrue(torch.equal(result.s, torch.zeros_like(result.s)))
        self.assertTrue(torch.equal(result.mean_sin, torch.sin(phase)))
        self.assertTrue(torch.equal(result.mean_cos, torch.cos(phase)))
        self.assertTrue(torch.equal(result.std_sin, torch.zeros_like(result.std_sin)))
        self.assertTrue(torch.equal(result.std_cos, torch.zeros_like(result.std_cos)))
        self.assertTrue(bool(result.zero_radius.all()))

    def test_zero_radius_backward_has_finite_gradients(self) -> None:
        center = self.center.clone().requires_grad_(True)
        axes = torch.zeros_like(self.axes, requires_grad=True)
        frequency = self.cycles.clone().requires_grad_(True)
        result = self.query(center, axes, frequency_cycles=frequency)
        result.moments.sum().backward()
        for gradient in (center.grad, axes.grad, frequency.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_frequency_permutation_only_permutates_frequency_axis(self) -> None:
        permutation = torch.tensor([2, 0, 1], dtype=torch.long)
        original = self.query(self.center, self.axes)
        permuted = self.query(self.center, self.axes, frequency_cycles=self.cycles[permutation])
        self.assertTrue(torch.allclose(permuted.moments, original.moments[:, permutation], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(permuted.s, original.s[:, permutation], atol=1e-6, rtol=1e-6))

    def test_lookup_range_gate_covers_both_s_and_two_s(self) -> None:
        self.assertAlmostEqual(float(self.query.lookup_chi(torch.tensor(0.0))), 1.0, places=7)
        with self.assertRaisesRegex(ValueError, "strict range"):
            self.query.lookup_chi(torch.tensor(CHI_TABLE_MAX_ARGUMENT + 1.0))

        # This has s in range but 2*s outside the table, which must also fail.
        axes = torch.zeros((1, 9, 2), dtype=torch.float32)
        axes[0, 0, 0] = 1.0
        frequency = torch.zeros((1, 9), dtype=torch.float32)
        frequency[0, 0] = 30.0
        with self.assertRaisesRegex(ValueError, "strict range"):
            self.query(self.center, axes, frequency_cycles=frequency)


if __name__ == "__main__":
    unittest.main()
