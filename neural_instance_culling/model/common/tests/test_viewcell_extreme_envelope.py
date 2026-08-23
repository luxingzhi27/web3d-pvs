from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.viewcell_extreme_envelope import (
    VIEWCELL_EXTREME_ENVELOPE_DIM,
    VIEWCELL_SUPPORT_ENVELOPE_DIM,
    viewcell_extreme_envelope,
    viewcell_support_envelope,
)


class ViewCellExtremeEnvelopeTests(unittest.TestCase):
    def test_linear_disk_samples_stay_inside_reported_interval(self) -> None:
        torch.manual_seed(7)
        center = torch.empty((5, 9)).uniform_(-0.6, 0.6)
        axes = torch.empty((5, 9, 2)).uniform_(-0.2, 0.2)
        result = viewcell_extreme_envelope(center, axes)

        angle = torch.linspace(0.0, 2.0 * torch.pi, 257)[:-1]
        unit = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        samples = center[:, None, :] + torch.einsum("bda,sa->bsd", axes, unit)
        self.assertTrue(torch.all(samples >= result.lower[:, None, :] - 1e-6))
        self.assertTrue(torch.all(samples <= result.upper[:, None, :] + 1e-6))
        self.assertEqual(result.features.shape, (5, VIEWCELL_EXTREME_ENVELOPE_DIM))

    def test_zero_disk_has_zero_range_and_equal_bounds(self) -> None:
        center = torch.linspace(-0.8, 0.8, 9).reshape(1, 9)
        result = viewcell_extreme_envelope(center, torch.zeros((1, 9, 2)))
        torch.testing.assert_close(result.half_range, torch.zeros_like(center))
        torch.testing.assert_close(result.lower, center)
        torch.testing.assert_close(result.upper, center)

    def test_best_overlap_is_not_worse_than_worst_overlap(self) -> None:
        center = torch.zeros((3, 9))
        center[:, 7:9] = -0.5
        axes = torch.zeros((3, 9, 2))
        axes[:, 5, 0] = torch.tensor([0.0, 0.2, 0.4])
        axes[:, 6, 1] = torch.tensor([0.0, 0.1, 0.3])
        result = viewcell_extreme_envelope(center, axes)
        best = result.overlap_margins[:, :2]
        worst = result.overlap_margins[:, 2:]
        self.assertTrue(torch.all(best >= worst))

    def test_rejects_non_finite_and_wrong_shapes(self) -> None:
        with self.assertRaises(ValueError):
            viewcell_extreme_envelope(torch.zeros((2, 8)), torch.zeros((2, 9, 2)))
        invalid = torch.zeros((1, 9))
        invalid[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            viewcell_extreme_envelope(invalid, torch.zeros((1, 9, 2)))

    def test_joint_support_bounds_sampled_disk_projections(self) -> None:
        torch.manual_seed(11)
        center = torch.empty((4, 9)).uniform_(-0.5, 0.5)
        axes = torch.empty((4, 9, 2)).uniform_(-0.12, 0.12)
        directions = torch.randn((16, 9))
        result = viewcell_support_envelope(center, axes, directions)

        angle = torch.linspace(0.0, 2.0 * torch.pi, 513)[:-1]
        unit_disk_boundary = torch.stack(
            [torch.cos(angle), torch.sin(angle)], dim=-1
        )
        samples = center[:, None, :] + torch.einsum(
            "bda,sa->bsd", axes, unit_disk_boundary
        )
        direction_norm = torch.linalg.vector_norm(directions, dim=-1)
        projected = torch.einsum("bsd,fd->bsf", samples, directions)
        projected = projected / direction_norm.reshape(1, 1, -1) / 3.0
        self.assertTrue(torch.all(projected >= result.lower[:, None, :] - 1e-6))
        self.assertTrue(torch.all(projected <= result.upper[:, None, :] + 1e-6))
        self.assertEqual(result.features.shape, (4, VIEWCELL_SUPPORT_ENVELOPE_DIM))

    def test_zero_joint_direction_is_finite_zero(self) -> None:
        result = viewcell_support_envelope(
            torch.zeros((2, 9)),
            torch.zeros((2, 9, 2)),
            torch.zeros((16, 9)),
        )
        self.assertTrue(torch.equal(result.features, torch.zeros((2, 32))))


if __name__ == "__main__":
    unittest.main()
