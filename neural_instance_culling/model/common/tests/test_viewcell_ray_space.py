from __future__ import annotations

import unittest
import math

import torch

from neural_instance_culling.model.common.viewcell_ray_space import (
    build_horizontal_disk_ray_query,
    normalized_relative_log_depth,
)


class ViewCellRaySpaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        self.view = torch.tensor([[0.15, -0.10, -0.98, 1.1, 0.65]], dtype=torch.float32)
        self.aabb = torch.tensor([[-2.0, -1.0, -12.0, 3.0, 4.0, -7.0]], dtype=torch.float32)

    def test_disk_axes_match_centered_finite_difference(self) -> None:
        radius = 2.0
        query = build_horizontal_disk_ray_query(self.camera, self.view, self.aabb, radius)
        epsilon = 1e-3
        columns = []
        for world_axis in (0, 2):
            offset = torch.zeros_like(self.camera)
            offset[:, world_axis] = epsilon
            plus = build_horizontal_disk_ray_query(
                self.camera + offset, self.view, self.aabb, 0.0
            ).center_view
            minus = build_horizontal_disk_ray_query(
                self.camera - offset, self.view, self.aabb, 0.0
            ).center_view
            columns.append((plus - minus) / (2.0 * epsilon) * radius)
        finite_axes = torch.stack(columns, dim=-1)
        row_budget = (1.0 - query.center_view.abs()).clamp_min(0.0)
        row_norm = torch.linalg.vector_norm(finite_axes, dim=-1)
        scale = torch.where(
            row_norm > row_budget,
            row_budget / row_norm.clamp_min(1e-12),
            torch.ones_like(row_norm),
        )
        bounded_finite_axes = finite_axes * scale.unsqueeze(-1)
        self.assertTrue(
            torch.allclose(query.disk_axes, bounded_finite_axes, atol=3e-3, rtol=2e-2),
            (query.disk_axes - bounded_finite_axes).abs().max().item(),
        )

    def test_zero_radius_has_zero_axes(self) -> None:
        query = build_horizontal_disk_ray_query(self.camera, self.view, self.aabb, 0.0)
        self.assertTrue(torch.equal(query.disk_axes, torch.zeros_like(query.disk_axes)))
        self.assertEqual(tuple(query.center_view.shape), (1, 9))

    def test_singular_jacobian_is_bounded_by_the_feature_domain(self) -> None:
        camera = torch.zeros((1, 3), dtype=torch.float32)
        view = torch.tensor([[0.0, 0.0, -1.0, 1.0, 0.6]], dtype=torch.float32)
        near_side_aabb = torch.tensor([[0.9, -0.1, -0.02, 1.1, 0.1, 0.02]], dtype=torch.float32)
        query = build_horizontal_disk_ray_query(camera, view, near_side_aabb, 2.0)
        row_norm = torch.linalg.vector_norm(query.disk_axes, dim=-1)
        row_budget = 1.0 - query.center_view.abs()
        self.assertTrue(bool(torch.all(row_norm <= row_budget + 1e-6)))
        spectral_norm = torch.linalg.matrix_norm(query.disk_axes, ord=2, dim=(-2, -1))
        worst_two_s = 4.0 * math.pi * 8.0 * float(spectral_norm.max())
        self.assertLess(worst_two_s, 320.0)

    def test_relative_log_depth_is_monotone(self) -> None:
        distance = torch.tensor([1.0, 2.0, 5.0, 20.0])
        value = normalized_relative_log_depth(distance, torch.ones_like(distance), 0.0, 4.0)
        self.assertTrue(bool(torch.all(value[1:] >= value[:-1])))
        self.assertTrue(bool(torch.all((value >= 0.0) & (value <= 1.0))))

    def test_invalid_quantiles_fail(self) -> None:
        with self.assertRaises(ValueError):
            normalized_relative_log_depth(torch.tensor([1.0]), torch.tensor([1.0]), 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
