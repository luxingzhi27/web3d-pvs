from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.viewcell_tail_boundary_deficit_loss import (
    viewcell_tail_boundary_deficit_loss,
)


class ViewcellTailBoundaryDeficitLossTests(unittest.TestCase):
    @staticmethod
    def _inputs(
        *,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if residual is None:
            residual = torch.zeros(8, dtype=torch.float32)
        base = torch.tensor([-3.0, 1.0, 2.0, -1.0, 0.0, 3.0, 1.5, -2.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        hit_rates = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0])
        weights = torch.tensor([1.0, 10.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
        offsets = torch.tensor([0, 4, 8], dtype=torch.long)
        return residual, base, target, hit_rates, weights, offsets

    @staticmethod
    def _all_tail_kwargs() -> dict[str, float | int]:
        return {
            "positive_mass_fraction": 1.0,
            "positive_count_cap": 8,
            "min_positive_count": 1,
            "negative_fraction": 1.0,
            "negative_count_cap": 8,
            "min_negative_count": 1,
            "gap_weight": 0.0,
            "residual_l2_weight": 0.0,
        }

    def test_outputs_are_finite_and_report_detailed_stats(self) -> None:
        inputs = self._inputs(
            residual=torch.tensor(
                [0.2, -0.1, 0.3, -0.4, 0.1, 0.0, -0.2, 0.4],
                requires_grad=True,
            )
        )
        loss, stats = viewcell_tail_boundary_deficit_loss(*inputs)

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(stats["poseCount"], 2.0)
        self.assertGreater(stats["selectedPositiveCount"], 0.0)
        self.assertGreater(stats["selectedNegativeCount"], 0.0)
        for value in stats.values():
            if isinstance(value, torch.Tensor):
                self.assertTrue(bool(torch.isfinite(value).all()))
            else:
                self.assertTrue(bool(torch.isfinite(torch.tensor(value))))

    def test_positive_and_negative_gradients_point_toward_boundary(self) -> None:
        residual = torch.zeros(4, requires_grad=True)
        base = torch.tensor([0.0, 1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        hit_rates = torch.tensor([0.0, 1.0, 1.0, 1.0])
        weights = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss, _ = viewcell_tail_boundary_deficit_loss(
            residual,
            base,
            target,
            hit_rates,
            weights,
            torch.tensor([0, 4]),
            **self._all_tail_kwargs(),
        )
        loss.backward()

        self.assertLess(float(residual.grad[0]), 0.0)
        self.assertLess(float(residual.grad[1]), 0.0)
        self.assertGreater(float(residual.grad[2]), 0.0)
        self.assertGreater(float(residual.grad[3]), 0.0)
        self.assertGreater(abs(float(residual.grad[0])), abs(float(residual.grad[1])))

    def test_positive_and_negative_classes_are_balanced_before_pose_average(self) -> None:
        common = {
            "target": torch.tensor([1.0, 0.0]),
            "hit_rates": torch.tensor([0.0, 1.0]),
            "weights": torch.tensor([1.0, 0.0]),
            "offsets": torch.tensor([0, 2]),
        }
        loss_one_negative, stats_one_negative = viewcell_tail_boundary_deficit_loss(
            torch.zeros(2),
            torch.tensor([0.0, 3.0]),
            common["target"],
            common["hit_rates"],
            common["weights"],
            common["offsets"],
            **self._all_tail_kwargs(),
        )

        loss_many_negatives, stats_many_negatives = (
            viewcell_tail_boundary_deficit_loss(
                torch.zeros(5),
                torch.tensor([0.0, 3.0, 3.0, 3.0, 3.0]),
                torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]),
                torch.zeros(5),
                torch.tensor([0, 5]),
                **self._all_tail_kwargs(),
            )
        )

        torch.testing.assert_close(loss_one_negative, loss_many_negatives)
        torch.testing.assert_close(
            stats_one_negative["lossPositiveDeficit"],
            stats_many_negatives["lossPositiveDeficit"],
        )
        torch.testing.assert_close(
            stats_one_negative["lossNegativeDeficit"],
            stats_many_negatives["lossNegativeDeficit"],
        )

    def test_no_qualifying_pose_returns_differentiable_zero(self) -> None:
        residual = torch.tensor([0.4, -0.3, 0.2], requires_grad=True)
        loss, stats = viewcell_tail_boundary_deficit_loss(
            residual,
            torch.tensor([0.0, 1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
            torch.ones(3),
            torch.ones(3),
            torch.tensor([0, 3]),
            residual_l2_weight=10.0,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(stats["poseCount"], 0.0)
        loss.backward()
        torch.testing.assert_close(residual.grad, torch.zeros_like(residual))

    def test_invalid_shapes_values_offsets_and_parameters_are_rejected(self) -> None:
        inputs = self._inputs()
        with self.assertRaisesRegex(ValueError, "align"):
            viewcell_tail_boundary_deficit_loss(inputs[0], inputs[1][:-1], *inputs[2:])

        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = inputs[0].clone()
            invalid[0] = float("nan")
            viewcell_tail_boundary_deficit_loss(invalid, *inputs[1:])

        with self.assertRaisesRegex(ValueError, "binary"):
            invalid = inputs[2].clone()
            invalid[0] = 0.5
            viewcell_tail_boundary_deficit_loss(
                inputs[0], inputs[1], invalid, *inputs[3:]
            )

        with self.assertRaisesRegex(ValueError, "visible_hit_rates"):
            invalid = inputs[3].clone()
            invalid[0] = 1.1
            viewcell_tail_boundary_deficit_loss(
                inputs[0], inputs[1], inputs[2], invalid, inputs[4], inputs[5]
            )

        with self.assertRaisesRegex(ValueError, "visible_weights"):
            invalid = inputs[4].clone()
            invalid[0] = -1.0
            viewcell_tail_boundary_deficit_loss(
                inputs[0], inputs[1], inputs[2], inputs[3], invalid, inputs[5]
            )

        with self.assertRaisesRegex(ValueError, "pose_offsets"):
            viewcell_tail_boundary_deficit_loss(
                *inputs[:-1], torch.tensor([1, 8], dtype=torch.long)
            )

        with self.assertRaisesRegex(ValueError, "positive_mass_fraction"):
            viewcell_tail_boundary_deficit_loss(
                *inputs, positive_mass_fraction=0.0
            )
        with self.assertRaisesRegex(ValueError, "negative_count_cap"):
            viewcell_tail_boundary_deficit_loss(*inputs, negative_count_cap=0)
        with self.assertRaisesRegex(ValueError, "huber_beta"):
            viewcell_tail_boundary_deficit_loss(*inputs, huber_beta=0.0)
        with self.assertRaisesRegex(ValueError, "min_positive_count"):
            viewcell_tail_boundary_deficit_loss(
                *inputs, positive_count_cap=1, min_positive_count=2
            )


if __name__ == "__main__":
    unittest.main()
