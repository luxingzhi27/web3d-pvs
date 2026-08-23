from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.viewcell_boundary_opportunity_loss import (
    viewcell_boundary_opportunity_loss,
)


class ViewcellBoundaryOpportunityLossTests(unittest.TestCase):
    @staticmethod
    def _inputs(*, requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
        uplift = torch.tensor(
            [0.2, 0.1, 0.3, 0.4, 0.5, 0.6],
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        base = torch.tensor([-4.0, 2.0, 1.0, 0.5, -1.0, -2.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        hit_rates = torch.tensor([0.05, 0.80, 0.0, 0.0, 0.0, 0.0])
        weights = torch.tensor([10.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        offsets = torch.tensor([0, 6], dtype=torch.long)
        return uplift, base, target, hit_rates, weights, offsets

    def test_selects_frozen_positive_tail_and_negative_quantile_boundary(self) -> None:
        inputs = self._inputs(requires_grad=True)
        loss, stats = viewcell_boundary_opportunity_loss(
            *inputs,
            positive_tail_mass_fraction=0.05,
            negative_top_fraction=0.25,
            min_positive_count=1,
            min_negative_count=1,
        )

        self.assertEqual(stats["tailPositiveCount"], 1.0)
        self.assertEqual(stats["tailRarePositiveCount"], 1.0)
        self.assertEqual(stats["hardNegativeCount"], 1.0)
        self.assertAlmostEqual(
            float(stats["opportunityNegativeBoundaryLogit"]), 1.0
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        gradient = inputs[0].grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertLess(float(gradient[0]), 0.0)
        self.assertGreater(float(gradient[2]), 0.0)
        self.assertGreater(float(gradient[3]), 0.0)
        self.assertGreater(float(gradient[4]), 0.0)
        self.assertGreater(float(gradient[5]), 0.0)
        self.assertEqual(float(gradient[1]), 0.0)

    def test_all_negatives_are_anchored_not_only_the_hard_tail(self) -> None:
        inputs = self._inputs(requires_grad=True)
        loss, stats = viewcell_boundary_opportunity_loss(
            *inputs,
            negative_top_fraction=0.25,
            min_positive_count=1,
            min_negative_count=1,
            positive_margin_weight=0.0,
            hard_negative_uplift_weight=0.0,
            all_negative_uplift_weight=1.0,
        )
        self.assertGreater(float(stats["lossOpportunityAllNegativeUplift"]), 0.0)
        loss.backward()
        gradient = inputs[0].grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        torch.testing.assert_close(gradient[:2], torch.zeros_like(gradient[:2]))
        self.assertTrue(bool(torch.all(gradient[2:] > 0.0)))

    def test_hard_negative_anchor_adds_gradient_only_to_upper_tail(self) -> None:
        inputs = self._inputs(requires_grad=True)
        loss, _ = viewcell_boundary_opportunity_loss(
            *inputs,
            negative_top_fraction=0.25,
            min_positive_count=1,
            min_negative_count=1,
            positive_margin_weight=0.0,
            all_negative_uplift_weight=0.0,
            hard_negative_uplift_weight=1.0,
        )
        loss.backward()
        gradient = inputs[0].grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(float(gradient[2]), 0.0)
        torch.testing.assert_close(
            gradient[torch.tensor([0, 1, 3, 4, 5])],
            torch.zeros(5),
        )

    def test_negative_anchor_has_zero_gradient_at_zero_uplift(self) -> None:
        inputs = list(self._inputs())
        uplift = torch.zeros_like(inputs[0], requires_grad=True)
        inputs[0] = uplift
        loss, _ = viewcell_boundary_opportunity_loss(
            *inputs,
            positive_margin_weight=0.0,
            all_negative_uplift_weight=4.0,
            hard_negative_uplift_weight=4.0,
        )
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        torch.testing.assert_close(uplift.grad, torch.zeros_like(uplift))

    def test_missing_class_is_finite_and_negative_anchor_remains_connected(self) -> None:
        uplift = torch.tensor([0.2, 0.3], requires_grad=True)
        loss, stats = viewcell_boundary_opportunity_loss(
            uplift,
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 0.0]),
            torch.zeros(2),
            torch.zeros(2),
            torch.tensor([0, 2]),
        )
        self.assertEqual(stats["tailPositiveCount"], 0.0)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertTrue(bool(torch.all(uplift.grad > 0.0)))

    def test_invalid_shapes_values_and_parameters_are_rejected(self) -> None:
        inputs = self._inputs()
        with self.assertRaisesRegex(ValueError, "align"):
            viewcell_boundary_opportunity_loss(inputs[0], inputs[1][:-1], *inputs[2:])
        invalid = inputs[0].clone()
        invalid[0] = -0.1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            viewcell_boundary_opportunity_loss(invalid, *inputs[1:])
        with self.assertRaisesRegex(ValueError, "positive_tail_mass_fraction"):
            viewcell_boundary_opportunity_loss(
                *inputs, positive_tail_mass_fraction=0.0
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            viewcell_boundary_opportunity_loss(*inputs, temperature=0.0)
        with self.assertRaisesRegex(ValueError, "pose_offsets"):
            viewcell_boundary_opportunity_loss(
                *inputs[:-1], torch.tensor([1, 6], dtype=torch.long)
            )


if __name__ == "__main__":
    unittest.main()
