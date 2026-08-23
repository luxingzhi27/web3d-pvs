from __future__ import annotations

import math
import unittest

import torch

from neural_instance_culling.model.common.weighted_neyman_pearson_operating_loss import (
    WeightedNeymanPearsonDualState,
    weighted_neyman_pearson_operating_loss,
)


class WeightedNeymanPearsonOperatingLossTests(unittest.TestCase):
    @staticmethod
    def _target() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([8.0, 2.0, 0.0, 0.0]),
        )

    def _loss(
        self,
        logits: torch.Tensor,
        *,
        boundary: float = 0.0,
        dual: float = 1.0,
        recall_target: float = 0.9,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        target, weights = self._target()
        return weighted_neyman_pearson_operating_loss(
            logits,
            target,
            weights,
            torch.tensor(boundary),
            1.0,
            dual,
            recall_target,
        )

    def test_lowering_negative_logits_reduces_objective(self) -> None:
        high_negative, _ = self._loss(
            torch.tensor([2.0, 1.0, 2.0, 1.0]), dual=0.0
        )
        low_negative, stats = self._loss(
            torch.tensor([2.0, 1.0, -2.0, -3.0]), dual=0.0
        )
        self.assertLess(float(low_negative), float(high_negative))
        self.assertLess(float(stats["soft_fpr"]), 0.5)

    def test_raising_positive_logits_reduces_recall_violation(self) -> None:
        low_positive, low_stats = self._loss(
            torch.tensor([-2.0, -1.0, -2.0, -3.0]), dual=2.0
        )
        high_positive, high_stats = self._loss(
            torch.tensor([2.0, 1.0, -2.0, -3.0]), dual=2.0
        )
        self.assertLess(
            float(high_stats["weighted_recall_violation"]),
            float(low_stats["weighted_recall_violation"]),
        )
        self.assertLess(float(high_positive), float(low_positive))

    def test_all_positive_prediction_is_penalized_by_fpr(self) -> None:
        all_positive, positive_stats = self._loss(
            torch.full((4,), 12.0), dual=0.0
        )
        all_negative, negative_stats = self._loss(
            torch.full((4,), -12.0), dual=0.0
        )
        self.assertGreater(float(positive_stats["soft_fpr"]), 0.99)
        self.assertLess(float(negative_stats["soft_fpr"]), 1e-4)
        self.assertGreater(float(all_positive), float(all_negative))

    def test_boundary_has_opposing_finite_fpr_and_recall_gradients(self) -> None:
        target, weights = self._target()
        boundary = torch.tensor(0.0, requires_grad=True)
        _loss, stats = weighted_neyman_pearson_operating_loss(
            torch.tensor([1.0, 0.5, 0.5, 1.0]),
            target,
            weights,
            boundary,
            0.75,
            2.0,
            0.9,
        )
        fpr_gradient = torch.autograd.grad(
            stats["soft_fpr"], boundary, retain_graph=True
        )[0]
        recall_penalty_gradient = torch.autograd.grad(
            stats["dual_penalty"], boundary
        )[0]
        self.assertTrue(bool(torch.isfinite(fpr_gradient)))
        self.assertTrue(bool(torch.isfinite(recall_penalty_gradient)))
        self.assertLess(float(fpr_gradient), 0.0)
        self.assertGreater(float(recall_penalty_gradient), 0.0)

    def test_dual_state_ascends_from_detached_violation_and_clamps(self) -> None:
        state = WeightedNeymanPearsonDualState(
            multiplier=0.2, learning_rate=0.5, maximum=1.0
        )
        violation = torch.tensor(0.8, requires_grad=True)
        state.update(violation)
        self.assertAlmostEqual(state.multiplier, 0.6)
        state.update(torch.tensor(10.0))
        self.assertEqual(state.multiplier, 1.0)
        state.update(torch.tensor(-10.0))
        self.assertEqual(state.multiplier, 0.0)
        self.assertEqual(
            state.as_dict(),
            {"multiplier": 0.0, "learning_rate": 0.5, "maximum": 1.0},
        )

    def test_zero_negative_weights_are_valid_but_zero_positive_mass_is_rejected(self) -> None:
        loss, stats = weighted_neyman_pearson_operating_loss(
            torch.zeros(4),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([2.0, 0.0, 0.0, 0.0]),
            torch.tensor(0.0),
            1.0,
            1.0,
            0.9,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertAlmostEqual(float(stats["positive_weight_mass"]), 2.0)

        with self.assertRaisesRegex(ValueError, "positive visible weights"):
            weighted_neyman_pearson_operating_loss(
                torch.zeros(4),
                torch.tensor([1.0, 1.0, 0.0, 0.0]),
                torch.zeros(4),
                torch.tensor(0.0),
                1.0,
                1.0,
                0.9,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            weighted_neyman_pearson_operating_loss(
                torch.zeros(4),
                torch.tensor([1.0, 1.0, 0.0, 0.0]),
                torch.tensor([1.0, -1.0, 0.0, 0.0]),
                torch.tensor(0.0),
                1.0,
                1.0,
                0.9,
            )

    def test_malformed_binary_inputs_and_nonfinite_values_are_rejected(self) -> None:
        target, weights = self._target()
        valid = (
            torch.zeros(4),
            target,
            weights,
            torch.tensor(0.0),
            1.0,
            1.0,
            0.9,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = list(valid)
            invalid[0] = torch.tensor([0.0, float("nan"), 0.0, 0.0])
            weighted_neyman_pearson_operating_loss(*invalid)
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = list(valid)
            invalid[2] = torch.tensor([8.0, 2.0, float("inf"), 0.0])
            weighted_neyman_pearson_operating_loss(*invalid)
        with self.assertRaisesRegex(ValueError, "binary"):
            invalid = list(valid)
            invalid[1] = torch.tensor([1.0, 0.5, 0.0, 0.0])
            weighted_neyman_pearson_operating_loss(*invalid)
        with self.assertRaisesRegex(ValueError, "align"):
            invalid = list(valid)
            invalid[2] = torch.ones(3)
            weighted_neyman_pearson_operating_loss(*invalid)
        with self.assertRaisesRegex(ValueError, "boundary_logit"):
            invalid = list(valid)
            invalid[3] = torch.tensor(float("inf"))
            weighted_neyman_pearson_operating_loss(*invalid)

    def test_every_returned_diagnostic_is_finite(self) -> None:
        loss, stats = self._loss(torch.tensor([1.0, 0.5, -0.5, -1.0]))
        values = [loss, *stats.values()]
        for value in values:
            if isinstance(value, torch.Tensor):
                self.assertTrue(bool(torch.isfinite(value).all()))
            else:
                self.assertTrue(math.isfinite(float(value)))


if __name__ == "__main__":
    unittest.main()
