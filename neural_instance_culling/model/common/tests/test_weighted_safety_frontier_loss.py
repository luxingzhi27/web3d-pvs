from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.weighted_safety_frontier_loss import (
    weighted_safety_frontier_pair_loss,
)


class WeightedSafetyFrontierPairLossTests(unittest.TestCase):
    @staticmethod
    def _frontier() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {
            101: (torch.tensor([10, 11]), torch.tensor([12, 13])),
            202: (torch.tensor([20]), torch.tensor([21, 22])),
        }

    @staticmethod
    def _inputs(
        *, requires_grad: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.tensor(
            [-0.5, 0.1, -0.2, 0.3, 0.8, -0.4, 0.2],
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        instance_ids = torch.tensor([10, 11, 12, 13, 20, 21, 22], dtype=torch.long)
        visible_weights = torch.tensor([9.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0])
        pose_indices = torch.tensor([101, 202], dtype=torch.long)
        pose_offsets = torch.tensor([0, 4, 7], dtype=torch.long)
        return logits, instance_ids, visible_weights, pose_indices, pose_offsets

    def test_gradient_direction_is_positive_down_and_negative_up(self) -> None:
        logits = torch.zeros(3, requires_grad=True)
        loss, stats = weighted_safety_frontier_pair_loss(
            logits,
            torch.tensor([1, 2, 3]),
            torch.tensor([5.0, 0.0, 0.0]),
            torch.tensor([9]),
            torch.tensor([0, 3]),
            {9: (torch.tensor([1]), torch.tensor([2, 3]))},
            margin=0.2,
            temperature=0.25,
            pose_cvar_weight=0.0,
        )
        loss.backward()

        self.assertLess(float(logits.grad[0]), 0.0)
        self.assertGreater(float(logits.grad[1]), 0.0)
        self.assertGreater(float(logits.grad[2]), 0.0)
        self.assertEqual(stats["pairCount"], 2.0)

    def test_loss_and_gap_diagnostics_are_invariant_to_global_logit_shift(self) -> None:
        inputs = self._inputs()
        first_loss, first_stats = weighted_safety_frontier_pair_loss(
            *inputs,
            self._frontier(),
            pose_cvar_fraction=0.5,
            pose_cvar_weight=0.7,
            batch_global_pair_weight=0.3,
        )
        shifted_loss, shifted_stats = weighted_safety_frontier_pair_loss(
            inputs[0] + 37.0,
            *inputs[1:],
            self._frontier(),
            pose_cvar_fraction=0.5,
            pose_cvar_weight=0.7,
            batch_global_pair_weight=0.3,
        )

        torch.testing.assert_close(first_loss, shifted_loss)
        torch.testing.assert_close(first_stats["meanGap"], shifted_stats["meanGap"])
        torch.testing.assert_close(first_stats["worstGap"], shifted_stats["worstGap"])

    def test_selection_uses_fixed_ids_and_ignores_unlisted_batch_members(self) -> None:
        logits = torch.zeros(5, requires_grad=True)
        loss, stats = weighted_safety_frontier_pair_loss(
            logits,
            torch.tensor([100, 200, 300, 400, 500]),
            torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0]),
            torch.tensor([7]),
            torch.tensor([0, 5]),
            {
                7: (
                    torch.tensor([200, 999]),
                    torch.tensor([400, 888]),
                )
            },
            pose_cvar_weight=0.0,
        )
        loss.backward()

        self.assertEqual(stats["poseCount"], 1.0)
        self.assertEqual(stats["positiveCount"], 1.0)
        self.assertEqual(stats["negativeCount"], 1.0)
        self.assertEqual(stats["pairCount"], 1.0)
        torch.testing.assert_close(logits.grad[[0, 2, 4]], torch.zeros(3))
        self.assertLess(float(logits.grad[1]), 0.0)
        self.assertGreater(float(logits.grad[3]), 0.0)

    def test_pose_mean_cvar_and_batch_global_components_are_reported(self) -> None:
        logits = torch.tensor([0.0, 0.0, 0.0, 2.0])
        loss, stats = weighted_safety_frontier_pair_loss(
            logits,
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.tensor([10, 20]),
            torch.tensor([0, 2, 4]),
            {
                10: (torch.tensor([1]), torch.tensor([2])),
                20: (torch.tensor([3]), torch.tensor([4])),
            },
            margin=0.0,
            temperature=1.0,
            pose_cvar_fraction=0.5,
            pose_cvar_weight=1.0,
            batch_global_pair_weight=1.0,
        )

        pose_mean = stats["lossPoseMean"]
        pose_cvar = stats["lossPoseCvar"]
        global_pair = stats["lossBatchGlobalPair"]
        self.assertGreaterEqual(float(pose_cvar), float(pose_mean))
        torch.testing.assert_close(loss, pose_mean + pose_cvar + global_pair)
        self.assertEqual(stats["poseCount"], 2.0)
        self.assertEqual(stats["globalPairCount"], 4.0)
        self.assertGreater(float(stats["worstGap"]), -3.0)

    def test_empty_frontier_returns_differentiable_zero(self) -> None:
        logits = torch.tensor([0.4, -0.2], requires_grad=True)
        loss, stats = weighted_safety_frontier_pair_loss(
            logits,
            torch.tensor([1, 2]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([5]),
            torch.tensor([0, 2]),
            {},
        )

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(stats["poseCount"], 0.0)
        self.assertEqual(stats["pairCount"], 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_invalid_shapes_values_and_parameters_are_rejected(self) -> None:
        inputs = self._inputs()
        with self.assertRaisesRegex(ValueError, "align"):
            weighted_safety_frontier_pair_loss(
                inputs[0], inputs[1], inputs[2][:-1], inputs[3], inputs[4], self._frontier()
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            invalid_logits = inputs[0].clone()
            invalid_logits[0] = float("nan")
            weighted_safety_frontier_pair_loss(
                invalid_logits, *inputs[1:], self._frontier()
            )

        with self.assertRaisesRegex(ValueError, "pose_offsets"):
            weighted_safety_frontier_pair_loss(
                inputs[0], inputs[1], inputs[2], inputs[3], torch.tensor([1, 7]), self._frontier()
            )

        with self.assertRaisesRegex(ValueError, "temperature"):
            weighted_safety_frontier_pair_loss(
                *inputs, self._frontier(), temperature=0.0
            )
        with self.assertRaisesRegex(ValueError, "pose_cvar_fraction"):
            weighted_safety_frontier_pair_loss(
                *inputs, self._frontier(), pose_cvar_fraction=0.0
            )
        with self.assertRaisesRegex(ValueError, "positive_importance_power"):
            weighted_safety_frontier_pair_loss(
                *inputs, self._frontier(), positive_importance_power=-1.0
            )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            weighted_safety_frontier_pair_loss(
                *inputs,
                {101: (torch.tensor([10]), torch.tensor([10]))},
            )


if __name__ == "__main__":
    unittest.main()
