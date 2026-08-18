from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.pose_balanced_frontier_loss import (
    dynamic_weighted_safety_frontier_loss,
    pose_balanced_binary_cross_entropy,
    pose_balanced_frontier_visibility_loss,
)


class PoseBalancedFrontierLossTests(unittest.TestCase):
    def test_pose_balanced_bce_does_not_let_a_large_negative_class_dominate(self) -> None:
        logits = torch.zeros(14)
        labels = torch.tensor(
            [1.0, 0.0, 1.0, 0.0] + [0.0] * 10,
            dtype=torch.float32,
        )
        weights = torch.tensor([4.0, 0.0, 1.0, 0.0] + [0.0] * 10)
        loss, parts = pose_balanced_binary_cross_entropy(
            logits,
            labels,
            torch.tensor([0, 2, 14]),
            weights,
        )

        torch.testing.assert_close(loss, torch.tensor(0.69314718))
        self.assertEqual(parts["poseBalancedPoseCount"], 2.0)
        self.assertEqual(parts["poseBalancedMixedPoseCount"], 2.0)

    def test_frontier_updates_only_low_positive_and_high_negative_members(self) -> None:
        logits = torch.tensor(
            [-2.0, 1.0, 0.5, -0.5, -1.0, 2.0],
            requires_grad=True,
        )
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        weights = torch.tensor([1.0, 99.0, 0.0, 0.0, 0.0, 0.0])
        loss, parts = dynamic_weighted_safety_frontier_loss(
            logits,
            labels,
            torch.tensor([0, 6]),
            weights,
            positive_mass_fraction=0.005,
            positive_count_cap=4,
            negative_top_fraction=0.01,
            negative_count_cap=4,
            margin=0.5,
            temperature=0.25,
        )
        loss.backward()

        self.assertLess(float(logits.grad[0]), 0.0)
        self.assertEqual(float(logits.grad[1]), 0.0)
        self.assertEqual(float(logits.grad[2]), 0.0)
        self.assertEqual(float(logits.grad[3]), 0.0)
        self.assertEqual(float(logits.grad[4]), 0.0)
        self.assertGreater(float(logits.grad[5]), 0.0)
        self.assertEqual(parts["dynamicSafetyFrontierPositiveCount"], 1.0)
        self.assertEqual(parts["dynamicSafetyFrontierNegativeCount"], 1.0)

    def test_frontier_membership_changes_with_current_scores(self) -> None:
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        weights = torch.tensor([1.0, 1.0, 0.0, 0.0])
        first = torch.tensor([-1.0, 1.0, -1.0, 1.0], requires_grad=True)
        second = torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True)
        first_loss, _ = dynamic_weighted_safety_frontier_loss(
            first, labels, torch.tensor([0, 4]), weights
        )
        second_loss, _ = dynamic_weighted_safety_frontier_loss(
            second, labels, torch.tensor([0, 4]), weights
        )
        first_loss.backward()
        second_loss.backward()

        self.assertNotEqual(float(first.grad[0]), 0.0)
        self.assertEqual(float(first.grad[1]), 0.0)
        self.assertEqual(float(second.grad[0]), 0.0)
        self.assertNotEqual(float(second.grad[1]), 0.0)

    def test_combined_loss_is_exact_sum_of_two_registered_terms(self) -> None:
        logits = torch.tensor([-0.5, 0.2, 0.7, -0.4])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        weights = torch.tensor([4.0, 0.0, 1.0, 0.0])
        loss, parts = pose_balanced_frontier_visibility_loss(
            logits,
            labels,
            torch.tensor([0, 2, 4]),
            weights,
            frontier_weight=0.4,
        )

        torch.testing.assert_close(
            loss,
            parts["lossVisibilityPoseBalanced"]
            + parts["lossVisibilityFrontierWeighted"],
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        logits = torch.zeros(2)
        labels = torch.tensor([1.0, 0.0])
        weights = torch.tensor([1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "pose_offsets"):
            pose_balanced_binary_cross_entropy(
                logits, labels, torch.tensor([1, 2]), weights
            )
        with self.assertRaisesRegex(ValueError, "negative fraction"):
            dynamic_weighted_safety_frontier_loss(
                logits,
                labels,
                torch.tensor([0, 2]),
                weights,
                negative_top_fraction=0.0,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            pose_balanced_frontier_visibility_loss(
                logits,
                labels,
                torch.tensor([0, 2]),
                weights,
                frontier_weight=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
