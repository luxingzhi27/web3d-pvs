from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.cross_pose_operating_loss import (
    CrossPoseOperatingDualState,
    cross_pose_recall_constrained_operating_loss,
    pose_asymmetric_binary_cross_entropy,
)


class CrossPoseOperatingLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.weights = torch.tensor([4.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.offsets = torch.tensor([0, 2, 6])

    def test_positive_share_changes_pose_normalized_bce(self) -> None:
        logits = torch.tensor([-1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
        low, _ = pose_asymmetric_binary_cross_entropy(
            logits,
            self.labels,
            self.offsets,
            self.weights,
            positive_class_fraction=0.2,
        )
        high, _ = pose_asymmetric_binary_cross_entropy(
            logits,
            self.labels,
            self.offsets,
            self.weights,
            positive_class_fraction=0.8,
        )
        self.assertAlmostEqual(float(low), float(high), places=6)

        logits = torch.tensor([-3.0, -1.0, -3.0, -1.0, -1.0, -1.0])
        low, _ = pose_asymmetric_binary_cross_entropy(
            logits,
            self.labels,
            self.offsets,
            self.weights,
            positive_class_fraction=0.2,
        )
        high, _ = pose_asymmetric_binary_cross_entropy(
            logits,
            self.labels,
            self.offsets,
            self.weights,
            positive_class_fraction=0.8,
        )
        self.assertGreater(float(high), float(low))

    def test_recall_oversatisfaction_has_exactly_zero_safety_penalty(self) -> None:
        loss, parts = cross_pose_recall_constrained_operating_loss(
            torch.tensor([8.0, -4.0, 8.0, -4.0, -3.0, -2.0]),
            self.labels,
            self.offsets,
            self.weights,
            torch.tensor(0.0),
            weighted_recall_target=0.99,
            dual_multiplier=10.0,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(parts["crossPoseWeightedRecallActiveViolation"]), 0.0)
        self.assertEqual(float(parts["crossPoseSafetyPenalty"]), 0.0)

    def test_lowering_global_hard_negatives_reduces_loss(self) -> None:
        high, high_parts = cross_pose_recall_constrained_operating_loss(
            torch.tensor([4.0, 3.0, 4.0, 2.0, 1.0, 0.0]),
            self.labels,
            self.offsets,
            self.weights,
            torch.tensor(0.0),
            weighted_recall_target=0.9,
        )
        low, low_parts = cross_pose_recall_constrained_operating_loss(
            torch.tensor([4.0, -3.0, 4.0, -2.0, -1.0, -4.0]),
            self.labels,
            self.offsets,
            self.weights,
            torch.tensor(0.0),
            weighted_recall_target=0.9,
        )
        self.assertLess(float(low), float(high))
        self.assertLess(
            float(low_parts["crossPoseHardNegativeFpr"]),
            float(high_parts["crossPoseHardNegativeFpr"]),
        )

    def test_boundary_receives_finite_gradient(self) -> None:
        boundary = torch.tensor(0.0, requires_grad=True)
        loss, _ = cross_pose_recall_constrained_operating_loss(
            torch.tensor([1.0, 0.5, 1.0, 0.2, 0.0, -0.2]),
            self.labels,
            self.offsets,
            self.weights,
            boundary,
            weighted_recall_target=0.99,
        )
        loss.backward()
        self.assertTrue(bool(torch.isfinite(boundary.grad)))
        self.assertNotEqual(float(boundary.grad), 0.0)

    def test_dual_state_uses_signed_violation_and_clamps(self) -> None:
        state = CrossPoseOperatingDualState(
            multiplier=1.0, learning_rate=0.5, maximum=2.0
        )
        state.update(torch.tensor(1.0))
        self.assertEqual(state.multiplier, 1.5)
        state.update(torch.tensor(-10.0))
        self.assertEqual(state.multiplier, 0.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary"):
            cross_pose_recall_constrained_operating_loss(
                torch.zeros(6),
                torch.tensor([0.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
                self.offsets,
                self.weights,
                torch.tensor(0.0),
            )
        with self.assertRaisesRegex(ValueError, "positive mass"):
            cross_pose_recall_constrained_operating_loss(
                torch.zeros(6),
                self.labels,
                self.offsets,
                torch.zeros(6),
                torch.tensor(0.0),
            )


if __name__ == "__main__":
    unittest.main()
