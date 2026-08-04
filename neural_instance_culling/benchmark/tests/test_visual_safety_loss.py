import sys
import unittest
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
sys.path.insert(0, str(MODEL_DIR))

from common.pose_set_loss import pose_subpose_robust_safety_loss, pose_visual_safety_loss


class VisualSafetyLossTest(unittest.TestCase):
    def test_high_weight_positive_receives_stronger_gradient(self):
        logits = torch.zeros((4, 1), requires_grad=True)
        target = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        offsets = torch.tensor([0, 4])
        weights = torch.tensor([[100.0], [10.0], [0.0], [0.0]])
        loss, _ = pose_visual_safety_loss(
            logits,
            target,
            offsets,
            weights,
            weight_power=1.0,
            tail_k=1,
            tail_margin=1.0,
            tail_weight=0.5,
        )
        loss.backward()
        self.assertLess(float(logits.grad[0]), float(logits.grad[1]))

    def test_weight_scale_does_not_change_normalized_mass_term(self):
        logits = torch.tensor([[0.2], [-0.4], [0.0]], requires_grad=True)
        target = torch.tensor([[1.0], [1.0], [0.0]])
        offsets = torch.tensor([0, 3])
        first, first_parts = pose_visual_safety_loss(
            logits, target, offsets, torch.tensor([[100.0], [20.0], [0.0]]), tail_weight=0.0
        )
        second, second_parts = pose_visual_safety_loss(
            logits, target, offsets, torch.tensor([[10000.0], [2000.0], [0.0]]), tail_weight=0.0
        )
        self.assertAlmostEqual(first_parts["lossVisualSafetyMass"], second_parts["lossVisualSafetyMass"], places=7)
        self.assertAlmostEqual(float(first), float(second), places=7)

    def test_zero_weights_fall_back_to_positive_miss_loss(self):
        logits = torch.zeros((2, 1), requires_grad=True)
        target = torch.tensor([[1.0], [0.0]])
        offsets = torch.tensor([0, 2])
        loss, parts = pose_visual_safety_loss(logits, target, offsets, torch.zeros((2, 1)), tail_weight=0.0)
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(parts["lossVisualSafetyMass"], 0.0)

    def test_rare_positive_receives_bounded_extra_protection(self):
        logits = torch.zeros((3, 1), requires_grad=True)
        target = torch.tensor([[1.0], [1.0], [0.0]])
        offsets = torch.tensor([0, 3])
        weights = torch.tensor([[100.0], [100.0], [0.0]])
        hit_rates = torch.tensor([[0.0], [1.0], [0.0]])
        loss, parts = pose_subpose_robust_safety_loss(
            logits,
            target,
            offsets,
            weights,
            hit_rates,
            rare_weight=1.0,
            frequency_power=0.5,
            tail_weight=0.0,
        )
        loss.backward()
        self.assertLess(float(logits.grad[0]), float(logits.grad[1]))
        self.assertGreater(parts["subposeRobustRareMean"], 0.0)

    def test_subpose_robust_loss_does_not_reward_negative_predictions(self):
        logits = torch.zeros((3, 1), requires_grad=True)
        target = torch.tensor([[1.0], [0.0], [0.0]])
        offsets = torch.tensor([0, 3])
        loss, _ = pose_subpose_robust_safety_loss(
            logits,
            target,
            offsets,
            torch.tensor([[10.0], [0.0], [0.0]]),
            torch.tensor([[0.2], [0.0], [0.0]]),
            tail_weight=0.0,
        )
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[1]), 0.0, places=7)
        self.assertAlmostEqual(float(logits.grad[2]), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
